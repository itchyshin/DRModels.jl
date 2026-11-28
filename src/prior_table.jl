# Literal-only parsing for the prior-table facade. Target expansion and model
# eligibility are deliberately separate: parsing must not discard SD intent.
const _PRIOR_NUMBER = raw"[+-]?(?:(?:[0-9]+(?:\.[0-9]*)?)|(?:\.[0-9]+))(?:[eE][+-]?[0-9]+)?"
const _PRIOR_WS = raw"[ \t\r\n\f\v]*"
const _PRIOR_CALL = Regex("^" * _PRIOR_WS * "(normal|student_t|ridge|gamma|lkj)" * _PRIOR_WS * raw"\(" * _PRIOR_WS * "(" * _PRIOR_NUMBER * ")" * _PRIOR_WS * "(?:," * _PRIOR_WS * "(" * _PRIOR_NUMBER * ")" * _PRIOR_WS * ")?(?:," * _PRIOR_WS * "(" * _PRIOR_NUMBER * ")" * _PRIOR_WS * ")?" * raw"\)" * _PRIOR_WS * raw"$")

function _parse_prior_table(table)
    Tables.istable(table) || throw(ArgumentError("priors table must implement Tables.jl"))
    columns = Tables.columns(table)
    names = collect(Tables.columnnames(columns))
    length(unique(names)) == length(names) || throw(ArgumentError("duplicate prior table columns"))
    required = (:prior, :class, :coef)
    all(in(names), required) || throw(ArgumentError("prior table requires prior, class and coef columns"))
    all(in((required..., :dpar, :group, :scale, :from_coef, :to_coef)), names) || throw(ArgumentError("unknown prior table column"))
    cols = Dict(n => Tables.getcolumn(columns, n) for n in names)
    n = length(cols[:prior])
    n > 0 || throw(ArgumentError("prior table must not be empty"))
    all(c -> length(c) == n, values(cols)) || throw(ArgumentError("prior table columns must have equal lengths"))
    result = NamedTuple[]
    for i in 1:n
        all(c -> c[i] isa AbstractString, values(cols)) || throw(ArgumentError("prior table cells must be non-missing strings"))
        prior, class, coef = (String(cols[k][i]) for k in required)
        dpar = haskey(cols, :dpar) ? String(cols[:dpar][i]) : ""
        group = haskey(cols, :group) ? String(cols[:group][i]) : ""
        scale = haskey(cols, :scale) ? String(cols[:scale][i]) :
            class == "fixef" ? "link" : class == "ranef_sd" ? "natural_sd" : ""
        from_coef = haskey(cols, :from_coef) ? String(cols[:from_coef][i]) : ""
        to_coef = haskey(cols, :to_coef) ? String(cols[:to_coef][i]) : ""
        isempty(dpar) && (dpar = "mu")
        class in ("fixef", "ranef_sd", "residual_sd", "correlation") || throw(ArgumentError("unsupported prior class $class"))
        m = match(_PRIOR_CALL, prior)
        m === nothing && throw(ArgumentError("invalid prior distribution syntax"))
        kind = Symbol(m.captures[1])
        arity = kind === :student_t ? 3 : kind in (:ridge, :lkj) ? 1 : 2
        raw_nums = filter(!isnothing, m.captures[2:4])
        length(raw_nums) == arity || throw(ArgumentError("wrong prior distribution arity"))
        nums = map(x -> tryparse(Float64, x), raw_nums)
        all(x -> x !== nothing && isfinite(x), nums) || throw(ArgumentError("prior parameters must be finite numbers"))
        a = nums[1]
        if kind === :normal
            class == "fixef" || throw(ArgumentError("normal requires fixef"))
            sd = nums[2]
            sd > 0 && isfinite(inv(sd * sd)) && inv(sd * sd) > 0 || throw(ArgumentError("normal sd is outside the representable precision range"))
        elseif kind === :student_t
            class == "fixef" || throw(ArgumentError("student_t requires fixef"))
            nums[2] > 0 && isfinite(inv(nums[2] * nums[2])) && inv(nums[2] * nums[2]) > 0 && nums[3] > 0 ||
                throw(ArgumentError("Student-t scale and degrees of freedom must be positive and representable"))
        elseif kind === :ridge
            class == "fixef" && a >= 0 || throw(ArgumentError("ridge requires fixef and nonnegative lambda"))
        elseif kind === :gamma
            class in ("ranef_sd", "residual_sd") || throw(ArgumentError("gamma requires ranef_sd or residual_sd"))
            shape = nums[2]; rate = shape / a
            a > 0 && shape > 1 && isfinite(rate) && rate > 0 || throw(ArgumentError("invalid Gamma mean, shape or rate"))
        else
            class == "correlation" && a > 1 || throw(ArgumentError("LKJ requires correlation and eta greater than 1"))
        end
        if class == "correlation"
            kind === :lkj || throw(ArgumentError("correlation priors require lkj(eta)"))
            haskey(cols, :scale) && scale == "natural_correlation" ||
                throw(ArgumentError("correlation priors require scale=natural_correlation"))
            if dpar == "rho12"
                coef == "(Intercept)" && isempty(group) && isempty(from_coef) && isempty(to_coef) ||
                    throw(ArgumentError("rho12 correlation priors require coef=(Intercept) and no group or pair selectors"))
            elseif dpar == "mu"
                valid_pair = isempty(coef) && !isempty(group) && !isempty(from_coef) &&
                    !isempty(to_coef) && from_coef != to_coef &&
                    (from_coef == "(Intercept)" || to_coef == "(Intercept)")
                valid_pair || throw(ArgumentError("random-slope correlation priors require dpar=mu, group, and distinct (Intercept)/slope from_coef/to_coef selectors"))
                if from_coef != "(Intercept)"
                    from_coef, to_coef = to_coef, from_coef
                end
            else
                throw(ArgumentError("correlation dpar must be rho12 or mu"))
            end
            push!(result, (kind=kind, class=class, coef=coef, dpar=dpar, group=group,
                           from_coef=from_coef, to_coef=to_coef,
                           scale=scale, arguments=Tuple(nums)))
        elseif class == "residual_sd"
            kind === :gamma && dpar == "sigma" && coef == "(Intercept)" && isempty(group) ||
                throw(ArgumentError("residual_sd Gamma priors require dpar=sigma, coef=(Intercept), and no group"))
            haskey(cols, :scale) && scale == "natural_sd" ||
                throw(ArgumentError("residual_sd priors require scale=natural_sd"))
            (!isempty(from_coef) || !isempty(to_coef)) &&
                throw(ArgumentError("from_coef/to_coef require class=correlation"))
            push!(result, (kind=kind, class=class, coef=coef, dpar=dpar, group=group,
                           scale=scale, arguments=Tuple(nums)))
        else
            (!isempty(from_coef) || !isempty(to_coef)) &&
                throw(ArgumentError("from_coef/to_coef require class=correlation"))
            class == "fixef" && (!isempty(group) || coef == "all") && throw(ArgumentError("invalid fixed-effect selector"))
            if class == "ranef_sd"
                haskey(cols, :scale) && !isempty(scale) && scale != "natural_sd" &&
                    throw(ArgumentError("ranef_sd priors require scale=natural_sd"))
            elseif class == "fixef"
                haskey(cols, :scale) && !isempty(scale) && scale != "link" &&
                    throw(ArgumentError("fixef priors require scale=link"))
            else
                haskey(cols, :scale) && !isempty(scale) &&
                    throw(ArgumentError("scale is only supported for SD and correlation targets"))
            end
            push!(result, (kind=kind, class=class, coef=coef, dpar=dpar, group=group,
                           scale=class == "ranef_sd" && isempty(scale) ? "natural_sd" : scale,
                           arguments=Tuple(nums)))
        end
    end
    return result
end

# `groups` must come from the already validated model, not a filtered subset of
# unsupported terms. Expand each SD target once, independently of group levels.
function _expand_prior_table(rows, groups)
    length(unique(groups)) == length(groups) || throw(ArgumentError("ambiguous random-intercept groups"))
    output = NamedTuple[]
    seen = Set{String}()
    residual_sd_seen = false
    seen_correlations = Set{Tuple{String,String,String,String}}()
    for row in rows
        if row.class == "correlation"
            key = (row.dpar, row.group, row.from_coef, row.to_coef)
            key in seen_correlations && throw(ArgumentError("overlapping correlation prior target $(key)"))
            push!(seen_correlations, key)
            push!(output, row)
            continue
        elseif row.class == "residual_sd"
            residual_sd_seen && throw(ArgumentError("overlapping residual SD prior target sigma:(Intercept)"))
            residual_sd_seen = true
            push!(output, row)
            continue
        elseif row.class == "fixef"
            push!(output, row)
            continue
        end
        row.dpar == "mu" || throw(ArgumentError("SD priors require dpar=mu"))
        selected = if isempty(row.group)
            if isempty(row.coef)
                groups
            else
                row.coef == "(Intercept)" && throw(ArgumentError("use group for a grouping variable named (Intercept)"))
                [row.coef]
            end
        else
            row.coef in ("", "(Intercept)", row.group) || throw(ArgumentError("conflicting group and coefficient selectors"))
            [row.group]
        end
        isempty(selected) && throw(ArgumentError("SD prior matched no random intercept"))
        for group in selected
            group in groups || throw(ArgumentError("unknown random-intercept group $group"))
            group in seen && throw(ArgumentError("overlapping SD prior target $group"))
            push!(seen, group)
            push!(output, merge(row, (group=group, coef="(Intercept)")))
        end
    end
    return output
end

function _prior_table_specs(rows, groups)
    expanded = _expand_prior_table(rows, groups)
    map(expanded) do row
        if row.class == "correlation"
            drm_prior_lkj(dpar=Symbol(row.dpar), eta=row.arguments[1], group=row.group,
                          coef=row.coef, from_coef=row.from_coef,
                          to_coef=row.to_coef, scale=Symbol(row.scale))
        elseif row.class == "ranef_sd"
            drm_prior_sd_gamma(dpar=row.dpar, group=row.group, shape=row.arguments[2], rate=row.arguments[2] / row.arguments[1])
        elseif row.class == "residual_sd"
            drm_prior_residual_sd_gamma(dpar=Symbol(row.dpar), coef=row.coef,
                shape=row.arguments[2], rate=row.arguments[2] / row.arguments[1])
        elseif row.kind === :normal
            drm_prior_normal(dpar=row.dpar, coef=isempty(row.coef) ? "all" : row.coef,
                             intercept=isempty(row.coef) ? "exclude" : nothing,
                             mean=row.arguments[1], sd=row.arguments[2])
        elseif row.kind === :student_t
            drm_prior_student_t(dpar=row.dpar, coef=isempty(row.coef) ? "all" : row.coef,
                                intercept=isempty(row.coef) ? "exclude" : nothing,
                                location=row.arguments[1], scale=row.arguments[2], df=row.arguments[3])
        else
            drm_prior_ridge(dpar=row.dpar, coef=isempty(row.coef) ? "all" : row.coef,
                            intercept=isempty(row.coef) ? "exclude" : nothing,
                            lambda=row.arguments[1])
        end
    end
end

# Present table-resolved targets in the table contract's vocabulary. The native
# resolver records remain unchanged for fitting, replay, and bridge marshalling.
function _prior_table_preview_row(row)
    hasproperty(row, :class) || return row
    native_class = row.class
    public_class = native_class === :random_effect_sd ? :ranef_sd :
        native_class in (:residual_correlation, :random_effect_correlation) ? :correlation :
        native_class
    public_scale = native_class === :random_effect_sd ? :natural_sd : row.scale
    drop_block = native_class === :random_effect_correlation
    fields = Pair{Symbol,Any}[]
    for name in keys(row)
        name === :block && drop_block && continue
        value = name === :class ? public_class : name === :scale ? public_scale : getproperty(row, name)
        push!(fields, name => value)
    end
    names = Tuple(first.(fields))
    values = Tuple(last.(fields))
    return NamedTuple{names}(values)
end

"""Resolve priors against a model and return a canonical target preview.

Table previews use classes `:ranef_sd`, `:residual_sd`, and `:correlation`,
with scales `:natural_sd` or `:natural_correlation`. Random-effect correlation
targets are identified by `group`, `from_coef`, and `to_coef`. The same preview
labels are returned for table and constructor inputs; native resolver records
remain private to fitting, replay, and bridge paths.
"""
function resolve_priors(f, fam; data, priors)
    resolved = DRModels._resolve_priors_native(f, fam; data=data, priors=priors)
    return map(_prior_table_preview_row, resolved)
end

# Validate the complete model before expanding broad selectors. The resolved
# preview supplies exactly the selected SD identities in model order.
function _table_priors_for_model(f, fam, data, table)
    rows = _parse_prior_table(table)
    resolved = _resolve_priors_native(f, fam; data=data, priors=table)
    groups = String[r.group for r in resolved if hasproperty(r, :group)]
    specs = _prior_table_specs(rows, groups)
    fixed_rows = filter(r -> hasproperty(r, :precision), resolved)
    replay = AbstractDrmPrior[]
    for spec in specs
        if _is_coefficient_prior(spec)
            selected = filter(fixed_rows) do row
                row.dpar == spec.dpar && row.kind == spec.kind &&
                    row.mean == spec.mean && row.precision == spec.precision &&
                    row.constant == spec.constant && isequal(row.df, spec.df) &&
                    (spec.coef == "all" ? row.reason == :all : row.coef == spec.coef)
            end
            isempty(selected) && throw(ArgumentError("prior table fixed target could not be replayed"))
            append!(replay, [FixedPrior(row.kind, row.dpar, row.coef, nothing,
                                        row.mean, row.precision, row.constant, row.df) for row in selected])
        else
            push!(replay, spec)
        end
    end
    replay
end
