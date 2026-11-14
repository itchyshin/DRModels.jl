# Literal-only parsing for the prior-table facade. Target expansion and model
# eligibility are deliberately separate: parsing must not discard SD intent.
const _PRIOR_NUMBER = raw"[+-]?(?:(?:[0-9]+(?:\.[0-9]*)?)|(?:\.[0-9]+))(?:[eE][+-]?[0-9]+)?"
const _PRIOR_WS = raw"[ \t\r\n\f\v]*"
const _PRIOR_CALL = Regex("^" * _PRIOR_WS * "(normal|ridge|gamma)" * _PRIOR_WS * raw"\(" * _PRIOR_WS * "(" * _PRIOR_NUMBER * ")" * _PRIOR_WS * "(?:," * _PRIOR_WS * "(" * _PRIOR_NUMBER * ")" * _PRIOR_WS * ")?" * raw"\)" * _PRIOR_WS * raw"$")

function _parse_prior_table(table)
    Tables.istable(table) || throw(ArgumentError("priors table must implement Tables.jl"))
    columns = Tables.columns(table)
    names = collect(Tables.columnnames(columns))
    length(unique(names)) == length(names) || throw(ArgumentError("duplicate prior table columns"))
    required = (:prior, :class, :coef)
    all(in(names), required) || throw(ArgumentError("prior table requires prior, class and coef columns"))
    all(in((required..., :dpar, :group)), names) || throw(ArgumentError("unknown prior table column"))
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
        isempty(dpar) && (dpar = "mu")
        class in ("fixef", "ranef_sd") || throw(ArgumentError("unsupported prior class $class"))
        m = match(_PRIOR_CALL, prior)
        m === nothing && throw(ArgumentError("invalid prior distribution syntax"))
        kind = Symbol(m.captures[1])
        (kind === :ridge ? m.captures[3] === nothing : m.captures[3] !== nothing) || throw(ArgumentError("wrong prior distribution arity"))
        nums = map(x -> tryparse(Float64, x), filter(!isnothing, m.captures[2:3]))
        all(x -> x !== nothing && isfinite(x), nums) || throw(ArgumentError("prior parameters must be finite numbers"))
        a = nums[1]
        if kind === :normal
            class == "fixef" || throw(ArgumentError("normal requires fixef"))
            sd = nums[2]
            sd > 0 && isfinite(inv(sd * sd)) && inv(sd * sd) > 0 || throw(ArgumentError("normal sd is outside the representable precision range"))
        elseif kind === :ridge
            class == "fixef" && a >= 0 || throw(ArgumentError("ridge requires fixef and nonnegative lambda"))
        else
            class == "ranef_sd" || throw(ArgumentError("gamma requires ranef_sd"))
            shape = nums[2]; rate = shape / a
            a > 0 && shape > 1 && isfinite(rate) && rate > 0 || throw(ArgumentError("invalid Gamma mean, shape or rate"))
        end
        class == "fixef" && (!isempty(group) || coef == "all") && throw(ArgumentError("invalid fixed-effect selector"))
        push!(result, (kind=kind, class=class, coef=coef, dpar=dpar, group=group, arguments=Tuple(nums)))
    end
    return result
end

# `groups` must come from the already validated model, not a filtered subset of
# unsupported terms. Expand each SD target once, independently of group levels.
function _expand_prior_table(rows, groups)
    length(unique(groups)) == length(groups) || throw(ArgumentError("ambiguous random-intercept groups"))
    output = NamedTuple[]
    seen = Set{String}()
    for row in rows
        if row.class == "fixef"
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
        if row.class == "ranef_sd"
            drm_prior_sd_gamma(dpar=row.dpar, group=row.group, shape=row.arguments[2], rate=row.arguments[2] / row.arguments[1])
        elseif row.kind === :normal
            drm_prior_normal(dpar=row.dpar, coef=isempty(row.coef) ? "all" : row.coef,
                             intercept=isempty(row.coef) ? "exclude" : nothing,
                             mean=row.arguments[1], sd=row.arguments[2])
        else
            drm_prior_ridge(dpar=row.dpar, coef=isempty(row.coef) ? "all" : row.coef,
                            intercept=isempty(row.coef) ? "exclude" : nothing,
                            lambda=row.arguments[1])
        end
    end
end

# Validate the complete model before expanding broad selectors. The resolved
# preview supplies exactly the selected SD identities in model order.
function _table_priors_for_model(f, fam, data, table)
    rows = _parse_prior_table(table)
    resolved = resolve_priors(f, fam; data=data, priors=table)
    groups = String[r.group for r in resolved if hasproperty(r, :group)]
    specs = _prior_table_specs(rows, groups)
    fixed_rows = filter(r -> hasproperty(r, :precision), resolved)
    replay = AbstractDrmPrior[]
    for spec in specs
        if spec isa FixedPrior
            selected = filter(fixed_rows) do row
                row.dpar == spec.dpar && row.kind == spec.kind &&
                    row.mean == spec.mean && row.precision == spec.precision &&
                    row.constant == spec.constant &&
                    (spec.coef == "all" ? row.reason == :all : row.coef == spec.coef)
            end
            isempty(selected) && throw(ArgumentError("prior table fixed target could not be replayed"))
            append!(replay, [FixedPrior(row.kind, row.dpar, row.coef, nothing,
                                        row.mean, row.precision, row.constant) for row in selected])
        else
            push!(replay, spec)
        end
    end
    all(p -> p isa FixedPrior, replay) ? FixedPrior[p for p in replay] : replay
end
