# joint_missing_frontend.jl — narrow formula admission for one or two missing predictors.
#
# The numerical likelihood lives in joint_missing_predictor.jl.  This file only
# turns a validated, native StatsModels formula into that prepared representation.

using StatsModels: DummyCoding, CategoricalTerm, ContinuousTerm, InteractionTerm,
    MatrixTerm, InterceptTerm, ContrastsMatrix

"""
    mi(x)

Marker for exactly one additive predictor whose missing values are modelled by
the joint missing-predictor frontend. It is valid only inside `@formula`; direct
evaluation is an error.
"""
mi(args...) = throw(ArgumentError("`mi()` is a formula marker; use it only inside `@formula(...)`"))

"""Validated controls for the joint missing-predictor formula admission."""
struct JointMissingControl
    response::Symbol
    predictor::Symbol
end

function _joint_control_value(value, name::AbstractString, allowed::Tuple)
    value isa Symbol || value isa AbstractString ||
        throw(ArgumentError("miss_control: `$name` must be one of $(join(string.(allowed), ", "))"))
    result = Symbol(value)
    result in allowed ||
        throw(ArgumentError("miss_control: `$name` must be one of $(join(string.(allowed), ", ")) (got `$value`)"))
    return result
end

"""
    miss_control(; response = "fail", predictor = "fail")

Declare missing-data handling for the joint frontend. Its currently verified
admission requires `predictor = "model"` and, when responses are missing,
`response = "include"`; the defaults
fail rather than silently changing a conventional `drm()` fit.
"""
function miss_control(; response = "fail", predictor = "fail")
    return JointMissingControl(
        _joint_control_value(response, "response", (:fail, :include)),
        _joint_control_value(predictor, "predictor", (:fail, :model)),
    )
end

"""Predictor-only multinomial-logit marker for finite categorical imputation."""
struct CategoricalLogit end

"""Formula and family for the one predictor distribution in a joint fit."""
struct JointImputeModel{F,L}
    formula::FormulaTerm
    family::F
    levels::L
end

# Source compatibility for callers that constructed the original two-field
# wrapper directly before finite-state predictor families were admitted.
JointImputeModel(formula::FormulaTerm, family) = JointImputeModel(formula, family, nothing)

"""
    impute_model(formula; family = Gaussian(), levels = nothing)

Wrap a predictor model when its distribution is not the default Gaussian.
The bounded frontend admits Gaussian, Bernoulli (`Binomial()`), ordinal
(`CumulativeLogit()`), and categorical (`CategoricalLogit()`) predictors.
`levels` is required for textual ordinal data, recommended for categorical
data, and rejected for Gaussian/Bernoulli predictors.
"""
function impute_model(formula::FormulaTerm; family = Gaussian(), levels = nothing)
    family isa Gaussian || family isa Binomial || family isa CumulativeLogit || family isa CategoricalLogit ||
        throw(ArgumentError("impute_model: `family` must be Gaussian(), Binomial(), CumulativeLogit(), or CategoricalLogit()"))
    (family isa Gaussian || family isa Binomial) && levels !== nothing &&
        throw(ArgumentError("impute_model: `levels` applies only to CumulativeLogit() or CategoricalLogit() predictor models"))
    if levels !== nothing
        labels = string.(collect(levels))
        length(labels) >= 3 || throw(ArgumentError("impute_model: finite-state predictors require at least three declared levels"))
        all(!isempty, labels) && length(unique(labels)) == length(labels) ||
            throw(ArgumentError("impute_model: `levels` must be distinct nonempty labels"))
        return JointImputeModel(formula, family, labels)
    end
    return JointImputeModel(formula, family, nothing)
end

"""User-facing wrapper around the prepared exact joint fit."""
struct JointDrmFit
    prepared::PreparedJointFit
    formula::DrmFormula
    variable::Symbol
end

"""
    JointTwoDrmFit

Formula-facing wrapper for the prepared two-independent-Gaussian-predictor
kernel. Raw coefficients and `vcov` preserve the kernel order
`beta, b1, b2, delta, alpha1, logtau1, alpha2, logtau2`; natural predictor
SDs are exposed only through `coef(fit, :sigma_mi_<variable>)`.
"""
struct JointTwoDrmFit
    prepared::PreparedTwoJointGaussianFit
    formula::DrmFormula
    variables::NTuple{2,Symbol}
end

# Applied schemas and state-coordinate metadata retained by a direct finite
# formula fit.  New data must be evaluated against these objects: building a
# schema from a singleton factor batch would otherwise change no-intercept and
# treatment coordinates before multiplication by the fitted beta block.
struct _JointFiniteStatePredictionPlan{F}
    formula::F
    variable::Symbol
    levels::Vector{String}
    predictor::Symbol
    permutation::Vector{Int}
    marker_columns::UnitRange{Int}
    full_marker::Bool
    names::Vector{String}
end

struct _JointFiniteSigmaPredictionPlan{F}
    formula::F
    names::Vector{String}
end

struct _JointFinitePredictionPlan{M,S}
    mean::M
    sigma::S
end

"""
    JointFiniteDrmFit

Formula-facing wrapper for one ordinal or categorical missing predictor.  Its
raw coefficient vector and covariance keep the kernel order `beta, delta,
alpha, cutraw`; ordinal raw cutpoints are deliberately separate from the
predictor-logit coefficients and `cutpoints(fit)` returns their natural scale.
Direct formula fits retain applied mean and sigma schemas for known-state
new-data prediction; the three-argument constructor remains available for
manually wrapped prepared fits but those fits cannot predict new data.
"""
struct JointFiniteDrmFit{P}
    prepared::PreparedFiniteJointFit
    formula::DrmFormula
    variable::Symbol
    prediction::P
end

JointFiniteDrmFit(prepared::PreparedFiniteJointFit, formula::DrmFormula, variable::Symbol) =
    JointFiniteDrmFit(prepared, formula, variable, nothing)

_joint_frontend_terms(rhs) = rhs isa Tuple ? collect(rhs) : Any[rhs]

function _joint_mi_count(term)
    if term isa Tuple || term isa AbstractVector
        return sum(_joint_mi_count, term)
    elseif term isa FunctionTerm
        return (term.f === mi ? 1 : 0) + sum(_joint_mi_count, term.args)
    elseif hasproperty(term, :terms)
        return sum(_joint_mi_count, getproperty(term, :terms))
    elseif hasproperty(term, :args)
        return sum(_joint_mi_count, getproperty(term, :args))
    end
    return 0
end

_joint_contains_mi(term) = _joint_mi_count(term) > 0

"""Whether any formula axis contains the `mi()` marker."""
_has_joint_mi(f::DrmFormula) = any(_joint_contains_mi(pair.second) for pair in f.forms)
_has_joint_mi(f::FormulaTerm) = _joint_contains_mi(f.rhs)

function _joint_term_symbols(term)
    if term isa Term
        return Symbol[term.sym]
    elseif term isa FunctionTerm
        return reduce(vcat, (_joint_term_symbols(arg) for arg in term.args); init = Symbol[])
    elseif hasproperty(term, :terms)
        return reduce(vcat, (_joint_term_symbols(arg) for arg in getproperty(term, :terms)); init = Symbol[])
    elseif hasproperty(term, :args)
        return reduce(vcat, (_joint_term_symbols(arg) for arg in getproperty(term, :args)); init = Symbol[])
    end
    return Symbol[]
end

_joint_frontend_missing(value) = ismissing(value) || (value isa AbstractFloat && isnan(value))

function _joint_require_complete_rhs(rhs, data, label::AbstractString)
    for name in unique(reduce(vcat, (_joint_term_symbols(t) for t in _joint_frontend_terms(rhs)); init = Symbol[]))
        values = _table_column(data, name)
        any(_joint_frontend_missing, values) &&
            throw(ArgumentError("joint missing-predictor frontend: `$name` has missing values in $label; only the marked predictor and response may be incomplete"))
    end
    return nothing
end

function _joint_require_exogenous(rhs, response::Symbol, predictor::Symbol, label::AbstractString)
    for term in _joint_frontend_terms(rhs), name in _joint_term_symbols(term)
        name in (response, predictor) && throw(ArgumentError(
            "joint missing-predictor frontend: $label must use exogenous covariates; `$name` is a modelled response or predictor"))
    end
    return nothing
end

function _joint_require_exogenous(rhs, response::Symbol, predictors::NTuple{2,Symbol}, label::AbstractString)
    for term in _joint_frontend_terms(rhs), name in _joint_term_symbols(term)
        (name === response || name in predictors) && throw(ArgumentError(
            "joint missing-predictor frontend: $label must use exogenous covariates; `$name` is a modelled response or predictor"))
    end
    return nothing
end

function _joint_validate_fixed_term(term, label::AbstractString)
    if term isa FunctionTerm
        forbidden = term.f === mi || term.f === (|) || term.f === meta_V ||
            term.f === phylo || term.f === relmat || term.f === animal || term.f === spatial ||
            nameof(term.f) === :offset
        forbidden && throw(ArgumentError("joint missing-predictor frontend: $label cannot contain mi(), offset(), random effects, structured effects, or meta_V()"))
        foreach(arg -> _joint_validate_fixed_term(arg, label), term.args)
    elseif hasproperty(term, :terms)
        foreach(arg -> _joint_validate_fixed_term(arg, label), getproperty(term, :terms))
    elseif !(term isa Term || term isa ConstantTerm)
        throw(ArgumentError("joint missing-predictor frontend: $label contains an unsupported formula term"))
    end
    return nothing
end

function _joint_validate_fixed_rhs(rhs, label::AbstractString)
    foreach(term -> _joint_validate_fixed_term(term, label), _joint_frontend_terms(rhs))
    return nothing
end

function _joint_mean_parts(rhs)
    terms = _joint_frontend_terms(rhs)
    total = sum(_joint_mi_count, terms)
    total == 1 || throw(ArgumentError("joint missing-predictor frontend: mean formula needs exactly one bare additive `mi(x)` term (found $total)"))
    marked = Any[]
    fixed = Any[]
    for term in terms
        if term isa FunctionTerm && term.f === mi
            length(term.args) == 1 && term.args[1] isa Term ||
                throw(ArgumentError("joint missing-predictor frontend: `mi()` must have one bare predictor name, e.g. `mi(x)`"))
            push!(marked, term)
        elseif _joint_contains_mi(term)
            throw(ArgumentError("joint missing-predictor frontend: `mi(x)` must be a bare additive term; nested or interacted markers are not implemented"))
        else
            _joint_validate_fixed_term(term, "mean formula")
            push!(fixed, term)
        end
    end
    length(marked) == 1 || throw(ArgumentError("joint missing-predictor frontend: multiple `mi()` terms are not implemented"))
    # StatsModels supplies an implicit intercept for `y ~ mi(x)`. Retain it
    # explicitly after removing the marker; `0 + mi(x)` remains zero-width and
    # is refused after design construction below.
    fixed_rhs = isempty(fixed) ? ConstantTerm(1) : length(fixed) == 1 ? fixed[1] : Tuple(fixed)
    return marked[1].args[1].sym, fixed_rhs
end

function _joint_two_mean_parts(rhs)
    terms = _joint_frontend_terms(rhs)
    total = sum(_joint_mi_count, terms)
    total == 2 || throw(ArgumentError("joint missing-predictor frontend: two-predictor mean formula needs exactly two bare additive `mi(x)` terms (found $total)"))
    variables = Symbol[]
    fixed = Any[]
    for term in terms
        if term isa FunctionTerm && term.f === mi
            length(term.args) == 1 && term.args[1] isa Term ||
                throw(ArgumentError("joint missing-predictor frontend: `mi()` must have one bare predictor name, e.g. `mi(x)`"))
            push!(variables, term.args[1].sym)
        elseif _joint_contains_mi(term)
            throw(ArgumentError("joint missing-predictor frontend: `mi(x)` must be a bare additive term; nested or interacted markers are not implemented"))
        else
            _joint_validate_fixed_term(term, "mean formula")
            push!(fixed, term)
        end
    end
    length(unique(variables)) == 2 ||
        throw(ArgumentError("joint missing-predictor frontend: the two `mi()` terms must mark distinct predictors"))
    fixed_rhs = isempty(fixed) ? ConstantTerm(1) : length(fixed) == 1 ? fixed[1] : Tuple(fixed)
    return (variables[1], variables[2]), fixed_rhs
end

function _joint_impute_spec(impute, variable::Symbol)
    impute isa NamedTuple ||
        throw(ArgumentError("joint missing-predictor frontend: `impute` must be a one-entry named tuple, e.g. `(x = @formula(x ~ z),)`"))
    names = Tuple(keys(impute))
    names == (variable,) ||
        throw(ArgumentError("joint missing-predictor frontend: `impute` must contain exactly the marked predictor `$variable`"))
    spec = getproperty(impute, variable)
    spec = spec isa FormulaTerm ? impute_model(spec) : spec
    spec isa JointImputeModel ||
        throw(ArgumentError("joint missing-predictor frontend: `$variable` must map to `@formula($variable ~ ...)` or `impute_model(...)`"))
    spec.formula.lhs isa Term && spec.formula.lhs.sym === variable ||
        throw(ArgumentError("joint missing-predictor frontend: imputation formula lhs must be `$variable`"))
    return spec
end

function _joint_two_impute_specs(impute, variables::NTuple{2,Symbol})
    impute isa NamedTuple ||
        throw(ArgumentError("joint missing-predictor frontend: `impute` must be a two-entry named tuple for the marked predictors"))
    names = Tuple(keys(impute))
    length(names) == 2 && Set(names) == Set(variables) ||
        throw(ArgumentError("joint missing-predictor frontend: `impute` must contain exactly `$(variables[1])` and `$(variables[2])`"))
    specs = ntuple(2) do j
        variable = variables[j]
        spec = getproperty(impute, variable)
        spec = spec isa FormulaTerm ? impute_model(spec) : spec
        spec isa JointImputeModel ||
            throw(ArgumentError("joint missing-predictor frontend: `$variable` must map to `@formula($variable ~ ...)` or `impute_model(...)`"))
        spec.formula.lhs isa Term && spec.formula.lhs.sym === variable ||
            throw(ArgumentError("joint missing-predictor frontend: imputation formula lhs must be `$variable`"))
        spec.family isa Gaussian ||
            throw(ArgumentError("joint missing-predictor frontend: two-predictor admission requires Gaussian() for both predictor models"))
        spec
    end
    return specs
end

function _joint_missing_vector(values)
    return Union{Missing,Float64}[value isa AbstractFloat && isnan(value) ? missing : Float64(value) for value in values]
end

function _joint_refuse_nondefault(method, algorithm, K, A, tree, coords, profile_ci, phylo_coupled, penalty, sparse)
    method === :ML || throw(ArgumentError("joint missing-predictor frontend: only `method = :ML` is implemented; REML is not available"))
    algorithm === :auto || throw(ArgumentError("joint missing-predictor frontend: `algorithm` is not forwarded; the verified prepared route uses LBFGS"))
    K === nothing && A === nothing && tree === nothing && coords === nothing &&
        !profile_ci && !phylo_coupled && penalty === nothing && sparse === nothing ||
        throw(ArgumentError("joint missing-predictor frontend: random/structured, profile, penalty, and sparse options are not implemented for this formula admission"))
    return nothing
end

_joint_finite_family(family) = family isa CumulativeLogit || family isa CategoricalLogit

function _joint_finite_label(value)
    value isa Real && isfinite(Float64(value)) && isinteger(value) && return string(Int(value))
    return string(value)
end

function _joint_finite_observed(values, variable::Symbol)
    observed = Any[]
    for value in values
        _joint_frontend_missing(value) && continue
        push!(observed, value)
    end
    isempty(observed) && throw(ArgumentError(
        "joint missing-predictor frontend: `$variable` needs observed finite-state predictor values"))
    return observed
end

function _joint_finite_numeric_levels(observed, variable::Symbol)
    # Numeric finite predictors use the native integer-code convention.  Do not
    # let non-integral or non-finite numbers slip into categorical lexical
    # labels (for example, `1.5` must not become the level "1.5").
    all(value -> value isa Real, observed) || return nothing
    all(value -> isfinite(Float64(value)) && isinteger(value), observed) ||
        throw(ArgumentError("joint missing-predictor frontend: inferred numeric `$variable` codes must be finite integers 1:K"))
    codes = Int.(observed)
    K = maximum(codes)
    unique_codes = sort(unique(codes))
    # `length(unique_codes) == K` proves exact 1:K coverage after the minimum
    # check, without materializing a potentially huge proposed range first.
    (K >= 3 && minimum(codes) == 1 && length(unique_codes) == K) ||
        throw(ArgumentError("joint missing-predictor frontend: inferred numeric `$variable` codes must be contiguous 1:K with K >= 3"))
    return string.(1:K)
end

function _joint_finite_levels(spec::JointImputeModel, values, variable::Symbol)
    observed = _joint_finite_observed(values, variable)
    labels = if spec.levels !== nothing
        copy(spec.levels)
    elseif spec.family isa CumulativeLogit
        inferred = _joint_finite_numeric_levels(observed, variable)
        inferred === nothing &&
            throw(ArgumentError("joint missing-predictor frontend: textual ordinal `$variable` requires explicit ordered `levels`"))
        inferred
    else
        # Categorical labels have no scientific ordering.  Sorting their printed
        # form gives a reproducible baseline for labels. Numeric codes instead
        # use the native 1:K convention, never lexical order such as 1,10,2.
        inferred = _joint_finite_numeric_levels(observed, variable)
        inferred === nothing ? sort(unique(_joint_finite_label.(observed))) : inferred
    end
    length(labels) >= 3 || throw(ArgumentError(
        "joint missing-predictor frontend: finite-state `$variable` requires K >= 3 levels"))
    observed_labels = _joint_finite_label.(observed)
    all(label -> label in labels, observed_labels) || throw(ArgumentError(
        "joint missing-predictor frontend: observed `$variable` values are not all in declared `levels`"))
    all(label -> any(==(label), observed_labels), labels) || throw(ArgumentError(
        "joint missing-predictor frontend: every declared `$variable` level must occur among observed predictor rows"))
    return labels
end

function _joint_finite_predictor_design(data, variable::Symbol, rhs)
    n = length(_table_column(data, variable))
    # `_design` uses its LHS only to obtain a StatsModels schema/model matrix.
    # The finite predictor can be textual or missing, so install a local numeric
    # dummy rather than coercing its observed state labels through the response
    # path.  The original table is never modified.
    design_data = _replace_table_column(data, variable, zeros(Float64, n))
    _, X, names = _design(variable, rhs, design_data)
    return X, names
end

function _joint_finite_polynomials(K::Int)
    scores = collect(1.0:1.0:K)
    centered = scores .- mean(scores)
    # Keep the polynomial domain fixed on [-1, 1]. Dividing by its norm makes
    # high powers artificially tiny before re-orthogonalization.
    scaled = centered ./ maximum(abs, centered)
    basis = Vector{Vector{Float64}}()
    push!(basis, ones(Float64, K) ./ sqrt(K))
    for degree in 1:(K - 1)
        vector = scaled .^ degree
        # Twice-repeated modified Gram--Schmidt retains orthogonality for
        # K > 3 without making the contrast signs depend on QR/BLAS details.
        for _ in 1:2, previous in basis
            vector .-= dot(previous, vector) .* previous
        end
        scale = norm(vector)
        scale > sqrt(eps(Float64)) || throw(ArgumentError(
            "joint missing-predictor frontend: could not construct ordinal polynomial contrasts"))
        vector ./= scale
        # Fix the QR/Gram-Schmidt sign convention independently of BLAS: the
        # leading (highest-score) coefficient is positive, giving .L/.Q signs
        # used by the generated parity payload.
        vector[end] < 0 && (vector .*= -1)
        push!(basis, vector)
    end
    return hcat(basis[2:end]...)
end

function _joint_finite_state_design(Xfixed::AbstractMatrix, variable::Symbol,
                                    labels::Vector{String}, predictor::Symbol;
                                    insertion::Integer = size(Xfixed, 2),
                                    full_states::Bool = false)
    n, p = size(Xfixed)
    0 <= insertion <= p || throw(ArgumentError("joint missing-predictor frontend: invalid finite-state mean insertion"))
    K = length(labels)
    nstate = full_states ? K : K - 1
    contrast = if full_states
        Matrix{Float64}(I, K, K)
    elseif predictor === :ordinal
        _joint_finite_polynomials(K)
    else
        C = zeros(Float64, K, K - 1)
        for k in 2:K
            C[k, k - 1] = 1.0
        end
        C
    end
    Xstate = Array{Float64}(undef, n, K, p + nstate)
    for i in 1:n, k in 1:K
        insertion > 0 && (Xstate[i, k, 1:insertion] .= Xfixed[i, 1:insertion])
        Xstate[i, k, (insertion + 1):(insertion + nstate)] .= contrast[k, :]
        insertion < p && (Xstate[i, k, (insertion + nstate + 1):end] .= Xfixed[i, (insertion + 1):end])
    end
    term_names = if full_states
        ["mi($(variable))$(label)" for label in labels]
    elseif predictor === :ordinal
        suffix = ["L", "Q", "C"]
        [j <= length(suffix) ? "mi($(variable)).$(suffix[j])" : "mi($(variable))^$j" for j in 1:(K - 1)]
    else
        ["mi($(variable))$(labels[k])" for k in 2:K]
    end
    return Xstate, term_names
end

function _joint_finite_state_rhs(rhs, variable::Symbol)
    terms = _joint_frontend_terms(rhs)
    rewritten = Any[]
    replaced = false
    for term in terms
        if term isa FunctionTerm && term.f === mi
            length(term.args) == 1 && term.args[1] isa Term && term.args[1].sym === variable ||
                throw(ArgumentError("joint missing-predictor frontend: finite-state state design needs one bare `mi($variable)` term"))
            push!(rewritten, Term(variable))
            replaced = true
        elseif _joint_contains_mi(term)
            throw(ArgumentError("joint missing-predictor frontend: finite-state state design cannot expand interacted or nested `mi()`"))
        else
            push!(rewritten, term)
        end
    end
    replaced || throw(ArgumentError("joint missing-predictor frontend: finite-state state design found no `mi($variable)` term"))
    return length(rewritten) == 1 ? rewritten[1] : Tuple(rewritten)
end

function _joint_finite_state_data(data, variable::Symbol, labels::Vector{String})
    columns = Tables.columntable(data)
    names = Tuple(Symbol(name) for name in keys(columns))
    variable in names || throw(ArgumentError("joint missing-predictor frontend: `$variable` is absent from state-design data"))
    n = length(_table_column(data, variable))
    K = length(labels)
    values = map(names) do name
        name === variable && return [labels[k] for _ in 1:n for k in 1:K]
        column = collect(getproperty(columns, name))
        length(column) == n || throw(ArgumentError("joint missing-predictor frontend: state-design columns have inconsistent lengths"))
        return repeat(column; inner = K)
    end
    return NamedTuple{names}(Tuple(values))
end

function _joint_finite_state_column_map(Xstate::Array{Float64,3}, K::Int)
    n, actual_K, p = size(Xstate)
    actual_K == K || throw(ArgumentError("joint missing-predictor frontend: state-design dimension disagrees with predictor levels"))
    state_for_column = Dict{Int,Int}()
    for column in 1:p
        any(state -> !all(@view(Xstate[:, state, column]) .== @view(Xstate[:, 1, column])), 2:K) || continue
        energy = [sum(abs2, @view(Xstate[:, state, column])) for state in 1:K]
        state = argmax(energy)
        maximum(energy) > 0 || throw(ArgumentError("joint missing-predictor frontend: finite-state marker column is empty"))
        for other in 1:K
            target = other == state ? 1.0 : 0.0
            maximum(abs, @view(Xstate[:, other, column]) .- target) <= 32 * eps(Float64) ||
                throw(ArgumentError("joint missing-predictor frontend: state design must contain a bare additive marker without state interactions"))
        end
        haskey(state_for_column, state) &&
            throw(ArgumentError("joint missing-predictor frontend: finite-state state design has duplicate marker columns"))
        state_for_column[state] = column
    end
    length(state_for_column) in (K - 1, K) ||
        throw(ArgumentError("joint missing-predictor frontend: finite-state state design has an unsupported marker rank"))
    expected = length(state_for_column) == K ? collect(1:K) : collect(2:K)
    all(haskey(state_for_column, state) for state in expected) ||
        throw(ArgumentError("joint missing-predictor frontend: finite-state marker contrasts do not match declared levels"))
    return [state_for_column[state] for state in expected], length(expected) == K
end

# StatsModels' display names separate a categorical variable and its level as
# `"a: level"`, while R's model matrix uses `"alevel"`; interaction components
# are then joined with `:`.  Rebuild only the structured categorical and
# interaction names from the already-applied schema rather than stripping
# punctuation from a rendered string.  Thus a genuine level such as
# `"level: high & dry"` is preserved verbatim.
_joint_finite_native_names(::InterceptTerm{true}) = ["(Intercept)"]
_joint_finite_native_names(::InterceptTerm{false}) = String[]
_joint_finite_native_names(term::ContinuousTerm) = [string(term.sym)]
_joint_finite_native_level_name(level) = level isa Bool ? (level ? "TRUE" : "FALSE") : string(level)
_joint_finite_native_names(term::CategoricalTerm) =
    [string(term.sym, _joint_finite_native_level_name(level)) for level in term.contrasts.coefnames]
_joint_finite_native_names(term::MatrixTerm) =
    reduce(vcat, (_joint_finite_native_names(part) for part in term.terms); init = String[])

function _joint_finite_native_names(term::InteractionTerm)
    result = [""]
    for part in term.terms
        result = [isempty(prefix) ? name : string(prefix, ":", name)
                  for name in _joint_finite_native_names(part) for prefix in result]
    end
    return result
end

_joint_finite_native_names(term) = String.(vec(coefnames(term)))

function _joint_finite_state_hints(state_rhs, data, variable::Symbol,
                                   labels::Vector{String})
    hints = Dict{Symbol,Any}(variable => DummyCoding(levels = labels))
    used = unique(reduce(vcat, (_joint_term_symbols(term) for term in _joint_frontend_terms(state_rhs)); init = Symbol[]))
    for name in used
        name === variable && continue
        values = _table_column(data, name)
        complete = [value for value in values if !_joint_frontend_missing(value)]
        if all(value -> value isa Bool, complete)
            # R treats logical fixed columns as a two-level factor even when
            # this data slice observes only FALSE or only TRUE.  Julia
            # Bool <: Real, so without this explicit two-level hint
            # StatsModels would silently use a continuous 0/1 column.
            # A `DummyCoding(levels = ...)` contrast hint still asks
            # StatsModels to derive levels from the singleton data column.
            # A concrete CategoricalTerm instead carries the native two-level
            # contrast matrix into schema application unchanged.
            hints[name] = CategoricalTerm(
                name, ContrastsMatrix(DummyCoding(), Bool[false, true]))
        elseif any(value -> !(value isa Real || value isa AbstractString || value isa Symbol), complete)
            throw(ArgumentError(
                "joint missing-predictor frontend: used fixed covariate `$name` has a non-plain categorical value; it needs a typed contrast contract (finite-state fixed designs currently admit numeric, plain strings, symbols, and Bool covariates)"))
        end
    end
    return hints
end

"""
    _joint_finite_native_state_design(formula, data, variable, levels, predictor)

Construct the row-then-state mean design by evaluating the complete mean formula
after replacing its bare `mi(variable)` marker with each declared state.  The
single expanded StatsModels schema chooses no-intercept factor coding for all
terms together: the first categorical term is full rank and later terms use
treatment contrasts.  Ordinal marker columns are then replaced by the fixed
polynomial contrasts unless that marker itself is the full-rank first factor.
"""
function _joint_finite_state_prediction_plan(f::DrmFormula, data, variable::Symbol,
                                              labels::Vector{String}, predictor::Symbol)
    predictor in (:ordinal, :categorical) ||
        throw(ArgumentError("joint missing-predictor frontend: unsupported finite-state predictor `$predictor`"))
    rhs = Dict(f.forms)[:mu]
    state_rhs = _joint_finite_state_rhs(rhs, variable)
    hints = _joint_finite_state_hints(state_rhs, data, variable, labels)
    expanded = _joint_finite_state_data(data, variable, labels)
    # The declared finite-state labels are semantic model levels: in
    # particular, the first label is the treatment baseline.  Plain `String`
    # data would let StatsModels sort levels lexically, which changes both the
    # baseline and the raw-coordinate meaning whenever declared order is not
    # lexical.  Apply the ordinary full-rank formula schema with an explicit
    # contrast hint for the marker and for formula-used Boolean fixed columns.
    # Other complete exogenous terms retain their normal StatsModels coding and
    # interaction behavior.
    raw_response = _table_column(expanded, f.response)
    y_response, observed_response = _coerce_response_column(raw_response)
    design_data = all(observed_response) ? expanded :
        _replace_table_column(expanded, f.response,
                              ifelse.(observed_response, y_response, 0.0))
    ft = FormulaTerm(Term(f.response), state_rhs)
    ft = apply_schema(ft, schema(ft, design_data, hints), StatisticalModel)
    _, X = modelcols(ft, design_data)
    Xflat = X isa AbstractMatrix ? Matrix{Float64}(X) :
        reshape(Float64.(collect(X)), :, 1)
    names = _joint_finite_native_names(ft.rhs)
    n = length(_table_column(data, variable))
    K = length(labels)
    size(Xflat, 1) == n * K || throw(ArgumentError("joint missing-predictor frontend: expanded state design has invalid row count"))
    p = size(Xflat, 2)
    length(names) == p || throw(ArgumentError(
        "joint missing-predictor frontend: generated finite-state names disagree with the state design"))
    Xstate = Array{Float64}(undef, n, K, p)
    for i in 1:n, state in 1:K
        Xstate[i, state, :] .= Xflat[(i - 1) * K + state, :]
    end
    marker_columns, full_marker = _joint_finite_state_column_map(Xstate, K)
    physical_marker_columns = sort(marker_columns)
    contiguous = physical_marker_columns == collect(first(physical_marker_columns):(first(physical_marker_columns) + length(physical_marker_columns) - 1))
    contiguous || throw(ArgumentError("joint missing-predictor frontend: finite-state marker columns are unexpectedly interleaved"))
    first_marker = first(physical_marker_columns)
    permutation = vcat(collect(1:(first_marker - 1)), marker_columns,
                       collect((last(physical_marker_columns) + 1):p))
    Xstate = Xstate[:, :, permutation]
    names = names[permutation]
    marker_range = first_marker:(first_marker + length(marker_columns) - 1)
    marker_names = if full_marker
        ["mi($(variable))$(label)" for label in labels]
    elseif predictor === :ordinal
        suffix = ["L", "Q", "C"]
        [column <= length(suffix) ? "mi($(variable)).$(suffix[column])" : "mi($(variable))^$column" for column in 1:(K - 1)]
    else
        ["mi($(variable))$(labels[state])" for state in 2:K]
    end
    if predictor === :ordinal && !full_marker
        contrast = _joint_finite_polynomials(K)
        for state in 1:K, column in 1:(K - 1)
            Xstate[:, state, marker_range[column]] .= contrast[state, column]
        end
    end
    names[marker_range] = marker_names
    plan = _JointFiniteStatePredictionPlan(ft, variable, copy(labels), predictor,
        permutation, marker_range, full_marker, copy(names))
    return Xstate, names, plan
end

function _joint_finite_native_state_design(f::DrmFormula, data, variable::Symbol,
                                           labels::Vector{String}, predictor::Symbol)
    Xstate, names, _ = _joint_finite_state_prediction_plan(f, data, variable, labels, predictor)
    return Xstate, names
end

# Unlike the finite mean design, sigma never contains the modelled predictor in
# this admission.  It still needs an applied schema so a singleton factor in
# `newdata` uses the fitted coordinate system rather than learning a new one.
function _joint_finite_sigma_prediction_plan(response::Symbol, rhs, data)
    raw_response = _table_column(data, response)
    y_response, observed_response = _coerce_response_column(raw_response)
    design_data = all(observed_response) ? data :
        _replace_table_column(data, response,
                              ifelse.(observed_response, y_response, 0.0))
    ft = FormulaTerm(Term(response), rhs)
    ft = apply_schema(ft, schema(ft, design_data), StatisticalModel)
    _, X = modelcols(ft, design_data)
    Xmat = X isa AbstractMatrix ? Matrix{Float64}(X) :
        reshape(Float64.(collect(X)), :, 1)
    names = String.(vec(coefnames(ft.rhs)))
    return Xmat, names, _JointFiniteSigmaPredictionPlan(ft, copy(names))
end

function _joint_finite_mean_insertion(f::DrmFormula, data)
    rhs = Dict(f.forms)[:mu]
    terms = _joint_frontend_terms(rhs)
    marker = findfirst(term -> term isa FunctionTerm && term.f === mi, terms)
    marker === nothing && throw(ArgumentError("joint missing-predictor frontend: no finite `mi()` marker found"))
    prefix = Any[]
    # Keep the original explicit intercept convention even when a constant sits
    # after the marker; StatsModels otherwise reinstates an implicit intercept.
    append!(prefix, (term for term in terms if term isa ConstantTerm))
    append!(prefix, (term for term in terms[1:(marker - 1)] if !(term isa ConstantTerm)))
    prefix_rhs = isempty(prefix) ? ConstantTerm(1) : length(prefix) == 1 ? prefix[1] : Tuple(prefix)
    _, Xprefix, _ = _design(f.response, prefix_rhs, data)
    return size(Xprefix, 2)
end

function _fit_finite_joint_formula(f::DrmFormula, data, variable::Symbol, fixed_mu, rhs_sigma,
                                   spec::JointImputeModel;
                                   missing::JointMissingControl,
                                   g_tol::Real)
    predictor = spec.family isa CumulativeLogit ? :ordinal : :categorical
    raw_x = _table_column(data, variable)
    labels = _joint_finite_levels(spec, raw_x, variable)
    observed_x = BitVector(!_joint_frontend_missing(value) for value in raw_x)
    any(.!observed_x) || throw(ArgumentError(
        "joint missing-predictor frontend: finite-state `$variable` admission requires at least one missing predictor row"))

    y, _, _ = _design(f.response, fixed_mu, data)
    missing.response === :fail && any(isnan, y) &&
        throw(ArgumentError("joint missing-predictor frontend: response has missing values; use `miss_control(response = \"include\", predictor = \"model\")`"))
    Xsigma, sigma_names, sigma_plan = _joint_finite_sigma_prediction_plan(f.response, rhs_sigma, data)
    Xpredictor, predictor_names = _joint_finite_predictor_design(data, variable, spec.formula.rhs)
    if predictor === :ordinal
        intercept = findfirst(==("(Intercept)"), predictor_names)
        if intercept !== nothing
            keep = setdiff(1:length(predictor_names), intercept)
            Xpredictor = Xpredictor[:, keep]
            predictor_names = predictor_names[keep]
        end
    end
    Xstate, mu_names, mean_plan = _joint_finite_state_prediction_plan(f, data, variable, labels, predictor)
    pstate = size(Xstate, 3)
    pstate > 0 || throw(ArgumentError("joint missing-predictor frontend: finite-state mean design is empty"))
    n = length(y)
    size(Xstate, 1) == n && size(Xsigma, 1) == n && size(Xpredictor, 1) == n && length(raw_x) == n ||
        throw(ArgumentError("joint missing-predictor frontend: response and design rows must agree"))
    observed_count = count(observed_x)
    q = size(Xpredictor, 2)
    needed = predictor === :ordinal ? q + length(labels) - 1 : q * (length(labels) - 1)
    observed_count > needed || throw(ArgumentError(
        "joint missing-predictor frontend: `$variable` has $observed_count observed rows, insufficient for its $needed finite-state predictor parameters"))

    kernel_x = Union{Missing,String}[observed_x[i] ? _joint_finite_label(raw_x[i]) : Base.missing for i in eachindex(raw_x)]
    prepared = prepared_joint_model(_joint_missing_vector(y), kernel_x, Xstate, Xsigma, Xpredictor;
        predictor = predictor, levels = labels, variable = variable,
        mu_names = mu_names, sigma_names = sigma_names,
        predictor_names = predictor_names, original_row = collect(1:n))
    fit = fit_prepared_joint(prepared; g_tol = g_tol)
    prediction = _JointFinitePredictionPlan(mean_plan, sigma_plan)
    return JointFiniteDrmFit(fit, f, variable, prediction)
end

function _fit_two_joint_formula(f::DrmFormula, data;
                                impute,
                                missing,
                                g_tol::Real,
                                method::Symbol,
                                algorithm::Symbol,
                                K,
                                A,
                                tree,
                                coords,
                                profile_ci::Bool,
                                phylo_coupled::Bool,
                                penalty,
                                sparse)
    f.response2 === nothing || throw(ArgumentError("joint missing-predictor frontend: bivariate responses are not implemented"))
    _joint_refuse_nondefault(method, algorithm, K, A, tree, coords, profile_ci, phylo_coupled, penalty, sparse)
    missing isa JointMissingControl ||
        throw(ArgumentError("joint missing-predictor frontend: `missing` must be created with miss_control(...)"))
    missing.response in (:fail, :include) ||
        throw(ArgumentError("joint missing-predictor frontend: response control must be :fail or :include"))
    missing.predictor === :model ||
        throw(ArgumentError("joint missing-predictor frontend: use `missing = miss_control(predictor = \"model\")`"))
    isfinite(g_tol) && g_tol > 0 || throw(ArgumentError("joint missing-predictor frontend: `g_tol` must be finite and positive"))

    rhs = Dict(f.forms)
    Set(keys(rhs)) == Set((:mu, :sigma)) ||
        throw(ArgumentError("joint missing-predictor frontend: only `mu` and `sigma` formulas are implemented"))
    variables, fixed_mu = _joint_two_mean_parts(rhs[:mu])
    f.response in variables && throw(ArgumentError(
        "joint missing-predictor frontend: response and modelled predictors must be different variables"))
    _joint_contains_mi(rhs[:sigma]) &&
        throw(ArgumentError("joint missing-predictor frontend: `mi()` is valid on the mean formula only, not `sigma`"))
    _joint_require_exogenous(fixed_mu, f.response, variables, "the mean design")
    _joint_require_exogenous(rhs[:sigma], f.response, variables, "the sigma design")
    _joint_validate_fixed_rhs(rhs[:sigma], "sigma formula")
    _joint_require_complete_rhs(fixed_mu, data, "the mean design")
    _joint_require_complete_rhs(rhs[:sigma], data, "the sigma design")

    specs = _joint_two_impute_specs(impute, variables)
    for j in 1:2
        rhsj = specs[j].formula.rhs
        _joint_require_exogenous(rhsj, f.response, variables, "predictor design for `$(variables[j])`")
        _joint_validate_fixed_rhs(rhsj, "predictor model for `$(variables[j])`")
        _joint_require_complete_rhs(rhsj, data, "predictor design for `$(variables[j])`")
    end

    y, Xmu, mu_names = _design(f.response, fixed_mu, data)
    missing.response === :fail && any(isnan, y) &&
        throw(ArgumentError("joint missing-predictor frontend: response has missing values; use `miss_control(response = \"include\", predictor = \"model\")`"))
    _, Xsigma, sigma_names = _design(f.response, rhs[:sigma], data)
    x1, Xpredictor1, predictor_names1 = _design(variables[1], specs[1].formula.rhs, data)
    x2, Xpredictor2, predictor_names2 = _design(variables[2], specs[2].formula.rhs, data)
    size(Xmu, 2) > 0 ||
        throw(ArgumentError("joint missing-predictor frontend: removing the two `mi()` terms leaves a zero-width mean design, which the prepared engine does not support"))
    n = length(y)
    length(x1) == n && length(x2) == n && size(Xmu, 1) == n && size(Xsigma, 1) == n &&
        size(Xpredictor1, 1) == n && size(Xpredictor2, 1) == n ||
        throw(ArgumentError("joint missing-predictor frontend: response and design rows must agree"))
    x = hcat(_joint_missing_vector(x1), _joint_missing_vector(x2))
    prepared = prepared_joint_model(_joint_missing_vector(y), x, Xmu, Xsigma,
                                    (Xpredictor1, Xpredictor2);
                                    predictor = :gaussian, predictor_variables = variables,
                                    mu_names = mu_names, sigma_names = sigma_names,
                                    predictor_names = (predictor_names1, predictor_names2),
                                    original_row = collect(1:n))
    return JointTwoDrmFit(fit_prepared_joint(prepared; g_tol = g_tol), f, variables)
end

"""
    _fit_joint_formula(f, data; impute, missing, ...)

Build a prepared exact joint model from one Gaussian response formula containing
one or two bare additive `mi(x)` mean terms. This deliberately partial frontend
admits fixed Gaussian ML responses with complete remaining covariates. The
one-predictor route permits Gaussian or Bernoulli predictor models; the
two-predictor route requires two independent Gaussian predictor models. It
neither drops rows nor fills modelled predictors before fitting.
"""
function _fit_joint_formula(f::DrmFormula, data;
                            impute = nothing,
                            missing = miss_control(),
                            g_tol::Real = 1e-8,
                            method::Symbol = :ML,
                            algorithm::Symbol = :auto,
                            K = nothing,
                            A = nothing,
                            tree = nothing,
                            coords = nothing,
                            profile_ci::Bool = false,
                            phylo_coupled::Bool = false,
                            penalty = nothing,
                            sparse = nothing)
    rhs_for_route = Dict(f.forms)
    if haskey(rhs_for_route, :mu) && _joint_mi_count(rhs_for_route[:mu]) == 2
        return _fit_two_joint_formula(f, data; impute = impute, missing = missing,
            g_tol = g_tol, method = method, algorithm = algorithm, K = K, A = A,
            tree = tree, coords = coords, profile_ci = profile_ci,
            phylo_coupled = phylo_coupled, penalty = penalty, sparse = sparse)
    end
    f.response2 === nothing || throw(ArgumentError("joint missing-predictor frontend: bivariate responses are not implemented"))
    _joint_refuse_nondefault(method, algorithm, K, A, tree, coords, profile_ci, phylo_coupled, penalty, sparse)
    missing isa JointMissingControl ||
        throw(ArgumentError("joint missing-predictor frontend: `missing` must be created with miss_control(...)"))
    missing.response in (:fail, :include) ||
        throw(ArgumentError("joint missing-predictor frontend: response control must be :fail or :include"))
    missing.predictor === :model ||
        throw(ArgumentError("joint missing-predictor frontend: use `missing = miss_control(predictor = \"model\")`"))
    isfinite(g_tol) && g_tol > 0 || throw(ArgumentError("joint missing-predictor frontend: `g_tol` must be finite and positive"))

    rhs = Dict(f.forms)
    Set(keys(rhs)) == Set((:mu, :sigma)) ||
        throw(ArgumentError("joint missing-predictor frontend: only `mu` and `sigma` formulas are implemented"))
    variable, fixed_mu = _joint_mean_parts(rhs[:mu])
    variable !== f.response || throw(ArgumentError(
        "joint missing-predictor frontend: response and modelled predictor must be different variables"))
    _joint_require_exogenous(fixed_mu, f.response, variable, "the mean design")
    _joint_require_exogenous(rhs[:sigma], f.response, variable, "the sigma design")
    _joint_contains_mi(rhs[:sigma]) &&
        throw(ArgumentError("joint missing-predictor frontend: `mi()` is valid on the mean formula only, not `sigma`"))
    _joint_validate_fixed_rhs(rhs[:sigma], "sigma formula")
    _joint_require_complete_rhs(fixed_mu, data, "the mean design")
    _joint_require_complete_rhs(rhs[:sigma], data, "the sigma design")

    spec = _joint_impute_spec(impute, variable)
    _joint_require_exogenous(spec.formula.rhs, f.response, variable, "the predictor design")
    _joint_validate_fixed_rhs(spec.formula.rhs, "the predictor model")
    _joint_require_complete_rhs(spec.formula.rhs, data, "the predictor design")
    if _joint_finite_family(spec.family)
        return _fit_finite_joint_formula(f, data, variable, fixed_mu, rhs[:sigma], spec;
            missing = missing, g_tol = g_tol)
    end
    predictor = spec.family isa Gaussian ? :gaussian : spec.family isa Binomial ? :bernoulli :
        throw(ArgumentError("joint missing-predictor frontend: predictor family must be Gaussian() or Binomial()"))
    if predictor === :bernoulli
        observed_labels = _joint_finite_label.(_joint_finite_observed(_table_column(data, variable), variable))
        length(unique(observed_labels)) <= 2 || throw(ArgumentError(
            "joint missing-predictor frontend: Bernoulli `$variable` permits at most two observed predictor levels"))
    end

    y, Xmu, mu_names = _design(f.response, fixed_mu, data)
    missing.response === :fail && any(isnan, y) &&
        throw(ArgumentError("joint missing-predictor frontend: response has missing values; use `miss_control(response = \"include\", predictor = \"model\")`"))
    _, Xsigma, sigma_names = _design(f.response, rhs[:sigma], data)
    x, Xpredictor, predictor_names = _design(variable, spec.formula.rhs, data)
    size(Xmu, 2) > 0 ||
        throw(ArgumentError("joint missing-predictor frontend: `0 + mi($variable)` leaves a zero-width mean design, which the prepared engine does not support"))
    n = length(y)
    length(x) == n && size(Xmu, 1) == n && size(Xsigma, 1) == n && size(Xpredictor, 1) == n ||
        throw(ArgumentError("joint missing-predictor frontend: response and design rows must agree"))
    prepared = prepared_joint_model(_joint_missing_vector(y), _joint_missing_vector(x), Xmu, Xsigma, Xpredictor;
                                    predictor = predictor, mu_names = mu_names, sigma_names = sigma_names,
                                    predictor_names = predictor_names, original_row = collect(1:n))
    return JointDrmFit(fit_prepared_joint(prepared; g_tol = g_tol), f, variable)
end

"""All raw fitted coefficients in the prepared-engine order."""
coef(fit::JointDrmFit) = coef(fit.prepared.fit)

"""
    coef(fit::JointDrmFit, parameter)

`:mi_<variable>` maps to the internal predictor-model block. For a Gaussian predictor, `:sigma_mi_<variable>` returns
the natural predictor SD, while `:logsd_mi_<variable>` returns its raw
log-covariance coordinate.
"""
function coef(fit::JointDrmFit, parameter::Symbol)
    mi_name = Symbol(:mi_, fit.variable)
    raw_sd_name = Symbol(:logsd_mi_, fit.variable)
    natural_sd_name = Symbol(:sigma_mi_, fit.variable)
    if parameter === mi_name
        return coef(fit.prepared.fit, :mi_x)
    elseif parameter === raw_sd_name
        return coef(fit.prepared.fit, :logsd_mi_x)
    elseif parameter === natural_sd_name
        fit.prepared.prepared.predictor === :gaussian ||
            throw(ArgumentError("joint missing-predictor frontend: Bernoulli `$((fit.variable))` has no predictor SD"))
        return exp.(coef(fit.prepared.fit, :logsd_mi_x))
    end
    return coef(fit.prepared.fit, parameter)
end

vcov(fit::JointDrmFit) = vcov(fit.prepared.fit)
loglik(fit::JointDrmFit) = loglik(fit.prepared.fit)
nobs(fit::JointDrmFit) = nobs(fit.prepared.fit)
is_converged(fit::JointDrmFit) = is_converged(fit.prepared.fit)
convergence_status(fit::JointDrmFit) = convergence_status(fit.prepared.fit)
_converged_nondegenerate(fit::JointDrmFit) = _converged_nondegenerate(fit.prepared.fit)
niterations(fit::JointDrmFit) = niterations(fit.prepared.fit)
family(::JointDrmFit) = Gaussian()

"""Conditional imputation for the marker variable in a formula-based joint fit."""
function imputed(fit::JointDrmFit; variable = nothing, rows = :missing, se::Bool = true)
    requested = variable isa AbstractString ? Symbol(variable) : variable
    requested === nothing || requested isa Symbol ||
        throw(ArgumentError("joint missing-predictor frontend: `variable` must be a Symbol, String, or nothing"))
    requested === nothing || requested === fit.variable ||
        throw(ArgumentError("joint missing-predictor frontend: this fit models `$(fit.variable)`, not `$variable`"))
    return _imputed_joint(fit.prepared, fit.variable; rows = rows, se = se)
end

"""All raw fitted coefficients in the two-predictor prepared-engine order."""
coef(fit::JointTwoDrmFit) = coef(fit.prepared.fit)

function coef(fit::JointTwoDrmFit, parameter::Symbol)
    for variable in fit.variables
        mi_name = Symbol(:mi_, variable)
        raw_sd_name = Symbol(:logsd_mi_, variable)
        natural_sd_name = Symbol(:sigma_mi_, variable)
        if parameter === mi_name || parameter === raw_sd_name
            return coef(fit.prepared.fit, parameter)
        elseif parameter === natural_sd_name
            return exp.(coef(fit.prepared.fit, raw_sd_name))
        end
    end
    return coef(fit.prepared.fit, parameter)
end

# Keep the covariance on raw coordinates. In particular, logtau coordinates
# are not delta-transformed merely because `coef(..., :sigma_mi_*)` is natural.
vcov(fit::JointTwoDrmFit) = vcov(fit.prepared.fit)
loglik(fit::JointTwoDrmFit) = loglik(fit.prepared.fit)
nobs(fit::JointTwoDrmFit) = nobs(fit.prepared.fit)
is_converged(fit::JointTwoDrmFit) = is_converged(fit.prepared.fit)
convergence_status(fit::JointTwoDrmFit) = convergence_status(fit.prepared.fit)
_converged_nondegenerate(fit::JointTwoDrmFit) = _converged_nondegenerate(fit.prepared.fit)
niterations(fit::JointTwoDrmFit) = niterations(fit.prepared.fit)
family(::JointTwoDrmFit) = Gaussian()

"""Conditional imputation for one explicitly selected two-predictor marker variable."""
function imputed(fit::JointTwoDrmFit; variable = nothing, rows = :missing, se::Bool = true)
    requested = variable isa AbstractString ? Symbol(variable) : variable
    requested isa Symbol ||
        throw(ArgumentError("joint missing-predictor frontend: two-predictor fits require `variable`"))
    requested in fit.variables ||
        throw(ArgumentError("joint missing-predictor frontend: this fit models `$(fit.variables[1])` and `$(fit.variables[2])`, not `$variable`"))
    return imputed(fit.prepared; variable = requested, rows = rows, se = se)
end

"""All raw fitted finite-state coefficients in `beta, delta, alpha, cutraw` order."""
coef(fit::JointFiniteDrmFit) = coef(fit.prepared.fit)

function coef(fit::JointFiniteDrmFit, parameter::Symbol)
    parameter === Symbol(:mi_, fit.variable) && return coef(fit.prepared.fit, parameter)
    parameter === Symbol(:rawcut_, fit.variable) || return coef(fit.prepared.fit, parameter)
    fit.prepared.prepared.predictor === :ordinal ||
        throw(ArgumentError("joint missing-predictor frontend: categorical `$(fit.variable)` has no ordinal cutpoints"))
    return coef(fit.prepared.fit, parameter)
end

"""
    cutpoints(fit::JointFiniteDrmFit)

Natural cumulative-logit cutpoints for an ordinal finite predictor.  This is a
separate transformed view of raw `coef(fit, :rawcut_<variable>)`; `vcov(fit)`
continues to use the raw kernel coordinates.
"""
function cutpoints(fit::JointFiniteDrmFit)
    model = fit.prepared.prepared
    model.predictor === :ordinal ||
        throw(ArgumentError("joint missing-predictor frontend: categorical `$(fit.variable)` has no ordered cutpoints"))
    raw = coef(fit.prepared.fit, Symbol(:rawcut_, fit.variable))
    result = similar(raw)
    result[1] = raw[1]
    for j in 2:length(raw)
        result[j] = result[j - 1] + exp(raw[j])
    end
    return result
end

# Raw covariance is intentionally not delta-transformed for `cutpoints`.
vcov(fit::JointFiniteDrmFit) = vcov(fit.prepared.fit)
loglik(fit::JointFiniteDrmFit) = loglik(fit.prepared.fit)
nobs(fit::JointFiniteDrmFit) = nobs(fit.prepared.fit)
is_converged(fit::JointFiniteDrmFit) = is_converged(fit.prepared.fit)
convergence_status(fit::JointFiniteDrmFit) = convergence_status(fit.prepared.fit)
_converged_nondegenerate(fit::JointFiniteDrmFit) = _converged_nondegenerate(fit.prepared.fit)
niterations(fit::JointFiniteDrmFit) = niterations(fit.prepared.fit)
family(::JointFiniteDrmFit) = Gaussian()
fitted(fit::JointFiniteDrmFit) = fitted(fit.prepared.fit)
joint_missing_summary(fit::JointFiniteDrmFit) = joint_missing_summary(fit.prepared)

function _joint_finite_retained_prediction_plan(fit::JointFiniteDrmFit)
    plan = fit.prediction
    plan isa _JointFinitePredictionPlan || throw(ArgumentError(
        "joint missing-predictor frontend: this manually wrapped finite fit did not retain applied schemas for `newdata` prediction"))
    return plan
end

function _joint_finite_prediction_table(newdata)
    columns = try
        Tables.columntable(newdata)
    catch err
        throw(ArgumentError(
            "joint missing-predictor frontend: `newdata` must be a Tables-compatible column table ($(sprint(showerror, err)))"))
    end
    names = Tuple(Symbol(name) for name in keys(columns))
    isempty(names) && throw(ArgumentError(
        "joint missing-predictor frontend: `newdata` must contain at least one column"))
    n = try
        length(getproperty(columns, names[1]))
    catch err
        throw(ArgumentError(
            "joint missing-predictor frontend: every `newdata` column must be a vector ($(sprint(showerror, err)))"))
    end
    for name in names
        length(getproperty(columns, name)) == n || throw(ArgumentError(
            "joint missing-predictor frontend: `newdata` columns have inconsistent lengths"))
    end
    return NamedTuple{names}(Tuple(getproperty(columns, name) for name in names)), n
end

function _joint_finite_prediction_response_data(data::NamedTuple, response::Symbol, n::Int)
    names = Tuple(keys(data))
    response in names && return _replace_table_column(data, response, zeros(Float64, n))
    return merge(data, NamedTuple{(response,)}((zeros(Float64, n),)))
end

function _joint_finite_known_state_rows(data::NamedTuple,
                                        plan::_JointFiniteStatePredictionPlan)
    variable = plan.variable
    variable in keys(data) || throw(ArgumentError(
        "joint missing-predictor frontend: mu prediction requires known `$variable` values in `newdata`"))
    raw = getproperty(data, variable)
    states = Vector{Int}(undef, length(raw))
    for i in eachindex(raw)
        value = raw[i]
        _joint_frontend_missing(value) && throw(ArgumentError(
            "joint missing-predictor frontend: mu prediction requires nonmissing `$variable` values; new-data imputation is not implemented"))
        state = findfirst(==(_joint_finite_label(value)), plan.levels)
        state === nothing && throw(ArgumentError(
            "joint missing-predictor frontend: `$variable` value `$(value)` is not one of the fitted finite-state levels"))
        states[i] = state
    end
    return states
end

function _joint_finite_prediction_matrix(plan::_JointFiniteStatePredictionPlan,
                                         data::NamedTuple, response::Symbol)
    states = _joint_finite_known_state_rows(data, plan)
    n = length(states)
    expanded = _joint_finite_state_data(data, plan.variable, plan.levels)
    design_data = _joint_finite_prediction_response_data(expanded, response, n * length(plan.levels))
    Xflat = try
        _, values = modelcols(plan.formula, design_data)
        values isa AbstractMatrix ? Matrix{Float64}(values) :
            reshape(Float64.(collect(values)), :, 1)
    catch err
        throw(ArgumentError(
            "joint missing-predictor frontend: `newdata` does not match the retained mean schema ($(sprint(showerror, err)))"))
    end
    K = length(plan.levels)
    p = length(plan.names)
    size(Xflat, 1) == n * K || throw(ArgumentError(
        "joint missing-predictor frontend: retained mean schema produced an invalid row count"))
    size(Xflat, 2) == p || throw(ArgumentError(
        "joint missing-predictor frontend: retained mean schema changed its coefficient coordinates"))
    length(plan.permutation) == p || throw(ArgumentError(
        "joint missing-predictor frontend: retained mean coordinate permutation is invalid"))
    Xflat = Xflat[:, plan.permutation]
    Xstate = Array{Float64}(undef, n, K, p)
    for i in 1:n, state in 1:K
        Xstate[i, state, :] .= Xflat[(i - 1) * K + state, :]
    end
    if plan.predictor === :ordinal && !plan.full_marker
        contrast = _joint_finite_polynomials(K)
        length(plan.marker_columns) == K - 1 || throw(ArgumentError(
            "joint missing-predictor frontend: retained ordinal marker coordinates are invalid"))
        for state in 1:K, column in 1:(K - 1)
            Xstate[:, state, plan.marker_columns[column]] .= contrast[state, column]
        end
    end
    X = Matrix{Float64}(undef, n, p)
    for i in 1:n
        X[i, :] .= Xstate[i, states[i], :]
    end
    all(isfinite, X) || throw(ArgumentError(
        "joint missing-predictor frontend: retained mean schema produced a non-finite new-data design"))
    return X
end

function _joint_finite_prediction_matrix(plan::_JointFiniteSigmaPredictionPlan,
                                         data::NamedTuple, response::Symbol)
    n = length(first(values(data)))
    design_data = _joint_finite_prediction_response_data(data, response, n)
    X = try
        _, values = modelcols(plan.formula, design_data)
        values isa AbstractMatrix ? Matrix{Float64}(values) :
            reshape(Float64.(collect(values)), :, 1)
    catch err
        throw(ArgumentError(
            "joint missing-predictor frontend: `newdata` does not match the retained sigma schema ($(sprint(showerror, err)))"))
    end
    size(X, 1) == n || throw(ArgumentError(
        "joint missing-predictor frontend: retained sigma schema produced an invalid row count"))
    size(X, 2) == length(plan.names) || throw(ArgumentError(
        "joint missing-predictor frontend: retained sigma schema changed its coefficient coordinates"))
    all(isfinite, X) || throw(ArgumentError(
        "joint missing-predictor frontend: retained sigma schema produced a non-finite new-data design"))
    return X
end

function _joint_finite_prediction_se(X::AbstractMatrix, V::AbstractMatrix,
                                     range::UnitRange{Int})
    size(V, 1) == size(V, 2) || throw(ArgumentError(
        "joint missing-predictor frontend: covariance has invalid dimensions"))
    last(range) <= size(V, 1) || throw(ArgumentError(
        "joint missing-predictor frontend: covariance lacks the requested parameter block"))
    all(isfinite, V) || throw(ArgumentError(
        "joint missing-predictor frontend: covariance is unavailable for new-data standard errors"))
    isapprox(V, transpose(V); rtol = 1e-8, atol = 0.0) || throw(ArgumentError(
        "joint missing-predictor frontend: covariance must be symmetric for new-data standard errors"))
    try
        cholesky(Symmetric(V); check = true)
    catch err
        throw(ArgumentError(
            "joint missing-predictor frontend: covariance must be positive definite for new-data standard errors ($(sprint(showerror, err)))"))
    end
    Vblock = view(V, range, range)
    size(X, 2) == length(range) || throw(ArgumentError(
        "joint missing-predictor frontend: prediction design and covariance block disagree"))
    result = Vector{Float64}(undef, size(X, 1))
    for i in axes(X, 1)
        row = view(X, i, :)
        variance = dot(row, Vblock, row)
        isfinite(variance) || throw(ArgumentError(
            "joint missing-predictor frontend: new-data prediction variance is non-finite"))
        # A symmetric observed-information inverse can acquire a negative
        # quadratic only through floating-point cancellation at this scale.
        # Preserve that roundoff allowance, but never turn a material negative
        # variance into a public zero standard error.
        scale = dot(abs.(row), abs.(Vblock) * abs.(row))
        isfinite(scale) || throw(ArgumentError(
            "joint missing-predictor frontend: prediction-variance scale is non-finite"))
        variance < -128 * eps(Float64) * scale && throw(ArgumentError(
            "joint missing-predictor frontend: covariance gives a materially negative prediction variance"))
        result[i] = sqrt(max(0.0, variance))
    end
    all(isfinite, result) || throw(ArgumentError(
        "joint missing-predictor frontend: new-data standard errors are non-finite"))
    return result
end

"""
    predict(fit::JointFiniteDrmFit, newdata; dpar = :mu, type = :response, se = false)

Known-state plug-in prediction for a formula-fitted ordinal or categorical
missing predictor.  Mean predictions require every new-data marker value to be
an observed fitted level and are evaluated with the retained formula schema;
they never condition on a supplied new response.  Sigma predictions use their
own retained schema and therefore do not require the marker column.  Missing
new-data marker values and unknown levels remain outside this bounded route.
"""
function predict(fit::JointFiniteDrmFit, newdata;
                 dpar = :mu, type = :response, se = false)
    dpar isa Symbol && dpar in (:mu, :sigma) || throw(ArgumentError(
        "joint missing-predictor frontend: `dpar` must be :mu or :sigma"))
    type isa Symbol && type in (:response, :link) || throw(ArgumentError(
        "joint missing-predictor frontend: `type` must be :response or :link"))
    se isa Bool || throw(ArgumentError(
        "joint missing-predictor frontend: `se` must be Bool"))
    retained = _joint_finite_retained_prediction_plan(fit)
    data, _ = _joint_finite_prediction_table(newdata)
    plan = dpar === :mu ? retained.mean : retained.sigma
    X = _joint_finite_prediction_matrix(plan, data, fit.formula.response)
    η = X * coef(fit.prepared.fit, dpar)
    all(isfinite, η) || throw(ArgumentError(
        "joint missing-predictor frontend: new-data linear predictor is non-finite"))
    prediction = dpar === :sigma && type === :response ? exp.(η) : η
    all(isfinite, prediction) || throw(ArgumentError(
        "joint missing-predictor frontend: new-data prediction is non-finite"))
    se || return prediction

    fit.prepared.metadata.covariance_status === :observed_information_inverse || throw(ArgumentError(
        "joint missing-predictor frontend: covariance status does not support new-data standard errors"))
    V = vcov(fit.prepared.fit)
    range = _block_range(fit.prepared.fit, dpar)
    link_se = _joint_finite_prediction_se(X, V, range)
    result_se = dpar === :sigma && type === :response ? exp.(η) .* link_se : link_se
    all(isfinite, result_se) || throw(ArgumentError(
        "joint missing-predictor frontend: new-data standard errors are non-finite"))
    return (; prediction, se = result_se)
end

"""Conditional finite-state imputation for the marker variable in a formula-based fit."""
function imputed(fit::JointFiniteDrmFit; variable = nothing, rows = :missing, se::Bool = true)
    requested = variable isa AbstractString ? Symbol(variable) : variable
    requested === nothing || requested isa Symbol ||
        throw(ArgumentError("joint missing-predictor frontend: `variable` must be a Symbol, String, or nothing"))
    requested === nothing || requested === fit.variable ||
        throw(ArgumentError("joint missing-predictor frontend: this fit models `$(fit.variable)`, not `$variable`"))
    return imputed(fit.prepared; variable = fit.variable, rows = rows, se = se)
end
