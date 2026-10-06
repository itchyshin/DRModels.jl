# Exact prepared likelihood for one ordinal or nominal finite-state missing
# predictor.  This is a Julia implementation of the written probability and
# conditional-moment contract; it does not depend on the GPL R implementation.

"""Validated Gaussian-response model with one ordinal or categorical predictor."""
struct PreparedFiniteJointModel
    predictor::Symbol
    y::Vector{Union{Missing,Float64}}
    x::Vector{Union{Missing,Int}}
    X_mu_state::Array{Float64,3}
    Xsigma::Matrix{Float64}
    Xpredictor::Matrix{Float64}
    levels::Vector{String}
    variable::Symbol
    mu_names::Vector{String}
    sigma_names::Vector{String}
    predictor_names::Vector{String}
    original_row::Vector{Int}
    observed_y::BitVector
    observed_x::BitVector
end

"""Row-level posterior state summaries for a finite-state prepared fit."""
struct JointFiniteMissingMetadata
    predictor::Symbol
    variable::Symbol
    levels::Vector{String}
    original_row::Vector{Int}
    observed_y::BitVector
    observed_x::BitVector
    conditional_probabilities::Matrix{Float64}
    conditional_mean::Vector{Float64}
    conditional_variance::Vector{Float64}
    conditional_status::Vector{Symbol}
    all_rows::Int
    predictor_only_rows::Int
    uncertainty_status::Symbol
    optimizer_status::Symbol
    covariance_status::Symbol
end

"""Fitted exact prepared ordinal or categorical missing-predictor model."""
struct PreparedFiniteJointFit
    fit::DrmFit
    prepared::PreparedFiniteJointModel
    metadata::JointFiniteMissingMetadata
end

function _finite_joint_matrix(A, n::Int, label::AbstractString; allow_zero::Bool = false)
    size(A, 1) == n || throw(ArgumentError("$label must have $n rows"))
    (allow_zero || size(A, 2) > 0) || throw(ArgumentError("$label must have at least one column"))
    B = Matrix{Float64}(A)
    all(isfinite, B) || throw(ArgumentError("$label contains a non-finite value"))
    return B
end

function _finite_joint_state_array(A, n::Int, K::Int)
    ndims(A) == 3 && size(A, 1) == n && size(A, 2) == K && size(A, 3) > 0 ||
        throw(ArgumentError("X_mu_state must be an n-by-K-by-p array with p > 0"))
    B = Array{Float64,3}(A)
    all(isfinite, B) || throw(ArgumentError("X_mu_state contains a non-finite value"))
    return B
end

function _finite_joint_levels(levels)
    values = string.(collect(levels))
    length(values) >= 2 || throw(ArgumentError("levels must contain at least two states"))
    all(!isempty, values) && length(unique(values)) == length(values) ||
        throw(ArgumentError("levels must contain distinct nonempty labels"))
    return values
end

function _finite_joint_value(v, labels::Vector{String}, label::AbstractString)
    v === missing && return missing
    if v isa AbstractFloat
        isfinite(v) || throw(ArgumentError("$label contains a non-finite value"))
    elseif !(v isa Real || v isa AbstractString || v isa Symbol || v isa Bool)
        throw(ArgumentError("$label must contain missing values or declared state labels"))
    end
    idx = findfirst(==(string(v)), labels)
    idx === nothing && throw(ArgumentError("$label contains `$v`, which is not a declared level"))
    return idx
end

"""
    prepared_joint_model(y, x, X_mu_state, Xsigma, Xpredictor;
                         predictor, levels, variable, ...)

Build the prepared finite-state Gaussian-response joint model.  `X_mu_state`
contains the complete mean design for every observation-state combination in
state order; it must not be reconstructed from an expected score or a modal
state.  The prepared raw parameter order is `beta, delta, alpha, cutraw`.
For ordinal predictors `alpha` has `q` entries and `cutraw` has `K - 1`;
for categorical predictors `alpha` is level-major over the `K - 1`
non-baseline logits and no cutpoint block is present.
"""
function prepared_joint_model(y, x, X_mu_state::AbstractArray{<:Any,3}, Xsigma, Xpredictor;
                              predictor::Symbol,
                              levels,
                              variable::Symbol = :x,
                              mu_names = ["mu[$j]" for j in 1:size(X_mu_state, 3)],
                              sigma_names = ["sigma[$j]" for j in 1:size(Xsigma, 2)],
                              predictor_names = ["x[$j]" for j in 1:size(Xpredictor, 2)],
                              original_row = collect(eachindex(y)))
    predictor in (:ordinal, :categorical) ||
        throw(ArgumentError("finite prepared route requires predictor = :ordinal or :categorical"))
    n = length(y)
    n > 0 || throw(ArgumentError("prepared joint data must contain at least one row"))
    length(x) == n || throw(ArgumentError("x and y must have equal length"))
    labels = _finite_joint_levels(levels)
    K = length(labels)
    isempty(String(variable)) && throw(ArgumentError("variable must be a nonempty Symbol"))
    yy = Union{Missing,Float64}[_joint_float_or_missing(v, "y") for v in y]
    xx = Union{Missing,Int}[_finite_joint_value(v, labels, "x") for v in x]
    Xstate = _finite_joint_state_array(X_mu_state, n, K)
    Xσ = _joint_matrix(Xsigma, n, "Xsigma")
    Xp = _finite_joint_matrix(Xpredictor, n, "Xpredictor"; allow_zero = predictor === :ordinal)
    predictor === :categorical && size(Xp, 2) == 0 &&
        throw(ArgumentError("categorical predictors require a nonempty Xpredictor design"))
    if predictor === :ordinal && size(Xp, 2) > 0 && rank(hcat(ones(n), Xp)) == rank(Xp)
        throw(ArgumentError("ordinal Xpredictor must not span an intercept"))
    end
    raw_rows = collect(original_row)
    all(v -> v isa Integer && !(v isa Bool) && v > 0, raw_rows) ||
        throw(ArgumentError("original_row must contain positive integer entries"))
    rows = Int.(raw_rows)
    length(rows) == n || throw(ArgumentError("original_row must have $n entries"))
    length(unique(rows)) == n || throw(ArgumentError("original_row must contain unique positive entries"))
    return PreparedFiniteJointModel(
        predictor, yy, xx, Xstate, Xσ, Xp, labels, variable,
        _joint_names(mu_names, size(Xstate, 3), "mu_names"),
        _joint_names(sigma_names, size(Xσ, 2), "sigma_names"),
        _joint_names(predictor_names, size(Xp, 2), "predictor_names"),
        rows, BitVector(v !== missing for v in yy), BitVector(v !== missing for v in xx),
    )
end

_finite_joint_nbeta(model::PreparedFiniteJointModel) = size(model.X_mu_state, 3)
_finite_joint_ndelta(model::PreparedFiniteJointModel) = size(model.Xsigma, 2)
_finite_joint_npredictor(model::PreparedFiniteJointModel) = size(model.Xpredictor, 2)
_finite_joint_nalpha(model::PreparedFiniteJointModel) =
    model.predictor === :ordinal ? _finite_joint_npredictor(model) :
    _finite_joint_npredictor(model) * (length(model.levels) - 1)
_finite_joint_ncut(model::PreparedFiniteJointModel) =
    model.predictor === :ordinal ? length(model.levels) - 1 : 0
_finite_joint_ntheta(model::PreparedFiniteJointModel) = _finite_joint_nbeta(model) +
    _finite_joint_ndelta(model) + _finite_joint_nalpha(model) + _finite_joint_ncut(model)

function _finite_joint_ranges(model::PreparedFiniteJointModel)
    p, r, a = _finite_joint_nbeta(model), _finite_joint_ndelta(model), _finite_joint_nalpha(model)
    beta = 1:p
    delta = (p + 1):(p + r)
    alpha = (p + r + 1):(p + r + a)
    cutraw = (p + r + a + 1):_finite_joint_ntheta(model)
    return (; beta, delta, alpha, cutraw)
end

function _finite_joint_parameters(model::PreparedFiniteJointModel, theta::AbstractVector)
    length(theta) == _finite_joint_ntheta(model) ||
        throw(ArgumentError("theta has length $(length(theta)); expected $(_finite_joint_ntheta(model))"))
    all(isfinite, theta) || throw(ArgumentError("theta contains a non-finite value"))
    ranges = _finite_joint_ranges(model)
    return @view(theta[ranges.beta]), @view(theta[ranges.delta]),
           @view(theta[ranges.alpha]), @view(theta[ranges.cutraw])
end

function _finite_joint_state_means(model::PreparedFiniteJointModel, beta)
    T = promote_type(eltype(beta), Float64)
    n, K = length(model.y), length(model.levels)
    output = Matrix{T}(undef, n, K)
    @inbounds for i in 1:n, k in 1:K
        output[i, k] = dot(@view(model.X_mu_state[i, k, :]), beta)
    end
    return output
end

function _finite_joint_log1mexp(x)
    x < -log(2) ? log1p(-exp(x)) : log(-expm1(x))
end

"""Log ordinal probabilities from `c1, logspacing...`, without sigmoid subtraction."""
function _finite_joint_ordinal_logprobabilities(eta, cutraw)
    K = length(cutraw) + 1
    T = promote_type(typeof(eta), eltype(cutraw), Float64)
    output = Vector{T}(undef, K)
    previous = cutraw[1] - eta
    output[1] = _joint_logsigmoid(previous)
    @inbounds for k in 2:(K - 1)
        raw_spacing = cutraw[k]
        spacing = exp(raw_spacing)
        if raw_spacing < -36
            # exp(d) - 1 ≈ d on this scale.  Use the raw log spacing so
            # t = -1000 retains a finite log probability and AD derivative.
            output[k] = _joint_logsigmoid(previous) +
                        _joint_logsigmoid(-previous) + raw_spacing
        else
            # σ(a+d)-σ(a) = σ(a+d) σ(-a) [1-exp(-d)].  This form is
            # stable for both a narrow spacing and d = exp(10) or larger.
            output[k] = _joint_logsigmoid(previous + spacing) +
                        _joint_logsigmoid(-previous) + _finite_joint_log1mexp(-spacing)
        end
        previous += spacing
    end
    output[K] = _joint_logsigmoid(-previous)
    return output
end

function _finite_joint_logsumexp(values)
    m = maximum(values)
    return m + log(sum(exp(v - m) for v in values))
end

function _finite_joint_logprobabilities(model::PreparedFiniteJointModel, alpha, cutraw)
    n, K = length(model.y), length(model.levels)
    T = promote_type(eltype(alpha), eltype(cutraw), Float64)
    output = Matrix{T}(undef, n, K)
    q = _finite_joint_npredictor(model)
    @inbounds for i in 1:n
        if model.predictor === :ordinal
            eta = q == 0 ? zero(T) : dot(@view(model.Xpredictor[i, :]), @view(alpha[1:q]))
            output[i, :] .= _finite_joint_ordinal_logprobabilities(eta, cutraw)
        else
            output[i, 1] = zero(T)
            for k in 2:K
                first = (k - 2) * q + 1
                output[i, k] = dot(@view(model.Xpredictor[i, :]), @view(alpha[first:(first + q - 1)]))
            end
            normalizer = _finite_joint_logsumexp(@view output[i, :])
            for k in 1:K
                output[i, k] -= normalizer
            end
        end
    end
    return output
end

function _finite_joint_linear_predictors(model::PreparedFiniteJointModel, theta::AbstractVector)
    beta, delta, alpha, cutraw = _finite_joint_parameters(model, theta)
    return _finite_joint_state_means(model, beta), model.Xsigma * delta,
           _finite_joint_logprobabilities(model, alpha, cutraw)
end

"""Exact prepared-row likelihood for an ordinal or categorical missing predictor."""
function prepared_joint_rowloglik(model::PreparedFiniteJointModel, theta::AbstractVector)
    state_means, log_sigma, logprobabilities = _finite_joint_linear_predictors(model, theta)
    T = promote_type(eltype(theta), Float64)
    result = Vector{T}(undef, length(model.y))
    @inbounds for i in eachindex(result)
        if model.observed_x[i]
            k = model.x[i]
            result[i] = logprobabilities[i, k]
            model.observed_y[i] && (result[i] += _joint_logpdf_normal(model.y[i], state_means[i, k], exp(log_sigma[i])))
        elseif model.observed_y[i]
            weights = Vector{T}(undef, length(model.levels))
            for k in eachindex(weights)
                weights[k] = logprobabilities[i, k] +
                             _joint_logpdf_normal(model.y[i], state_means[i, k], exp(log_sigma[i]))
            end
            result[i] = _finite_joint_logsumexp(weights)
        else
            result[i] = zero(T)
        end
    end
    return result
end

prepared_joint_nll(model::PreparedFiniteJointModel, theta::AbstractVector) =
    -sum(prepared_joint_rowloglik(model, theta))

"""
    prepared_joint_conditional_moments(model, theta)

Return posterior state probabilities in original prepared-row order.  Ordinal
`mean` and `variance` are the conditional score mean and variance.  Categorical
`mean` is the first modal state code and its variance is unavailable (`NaN`).
"""
function prepared_joint_conditional_moments(model::PreparedFiniteJointModel, theta::AbstractVector)
    state_means, log_sigma, logprobabilities = _finite_joint_linear_predictors(model, theta)
    T = promote_type(eltype(theta), Float64)
    n, K = length(model.y), length(model.levels)
    probabilities = Matrix{T}(undef, n, K)
    mean = Vector{T}(undef, n)
    variance = Vector{T}(undef, n)
    status = Vector{Symbol}(undef, n)
    @inbounds for i in 1:n
        if model.observed_x[i]
            fill!(@view(probabilities[i, :]), zero(T))
            probabilities[i, model.x[i]] = one(T)
            mean[i] = T(model.x[i])
            variance[i] = model.predictor === :ordinal ? zero(T) : T(NaN)
            status[i] = :observed
        else
            logweight = if model.observed_y[i]
                [logprobabilities[i, k] + _joint_logpdf_normal(model.y[i], state_means[i, k], exp(log_sigma[i]))
                 for k in 1:K]
            else
                collect(@view(logprobabilities[i, :]))
            end
            normalizer = _finite_joint_logsumexp(logweight)
            for k in 1:K
                probabilities[i, k] = exp(logweight[k] - normalizer)
            end
            if model.predictor === :ordinal
                value = zero(T)
                for k in 1:K
                    value += T(k) * probabilities[i, k]
                end
                mean[i] = value
                variance[i] = sum((T(k) - value)^2 * probabilities[i, k] for k in 1:K)
                status[i] = model.observed_y[i] ? :ordinal_posterior : :predictor_only
            else
                best = 1
                for k in 2:K
                    probabilities[i, k] > probabilities[i, best] && (best = k)
                end
                mean[i] = T(best)
                variance[i] = T(NaN)
                status[i] = model.observed_y[i] ? :categorical_posterior : :predictor_only
            end
        end
    end
    return (probabilities = probabilities, mean = mean, variance = variance, status = status)
end

function prepared_joint_initial(model::PreparedFiniteJointModel)
    theta = zeros(Float64, _finite_joint_ntheta(model))
    ranges = _finite_joint_ranges(model)
    complete = findall(model.observed_y .& model.observed_x)
    if !isempty(complete)
        X = reduce(vcat, [reshape(@view(model.X_mu_state[i, model.x[i], :]), 1, :) for i in complete])
        response = Float64[model.y[i] for i in complete]
        beta = try X \ response catch; zeros(size(X, 2)) end
        theta[ranges.beta] .= beta
        residual = response - X * beta
        sd = length(residual) > 1 ? std(residual) : 1.0
        theta[first(ranges.delta)] = log(isfinite(sd) && sd > sqrt(eps(Float64)) ? sd : sqrt(eps(Float64)))
    end
    if model.predictor === :ordinal
        theta[first(ranges.cutraw)] = 0.0
        length(ranges.cutraw) > 1 && (theta[(first(ranges.cutraw) + 1):last(ranges.cutraw)] .= 0.0)
    end
    return theta
end

function _finite_joint_copy_model(model::PreparedFiniteJointModel)
    return PreparedFiniteJointModel(model.predictor, copy(model.y), copy(model.x), copy(model.X_mu_state),
        copy(model.Xsigma), copy(model.Xpredictor), copy(model.levels), model.variable,
        copy(model.mu_names), copy(model.sigma_names), copy(model.predictor_names), copy(model.original_row),
        copy(model.observed_y), copy(model.observed_x))
end

function _finite_joint_blocks(model::PreparedFiniteJointModel)
    ranges = _finite_joint_ranges(model)
    blocks = Pair{Symbol,UnitRange{Int}}[:mu => ranges.beta, :sigma => ranges.delta,
                                          Symbol(:mi_, model.variable) => ranges.alpha]
    alpha_names = model.predictor === :ordinal ? model.predictor_names :
        [string(level, ":", term) for level in model.levels[2:end] for term in model.predictor_names]
    names = Pair{Symbol,Vector{String}}[:mu => model.mu_names, :sigma => model.sigma_names,
                                         Symbol(:mi_, model.variable) => alpha_names]
    if model.predictor === :ordinal
        push!(blocks, Symbol(:rawcut_, model.variable) => ranges.cutraw)
        push!(names, Symbol(:rawcut_, model.variable) =>
              [k == 1 ? "cut1" : "log_spacing$k" for k in 1:length(ranges.cutraw)])
    end
    return blocks, names
end

"""Fit the finite-state joint likelihood by forward-mode AD and LBFGS."""
function fit_prepared_joint(model::PreparedFiniteJointModel;
                            initial::AbstractVector = prepared_joint_initial(model),
                            g_tol::Real = 1e-8)
    frozen = _finite_joint_copy_model(model)
    theta0 = Float64.(collect(initial))
    length(theta0) == _finite_joint_ntheta(frozen) || throw(ArgumentError("initial has wrong length"))
    all(isfinite, theta0) || throw(ArgumentError("initial contains a non-finite value"))
    isfinite(g_tol) && g_tol > 0 || throw(ArgumentError("g_tol must be finite and positive"))
    nll = theta -> prepared_joint_nll(frozen, theta)
    result = Optim.optimize(nll, theta0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    theta = Float64.(Optim.minimizer(result))
    nll_value = nll(theta)
    converged = drm_optim_converged(result) && isfinite(nll_value) && all(isfinite, theta)
    optimizer_status = converged ? :converged : :not_converged
    covariance, covariance_status = _joint_hessian_covariance(nll, theta)
    moments = prepared_joint_conditional_moments(frozen, theta)
    state_means = _finite_joint_state_means(frozen, @view theta[_finite_joint_ranges(frozen).beta])
    mu = [sum(moments.probabilities[i, k] * state_means[i, k] for k in 1:length(frozen.levels))
          for i in eachindex(frozen.y)]
    blocks, names = _finite_joint_blocks(frozen)
    ranges = _finite_joint_ranges(frozen)
    base = DrmFit(PreparedJointGaussian(), blocks, names, theta, covariance, -nll_value,
        count(frozen.observed_y), converged,
        Dict{Symbol,Vector{Float64}}(:mu => Float64.(mu)),
        Dict{Symbol,Vector{Float64}}(:mu => Float64[v === missing ? NaN : v for v in frozen.y]),
        Dict{Symbol,Vector{Float64}}(:sigma => exp.(frozen.Xsigma * theta[ranges.delta])))
    base = _withiterations(_withnll(base, nll), Optim.iterations(result))
    metadata = JointFiniteMissingMetadata(frozen.predictor, frozen.variable, copy(frozen.levels),
        copy(frozen.original_row), copy(frozen.observed_y), copy(frozen.observed_x),
        Float64.(moments.probabilities), Float64.(moments.mean), Float64.(moments.variance),
        copy(moments.status), length(frozen.y), count(i -> !frozen.observed_y[i] && frozen.observed_x[i], eachindex(frozen.y)),
        :not_computed, optimizer_status, covariance_status)
    return PreparedFiniteJointFit(base, frozen, metadata)
end

function joint_missing_summary(fit::PreparedFiniteJointFit)
    m = fit.metadata
    return (predictor = m.predictor, variable = m.variable, levels = copy(m.levels),
            original_row = copy(m.original_row), observed_y = copy(m.observed_y), observed_x = copy(m.observed_x),
            conditional_probabilities = copy(m.conditional_probabilities), conditional_mean = copy(m.conditional_mean),
            conditional_variance = copy(m.conditional_variance), conditional_status = copy(m.conditional_status),
            all_rows = m.all_rows, predictor_only_rows = m.predictor_only_rows,
            uncertainty_status = m.uncertainty_status, optimizer_status = m.optimizer_status,
            covariance_status = m.covariance_status)
end

function _finite_joint_imputation_table(fit::PreparedFiniteJointFit; rows = :missing, se::Bool = true)
    choice = rows isa Symbol ? rows : rows isa AbstractString ? Symbol(rows) : nothing
    choice in (:missing, :all) || throw(ArgumentError("imputed: rows must be :missing or :all"))
    model, metadata = fit.prepared, fit.metadata
    ids = choice === :all ? collect(eachindex(model.y)) : findall(.!model.observed_x)
    n = length(model.y)
    std_error = fill(NaN, n)
    status = fill(_joint_uncertainty_status(metadata.covariance_status), n)
    if metadata.covariance_status === :observed_information_inverse
        fill!(status, "ok")
        if se
            if model.predictor === :ordinal
                for i in findall(.!model.observed_x)
                    value = metadata.conditional_variance[i]
                    if isfinite(value) && value >= 0
                        std_error[i] = sqrt(value)
                    else
                        status[i] = "route_conditional_se_unavailable"
                    end
                end
            else
                for i in findall(.!model.observed_x)
                    status[i] = "route_conditional_se_unavailable"
                end
            end
        end
    end
    source = [model.observed_x[i] ? "observed" :
              model.predictor === :ordinal ? "conditional_expected_score" : "conditional_modal_category" for i in ids]
    sd = Union{Missing,Float64}[isfinite(std_error[i]) ? std_error[i] : missing for i in ids]
    return (variable = fill(String(model.variable), length(ids)), original_row = copy(model.original_row[ids]),
            model_row = ids, observed = copy(model.observed_x[ids]), estimate = copy(metadata.conditional_mean[ids]),
            std_error = sd, source = source, uncertainty_status = copy(status[ids]))
end

function imputed(fit::PreparedFiniteJointFit; variable = nothing, rows = :missing, se::Bool = true)
    requested = variable === nothing ? fit.prepared.variable : variable isa AbstractString ? Symbol(variable) : variable
    requested === fit.prepared.variable || throw(ArgumentError("imputed: no modelled predictor named `$requested`"))
    se isa Bool || throw(ArgumentError("imputed: `se` must be Bool"))
    return _finite_joint_imputation_table(fit; rows = rows, se = se)
end
