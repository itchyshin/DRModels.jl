# joint_missing_predictor.jl — prepared exact prototypes for one missing predictor.
#
# This is deliberately a Julia-only prepared-data layer.  It neither parses R
# formulas nor changes the formula front end; its purpose is to make the row
# likelihood and conditional-moment contract independently testable.

"""Tag for the prepared Gaussian-predictor prototype."""
struct PreparedJointGaussian end

"""Tag for the prepared Bernoulli-predictor prototype."""
struct PreparedJointBernoulli end

"""
    PreparedJointModel

Validated, formula-free input for a Gaussian response with one potentially
missing Gaussian or Bernoulli predictor. `Xmu` excludes that predictor; its
coefficient is the separate `b` coordinate in the fixed parameter order
`beta, b, delta, alpha, kappa` (with `kappa` only for a Gaussian predictor).
"""
struct PreparedJointModel
    predictor::Symbol
    y::Vector{Union{Missing,Float64}}
    x::Vector{Union{Missing,Float64}}
    Xmu::Matrix{Float64}
    Xsigma::Matrix{Float64}
    Xpredictor::Matrix{Float64}
    mu_names::Vector{String}
    sigma_names::Vector{String}
    predictor_names::Vector{String}
    original_row::Vector{Int}
    row_state::Vector{Symbol}
    observed_y::BitVector
    observed_x::BitVector
end

"""Original-row and conditional-moment metadata for a prepared joint fit."""
struct JointMissingMetadata
    predictor::Symbol
    original_row::Vector{Int}
    row_state::Vector{Symbol}
    observed_y::BitVector
    observed_x::BitVector
    conditional_mean::Vector{Float64}
    conditional_variance::Vector{Float64}
    conditional_status::Vector{Symbol}
    all_rows::Int
    predictor_only_rows::Int
    uncertainty_status::Symbol
    optimizer_status::Symbol
    covariance_status::Symbol
end

"""
    PreparedJointFit

Result wrapper for the prepared prototype. `fit` is a normal `DrmFit`; the
separate `metadata` field prevents missing-data information from being confused
with random effects. Generic `confint`, `simulate`, and `AIC` forwarding is
intentionally absent.
"""
struct PreparedJointFit
    fit::DrmFit
    prepared::PreparedJointModel
    metadata::JointMissingMetadata
end

"""
    joint_missing_summary(fit::PreparedJointFit)

Return a copy-safe named summary of prepared-row conditional moments and their
data-only masks. This summary contains no `std_error`: its moments condition on fitted
parameters. Use [`imputed`](@ref) for native-shaped uncertainty summaries.
"""
function joint_missing_summary(fit::PreparedJointFit)
    m = fit.metadata
    return (predictor = m.predictor,
            original_row = copy(m.original_row),
            row_state = copy(m.row_state),
            observed_y = copy(m.observed_y),
            observed_x = copy(m.observed_x),
            conditional_mean = copy(m.conditional_mean),
            conditional_variance = copy(m.conditional_variance),
            conditional_status = copy(m.conditional_status),
            all_rows = m.all_rows,
            predictor_only_rows = m.predictor_only_rows,
            uncertainty_status = m.uncertainty_status,
            optimizer_status = m.optimizer_status,
            covariance_status = m.covariance_status)
end

_joint_missing(v) = v === missing

function _joint_float_or_missing(v, label::AbstractString)
    v === missing && return missing
    v isa Real || throw(ArgumentError("$label must contain only real values or missing"))
    value = Float64(v)
    isfinite(value) || throw(ArgumentError("$label contains a non-finite value"))
    return value
end

function _joint_matrix(A, n::Int, label::AbstractString)
    size(A, 1) == n || throw(ArgumentError("$label must have $n rows"))
    size(A, 2) > 0 || throw(ArgumentError("$label must have at least one column"))
    B = Matrix{Float64}(A)
    all(isfinite, B) || throw(ArgumentError("$label contains a non-finite value"))
    return B
end

function _joint_names(names, p::Int, label::AbstractString)
    values = String.(collect(names))
    length(values) == p || throw(ArgumentError("$label must have $p entries"))
    length(unique(values)) == p || throw(ArgumentError("$label must be unique"))
    return values
end

"""
    prepared_joint_model(y, x, Xmu, Xsigma, Xpredictor;
                         predictor, mu_names, sigma_names, predictor_names,
                         original_row)

Construct a complete-design, original-row-preserving prepared model. Missingness
is permitted only in `y` and `x`; masks are data properties, never parameters.
"""
function prepared_joint_model(y, x, Xmu, Xsigma, Xpredictor;
                              predictor::Symbol,
                              mu_names = ["mu[$j]" for j in 1:size(Xmu, 2)],
                              sigma_names = ["sigma[$j]" for j in 1:size(Xsigma, 2)],
                              predictor_names = ["x[$j]" for j in 1:size(Xpredictor, 2)],
                              original_row = collect(eachindex(y)))
    predictor in (:gaussian, :bernoulli) ||
        throw(ArgumentError("predictor must be :gaussian or :bernoulli"))
    n = length(y)
    n > 0 || throw(ArgumentError("prepared joint data must contain at least one row"))
    length(x) == n || throw(ArgumentError("x and y must have equal length"))
    yy = Union{Missing,Float64}[_joint_float_or_missing(v, "y") for v in y]
    xx = Union{Missing,Float64}[_joint_float_or_missing(v, "x") for v in x]
    if predictor === :bernoulli
        all(v -> v === missing || v == 0.0 || v == 1.0, xx) ||
            throw(ArgumentError("Bernoulli x must contain only 0, 1, or missing"))
    end
    Xμ = _joint_matrix(Xmu, n, "Xmu")
    Xσ = _joint_matrix(Xsigma, n, "Xsigma")
    Xx = _joint_matrix(Xpredictor, n, "Xpredictor")
    rows = Int.(collect(original_row))
    length(rows) == n || throw(ArgumentError("original_row must have $n entries"))
    all(>(0), rows) || throw(ArgumentError("original_row must be positive"))
    length(unique(rows)) == n || throw(ArgumentError("original_row must be unique"))
    obs_y = BitVector(v !== missing for v in yy)
    obs_x = BitVector(v !== missing for v in xx)
    state = Vector{Symbol}(undef, n)
    @inbounds for i in 1:n
        state[i] = obs_x[i] ? (obs_y[i] ? :complete : :x_observed_y_missing) :
                   (obs_y[i] ? :x_missing_y_observed : :both_missing)
    end
    mu_labels = _joint_names(mu_names, size(Xμ, 2), "mu_names")
    "mi(x)" in mu_labels && throw(ArgumentError("mu_names may not contain reserved appended name mi(x)"))
    return PreparedJointModel(predictor, yy, xx, Xμ, Xσ, Xx,
                              mu_labels,
                              _joint_names(sigma_names, size(Xσ, 2), "sigma_names"),
                              _joint_names(predictor_names, size(Xx, 2), "predictor_names"),
                              rows, state, obs_y, obs_x)
end

_joint_nbeta(model::PreparedJointModel) = size(model.Xmu, 2)
_joint_ndelta(model::PreparedJointModel) = size(model.Xsigma, 2)
_joint_nalpha(model::PreparedJointModel) = size(model.Xpredictor, 2)
_joint_ntheta(model::PreparedJointModel) = _joint_nbeta(model) + 1 + _joint_ndelta(model) +
    _joint_nalpha(model) + (model.predictor === :gaussian ? 1 : 0)

function _joint_parameters(model::PreparedJointModel, theta::AbstractVector)
    length(theta) == _joint_ntheta(model) ||
        throw(ArgumentError("theta has length $(length(theta)); expected $(_joint_ntheta(model))"))
    all(isfinite, theta) || throw(ArgumentError("theta contains a non-finite value"))
    p = _joint_nbeta(model)
    r = _joint_ndelta(model)
    q = _joint_nalpha(model)
    beta = @view theta[1:p]
    b = theta[p + 1]
    delta = @view theta[p+2:p+1+r]
    alpha = @view theta[p+2+r:p+1+r+q]
    kappa = model.predictor === :gaussian ? theta[end] : nothing
    return beta, b, delta, alpha, kappa
end

_joint_logpdf_normal(y, mean, sd) = -0.5 * log(2π) - log(sd) - 0.5 * ((y - mean) / sd)^2

function _joint_log1pexp(z)
    z >= zero(z) ? z + log1p(exp(-z)) : log1p(exp(z))
end

_joint_logsigmoid(z) = z >= zero(z) ? -log1p(exp(-z)) : z - log1p(exp(z))
_joint_logistic(z) = z >= zero(z) ? inv(one(z) + exp(-z)) : exp(z) / (one(z) + exp(z))
_joint_logsumexp2(a, b) = a >= b ? a + log1p(exp(b - a)) : b + log1p(exp(a - b))

function _joint_linear_predictors(model::PreparedJointModel, theta::AbstractVector)
    beta, b, delta, alpha, kappa = _joint_parameters(model, theta)
    return model.Xmu * beta, b, model.Xsigma * delta, model.Xpredictor * alpha, kappa
end

"""
    prepared_joint_rowloglik(model, theta)

Return exact marginal log-likelihood contributions in original prepared-row
order. Missing `x` is integrated analytically for a Gaussian predictor and by
finite logit-stable enumeration for a Bernoulli predictor.
"""
function prepared_joint_rowloglik(model::PreparedJointModel, theta::AbstractVector)
    a, b, log_sigma, eta_x, kappa = _joint_linear_predictors(model, theta)
    T = promote_type(eltype(theta), Float64)
    result = Vector{T}(undef, length(model.y))
    tau = model.predictor === :gaussian ? exp(kappa) : nothing
    @inbounds for i in eachindex(result)
        sigma = exp(log_sigma[i])
        state = model.row_state[i]
        if model.predictor === :gaussian
            if state === :complete
                result[i] = _joint_logpdf_normal(model.x[i], eta_x[i], tau) +
                    _joint_logpdf_normal(model.y[i], a[i] + b * model.x[i], sigma)
            elseif state === :x_missing_y_observed
                marginal_sd = hypot(sigma, b * tau)
                result[i] = _joint_logpdf_normal(model.y[i], a[i] + b * eta_x[i], marginal_sd)
            elseif state === :x_observed_y_missing
                result[i] = _joint_logpdf_normal(model.x[i], eta_x[i], tau)
            else
                result[i] = zero(T)
            end
        else
            if state === :complete
                logpx = model.x[i] == 1.0 ? _joint_logsigmoid(eta_x[i]) : _joint_logsigmoid(-eta_x[i])
                result[i] = logpx + _joint_logpdf_normal(model.y[i], a[i] + b * model.x[i], sigma)
            elseif state === :x_missing_y_observed
                # Keep full weights for the likelihood: adding a large
                # analytic log-odds to logw0 can cancel its normalizer.
                logw0 = _joint_logsigmoid(-eta_x[i]) + _joint_logpdf_normal(model.y[i], a[i], sigma)
                logw1 = _joint_logsigmoid(eta_x[i]) + _joint_logpdf_normal(model.y[i], a[i] + b, sigma)
                result[i] = _joint_logsumexp2(logw0, logw1)
            elseif state === :x_observed_y_missing
                result[i] = model.x[i] == 1.0 ? _joint_logsigmoid(eta_x[i]) : _joint_logsigmoid(-eta_x[i])
            else
                result[i] = zero(T)
            end
        end
    end
    return result
end

prepared_joint_nll(model::PreparedJointModel, theta::AbstractVector) = -sum(prepared_joint_rowloglik(model, theta))

"""
    prepared_joint_conditional_moments(model, theta)

Return named vectors `(mean, variance, status)` for `x` conditional on the
available row data. The returned values retain the original prepared-row order.
"""
function prepared_joint_conditional_moments(model::PreparedJointModel, theta::AbstractVector)
    a, b, log_sigma, eta_x, kappa = _joint_linear_predictors(model, theta)
    T = promote_type(eltype(theta), Float64)
    n = length(model.y)
    xmean = Vector{T}(undef, n)
    xvar = Vector{T}(undef, n)
    status = Vector{Symbol}(undef, n)
    tau = model.predictor === :gaussian ? exp(kappa) : nothing
    @inbounds for i in 1:n
        state = model.row_state[i]
        sigma = exp(log_sigma[i])
        if state === :complete || state === :x_observed_y_missing
            xmean[i] = model.x[i]
            xvar[i] = zero(T)
            status[i] = :observed
        elseif model.predictor === :gaussian
            if state === :x_missing_y_observed
                marginal_sd = hypot(sigma, b * tau)
                # The two ratios keep the conditional mean/SD stable when the
                # two scales have very different magnitudes.
                gain = (b * tau / marginal_sd) * (tau / marginal_sd)
                prior_weight = (sigma / marginal_sd)^2
                xmean[i] = prior_weight * eta_x[i] + gain * (model.y[i] - a[i])
                posterior_sd = tau * (sigma / marginal_sd)
                xvar[i] = posterior_sd * posterior_sd
                status[i] = :gaussian_posterior
            else
                xmean[i] = eta_x[i]
                xvar[i] = tau * tau
                status[i] = :predictor_only
            end
        else
            p = _joint_logistic(eta_x[i])
            if state === :x_missing_y_observed
                logodds = eta_x[i] + (b / sigma) * ((model.y[i] - a[i] - b / 2) / sigma)
                q = _joint_logistic(logodds)
                xmean[i] = q
                q0 = _joint_logistic(-logodds)
                xvar[i] = q0 * q
                status[i] = :bernoulli_posterior
            else
                xmean[i] = p
                p0 = _joint_logistic(-eta_x[i])
                xvar[i] = p0 * p
                status[i] = :predictor_only
            end
        end
    end
    return (mean = xmean, variance = xvar, status = status)
end

"""A finite, neutral initial parameter vector in the documented fixed order."""
function prepared_joint_initial(model::PreparedJointModel)
    theta = zeros(Float64, _joint_ntheta(model))
    observed = findall(model.observed_y .& model.observed_x)
    if !isempty(observed)
        X = hcat(model.Xmu[observed, :], Float64[model.x[i] for i in observed])
        response = Float64[model.y[i] for i in observed]
        coeff = try
            X \ response
        catch
            zeros(size(X, 2))
        end
        theta[1:length(coeff)] .= coeff
        residual = response - X * coeff
        residual_sd = length(residual) > 1 ? std(residual) : 1.0
        theta[_joint_nbeta(model)+2] = log(isfinite(residual_sd) && residual_sd > sqrt(eps(Float64)) ?
                                            residual_sd : sqrt(eps(Float64)))
    end
    return theta
end

function _joint_hessian_covariance(nll, theta::Vector{Float64})
    H = try
        ForwardDiff.hessian(nll, theta)
    catch
        return fill(NaN, length(theta), length(theta)), :hessian_unavailable
    end
    if !all(isfinite, H)
        return fill(NaN, length(theta), length(theta)), :hessian_unavailable
    end
    Hs = Matrix(Symmetric((H + H') / 2))
    factor = cholesky(Symmetric(Hs); check = false)
    if !issuccess(factor)
        return fill(NaN, length(theta), length(theta)), :hessian_not_positive_definite
    end
    covariance = try
        Matrix(factor \ I)
    catch
        fill(NaN, length(theta), length(theta))
    end
    all(isfinite, covariance) || return fill(NaN, length(theta), length(theta)), :hessian_unavailable
    return Matrix(Symmetric((covariance + covariance') / 2)), :observed_information_inverse
end

function _joint_copy_model(model::PreparedJointModel)
    return PreparedJointModel(model.predictor, copy(model.y), copy(model.x),
                              copy(model.Xmu), copy(model.Xsigma), copy(model.Xpredictor),
                              copy(model.mu_names), copy(model.sigma_names), copy(model.predictor_names),
                              copy(model.original_row), copy(model.row_state),
                              copy(model.observed_y), copy(model.observed_x))
end

"""
    fit_prepared_joint(model; initial = prepared_joint_initial(model), g_tol = 1e-8)

Fit the exact prepared prototype by forward-mode AD and LBFGS. Optimizer and
covariance status are recorded separately; no failed or non-finite calculation
is converted into a finite likelihood.
"""
function fit_prepared_joint(model::PreparedJointModel;
                            initial::AbstractVector = prepared_joint_initial(model),
                            g_tol::Real = 1e-8)
    model = _joint_copy_model(model)
    theta0 = Float64.(collect(initial))
    length(theta0) == _joint_ntheta(model) ||
        throw(ArgumentError("initial has wrong length"))
    all(isfinite, theta0) || throw(ArgumentError("initial contains a non-finite value"))
    isfinite(g_tol) && g_tol > 0 || throw(ArgumentError("g_tol must be finite and positive"))
    nll = theta -> prepared_joint_nll(model, theta)
    result = Optim.optimize(nll, theta0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    theta = Float64.(Optim.minimizer(result))
    nll_value = nll(theta)
    converged = drm_optim_converged(result) && isfinite(nll_value) && all(isfinite, theta)
    optimizer_status = converged ? :converged : :not_converged
    covariance, covariance_status = _joint_hessian_covariance(nll, theta)
    moments = prepared_joint_conditional_moments(model, theta)
    family = model.predictor === :gaussian ? PreparedJointGaussian() : PreparedJointBernoulli()
    p = _joint_nbeta(model)
    r = _joint_ndelta(model)
    q = _joint_nalpha(model)
    delta_range = (p + 2):(p + 1 + r)
    alpha_range = (p + 2 + r):(p + 1 + r + q)
    blocks = Pair{Symbol,UnitRange{Int}}[:mu => (1:(p + 1)), :sigma => delta_range,
                                          :mi_x => alpha_range]
    names = Pair{Symbol,Vector{String}}[:mu => [model.mu_names; "mi(x)"],
                                         :sigma => model.sigma_names,
                                         :mi_x => model.predictor_names]
    if model.predictor === :gaussian
        push!(blocks, :logsd_mi_x => (length(theta):length(theta)))
        push!(names, :logsd_mi_x => ["log_sd"])
    end
    yobs = Float64[v === missing ? NaN : v for v in model.y]
    means = Dict{Symbol,Vector{Float64}}(:mu => Float64.(model.Xmu * theta[1:p] .+ theta[p + 1] .* moments.mean))
    obs = Dict{Symbol,Vector{Float64}}(:mu => yobs)
    scales = Dict{Symbol,Vector{Float64}}(:sigma => exp.(model.Xsigma * theta[delta_range]))
    base = DrmFit(family, blocks, names, theta, covariance, -nll_value,
                  count(model.observed_y), converged, means, obs, scales)
    base = _withiterations(_withnll(base, nll), Optim.iterations(result))
    metadata = JointMissingMetadata(model.predictor, copy(model.original_row), copy(model.row_state),
                                    copy(model.observed_y), copy(model.observed_x), Float64.(moments.mean),
                                    Float64.(moments.variance), copy(moments.status), length(model.y),
                                    count(==( :x_observed_y_missing), model.row_state), :not_computed,
                                    optimizer_status, covariance_status)
    return PreparedJointFit(base, model, metadata)
end
