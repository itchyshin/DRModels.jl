# Exact prepared likelihood for two conditionally independent Gaussian missing predictors.
# This is a separate type so the verified one-predictor prepared contract remains unchanged.

"""Validated, formula-free Gaussian-response model with two independent Gaussian predictors."""
struct PreparedTwoJointGaussianModel
    y::Vector{Union{Missing,Float64}}
    x::Matrix{Union{Missing,Float64}}
    Xmu::Matrix{Float64}
    Xsigma::Matrix{Float64}
    Xpredictor::NTuple{2,Matrix{Float64}}
    predictor_variables::NTuple{2,Symbol}
    mu_names::Vector{String}
    sigma_names::Vector{String}
    predictor_names::NTuple{2,Vector{String}}
    original_row::Vector{Int}
    observed_y::BitVector
    observed_x::NTuple{2,BitVector}
end

"""Row-level conditional moments for two Gaussian missing predictors."""
struct JointTwoMissingMetadata
    predictor_variables::NTuple{2,Symbol}
    original_row::Vector{Int}
    observed_y::BitVector
    observed_x::NTuple{2,BitVector}
    conditional_mean::Matrix{Float64}
    conditional_covariance::Array{Float64,3}
    conditional_status::Matrix{Symbol}
    all_rows::Int
    predictor_only_rows::Int
    uncertainty_status::Symbol
    optimizer_status::Symbol
    covariance_status::Symbol
end

"""Fitted exact prepared model for two independent Gaussian missing predictors."""
struct PreparedTwoJointGaussianFit
    fit::DrmFit
    prepared::PreparedTwoJointGaussianModel
    metadata::JointTwoMissingMetadata
end

function _two_joint_tuple_matrices(Xpredictor, n::Int)
    Xpredictor isa Tuple && length(Xpredictor) == 2 ||
        throw(ArgumentError("Xpredictor must be a two-element tuple"))
    return (_joint_matrix(Xpredictor[1], n, "Xpredictor[1]"),
            _joint_matrix(Xpredictor[2], n, "Xpredictor[2]"))
end

function _two_joint_tuple_names(names, sizes::Tuple{Int,Int}, label::AbstractString)
    names isa Tuple && length(names) == 2 ||
        throw(ArgumentError("$label must be a two-element tuple"))
    return (_joint_names(names[1], sizes[1], "$label[1]"),
            _joint_names(names[2], sizes[2], "$label[2]"))
end

"""
    prepared_joint_model(y, x::AbstractMatrix, Xmu, Xsigma, Xpredictor;
                         predictor = :gaussian, predictor_variables = (:x1, :x2), ...)

Construct the exact prepared model for two conditionally independent Gaussian
predictors. `Xmu` excludes both marked predictor columns. The fixed parameter
order is `beta, b1, b2, delta, alpha1, logtau1, alpha2, logtau2`.
"""
function prepared_joint_model(y, x::AbstractMatrix, Xmu, Xsigma, Xpredictor;
                              predictor::Symbol = :gaussian,
                              predictor_variables = (:x1, :x2),
                              mu_names = ["mu[$j]" for j in 1:size(Xmu, 2)],
                              sigma_names = ["sigma[$j]" for j in 1:size(Xsigma, 2)],
                              predictor_names = (["x1[$j]" for j in 1:size(Xpredictor[1], 2)],
                                                 ["x2[$j]" for j in 1:size(Xpredictor[2], 2)]),
                              original_row = collect(eachindex(y)))
    predictor === :gaussian ||
        throw(ArgumentError("two-predictor prepared route requires predictor = :gaussian"))
    n = length(y)
    n > 0 || throw(ArgumentError("prepared joint data must contain at least one row"))
    size(x) == (n, 2) || throw(ArgumentError("x must be an n-by-2 matrix"))
    predictor_variables isa Tuple && length(predictor_variables) == 2 ||
        throw(ArgumentError("predictor_variables must be a two-element tuple"))
    variables = (Symbol(predictor_variables[1]), Symbol(predictor_variables[2]))
    all(v -> !isempty(String(v)), variables) && variables[1] != variables[2] ||
        throw(ArgumentError("predictor_variables must contain two distinct nonempty names"))
    yy = Union{Missing,Float64}[_joint_float_or_missing(v, "y") for v in y]
    xx = Matrix{Union{Missing,Float64}}(undef, n, 2)
    for j in 1:2, i in 1:n
        xx[i, j] = _joint_float_or_missing(x[i, j], "x")
    end
    Xμ = _joint_matrix(Xmu, n, "Xmu")
    Xσ = _joint_matrix(Xsigma, n, "Xsigma")
    Xx = _two_joint_tuple_matrices(Xpredictor, n)
    raw_rows = collect(original_row)
    all(v -> v isa Integer && v > 0, raw_rows) ||
        throw(ArgumentError("original_row must contain positive integer entries"))
    rows = Int.(raw_rows)
    length(rows) == n || throw(ArgumentError("original_row must have $n entries"))
    length(unique(rows)) == n ||
        throw(ArgumentError("original_row must contain unique positive entries"))
    obs_y = BitVector(v !== missing for v in yy)
    obs_x = (BitVector(xx[i, 1] !== missing for i in 1:n),
             BitVector(xx[i, 2] !== missing for i in 1:n))
    mu_labels = _joint_names(mu_names, size(Xμ, 2), "mu_names")
    for variable in variables
        "mi($variable)" in mu_labels &&
            throw(ArgumentError("mu_names may not contain reserved appended name mi($variable)"))
    end
    return PreparedTwoJointGaussianModel(
        yy, xx, Xμ, Xσ, Xx, variables,
        mu_labels,
        _joint_names(sigma_names, size(Xσ, 2), "sigma_names"),
        _two_joint_tuple_names(predictor_names, (size(Xx[1], 2), size(Xx[2], 2)), "predictor_names"),
        rows, obs_y, obs_x,
    )
end

_two_joint_nbeta(model::PreparedTwoJointGaussianModel) = size(model.Xmu, 2)
_two_joint_ndelta(model::PreparedTwoJointGaussianModel) = size(model.Xsigma, 2)
_two_joint_nalpha(model::PreparedTwoJointGaussianModel, j::Int) = size(model.Xpredictor[j], 2)
_two_joint_ntheta(model::PreparedTwoJointGaussianModel) =
    _two_joint_nbeta(model) + 2 + _two_joint_ndelta(model) +
    _two_joint_nalpha(model, 1) + 1 + _two_joint_nalpha(model, 2) + 1

function _two_joint_ranges(model::PreparedTwoJointGaussianModel)
    p, r = _two_joint_nbeta(model), _two_joint_ndelta(model)
    q1, q2 = _two_joint_nalpha(model, 1), _two_joint_nalpha(model, 2)
    b = (p + 1):(p + 2)
    delta = (p + 3):(p + 2 + r)
    alpha1 = (last(delta) + 1):(last(delta) + q1)
    kappa1 = last(alpha1) + 1
    alpha2 = (kappa1 + 1):(kappa1 + q2)
    kappa2 = last(alpha2) + 1
    return (; b, delta, alpha1, kappa1, alpha2, kappa2)
end

function _two_joint_parameters(model::PreparedTwoJointGaussianModel, theta::AbstractVector)
    length(theta) == _two_joint_ntheta(model) ||
        throw(ArgumentError("theta has length $(length(theta)); expected $(_two_joint_ntheta(model))"))
    all(isfinite, theta) || throw(ArgumentError("theta contains a non-finite value"))
    p = _two_joint_nbeta(model)
    ranges = _two_joint_ranges(model)
    return @view(theta[1:p]), @view(theta[ranges.b]), @view(theta[ranges.delta]),
           @view(theta[ranges.alpha1]), theta[ranges.kappa1],
           @view(theta[ranges.alpha2]), theta[ranges.kappa2]
end

function _two_joint_linear_predictors(model::PreparedTwoJointGaussianModel, theta::AbstractVector)
    beta, b, delta, alpha1, kappa1, alpha2, kappa2 = _two_joint_parameters(model, theta)
    return model.Xmu * beta, b, model.Xsigma * delta,
           (model.Xpredictor[1] * alpha1, model.Xpredictor[2] * alpha2),
           (exp(kappa1), exp(kappa2))
end

"""Exact marginal row log likelihood in prepared-row order for the two-Gaussian route."""
function prepared_joint_rowloglik(model::PreparedTwoJointGaussianModel, theta::AbstractVector)
    a, b, log_sigma, eta, tau = _two_joint_linear_predictors(model, theta)
    T = promote_type(eltype(theta), Float64)
    result = Vector{T}(undef, length(model.y))
    @inbounds for i in eachindex(result)
        contribution = zero(T)
        mean_y = a[i]
        # Work on the SD scale: a finite `sigma` (for example 1e200) has a
        # well-defined Normal log density even though sigma^2 overflows.
        marginal_sd = exp(log_sigma[i])
        for j in 1:2
            if model.observed_x[j][i]
                xij = model.x[i, j]
                contribution += _joint_logpdf_normal(xij, eta[j][i], tau[j])
                mean_y += b[j] * xij
            else
                mean_y += b[j] * eta[j][i]
                marginal_sd = hypot(marginal_sd, b[j] * tau[j])
            end
        end
        result[i] = model.observed_y[i] ? contribution +
            _joint_logpdf_normal(model.y[i], mean_y, marginal_sd) : contribution
    end
    return result
end

prepared_joint_nll(model::PreparedTwoJointGaussianModel, theta::AbstractVector) =
    -sum(prepared_joint_rowloglik(model, theta))

"""
    prepared_joint_conditional_moments(model::PreparedTwoJointGaussianModel, theta)

Return marginal means, full 2-by-2 row posterior covariance matrices, and
per-predictor statuses. Off-diagonal covariance is retained when both predictors
are missing and the response is observed.
"""
function prepared_joint_conditional_moments(model::PreparedTwoJointGaussianModel, theta::AbstractVector)
    a, b, log_sigma, eta, tau = _two_joint_linear_predictors(model, theta)
    T = promote_type(eltype(theta), Float64)
    n = length(model.y)
    mean = Matrix{T}(undef, n, 2)
    covariance = zeros(T, n, 2, 2)
    status = Matrix{Symbol}(undef, n, 2)
    @inbounds for i in 1:n
        missing = Int[]
        base = a[i]
        for j in 1:2
            if model.observed_x[j][i]
                mean[i, j] = model.x[i, j]
                status[i, j] = :observed
                base += b[j] * model.x[i, j]
            else
                push!(missing, j)
                mean[i, j] = eta[j][i]
                covariance[i, j, j] = tau[j]^2
                status[i, j] = :predictor_only
            end
        end
        isempty(missing) && continue
        model.observed_y[i] || continue
        # `marginal_sd` is sqrt(sigma^2 + sum(b_j tau_j)^2), evaluated with
        # `hypot` to preserve a finite representable SD without squaring it.
        # `response_residual` deliberately excludes missing-predictor means;
        # the weighted update below avoids eta + gain*(y - a - b*eta), whose
        # subtraction loses the posterior mean at large scales.
        marginal_sd = exp(log_sigma[i])
        for j in missing
            marginal_sd = hypot(marginal_sd, b[j] * tau[j])
        end
        response_residual = model.y[i] - base
        scaled_btau = zeros(T, 2)
        for j in missing
            scaled_btau[j] = b[j] * tau[j] / marginal_sd
        end
        for j in missing
            other_sd = exp(log_sigma[i])
            for k in missing
                k == j || (other_sd = hypot(other_sd, b[k] * tau[k]))
            end
            other_ratio = other_sd / marginal_sd
            gain = (tau[j] / marginal_sd) * scaled_btau[j]
            mean[i, j] = other_ratio^2 * eta[j][i] + gain * response_residual
            for k in missing
                k == j || (mean[i, j] -= gain * b[k] * eta[k][i])
            end
            status[i, j] = :gaussian_posterior
            posterior_sd = tau[j] * other_ratio
            covariance[i, j, j] = posterior_sd * posterior_sd
        end
        for j in missing, k in missing
            j < k || continue
            value = -(tau[j] * scaled_btau[j]) * (tau[k] * scaled_btau[k])
            covariance[i, j, k] = value
            covariance[i, k, j] = value
        end
    end
    return (mean = mean, covariance = covariance,
            variance = hcat(copy(@view covariance[:, 1, 1]), copy(@view covariance[:, 2, 2])),
            status = status)
end

function prepared_joint_initial(model::PreparedTwoJointGaussianModel)
    theta = zeros(Float64, _two_joint_ntheta(model))
    ranges = _two_joint_ranges(model)
    complete = findall(model.observed_y .& model.observed_x[1] .& model.observed_x[2])
    if !isempty(complete)
        X = hcat(model.Xmu[complete, :], Float64[model.x[i, 1] for i in complete],
                 Float64[model.x[i, 2] for i in complete])
        response = Float64[model.y[i] for i in complete]
        coeff = try X \ response catch; zeros(size(X, 2)) end
        theta[1:length(coeff)] .= coeff
        residual = response - X * coeff
        sigma = length(residual) > 1 ? std(residual) : 1.0
        theta[first(ranges.delta)] = log(isfinite(sigma) && sigma > sqrt(eps(Float64)) ? sigma : sqrt(eps(Float64)))
    end
    for j in 1:2
        observed = findall(model.observed_x[j])
        isempty(observed) && continue
        X = model.Xpredictor[j][observed, :]
        response = Float64[model.x[i, j] for i in observed]
        coeff = try X \ response catch; zeros(size(X, 2)) end
        alpha = j == 1 ? ranges.alpha1 : ranges.alpha2
        theta[alpha] .= coeff
        residual = response - X * coeff
        tau = length(residual) > 1 ? std(residual) : 1.0
        theta[j == 1 ? ranges.kappa1 : ranges.kappa2] =
            log(isfinite(tau) && tau > sqrt(eps(Float64)) ? tau : sqrt(eps(Float64)))
    end
    return theta
end

function _two_joint_copy_model(model::PreparedTwoJointGaussianModel)
    return PreparedTwoJointGaussianModel(copy(model.y), copy(model.x), copy(model.Xmu), copy(model.Xsigma),
        (copy(model.Xpredictor[1]), copy(model.Xpredictor[2])), model.predictor_variables,
        copy(model.mu_names), copy(model.sigma_names),
        (copy(model.predictor_names[1]), copy(model.predictor_names[2])), copy(model.original_row),
        copy(model.observed_y), (copy(model.observed_x[1]), copy(model.observed_x[2])))
end

function fit_prepared_joint(model::PreparedTwoJointGaussianModel;
                            initial::AbstractVector = prepared_joint_initial(model),
                            g_tol::Real = 1e-8)
    frozen = _two_joint_copy_model(model)
    theta0 = Float64.(collect(initial))
    length(theta0) == _two_joint_ntheta(frozen) || throw(ArgumentError("initial has wrong length"))
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
    p = _two_joint_nbeta(frozen)
    ranges = _two_joint_ranges(frozen)
    blocks = Pair{Symbol,UnitRange{Int}}[
        :mu => (1:(p + 2)), :sigma => ranges.delta,
        Symbol(:mi_, frozen.predictor_variables[1]) => ranges.alpha1,
        Symbol(:logsd_mi_, frozen.predictor_variables[1]) => (ranges.kappa1:ranges.kappa1),
        Symbol(:mi_, frozen.predictor_variables[2]) => ranges.alpha2,
        Symbol(:logsd_mi_, frozen.predictor_variables[2]) => (ranges.kappa2:ranges.kappa2),
    ]
    names = Pair{Symbol,Vector{String}}[
        :mu => [frozen.mu_names; "mi($(frozen.predictor_variables[1]))"; "mi($(frozen.predictor_variables[2]))"],
        :sigma => frozen.sigma_names,
        Symbol(:mi_, frozen.predictor_variables[1]) => frozen.predictor_names[1],
        Symbol(:logsd_mi_, frozen.predictor_variables[1]) => ["log_sd"],
        Symbol(:mi_, frozen.predictor_variables[2]) => frozen.predictor_names[2],
        Symbol(:logsd_mi_, frozen.predictor_variables[2]) => ["log_sd"],
    ]
    means = Dict{Symbol,Vector{Float64}}(:mu => Float64.(frozen.Xmu * theta[1:p] .+
        theta[ranges.b[1]] .* moments.mean[:, 1] .+ theta[ranges.b[2]] .* moments.mean[:, 2]))
    observed = Dict{Symbol,Vector{Float64}}(:mu => Float64[v === missing ? NaN : v for v in frozen.y])
    scales = Dict{Symbol,Vector{Float64}}(:sigma => exp.(frozen.Xsigma * theta[ranges.delta]))
    base = DrmFit(PreparedJointGaussian(), blocks, names, theta, covariance, -nll_value,
        count(frozen.observed_y), converged, means, observed, scales)
    base = _withiterations(_withnll(base, nll), Optim.iterations(result))
    metadata = JointTwoMissingMetadata(frozen.predictor_variables, copy(frozen.original_row), copy(frozen.observed_y),
        (copy(frozen.observed_x[1]), copy(frozen.observed_x[2])), Float64.(moments.mean), Float64.(moments.covariance),
        copy(moments.status), length(frozen.y), count(i -> !frozen.observed_y[i] &&
        (frozen.observed_x[1][i] || frozen.observed_x[2][i]), eachindex(frozen.y)), :not_computed,
        optimizer_status, covariance_status)
    return PreparedTwoJointGaussianFit(base, frozen, metadata)
end

function joint_missing_summary(fit::PreparedTwoJointGaussianFit)
    m = fit.metadata
    return (predictor_variables = m.predictor_variables, original_row = copy(m.original_row),
            observed_y = copy(m.observed_y), observed_x = (copy(m.observed_x[1]), copy(m.observed_x[2])),
            conditional_mean = copy(m.conditional_mean), conditional_covariance = copy(m.conditional_covariance),
            conditional_status = copy(m.conditional_status), all_rows = m.all_rows,
            predictor_only_rows = m.predictor_only_rows, uncertainty_status = m.uncertainty_status,
            optimizer_status = m.optimizer_status, covariance_status = m.covariance_status)
end

function _two_joint_imputation_uncertainty(model::PreparedTwoJointGaussianModel, theta, covariance;
                                            predictor_index::Integer,
                                            se::Bool = true,
                                            covariance_status::Symbol = :observed_information_inverse)
    predictor_index in 1:2 || throw(ArgumentError("predictor_index must be 1 or 2"))
    se isa Bool || throw(ArgumentError("se must be Bool"))
    moments = prepared_joint_conditional_moments(model, theta)
    j = Int(predictor_index)
    statuses = fill(_joint_uncertainty_status(covariance_status), length(model.y))
    parameter_variance = zeros(Float64, length(model.y))
    std_error = fill(NaN, length(model.y))
    result() = (estimate = Float64.(moments.mean[:, j]), std_error = std_error,
                conditional_variance = Float64.(moments.covariance[:, j, j]),
                parameter_variance = parameter_variance, uncertainty_status = statuses)
    (!se || covariance_status !== :observed_information_inverse) && return result()
    if size(covariance) == (length(theta), length(theta)) && all(isfinite, covariance)
        V = Matrix{Float64}(covariance)
        if isapprox(V, V'; rtol = 1e-12, atol = 1e-12) && issuccess(cholesky(Symmetric(V); check = false))
            missing_ids = findall(.!model.observed_x[j])
            if !isempty(missing_ids)
                J = try
                    ForwardDiff.jacobian(t -> prepared_joint_conditional_moments(model, t).mean[missing_ids, j], theta)
                catch
                    fill(NaN, length(missing_ids), length(theta))
                end
                parameter_variance[missing_ids] .= vec(sum((J * V) .* J; dims = 2))
            end
        else
            fill!(statuses, "sdreport_non_pd_hessian")
        end
    else
        fill!(statuses, "sdreport_unavailable")
    end
    if se && all(==("ok"), statuses)
        for i in findall(.!model.observed_x[j])
            variance = moments.covariance[i, j, j] + parameter_variance[i]
            if isfinite(variance) && variance >= 0
                std_error[i] = sqrt(variance)
            else
                statuses[i] = "route_conditional_se_unavailable"
            end
        end
    end
    return result()
end

function _two_joint_imputation_table(fit::PreparedTwoJointGaussianFit, j::Int; rows = :missing, se::Bool = true)
    choice = rows isa Symbol ? rows : rows isa AbstractString ? Symbol(rows) : nothing
    choice in (:missing, :all) || throw(ArgumentError("imputed: rows must be :missing or :all"))
    model, metadata = fit.prepared, fit.metadata
    ids = choice === :all ? collect(eachindex(model.y)) : findall(.!model.observed_x[j])
    values = _two_joint_imputation_uncertainty(model, fit.fit.theta, fit.fit.vcov;
        predictor_index = j, se = se, covariance_status = metadata.covariance_status)
    sd = Union{Missing,Float64}[isfinite(values.std_error[i]) ? values.std_error[i] : missing for i in ids]
    source = [model.observed_x[j][i] ? "observed" : "conditional_mode" for i in ids]
    return (variable = fill(String(model.predictor_variables[j]), length(ids)), original_row = copy(model.original_row[ids]),
            model_row = ids, observed = copy(model.observed_x[j][ids]), estimate = copy(values.estimate[ids]),
            std_error = sd, source = source, uncertainty_status = copy(values.uncertainty_status[ids]))
end

function imputed(fit::PreparedTwoJointGaussianFit; variable = nothing, rows = :missing, se::Bool = true)
    requested = variable isa AbstractString ? Symbol(variable) : variable
    requested isa Symbol || throw(ArgumentError("imputed: two-predictor fits require `variable`"))
    j = findfirst(==(requested), fit.prepared.predictor_variables)
    j === nothing && throw(ArgumentError("imputed: no modelled predictor named `$requested`"))
    se isa Bool || throw(ArgumentError("imputed: `se` must be Bool"))
    return _two_joint_imputation_table(fit, j; rows = rows, se = se)
end
