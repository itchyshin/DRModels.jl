using Test
using DRModels
using LinearAlgebra
using ForwardDiff

# BLAS is pinned to one thread for reproducible linear algebra in this file.
# The Julia thread count is deliberately NOT constrained: nothing here is
# thread-sensitive (no @threads/@spawn in the tests or in src/joint_missing*.jl),
# and the old `|| error(...)` aborted every later file in the shard under a
# multi-threaded `Pkg.test`. How evidence was gathered belongs in the receipt,
# not as a precondition on the tests.
BLAS.set_num_threads(1)

_normal_logpdf(y, mean, covariance) = begin
    length(y) == 0 && return 0.0
    factor = cholesky(Symmetric(covariance))
    residual = y - mean
    -0.5 * (length(y) * log(2π) + 2sum(log, diag(factor.L)) + dot(residual, factor \ residual))
end

function _two_joint_fixture(; predictor_variables = (:x1, :x2), swapped = false)
    # The eight rows enumerate every (observed y, observed x1, observed x2)
    # state. Predictor designs deliberately have different widths.
    masks = ((false, false, false), (true, false, false), (false, true, false),
             (true, true, false), (false, false, true), (true, false, true),
             (false, true, true), (true, true, true))
    z = collect(range(-0.8, 0.7; length = length(masks)))
    x1_full = 0.2 .+ 0.6 .* z .+ [0.1, -0.2, 0.25, -0.1, 0.15, -0.15, 0.2, -0.05]
    x2_full = -0.3 .+ 0.4 .* z .+ [-0.15, 0.2, -0.1, 0.25, -0.2, 0.1, 0.15, -0.25]
    y_full = 0.4 .- 0.25 .* z .+ 0.8 .* x1_full .- 0.55 .* x2_full .+
             [0.05, -0.08, 0.04, -0.03, 0.06, -0.05, 0.02, -0.04]
    y = Union{Missing,Float64}[masks[i][1] ? y_full[i] : missing for i in eachindex(masks)]
    x = Matrix{Union{Missing,Float64}}(undef, length(masks), 2)
    for i in eachindex(masks)
        x[i, 1] = masks[i][2] ? x1_full[i] : missing
        x[i, 2] = masks[i][3] ? x2_full[i] : missing
    end
    Xmu = hcat(ones(length(z)), z)
    Xsigma = ones(length(z), 1)
    Xp1, Xp2 = hcat(ones(length(z)), z), ones(length(z), 1)
    if swapped
        x = x[:, [2, 1]]
        Xp1, Xp2 = Xp2, Xp1
    end
    predictor_names = swapped ? (("x2_0",), ("x1_0", "x1_z")) :
        (("x1_0", "x1_z"), ("x2_0",))
    model = DRModels.prepared_joint_model(y, x, Xmu, Xsigma, (Xp1, Xp2);
        predictor = :gaussian, predictor_variables = predictor_variables,
        mu_names = ["(Intercept)", "z"], sigma_names = ["(Intercept)"],
        predictor_names = predictor_names,
        original_row = [41, 9, 73, 2, 58, 30, 101, 14])
    theta = swapped ?
        [0.4, -0.25, -0.55, 0.8, log(0.7), -0.3, log(0.75), 0.2, 0.6, log(0.9)] :
        [0.4, -0.25, 0.8, -0.55, log(0.7), 0.2, 0.6, log(0.9), -0.3, log(0.75)]
    return model, theta, masks
end

function _dense_row_loglik(model, theta, i)
    p = size(model.Xmu, 2)
    ranges = DRModels._two_joint_ranges(model)
    beta, b = theta[1:p], theta[ranges.b]
    delta, alpha1, kappa1 = theta[ranges.delta], theta[ranges.alpha1], theta[ranges.kappa1]
    alpha2, kappa2 = theta[ranges.alpha2], theta[ranges.kappa2]
    eta1, eta2 = dot(model.Xpredictor[1][i, :], alpha1), dot(model.Xpredictor[2][i, :], alpha2)
    tau1, tau2 = exp(kappa1), exp(kappa2)
    mu_y = dot(model.Xmu[i, :], beta) + b[1] * eta1 + b[2] * eta2
    sigma = exp(dot(model.Xsigma[i, :], delta))
    covariance = [tau1^2 0.0 b[1] * tau1^2;
                  0.0 tau2^2 b[2] * tau2^2;
                  b[1] * tau1^2 b[2] * tau2^2 sigma^2 + (b[1] * tau1)^2 + (b[2] * tau2)^2]
    values = Float64[]
    means = Float64[]
    ids = Int[]
    for j in 1:2
        if model.observed_x[j][i]
            push!(values, model.x[i, j]); push!(means, j == 1 ? eta1 : eta2); push!(ids, j)
        end
    end
    if model.observed_y[i]
        push!(values, model.y[i]); push!(means, mu_y); push!(ids, 3)
    end
    return _normal_logpdf(values, means, covariance[ids, ids])
end

function _dense_conditional(model, theta, i)
    p = size(model.Xmu, 2)
    ranges = DRModels._two_joint_ranges(model)
    beta, b = theta[1:p], theta[ranges.b]
    delta, alpha1, kappa1 = theta[ranges.delta], theta[ranges.alpha1], theta[ranges.kappa1]
    alpha2, kappa2 = theta[ranges.alpha2], theta[ranges.kappa2]
    eta = [dot(model.Xpredictor[1][i, :], alpha1), dot(model.Xpredictor[2][i, :], alpha2)]
    tau = exp.([kappa1, kappa2])
    mu = [eta[1], eta[2], dot(model.Xmu[i, :], beta) + dot(b, eta)]
    sigma = exp(dot(model.Xsigma[i, :], delta))
    covariance = [tau[1]^2 0.0 b[1] * tau[1]^2;
                  0.0 tau[2]^2 b[2] * tau[2]^2;
                  b[1] * tau[1]^2 b[2] * tau[2]^2 sigma^2 + sum((b .* tau).^2)]
    missing_ids = findall(j -> !model.observed_x[j][i], 1:2)
    observed_ids = [j for j in 1:2 if model.observed_x[j][i]]
    observed_values = Float64[model.x[i, j] for j in observed_ids]
    if model.observed_y[i]
        push!(observed_ids, 3); push!(observed_values, model.y[i])
    end
    output_mean, output_covariance = copy(eta), zeros(2, 2)
    for j in 1:2
        model.observed_x[j][i] && (output_mean[j] = model.x[i, j])
    end
    isempty(missing_ids) && return output_mean, output_covariance
    if isempty(observed_ids)
        output_covariance[missing_ids, missing_ids] .= covariance[missing_ids, missing_ids]
        return output_mean, output_covariance
    end
    Cmo = covariance[missing_ids, observed_ids]
    Coo = covariance[observed_ids, observed_ids]
    gain = Cmo / Coo
    output_mean[missing_ids] .= mu[missing_ids] + gain * (observed_values - mu[observed_ids])
    output_covariance[missing_ids, missing_ids] .= covariance[missing_ids, missing_ids] - gain * covariance[observed_ids, missing_ids]
    return output_mean, output_covariance
end

function _central_gradient(f, theta; h = 1e-6)
    output = similar(theta)
    for j in eachindex(theta)
        step = h * max(1.0, abs(theta[j]))
        plus, minus = copy(theta), copy(theta)
        plus[j] += step; minus[j] -= step
        output[j] = (f(plus) - f(minus)) / (2step)
    end
    return output
end

@testset "prepared two-Gaussian missing-predictor API" begin
    @test isdefined(DRModels, :PreparedTwoJointGaussianModel)
    @test isdefined(DRModels, :PreparedTwoJointGaussianFit)
    @test isdefined(DRModels, :prepared_joint_model)
end

@testset "two Gaussian exact likelihood and conditional 2-by-2 moments" begin
    model, theta, masks = _two_joint_fixture()
    @test model.original_row == [41, 9, 73, 2, 58, 30, 101, 14]
    @test Tuple(model.observed_x[1]) == Tuple(mask[2] for mask in masks)
    @test Tuple(model.observed_x[2]) == Tuple(mask[3] for mask in masks)
    @test DRModels._two_joint_ntheta(model) == 10
    rows = DRModels.prepared_joint_rowloglik(model, theta)
    @test rows ≈ [_dense_row_loglik(model, theta, i) for i in eachindex(rows)] atol = 1e-12
    @test rows[1] == 0.0 # all missing: it contributes no likelihood
    moments = DRModels.prepared_joint_conditional_moments(model, theta)
    @test size(moments.mean) == (8, 2)
    @test size(moments.covariance) == (8, 2, 2)
    @test size(moments.variance) == (8, 2)
    for i in 1:8
        dense_mean, dense_covariance = _dense_conditional(model, theta, i)
        @test moments.mean[i, :] ≈ dense_mean atol = 1e-12
        @test moments.covariance[i, :, :] ≈ dense_covariance atol = 1e-12
    end
    # Both predictors missing with observed y: changing one slope flips the posterior covariance sign.
    @test moments.covariance[2, 1, 2] > 0
    flipped = copy(theta); flipped[4] *= -1
    @test DRModels.prepared_joint_conditional_moments(model, flipped).covariance[2, 1, 2] < 0
    zero_slope = copy(theta); zero_slope[3] = 0.0
    prior = DRModels.prepared_joint_conditional_moments(model, zero_slope)
    @test prior.covariance[2, 1, 2] == 0.0
    @test prior.variance[2, 1] ≈ exp(2theta[8]) atol = 1e-12
end

@testset "two Gaussian derivatives, masks, and input validation" begin
    model, theta, _ = _two_joint_fixture()
    nll = t -> DRModels.prepared_joint_nll(model, t)
    @test isapprox(ForwardDiff.gradient(nll, theta), _central_gradient(nll, theta); atol = 1e-6, rtol = 1e-6)
    @test DRModels.prepared_joint_nll(model, theta) ≈ -sum(DRModels.prepared_joint_rowloglik(model, theta)) atol = 1e-12
    H = ForwardDiff.hessian(nll, theta)
    direction = collect(range(-0.35, 0.4; length = length(theta)))
    h = 1e-5
    directional_fd = (ForwardDiff.gradient(nll, theta .+ h .* direction) -
                      ForwardDiff.gradient(nll, theta .- h .* direction)) ./ (2h)
    @test isapprox(H * direction, directional_fd; atol = 1e-5, rtol = 1e-5)
    # A broad but finite scale range and zero slope remain well-defined.
    scaled = copy(theta); scaled[3] = 0.0; scaled[8] = log(1e-6); scaled[10] = log(1e6)
    @test isfinite(DRModels.prepared_joint_nll(model, scaled))
    @test all(isfinite, DRModels.prepared_joint_conditional_moments(model, scaled).covariance)
    @test_throws ArgumentError DRModels.prepared_joint_model(model.y, model.x, model.Xmu, model.Xsigma,
        model.Xpredictor; predictor = :bernoulli)
    @test_throws ArgumentError DRModels.prepared_joint_model(model.y, model.x, model.Xmu, model.Xsigma,
        model.Xpredictor; predictor_variables = (:x, :x))
    @test_throws ArgumentError DRModels.prepared_joint_model(model.y, model.x, model.Xmu, model.Xsigma,
        model.Xpredictor; original_row = [1, 2, 3, 4, 5, 6, 7, 1])
    @test_throws ArgumentError DRModels.prepared_joint_model(model.y, model.x, model.Xmu, model.Xsigma,
        model.Xpredictor; original_row = [1, 2, 3, 4, 5, 6, 7, 8.5])
    @test_throws ArgumentError DRModels.prepared_joint_model(model.y, model.x, model.Xmu, model.Xsigma,
        model.Xpredictor; mu_names = ["mi(x1)", "z"])
end

@testset "two Gaussian stable large-scale arithmetic and predictor permutation" begin
    model, theta, _ = _two_joint_fixture()
    # A representable Normal SD must not be squared before its row log density.
    sigma_large = DRModels.prepared_joint_model([1.0], reshape(Union{Missing,Float64}[missing, 0.0], 1, 2),
        ones(1, 1), ones(1, 1), (ones(1, 1), ones(1, 1)); predictor_variables = (:x1, :x2))
    theta_large = [0.0, 1.0, 0.0, log(1e200), 0.0, log(1.0), 0.0, log(1.0)]
    @test isfinite(DRModels.prepared_joint_rowloglik(sigma_large, theta_large)[1])
    # For tau = 1e8, b = 1, sigma = 1, the posterior variance is one, not
    # the cancellation-prone difference 1e16 - 1e16.
    x1_large = DRModels.prepared_joint_model([0.0], reshape(Union{Missing,Float64}[missing, 0.0], 1, 2),
        ones(1, 1), ones(1, 1), (ones(1, 1), ones(1, 1)); predictor_variables = (:x1, :x2))
    theta_x1_large = [0.0, 1.0, 0.0, 0.0, 0.0, log(1e8), 0.0, log(1.0)]
    moments = DRModels.prepared_joint_conditional_moments(x1_large, theta_x1_large)
    @test moments.variance[1, 1] ≈ 1.0 rtol = 1e-12
    theta_x1_extreme = copy(theta_x1_large); theta_x1_extreme[6] = log(1e100)
    @test DRModels.prepared_joint_conditional_moments(x1_large, theta_x1_extreme).variance[1, 1] ≈ 1.0 rtol = 1e-12
    large_prior_mean = copy(theta_x1_large); large_prior_mean[5] = 1e16
    @test DRModels.prepared_joint_conditional_moments(x1_large, large_prior_mean).mean[1, 1] ≈ 1.0 rtol = 1e-12

    swapped, swapped_theta, _ = _two_joint_fixture(; predictor_variables = (:x2, :x1), swapped = true)
    base_rows = DRModels.prepared_joint_rowloglik(model, theta)
    swapped_rows = DRModels.prepared_joint_rowloglik(swapped, swapped_theta)
    @test swapped_rows ≈ base_rows atol = 1e-12
    base_moments = DRModels.prepared_joint_conditional_moments(model, theta)
    swapped_moments = DRModels.prepared_joint_conditional_moments(swapped, swapped_theta)
    @test swapped_moments.mean[:, 1] ≈ base_moments.mean[:, 2] atol = 1e-12
    @test swapped_moments.mean[:, 2] ≈ base_moments.mean[:, 1] atol = 1e-12
    @test swapped_moments.covariance[:, 1, 1] ≈ base_moments.covariance[:, 2, 2] atol = 1e-12
    @test swapped_moments.covariance[:, 2, 2] ≈ base_moments.covariance[:, 1, 1] atol = 1e-12
    @test swapped_moments.covariance[:, 1, 2] ≈ base_moments.covariance[:, 2, 1] atol = 1e-12
end

function _synthetic_two_fit(model, theta, covariance;
                            covariance_status::Symbol = :observed_information_inverse)
    ranges = DRModels._two_joint_ranges(model)
    p = size(model.Xmu, 2)
    moments = DRModels.prepared_joint_conditional_moments(model, theta)
    blocks = Pair{Symbol,UnitRange{Int}}[
        :mu => (1:(p + 2)), :sigma => ranges.delta,
        Symbol(:mi_, model.predictor_variables[1]) => ranges.alpha1,
        Symbol(:logsd_mi_, model.predictor_variables[1]) => (ranges.kappa1:ranges.kappa1),
        Symbol(:mi_, model.predictor_variables[2]) => ranges.alpha2,
        Symbol(:logsd_mi_, model.predictor_variables[2]) => (ranges.kappa2:ranges.kappa2),
    ]
    names = Pair{Symbol,Vector{String}}[
        :mu => [model.mu_names; "mi($(model.predictor_variables[1]))"; "mi($(model.predictor_variables[2]))"],
        :sigma => model.sigma_names,
        Symbol(:mi_, model.predictor_variables[1]) => model.predictor_names[1],
        Symbol(:logsd_mi_, model.predictor_variables[1]) => ["log_sd"],
        Symbol(:mi_, model.predictor_variables[2]) => model.predictor_names[2],
        Symbol(:logsd_mi_, model.predictor_variables[2]) => ["log_sd"],
    ]
    nll = t -> DRModels.prepared_joint_nll(model, t)
    raw = DRModels.DrmFit(DRModels.PreparedJointGaussian(), blocks, names, copy(theta), Matrix{Float64}(covariance),
        -nll(theta), count(model.observed_y), true,
        Dict{Symbol,Vector{Float64}}(:mu => copy(moments.mean[:, 1])),
        Dict{Symbol,Vector{Float64}}(:mu => Float64[v === missing ? NaN : v for v in model.y]),
        Dict{Symbol,Vector{Float64}}(:sigma => exp.(model.Xsigma * theta[ranges.delta])))
    raw = DRModels._withnll(raw, nll)
    metadata = DRModels.JointTwoMissingMetadata(model.predictor_variables, copy(model.original_row), copy(model.observed_y),
        (copy(model.observed_x[1]), copy(model.observed_x[2])), Float64.(moments.mean), Float64.(moments.covariance),
        copy(moments.status), length(model.y), count(i -> !model.observed_y[i] &&
        (model.observed_x[1][i] || model.observed_x[2][i]), eachindex(model.y)), :not_computed,
        :converged, covariance_status)
    return DRModels.PreparedTwoJointGaussianFit(raw, model, metadata)
end

@testset "two Gaussian imputation uses the full raw covariance" begin
    model, theta, _ = _two_joint_fixture()
    # This is deliberately an arbitrary evaluation point, not an optimizer result.
    arbitrary = theta .+ [0.03, -0.02, 0.04, 0.01, -0.03, 0.02, -0.01, 0.04, 0.03, -0.02]
    A = reshape(Float64.(1:100), 10, 10) ./ 100
    V = A * A' + 0.2I
    @test abs(V[1, end]) > 1e-8
    fit = _synthetic_two_fit(model, arbitrary, V)
    for (j, variable) in enumerate((:x1, :x2))
        values = DRModels._two_joint_imputation_uncertainty(model, arbitrary, V;
            predictor_index = j, covariance_status = :observed_information_inverse)
        ids = findall(.!model.observed_x[j])
        J = reduce(vcat, permutedims.([_central_gradient(t -> _dense_conditional(model, t, i)[1][j], arbitrary) for i in ids]))
        conditional = DRModels.prepared_joint_conditional_moments(model, arbitrary).covariance[ids, j, j]
        expected_parameter = vec(sum((J * V) .* J; dims = 2))
        @test values.parameter_variance[ids] ≈ expected_parameter atol = 1e-6
        @test values.std_error[ids].^2 ≈ conditional .+ expected_parameter atol = 1e-6
        @test all(isnan, values.std_error[model.observed_x[j]])
        table = DRModels.imputed(fit; variable = variable, rows = :all)
        @test table.variable == fill(String(variable), length(model.y))
        @test table.original_row == model.original_row
        @test all(ismissing, table.std_error[model.observed_x[j]])
        @test_throws ArgumentError DRModels.imputed(fit; variable = :unknown)
    end
    # Permute both covariance axes, including the off-block entries, and
    # select the same scientific predictor after swapping its position.
    swapped, _, _ = _two_joint_fixture(; predictor_variables = (:x2, :x1), swapped = true)
    perm = [1, 2, 4, 3, 5, 9, 10, 6, 7, 8]
    base_uncertainty = DRModels._two_joint_imputation_uncertainty(model, arbitrary, V; predictor_index = 1)
    swap_uncertainty = DRModels._two_joint_imputation_uncertainty(swapped, arbitrary[perm], V[perm, perm]; predictor_index = 2)
    @test swap_uncertainty.estimate ≈ base_uncertainty.estimate atol = 1e-12
    ids1 = findall(.!model.observed_x[1])
    @test swap_uncertainty.std_error[ids1] ≈ base_uncertainty.std_error[ids1] atol = 1e-10
    @test_throws ArgumentError DRModels.imputed(fit)
    @test_throws ArgumentError DRModels.imputed(fit; variable = :x1, rows = :other)
    non_pd = DRModels._two_joint_imputation_uncertainty(model, arbitrary, -Matrix(V); predictor_index = 1)
    @test all(==("sdreport_non_pd_hessian"), non_pd.uncertainty_status)
    @test all(isnan, non_pd.std_error)
    unavailable = DRModels._two_joint_imputation_uncertainty(model, arbitrary, fill(NaN, 10, 10);
        predictor_index = 2, covariance_status = :hessian_unavailable)
    @test all(==("sdreport_failed"), unavailable.uncertainty_status)
    @test all(isnan, unavailable.std_error)
    unavailable_no_se = DRModels._two_joint_imputation_uncertainty(model, arbitrary, V;
        predictor_index = 2, se = false, covariance_status = :hessian_unavailable)
    @test all(==("sdreport_failed"), unavailable_no_se.uncertainty_status)
    @test all(isnan, unavailable_no_se.std_error)
end

@testset "two Gaussian fit freezes inputs and reports covariance status" begin
    n = 72
    z = collect(range(-1.0, 1.0; length = n))
    x1_full = 0.2 .+ 0.5 .* z .+ 0.16 .* sin.(1:n)
    x2_full = -0.1 .+ 0.35 .* z .+ 0.13 .* cos.(1:n)
    y_full = 0.3 .- 0.2 .* z .+ 0.75 .* x1_full .- 0.45 .* x2_full .+ 0.08 .* sin.(2 .* (1:n))
    y = Union{Missing,Float64}[i % 11 == 0 ? missing : y_full[i] for i in 1:n]
    x = Matrix{Union{Missing,Float64}}(undef, n, 2)
    for i in 1:n
        x[i, 1] = i % 5 == 0 ? missing : x1_full[i]
        x[i, 2] = i % 7 == 0 ? missing : x2_full[i]
    end
    model = DRModels.prepared_joint_model(y, x, hcat(ones(n), z), ones(n, 1),
        (hcat(ones(n), z), ones(n, 1)); predictor_variables = (:x1, :x2), original_row = collect(1001:(1000 + n)))
    fitted = DRModels.fit_prepared_joint(model; g_tol = 1e-7)
    @test fitted isa DRModels.PreparedTwoJointGaussianFit
    @test fitted.fit.nobs == count(model.observed_y)
    @test fitted.metadata.predictor_only_rows == count(i -> !model.observed_y[i] &&
        (model.observed_x[1][i] || model.observed_x[2][i]), eachindex(model.y))
    @test fitted.metadata.covariance_status in (:observed_information_inverse, :hessian_not_positive_definite, :hessian_unavailable)
    nll_before = fitted.fit.nll(copy(fitted.fit.theta))
    model.Xmu[1, 1] = 1e6; model.x[1, 1] = -1e6
    @test fitted.fit.nll(fitted.fit.theta) == nll_before
    summary = DRModels.joint_missing_summary(fitted)
    summary.original_row[1] = -1
    @test fitted.metadata.original_row[1] == 1001
end

println("S9_TWO_GAUSSIAN_TARGETED_PASS")
