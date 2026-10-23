using Test
using DRModels
using ForwardDiff
using LinearAlgebra
using QuadGK

const _joint_source = joinpath(@__DIR__, "..", "src", "joint_missing_predictor.jl")
if !isdefined(DRModels, :PreparedJointModel)
    Base.include(DRModels, _joint_source)
end

@testset "prepared joint missing-predictor API" begin
    @test isdefined(DRModels, :PreparedJointModel)
    @test isdefined(DRModels, :prepared_joint_model)
    @test isdefined(DRModels, :prepared_joint_rowloglik)
    @test isdefined(DRModels, :prepared_joint_conditional_moments)
    @test isdefined(DRModels, :fit_prepared_joint)
    @test isdefined(DRModels, :PreparedJointFit)
    @test isdefined(DRModels, :joint_missing_summary)
end

function _joint_fixture(predictor::Symbol; eta_extreme = 0.0)
    y = Union{Missing,Float64}[1.7, -0.2, missing, missing]
    x = predictor === :gaussian ? Union{Missing,Float64}[0.8, missing, 1.0, missing] :
        Union{Missing,Float64}[1.0, missing, 1.0, missing]
    z = [-0.6, 0.3, 0.8, -0.1]
    Xmu = hcat(ones(4), z)
    # Deliberately non-intercept-only: the prepared layer must apply this
    # residual-scale design to each row without a special route.
    Xsigma = hcat(ones(4), z)
    Xpredictor = hcat(ones(4), z)
    model = DRModels.prepared_joint_model(y, x, Xmu, Xsigma, Xpredictor;
        predictor = predictor,
        mu_names = ["(Intercept)", "z"],
        sigma_names = ["(Intercept)", "z"],
        predictor_names = ["(Intercept)", "z"],
        original_row = [17, 4, 81, 29])
    theta = predictor === :gaussian ?
        [0.2, -0.35, 0.7, 0.1, -0.2, eta_extreme, 0.25, log(0.9)] :
        [0.2, -0.35, 0.7, 0.1, -0.2, eta_extreme, 0.25]
    return model, theta
end

_lognormal(y, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((y - m) / s)^2

function _central_gradient(f, theta; h = 1e-6)
    out = similar(theta)
    for j in eachindex(theta)
        step = h * max(1.0, abs(theta[j]))
        plus, minus = copy(theta), copy(theta)
        plus[j] += step
        minus[j] -= step
        out[j] = (f(plus) - f(minus)) / (2step)
    end
    return out
end

@testset "Gaussian predictor exact row likelihood and moments" begin
    model, theta = _joint_fixture(:gaussian)
    row = DRModels.prepared_joint_rowloglik(model, theta)
    a = model.Xmu * theta[1:2]
    sigma = exp.(model.Xsigma * theta[4:5])
    m = model.Xpredictor * theta[6:7]
    tau = exp(theta[8])
    @test model.row_state == [:complete, :x_missing_y_observed,
                              :x_observed_y_missing, :both_missing]
    @test model.original_row == [17, 4, 81, 29]
    @test row[1] ≈ _lognormal(model.x[1], m[1], tau) +
                    _lognormal(model.y[1], a[1] + theta[3] * model.x[1], sigma[1]) atol = 1e-12
    @test row[2] ≈ _lognormal(model.y[2], a[2] + theta[3] * m[2], hypot(sigma[2], theta[3] * tau)) atol = 1e-12
    @test row[3] ≈ _lognormal(model.x[3], m[3], tau) atol = 1e-12
    @test row[4] == 0.0

    moments = DRModels.prepared_joint_conditional_moments(model, theta)
    post_sd = inv(hypot(inv(tau), theta[3] / sigma[2]))
    post_var = post_sd^2
    post_mean = post_var * (m[2] / tau^2 + theta[3] * (model.y[2] - a[2]) / sigma[2]^2)
    @test moments.status == [:observed, :gaussian_posterior, :observed, :predictor_only]
    @test moments.mean[2] ≈ post_mean atol = 1e-12
    @test moments.variance[2] ≈ post_var atol = 1e-12
    @test moments.mean[4] ≈ m[4] atol = 1e-12
    @test moments.variance[4] ≈ tau^2 atol = 1e-12

    # Independent numerical integration confirms the closed-form Gaussian
    # posterior moment on the missing-x / observed-y row.
    kernel(x) = exp(_lognormal(x, m[2], tau) + _lognormal(model.y[2], a[2] + theta[3] * x, sigma[2]))
    lo, hi = m[2] - 12tau, m[2] + 12tau
    mass, _ = quadgk(kernel, lo, hi)
    qmean, _ = quadgk(x -> x * kernel(x), lo, hi)
    qsecond, _ = quadgk(x -> x^2 * kernel(x), lo, hi)
    @test moments.mean[2] ≈ qmean / mass atol = 1e-10
    @test moments.variance[2] ≈ qsecond / mass - (qmean / mass)^2 atol = 1e-10

    theta_b0 = copy(theta); theta_b0[3] = 0.0
    prior = DRModels.prepared_joint_conditional_moments(model, theta_b0)
    @test prior.mean[2] ≈ m[2] atol = 1e-12
    @test prior.variance[2] ≈ tau^2 atol = 1e-12
end

@testset "Bernoulli predictor finite-state likelihood and moments" begin
    model, theta = _joint_fixture(:bernoulli)
    row = DRModels.prepared_joint_rowloglik(model, theta)
    a = model.Xmu * theta[1:2]
    sigma = exp.(model.Xsigma * theta[4:5])
    eta = model.Xpredictor * theta[6:7]
    p = 1 / (1 + exp(-eta[2]))
    w0 = (1 - p) * exp(_lognormal(model.y[2], a[2], sigma[2]))
    w1 = p * exp(_lognormal(model.y[2], a[2] + theta[3], sigma[2]))
    q = w1 / (w0 + w1)
    @test row[2] ≈ log(w0 + w1) atol = 1e-12
    moments = DRModels.prepared_joint_conditional_moments(model, theta)
    @test moments.status == [:observed, :bernoulli_posterior, :observed, :predictor_only]
    @test moments.mean[2] ≈ q atol = 1e-12
    @test moments.variance[2] ≈ q * (1 - q) atol = 1e-12
    @test moments.mean[4] ≈ 1 / (1 + exp(-eta[4])) atol = 1e-12

    dominant = DRModels.prepared_joint_model([1e9], [missing], ones(1, 1), ones(1, 1), ones(1, 1);
        predictor = :bernoulli, original_row = [903])
    dominant_theta = [0.0, 1e9, 0.0, 0.0]
    dominant_row = DRModels.prepared_joint_rowloglik(dominant, dominant_theta)
    @test dominant_row[1] ≈ -log(2) + _lognormal(1e9, 1e9, 1.0) atol = 1e-12
    @test isfinite(dominant_row[1])
end

@testset "AD derivatives and observed-information guard" begin
    model, theta = _joint_fixture(:gaussian)
    nll = t -> DRModels.prepared_joint_nll(model, t)
    ad_gradient = ForwardDiff.gradient(nll, theta)
    fd_gradient = _central_gradient(nll, theta)
    @test isapprox(ad_gradient, fd_gradient; atol = 1e-6, rtol = 1e-6)
    H = ForwardDiff.hessian(nll, theta)
    direction = collect(range(-0.4, 0.3; length = length(theta)))
    h = 1e-5
    directional_fd = (ForwardDiff.gradient(nll, theta .+ h .* direction) -
                      ForwardDiff.gradient(nll, theta .- h .* direction)) ./ (2h)
    @test isapprox(H * direction, directional_fd; atol = 1e-5, rtol = 1e-5)

    bernoulli_model, bernoulli_theta = _joint_fixture(:bernoulli)
    bernoulli_nll = t -> DRModels.prepared_joint_nll(bernoulli_model, t)
    @test isapprox(ForwardDiff.gradient(bernoulli_nll, bernoulli_theta),
                   _central_gradient(bernoulli_nll, bernoulli_theta); atol = 1e-6, rtol = 1e-6)
    bernoulli_H = ForwardDiff.hessian(bernoulli_nll, bernoulli_theta)
    bernoulli_direction = collect(range(-0.3, 0.4; length = length(bernoulli_theta)))
    bernoulli_fd = (ForwardDiff.gradient(bernoulli_nll, bernoulli_theta .+ h .* bernoulli_direction) -
                    ForwardDiff.gradient(bernoulli_nll, bernoulli_theta .- h .* bernoulli_direction)) ./ (2h)
    @test isapprox(bernoulli_H * bernoulli_direction, bernoulli_fd; atol = 1e-5, rtol = 1e-5)

    z = collect(range(-1.0, 1.0; length = 30))
    x = Union{Missing,Float64}[0.3 + 0.6 * z[i] + 0.15 * sin(i) for i in 1:30]
    y = Union{Missing,Float64}[0.2 + 0.5 * z[i] + 0.8 * x[i] + 0.12 * cos(i) for i in 1:30]
    x[25] = missing; y[7] = missing; y[30] = missing
    fit_model = DRModels.prepared_joint_model(y, x, hcat(ones(30), z), hcat(ones(30)), hcat(ones(30), z);
        predictor = :gaussian, original_row = collect(101:130))
    fitted = DRModels.fit_prepared_joint(fit_model; g_tol = 1e-7)
    @test fitted isa DRModels.PreparedJointFit
    @test fitted.fit.nobs == 28
    @test fitted.metadata.all_rows == 30
    @test fitted.metadata.predictor_only_rows == 2
    @test fitted.metadata.uncertainty_status == :not_computed
    @test fitted.fit.converged
    @test fitted.metadata.optimizer_status == :converged
    @test fitted.metadata.covariance_status == :observed_information_inverse
    fitted_gradient = ForwardDiff.gradient(fitted.fit.nll, fitted.fit.theta)
    @test norm(fitted_gradient, Inf) < 1e-5
    @test isapprox(ForwardDiff.hessian(fitted.fit.nll, fitted.fit.theta) * fitted.fit.vcov,
                   I; atol = 1e-4, rtol = 1e-4)
    fitted_nll = fitted.fit.nll(copy(fitted.fit.theta))
    fit_model.Xmu[1, 1] = 1e6
    fit_model.y[1] = -1e6
    @test fitted.fit.nll(fitted.fit.theta) == fitted_nll
    @test fitted.prepared.original_row[1] == 101
    summary = DRModels.joint_missing_summary(fitted)
    summary.original_row[1] = -1
    @test fitted.metadata.original_row[1] == 101
    @test summary.uncertainty_status == :not_computed
end

@testset "finite extremes and fail-closed input validation" begin
    extreme_model, extreme_theta = _joint_fixture(:bernoulli; eta_extreme = 1000.0)
    @test all(isfinite, DRModels.prepared_joint_rowloglik(extreme_model, extreme_theta))
    moments = DRModels.prepared_joint_conditional_moments(extreme_model, extreme_theta)
    @test 0 <= moments.mean[2] <= 1
    @test 0 <= moments.variance[2] <= 0.25
    low_model, low_theta = _joint_fixture(:bernoulli; eta_extreme = -1000.0)
    @test all(isfinite, DRModels.prepared_joint_rowloglik(low_model, low_theta))
    @test 0 <= DRModels.prepared_joint_conditional_moments(low_model, low_theta).variance[2] <= 0.25

    # A naive `(tau * sigma) / marginal_sd` overflows here although the
    # posterior SD is representable. The ratio order is part of the contract.
    gauss_model, gauss_theta = _joint_fixture(:gaussian)
    gauss_theta[3] = 1e100
    gauss_theta[4:5] .= log(1e160), 0.0
    gauss_theta[end] = log(1e160)
    large_scale = DRModels.prepared_joint_conditional_moments(gauss_model, gauss_theta)
    @test isfinite(large_scale.variance[2])
    @test large_scale.variance[2] ≈ 1e120 rtol = 1e-10

    # Centering at a large predictor prior loses the small posterior result.
    large_prior = DRModels.prepared_joint_model([1.0], [missing], ones(1, 1), ones(1, 1), ones(1, 1);
        predictor = :gaussian)
    large_theta = [0.0, 1.0, 0.0, 1e16, log(1e8)]
    large_mean = DRModels.prepared_joint_conditional_moments(large_prior, large_theta).mean[1]
    bm, bt, bs, bb, by = BigFloat(1e16), BigFloat(1e8), BigFloat(1), BigFloat(1), BigFloat(1)
    expected_big = (bs^2 / (bs^2 + bb^2 * bt^2)) * bm +
                   (bb * bt^2 / (bs^2 + bb^2 * bt^2)) * by
    @test large_mean ≈ Float64(expected_big) atol = 1e-12

    model, theta = _joint_fixture(:gaussian)
    @test_throws ArgumentError DRModels.prepared_joint_rowloglik(model, theta[1:end-1])
    nonfinite_theta = copy(theta); nonfinite_theta[1] = Inf
    @test_throws ArgumentError DRModels.prepared_joint_nll(model, nonfinite_theta)
    all_missing = DRModels.prepared_joint_model([missing], [missing], ones(1, 1), ones(1, 1), ones(1, 1);
        predictor = :gaussian)
    @test_throws ArgumentError DRModels.prepared_joint_rowloglik(all_missing, [Inf, 0.0, 0.0, 0.0, 0.0])
    @test_throws ArgumentError DRModels.fit_prepared_joint(model; g_tol = Inf)
    @test_throws ArgumentError DRModels.prepared_joint_model([1.0], [0.2], ones(1, 1), ones(1, 1), ones(1, 1);
        predictor = :bernoulli)
    @test_throws ArgumentError DRModels.prepared_joint_model([1.0], [missing], fill(Inf, 1, 1), ones(1, 1), ones(1, 1);
        predictor = :gaussian)
    @test_throws ArgumentError DRModels.prepared_joint_model([1.0, missing], [0.0, missing], ones(2, 1), ones(2, 1), ones(2, 1);
        predictor = :bernoulli, original_row = [4, 4])
    @test_throws ArgumentError DRModels.prepared_joint_model([1.0], [0.0], ones(1, 1), ones(1, 1), ones(1, 1);
        predictor = :gaussian, mu_names = ["mi(x)"])
end

println("S9_JOINT_PURE_PASS")
