using DRModels
using ForwardDiff
using LinearAlgebra
using SpecialFunctions: loggamma
using Test

const rho_prior_data = (
    y1=[0.5, -0.2, 0.9, 1.1, -0.3, 0.4],
    y2=[0.1, 0.4, -0.5, 0.3, 0.8, -0.2],
)
const rho_prior_formula = bf(
    mu1=@formula(y1 ~ 1), mu2=@formula(y2 ~ 1),
    sigma1=@formula(sigma1 ~ 1), sigma2=@formula(sigma2 ~ 1),
    rho12=@formula(rho12 ~ 1),
)
const rho_prior_table = (
    prior=["lkj(2)"], class=["correlation"], coef=["(Intercept)"],
    dpar=["rho12"], group=[""], scale=["natural_correlation"],
)

function rho_prior_dense_data_nll(theta, guard)
    rho = guard * tanh(theta[5])
    om = 1 - rho^2
    sum(begin
        z1 = (rho_prior_data.y1[i] - theta[1]) * exp(-theta[3])
        z2 = (rho_prior_data.y2[i] - theta[2]) * exp(-theta[4])
        log(2π) + theta[3] + theta[4] + 0.5log(om) +
            0.5 * (z1^2 - 2rho*z1*z2 + z2^2) / om
    end for i in eachindex(rho_prior_data.y1))
end

@testset "bivariate fixed coefficient priors compose with rho12 LKJ" begin
    mixed_table = (
        prior=["normal(0,2)", "student_t(0,1.5,4)", "lkj(2)"],
        class=["fixef", "fixef", "correlation"],
        coef=["(Intercept)", "(Intercept)", "(Intercept)"],
        dpar=["mu1", "mu2", "rho12"], group=["", "", ""],
        scale=["link", "link", "natural_correlation"],
    )
    fit = drm(rho_prior_formula, Gaussian(); data=rho_prior_data, priors=mixed_table)
    rows = resolve_priors(rho_prior_formula, Gaussian(); data=rho_prior_data,
        priors=mixed_table)
    @test length(rows) == 3
    @test [r.dpar for r in rows] == [:mu1, :mu2, :rho12]
    @test fit.penalty.kind === :mixed_priors
    @test fit.penalty.correlation_row.class === :residual_correlation
    @test fit.estim_method === :MAP
    @test fit.loglik ≈ -fit.penalty.data_nll(fit.theta) atol=1e-10
    @test fit.penalty.objective ≈ fit.nll(fit.theta) atol=1e-10
    @test fit.penalty.log_prior ≈ -(fit.penalty.fixed_penalty + fit.penalty.correlation_penalty) atol=1e-10

    # Frozen dense Gaussian likelihood, plus independently normalized Normal,
    # Student-t, and natural-correlation LKJ densities.
    for theta in ([0.2, -0.1, -0.3, 0.15, 0.25],
                  [-0.2, 0.3, -0.1, -0.4, -0.5],
                  [0.0, 0.0, 0.1, -0.2, 0.0],
                  [0.4, 0.2, -0.5, -0.1, 0.7],
                  [-0.1, -0.2, -0.2, 0.25, -0.3])
        rho = fit.penalty.correlation_row.guard * tanh(theta[5])
        n_fixed1 = log(2.0) + 0.5log(2π) + 0.5 * (theta[1] / 2)^2
        z2 = theta[2] / 1.5
        n_fixed2 = log(1.5) + 0.5log(4π) + loggamma(2.0) - loggamma(2.5) +
            2.5 * log1p(z2^2 / 4)
        n_rho = -log1p(-rho^2) + log(2 * (fit.penalty.correlation_row.guard -
            fit.penalty.correlation_row.guard^3 / 3))
        expected = rho_prior_dense_data_nll(theta, fit.penalty.correlation_row.guard) +
            n_fixed1 + n_fixed2 + n_rho
        @test fit.nll(theta) ≈ expected atol=1e-10
        grad_ad = ForwardDiff.gradient(fit.nll, theta)
        hess_ad = ForwardDiff.hessian(fit.nll, theta)
        h = 1e-4
        grad_fd = similar(grad_ad); hess_fd = similar(hess_ad); f0 = fit.nll(theta)
        for i in eachindex(theta)
            ei = zeros(length(theta)); ei[i] = h
            plus = fit.nll(theta .+ ei); minus = fit.nll(theta .- ei)
            grad_fd[i] = (plus - minus) / (2h)
            hess_fd[i, i] = (plus - 2 * f0 + minus) / h^2
            for j in 1:(i - 1)
                ej = zeros(length(theta)); ej[j] = h
                hess_fd[i, j] = hess_fd[j, i] =
                    (fit.nll(theta .+ ei .+ ej) - fit.nll(theta .+ ei .- ej) -
                     fit.nll(theta .- ei .+ ej) + fit.nll(theta .- ei .- ej)) / (4h^2)
            end
        end
        @test maximum(abs.(grad_ad .- grad_fd)) <= 1e-6
        @test maximum(abs.(hess_ad .- hess_fd) ./ max.(1, abs.(hess_fd))) <= 1e-5
    end
    @test_throws ArgumentError confint(fit)
    replay = update(fit, rho_prior_formula; data=rho_prior_data)
    @test replay.penalty.specs == fit.penalty.specs
    @test replay.penalty.input_table == mixed_table
    @test replay.theta ≈ fit.theta atol=1e-7

    bad_residual = drm_prior_residual_sd_gamma(dpar=:sigma, coef="(Intercept)", shape=2, rate=3)
    @test_throws ArgumentError drm(rho_prior_formula, Gaussian(); data=rho_prior_data,
        priors=drm_priors(bad_residual))
end

@testset "bivariate residual LKJ prior fit" begin
    plain = drm(rho_prior_formula, Gaussian(); data=rho_prior_data)
    fit = drm(rho_prior_formula, Gaussian(); data=rho_prior_data, priors=rho_prior_table)
    constructor_fit = drm(rho_prior_formula, Gaussian(); data=rho_prior_data,
        priors=drm_priors(drm_prior_lkj(dpar=:rho12, eta=2)))
    resolved = resolve_priors(rho_prior_formula, Gaussian(); data=rho_prior_data,
        priors=rho_prior_table)
    row = fit.penalty.correlation_row

    @test fit.estim_method === :MAP
    @test fit.penalty.kind === :correlation_priors
    @test row.class === :residual_correlation
    @test row.dpar === :rho12
    @test row.block === :bivariate_residual
    @test row.scale === :natural_correlation
    @test row.guard == DRModels.DRM_RHO12_PRIOR_GUARD
    @test fit.loglik ≈ -fit.penalty.data_nll(fit.theta) atol=1e-10
    @test fit.penalty.objective ≈ fit.nll(fit.theta) atol=1e-10
    @test fit.penalty.log_prior ≈ -fit.penalty.correlation_penalty atol=1e-10
    @test constructor_fit.theta ≈ fit.theta atol=1e-7
    @test only(resolved).class === :correlation
    @test only(resolved).scale === :natural_correlation
    constructor_resolved = resolve_priors(rho_prior_formula, Gaussian(); data=rho_prior_data,
        priors=drm_priors(drm_prior_lkj(dpar=:rho12, eta=2)))
    @test only(constructor_resolved) == only(resolved)
    @test only(DRModels._resolve_priors_native(rho_prior_formula, Gaussian();
        data=rho_prior_data,
        priors=drm_priors(drm_prior_lkj(dpar=:rho12, eta=2)))).class === :residual_correlation
    @test only(resolved).index == length(fit.theta)
    @test plain.nll(plain.theta) ≈
        rho_prior_dense_data_nll(plain.theta, DRModels.RHO_GUARD) atol=1e-10

    # Independent row-wise bivariate-normal NLL plus one normalized LKJ(2)
    # density on the guarded natural correlation. The fit objective remains
    # replayable at arbitrary interior points.
    for theta in ([0.2, -0.1, -0.3, 0.15, 0.25],
                  [-0.2, 0.3, -0.1, -0.4, -0.5],
                  [0.0, 0.0, 0.1, -0.2, 0.0],
                  [0.4, 0.2, -0.5, -0.1, 0.7],
                  [-0.1, -0.2, -0.2, 0.25, -0.3])
        rho = row.guard * tanh(theta[5])
        data_nll = rho_prior_dense_data_nll(theta, row.guard)
        prior_nll = -log1p(-rho^2) + log(2 * (row.guard - row.guard^3 / 3))
        @test fit.nll(theta) ≈ data_nll + prior_nll atol=1e-10

        grad_ad = ForwardDiff.gradient(fit.nll, theta)
        hess_ad = ForwardDiff.hessian(fit.nll, theta)
        h = 1e-4
        grad_fd = similar(grad_ad)
        hess_fd = similar(hess_ad)
        f0 = fit.nll(theta)
        for i in eachindex(theta)
            ei = zeros(length(theta)); ei[i] = h
            plus = fit.nll(theta .+ ei)
            minus = fit.nll(theta .- ei)
            grad_fd[i] = (plus - minus) / (2h)
            hess_fd[i, i] = (plus - 2 * f0 + minus) / h^2
            for j in 1:(i - 1)
                ej = zeros(length(theta)); ej[j] = h
                hess_fd[i, j] = hess_fd[j, i] =
                    (fit.nll(theta .+ ei .+ ej) - fit.nll(theta .+ ei .- ej) -
                     fit.nll(theta .- ei .+ ej) + fit.nll(theta .- ei .- ej)) / (4h^2)
            end
        end
        @test maximum(abs.(grad_ad .- grad_fd)) <= 1e-6
        @test maximum(abs.(hess_ad .- hess_fd) ./ max.(1, abs.(hess_fd))) <= 1e-5
    end

    gradient = ForwardDiff.gradient(fit.nll, fit.theta)
    @test all(isfinite, gradient)
    @test norm(fit.theta - plain.theta) > 0
    @test fit.loglik != plain.loglik

    replay = update(fit, rho_prior_formula; data=rho_prior_data)
    @test replay.estim_method === :MAP
    @test replay.penalty.kind === :correlation_priors
    @test replay.penalty.specs == fit.penalty.specs
    @test replay.penalty.input_table == rho_prior_table
    @test replay.theta ≈ fit.theta atol=1e-7
    @test_throws ArgumentError confint(fit)

    @test_throws ArgumentError drm(rho_prior_formula, Gaussian();
        data=rho_prior_data, priors=rho_prior_table, method=:REML)
    @test_throws ArgumentError drm(rho_prior_formula, Gaussian();
        data=rho_prior_data, priors=rho_prior_table, V=zeros(6, 2, 2))
    @test_throws ArgumentError drm(rho_prior_formula, Gaussian();
        data=rho_prior_data, priors=rho_prior_table, weights=fill(2.0, 6))
    varying = bf(mu1=@formula(y1 ~ 1), mu2=@formula(y2 ~ 1),
        sigma1=@formula(sigma1 ~ 1), sigma2=@formula(sigma2 ~ 1),
        rho12=@formula(rho12 ~ 1 + x))
    @test_throws ArgumentError drm(varying, Gaussian();
        data=merge(rho_prior_data, (x=collect(range(-1, 1; length=6)),)),
        priors=rho_prior_table)
    structured = bf(
        mu1=@formula(y1 ~ 1 + phylo(1 | site)),
        mu2=@formula(y2 ~ 1 + phylo(1 | site)),
        sigma1=@formula(sigma1 ~ 1), sigma2=@formula(sigma2 ~ 1),
        rho12=@formula(rho12 ~ 1),
    )
    @test_throws ArgumentError drm(structured, Gaussian();
        data=merge(rho_prior_data, (site=[:a, :a, :b, :b, :c, :c],)),
        priors=rho_prior_table)
end
