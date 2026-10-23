# Bivariate Gaussian location–scale with predictor-dependent residual
# correlation ρ12 — fixed effects. Recovery test (Curie): simulate a correlated
# bivariate response with known coefficients on μ1, μ2, log σ1, log σ2, and
# atanh(ρ12), fit via the keyword `bf(mu1=…, mu2=…, …)` front end, recover.
using DRModels
using Test, Random, Statistics

@testset "Bivariate Gaussian location-scale + rho12 — recovery" begin
    Random.seed!(20260531)
    n = 6000
    x = randn(n)
    βμ1 = [0.3, 0.5]
    βμ2 = [-0.2, 0.4]
    βσ1 = [-0.1, 0.2]      # on log σ1
    βσ2 = [0.0, -0.3]      # on log σ2
    βρ = [0.4, 0.3]        # on atanh(ρ12)  (ρ = tanh(η))

    μ1 = βμ1[1] .+ βμ1[2] .* x
    μ2 = βμ2[1] .+ βμ2[2] .* x
    σ1 = exp.(βσ1[1] .+ βσ1[2] .* x)
    σ2 = exp.(βσ2[1] .+ βσ2[2] .* x)
    ρ = tanh.(βρ[1] .+ βρ[2] .* x)

    z1 = randn(n); z2 = randn(n)
    y1 = μ1 .+ σ1 .* z1
    y2 = μ2 .+ σ2 .* (ρ .* z1 .+ sqrt.(1 .- ρ .^ 2) .* z2)
    data = (; y1, y2, x)

    fit = drm(bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
                 sigma1 = @formula(sigma1 ~ x), sigma2 = @formula(sigma2 ~ x),
                 rho12 = @formula(rho12 ~ x)), Gaussian(); data = data)

    @test coef(fit, :mu1) ≈ βμ1 atol = 0.06
    @test coef(fit, :mu2) ≈ βμ2 atol = 0.06
    @test coef(fit, :sigma1) ≈ βσ1 atol = 0.06
    @test coef(fit, :sigma2) ≈ βσ2 atol = 0.06
    @test coef(fit, :rho12) ≈ βρ atol = 0.10
    @test isfinite(loglik(fit))
end

# #324.3: the residual-σ seed must stay finite/sensible when the mean design is
# saturated (observed rows == mu coefficients → residuals are exactly 0). The
# pre-fix seed `log(sqrt(mean(r²)) + eps())` collapsed to log(eps()) ≈ −36, an
# extreme start where exp(−ls) overflows on the first nll evaluation. `_seed_ls`
# floors the seed variance at a small fraction of var(y) (with a 1e-8 backstop).
@testset "Bivariate residual-σ seed floor for saturated design (#324.3)" begin
    # Zero-residual (saturated) case: seed is floored, finite, and data-scaled.
    yobs = [1.0, 2.5, -0.7]
    s = DRModels._seed_ls(zeros(3), yobs)
    @test isfinite(s)
    @test s > -20                              # not the pre-fix log(eps()) ≈ −36
    @test s ≈ 0.5 * log(max(1e-6 * var(yobs), 1e-8))
    # Constant-response backstop (var(y) == 0) still finite via the 1e-8 floor.
    @test isfinite(DRModels._seed_ls(zeros(3), [2.0, 2.0, 2.0]))
    # Non-saturated case is unchanged: seed = 0.5·log(mean(r²)).
    r = [0.3, -0.4, 0.1, 0.2]
    @test DRModels._seed_ls(r, randn(4)) ≈ 0.5 * log(sum(r .^ 2) / 4)

    # End-to-end saturated bivariate fit completes with a finite logLik (the
    # pre-fix start overflowed the first objective evaluation).
    data = (; y1 = [0.5, -0.3, 1.2], y2 = [0.1, 0.4, -0.2],
            x = [-1.0, 0.0, 1.0], x2 = [1.0, 0.0, 1.0])
    fit_sat = drm(bf(mu1 = @formula(y1 ~ x + x2), mu2 = @formula(y2 ~ x),
                     sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                     rho12 = @formula(rho12 ~ 1)), Gaussian(); data = data)
    @test isfinite(loglik(fit_sat))
end
