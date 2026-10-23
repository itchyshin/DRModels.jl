# Gaussian random intercept (1 | g) on the mean. For a mean random effect the
# marginal stays exactly Gaussian: y ~ N(Xβ, D + σ_b² ZZ′), fit in closed form
# (no Laplace). Recovery test (Curie/Fisher): recover the fixed effects, the
# residual σ, and the random-intercept SD σ_b.
using DRModels
using Test, Random

@testset "Gaussian random intercept (1|g) — recovery" begin
    Random.seed!(20260601)
    G = 80; m = 20; n = G * m
    g = repeat(1:G, inner = m)
    x = randn(n)
    β = [0.5, -0.4]      # (Intercept), x
    σ = 0.7              # residual SD
    σb = 0.9             # random-intercept SD
    b = σb .* randn(G)
    y = β[1] .+ β[2] .* x .+ b[g] .+ σ .* randn(n)
    data = (; y, x, g)

    fit = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = data)

    @test coef(fit, :mu) ≈ β atol = 0.1
    @test exp(coef(fit, :sigma)[1]) ≈ σ atol = 0.1     # residual SD
    @test re_sd(fit)[:g] ≈ σb atol = 0.15              # random-intercept SD
    @test isfinite(loglik(fit))
end

@testset "Gaussian independent random slope (0 + x | g) — recovery" begin
    Random.seed!(20260604)
    G = 90; m = 25; n = G * m
    g = repeat(1:G, inner = m)
    x = randn(n)
    β = [0.4, 0.6]
    σ = 0.5              # residual SD
    σb = 0.7             # random-SLOPE SD
    b = σb .* randn(G)
    y = β[1] .+ β[2] .* x .+ b[g] .* x .+ σ .* randn(n)    # slope varies by group
    data = (; y, x, g)

    fit = drm(bf(@formula(y ~ x + (0 + x | g)), @formula(sigma ~ 1)), Gaussian(); data = data)

    @test coef(fit, :mu)[1] ≈ β[1] atol = 0.1                  # intercept (no RE on it)
    # the fixed slope estimates β₁ + the realized mean random slope (finite G):
    @test coef(fit, :mu)[2] ≈ β[2] + sum(b) / G atol = 0.1
    @test exp(coef(fit, :sigma)[1]) ≈ σ atol = 0.1             # residual SD
    @test re_sd(fit)[:g] ≈ σb atol = 0.15                      # random-slope SD
    @test isfinite(loglik(fit))
end

# #1114: all-singleton groups + `sigma ~ z` let the residual sigma run off to 0, so the
# ForwardDiff Hessian overflows to NaN. The (1 | g) route must follow the #968 convention
# (warn, all-NaN vcov, converged = false) instead of the bare LAPACK ArgumentError.
@testset "Gaussian (1|g) + sigma ~ z, all-singleton groups: non-finite Hessian (#1114)" begin
    Random.seed!(20261016100 + 87)   # rep 87 of this t2-style cell throws on main
    n = 50
    x = randn(n); z = randn(n); eps = randn(n); u = 0.5 .* randn(n)
    y = 1 .+ 0.5 .* x .+ u .+ exp.(-0.5 .+ 0.4 .* z) .* eps
    data = (; y, x, z, g = string.(1:n))
    fit = @test_logs (:warn,) match_mode = :any drm(
        bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ z)), Gaussian(); data = data)
    @test !is_converged(fit)
    @test all(isnan, vcov(fit))
end
