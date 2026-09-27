# Twin gap #723: ZeroOneBeta() random intercept (1|g) on the mean was blocked
# ("ZeroOneBeta() currently supports fixed effects only") even though drmTMB
# fits `blotch ~ variety + (1|site), sigma ~ 1, zoi ~ 1, coi ~ 1` (leafblotch).
# Recovery test on a known DGP mirroring `test_beta_re.jl`'s pattern extended
# with zoi/coi atoms: b_g ~ N(0, σ_b²) on the logit mean, integrated out by
# 32-node Gauss–Hermite quadrature (same scheme as `_fit_beta_ranef`); zoi/coi/
# sigma stay fixed-effects-only, as in drmTMB's admitted ZOB + ordinary RI.
using DRModels
using Test, Random
import Distributions

@testset "ZeroOneBeta random intercept (1|g) — recovery (#723)" begin
    Random.seed!(20260723)
    G = 50; m = 40; n = G * m
    g = repeat(1:G, inner = m); x = randn(n)
    β = [0.2, 0.6]; φ = 15.0; σb = 0.5; zoi = 0.15; coi = 0.35
    bg = σb .* randn(G)
    μ = 1 ./ (1 .+ exp.(-(β[1] .+ β[2] .* x .+ bg[g])))
    y = Vector{Float64}(undef, n)
    for i in 1:n
        if rand() < zoi
            y[i] = rand() < coi ? 1.0 : 0.0
        else
            y[i] = rand(Distributions.Beta(μ[i] * φ, (1 - μ[i]) * φ))
        end
    end
    data = (; y, x, g)

    fit = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1),
                 @formula(zoi ~ 1), @formula(coi ~ 1)),
        ZeroOneBeta(); data = data)

    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.15                      # logit-mean slope
    @test exp(-2 * coef(fit, :sigma)[1]) ≈ φ atol = 6.0              # precision φ = 1/σ²
    @test 1 / (1 + exp(-coef(fit, :zoi)[1])) ≈ zoi atol = 0.05       # boundary probability
    @test 1 / (1 + exp(-coef(fit, :coi)[1])) ≈ coi atol = 0.08       # P(1 | boundary)
    @test re_sd(fit)[:g] ≈ σb atol = 0.20                            # group random-intercept SD
    @test isfinite(loglik(fit))
    @test fit.converged
end

@testset "ZeroOneBeta(): other formulas still refuse random effects (#723)" begin
    Random.seed!(1)
    n = 200; x = randn(n); g = rand(1:10, n)
    y = clamp.(rand(n) .* 0.9 .+ 0.05, 0.0, 1.0)
    data = (; y, x, g)
    @test_throws ErrorException drm(
        bf(@formula(y ~ x), @formula(sigma ~ 1 + (1 | g)), @formula(zoi ~ 1), @formula(coi ~ 1)),
        ZeroOneBeta(); data = data)
end
