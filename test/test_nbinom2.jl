# Negative-binomial (NB2) family: overdispersed counts. Log link on the mean μ;
# the `sigma` slot carries `log σ` with size θ = 1/σ² = exp(-2·coef(:sigma))
# (#315/#316, matching drmTMB). Var = μ + μ²/θ; as θ → ∞ it tends to Poisson.
# Fixed effects, ML. Mirrors drmTMB's `nbinom2`.
using DRModels
using Test, Random
import Distributions          # qualified — DRModels has its own family type

@testset "Negative binomial (NB2, overdispersed counts) — recovery" begin
    Random.seed!(20260616)
    n = 4000
    x = randn(n)
    β = [0.4, 0.5]; θ = 2.5                      # log μ = 0.4 + 0.5 x; dispersion θ
    μ = exp.(β[1] .+ β[2] .* x)
    y = Float64.([rand(Distributions.NegativeBinomial(θ, θ / (θ + μi))) for μi in μ])
    data = (; y, x)

    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), NegBinomial2(); data = data)

    @test coef(fit, :mu)[1] ≈ β[1] atol = 0.08    # log-mean intercept
    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.08    # log-mean slope
    @test exp(-2 * coef(fit, :sigma)[1]) ≈ θ atol = 0.8  # dispersion (size) — weakly identified
    @test isfinite(loglik(fit))
    @test all(fitted(fit) .> 0)
end

@testset "NB2 FE sigma ~ x does not DomainError under ForwardDiff (#385)" begin
    # Regression: Dual(p=1.0) fails Distributions' `p <= one(p)` during L-BFGS.
    Random.seed!(20260611)
    n = 200
    x = randn(n)
    μ = exp.(0.3 .+ 0.45 .* x)
    ησ = -0.10 .+ 0.25 .* x
    θ = exp.(-2 .* ησ)
    y = Float64.([rand(Distributions.NegativeBinomial(ti, ti / (ti + μi)))
                  for (ti, μi) in zip(θ, μ)])
    data = (; y, x)
    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), NegBinomial2(); data = data)
    @test length(coef(fit, :sigma)) == 2
    @test isfinite(loglik(fit))
    @test all(isfinite, coef(fit, :mu))
    @test all(isfinite, coef(fit, :sigma))
end
