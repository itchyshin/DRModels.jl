# Beta family: responses on the open interval (0,1) — proportions, rates. Logit
# link on the mean μ; the `sigma` slot carries σ with the drmTMB precision
# mapping φ = 1/σ² (so a Beta(μφ, (1-μ)φ) likelihood). Fixed effects, ML.
using DRModels
using Test, Random
import Distributions          # qualified — DRModels has its own `Beta` family type

@testset "Beta (proportions, logit link) — recovery" begin
    Random.seed!(20260617)
    n = 3000
    x = randn(n)
    β = [0.3, 0.8]; φ = 15.0                         # logit μ = 0.3 + 0.8x; precision φ
    μ = 1 ./ (1 .+ exp.(-(β[1] .+ β[2] .* x)))
    y = Float64.([rand(Distributions.Beta(μi * φ, (1 - μi) * φ)) for μi in μ])
    data = (; y, x)

    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Beta(); data = data)

    @test coef(fit, :mu)[1] ≈ β[1] atol = 0.08       # logit-mean intercept
    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.08       # logit-mean slope
    φ̂ = exp(-2 * coef(fit, :sigma)[1])               # φ = 1/σ²
    @test φ̂ ≈ φ atol = 4.0                            # precision — weakly identified
    @test isfinite(loglik(fit))
    @test all(0 .< fitted(fit) .< 1)                 # fitted means are proportions
end
