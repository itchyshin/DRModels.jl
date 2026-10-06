# Tweedie family (compound Poisson–Gamma, 1 < p < 2): semicontinuous responses —
# a point mass at exactly 0 plus a continuous positive part (biomass, rainfall,
# insurance loss). Log link on the mean μ; `sigma` is √dispersion (φ = σ²); `nu`
# is the power p, estimated on a logit-(1,2) link. Density via the Dunn–Smyth
# series. Mirrors drmTMB's `tweedie`. Fixed effects, ML.
using DRModels
using Test, Random
import Distributions

# simulate a Tweedie variate as a compound Poisson–Gamma
function _rtweedie(μ, φ, p)
    λ = μ^(2 - p) / (φ * (2 - p)); γ = φ * (p - 1) * μ^(p - 1); sh = (2 - p) / (p - 1)
    N = rand(Distributions.Poisson(λ))
    N == 0 ? 0.0 : rand(Distributions.Gamma(N * sh, γ))
end

@testset "Tweedie (semicontinuous, 1<p<2) — recovery" begin
    Random.seed!(20260625)
    n = 4000; x = randn(n)
    β = [0.5, 0.3]; φ = 2.0; p = 1.5
    μ = exp.(β[1] .+ β[2] .* x)
    y = [_rtweedie(μ[i], φ, p) for i in 1:n]
    @test count(==(0.0), y) > 0           # genuine exact zeros
    data = (; y, x)

    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)), Tweedie(); data = data)

    @test coef(fit, :mu)[1] ≈ β[1] atol = 0.08         # log-mean intercept
    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.08         # log-mean slope
    @test exp(2 * coef(fit, :sigma)[1]) ≈ φ atol = 0.4 # dispersion φ = σ²
    @test 1 + 1 / (1 + exp(-coef(fit, :nu)[1])) ≈ p atol = 0.12   # power p via logit-(1,2)
    @test isfinite(loglik(fit))
end
