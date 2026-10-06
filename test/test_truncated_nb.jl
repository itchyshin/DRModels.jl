# Truncated negative-binomial (NB2) family: strictly-positive counts (≥ 1) — no
# zeros possible (litter sizes, group sizes given presence). The likelihood is the
# zero-truncated NB2, P(k) = NB(k)/(1-NB(0)). Mirrors drmTMB's `truncated_nbinom2`.
using DRModels
using Test, Random
import Distributions

rtnb(r, p) = (while true; k = rand(Distributions.NegativeBinomial(r, p)); k > 0 && return k; end)

@testset "Truncated NB2 (positive counts ≥ 1) — recovery" begin
    Random.seed!(20260622)
    n = 5000; x = randn(n)
    βμ = [0.8, 0.3]; θ = 3.0
    μ = exp.(βμ[1] .+ βμ[2] .* x)
    y = Float64.([rtnb(θ, θ / (θ + μ[i])) for i in 1:n])     # zero-truncated NB draws
    @test all(y .>= 1)
    data = (; y, x)

    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), TruncatedNegBinomial2(); data = data)

    @test coef(fit, :mu)[1] ≈ βμ[1] atol = 0.15          # log-mean of the untruncated NB
    @test coef(fit, :mu)[2] ≈ βμ[2] atol = 0.10
    @test exp(-2 * coef(fit, :sigma)[1]) ≈ θ atol = 1.5        # dispersion — weakly identified under truncation
    @test isfinite(loglik(fit))
end
