# Twin gap #760: residuals(fit; type = :quantile) on a mixed fit judged every
# row against fitted(fit) at the random intercept fixed at 0 (means[:mu], the
# _fit_*_ranef routes' fixed-effect-only mean), not a distribution that
# accounts for the random effect. On a correctly specified GLMM (chick
# survival, Binomial(1|BroodNo)) that inflates the PIT residuals' SD above 1
# purely because the reference ignores the brood effect (drmTMB #760 report:
# engine SD 1.0538 vs a hand-computed marginal SD 1.0144).
#
# `ranef()` (conditional modes) is not yet available for a non-Gaussian GLMM
# (gaussian_ranef.jl docstring), so the fix judges each row against the
# MARGINAL distribution: the random intercept integrated out by the same
# 32-node Gauss-Hermite quadrature the ranef fitter itself uses (documented in
# `_ranef_marginal_mix`, src/quantile_residuals.jl).
using DRModels
using Test, Random, Statistics
import Distributions

@testset "residuals(type=:quantile) on a Binomial GLMM uses the marginal reference (#760)" begin
    Random.seed!(20260760)
    G = 300; m = 20; n = G * m
    g = repeat(1:G, inner = m); x = randn(n)
    β = [0.1, 0.5]; σb = 0.8
    bg = σb .* randn(G)
    η = β[1] .+ β[2] .* x .+ bg[g]
    p = 1 ./ (1 .+ exp.(-η))
    y = Float64.(rand.(Distributions.Bernoulli.(p)))
    data = (; y, x, g)

    fit = drm(bf(@formula(y ~ x + (1 | g))), Binomial(); data = data)
    @test fit.converged

    q = residuals(fit; type = :quantile, rng = MersenneTwister(4))
    @test length(q) == n
    @test all(isfinite, q)

    # std(q) should sit near 1 (correctly calibrated against the marginal
    # reference), not inflated the way the b=0 reference was (the issue's own
    # engine-vs-hand-computed comparison: 1.0538 vs 1.0144 on n=1600).
    @test abs(std(q) - 1.0) < 0.08   # MC SE ≈ 1/√(2n) ≈ 0.0091 at n=6000; generous margin
                                      # for the σ_b plug-in vs the DGP's true σ_b

    # Direction check: reconstructing residuals against the WRONG (b = 0 /
    # fixed-effect-only) reference reproduces the documented inflation. Uses
    # the same discrete-PIT randomisation the engine uses (Binomial(1, μ)).
    rng3 = MersenneTwister(4)
    q_wrong = Vector{Float64}(undef, n)
    for i in 1:n
        d = Distributions.Binomial(1, fit.means[:mu][i])
        yi = round(Int, y[i])
        a = Distributions.cdf(d, yi - 1); b = Distributions.cdf(d, yi)
        u = clamp(a + (b - a) * rand(rng3), eps(), 1 - eps())
        q_wrong[i] = Distributions.quantile(Distributions.Normal(), u)
    end
    @test std(q_wrong) > std(q)             # the fixed-effect-only reference over-disperses
    @test std(q_wrong) - 1.0 > std(q) - 1.0 # ... in the documented direction
end
