# test_unbounded_scale_spike.jl -- DRModels.jl#1059 (parity with drmTMB#1469).
#
# A Gaussian `sigma ~ x` fit can drive the scale of ONE observation to zero while
# the location predictor passes through that observation exactly. The likelihood
# then has no finite maximum, and `Optim.converged` still says true. The old
# `_nondegenerate_fit` bar compared the LARGEST fitted scale with the response
# scale, so it only caught the case where every scale collapsed; here the other
# scales stay ordinary, so the fit passed as converged.
#
# The rule is unit-free: a ratio of fitted scales above 1e4 together with the
# smallest-scale observation fitted exactly (|y - mu| / sigma < 1e-3). These
# tests build the per-row vectors directly, so they are deterministic and do not
# depend on where an optimiser stops; they assert on the public `is_converged`.

using Test
using Random
using DRModels
using DRModels: is_converged

@testset "#1059 an exactly fitted, collapsing scale is degenerate" begin
    rng = MersenneTwister(1059)
    n = 40
    x = randn(rng, n)
    y = 0.3 .+ 0.5 .* x .+ randn(rng, n) .* exp.(0.1 .* x)
    f = bf(DRModels.@formula(y ~ x), DRModels.@formula(sigma ~ x))
    template = drm(f, Gaussian(); data = (y = y, x = x))
    @test is_converged(template)               # the honest fit is fine

    # Build a fit that has the same optimiser flag but the per-row signature of
    # the spike: row 1 is fitted exactly with a vanishing scale.
    function with_rows(fit; means, obs, scales)
        DRModels.DrmFit(fit.family, fit.blocks, fit.coefnames, fit.theta, fit.vcov,
            fit.loglik, fit.nobs, true, Dict(:mu => means), Dict(:mu => obs),
            Dict(:sigma => scales))
    end
    μ = 0.3 .+ 0.5 .* x
    s = exp.(0.1 .* x)
    ynow = μ .+ s .* randn(MersenneTwister(2), n)

    spike_s = copy(s); spike_s[1] = 1e-9
    spike_y = copy(ynow); spike_y[1] = μ[1] + 1e-13          # |z| = 1e-4 < 1e-3
    spike = with_rows(template; means = μ, obs = spike_y, scales = spike_s)
    @test !is_converged(spike)

    # Unit-free: the same fit expressed in other units gets the same verdict.
    for c in (1e-3, 1e3)
        scaled = with_rows(template; means = c .* μ, obs = c .* spike_y,
                           scales = c .* spike_s)
        @test !is_converged(scaled)
        ordinary = with_rows(template; means = c .* μ, obs = c .* ynow, scales = c .* s)
        @test is_converged(ordinary)
    end

    # A wide scale ratio alone is not enough: the smallest-scale row must be
    # fitted exactly. Here its standardized residual is 1.2, an ordinary draw.
    wide_y = copy(ynow); wide_y[1] = μ[1] + 1.2 * spike_s[1]
    wide = with_rows(template; means = μ, obs = wide_y, scales = spike_s)
    @test is_converged(wide)

    # An exact fit alone is not enough either: the scales must span > 1e4.
    mild_s = copy(s); mild_s[1] = 0.05                        # ratio ~ 30
    mild_y = copy(ynow); mild_y[1] = μ[1] + 1e-6
    @test is_converged(with_rows(template; means = μ, obs = mild_y, scales = mild_s))

    # A fit with conditional random effects keeps its earlier behaviour (the
    # exact-fit test needs the marginal residual, which this route does not store).
    with_re = DRModels._withranef(spike, Dict(:g => zeros(3)))
    @test is_converged(with_re)
end
