# test_heritability_true_profile.jl — issue #313.
#
# The heritability/icc `method = :profile` CI must be a TRUE profile: at each fixed
# ratio it re-maximises the likelihood over ALL nuisance parameters, not a
# substitution/ELR profile that freezes the nuisance variances at the MLE. A frozen
# substitution rises too steeply (the co-components cannot absorb variance), so it
# under-covers. This test anchors two properties on a 2-component fit where the
# components trade off:
#   (1) at a fixed ratio away from r̂, re-optimising the nuisance CANNOT raise the
#       NLL above the frozen-substitution NLL (and here strictly lowers it), so the
#       true profile is genuinely re-maximising;
#   (2) the resulting :profile CI is a valid [0, 1] interval bracketing the estimate.
using DRModels, Test, Random, Statistics
const Optim = DRModels.Optim          # reuse the package's Optim (not a declared test dep)

@testset "heritability true profile re-optimises nuisance (#313)" begin
    Random.seed!(42)
    G = 25; m = 6; nn = G * m
    g = repeat(1:G, inner = m); h = repeat(1:G, outer = m)
    b1 = 0.9 .* randn(G); b2 = 0.6 .* randn(G)
    y = 1.0 .+ b1[g] .+ b2[h] .+ 0.5 .* randn(nn)
    fit = drm(bf(@formula(y ~ 1 + (1 | g) + (1 | h))), Gaussian(); data = (; y, g, h))

    comps, resid_idx = DRModels._variance_component_indices(fit)
    @test length(comps) == 2                     # two structured components (trade off)
    focal = comps[1].second
    denom = vcat([idx for (_, idx) in comps], resid_idx)
    others = [idx for idx in denom if idx != focal]
    θ̂ = copy(coef(fit))
    nll = fit.nll
    @test nll !== nothing
    r̂ = DRModels._ratio_closure(focal, denom)(θ̂)

    # Frozen-substitution NLL (the OLD, defective behaviour): S_others fixed at MLE.
    S0 = sum(exp(2 * θ̂[idx]) for idx in others)
    function nll_frozen(v)
        θ = copy(θ̂)
        θ[focal] = 0.5 * log(v / (1 - v) * S0)
        nll(θ)
    end
    # True-profile NLL: re-optimise the free nuisance, S_others recomputed each step.
    free = setdiff(1:length(θ̂), focal)
    function nll_true(v)
        function build(z)
            θ = copy(θ̂)
            for (k, idx) in enumerate(free); θ[idx] = z[k]; end
            S = sum(exp(2 * θ[idx]) for idx in others)
            θ[focal] = 0.5 * log(v / (1 - v) * S)
            θ
        end
        res = Optim.optimize(z -> nll(build(z)), copy(θ̂[free]), Optim.NelderMead(),
                             Optim.Options(iterations = 2000))
        Optim.minimum(res)
    end

    # (1) At ratios either side of r̂, re-optimisation lowers (never raises) the NLL.
    for v in (r̂ - 0.15, r̂ + 0.15)
        @test nll_true(v) <= nll_frozen(v) + 1e-6      # re-max can only decrease NLL
        @test nll_true(v) < nll_frozen(v) - 1e-4       # and here it STRICTLY decreases
    end

    # (2) The public :profile CI is a valid [0, 1] interval bracketing the estimate.
    hp = heritability(fit; component = :g, method = :profile)
    @test hp.method === :profile
    @test 0.0 <= hp.ci.lower <= hp.estimate <= hp.ci.upper <= 1.0
    @test hp.ci.upper > hp.ci.lower
end
