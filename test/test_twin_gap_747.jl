# test_twin_gap_747.jl -- DRModels.jl #746 / #747 (twins of drmTMB #1289 / #1290).
#
# THE DEFECT. Gaussian location-scale with a mean random intercept `(1 | g)` and
# `sigma ~ x` (or `sigma ~ x + x2`), routed through the exact Woodbury marginal in
# `_fit_ranef_gaussian` (src/gaussian_ranef.jl), returned logLik ~ +1e24 ... +1e125
# with sigma coefficients at +-O(10-50) on ~20% of draws. drmTMB (TMB Laplace, which
# is exact for this Gaussian model) fits the same data sensibly.
#
# ROOT CAUSE. The quadratic r'V^-1 r was formed as the DIFFERENCE
# q1 - q2 = r'D^-1 r - sum_k C_k^2 / M_k. Both terms scale like 1/D_min; once one
# sigma_i is small, their rounding error exceeds the true O(100) value, the computed
# nll goes hugely NEGATIVE, and LBFGS descends into that rounding hole. The
# likelihood itself is bounded. Fix: `_re_quad_stable` evaluates the identical
# quantity as a sum of non-negative terms (penalised RSS at the conditional mode,
# with one refinement step) -- used by the ML and REML objectives of
# `_fit_ranef_gaussian` and `_fit_ranef_gaussian_lss` (the `sd(g) ~ z` route).
#
# MEASURED (macOS/aarch64, Julia 1.10, 2026-09-27) on the StableRNG DGM in
# twin_gap_747_dgm.jl, seeds 1:30 per shape (60 fits):
#   origin/main  12/60 absurd (logLik +1.1e3 ... +6.8e125), all |sigma coef| >= 9.
#   this fix      0/60; every logLik matches drmTMB 0.7.1 to <= 3.6e-13, all
#                 gradient-converged.
# The reference values below are drmTMB 0.7.1 generated OUTPUTS on the same data
# (no drmTMB source), integrator on both sides = exact Gaussian marginal (TMB
# Laplace is exact here; DRModels.jl uses the closed-form Woodbury marginal).
#
#   julia --project=test -e 'using DRModels, Test; include("test/test_twin_gap_747.jl")'

module TestTwinGap747

using DRModels
using Test
using LinearAlgebra
using Random
using Logging

include(joinpath(@__DIR__, "twin_gap_747_dgm.jl"))

# (shape, seed, drmTMB logLik, drmTMB sigma intercept, drmTMB sigma slope on x).
# Every seed except bal/2 and unb/3 (healthy controls) returned an absurd
# positive logLik on origin/main before this fix.
const REF = [
    (:bal, 1, -101.1195894854, -0.413837, 0.234239),
    (:bal, 2, -95.3520852876, -0.438908, 0.173047),
    (:bal, 6, -106.3460873516, -0.477073, 0.297359),
    (:bal, 12, -106.0343404485, -0.432656, 0.337464),
    (:bal, 13, -116.4045101880, -0.356147, 0.399190),
    (:bal, 17, -99.4180508321, -0.422816, 0.238736),
    (:bal, 24, -101.7792990744, -0.451936, 0.219991),
    (:bal, 26, -90.9259689431, -0.606338, 0.222320),
    (:bal, 29, -100.4147722397, -0.475224, 0.215843),
    (:unb, 3, -92.2230604035, -0.417080, 0.271087),
    (:unb, 4, -111.7134759180, -0.170878, 0.303669),
    (:unb, 11, -110.5815551494, -0.169800, 0.249622),
    (:unb, 13, -114.0369732168, -0.062672, 0.154238),
    (:unb, 18, -106.8422849179, -0.273581, 0.183128),
]

# Dense reference r'V^-1 r for V = D + Z diag(sb2) Z', Z_ik = w_i [g_i = k].
function _dense_quad(r, invD, w, gidx, sb2)
    n = length(r); G = length(sb2)
    Z = zeros(eltype(r), n, G)
    for i in 1:n
        Z[i, gidx[i]] = w === nothing ? one(eltype(r)) : w[i]
    end
    V = Diagonal(1 ./ invD) + Z * Diagonal(sb2) * Z'
    return dot(r, Symmetric(V) \ r)
end

function _SC(r, invD, w, gidx, G)
    S = zeros(G); C = zeros(G)
    for i in eachindex(r)
        wi = w === nothing ? 1.0 : w[i]
        S[gidx[i]] += wi^2 * invD[i]; C[gidx[i]] += wi * r[i] * invD[i]
    end
    return S, C
end

@testset "#746/#747: cancellation-free Woodbury quadratic" begin
    rng = MersenneTwister(747)
    n, G = 40, 5
    gidx = repeat(1:G, inner = 8)
    r = randn(rng, n)
    w = 1 .+ rand(rng, n)
    sb2 = 0.2 .+ rand(rng, G)

    # (a) Well-conditioned: equals the dense r'V^-1 r (and hence the old q1 - q2).
    invD = exp.(randn(rng, n))
    for ww in (nothing, w)
        S, C = _SC(r, invD, ww, gidx, G)
        q = DRModels._re_quad_stable(r, invD, ww, gidx, 1 ./ sb2, S, C)
        @test q ≈ _dense_quad(r, invD, ww, gidx, sb2) rtol = 1e-12
    end

    # (b) The runaway geometry: one residual precision per group ~1e120. The old
    # difference q1 - q2 is pure rounding noise here (it can be hugely negative);
    # the stable form must stay non-negative and agree with a BigFloat dense solve.
    invDx = copy(invD)
    for k in 1:G
        invDx[8 * (k - 1) + 1] = 1e120
    end
    S, C = _SC(r, invDx, nothing, gidx, G)
    q = DRModels._re_quad_stable(r, invDx, nothing, gidx, 1 ./ sb2, S, C)
    qbig = _dense_quad(big.(r), big.(invDx), nothing, gidx, big.(sb2))
    @test q >= 0
    @test q ≈ Float64(qbig) rtol = 1e-8
    q_old = sum(r .^ 2 .* invDx) - sum(C .^ 2 ./ (1 ./ sb2 .+ S))
    @test abs(q_old - Float64(qbig)) > 1e6 * abs(q - Float64(qbig))   # documents the old failure
end

@testset "#746/#747: Gaussian (1|g) + sigma ~ x twins drmTMB (no runaway)" begin
    for (kind, seed, ll_r, s0_r, s1_r) in REF
        d = _tg747_draw(seed, kind)
        fit = with_logger(NullLogger()) do
            drm(_tg747_formula(kind), Gaussian(); data = d)
        end
        ll = loglik(fit)
        sc = coef(fit, :sigma)
        @testset "$kind seed $seed" begin
            @test isfinite(ll)
            @test fit.converged
            @test ll ≈ ll_r atol = 1e-6
            @test sc[1] ≈ s0_r atol = 1e-4
            @test sc[2] ≈ s1_r atol = 1e-4
        end
    end
end

@testset "#746/#747: REML objective shares the stable quadratic" begin
    # bal seed 17 was +4.2e124 under ML on origin/main; REML reuses `nll_ml`.
    d = _tg747_draw(17, :bal)
    fit = with_logger(NullLogger()) do
        drm(_tg747_formula(:bal), Gaussian(); data = d, method = :REML)
    end
    @test isfinite(loglik(fit))
    @test loglik(fit) < 0
    @test fit.converged
    @test all(abs.(coef(fit, :sigma)) .< 2)
end

end # module
