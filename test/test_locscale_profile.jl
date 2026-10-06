# Profile-likelihood CIs for the location–scale fit (#202). The constrained inner
# solve is a trust-region Newton (robust on boundary steps) and is warm-started +
# Wald-seeded (fast); the χ²₁ crossing is found by a Venzon–Moolgavkar-style
# guarded-Newton root-find (deviance + envelope-theorem slope, bracket-safeguarded
# — #227 item C15, #209). Gates:
# (1) for the well-identified mean slope the profile CI ≈ the Wald CI (the
#     likelihood is near-quadratic) — and the profile NLL at an endpoint sits on
#     the χ²₁ threshold (the defining property);
# (2) the new VM endpoints match a reference bracket+bisection search to rtol 1e-2
#     on the well-identified mean slope (same root, fewer constrained solves);
# (3) a VARIANCE parameter (log L11) — where Wald is least trustworthy — yields a
#     finite, bracketed CI containing the estimate.
using DRModels
using Test, Random, LinearAlgebra, SparseArrays
import Distributions

_nb2_draw_pr(η, ψ) = (r = exp(ψ); μ = exp(η);
                      Float64(rand(Distributions.NegativeBinomial(r, r / (r + μ)))))

# Reference endpoint search: the pre-VM bracket-EXPANSION + fixed BISECTION
# schedule, kept here so the new guarded-Newton root-find can be checked to agree
# with it on a well-identified parameter (it must find the SAME χ²₁ crossing).
function _ls_profile_endpoint_bisect(kind, y, Xμ, Xψ, gidx, G, Q, θ̂; idx, dir,
                                     nll_min, se, level = 0.95,
                                     maxexpand = 40, nbisect = 30)
    thr = nll_min + Distributions.quantile(Distributions.Chisq(1), level) / 2
    z = Distributions.quantile(Distributions.Normal(), 1 - (1 - level) / 2)
    step = max(z * se, 1e-3)
    evalg(val) = (DRModels._ls_profile_nll(kind, y, Xμ, Xψ, gidx, G, Q, θ̂, idx, val)[1]) - thr
    a = θ̂[idx]; b = a; gb = -1.0
    for _ in 1:maxexpand
        b = θ̂[idx] + dir * step
        gb = evalg(b)
        gb > 0 && break
        a = b; step *= 1.6
    end
    gb > 0 || return dir > 0 ? Inf : -Inf
    for _ in 1:nbisect
        mid = (a + b) / 2
        evalg(mid) > 0 ? (b = mid) : (a = mid)
    end
    return (a + b) / 2
end

# Heavy reference cross-check: a 30-bisection endpoint search + the variance-
# boundary VM profile run several minutes of constrained Laplace re-solves, so it
# is opt-in to keep routine local `Pkg.test()` from stalling here (#202 perf
# follow-up). Routine profile-CI coverage comes from the σ-phylo boundary testset
# (test_gaussian_locscale_phylo_boundary.jl) and the public confint(:profile)
# testset below. Set DRM_SLOW_TESTS=1 to run this cross-check.
if get(ENV, "DRM_SLOW_TESTS", "0") == "1"
@testset "location–scale profile-likelihood CI" begin
    Random.seed!(2718)
    G = 20; m = 20; n = G * m
    species = repeat(1:G, inner = m)
    x = randn(n)
    Λt = [0.25 0.05; 0.05 0.16]
    LΛ = cholesky(Symmetric(Λt)).L
    A = [LΛ * randn(2) for _ in 1:G]
    Xμ = hcat(ones(n), x); Xψ = ones(n, 1)
    y = [_nb2_draw_pr(0.5 + 0.4x[i] + A[species[i]][1], 0.3 + A[species[i]][2]) for i in 1:n]
    Q = sparse(1.0 * I, G, G)

    fit = DRModels._fit_locscale(Val(:nb2), y, Xμ, Xψ, species, G, Q; se = true)
    nllmin = fit.nll

    # (1) Mean slope (idx = 2): profile CI ≈ Wald CI, and endpoint on the threshold.
    ci = DRModels._ls_profile_ci(Val(:nb2), y, Xμ, Xψ, species, G, Q, fit.θ; idx = 2, nll_min = nllmin)
    @test ci.lower < fit.θ[2] < ci.upper
    @test ci.lower ≈ fit.θ[2] - 1.96 * fit.se[2] rtol = 0.2
    @test ci.upper ≈ fit.θ[2] + 1.96 * fit.se[2] rtol = 0.2
    chi = Distributions.quantile(Distributions.Chisq(1), 0.95)
    dev_hi, _, ok_hi = DRModels._ls_profile_nll(Val(:nb2), y, Xμ, Xψ, species, G, Q, fit.θ, 2, ci.upper)
    @test ok_hi
    @test 2 * (dev_hi - nllmin) ≈ chi rtol = 1e-2

    # (2) VM guarded-Newton endpoints match the high-precision (30-bisection)
    #     reference. The guarded-Newton converges the DEVIANCE to ~1% of the χ²₁
    #     threshold (cf. the rtol=1e-2 deviance check above) — i.e. ~0.4% in the
    #     parameter on this endpoint — while the bisection nails the root to ~1e-9,
    #     so the cross-method agreement is at the Newton's deviance-convergence scale
    #     (rtol 1e-2), not tighter. The CI's χ²₁ validity is the assertion above; this
    #     is a same-root consistency check. (Tightening the VM stop is a follow-up.)
    lo_ref = _ls_profile_endpoint_bisect(Val(:nb2), y, Xμ, Xψ, species, G, Q, fit.θ;
                                         idx = 2, dir = -1.0, nll_min = nllmin, se = fit.se[2])
    hi_ref = _ls_profile_endpoint_bisect(Val(:nb2), y, Xμ, Xψ, species, G, Q, fit.θ;
                                         idx = 2, dir = +1.0, nll_min = nllmin, se = fit.se[2])
    @test ci.lower ≈ lo_ref rtol = 1e-2
    @test ci.upper ≈ hi_ref rtol = 1e-2

    # (3) Variance parameter (idx = 4 = log L11): finite bracketed CI.
    civ = DRModels._ls_profile_ci(Val(:nb2), y, Xμ, Xψ, species, G, Q, fit.θ; idx = 4, nll_min = nllmin)
    @test isfinite(civ.lower) && isfinite(civ.upper)
    @test civ.lower < fit.θ[4] < civ.upper
end
else
    @info "locscale profile reference cross-check skipped (~minutes); set DRM_SLOW_TESTS=1 to run"
end

# Public-API surface: confint(:profile) on a drm()-fitted location–scale model
# routes to the robust profiler (#202, #209 item 2). The DrmFit covariance block
# is in :recov order; the router permutes to the engine packing — so the per-
# coefficient CIs must line up with the right parameters.
#
# This profiles the FULL parameter vector including the covariance block, whose
# near-boundary inner solves run ~8+ min on Apple BLAS — opt-in so it does not
# dominate routine `Pkg.test()`. Set DRM_SLOW_TESTS=1 to run. (Perf follow-up:
# cache the inner-mode factorisation across warm-started profile points.)
if get(ENV, "DRM_SLOW_TESTS", "0") == "1"
@testset "location–scale profile CI via public confint(:profile)" begin
    Random.seed!(2026)
    G = 25; m = 25; n = G * m
    species = repeat(1:G, inner = m)
    x = randn(n)
    Λt = [0.25 0.05; 0.05 0.16]
    LΛ = cholesky(Symmetric(Λt)).L
    A = [LΛ * randn(2) for _ in 1:G]
    y = [_nb2_draw_pr(0.5 + 0.4x[i] + A[species[i]][1], 0.3 + 0.2x[i] + A[species[i]][2])
         for i in 1:n]
    data = (; y, x, species)
    fit = drm(bf(@formula(y ~ x + (1 | p | species)),
                 @formula(sigma ~ x + (1 | p | species))), NegBinomial2(); data = data)

    wald = confint(fit)                                   # :wald (all coefs)
    # Profile the FULL parameter vector — including the variance/covariance block.
    # Profiling a covariance parameter toward its boundary can drive Λ near-
    # singular; the VM root-finder treats that infeasible region as a boundary
    # (endpoint → ±Inf for an unbounded direction) rather than crashing, so the
    # full-vector call must run without error (#227 item C15, #209).
    prof = confint(fit; method = :profile)
    @test length(prof) == length(wald)
    @test [r.coef for r in prof] == [r.coef for r in wald]
    # Every endpoint is well-defined (finite or ±Inf), never NaN, and brackets the
    # estimate where finite (an unbounded side gives ±Inf, still a valid bracket).
    for r in prof
        @test !isnan(r.lower) && !isnan(r.upper)
        @test r.lower ≤ r.estimate ≤ r.upper
        @test isfinite(r.estimate)
    end

    # Well-identified mean slope: profile ≈ Wald (near-quadratic likelihood) and a
    # finite, bracketing interval.
    prof_mu = [r for r in prof if r.param === :mu]
    waldmu = [r for r in wald if r.param === :mu]
    pslope = first(r for r in prof_mu if r.coef == "x")
    wslope = first(r for r in waldmu if r.coef == "x")
    @test isfinite(pslope.lower) && isfinite(pslope.upper)
    @test pslope.lower < pslope.estimate < pslope.upper
    @test pslope.lower ≈ wslope.lower rtol = 0.35
    @test pslope.upper ≈ wslope.upper rtol = 0.35

    # profile_result audit surface reports the locscale backend, on the full vector.
    res = profile_result(fit)
    @test res.autodiff === :locscale
    @test res.attempted == length(prof)
    @test [r.coef for r in res.ci] == [r.coef for r in prof]
    # The lower/upper unbounded flags agree with the returned endpoints.
    for (s, r) in zip(res.stats, res.ci)
        @test s.lower_unbounded == !isfinite(r.lower)
        @test s.upper_unbounded == !isfinite(r.upper)
    end

    # check_drm reports a real gradient norm for the location–scale fit
    # (exact analytic gradient; ForwardDiff can't pierce the Float64 inner solve).
    rep = check_drm(fit)
    @test isfinite(rep.max_abs_grad)
end
else
    @info "locscale full-vector confint(:profile) testset skipped (~8+ min); set DRM_SLOW_TESTS=1 to run"
end
