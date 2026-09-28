# test_twin_gap_747_lss_reml.jl -- docs regression for the location-scale-scale
# REML route after #746/#747 (docs/src/tutorials/location-scale-scale.md, "REML
# on location-scale-scale models").
#
# THE DEFECT (found while repairing the docs CI failure on PR #835). `_fit_ranef_gaussian_lss`'s
# REML objective (src/gaussian_lss.jl) shares `_re_quad_stable` (src/gaussian_ranef.jl)
# with the plain `_fit_ranef_gaussian` ML/REML objectives. An unconstrained LBFGS line-search
# probe can push a group's log sd(g) coefficient to a large negative value, driving
# sigma_b,k -> 0 exactly and invsigb2[k] = 1/sigma_b,k^2 -> Inf. `_re_quad_stable`'s
# one-step Newton refinement then computed `u[k] * invsigb2[k]` with `u[k] == 0.0`
# (from `C[k] / M[k]` with `M[k] == Inf`), i.e. `0.0 * Inf == NaN`, which failed
# LineSearches.HagerZhang's `isfinite(phi_c) && isfinite(dphi_c)` assertion during
# `drm(...; method = :REML)` on the tutorial's `sd(id) ~ sex` model (CI run
# 36352178108, docs job). FIX: skip the refinement/quadratic contribution for a
# group `k` when `invsigb2[k]` is not finite -- sigma_b,k -> 0 pins the random
# intercept at its unrefined value (already 0 from `C[k] / M[k]`) and contributes
# nothing to the quadratic form in that limit, so no correctness is lost.
#
# WHY THIS TEST NO LONGER PINS EXACT COEFFICIENTS (found while chasing a
# Julia-1.13-only CI failure, 2026-09-27). The original version of this file
# pinned `coef(fit_reml, :sigma)` / `:sd` (and even `ll` / `:mu`) to values
# measured on Julia 1.10 at `atol = 1e-6`. On Julia 1.13 every one of those
# assertions failed, and by far more than floating-point noise: `ll` differed
# by ~4.1 (-408.179 on 1.13 vs -412.301 on 1.10), `mu` by up to ~0.27, and `sd`
# by up to ~0.41. Investigation (see PR discussion) showed this is NOT a
# platform-fragile-but-otherwise-flat direction: refining Julia 1.10's answer
# with extra LBFGS iterations under the *exact same* objective (evaluated with
# `fit_reml.nll`, i.e. still using the #746/#747/#835-fixed `_re_quad_stable`)
# moves it, to 10 decimal places, onto Julia 1.13's answer. That means Julia
# 1.10's pinned point was never actually a stationary point of the REML
# objective in the first place -- LBFGS's internally tracked gradient norm
# satisfied `g_tol` near the sigma_b -> 0 boundary (the same ill-conditioned
# region the #835 guard patches) well before reaching the true optimum, an
# artifact of cancellation, not a genuine convergence. Raising `g_tol` to
# 1e-14 on Julia 1.10 does not change this (checked directly): it is a stable
# floating-point-dependent local optimum, not an iteration-budget problem.
# Separately, checking out the PR's pre-fix base commit (bdc4e8a81) and
# re-running this exact repro shows it does NOT throw on either Julia 1.10 or
# 1.13 -- this synthetic seed/model never actually drives any group's
# sigma_b,k all the way to the exact 0*Inf line the #835 guard patches, so
# this reduced repro cannot discriminate pre- from post-fix by a thrown
# exception either. Given both `ll` and every coefficient block move together
# across Julia versions, per-block "profile flatness" tolerances would be
# arbitrary; the only property that is actually invariant here is that the
# REPORTED `ll`/coefficients are self-consistent with the REML objective
# itself (no cancellation garbage, regardless of which of the (at least) two
# stationary points LBFGS lands on), which is what is tested below.
#
#   julia --project=test -e 'using DRModels, Test; include("test/test_twin_gap_747_lss_reml.jl")'

module TestTwinGap747LssReml

using DRModels
using Test
using Random
using LinearAlgebra: cholesky, Symmetric, logdet, issuccess

@testset "#835 docs regression: lss REML sd(id) ~ sex does not hit 0*Inf" begin
    rng = Random.MersenneTwister(20260715)
    n_id, n_each = 80, 6
    sex = repeat([0.0, 1.0], inner = n_id ÷ 2)
    b = randn(rng, n_id) .* [0.65, 0.40][Int.(sex) .+ 1]
    id = repeat(1:n_id, inner = n_each)
    sexl = sex[id]
    y = [0.35, 0.70][Int.(sexl) .+ 1] .+ b[id] .+
        randn(rng, n_id * n_each) .* [0.35, 0.60][Int.(sexl) .+ 1]
    dat = (; y, sex = sexl, id)

    fit_reml = drm(bf(@formula(y ~ sex + (1 | id)),
                      @formula(sigma ~ sex),
                      @formula(sd(id) ~ sex)),
                   Gaussian(); data = dat, method = :REML)

    ll = reml_loglik(fit_reml)
    @test isfinite(ll)
    @test fit_reml.converged

    # Self-consistency: recompute the REML objective (mirrors `nll_reml` in
    # src/gaussian_lss.jl) at the fitted `theta`, reusing `fit_reml.nll` (the
    # ML nll closure, which already contains the #746/#747 cancellation-free
    # quadratic form and the #835 0*Inf guard) plus the Woodbury profiling
    # correction for the mean block. If the reported `ll` disagreed with this
    # from-scratch recomputation -- as it would under the historical bug,
    # which sent the objective to +1e44 .. +1e133 (see header) via 0*Inf/
    # cancellation garbage -- this catches it at essentially machine
    # precision, independent of which stationary point was reached.
    Xmu = hcat(ones(length(y)), sexl)
    Xsigma = hcat(ones(length(y)), sexl)
    Zg = hcat(ones(n_id), sex)
    pmu = 2
    function reml_obj_at(theta)
        betasigma = theta[(pmu + 1):(pmu + size(Xsigma, 2))]
        alpha = theta[(pmu + size(Xsigma, 2) + 1):end]
        etasigma = Xsigma * betasigma
        etasigmab = Zg * alpha
        S = zeros(n_id)
        ZtDinvX = zeros(n_id, pmu)
        XtDinvX = zeros(pmu, pmu)
        @inbounds for i in 1:length(y)
            invD = exp(-2 * etasigma[i])
            k = id[i]
            S[k] += invD
            for j in 1:pmu
                xj = Xmu[i, j]
                ZtDinvX[k, j] += invD * xj
                for l in 1:pmu
                    XtDinvX[j, l] += invD * xj * Xmu[i, l]
                end
            end
        end
        XtVinvX = copy(XtDinvX)
        @inbounds for k in 1:n_id
            sb2 = exp(2 * etasigmab[k])
            invMk = 1 / (1 / sb2 + S[k])
            for j in 1:pmu, l in 1:pmu
                XtVinvX[j, l] -= ZtDinvX[k, j] * invMk * ZtDinvX[k, l]
            end
        end
        chol = cholesky(Symmetric(XtVinvX); check = false)
        issuccess(chol) || return fit_reml.nll(theta) + 1e8
        return fit_reml.nll(theta) + 0.5 * logdet(chol) - 0.5 * pmu * log(2π)
    end
    @test isapprox(-reml_obj_at(fit_reml.theta), ll; atol = 1e-6, rtol = 0)

    # Sanity band, not a platform-specific pin: wide enough to hold both the
    # original Julia-1.10 answer (ll = -412.3006010571017) and the Julia-1.13
    # answer measured 2026-09-27 (ll = -408.1790006287666, a genuinely
    # different, deeper stationary point -- see header), but many orders of
    # magnitude away from the historical bug's +1e44 .. +1e133.
    @test -450 < ll < -350
    @test all(isfinite, coef(fit_reml, :mu))
    @test all(isfinite, coef(fit_reml, :sigma))
    @test all(isfinite, coef(fit_reml, :sd))
end

end # module
