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
# This test fits the tutorial's exact model+seed and checks (a) the fit is
# finite/converged and (b) its REML log-likelihood and coefficients match a value
# pinned from origin/main (bdc4e8a81, 2026-09-27) to 1e-6 -- main does not hit the
# NaN (its optimizer path never reaches the sigma_b,k -> 0 line-search probe for
# this seed), so it is the correctness reference.
#
#   julia --project=test -e 'using DRModels, Test; include("test/test_twin_gap_747_lss_reml.jl")'

module TestTwinGap747LssReml

using DRModels
using Test
using Random

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

    # Pinned from origin/main (bdc4e8a81), same seed/model, 2026-09-27.
    ll_main = -412.3006010571017
    mu_main = [0.3655607713168836, 0.44118035047240606]
    sigma_main = [-1.0271225674164246, 0.5409939286056326]
    sd_main = [-0.3971922203086229, -0.3420955459529354]

    @test isapprox(ll, ll_main; atol = 1e-6, rtol = 0)
    @test isapprox(coef(fit_reml, :mu), mu_main; atol = 1e-6, rtol = 0)
    @test isapprox(coef(fit_reml, :sigma), sigma_main; atol = 1e-6, rtol = 0)
    @test isapprox(coef(fit_reml, :sd), sd_main; atol = 1e-6, rtol = 0)
end

end # module
