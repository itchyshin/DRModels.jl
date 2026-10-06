# test_bridge_response_mask_inference.jl — issue #646.
#
# The missing-response ("response mask") Gaussian route fits the OBSERVED rows
# and then restores means/obs/scales over the FULL design, so the returned
# `DrmFit` deliberately carries `nobs = 54` alongside length-60 per-row vectors.
# Two consumers read that pairing wrongly, and both defects were invisible to
# the existing tests because nothing exercised inference on a masked fit.
#
# MEASURED at the pin (drmjl-430ef64cc) on the drmTMB fixture below
# (set.seed(1); n = 60; y[1:6] <- NA; bf(y ~ x, sigma ~ x), Gaussian):
#
#   1. `is_converged(fit)` == false on a fit the optimiser had genuinely solved:
#        fit.converged (raw Optim.converged) = true
#        |grad|inf at the optimum            = 6.41e-12   (g_tol = 1e-8)
#        theta identical to the complete-case fit, max abs diff = 0.0
#      `_nondegenerate_fit` took its scale bar as `std(fit.obs[:mu])`, and
#      `obs[:mu]` still carries NaN in the 6 masked positions, so
#        yscale = NaN  ->  smax > 1e-6 * max(NaN, eps) == false
#      for ANY smax (here 0.9546). Every `>` against NaN is false under
#      IEEE-754, so the bar rejected every masked fit unconditionally.
#
#   2. `bootstrap_result` threw "all B bootstrap replicates failed": every
#      replicate died in `_simulate_once` with
#        DimensionMismatch: arrays could not be broadcast to a common size;
#        got a dimension with lengths 60 and 54
#      because the draw count was `fit.nobs` (54) while `means`/`scales` are
#      length 60. Deterministic, so all B replicates failed identically.
#
#   3. The iteration count was silently lost: `_with_full_fixed_gaussian_rows`
#      used the 19-arg compatibility constructor, which defaults `iterations`
#      to -1 ("not recorded"). The identical complete-case fit reported 7.
#
# The fixture is a plain Float64 vector with NaN in the masked positions --
# `_coerce_response_column` maps both `missing` and NaN to that, so this is the
# same object the R bridge produces from `NA_real_`.

using Test
using Random
using Statistics
using DRModels
using DRModels: is_converged, _nondegenerate_fit, niterations

@testset "#646 response-mask inference (convergence flag + bootstrap)" begin
    # The drmTMB fixture, generated once and inlined so the test does not
    # depend on R's RNG stream.
    rng = MersenneTwister(646)
    n = 60
    x = randn(rng, n)
    y = 0.3 .+ 0.5 .* x .+ randn(rng, n) .* exp.(0.1 .* x)
    y_masked = copy(y)
    y_masked[1:6] .= NaN
    n_observed = count(!isnan, y_masked)
    @test n_observed == 54

    f = bf(DRModels.@formula(y ~ x), DRModels.@formula(sigma ~ x))
    fit = drm(f, Gaussian(); data = (y = y_masked, x = x))

    # The fit really is on the observed rows only.
    @test nobs(fit) == n_observed
    # ... while the per-row vectors span the full design.
    @test length(fit.means[:mu]) == n
    @test length(fit.scales[:sigma]) == n
    @test count(isnan, fit.obs[:mu]) == 6

    # DEFECT 1: the converged fit must report converged.
    @test fit.converged                      # raw optimiser flag
    @test _nondegenerate_fit(fit)            # the bar that used to fail on NaN
    @test is_converged(fit)                  # the public accessor the bridge exports

    # The masked fit is the complete-case fit on the observed rows: same
    # optimum, so the convergence verdict must agree too. This is what makes
    # the false negative provably a mapping bug and not a real near-miss.
    obs = .!isnan.(y_masked)
    fit_cc = drm(f, Gaussian(); data = (y = y_masked[obs], x = x[obs]))
    @test is_converged(fit_cc)
    @test fit.theta ≈ fit_cc.theta atol = 1e-10
    @test isapprox(fit.loglik, fit_cc.loglik; atol = 1e-10)

    # DEFECT 3: the iteration count survives the full-row rebuild.
    @test niterations(fit) == niterations(fit_cc)
    @test niterations(fit) > 0

    # DEFECT 2: simulate draws over the full design, and a bootstrap on a
    # masked response completes instead of losing every replicate.
    #
    # The DRAW still spans the full design and is finite everywhere -- it must,
    # because `means`/`scales` are length 60. The mask is re-imposed one step
    # later, when `_bootstrap_data` builds the replicate table, so every
    # replicate refits on the same rows the seed fit observed (drmTMB #1188).
    ysim = DRModels.simulate(fit; rng = MersenneTwister(1))
    @test length(ysim) == n
    @test all(isfinite, ysim)

    # drmTMB #1188: mask-preserving replicates. Before this fix the replicate
    # table carried no missing values at all and each refit used all 60 rows,
    # so the bootstrap spread ignored the missing fraction entirely. MEASURED
    # on the R side of the same defect (S = 200 datasets, B = 99 replicates,
    # nominal 95% percentile interval on `fixef:mu:x`, truth 0.5): coverage
    # 0.895 / 0.820 / 0.720 before the fix at 10% / 30% / 50% masked, and
    # 0.910 / 0.895 / 0.910 after, against a Wald reference of
    # 0.920 / 0.925 / 0.915.
    datab = DRModels._bootstrap_data(fit.formula, (y = y_masked, x = x), copy(ysim))
    @test findall(isnan, datab.y) == findall(isnan, y_masked)
    @test datab.y[obs] == ysim[obs]
    @test nobs(drm(f, Gaussian(); data = datab)) == n_observed

    boot = bootstrap_result(
        fit;
        data = (y = y_masked, x = x),
        B = 5,
        rng = MersenneTwister(2),
        failures = :error,
        check_converged = true,
    )
    @test boot.used == 5
    @test isempty(boot.failures)
    @test length(boot.summary) == length(coef(fit))
    @test all(isfinite, [r.estimate for r in boot.summary])

    # A degenerate fit must still be rejected: the NaN-tolerant scale bar
    # widens `_nondegenerate_fit`'s input, it must not disable the check.
    degenerate = DRModels.DrmFit(
        fit.family, fit.blocks, fit.coefnames, fit.theta, fit.vcov,
        fit.loglik, fit.nobs, true, fit.means, fit.obs,
        Dict(:sigma => fill(1e-12, n)),
    )
    @test !_nondegenerate_fit(degenerate)
    @test !is_converged(degenerate)
end
