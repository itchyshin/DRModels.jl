# Refs #1002 / #1053: lrtest and lrt_boundary warn when a compared fit fails the
# shared convergence checks, exempt a boundary fit only after its stationarity
# check passes, and refuse fits estimated on different observed rows.
using DRModels
using Test, Random

# Test-local copy of a fit with `converged = false`; `scales` can be overridden.
_noconv(f::DRModels.DrmFit; scales = f.scales) = DRModels.DrmFit(f.family, f.blocks, f.coefnames,
    f.theta, f.vcov, f.loglik, f.nobs, false, f.means, f.obs, scales, f.formula, f.nll,
    f.nllgrad, f.ranef, f.estim_method, f.reml_loglik, f.ml_loglik, f.marginal,
    f.phylo_penalty, f.penalty, f.iterations, f.phylo_scale, f.uncertainty, f.geometry)
_converged(f::DRModels.DrmFit; scales = f.scales) = DRModels.DrmFit(f.family, f.blocks, f.coefnames,
    f.theta, f.vcov, f.loglik, f.nobs, true, f.means, f.obs, scales, f.formula, f.nll,
    f.nllgrad, f.ranef, f.estim_method, f.reml_loglik, f.ml_loglik, f.marginal,
    f.phylo_penalty, f.penalty, f.iterations, f.phylo_scale, f.uncertainty, f.geometry)

@testset "compare guards: unconverged and observed rows (#1002)" begin
    x = randn(MersenneTwister(31), 60)
    y = 0.2 .+ 0.5 .* x .+ randn(MersenneTwister(32), 60)
    data = (; y, x)
    full = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data)
    reduced = drm(bf(@formula(y ~ 1), @formula(sigma ~ 1)), Gaussian(); data)

    @testset "converged fits: no warning" begin
        @test_logs lrtest(reduced, full)
        @test_logs lrt_boundary(full, reduced)
    end

    @testset "unconverged fit warns, names the fit, result still returned" begin
        bad = _noconv(full)
        @test_warn "full fit did not converge" lrtest(reduced, bad)
        t = (@test_logs (:warn,) match_mode = :any lrtest(reduced, bad))
        @test t.statistic ≈ 2 * (loglik(full) - loglik(reduced))
        @test_warn "reduced fit did not converge" lrtest(_noconv(reduced), full)
        @test_warn "reduced and full fit did not converge" lrtest(_noconv(reduced), bad)
        @test_warn "did not converge" lrt_boundary(bad, reduced)
    end

    @testset "unmapped wall and optimizer failure have distinct statuses" begin
        wall_scales = merge(full.scales, Dict(:cond_logsd_wall => [-9.0]))
        failed_at_boundary = _noconv(full; scales = wall_scales)
        @test DRModels.convergence_status(failed_at_boundary) === :not_converged
        @test_warn "full fit did not converge" lrtest(reduced, failed_at_boundary)

        wall_unmapped = _converged(full; scales = wall_scales)
        @test DRModels.convergence_status(wall_unmapped) === :degenerate
        # The optimizer flag alone does not certify a wall whose active
        # conditional-mode coordinate cannot be mapped for the KKT check.
        @test_warn "full fit did not converge" lrtest(reduced, wall_unmapped)
    end

    @testset "same nobs, different observed rows is refused" begin
        y2 = copy(y); y2[10] = y2[10] + 5.0       # same nobs, one different observation
        other = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; y = y2, x))
        @test_throws ArgumentError lrtest(reduced, other)
    end
end
