# Refs #1002 / #1053: lrtest and lrt_boundary warn when a compared fit did not
# converge, stay silent for a boundary fit (own status), and keep refusing fits
# estimated on different observed rows even when nobs is equal.
using DRModels
using Test, Random

# Test-local copy of a fit with `converged = false`; `scales` can be overridden.
_noconv(f::DRModels.DrmFit; scales = f.scales) = DRModels.DrmFit(f.family, f.blocks, f.coefnames,
    f.theta, f.vcov, f.loglik, f.nobs, false, f.means, f.obs, scales, f.formula, f.nll,
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

    @testset "boundary fit is not called unconverged" begin
        wall = _noconv(full; scales = merge(full.scales, Dict(:cond_logsd_wall => [-9.0])))
        @test DRModels.convergence_status(wall) === :boundary
        @test_logs lrtest(reduced, wall)
    end

    @testset "same nobs, different observed rows is refused" begin
        y2 = copy(y); y2[10] = y2[10] + 5.0       # same nobs, one different observation
        other = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; y = y2, x))
        @test_throws ArgumentError lrtest(reduced, other)
    end
end
