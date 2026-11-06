# Refs #1002: `lrt_boundary` runs the same marginal / MAP guards as `lrtest`
# (the issue was filed when it did not). Documents current behaviour; no
# behaviour change. The remaining gap -- neither verb reads `converged` -- is
# pinned with @test_broken.
using DRModels
using Test, Random

# Test-local copy of a fit with `converged = false` (no src helper exists).
_unconverged(f::DRModels.DrmFit) = DRModels.DrmFit(f.family, f.blocks, f.coefnames, f.theta,
    f.vcov, f.loglik, f.nobs, false, f.means, f.obs, f.scales, f.formula, f.nll, f.nllgrad,
    f.ranef, f.estim_method, f.reml_loglik, f.ml_loglik, f.marginal, f.phylo_penalty,
    f.penalty, f.iterations, f.phylo_scale, f.uncertainty, f.geometry)

_gmsg(f) = try; f(); "" catch e; e isa ArgumentError ? e.msg : rethrow(); end

@testset "lrt_boundary guards match lrtest (#1002)" begin
    g = repeat(1:8, inner = 6)
    x = randn(MersenneTwister(12), 48)
    y = 0.2 .+ 0.3 .* x .+ randn(MersenneTwister(13), 48)
    data = (; y, x, g)
    full = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data)
    reduced = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data)
    other = drm(bf(@formula(y ~ 1 + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data)

    @testset "marginal mismatch refused by both" begin
        lap = DRModels._withmarginal(other, :Laplace)
        @test occursin("marginal", _gmsg(() -> lrt_boundary(full, lap)))
        @test occursin("marginal", _gmsg(() -> lrtest(lap, full)))
        va = DRModels._withmarginal(other, :VA)
        @test occursin("ELBO", _gmsg(() -> lrt_boundary(full, va)))
    end

    @testset "MAP refused by both" begin
        mapped = DRModels._withmap(full, 0.1, nothing)
        @test occursin("MAP", _gmsg(() -> lrt_boundary(mapped, reduced)))
        @test occursin("MAP", _gmsg(() -> lrtest(reduced, mapped)))
    end

    @testset "converged flag is not read (gap, #1002)" begin
        # Desired: a fit flagged non-converged is refused or warned about.
        # Today nothing happens, so this stays broken until a design is chosen.
        bad = _unconverged(full)
        @test_broken !isempty(_gmsg(() -> lrt_boundary(bad, reduced)))
    end
end
