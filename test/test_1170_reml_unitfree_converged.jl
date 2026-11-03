# #1170: the fixed-effect Gaussian REML route judges convergence by the unit-free
# Newton step (as the ML route does), not by the absolute `g_tol` gradient test.
using DRModels
using Test, Random

# The near-zero residual SD cell of #983: y = 1 + 0.5 x + sd * eps, n = 100.
function _gen_1170(rep, sd; n = 100)
    Random.seed!(20260927 + 15000 + rep)
    x = randn(n)
    return (; y = 1 .+ 0.5 .* x .+ sd .* randn(n), x)
end

@testset "#1170 Gaussian REML unit-free converged flag" begin
    f = bf(@formula(y ~ x), @formula(sigma ~ 1))

    # A fit at the optimum is not unconverged just because the raw gradient
    # misses the absolute tolerance. Rep 17 at sd = 3e-8 is such a fit.
    fit = drm(f, Gaussian(); data = _gen_1170(17, 3e-8), method = :REML)
    @test fit.estim_method == :REML
    @test isapprox(fit.theta[1], 1.0; atol = 1e-6)
    @test isapprox(fit.theta[2], 0.5; atol = 1e-6)
    @test fit.converged

    # The #983 cell itself (sd about 1e-4) stays converged.
    for rep in 1:5
        @test drm(f, Gaussian(); data = _gen_1170(rep, exp(-9.2)), method = :REML).converged
    end

    # A truncated fit still reports unconverged: `g_tol = 1e6` stops the
    # optimiser at its starting point, far from the REML optimum.
    Random.seed!(1)
    n = 100
    x = randn(n)
    y = 5 .+ 3 .* randn(n)
    fit_ok = drm(f, Gaussian(); data = (; y, x), method = :REML)
    @test fit_ok.converged
    fit_cut = drm(f, Gaussian(); data = (; y, x), method = :REML, g_tol = 1e6)
    @test !fit_cut.converged
end
