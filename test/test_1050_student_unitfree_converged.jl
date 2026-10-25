# #1050 (Student fixed-effect route): `converged` is the unit-free Newton step in
# SE units, not the absolute `g_tol = 1e-8` gradient test. At the Gaussian limit
# the degrees of freedom run to infinity along a flat direction, the optimiser
# stops on its f test with |g|inf a little above 1e-8, and the fit is the MLE.
using DRModels
using Test, Random

function _gen_1050_student(rep; n = 100)
    Random.seed!(20261025 + rep)
    x = randn(n)
    return (; y = 1 .+ 0.5 .* x .+ randn(n), x)
end

@testset "#1050 Student unit-free converged flag" begin
    f = bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1))

    # Reps 46 and 50 stop with |g|inf of 5e-8 and 2e-7 at a valid optimum.
    for rep in (46, 50)
        fit = drm(f, Student(); data = _gen_1050_student(rep))
        g = DRModels.ForwardDiff.gradient(fit.nll, fit.theta)
        @test maximum(abs, g) > 1e-8                    # the old absolute test fails
        @test DRModels._unitfree_converged(
            DRModels.ForwardDiff.hessian(fit.nll, fit.theta), g)
        @test fit.converged
    end

    # An interior fit (finite nu) stays converged.
    Random.seed!(7)
    n = 300; x = randn(n)
    chi = vec(sum(abs2, randn(n, 4); dims = 2))              # t_4 noise: chi-square_4 mixing
    y = 1 .+ 0.5 .* x .+ 0.7 .* randn(n) ./ sqrt.(chi ./ 4)
    fit_ok = drm(f, Student(); data = (; y, x))
    @test fit_ok.converged

    # A truncated fit still reports unconverged: `g_tol = 1e6` stops the
    # optimiser at its starting point, far from the optimum.
    fit_cut = drm(f, Student(); data = (; y, x), g_tol = 1e6)
    @test !fit_cut.converged
end
