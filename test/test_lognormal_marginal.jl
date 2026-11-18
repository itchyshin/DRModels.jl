# #1126: LogNormal drm() accepts `marginal =` like Poisson/Gamma/NB2/Binomial/Beta/
# Student. log y is exactly Gaussian given the random intercept, so the Laplace
# approximation is exact there: `:Laplace` equals the default `:LA` fit; shapes /
# integrators LogNormal cannot honour fail with a clear ArgumentError, never a
# raw MethodError.
using DRModels
using Test, Random

@testset "LogNormal marginal= keyword (#1126)" begin
    Random.seed!(20261016611)
    dat = (y = abs.(randn(40)) .+ 0.2, x = randn(40), g = string.(repeat(1:8, inner = 5)))
    form = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1))

    fla = drm(form, LogNormal(); data = dat)
    @test fla.marginal === :LA
    @test drm(form, LogNormal(); data = dat, marginal = :LA).marginal === :LA

    fit = drm(form, LogNormal(); data = dat, marginal = :Laplace)   # was a MethodError
    @test fit.marginal === :Laplace
    @test loglik(fit) ≈ loglik(fla) atol = 1e-6                     # Laplace is exact here
    @test coef(fit, :mu) ≈ coef(fla, :mu) atol = 1e-4
    @test re_sd(fit)[:g] ≈ re_sd(fla)[:g] atol = 1e-4

    # clear refusals, not MethodErrors
    @test_throws ArgumentError drm(form, LogNormal(); data = dat, marginal = :AGHQ)
    @test_throws ArgumentError drm(form, LogNormal(); data = dat, marginal = :VA)
    @test_throws ArgumentError drm(form, LogNormal(); data = dat, marginal = :bogus)
    @test_throws ArgumentError drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), LogNormal();
                                   data = dat, marginal = :Laplace)  # no random effect
end
