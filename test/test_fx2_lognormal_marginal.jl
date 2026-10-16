# Red-proof for sweep-oct16 t6: LogNormal drm() has no `marginal=` keyword.
# Poisson / Gamma / NB2 / Binomial / Beta / Student accept it. After the fix,
# LogNormal accepts `:LA` (default) and `:Laplace` on an ordinary `(1 | g)`
# (the existing AGHQ-K=1 path is already exact Laplace for this family).
using DRModels
using Test, Random

@testset "FX2 LogNormal marginal=" begin
    Random.seed!(20261016)
    n = 40
    x = randn(n)
    g = string.(repeat(1:8, inner = 5))
    y = exp.(0.2 .+ 0.3 .* x .+ 0.25 .* randn(n))
    dat = (; y, x, g)
    form_fe = bf(@formula(y ~ x), @formula(sigma ~ 1))
    form_re = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1))

    @testset "keyword is accepted" begin
        fit_la = drm(form_fe, LogNormal(); data = dat, marginal = :LA)
        @test is_converged(fit_la) || isfinite(loglik(fit_la))
        @test fit_la.marginal === :LA

        fit_re = drm(form_re, LogNormal(); data = dat, marginal = :Laplace)
        @test isfinite(loglik(fit_re))
        @test fit_re.marginal === :Laplace

        fit_re_la = drm(form_re, LogNormal(); data = dat, marginal = :LA)
        @test isfinite(loglik(fit_re_la))
        @test fit_re_la.marginal === :LA
    end

    @testset "refusals match the Student / Gamma shape" begin
        @test_throws ArgumentError drm(form_fe, LogNormal(); data = dat, marginal = :Laplace)
        @test_throws ArgumentError drm(form_fe, LogNormal(); data = dat, marginal = :VA)
        @test_throws ArgumentError drm(form_re, LogNormal(); data = dat, marginal = :bogus)
    end

    @testset "sibling Gamma still accepts marginal=:Laplace" begin
        yg = exp.(0.2 .+ 0.3 .* x .+ 0.2 .* randn(n))
        fg = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)),
                 Gamma(); data = (; y = yg, x, g), marginal = :Laplace)
        @test fg.marginal === :Laplace
        @test isfinite(loglik(fg))
    end
end
