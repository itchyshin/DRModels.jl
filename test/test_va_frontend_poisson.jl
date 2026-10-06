# Poisson public VA front end (#136 Arc 0): `drm(...; marginal=:VA)` must reach
# the existing `_fit_poisson_ranef_va` kernel. Until this slice the kernel lived
# only as an internal proof-function. Here we check the WIRING:
#
#   (1) Routing is lossless. A single random intercept `(1 | g)` under
#       `marginal = :VA` must match calling `_fit_poisson_ranef_va` directly
#       (coef + ELBO). The VA path is deterministic, so this is an exact identity.
#   (2) `DrmFit.marginal` is tagged `:VA` (default `:LA`). Mixed LA/VA AIC/LRT
#       must error — ELBO ≠ logLik.
#   (3) Unsupported VA (FE-only, corr, crossed, phylo, zi, hu) errors citing #136
#       rather than silently falling back to Laplace.
#   (4) `method = :VA` is rejected with a pointer to `marginal`.
#   (5) Default Laplace is unchanged (`marginal = :LA` ≡ omitting the keyword).
#
# Issue #136 stays OPEN (Rung 1 wires Binomial/NB2/Gamma/Beta `(1|g)`; 136e later).

using DRModels
using Test
using Random
import Distributions

const DV = DRModels

@testset "Poisson public VA frontend (#136 Arc 0)" begin

    @testset "marginal=:VA routes to _fit_poisson_ranef_va" begin
        rng = MersenneTwister(20260611)
        G = 60; per = 12; n = G * per
        g = repeat(1:G, inner = per); x = randn(rng, n)
        β = [0.5, -0.4]; σb = 0.6
        bg = σb .* randn(rng, G)
        λ = exp.(β[1] .+ β[2] .* x .+ bg[g])
        y = Float64.([rand(rng, Distributions.Poisson(λi)) for λi in λ])
        data = (; y, x, g)

        fit_va = drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = data, marginal = :VA)
        fit_la = drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = data)
        gidx, _ = DV._group_index(g)
        ker = DV._fit_poisson_ranef_va(Poisson(), y, hcat(ones(n), x), gidx, G,
                                       ["(Intercept)", "x"], :g, 1e-8)

        @test is_converged(fit_va)
        @test coef(fit_va) ≈ coef(ker) atol = 1e-10
        @test loglik(fit_va) ≈ loglik(ker) atol = 1e-8
        @test fit_va.marginal === :VA
        @test fit_la.marginal === :LA
        @test coef(fit_va, :mu)[1] ≈ coef(fit_la, :mu)[1] atol = 0.20
        @test coef(fit_va, :mu)[2] ≈ coef(fit_la, :mu)[2] atol = 0.05
        @test re_sd(fit_va)[:g] ≈ re_sd(fit_la)[:g] atol = 0.12
        @test family(fit_va) isa Poisson
        @test isfinite(loglik(fit_va))
    end

    @testset "method=:VA is rejected with pointer to marginal" begin
        rng = MersenneTwister(3)
        G = 20; per = 8; n = G * per
        g = repeat(1:G, inner = per); x = randn(rng, n)
        y = Float64.([rand(rng, Distributions.Poisson(exp(0.3 + 0.4 * xi))) for xi in x])
        data = (; y, x, g)
        err = try
            drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = data, method = :VA)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("marginal", sprint(showerror, err))
        @test occursin("method", sprint(showerror, err))
    end

    @testset "unsupported VA rejects citing #136" begin
        rng = MersenneTwister(7)
        G = 20; per = 10; n = G * per
        g = repeat(1:G, inner = per); h = repeat(1:G, inner = per)[randperm(rng, n)]
        x = randn(rng, n)
        y = Float64.([rand(rng, Distributions.Poisson(exp(0.3 + 0.5 * xi))) for xi in x])
        data = (; y, x, g, h)

        function _va_err(expr)
            try
                expr()
                nothing
            catch e
                e
            end
        end

        e_fe = _va_err(() -> drm(bf(@formula(y ~ x)), Poisson(); data = data, marginal = :VA))
        e_corr = _va_err(() -> drm(bf(@formula(y ~ x + (1 + x | g))), Poisson(); data = data, marginal = :VA))
        e_cross = _va_err(() -> drm(bf(@formula(y ~ x + (1 | g) + (1 | h))), Poisson(); data = data, marginal = :VA))
        e_zi = _va_err(() -> drm(bf(@formula(y ~ x), @formula(zi ~ 1)), Poisson(); data = data, marginal = :VA))
        e_hu = _va_err(() -> drm(bf(@formula(y ~ x), @formula(hu ~ 1)), Poisson(); data = data, marginal = :VA))
        e_bad = _va_err(() -> drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = data, marginal = :nope))

        phy = random_balanced_tree(8; branch_length = 0.2)
        species = repeat(1:8, inner = 3)
        yp = Float64.([rand(rng, Distributions.Poisson(exp(0.2))) for _ in species])
        e_phy = _va_err(() -> drm(bf(@formula(y ~ 1 + phylo(1 | species))), Poisson();
                                  data = (; y = yp, species), tree = phy, se = false,
                                  marginal = :VA))

        for e in (e_fe, e_corr, e_cross, e_zi, e_hu, e_phy)
            @test e isa ArgumentError
            msg = sprint(showerror, e)
            @test occursin("136", msg)
            @test occursin("VA", msg) || occursin("variational", lowercase(msg))
        end
        @test e_bad isa ArgumentError

        # Default Laplace still fits a plain FE Poisson on the same data.
        # (Do not smoke zi/hu here — this y is not zero-inflated, so the ZIP Hessian
        # can be singular; those paths are covered in test_zi.jl / test_hurdle.jl.)
        @test is_converged(drm(bf(@formula(y ~ x)), Poisson(); data = data))
    end

    @testset "mixed LA/VA AIC and LRT error" begin
        rng = MersenneTwister(11)
        G = 40; per = 8; n = G * per
        g = repeat(1:G, inner = per); x = randn(rng, n)
        y = Float64.([rand(rng, Distributions.Poisson(exp(0.3 + 0.5 * x[i] + 0.5 * randn(rng)))) for i in 1:n])
        data = (; y, x, g)
        fit_va = drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = data, marginal = :VA)
        fit_la = drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = data)
        @test_throws ArgumentError lrtest(fit_la, fit_va)
        @test_throws ArgumentError anova(fit_la, fit_va)
        @test_throws ArgumentError aic(fit_va)
        @test_throws ArgumentError bic(fit_va)
        @test_throws ArgumentError aicc(fit_va)
        @test isfinite(aic(fit_la)) && isfinite(bic(fit_la))
    end

    @testset "marginal=:LA reproduces the default Laplace fit" begin
        rng = MersenneTwister(99)
        G = 40; per = 8; n = G * per
        g = repeat(1:G, inner = per); x = randn(rng, n)
        y = Float64.([rand(rng, Distributions.Poisson(exp(0.3 + 0.5 * x[i] + 0.5 * randn(rng)))) for i in 1:n])
        data = (; y, x, g)
        f_default = drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = data)
        f_la = drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = data, marginal = :LA)
        @test coef(f_default) == coef(f_la)
        @test loglik(f_default) == loglik(f_la)
        @test f_default.marginal === :LA
        @test f_la.marginal === :LA
    end
end
