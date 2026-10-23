# Public VA front end (#136 Rung 1): `drm(...; marginal=:VA)` for families that
# already have `(1 | g)` ELBO kernels — Binomial, NegBinomial2, Gamma, Beta.
# Same wiring contract as Poisson Arc 0 (`test_va_frontend_poisson.jl`):
#
#   (1) Routing is lossless vs the internal `_fit_*_ranef_va` kernel.
#   (2) `DrmFit.marginal` is tagged `:VA` (default `:LA`).
#   (3) Unsupported VA errors citing #136 (no silent Laplace fallback).
#   (4) `method = :VA` is rejected with a pointer to `marginal` (shared helper;
#       one family is enough once Poisson already covers the same message).
#   (5) Default Laplace is unchanged (`marginal = :LA` ≡ omitting the keyword).
#
# Mixed LA/VA AIC/LRT is guarded in `comparison.jl` / `aic`; Poisson Arc 0 covers
# the Poisson public path. Rung 2 adds one Rung-1 family (NB2) so the guard is
# not Poisson-only. Issue #136 stays OPEN (phylo/crossed/ZI public VA + 136e later).

using DRModels
using Test
using Random
import Distributions

const DV = DRModels

function _va_err(expr)
    try
        expr()
        nothing
    catch e
        e
    end
end

function _assert_va_reject(e)
    @test e isa ArgumentError
    msg = sprint(showerror, e)
    @test occursin("136", msg)
    @test occursin("VA", msg) || occursin("variational", lowercase(msg))
end

@testset "Rung 1 public VA frontend (#136)" begin

    @testset "Binomial marginal=:VA routes to _fit_binomial_ranef_va" begin
        rng = MersenneTwister(20260808)
        G = 40; per = 10; n = G * per
        g = repeat(1:G, inner = per); x = randn(rng, n)
        ntr = fill(8.0, n)
        β = [0.4, -0.6]; σb = 0.7
        bg = σb .* randn(rng, G)
        μ = DV._logistic.(β[1] .+ β[2] .* x .+ bg[g])
        s = Float64.([rand(rng, Distributions.Binomial(8, μi)) for μi in μ])
        fail = ntr .- s
        data = (; s, fail, x, g)

        fit_va = drm(bf(@formula(cbind(s, fail) ~ x + (1 | g))), Binomial();
                     data = data, marginal = :VA)
        fit_la = drm(bf(@formula(cbind(s, fail) ~ x + (1 | g))), Binomial(); data = data)
        gidx, _ = DV._group_index(g)
        ker = DV._fit_binomial_ranef_va(Binomial(), s, ntr, hcat(ones(n), x), gidx, G,
                                        ["(Intercept)", "x"], :g, 1e-8)

        @test is_converged(fit_va)
        @test coef(fit_va) ≈ coef(ker) atol = 1e-10
        @test loglik(fit_va) ≈ loglik(ker) atol = 1e-8
        @test fit_va.marginal === :VA
        @test fit_la.marginal === :LA
        @test family(fit_va) isa Binomial
        @test isfinite(loglik(fit_va))
    end

    @testset "NB2 marginal=:VA routes to _fit_nb2_ranef_va" begin
        rng = MersenneTwister(202608081)
        G = 40; per = 10; n = G * per
        g = repeat(1:G, inner = per); x = randn(rng, n)
        β = [0.4, 0.5]; θsz = 3.0; σb = 0.5
        bg = σb .* randn(rng, G)
        μ = exp.(β[1] .+ β[2] .* x .+ bg[g])
        y = Float64.([rand(rng, Distributions.NegativeBinomial(θsz, θsz / (θsz + μi))) for μi in μ])
        data = (; y, x, g)

        fit_va = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), NegBinomial2();
                     data = data, marginal = :VA)
        fit_la = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), NegBinomial2(); data = data)
        gidx, _ = DV._group_index(g)
        ker = DV._fit_nb2_ranef_va(NegBinomial2(), y, hcat(ones(n), x), ones(n, 1), gidx, G,
                                   ["(Intercept)", "x"], ["(Intercept)"], :g, 1e-8)

        @test is_converged(fit_va)
        @test coef(fit_va) ≈ coef(ker) atol = 1e-10
        @test loglik(fit_va) ≈ loglik(ker) atol = 1e-8
        @test fit_va.marginal === :VA
        @test fit_la.marginal === :LA
        @test family(fit_va) isa NegBinomial2
        @test isfinite(loglik(fit_va))
    end

    @testset "Gamma marginal=:VA routes to _fit_gamma_ranef_va" begin
        rng = MersenneTwister(202608082)
        G = 40; per = 10; n = G * per
        g = repeat(1:G, inner = per); x = randn(rng, n)
        β = [0.4, 0.5]; α = 4.0; σb = 0.5
        bg = σb .* randn(rng, G)
        μ = exp.(β[1] .+ β[2] .* x .+ bg[g])
        y = Float64.([rand(rng, Distributions.Gamma(α, μi / α)) for μi in μ])
        data = (; y, x, g)

        fit_va = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gamma();
                     data = data, marginal = :VA)
        fit_la = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gamma(); data = data)
        gidx, _ = DV._group_index(g)
        ker = DV._fit_gamma_ranef_va(Gamma(), y, hcat(ones(n), x), ones(n, 1), gidx, G,
                                     ["(Intercept)", "x"], ["(Intercept)"], :g, 1e-8)

        @test is_converged(fit_va)
        @test coef(fit_va) ≈ coef(ker) atol = 1e-10
        @test loglik(fit_va) ≈ loglik(ker) atol = 1e-8
        @test fit_va.marginal === :VA
        @test fit_la.marginal === :LA
        @test family(fit_va) isa Gamma
        @test isfinite(loglik(fit_va))
    end

    @testset "Beta marginal=:VA routes to _fit_beta_ranef_va" begin
        rng = MersenneTwister(202608083)
        G = 40; per = 10; n = G * per
        g = repeat(1:G, inner = per); x = randn(rng, n)
        β = [0.3, -0.6]; φ = 12.0; σb = 0.5
        bg = σb .* randn(rng, G)
        μ = DV._logistic.(β[1] .+ β[2] .* x .+ bg[g])
        y = Float64.([rand(rng, Distributions.Beta(μi * φ, (1 - μi) * φ)) for μi in μ])
        data = (; y, x, g)

        fit_va = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Beta();
                     data = data, marginal = :VA)
        fit_la = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Beta(); data = data)
        gidx, _ = DV._group_index(g)
        ker = DV._fit_beta_ranef_va(Beta(), y, hcat(ones(n), x), ones(n, 1), gidx, G,
                                    ["(Intercept)", "x"], ["(Intercept)"], :g, 1e-8)

        @test is_converged(fit_va)
        @test coef(fit_va) ≈ coef(ker) atol = 1e-10
        @test loglik(fit_va) ≈ loglik(ker) atol = 1e-8
        @test fit_va.marginal === :VA
        @test fit_la.marginal === :LA
        @test family(fit_va) isa Beta
        @test isfinite(loglik(fit_va))
    end

    @testset "mixed LA/VA AIC and LRT error (NB2)" begin
        rng = MersenneTwister(1363)
        G = 30; per = 8; n = G * per
        g = repeat(1:G, inner = per); x = randn(rng, n)
        y = Float64.([rand(rng, Distributions.NegativeBinomial(4.0, 4.0 / (4.0 + exp(0.3 + 0.4 * x[i] + 0.4 * randn(rng)))))
                      for i in 1:n])
        data = (; y, x, g)
        fit_va = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), NegBinomial2();
                     data = data, marginal = :VA)
        fit_la = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), NegBinomial2(); data = data)
        @test_throws ArgumentError lrtest(fit_la, fit_va)
        @test_throws ArgumentError anova(fit_la, fit_va)
        @test_throws ArgumentError aic(fit_va)
        @test_throws ArgumentError bic(fit_va)
        @test_throws ArgumentError aicc(fit_va)
        @test isfinite(aic(fit_la)) && isfinite(bic(fit_la))
    end

    @testset "method=:VA rejected with pointer to marginal (Binomial)" begin
        rng = MersenneTwister(3)
        G = 16; per = 6; n = G * per
        g = repeat(1:G, inner = per); x = randn(rng, n)
        ntr = fill(6.0, n)
        μ = DV._logistic.(0.2 .+ 0.4 .* x)
        s = Float64.([rand(rng, Distributions.Binomial(6, μi)) for μi in μ])
        fail = ntr .- s
        data = (; s, fail, x, g)
        err = _va_err(() -> drm(bf(@formula(cbind(s, fail) ~ x + (1 | g))), Binomial();
                                data = data, method = :VA))
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin("marginal", msg)
        @test occursin("method", msg)
    end

    @testset "unsupported VA rejects citing #136" begin
        rng = MersenneTwister(7)
        G = 16; per = 8; n = G * per
        g = repeat(1:G, inner = per); h = repeat(1:G, inner = per)[randperm(rng, n)]
        x = randn(rng, n)

        ntr = fill(6.0, n)
        μb = DV._logistic.(0.2 .+ 0.4 .* x)
        s = Float64.([rand(rng, Distributions.Binomial(6, μb_i)) for μb_i in μb])
        fail = ntr .- s
        ynb = Float64.([rand(rng, Distributions.NegativeBinomial(3.0, 3.0 / (3.0 + exp(0.3 + 0.4 * xi)))) for xi in x])
        yg = Float64.([rand(rng, Distributions.Gamma(4.0, exp(0.3 + 0.4 * xi) / 4.0)) for xi in x])
        yβ = Float64.([rand(rng, Distributions.Beta(0.4 * 12, 0.6 * 12)) for _ in 1:n])
        data_b = (; s, fail, x, g, h)
        data_nb = (; y = ynb, x, g, h)
        data_g = (; y = yg, x, g, h)
        data_β = (; y = yβ, x, g, h)

        e_fe_b = _va_err(() -> drm(bf(@formula(cbind(s, fail) ~ x)), Binomial();
                                   data = data_b, marginal = :VA))
        e_cross_b = _va_err(() -> drm(bf(@formula(cbind(s, fail) ~ x + (1 | g) + (1 | h))),
                                      Binomial(); data = data_b, marginal = :VA))
        e_corr_b = _va_err(() -> drm(bf(@formula(cbind(s, fail) ~ x + (1 + x | g))),
                                     Binomial(); data = data_b, marginal = :VA))

        e_fe_nb = _va_err(() -> drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), NegBinomial2();
                                    data = data_nb, marginal = :VA))
        e_corr_nb = _va_err(() -> drm(bf(@formula(y ~ x + (1 + x | g)), @formula(sigma ~ 1)),
                                      NegBinomial2(); data = data_nb, marginal = :VA))
        e_sig_nb = _va_err(() -> drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ x)),
                                     NegBinomial2(); data = data_nb, marginal = :VA))
        e_zi_nb = _va_err(() -> drm(bf(@formula(y ~ x + (1 | g)), @formula(zi ~ 1)),
                                    NegBinomial2(); data = data_nb, marginal = :VA))

        e_fe_g = _va_err(() -> drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gamma();
                                   data = data_g, marginal = :VA))
        e_cross_g = _va_err(() -> drm(bf(@formula(y ~ x + (1 | g) + (1 | h)), @formula(sigma ~ 1)),
                                      Gamma(); data = data_g, marginal = :VA))
        e_sig_g = _va_err(() -> drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ x)),
                                    Gamma(); data = data_g, marginal = :VA))

        e_fe_β = _va_err(() -> drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Beta();
                                   data = data_β, marginal = :VA))
        e_corr_β = _va_err(() -> drm(bf(@formula(y ~ x + (1 + x | g)), @formula(sigma ~ 1)),
                                     Beta(); data = data_β, marginal = :VA))
        e_sig_β = _va_err(() -> drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ x)),
                                    Beta(); data = data_β, marginal = :VA))

        phy = random_balanced_tree(8; branch_length = 0.2)
        species = repeat(1:8, inner = 3)
        ygp = Float64.([rand(rng, Distributions.Gamma(4.0, 1.0 / 4.0)) for _ in species])
        e_phy_g = _va_err(() -> drm(bf(@formula(y ~ 1 + phylo(1 | species)), @formula(sigma ~ 1)),
                                    Gamma(); data = (; y = ygp, species), tree = phy, se = false,
                                    marginal = :VA))

        for e in (e_fe_b, e_cross_b, e_corr_b, e_fe_nb, e_corr_nb, e_sig_nb, e_zi_nb,
                  e_fe_g, e_cross_g, e_sig_g, e_fe_β, e_corr_β, e_sig_β, e_phy_g)
            _assert_va_reject(e)
        end
    end

    @testset "marginal=:LA reproduces the default Laplace fit" begin
        rng = MersenneTwister(99)
        G = 24; per = 6; n = G * per
        g = repeat(1:G, inner = per); x = randn(rng, n)
        ntr = fill(6.0, n)
        μb = DV._logistic.(0.2 .+ 0.4 .* x .+ 0.4 .* randn(rng, n))
        s = Float64.([rand(rng, Distributions.Binomial(6, clamp(μi, 1e-4, 1 - 1e-4))) for μi in μb])
        fail = ntr .- s
        ynb = Float64.([rand(rng, Distributions.NegativeBinomial(3.0, 3.0 / (3.0 + exp(0.3 + 0.4 * x[i] + 0.4 * randn(rng))))) for i in 1:n])
        yg = Float64.([rand(rng, Distributions.Gamma(4.0, exp(0.3 + 0.4 * x[i] + 0.3 * randn(rng)) / 4.0)) for i in 1:n])
        yβ = Float64.([let μ = DV._logistic(0.2 - 0.4 * x[i] + 0.3 * randn(rng));
                           rand(rng, Distributions.Beta(μ * 10, (1 - μ) * 10))
                       end for i in 1:n])

        pairs = (
            (bf(@formula(cbind(s, fail) ~ x + (1 | g))), Binomial(), (; s, fail, x, g)),
            (bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), NegBinomial2(), (; y = ynb, x, g)),
            (bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gamma(), (; y = yg, x, g)),
            (bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Beta(), (; y = yβ, x, g)),
        )
        for (fml, fam, dat) in pairs
            f_default = drm(fml, fam; data = dat)
            f_la = drm(fml, fam; data = dat, marginal = :LA)
            @test coef(f_default) == coef(f_la)
            @test loglik(f_default) == loglik(f_la)
            @test f_default.marginal === :LA
            @test f_la.marginal === :LA
        end
    end
end
