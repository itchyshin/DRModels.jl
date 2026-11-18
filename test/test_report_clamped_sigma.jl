# `sigma(fit)` must be the scale the likelihood SCORED, not a raw `exp(Xσ θ̂)` the
# model never evaluated (Dinnage audit M2; vault decision D-268; #324 follow-up).
#
# Every NB2 / Gamma / Beta fitter guards the log-scale linear predictor before
# exponentiating it — `_softclamp` on the fixed-effects routes (NB2 ±20, Gamma and
# Beta ±15, margin 3; NB2 ranef routes too), a hard `clamp` on the Gamma / Beta ranef, zi / hurdle / truncated
# routes. Post-fit the same routes stored `exp.(Xσ * θ̂)`, i.e. the UNguarded
# predictor, so on any row where the guard bit, `sigma(fit)` (and `simulate`,
# `residuals`, `predict`, all of which read `fit.scales[:sigma]`) described a
# scale the likelihood never used, and `loglik(fit)` could not be reproduced from
# the reported parameters. DRM.jl's multi-RE Gaussian route already reported the
# guarded value (`src/gaussian_ranef.jl`); this is that design applied to the rest.
# R twin: drmTMB's `drm_clamped_sigma_eta()` (commit 0526baa2f).
#
# NOTE ON THE BANDS. These are DRM.jl's own bands (NB2 ±20, Gamma/Beta ±15), NOT
# drmTMB's ±12 — a deliberate, documented difference (see `_softclamp` in
# src/negbinomial.jl and docs/design/03-likelihoods.md). This file asserts INTERNAL
# consistency (reported scale == scored scale), not band equality with R.
using DRModels
using Test, Random
import Distributions

# σ the likelihood scored, rebuilt from the fitted coefficients by hand.
_scored_sigma_soft(Xσ, θσ, lo, hi, m) = exp.(DRModels._softclamp.(Xσ * θσ, lo, hi, m))

@testset "reported sigma is the scale the likelihood scored (D-268, audit M2)" begin

    @testset "NB2 fixed effects — sigma predictor outside the ±20 band" begin
        # Six rows carry an extreme `sigma` covariate (z = 30) and y = 0. Zeros are
        # flat in η_σ for NB2 (P(0) → 1 as the size → 0), so they exert no pull on
        # the slope; the slope is pinned by the z ∈ {0,1,2} rows, which puts those
        # six rows at η_σ ≈ 27 — well past the ±20 identity band, where the
        # likelihood scores `_softclamp(η_σ) ≈ 23`, not 27.
        Random.seed!(3)
        n = 300
        z = vcat(repeat([0.0, 1.0, 2.0], inner = 98), fill(30.0, 6))
        β0, β1, μtrue = -0.3, 0.9, 6.0
        y = Float64[]
        for zi in z
            if zi == 30.0
                push!(y, 0.0)
            else
                r = exp(-2 * (β0 + β1 * zi))
                push!(y, Float64(rand(Distributions.NegativeBinomial(r, r / (r + μtrue)))))
            end
        end
        fit = drm(bf(@formula(y ~ 1), @formula(sigma ~ z)), NegBinomial2(); data = (; y, z))

        Xσ = hcat(ones(n), z)
        ηraw = Xσ * coef(fit, :sigma)
        @test maximum(ηraw) > 20.0                      # fixture really leaves the band
        σ̂ = sigma(fit)

        # (b) the reported scale IS the scored scale, row by row
        @test σ̂ ≈ _scored_sigma_soft(Xσ, coef(fit, :sigma), -20.0, 20.0, 3.0) rtol = 1e-12
        # and it is NOT the raw one on the clamped rows (this is what used to ship)
        @test exp(ηraw[end]) / σ̂[end] > 10.0

        # (a) loglik(fit) is reproducible from what the fit reports. This one holds
        # on the old code too — at a size of ~1e-20 the raw and the guarded scale
        # give the same near-degenerate NB2 mass for y = 0 — so the red above is
        # what pins this route; this is the invariant the fix must not break.
        μ̂ = fitted(fit)
        r̂ = 1 ./ σ̂ .^ 2
        ll = sum(Distributions.logpdf(Distributions.NegativeBinomial(r̂[i], r̂[i] / (r̂[i] + μ̂[i])),
                                      round(Int, y[i])) for i in 1:n)
        @test ll ≈ loglik(fit) rtol = 1e-10
    end

    @testset "Gamma fixed effects — sigma predictor outside the ±15 band" begin
        # Data with a coefficient of variation of ~2e-9: the MLE log σ ≈ -20, past
        # the ±15 band, so the likelihood scores the soft-clamped value (≈ -17.5).
        Random.seed!(20260915)
        n = 150
        y = 2.0 .* (1 .+ 2e-9 .* randn(n))
        fit = drm(bf(@formula(y ~ 1), @formula(sigma ~ 1)), Gamma(); data = (; y))

        Xσ = ones(n, 1)
        ηraw = Xσ * coef(fit, :sigma)
        @test minimum(ηraw) < -15.0
        σ̂ = sigma(fit)
        @test σ̂ ≈ _scored_sigma_soft(Xσ, coef(fit, :sigma), -15.0, 15.0, 3.0) rtol = 1e-12
        @test σ̂[1] / exp(ηraw[1]) > 2.0                  # raw was an order of magnitude off

        # NOTE: no loglik round-trip here, deliberately. Where the Gamma/Beta guard
        # binds the shape/precision is σ⁻² ≈ 10¹⁵, and recovering it by squaring a
        # reported σ loses the last bits of an exponent the density is extremely
        # sensitive to (one ulp in φ moves the Beta loglik by ~300 nats at n = 150).
        # The likelihood is reproducible from η_σ, not from a round trip through σ;
        # the in-band testset below does assert the round trip, where it is
        # numerically meaningful. See the NB2 case above for the out-of-band check.
    end

    @testset "Beta fixed effects — sigma predictor outside the ±15 band" begin
        Random.seed!(20260915)
        n = 150
        y = 0.45 .+ 2e-9 .* randn(n)
        fit = drm(bf(@formula(y ~ 1), @formula(sigma ~ 1)), Beta(); data = (; y))

        Xσ = ones(n, 1)
        ηraw = Xσ * coef(fit, :sigma)
        @test minimum(ηraw) < -15.0
        σ̂ = sigma(fit)
        @test σ̂ ≈ _scored_sigma_soft(Xσ, coef(fit, :sigma), -15.0, 15.0, 3.0) rtol = 1e-12
        @test σ̂[1] / exp(ηraw[1]) > 1.2

        # No loglik round-trip: same ill-conditioning as the Gamma case above.
    end

    @testset "in-band fits are bit-for-bit unchanged (the guard is identity there)" begin
        # Ordinary fits never approach the band, so the reported scale must be
        # EXACTLY the raw `exp(Xσ θ̂)` it always was — `==`, not `≈`.
        Random.seed!(7)
        n = 400
        x = randn(n)
        X = hcat(ones(n), x)

        ycount = Float64[rand(Distributions.NegativeBinomial(3.0, 3.0 / (3.0 + exp(0.4 + 0.3 * xi)))) for xi in x]
        fnb = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), NegBinomial2(); data = (; y = ycount, x))
        @test all(abs.(X * coef(fnb, :sigma)) .< 20.0)
        @test sigma(fnb) == exp.(X * coef(fnb, :sigma))
        rnb = 1 ./ sigma(fnb) .^ 2
        @test sum(Distributions.logpdf(Distributions.NegativeBinomial(rnb[i], rnb[i] / (rnb[i] + fitted(fnb)[i])),
                                       round(Int, ycount[i])) for i in 1:n) ≈ loglik(fnb) rtol = 1e-12

        ygam = Float64[rand(Distributions.Gamma(6.0, exp(0.2 + 0.3 * xi) / 6.0)) for xi in x]
        fg = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gamma(); data = (; y = ygam, x))
        @test all(abs.(X * coef(fg, :sigma)) .< 15.0)
        @test sigma(fg) == exp.(X * coef(fg, :sigma))
        αg = 1 ./ sigma(fg) .^ 2
        @test sum(Distributions.logpdf(Distributions.Gamma(αg[i], fitted(fg)[i] / αg[i]), ygam[i])
                  for i in 1:n) ≈ loglik(fg) rtol = 1e-12

        mb = 1 ./ (1 .+ exp.(-(0.1 .+ 0.5 .* x)))
        ybet = Float64[rand(Distributions.Beta(mi * 14.0, (1 - mi) * 14.0)) for mi in mb]
        fb = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Beta(); data = (; y = ybet, x))
        @test all(abs.(X * coef(fb, :sigma)) .< 15.0)
        @test sigma(fb) == exp.(X * coef(fb, :sigma))
        φb = 1 ./ sigma(fb) .^ 2
        @test sum(Distributions.logpdf(Distributions.Beta(fitted(fb)[i] * φb[i], (1 - fitted(fb)[i]) * φb[i]), ybet[i])
                  for i in 1:n) ≈ loglik(fb) rtol = 1e-12
    end

    @testset "hard-guarded NB2 routes report the guarded scale too" begin
        # The zero-inflated / hurdle / truncated NB2 routes hard-`clamp`
        # η_σ at ±20 in their likelihoods; they must report `exp(clamp(...))` for
        # the same reason. In band this is an identity, which is the state every
        # realistic fit is in — assert that it holds exactly.
        Random.seed!(11)
        n = 400
        x = randn(n)
        X = hcat(ones(n), x)
        μ = exp.(0.5 .+ 0.3 .* x)
        r = 2.5
        ynb = Float64[rand(Distributions.NegativeBinomial(r, r / (r + μi))) for μi in μ]
        yzi = [rand() < 0.2 ? 0.0 : ynb[i] for i in 1:n]

        fzi = drm(bf(@formula(y ~ x), @formula(sigma ~ x), @formula(zi ~ 1)), NegBinomial2();
                  data = (; y = yzi, x))
        # a zi fit has two scale slots, so `sigma()` returns the whole Dict
        @test sigma(fzi)[:sigma] == exp.(clamp.(X * coef(fzi, :sigma), -20.0, 20.0))

        ytr = Float64[max(v, 1.0) for v in ynb]
        ftr = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), TruncatedNegBinomial2();
                  data = (; y = ytr, x))
        @test sigma(ftr) == exp.(clamp.(X * coef(ftr, :sigma), -20.0, 20.0))
    end
end
