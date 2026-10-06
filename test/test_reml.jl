# test_reml.jl — REML for the fixed-effect Gaussian location–scale model
# (issue #11, slice 2). Verifies:
#   1. ML is unchanged when method = :ML (the default) — byte-for-byte vs no kwarg.
#   2. The defining REML property: the REML residual variance is LARGER than ML
#      (the n vs n−pμ divisor) — the analog of diag(Λ_REML) ≥ diag(Λ_ML).
#   3. FD-gradient self-consistency of the restricted objective at the optimum.
#   4. The model-selection guard fires across different MEAN structures under REML
#      and stays silent for a variance-only difference.
#   5. The fit stores reml_loglik + ml_loglik + estimation_method = :REML.
using DRModels                         # re-exports @formula
using Test, Random, Statistics, LinearAlgebra
import ForwardDiff

@testset "REML (fixed-effect Gaussian location–scale, method=:REML)" begin
    Random.seed!(20260607)
    n = 60                      # small n makes the n vs n−p REML gap clearly visible
    x = randn(n)
    z = randn(n)
    y = 1.5 .+ 0.7 .* x .- 0.4 .* z .+ 1.3 .* randn(n)
    data = (; y, x, z)

    @testset "method=:ML is the unchanged default" begin
        f_default = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian(); data)
        f_ml      = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian(); data, method = :ML)
        @test coef(f_default) == coef(f_ml)            # byte-for-byte
        @test loglik(f_default) == loglik(f_ml)
        @test estimation_method(f_default) === :ML
        @test isnan(reml_loglik(f_default))
        @test ml_loglik(f_default) == loglik(f_default)
    end

    @testset "REML residual variance ≥ ML (the defining property)" begin
        # Homoscedastic σ ~ 1: closed-form check that σ²_REML = SSR/(n−pμ) >
        # σ²_ML = SSR/n. Here pμ = 2 (intercept + x).
        fml   = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian(); data, method = :ML)
        freml = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian(); data, method = :REML)

        σ²_ml   = exp(2 * coef(fml,   :sigma)[1])
        σ²_reml = exp(2 * coef(freml, :sigma)[1])
        @test σ²_reml > σ²_ml                          # REML inflates the variance

        # Quantitatively matches the n/(n−pμ) ratio.
        pμ = length(coef(fml, :mu))
        @test isapprox(σ²_reml / σ²_ml, n / (n - pμ); rtol = 1e-4)

        # The mean coefficients are unchanged at the optimum (β̂_μ is the same WLS
        # estimate ML would give at the REML σ — only the scale block differs).
        @test isapprox(coef(freml, :mu), coef(fml, :mu); rtol = 1e-5)

        # Metadata: REML fit records both log-likelihoods + the method marker.
        @test estimation_method(freml) === :REML
        @test isfinite(reml_loglik(freml))
        @test loglik(freml) == reml_loglik(freml)
        @test isfinite(ml_loglik(freml))
        # At the REML estimate the plain ML log-lik is ≤ the ML-optimal one.
        @test ml_loglik(freml) <= loglik(fml) + 1e-8
    end

    @testset "REML property holds with heteroscedastic σ" begin
        # σ depends on x: the REML correction inflates the overall residual scale
        # relative to ML. The robust aggregate statement is that the AVERAGE fitted
        # variance is larger under REML (the n vs n−pμ degrees-of-freedom effect),
        # which is the practical bias-correction guarantee.
        fml   = drm(bf(@formula(y ~ 1 + x + z), @formula(sigma ~ 1 + x)), Gaussian(); data, method = :ML)
        freml = drm(bf(@formula(y ~ 1 + x + z), @formula(sigma ~ 1 + x)), Gaussian(); data, method = :REML)
        mean_var_ml   = mean(abs2, sigma(fml))
        mean_var_reml = mean(abs2, sigma(freml))
        @test mean_var_reml > mean_var_ml
    end

    @testset "FD-gradient self-consistency at the REML optimum" begin
        # Rebuild the restricted objective over β_σ and check its FD gradient ≈ 0
        # at the fitted estimate (the slice-2 analog of reml_q4's Gate B).
        Xμ = hcat(ones(n), x)
        Xσ = reshape(ones(n), n, 1)
        pμ = size(Xμ, 2)
        const_2pi = 0.5 * n * log(2π)
        function nll_reml(βσ)
            ησ = Xσ * βσ
            w = exp.(-2 .* ησ)
            XtW = Xμ' * (w .* Xμ)
            βμ = XtW \ (Xμ' * (w .* y))
            r = y .- Xμ * βμ
            s = zero(eltype(βσ))
            @inbounds for i in 1:n
                s += ησ[i] + 0.5 * r[i] * r[i] * exp(-2 * ησ[i])
            end
            return s + const_2pi + 0.5 * logdet(XtW) - 0.5 * pμ * log(2π)
        end
        freml = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian(); data, method = :REML)
        βσ̂ = coef(freml, :sigma)
        g = ForwardDiff.gradient(nll_reml, βσ̂)
        @test maximum(abs, g) < 1e-4
    end

    @testset "model-selection guard (the REML trap)" begin
        full_reml = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian(); data, method = :REML)
        # Same MEAN structure (μ = 1 + x), different VARIANCE structure → VALID.
        var_only_reml = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x)), Gaussian(); data, method = :REML)
        # Different MEAN structure (μ = 1) → INVALID for a REML comparison.
        diff_mean_reml = drm(bf(@formula(y ~ 1), @formula(sigma ~ 1)), Gaussian(); data, method = :REML)

        # Variance-only REML comparison: allowed (no error). full has fewer params
        # than var_only here (sigma 1 vs 1+x), so order (full_reml, var_only_reml).
        @test lrtest(full_reml, var_only_reml) isa NamedTuple

        # Cross-mean-structure REML comparison: must ERROR (the classic trap).
        @test_throws ArgumentError lrtest(diff_mean_reml, full_reml)
        @test_throws ArgumentError anova(diff_mean_reml, full_reml)

        # All-ML comparison across mean structures stays allowed (no guard).
        full_ml = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian(); data)
        red_ml  = drm(bf(@formula(y ~ 1),     @formula(sigma ~ 1)), Gaussian(); data)
        @test lrtest(red_ml, full_ml) isa NamedTuple

        # aic/bic on a REML fit emit a one-time warning but still return a number.
        @test isfinite(aic(full_reml))
        @test isfinite(bic(full_reml))
    end

    @testset "unsupported REML cells are rejected clearly" begin
        # Ordinary mean (1 | g) REML is #439 (`test/test_reml_ordinary_ranef.jl`).
        # Slopes and unknown method symbols stay rejected here.
        Random.seed!(7)
        g = repeat(1:10, inner = 6)
        x2 = randn(60)
        yy = 1.0 .+ randn(10)[g] .+ 0.5 .* randn(60)
        d2 = (; y = yy, g = g, x = x2)
        @test_throws ArgumentError drm(bf(@formula(y ~ 1 + x + (0 + x | g)), @formula(sigma ~ 1)),
                                       Gaussian(); data = d2, method = :REML)
        # An unknown method symbol errors.
        @test_throws ArgumentError drm(bf(@formula(y ~ 1), @formula(sigma ~ 1)),
                                       Gaussian(); data, method = :restricted)
    end
end
