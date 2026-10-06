using DRModels
using Test, Random
using StableRNGs   # cross-Julia-version reproducible streams (MersenneTwister differs on 1.13)

# Sentinel handling in the cross-family route. `fit_mixed_family` maps a
# non-finite objective to a 1e10 plateau; the profile CI, the bootstrap and
# AIC/BIC must not read that plateau as data.

function _rp_sent(rng, λ)
    L = exp(-λ); k = 0; p = 1.0
    while true
        k += 1; p *= rand(rng); p <= L && return k - 1
    end
end

@testset "mixed_family sentinel guards" begin
    @testset "(a) profile CI: sentinel inner solve is unresolved, not crossed" begin
        # Flat objective: dev(t) = t^2/4, so chi2(1, .95) is crossed only at
        # t = 3.92 > atanh(0.999), i.e. the TRUE upper profile arm is the bound
        # 0.999. A cliff at t > 1.5 maps to the 1e10 plateau. The penalised inner
        # solve at rho0 = 0.999 is dragged onto that plateau.
        cliff(θ) = θ[1] > 1.5 ? sum(θ) * 0 + 1e10 : θ[1]^2 / 8
        lo, hi = DRModels._mf_profile_ci(cliff, θ -> tanh(θ[1]), [0.0], 0.0, 0.95)
        @test lo ≈ -0.999                 # clean arm untouched
        @test isnan(hi)                   # unresolved arm flagged NaN (was ≈ tanh(1.5) = 0.905)

        # Sentinel everywhere except at the optimum: the inner solve cannot move,
        # so it can never reach rho0 and both arms are unresolved (was the bounds
        # (-0.999, 0.999), read off a dev of 0).
        island(θ) = θ[1] == 0.0 ? θ[1]^2 / 8 : sum(θ) * 0 + 1e10
        @test all(isnan, DRModels._mf_profile_ci(island, θ -> tanh(θ[1]), [0.0], 0.0, 0.95))

        # No cliff: an ordinary interior crossing is still found.
        smooth(θ) = θ[1]^2 / 2
        lo2, hi2 = DRModels._mf_profile_ci(smooth, θ -> tanh(θ[1]), [0.0], 0.0, 0.95)
        @test isapprox(hi2, tanh(sqrt(3.841458820694124)); atol = 3e-3)
        @test isapprox(lo2, -tanh(sqrt(3.841458820694124)); atol = 3e-3)
    end

    @testset "(b) bootstrap skips non-converged / sentinel refits and reports the count" begin
        good = [(; rho_latent = 0.40 + 0.01 * i, converged = true, loglik = -100.0) for i in 1:12]
        bad_nc = [(; rho_latent = 0.999, converged = false, loglik = -100.0) for _ in 1:4]
        bad_sen = [(; rho_latent = -0.999, converged = true, loglik = -1e10) for _ in 1:4]
        fits = Any[good; bad_nc; bad_sen; nothing]
        ci, kept = DRModels._mf_boot_ci(fits, 21, 0.95)
        @test kept == 12
        @test 0.40 <= ci[1] < ci[2] <= 0.52   # was (-0.999, 0.999)-contaminated
    end

    @testset "(c) mf_aic / mf_bic: NaN + warning on a sentinel or non-converged fit" begin
        rng = StableRNG(11)
        n = 120
        x = randn(rng, n); X = hcat(ones(n), x); u = randn(rng, n)
        y1 = X * [0.5, 0.8] .+ 0.7 .* u .+ 0.5 .* randn(rng, n)
        η2 = X * [0.3, -0.5] .+ 0.6 .* u
        y2 = Float64[_rp_sent(rng, exp(clamp(η2[i], -20.0, 20.0))) for i in 1:n]
        fit = DRModels.fit_mixed_family(y1 = y1, X1 = X, fam1 = Gaussian(),
                                        y2 = y2, X2 = X, fam2 = DRModels.Poisson(),
                                        confint = false)
        @test isfinite(DRModels.mf_aic(fit)) && isfinite(DRModels.mf_bic(fit; nobs = n))

        plateau = merge(fit, (; loglik = -1e10))
        v = @test_logs (:warn,) DRModels.mf_aic(plateau)
        @test isnan(v)
        v = @test_logs (:warn,) DRModels.mf_bic(plateau; nobs = n)
        @test isnan(v)
        nc = merge(fit, (; converged = false))
        v = @test_logs (:warn,) DRModels.mf_aic(nc)
        @test isnan(v)
        v = @test_logs (:warn,) DRModels.mf_bic(nc; nobs = n)
        @test isnan(v)
    end

    @testset "ordinary fit unchanged (profile + bootstrap), to 1e-8" begin
        rng = StableRNG(20260929)
        n = 200; x = randn(rng, n); X = hcat(ones(n), x); u = randn(rng, n)
        y1 = X * [0.5, 0.8] .+ 0.7 .* u .+ 0.5 .* randn(rng, n)
        η2 = X * [0.3, -0.5] .+ 0.6 .* u
        y2 = Float64[_rp_sent(rng, exp(clamp(η2[i], -20.0, 20.0))) for i in 1:n]
        f = DRModels.fit_mixed_family(y1 = y1, X1 = X, fam1 = Gaussian(), y2 = y2, X2 = X,
                                      fam2 = DRModels.Poisson(), profile = true, B = 12,
                                      rng = StableRNG(1))
        # The profile endpoints come from a bisection that stops at width < 1e-3, so they
        # are quantised: every decision had a deviance margin >= 4e-3 (measured), and the
        # 1e-8 pin is met to ~1e-15 across OPENBLAS_CORETYPE variants. A 1e-8 miss
        # therefore means a changed bisection path (e.g. an arm lost to NaN), not noise.
        # Values recorded from the pre-change code (base src, StableRNG data), identical on Julia 1.10 and 1.13.
        @test f.rho_latent ≈ 0.4652924608401332 atol = 1e-8
        @test all(isapprox.(f.rho_ci_wald, (0.3367448322273234, 0.5768132074484522); atol = 1e-8))
        @test all(isapprox.(f.rho_ci_profile, (0.33480741489124444, 0.5770895967286014); atol = 1e-8))
        @test all(isapprox.(f.rho_ci_boot, (0.33588692279269083, 0.5249108628415716); atol = 1e-8))
        @test f.loglik ≈ -581.4048758300801 atol = 1e-8
        @test f.n_boot_kept == 12
    end
end
