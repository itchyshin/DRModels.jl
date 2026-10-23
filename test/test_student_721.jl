# #721 (twin: drmTMB #1265) — Student() fixed-effects loglik must be finite and
# correct on the near-Gaussian ν ridge. On `datasets::trees` the optimiser used to
# walk ην to ≈36.8 (ν ≈ 1e16), where `Distributions.TDist` logpdf loses all
# precision (loggamma cancellation), and the fit reported loglik ≈ +1924 with
# converged status. drmTMB reports logLik ≈ -87.82 on the same model.
using DRModels
using Test, Random
using Distributions: TDist, Normal, logpdf

# R `datasets::trees` (Girth, Volume); n = 31.
const TREES_GIRTH_721 = [8.3, 8.6, 8.8, 10.5, 10.7, 10.8, 11.0, 11.0, 11.1, 11.2, 11.3, 11.4,
    11.4, 11.7, 12.0, 12.9, 12.9, 13.3, 13.7, 13.8, 14.0, 14.2, 14.5, 16.0, 16.3, 17.3, 17.5,
    17.9, 18.0, 18.0, 20.6]
const TREES_VOLUME_721 = [10.3, 10.3, 10.2, 16.4, 18.8, 19.7, 15.6, 18.2, 22.6, 19.9, 24.2,
    21.0, 21.4, 21.3, 19.1, 22.2, 33.8, 27.4, 25.7, 24.9, 34.5, 31.7, 36.3, 38.3, 42.6, 55.4,
    55.7, 58.3, 51.5, 51.0, 77.0]

@testset "#721 Student FE on trees: finite, correct loglik" begin
    data = (; Volume = TREES_VOLUME_721, Girth = TREES_GIRTH_721)
    fit = drm(bf(@formula(Volume ~ Girth), @formula(sigma ~ 1), @formula(nu ~ 1)), Student(); data = data)
    μ̂ = coef(fit, :mu); lσ = coef(fit, :sigma)[1]
    @test μ̂[1] ≈ -36.94436 atol = 0.01          # drmTMB reference
    @test μ̂[2] ≈ 5.065921 atol = 0.001
    @test lσ ≈ 1.414041 atol = 1e-3
    @test isfinite(loglik(fit))
    @test loglik(fit) < 0                          # the old bug reported +1924
    @test loglik(fit) ≈ -87.82236 atol = 0.01    # drmTMB logLik
    # Independent check: the MLE sits on the Gaussian ridge, so the reported
    # loglik must equal the Normal loglik at the reported μ, σ.
    r = (data.Volume .- (μ̂[1] .+ μ̂[2] .* data.Girth)) ./ exp(lσ)
    ll_norm = sum(logpdf.(Normal(), r)) - length(r) * lσ
    @test loglik(fit) ≈ ll_norm atol = 0.01
    # The objective itself must stay on the flat Gaussian ridge as ην grows
    # (optimiser-path independent: the issue's run stopped at ην ≈ 36.8, where the
    # old objective read nll ≈ -1924).
    θ̂ = copy(fit.theta)
    for ην in (20.0, 36.8, 60.0, 800.0)
        θ̂[end] = ην
        @test fit.nll(θ̂) ≈ -ll_norm atol = 0.01
    end
end

@testset "#721 Student logpdf is stable for huge ν" begin
    # Normal limit: ην large ⇒ standardized Student logpdf → Normal logpdf.
    for ην in (15.0, 30.0, 36.8, 60.0, 800.0), z in (0.0, 0.5, 3.0)
        v = DRModels._student_logpdf_std(z, ην)
        @test isfinite(v)
        @test v ≈ logpdf(Normal(), z) atol = 1e-5
    end
    # Agrees with Distributions.TDist where TDist is accurate (moderate ν).
    for ην in (-3.0, 0.0, log(3.0), 2.0, 5.0, 9.0), z in (0.0, 0.7, -2.5, 40.0)
        @test DRModels._student_logpdf_std(z, ην) ≈ logpdf(TDist(2 + exp(ην)), z) rtol = 1e-9
    end
end

@testset "#721 heavy-tailed response still recovers small ν" begin
    rng = MersenneTwister(721)
    n = 4000
    x = randn(rng, n)
    y = 1.0 .+ 0.5 .* x .+ 0.7 .* rand(rng, TDist(4.0), n)
    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)), Student(); data = (; y, x))
    ν̂ = 2 + exp(coef(fit, :nu)[1])
    @test 3.0 < ν̂ < 5.5
    @test coef(fit, :mu) ≈ [1.0, 0.5] atol = 0.06
    @test exp(coef(fit, :sigma)[1]) ≈ 0.7 atol = 0.06
    @test isfinite(loglik(fit))
    @test fit.converged
end
