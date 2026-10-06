# eta-clamp twin-parity guard (#324). drmTMB does NOT clamp the mean predictor
# and soft-clamps only the log-scale (identity, then a smooth tanh tail). DRModels.jl
# previously HARD-clamped the mean and scale predictors in the NB2, Gamma and Beta
# density paths, so the two twins' likelihood surfaces (loglik AND gradient)
# diverged at extreme eta: beyond the hard bound the gradient was a flat zero,
# which drmTMB never produces. The fix replaces those hard clamps with
# `DRModels._softclamp`, identity across each family's original hard band (so no
# legitimate fit is distorted — including a near-Poisson NB2 whose dispersion
# eta_sigma sits near the old boundary) and C1-smooth beyond.
using DRModels
using Test, Random, ForwardDiff
import Distributions

_logistic_test(η) = 1 / (1 + exp(-η))

@testset "eta-clamp twin parity (#324): soft-clamp ports drmTMB's guard" begin
    sc = DRModels._softclamp

    @testset "identity inside the band (bit-for-bit, like drmTMB)" begin
        for x in (-12.0, -11.0, -5.0, -1.0, 0.0, 0.7, 3.0, 11.9, 12.0)
            @test sc(x, -12.0, 12.0, 3.0) === x          # exact identity, not just ≈
        end
        # a wide mean band is also identity across the whole well-posed regime
        for x in (-17.0, -8.0, 0.0, 8.0, 17.0)
            @test sc(x, -17.0, 17.0, 3.0) === x
        end
    end

    @testset "bounded + C1-smooth (nonzero gradient) beyond the band" begin
        lo, hi, m = -12.0, 12.0, 3.0
        # saturates within `margin`: overall range [lo-m, hi+m] = [-15, 15]
        # (tanh(∞)=1 in Float64, so a full runaway lands exactly on the endpoint)
        @test sc(1e6, lo, hi, m) <= hi + m
        @test sc(1e6, lo, hi, m) > hi
        @test sc(-1e6, lo, hi, m) >= lo - m
        @test sc(-1e6, lo, hi, m) < lo
        # unlike a hard clamp, the gradient stays strictly positive past the bound
        g = ForwardDiff.derivative(x -> sc(x, lo, hi, m), 20.0)
        @test g > 0
        @test ForwardDiff.derivative(x -> sc(x, lo, hi, m), -20.0) > 0
        # continuity at the knot (C0) and derivative continuity (C1 → 1 from inside)
        @test sc(hi, lo, hi, m) ≈ sc(nextfloat(hi), lo, hi, m) atol = 1e-9
        @test ForwardDiff.derivative(x -> sc(x, lo, hi, m), hi - 1e-6) ≈ 1.0 atol = 1e-4
        @test ForwardDiff.derivative(x -> sc(x, lo, hi, m), hi + 1e-6) ≈ 1.0 atol = 1e-2
    end

    @testset "the old hard clamp diverged from drmTMB at extreme eta_mu" begin
        # drmTMB leaves the mean unclamped; the old DRModels.jl hard clamp zeroed the
        # gradient beyond the bound. Reproduce the divergence the fix removes.
        r = exp(-2 * 0.0); y = 5
        nb_none(b) = -Distributions.logpdf(Distributions.NegativeBinomial(r, r / (r + exp(b))), y)
        nb_hard(b) = -Distributions.logpdf(Distributions.NegativeBinomial(r, r / (r + exp(clamp(b, -20.0, 20.0)))), y)
        nb_soft(b) = -Distributions.logpdf(Distributions.NegativeBinomial(r, r / (r + exp(sc(b, -20.0, 20.0, 3.0)))), y)
        b = 25.0
        gnone = ForwardDiff.derivative(nb_none, b)
        ghard = ForwardDiff.derivative(nb_hard, b)
        # OLD behaviour: hard clamp -> flat gradient (diverges from drmTMB)
        @test ghard == 0.0
        @test abs(gnone) > 1e-6
        # NEW behaviour: soft guard keeps a live (nonzero) gradient like drmTMB
        gsoft = ForwardDiff.derivative(nb_soft, b)
        @test gsoft != 0.0
    end

    @testset "guard is inert on well-posed fits (results unchanged)" begin
        Random.seed!(20260703)
        n = 1500
        x = randn(n)

        # NB2: recovers the mean coefficients; guard never bites.
        θ = 3.0
        μ = exp.(0.3 .+ 0.4 .* x)
        ynb = Float64.([rand(Distributions.NegativeBinomial(θ, θ / (θ + μi))) for μi in μ])
        fnb = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), NegBinomial2(); data = (; y = ynb, x))
        @test coef(fnb, :mu)[1] ≈ 0.3 atol = 0.12
        @test coef(fnb, :mu)[2] ≈ 0.4 atol = 0.12
        @test isfinite(loglik(fnb))

        # Gamma: log-mean recovery.
        α = 4.0
        μg = exp.(0.2 .+ 0.5 .* x)
        yg = Float64.([rand(Distributions.Gamma(α, μi / α)) for μi in μg])
        fg = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gamma(); data = (; y = yg, x))
        @test coef(fg, :mu)[1] ≈ 0.2 atol = 0.12
        @test coef(fg, :mu)[2] ≈ 0.5 atol = 0.12
        @test isfinite(loglik(fg))

        # Beta: logit-mean recovery.
        φ = 12.0
        μb = _logistic_test.(0.1 .+ 0.6 .* x)
        yb = Float64.([rand(Distributions.Beta(m * φ, (1 - m) * φ)) for m in μb])
        fb = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Beta(); data = (; y = yb, x))
        @test coef(fb, :mu)[1] ≈ 0.1 atol = 0.15
        @test coef(fb, :mu)[2] ≈ 0.6 atol = 0.15
        @test isfinite(loglik(fb))
    end

    @testset "near-Poisson NB2 with boundary dispersion fits (regression, #324 CI)" begin
        # Exact dataset from test_bootstrap_nongaussian.jl:47 (same threaded rng
        # sequence). Its ML dispersion runs to eta_sigma ≈ -14.6 (near-Poisson). A
        # scale band tighter than the original hard [-20,20] (e.g. drmTMB's Gaussian
        # [-12,12]) would saturate that direction, flatten the sigma-axis curvature,
        # and make the Hessian inversion throw SingularException on the Linux/x86
        # BLAS path (green on macOS/arm64). Keeping the identity band at the original
        # [-20,20] leaves this fit undistorted and invertible on both paths.
        rng = MersenneTwister(20260804)
        n = 180
        x = randn(rng, n)
        ntr = fill(8, n)
        p = @. 1 / (1 + exp(-(0.2 + 0.7x)))
        # advance the rng exactly as the source testset does before the NB2 draw
        _ = Float64[rand(rng, Distributions.Binomial(ntr[i], p[i])) for i in 1:n]
        _ = Float64[rand(rng, Distributions.BetaBinomial(ntr[i], p[i] * 12.0, (1 - p[i]) * 12.0)) for i in 1:n]
        μ = exp.(0.3 .+ 0.4 .* x)
        ynb = Float64[rand(rng, Distributions.NegativeBinomial(4.0, 4.0 / (4.0 + μ[i]))) for i in 1:n]
        local fit_nb
        @test (fit_nb = drm(bf(@formula(ynb ~ x), @formula(sigma ~ 1)), NegBinomial2();
                            data = (; ynb, x)); true)   # must not throw (was SingularException)
        @test all(isfinite, fit_nb.vcov)
        @test all(isfinite, coef(fit_nb, :mu))
    end
end
