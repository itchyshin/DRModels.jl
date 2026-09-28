# Zero-truncated NB2 randomized quantile residuals must never return NaN, even
# at extreme dispersion (log σ ≲ -20) or extreme small μ.
#
# `_quantile_residuals` (quantile_residuals.jl) previously built the
# zero-truncated CDF as F_t(k) = (Distributions.cdf(NB, k) − F0) / (1 − F0).
# At extreme dispersion the NB2 size r = 1/σ² grows so large that r + μ rounds
# to EXACTLY r in float64: p = r/(r+μ) rounds to exactly 1.0, so
# `Distributions.cdf` returns exactly 1.0 for every k — both the numerator and
# the denominator go to 0.0, and 0/0 = NaN. The same underflow happens when μ
# itself underflows to ~0 (μ/r == 0.0 exactly), regardless of σ. This mirrors
# the cancellation #866/#874 fixed in the hurdle/truncated NB2 LIKELIHOOD via
# `_nb2_logpmf` / `_log1mexp`; this test locks down the analogous fix in the
# quantile-residual driver.
#
# A minimal `DrmFit` is built by hand (via the 11-arg `DrmFit` constructor) so
# extreme (log σ, μ) combinations can be probed directly, without needing an
# optimiser to actually converge to that corner of the parameter space.
using DRModels
using Test, Random, Statistics
import Distributions
using StableRNGs

# A bare-bones univariate DrmFit exposing only what `_quantile_residuals`
# reads: family, means[:mu], obs[:mu], scales. `blocks`/`coefnames`/`theta`/
# `vcov`/`loglik` are placeholders — the driver never touches them.
function _fake_fit(fam, mu::Vector{Float64}, y::Vector{Float64}, scales::Dict{Symbol,Vector{Float64}})
    n = length(y)
    blocks = Pair{Symbol,UnitRange{Int}}[:mu => 1:1]
    coefnames = Pair{Symbol,Vector{String}}[:mu => ["(Intercept)"]]
    theta = [0.0]
    vcov = fill(NaN, 1, 1)
    return DRModels.DrmFit(fam, blocks, coefnames, theta, vcov, NaN, n, true,
                            Dict(:mu => mu), Dict(:mu => y), scales)
end

@testset "TruncatedNegBinomial2 quantile residuals: no NaN at extreme dispersion" begin
    @testset "extreme log σ (was NaN before the fix)" begin
        y = [1.0, 2.0, 5.0, 10.0]
        for logsigma in (-20.0, -30.0, -50.0), mu in (0.5, 5.0, 50.0)
            sigma = fill(exp(logsigma), length(y))
            mus = fill(mu, length(y))
            fit = _fake_fit(TruncatedNegBinomial2(), mus, y, Dict(:sigma => sigma))
            r = DRModels._quantile_residuals(fit, StableRNG(1))
            @test all(isfinite, r)
        end
    end

    @testset "extreme small μ (μ/r underflows to 0 exactly)" begin
        y = [1.0, 2.0, 5.0]
        for logsigma in (-20.0, -50.0), mu in (1e-300, 1e-320)
            sigma = fill(exp(logsigma), length(y))
            mus = fill(mu, length(y))
            fit = _fake_fit(TruncatedNegBinomial2(), mus, y, Dict(:sigma => sigma))
            r = DRModels._quantile_residuals(fit, StableRNG(2))
            @test all(isfinite, r)
        end
    end

    @testset "ordinary parameters: unchanged to 1e-10 (fixed RNG)" begin
        # A pre-fix and post-fix build must agree bit-for-bit (to tolerance) at
        # ordinary (non-extreme) dispersion: the fix must not perturb the
        # already-correct code path.
        y = [1.0, 2.0, 3.0, 4.0, 7.0, 12.0]
        mus = fill(4.0, length(y))
        sigma = fill(exp(-0.5), length(y))   # ordinary log σ = -0.5
        fit = _fake_fit(TruncatedNegBinomial2(), mus, y, Dict(:sigma => sigma))
        r = DRModels._quantile_residuals(fit, StableRNG(42))

        # Reference values computed directly from the pre-fix cdf-ratio formula
        # (Distributions.cdf is numerically exact — no cancellation — at this
        # ordinary dispersion, so it is a valid ground truth here).
        r_ref = let rng = StableRNG(42), std_normal = Distributions.Normal()
            φ = 1 / (exp(-0.5)^2)
            map(y) do yi
                d = Distributions.NegativeBinomial(φ, φ / (φ + 4.0))
                F0 = Distributions.cdf(d, 0)
                denom = 1 - F0
                yiI = round(Int, yi)
                a = (Distributions.cdf(d, yiI - 1) - F0) / denom
                b = (Distributions.cdf(d, yiI) - F0) / denom
                u = clamp(a + (b - a) * rand(rng), eps(), 1 - eps())
                Distributions.quantile(std_normal, u)
            end
        end
        @test all(isfinite, r)
        @test r ≈ r_ref atol=1e-10
    end

    @testset "reproducible under a fixed RNG" begin
        y = [1.0, 3.0, 6.0]
        mus = fill(2.0, length(y))
        sigma = fill(exp(-25.0), length(y))
        fit = _fake_fit(TruncatedNegBinomial2(), mus, y, Dict(:sigma => sigma))
        r1 = DRModels._quantile_residuals(fit, StableRNG(7))
        r2 = DRModels._quantile_residuals(fit, StableRNG(7))
        @test r1 == r2
        @test all(isfinite, r1)
    end
end

@testset "sweep: other discrete drivers stay finite at extreme parameters (no regression)" begin
    # These families/drivers were checked and are NOT NaN sources (they saturate
    # to a large-but-finite residual via the existing `clamp(..., lo, hi)`), but
    # are swept here as a regression guard alongside the TruncatedNegBinomial2 fix.
    @testset "NegBinomial2 (plain, non-truncated)" begin
        y = [0.0, 2.0, 5.0]
        for logsigma in (-20.0, -30.0, -50.0), mu in (0.5, 5.0, 50.0)
            sigma = fill(exp(logsigma), length(y))
            mus = fill(mu, length(y))
            fit = _fake_fit(NegBinomial2(), mus, y, Dict(:sigma => sigma))
            r = DRModels._quantile_residuals(fit, StableRNG(1))
            @test all(isfinite, r)
        end
    end

    @testset "NegBinomial2 hurdle-tagged fit (scales carries :hu)" begin
        y = [0.0, 2.0, 5.0]
        for logsigma in (-20.0, -30.0, -50.0), mu in (0.5, 5.0, 50.0)
            sigma = fill(exp(logsigma), length(y))
            mus = fill(mu, length(y))
            fit = _fake_fit(NegBinomial2(), mus, y, Dict(:sigma => sigma, :hu => fill(0.3, length(y))))
            r = DRModels._quantile_residuals(fit, StableRNG(1))
            @test all(isfinite, r)
        end
    end

    @testset "Poisson hurdle-tagged fit (no σ; extreme μ)" begin
        y = [0.0, 2.0, 5.0]
        for mu in (1e-15, 1e15)
            mus = fill(mu, length(y))
            fit = _fake_fit(DRModels.Poisson(), mus, y, Dict(:hu => fill(0.3, length(y))))
            r = DRModels._quantile_residuals(fit, StableRNG(1))
            @test all(isfinite, r)
        end
    end

    @testset "BetaBinomial (extreme φ = σ⁻²)" begin
        y = [0.0, 5.0, 10.0]
        trials = fill(20.0, length(y))
        for logsigma in (-20.0, -30.0, -50.0), m in (0.01, 0.5, 0.99)
            sigma = fill(exp(logsigma), length(y))
            mus = fill(m, length(y))
            fit = _fake_fit(DRModels.BetaBinomial(), mus, y, Dict(:sigma => sigma, :trials => trials))
            r = DRModels._quantile_residuals(fit, StableRNG(1))
            @test all(isfinite, r)
        end
    end
end
