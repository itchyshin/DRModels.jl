using DRModels
using Test, DelimitedFiles
const DRM = DRModels

# Cross-family profile CI for rho: an unresolved far point must not discard a
# resolvable arm.
#
# `_mf_profile_ci` decides whether an arm crosses the chi-square cutoff from one
# inner solve at rho0 = +-0.999. On the NB2 x Gaussian fixture (drmTMB
# test-xfam-bridge.R, n = 150) that solve stalls (gradient residual 1.9e-4 after
# 252 iterations, nll 1107.8), and the whole upper arm was returned NaN without
# ever bisecting, although a finite bound exists (profile crossing ~0.7993; Wald
# upper 0.791). Before the fix: (0.4699741, NaN).
#
# Data are committed CSVs (generated in R; see the fixture README), so the test
# does not depend on the Julia version's RNG.

const _MFPF_DIR = joinpath(@__DIR__, "fixtures", "mf_profile_far_point")

function _mfpf_fit(name, fam1, fam2)
    m = readdlm(joinpath(_MFPF_DIR, name * ".csv"), ',', Float64; skipstart = 1)
    y1, y2, x = m[:, 1], m[:, 2], m[:, 3]
    X = hcat(ones(length(x)), x)
    return DRM.fit_mixed_family(; y1 = y1, X1 = X, fam1 = fam1, y2 = y2, X2 = X,
                            fam2 = fam2, profile = true)
end

@testset "mixed_family profile CI: far-point failure does not drop an arm" begin
    @testset "NB2 x Gaussian: finite upper bound near the profile crossing" begin
        f = _mfpf_fit("nb2_gaussian", DRM.NegBinomial2(), DRM.Gaussian())
        @test f.rho_latent ≈ 0.65901 atol = 1e-5
        @test f.loglik ≈ -449.180116 atol = 1e-5
        lo, hi = f.rho_ci_profile
        @test isfinite(lo) && isfinite(hi)
        @test lo ≈ 0.4699741 atol = 1e-6            # lower arm unchanged
        @test f.rho_latent <= hi
        @test hi > lo
        # Crossing of the chi-square(1, .95) cutoff measured at ~0.7993 with tight
        # inner solves; the bisection stops at a 1e-3 bracket.
        @test 0.797 < hi < 0.801
        @test hi > f.rho_ci_wald[2]                 # profile is wider than Wald here
    end

    @testset "synthetic: MLE next to the boundary still reaches the far point" begin
        # theta-hat at rho = tanh(3) = 0.995, only 0.004 from the 0.999 cutoff: the
        # far point must still be tried (a "beyond rho-hat by 0.01" rule on it
        # would have returned NaN). The curvature is flat enough that the upper
        # arm never crosses (bound 0.999) and the lower arm crosses at theta = -0.92.
        flat(θ) = (θ[1] - 3.0)^2 / 8
        lo, hi = DRModels._mf_profile_ci(flat, θ -> tanh(θ[1]), [3.0], tanh(3.0), 0.95)
        @test hi ≈ 0.999
        @test isapprox(lo, tanh(3.0 - 2 * sqrt(3.841458820694124)); atol = 3e-3)
    end

    @testset "other cross-family fits unchanged" begin
        g = _mfpf_fit("gaussian_poisson", DRM.Gaussian(), DRM.Poisson())
        @test g.rho_ci_profile[1] ≈ 0.3268811 atol = 1e-6
        @test g.rho_ci_profile[2] ≈ 0.5857561 atol = 1e-6
        h = _mfpf_fit("gamma_poisson", DRM.Gamma(), DRM.Poisson())
        @test h.rho_ci_profile[1] ≈ 0.0360536 atol = 1e-6
        @test h.rho_ci_profile[2] ≈ 0.3253959 atol = 1e-6
    end
end
