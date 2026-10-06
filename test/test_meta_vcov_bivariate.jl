# test_meta_vcov_bivariate.jl — A8: bivariate meta-analysis with known sampling
# covariance, the engine path that unblocks drmTMB's `meta_vcov_bivariate`.
#
# WHY THIS EXISTS. The A4d design pass refused to port `meta_vcov_bivariate`
# because "the output would have no consumer": DRModels.jl's `meta_V` was
# diagonal-univariate only and the bivariate route ignored it. This slice builds
# the consumer — per-study known 2x2 sampling covariance added to the bivariate
# residual model, S_i = V_i + [[sigma1^2, rho*sigma1*sigma2], [., sigma2^2]] —
# and then the constructor is worth porting.
#
# THE CORRECTNESS ANCHOR is deliberately independent of drmTMB: the fitted
# log-likelihood must match a from-scratch dense-MVN computation over the stacked
# 2n response (a port of drmTMB's own "matches a base R MVN calculation" test).
# Cross-implementation parity vs native drmTMB lives in tools/parity_biv_meta.R;
# THIS file must not be able to pass by both implementations sharing a bug.

using DRModels
using Test
using Random
using LinearAlgebra
using Statistics

# The drmTMB test DGP (test-biv-gaussian.R, new_biv_gaussian_known_v_data),
# ported: per-row S_i = V_i + heterogeneity, sampling_cor inside V, residual_rho
# in the heterogeneity — two DIFFERENT correlations the fit must separate.
function _biv_meta_fixture(; n = 160, residual_rho = -0.35, sampling_cor = 0.6,
                           sigma1 = 0.45, sigma2 = 0.55, seed = 20260516)
    rng = MersenneTwister(seed)
    x = randn(rng, n)
    mu1 = 0.2 .+ 0.5 .* x
    mu2 = -0.1 .- 0.35 .* x
    v1 = 0.01 .+ 0.03 .* rand(rng, n)
    v2 = 0.01 .+ 0.04 .* rand(rng, n)
    y1 = similar(x); y2 = similar(x)
    for i in 1:n
        c12 = sampling_cor * sqrt(v1[i] * v2[i]) + residual_rho * sigma1 * sigma2
        S = [v1[i] + sigma1^2  c12; c12  v2[i] + sigma2^2]
        z = cholesky(Symmetric(S)).L * randn(rng, 2)
        y1[i] = mu1[i] + z[1]
        y2[i] = mu2[i] + z[2]
    end
    (data = (; x, y1, y2), v1 = v1, v2 = v2,
     residual_rho = residual_rho, sampling_cor = sampling_cor,
     sigma1 = sigma1, sigma2 = sigma2)
end

const _BF_BIV = bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
                   sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                   rho12 = @formula(rho12 ~ 1))

@testset "meta_vcov_bivariate constructor (drmTMB parity)" begin
    v1 = [0.01, 0.02, 0.03]; v2 = [0.02, 0.01, 0.04]

    @testset "cor12 route computes the covariance" begin
        V = meta_vcov_bivariate(v1, v2; cor12 = 0.5)
        @test V isa MetaVcovBivariate
        @test V.v1 == v1 && V.v2 == v2
        @test V.cov12 ≈ 0.5 .* sqrt.(v1 .* v2)
    end

    @testset "cov12 route stores as given; default is independence" begin
        V = meta_vcov_bivariate(v1, v2; cov12 = [0.001, 0.002, 0.003])
        @test V.cov12 == [0.001, 0.002, 0.003]
        @test all(meta_vcov_bivariate(v1, v2).cov12 .== 0.0)
    end

    @testset "the dense drmTMB-shaped matrix: 2n x 2n, y1/y2-paired blocks" begin
        V = meta_vcov_bivariate(v1, v2; cor12 = 0.5)
        M = Matrix(V)
        @test size(M) == (6, 6)
        for i in 1:3
            r = 2i - 1
            @test M[r, r] == v1[i]
            @test M[r+1, r+1] == v2[i]
            @test M[r, r+1] == M[r+1, r] == V.cov12[i]
        end
        # off-block entries are exactly zero
        @test M[1, 3] == 0.0 && M[2, 5] == 0.0
    end

    @testset "validation mirrors drmTMB's" begin
        @test_throws ArgumentError meta_vcov_bivariate(Float64[], Float64[])
        @test_throws ArgumentError meta_vcov_bivariate(v1, v2[1:2])
        @test_throws ArgumentError meta_vcov_bivariate([-0.01, 0.02, 0.03], v2)
        @test_throws ArgumentError meta_vcov_bivariate(v1, v2; cor12 = 1.5)
        @test_throws ArgumentError meta_vcov_bivariate(v1, v2; cor12 = 0.3,
                                                       cov12 = [0.0, 0.0, 0.0])
        # a cov12 that makes a sampling block non-PSD is refused up front
        @test_throws ArgumentError meta_vcov_bivariate(v1, v2;
                                                       cov12 = [0.5, 0.0, 0.0])
    end

    @testset "a dense matrix round-trips; off-block contamination refuses" begin
        V = meta_vcov_bivariate(v1, v2; cor12 = 0.4)
        M = Matrix(V)
        W = MetaVcovBivariate(M)          # dense -> compact
        @test W.v1 ≈ V.v1 && W.v2 ≈ V.v2 && W.cov12 ≈ V.cov12
        M2 = copy(M); M2[1, 3] = 1e-3     # cross-study covariance: unsupported
        @test_throws ArgumentError MetaVcovBivariate(M2)
    end
end

@testset "bivariate known-V engine (A8)" begin
    fx = _biv_meta_fixture()
    V = meta_vcov_bivariate(fx.v1, fx.v2; cor12 = fx.sampling_cor)

    @testset "loglik matches an independent dense-MVN computation" begin
        fit = drm(_BF_BIV, Gaussian(); data = fx.data, V = V)
        @test is_converged(fit)
        s1 = fit.scales[:sigma1][1]; s2 = fit.scales[:sigma2][1]
        ρ = fit.scales[:rho12][1]
        n = length(fx.data.y1)
        # stacked 2n MVN with dense Sigma = V + heterogeneity — from scratch,
        # sharing NOTHING with the fitter's per-row expressions
        μ1 = fit.means[:mu1]; μ2 = fit.means[:mu2]
        y = Vector{Float64}(undef, 2n); μ = similar(y)
        Σ = Matrix(V)
        for i in 1:n
            r = 2i - 1
            y[r] = fx.data.y1[i]; y[r+1] = fx.data.y2[i]
            μ[r] = μ1[i];         μ[r+1] = μ2[i]
            Σ[r, r]     += s1^2
            Σ[r+1, r+1] += s2^2
            Σ[r, r+1]   += ρ * s1 * s2
            Σ[r+1, r]   += ρ * s1 * s2
        end
        C = cholesky(Symmetric(Σ))
        resid = y .- μ
        ll_direct = -0.5 * (2n * log(2π) + logdet(C) + dot(resid, C \ resid))
        @test loglik(fit) ≈ ll_direct atol = 1e-6
    end

    @testset "residual rho12 is separated from the sampling correlation" begin
        # The fixture bakes in BOTH correlations. If the engine ignored V, the
        # fitted rho12 would absorb the sampling correlation (pulled toward
        # +0.6-ish); with V consumed it must sit near the true -0.35 instead.
        fx2 = _biv_meta_fixture(n = 300, seed = 20260517)
        V2 = meta_vcov_bivariate(fx2.v1, fx2.v2; cor12 = fx2.sampling_cor)
        fit = drm(_BF_BIV, Gaussian(); data = fx2.data, V = V2)
        ρ̂ = fit.scales[:rho12][1]
        @test ρ̂ < 0                       # sign alone rules out absorption
        # Finite-sample recovery band, not a collapse check. Same
        # MersenneTwister(20260517) fixture draws different randn streams on
        # Julia 1.10 vs 1.12, so |ρ̂+0.35| is 0.068 on 1.10 and 0.183 on 1.12
        # (Linux CI job 95705490872 and local 1.12.6, bit-identical). Sign and
        # the no-V contrast below still separate residual rho from sampling
        # cor = +0.6; a "closer to DGP than to sampling" rewrite is too loose
        # (the no-V 1.12 fit also passes it). 0.15 → 0.20 covers the measured
        # 1.12 miss without hiding a real absorption.
        @test abs(ρ̂ - fx2.residual_rho) < 0.20
        novi = drm(_BF_BIV, Gaussian(); data = fx2.data)
        # and WITHOUT V the estimate visibly shifts toward the contaminated blend
        @test fit.scales[:rho12][1] < novi.scales[:rho12][1]
    end

    @testset "the dense drmTMB-shaped matrix is accepted directly" begin
        fit_c = drm(_BF_BIV, Gaussian(); data = fx.data, V = V)
        fit_m = drm(_BF_BIV, Gaussian(); data = fx.data, V = Matrix(V))
        @test loglik(fit_c) ≈ loglik(fit_m) atol = 1e-10
        @test coef(fit_c) ≈ coef(fit_m) atol = 1e-10
    end

    @testset "refusals" begin
        # wrong length
        @test_throws ArgumentError drm(_BF_BIV, Gaussian(); data = fx.data,
            V = meta_vcov_bivariate(fx.v1[1:10], fx.v2[1:10]))
        # V with a structured bivariate route: not in this slice
        bfq2 = bf(mu1 = @formula(y1 ~ x + phylo(1 | g)),
                  mu2 = @formula(y2 ~ x + phylo(1 | g)),
                  sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                  rho12 = @formula(rho12 ~ 1))
        g = repeat(1:16, inner = 10)
        dat2 = (; fx.data..., g)
        @test_throws ArgumentError drm(bfq2, Gaussian(); data = dat2, V = V,
                                       tree = random_balanced_tree(16))
    end
end
