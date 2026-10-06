using DRModels
using Test, Random, Statistics, LinearAlgebra

# Knuth Poisson sampler (no Distributions dependency in the test).
function _rpois_pf(rng, λ)
    L = exp(-λ); k = 0; p = 1.0
    while true
        k += 1
        p *= rand(rng)
        p <= L && return k - 1
    end
end

@testset "cross-family post-fit accessors (mf_*)" begin
    rng = MersenneTwister(20260610)
    n = 300
    x = randn(rng, n)
    X1 = hcat(ones(n), x); X2 = hcat(ones(n), x)
    β1 = [0.5, 0.8]; β2 = [0.3, -0.5]
    λ1 = 0.7; λ2 = 0.6; σ1 = 0.5
    u = randn(rng, n)
    y1 = X1 * β1 .+ λ1 .* u .+ σ1 .* randn(rng, n)
    η2 = X2 * β2 .+ λ2 .* u
    y2 = Float64[_rpois_pf(rng, exp(clamp(η2[i], -20.0, 20.0))) for i in 1:n]

    fit = DRModels.fit_mixed_family(y1 = y1, X1 = X1, fam1 = Gaussian(),
                               y2 = y2, X2 = X2, fam2 = DRModels.Poisson(),
                               confint = false)
    @test fit.converged

    # parameter count: β1(2) + β2(2) + 2 loadings + βσ1(1, Gaussian) + βσ2(0, Poisson)
    @test DRModels._mf_nparams(fit) == 7

    @testset "mf_coef" begin
        tbl = mf_coef(fit)
        # three equal-length parallel vectors
        @test length(tbl.axis) == length(tbl.term) == length(tbl.estimate)
        # rows = nparams (β1,β2,βσ1,βσ2,λ1,λ2) + 1 for rho
        @test length(tbl.term) == DRModels._mf_nparams(fit) + 1
        @test "rho" in tbl.term
        @test "lambda1" in tbl.term && "lambda2" in tbl.term
        @test "bsig1[1]" in tbl.term            # Gaussian dispersion coefficient present
        @test !any(startswith.(tbl.term, "bsig2")) # Poisson axis is dispersionless
        @test all(isfinite, tbl.estimate)
        # the rho row matches the fit field
        @test tbl.estimate[findfirst(==("rho"), tbl.term)] == fit.rho_latent
        # the b1[1] row matches the fit's first fixed effect
        @test tbl.estimate[findfirst(==("b1[1]"), tbl.term)] == fit.β1[1]
    end

    @testset "mf_aic / mf_bic" begin
        a = mf_aic(fit)
        b = mf_bic(fit; nobs = n)
        @test isfinite(a) && isfinite(b)
        # -2logL + 2k vs -2logL + k·log n: bic > aic whenever log(n) > 2, i.e. n ≥ 8.
        @test b > a
        # explicit formula agreement
        k = DRModels._mf_nparams(fit)
        @test a ≈ -2 * fit.loglik + 2 * k
        @test b ≈ -2 * fit.loglik + k * log(n)
        @test_throws ArgumentError mf_bic(fit; nobs = 0)
    end

    @testset "mf_fitted (u = 0 convention)" begin
        ft = mf_fitted(fit, X1, X2)
        @test length(ft.mu1) == n && length(ft.mu2) == n
        @test all(isfinite, ft.mu1) && all(isfinite, ft.mu2)
        # Gaussian axis: identity link ⇒ μ1 = X1 β1 exactly.
        @test ft.mu1 ≈ X1 * fit.β1
        # Poisson axis: log link ⇒ μ2 = exp(X2 β2), and strictly positive.
        @test ft.mu2 ≈ exp.(X2 * fit.β2)
        @test all(>(0), ft.mu2)
        # wrong design width is rejected
        @test_throws DimensionMismatch mf_fitted(fit, X1[:, 1:1], X2)
    end

    @testset "mf_summary runs" begin
        buf = IOBuffer()
        ret = mf_summary(fit; nobs = n, io = buf)
        @test ret === fit                       # returns the fit
        s = String(take!(buf))
        @test occursin("rho", s)
        @test occursin("AIC", s) && occursin("BIC", s)
        @test occursin("logLik", s)
        # without nobs: still runs, BIC omitted
        buf2 = IOBuffer()
        mf_summary(fit; io = buf2)
        s2 = String(take!(buf2))
        @test occursin("AIC", s2) && !occursin("BIC", s2)
    end
end
