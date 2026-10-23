# Sparse all-node solvers for the Gaussian phylogenetic mean cell.
#
# The EM path fits the Gaussian phylogenetic-MEAN cell — a single phylo(1 | g)
# structured mean random effect with a constant residual scale — by closed-form
# E/M steps with exact O(p) Takahashi traces. The default sparse L-BFGS path uses
# the same all-node marginal with exact two-variance Takahashi gradients. On a
# balanced tree both sparse routes match the legacy dense GLS/LBFGS surface, so
# the correctness anchor is: same μ coefficients, same residual σ, and same
# marginal logLik (rtol ~1e-3).
using DRModels
using Test, Random, LinearAlgebra

@testset "Gaussian phylo mean: :auto uses sparse L-BFGS and :em remains available" begin
    Random.seed!(20260607)
    # A power-of-two leaf count makes `random_balanced_tree` perfectly balanced
    # (no leftover odd node carried up), so every leaf is the SAME depth from the
    # root. Then the Brownian leaf covariance C has a constant diagonal, i.e.
    # C = c·Kc with Kc the correlation. The default GLS fitter parametrizes the
    # phylo effect as σ_s²·Kc while the EM uses σ²_phy·C; with a constant diagonal
    # these are the SAME marginal model (σ_s² = c·σ²_phy), so β, residual σ, and
    # logLik coincide — the correctness anchor. (On an unbalanced tree the two
    # marginals genuinely differ and only logLik/σ stay close, not β.)
    G = 64
    phy = random_balanced_tree(G; branch_length=0.3)
    C = sigma_phy_dense(phy; σ²_phy=1.0)            # phylo covariance over leaves
    d = sqrt.(diag(C))
    Kc = C ./ (d * d')            # correlation
    m = 4
    n = G * m
    species = repeat(1:G; inner=m)
    x = randn(n)
    β = [0.2, 0.5]
    σ = 0.4
    σs = 0.9
    u = σs .* (cholesky(Symmetric(Kc)).L * randn(G))
    y = β[1] .+ β[2] .* x .+ u[species] .+ σ .* randn(n)
    data = (; y, x, species)

    form = bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1))
    fit_def = drm(form, Gaussian(); data=data, tree=phy)                       # :auto -> sparse L-BFGS
    fit_em = drm(form, Gaussian(); data=data, tree=phy, algorithm=:em)
    fit_gls = drm(form, Gaussian(); data=data, tree=phy, algorithm=:gls)
    fit_slb = drm(form, Gaussian(); data=data, tree=phy, algorithm=:sparse_lbfgs)

    # (a) Default and forced sparse-LBFGS routes are identical.
    @test coef(fit_slb, :mu) ≈ coef(fit_def, :mu) rtol = 1e-6
    @test exp(coef(fit_slb, :sigma)[1]) ≈ exp(coef(fit_def, :sigma)[1]) rtol = 1e-6
    @test loglik(fit_slb) ≈ loglik(fit_def) rtol = 1e-6
    @test fit_def.nll !== nothing
    @test all(isfinite, diag(vcov(fit_def))[1:2])
    # #556: the sparse route's variance block is now COMPUTED (profiled
    # curvature), not NaN-by-construction — assert finite and positive.
    @test all(isfinite, diag(vcov(fit_def))[3:4])
    @test all(>(0), diag(vcov(fit_def))[3:4])

    # (b) Explicit EM and legacy dense GLS remain available and reach the
    # same likelihood surface on this balanced-tree anchor.
    @test coef(fit_em, :mu) ≈ coef(fit_def, :mu) rtol = 1e-3
    @test exp(coef(fit_em, :sigma)[1]) ≈ exp(coef(fit_def, :sigma)[1]) rtol = 1e-3
    @test loglik(fit_em) ≈ loglik(fit_def) rtol = 1e-3
    @test fit_em.converged
    @test coef(fit_def, :mu) ≈ coef(fit_gls, :mu) rtol = 1e-3
    @test exp(coef(fit_def, :sigma)[1]) ≈ exp(coef(fit_gls, :sigma)[1]) rtol = 1e-3
    @test loglik(fit_def) ≈ loglik(fit_gls) rtol = 1e-3

    # (c) Sparse L-BFGS stores the full sparse objective for profiling.
    @test isfinite(fit_slb.nll(fit_slb.theta))
    @test fit_slb.nllgrad !== nothing
    θ_probe = fit_slb.theta .+ [0.01, -0.02, 0.015, -0.01]
    g_stored = zeros(length(θ_probe))
    fit_slb.nllgrad(g_stored, θ_probe)
    h = 1e-5
    g_fd = similar(g_stored)
    for j in eachindex(θ_probe)
        step = zeros(length(θ_probe))
        step[j] = h
        g_fd[j] = (fit_slb.nll(θ_probe .+ step) - fit_slb.nll(θ_probe .- step)) / (2h)
    end
    @test maximum(abs.(g_stored .- g_fd)) < 1e-4

    prof_serial = profile_result(fit_def; parm=:resd)
    prof_threaded = profile_result(fit_def; parm=:resd, threads=true)
    @test prof_serial.autodiff === :loconly
    @test prof_threaded.autodiff === :loconly
    @test prof_threaded.ci == prof_serial.ci
    @test prof_threaded.threaded == (Threads.nthreads() > 1)
    @test all(isfinite(r.lower) && isfinite(r.upper) for r in prof_serial.ci)

    # (d) accessors / summary run on the EM fit.
    @test nobs(fit_em) == n
    @test length(coef(fit_em)) == length(coef(fit_def))
    @test isfinite(loglik(fit_em))
    @test haskey(re_sd(fit_em), :species)              # EM reports its Brownian phylo SD
    @test re_sd(fit_em)[:species] > 0
    @test fitted(fit_em) ≈ fitted(fit_def) rtol = 1e-3
    @test length(sigma(fit_em)) == n
    # EM yields no coefficient vcov → NaNs (documented).
    @test all(isnan, vcov(fit_em))
    @test sprint(show, fit_em) isa String              # show() runs
end

@testset "conjugate EM (#12): :em rejects unsupported cells" begin
    Random.seed!(99)
    G = 24
    phy = random_balanced_tree(G; branch_length=0.3)
    m = 4
    n = G * m
    species = repeat(1:G; inner=m)
    x = randn(n)
    y = 0.2 .+ 0.5 .* x .+ 0.4 .* randn(n)
    data = (; y, x, species)

    # (b) non-constant sigma is unsupported.
    @test_throws ArgumentError drm(
        bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ x)),
        Gaussian();
        data=data,
        tree=phy,
        algorithm=:sparse_lbfgs,
    )

    # No structured mean RE at all is unsupported.
    @test_throws ArgumentError drm(
        bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data=data, algorithm=:em
    )

    # A plain (non-phylo) random intercept is unsupported.
    @test_throws ArgumentError drm(
        bf(@formula(y ~ x + (1 | species)), @formula(sigma ~ 1)),
        Gaussian();
        data=data,
        algorithm=:em,
    )

    # An unknown algorithm symbol is rejected up front.
    @test_throws ArgumentError drm(
        bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data=data, algorithm=:nope
    )
end
