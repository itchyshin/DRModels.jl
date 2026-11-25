# test_cov_1125n4_rssig_kernel.jl — RS+sigma-RE Laplace kernel (src/gaussian_rs_sigma_ranef.jl).
using DRModels
using Test, LinearAlgebra, ForwardDiff, Random

@testset "cov 1125n4: _rssig_terms consistency" begin
    Random.seed!(42)
    m = 8
    r = randn(m)
    x = randn(m)
    η = fill(-0.2, m)
    u0, u1, δ = 0.1, -0.05, 0.02
    Si11, Si12, Si22, invτ2 = 0.5, 0.1, 0.8, 4.0
    h, g, H = DRModels._rssig_terms(r, x, η, u0, u1, δ, Si11, Si12, Si22, invτ2)
    @test isfinite(h) && all(isfinite, g) && all(isfinite, H)
    @test issymmetric(H)
    @test all(eigvals(Symmetric(H)) .> 0)
    # central FD on u0
    h0, g0, _ = DRModels._rssig_terms(r, x, η, u0, u1, δ, Si11, Si12, Si22, invτ2)
    eps = 1e-6
    hp, _, _ = DRModels._rssig_terms(r, x, η, u0 + eps, u1, δ, Si11, Si12, Si22, invτ2)
    hm, _, _ = DRModels._rssig_terms(r, x, η, u0 - eps, u1, δ, Si11, Si12, Si22, invτ2)
    @test g[1] ≈ (hp - hm) / (2eps) rtol = 1e-4
    # ForwardDiff through mode objective slice
    f(θ) = first(DRModels._rssig_terms(r, x, η, θ[1], θ[2], θ[3], Si11, Si12, Si22, invτ2))
    gfd = ForwardDiff.gradient(f, [u0, u1, δ])
    @test g ≈ gfd rtol = 1e-6 atol = 1e-8
end

@testset "cov 1125n4: _rssig_mode_primal" begin
    Random.seed!(7)
    m = 12
    r, x, η = randn(m), randn(m), fill(0.0, m)
    Si11, Si12, Si22, invτ2 = 1.0, 0.0, 1.0, 2.0
    v0 = DRModels._rssig_mode_primal(r, x, η, Si11, Si12, Si22, invτ2, zeros(3))
    @test length(v0) == 3 && all(isfinite, v0)
    h0, g0, _ = DRModels._rssig_terms(r, x, η, v0[1], v0[2], v0[3], Si11, Si12, Si22, invτ2)
    @test norm(g0, Inf) < 1e-8 * (1 + abs(h0)) + 1e-10
    # second start should not increase penalised objective at convergence
    v1 = DRModels._rssig_mode_primal(r, x, η, Si11, Si12, Si22, invτ2, [0.5, -0.2, 0.1])
    h1, _, _ = DRModels._rssig_terms(r, x, η, v1[1], v1[2], v1[3], Si11, Si12, Si22, invτ2)
    @test h1 ≤ h0 + 1e-10
end

@testset "cov 1125n4: Gaussian (1+x|g) + sigma (1|g) fit" begin
    Random.seed!(1125)
    n = 200
    g = repeat(1:10, outer = 20)
    x = randn(n)
    y = 0.5 .+ 0.7 .* x .+ randn(n)
    d = (; y, x, g)
    fit = drm(bf(@formula(y ~ 1 + x + (1 + x | g)), @formula(sigma ~ 1 + (1 | g))),
              Gaussian(); data = d)
    @test fit.converged
    @test isfinite(loglik(fit))
    @test haskey(Dict(fit.blocks), :recov)
    @test any(==(:resd), first.(fit.blocks))
end