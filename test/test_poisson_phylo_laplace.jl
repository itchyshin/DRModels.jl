using DRModels
using Test, Random, LinearAlgebra
import Distributions

@testset "Poisson phylo random intercept — sparse Laplace route" begin
    Random.seed!(20260603)
    p = 32
    m = 8
    phy = random_balanced_tree(p; branch_length = 0.20)
    species = repeat(1:p, inner = m)
    n = length(species)
    x = randn(n)
    β = [0.15, 0.35]
    σphy = 0.45
    C = sigma_phy_dense(phy; σ²_phy = σphy^2)
    u = cholesky(Symmetric(C)).L * randn(p)
    λ = exp.(β[1] .+ β[2] .* x .+ u[species])
    y = Float64.([rand(Distributions.Poisson(λi)) for λi in λ])

    fit = drm(bf(@formula(y ~ x + phylo(1 | species))), Poisson();
              data = (; y, x, species), tree = phy, se = false)

    @test fit.converged
    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.20
    @test re_sd(fit)[:species] > 0.05
    @test isfinite(loglik(fit))
    @test all(fitted(fit) .> 0)
end

@testset "Poisson phylo sparse Laplace gradient" begin
    Random.seed!(20260604)
    p = 8
    phy = random_balanced_tree(p; branch_length = 0.25)
    species = repeat(1:p, inner = 3)
    x = randn(length(species))
    y = Float64.([rand(Distributions.Poisson(exp(0.2 + 0.25xi))) for xi in x])

    fit = drm(bf(@formula(y ~ x + phylo(1 | species))), Poisson();
              data = (; y, x, species), tree = phy, se = false)
    θ = coef(fit)
    g = zeros(length(θ))
    fit.nllgrad(g, θ)

    h = 1e-4
    fd = similar(g)
    for k in eachindex(θ)
        e = zeros(length(θ))
        e[k] = h
        fd[k] = (fit.nll(θ .+ e) - fit.nll(θ .- e)) / (2h)
    end
    @test g ≈ fd rtol = 2e-3 atol = 2e-3
end
