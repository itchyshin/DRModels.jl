using DRModels
using Test, Random, LinearAlgebra
import Distributions

_binom_logistic(x) = 1 / (1 + exp(-x))

@testset "Binomial phylo cbind response - sparse Laplace route" begin
    Random.seed!(20260611)
    p = 32
    m = 8
    phy = random_balanced_tree(p; branch_length = 0.20)
    species = repeat(1:p, inner = m)
    n = length(species)
    x = randn(n)
    β = [-0.20, 0.50]
    σphy = 0.35
    C = sigma_phy_dense(phy; σ²_phy = σphy^2)
    u = cholesky(Symmetric(C)).L * randn(p)
    prob = _binom_logistic.(β[1] .+ β[2] .* x .+ u[species])
    trials = fill(10, n)
    successes = Float64.([rand(Distributions.Binomial(trials[i], prob[i])) for i in 1:n])
    failures = Float64.(trials) .- successes

    fit = drm(bf(@formula(cbind(successes, failures) ~ x + phylo(1 | species))),
              Binomial(); data = (; successes, failures, x, species), tree = phy, se = false)

    @test fit.converged
    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.30
    @test re_sd(fit)[:species] > 0.03
    @test isfinite(loglik(fit))
    @test all(0 .< fitted(fit) .< 1)

    @test_throws ErrorException drm(
        bf(@formula(cbind(successes, failures) ~ x + phylo(1 | species) + (1 | block))),
        Binomial(); data = (; successes, failures, x, species, block = repeat(1:4, inner = n ÷ 4)),
        tree = phy, se = false
    )
end

@testset "Binomial phylo Bernoulli response - sparse Laplace route" begin
    Random.seed!(20260612)
    p = 24
    m = 18
    phy = random_balanced_tree(p; branch_length = 0.20)
    species = repeat(1:p, inner = m)
    x = randn(length(species))
    β = [0.10, -0.45]
    C = sigma_phy_dense(phy; σ²_phy = 0.30^2)
    u = cholesky(Symmetric(C)).L * randn(p)
    prob = _binom_logistic.(β[1] .+ β[2] .* x .+ u[species])
    y = Float64.([rand(Distributions.Binomial(1, pi)) for pi in prob])

    fit = drm(bf(@formula(y ~ x + phylo(1 | species))),
              Binomial(); data = (; y, x, species), tree = phy, se = false)

    @test fit.converged
    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.35
    @test re_sd(fit)[:species] > 0.02
    @test isfinite(loglik(fit))
    @test all(0 .< fitted(fit) .< 1)
end

@testset "Binomial phylo sparse Laplace gradient" begin
    Random.seed!(20260613)
    p = 8
    phy = random_balanced_tree(p; branch_length = 0.25)
    species = repeat(1:p, inner = 5)
    x = randn(length(species))
    C = sigma_phy_dense(phy; σ²_phy = 0.30^2)
    u = cholesky(Symmetric(C)).L * randn(p)
    prob = _binom_logistic.(-0.10 .+ 0.40 .* x .+ u[species])
    trials = fill(8, length(species))
    successes = Float64.([rand(Distributions.Binomial(trials[i], prob[i])) for i in eachindex(trials)])
    failures = Float64.(trials) .- successes

    fit = drm(bf(@formula(cbind(successes, failures) ~ x + phylo(1 | species))),
              Binomial(); data = (; successes, failures, x, species), tree = phy, se = false)
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
    @test g ≈ fd rtol = 5e-3 atol = 5e-3
end
