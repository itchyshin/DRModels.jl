# test_betabinomial_phylo_laplace.jl — public-API recovery + FD-gradient checks
# for BetaBinomial()'s phylo(1 | grp) sparse-Laplace route (#166), constant-σ
# (overdispersion) only. Mirrors test_binomial_phylo_laplace.jl /
# test_gamma_beta_phylo_laplace.jl's shape; the exact FD-vs-analytic ≤ 1e-6
# gate on the low-level kernel lives in test_nongaussian_phylo_grad_gate.jl.

using DRModels
using Test, Random, LinearAlgebra
import Distributions

_bbphy_logistic(x) = 1 / (1 + exp(-x))

@testset "BetaBinomial phylo random intercept - sparse Laplace route" begin
    Random.seed!(20260802)
    p = 32
    m = 10
    phy = random_balanced_tree(p; branch_length = 0.20)
    species = repeat(1:p, inner = m)
    n = length(species)
    x = randn(n)
    β = [-0.10, 0.45]
    precision = 16.0
    σphy = 0.35
    C = sigma_phy_dense(phy; σ²_phy = σphy^2)
    u = cholesky(Symmetric(C)).L * randn(p)
    μ = _bbphy_logistic.(β[1] .+ β[2] .* x .+ u[species])
    trials = fill(10, n)
    successes = Float64.([rand(Distributions.BetaBinomial(trials[i], μ[i] * precision, (1 - μ[i]) * precision)) for i in 1:n])
    failures = Float64.(trials) .- successes

    fit = drm(bf(@formula(cbind(successes, failures) ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
              BetaBinomial(); data = (; successes, failures, x, species), tree = phy, se = false)

    @test fit.converged
    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.30
    @test 0.03 < exp(coef(fit, :sigma)[1]) < 1.0
    @test re_sd(fit)[:species] > 0.03
    @test isfinite(loglik(fit))
    @test all(0 .< fitted(fit) .< 1)

    # Phylo structured effects cannot yet combine with ordinary random effects.
    @test_throws ErrorException drm(
        bf(@formula(cbind(successes, failures) ~ x + phylo(1 | species) + (1 | block)), @formula(sigma ~ 1)),
        BetaBinomial(); data = (; successes, failures, x, species, block = repeat(1:4, inner = n ÷ 4)),
        tree = phy, se = false
    )

    # Nonconstant-sigma is out of scope for #166 — must error, not silently
    # dispatch to a different route.
    @test_throws ErrorException drm(
        bf(@formula(cbind(successes, failures) ~ x + phylo(1 | species)), @formula(sigma ~ x)),
        BetaBinomial(); data = (; successes, failures, x, species), tree = phy, se = false
    )
end

@testset "BetaBinomial phylo sparse Laplace gradient" begin
    Random.seed!(20260803)
    p = 8
    phy = random_balanced_tree(p; branch_length = 0.25)
    species = repeat(1:p, inner = 5)
    x = randn(length(species))
    precision = 12.0
    C = sigma_phy_dense(phy; σ²_phy = 0.25^2)
    u = cholesky(Symmetric(C)).L * randn(p)
    μ = _bbphy_logistic.(0.05 .+ 0.35 .* x .+ u[species])
    trials = fill(8, length(species))
    successes = Float64.([rand(Distributions.BetaBinomial(trials[i], μ[i] * precision, (1 - μ[i]) * precision)) for i in eachindex(trials)])
    failures = Float64.(trials) .- successes

    fit = drm(bf(@formula(cbind(successes, failures) ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
              BetaBinomial(); data = (; successes, failures, x, species), tree = phy, se = false)
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
