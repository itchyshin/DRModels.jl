using DRModels
using Test, Random, LinearAlgebra
import Distributions

_gb_logistic(x) = 1 / (1 + exp(-x))

@testset "Gamma phylo random intercept - sparse Laplace route" begin
    Random.seed!(20260607)
    p = 32
    m = 8
    phy = random_balanced_tree(p; branch_length = 0.20)
    species = repeat(1:p, inner = m)
    n = length(species)
    x = randn(n)
    β = [0.20, 0.30]
    sigma = 0.45
    shape = 1 / sigma^2
    σphy = 0.40
    C = sigma_phy_dense(phy; σ²_phy = σphy^2)
    u = cholesky(Symmetric(C)).L * randn(p)
    μ = exp.(β[1] .+ β[2] .* x .+ u[species])
    y = Float64.([rand(Distributions.Gamma(shape, μi / shape)) for μi in μ])

    fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
              Gamma(); data = (; y, x, species), tree = phy, se = false)

    @test fit.converged
    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.25
    @test 0.05 < exp(coef(fit, :sigma)[1]) < 1.5
    @test re_sd(fit)[:species] > 0.05
    @test isfinite(loglik(fit))
    @test all(fitted(fit) .> 0)

    # Covariate dispersion now routes to the per-observation log-σ path (#164)
    # instead of erroring (the dedicated recovery/gradient gates on x-dependent
    # dispersion live in test_164_gamma_hetero.jl); here we only assert the path
    # is reachable and returns a 1-column σ block on this slope-only design.
    fit_het = drm(
        bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 0 + x)),
        Gamma(); data = (; y, x, species), tree = phy, se = false
    )
    @test length(coef(fit_het, :sigma)) == 1
    @test isfinite(loglik(fit_het))
end

@testset "Beta phylo random intercept - sparse Laplace route" begin
    Random.seed!(20260608)
    p = 32
    m = 10
    phy = random_balanced_tree(p; branch_length = 0.20)
    species = repeat(1:p, inner = m)
    n = length(species)
    x = randn(n)
    β = [-0.10, 0.45]
    precision = 18.0
    σphy = 0.35
    C = sigma_phy_dense(phy; σ²_phy = σphy^2)
    u = cholesky(Symmetric(C)).L * randn(p)
    μ = _gb_logistic.(β[1] .+ β[2] .* x .+ u[species])
    y = Float64.([rand(Distributions.Beta(μi * precision, (1 - μi) * precision)) for μi in μ])

    fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
              Beta(); data = (; y, x, species), tree = phy, se = false)

    @test fit.converged
    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.30
    @test 0.03 < exp(coef(fit, :sigma)[1]) < 1.0
    @test re_sd(fit)[:species] > 0.05
    @test isfinite(loglik(fit))
    @test all(0 .< fitted(fit) .< 1)

    # Covariate dispersion now routes to the per-observation log-σ path (#164)
    # instead of erroring (the dedicated recovery/gradient gates on x-dependent
    # dispersion live in test_164_gamma_hetero.jl); here we only assert the path
    # is reachable and returns a 1-column σ block on this slope-only design.
    fit_het = drm(
        bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 0 + x)),
        Beta(); data = (; y, x, species), tree = phy, se = false
    )
    @test length(coef(fit_het, :sigma)) == 1
    @test isfinite(loglik(fit_het))
end

@testset "Gamma phylo sparse Laplace gradient" begin
    Random.seed!(20260609)
    p = 8
    phy = random_balanced_tree(p; branch_length = 0.25)
    species = repeat(1:p, inner = 4)
    x = randn(length(species))
    sigma = 0.50
    shape = 1 / sigma^2
    C = sigma_phy_dense(phy; σ²_phy = 0.30^2)
    u = cholesky(Symmetric(C)).L * randn(p)
    μ = exp.(0.20 .+ 0.25 .* x .+ u[species])
    y = Float64.([rand(Distributions.Gamma(shape, μi / shape)) for μi in μ])

    fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
              Gamma(); data = (; y, x, species), tree = phy, se = false)
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

@testset "Beta phylo sparse Laplace gradient" begin
    Random.seed!(20260610)
    p = 8
    phy = random_balanced_tree(p; branch_length = 0.25)
    species = repeat(1:p, inner = 5)
    x = randn(length(species))
    precision = 14.0
    C = sigma_phy_dense(phy; σ²_phy = 0.25^2)
    u = cholesky(Symmetric(C)).L * randn(p)
    μ = _gb_logistic.(0.05 .+ 0.35 .* x .+ u[species])
    y = Float64.([rand(Distributions.Beta(μi * precision, (1 - μi) * precision)) for μi in μ])

    fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
              Beta(); data = (; y, x, species), tree = phy, se = false)
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
    @test g ≈ fd rtol = 7e-3 atol = 7e-3
end
