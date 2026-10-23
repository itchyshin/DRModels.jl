using DRModels
using Test, Random, LinearAlgebra
import Distributions

@testset "NB2 phylo random intercept — sparse Laplace route" begin
    Random.seed!(20260605)
    p = 32
    m = 8
    phy = random_balanced_tree(p; branch_length = 0.20)
    species = repeat(1:p, inner = m)
    n = length(species)
    x = randn(n)
    β = [0.25, 0.35]
    size = 4.0
    σphy = 0.45
    C = sigma_phy_dense(phy; σ²_phy = σphy^2)
    u = cholesky(Symmetric(C)).L * randn(p)
    μ = exp.(β[1] .+ β[2] .* x .+ u[species])
    y = Float64.([rand(Distributions.NegativeBinomial(size, size / (size + μi))) for μi in μ])

    fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
              NegBinomial2(); data = (; y, x, species), tree = phy, se = false)

    @test fit.converged
    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.25
    @test 0.5 < exp(-2 * coef(fit, :sigma)[1]) < 20.0
    @test re_sd(fit)[:species] > 0.05
    @test isfinite(loglik(fit))
    @test all(fitted(fit) .> 0)

    # Covariate dispersion on the phylo route is now supported (#164): a
    # non-constant `sigma` formula fits via the per-observation log-size path
    # instead of hard-erroring. (Full FD gate + recovery in
    # test_164_mean_re_covariate_sigma.jl.)
    fit_disp = drm(
        bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 0 + x)),
        NegBinomial2(); data = (; y, x, species), tree = phy, se = false
    )
    @test fit_disp.converged
    @test length(coef(fit_disp, :sigma)) == 1
    @test isfinite(loglik(fit_disp))
end

@testset "NB2 phylo sparse Laplace gradient" begin
    Random.seed!(20260606)
    p = 8
    phy = random_balanced_tree(p; branch_length = 0.25)
    species = repeat(1:p, inner = 4)
    x = randn(length(species))
    size = 5.0
    σphy = 0.30
    C = sigma_phy_dense(phy; σ²_phy = σphy^2)
    u = cholesky(Symmetric(C)).L * randn(p)
    μ = exp.(0.25 .+ 0.30 .* x .+ u[species])
    y = Float64.([rand(Distributions.NegativeBinomial(size, size / (size + μi))) for μi in μ])

    fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
              NegBinomial2(); data = (; y, x, species), tree = phy, se = false)
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
    @test g ≈ fd rtol = 4e-3 atol = 4e-3
end
