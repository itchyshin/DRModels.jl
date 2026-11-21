# Gradient / Hessian on a BetaBinomial fit (src/fit_derivatives.jl, uncovered family).
using DRModels
using Test, Random, LinearAlgebra
import Distributions

@testset "fit_derivatives: BetaBinomial fixed effects" begin
    Random.seed!(202611202)
    n = 120
    x = randn(n)
    p = 1 / (1 .+ exp.(-0.2 .- 0.35 .* x))
    ntrials = fill(15, n)
    y = [rand(Distributions.BetaBinomial(ntrials[i], 2.0, 2.0)) / ntrials[i] for i in 1:n]
    s = round.(Int, y .* ntrials)
    f = round.(Int, ntrials .- s)
    fit = drm(bf(@formula(cbind(s, f) ~ x)), BetaBinomial(); data = (; s, f, x))
    @test fit.converged

    θ = Vector{Float64}(coef(fit))
    g = gradient(fit)
    @test length(g) == length(θ)
    @test all(isfinite, g)
    @test g ≈ DRModels._fd_gradient(fit.nll, θ) atol = 1e-4

    H = hessian(fit)
    @test H ≈ H' rtol = 0
    @test isposdef(Symmetric(H))
    @test inv(H) ≈ vcov(fit) rtol = 1e-5
end