# sigma(fit): fitted scale / dispersion accessor, mirroring drmTMB's sigma().
using DRModels
using Test, Random

@testset "sigma() — fitted scale accessor" begin
    @testset "univariate constant scale → σ vector" begin
        Random.seed!(20260601)
        n = 400; x = randn(n); σ = 0.6
        y = 1.0 .+ 0.5 .* x .+ σ .* randn(n)
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
        s = sigma(fit)
        @test s isa Vector{Float64}
        @test length(s) == n
        @test all(s .> 0)
        @test all(≈(s[1]), s)               # constant scale ⇒ all equal
        @test s[1] ≈ σ atol = 0.1           # recovers the truth
    end

    @testset "heteroscedastic σ ~ x → varying scale" begin
        Random.seed!(20260602)
        n = 600; x = randn(n)
        y = 0.5 .+ 0.7 .* x .+ exp.(0.3 .* x) .* randn(n)
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian(); data = (; y, x))
        s = sigma(fit)
        @test s isa Vector{Float64}
        @test length(s) == n
        @test maximum(s) > minimum(s)       # not constant — tracks x
        # larger x ⇒ larger fitted σ (positive log-σ slope recovered)
        @test s[argmax(x)] > s[argmin(x)]
    end

    @testset "bivariate co-scale → Dict(:sigma1,:sigma2)" begin
        Random.seed!(20260603)
        n = 300; x = randn(n)
        y = 0.5 .+ 0.4 .* x .+ 0.5 .* randn(n)
        y2 = -0.2 .+ 0.3 .* x .+ 0.7 .* randn(n)
        fit = drm(bf(mu1 = @formula(y ~ x), mu2 = @formula(y2 ~ x),
                     sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                     rho12 = @formula(rho12 ~ 1)), Gaussian(); data = (; y, y2, x))
        s = sigma(fit)
        @test s isa Dict
        @test haskey(s, :sigma1) && haskey(s, :sigma2)
        @test all(s[:sigma1] .> 0) && all(s[:sigma2] .> 0)
    end

    @testset "Poisson has no free dispersion → empty" begin
        Random.seed!(20260604)
        n = 300; x = randn(n)
        y = [rand() < 1 / (1 + exp(-(0.2 + 0.3 * xi))) ? 1 : 0 for xi in x]   # small counts
        fit = drm(bf(@formula(y ~ x)), Poisson(); data = (; y, x))
        @test sigma(fit) isa Dict
        @test isempty(sigma(fit))
    end
end
