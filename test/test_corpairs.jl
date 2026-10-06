# corpairs(fit): fitted between-response residual correlation, drmTMB's corpairs().
using DRModels
using Test, Random

@testset "corpairs() — between-response residual correlation" begin
    @testset "bivariate constant ρ12 → vector, recovers truth" begin
        Random.seed!(20260601)
        n = 800; x = randn(n)
        ρ = 0.5; s1 = 0.6; s2 = 0.8
        z1 = randn(n); z2 = randn(n)
        e1 = s1 .* z1
        e2 = s2 .* (ρ .* z1 .+ sqrt(1 - ρ^2) .* z2)
        y = 0.3 .+ 0.5 .* x .+ e1
        y2 = -0.1 .+ 0.2 .* x .+ e2
        fit = drm(bf(mu1 = @formula(y ~ x), mu2 = @formula(y2 ~ x),
                     sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                     rho12 = @formula(rho12 ~ 1)), Gaussian(); data = (; y, y2, x))
        c = corpairs(fit)
        @test c isa Vector{Float64}
        @test length(c) == n
        @test all(-1 .< c .< 1)
        @test all(≈(c[1]), c)               # rho12 ~ 1 ⇒ constant
        @test c[1] ≈ ρ atol = 0.1           # recovers the truth
    end

    @testset "bivariate ρ12 ~ x → varying correlation" begin
        Random.seed!(20260602)
        n = 600; x = randn(n)
        y = 0.3 .+ 0.5 .* x .+ 0.6 .* randn(n)
        y2 = -0.1 .+ 0.2 .* x .+ 0.7 .* randn(n)
        fit = drm(bf(mu1 = @formula(y ~ x), mu2 = @formula(y2 ~ x),
                     sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                     rho12 = @formula(rho12 ~ x)), Gaussian(); data = (; y, y2, x))
        c = corpairs(fit)
        @test c isa Vector{Float64}
        @test length(c) == n
        @test all(-1 .< c .< 1)
    end

    @testset "univariate → empty" begin
        Random.seed!(20260603)
        n = 200; x = randn(n); y = 1.0 .+ 0.5 .* x .+ 0.4 .* randn(n)
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
        @test corpairs(fit) isa Dict
        @test isempty(corpairs(fit))
    end
end
