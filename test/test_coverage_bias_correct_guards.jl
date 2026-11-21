# Guard rails and degenerate cases in src/bias_correct.jl (not in test_bias_correct.jl).
using DRModels
using Test, Random, LinearAlgebra
import Distributions

@testset "bias_correct guards and degenerate variance" begin
    @testset "DimensionMismatch when V and θ disagree in length" begin
        @test_throws DimensionMismatch bias_correct([1.0, 2.0], reshape([1.0], 1, 1), θ -> θ[1])
        @test_throws DimensionMismatch begin
            θ̂ = [1.0]
            V = Matrix(1.0I, 2, 2)
            bias_correct(θ̂, V, θ -> θ[1])
        end
    end

    @testset "non-PD direction ⇒ se = Inf (unbounded Wald CI)" begin
        θ̂ = [0.0, 0.0]
        V = [0.1 0.2; 0.2 0.1]   # indefinite along [1, -1]
        bc = bias_correct(θ̂, V, θ -> θ[1] - θ[2])
        @test bc.se == Inf
        @test bc.ci.lower == -Inf && bc.ci.upper == Inf
        @test isfinite(bc.estimate) && isfinite(bc.corrected)
    end

    @testset "MAP + fixed priors are refused on DrmFit" begin
        Random.seed!(20261120)
        n = 80
        x = randn(n)
        y = 0.2 .+ 0.5x .+ randn(n)
        pri = drm_priors(drm_prior_normal(dpar = "mu", coef = "(Intercept)", mean = 0, sd = 1))
        fit = drm(bf(@formula(y ~ x)), Gaussian(); data = (; y, x), priors = pri)
        @test fit.estim_method === :MAP
        err = sprint(showerror, try bias_correct(fit, θ -> θ[1]) catch e; e end)
        @test occursin("bias_correct", err)
        @test occursin("MAP", err)
    end
end