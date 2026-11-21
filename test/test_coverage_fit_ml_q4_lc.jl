# Numerical helpers in src/fit_ml_q4.jl: log-Cholesky map and non-PD Λ scoring.
using DRModels
using Test, LinearAlgebra, Random

@testset "fit_ml_q4 log-Cholesky parameterisation" begin
    Random.seed!(41)
    Λ = let v = randn(10)
        DRModels.lc_to_Λ(v)
    end
    @test isposdef(Symmetric(Λ))
    v = DRModels.Λ_to_lc(Λ)
    @test length(v) == 10
    Λ2 = DRModels.lc_to_Λ(v)
    @test Λ2 ≈ Λ rtol = 1e-12

    @testset "L_given_Λ returns -Inf for non-PD Λ (line search guard)" begin
        prob, Q = let p = 6
            phy = random_balanced_tree(p; branch_length = 0.2)
            n = p
            y1 = randn(n); y2 = randn(n)
            X = hcat(ones(n), randn(n))
            make_problem(phy, y1, y2, X, X, X, X, X)
        end
        β = (mu1 = zeros(2), mu2 = zeros(2), s1 = [0.0], s2 = [0.0], rho = [0.0])
        badΛ = zeros(4, 4)   # not PD
        @test DRModels.L_given_Λ(prob, Q, β, badΛ) == -Inf
        @test DRModels._lg_chol_L(badΛ) === nothing
    end
end