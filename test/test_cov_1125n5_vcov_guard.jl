# test_cov_1125n5_vcov_guard.jl — Hessian / Gram vcov guards (src/vcov_guard.jl).
using DRModels
using Test, LinearAlgebra, Logging

@testset "cov 1125n5: _vcov_from_hessian well-conditioned" begin
    H = [4.0 0.5; 0.5 9.0]
    V = @test_logs match_mode = :any DRModels._vcov_from_hessian(H)
    @test issymmetric(V)
    @test isposdef(Symmetric(V))
    @test V ≈ inv(H) rtol = 1e-10
    V0 = DRModels._vcov_from_hessian(zeros(0, 0))
    @test size(V0) == (0, 0)
end

@testset "cov 1125n5: _vcov_from_hessian singular and saddle" begin
    Hsing = Diagonal([1.0, 1e-20])
    V = @test_logs (:warn,) match_mode = :any DRModels._vcov_from_hessian(Hsing)
    @test all(isfinite, V)
    Hsad = [1.0 0.0; 0.0 -2.0]
    Vbad = @test_logs (:warn,) match_mode = :any DRModels._vcov_from_hessian(Hsad)
    @test all(isnan, Vbad)
    msg = try
        DRModels._vcov_from_hessian([1.0 NaN; 0.2 3.0]; context = "probe")
        ""
    catch e
        e isa ArgumentError ? e.msg : rethrow()
    end
    @test occursin("Hessian", msg)
    @test occursin("NaN", msg) || occursin("non-finite", msg)
end

@testset "cov 1125n5: _vcov_from_hessian_checked" begin
    H = [2.0 0.1; 0.1 3.0]
    V, ok = DRModels._vcov_from_hessian_checked(H, true)
    @test ok
    @test isposdef(Symmetric(V))
    V2, ok2 = @test_logs (:warn,) match_mode = :any begin
        DRModels._vcov_from_hessian_checked([1.0 Inf; 0.0 2.0], true)
    end
    @test !ok2
    @test all(isnan, V2)
end

@testset "cov 1125n5: _vcov_from_gram" begin
    G = [3.0 0.2; 0.2 5.0]
    V = DRModels._vcov_from_gram(G)
    @test V ≈ inv(G) rtol = 1e-10
    Gbad = [1.0 1.0; 1.0 1.0]
    Vw = @test_logs (:warn,) match_mode = :any DRModels._vcov_from_gram(Gbad)
    @test size(Vw) == (2, 2)
end

@testset "cov 1125n5: _fd_hessian_from_values" begin
    θ = [0.3, -0.1]
    f(θ) = θ[1]^2 + 2 * θ[2]^2 + θ[1] * θ[2]
    H, ok = DRModels._fd_hessian_from_values(f, θ; hstep = 1e-5)
    @test ok
    Htrue = [2.0 1.0; 1.0 4.0]
    @test H ≈ Htrue rtol = 1e-4
    fbad(θ) = 1e20
    H2, ok2 = DRModels._fd_hessian_from_values(fbad, θ; sentinel = 1e16)
    @test !ok2
    @test all(iszero, H2)
end

@testset "cov 1125n5: _vcov_from_fd_hessian" begin
    H = [2.0 0.0; 0.0 3.0]
    V = DRModels._vcov_from_fd_hessian(H)
    @test V ≈ inv(H) rtol = 1e-10
    Vnan = DRModels._vcov_from_fd_hessian(fill(NaN, 2, 2))
    @test all(isnan, Vnan)
end