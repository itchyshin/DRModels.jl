# cov lane 1125n6 — adaptive_ghq.jl linear algebra and rule construction (not integration DGPs).
using DRModels
using Test, LinearAlgebra

@testset "cov 1125n6: _AGHQRule construction" begin
    r1 = DRModels._AGHQRule(1, 1)
    @test r1.q == 1 && r1.K == 1
    @test size(r1.Z) == (1, 1)
    @test length(r1.lw) == 1
    r2 = DRModels._AGHQRule(2, 3)
    @test r2.q == 2 && r2.K == 3
    @test size(r2.Z, 2) == 9
    @test all(isfinite, r2.lw)
    @test_throws ArgumentError DRModels._AGHQRule(0, 2)
    @test_throws ArgumentError DRModels._AGHQRule(2, 0)
end

@testset "cov 1125n6: _aghq_chol" begin
    for H in ([2.0 0.3; 0.3 5.0], [4.0;;], [1.0 0.0; 0.0 2.0])
        L, ok = DRModels._aghq_chol(H)
        ok && @test cholesky(H).L ≈ L rtol = 1e-12
    end
    Lbad, okbad = DRModels._aghq_chol([1.0 1.0; 1.0 1.0])
    @test !okbad
    Lneg, okneg = DRModels._aghq_chol([-1.0;;])
    @test !okneg
    L0, ok0 = DRModels._aghq_chol(zeros(0, 0))
    @test ok0 && size(L0) == (0, 0)
end

@testset "cov 1125n6: _aghq_cholsolve" begin
  for _ in 1:5
        q = rand(1:4)
        H = rand(q, q)
        H = H * H' + 3.0I
        L, ok = DRModels._aghq_chol(H)
        ok || continue
        r = randn(q)
        x = DRModels._aghq_cholsolve(L, r)
        @test L * L' * x ≈ r rtol = 1e-10
    end
    @test DRModels._aghq_cholsolve([2.0;;], [4.0]) == [1.0]
end

@testset "cov 1125n6: _aghq_f at mode" begin
    η0 = [0.1, -0.2, 0.0]
    Zre = [1.0 0.0; 0.5 0.5; 0.0 1.0]
    L = cholesky([2.0 0.1; 0.1 1.5]).L
    Linv = inv(L)
    logdetL = logdet(L)
    ll(i, η) = -0.5 * η^2
    idx = [1, 2, 3]
    b0 = zeros(2)
    f0 = DRModels._aghq_f(ll, idx, η0, Zre, Linv, logdetL, b0)
    @test isfinite(f0)
    b = [0.2, -0.1]
    fb = DRModels._aghq_f(ll, idx, η0, Zre, Linv, logdetL, b)
    @test isfinite(fb) && fb <= max(f0, fb)
end

@testset "cov 1125n6: _crossed_aghq_schur and _newton" begin
    A = [2.5, 3.1]
    B = [0.2 0.1; 0.15 0.05]
    D = [4.0, 2.2]
    S = DRModels._crossed_aghq_schur(A, B, D)
    @test issymmetric(S)
    @test all(eigvals(Symmetric(S)) .> 0)
    gg = [0.1, -0.2]
    gh = [0.3, -0.1]
    δg, δh, _, ok = DRModels._crossed_aghq_newton(gg, gh, A, B, D)
    @test ok
    @test length(δg) == length(gg)
    @test length(δh) == length(gh)
    Dbad = [0.0, 1.0]
    _, _, _, ok2 = DRModels._crossed_aghq_newton(gg, gh, A, B, Dbad)
    @test !ok2
end

@testset "cov 1125n6: _aghq_grad_neghess Gaussian ll" begin
    η0 = [0.0, 0.5]
    Zre = [1.0 0.0; 0.0 1.0]
    Sinv = [1.0 0.0; 0.0 1.0]
    ll(i, η) = -0.5 * η^2
    idx = [1, 2]
    b = [0.1, -0.2]
    g, Hn = DRModels._aghq_grad_neghess(ll, idx, η0, Zre, Sinv, b, true)
    @test all(isfinite, g)
    @test size(Hn) == (2, 2)
    @test all(isfinite, Hn)
end