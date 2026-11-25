# test_cov_1125n4_separation_kernel.jl — separation NNLS kernels (src/separation.jl).
using DRModels
using Test, LinearAlgebra, Random

@testset "cov 1125n4: _sep_lstsq" begin
    E = [1.0 0.0; 1.0 1.0; 1.0 2.0]
    z = DRModels._sep_lstsq(E, [1.0, 2.0, 3.0])
    @test length(z) == 2
    @test E * z ≈ [1.0, 2.0, 3.0] atol = 1e-10
    z0 = DRModels._sep_lstsq(zeros(0, 3), ones(3))
    @test z0 == zeros(3)
    # rank-deficient: extra column alias
    X = [1.0 1.0; 1.0 2.0; 1.0 3.0]
    z = DRModels._sep_lstsq(X, [1.0, 1.0, 1.0])
    @test norm(X[:, 1:1] * z[1:1] - ones(3)) < 1e-8 || norm(X * z - ones(3)) < 0.5
end

@testset "cov 1125n4: _sep_constraints scaling invariance" begin
    X = [1.0 2.0; 1.0 -2.0; 1.0 0.5]
    s, ntr = [1.0, 0.0, 2.0], ones(3)
    A1, k1, a1, ac1 = DRModels._sep_constraints(X, s, ntr)
    Xbig = 1e6 .* X
    A2, k2, _, _ = DRModels._sep_constraints(Xbig, s, ntr)
    @test k1 == k2
    d1 = DRModels._detect_separation(X, s, ntr)
    d2 = DRModels._detect_separation(Xbig, s, ntr)
    @test d1.separated == d2.separated
    @test d1.flagged == d2.flagged
end

@testset "cov 1125n4: _sep_nnls residual cone" begin
  A = [1.0 0.0; -1.0 0.0; 0.0 1.0; 0.0 -1.0]
    c = [0.0, 0.0]
    nn = DRModels._sep_nnls(A, c, 500)
    @test nn.converged
    @test norm(nn.r) ≤ DRModels._SEP_RESID_TOL * sqrt(2) + 1e-10
    @test all(A * nn.r .≥ -1e-8)
end

@testset "cov 1125n4: detectseparation edge designs" begin
    @test !DRModels._detect_separation(zeros(0, 2), Float64[], Float64[]).separated
    X = hcat(ones(4), [1, 2, 3, 4])
    s = [0.0, 0.0, 4.0, 4.0]
    d = DRModels._detect_separation(X, s, ones(4))
    @test d.separated
    @test 2 in d.flagged
    # pathological duplicated column: rank drops to intercept-only (no slope screen)
    @test !DRModels._detect_separation(hcat(ones(6), fill(1.0, 6)), [0, 0, 0, 3, 3, 3], ones(6)).separated
    # NNLS iteration budget can exhaust before convergence (surfaced as inconclusive upstream)
    A = randn(40, 6)
    A = A ./ [norm(A[i, :]) for i in 1:size(A, 1)]
    nn = DRModels._sep_nnls(A, randn(6), 3)
    @test nn.converged == false
    @test nn.iter > 0
end