# test_cov_1124d_whitened_coords.jl — lane 1124d: whitened coordinate maps for the
# q=4 augmented engine (public engine helpers; no REML routes).
using DRModels
using Test, LinearAlgebra, SparseArrays, Random

@testset "cov 1124d: whitened_to_u / u_to_whitened round-trip" begin
    Random.seed!(1124)
    p = 8
    phy = random_balanced_tree(p; branch_length = 0.2)
    keep = setdiff(1:phy.n_total, [phy.root_index])
    Q_cond = phy.Q_topology[keep, keep]
    Λ = [0.28 0.08 0.04 0.0
         0.08 0.26 0.0 0.03
         0.04 0.0 0.14 0.02
         0.0 0.03 0.02 0.13]
    Λ = (Λ + Λ') / 2
    @test isposdef(Symmetric(Λ))
    W = DRModels.whitened_prior(Q_cond, cholesky(Symmetric(Λ)).L)
    nu = 4 * length(keep)
    v0 = randn(nu)
    u = DRModels.whitened_to_u(W, v0)
    v = DRModels.u_to_whitened(W, u)
    @test v ≈ v0 rtol = 1e-12
    @test length(u) == nu
    @test issparse(W.Pw)
    @test size(W.L) == (4, 4)
    @test W.L * W.L' ≈ Λ rtol = 1e-12
end