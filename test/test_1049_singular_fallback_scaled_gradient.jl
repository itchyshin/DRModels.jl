# DRModels.jl#1049: the singular-Hessian fallback of `_unitfree_converged` must
# judge the gradient in the correlation-scaled units (g .* D, D = diag(H)^(-1/2)),
# not the raw gradient. A raw `max |g| <= 1e-3` moves with parameter units.
using DRModels, Test, LinearAlgebra

@testset "#1049 singular-Hessian fallback is unit-free" begin
    # Collinear in correlation form (rank 1), so the singular branch is taken.
    Hs = [1.0 1.0; 1.0 1.0]
    @test DRModels._unitfree_converged(Hs, [1e-6, 1e-6])
    @test !DRModels._unitfree_converged(Hs, [1e-2, 1e-2])
    # Reparameterise theta = D phi: H -> D H D, g -> D g. The verdict must not move.
    for s in (1e-5, 1e-2, 1.0, 1e2, 1e5)
        D = Diagonal([s, s])
        @test DRModels._unitfree_converged(D * Hs * D, D * [1e-6, 1e-6])
        @test !DRModels._unitfree_converged(D * Hs * D, D * [1e-2, 1e-2])
    end
    # Mixed scales: unit-free verdict depends on the scaled gradient only.
    D = Diagonal([1e5, 1e-3])
    @test DRModels._unitfree_converged(D * Hs * D, D * [1e-6, 1e-6])
    @test !DRModels._unitfree_converged(D * Hs * D, D * [1e-2, 1e-2])
    # The negative-eigenvalue saddle rejection is unchanged.
    @test !DRModels._unitfree_converged([1.0 0.0; 0.0 -1.0], [0.0, 0.0])
end
