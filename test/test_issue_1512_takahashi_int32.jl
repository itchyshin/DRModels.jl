# refs #1512 — `_csc_rowidx` must accept Int32 CSC index arrays (CHOLMOD on
# ILP32 / macOS arm64), not only `Vector{Int}`.
using DRModels
using SparseArrays, LinearAlgebra
using Test

@testset "takahashi CSC index type (#1512)" begin
    colptr = Int32[1, 2, 4]
    rowval = Int32[1, 1, 2]
    @test DRModels._csc_rowidx(colptr, rowval, Int32(1), Int32(1)) == 1
    @test DRModels._csc_rowidx(colptr, rowval, Int32(2), Int32(1)) == 2
    @test DRModels._csc_rowidx(colptr, rowval, Int32(2), Int32(2)) == 3
    @test DRModels._csc_rowidx(colptr, rowval, Int32(1), Int32(2)) == -1

    n = 12
    A = sprand(n, n, 0.15) + 4.0 * I
    A = A + A'
    F = cholesky(Symmetric(A); check=false)
    d = DRModels.takahashi_diag(F)
    sel = DRModels.takahashi_selinv(F)
    L = sparse(F.L)
    cp32 = Int32.(L.colptr)
    rv32 = Int32.(L.rowval)
    for j in 1:(length(cp32) - 1)
        cs = cp32[j]
        ce = cp32[j + 1] - 1
        for p in cs:ce
            i = rv32[p]
            @test DRModels._csc_rowidx(cp32, rv32, j, i) == p
        end
    end
    @test length(d) == n
    @test size(sel) == (n, n)
    @test maximum(abs.(diag(sel) .- d)) < 1e-10
end