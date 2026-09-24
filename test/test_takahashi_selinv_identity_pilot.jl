# Lane C pilot: H² SIMD Takahashi identity on ONE matrix shape (banded SPD).
# Does not wire SelectedInversion.jl. Does not claim a speedup.
#
# Cell: n=64 tridiagonal Laplacian + 8×8 dense corner (creates non-trivial
# cliques so scatter / aligned-tail paths can fire). Gates match HSquared.jl
# #361/#363 contracts on a small local cell.

using Test
using SparseArrays
using LinearAlgebra
using DRModels

"""One SPD cell: banded Laplacian with a dense corner."""
function _pilot_banded_spd(n::Int = 64, corner::Int = 8)
    d = fill(2.0, n)
    e = fill(-1.0, n - 1)
    A = spdiagm(-1 => e, 0 => d, 1 => e)
    # Dense corner so the Cholesky factor has wider cliques than pure tridiagonal.
    Ac = Matrix(A)
    Ac[1:corner, 1:corner] .+= 0.25
    for i in 1:corner
        Ac[i, i] += 1.0
    end
    return sparse(Symmetric(Ac + I))
end

@testset "Lane C: H² SIMD selinv identity (one banded cell)" begin
    A = _pilot_banded_spd()
    ch = cholesky(A)
    Ainv = inv(Symmetric(Matrix(A)))

    # (1) strict_order scatter is BITWISE equal to per_pair reference
    z_strict = DRModels._selinv_zvals(ch; strict_order = true)[1]
    z_pair = DRModels._selinv_zvals(ch; per_pair = true)[1]
    @test reinterpret(UInt64, z_strict) == reinterpret(UInt64, z_pair)

    # (2) default (SIMD-allowed) vs strict_order within H² measured rtol
    z_default = DRModels._selinv_zvals(ch)[1]
    denom = max.(abs.(z_strict), 1e-300)
    max_rel = maximum(abs.(z_default .- z_strict) ./ denom)
    @test max_rel <= 1.3e-15

    # (3) public API vs dense inv at the selected pattern
    Sel = DRModels.takahashi_selinv(ch)
    rows = rowvals(Sel)
    vals = nonzeros(Sel)
    maxerr = 0.0
    ncheck = 0
    for j in 1:size(Sel, 2)
        for idx in nzrange(Sel, j)
            i = rows[idx]
            maxerr = max(maxerr, abs(vals[idx] - Ainv[i, j]))
            ncheck += 1
        end
    end
    @test ncheck > 0
    @test maxerr < 1e-10

    # (4) diagonal helper
    @test maximum(abs.(DRModels.takahashi_diag(ch) .- diag(Ainv))) < 1e-10
end
