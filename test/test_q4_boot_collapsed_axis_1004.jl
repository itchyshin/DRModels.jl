# #1004: a collapsed (SD exactly 0) among-axis in the q=4 bootstrap leaves its
# correlations undefined (NaN), not a fake 0.0 draw, and a nonzero prior ridge
# is returned to the caller instead of being applied silently.
using DRModels, Test, LinearAlgebra, SparseArrays
import Statistics

@testset "bootstrap_sigma_a — collapsed axis gives NaN, ridge reported (#1004)" begin
    pairs = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    Σ = Matrix(Diagonal([1.0, 1.0, 0.0, 1.0]))
    Σ[1, 2] = Σ[2, 1] = 0.5
    r = DRModels._q4_cor_offdiag(Σ, pairs)
    @test r[1] ≈ 0.5
    @test isnan(r[2]) && isnan(r[4]) && isnan(r[6])   # pairs touching axis 3
    @test !isnan(r[3]) && !isnan(r[5])

    # NaN draws are left out of the percentile interval, not counted as zeros.
    col = vcat(fill(0.8, 18), fill(NaN, 2))
    row = DRModels._q4_boot_row(:c, 0.8, col, 0.025)
    @test row.lower ≈ 0.8 && row.upper ≈ 0.8
    @test row.std_error == 0.0
    @test all(isnan, (DRModels._q4_boot_row(:c, 0.8, fill(NaN, 5), 0.025).lower,))

    # Ridge is returned: 0.0 for a regular Σ_a, > 0 for a singular one.
    Q = sparse(1.0I, 3, 3)
    _, ridge0 = DRModels._q4_re_prior_chol(Q, Matrix(1.0I, 4, 4))
    @test ridge0 == 0.0
    Σs = Matrix(Diagonal([1.0, 1.0, 0.0, 1.0]))   # exactly singular
    _, ridge1 = DRModels._q4_re_prior_chol(Q, Σs)
    @test ridge1 > 0.0
end
