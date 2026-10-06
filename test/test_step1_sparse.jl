# test_step1_sparse.jl — Step 1 foundation: confirm the ported sparse Q
# reproduces the R-side ape::vcv Σ_phy, and that sparse Cholesky +
# Takahashi selected-inversion match dense linear algebra.

using DRModels
using Test, LinearAlgebra, SparseArrays
using DelimitedFiles: readdlm

const D = DRModels
const FIX = joinpath(@__DIR__, "..", "bench", "fixtures")

# ---------------------------------------------------------------------------
# Helper: align Julia's Newick-order leaves to the R Σ_phy / data order.
# R wrote Σ_phy = vcv(tree)[tip.label, tip.label] and species = tip.label,
# so data-row i ↔ Σ_phy row i ↔ species name df.species[i].
# ---------------------------------------------------------------------------
function leaf_perm(phy, species_order::Vector{String})
    # position of phy leaf k within the R species ordering
    name_to_rpos = Dict(s => i for (i, s) in enumerate(species_order))
    # perm[k] = R-row for Julia-leaf k
    return [name_to_rpos[phy.leaf_names[k]] for k in 1:phy.n_leaves]
end

@testset "Step 1: sparse phylogenetic infrastructure" begin

    # --- Load the q4_p100 tree + R-side artifacts ----------------------------
    newick = read(joinpath(FIX, "q4_p100_tree.nwk"), String)
    phy = augmented_phy(newick)
    @test phy.n_leaves == 100
    @test phy.n_total == 2 * 100 - 1
    @test nnz(phy.Q_topology) < 20 * phy.n_leaves   # ~8p sparse

    raw, header = readdlm(joinpath(FIX, "q4_p100.csv"), ','; header = true)
    cols = Symbol.(strip.(string.(vec(header))))
    species_col = findfirst(==(:species), cols)
    species_order = String.(strip.(string.(raw[:, species_col])))   # = tree$tip.label order
    @test length(species_order) == 100
    @test Set(species_order) == Set(phy.leaf_names)  # same leaves, maybe reordered

    Σ_R = readdlm(joinpath(FIX, "q4_p100_sigma_phy.csv"), ',', Float64; skipstart = 1)
    @test size(Σ_R) == (100, 100)

    # --- (a) sparse Q reproduces the dense ape::vcv Σ_phy --------------------
    Σ_jl_newick = sigma_phy_dense(phy; σ²_phy = 1.0)   # leaves in Newick order
    perm = leaf_perm(phy, species_order)
    # reorder Julia's matrix into R's ordering: Σ_R[perm[k], perm[l]] == Σ_jl[k,l]
    Σ_jl_Rorder = similar(Σ_jl_newick)
    for k in 1:100, l in 1:100
        Σ_jl_Rorder[perm[k], perm[l]] = Σ_jl_newick[k, l]
    end
    relerr = maximum(abs.(Σ_jl_Rorder .- Σ_R)) / maximum(abs.(Σ_R))
    @test relerr < 1e-8

    # --- (b) sparse Cholesky logdet matches dense ---------------------------
    # Build a PD sparse matrix from the augmented Q: M = Q_cond + I (root removed)
    keep = setdiff(1:phy.n_total, [phy.root_index])
    Qc = phy.Q_topology[keep, keep]
    M = Qc + 1.0I
    Ms = sparse(Symmetric(M))
    ch = cholesky(Ms)
    ld_sparse = logdet(ch)
    ld_dense = logdet(Symmetric(Matrix(M)))
    @test isapprox(ld_sparse, ld_dense; rtol = 1e-10)

    # --- (c) Takahashi selected inverse matches dense inv at the pattern ----
    Minv_dense = inv(Symmetric(Matrix(M)))
    Sel = takahashi_selinv(ch)            # sparse, entries at L+L' pattern
    # check the entries Takahashi computed against dense inv
    rows = rowvals(Sel); vals = nonzeros(Sel)
    maxerr = 0.0; ncheck = 0
    for j in 1:size(Sel, 2)
        for idx in nzrange(Sel, j)
            i = rows[idx]
            maxerr = max(maxerr, abs(vals[idx] - Minv_dense[i, j]))
            ncheck += 1
        end
    end
    @test ncheck > 0
    @test maxerr < 1e-8

    # --- (d) diagonal of inverse via takahashi_diag -------------------------
    d_sel = D.takahashi_diag(ch)
    d_dense = diag(Minv_dense)
    @test maximum(abs.(d_sel .- d_dense)) < 1e-8
end
