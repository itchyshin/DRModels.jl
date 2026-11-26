# cov lane 1125n9 — Newick / topology guards (src/sparse_phy.jl).
using DRModels
using Test, LinearAlgebra, SparseArrays

_msg(f) = try
    f()
    ""
catch e
    e isa ArgumentError ? e.msg : sprint(showerror, e)
end

@testset "cov 1125n9: _phy_branch_length" begin
    @test DRModels._phy_branch_length(0.25, "tip A") == 0.25
    @test_throws ArgumentError DRModels._phy_branch_length(0, "root")
    @test_throws ArgumentError DRModels._phy_branch_length(-1.0, "edge")
    @test_throws ArgumentError DRModels._phy_branch_length(NaN, "x")
    @test_throws ArgumentError DRModels._phy_branch_length(true, "bad")
    tiny = nextfloat(0.0) * 1e300  # still must fail finiteness / inv check when too small
    msg = _msg(() -> DRModels._phy_branch_length(1e-400, "tiny"))
    @test !isempty(msg)
end

@testset "cov 1125n9: _phy_require_finite_lengths" begin
    @test DRModels._phy_require_finite_lengths([0.1, 0.2]) === nothing
    @test_throws ArgumentError DRModels._phy_require_finite_lengths([0.1, Inf])
end

@testset "cov 1125n9: _phy_validate_leaf_names" begin
    @test DRModels._phy_validate_leaf_names(["a", "b", "c"], 3) == ["a", "b", "c"]
    @test_throws ArgumentError DRModels._phy_validate_leaf_names(["a", "a"], 2)
    @test_throws ArgumentError DRModels._phy_validate_leaf_names([""], 1)
    @test_throws ArgumentError DRModels._phy_validate_leaf_names(["a"], 2)
end

@testset "cov 1125n9: _phy_validate_topology" begin
    parent = [0, 1, 1]
    leaf = [false, true, true]
    root, kids = DRModels._phy_validate_topology(parent, leaf)
    @test root == 1
    @test kids[1] == [2, 3]
    bad_parent = [0, 1, 5]
    @test_throws ArgumentError DRModels._phy_validate_topology(bad_parent, leaf)
    cycle = [2, 3, 2]
    @test_throws ArgumentError DRModels._phy_validate_topology(cycle, [false, false, false])
    two_roots = [0, 0, 1]
    @test_throws ArgumentError DRModels._phy_validate_topology(two_roots, leaf)
    unary = [0, 1, 1]
    unary_leaf = [false, false, true]
    @test_throws ArgumentError DRModels._phy_validate_topology(unary, unary_leaf)
    self = [1, 1, 1]
    @test_throws ArgumentError DRModels._phy_validate_topology(self, [true, true, true])
end

@testset "cov 1125n9: augmented_tree_precision" begin
    phy = DRModels.augmented_phy("((a:0.5,b:0.5):0.5,(c:0.5,d:0.5):0.5);")
    Q, leaf_pos, q = DRModels.augmented_tree_precision(phy)
    @test size(Q, 1) == q
    @test issparse(Q)
    @test length(leaf_pos) == phy.n_leaves
    @test length(phy.branch_lengths) == phy.n_total - 1
end

@testset "cov 1125n9: augmented_phy labels and quotes" begin
    phy = DRModels.augmented_phy("('taxon A':0.3,'taxon B':0.3);")
    @test phy.n_leaves == 2
    @test "taxon A" in phy.leaf_names
    phy2 = DRModels.augmented_phy("((a:0.5,b:0.5):0.5,(c:0.5,d:0.5):0.5);")
    @test phy2.n_leaves == 4
    Qsub, _, qsub = DRModels.augmented_tree_precision(phy2)
    @test size(Qsub, 1) == qsub
    @test_throws ArgumentError DRModels.augmented_phy("(a:0,b:1);")
    msg = _msg(() -> DRModels.augmented_phy("(two words:0.5,b:0.5);"))
    @test occursin("whitespace", msg) || occursin("quote", msg)
end