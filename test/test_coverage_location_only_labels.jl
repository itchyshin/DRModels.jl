# DJ1: the last-resort numeric label path of `_phylo_mean_leaf_index`.
# `_loconly_marginal_nll` and `_fit_structured_gaussian_sparse_lbfgs` are
# rewritten by open drafts and are not called here.
using DRModels, Test

@testset "Gaussian phylo-mean labels: names, tip indices, and a refused code" begin
    phy = random_balanced_tree(4; branch_length = 0.2)
    names = string.(phy.leaf_names)
    @test DRModels._phylo_mean_leaf_index(phy, names) == collect(1:4)
    @test DRModels._phylo_mean_leaf_index(phy, [1, 2, 3, 4]) == [1, 2, 3, 4]
    # Float codes in 1:n_leaves are tip indices (the R path).
    @test DRModels._phylo_mean_leaf_index(phy, [1.0, 2.0, 3.0, 4.0]) == [1, 2, 3, 4]
    # Integer codes outside 1:n_leaves, and not one level per tip, are the
    # documented ArgumentError. Two distinct codes on a 4-leaf tree.
    err = try
        DRModels._phylo_mean_leaf_index(phy, [9, 9, 10, 10])
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("4", err.msg)
    @test occursin("2", err.msg)
    @test occursin("leaf", err.msg)
end
