using DRModels
using Test
using LinearAlgebra
using SparseArrays

@testset "rooted multifurcating phylogenies" begin
    @testset "Newick star retains each supplied branch length" begin
        phy = DRModels.augmented_phy("(A:1.0,B:2.0,C:3.0);")
        @test phy.n_leaves == 3
        @test phy.n_total == 4
        @test phy.root_index == 4
        @test phy.leaf_indices == [1, 2, 3]
        @test phy.leaf_names == ["A", "B", "C"]
        @test phy.branch_lengths == [1.0, 2.0, 3.0]
        @test issparse(phy.Q_topology)
        @test nnz(phy.Q_topology) == 3 * phy.n_total - 2
        @test isapprox(Matrix(phy.Q_topology), Matrix(phy.Q_topology)'; atol = 0, rtol = 0)
        @test DRModels.phylo_tree_height(phy) == 3.0
        @test DRModels.sigma_phy_dense(phy) ≈ Diagonal([1.0, 2.0, 3.0]) atol = 1e-12 rtol = 0
        Q, _, _ = DRModels.augmented_tree_precision(phy)
        @test logdet(Symmetric(Matrix(Q))) ≈ -sum(log, phy.branch_lengths) atol = 1e-12 rtol = 0
    end

    @testset "mixed multifurcation gives shared-path covariance" begin
        # A/B/C share the root-to-internal branch of length four. D and E
        # share only the root, which is conditioned to zero.
        phy = DRModels.augmented_phy("((A:1.0,B:2.0,C:3.0):4.0,D:5.0,E:6.0);")
        expected = [5.0 4.0 4.0 0.0 0.0;
                    4.0 6.0 4.0 0.0 0.0;
                    4.0 4.0 7.0 0.0 0.0;
                    0.0 0.0 0.0 5.0 0.0;
                    0.0 0.0 0.0 0.0 6.0]
        @test phy.n_leaves == 5
        @test phy.n_total == 7
        @test phy.leaf_names == ["A", "B", "C", "D", "E"]
        @test nnz(phy.Q_topology) == 3 * phy.n_total - 2
        @test DRModels.phylo_tree_height(phy) == 7.0
        @test DRModels.sigma_phy_dense(phy) ≈ expected atol = 1e-12 rtol = 0
        Q, leaf_pos, q = DRModels.augmented_tree_precision(phy)
        @test q == phy.n_total - 1
        @test size(Q) == (q, q)
        @test length(leaf_pos) == 5
        @test logdet(Symmetric(Matrix(Q))) ≈ -sum(log, phy.branch_lengths) atol = 1e-12 rtol = 0
    end

    @testset "binary behavior is unchanged" begin
        phy = DRModels.augmented_phy("((A:1.0,B:2.0):3.0,C:4.0);")
        @test phy.n_total == 5
        @test phy.leaf_names == ["A", "B", "C"]
        @test DRModels.sigma_phy_dense(phy) ≈
              [4.0 3.0 0.0; 3.0 5.0 0.0; 0.0 0.0 4.0] atol = 1e-12 rtol = 0
    end

    @testset "edge order and nonterminal root id do not alter rooted covariance" begin
        # Leaves are 1:5; root six has an internal child seven. The input is
        # deliberately reverse topological order and root is not the max id.
        valid = [(6, 7, 4.0), (7, 1, 1.0), (7, 2, 2.0),
                 (7, 3, 3.0), (6, 4, 5.0), (6, 5, 6.0)]
        phy = DRModels.make_phy(reverse(valid), 5; root_index = 6,
                           leaf_names = ["A", "B", "C", "D", "E"])
        @test phy.root_index == 6
        @test phy.leaf_indices == collect(1:5)
        @test phy.leaf_names == ["A", "B", "C", "D", "E"]
        @test DRModels.sigma_phy_dense(phy) ≈
              [5.0 4.0 4.0 0.0 0.0;
               4.0 6.0 4.0 0.0 0.0;
               4.0 4.0 7.0 0.0 0.0;
               0.0 0.0 0.0 5.0 0.0;
               0.0 0.0 0.0 0.0 6.0] atol = 1e-12 rtol = 0
        @test_throws ArgumentError DRModels.make_phy(valid, 5; root_index = 7)
    end

    @testset "topology, branch, and tip-label validation" begin
        # Node ids must be contiguous and bounded before any node-indexed
        # allocation: this sparse edge list must not allocate one million rows.
        @test_throws ArgumentError DRModels.make_phy([(1_000_000, 1, 1.0),
                                                  (1_000_000, 2, 1.0)], 2)
        @test_throws ArgumentError DRModels.make_phy([(3.0, 1, 1.0), (3, 2, 1.0)], 2)
        @test_throws ArgumentError DRModels.make_phy([(true, 1, 1.0), (3, 2, 1.0)], 2)
        @test_throws ArgumentError DRModels.make_phy([(-3, 1, 1.0), (3, 2, 1.0)], 2)
        @test_throws ArgumentError DRModels.make_phy([(5, 1, 1.0), (5, 2, 1.0),
                                                  (5, 3, 1.0)], 3)
        @test_throws ArgumentError DRModels.make_phy([(3, 1, 1.0), (4, 2, 1.0)], 2)
        @test_throws ArgumentError DRModels.make_phy([(3, 1, 1.0), (3, 2, 1.0),
                                                  (1, 2, 1.0)], 2)
        @test_throws ArgumentError DRModels.make_phy([(5, 4, 1.0), (5, 2, 1.0),
                                                  (5, 3, 1.0), (4, 1, 1.0)], 3)
        @test_throws ArgumentError DRModels.make_phy([(3, 1, 1.0), (3, 2, 1.0),
                                                  (3, 2, 1.0)], 2)
        @test_throws ArgumentError DRModels.make_phy([(3, 3, 1.0), (3, 1, 1.0),
                                                  (3, 2, 1.0)], 2)
        @test_throws ArgumentError DRModels.make_phy([(5, 4, 1.0), (4, 5, 1.0),
                                                  (5, 1, 1.0), (5, 2, 1.0),
                                                  (5, 3, 1.0)], 3)
        @test_throws ArgumentError DRModels.make_phy([(8, 1, 1.0), (8, 2, 1.0),
                                                  (6, 7, 1.0), (6, 3, 1.0),
                                                  (7, 6, 1.0), (7, 4, 1.0),
                                                  (7, 5, 1.0)], 5)
        for bad in (0.0, -1.0, Inf, NaN, nextfloat(0.0))
            @test_throws ArgumentError DRModels.make_phy([(3, 1, bad), (3, 2, 1.0)], 2)
        end
        @test_throws ArgumentError DRModels.make_phy([(3, 1, 1.0e-308),
                                                  (3, 2, 1.0e-308)], 2)
        @test_throws ArgumentError DRModels.augmented_phy("(A:1.0,A:2.0,C:3.0);")
        @test_throws ArgumentError DRModels.augmented_phy("(:1.0,B:2.0,C:3.0);")
        @test_throws ArgumentError DRModels.augmented_phy("((A:1.0):2.0,B:3.0,C:4.0);")
        @test_throws ArgumentError DRModels.make_phy([(3, 1, 1.0), (3, 2, 1.0)], 2;
                                                  leaf_names = ["A", "A"])
        @test_throws ArgumentError DRModels.make_phy([(3, 1, 1.0), (3, 2, 1.0)], 2;
                                                  leaf_names = ["A"])
    end

    @testset "many-tip star height" begin
        p = 4096
        phy = DRModels.make_phy([(p + 1, leaf, 0.25) for leaf in 1:p], p;
                           root_index = p + 1)
        @test phy.n_total == p + 1
        @test DRModels.phylo_tree_height(phy) == 0.25
    end
end

println("PHYLO_POLYTOMY_TEST_READY")
