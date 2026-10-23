# test_phylo_tree_height.jl — the tree-scale convention, made visible.
#
# WHY THIS EXISTS. DRModels.jl builds its phylogenetic covariance from the branch
# lengths AS SUPPLIED, so a tree of height h implies tip variance h and the
# reported `sd_phylo` carries a factor sqrt(h). R's drmTMB standardises via
# `ape::vcv(tree, corr = TRUE)`, whose tips always have variance 1. The two agree
# only when h == 1.
#
# That difference is not hypothetical. It cost two debugging rounds in one day:
# once in the phylo-penalty parity fixture (where identical log-likelihoods sat
# beside SDs differing by exactly sqrt(h)), and once in a Binomial recovery study
# where it masqueraded as a ~30% variance-component bias in the package. Both
# times the arithmetic was right and the SCALE was the bug. Hence a cheap,
# user-callable height and a warning that names the factor.

using DRModels
using Test
using LinearAlgebra

@testset "phylo tree height + scale warning" begin

    @testset "height matches the dense covariance's tip variance" begin
        # `sigma_phy_dense` inverts a dense matrix (O(p^3)); `phylo_tree_height`
        # is an O(p) walk over the sparse topology. They must agree, or the cheap
        # one is not a substitute for the expensive one.
        for (G, bl) in ((16, 0.25), (32, 0.5), (8, 1.0), (16, 0.125))
            phy = random_balanced_tree(G; branch_length = bl)
            h_fast = phylo_tree_height(phy)
            h_dense = maximum(diag(sigma_phy_dense(phy; σ²_phy = 1.0)))
            @test isapprox(h_fast, h_dense; rtol = 1e-8)
            @test h_fast > 0
        end
    end

    @testset "the balanced-tree height is depth x branch length" begin
        # G = 16 leaves is 4 levels, so branch_length 0.25 gives height exactly 1 —
        # the case that needs NO warning, and the setting a user should rescale to.
        @test isapprox(phylo_tree_height(random_balanced_tree(16; branch_length = 0.25)), 1.0)
        @test isapprox(phylo_tree_height(random_balanced_tree(16; branch_length = 0.5)), 2.0)
    end

    @testset "the warning fires only off unit height, and only once" begin
        unit = random_balanced_tree(16; branch_length = 0.25)   # h == 1
        off  = random_balanced_tree(16; branch_length = 0.5)    # h == 2

        # A unit-height tree is the agreeing case: silence.
        @test_logs DRModels._warn_if_tree_not_unit_height(unit)

        # Off-unit warns, and the message must name the FACTOR, not merely
        # complain -- a warning a user cannot act on is noise.
        empty!(DRModels._PHYLO_HEIGHT_WARNED)
        @test_logs (:warn, r"sqrt") DRModels._warn_if_tree_not_unit_height(off)

        # ...and not again for the same tree. The scale is a reporting
        # convention, not an error; warning on every fit in a loop would train
        # users to ignore it.
        @test_logs DRModels._warn_if_tree_not_unit_height(off)
    end

    @testset "a diagnostic never breaks a fit" begin
        # Whatever it is handed, the guard returns rather than throwing.
        @test DRModels._warn_if_tree_not_unit_height(random_balanced_tree(8; branch_length = 0.1)) === nothing
    end
end
