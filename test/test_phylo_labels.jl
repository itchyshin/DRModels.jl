using DRModels
using Test
using LinearAlgebra

function _phylo_label_error(newick)
    err = try
        DRModels.augmented_phy(newick)
        nothing
    catch caught
        caught
    end
    @test err isa ArgumentError
    return sprint(showerror, err)
end

@testset "quoted Newick tip labels" begin
    @testset "quoted labels preserve names, order, and covariance" begin
        labels = ["Homo sapiens", "A,B: C(D)[E];", "O'Brien", "Δ_日本"]
        phy = DRModels.augmented_phy(
            "(('Homo sapiens':1.0,'A,B: C(D)[E];':2.0)'inner label':3.0," *
            "'O''Brien':4.0,'Δ_日本':5.0);"
        )
        @test phy.leaf_names == labels
        @test phy.leaf_indices == [1, 2, 3, 4]
        @test DRModels.sigma_phy_dense(phy) ≈
              [4.0 3.0 0.0 0.0;
               3.0 5.0 0.0 0.0;
               0.0 0.0 4.0 0.0;
               0.0 0.0 0.0 5.0] atol = 1e-12 rtol = 0
    end

    @testset "quoted whitespace is literal while token whitespace is ignored" begin
        phy = DRModels.augmented_phy(
            " \n ( 'two words' : 1.0 , 'tab\tname' : 2.0 , " *
            "'line\nbreak' : 3.0 ) ; \n"
        )
        @test phy.leaf_names == ["two words", "tab\tname", "line\nbreak"]
        @test DRModels.sigma_phy_dense(phy) ≈ Diagonal([1.0, 2.0, 3.0]) atol = 1e-12 rtol = 0
    end

    @testset "unquoted token whitespace remains admissible" begin
        phy = DRModels.augmented_phy(" ( A : 1.0 , B : 2.0 , C : 3.0 ) ; ")
        @test phy.leaf_names == ["A", "B", "C"]
        @test DRModels.sigma_phy_dense(phy) ≈ Diagonal([1.0, 2.0, 3.0]) atol = 1e-12 rtol = 0
    end

    @testset "simple labels retain their existing spelling" begin
        phy = DRModels.augmented_phy("(simple:1.0,under_score:2.0,dot.name-3:3.0);")
        @test phy.leaf_names == ["simple", "under_score", "dot.name-3"]
    end

    @testset "quoted spaces do not collide with simple labels" begin
        phy = DRModels.augmented_phy(
            "('A B':1.0,AB:2.0,A_B:3.0,' leading':4.0,'trailing ':5.0);"
        )
        @test phy.leaf_names == ["A B", "AB", "A_B", " leading", "trailing "]
        @test length(unique(phy.leaf_names)) == 5
    end

    @testset "malformed quoted and unquoted labels fail clearly" begin
        @test occursin("unterminated quoted label", _phylo_label_error("('unterminated:1.0,B:2.0);"))
        @test occursin("expected delimiter", _phylo_label_error("('A'junk:1.0,B:2.0);"))
        @test occursin("unquoted label", _phylo_label_error("(two words:1.0,B:2.0);"))
        @test occursin("extra characters", _phylo_label_error("('A B':1.0,C:2.0); trailing"))
        @test occursin("unique", _phylo_label_error("('O''Brien':1.0,'O''Brien':2.0);"))
        @test occursin("nonempty", _phylo_label_error("('':1.0,B:2.0);"))
        @test occursin("single quote", _phylo_label_error("(A':1.0,B:2.0);"))
        @test occursin("NUL", _phylo_label_error("(A:1.0,B:2.0);\0junk"))
        @test occursin("NUL", _phylo_label_error("(A\0B:1.0,C:2.0);"))
        @test occursin("NUL", _phylo_label_error("('A\0B':1.0,C:2.0);"))
    end
end

println("PHYLO_LABELS_TEST_READY")
