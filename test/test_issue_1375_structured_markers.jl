using DRModels
using StatsModels
using Test

@testset "#1375 structured markers recognise temporal" begin
    @testset "_split_bivariate_q4_rhs" begin
        fixed, structured = DRModels._split_bivariate_q4_rhs(
            @formula(y ~ x + phylo(1 | g)).rhs, :mu1)
        @test fixed == @formula(y ~ x).rhs
        @test structured == (:phylo, :g, nothing)

        fixed, structured = DRModels._split_bivariate_q4_rhs(
            @formula(y ~ x + temporal(1 | g, t, ar1)).rhs, :mu1)
        @test fixed == @formula(y ~ x).rhs
        @test structured == (:temporal, :g, nothing)
    end

    @testset "_parse_structured_slope" begin
        @test DRModels._parse_structured_slope(@formula(y ~ phylo(1 + x | g)).rhs) ==
              (:phylo, :x, :g)
        @test DRModels._parse_structured_slope(@formula(y ~ temporal(1 + x | g)).rhs) ==
              (:temporal, :x, :g)
    end

    @testset "membership lists" begin
        @test length(DRModels._STRUCTURED_MARKERS) == 4
        @test length(DRModels._STRUCTURED_OR_TEMPORAL_MARKERS) == 5
        @test DRModels.temporal in DRModels._STRUCTURED_OR_TEMPORAL_MARKERS
        @test !(DRModels.temporal in DRModels._STRUCTURED_MARKERS)
    end
end