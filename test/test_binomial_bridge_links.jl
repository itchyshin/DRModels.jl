using Test
using DRModels

@testset "bridge tags preserve supported Binomial links" begin
    @test DRModels._bridge_family("binomial").link === :logit
    @test DRModels._bridge_family("binomial_probit").link === :probit
    @test DRModels._bridge_family("binomial_cloglog").link === :cloglog
    @test_throws ArgumentError DRModels._bridge_family("binomial_cauchit")
end
