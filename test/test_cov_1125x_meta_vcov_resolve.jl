# test_cov_1125x_meta_vcov_resolve.jl — meta V dense constructor / resolver (src/meta_vcov_bivariate.jl).
using DRModels
using Test

_msg(f) = try
    f()
    ""
catch e
    e isa ArgumentError || rethrow()
    e.msg
end

@testset "meta_vcov_bivariate: length and dense constructor errors" begin
    v1 = [0.1, 0.2, 0.3]
    v2 = [0.4, 0.5, 0.6]
    V = meta_vcov_bivariate(v1, v2; cor12 = 0.0)
    @test length(V) == 3
    @test length(V) == length(V.v1)

    msg = _msg(() -> MetaVcovBivariate(zeros(3, 3)))
    @test occursin("2n × 2n", msg)
    M3 = zeros(4, 4); M3[1, 1] = 1.0; M3[2, 2] = 2.0; M3[1, 2] = 0.1; M3[2, 1] = 0.2
    msg2 = _msg(() -> MetaVcovBivariate(M3))
    @test occursin("not symmetric", msg2)
end

@testset "meta_vcov: _resolve_biv_meta_v" begin
    v1 = [0.2, 0.3]
    v2 = [0.4, 0.5]
    V = meta_vcov_bivariate(v1, v2)
    @test DRModels._resolve_biv_meta_v(nothing, 2) === nothing
    @test DRModels._resolve_biv_meta_v(V, 2) === V
    msg = _msg(() -> DRModels._resolve_biv_meta_v(V, 5))
    @test occursin("5 rows", msg) || occursin("5 studies", msg)
    msg2 = _msg(() -> DRModels._resolve_biv_meta_v("not-a-V", 2))
    @test occursin("meta_vcov_bivariate", msg2)
    M = Matrix(V)
    @test DRModels._resolve_biv_meta_v(M, 2) isa MetaVcovBivariate
end