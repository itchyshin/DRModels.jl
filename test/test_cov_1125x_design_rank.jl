# test_cov_1125x_design_rank.jl — design-rank helper edge cases (src/design_rank.jl).
using DRModels
using Test
using LinearAlgebra

@testset "design_rank: _design_rank" begin
    Xfull = hcat(ones(5), collect(1.0:5))
    @test DRModels._design_rank(Xfull, ["(Intercept)", "x"]) === nothing

    z = 2 .* collect(1.0:5) .+ 1
    Xbad = hcat(ones(5), collect(1.0:5), z)
    r = DRModels._design_rank(Xbad, ["(Intercept)", "x", "z"])
    @test r !== nothing
    @test r.rank == 2
    @test r.ncol == 3
    @test r.aliased == ["z"]

    @test DRModels._design_rank(zeros(0, 2), ["a", "b"]) === nothing
    @test DRModels._design_rank(hcat(ones(3), [NaN, 2.0, 3.0]), ["(Intercept)", "x"]) === nothing

    msg = DRModels._design_rank_message([(param = :sigma, rank = 2, ncol = 3, aliased = ["z"])])
    @test occursin("fixed_effect_rank_deficiency", msg)
    @test occursin("`sigma`", msg)
    @test occursin("rank 2 of 3", msg)
end

@testset "design_rank: _check_fixed_design_rank records alias" begin
    delete!(task_local_storage(), DRModels._DESIGN_RANK_KEY)
    X = hcat(ones(4), [1.0, 1.0, 1.0, 1.0])  # duplicate intercept column
    @test_logs (:warn,) match_mode = :any begin
        DRModels._check_fixed_design_rank((:mu, Matrix{Float64}(X), String["(Intercept)", "dup"]))
    end
    found = get(task_local_storage(), DRModels._DESIGN_RANK_KEY, nothing)
    @test found !== nothing
    @test only(found).param == :mu
    delete!(task_local_storage(), DRModels._DESIGN_RANK_KEY)

    @test_logs DRModels._check_fixed_design_rank((:sigma, Matrix{Float64}(hcat(ones(3), [1.0, 2.0, 3.0])),
                                                   String["(Intercept)", "t"]))
    @test get(task_local_storage(), DRModels._DESIGN_RANK_KEY, nothing) === nothing
end