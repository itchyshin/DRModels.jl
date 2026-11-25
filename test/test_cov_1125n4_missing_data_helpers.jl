# test_cov_1125n4_missing_data_helpers.jl — row-drop / NA helpers (src/missing_data.jl).
using DRModels
using Test, StatsModels, Tables

_msg(f) = try
    f()
    ""
catch e
    e isa ArgumentError ? e.msg : sprint(showerror, e)
end

@testset "cov 1125n4: missing_data cell detection" begin
    @test DRModels._is_missing_cell(missing)
    @test DRModels._is_missing_cell(NaN)
    @test DRModels._is_missing_cell(Inf)
    @test DRModels._is_missing_cell(-Inf)
    @test !DRModels._is_missing_cell(0.0)
    @test !DRModels._is_missing_cell(1)
    @test !DRModels._is_missing_cell("a")
    @test !DRModels._is_missing_cell(false)
end

@testset "cov 1125n4: _narrow_nomissing" begin
    v = Vector{Union{Missing,Float64}}([1.0, 2.0, 3.0])
    w = DRModels._narrow_nomissing(v)
    @test w isa Vector{Float64}
    @test w == [1.0, 2.0, 3.0]
    kept = DRModels._narrow_nomissing([1.0, missing])
    @test kept[1] == 1.0 && ismissing(kept[2])
    a = Any[1, 2.0, "x"]
    @test DRModels._narrow_nomissing(a) === a
    keep = [true, false, true]
  col = Union{Missing,Float64}[1.0, missing, 3.0]
    @test DRModels._subset_kept_rows(col, keep) == [1.0, 3.0]
end

@testset "cov 1125n4: _predictor_columns" begin
    f = bf(@formula(y ~ 1 + x + z + (1 | g)), @formula(sigma ~ 1))
    pc = DRModels._predictor_columns(f)
    @test :x in pc && :z in pc
    @test :y ∉ pc && :g ∉ pc
    f2 = bf(@formula(y ~ x + meta_V(v)), @formula(sigma ~ 1))
    @test :v ∉ DRModels._predictor_columns(f2)
    @test :x in DRModels._predictor_columns(f2)
end

@testset "cov 1125n4: _bivariate_response_policy" begin
    @test DRModels._bivariate_response_policy(nothing) === :drop
    inc = miss_control(response = "include")
    @test DRModels._bivariate_response_policy(inc) === :include
    drop = miss_control(response = "drop")
    @test DRModels._bivariate_response_policy(drop) === :drop
    msg = _msg(() -> DRModels._bivariate_response_policy(miss_control(predictor = "model")))
    @test occursin("predictor", msg) && occursin("bivariate", msg)
    msg2 = _msg(() -> DRModels._bivariate_response_policy(:not_control))
    @test occursin("miss_control", msg2)
end

@testset "cov 1125n4: _drop_missing_predictor_rows" begin
    dat = (; y = [1.0, 2.0, 3.0], x = Union{Missing,Float64}[1.0, missing, 3.0])
    f = bf(@formula(y ~ x), @formula(sigma ~ 1))
    out = @test_logs (:warn,) match_mode = :any begin
        DRModels._drop_missing_predictor_rows(f, dat)
    end
    @test length(out.x) == 2
    allx = (; y = [1.0, 2.0], x = Union{Missing,Float64}[missing, missing])
    msg = _msg(() -> DRModels._drop_missing_predictor_rows(f, allx))
    @test occursin("nothing left", msg)
    fmi = bf(@formula(y ~ mi(x)), @formula(sigma ~ 1))
    @test DRModels._drop_missing_predictor_rows(fmi, dat) === dat
end

@testset "cov 1125n4: relmat level refusal on drop" begin
    dat = (y = [1.0, 2.0, 3.0],
           x = Union{Missing,Float64}[1.0, missing, 1.0],
           g = ["a", "b", "a"])
    K = [1.0 0.5; 0.5 1.0]
    f = bf(@formula(y ~ x + relmat(1 | g)), @formula(sigma ~ 1))
    msg = _msg(() -> DRModels._drop_missing_predictor_rows(f, dat))
    @test occursin("levels", msg) || occursin("order", msg) || occursin("relmat", msg)
end