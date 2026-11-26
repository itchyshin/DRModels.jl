# cov lane 1125n9 — finite input guards and converged backstop (src/finite_inputs.jl).
using DRModels
using Test

_msg(f) = try
    f()
    ""
catch e
    e isa ArgumentError ? e.msg : sprint(showerror, e)
end

@testset "cov 1125n9: _input_offender and _skip_unobserved_row" begin
    @test DRModels._input_offender(1.0; allow_missing = true) === nothing
    @test DRModels._input_offender(missing; allow_missing = true) === nothing
    @test DRModels._input_offender(missing; allow_missing = false) == "missing"
    @test DRModels._input_offender(NaN; allow_missing = true) === nothing
    @test DRModels._input_offender(NaN; allow_missing = false) == "NaN"
    @test DRModels._input_offender(Inf; allow_missing = false) == "Inf"
    @test DRModels._input_offender(-Inf32; allow_missing = false) == "-Inf"
    @test DRModels._input_offender("ok"; allow_missing = false) === nothing
    obs = BitVector([true, false, true])
    @test !DRModels._skip_unobserved_row(obs, 1)
    @test DRModels._skip_unobserved_row(obs, 2)
    @test !DRModels._skip_unobserved_row(obs, 4)
    @test !DRModels._skip_unobserved_row(nothing, 1)
end

@testset "cov 1125n9: _require_finite_response" begin
    @test DRModels._require_finite_response([1.0, 2.0], :y;
        allow_missing = true, context = "drm") === nothing
  msg = _msg(() -> DRModels._require_finite_response([1.0, Inf], :y;
        allow_missing = true, context = "associate"))
    @test occursin("associate", msg) && occursin("Inf", msg)
    msg2 = _msg(() -> DRModels._require_finite_response([missing, 1.0], :count;
        allow_missing = false, context = "drm"))
    @test occursin("missing", msg2)
    msg3 = _msg(() -> DRModels._require_finite_response([NaN], :y;
        allow_missing = false, context = "drm"))
    @test occursin("NaN", msg3)
end

@testset "cov 1125n9: _require_finite_predictors with observed mask" begin
    X = [1.0 2.0; 3.0 4.0]
    DRModels._require_finite_predictors(X, ["x1", "x2"]; context = "drm")
    obs = BitVector([true, false])
    DRModels._require_finite_predictors(X, ["block"]; context = "drm", observed = obs)
    Xbad = copy(X); Xbad[2, 2] = NaN
    msg = _msg(() -> DRModels._require_finite_predictors(Xbad, ["x1", "x2"]; context = "drm"))
    @test occursin("predictor", msg) && occursin("x2", msg)
    msg2 = _msg(() -> DRModels._require_finite_predictors([1.0, Inf], "offset";
        context = "drm", observed = BitVector([true, true])))
    @test occursin("offset", msg2)
    vbad = [1.0, NaN]
    msg3 = _msg(() -> DRModels._require_finite_predictors(vbad, "z";
        context = "drm", observed = BitVector([true, true])))
    @test occursin("row 2", msg3)
end

@testset "cov 1125n9: weights, offset, coords arrays" begin
    @test_throws ArgumentError DRModels._require_finite_inputs(; weights = [NaN])
    @test_throws ArgumentError DRModels._require_finite_inputs(; weights = [Inf])
    @test DRModels._require_finite_inputs(; weights = [1.0, 2.0]) === nothing
    @test DRModels._require_finite_inputs(; offset = zeros(3)) === nothing
    M = [1.0 2.0; 3.0 4.0]
    @test DRModels._require_finite_inputs(; coords = M) === nothing
    @test_throws ArgumentError DRModels._require_finite_inputs(;
        coords = [1.0 Inf; 2.0 3.0])
    tbl = (x = [1.0, 2.0], y = [3.0, 4.0])
    M = DRModels._finite_check_values(tbl)
    @test size(M) == (2, 2) && all(isfinite, M)
    @test DRModels._finite_check_values([1, 2, 3]) == [1, 2, 3]
    @test DRModels._finite_check_values(42) === nothing
end

@testset "cov 1125n9: _finite_user_matrix" begin
    K = DRModels._finite_user_matrix([1.0 0.2; 0.2 1.0], "K")
    @test K ≈ [1.0 0.2; 0.2 1.0]
    @test_throws ArgumentError DRModels._finite_user_matrix([1.0 Inf], "A")
end

@testset "cov 1125n9: loglik sentinels and _report_converged" begin
    @test DRModels._loglik_is_sentinel(-1e15)
    @test DRModels._loglik_is_sentinel(-1e16)
    @test DRModels._loglik_is_sentinel(NaN)
    @test !DRModels._loglik_is_sentinel(-1e14)
    @test DRModels._loglik_is_failed(-1e18, true)
    @test DRModels._loglik_is_failed(1e16, false)
    @test !DRModels._loglik_is_failed(1e16, true)
    @test DRModels._coefficients_finite([0.0, 1.0])
    @test !DRModels._coefficients_finite([0.0, NaN])
    @test DRModels._report_converged(true, -10.0, [0.1])
    @test !DRModels._report_converged(true, -1e18, [0.1])
    @test !DRModels._report_converged(false, -10.0, [0.1])
end