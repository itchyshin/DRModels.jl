# cov lane 1125n8 — bridge inference parsers (src/bridge.jl).
using DRModels
using Test

function _argmsg(f)
    try
        f()
        error("expected an ArgumentError")
    catch err
        err isa ArgumentError || rethrow()
        return err.msg
    end
end

@testset "cov 1125n8: _bridge_parse_fixef_parm" begin
    t = DRModels._bridge_parse_fixef_parm("fixef:mu:x")
    @test t.param == :mu && t.coef == "x"
    t2 = DRModels._bridge_parse_fixef_parm("FIXEF:sigma:(Intercept)")
    @test t2.param == :sigma
    t3 = DRModels._bridge_parse_fixef_parm("fixef:mu:x:z")
    @test t3.coef == "x:z"
    @test_throws ArgumentError DRModels._bridge_parse_fixef_parm("sd:mu")
    @test_throws ArgumentError DRModels._bridge_parse_fixef_parm(42)
end

@testset "cov 1125n8: _bridge_parse_inference_parm" begin
    sd = DRModels._bridge_parse_inference_parm("sd:sigma")
    @test sd.kind == :sd && sd.param == :resd_sigma
    fx = DRModels._bridge_parse_inference_parm("fixef:mu:(Intercept)")
    @test fx.kind == :fixef && fx.param == :mu
    @test DRModels._bridge_parse_sd_parm("sd:resd_mu") === :resd_mu
    @test_throws ArgumentError DRModels._bridge_parse_inference_parm("rho12:mu")
    @test_throws ArgumentError DRModels._bridge_parse_inference_parm("sd:rho12")
end

@testset "cov 1125n8: _bridge_pick_fixef_row" begin
    rows = [(param = :mu, coef = "x", estimate = 0.2),
            (param = :sigma, coef = "(Intercept)", estimate = -0.5)]
    r = DRModels._bridge_pick_fixef_row(rows, (param = :mu, coef = "x"))
    @test r.estimate == 0.2
    msg = _argmsg(() -> DRModels._bridge_pick_fixef_row(rows, (param = :mu, coef = "z")))
    @test occursin("no row", msg)
    @test occursin("fixef:mu:z", msg)
end

@testset "cov 1125n8: _bridge_fit unknown options" begin
    msg = _argmsg(() -> DRModels._bridge_fit(
        nothing, DRModels.Gaussian(), (;);
        tree = nothing, K = nothing, A = nothing, Ainv = nothing, coords = nothing,
        options = Dict("typo_key" => 1)))
    @test occursin("unknown option", msg)
    @test occursin("typo_key", msg)
end

@testset "cov 1125n8: _bridge_align_labelled" begin
    M = [1.0 2.0; 3.0 4.0]
    out = DRModels._align_labelled_matrix(M, ["a", "b"], ["a", "b"], "K")
    @test out == M
    perm = DRModels._align_labelled_matrix(M, ["a", "b"], ["b", "a"], "K")
    @test perm[1, 1] == M[2, 2]
    msg = _argmsg(() -> DRModels._align_labelled_matrix(M, ["a"], ["a", "b"], "K"))
    @test occursin("K", msg)
    msg2 = _argmsg(() -> DRModels._align_labelled_matrix(M, ["a", "a"], ["a"], "K"))
    @test occursin("duplicated", msg2)
end