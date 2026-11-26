# cov lane 1125n8 — shared CI / bootstrap argument checks (src/ci_args.jl).
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

@testset "cov 1125n8: _validate_ci_level" begin
    @test DRModels._validate_ci_level(0.95) === 0.95
    @test DRModels._validate_ci_level(0.5; what = "probe") === 0.5
    for bad in (0.0, 1.0, -0.1, NaN, true)
        msg = _argmsg(() -> DRModels._validate_ci_level(bad; what = "confint"))
        @test occursin("confint: level", msg)
    end
end

@testset "cov 1125n8: _validate_bootstrap_B" begin
    @test DRModels._validate_bootstrap_B(10) === 10
    @test DRModels._validate_bootstrap_B(3.0) === 3
    for bad in (0, -1, 1.5, true, NaN)
        msg = _argmsg(() -> DRModels._validate_bootstrap_B(bad))
        @test occursin("B must be an integer", msg)
    end
end

@testset "cov 1125n8: bootstrap boundary detection" begin
  vals = fill(1e-12, 30)
    @test DRModels._bootstrap_share_at_bound(vals, v -> v < 1e-4)
    @test DRModels._bootstrap_param_at_boundary(:resd, fill(-10.0, 25))
    @test DRModels._bootstrap_param_at_boundary(:rho12, fill(8.0, 25))
    @test DRModels._bootstrap_param_at_boundary(:sd_mu1, fill(1e-8, 30))
    @test DRModels._bootstrap_param_at_boundary(:cor_mu1_mu2, fill(0.999, 30))
    @test !DRModels._bootstrap_param_at_boundary(:mu, [0.0, 1.0])
end

@testset "cov 1125n8: _bootstrap_status and _finish_bootstrap_status" begin
    @test DRModels._bootstrap_status(0, 0, false) == "bootstrap_unavailable"
    @test DRModels._bootstrap_status(5, 2, false) == "bootstrap_incomplete"
    @test DRModels._bootstrap_status(5, 0, true) == "bootstrap_at_boundary"
    @test DRModels._bootstrap_status(5, 0, false) == "bootstrap"
    st = @test_logs (:warn,) match_mode = :any begin
        DRModels._finish_bootstrap_status(25, 5, [:sd_mu1])
    end
    @test st == "bootstrap_incomplete" || st == "bootstrap_at_boundary"
end

@testset "cov 1125n8: _bridge_bootstrap_status" begin
    res = (used = 4, failed = 0, boundary_params = ["sd_mu1:phylo"])
    @test DRModels._bridge_bootstrap_status(res) in ("bootstrap", "bootstrap_at_boundary")
    @test DRModels._bridge_bootstrap_status(res, :sd_mu1, "phylo") in
          ("bootstrap", "bootstrap_at_boundary")
    @test DRModels._bridge_bootstrap_status(res, :mu, "x") == "bootstrap"
end