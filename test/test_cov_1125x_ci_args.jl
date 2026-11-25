# test_cov_1125x_ci_args.jl — shared CI / bootstrap helper contracts (src/ci_args.jl).
using DRModels
using Test

_msg(f) = try
    f()
    ""
catch e
    e isa ArgumentError || rethrow()
    e.msg
end

@testset "ci_args: level and B validation" begin
    @test DRModels._validate_ci_level(0.95) === 0.95
    @test DRModels._validate_bootstrap_B(4.0) === 4
    @test DRModels._validate_ci_level(0.9; what = "confint") === 0.9

    for bad in (-0.1, 0.0, 1.0, NaN, Inf, true)
        msg = _msg(() -> DRModels._validate_ci_level(bad))
        @test occursin("level", msg) && occursin(repr(bad), msg)
        msg2 = _msg(() -> DRModels._validate_ci_level(bad; what = "icc"))
        @test occursin("icc: level", msg2)
    end

    for bad in (0, -1, 1.2, NaN, true)
        msg = _msg(() -> DRModels._validate_bootstrap_B(bad))
        @test occursin("B must be an integer", msg)
        @test occursin(repr(bad), msg)
    end
end

@testset "ci_args: bootstrap boundary detection" begin
    tiny_log = fill(log(1e-6), 25)
    @test DRModels._bootstrap_param_at_boundary(:resd, tiny_log)
    @test DRModels._bootstrap_param_at_boundary(:sd_phylo, tiny_log)
    @test !DRModels._bootstrap_param_at_boundary(:resd, fill(0.0, 25))

    rho_hi = fill(atanh(0.99), 30)
    @test DRModels._bootstrap_param_at_boundary(:rho12, rho_hi)
    @test !DRModels._bootstrap_param_at_boundary(:rho12, zeros(30))

    @test DRModels._bootstrap_param_at_boundary(:sd_mu1, fill(1e-6, 25))
    @test DRModels._bootstrap_param_at_boundary(:cor_mu1_mu2, fill(0.99, 25))
    @test !DRModels._bootstrap_param_at_boundary(:mu, fill(0.3, 25))
end

@testset "ci_args: bootstrap status strings" begin
    @test DRModels._bootstrap_status(1, 0, false) == "bootstrap_unavailable"
    @test DRModels._bootstrap_status(2, 0, false) == "bootstrap"
    @test DRModels._bootstrap_status(10, 3, false) == "bootstrap_incomplete"
    @test DRModels._bootstrap_status(30, 0, true) == "bootstrap_at_boundary"

    whole = (; used = 40, failed = 2, boundary_params = ["resd:(Intercept)"])
    @test DRModels._bridge_bootstrap_status(whole) == "bootstrap_at_boundary"
    @test DRModels._bridge_bootstrap_status(whole, :resd, "(Intercept)") ==
        "bootstrap_at_boundary"
    @test DRModels._bridge_bootstrap_status(whole, :mu, "x") == "bootstrap_incomplete"

    incomplete = (; used = 40, failed = 2, boundary_params = String[])
    @test DRModels._bridge_bootstrap_status(incomplete) == "bootstrap_incomplete"

    bare = (; used = 5, failed = 0, boundary_params = ["resd"])
    @test DRModels._bridge_bootstrap_status(bare, :resd) == "bootstrap_at_boundary"
end

@testset "ci_args: _finish_bootstrap_status warnings" begin
  @test_logs (:warn,) match_mode = :any DRModels._finish_bootstrap_status(12, 2, ["resd"])
    st = DRModels._finish_bootstrap_status(8, 0, String[])
    @test st == "bootstrap"
end