# Targeted coverage for src/[a-f]*.jl helpers (Cursor coverage 27 Nov 2026).
# Exercises documented contracts on finite-input guards, bootstrap status plumbing,
# design-rank messaging, and comparison variance-component labels — without touching src/.
using DRModels
using Test, Random, Logging, Tables

@testset "coverage a–f: finite_inputs / ci_args / design_rank / comparison" begin
    @testset "finite_inputs: failed log-likelihood rules and observed-row mask" begin
        @test !DRModels._loglik_is_failed(-20.0, true)
        @test DRModels._loglik_is_failed(-1e16, true)
        @test DRModels._loglik_is_failed(NaN, true)
        @test DRModels._loglik_is_failed(-1e16, false)    # sentinel, regardless of converged
        @test DRModels._loglik_is_failed(2e15, false)
        @test !DRModels._loglik_is_failed(2e15, true)

        blocks = [:mu => 1:1]
        names = [:mu => ["(Intercept)"]]
        V = reshape([1.0], 1, 1)
        means = Dict(:mu => [0.1])
        obs = Dict(:mu => [0.1])
        scales = Dict(:sigma => [1.0])
        huge = DrmFit(Gaussian(), blocks, names, [0.2], V, 5e15, 1, false,
                      means, obs, scales)
        @test DRModels._display_loglik(huge) === NaN
        fine = DrmFit(Gaussian(), blocks, names, [0.2], V, -3.0, 1, true,
                     means, obs, scales)
        @test DRModels._display_loglik(fine) == -3.0

        e = try
            DRModels._require_finite_inputs(;
                response=(Float64[1.0, Inf], "y"),
                context = "test")
            nothing
        catch err
            err
        end
        @test e isa ArgumentError
        @test occursin("Inf", e.msg)
        @test occursin("missing and NaN", e.msg)

        # Row 2 is unobserved — a NaN predictor there must not error (#1045).
        DRModels._require_finite_inputs(;
            predictors=([1.0 2.0; NaN 3.0], ["x1", "x2"]),
            observed = [true, false],
            context = "test")
        err = try
            DRModels._require_finite_inputs(;
                predictors=([1.0 NaN; 2.0 3.0], ["x1", "x2"]),
                observed = [true, true],
                context = "test")
            nothing
        catch ex
            ex
        end
        @test err isa ArgumentError
        @test occursin("predictor", err.msg)
        @test occursin("row 1", err.msg)

        tbl = (; x1 = [1.0, 2.0], x2 = [0.5, -0.5])
        DRModels._require_finite_inputs(; predictors=(tbl, ["x1", "x2"]), context = "drm")
        DRModels._require_finite_data_column(tbl, :x1)
        @test DRModels._finite_user_matrix([1.0 2.0; 3.0 4.0], "K") == [1.0 2.0; 3.0 4.0]
    end

    @testset "ci_args: bootstrap boundary share and bridge status" begin
        DR = DRModels
        vals = fill(log(1e-8), 25)
        @test DR._bootstrap_share_at_bound(vals, v -> exp(v) < DR._BOOTSTRAP_SD_BOUNDARY)
        @test !DR._bootstrap_share_at_bound(vals[1:10], v -> exp(v) < DR._BOOTSTRAP_SD_BOUNDARY)
        @test DR._bootstrap_param_at_boundary(:resd, vals)
        @test !DR._bootstrap_param_at_boundary(:sigma, vals)
        rho_vals = fill(atanh(0.99), 25)
        @test DR._bootstrap_param_at_boundary(:rho12, rho_vals)
        nat_sd = fill(1e-8, 25)
        @test DR._bootstrap_param_at_boundary(:sd_sp, nat_sd)
        cor_vals = fill(0.99, 25)
        @test DR._bootstrap_param_at_boundary(:cor_leaf, cor_vals)

        @test DR._bootstrap_status(1, 0, false) == "bootstrap_unavailable"
        @test DR._bootstrap_status(5, 0, false) == "bootstrap"
        @test DR._bootstrap_status(5, 2, false) == "bootstrap_incomplete"
        @test DR._bootstrap_status(5, 0, true) == "bootstrap_at_boundary"

        res = (; used = 30, failed = 1, boundary_params = ["resd:g:(Intercept)"])
        @test DR._bridge_bootstrap_status(res) == "bootstrap_at_boundary"
        @test DR._bridge_bootstrap_status(res, :resd, "g:(Intercept)") == "bootstrap_at_boundary"
        @test DR._bridge_bootstrap_status(res, :mu, "(Intercept)") == "bootstrap_incomplete"
        clean = (; used = 40, failed = 0, boundary_params = String[])
        @test DR._bridge_bootstrap_status(clean) == "bootstrap"

        logs, st = Test.collect_test_logs() do
            DR._finish_bootstrap_status(40, 4, ["sd:g:(Intercept)"])
        end
        @test st == "bootstrap_at_boundary"
        @test any(l -> occursin("bootstrap_at_boundary", string(l.message)), logs)
    end

    @testset "design_rank: message text and pending finding registry" begin
        X = hcat(ones(6), collect(1:6), 2 .* collect(1:6) .+ 1)
        r = DRModels._design_rank(X, ["(Intercept)", "x", "z"])
        @test r !== nothing
        @test r.rank == 2
        msg = DRModels._design_rank_message([(param = :mu, rank = r.rank, ncol = r.ncol, aliased = r.aliased)])
        @test occursin("fixed_effect_rank_deficiency", msg)
        @test occursin("z", msg)
        DRModels._check_fixed_design_rank((:mu, X, ["(Intercept)", "x", "z"]))
        @test DRModels._design_rank_pending()
    end

    @testset "comparison: variance-component block labels on a RE fit" begin
        Random.seed!(1127)
        g = repeat(1:4, inner = 5)
        x = randn(20)
        y = 0.2 .+ 0.4 .* x .+ 0.3 .* randn(20)
        full = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian();
                 data = (; y, x, g))
        reduced = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian();
                      data = (; y, x, g))
        vc_full = DRModels._variance_component_blocks(full)
        vc_red = DRModels._variance_component_blocks(reduced)
        @test :resd in vc_full
        @test isempty(vc_red)
        nest = DRModels._family_boundary_nest(Poisson(), NegBinomial2())
        @test nest !== nothing
        @test nest.added === :sigma && nest.q == 1
        @test DRModels._family_boundary_nest(reduced.family, full.family) === nothing
    end
end