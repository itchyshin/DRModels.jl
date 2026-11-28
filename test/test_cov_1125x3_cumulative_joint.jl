# Round 3 — src/cumulative_predict.jl + joint_missing_bridge.jl + joint_missing_finite_bridge.jl.
using DRModels, Test, TOML, LinearAlgebra

@testset "cov 1125x3 cumulative predict design mismatch" begin
    d = (; y = repeat([1.0, 2.0, 3.0], inner = 5),
         x = randn(15),
         f = repeat(["a", "b", "c"], inner = 5))
    fit = drm(bf(@formula(y ~ x + f)), CumulativeLogit(); data = d)
  nd = (; x = [0.0], f = ["z"])
    err = try predict(fit, nd); nothing catch e; e end
    @test err isa ArgumentError && occursin("predict", err.msg)
    good = (; x = [0.0, 1.0], f = ["a", "c"])
    η = predict(fit, good; type = :link)
    @test length(η) == 2
    @test all(isfinite, η)
    pp = predict_parameters(fit, good)
    @test pp[:mu] ≈ η atol = 1e-12
    fx = drm(bf(@formula(y ~ x)), CumulativeLogit(); data = d)
    @test predict(fx, (; x = [1.0, -1.0])) ≈ coef(fx, :mu)[1] .* [1.0, -1.0]
end

ref = TOML.parsefile(joinpath(@__DIR__, "fixtures", "joint_missing_predictor", "native_reference.toml"))
function _jm_payload(kind)
    d = ref[kind]
    n = length(d["x"])
    Dict{String, Any}(
        "schema" => "joint_missing_v1", "predictor" => kind, "variable" => "body_mass",
        "y" => copy(d["y"]), "x" => copy(d["x"]),
        "observed_y" => copy(d["y_observed"]), "observed_x" => copy(d["x_observed"]),
        "X_mu" => hcat(ones(n), d["x"], d["z"]),
        "X_sigma" => ones(n, 1), "X_predictor" => hcat(ones(n), d["z"]),
        "mu_col" => 2, "mu_names" => ["(Intercept)", "mi(body_mass)", "z"],
        "sigma_names" => ["(Intercept)"], "predictor_names" => ["(Intercept)", "z"],
        "original_row" => copy(d["original_row"]), "options" => Dict("g_tol" => 1e-8))
end

@testset "cov 1125x3 joint bridge helper refusals" begin
    p = _jm_payload("gaussian")
    @test_throws ArgumentError DRModels._prepare_joint_bridge(Dict("schema" => "nope"))
    bad = deepcopy(p)
    bad["observed_y"] = fill(2, length(p["y"]))
    @test_throws ArgumentError DRModels._prepare_joint_bridge(bad)
    bad = deepcopy(p)
    bad["X_mu"] = fill(NaN, size(p["X_mu"])...)
    @test_throws ArgumentError DRModels._prepare_joint_bridge(bad)
    bad = deepcopy(p)
    bad["options"] = Dict("g_tol" => -1.0)
    @test_throws ArgumentError DRModels._prepare_joint_bridge(bad)
    bad = deepcopy(p)
    bad["predictor"] = "poisson"
    @test_throws ArgumentError DRModels._prepare_joint_bridge(bad)
    @test_throws ArgumentError DRModels._joint_bridge_dict([], "x")
    @test_throws ArgumentError DRModels._joint_bridge_mask(ones(3), 4, "m")
    @test_throws ArgumentError DRModels._joint_bridge_mask([0, 2], 2, "m")
    @test_throws ArgumentError DRModels._joint_bridge_names(["", "a"], 2, "n")
    @test_throws ArgumentError DRModels._joint_bridge_names(["a", "a"], 2, "n")
end

@testset "cov 1125x3 joint bridge gaussian + bernoulli" begin
    for kind in ("gaussian", "bernoulli")
        out = DRModels.drm_bridge_joint(_jm_payload(kind))
        @test out["schema"] == "joint_missing_result_v1"
        @test out["converged"]
        @test length(out["coefficients"]) == length(out["coef_names"])
        @test all(isfinite, out["coefficients"])
    end
end

function _finite_payload(kind)
    c = TOML.parsefile(joinpath(@__DIR__, "..", "docs/dev-log/evidence/julia-r-parity/finite-state/finite-reference-003.toml"))[kind]
    mat(v) = reduce(vcat, permutedims.(Float64.(r) for r in v))
    Xstate = mat(c["state_design"])
    Xmu = copy(Xstate[1:c["K"]:end, :])
    for i in 1:c["n"]
        if c["observed_x"][i]
            Xmu[i, :] .= Xstate[(i - 1) * c["K"] + c["x"][i], :]
        end
    end
    Dict{String, Any}(
        "schema" => "joint_missing_finite_v1", "predictor" => kind, "variable" => "x",
        "levels" => c["levels"], "y" => c["y"], "x" => c["x"],
        "observed_y" => c["observed_y"], "observed_x" => c["observed_x"],
        "X_mu" => Xmu, "X_mu_state" => Xstate, "state_layout" => "row_then_state",
        "X_sigma" => mat(c["X_sigma"]), "X_predictor" => mat(c["X_predictor"]),
        "mu_names" => c["mu_names"], "sigma_names" => c["sigma_names"],
        "predictor_names" => c["predictor_names"], "original_row" => c["original_row"],
        "options" => Dict("g_tol" => 1e-8))
end

@testset "cov 1125x3 finite joint bridge" begin
    for kind in ("ordinal", "categorical")
        r = DRModels.drm_bridge_joint(_finite_payload(kind))
        @test r["schema"] == "joint_missing_finite_result_v1"
        @test maximum(abs, sum(r["conditional_probabilities"]; dims = 2) .- 1) < 1e-10
        @test r["original_row"] == collect(1:180)
    end
end