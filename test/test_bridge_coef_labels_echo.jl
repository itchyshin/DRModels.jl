# Coefficient-label echo (#563): R sends `options["coef_labels"]`
# (design 258 §7.1, a `Dict{String,Vector{String}}` keyed by dpar — base-R
# `model.matrix()` column names, in R's order, fixed-effect part only).
# Julia must echo those names verbatim (dpar-prefixed, matching the existing
# `"<dpar>_<term>"` `bridge_formula_labels_v1` convention) as the public
# `coef_names`/`vcov_names`, keep its own spelling in `raw_coef_names`, and
# fail closed on any count mismatch, unknown dpar, or a dpar with columns but
# no supplied labels — never guessing, padding, or reordering. When
# `coef_labels` is absent, behaviour must be identical to today's
# self-rendered base-R names (design 258 §7.1-7.3, confirmed by the drmTMB
# lane 2026-09-02).
using DRModels
using Test

const _ECHO_N = 40
const _ECHO_X = collect(range(-2.0, 2.0; length = _ECHO_N))
const _ECHO_Y = 0.5 .+ 0.3 .* _ECHO_X .+ [0.05 * sin(3.0 * i) for i in 1:_ECHO_N]
const _ECHO_DATA = (; y = _ECHO_Y, x = _ECHO_X)

# mu: 3 fixed-effect columns ((Intercept), x, I(x^2)); sigma defaults to an
# intercept-only Gaussian scale (1 column) when no sigma formula is given.
const _ECHO_FORMULA = "y ~ x + I(x^2)"

# Run `drm_bridge`, capturing any thrown exception instead of letting it
# abort `include(...)` — lets every case report exactly what happened.
function _echo_try(; options = Dict{String,Any}())
    try
        return (:ok, drm_bridge(; formula = _ECHO_FORMULA, family = "gaussian",
                                 data = _ECHO_DATA, options = options))
    catch e
        return (:error, e, sprint(showerror, e))
    end
end

@testset "bridge coef_labels echo (design 258 §7, #563)" begin

    @testset "(a) echo: verbatim public names, raw stays Julia's own" begin
        _, baseline = _echo_try()

        # Base-R names, exactly as drmTMB's `model.matrix()` spells them: the
        # echo pastes them verbatim, and (since 2026-09-04) refuses any
        # spelling DRModels.jl's own rendering of the same design disagrees with —
        # see test_bridge_formula_constructs.jl for the refusal cases.
        options = Dict{String,Any}("coef_labels" => Dict(
            "mu" => ["(Intercept)", "x", "I(x^2)"],
            "sigma" => ["(Intercept)"],
        ))
        status, out = _echo_try(; options = options)
        @test status === :ok
        status === :ok || return

        @test out["coef_names"] == [
            "mu_(Intercept)", "mu_x", "mu_I(x^2)", "sigma_(Intercept)",
        ]
        @test out["vcov_names"] == out["coef_names"]
        @test out["coef_label_contract"] == "bridge_formula_labels_v1"

        # raw_coef_names keeps Julia's own spelling, unaffected by the echo.
        @test out["raw_coef_names"] == baseline["raw_coef_names"]
        @test out["raw_coef_names"] != out["coef_names"]

        # coef_name_map is a consistent public -> raw bijection.
        mapping = out["coef_name_map"]
        @test Set(keys(mapping)) == Set(out["coef_names"])
        @test Set(values(mapping)) == Set(out["raw_coef_names"])
        @test [mapping[n] for n in out["coef_names"]] == out["raw_coef_names"]

        # Numbers are untouched by the echo.
        @test out["coefficients"] ≈ baseline["coefficients"]
        @test out["vcov"] ≈ baseline["vcov"]
    end

    @testset "(b) count mismatch aborts, naming dpar and both counts" begin
        options = Dict{String,Any}("coef_labels" => Dict(
            "mu" => ["(Intercept)", "x_renamed"],  # 2 supplied, mu has 3 columns
            "sigma" => ["sigma_only"],
        ))
        status, e, msg = _echo_try(; options = options)
        @test status === :error
        @test status !== :error || e isa ErrorException
        @test status !== :error || occursin("mu", msg)
        @test status !== :error || occursin("2", msg)
        @test status !== :error || occursin("3", msg)
    end

    @testset "(c) unknown dpar key aborts, naming it" begin
        options = Dict{String,Any}("coef_labels" => Dict(
            "mu" => ["(Intercept)", "x_renamed", "ix2_renamed"],
            "sigma" => ["sigma_only"],
            "notadpar" => ["whatever"],
        ))
        status, e, msg = _echo_try(; options = options)
        @test status === :error
        @test status !== :error || e isa ErrorException
        @test status !== :error || occursin("notadpar", msg)
    end

    @testset "(c2) a dpar with columns but no labels aborts, naming it" begin
        options = Dict{String,Any}("coef_labels" => Dict(
            "mu" => ["(Intercept)", "x_renamed", "ix2_renamed"],
            # sigma omitted entirely, though it has 1 fixed-effect column.
        ))
        status, e, msg = _echo_try(; options = options)
        @test status === :error
        @test status !== :error || e isa ErrorException
        @test status !== :error || occursin("sigma", msg)
    end

    @testset "(d) absent coef_labels: identical map to today" begin
        _, out1 = _echo_try()
        _, out2 = _echo_try(; options = Dict{String,Any}())
        @test out1["coef_names"] == out2["coef_names"]
        @test out1["raw_coef_names"] == out2["raw_coef_names"]
        @test out1["coef_name_map"] == out2["coef_name_map"]
    end

    @testset "(f) scalar String for a length-1 block echoes like [String]" begin
        options_vec = Dict{String,Any}("coef_labels" => Dict(
            "mu" => ["(Intercept)", "x", "I(x^2)"],
            "sigma" => ["(Intercept)"],
        ))
        options_scalar = Dict{String,Any}("coef_labels" => Dict(
            "mu" => ["(Intercept)", "x", "I(x^2)"],
            "sigma" => "(Intercept)",  # bare String, not [String] -- JuliaCall unboxing
        ))
        status_vec, out_vec = _echo_try(; options = options_vec)
        status_scalar, out_scalar = _echo_try(; options = options_scalar)
        @test status_vec === :ok
        @test status_scalar === :ok
        status_vec === :ok && status_scalar === :ok || return
        @test out_scalar["coef_names"] == out_vec["coef_names"]
        @test out_scalar["vcov_names"] == out_vec["vcov_names"]
        @test out_scalar["raw_coef_names"] == out_vec["raw_coef_names"]
        @test out_scalar["coef_name_map"] == out_vec["coef_name_map"]
    end

    @testset "(g) scalar String for a 2+ column block fails closed like a 1-vector" begin
        options = Dict{String,Any}("coef_labels" => Dict(
            "mu" => "just_one_name",  # bare String, mu has 3 columns
            "sigma" => ["sigma_only"],
        ))
        status, e, msg = _echo_try(; options = options)
        @test status === :error
        @test status !== :error || e isa ErrorException
        @test status !== :error || occursin("mu", msg)
        @test status !== :error || occursin("1", msg)
        @test status !== :error || occursin("3", msg)
    end

    @testset "(h) non-String, non-vector value fails closed" begin
        options = Dict{String,Any}("coef_labels" => Dict(
            "mu" => ["(Intercept)", "x_renamed", "ix2_renamed"],
            "sigma" => 5,  # neither a String nor a Vector{String}
        ))
        status, e, msg = _echo_try(; options = options)
        @test status === :error
        @test status !== :error || e isa ErrorException
        @test status !== :error || occursin("sigma", msg)
    end

    @testset "(e) bivariate model: echo across mu1/mu2/sigma1/sigma2/rho12" begin
        y2 = 0.4 .+ 0.2 .* _ECHO_X .+ [0.03 * cos(2.0 * i) for i in 1:_ECHO_N]
        bdata = (; y1 = _ECHO_Y, y2 = y2, x = _ECHO_X)
        bformula = Dict(
            :mu1 => "y1 ~ x", :mu2 => "y2 ~ x",
            :sigma1 => "sigma1 ~ 1", :sigma2 => "sigma2 ~ 1",
            :rho12 => "rho12 ~ 1",
        )
        options = Dict{String,Any}("coef_labels" => Dict(
            "mu1" => ["(Intercept)", "x"], "mu2" => ["(Intercept)", "x"],
            "sigma1" => ["(Intercept)"], "sigma2" => ["(Intercept)"], "rho12" => ["(Intercept)"],
        ))
        status = :error
        out = nothing
        try
            out = drm_bridge(; formula = bformula, family = "biv_gaussian",
                              data = bdata, options = options)
            status = :ok
        catch e
            @test false  # unexpected rejection; surface it via the assertion message
            rethrow()
        end
        @test status === :ok
        status === :ok || return
        @test out["coef_names"] == [
            "mu1_(Intercept)", "mu1_x", "mu2_(Intercept)", "mu2_x",
            "sigma1_(Intercept)", "sigma2_(Intercept)", "rho12_(Intercept)",
        ]
        @test out["vcov_names"] == out["coef_names"]
    end
end

println("BRIDGE_COEF_LABELS_ECHO_DONE")
