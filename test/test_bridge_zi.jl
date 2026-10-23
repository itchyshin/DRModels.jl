# The `zi` (and `hu`) count-mixture parts through the MARSHALLING BOUNDARY.
#
# `test_zi.jl` / `test_hurdle.jl` exercise these mixtures through native `drm`.
# Nothing exercised them through `drm_bridge` -- the boundary drmTMB's
# `engine = "julia"` actually crosses. drmTMB banks a zero-inflated Poisson
# parity receipt against this route (`capability_id zi_poisson` in
# docs/dev-log/evidence/parity-fixtures.tsv), so the route was load-bearing
# with no test of its own on this side.
#
# The admission mechanism under test: zero-inflation is a FORMULA property, not
# a family. `_bridge_family` never sees a `zi_*` tag -- the family stays
# `poisson`/`nbinom2` and `_bridge_formula` routes a keyed `zi` entry into the
# `bf(...)` bundle. drmTMB spells it identically (`drm_family_type()` returns
# `"poisson"` for a ZIP; `"zi_poisson"` is only its post-fit `model_type`).
#
# The second half pins the REFUSAL: `family = "zi_poisson"` must throw, and the
# message must name the spelling that works. It is deliberately NOT an alias --
# see `_bridge_mixture_family_hint` in src/bridge.jl.
using DRModels
using Test, Random
import Distributions

# Bridge coefficients arrive as a name => value Dict; native `coef(fit, :dpar)`
# arrives as a per-dpar vector in design-matrix column order. Compare BY NAME.
function _bridge_vs_native_coef(res, fit, dpars)
    bridge = res["coef"]
    worst = 0.0
    seen = 0
    for dp in dpars
        native = coef(fit, dp)
        # `<dpar>_<term>` is the bridge's public block spelling (design 258).
        keys_dp = sort([k for k in keys(bridge) if startswith(k, string(dp) * "_")])
        @test length(keys_dp) == length(native)
        # Column order within a block is the design-matrix order on both sides.
        ordered = [k for k in ["$(dp)_(Intercept)", "$(dp)_x"] if k in keys_dp]
        @test length(ordered) == length(keys_dp)   # no unexpected extra term
        for (i, k) in enumerate(ordered)
            worst = max(worst, abs(bridge[k] - native[i]))
            seen += 1
        end
    end
    return (worst, seen)
end

@testset "Bridge boundary: zero-inflation (`zi`) count mixtures" begin

    @testset "ZIP -- drm_bridge(family=\"poisson\", zi entry) == native drm" begin
        Random.seed!(20260905)
        n = 400
        x = randn(n)
        λ = exp.(0.6 .+ 0.4 .* x)
        ηπ = -0.8 .+ 0.5 .* x
        π = 1 ./ (1 .+ exp.(-ηπ))
        y = Float64.([rand() < π[i] ? 0 : rand(Distributions.Poisson(λ[i])) for i in 1:n])

        res = DRModels.drm_bridge(; formula = Dict("mu" => "y ~ x", "zi" => "zi ~ x"),
                               family = "poisson", data = Dict("y" => y, "x" => x))
        fit = drm(bf(@formula(y ~ x), @formula(zi ~ x)), Poisson(); data = (; y, x))

        # The bridge fitted the ZERO-INFLATED model, not a plain Poisson: four
        # coefficients in two named blocks, and both blocks are present.
        @test length(res["coef"]) == 4
        @test haskey(res["coef"], "mu_(Intercept)")
        @test haskey(res["coef"], "zi_(Intercept)")
        @test haskey(res["coef"], "zi_x")

        worst, seen = _bridge_vs_native_coef(res, fit, (:mu, :zi))
        @test seen == 4                       # every coefficient really compared
        @test worst < 1e-8
        @test isfinite(res["loglik"])
        @test res["loglik"] ≈ loglik(fit) atol = 1e-8

        # A plain Poisson on the same data is a DIFFERENT, worse fit -- so the
        # `zi` entry demonstrably changed the likelihood rather than being
        # silently dropped somewhere in the marshalling.
        plain = DRModels.drm_bridge(; formula = Dict("mu" => "y ~ x"),
                                 family = "poisson", data = Dict("y" => y, "x" => x))
        @test length(plain["coef"]) == 2
        @test res["loglik"] > plain["loglik"] + 1.0

        # The zero-inflation probability is recovered on the response scale.
        @test 1 / (1 + exp(-res["coef"]["zi_(Intercept)"])) ≈ 1 / (1 + exp(0.8)) atol = 0.12
    end

    @testset "ZINB -- drm_bridge(family=\"nbinom2\", sigma + zi entries) == native drm" begin
        Random.seed!(20260906)
        n = 400
        x = randn(n)
        θ = 3.0
        μ = exp.(0.6 .+ 0.4 .* x)
        πz = 0.30
        y = Float64.([rand() < πz ? 0 : rand(Distributions.NegativeBinomial(θ, θ / (θ + μ[i]))) for i in 1:n])

        res = DRModels.drm_bridge(; formula = Dict("mu" => "y ~ x", "sigma" => "sigma ~ 1",
                                              "zi" => "zi ~ 1"),
                               family = "nbinom2", data = Dict("y" => y, "x" => x))
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(zi ~ 1)),
                  NegBinomial2(); data = (; y, x))

        @test length(res["coef"]) == 4        # mu (2) + sigma (1) + zi (1)
        worst, seen = _bridge_vs_native_coef(res, fit, (:mu, :sigma, :zi))
        @test seen == 4
        @test worst < 1e-8
        @test res["loglik"] ≈ loglik(fit) atol = 1e-8
    end

    @testset "`zi_poisson` is REFUSED, with the working spelling named" begin
        # Not an alias, on purpose: `zi_poisson -> Poisson()` would fit a PLAIN
        # Poisson without error whenever the caller omitted the `zi ~` part.
        @test_throws ArgumentError DRModels._bridge_family("zi_poisson")

        msg = try
            DRModels._bridge_family("zi_poisson")
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("unsupported family `zi_poisson`", msg)
        @test occursin("family = \"poisson\"", msg)   # the spelling that works
        @test occursin("zi", msg)
        @test occursin("formula", msg)

        # Reached through the public entry point too, not only the internal.
        Random.seed!(20260907)
        n = 60
        x = randn(n)
        y = Float64.(rand(Distributions.Poisson(2.0), n))
        @test_throws ArgumentError DRModels.drm_bridge(;
            formula = Dict("mu" => "y ~ x", "zi" => "zi ~ 1"),
            family = "zi_poisson", data = Dict("y" => y, "x" => x))
    end

    @testset "the hint covers the other mixture spellings, and only those" begin
        for (tag, want) in (("zi_nbinom2", "nbinom2"), ("zero_inflated_poisson", "poisson"),
                            ("hurdle_nbinom2", "nbinom2"))
            msg = try
                DRModels._bridge_family(tag)
                ""
            catch e
                sprint(showerror, e)
            end
            @test occursin("unsupported family `$tag`", msg)
            @test occursin("family = \"$want\"", msg)
        end
        # `hu` is named for a hurdle tag, `zi` for a zero-inflated one.
        @test occursin("`hu` entry", sprint(showerror,
            try DRModels._bridge_family("hurdle_nbinom2") catch e; e end))
        @test occursin("`zi` entry", sprint(showerror,
            try DRModels._bridge_family("zi_poisson") catch e; e end))

        # NO hint leakage: a genuinely unknown family, and a mixture prefix on a
        # NON-count family, still get the plain message with no suggestion.
        for tag in ("weibull", "zi_gaussian", "hurdle_beta")
            msg = try
                DRModels._bridge_family(tag)
                ""
            catch e
                sprint(showerror, e)
            end
            @test occursin("unsupported family `$tag`", msg)
            @test !occursin("formula parts here", msg)
        end
    end
end
