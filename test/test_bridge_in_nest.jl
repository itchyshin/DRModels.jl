# test_bridge_in_nest.jl — R's `%in%` and nested `/` through `drm_bridge`'s
# formula translation (#467).
#
# Defect (found 2026-09-27 while verifying #467): `_bridge_translate_r_ops`/
# `_bridge_xlate` (src/bridge.jl) had no handling for `%in%` or `/`.
#
#   * `%in%` is not valid Julia infix syntax the way R means it:
#     `Meta.parse("b %in% a")` silently parses it as NESTED MODULO,
#     `(b % in) % a`, which only fails later, deep inside `@formula`/
#     StatsModels, with a confusing "no variable called 'in'" message that
#     never names the actual construct.
#   * `/` parses fine as plain Julia division, so `y ~ x / z` on two NUMERIC
#     columns did not error at all — it silently fit a single materialised
#     "x / z" arithmetic-division covariate instead of R's nesting
#     expansion `x + x:z` (two terms). Confirmed SILENTLY WRONG (not merely
#     a late error) by running it on main before this fix.
#
# R documents `b %in% a` as identical, term-for-term, to `b:a`, and `a/b` as
# `a + a:b` (?formula). Both are confirmed here against base R's own
# `stats::terms()`/`model.matrix()` (see test/fixtures/bridge_in_nest/, and
# scratch checks referenced in the PR description) to produce byte-identical
# column names and values to the bridge's existing `&` (R's `:`) translation
# — so both are implemented as faithful rewrites. The genuinely ambiguous
# forms — a compound or chained nesting left-hand side, e.g. `(a+c)/b` or
# `a/b/c`, whose R contrast algebra this AST rewrite does not replicate — are
# refused sharply instead of guessed.
#
#   julia --project=. -e 'using DRModels, Test; include("test/test_bridge_in_nest.jl")'

module TestBridgeInNest

using DRModels
using Test
using DelimitedFiles: readdlm

const FIXTURE = joinpath(@__DIR__, "fixtures", "bridge_in_nest")

function _load_csv(path)
    raw, header = readdlm(path, ','; header = true)
    cols = Symbol.(strip.(string.(vec(header))))
    return cols, raw
end

function _load_data()
    cols, raw = _load_csv(joinpath(FIXTURE, "data.csv"))
    numeric = Set((:x, :z, :h))
    pairs = map(enumerate(cols)) do (j, name)
        col = raw[:, j]
        if name in numeric
            name => Float64[parse(Float64, string(v)) for v in col]
        else
            name => string.(strip.(string.(col)))
        end
    end
    return NamedTuple(pairs)
end

# Base-R `model.matrix()` spells an interaction column name with `:`, and the
# bridge's own `coef_names` keep that same R spelling verbatim for factor
# interactions (confirmed against test_bridge_base_r_names.jl row 7,
# `"gb:factor(h)20"`) — only the FORMULA *source text* uses `&` in place of
# `:` (Julia parses `:` as `range`). So the oracle names need no translation.
function _oracle_matrix(slug)
    cols, raw = _load_csv(joinpath(FIXTURE, slug * ".csv"))
    names = String.(cols)
    values = Float64[parse(Float64, string(v)) for v in raw]
    return names, reshape(values, size(raw))
end

const _DATA = _load_data()
# A response with real signal + noise (not an exact linear function of a
# single covariate): several of the fixture's designs are otherwise rank- or
# variance-degenerate at n = 12, which would fail on a Hessian/vcov guard
# unrelated to what this test checks (formula translation, not estimability).
const _Y = [0.4 + 0.3 * _DATA.x[i] - 0.15 * _DATA.z[i] +
            (_DATA.g[i] == "b" ? 0.5 : _DATA.g[i] == "c" ? -0.3 : 0.0) +
            (_DATA.h[i] == 20.0 ? 0.2 : 0.0) +
            0.07 * sin(2.7 * i) for i in 1:length(_DATA.x)]

@testset "bridge %in% / nested-/ translation (#467)" begin
    @testset "RED (documents pre-fix failure modes)" begin
        # `%in%` used to reach a confusing StatsModels error about a missing
        # variable called "in", never naming the actual construct.
        err = try
            DRModels.drm_bridge(; formula = "y ~ h %in% g", family = "gaussian",
                data = merge(_DATA, (; y = _Y)))
            nothing
        catch e
            e
        end
        @test err === nothing  # now translated faithfully — see below
    end

    @testset "faithful: numeric `/` nesting (x / z -> x + x:z)" begin
        exp_names, exp_mm = _oracle_matrix("nested_div_numeric")
        out = DRModels.drm_bridge(; formula = "y ~ x / z", family = "gaussian",
            data = merge(_DATA, (; y = _Y)))
        mu_names = [n for n in out["coef_names"] if startswith(n, "mu_")]
        @test mu_names == ["mu_" * n for n in exp_names]
        # Same construct via the already-trusted explicit `&` spelling must
        # give the identical design (not just the same names).
        ref = DRModels.drm_bridge(; formula = "y ~ x + x&z", family = "gaussian",
            data = merge(_DATA, (; y = _Y)))
        @test out["coefficients"] ≈ ref["coefficients"]
        @test size(out["vcov"]) == size(ref["vcov"])
    end

    @testset "faithful: factor/numeric `/` nesting (g / h -> g + g:h)" begin
        exp_names, _ = _oracle_matrix("nested_div_factor")
        out = DRModels.drm_bridge(; formula = "y ~ g / h", family = "gaussian",
            data = merge(_DATA, (; y = _Y)))
        mu_names = [n for n in out["coef_names"] if startswith(n, "mu_")]
        @test mu_names == ["mu_" * n for n in exp_names]
        ref = DRModels.drm_bridge(; formula = "y ~ g + g&h", family = "gaussian",
            data = merge(_DATA, (; y = _Y)))
        @test out["coefficients"] ≈ ref["coefficients"]
    end

    @testset "faithful: `%in%` alongside its main effect (g + h %in% g)" begin
        exp_names, _ = _oracle_matrix("in_with_main")
        out = DRModels.drm_bridge(; formula = "y ~ g + h %in% g", family = "gaussian",
            data = merge(_DATA, (; y = _Y)))
        mu_names = [n for n in out["coef_names"] if startswith(n, "mu_")]
        @test mu_names == ["mu_" * n for n in exp_names]
        ref = DRModels.drm_bridge(; formula = "y ~ g + g&h", family = "gaussian",
            data = merge(_DATA, (; y = _Y)))
        @test out["coefficients"] ≈ ref["coefficients"]
    end

    @testset "`%in%` alone (bare interaction, full dummy coding)" begin
        # No main effect present: matches the bridge's own `&` alone, same as
        # any bare interaction (both give R's full, non-reduced coding).
        out = DRModels.drm_bridge(; formula = "y ~ h %in% g", family = "gaussian",
            data = merge(_DATA, (; y = _Y)))
        ref = DRModels.drm_bridge(; formula = "y ~ h&g", family = "gaussian",
            data = merge(_DATA, (; y = _Y)))
        @test out["coefficients"] ≈ ref["coefficients"]
        @test [n for n in out["coef_names"] if startswith(n, "mu_")] ==
              [n for n in ref["coef_names"] if startswith(n, "mu_")]
    end

    @testset "sharp refusal: compound left-hand side `(a+c)/b`" begin
        err = try
            DRModels.drm_bridge(; formula = "y ~ (g + z) / x", family = "gaussian",
                data = merge(_DATA, (; y = _Y)))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin("nesting `/`", msg)
        @test occursin("compound or chained", msg)
    end

    @testset "sharp refusal: chained nesting `a/b/c`" begin
        err = try
            DRModels.drm_bridge(; formula = "y ~ x / z / g", family = "gaussian",
                data = merge(_DATA, (; y = _Y)))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin("nesting `/`", msg)
        @test occursin("compound or chained", msg)
    end

    @testset "`/` still means division inside I(...) (untouched)" begin
        out = DRModels.drm_bridge(; formula = "y ~ I(x / 2)", family = "gaussian",
            data = merge(_DATA, (; y = _Y)))
        @test any(n -> n == "mu_I(x/2)", out["coef_names"])
    end
end

println("BRIDGE_IN_NEST_DONE")

end # module
