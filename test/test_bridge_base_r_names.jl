# RED-first measurement: does DRModels.jl's bridge_formula_labels_v1 `coef_names`
# already match base-R `stats::model.matrix()` spelling for the ten formula
# constructs in drmTMB design 258 §2?  This is a MEASUREMENT test, not a
# fix — every construct is exercised even when the bridge is known/expected
# to reject or mis-render it, so a failure records exactly what happened
# (rejection message or wrong spelling/order) rather than skipping the case.
#
# Rows 1-6 are DRModels.jl's #467 cohort (already partially covered by
# test_bridge_formula_labels.jl); rows 7-10 were added later from drmTMB's
# non-shipping `public-004.json` oracle and are NOT covered by any existing
# DRModels.jl fixture. Row 7 in particular exercises R's term-ORDER algorithm for
# a two-factor crossed design and is flagged in the brief as the
# highest-risk untested case.
#
# A second combined @testset measures whether a `coef_labels` payload field
# (design 258 §7.1's per-dpar base-R name list, forwarded here via
# `options["coef_labels"]` since there is no dedicated `drm_bridge` kwarg —
# see the brief §1/§6) has ANY effect on the returned `coef_names`. Today
# nothing in src/bridge.jl reads `opts[:coef_labels]`, so this measures the
# field's current inertness per the coordinator's 2026-09-02 update citing
# drmTMB claude/rev-parity-c2-label-producer @ af1790492.
using DRModels
using Test

# ---------------------------------------------------------------------------
# Shared synthetic data.  One dataset services every construct; unused
# columns are simply ignored by whichever formula does not reference them.
# `g` (3 levels a/b/c, alphabetical baseline "a") and `h` (2 numeric levels
# 10/20, `factor(h)` baseline "10" under R's ascending-numeric-value default)
# are the row-7 crossed-factor columns. `grp` (3 levels hi/lo/mid,
# alphabetical baseline "hi") is row 2's plain factor column.
# ---------------------------------------------------------------------------
const _N = 60
const _X = collect(range(-2.0, 2.0; length = _N))
const _Z = [sin(0.3 * i) + 0.05 * i for i in 1:_N]
const _G = [["a", "b", "c"][mod1(i, 3)] for i in 1:_N]
const _H = [iseven(i) ? 20.0 : 10.0 for i in 1:_N]
const _GRP = [["hi", "lo", "mid"][mod1(i, 3)] for i in 1:_N]
const _Y = 0.6 .+ 0.35 .* _X .- 0.08 .* _X .^ 2 .+ 0.25 .* _Z .+
    [g == "b" ? 0.4 : g == "c" ? -0.25 : 0.0 for g in _G] .+
    [h == 20.0 ? 0.3 : 0.0 for h in _H] .+
    [grp == "lo" ? 0.2 : grp == "mid" ? -0.15 : 0.0 for grp in _GRP] .+
    [0.05 * sin(7.0 * i) for i in 1:_N]

const _DATA = (; y = _Y, x = _X, z = _Z, g = _G, h = _H, grp = _GRP)

# (id, description, formula, dpar, expected base-R names — unprefixed, in
# `coefficients`/`vcov` order.)
const _CONSTRUCTS = [
    (id = 1, formula = "y ~ x + I(x^2)",
     expected = ["(Intercept)", "x", "I(x^2)"]),
    (id = 2, formula = "y ~ factor(grp)",
     expected = ["(Intercept)", "factor(grp)lo", "factor(grp)mid"]),
    (id = 3, formula = "y ~ poly(x, 3)",
     expected = ["(Intercept)", "poly(x, 3)1", "poly(x, 3)2", "poly(x, 3)3"]),
    (id = 4, formula = "y ~ z * poly(x, 2)",
     expected = ["(Intercept)", "z", "poly(x, 2)1", "poly(x, 2)2",
                 "z:poly(x, 2)1", "z:poly(x, 2)2"]),
    (id = 5, formula = "y ~ (x + z)^2",
     expected = ["(Intercept)", "x", "z", "x:z"]),
    (id = 6, formula = "y ~ scale(x)",
     expected = ["(Intercept)", "scale(x)"]),
    (id = 7, formula = "y ~ g + factor(h) + factor(h):g",
     # Oracle measured with base R 2026-09-02: colnames(model.matrix(~ g + factor(h) + factor(h):g, d))
     # gives REDUCED coding (both main effects present); design 258 §2 row 7 listed a 9-name
     # full-dummy expansion, which R does not produce for this formula.
     expected = ["(Intercept)", "gb", "gc", "factor(h)20",
                 "gb:factor(h)20", "gc:factor(h)20"]),
    (id = 8, formula = "y ~ I(+x)",
     expected = ["(Intercept)", "I(+x)"]),
    (id = 9, formula = "y ~ I(x + (z + 2))",
     expected = ["(Intercept)", "I(x + (z + 2))"]),
    (id = 10, formula = "y ~ I(x^2)",
     expected = ["(Intercept)", "I(x^2)"]),
]

# Call `drm_bridge`, catching any rejection so the RED report carries the
# exact ArgumentError/ErrorException text rather than aborting the run.
function _try_bridge(; formula, options = Dict{String,Any}())
    try
        out = drm_bridge(; formula = formula, family = "gaussian", data = _DATA,
                          options = options)
        return (:ok, out)
    catch e
        return (:error, sprint(showerror, e))
    end
end

# Every construct here supplies only the `mu` formula and lets `sigma`
# default (an intercept-only Gaussian scale), so `out["coef_names"]` also
# carries `sigma_(Intercept)`. Restrict the comparison to the `mu`-prefixed
# block — the per-dpar selector design 258 §7.1's `coef_labels` payload
# itself uses (keyed by dpar string) — not the whole coordinate vector.
_dpar_block(names, dpar::AbstractString) =
    [n for n in names if startswith(n, dpar * "_")]

# Everything below is nested inside ONE top-level @testset: an un-nested
# `@testset` that fails throws immediately after printing its own summary,
# which would abort `include(...)` after row 1 and hide rows 2-10. Nesting
# keeps every row's result visible in one final report.
@testset "bridge base-R coefficient names (design 258 §2, RED measurement)" begin
    # -----------------------------------------------------------------------
    # Part A: does Julia's OWN self-rendered `coef_names` already match
    # base-R spelling, exactly, in order? One @testset per construct.
    # -----------------------------------------------------------------------
    for c in _CONSTRUCTS
        @testset "row $(c.id) own-rendering: $(c.formula)" begin
            status, result = _try_bridge(; formula = c.formula)
            expected_full = ["mu_" * n for n in c.expected]
            observed = status === :ok ? _dpar_block(result["coef_names"], "mu") :
                ["REJECTED: $(result)"]
            @test observed == expected_full
        end
    end

    # -----------------------------------------------------------------------
    # Part B: is a `coef_labels` payload (design 258 §7.1's per-dpar base-R
    # name list) honored VERBATIM when present? Forwarded via
    # `options["coef_labels"]` — the only extensibility point `drm_bridge`
    # currently exposes, since no dedicated kwarg exists (brief §1/§6;
    # `git grep coef_labels -- '*.jl'` on origin/main is empty). One
    # combined @testset, one @test per construct inside it, per the
    # coordinator's 2026-09-02 update (drmTMB
    # claude/rev-parity-c2-label-producer @ af1790492).
    # -----------------------------------------------------------------------
    @testset "coef_labels payload forwarded verbatim (design 258 §7)" begin
        for c in _CONSTRUCTS
            expected_full = ["mu_" * n for n in c.expected]
            # #563's fail-closed contract aborts when a dpar with fixed-effect
            # columns has no supplied labels; every construct here also gets
            # an auto-defaulted intercept-only `sigma` (see Part A's comment
            # above), so a real payload names both dpars, not just `mu`.
            options = Dict{String,Any}("coef_labels" => Dict(
                "mu" => c.expected, "sigma" => ["(Intercept)"]))
            status, result = _try_bridge(; formula = c.formula, options = options)
            observed = status === :ok ? _dpar_block(result["coef_names"], "mu") :
                ["REJECTED: $(result)"]
            @test observed == expected_full
        end
    end
end

println("BRIDGE_BASE_R_NAMES_DONE")
