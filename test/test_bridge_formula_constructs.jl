# Formula constructs through the bridge with R-contrast fidelity (A6 of the
# drmTMB <-> DRModels.jl parity programme; DRModels.jl #467 + the #609 factors case;
# design 258 in drmTMB).
#
# drmTMB sends `options["coef_labels"]` -- base-R `model.matrix()` column
# names per dpar -- and `_bridge_echo_coef_labels` pastes them onto the
# fitted columns POSITIONALLY. Measured through drmTMB against DRModels.jl
# 430ef64cc on 2026-09-05: `factor()`, `I(x^2)`, `poly(x, 2)`, `(x + z)^2`
# and `- term` already agree with `engine = "tmb"` name-for-name to <= 3e-11,
# but a design the two engines build DIFFERENTLY with the SAME column count
# -- a character column whose R (locale-collated) level order is not Julia's
# codepoint order (max|coef diff| 0.462), an ordered factor R codes with
# contr.poly (1.180), a contr.sum factor (1.757), or a factor whose level
# order was reversed on the Julia side only -- passed the echo and reported
# DRModels.jl's coefficients under R's names with NO error.
#
# `_bridge_check_coef_labels_fidelity` closes that: every regression block
# DRModels.jl can render itself must render to exactly the supplied base-R names,
# in order, else the fit is refused naming the dpar and BOTH spellings.
# Blocks DRModels.jl cannot render (the `raw, raw` fallback), and blocks with no
# formula counterpart, are still echoed verbatim -- the count check in
# test_bridge_coef_labels_echo.jl is unchanged.
using DRModels
using Test

const _FC_N = 60
const _FC_X = collect(range(-2.0, 2.0; length = _FC_N))
const _FC_Z = [sin(0.3 * i) + 0.05 * i for i in 1:_FC_N]
# Three levels whose Julia `sort(unique(...))` order is hi < lo < mid, so
# DRModels.jl's baseline is "hi" and its dummy columns are lo, mid. An R user
# with `factor(grp, levels = c("lo", "mid", "hi"))` gets baseline "lo" and
# columns mid, hi -- same count, different design.
const _FC_GRP = [["hi", "lo", "mid"][mod1(i, 3)] for i in 1:_FC_N]
const _FC_Y = 0.6 .+ 0.35 .* _FC_X .- 0.08 .* _FC_X .^ 2 .+ 0.25 .* _FC_Z .+
    [grp == "lo" ? 0.2 : grp == "mid" ? -0.15 : 0.0 for grp in _FC_GRP] .+
    [0.05 * sin(7.0 * i) for i in 1:_FC_N]
const _FC_Y2 = 0.2 .- 0.15 .* _FC_X .+ [0.04 * cos(5.0 * i) for i in 1:_FC_N]
# A logical covariate: R names its one treatment column `flagTRUE`; Julia
# keeps the Bool vector continuous (same 0/1 column) and must render R's name.
const _FC_FLAG = [_FC_X[i] > 0 for i in 1:_FC_N]
const _FC_DATA = (; y = _FC_Y, y2 = _FC_Y2, x = _FC_X, z = _FC_Z, grp = _FC_GRP, flag = _FC_FLAG)

function _fc_try(formula; family = "gaussian", options = Dict{String,Any}())
    try
        return (:ok, drm_bridge(; formula, family, data = _FC_DATA, options))
    catch e
        return (:error, sprint(showerror, e))
    end
end

_fc_block(names, dpar) =
    [n[nextind(n, firstindex(n), length(dpar) + 1):end] for n in names if startswith(n, dpar * "_")]

_fc_options(labels::Dict{String,Vector{String}}) =
    Dict{String,Any}("coef_labels" => Dict{String,Any}(k => v for (k, v) in labels))

# (label, formula, expected base-R names per dpar -- `sigma` defaults to an
# intercept-only block for a Gaussian fit that names no sigma formula.)
const _FC_CONSTRUCTS = [
    (label = "factor()", formula = "y ~ x + factor(grp)",
     expected = Dict("mu" => ["(Intercept)", "x", "factor(grp)lo", "factor(grp)mid"],
                     "sigma" => ["(Intercept)"])),
    (label = "bare string column", formula = "y ~ x + grp",
     expected = Dict("mu" => ["(Intercept)", "x", "grplo", "grpmid"],
                     "sigma" => ["(Intercept)"])),
    (label = "factor interaction", formula = "y ~ x * grp",
     expected = Dict("mu" => ["(Intercept)", "x", "grplo", "grpmid", "x:grplo", "x:grpmid"],
                     "sigma" => ["(Intercept)"])),
    (label = "I(x^2)", formula = "y ~ x + I(x^2)",
     expected = Dict("mu" => ["(Intercept)", "x", "I(x^2)"], "sigma" => ["(Intercept)"])),
    (label = "poly(x, 2)", formula = "y ~ poly(x, 2)",
     expected = Dict("mu" => ["(Intercept)", "poly(x, 2)1", "poly(x, 2)2"],
                     "sigma" => ["(Intercept)"])),
    (label = "(x + z)^2", formula = "y ~ (x + z)^2",
     expected = Dict("mu" => ["(Intercept)", "x", "z", "x:z"], "sigma" => ["(Intercept)"])),
    (label = "- term", formula = "y ~ x + z - z",
     expected = Dict("mu" => ["(Intercept)", "x"], "sigma" => ["(Intercept)"])),
    (label = "sigma-side factor", formula = "y ~ x; sigma ~ grp",
     expected = Dict("mu" => ["(Intercept)", "x"], "sigma" => ["(Intercept)", "grplo", "grpmid"])),
    (label = "logical covariate", formula = "y ~ x + flag",
     expected = Dict("mu" => ["(Intercept)", "x", "flagTRUE"], "sigma" => ["(Intercept)"])),
    (label = "logical in an interaction", formula = "y ~ x * flag",
     expected = Dict("mu" => ["(Intercept)", "x", "flagTRUE", "x:flagTRUE"], "sigma" => ["(Intercept)"])),
    (label = "logical under factor()", formula = "y ~ x + factor(flag)",
     expected = Dict("mu" => ["(Intercept)", "x", "factor(flag)TRUE"], "sigma" => ["(Intercept)"])),
]

# One disagreement per row: the SAME column count as DRModels.jl's design, a
# spelling R would produce for a design DRModels.jl did not build. `mentions`
# must all appear in the refusal so the user sees both spellings.
const _FC_DISAGREEMENTS = [
    (label = "level order (R levels lo, mid, hi -> baseline lo)",
     formula = "y ~ x + grp", dpar = "mu",
     labels = Dict("mu" => ["(Intercept)", "x", "grpmid", "grphi"], "sigma" => ["(Intercept)"]),
     mentions = ["coef_labels[\"mu\"]", "grpmid", "grphi", "grplo", "engine = \"tmb\""]),
    (label = "level order under factor() (R levels mid, hi, lo)",
     formula = "y ~ x + factor(grp)", dpar = "mu",
     labels = Dict("mu" => ["(Intercept)", "x", "factor(grp)hi", "factor(grp)lo"], "sigma" => ["(Intercept)"]),
     mentions = ["coef_labels[\"mu\"]", "factor(grp)hi", "factor(grp)mid"]),
    (label = "ordered factor (contr.poly spelling .L/.Q)",
     formula = "y ~ x + grp", dpar = "mu",
     labels = Dict("mu" => ["(Intercept)", "x", "grp.L", "grp.Q"], "sigma" => ["(Intercept)"]),
     mentions = ["coef_labels[\"mu\"]", "grp.L", "grplo"]),
    (label = "contr.sum spelling (grp1/grp2)",
     formula = "y ~ x + grp", dpar = "mu",
     labels = Dict("mu" => ["(Intercept)", "x", "grp1", "grp2"], "sigma" => ["(Intercept)"]),
     mentions = ["coef_labels[\"mu\"]", "grp1", "grplo"]),
    (label = "interaction component order (grplo:x for x:grplo)",
     formula = "y ~ x * grp", dpar = "mu",
     labels = Dict("mu" => ["(Intercept)", "x", "grplo", "grpmid", "grplo:x", "grpmid:x"],
                   "sigma" => ["(Intercept)"]),
     mentions = ["coef_labels[\"mu\"]", "grplo:x", "x:grplo"]),
    (label = "term order (z before x for y ~ x + z)",
     formula = "y ~ x + z", dpar = "mu",
     labels = Dict("mu" => ["(Intercept)", "z", "x"], "sigma" => ["(Intercept)"]),
     mentions = ["coef_labels[\"mu\"]"]),
    (label = "sigma-side level order",
     formula = "y ~ x; sigma ~ grp", dpar = "sigma",
     labels = Dict("mu" => ["(Intercept)", "x"], "sigma" => ["(Intercept)", "grpmid", "grphi"]),
     mentions = ["coef_labels[\"sigma\"]", "grpmid", "grphi", "grplo"]),
]

@testset "bridge formula constructs with R-contrast fidelity (A6, #467/#609)" begin

    @testset "(a) own rendering: base-R spelling per construct" begin
        for c in _FC_CONSTRUCTS
            @testset "$(c.label): $(c.formula)" begin
                status, out = _fc_try(c.formula)
                @test status === :ok
                status === :ok || return
                for (dpar, expected) in c.expected
                    @test _fc_block(out["coef_names"], dpar) == expected
                end
            end
        end
    end

    @testset "(b) R's spelling supplied: accepted, echoed verbatim, numbers unchanged" begin
        for c in _FC_CONSTRUCTS
            @testset "$(c.label): $(c.formula)" begin
                _, baseline = _fc_try(c.formula)
                status, out = _fc_try(c.formula; options = _fc_options(c.expected))
                @test status === :ok
                status === :ok || (println("REFUSED: ", out); return)
                for (dpar, expected) in c.expected
                    @test _fc_block(out["coef_names"], dpar) == expected
                end
                @test out["vcov_names"] == out["coef_names"]
                @test out["coef_label_contract"] == "bridge_formula_labels_v1"
                @test out["raw_coef_names"] == baseline["raw_coef_names"]
                @test out["coefficients"] ≈ baseline["coefficients"]
                @test out["vcov"] ≈ baseline["vcov"]
            end
        end
    end

    @testset "(c) a design DRModels.jl did not build is refused BY NAME, naming both spellings" begin
        for d in _FC_DISAGREEMENTS
            @testset "$(d.label)" begin
                # Same column count: the count check alone passes this.
                _, baseline = _fc_try(d.formula)
                @test length(_fc_block(baseline["coef_names"], d.dpar)) == length(d.labels[d.dpar])
                status, msg = _fc_try(d.formula; options = _fc_options(d.labels))
                @test status === :error
                status === :error || (println("NOT REFUSED: ", msg["coef_names"]); return)
                @test occursin("does not match the design DRModels.jl built", msg)
                for m in d.mentions
                    @test occursin(m, msg)
                end
                # A design disagreement, not a count complaint.
                @test !occursin("supplies", msg)
            end
        end
    end

    @testset "(d) bivariate: mu1's factor rendered, accepted when R agrees, refused when not" begin
        formula = Dict("mu1" => "y ~ x + grp", "mu2" => "y2 ~ x",
                       "sigma1" => "sigma1 ~ 1", "sigma2" => "sigma2 ~ 1", "rho12" => "rho12 ~ 1")
        agree = Dict("mu1" => ["(Intercept)", "x", "grplo", "grpmid"], "mu2" => ["(Intercept)", "x"],
                     "sigma1" => ["(Intercept)"], "sigma2" => ["(Intercept)"], "rho12" => ["(Intercept)"])
        disagree = Dict("mu1" => ["(Intercept)", "x", "grpmid", "grphi"], "mu2" => ["(Intercept)", "x"],
                        "sigma1" => ["(Intercept)"], "sigma2" => ["(Intercept)"], "rho12" => ["(Intercept)"])
        status_ok, out = _fc_try(formula; family = "biv_gaussian", options = _fc_options(agree))
        @test status_ok === :ok
        status_ok === :ok && @test _fc_block(out["coef_names"], "mu1") == agree["mu1"]
        status_bad, msg = _fc_try(formula; family = "biv_gaussian", options = _fc_options(disagree))
        @test status_bad === :error
        status_bad === :error && @test occursin("coef_labels[\"mu1\"]", msg)
        status_bad === :error && @test occursin("grphi", msg) && occursin("grplo", msg)
    end

    # The location-scale-scale `sd_<group>` block is rendered by
    # `_bridge_lss_public_to_raw!`, which `_bridge_rendered_regression_blocks`
    # skips by construction, so before `_bridge_check_lss_coef_labels_fidelity`
    # it was echoed positionally with no design comparison at all. Measured
    # through drmTMB origin/main 2fcbb0fbf against DRModels.jl aee371cc9 on
    # 2026-09-05: `sd(study) ~ <character column>` converged on both engines to
    # an identical logLik (-69.917488, diff 2.98e-13) under identical names,
    # with `mu` and `sigma` faithful to 2.1e-11 and the `sd` block off by
    # 1.3853 -- the baseline had moved. Same column COUNT, so (e) never fired.
    @testset "(f) an LSS sd(group) block DRModels.jl did not build is refused BY NAME" begin
        n = 96
        group = [string("s", mod1(i, 12)) for i in 1:n]
        # Codepoint order is "Beta" < "alpha" < "gamma", so DRModels.jl's baseline
        # is "Beta" and its columns are alpha, gamma. An R user whose locale
        # collates alpha < Beta < gamma sends baseline "alpha", columns
        # Beta, gamma -- same count, different design.
        lab = [["Beta", "alpha", "gamma"][mod1(mod1(i, 12), 3)] for i in 1:n]
        gx = [0.15 * mod1(i, 12) for i in 1:n]
        y = 0.4 .+ 0.3 .* _FC_X[mod1.(1:n, _FC_N)] .+ gx .+
            [0.05 * sin(3.0 * i) for i in 1:n]
        data = (; y, x = _FC_X[mod1.(1:n, _FC_N)], g = group, lab = lab)
        formula = "y ~ x + (1 | g); sigma ~ 1; sd(g) ~ lab"
        base = try
            drm_bridge(; formula, family = "gaussian", data)
        catch e
            @test false
            rethrow(e)
        end
        rendered = _fc_block(base["coef_names"], "sd")
        @test rendered == ["(Intercept)", "labalpha", "labgamma"]

        agree = Dict("mu" => ["(Intercept)", "x"], "sigma" => ["(Intercept)"],
                     "sd" => ["(Intercept)", "labalpha", "labgamma"])
        disagree = Dict("mu" => ["(Intercept)", "x"], "sigma" => ["(Intercept)"],
                        "sd" => ["(Intercept)", "labBeta", "labgamma"])
        # Same column count on the sd block: the count check alone passes both.
        @test length(agree["sd"]) == length(disagree["sd"]) == length(rendered)

        ok, out = try
            (:ok, drm_bridge(; formula, family = "gaussian", data,
                             options = _fc_options(agree)))
        catch e
            (:error, sprint(showerror, e))
        end
        @test ok === :ok
        if ok === :ok
            @test _fc_block(out["coef_names"], "sd") == agree["sd"]
            @test out["coefficients"] ≈ base["coefficients"]
            @test out["raw_coef_names"] == base["raw_coef_names"]
        else
            println("GREEN CONTROL REFUSED: ", out)
        end

        bad, msg = try
            (:ok, drm_bridge(; formula, family = "gaussian", data,
                             options = _fc_options(disagree)))
        catch e
            (:error, sprint(showerror, e))
        end
        @test bad === :error
        if bad === :error
            @test occursin("does not match the design DRModels.jl built", msg)
            @test occursin("sd_g", msg)
            @test occursin("labBeta", msg)
            @test occursin("labalpha", msg)
            # A design disagreement, not a count complaint.
            @test !occursin("supplies", msg)
        else
            println("NOT REFUSED: ", msg["coef_names"])
        end
    end

    @testset "(e) the count check still fires first, unchanged" begin
        status, msg = _fc_try("y ~ x + grp";
            options = _fc_options(Dict("mu" => ["(Intercept)", "x", "grplo"], "sigma" => ["(Intercept)"])))
        @test status === :error
        status === :error && @test occursin("supplies 3 names", msg)
        status === :error && @test !occursin("does not match the design", msg)
        # Too FEW names: no unused-level hint, because that cause cannot
        # produce a short supply.
        status === :error && @test !occursin("droplevels", msg)
    end

    @testset "(f) too MANY names on a coded block names the unused-level cause" begin
        # The shape drmTMB produces for a factor with a level no row uses:
        # R's `model.matrix()` sends one name per DECLARED level, DRModels.jl
        # builds one per OBSERVED level. Measured through drmTMB on
        # 2026-09-05, the bare count message named neither the column nor the
        # fix (DRModels.jl #467/#609).
        status, msg = _fc_try("y ~ x + grp";
            options = _fc_options(Dict(
                "mu" => ["(Intercept)", "x", "grplo", "grpmid", "grpunused"],
                "sigma" => ["(Intercept)"])))
        @test status === :error
        status === :error && @test occursin("supplies 5 names", msg)
        status === :error && @test occursin("droplevels", msg)
        status === :error && @test occursin("factor level with no rows", msg)
        # It names the coded columns DRModels.jl actually built.
        status === :error && @test occursin("mu_grp: lo", msg)

        # A block with no coded column gets the bare count message: the hint
        # would be wrong there.
        status2, msg2 = _fc_try("y ~ x + z";
            options = _fc_options(Dict(
                "mu" => ["(Intercept)", "x", "z", "extra"], "sigma" => ["(Intercept)"])))
        @test status2 === :error
        status2 === :error && @test occursin("supplies 4 names", msg2)
        status2 === :error && @test !occursin("droplevels", msg2)
    end
end

println("BRIDGE_FORMULA_CONSTRUCTS_DONE")
