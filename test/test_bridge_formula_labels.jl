# Public R-formula coefficient labels for the Julia bridge.  The numerical
# design remains Julia-owned; this test only exercises the exported name and
# public-selector contract.
using DRModels
using Test
using Statistics
using LinearAlgebra: I
using StatsModels: FormulaTerm, Term, apply_schema, coefnames, schema
import StatsAPI: StatisticalModel

@testset "bridge formula-derived coefficient labels" begin
    n = 96
    x = collect(range(-1.4, 1.4; length = n))
    z = [sin(0.41 * i) + 0.03 * cos(0.17 * i) for i in 1:n]
    g = [isodd(i) ? "a" : "b" for i in 1:n]
    y = 0.35 .+ 0.28 .* x .- 0.11 .* x.^2 .+
        0.22 .* (g .== "b") .+ 0.07 .* sin.(3 .* x) .+ 0.04 .* z
    dat = (; y, x, z, g)
    formula = "y ~ x + I(x^2) + g + x:g; sigma ~ scale(x) + poly(z, 2)"

    out = drm_bridge(; formula, family = "gaussian", data = dat)

    # Versioned metadata makes this a bridge contract, not an accidental
    # post-hoc display rewrite.  The raw vector remains available solely to
    # route the public fixed-effect selector through a fresh Julia refit.
    @test out["coef_label_contract"] == "bridge_formula_labels_v1"
    @test haskey(out, "raw_coef_names")
    @test haskey(out, "coef_name_map")
    @test out["coef_names"] == out["vcov_names"]
    @test length(out["coef_names"]) == length(out["raw_coef_names"])
    @test length(unique(out["coef_names"])) == length(out["coef_names"])
    @test length(unique(out["raw_coef_names"])) == length(out["raw_coef_names"])
    @test size(out["vcov"]) == (length(out["coef_names"]), length(out["coef_names"]))

    # Native R model.matrix spellings, including categorical treatment coding
    # and the interaction.  Synthetic Julia materialization names must not
    # leak into the public coefficient/covariance surface.
    expected = [
        "mu_(Intercept)", "mu_x", "mu_I(x^2)", "mu_gb", "mu_x:gb",
        "sigma_(Intercept)", "sigma_scale(x)",
        "sigma_poly(z, 2)1", "sigma_poly(z, 2)2",
    ]
    @test out["coef_names"] == expected
    @test all(!occursin("__bridge_", name) for name in out["coef_names"])

    # The reverse mapping is exact and keyed by the public full coefficient
    # name; its values are the actual returned raw names, never guessed from a
    # synthetic column ordinal.
    mapping = out["coef_name_map"]
    @test Set(keys(mapping)) == Set(out["coef_names"])
    @test Set(values(mapping)) == Set(out["raw_coef_names"])
    @test startswith(mapping["mu_I(x^2)"], "mu___bridge_I_")
    @test startswith(mapping["sigma_scale(x)"], "sigma___bridge_scale_")
    @test startswith(mapping["sigma_poly(z, 2)1"], "sigma___bridge_poly2c1_")
    @test startswith(mapping["sigma_poly(z, 2)2"], "sigma___bridge_poly2c2_")

    # Public bridge inference accepts the same native selector exposed in the
    # coefficient table, resolves it unambiguously to the raw refit name, and
    # returns the public term rather than leaking that raw selector back out.
    inferred = drm_bridge_inference(
        formula = formula, family = "gaussian", data = dat,
        method = "profile", level = 0.80, parm = "fixef:mu:I(x^2)",
    )
    @test inferred["param"] == "mu"
    @test inferred["coef"] == "I(x^2)"

    # CumulativeLogit retains the original mean formula for prediction, but
    # its fitted location block deliberately removes the intercept because the
    # cutpoints identify location.  Formula-label export must apply that known
    # family-specific projection and leave the raw cutpoint coordinates alone.
    ordinal_x = collect(range(-1.2, 1.2; length = n))
    ordinal_y = Float64[1 + mod(i + (i ÷ 7), 3) for i in 1:n]
    ordinal = drm_bridge(
        formula = "y ~ x", family = "cumulative_logit",
        data = (; y = ordinal_y, x = ordinal_x),
    )
    ordinal_expected = ["mu_x", "cutpoints_theta1", "cutpoints_theta2"]
    @test ordinal["coef_names"] == ordinal_expected
    @test ordinal["raw_coef_names"] == ordinal_expected
    @test ordinal["vcov_names"] == ordinal_expected
    @test ordinal["coef_name_map"] == Dict(name => name for name in ordinal_expected)

    # Render interactions from typed factor components, rather than editing the
    # raw StatsModels text.  Both surviving level labels contain the characters
    # that a global `": "` or `" & "` replacement would corrupt.
    punctuated = (
        y = collect(1.0:4.0),
        a = ["a", "b: c", "a", "b: c"],
        b = ["a", "a", "d & e", "d & e"],
    )
    form, augmented, label_plan = DRModels._bridge_formula(
        "y ~ a:b; sigma ~ 1", "gaussian", punctuated; labels = true)
    rhs = Dict(form.forms)[:mu]
    typed = apply_schema(FormulaTerm(Term(:y), rhs),
                         schema(FormulaTerm(Term(:y), rhs), augmented),
                         StatisticalModel)
    @test String.(vec(coefnames(typed.rhs))) == [
        "(Intercept)", "a: a & b: a", "a: b: c & b: a",
        "a: a & b: d & e", "a: b: c & b: d & e",
    ]
    @test DRModels._bridge_public_term_labels(typed.rhs, label_plan.atoms) ==
          ["(Intercept)", "aa:ba", "ab: c:ba", "aa:bd & e", "ab: c:bd & e"]

    # Interaction labels follow formula-wide first appearance, so an otherwise
    # local `x:g` writes `ga:x` when `g` appeared earlier as a main effect.
    ordered = (; y = collect(1.0:6.0), x = collect(1.0:6.0),
               g = repeat(["a", "b", "c"], 2))
    orderedform, orderedaug, orderedplan = DRModels._bridge_formula(
        "y ~ 0 + g + x:g; sigma ~ 1", "gaussian", ordered; labels = true)
    orderedrhs = Dict(orderedform.forms)[:mu]
    orderedtyped = apply_schema(FormulaTerm(Term(:y), orderedrhs),
                                schema(FormulaTerm(Term(:y), orderedrhs), orderedaug),
                                StatisticalModel)
    @test DRModels._bridge_public_term_labels(
        orderedtyped.rhs, orderedplan.atoms,
        DRModels._bridge_formula_symbol_order(orderedrhs)) ==
        ["ga", "gb", "gc", "ga:x", "gb:x", "gc:x"]

    # Keep the raw Kronecker enumeration for two width-two factors, but write
    # each interaction tuple in first-appearance order.  Pairing each public
    # label with its raw column catches a silent covariance/estimate relabel.
    crossdata = (; y = collect(1.0:9.0),
                 a = repeat(["a", "b", "c"], inner = 3),
                 b = repeat(["a", "b", "c"], 3))
    crossform, crossaug, crossplan = DRModels._bridge_formula(
        "y ~ a + b + b:a; sigma ~ 1", "gaussian", crossdata; labels = true)
    crossrhs = Dict(crossform.forms)[:mu]
    crosstyped = apply_schema(FormulaTerm(Term(:y), crossrhs),
                              schema(FormulaTerm(Term(:y), crossrhs), crossaug),
                              StatisticalModel)
    crossraw = String.(vec(coefnames(crosstyped.rhs)))
    crosspublic = DRModels._bridge_public_term_labels(
        crosstyped.rhs, crossplan.atoms,
        DRModels._bridge_formula_symbol_order(crossrhs))
    @test crossraw[end-3:end] == [
        "b: b & a: b", "b: c & a: b", "b: b & a: c", "b: c & a: c",
    ]
    @test crosspublic[end-3:end] == ["ab:bb", "ab:bc", "ac:bb", "ac:bc"]

    # Repeated materialisation is one exact source term, even across dpars.
    _, _, repeated = DRModels._bridge_formula(
        "y ~ scale(x); sigma ~ scale(x)", "gaussian", dat; labels = true)
    @test length(repeated.atoms) == 1
    @test only(values(repeated.atoms)) == "scale(x)"

    # The label renderer carries original `I(...)` spelling provenance through
    # parsing, so a public selector never loses an explicit group or unary +.
    # The spellings match R's deparsed model-matrix labels.
    intrinsic = Dict(
        "x^2" => "x^2",
        "x*z" => "x * z",
        "(x+z)/2" => "(x + z)/2",
        "x/z" => "x/z",
        "x-(z+2)" => "x - (z + 2)",
        "(x/z)^2" => "(x/z)^2",
        "x^(z+2)" => "x^(z + 2)",
        "-x^2" => "-x^2",
        "(-x)^2" => "(-x)^2",
        "x/(z/2)" => "x/(z/2)",
        "x+z+2" => "x + z + 2",
        "x*z*2" => "x * z * 2",
        "+x" => "+x",
        "x+(z+2)" => "x + (z + 2)",
        "x^2.0" => "x^2",
        "x*(z/2)" => "x * (z/2)",
        # R deparse switches at these decimal/scientific boundaries.  Preserve
        # its rendered numeric literal even when Julia's `%g` would choose the
        # shorter-looking but public-selector-incompatible alternative.
        "10000" => "10000",
        "100000" => "1e+05",
        "1e-3" => "0.001",
        "1e-4" => "1e-04",
    )
    label_data = (; y = fill(1.0, 4), x = fill(2.0, 4), z = fill(1.0, 4))
    for (input, expected_label) in intrinsic
        _, _, source_plan = DRModels._bridge_formula(
            "y ~ I($input); sigma ~ 1", "gaussian", label_data; labels = true)
        @test only(values(source_plan.atoms)) == "I($expected_label)"
    end

    # R 4.6.0 generated this retained grid through `model.matrix()` under
    # scipen = 0.  It exercises signs, decimal/scientific ties, mixed
    # mantissas and the finite Float64 endpoints.  Both the I() literal and a
    # numeric factor level must use the same independently generated spelling.
    numeric_fixture = joinpath(@__DIR__, "..", "docs", "dev-log", "evidence",
                               "julia-r-parity", "coefficient-labels",
                               "native-numeric-labels.tsv")
    numeric_rows = readlines(numeric_fixture)
    @test first(numeric_rows) == "input\tdeparse\tfactor"
    for row in numeric_rows[2:end]
        input, deparse, factor_label = split(row, '\t')
        value = parse(Float64, input)
        @test DRModels._bridge_r_number_label(input) == deparse
        @test DRModels._bridge_r_factor_level_label(value) == factor_label
    end

    # Existing bridge admission allows scalar functions around an I() term.
    # Label rendering must recurse through the typed FunctionTerm and preserve
    # the exact materialised atom provenance rather than rejecting or leaking
    # the private generated name.
    nestedform, nestedaug, nestedplan = DRModels._bridge_formula(
        "y ~ log1p(1 + I(x^2)); sigma ~ 1", "gaussian", label_data; labels = true)
    nestedrhs = Dict(nestedform.forms)[:mu]
    nestedraw, nestedpublic = DRModels._bridge_render_formula_block(
        nestedform, :mu, nestedrhs, nestedplan)
    @test nestedraw == ["(Intercept)", "log1p(1 + __bridge_I_1)"]
    @test nestedpublic == ["(Intercept)", "log1p(1 + I(x^2))"]

    # Retained R 4.6.0 model-matrix fixture for scalar functions enclosing one
    # or more I() atoms.  It locks both the exported source spelling (including
    # semantically redundant parentheses) and the evaluated design column.
    nested_fixture = joinpath(@__DIR__, "..", "docs", "dev-log", "evidence",
                              "julia-r-parity", "coefficient-labels",
                              "native-nested-labels.tsv")
    nested_rows = readlines(nested_fixture)
    @test first(nested_rows) == "expression\texpected\tv1\tv2\tv3\tv4\tv5\tv6\tv7\tv8"
    nested_data = (; y = collect(1.0:8.0),
                   x = collect(range(0.2, 1.6; length = 8)),
                   z = collect(range(-0.7, 0.7; length = 8)))
    for row in nested_rows[2:end]
        fields = split(row, '\t')
        expression, expected = fields[1:2]
        expected_values = parse.(Float64, fields[3:end])
        nested_form, nested_augmented, nested_labels = DRModels._bridge_formula(
            "y ~ $expression; sigma ~ 1", "gaussian", nested_data; labels = true)
        nested_rhs = Dict(nested_form.forms)[:mu]
        raw_names, public_names = DRModels._bridge_render_formula_block(
            nested_form, :mu, nested_rhs, nested_labels)
        _, nested_X, nested_design_names = DRModels._design(:y, nested_rhs, nested_augmented)
        @test raw_names[2] == nested_design_names[2]
        @test public_names[2] == expected
        @test nested_X[:, 2] ≈ expected_values atol = 1e-12
    end

    # A separate R fixture covers scalar source expressions with scale(), mixed
    # scale/I atoms, and no materialisation at all.  Exact labels must retain
    # source parentheses even when the Julia AST regards them as redundant.
    scalar_fixture = joinpath(@__DIR__, "..", "docs", "dev-log", "evidence",
                              "julia-r-parity", "coefficient-labels",
                              "native-scalar-labels.tsv")
    scalar_rows = readlines(scalar_fixture)
    @test first(scalar_rows) == "expression\texpected\tv1\tv2\tv3\tv4\tv5\tv6\tv7\tv8"
    for row in scalar_rows[2:end]
        fields = split(row, '\t')
        expression, expected = fields[1:2]
        expected_values = parse.(Float64, fields[3:end])
        scalar_form, scalar_augmented, scalar_labels = DRModels._bridge_formula(
            "y ~ $expression; sigma ~ 1", "gaussian", nested_data; labels = true)
        scalar_rhs = Dict(scalar_form.forms)[:mu]
        raw_names, public_names = DRModels._bridge_render_formula_block(
            scalar_form, :mu, scalar_rhs, scalar_labels)
        _, scalar_X, scalar_design_names = DRModels._design(:y, scalar_rhs, scalar_augmented)
        @test raw_names[2] == scalar_design_names[2]
        @test public_names[2] == expected
        @test scalar_X[:, 2] ≈ expected_values atol = 1e-12
    end

    # Generic scalar calls remain admitted. Source provenance must not narrow
    # comparisons or comma-separated arguments merely because their label is
    # now canonicalized. R's model matrix keeps the enclosing spelling.
    scalar_logic = (
        x = collect(range(-1.5, 1.5; length = 8)),
        y = collect(1.0:8.0),
    )
    for (expression, expected, values) in (
        ("ifelse(x > 0, x, 0)", "ifelse(x > 0, x, 0)",
         ifelse.(scalar_logic.x .> 0, scalar_logic.x, 0.0)),
        ("ifelse(x > 0, I(x^2), scale(x))", "ifelse(x > 0, I(x^2), scale(x))",
         ifelse.(scalar_logic.x .> 0, scalar_logic.x .^ 2,
                 (scalar_logic.x .- Statistics.mean(scalar_logic.x)) ./ Statistics.std(scalar_logic.x))),
    )
        logic_form, logic_augmented, logic_labels = DRModels._bridge_formula(
            "y ~ $expression; sigma ~ 1", "gaussian", scalar_logic; labels = true)
        logic_rhs = Dict(logic_form.forms)[:mu]
        _, logic_public = DRModels._bridge_render_formula_block(
            logic_form, :mu, logic_rhs, logic_labels)
        _, logic_X, _ = DRModels._design(:y, logic_rhs, logic_augmented)
        @test logic_public[2] == expected
        @test logic_X[:, 2] ≈ values atol = 1e-12
    end

    # Retained R 4.6.0 source labels and values for comparisons inside the
    # same admitted `ifelse()` class. This keeps the new lexer honest about
    # <=, !=, and parenthesized boolean conjunctions.
    conditional_fixture = joinpath(@__DIR__, "..", "docs", "dev-log", "evidence",
                                   "julia-r-parity", "coefficient-labels",
                                   "native-conditional-labels.tsv")
    conditional_rows = readlines(conditional_fixture)
    @test first(conditional_rows) == "expression\texpected\tv1\tv2\tv3\tv4\tv5\tv6\tv7\tv8"
    for row in conditional_rows[2:end]
        fields = split(row, '\t')
        expression, expected = fields[1:2]
        expected_values = parse.(Float64, fields[3:end])
        conditional_form, conditional_augmented, conditional_labels = DRModels._bridge_formula(
            "y ~ $expression; sigma ~ 1", "gaussian", nested_data; labels = true)
        conditional_rhs = Dict(conditional_form.forms)[:mu]
        _, conditional_public = DRModels._bridge_render_formula_block(
            conditional_form, :mu, conditional_rhs, conditional_labels)
        _, conditional_X, _ = DRModels._design(:y, conditional_rhs, conditional_augmented)
        @test conditional_public[2] == expected
        @test conditional_X[:, 2] ≈ expected_values atol = 1e-12
    end

    # Repeated expressions have one provenance scope per formula part.  The
    # exact public spellings below are both valid selectors because each dpar
    # owns an independent design column; a different spelling repeated within
    # one RHS remains an ambiguity rather than a silently collinear alias.
    scopedform, _, scopedplan = DRModels._bridge_formula(
        "y ~ I(x^2); sigma ~ I((x^2))", "gaussian", label_data; labels = true)
    @test Set(values(scopedplan.atoms)) == Set(["I(x^2)", "I((x^2))"])
    @test length(scopedplan.atoms) == 2
    @test_throws ArgumentError DRModels._bridge_formula(
        "y ~ I(x^2) + I((x^2)); sigma ~ 1", "gaussian", label_data; labels = true)

    # Distinct outer source spellings can share the same inner I() spelling
    # across dpars.  Their atoms must still be scoped apart so the translated
    # FunctionTerm keys cannot overwrite `mu` provenance with `sigma`.
    outer_scoped, _, outer_plan = DRModels._bridge_formula(
        "y ~ log1p(I(x^2)); sigma ~ log1p((I(x^2)))", "gaussian", label_data;
        labels = true)
    @test length(outer_plan.atoms) == 2
    @test Set(value for part in values(outer_plan.function_labels)
                   for value in values(part)) ==
          Set(["log1p(I(x^2))", "log1p((I(x^2)))"])
    outer_forms = Dict(outer_scoped.forms)
    _, outer_mu_public = DRModels._bridge_render_formula_block(
        outer_scoped, :mu, outer_forms[:mu], outer_plan)
    _, outer_sigma_public = DRModels._bridge_render_formula_block(
        outer_scoped, :sigma, outer_forms[:sigma], outer_plan)
    @test outer_mu_public[2] == "log1p(I(x^2))"
    @test outer_sigma_public[2] == "log1p((I(x^2)))"

    scale_scoped, _, scale_plan = DRModels._bridge_formula(
        "y ~ exp(scale(x)); sigma ~ exp((scale(x)))", "gaussian", nested_data;
        labels = true)
    @test length(scale_plan.atoms) == 2
    scale_forms = Dict(scale_scoped.forms)
    _, scale_mu_public = DRModels._bridge_render_formula_block(
        scale_scoped, :mu, scale_forms[:mu], scale_plan)
    _, scale_sigma_public = DRModels._bridge_render_formula_block(
        scale_scoped, :sigma, scale_forms[:sigma], scale_plan)
    @test scale_mu_public[2] == "exp(scale(x))"
    @test scale_sigma_public[2] == "exp((scale(x)))"

    # Even when no bridge column is materialised, source spelling is scoped by
    # dpar.  The AST for these two scalar calls is identical after parsing.
    plain_scoped, _, plain_plan = DRModels._bridge_formula(
        "y ~ exp(x); sigma ~ exp((x))", "gaussian", label_data; labels = true)
    @test isempty(plain_plan.atoms)
    plain_forms = Dict(plain_scoped.forms)
    _, plain_mu_public = DRModels._bridge_render_formula_block(
        plain_scoped, :mu, plain_forms[:mu], plain_plan)
    _, plain_sigma_public = DRModels._bridge_render_formula_block(
        plain_scoped, :sigma, plain_forms[:sigma], plain_plan)
    @test plain_mu_public[2] == "exp(x)"
    @test plain_sigma_public[2] == "exp((x))"

    # Typed contrast labels distinguish Boolean and numeric levels from their
    # string look-alikes.  This is deliberately not a raw-name substitution.
    booldata = (; y = collect(1.0:4.0), g = Bool[false, true, false, true])
    boolform, boolaug, boolplan = DRModels._bridge_formula(
        "y ~ factor(g); sigma ~ 1", "gaussian", booldata; labels = true)
    boolrhs = Dict(boolform.forms)[:mu]
    booltyped = apply_schema(FormulaTerm(Term(:y), boolrhs),
                             schema(FormulaTerm(Term(:y), boolrhs), boolaug),
                             StatisticalModel)
    @test DRModels._bridge_public_term_labels(booltyped.rhs, boolplan.atoms) ==
          ["(Intercept)", "factor(g)TRUE"]

    for (high, rendered) in ((2.0, "2"), (10000.0, "10000"),
                             (100000.0, "1e+05"), (1e-3, "0.001"),
                             (1e-4, "1e-04"), (2e20, "2e+20"),
                             (1e-7, "1e-07"), (1.234567890123456, "1.23456789012346"))
        floatdata = (; y = collect(1.0:4.0), g = [0.0, high, 0.0, high])
        floatform, floataug, floatplan = DRModels._bridge_formula(
            "y ~ factor(g); sigma ~ 1", "gaussian", floatdata; labels = true)
        floatrhs = Dict(floatform.forms)[:mu]
        floattyped = apply_schema(FormulaTerm(Term(:y), floatrhs),
                                  schema(FormulaTerm(Term(:y), floatrhs), floataug),
                                  StatisticalModel)
        @test DRModels._bridge_public_term_labels(floattyped.rhs, floatplan.atoms) ==
              ["(Intercept)", "factor(g)" * rendered]
    end

    # A group-level LSS formula records `sd_<group>` in `DrmFormula` but its
    # fitted coordinates are a shared `:sd` block.  The group prefix remains
    # part of the public coordinate; only the typed materialised term changes.
    lssdata = (; y = [1.0, 1.2, 1.5, 1.7], x = [0.2, 0.2, 0.8, 0.8],
               g = ["a", "a", "b", "b"])
    lssform, _, lssplan = DRModels._bridge_formula(
        "y ~ 1 + (1 | g); sigma ~ 1; sd(g) ~ I(x^2)", "gaussian", lssdata;
        labels = true)
    lssrhs = only(last(p) for p in lssform.forms if first(p) == :sd_g)
    lssgroupdesign, lssgroupnames = DRModels._sd_group_design(
        :y, lssrhs, lssplan.data, [1, 1, 2, 2], 2, :g)
    @test lssgroupnames == ["(Intercept)", "__bridge_I_1"]
    @test lssgroupdesign[:, 2] ≈ [0.04, 0.64] atol = 1e-14
    lssraw, _ = DRModels._bridge_render_formula_block(lssform, :sd, lssrhs, lssplan)
    @test lssraw == ["(Intercept)", "__bridge_I_1"]
    lsspnames = ["g: " * name for name in lssraw]
    lssmap = Dict("sd_" * name => "sd_" * name for name in lsspnames)
    DRModels._bridge_lss_public_to_raw!(lssmap, lssform, Dict(:sd => lsspnames), lssplan)
    @test lssmap["sd_g: __bridge_I_1"] == "sd_g: I(x^2)"

    # A single `sd(g) ~ g` formula has ordinary raw categorical names starting
    # with `g: `; it is not a multi-component group prefix.
    lssfactor, _, lssfactorplan = DRModels._bridge_formula(
        "y ~ 1 + (1 | g); sigma ~ 1; sd(g) ~ g", "gaussian", lssdata;
        labels = true)
    lssfactorrhs = only(last(p) for p in lssfactor.forms if first(p) == :sd_g)
    factorraw, factorpublic = DRModels._bridge_render_formula_block(
        lssfactor, :sd, lssfactorrhs, lssfactorplan)
    lssfactormap = Dict("sd_" * name => "sd_" * name for name in factorraw)
    DRModels._bridge_lss_public_to_raw!(lssfactormap, lssfactor,
                                   Dict(:sd => factorraw), lssfactorplan)
    @test [lssfactormap["sd_" * name] for name in factorraw] ==
          ["sd_" * name for name in factorpublic]

    # Re-rendering schemas must follow `_design` exactly: missing/NaN response
    # values become placeholders only in the temporary schema table.
    missingdata = (; y = Any[1.0, missing, NaN, 2.0], x = [1.0, 2.0, 3.0, 4.0])
    schema_data = DRModels._bridge_label_schema_data(:y, missingdata)
    @test schema_data.y == [1.0, 0.0, 0.0, 2.0]
    @test ismissing(missingdata.y[2])
    @test isnan(missingdata.y[3])

    # Coordinate axes are a strict partition; neither omitted/duplicated
    # coefficients nor a covariance with a different coordinate shape cross
    # the primitive bridge boundary.
    blocks = Pair{Symbol,UnitRange{Int}}[:mu => 1:1, :sigma => 2:2]
    names = Pair{Symbol,Vector{String}}[:mu => ["x"], :sigma => ["(Intercept)"]]
    @test isnothing(DRModels._bridge_validate_coordinate_axes(blocks, names, 2, Matrix{Float64}(I, 2, 2)))
    @test_throws ErrorException DRModels._bridge_validate_coordinate_axes(blocks, names, 2, ones(1, 1))
    @test_throws ErrorException DRModels._bridge_validate_coordinate_axes(
        Pair{Symbol,UnitRange{Int}}[:mu => 1:1, :sigma => 1:1], names, 2, nothing)
    @test_throws ErrorException DRModels._bridge_validate_coordinate_axes(
        Pair{Symbol,UnitRange{Int}}[:sigma => 2:2, :mu => 1:1],
        Pair{Symbol,Vector{String}}[:sigma => ["(Intercept)"], :mu => ["x"]],
        2, [1.0 0.0; 0.0 9.0])
end

println("BRIDGE_FORMULA_LABELS_PASS")
