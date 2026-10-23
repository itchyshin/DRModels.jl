# R→Julia formula translation across the drm_bridge string boundary (Ayumi LS#2,
# then #467).
#
# The R side sends model formulas as strings; R writes interactions with `:`,
# which Julia parses as the (lower-precedence) range operator, mis-associating
# the `+` chain and pulling a trailing phylo()/(1|g) term inside a Colon term the
# engine cannot read. The bridge rewrites `:` → `&` (string level).
#
# Six more R constructs `@formula` cannot evaluate as the R user intends:
# `scale()`, `I()`, `factor()`, `(...)^k` crossing, and general `- term`
# removal are now translated into faithful `@formula`-native equivalents
# (materialised columns for scale/I/factor; expanded `+`/`&` terms for `^`/`-`)
# — see #467 and `_bridge_xlate` in src/bridge.jl. R-parity fixtures proving
# byte-identical numeric agreement with drmTMB live under
# test/parity/fixtures/bridge-*/ (gated, `DRM_PARITY_TESTS=1`); this file
# checks the translation's STRUCTURE: what fits, what still throws, and that a
# rejection is caught at ANY nesting depth. `poly()` LANDED 2026-08-25 (#492):
# R's `raw = FALSE` orthogonal basis is deterministic and transcribes to 9.99e-16,
# so the old blanket rejection was replaced by a narrow one. What stays rejected is
# what genuinely cannot be reproduced by this rewrite — see the `poly()` testsets.
using DRModels
using Test

@testset "bridge: R formula operator translation" begin
    p = 16; m = 5
    phy = random_balanced_tree(p; branch_length = 0.20)
    species = ["sp$(i)" for i in repeat(1:p, inner = m)]
    n = length(species)
    rng_x1 = Float64[sin(i) for i in 1:n]          # deterministic, no RNG dependence
    rng_x2 = Float64[cos(2i) for i in 1:n]
    rng_x3 = Float64[sin(3i + 1) for i in 1:n]
    noise  = Float64[0.4 * sin(7.1i + 2.3) for i in 1:n]   # deterministic pseudo-noise
    y = 0.3 .+ 0.5 .* rng_x1 .- 0.2 .* rng_x2 .+ 0.1 .* (rng_x1 .* rng_x2) .+ noise
    grp3 = ["a", "b", "c"][mod1.(1:n, 3)]
    dat = (; y, x1 = rng_x1, x2 = rng_x2, x3 = rng_x3, species, grp3)

    fitok(f) = isfinite(drm_bridge(; formula = f, family = "gaussian",
                                   data = dat, tree = phy)["loglik"])
    ncoef(f) = length(drm_bridge(; formula = f, family = "gaussian",
                                 data = dat, tree = phy)["coef_names"])
    rejects(f) = (@test_throws ArgumentError drm_bridge(; formula = f, family = "gaussian",
                                                         data = dat, tree = phy))

    @testset "valid formulas still fit" begin
        @test fitok("y ~ x1; sigma ~ 1")
        @test fitok("y ~ x1 + phylo(1 | species); sigma ~ 1")
        @test fitok("y ~ x1 + phylo(1 | species); sigma ~ phylo(1 | species)")
    end

    @testset "univariate phylo random slope: the two-SD form fits, the rest reject" begin
        # Silent-data-loss fix: a `phylo(<not 1> | group)` marker translated
        # from an R formula string must hit the same univariate-route refusal
        # as the native Julia path, not silently fit the intercept-only model.
        # #620 landed the ONE admitted slope form on the Gaussian mean, so it
        # now fits across the bridge as well -- and must expose BOTH field SDs,
        # never the intercept-only fit this testset was written to catch.
        two_sd = drm_bridge(; formula = "y ~ x1 + phylo(1 + x1 | species); sigma ~ 1",
                            family = "gaussian", data = dat, tree = phy)
        @test isfinite(two_sd["loglik"])
        @test two_sd["coef_names"] ==
            ["mu_(Intercept)", "mu_x1", "sigma_(Intercept)", "resd_species", "resd_species:x1"]
        # every other slope shape still refuses
        rejects("y ~ x1 + phylo(0 + x1 | species); sigma ~ 1")
        rejects("y ~ x1 + phylo(1 + x1 + x2 | species); sigma ~ 1")
        rejects("y ~ x1; sigma ~ phylo(1 + x1 | species)")
    end

    @testset "R `:` interaction + phylo no longer crashes" begin
        # The exact shape of Ayumi's MethodError |(::Int64, ::String).
        @test fitok("y ~ x1 + x2 + x1:x2 + phylo(1 | species); sigma ~ 1")
        @test fitok("y ~ x1 + x1:x2 + phylo(1 | species); sigma ~ x1 + x1:x2")
        @test fitok("y ~ x1 + x2 + x3 + x1:x2:x3; sigma ~ 1")   # 3-way
    end

    @testset "R `*` crossing: valid fits, nested bans still caught" begin
        # `*` (crossing = main effects + interaction) is a valid Julia formula op.
        @test fitok("y ~ x1 * x2 + phylo(1 | species); sigma ~ 1")
        # I()/scale()/poly() are all LANDED now, so the exemplar for "a ban nested
        # at any depth is still caught" is a construct that remains unsupported —
        # and it must NOT leak a raw Julia error.
        for f in ("y ~ log1p(poly(x1, 2)); sigma ~ 1",            # poly under a scalar fn
                  "y ~ (x1 + poly(x1, 2))^2; sigma ~ 1",          # poly under `^`
                  "y ~ I(log(x1)); sigma ~ 1")                    # outside I()'s grammar
            rejects(f)
        end
    end

    @testset "poly() is landed, and its shape guards are narrow (#492)" begin
        # R's orthogonal (`raw = FALSE`) basis is deterministic; the transcription
        # matches `stats::poly(x, 3)` to 9.99e-16. Numeric agreement with drmTMB is
        # in test/parity/fixtures/bridge-poly*/ (gated); here: what fits, what does not.
        @test fitok("y ~ poly(x1, 3); sigma ~ 1")
        @test fitok("y ~ poly(x1, 2) + x2; sigma ~ 1")
        @test fitok("y ~ x2 * poly(x1, 2); sigma ~ 1")     # `*` matches R (fixture: bridge-poly-cross)
        @test fitok("y ~ x2 & poly(x1, 2); sigma ~ 1")     # R `:` — 3 columns both sides

        # k columns, not one: poly(x1, 3) must add exactly 3 mu coefficients.
        base = drm_bridge(; formula = "y ~ 1; sigma ~ 1", family = "gaussian", data = dat, tree = phy)
        p3   = drm_bridge(; formula = "y ~ poly(x1, 3); sigma ~ 1", family = "gaussian", data = dat, tree = phy)
        nmu(o) = count(k -> startswith(k, "mu_"), keys(o["coef"]))
        @test nmu(p3) == nmu(base) + 3

        # `(...)^k`: MEASURED mismatch, not a guess. R treats poly as ONE term, so
        # `(x + poly(x, 2))^2` gives 6 model-matrix columns; flattening poly into two
        # terms yields 7, the extra being poly1 & poly2.
        try
            drm_bridge(; formula = "y ~ (x1 + poly(x1, 2))^2; sigma ~ 1", family = "gaussian",
                       data = dat, tree = phy)
            @test false
        catch e
            @test e isa ArgumentError
            @test occursin("ONE term", sprint(showerror, e))
        end

        # A scalar function applied to a k-column basis: R maps it over the matrix
        # (k columns out); this rewrite would map it over their SUM (one column).
        try
            drm_bridge(; formula = "y ~ log1p(poly(x1, 2)); sigma ~ 1", family = "gaussian",
                       data = dat, tree = phy)
            @test false
        catch e
            @test e isa ArgumentError
            @test occursin("may only appear as a model term", sprint(showerror, e))
        end

        # Forms that cannot be reproduced, each named specifically rather than
        # swept under one blanket refusal.
        rejects("y ~ poly(x1, 2, raw = TRUE); sigma ~ 1")   # raw powers: use I(x^k)
        rejects("y ~ poly(x1 + x2, 2); sigma ~ 1")          # not a bare column
        rejects("y ~ poly(x1, 0); sigma ~ 1")               # degree must be >= 1
    end

    @testset "scale()/I()/factor() are now landed" begin
        @test fitok("y ~ scale(x1); sigma ~ 1")
        @test fitok("y ~ x1 + I(x1^2); sigma ~ 1")
        @test fitok("y ~ factor(grp3); sigma ~ 1")
        # Nested under `*` / a transform: previously banned, now must fit.
        @test fitok("y ~ x1 * I(x1^2); sigma ~ 1")
        @test fitok("y ~ x1 * scale(x2); sigma ~ 1")
        @test fitok("y ~ log1p(1 + I(x1^2)); sigma ~ 1")

        # Coefficient counts are naming-agnostic evidence the translation did
        # what it claims: scale(x1) contributes exactly one mu coefficient
        # (like a plain covariate); I(x1^2) adds one column beyond x1 itself;
        # factor(grp3) (3 levels) contributes 2 dummy coefficients beyond the
        # intercept, exactly like the bare column would if it were treated
        # categorically.
        base = ncoef("y ~ x1; sigma ~ 1")
        @test ncoef("y ~ scale(x1); sigma ~ 1") == base
        @test ncoef("y ~ x1 + I(x1^2); sigma ~ 1") == base + 1
        @test ncoef("y ~ factor(grp3); sigma ~ 1") == base + 1   # intercept + 2 dummies vs intercept + x1
    end

    @testset "(...)^k crossing expands to main effects + interactions" begin
        @test fitok("y ~ (x1 + x2)^2; sigma ~ 1")
        # (x1+x2)^2 = x1 + x2 + x1&x2: two main effects + one interaction,
        # i.e. exactly one more coefficient than plain `*` crossing gives.
        @test ncoef("y ~ (x1 + x2)^2; sigma ~ 1") == ncoef("y ~ x1 * x2; sigma ~ 1")
        # (x1+x2+x3)^2 = 3 mains + 3 pairwise interactions (no 3-way, since k=2).
        @test ncoef("y ~ (x1 + x2 + x3)^2; sigma ~ 1") ==
              1 + 3 + 3 + 1   # mu: intercept + mains + pairwise, plus sigma's intercept
    end

    @testset "general `- term` removal (mechanical)" begin
        with_x2 = ncoef("y ~ x1 + x2; sigma ~ 1")
        @test ncoef("y ~ x1 + x2 - x2; sigma ~ 1") == with_x2 - 1
        @test fitok("y ~ x1 + x2 - x2; sigma ~ 1")   # equivalent to `y ~ x1`
        # Chained removal + trailing `-1` (no intercept).
        @test ncoef("y ~ x1 + x2 - x2 - 1; sigma ~ 1") == with_x2 - 2
        # Removing a term that was never present is a silent no-op, matching
        # R's own `terms()` behaviour for `y ~ x1 + x2 - x3`.
        @test ncoef("y ~ x1 + x2 - x3; sigma ~ 1") == with_x2
        # `^`-expanded interaction can be removed by name afterwards.
        @test ncoef("y ~ (x1 + x2)^2 - x1&x2; sigma ~ 1") == ncoef("y ~ x1 + x2; sigma ~ 1")
    end

    @testset "`-` term removal combined with unexpanded `*` is refused, not guessed" begin
        # `(x1+x2)*x3 - x1&x3` would need `*` expanded first to find `x1:x3`
        # inside it; guessing wrong here would silently mis-model, so it stays
        # a loud rejection rather than a best-effort guess.
        rejects("y ~ (x1 + x2) * x3 - x1&x3; sigma ~ 1")
        rejects("y ~ (x1 * x2)^2; sigma ~ 1")   # same risk, on the `^` side
    end

    @testset "`^` misuse still rejected" begin
        rejects("y ~ x1^0.5; sigma ~ 1")     # non-integer exponent
        rejects("y ~ x1^0; sigma ~ 1")       # k must be >= 1
    end

    @testset "scale()/factor()/I() argument-shape guards" begin
        rejects("y ~ scale(x1 + x2); sigma ~ 1")      # not a bare column
        rejects("y ~ factor(x1 + x2); sigma ~ 1")     # not a bare column
        rejects("y ~ I(log(x1)); sigma ~ 1")           # outside the safe +-*/^ grammar
        rejects("y ~ I(x1, x2); sigma ~ 1")            # not exactly one expression
        rejects("y ~ I(nosuchcolumn); sigma ~ 1")      # unknown column reference
    end

    @testset "R `- 1` drops the intercept" begin
        with_int = drm_bridge(; formula = "y ~ x1; sigma ~ 1", family = "gaussian", data = dat, tree = phy)
        no_int   = drm_bridge(; formula = "y ~ x1 - 1; sigma ~ 1", family = "gaussian", data = dat, tree = phy)
        # naming-agnostic: dropping the intercept removes exactly one mu coefficient.
        @test length(no_int["coef_names"]) == length(with_int["coef_names"]) - 1
    end
end
