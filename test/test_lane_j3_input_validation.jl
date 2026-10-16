# test_lane_j3_input_validation.jl — fit-entry argument and data checks.
#
# Each testset is a red control for an input that used to be accepted and then
# dropped or misread without a word (code-review issues #1027, #1026, #1022,
# #1020, #1014, #1013, #1010, #966, #1017). Every refusal must be an
# ArgumentError whose message names the offending argument.
using DRModels
using Test, Random, LinearAlgebra

_j3_msg(f) = try
    f()
    ""
catch e
    sprint(showerror, e)
end

function _j3_rpois(rng, l)
    L = exp(-l); k = 0; p = 1.0
    while true
        p *= rand(rng)
        p <= L && return k
        k += 1
    end
end

@testset "lane J3: fit-entry input validation" begin

    @testset "#1027: a formula for a parameter the family does not have is refused" begin
        rng = MersenneTwister(5)
        n = 120; x = randn(rng, n); z = randn(rng, n)
        y = exp.(0.3 .+ 0.4x .+ 0.3randn(rng, n))
        c = Float64.(rand(rng, 0:6, n))
        u = clamp.(rand(rng, n), 0.02, 0.98)
        d = (y = y, c = c, u = u, x = x, z = z)
        cases = [
            ("sigma", DRModels.Poisson(), bf(@formula(c ~ x), @formula(sigma ~ z))),
            ("nu",    DRModels.Poisson(), bf(@formula(c ~ x), @formula(nu ~ z))),
            ("nu",    NegBinomial2(),     bf(@formula(c ~ x), @formula(nu ~ z))),
            ("nu",    Gamma(),            bf(@formula(y ~ x), @formula(nu ~ z))),
            ("zi",    Gamma(),            bf(@formula(y ~ x), @formula(zi ~ z))),
            ("nu",    Beta(),             bf(@formula(u ~ x), @formula(nu ~ z))),
            ("nu",    Gaussian(),         bf(@formula(y ~ x), @formula(nu ~ z))),
            ("zi",    Student(),          bf(@formula(y ~ x), @formula(zi ~ 1))),
            ("sigma", CumulativeLogit(),  bf(@formula(c ~ x), @formula(sigma ~ z))),
        ]
        for (p, fam, fm) in cases
            @test_throws ArgumentError drm(fm, fam; data = d)
            msg = _j3_msg(() -> drm(fm, fam; data = d))
            @test occursin("has no `$p` parameter", msg)
        end
        # Parameters the family does have, and the constant sigma `bf` adds, still fit.
        @test drm(bf(@formula(c ~ x), @formula(zi ~ 1)), DRModels.Poisson();
                  data = d, se = false) isa DrmFit
        @test drm(bf(@formula(c ~ x), @formula(sigma ~ 1)), DRModels.Poisson();
                  data = d, se = false) isa DrmFit
        @test drm(bf(@formula(c ~ x), @formula(sigma ~ z)), NegBinomial2();
                  data = d, se = false) isa DrmFit
    end

    @testset "#1026: phylo labels that are not tip names are refused, not matched by order" begin
        tr = "(((a:1,b:1):1,(c:1,d:1):1):1,(e:1,f:1):2);"
        sp = ["a", "b", "c", "d", "e", "f"]
        g = repeat(reverse(sp), inner = 8)                # first-seen order f, e, …, a
        # direct: both tip matchers
        @test_throws ArgumentError DRModels._poisson_phylo_setup(tr, uppercase.(g))
        @test_throws ArgumentError DRModels._poisson_phylo_setup(tr, g .* "_x")
        phy = DRModels.augmented_phy(tr)
        @test_throws ArgumentError DRModels._phylo_mean_leaf_index(phy, uppercase.(g))
        msg = _j3_msg(() -> DRModels._phylo_mean_leaf_index(phy, uppercase.(g)))
        @test occursin("tip names", msg) && occursin("\"F\"", msg)   # first unmatched label, first-seen order
        # partial name match (some labels are tips, one is not) is refused too
        gp = copy(g); gp[gp .== "f"] .= "F"
        @test_throws ArgumentError DRModels._poisson_phylo_setup(tr, gp)
        # names that ARE tips are matched by name, whatever their order
        leaf = DRModels._phylo_mean_leaf_index(phy, g)
        @test phy.leaf_names[leaf] == g
        # purely numeric codes (no name to anchor on) keep the positional fallback
        codes = repeat(101:106, inner = 2)
        @test length(DRModels._phylo_mean_leaf_index(phy, codes)) == length(codes)

        # integer-valued Float64 codes (R doubles) are tip indices, like the same codes as Int
        gf = repeat([6.0, 5.0, 4.0, 3.0, 2.0, 1.0], inner = 2)
        @test DRModels._phylo_mean_leaf_index(phy, gf) == DRModels._phylo_mean_leaf_index(phy, Int.(gf))
        @test DRModels._phylo_mean_leaf_index(phy, gf) == Int.(gf)

        # end to end on a Laplace route
        rng = MersenneTwister(11)
        n = length(g); x = randn(rng, n)
        ueff = Dict("a" => 1.5, "b" => 1.4, "c" => -1.0, "d" => -1.1, "e" => 0.2, "f" => 0.1)
        y = Float64[_j3_rpois(rng, exp(0.3 + 0.2x[i] + ueff[g[i]])) for i in 1:n]
        fm = bf(@formula(y ~ x + phylo(1 | g)))
        @test drm(fm, DRModels.Poisson(); data = (y = y, x = x, g = g), tree = tr,
                  se = false) isa DrmFit
        @test_throws ArgumentError drm(fm, DRModels.Poisson();
                                       data = (y = y, x = x, g = uppercase.(g)), tree = tr, se = false)
    end

    @testset "#1022: ragged formula columns are refused before any @inbounds loop" begin
        rng = MersenneTwister(6)
        n = 40; x = randn(rng, n); g = repeat(1:8, inner = 5)
        y = 1 .+ 0.5x .+ randn(rng, n)
        f1 = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1))
        for gg in (g[1:30], g[1:1])
            @test_throws ArgumentError drm(f1, Gaussian(); data = (y = y, x = x, g = gg))
        end
        msg = _j3_msg(() -> drm(f1, Gaussian(); data = (y = y, x = x, g = g[1:30])))
        @test occursin("column `g` has 30 value(s)", msg)
        @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 + x | g)), @formula(sigma ~ 1)),
                                       Gaussian(); data = (y = y, x = x, g = g[1:1]))
        # meta_V: a length-1 variance column used to broadcast and fit silently
        v = fill(0.1, n)
        fm = bf(@formula(y ~ x + meta_V(v)), @formula(sigma ~ 1))
        @test drm(fm, Gaussian(); data = (y = y, x = x, v = v)) isa DrmFit
        @test_throws ArgumentError drm(fm, Gaussian(); data = (y = y, x = x, v = v[1:1]))
        @test_throws ArgumentError drm(fm, Gaussian(); data = (y = y, x = x, v = v[1:15]))
        # a non-Gaussian route goes through the same check
        cnt = Float64.(rand(rng, 0:4, n))
        @test_throws ArgumentError drm(bf(@formula(cnt ~ x + (1 | g))), DRModels.Poisson();
                                       data = (cnt = cnt, x = x, g = g[1:30]))
    end

    @testset "#1020: bivariate K must be symmetric and finite (both triangles read)" begin
        rng = MersenneTwister(82)
        G = 10; m = 6; n = G * m
        g = repeat(1:G, inner = m); x = randn(rng, n)
        A0 = randn(rng, G, G); K0 = Matrix(Symmetric(A0 * A0' / G + 0.5I))
        u = cholesky(K0).L * randn(rng, G, 2) * 0.6
        y1 = 1 .+ 0.5x .+ u[g, 1] .+ randn(rng, n); y2 = -0.5 .+ 0.3x .+ u[g, 2] .+ randn(rng, n)
        form = bf(mu1 = @formula(y1 ~ x + relmat(1 | g)), mu2 = @formula(y2 ~ x + relmat(1 | g)),
                  sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                  rho12 = @formula(rho12 ~ 1))
        dat = (y1 = y1, y2 = y2, x = x, g = g)
        Kl = copy(K0); Kl[5, 2] = NaN
        Kd = copy(K0); Kd[5, 2] += 0.3
        Kn = copy(K0); Kn[3, 3] = NaN
        for Kbad in (Kl, Kd, Kn)
            @test_throws ArgumentError drm(form, Gaussian(); data = dat, K = Kbad)
            @test_throws ArgumentError DRModels._q4_structured_precision(:relmat, :g, G;
                K = Kbad, A = nothing, coords = nothing, spatial_range = nothing)
        end
        @test occursin("not symmetric", _j3_msg(() -> drm(form, Gaussian(); data = dat, K = Kd)))
        @test occursin("non-finite", _j3_msg(() -> drm(form, Gaussian(); data = dat, K = Kn)))
        @test drm(form, Gaussian(); data = dat, K = K0) isa DrmFit
    end

    @testset "#1014: meta_V without a random effect refuses negative or non-finite variances" begin
        rng = MersenneTwister(3)
        n = 40; x = randn(rng, n); y = 1 .+ 0.5x .+ randn(rng, n)
        fm = bf(@formula(y ~ x + meta_V(v)), @formula(sigma ~ 1))
        vneg1 = [i == 1 ? -0.1 : 0.05 for i in 1:n]
        @test_throws ArgumentError drm(fm, Gaussian(); data = (y = y, x = x, v = vneg1))
        @test occursin("non-negative",
                       _j3_msg(() -> drm(fm, Gaussian(); data = (y = y, x = x, v = vneg1))))
        @test_throws ArgumentError drm(fm, Gaussian(); data = (y = y, x = x, v = fill(-0.05, n)))
        @test drm(fm, Gaussian(); data = (y = y, x = x, v = fill(0.05, n))) isa DrmFit
        # a zero-variance study is legitimate (drmTMB accepts it): v + sigma^2 stays positive
        v0 = [0.0; fill(0.05, n - 1)]
        @test drm(fm, Gaussian(); data = (y = y, x = x, v = v0)) isa DrmFit
        @test_throws ArgumentError drm(fm, Gaussian(); data = (y = y, x = x, v = [NaN; fill(0.05, n - 1)]))
    end

    @testset "#1013: drm_listwise takes temporal() formulas and refuses ragged tables" begin
        rng = MersenneTwister(13)
        ids = repeat(1:20, inner = 8); occ = repeat(1:8, 20); x = randn(rng, 160)
        y = Union{Missing,Float64}[1 .+ 0.5x .+ randn(rng, 160);]; y[5] = missing
        f = bf(@formula(y ~ x + temporal(1 | id, occ, ar1)), @formula(sigma ~ 1))
        clean = drm_listwise(f, (y = y, x = x, id = ids, occ = occ); verbose = false)
        @test length(clean.y) == 159
        @test !any(ismissing, clean.y)
        # a genuinely absent column is still reported
        @test_throws ArgumentError drm_listwise(f, (y = y, x = x, id = ids))
        # ragged table: refused before the unchecked scan
        fr = bf(@formula(y ~ x), @formula(sigma ~ 1))
        @test_throws ArgumentError drm_listwise(fr, (y = [1.0, 2.0, 3.0, 4.0], x = [1.0, 2.0]))
    end

    @testset "#1010: tree / K / A / coords with no matching marker are refused" begin
        rng = MersenneTwister(1)
        x = randn(rng, 160); y = 1 .+ 0.5x .+ randn(rng, 160); g = repeat(1:20, inner = 8)
        dat = (y = y, x = x, g = g)
        f = bf(@formula(y ~ x), @formula(sigma ~ 1))
        tr = "((a:1,b:1):1,(c:1,d:1):1);"
        for kw in ((; tree = tr), (; K = ones(3, 3)), (; A = ones(2, 2)), (; coords = ones(4, 2)))
            @test_throws ArgumentError drm(f, Gaussian(); data = dat, kw...)
            @test occursin("`$(first(keys(kw))) = …` was supplied",
                           _j3_msg(() -> drm(f, Gaussian(); data = dat, kw...)))
        end
        fre = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1))
        @test_throws ArgumentError drm(fre, Gaussian(); data = dat, K = Matrix(1.0I, 20, 20))
        @test_throws ArgumentError drm(fre, Gaussian(); data = dat, tree = tr)
        cnt = Float64.(rand(rng, 0:4, 160))
        fp = bf(@formula(cnt ~ x))
        @test_throws ArgumentError drm(fp, DRModels.Poisson(); data = (cnt = cnt, x = x), tree = tr)
        @test_throws ArgumentError drm(fp, DRModels.Poisson(); data = (cnt = cnt, x = x), K = ones(2, 2))
        # the matching marker still consumes the keyword
        @test drm(bf(@formula(y ~ x + relmat(1 | g)), @formula(sigma ~ 1)), Gaussian();
                  data = dat, K = Matrix(1.0I, 20, 20)) isa DrmFit
    end

    @testset "#966: cross-family formula route refuses missing responses and checks the sentinel" begin
        rng = MersenneTwister(31)
        n = 120; x = randn(rng, n); z = randn(rng, n)
        y1 = 0.3 .+ 0.7 .* x .+ z
        y2 = Float64.([rand(rng) < 1 / (1 + exp(-(0.2 + 0.5x[i] + z[i]))) for i in 1:n])
        y1m = copy(y1); y1m[4] = NaN
        bfx = bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
                 sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1))
        msg = _j3_msg(() -> drm(bfx, (Gaussian(), DRModels.Binomial());
                                data = (y1 = y1m, y2 = y2, x = x), confint = false))
        @test occursin("cross-family route needs complete", msg)
        @test occursin("`y1`", msg) && occursin("row 4", msg)
        # A clean formula-route fit is nowhere near the sentinel, and still returns its fields.
        fclean = drm(bfx, (Gaussian(), DRModels.Binomial()); data = (y1 = y1, y2 = y2, x = x), confint = false)
        @test fclean.loglik > -1e9
        @test fclean.converged isa Bool
    end

    @testset "#1017: r2_constant_sigma refuses structured / meta_V fits and handles NaN rows" begin
        rng = MersenneTwister(17)
        G = 20; g = repeat(1:G, inner = 6); n = length(g)
        x = randn(rng, n); b = 1.2 .* randn(rng, G)
        y = 1 .+ 0.5x .+ b[g] .+ 0.5randn(rng, n)
        frel = drm(bf(@formula(y ~ x + relmat(1 | g)), @formula(sigma ~ 1)), Gaussian();
                   data = (y = y, x = x, g = g), K = Matrix(1.0I, G, G))
        @test_throws ArgumentError r2_constant_sigma(frel)
        v = fill(0.1, n)
        fmeta = drm(bf(@formula(y ~ x + meta_V(v)), @formula(sigma ~ 1)), Gaussian();
                    data = (y = y, x = x, v = v))
        @test_throws ArgumentError r2_constant_sigma(fmeta)
        @test occursin("meta_V", _j3_msg(() -> r2_constant_sigma(fmeta)))
        # a NaN response row is dropped by the fit and by R², not called "zero variance"
        yn = 1 .+ 0.5x .+ randn(rng, n); yn[1] = NaN
        fn = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (y = yn, x = x))
        r = r2_constant_sigma(fn)
        keep = 2:n
        Xk = hcat(ones(n - 1), x[keep]); yk = yn[keep]
        bk = Xk \ yk
        ols = 1 - sum(abs2, yk .- Xk * bk) / sum(abs2, yk .- sum(yk) / length(yk))
        @test r ≈ ols atol = 1e-6
    end

    @testset "#938 / #1097: offset() is applied (coefficient 1) or refused, never freed" begin
        rng = MersenneTwister(938)
        n = 150; x = randn(rng, n); logb = 0.3 .* randn(rng, n); g = repeat(1:15, inner = 10)
        c = Float64[_j3_rpois(rng, exp(0.2 + 0.3x[i] + logb[i])) for i in 1:n]
        d = (c = c, x = x, logb = logb, g = g)
        # applied: no `offset` coefficient on NB2 (fixed and (1 | g)), Poisson, Gaussian
        fnb = drm(bf(@formula(c ~ x + offset(logb)), @formula(sigma ~ 1)),
                  NegBinomial2(); data = d, se = false)
        @test length(coef(fnb, :mu)) == 2
        fnr = drm(bf(@formula(c ~ x + offset(logb) + (1 | g)), @formula(sigma ~ 1)),
                  NegBinomial2(); data = d, se = false)
        @test length(coef(fnr, :mu)) == 2
        y = 1 .+ 0.5x .+ logb .+ randn(rng, n)
        fg = drm(bf(@formula(y ~ x + offset(logb)), @formula(sigma ~ 1)),
                 Gaussian(); data = merge(d, (y = y,)))
        @test length(coef(fg, :mu)) == 2
        fp = drm(bf(@formula(c ~ x + offset(logb))), DRModels.Poisson(); data = d, se = false)
        @test length(coef(fp, :mu)) == 2
        fpl = drm(bf(@formula(c ~ x + offset(logb) + (1 | g))), DRModels.Poisson();
                  data = d, marginal = :Laplace, se = false)
        @test length(coef(fpl, :mu)) == 2
        # refused, by name: a non-mean Poisson part, a Gaussian sigma offset, other families
        @test_throws ArgumentError drm(bf(@formula(c ~ x), @formula(zi ~ offset(logb))),
                                       DRModels.Poisson(); data = d, se = false)
        err = try
            drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + offset(logb))),
                Gaussian(); data = merge(d, (y = y,))); nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("offset(...)", err.msg)
        @test_throws ArgumentError drm(bf(@formula(c ~ x + offset(logb)), @formula(sigma ~ 1)),
                                       DRModels.Gamma(); data = merge(d, (c = c .+ 1,)))
    end

    @testset "#939: CumulativeLogit refuses a sigma formula with covariates" begin
        # Same guard as #1027 (its CumulativeLogit row above); pinned here by issue.
        rng = MersenneTwister(939)
        n = 150; x = randn(rng, n); z = randn(rng, n)
        k = Float64.(clamp.(round.(Int, 3 .+ x .+ randn(rng, n)), 1, 5))
        @test_throws ArgumentError drm(bf(@formula(k ~ x), @formula(sigma ~ z)), CumulativeLogit();
                                       data = (k = k, x = x, z = z))
    end

    @testset "#926: r2_constant_sigma refuses a scale random intercept" begin
        rng = MersenneTwister(926)
        G = 20; g = repeat(1:G, inner = 10); n = length(g); x = randn(rng, n)
        s = exp.(0.6 .* randn(rng, G))
        y = 1 .+ 0.5x .+ s[g] .* randn(rng, n)
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + (1 | g))), Gaussian();
                  data = (y = y, x = x, g = g))
        @test_throws ArgumentError r2_constant_sigma(fit)
    end

    @testset "#951: a sparse solver request on an unstructured Gaussian cell is refused" begin
        rng = MersenneTwister(951)
        n = 120; x = randn(rng, n); g = repeat(1:12, inner = 10)
        y = 1 .+ 0.5x .+ 0.6 .* randn(rng, 12)[g] .+ randn(rng, n)
        d = (y = y, x = x, g = g)
        ffix = bf(@formula(y ~ x), @formula(sigma ~ 1))
        fre = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1))
        @test_throws ArgumentError drm(ffix, Gaussian(); data = d, algorithm = :sparse)
        @test occursin("algorithm = :sparse",
                       _j3_msg(() -> drm(ffix, Gaussian(); data = d, algorithm = :sparse)))
        @test_throws ArgumentError drm(ffix, Gaussian(); data = d, sparse = true)
        @test_throws ArgumentError drm(fre, Gaussian(); data = d, algorithm = :sparse)
        # unstructured sigma random intercept and iid sd(g): same refusal (not silently dense)
        fsig = bf(@formula(y ~ x), @formula(sigma ~ 1 + (1 | g)))
        fboth = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1 + (1 | g)))
        fsd = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1), @formula(sd(g) ~ 1))
        for ff in (fsig, fboth, fsd)
            @test_throws ArgumentError drm(ff, Gaussian(); data = d, sparse = true)
            @test_throws ArgumentError drm(ff, Gaussian(); data = d, algorithm = :sparse)
        end
        # `:gls` / `:lbfgs` are the documented aliases for these cells' default fitter
        @test loglik(drm(ffix, Gaussian(); data = d, algorithm = :gls)) ≈
              loglik(drm(ffix, Gaussian(); data = d)) atol = 1e-10
    end

    @testset "#957: A is matched to the data by label (ids = … or a labelled table)" begin
        rng = MersenneTwister(957)
        G = 8; m = 6
        ids_sorted = ["a$(i)" for i in 1:G]                    # A's row labels, sorted
        B = randn(rng, G, G); A0 = Matrix(Symmetric(B * B' / G + 0.5I))   # A0 in sorted order
        first_seen = ids_sorted[[5, 2, 8, 1, 7, 3, 6, 4]]      # data visit animals in this order
        animal_col = repeat(first_seen, inner = m); n = length(animal_col)
        u = cholesky(A0).L * randn(rng, G)
        x = randn(rng, n)
        y = 1 .+ 0.5x .+ 0.8 .* u[indexin(animal_col, ids_sorted)] .+ randn(rng, n)
        d = (y = y, x = x, animal = animal_col)
        f = bf(@formula(y ~ x + animal(1 | animal)), @formula(sigma ~ 1))
        o = indexin(first_seen, ids_sorted)
        ll_ref = loglik(drm(f, Gaussian(); data = d, A = A0[o, o]))       # first-appearance order
        ll_ids = loglik(drm(f, Gaussian(); data = d, A = A0, ids = ids_sorted))
        @test ll_ids ≈ ll_ref atol = 1e-8
        Atab = NamedTuple{Tuple(Symbol.(ids_sorted))}(Tuple(collect(eachcol(A0))))
        ll_tab = loglik(drm(f, Gaussian(); data = d, A = Atab))
        @test ll_tab ≈ ll_ref atol = 1e-8
        # extra labels (animals not in the data) are dropped; a level missing from A is refused
        ids_plus = vcat(ids_sorted, "z9"); Aplus = Matrix(1.0I, G + 1, G + 1); Aplus[1:G, 1:G] .= A0
        @test loglik(drm(f, Gaussian(); data = d, A = Aplus, ids = ids_plus)) ≈ ll_ref atol = 1e-8
        @test_throws ArgumentError drm(f, Gaussian(); data = d, A = A0[1:G-1, 1:G-1],
                                       ids = ids_sorted[1:G-1])
        @test_throws ArgumentError drm(f, Gaussian(); data = d, A = A0, ids = ids_sorted[1:G-1])
        # `ids` without `A` is refused
        @test_throws ArgumentError drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian();
                                       data = d, ids = ids_sorted)
    end

    @testset "#974 / #963: readable refusals for univariate V = and animal() on sigma" begin
        rng = MersenneTwister(974)
        n = 10; x = randn(rng, n); y = randn(rng, n)
        msg = _j3_msg(() -> drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian();
                                data = (y = y, x = x), V = Matrix(0.01I, n, n)))
        @test occursin("`V = …`", msg) && occursin("bivariate", msg) && occursin("meta_V", msg)
        G = 5; g = repeat(1:G, inner = 6); n2 = length(g); x2 = randn(rng, n2); y2 = randn(rng, n2)
        fs = bf(@formula(y ~ x + animal(1 | g)), @formula(sigma ~ 1 + animal(1 | g)))
        msg2 = _j3_msg(() -> drm(fs, Gaussian(); data = (y = y2, x = x2, g = g), A = Matrix(1.0I, G, G)))
        @test occursin("#963", msg2) && occursin("on `sigma`", msg2)
    end
end
