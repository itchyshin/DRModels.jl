# test_lane_j3_input_validation.jl — fit-entry argument and data checks.
#
# Each testset is a red control for an input that used to be accepted and then
# dropped or misread without a word (code-review issues #1027, #1026, #1022,
# #1020, #1014, #1013, #1010, #966, #1017). Every refusal must be an
# ArgumentError whose message names the offending argument.
using DRModels
using DRModels.StatsModels
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

    @testset "#1014: meta_V without a random effect refuses non-positive variances" begin
        rng = MersenneTwister(3)
        n = 40; x = randn(rng, n); y = 1 .+ 0.5x .+ randn(rng, n)
        fm = bf(@formula(y ~ x + meta_V(v)), @formula(sigma ~ 1))
        vneg1 = [i == 1 ? -0.1 : 0.05 for i in 1:n]
        @test_throws ArgumentError drm(fm, Gaussian(); data = (y = y, x = x, v = vneg1))
        @test occursin("strictly positive",
                       _j3_msg(() -> drm(fm, Gaussian(); data = (y = y, x = x, v = vneg1))))
        @test_throws ArgumentError drm(fm, Gaussian(); data = (y = y, x = x, v = fill(-0.05, n)))
        @test drm(fm, Gaussian(); data = (y = y, x = x, v = fill(0.05, n))) isa DrmFit
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

    @testset "#966: cross-family route refuses missing responses; a sentinel optimum is not converged" begin
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
        # Raw-array entry: a NaN response either is refused up front or, if it
        # reaches the objective, every evaluation is the 1e10 sentinel and the
        # result must not claim convergence.
        X = hcat(ones(n), x)
        res = try
            DRModels.fit_mixed_family(y1 = y1m, X1 = X, fam1 = Gaussian(),
                                      y2 = y2, X2 = X, fam2 = DRModels.Binomial(),
                                      confint = false)
        catch e
            e
        end
        @test res isa Exception || res.converged == false
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
end
