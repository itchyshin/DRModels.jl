# Lane J4 regression tests (predict / residuals / density numerics), one
# testset per issue. Every numeric reference here is computed independently of
# the code under test: a full-range or BigFloat series, adaptive quadrature
# (QuadGK), or a closed-form Poisson log-likelihood.
#
#   #1028  Tweedie series window (peak search capped at n = 400, ±30 terms)
#   #982   NB2 fixed-effect logLik above the Poisson maximum at size → ∞
#   #998 / #1039  NB2 (1 | g) objective collapses at near-Poisson size
#   #1015  predict: every newdata error reported as an unseen factor level
#   #1006  predict / predict_parameters ignored offset()
#   #1000  bivariate (and non-Gaussian) predict rebuilt factor contrasts silently
#   #1016  coeftable docstring said z/p are NaN for sigma / resd rows
#   #999   VA kernels: `pσ == 1` admitted `sigma ~ 0 + x`; unclamped scale
#   #995 / #996 / #981  structured random intercept in quantile residuals

using DRModels
using Test, Random, Statistics, LinearAlgebra
import Distributions, QuadGK, ForwardDiff
using SpecialFunctions: loggamma

# --------------------------------------------------------------------------
# Independent Tweedie reference: the same Dunn–Smyth series summed over the
# FULL range n = 1..N (N far past the peak), in log-sum-exp form.
function _j4_tweedie_full(y, μ, φ, p)
    λ = μ^(2 - p) / (φ * (2 - p)); α = (2 - p) / (1 - p); γ = φ * (p - 1) * μ^(p - 1)
    nstar = y^(2 - p) / (φ * (2 - p))
    N = ceil(Int, 5 * nstar + 2000)
    z = log(λ) - α * log(y) + α * log(γ)
    t = [n * z - loggamma(n + 1.0) - loggamma(-n * α) for n in 1:N]
    m = maximum(t)
    return -λ - y / γ - log(y) + m + log(sum(exp.(t .- m)))
end

# Minimal quoted-CSV reader for the vendored fixtures (numbers or strings).
function _j4_csv(path)
    lines = readlines(path)
    hdr = Symbol.(replace.(split(lines[1], ","), "\"" => ""))
    cols = [String[] for _ in hdr]
    for l in lines[2:end], (j, v) in enumerate(split(l, ","))
        push!(cols[j], replace(v, "\"" => ""))
    end
    return NamedTuple{Tuple(hdr)}(Tuple(cols))
end

@testset "lane J4: predict / residuals / numerics" begin

@testset "#1028 Tweedie density: series window follows the peak" begin
    cases = [(2.0, 2.0, 1.0, 1.5),        # n* ~ 3 (unchanged regime)
             (20.0, 20.0, 0.05, 1.5),     # n* ~ 179
             (50.0, 50.0, 0.05, 1.5),     # n* ~ 283: old code off by 0.01
             (100.0, 100.0, 0.001, 1.5),  # n* ~ 20000: old code off by 35843 nats
             (500.0, 500.0, 0.05, 1.2),   # n* ~ 3607
             (1000.0, 800.0, 0.02, 1.6)]  # n* ~ 1981
    for (y, μ, φ, p) in cases
        pkg = DRModels._logpdf_tweedie(y, μ, φ, p)
        ref = _j4_tweedie_full(y, μ, φ, p)
        @test isfinite(pkg)
        @test pkg ≈ ref atol = 1e-8 rtol = 1e-11
    end

    # BigFloat reference for one mid-size case (independent of float64 rounding).
    setprecision(BigFloat, 256) do
        y, μ, φ, p = big(50.0), big(50.0), big(0.05), big(1.5)
        λ = μ^(2 - p) / (φ * (2 - p)); α = (2 - p) / (1 - p); γ = φ * (p - 1) * μ^(p - 1)
        z = log(λ) - α * log(y) + α * log(γ)
        s = sum(exp(n * z - loggamma(big(n) + 1) - loggamma(-n * α)) for n in 1:3000)
        ref = Float64(-λ - y / γ - log(y) + log(s))
        @test DRModels._logpdf_tweedie(50.0, 50.0, 0.05, 1.5) ≈ ref atol = 1e-9
    end

    # The density integrates to one and has mean μ (point mass at 0 included),
    # in a regime where the old window lost ~1% of the mass.
    μ, φ, p = 50.0, 0.05, 1.5
    f(y) = exp(DRModels._logpdf_tweedie(y, μ, φ, p))
    mass, _ = QuadGK.quadgk(f, 0.0, 20.0, 40.0, 50.0, 60.0, 80.0, 200.0; rtol = 1e-12)
    P0 = exp(DRModels._logpdf_tweedie(0.0, μ, φ, p))
    @test mass + P0 ≈ 1 atol = 1e-8
    m1, _ = QuadGK.quadgk(y -> y * f(y), 0.0, 20.0, 40.0, 50.0, 60.0, 80.0, 200.0; rtol = 1e-12)
    @test m1 ≈ μ rtol = 1e-8

    # Still differentiable (the window is chosen on detached values).
    g = ForwardDiff.derivative(m -> DRModels._logpdf_tweedie(50.0, m, 0.05, 1.5), 48.0)
    h = 1e-5
    fd = (DRModels._logpdf_tweedie(50.0, 48.0 + h, 0.05, 1.5) -
          DRModels._logpdf_tweedie(50.0, 48.0 - h, 0.05, 1.5)) / (2h)
    @test g ≈ fd rtol = 1e-6
    H = ForwardDiff.hessian(v -> DRModels._logpdf_tweedie(50.0, v[1], v[2], 1.5), [48.0, 0.05])
    @test all(isfinite, H)
end

@testset "#998 NB2 log-pmf at near-Poisson size (BigFloat reference)" begin
    setprecision(BigFloat, 512) do
        for r in (1e8, 1e13, 1e17), μ in (2.7, 150.0), k in (0, 1, 5, 40, 160)
            rb, μb = big(r), big(μ)
            ref = loggamma(k + rb) - loggamma(rb) - loggamma(big(k) + 1) +
                  rb * log(rb / (rb + μb)) + k * log(μb / (rb + μb))
            @test DRModels._nb2_logpmf(r, μ, k) ≈ Float64(ref) atol = 1e-8
        end
    end
end

@testset "#982 NB2 fixed effects: logLik never above the Poisson maximum" begin
    Random.seed!(982)
    n = 300; x = randn(n)
    # Under-dispersed counts (binomial): the NB2 MLE runs to size → ∞, the
    # regime of #982, so the NB2 logLik must approach the Poisson maximum from
    # below.
    y = Float64.(rand.(Distributions.Binomial.(20, 0.15 .* exp.(0.3 .* x) ./ (1 .+ 0.15 .* exp.(0.3 .* x)))))
    data = (; y, x)
    fp = drm(bf(@formula(y ~ x)), Poisson(); data = data)
    fn = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), NegBinomial2(); data = data)
    @test coef(fn, :sigma)[1] < -5             # really in the size → ∞ regime
    @test loglik(fn) ≤ loglik(fp) + 1e-6
    @test loglik(fn) ≥ loglik(fp) - 1e-3

    # The objective itself at log σ = -15 (size e^30) equals the closed-form
    # Poisson log-likelihood at the same β (true gap ~1e-11). Before the fix it
    # carried O(0.01-0.2) rounding noise.
    β = coef(fp, :mu)
    λ = exp.(β[1] .+ β[2] .* x)
    llpois = sum(Distributions.logpdf.(Distributions.Poisson.(λ), Int.(y)))
    for ls in (-12.0, -15.0, -18.0)
        @test -fn.nll([β; ls]) ≈ llpois atol = 1e-6
    end
end

@testset "#998 / #1039 NB2 (1 | g): objective finite, reproducible, Poisson limit" begin
    Random.seed!(1039)
    G = 25; m = 12; n = G * m
    g = repeat(1:G, inner = m); b = 0.3 .* randn(G)
    y = Float64.(rand.(Distributions.Poisson.(exp.(log(20.0) .+ b[g]))))
    data = (; y, g)
    fp = drm(bf(@formula(y ~ 1 + (1 | g))), Poisson(); data = data)
    fn = drm(bf(@formula(y ~ 1 + (1 | g)), @formula(sigma ~ 1)), NegBinomial2(); data = data)
    @test isfinite(loglik(fn))
    # Independent Poisson-GLMM marginal at (β0, σ_b): per group, adaptive
    # quadrature of Π_i Poisson(y_i; e^{β0+b}) · N(b; 0, σ_b²).
    β0 = log(20.0); lsb = log(0.3)
    nllp = 0.0
    for k in 1:G
        yk = y[g .== k]
        lf(bb) = sum(yi * (β0 + bb) - exp(β0 + bb) - loggamma(yi + 1) for yi in yk) +
                 Distributions.logpdf(Distributions.Normal(0, exp(lsb)), bb)
        grid = range(-3, 3; length = 601); c = maximum(lf.(grid))
        I, _ = QuadGK.quadgk(bb -> exp(lf(bb) - c), -3.0, 0.0, 3.0; rtol = 1e-12)
        nllp -= c + log(I)
    end
    for ls in (-12.0, -20.0, -25.0, -40.0)
        θ = [β0, ls, lsb]
        v1 = fn.nll(θ); v2 = fn.nll(θ)
        @test isfinite(v1)
        @test v1 ≈ v2 atol = 1e-8                 # reproducible (was not, #982 comment)
        @test v1 ≈ nllp atol = 1e-4               # the Poisson GLMM limit (was off by ~200-500)
    end
    # Dispersion gradient is not identically zero past the old hard clamp.
    gσ = ForwardDiff.gradient(fn.nll, [β0, -21.0, lsb])[2]
    @test isfinite(gσ)
    # The NB2 optimum is at least as good as its own Poisson limit at these
    # (β0, σ_b). (It may legitimately exceed the Poisson GLMM fit: Poisson
    # data can show sampling overdispersion, so no upper bound is asserted.)
    @test loglik(fn) ≥ -nllp - 1e-4
end

@testset "#1015 predict: newdata errors name the actual cause" begin
    Random.seed!(5)
    n = 60; x = randn(n); grp = repeat(["a", "b", "c"], inner = 20); off = fill(0.7, n)
    y = 1 .+ 0.5x .+ (grp .== "b") .* 0.8 .+ 0.3 .* randn(n)
    fit = drm(bf(@formula(y ~ x + grp), @formula(sigma ~ x)), Gaussian();
              data = (y = y, x = x, grp = grp))
    msg(f) = try
        f(); ""
    catch e
        e isa ArgumentError || rethrow(e)
        sprint(showerror, e)
    end
    m1 = msg(() -> predict(fit, (x = [0.0, 1.0],)))
    @test occursin("column `grp`", m1) && occursin("missing", m1)
    @test !occursin("not seen", m1)
    m2 = msg(() -> predict(fit, (x = [missing, 1.0], grp = ["a", "a"])))
    @test occursin("column `x`", m2) && occursin("missing", m2)
    @test !occursin("not seen", m2)
    m3 = msg(() -> predict(fit, (x = [0.0], grp = ["z"])))
    @test occursin("not seen when the model was fitted", m3) && occursin("z", m3)
    fo = drm(bf(@formula(y ~ x + offset(off)), @formula(sigma ~ 1)), Gaussian();
             data = (y = y, x = x, off = off))
    m4 = msg(() -> predict(fo, (x = [0.0, 1.0],)))
    @test occursin("column `off`", m4) && !occursin("not seen", m4)
    # Happy path unchanged.
    @test predict(fit, (x = [0.0], grp = ["b"])) ≈
          [coef(fit, :mu)[1] + coef(fit, :mu)[findfirst(==("grp: b"), last(fit.coefnames[1]))]]
end

@testset "#1006 predict applies a fitted Poisson offset()" begin
    Random.seed!(1006)
    n = 400; x = randn(n); expo = rand(n) .* 4 .+ 0.5; le = log.(expo)
    y = Float64.(rand.(Distributions.Poisson.(expo .* exp.(0.2 .+ 0.5 .* x))))
    data = (; y, x, le)
    fit = drm(bf(@formula(y ~ x + offset(le))), Poisson(); data = data)
    @test predict(fit, data) ≈ fitted(fit) rtol = 1e-10
    @test predict_parameters(fit, data)[:mu] ≈ fitted(fit) rtol = 1e-10
    β = coef(fit, :mu)
    nd = (x = [0.0, 1.0], le = [log(2.0), log(3.0)])
    @test predict(fit, nd) ≈ [2.0, 3.0] .* exp.(β[1] .+ β[2] .* [0.0, 1.0]) rtol = 1e-10
    @test predict(fit, nd; type = :link) ≈ β[1] .+ β[2] .* [0.0, 1.0] .+ nd.le rtol = 1e-10
    pse = predict(fit, nd; se = true)            # SE path builds the same design
    @test pse.prediction ≈ predict(fit, nd)
    err = try
        predict(fit, (x = [0.0],)); nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("offset column `le`", sprint(showerror, err))
end

@testset "#1000 predict refuses a factor design that does not match the fit" begin
    Random.seed!(1000)
    n = 120; x = randn(n); grp = repeat(["a", "b", "c"], outer = 40)
    y1 = 0.5 .+ x .+ (grp .== "b") .+ 0.3 .* randn(n)
    y2 = -0.5 .+ 0.5x .- (grp .== "c") .+ 0.3 .* randn(n)
    bfit = drm(bf(mu1 = @formula(y1 ~ x + grp), mu2 = @formula(y2 ~ x + grp)), Gaussian();
               data = (; y1, y2, x, grp))
    full = (x = [0.0, 0.0, 0.0], grp = ["a", "b", "c"])
    p = predict(bfit, full; type = :link)
    b1 = coef(bfit, :mu1)
    @test p[:mu1] ≈ [b1[1], b1[1] + b1[3], b1[1] + b1[4]]
    # Same column COUNT (three levels), different reference level and an
    # unseen level: used to be silently wrong.
    same = (x = [0.0, 0.0, 0.0], grp = ["b", "c", "d"])
    @test_throws ArgumentError predict(bfit, same)
    @test_throws ArgumentError predict_parameters(bfit, same)
    # Fewer levels: loud before too (DimensionMismatch), now an ArgumentError.
    @test_throws ArgumentError predict(bfit, (x = [0.0, 0.0], grp = ["b", "c"]))
    # Non-Gaussian fit (its fitter caches no training schema) gets the same guard.
    yc = Float64.(rand.(Distributions.Poisson.(exp.(0.5 .+ 0.3 .* (grp .== "b")))))
    pfit = drm(bf(@formula(y ~ grp)), Poisson(); data = (y = yc, grp = grp))
    @test predict(pfit, (grp = ["a", "b", "c"],)) ≈ fitted(pfit)[1:3] rtol = 1e-10
    @test_throws ArgumentError predict(pfit, (grp = ["b", "c"],))
end

@testset "#1016 coeftable docstring matches the z / p it returns" begin
    d = string(@doc DRModels.coeftable)
    @test occursin("σ_b = 1", d)
    @test !occursin("as `NaN` rather than a misleading test", d)
    Random.seed!(8)
    G = 25; m = 6; n = G * m
    g = repeat(1:G, inner = m); x = randn(n); u = randn(G) * 0.8
    y = 1 .+ 0.5x .+ u[g] .+ exp.(-0.3 .+ 0.2x) .* randn(n)
    fit = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ x)), Gaussian(); data = (; y, x, g))
    ct = DRModels.coeftable(fit)
    i = findfirst(==("resd: g"), ct.rownms)
    @test i !== nothing
    @test ct.cols[3][i] ≈ ct.cols[1][i] / ct.cols[2][i]    # documented: est / se on log σ_b
end

@testset "#999 VA kernels: constant sigma column only; scored scale reported" begin
    Random.seed!(999)
    G = 20; m = 10; n = G * m
    g = repeat(1:G, inner = m); x = randn(n)
    y = Float64.(rand.(Distributions.Poisson.(exp.(1.0 .+ 0.3 .* randn(G))[g])))
    @test_throws ErrorException drm(bf(@formula(y ~ 1 + (1 | g)), @formula(sigma ~ 0 + x)),
                                    NegBinomial2(); data = (; y, x, g), marginal = :VA)
    @test DRModels._va_constant_column(ones(5, 1))
    @test !DRModels._va_constant_column(reshape(collect(1.0:5.0), 5, 1))
    s = @test_logs (:warn, r"#959") DRModels._va_scored_sigma(ones(3, 1), [25.0], 20.0, "t")
    @test s ≈ fill(exp(20.0), 3)
    @test (@test_logs DRModels._va_scored_sigma(ones(3, 1), [0.5], 20.0, "t")) ≈ fill(exp(0.5), 3)
end

@testset "#995 / #996 / #981 structured random intercept: warned, not silent" begin
    Random.seed!(995)
    phy = random_balanced_tree(32; branch_length = 0.3)      # height 1.5
    @test phylo_tree_height(phy) ≈ 1.5
    species = repeat(1:32, inner = 6); xs = randn(length(species))
    C = sigma_phy_dense(phy)
    a = cholesky(Symmetric(0.25 .* C + 1e-10I)).L * randn(32)
    y = 0.2 .+ 0.4 .* xs .+ a[species] .+ 0.5 .* randn(length(species))
    fit = drm(bf(@formula(y ~ x + phylo(1 | species))), Gaussian();
              data = (; y, x = xs, species), tree = phy)
    @test DRModels._structured_mean_re(fit)
    @test_logs (:warn, r"#995") match_mode = :any residuals(fit; type = :quantile)

    G = 20; gg = repeat(1:G, inner = 10); xo = randn(10G)
    yo = 0.3 .+ xo .+ (0.5 .* randn(G))[gg] .+ 0.5 .* randn(10G)
    ofit = drm(bf(@formula(y ~ x + (1 | g))), Gaussian(); data = (y = yo, x = xo, g = gg))
    @test !DRModels._structured_mean_re(ofit)

    # #996: the non-Gaussian phylo setup now gives the tree-height warning.
    empty!(DRModels._PHYLO_HEIGHT_WARNED)
    @test_logs (:warn, r"maximum tip height") match_mode = :any DRModels._poisson_phylo_setup(phy, 1:32)

    # NB2 conditional distribution: Poisson limit once p would quantise (#998).
    d = DRModels._conditional_dist(NegBinomial2(), 1; μ = [150.0],
                                   scales = Dict(:sigma => [exp(-20.0)]), obs = Dict(:mu => [150.0]))
    @test d isa Distributions.Poisson
    d2 = DRModels._conditional_dist(NegBinomial2(), 1; μ = [150.0],
                                    scales = Dict(:sigma => [0.5]), obs = Dict(:mu => [150.0]))
    @test d2 isa Distributions.NegativeBinomial
end



@testset "#927 mean + sigma random intercepts with a sigma covariate fit" begin
    ss = _j4_csv(joinpath(@__DIR__, "fixtures", "lme4", "sleepstudy.csv"))
    y = parse.(Float64, ss.Reaction); Days = parse.(Float64, ss.Days)
    Subject = "S" .* ss.Subject
    fit = drm(bf(@formula(Reaction ~ 1 + Days + (1 | Subject)),
                 @formula(sigma ~ 1 + Days + (1 | Subject))), Gaussian();
              data = (; Reaction = y, Days, Subject))
    @test isfinite(loglik(fit))
    @test fit.nll(fit.theta) < DRModels.REML_NONPD_PENALTY   # optimum is not on the barrier
    # drmTMB 0.7.1 (TMB Laplace, same model) reports logLik -883.4328 (#927 comment).
    @test loglik(fit) ≈ -883.4328 atol = 0.05

    # The 2-D mode finder returns a MINIMUM (PD Hessian, zero gradient) on a
    # group whose residuals share one sign, where h(b, δ) has a saddle.
    r = [3.0, 3.2, 2.9, 3.1, 3.05]; η0 = fill(-1.0, 5)
    b, δ = DRModels._musig_mode_primal(r, η0, 0.25, 0.09)
    e = exp.(-2 .* (η0 .+ δ)); res = r .- b; D = sum(res .^ 2 .* e)
    Hbb = sum(e) + 1 / 0.25; Hbδ = 2 * sum(res .* e); Hδδ = 2D + 1 / 0.09
    @test Hbb * Hδδ - Hbδ^2 > 0
    @test abs(-sum(res .* e) + b / 0.25) < 1e-8
    @test abs(5 - D + δ / 0.09) < 1e-8
end


@testset "#953 CumulativeLogit crossed RE is not below the model it nests" begin
    w = _j4_csv(joinpath(@__DIR__, "fixtures", "ordinal", "wine.csv"))
    dat = (; rating = parse.(Int, w.rating), temp = w.temp, contact = w.contact,
           judge = w.judge, bottle = w.bottle)
    f2 = drm(bf(@formula(rating ~ 1 + temp + contact + (1 | judge) + (1 | bottle))),
             CumulativeLogit(); data = dat)
    # drmTMB 0.7.1 and ordinal::clmm (both Laplace) give -81.56541 for this
    # model AND for the single (1 | judge) model it nests (bottle SD ~ 2e-5);
    # DRModels.jl returned -81.7577 (#953).
    @test loglik(f2) ≥ -81.56541 - 5e-3
    @test loglik(f2) ≤ -81.56541 + 5e-2
end

end
