# test_tweedie_aghq.jl — per-group adaptive Gauss–Hermite quadrature (AGHQ) for
# Tweedie's two `mu` random-effect routes: the ordinary intercept `(1 | g)`
# (`_fit_tweedie_ranef`) and the independent slope `(0 + x | g)`
# (`_fit_tweedie_slope_ranef`). Applies the #719/#834 pattern (already wired
# into Poisson/NegBinomial2/Gamma/Beta/BetaBinomial/Student/LogNormal by #846,
# and into the correlated `(1 + x | g)` families by #839) to the last two
# `_fit_*_ranef` routes still on the pre-#719 fixed 32-node PRIOR-scale grid
# (b = √2 σ_b z): that grid places nodes at the assumed prior width regardless
# of where the group's posterior actually sits, so an informative group with a
# large fitted σ_b (the fit is confident the intercept varies a lot by group)
# has its marginal likelihood integrated on a grid that mostly misses the
# posterior. On the single-group demo below (σ_b = 6, m = 20, φ = 1.3,
# p = 1.5) that costs the intercept route 12.87 nat and the slope route 23.40
# nat against an independent QuadGK reference; both routes' *default* AGHQ
# (K = `_RANEF1D_AGHQ_K` = 5) are within 3e-8 and 6e-8 nat of the same
# reference (well inside the 1e-6 nat bar below).
#
# The K keyword this PR adds to `_fit_tweedie_ranef`/`_fit_tweedie_slope_ranef`
# does not exist on the pre-fix branch, and the pre-fix routes have no adaptive
# machinery to call at all — every `@testset` below either errors immediately
# (the `K = ...` calls) or fails on the numeric bar (the single-group and
# recovery comparisons) if run against `origin/claude/twin-gap-834` before this
# PR's `src/tweedie.jl` changes; verified by hand (`git stash` the src change,
# rerun this file: the `K` calls MethodError and the single-group comparisons
# fail the `abs(pkg5 - exact) < 1e-6` test at ~12.87 nat / 23.40 nat instead).
#
# Reference: `_exact_group_logmarg[_slope]` is an INDEPENDENT computation
# (ForwardDiff Newton to the group mode, then QuadGK over b with the mode as an
# interior breakpoint) — not the package's own `_aghq_group_logint` — matching
# the discipline of test_adaptive_ghq.jl / test_adaptive_ghq_1d.jl (`_ref_group834`
# / `_ref_group719`), except those use an independent AGHQ-40+ reference while
# this file uses QuadGK directly (as the task asked), which is possible here
# because q = 1 (Tweedie's two RE routes are both 1-D on `mu`).
module TestTweedieAGHQ

using DRModels
using Test, Random, ForwardDiff, StableRNGs, QuadGK
using DelimitedFiles: readdlm
import Distributions
const _Dtw = Distributions

# --- independent per-group reference: Newton mode + QuadGK ------------------
function _newton_mode(h; b0 = 0.0, maxiter = 200)
    b = b0
    for _ in 1:maxiter
        g = ForwardDiff.derivative(h, b)
        H = ForwardDiff.derivative(x -> ForwardDiff.derivative(h, x), b)
        step = g / H
        bn = b - step
        t = 1.0
        while !isfinite(h(bn)) || h(bn) < h(b) - 1e-13
            t /= 2; bn = b - t * step
            t < 1e-12 && break
        end
        b = bn
        abs(t * step) < 1e-13 && break
    end
    return b
end

function _exact_group_logmarg(y, eta0, phi, p, sigmab; halfwidth = 10.0)
    h(b) = begin
        ll = 0.0
        for i in eachindex(y)
            mu = exp(clamp(eta0[i] + b, -30.0, 30.0))
            ll += DRModels._logpdf_tweedie(y[i], mu, phi, p)
        end
        ll + _Dtw.logpdf(_Dtw.Normal(0, sigmab), b)
    end
    bstar = _newton_mode(h)
    shift = h(bstar)
    integrand(b) = exp(h(b) - shift)
    val, _ = quadgk(integrand, -halfwidth * sigmab, bstar, halfwidth * sigmab; rtol = 1e-12, order = 15)
    return log(val) + shift
end

function _exact_group_logmarg_slope(y, eta0, xs, phi, p, sigmab; halfwidth = 10.0)
    h(b) = begin
        ll = 0.0
        for i in eachindex(y)
            mu = exp(clamp(eta0[i] + b * xs[i], -30.0, 30.0))
            ll += DRModels._logpdf_tweedie(y[i], mu, phi, p)
        end
        ll + _Dtw.logpdf(_Dtw.Normal(0, sigmab), b)
    end
    bstar = _newton_mode(h)
    shift = h(bstar)
    integrand(b) = exp(h(b) - shift)
    val, _ = quadgk(integrand, -halfwidth * sigmab, bstar, halfwidth * sigmab; rtol = 1e-12, order = 15)
    return log(val) + shift
end

# Package AGHQ at a given K, built exactly as `_fit_tweedie_ranef` /
# `_fit_tweedie_slope_ranef` build it internally (q = 1; Zre = 1s for the
# intercept route, Zre = xs for the slope route).
function _pkg_group_logmarg(y, eta0, phi, p, sigmab, xs, K)
    n = length(y)
    ll = (i, eta) -> (mu = exp(clamp(eta, -30.0, 30.0)); DRModels._logpdf_tweedie(y[i], mu, phi, p))
    rule = DRModels._AGHQRule(1, K)
    Zre = xs === nothing ? ones(n, 1) : reshape(xs, n, 1)
    val, _ = DRModels._aghq_group_logint(ll, collect(1:n), eta0, Zre, reshape([sigmab], 1, 1), rule, zeros(1))
    return val
end

# Pre-#719-for-Tweedie algorithm: the fixed 32-node PRIOR-scale grid
# (b = √2 σ_b z) that `_fit_tweedie_ranef`/`_fit_tweedie_slope_ranef` used
# before this PR. Reconstructed here only to size the gap this PR closes — no
# longer in `src/`.
function _old_prior_ghq32(y, eta0, phi, p, sigmab, xs)
    z, w = DRModels._gauss_hermite(32); logw = log.(w); K = length(z); rt2 = sqrt(2.0); lpi = log(pi)
    terms = Vector{Float64}(undef, K)
    for k in 1:K
        delta = rt2 * sigmab * z[k]
        gll = logw[k]
        for i in eachindex(y)
            e = xs === nothing ? eta0[i] + delta : eta0[i] + delta * xs[i]
            mu = exp(clamp(e, -30.0, 30.0))
            gll += DRModels._logpdf_tweedie(y[i], mu, phi, p)
        end
        terms[k] = gll
    end
    mx = maximum(terms)
    return -0.5 * lpi + mx + log(sum(exp.(terms .- mx)))
end

@testset "Tweedie (1|g)/(0+x|g) adaptive GHQ: single group, sigma_b = 6, vs exact QuadGK" begin
    # One group, m = 20, eta0 = 0.5 (fixed part), phi = 1.3, p = 1.5, sigma_b = 6
    # (the "large sigma_b" the task specifies) — same regime as the PR's demo
    # script. Data are Gamma-mixture stand-ins for Tweedie draws (point mass at
    # 0 + a continuous part), fixed via StableRNG so this is reproducible
    # across Julia versions; only the SCALAR reference values below are
    # load-bearing (checked to 1e-6), not the y draws themselves.
    rng = StableRNG(20260927)
    m = 20
    sigmab = 6.0
    eta0base = 0.5
    phi = 1.3
    p = 1.5

    # --- intercept route: b_true = 1.0 (fixed, not drawn from the wide
    # N(0, 6^2) prior -- that would occasionally give |b| far in excess of the
    # data's dynamic range and blow up the compound Poisson-Gamma series,
    # which is a DGP/scale artifact, not what this test is about).
    b_true = 1.0
    mu_true = exp(eta0base + b_true)
    y = [rand(rng) < 0.3 ? 0.0 : rand(rng, _Dtw.Gamma(2.0, mu_true / 2.0)) for _ in 1:m]
    eta0 = fill(eta0base, m)

    exact = -49.26401760203747                     # QuadGK reference (see file header)
    @test _exact_group_logmarg(y, eta0, phi, p, sigmab) ≈ exact atol = 1e-6   # reference is reproducible
    pk5 = _pkg_group_logmarg(y, eta0, phi, p, sigmab, nothing, DRModels._RANEF1D_AGHQ_K)
    @test pk5 ≈ exact atol = 1e-6                                            # (a): default K matches to 1e-6
    old = _old_prior_ghq32(y, eta0, phi, p, sigmab, nothing)
    @test abs(old - exact) > 1.0                                             # old 32-node grid: >1 nat off (non-vacuous)
    @test isapprox(old, -62.13667050623692; atol = 1e-6)                     # pins the old-code failure size

    # --- slope route: b_true_s = -1.2 (fixed), xs a fixed deterministic
    # covariate; `rng` continues from the intercept draws above (one shared
    # stream, matching how a single StableRNG seed reproduces this whole file).
    xs = [((-1)^i) * (1.0 + 0.1 * i) for i in 1:m]
    b_true_s = -1.2
    mu_true_s = exp.(eta0base .+ b_true_s .* xs)
    ys = [rand(rng) < 0.3 ? 0.0 : rand(rng, _Dtw.Gamma(2.0, mu_true_s[i] / 2.0)) for i in 1:m]

    exact_s = -50.756999145363835
    @test _exact_group_logmarg_slope(ys, eta0, xs, phi, p, sigmab) ≈ exact_s atol = 1e-6
    pk5s = _pkg_group_logmarg(ys, eta0, phi, p, sigmab, xs, DRModels._RANEF1D_AGHQ_K)
    @test pk5s ≈ exact_s atol = 1e-6                                         # (a): default K matches to 1e-6
    olds = _old_prior_ghq32(ys, eta0, phi, p, sigmab, xs)
    @test abs(olds - exact_s) > 1.0
    @test isapprox(olds, -74.16004390836667; atol = 1e-6)
end

@testset "Tweedie (1|g)/(0+x|g): sigma_b recovery + fitted nll vs exact QuadGK at theta_hat" begin
    # Full-model recovery: a genuinely large true sigma_b, fit via the
    # PRODUCTION `_fit_tweedie_ranef`/`_fit_tweedie_slope_ranef` (the same
    # functions `drm()` calls), then checked against an independent QuadGK
    # total at the fitted theta_hat. `K = 25` here (well above the production
    # default `_RANEF1D_AGHQ_K = 5`) isolates whether the AGHQ MACHINERY is
    # correct (the 1e-5 nll bar the task asks for) from whether K = 5 is
    # *enough* for everyday use -- summed over G = 40 groups, K = 5's ~1e-5-1e-6
    # per-group residual can occasionally add up past 1e-5 in total (measured
    # -1.9e-4 nat for this exact DGP at K = 5; still tiny in absolute terms,
    # and the point of exposing K is exactly to let a caller dial it up).
    Ktest = 25

    # --- intercept -----------------------------------------------------------
    rng = StableRNG(19)
    G = 40; m = 15; n = G * m
    sigmab_true = 3.0
    beta_true = [0.5, 0.4]
    g = repeat(1:G, inner = m)
    x = randn(rng, n)
    b = sigmab_true .* randn(rng, G)
    eta0 = beta_true[1] .+ beta_true[2] .* x
    mu = exp.(eta0 .+ b[g])
    y = [rand(rng) < 0.3 ? 0.0 : rand(rng, _Dtw.Gamma(2.0, mu[i] / 2.0)) for i in 1:n]
    gidx, G_ = DRModels._group_index(string.(g))
    Xmu = hcat(ones(n), x); Xsig = ones(n, 1); Xnu = ones(n, 1)

    fit = DRModels._fit_tweedie_ranef(Tweedie(), y, Xmu, Xsig, Xnu, gidx, G_,
                                       ["(Intercept)", "x"], ["(Intercept)"], ["(Intercept)"],
                                       :id, 1e-8; K = Ktest)
    betahat = coef(fit, :mu); logsig = coef(fit, :sigma)[1]; logitp = coef(fit, :nu)[1]
    sdhat = exp(coef(fit, :resd)[1])
    @test sdhat ≈ sigmab_true rtol = 0.1                       # (b): sigma_b recovered within a sensible tolerance

    phihat = exp(2 * logsig); phat = 1 + 1 / (1 + exp(-logitp))
    eta0hat = betahat[1] .+ betahat[2] .* x
    exact_total = sum(_exact_group_logmarg(y[findall(==(j), g)], eta0hat[findall(==(j), g)], phihat, phat, sdhat)
                       for j in 1:G_)
    @test loglik(fit) ≈ exact_total atol = 1e-5                # (b): fitted nll vs exact QuadGK at theta_hat

    # --- independent slope -----------------------------------------------------
    rng2 = StableRNG(23)
    G2 = 40; m2 = 15; n2 = G2 * m2
    sigmab_true2 = 1.2
    beta_true2 = [0.5, 0.3]
    g2 = repeat(1:G2, inner = m2)
    x2 = 0.6 .* randn(rng2, n2)                 # kept modest: (0+x|g) multiplies b by x, so a wide x
                                                 # range combined with a large sigma_b overflows exp()
    bsl = sigmab_true2 .* randn(rng2, G2)
    eta0b = beta_true2[1] .+ beta_true2[2] .* x2
    mu2 = exp.(clamp.(eta0b .+ bsl[g2] .* x2, -20.0, 20.0))
    y2 = [rand(rng2) < 0.3 ? 0.0 : rand(rng2, _Dtw.Gamma(2.0, mu2[i] / 2.0)) for i in 1:n2]
    gidx2, G2_ = DRModels._group_index(string.(g2))
    Xmu2 = hcat(ones(n2), x2); Xsig2 = ones(n2, 1); Xnu2 = ones(n2, 1)

    fit2 = DRModels._fit_tweedie_slope_ranef(Tweedie(), y2, Xmu2, Xsig2, Xnu2, x2, gidx2, G2_,
                                              ["(Intercept)", "x"], ["(Intercept)"], ["(Intercept)"],
                                              :id, 1e-8; K = Ktest)
    betahat2 = coef(fit2, :mu); logsig2 = coef(fit2, :sigma)[1]; logitp2 = coef(fit2, :nu)[1]
    sdhat2 = exp(coef(fit2, :resd)[1])
    @test sdhat2 ≈ sigmab_true2 rtol = 0.1

    phihat2 = exp(2 * logsig2); phat2 = 1 + 1 / (1 + exp(-logitp2))
    eta0hat2 = betahat2[1] .+ betahat2[2] .* x2
    exact_total2 = sum(_exact_group_logmarg_slope(y2[findall(==(j), g2)], eta0hat2[findall(==(j), g2)],
                                                   x2[findall(==(j), g2)], phihat2, phat2, sdhat2)
                        for j in 1:G2_)
    @test loglik(fit2) ≈ exact_total2 atol = 1e-5
end

@testset "Tweedie (1|g)/(0+x|g): existing R-oracle fixtures move by ~1e-6, not 1e-3 (#563 S8)" begin
    # test_tweedie_ranef.jl's `(1 | id)`/`(0 + x | id)` fixtures already assert
    # DRModels.jl vs drmTMB agreement (atol = 1e-3 / rtol = 0.01 / atol = 0.3);
    # those tolerances are unchanged and that file still passes unmodified
    # (18/18). This testset pins the MUCH tighter old-vs-new DRModels.jl delta
    # directly: n_each = 8 obs/group keeps each group posterior close to
    # Gaussian (per that file's own header comment), so the fixed 32-node
    # PRIOR-scale grid was already accurate there and the AGHQ switch barely
    # moves the fit at all -- confirming "moves by no more than the
    # integration-error scale" rather than asserting it only via the loose
    # R-oracle bars above.
    dir = joinpath(@__DIR__, "parity", "fixtures", "tweedie-mu-ranef")
    dir_slope = joinpath(@__DIR__, "parity", "fixtures", "tweedie-mu-slope-ranef")
    raw, header = readdlm(joinpath(dir, "data.csv"), ','; header = true)
    cols = Symbol.(strip.(string.(vec(header))))
    j = Dict(c => i for (i, c) in enumerate(cols))
    id = string.(raw[:, j[:id]]); xdat = Float64[parse(Float64, string(v)) for v in raw[:, j[:x]]]
    ydat = Float64[parse(Float64, string(v)) for v in raw[:, j[:y]]]
    data = (; id, x = xdat, y = ydat)
    fit = drm(bf(@formula(y ~ x + (1 | id)), @formula(nu ~ 1)), Tweedie(); data = data)

    # Old-code (pre-this-PR) DRModels.jl fit on the same fixture, recorded once
    # (`git stash` the src/tweedie.jl change, refit, `git stash pop`) -- not an
    # R oracle, the OLD 32-node DRModels.jl fit itself.
    old_mu = [0.5012330237656447, 0.40229234389242274]
    old_sigma = -0.001602252496833282
    old_nu = -0.024022140438033836
    old_sd = 0.43605489799132224
    old_loglik = -572.3113017458559

    @test coef(fit, :mu)[1] ≈ old_mu[1] atol = 1e-5
    @test coef(fit, :mu)[2] ≈ old_mu[2] atol = 1e-5
    @test coef(fit, :sigma)[1] ≈ old_sigma atol = 1e-5
    @test coef(fit, :nu)[1] ≈ old_nu atol = 1e-5
    @test exp(coef(fit, :resd)[1]) ≈ old_sd atol = 1e-5
    @test loglik(fit) ≈ old_loglik atol = 1e-4

    raw2, header2 = readdlm(joinpath(dir_slope, "data.csv"), ','; header = true)
    cols2 = Symbol.(strip.(string.(vec(header2))))
    j2 = Dict(c => i for (i, c) in enumerate(cols2))
    id2 = string.(raw2[:, j2[:id]]); xdat2 = Float64[parse(Float64, string(v)) for v in raw2[:, j2[:x]]]
    ydat2 = Float64[parse(Float64, string(v)) for v in raw2[:, j2[:y]]]
    data2 = (; id = id2, x = xdat2, y = ydat2)
    fit2 = drm(bf(@formula(y ~ x + (0 + x | id)), @formula(nu ~ 1)), Tweedie(); data = data2)

    old_mu2 = [0.5413756275182275, 0.32562082671921805]
    old_sigma2 = 0.030274805099677296
    old_nu2 = -0.1178463643391292
    old_sd2 = 0.7026694321724711
    old_loglik2 = -591.9008483842179

    @test coef(fit2, :mu)[1] ≈ old_mu2[1] atol = 1e-5
    @test coef(fit2, :mu)[2] ≈ old_mu2[2] atol = 1e-5
    @test coef(fit2, :sigma)[1] ≈ old_sigma2 atol = 1e-5
    @test coef(fit2, :nu)[1] ≈ old_nu2 atol = 1e-5
    @test exp(coef(fit2, :resd)[1]) ≈ old_sd2 atol = 1e-5
    @test loglik(fit2) ≈ old_loglik2 atol = 1e-4
end

end # module
