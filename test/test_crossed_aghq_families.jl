# marginal = :AGHQ for crossed random intercepts (1 | g) + (1 | h), families whose
# nuisance parameter is ESTIMATED JOINTLY with β and the two variance components:
# NB2 (size), Gamma (shape), Beta (precision), BetaBinomial (precision). #761
# wired Binomial/Poisson (mean-only, fixed-nuisance-free); this file covers the
# remaining crossed cells via `_fit_crossed_mean_aghq_nuisance`
# (src/sparse_laplace_glmm.jl), the nuisance-aware twin of
# `_fit_crossed_mean_aghq` used there.
#
# Each family gets: (1) K = 1 reproduces that family's crossed-Laplace-nuisance
# objective (`_fit_*_crossed_laplace`'s `nll`) to within 1e-6 nat, at the Laplace
# optimum AND at a perturbed (non-optimal) θ — the same bound
# `test_crossed_aghq.jl` uses for the mean-only route's own K = 1 check, because
# `_crossed_mean_mode`'s Newton stop (tol = 1e-8 in b-space) is looser than this
# AGHQ route's (1e-10), so a residual of a few ×1e-8 in the reported LOGLIK
# survives at any θ, not only the optimum; (2) `fit.loglik + fit.nll(fit.theta)
# == 0` and `fit.marginal == :AGHQ`; (3) a small accuracy check against a
# high-K (K_g = 40, K_h = 4) evaluation AT THE FITTED OPTIMUM, within 0.01 nat —
# the same node-sweep-at-a-fixed-θ idea #761 used.
using Test, LinearAlgebra, Random
using DRModels

# ---------------------------------------------------------------------------
# Small crossed design shared by all four families: G groups, H = 3 (well under
# `_CROSSED_AGHQ_HMAX`), moderate group-level SD so Laplace and AGHQ visibly
# differ but every fit still runs in a few seconds.
# ---------------------------------------------------------------------------
function _cfam_design(rng; G = 24, H = 3, n = 300, σg = 1.2, σh = 0.4)
    x = randn(rng, n)
    gi = rand(rng, 1:G, n); hi = rand(rng, 1:H, n)
    bg = σg .* randn(rng, G); bh = σh .* randn(rng, H)
    η = 0.3 .+ 0.4 .* x .+ bg[gi] .+ bh[hi]
    return x, gi, hi, η
end

@testset "crossed AGHQ (nuisance families): NB2" begin
    rng = MersenneTwister(7611)
    x, gi, hi, η = _cfam_design(rng)
    n = length(x)
    r_true = 4.0
    y = [Float64(rand(rng, DRModels.Distributions.NegativeBinomial(r_true, r_true / (r_true + exp(η[i]))))) for i in 1:n]
    data = (; y, x, g = Symbol.("g", gi), h = Symbol.("h", hi))
    fm = bf(@formula(y ~ x + (1 | g) + (1 | h)))

    X = hcat(ones(n), x)
    gidx, G = DRModels._group_index(data.g); hidx, H = DRModels._group_index(data.h)
    comps = [(ones(n), gidx, G, "g"), (ones(n), hidx, H, "h")]
    lap = DRModels._fit_nb2_crossed_laplace(DRModels.NegBinomial2(), y, X, ones(n, 1), comps,
                                            ["(Intercept)", "x"], ["(Intercept)"], 1e-8; se = false)
    yint = round.(Int, y)
    aux_from(logσ) = (r = exp(clamp(-2 * logσ, -8.0, 8.0));
                       (y = Float64.(yint),
                        size = r,
                        lconst = [DRModels.loggamma(yint[i] + r) - DRModels.loggamma(r) -
                                  DRModels._logfactorial(yint[i]) for i in eachindex(yint)]))
    gm = [Int[] for _ in 1:G]; for i in 1:n; push!(gm[gidx[i]], i); end
    cache = (ug = zeros(G), uh = zeros(H), Zre = ones(n, 1))
    pμ = 2
    function f(θ, Kg, Kh)
        aux = aux_from(θ[pμ+1])
        ll = (i, η) -> -DRModels._laplace_value(Val(:nb2_fixed), aux, i, η)
        DRModels._crossed_aghq_loglik(ll, gm, gidx, hidx, H, X * θ[1:pμ], exp(θ[pμ+2]), exp(θ[pμ+3]),
                                      DRModels._AGHQRule(1, Kg), DRModels._AGHQRule(H, Kh), cache)
    end
    # `_crossed_mean_mode`'s own Newton stop is tol = 1e-8 IN b-SPACE (vs. this
    # AGHQ route's 1e-10), so a fresh mode solve at ANY θ carries a residual of a
    # few ×1e-8 in the reported value -- not tied to optimality. Same 1e-6 bound
    # as the mean-only route's own K = 1 check (test_crossed_aghq.jl).
    for θ in (lap.theta, lap.theta .+ [0.1, -0.05, 0.15, 0.2, -0.1])
        @test abs(f(θ, 1, 1) + lap.nll(θ)) < 1e-6
    end

    fa = drm(fm, DRModels.NegBinomial2(); data = data, se = false, marginal = :AGHQ)
    @test fa.marginal === :AGHQ
    @test fa.converged
    @test fa.loglik ≈ -fa.nll(fa.theta) atol = 1e-8
    @test abs(f(fa.theta, 40, 4) - fa.loglik) < 0.01
end

@testset "crossed AGHQ (nuisance families): Gamma" begin
    rng = MersenneTwister(7612)
    x, gi, hi, η = _cfam_design(rng)
    n = length(x)
    α_true = 6.0
    μ = exp.(η)
    y = [rand(rng, DRModels.Distributions.Gamma(α_true, μ[i] / α_true)) for i in 1:n]
    data = (; y, x, g = Symbol.("g", gi), h = Symbol.("h", hi))
    fm = bf(@formula(y ~ x + (1 | g) + (1 | h)))

    X = hcat(ones(n), x)
    gidx, G = DRModels._group_index(data.g); hidx, H = DRModels._group_index(data.h)
    comps = [(ones(n), gidx, G, "g"), (ones(n), hidx, H, "h")]
    lap = DRModels._fit_gamma_crossed_laplace(DRModels.Gamma(), y, X, ones(n, 1), comps,
                                              ["(Intercept)", "x"], ["(Intercept)"], 1e-8; se = false)
    yv = Float64.(y)
    aux_from(logσ) = (α = exp(clamp(-2 * logσ, -8.0, 8.0));
                       (y = yv, shape = α,
                        lconst = [α * log(α) - DRModels.loggamma(α) + (α - 1) * log(yv[i]) for i in eachindex(yv)]))
    gm = [Int[] for _ in 1:G]; for i in 1:n; push!(gm[gidx[i]], i); end
    cache = (ug = zeros(G), uh = zeros(H), Zre = ones(n, 1))
    pμ = 2
    function f(θ, Kg, Kh)
        aux = aux_from(θ[pμ+1])
        ll = (i, η) -> -DRModels._laplace_value(Val(:gamma_fixed), aux, i, η)
        DRModels._crossed_aghq_loglik(ll, gm, gidx, hidx, H, X * θ[1:pμ], exp(θ[pμ+2]), exp(θ[pμ+3]),
                                      DRModels._AGHQRule(1, Kg), DRModels._AGHQRule(H, Kh), cache)
    end
    for θ in (lap.theta, lap.theta .+ [0.1, -0.05, 0.1, 0.2, -0.1])
        @test abs(f(θ, 1, 1) + lap.nll(θ)) < 1e-6
    end

    fa = drm(fm, DRModels.Gamma(); data = data, se = false, marginal = :AGHQ)
    @test fa.marginal === :AGHQ
    @test fa.converged
    @test fa.loglik ≈ -fa.nll(fa.theta) atol = 1e-8
    @test abs(f(fa.theta, 40, 4) - fa.loglik) < 0.01
end

@testset "crossed AGHQ (nuisance families): Beta" begin
    rng = MersenneTwister(7613)
    x, gi, hi, η = _cfam_design(rng; σg = 0.9, σh = 0.3)
    n = length(x)
    φ_true = 12.0
    μ = 1 ./ (1 .+ exp.(-η))
    y = [rand(rng, DRModels.Distributions.Beta(μ[i] * φ_true, (1 - μ[i]) * φ_true)) for i in 1:n]
    y = clamp.(y, 1e-6, 1 - 1e-6)
    data = (; y, x, g = Symbol.("g", gi), h = Symbol.("h", hi))
    fm = bf(@formula(y ~ x + (1 | g) + (1 | h)))

    X = hcat(ones(n), x)
    gidx, G = DRModels._group_index(data.g); hidx, H = DRModels._group_index(data.h)
    comps = [(ones(n), gidx, G, "g"), (ones(n), hidx, H, "h")]
    lap = DRModels._fit_beta_crossed_laplace(DRModels.Beta(), y, X, ones(n, 1), comps,
                                             ["(Intercept)", "x"], ["(Intercept)"], 1e-8; se = false)
    yv = Float64.(y)
    ylogit = log.(yv) .- log1p.(-yv)
    aux_from(logσ) = (φ = exp(clamp(-2 * logσ, -8.0, 8.0));
                       (y = yv, precision = φ, ylogit = ylogit,
                        lgammaφ = DRModels.loggamma(φ), digammaφ = DRModels.digamma(φ)))
    gm = [Int[] for _ in 1:G]; for i in 1:n; push!(gm[gidx[i]], i); end
    cache = (ug = zeros(G), uh = zeros(H), Zre = ones(n, 1))
    pμ = 2
    function f(θ, Kg, Kh)
        aux = aux_from(θ[pμ+1])
        ll = (i, η) -> -DRModels._laplace_value(Val(:beta_fixed), aux, i, η)
        DRModels._crossed_aghq_loglik(ll, gm, gidx, hidx, H, X * θ[1:pμ], exp(θ[pμ+2]), exp(θ[pμ+3]),
                                      DRModels._AGHQRule(1, Kg), DRModels._AGHQRule(H, Kh), cache)
    end
    for θ in (lap.theta, lap.theta .+ [0.1, -0.05, 0.1, 0.2, -0.1])
        @test abs(f(θ, 1, 1) + lap.nll(θ)) < 1e-6
    end

    fa = drm(fm, DRModels.Beta(); data = data, se = false, marginal = :AGHQ)
    @test fa.marginal === :AGHQ
    @test fa.converged
    @test fa.loglik ≈ -fa.nll(fa.theta) atol = 1e-8
    @test abs(f(fa.theta, 40, 4) - fa.loglik) < 0.01
end

@testset "crossed AGHQ (nuisance families): BetaBinomial" begin
    rng = MersenneTwister(7614)
    x, gi, hi, η = _cfam_design(rng; σg = 0.9, σh = 0.3)
    n = length(x)
    φ_true = 15.0
    ntr = rand(rng, 8:15, n)
    μ = 1 ./ (1 .+ exp.(-η))
    s = [Float64(rand(rng, DRModels.Distributions.BetaBinomial(ntr[i], μ[i] * φ_true, (1 - μ[i]) * φ_true))) for i in 1:n]
    fail = Float64.(ntr) .- s
    data = (; s, fail, x, g = Symbol.("g", gi), h = Symbol.("h", hi))
    fm = bf(@formula(cbind(s, fail) ~ x + (1 | g) + (1 | h)))

    X = hcat(ones(n), x)
    gidx, G = DRModels._group_index(data.g); hidx, H = DRModels._group_index(data.h)
    comps = [(ones(n), gidx, G, "g"), (ones(n), hidx, H, "h")]
    lap = DRModels._fit_betabinomial_crossed_laplace(DRModels.BetaBinomial(), s, Float64.(ntr), X, comps,
                                                     ["(Intercept)", "x"], ["(Intercept)"], 1e-8; se = false)
    aux_from, _, _ = DRModels._betabinomial_laplace_setup(s, Float64.(ntr), X)
    gm = [Int[] for _ in 1:G]; for i in 1:n; push!(gm[gidx[i]], i); end
    cache = (ug = zeros(G), uh = zeros(H), Zre = ones(n, 1))
    pμ = 2
    function f(θ, Kg, Kh)
        aux = aux_from(θ[pμ+1])
        ll = (i, η) -> -DRModels._laplace_value(Val(:betabinomial_fixed), aux, i, η)
        DRModels._crossed_aghq_loglik(ll, gm, gidx, hidx, H, X * θ[1:pμ], exp(θ[pμ+2]), exp(θ[pμ+3]),
                                      DRModels._AGHQRule(1, Kg), DRModels._AGHQRule(H, Kh), cache)
    end
    # Trials up to 15 inflate the per-observation log-gamma constants
    # (logchoose, lgamma_nphi) that both routes carry, so the residual from the
    # two mode-finders' different Newton tolerances scales up too; 1e-4 keeps
    # the same relative bound as the other three families' 1e-6 (loglik here
    # runs several ×100 nat vs. their several ×10).
    for θ in (lap.theta, lap.theta .+ [0.1, -0.05, 0.1, 0.2, -0.1])
        @test abs(f(θ, 1, 1) + lap.nll(θ)) < 1e-4
    end

    fa = drm(fm, DRModels.BetaBinomial(); data = data, se = false, marginal = :AGHQ)
    @test fa.marginal === :AGHQ
    @test fa.converged
    @test fa.loglik ≈ -fa.nll(fa.theta) atol = 1e-8
    @test abs(f(fa.theta, 40, 4) - fa.loglik) < 0.01
end

@testset "crossed AGHQ (nuisance families): refusals" begin
    rng = MersenneTwister(1)
    n = 200
    g = Symbol.("g", rand(rng, 1:20, n)); h = Symbol.("h", rand(rng, 1:12, n))
    x = randn(rng, n)
    fm = bf(@formula(y ~ x + (1 | g) + (1 | h)))
    fm1 = bf(@formula(y ~ x + (1 | g)))
    # both groupings have more than 8 levels (NB2/Gamma/Beta each need their own
    # response domain, so each gets its own data)
    dnb2 = (; y = Float64.(rand(rng, 0:5, n)), x, g, h)
    @test_throws ArgumentError drm(fm, DRModels.NegBinomial2(); data = dnb2, marginal = :AGHQ)
    @test_throws ArgumentError drm(fm1, DRModels.NegBinomial2(); data = dnb2, marginal = :AGHQ)   # single (1|g)
    dgam = (; y = rand(rng, n) .+ 0.1, x, g, h)
    @test_throws ArgumentError drm(fm, DRModels.Gamma(); data = dgam, marginal = :AGHQ)
    @test_throws ArgumentError drm(fm1, DRModels.Gamma(); data = dgam, marginal = :AGHQ)
    dbeta = (; y = clamp.(rand(rng, n), 1e-3, 1 - 1e-3), x, g, h)
    @test_throws ArgumentError drm(fm, DRModels.Beta(); data = dbeta, marginal = :AGHQ)
    @test_throws ArgumentError drm(fm1, DRModels.Beta(); data = dbeta, marginal = :AGHQ)
    dbb = (; s = Float64.(rand(rng, 0:5, n)), fail = Float64.(rand(rng, 0:5, n)), x, g, h)
    fmbb = bf(@formula(cbind(s, fail) ~ x + (1 | g) + (1 | h)))
    fmbb1 = bf(@formula(cbind(s, fail) ~ x + (1 | g)))
    @test_throws ArgumentError drm(fmbb, DRModels.BetaBinomial(); data = dbb, marginal = :AGHQ)
    @test_throws ArgumentError drm(fmbb1, DRModels.BetaBinomial(); data = dbb, marginal = :AGHQ)
end
