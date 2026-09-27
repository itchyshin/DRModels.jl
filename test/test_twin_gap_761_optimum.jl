# Follow-up to issue #761: a crossed Binomial GLMM `(1 | g) + (1 | h)` could
# report `converged = true` at a log-likelihood BELOW its own nested
# single-grouping fit `(1 | g)`. At a true maximum this is impossible -- the
# nested model is exactly the h-SD -> 0 boundary of the bigger crossed model,
# so a genuine maximum of the crossed marginal likelihood can never be worse.
#
# Root cause: `_fit_binomial_crossed_laplace` (src/sparse_laplace_glmm.jl)
# integrates out BOTH random intercepts with a first-order Laplace
# approximation, while the single-grouping `(1 | g)` route (`_fit_binomial_ranef`,
# src/binomial.jl) integrates out its one random intercept with 32-node
# Gauss-Hermite quadrature -- a substantially more accurate approximation.
# Multi-start experiments (naive cold start `log(0.4)`/`log(0.4)`, a
# data-driven start embedding the nested GHQ optimum, and 10x the iteration
# budget) all land on the SAME crossed-Laplace optimum to 5+ significant
# figures, which rules out a local-optimum / bad-start bug in THIS optimizer:
# the ordering violation is a genuine Laplace-vs-GHQ accuracy gap, not an
# optimizer failing to find its own objective's minimum.
#
# Fix: `_fit_crossed_mean_laplace` (the shared crossed-mean Laplace driver,
# also used by the NB2/Gamma/Beta *-fixed crossed routes) now accepts
# `nested_candidates` -- boundary submodels embedded into its θ layout
# ([βμ; logσ_g; logσ_h]) -- and keeps whichever of {its own free optimum, a
# candidate} has the higher log-likelihood. `_fit_binomial_crossed_laplace`
# supplies two such candidates: the GHQ-32 fit of EACH grouping alone (via
# `_fit_binomial_ranef`), embedding the dropped grouping's log-SD at the same
# floor `eval_laplace` already clamps to (-8.0). A single-level grouping
# (fully aliased with the intercept) is never offered as a candidate.
using DRModels
using Test
using Random
using LinearAlgebra
import Distributions

logistic(x) = 1 / (1 + exp(-x))

function _crossed_components(g, h)
    gidx, G = DRModels._group_index(g)
    hidx, H = DRModels._group_index(h)
    return [
        (ones(length(g)), gidx, G, "g"),
        (ones(length(h)), hidx, H, "h"),
    ]
end

# A Bernoulli (ntr = 1) design with a SIZEABLE true grouping-g variance and a
# near-zero true grouping-h variance -- the shape that produced a ~17
# log-likelihood-unit violation before the fix (large σ_g makes the Laplace
# approximation's accuracy gap versus GHQ-32 large; h ≈ 0 puts the crossed
# model right at the boundary where it must reduce to the nested model).
function _boundary_data(seed)
    rng = MersenneTwister(seed)
    G, H, n = 300, 4, 1600
    x = randn(rng, n)
    g = [rand(rng, 1:G) for _ in 1:n]
    h = [rand(rng, 1:H) for _ in 1:n]
    β = [0.2, 0.45]; σg = 2.5; σh = 1e-3
    bg = σg .* randn(rng, G)
    bh = σh .* randn(rng, H)
    η = [β[1] + β[2] * x[i] + bg[g[i]] + bh[h[i]] for i in 1:n]
    p = logistic.(η)
    s = Float64.(rand.(rng, Distributions.Bernoulli.(p)))
    ntr = ones(n)
    gsym = [Symbol("g", gi) for gi in g]
    hsym = [Symbol("h", hi) for hi in h]
    X = hcat(ones(n), x)
    comps = _crossed_components(gsym, hsym)
    return s, ntr, X, comps, gsym, hsym, x
end

@testset "issue #761 follow-up: crossed fit is never below its own nested fit" begin

    @testset "crossed logLik >= nested (1|g) logLik, at the parameter level" begin
        s, ntr, X, comps, gsym, hsym, x = _boundary_data(1)

        fit_nested = DRModels._fit_binomial_ranef(
            DRModels.Binomial(), s, ntr, X, comps[1][2], comps[1][3], ["(Intercept)", "x"], :g, 1e-7)
        @test fit_nested.converged

        fit_crossed = DRModels._fit_binomial_crossed_laplace(
            DRModels.Binomial(), s, ntr, X, comps, ["(Intercept)", "x"], 1e-7)
        @test fit_crossed.converged

        # The literal invariant: a converged crossed fit must never be
        # reported below its converged nested fit (up to solver tolerance).
        @test fit_crossed.loglik >= fit_nested.loglik - 1e-8

        # On this fixture the fix makes them coincide exactly: the more
        # accurate GHQ-32 nested candidate is what the crossed route reports.
        @test fit_crossed.loglik ≈ fit_nested.loglik atol = 1e-6

        se = stderror(fit_crossed)
        @test all(isfinite, se[1:3])     # β and the identified σ_g are finite
        @test !isfinite(se[4])           # σ_h was never estimated -- Inf, not silently finite
    end

    @testset "crossed >= nested holds across several seeds (not a single lucky fixture)" begin
        for seed in 2:5
            s, ntr, X, comps, gsym, hsym, x = _boundary_data(seed)
            fit_nested = DRModels._fit_binomial_ranef(
                DRModels.Binomial(), s, ntr, X, comps[1][2], comps[1][3], ["(Intercept)", "x"], :g, 1e-7)
            fit_crossed = DRModels._fit_binomial_crossed_laplace(
                DRModels.Binomial(), s, ntr, X, comps, ["(Intercept)", "x"], 1e-7)
            @test fit_nested.converged
            @test fit_crossed.converged
            @test fit_crossed.loglik >= fit_nested.loglik - 1e-8
        end
    end

    @testset "drm() top-level default reaches the same fix" begin
        s, ntr, X, comps, gsym, hsym, x = _boundary_data(1)
        fail = ntr .- s
        data = (; s, fail, x, g = gsym, h = hsym)

        fit_nested = drm(bf(@formula(cbind(s, fail) ~ x + (1 | g))), Binomial(); data = data)
        fit_crossed = drm(bf(@formula(cbind(s, fail) ~ x + (1 | g) + (1 | h))), Binomial(); data = data)
        @test fit_nested.converged
        @test fit_crossed.converged
        @test fit_crossed.loglik >= fit_nested.loglik - 1e-8
    end

    @testset "a genuinely richer crossed model still beats its nested submodel" begin
        # Sanity check: the fix must not clamp crossed fits DOWN to the
        # nested candidate when h's variance is real and sizeable -- only
        # keep the candidate when it is actually the better (higher-loglik)
        # point on the same likelihood surface.
        rng = MersenneTwister(11)
        G, H, n = 28, 24, 2400
        x = randn(rng, n)
        g = [Symbol("g", rand(rng, 1:G)) for _ in 1:n]
        h = [Symbol("h", rand(rng, 1:H)) for _ in 1:n]
        gmap = Dict(Symbol("g", j) => j for j in 1:G)
        hmap = Dict(Symbol("h", j) => j for j in 1:H)
        X = hcat(ones(n), x)
        comps = _crossed_components(g, h)
        β = [0.2, 0.45]; σg = 0.45; σh = 0.6
        bg = σg .* randn(rng, G)
        bh = σh .* randn(rng, H)
        η = [β[1] + β[2] * x[i] + bg[gmap[g[i]]] + bh[hmap[h[i]]] for i in 1:n]
        ntr = fill(8.0, n)
        s = Float64.([rand(rng, Distributions.Binomial(round(Int, ntr[i]), logistic(η[i]))) for i in 1:n])

        fit_nested = DRModels._fit_binomial_ranef(
            DRModels.Binomial(), s, ntr, X, comps[1][2], comps[1][3], ["(Intercept)", "x"], :g, 1e-7)
        fit_crossed = DRModels._fit_binomial_crossed_laplace(
            DRModels.Binomial(), s, ntr, X, comps, ["(Intercept)", "x"], 1e-7)
        @test fit_crossed.converged
        @test fit_crossed.loglik > fit_nested.loglik + 1.0   # genuinely richer, not clamped to nested
        @test all(isfinite, stderror(fit_crossed))
    end
end
