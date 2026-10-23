# Regression for issue #761 -- a two-random-intercept-grouping Binomial GLMM
# ("crossed"/partially-crossed, e.g. `(1 | BroodNo) + (1 | Year)`) converged and
# reported `[Inf, Inf, Inf, Inf]` standard errors with NO warning at all.
#
# Root cause: `drm(...; se = true)` (the package default) dispatches Binomial's
# two-random-intercept-grouping formula to `_fit_binomial_crossed_laplace`
# (src/binomial.jl), which never forwarded its `se` argument down to the
# sparse-Laplace engine. `_fit_binomial_crossed_laplace`'s OWN keyword default
# in src/sparse_laplace_glmm.jl was `se::Bool = false` -- unlike every sibling
# crossed fitter (`_fit_poisson_crossed_laplace` defaults `se = true`) -- so the
# Hessian/vcov step was skipped entirely and `V` came back `fill(NaN, ...)`,
# which `stderror` (src/inference.jl) reports as `Inf` for every coordinate.
# `_vcov_from_hessian`'s own boundary warning (src/vcov_guard.jl) never had a
# chance to fire because it was never called.
#
# The fix flips `_fit_binomial_crossed_laplace`'s default to `se = true`
# (matching the Poisson/NB2 crossed fitters), so the existing, already-tested
# guard machinery (`_finite_hessian` / `_vcov_from_hessian`) actually runs and
# warns when it should.
using DRModels
using Test
using Random
using LinearAlgebra
using Logging
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

@testset "issue #761: crossed Binomial SEs are silently Inf with no warning" begin

    @testset "well-identified two-grouping fit: SEs are now finite, not silently Inf" begin
        rng = MersenneTwister(70)
        G, H, n = 28, 24, 2400
        x = randn(rng, n)
        g = [Symbol("g", rand(rng, 1:G)) for _ in 1:n]
        h = [Symbol("h", rand(rng, 1:H)) for _ in 1:n]
        gmap = Dict(Symbol("g", j) => j for j in 1:G)
        hmap = Dict(Symbol("h", j) => j for j in 1:H)
        X = hcat(ones(n), x)
        comps = _crossed_components(g, h)

        β = [0.2, 0.45]; σg = 0.45; σh = 0.35
        bg = σg .* randn(rng, G)
        bh = σh .* randn(rng, H)
        η = [β[1] + β[2] * x[i] + bg[gmap[g[i]]] + bh[hmap[h[i]]] for i in 1:n]
        ntr = fill(8.0, n)
        s = Float64.([rand(rng, Distributions.Binomial(round(Int, ntr[i]), logistic(η[i]))) for i in 1:n])

        # Called with NO `se` keyword -- exactly as every family dispatcher in
        # src/binomial.jl calls it -- so this exercises the actual default.
        fit = DRModels._fit_binomial_crossed_laplace(DRModels.Binomial(), s, ntr, X, comps,
                                                      ["(Intercept)", "x"], 1e-7)
        @test fit.converged
        se = stderror(fit)
        @test all(isfinite, se)          # NOT [Inf, Inf, Inf, Inf] as on main
        @test all(>(0), se)
    end

    @testset "genuine non-identifiability: a named warning fires, not a silent Inf" begin
        # H = 1: the second grouping has a single level, so its random
        # intercept is EXACTLY aliased with the fixed intercept -- the same
        # kind of boundary non-identifiability as issue #761's `Year` block
        # ("no brood spans two years"). This must warn, not fail silently.
        rng = MersenneTwister(70)
        G, n = 28, 2400
        x = randn(rng, n)
        g = [Symbol("g", rand(rng, 1:G)) for _ in 1:n]
        h = fill(:h1, n)
        gmap = Dict(Symbol("g", j) => j for j in 1:G)
        X = hcat(ones(n), x)
        comps = _crossed_components(g, h)

        β = [0.2, 0.45]; σg = 0.45
        bg = σg .* randn(rng, G)
        η = [β[1] + β[2] * x[i] + bg[gmap[g[i]]] for i in 1:n]
        ntr = fill(8.0, n)
        s = Float64.([rand(rng, Distributions.Binomial(round(Int, ntr[i]), logistic(η[i]))) for i in 1:n])

        fit = @test_logs (:warn, r"Hessian.*not (positive definite|trustworthy)"i) match_mode = :any DRModels._fit_binomial_crossed_laplace(
            DRModels.Binomial(), s, ntr, X, comps, ["(Intercept)", "x"], 1e-7)

        se = stderror(fit)
        # The identified fixed effects stay finite; the aliased random-intercept
        # coordinate(s) are reported as Inf via the package's existing
        # non-finite-variance convention (src/inference.jl `_boundary_se`) --
        # never a silent finite-looking value.
        @test all(isfinite, se[1:2])
        @test any(!isfinite, se)
    end

    @testset "drm() top-level default (se = true) reaches the same fix" begin
        rng = MersenneTwister(70)
        G, H, n = 28, 24, 2400
        x = randn(rng, n)
        g = [Symbol("g", rand(rng, 1:G)) for _ in 1:n]
        h = [Symbol("h", rand(rng, 1:H)) for _ in 1:n]
        gmap = Dict(Symbol("g", j) => j for j in 1:G)
        hmap = Dict(Symbol("h", j) => j for j in 1:H)
        β = [0.2, 0.45]; σg = 0.45; σh = 0.35
        bg = σg .* randn(rng, G)
        bh = σh .* randn(rng, H)
        η = [β[1] + β[2] * x[i] + bg[gmap[g[i]]] + bh[hmap[h[i]]] for i in 1:n]
        ntr = fill(8.0, n)
        s = Float64.([rand(rng, Distributions.Binomial(round(Int, ntr[i]), logistic(η[i]))) for i in 1:n])
        fail = ntr .- s

        fit = drm(bf(@formula(cbind(s, fail) ~ x + (1 | g) + (1 | h))), Binomial();
                  data = (; s, fail, x, g, h))
        @test fit.converged
        @test all(isfinite, stderror(fit))
    end
end
