# test_ll_sanity_fuzzer.jl — permanent property test for a bug CLASS: a route
# returning a wrong, often spuriously HIGH, log-likelihood at extreme
# parameters (catastrophic cancellation, prior evaluated off its support,
# non-adaptive quadrature blowing up). Key invariant: for a DISCRETE response
# (every family here except the continuous ones swept for comparison), every
# probability MASS is ≤ 1, so the marginal log-likelihood of any random-effects
# model is ≤ 0 too (it is an average, over the random effect, of quantities
# ≤ 1). A fitted or evaluated logLik > 0 (beyond ~1e-8) on a discrete response
# is therefore a PROVEN bug, not a judgment call. Independently of that bound,
# the objective must never be NaN, and never -Inf as an nll (== +Inf logLik).
#
# Sweep design: for each route we fit a tiny (n ≤ 60) dataset once (giving a
# converged θ̂ and its `fit.nll` closure — the same objects other tests obtain
# via `fit.nll(...)`, see e.g. test_poisson_phylo_laplace.jl), then perturb ONE
# coordinate of θ̂ at a time across a grid spanning every range the task asked
# for (variance-component log-SD, correlation log-Cholesky entries, fixed
# effects, nuisance log-params) while holding the rest at their fitted values.
# A full Cartesian product across every parameter would be combinatorially
# enormous for routes with 5-6 parameters and adds little signal beyond what a
# route's actual failure modes look like (per-parameter cancellations, not
# products of several extremes at once); three fully-extreme constant vectors
# per route are added on top for cheap interaction coverage.
#
# Violations found on origin/main (2026-09-27) were first re-asserted here as
# `@test_broken` (naming the route/θ/value). All 22 such cells are now FIXED and
# are plain `@test` cells again (batch-2 landing, 2026-09-29; the whole sweep
# re-run on the batch-2 branch reported zero violations):
#
#   - NegBinomial2 (1|g), `sigma` at log σ ≈ -20..-50 (NB2 dispersion
#     r = exp(-2σ) ~1e17-1e43 gave a NaN logpdf): fixed by #846.
#   - NegBinomial2 + hu~1 and TruncatedNegBinomial2 + hu~1, the same `sigma`
#     NaN on the fixed-effects path: fixed by #871/#874.
#   - CumulativeLogit (1|id), extreme fixed effects / cutpoint deltas / RE
#     log-SD gave NaN: fixed by #873/#877.
#
# The `_OPEN_BROKEN` / `_FIXED_BY` mechanism is kept (empty) so a future
# known-but-unfixed violation can be recorded the same way, while any NEW
# violation stays a hard `@test` failure.
using DRModels
using Test, Random, LinearAlgebra
import Distributions

const _GRID = (-50.0, -30.0, -20.0, -15.0, -5.0, 0.0, 3.0, 8.0, 20.0, 50.0)
const _ALLEXTREME = (-30.0, -5.0, 8.0)   # fully-extreme constant-θ vectors
const _logis(η) = 1 / (1 + exp(-η))

# `(route label, coordinate index, grid value)` cells that are KNOWN, currently
# -open violations on origin/main. `idx == 0` marks one of the `_ALLEXTREME`
# constant-vector checks (keyed by its grid value). Kept out of the plain
# `@test` sweep below and re-asserted individually as `@test_broken`.
const _OPEN_BROKEN = Set{Tuple{String,Int,Float64}}()

# Same shape, for a cell an OPEN pr already fixes on its branch (verified by
# checkout + re-run, see file header). Kept `@test_broken` here (not flipped to
# `@test`) until that PR actually merges into main.
const _FIXED_BY = Dict{Tuple{String,Int,Float64},String}()

# A throw at an extreme θ (DomainError/AssertionError instead of a finite-or-
# +Inf nll) is itself an instance of this bug class, so it is folded into the
# same NaN-shaped check rather than erroring the testset out.
_safe_nll(nllfun, θ) = try
    nllfun(θ)
catch
    NaN
end

function _assert_cell(label::String, key::Tuple{String,Int,Float64}, v::Real; discrete::Bool)
    ok = !isnan(v) && v != -Inf && (!discrete || -v <= 1e-8)
    if haskey(_FIXED_BY, key)
        @test_broken ok   # comment: $(label) θ$(key[2])=$(key[3]) — fixed by $(_FIXED_BY[key]), not yet on main
    elseif key in _OPEN_BROKEN
        @test_broken ok   # comment: $(label) θ$(key[2])=$(key[3]) -> $(v) (OPEN, see file header)
    else
        @test ok
    end
end

# Sweep one route's `nllfun = fit.nll`, `θ0 = fit.theta`.
function sweep_route(label::String, nllfun, θ0::Vector{Float64}; discrete::Bool = true)
    p = length(θ0)
    @testset "$label" begin
        @test !isnan(_safe_nll(nllfun, θ0))
        for i in 1:p, g in _GRID
            θ = copy(θ0); θ[i] = g
            v = _safe_nll(nllfun, θ)
            _assert_cell(label, (label, i, g), v; discrete = discrete)
        end
        for g in _ALLEXTREME
            θ = fill(g, p)
            v = _safe_nll(nllfun, θ)
            _assert_cell(label, (label, 0, g), v; discrete = discrete)
        end
    end
end

@testset "Likelihood sanity fuzzer — discrete logLik ≤ 0, no NaN, no -Inf nll" begin

    # ================= Poisson =================
    @testset "Poisson" begin
        Random.seed!(1); n = 40; x = randn(n)
        y = Float64.(rand.(Distributions.Poisson.(exp.(0.3 .+ 0.4 .* x))))
        fit = drm(bf(@formula(y ~ x)), Poisson(); data = (; y, x), se = false)
        sweep_route("Poisson fixed", fit.nll, fit.theta)

        Random.seed!(2); G = 6; m = 8; n = G * m
        g = repeat(1:G, inner = m); x = randn(n)
        bg = 0.3 .* randn(G)
        y = Float64.(rand.(Distributions.Poisson.(exp.(0.3 .+ 0.4 .* x .+ bg[g]))))
        fit = drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = (; y, x, g), se = false)
        sweep_route("Poisson (1|g)", fit.nll, fit.theta)

        Random.seed!(3); G = 6; m = 8; n = G * m
        g = repeat(1:G, inner = m); x = randn(n)
        b = [0.25 .* randn(G) 0.15 .* randn(G)]
        y = Float64.([rand(Distributions.Poisson(exp(0.3 + 0.4x[k] + b[g[k], 1] + b[g[k], 2] * x[k]))) for k in 1:n])
        fit = drm(bf(@formula(y ~ x + (1 + x | g))), Poisson(); data = (; y, x, g), se = false)
        sweep_route("Poisson (1+x|g)", fit.nll, fit.theta)

        # Crossed random intercepts: public formula routing for Poisson isn't
        # wired yet (per test_poisson_crossed_laplace.jl), so we call the
        # internal fitter directly, exactly as that test does.
        Random.seed!(4); Gc = 5; Hc = 4; nc = 40
        gg = rand(1:Gc, nc); hh = rand(1:Hc, nc); xx = randn(nc)
        bgg = 0.3 .* randn(Gc); bhh = 0.25 .* randn(Hc)
        yy = Float64.([rand(Distributions.Poisson(exp(0.2 + 0.3xx[i] + bgg[gg[i]] + bhh[hh[i]]))) for i in 1:nc])
        X = hcat(ones(nc), xx)
        gidx, Gfit = DRModels._group_index(gg)
        hidx, Hfit = DRModels._group_index(hh)
        comps = [(ones(nc), gidx, Gfit, "g"), (ones(nc), hidx, Hfit, "h")]
        fitx = DRModels._fit_poisson_crossed_laplace(DRModels.Poisson(), yy, X, comps, ["(Intercept)", "x"], 1e-7)
        sweep_route("Poisson crossed (1|g)+(1|h)", fitx.nll, fitx.theta)

        Random.seed!(5); p = 8; phy = random_balanced_tree(p; branch_length = 0.2)
        species = repeat(1:p, inner = 5); n = length(species); x = randn(n)
        C = sigma_phy_dense(phy; σ²_phy = 0.4^2)
        u = cholesky(Symmetric(C)).L * randn(p)
        y = Float64.([rand(Distributions.Poisson(exp(0.2 + 0.3x[i] + u[species[i]]))) for i in 1:n])
        fit = drm(bf(@formula(y ~ x + phylo(1 | species))), Poisson();
                  data = (; y, x, species), tree = phy, se = false)
        sweep_route("Poisson phylo(1|species)", fit.nll, fit.theta)

        Random.seed!(6); G = 8; m = 6; n = G * m
        pos = rand(G, 2) .* 6.0
        D = [sqrt(sum(abs2, pos[k, :] .- pos[l, :])) for k in 1:G, l in 1:G]
        Cr = exp.(-D ./ 0.8) + 1e-8I
        d = sqrt.(diag(Cr)); Cr = Symmetric(Cr ./ (d * d'))
        id = repeat(1:G, inner = m); x = randn(n)
        u = 0.4 .* (cholesky(Cr).L * randn(G))
        y = Float64.([rand(Distributions.Poisson(exp(0.2 + 0.3x[i] + u[id[i]]))) for i in 1:n])
        fit = drm(bf(@formula(y ~ x + relmat(1 | id))), Poisson();
                  data = (; y, x, id), K = Matrix(Cr), se = false)
        sweep_route("Poisson relmat(1|id)", fit.nll, fit.theta)

        Random.seed!(7); n = 60; x = randn(n)
        πz = _logis(-0.4); λ = exp.(0.5 .+ 0.3 .* x)
        y = Float64.([rand() < πz ? 0 : rand(Distributions.Poisson(λ[i])) for i in 1:n])
        fit = drm(bf(@formula(y ~ x), @formula(zi ~ 1)), Poisson(); data = (; y, x), se = false)
        sweep_route("Poisson + zi~1", fit.nll, fit.theta)

        Random.seed!(8); n = 60; x = randn(n)
        πz = _logis(0.4); λ = exp.(0.5 .+ 0.3 .* x)
        rtpois(l) = (while true; k = rand(Distributions.Poisson(l)); k > 0 && return k; end)
        y = Float64.([rand() < πz ? 0 : rtpois(λ[i]) for i in 1:n])
        fit = drm(bf(@formula(y ~ x), @formula(hu ~ 1)), Poisson(); data = (; y, x), se = false)
        sweep_route("Poisson + hu~1", fit.nll, fit.theta)
    end

    # ================= Binomial =================
    @testset "Binomial" begin
        Random.seed!(9); n = 50; x = randn(n); ntr = fill(10, n)
        μ = _logis.(0.2 .+ 0.4 .* x)
        s = Float64.([rand(Distributions.Binomial(ntr[i], μ[i])) for i in 1:n])
        fail = Float64.(ntr) .- s
        fit = drm(bf(@formula(cbind(s, fail) ~ x)), Binomial(); data = (; s, fail, x), se = false)
        sweep_route("Binomial fixed", fit.nll, fit.theta)

        Random.seed!(10); G = 6; m = 8; n = G * m
        g = repeat(1:G, inner = m); x = randn(n); ntr = fill(6, n)
        bg = 0.3 .* randn(G)
        μ = _logis.(0.2 .+ 0.4 .* x .+ bg[g])
        s = Float64.([rand(Distributions.Binomial(ntr[i], μ[i])) for i in 1:n])
        fail = Float64.(ntr) .- s
        fit = drm(bf(@formula(cbind(s, fail) ~ x + (1 | g))), Binomial(); data = (; s, fail, x, g), se = false)
        sweep_route("Binomial (1|g)", fit.nll, fit.theta)

        Random.seed!(11); p = 8; phy = random_balanced_tree(p; branch_length = 0.2)
        species = repeat(1:p, inner = 5); n = length(species); x = randn(n); ntr = fill(6, n)
        C = sigma_phy_dense(phy; σ²_phy = 0.4^2); u = cholesky(Symmetric(C)).L * randn(p)
        μ = _logis.(0.2 .+ 0.3 .* x .+ u[species])
        s = Float64.([rand(Distributions.Binomial(ntr[i], μ[i])) for i in 1:n])
        fail = Float64.(ntr) .- s
        fit = drm(bf(@formula(cbind(s, fail) ~ x + phylo(1 | species))), Binomial();
                  data = (; s, fail, x, species), tree = phy, se = false)
        sweep_route("Binomial phylo(1|species)", fit.nll, fit.theta)
    end

    # ================= NegBinomial2 =================
    @testset "NegBinomial2" begin
        Random.seed!(12); n = 50; x = randn(n); θd = 3.0
        μ = exp.(0.3 .+ 0.4 .* x)
        y = Float64.([rand(Distributions.NegativeBinomial(θd, θd / (θd + μ[i]))) for i in 1:n])
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), NegBinomial2(); data = (; y, x), se = false)
        sweep_route("NegBinomial2 fixed", fit.nll, fit.theta)

        Random.seed!(13); G = 6; m = 8; n = G * m
        g = repeat(1:G, inner = m); x = randn(n); θd = 3.0
        bg = 0.25 .* randn(G)
        μ = exp.(0.3 .+ 0.4 .* x .+ bg[g])
        y = Float64.([rand(Distributions.NegativeBinomial(θd, θd / (θd + μ[i]))) for i in 1:n])
        fit = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), NegBinomial2();
                  data = (; y, x, g), se = false)
        sweep_route("NegBinomial2 (1|g)", fit.nll, fit.theta)

        Random.seed!(14); p = 8; phy = random_balanced_tree(p; branch_length = 0.2)
        species = repeat(1:p, inner = 5); n = length(species); x = randn(n); θd = 3.0
        C = sigma_phy_dense(phy; σ²_phy = 0.4^2); u = cholesky(Symmetric(C)).L * randn(p)
        μ = exp.(0.3 .+ 0.4 .* x .+ u[species])
        y = Float64.([rand(Distributions.NegativeBinomial(θd, θd / (θd + μ[i]))) for i in 1:n])
        fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)), NegBinomial2();
                  data = (; y, x, species), tree = phy, se = false)
        sweep_route("NegBinomial2 phylo(1|species)", fit.nll, fit.theta)

        Random.seed!(15); n = 60; x = randn(n); θd = 3.0; πz = 0.3
        μ = exp.(0.3 .+ 0.4 .* x)
        y = Float64.([rand() < πz ? 0 : rand(Distributions.NegativeBinomial(θd, θd / (θd + μ[i]))) for i in 1:n])
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(zi ~ 1)), NegBinomial2();
                  data = (; y, x), se = false)
        sweep_route("NegBinomial2 + zi~1", fit.nll, fit.theta)

        Random.seed!(16); n = 60; x = randn(n); θd = 3.0; πz = 0.35
        rtnb(r, pp) = (while true; k = rand(Distributions.NegativeBinomial(r, pp)); k > 0 && return k; end)
        μ = exp.(0.4 .+ 0.3 .* x)
        y = Float64.([rand() < πz ? 0 : rtnb(θd, θd / (θd + μ[i])) for i in 1:n])
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(hu ~ 1)), NegBinomial2();
                  data = (; y, x), se = false)
        sweep_route("NegBinomial2 + hu~1", fit.nll, fit.theta)
    end

    # ================= TruncatedNegBinomial2 (fixed + hu only) =================
    @testset "TruncatedNegBinomial2" begin
        Random.seed!(17); n = 60; x = randn(n); θd = 3.0; πz = 0.35
        μ = exp.(0.4 .+ 0.3 .* x)
        rtnb(r, pp) = (while true; k = rand(Distributions.NegativeBinomial(r, pp)); k > 0 && return k; end)
        y = Float64.([rand() < πz ? 0 : rtnb(θd, θd / (θd + μ[i])) for i in 1:n])
        # No `se = false` kwarg on this constructor's `drm` method (fixed-effects
        # only route; matches test_hurdle.jl's own call).
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(hu ~ 1)), TruncatedNegBinomial2();
                  data = (; y, x))
        sweep_route("TruncatedNegBinomial2 + hu~1", fit.nll, fit.theta)
    end

    # ================= BetaBinomial =================
    @testset "BetaBinomial" begin
        Random.seed!(18); n = 50; x = randn(n); ntr = fill(8, n); prec = 15.0
        μ = _logis.(0.1 .+ 0.4 .* x)
        s = Float64.([rand(Distributions.BetaBinomial(ntr[i], μ[i] * prec, (1 - μ[i]) * prec)) for i in 1:n])
        fail = Float64.(ntr) .- s
        fit = drm(bf(@formula(cbind(s, fail) ~ x), @formula(sigma ~ 1)), BetaBinomial();
                  data = (; s, fail, x), se = false)
        sweep_route("BetaBinomial fixed", fit.nll, fit.theta)

        Random.seed!(19); G = 6; m = 8; n = G * m
        g = repeat(1:G, inner = m); x = randn(n); ntr = fill(6, n); prec = 15.0
        bg = 0.25 .* randn(G)
        μ = _logis.(0.1 .+ 0.4 .* x .+ bg[g])
        s = Float64.([rand(Distributions.BetaBinomial(ntr[i], μ[i] * prec, (1 - μ[i]) * prec)) for i in 1:n])
        fail = Float64.(ntr) .- s
        fit = drm(bf(@formula(cbind(s, fail) ~ x + (1 | g)), @formula(sigma ~ 1)), BetaBinomial();
                  data = (; s, fail, x, g), se = false)
        sweep_route("BetaBinomial (1|g)", fit.nll, fit.theta)

        Random.seed!(20); G = 6; m = 8; n = G * m
        g = repeat(1:G, inner = m); x = randn(n); ntr = fill(6, n); prec = 15.0
        b = [0.2 .* randn(G) 0.15 .* randn(G)]
        μ = [_logis(0.1 + 0.4x[k] + b[g[k], 1] + b[g[k], 2] * x[k]) for k in 1:n]
        s = Float64.([rand(Distributions.BetaBinomial(ntr[i], μ[i] * prec, (1 - μ[i]) * prec)) for i in 1:n])
        fail = Float64.(ntr) .- s
        fit = drm(bf(@formula(cbind(s, fail) ~ x + (1 + x | g)), @formula(sigma ~ 1)), BetaBinomial();
                  data = (; s, fail, x, g), se = false)
        sweep_route("BetaBinomial (1+x|g)", fit.nll, fit.theta)

        Random.seed!(21); Gc = 5; Hc = 4; nc = 40; prec = 15.0
        gg = rand(1:Gc, nc); hh = rand(1:Hc, nc); xx = randn(nc); ntr = fill(6, nc)
        bgg = 0.2 .* randn(Gc); bhh = 0.15 .* randn(Hc)
        gsym = [Symbol("g", j) for j in gg]; hsym = [Symbol("h", j) for j in hh]
        μ = [_logis(0.1 + 0.3xx[i] + bgg[gg[i]] + bhh[hh[i]]) for i in 1:nc]
        s = Float64.([rand(Distributions.BetaBinomial(ntr[i], μ[i] * prec, (1 - μ[i]) * prec)) for i in 1:nc])
        fail = Float64.(ntr) .- s
        fit = drm(bf(@formula(cbind(s, fail) ~ x + (1 | g) + (1 | h)), @formula(sigma ~ 1)),
                  BetaBinomial(); data = (; s, fail, x = xx, g = gsym, h = hsym), se = false)
        sweep_route("BetaBinomial crossed (1|g)+(1|h)", fit.nll, fit.theta)

        Random.seed!(22); p = 8; phy = random_balanced_tree(p; branch_length = 0.2)
        species = repeat(1:p, inner = 5); n = length(species); x = randn(n); ntr = fill(6, n); prec = 15.0
        C = sigma_phy_dense(phy; σ²_phy = 0.35^2); u = cholesky(Symmetric(C)).L * randn(p)
        μ = _logis.(0.1 .+ 0.3 .* x .+ u[species])
        s = Float64.([rand(Distributions.BetaBinomial(ntr[i], μ[i] * prec, (1 - μ[i]) * prec)) for i in 1:n])
        fail = Float64.(ntr) .- s
        fit = drm(bf(@formula(cbind(s, fail) ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
                  BetaBinomial(); data = (; s, fail, x, species), tree = phy, se = false)
        sweep_route("BetaBinomial phylo(1|species)", fit.nll, fit.theta)
    end

    # ================= CumulativeLogit (ordinal) =================
    @testset "CumulativeLogit" begin
        _ordinalize(η; cuts = (-0.5, 0.6)) = η <= cuts[1] ? 1.0 : (η <= cuts[2] ? 2.0 : 3.0)

        Random.seed!(23); n = 60; x = randn(n)
        η = 0.5 .* x .+ 0.3 .* randn(n)
        y = _ordinalize.(η)
        fit = drm(bf(@formula(y ~ x)), CumulativeLogit(); data = (; y, x), se = false)
        sweep_route("CumulativeLogit fixed", fit.nll, fit.theta)

        Random.seed!(24); G = 6; m = 8; n = G * m
        id = repeat(1:G, inner = m); x = randn(n)
        bg = 0.5 .* randn(G)
        η = 0.5 .* x .+ bg[id] .+ 0.2 .* randn(n)
        y = _ordinalize.(η)
        fit = drm(bf(@formula(y ~ x + (1 | id))), CumulativeLogit(); data = (; y, x, id), se = false)
        sweep_route("CumulativeLogit (1|id)", fit.nll, fit.theta)

        Random.seed!(25); p = 8; phy = random_balanced_tree(p; branch_length = 0.2)
        species = repeat(1:p, inner = 5); n = length(species); x = randn(n)
        C = sigma_phy_dense(phy; σ²_phy = 0.6^2); u = cholesky(Symmetric(C)).L * randn(p)
        η = 0.5 .* x .+ u[species] .+ 0.2 .* randn(n)
        y = _ordinalize.(η)
        fit = drm(bf(@formula(y ~ x + phylo(1 | species))), CumulativeLogit();
                  data = (; y, x, species), tree = phy, se = false)
        sweep_route("CumulativeLogit phylo(1|species)", fit.nll, fit.theta)
    end
end
