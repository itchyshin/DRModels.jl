# Cox–Reid scoping probe for non-Gaussian variance components (issue #441).
#
# SCOPE: a scoping probe, NOT the estimator. It answers three questions with numbers
# measured on THIS engine (no imported drmTMB figures):
#
#   Q1  Does DRModels.jl's non-Gaussian scalar-RE path show the ML finite-cluster
#       variance-component bias the vault predicts, and how big is it here?
#   Q2  Does a generic Cox–Reid adjusted profile likelihood — nll_ML + ½·logdet(I_ββ),
#       the same shape as the wired Gaussian `_glsp_reml_penalty` — remove it?
#   Q3  Does that generic penalty reduce EXACTLY to the validated Gaussian REML
#       (the #440 Woodbury `(1|g)` path)? If yes, the penalty is anchored and the
#       implement-G0 is a wiring job rather than a derivation job.
#
# ML stays the default everywhere; nothing here is wired into the module or the
# test suite. Scalar-per-cluster random effects only.
#
# Run:  julia --project=. bench/cox_reid_probe.jl

using DRModels
using LinearAlgebra
using Random
using Optim
using ForwardDiff
using Printf
using Distributions

const D = DRModels

# ---------------------------------------------------------------------------
# The generic Cox–Reid penalty.
#
# Cox & Reid (1987) adjusted profile likelihood: with β the (orthogonal-ish)
# fixed effects and θ the variance components,
#
#     ℓ_CR(θ) = ℓ_p(θ) − ½ log|I_ββ(θ)|,
#
# i.e. integrate β out under a flat prior by a Laplace approximation. In
# minimisation form the penalty ADDS to the nll. For a Gaussian LMM I_ββ =
# Xμ′V⁻¹Xμ and this is exactly Patterson–Thompson REML — which is why the same
# expression serves both and why Q3 is a real anchor rather than a formality.
#
# `_glsp_reml_penalty` (src/gaussian_locscale_phylo.jl) forms I_ββ by central
# finite differences of the route's ANALYTIC β-gradient block. The simple
# `(1|g)` routes have no analytic gradient closure but their nll is
# ForwardDiff-clean, so here we take the exact AD Hessian instead. Same
# quantity, cleaner where it is available.
# ---------------------------------------------------------------------------
const CR_SENTINEL = 1e18

function cr_penalty(nll, θ, pμ::Int)
    H = ForwardDiff.hessian(nll, θ)
    S = Symmetric(0.5 .* (H[1:pμ, 1:pμ] .+ H[1:pμ, 1:pμ]'))
    ch = cholesky(S; check = false)
    issuccess(ch) || return CR_SENTINEL      # non-PD β-information ⇒ large finite, never Inf
    return sum(log, diag(ch.U))              # = ½·logdet(I_ββ)
end

"""
Refit an ML objective under the Cox–Reid adjusted profile likelihood, warm-started
from the ML estimate. Nelder–Mead: the parameter space here is 3-dimensional and
this avoids nesting AD inside AD (the penalty already consumes a ForwardDiff
Hessian). Returns `(θ̂_cr, converged)`.
"""
function cr_refit(nll, θ̂_ml, pμ::Int)
    obj(θ) = nll(θ) + cr_penalty(nll, θ, pμ)
    res = try
        Optim.optimize(obj, copy(θ̂_ml), Optim.NelderMead(),
                       Optim.Options(g_tol = 1e-10, iterations = 2_000))
    catch err
        err isa InterruptException && rethrow(err)
        return copy(θ̂_ml), false
    end
    θ̂ = Optim.minimizer(res)
    ok = all(isfinite, θ̂) && obj(θ̂) < 1e16 && norm(θ̂ .- θ̂_ml) < 5.0
    return θ̂, ok
end

# ---------------------------------------------------------------------------
# Cell A — Poisson (1|g): the CLEAN isolation of the ML variance-component bias.
#
# Why this cell is the right probe target. DRModels.jl fits a plain Poisson `(1|g)`
# with 32-node Gauss–Hermite quadrature (`_fit_poisson_ranef`, src/poisson.jl),
# NOT a 1-point Laplace. The vault's AGHQ sweep converges by nq≈5 and then
# plateaus, so at 32 nodes the INTEGRAL error is already negligible on this
# route. Whatever downward bias remains in σ̂_b is therefore almost entirely the
# ML finite-cluster variance-component bias — the Cox–Reid lever, measured
# without AGHQ confounding it.
# ---------------------------------------------------------------------------
function simulate_poisson_re(rng, G::Int, n_each::Int, β, σb)
    n = G * n_each
    g = repeat(1:G, inner = n_each)
    x = randn(rng, n)
    b = σb .* randn(rng, G)
    λ = exp.(β[1] .+ β[2] .* x .+ b[g])
    y = Float64.([rand(rng, Distributions.Poisson(λi)) for λi in λ])
    return y, hcat(ones(n), x), g, n
end

function cell_a(; G::Int, n_each::Int, seeds::Int, σb::Float64 = 0.6,
                  β = [0.4, 0.3])
    ml = Float64[]
    cr = Float64[]
    nfail = 0
    for s in 1:seeds
        rng = MersenneTwister(20260818 + 1000G + s)
        y, Xμ, g, _ = simulate_poisson_re(rng, G, n_each, β, σb)
        gidx, ng = D._group_index(g)
        fit = D._fit_poisson_ranef(D.Poisson(), y, Xμ, gidx, ng,
                                   ["(Intercept)", "x"], :g, 1e-10)
        D.is_converged(fit) || (nfail += 1; continue)
        θ̂ = D.coef(fit)
        pμ = size(Xμ, 2)
        θ̂cr, ok = cr_refit(fit.nll, θ̂, pμ)
        ok || (nfail += 1; continue)
        push!(ml, exp(θ̂[pμ+1]))
        push!(cr, exp(θ̂cr[pμ+1]))
    end
    return (; G, n_each, σb, seeds, nfail,
            ml_mean = _mean(ml), cr_mean = _mean(cr),
            ml_bias_pct = 100 * (_mean(ml) - σb) / σb,
            cr_bias_pct = 100 * (_mean(cr) - σb) / σb,
            nrep = length(ml))
end

_mean(v) = isempty(v) ? NaN : sum(v) / length(v)

# ---------------------------------------------------------------------------
# Cell B — the Gaussian reduction anchor (Q3).
#
# Apply the SAME generic `cr_penalty` to a Gaussian `(1|g)` ML objective and
# check it lands on the independently-validated exact REML estimate from the
# #440 Woodbury path (`drm(...; method = :REML)`). If the generic penalty
# reproduces exact REML where exact REML is known, then applying it to a
# non-Gaussian marginal is Cox–Reid rather than an ad-hoc fudge.
# ---------------------------------------------------------------------------
function cell_b(; G::Int = 12, n_each::Int = 5, seed::Int = 4242)
    rng = MersenneTwister(seed)
    n = G * n_each
    g = repeat(1:G, inner = n_each)
    x = randn(rng, n)
    b = 0.8 .* randn(rng, G)
    y = 1.0 .+ 0.5 .* x .+ b[g] .+ 0.7 .* randn(rng, n)
    data = (; y, x, g)
    form = D.bf(D.@formula(y ~ x + (1 | g)))

    fit_ml   = D.drm(form, D.Gaussian(); data = data)
    fit_reml = D.drm(form, D.Gaussian(); data = data, method = :REML)

    θ̂ml = D.coef(fit_ml)
    pμ = length(D.coef(fit_ml, :mu))
    θ̂cr, ok = cr_refit(fit_ml.nll, θ̂ml, pμ)
    θ̂reml = D.coef(fit_reml)
    return (; ok, pμ,
            theta_ml = θ̂ml, theta_cr = θ̂cr, theta_reml = θ̂reml,
            max_abs_diff = maximum(abs.(θ̂cr .- θ̂reml)))
end

# ---------------------------------------------------------------------------
# Cell C — hook viability on the SPARSE LAPLACE route.
#
# Cells A/B use routes whose nll is ForwardDiff-clean. The routes named in #441
# — phylo / relmat / crossed, in src/sparse_laplace_glmm.jl — instead carry an
# EXACT analytic O(p) gradient with β_μ in positions 1:pμ, and `_withnll` already
# stores that gradient on the fit. That is precisely the `(obj, grad_fn, pμ)`
# signature the wired Gaussian `_glsp_reml_penalty` / `_glsp_reml_refit_clean`
# consume. This cell checks that claim by CALLING them, unmodified, against a
# non-Gaussian Laplace fit — the difference between "a wiring job" and "a
# derivation job" for the implement-G0.
# ---------------------------------------------------------------------------
function cell_c(; ntip::Int = 24, per::Int = 4, seed::Int = 909, σphy::Float64 = 0.7)
    rng = MersenneTwister(seed)
    tree = D.random_balanced_tree(ntip; branch_length = 0.25)
    species = repeat(1:ntip, inner = per)
    n = length(species)
    x = randn(rng, n)
    # Draw a genuinely tree-structured effect u ~ N(0, σ² Q⁻¹) from the route's OWN
    # precision, so the fitted model is not misspecified and σ̂ is not pinned at the
    # σ→0 boundary (where a restricted objective has nothing to correct).
    Q, leaf_node, _ = D._poisson_phylo_setup(tree, species)
    L = cholesky(Symmetric(Matrix(Q))).U
    u = σphy .* (L \ randn(rng, size(Q, 1)))
    η = 0.3 .+ 0.25 .* x .+ u[leaf_node]
    y = Float64.([rand(rng, Distributions.Poisson(exp(clamp(ηi, -20, 20)))) for ηi in η])
    data = (; y, x, species)
    form = D.bf(D.@formula(y ~ x + phylo(1 | species)))

    fit = D.drm(form, D.Poisson(); data = data, tree = tree, se = false)
    has_grad = fit.nllgrad !== nothing
    has_grad || return (; has_grad, reached = true)

    θ̂ = D.coef(fit)
    pμ = length(D.coef(fit, :mu))
    grad_fn = θ -> (g = zeros(length(θ)); fit.nllgrad(g, θ); copy(g))

    # The wired Gaussian penalty, called unmodified on a non-Gaussian Laplace fit.
    pen = D._glsp_reml_penalty(grad_fn, θ̂, pμ)
    # Independent cross-check of the same quantity from the nll VALUE surface.
    pen_fd = let h = 1e-4
        S = zeros(pμ, pμ)
        for j in 1:pμ, k in 1:pμ
            tpp = copy(θ̂); tpp[j] += h; tpp[k] += h
            tpm = copy(θ̂); tpm[j] += h; tpm[k] -= h
            tmp = copy(θ̂); tmp[j] -= h; tmp[k] += h
            tmm = copy(θ̂); tmm[j] -= h; tmm[k] -= h
            S[j, k] = (fit.nll(tpp) - fit.nll(tpm) - fit.nll(tmp) + fit.nll(tmm)) / (4h^2)
        end
        S .= 0.5 .* (S .+ S')
        0.5 * logdet(Symmetric(S))
    end

    # The wired Gaussian REML refit, called unmodified. Under a non-Gaussian
    # marginal the same objective IS the Cox–Reid adjusted profile likelihood.
    θ̂cr, conv, ml_nll, cr_nll, steps =
        D._glsp_reml_refit_clean(fit.nll, grad_fn, θ̂, pμ)

    return (; has_grad, reached = true, pμ, npar = length(θ̂), σphy,
            penalty = pen, penalty_fd = pen_fd,
            pen_rel_diff = abs(pen - pen_fd) / max(abs(pen_fd), 1.0),
            sigma_ml = exp(θ̂[end]), sigma_cr = exp(θ̂cr[end]),
            conv, ml_nll, cr_nll, steps)
end

# ---------------------------------------------------------------------------
# Cell D — cheap Laplace VC bias on the same Poisson-phylo spine as Cell C.
# Public `(1 | g)` is GHQ-32 (`_fit_poisson_ranef`); K=1 crossed Laplace
# redirects to that GHQ path. The 1-point Laplace marginal lives on phylo /
# relmat / K≥2 crossed. This cell is the Laplace-bias measurement #441 asked
# for — not an imported drmTMB figure. 12 seeds, ntip=16: Mac-cheap.
# ---------------------------------------------------------------------------
function cell_d(; ntip::Int = 16, per::Int = 4, seeds::Int = 12, σphy::Float64 = 0.7)
    ml = Float64[]
    cr = Float64[]
    nfail = 0
    for s in 1:seeds
        rng = MersenneTwister(20260818 + 17s)
        tree = D.random_balanced_tree(ntip; branch_length = 0.25)
        species = repeat(1:ntip, inner = per)
        n = length(species)
        x = randn(rng, n)
        Q, leaf_node, _ = D._poisson_phylo_setup(tree, species)
        L = cholesky(Symmetric(Matrix(Q))).U
        u = σphy .* (L \ randn(rng, size(Q, 1)))
        η = 0.3 .+ 0.25 .* x .+ u[leaf_node]
        y = Float64.([rand(rng, Distributions.Poisson(exp(clamp(ηi, -20, 20)))) for ηi in η])
        data = (; y, x, species)
        fit = D.drm(D.bf(D.@formula(y ~ x + phylo(1 | species))), D.Poisson();
                    data = data, tree = tree, se = false)
        D.is_converged(fit) && fit.nllgrad !== nothing || (nfail += 1; continue)
        θ̂ = D.coef(fit)
        pμ = length(D.coef(fit, :mu))
        grad_fn = θ -> (g = zeros(length(θ)); fit.nllgrad(g, θ); copy(g))
        θ̂cr, ok, _, _, _ = D._glsp_reml_refit_clean(fit.nll, grad_fn, θ̂, pμ)
        ok || (nfail += 1; continue)
        push!(ml, exp(θ̂[end]))
        push!(cr, exp(θ̂cr[end]))
    end
    return (; ntip, per, σphy, seeds, nfail, nrep = length(ml),
            ml_mean = _mean(ml), cr_mean = _mean(cr),
            ml_bias_pct = 100 * (_mean(ml) - σphy) / σphy,
            cr_bias_pct = 100 * (_mean(cr) - σphy) / σphy)
end

# ---------------------------------------------------------------------------

function main()
    println("=" ^ 78)
    println("Cox–Reid scoping probe (#441) — DRModels.jl")
    println("=" ^ 78)

    println("\n--- Cell B: Gaussian reduction anchor (generic CR penalty vs #440 exact REML) ---")
    b = cell_b()
    println("  refit converged : $(b.ok)     pμ = $(b.pμ)")
    @printf("  θ̂_ML   : %s\n", join((@sprintf("% .6f", v) for v in b.theta_ml), "  "))
    @printf("  θ̂_CR   : %s\n", join((@sprintf("% .6f", v) for v in b.theta_cr), "  "))
    @printf("  θ̂_REML : %s\n", join((@sprintf("% .6f", v) for v in b.theta_reml), "  "))
    @printf("  max |θ̂_CR − θ̂_REML| = %.3e\n", b.max_abs_diff)

    println("\n--- Cell A: Poisson (1|g), GHQ-32 integral, true σ_b = 0.6 ---")
    @printf("  %-6s %-8s %-7s %12s %12s %12s %12s\n",
            "G", "n_each", "nrep", "σ̂_ML", "bias_ML %", "σ̂_CR", "bias_CR %")
    for G in (10, 20, 40)
        r = cell_a(G = G, n_each = 6, seeds = 60)
        @printf("  %-6d %-8d %-7d %12.4f %12.2f %12.4f %12.2f\n",
                r.G, r.n_each, r.nrep, r.ml_mean, r.ml_bias_pct,
                r.cr_mean, r.cr_bias_pct)
    end

    println("\n--- Cell C: hook viability on the sparse-Laplace route (Poisson phylo) ---")
    c = cell_c()
    println("  analytic gradient on the fit : $(c.has_grad)")
    if c.has_grad
        @printf("  pμ = %d, npar = %d\n", c.pμ, c.npar)
        @printf("  _glsp_reml_penalty (analytic-grad FD) : %.8f\n", c.penalty)
        @printf("  independent value-surface FD          : %.8f  (rel diff %.2e)\n",
                c.penalty_fd, c.pen_rel_diff)
        @printf("  σ̂_phylo (true %.2f)  ML = %.5f   Cox–Reid = %.5f\n",
                c.σphy, c.sigma_ml, c.sigma_cr)
        @printf("  refit converged = %s, LBFGS steps = %d\n", c.conv, c.steps)
        @printf("  nll_ML = %.6f   nll_CR = %.6f\n", c.ml_nll, c.cr_nll)
    end

    println("\n--- Cell D: cheap Laplace VC bias (Poisson phylo, 12 seeds, true σ = 0.7) ---")
    d = cell_d()
    @printf("  ntip=%d per=%d nrep=%d nfail=%d\n", d.ntip, d.per, d.nrep, d.nfail)
    @printf("  σ̂_ML = %.4f  bias_ML = %+.2f%%     σ̂_CR = %.4f  bias_CR = %+.2f%%\n",
            d.ml_mean, d.ml_bias_pct, d.cr_mean, d.cr_bias_pct)
    println("\ndone.")
end

main()
