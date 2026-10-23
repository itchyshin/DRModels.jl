# test_nongaussian_phylo_grad_gate.jl — STANDING FD-vs-analytic gradient gates (#165)
# for the non-Gaussian PHYLOGENETIC sparse-Laplace routes that carry an exact
# implicit-function gradient via the `_laplace_v123`/`_laplace_v123_nuisance` d3
# kernels: NB2, Gamma, Binomial, Beta-binomial (#166) (and Beta, reported honestly).
#
# Companion to test_poisson_phylo_grad_gate.jl. Each route's analytic outer
# gradient (the exact implicit-logdet gradient that reuses `takahashi_selinv`)
# MUST match a central finite-difference gradient of the TRUE marginal NLL.
#
# Recipe (same as the q4 / Poisson-phylo gates): drive the inner Newton mode to a
# tight tolerance, warm-start every perturbed solve from the base-θ mode, and
# evaluate the gradient at a θ OFF the optimum so the implicit db̂/dθ terms are
# exercised. The inner-mode full-Newton-in-basin fix (in `_phylo_mean_mode`,
# gated on nonnegative data-Hessian weights) is what makes the tight mode — and
# hence the clean FD reference — attainable for the convex-data families.
#
# Convexity note: the binomial/nb2/gamma data terms have d² ≥ 0 everywhere, so the
# inner joint (nuisance held fixed) is strictly convex and the full Newton step in
# the basin is safe → these reach ≤ 1e-6. Beta's d² is NOT sign-definite, so its
# inner solve keeps the safeguarded line search; we report its achieved FD error
# at a looser-but-honest tolerance rather than forcing an unsafe full step.

using DRModels
using Test, Random, LinearAlgebra
import Distributions
using SpecialFunctions: loggamma, digamma

# ---- shared phylo design ---------------------------------------------------
function _phylo_setup_for_gate(seed; p = 12, m = 4, bl = 0.20)
    Random.seed!(seed)
    phy = random_balanced_tree(p; branch_length = bl)
    species = repeat(1:p, inner = m)
    n = length(species)
    x = randn(n)
    Q, leaf_node, _ = DRModels._poisson_phylo_setup(phy, species)
    q = size(Q, 1)
    logdetQ = logdet(cholesky(Symmetric(Q); check = false))
    σphy = 0.45
    C = sigma_phy_dense(phy; σ²_phy = σphy^2)
    u = cholesky(Symmetric(C)).L * randn(p)
    Xμ = hcat(ones(n), x)
    return (; phy, species, n, x, Q, leaf_node, q, logdetQ, u, Xμ)
end

const NTOL = 1e-10
const NMAX = 400
const FDH  = 1e-4

# Central-difference gradient of a nuisance-route marginal, warm-started.
function _fd_nuisance(kind, aux_from, n, Xμ, leaf_node, Q, logdetQ, θ, b_base)
    function mnll(t)
        v = DRModels._phylo_mean_laplace_nuisance_fg(
            kind, aux_from, n, Xμ, leaf_node, Q, logdetQ, Vector{Float64}(t);
            grad = false, b0 = copy(b_base), newton_tol = NTOL, newton_maxiter = NMAX,
        )[1]
        @assert isfinite(v) && v < 1e17 "marginal NLL infeasible (sentinel) at θ = $t"
        return v
    end
    g = zeros(length(θ))
    for k in eachindex(θ)
        tp = copy(θ); tp[k] += FDH
        tm = copy(θ); tm[k] -= FDH
        g[k] = (mnll(tp) - mnll(tm)) / (2FDH)
    end
    return g
end

function _fd_meanonly(kind, aux, n, Xμ, leaf_node, Q, logdetQ, θ, b_base)
    function mnll(t)
        v = DRModels._phylo_mean_laplace_fg(
            kind, aux, n, Xμ, leaf_node, Q, logdetQ, Vector{Float64}(t);
            grad = false, b0 = copy(b_base), newton_tol = NTOL, newton_maxiter = NMAX,
        )[1]
        @assert isfinite(v) && v < 1e17 "marginal NLL infeasible (sentinel) at θ = $t"
        return v
    end
    g = zeros(length(θ))
    for k in eachindex(θ)
        tp = copy(θ); tp[k] += FDH
        tm = copy(θ); tm[k] -= FDH
        g[k] = (mnll(tp) - mnll(tm)) / (2FDH)
    end
    return g
end

# ---- NB2 -------------------------------------------------------------------
@testset "NB2 phylo Laplace gradient gate (#165): FD-vs-exact ≤ 1e-6" begin
    s = _phylo_setup_for_gate(2041)
    sizep = 4.0
    μ = exp.(0.20 .+ 0.35 .* s.x .+ s.u[s.species])
    yint = [rand(Distributions.NegativeBinomial(sizep, sizep / (sizep + μi))) for μi in μ]
    y = Float64.(yint)
    function aux_from(logsigma)
        # ψ = log σ; size r = 1/σ² = exp(−2ψ) (drmTMB NB2 parameterization, matches
        # the production _nb2_laplace_setup). The engine's analytic gradient carries
        # the −2 chain factor, so the FD reference must read the slot as log σ too.
        r = exp(clamp(-2 * logsigma, -8.0, 8.0))
        lconst = [loggamma(yint[i] + r) - loggamma(r) - DRModels._logfactorial(yint[i]) for i in eachindex(yint)]
        return (y = y, size = r, lconst = lconst)
    end
    # θ = [βμ(2); logσ; logσphylo], OFF the optimum. θσ stored as −0.5 log(size);
    # pick size ≈ 3.0 ⇒ logσ = −0.5 log 3.
    θ = [0.10, 0.45, -0.5 * log(3.0), log(0.55)]
    val0, g_an, b_base, ok = DRModels._phylo_mean_laplace_nuisance_fg(
        Val(:nb2_fixed), aux_from, s.n, s.Xμ, s.leaf_node, s.Q, s.logdetQ, θ;
        grad = true, b0 = zeros(s.q), newton_tol = NTOL, newton_maxiter = NMAX,
    )
    @test ok && isfinite(val0) && val0 < 1e17
    g_fd = _fd_nuisance(Val(:nb2_fixed), aux_from, s.n, s.Xμ, s.leaf_node, s.Q, s.logdetQ, θ, b_base)
    max_abs_diff = maximum(abs, g_an .- g_fd)
    @info "NB2 phylo gradient gate" max_abs_diff g_an g_fd
    @test max_abs_diff ≤ 1e-6
end

# ---- Gamma -----------------------------------------------------------------
@testset "Gamma phylo Laplace gradient gate (#165): FD-vs-exact ≤ 1e-6" begin
    s = _phylo_setup_for_gate(2042)
    shape = 6.0
    μ = exp.(0.30 .+ 0.30 .* s.x .+ s.u[s.species])
    y = [rand(Distributions.Gamma(shape, μi / shape)) for μi in μ]
    function aux_from(logsigma)
        α = exp(clamp(-2 * logsigma, -8.0, 8.0))
        lconst = [α * log(α) - loggamma(α) + (α - 1) * log(y[i]) for i in eachindex(y)]
        return (y = y, shape = α, lconst = lconst)
    end
    # θσ stored as -0.5 log α; pick α ≈ 4.0 ⇒ logσ = -0.5 log 4.
    θ = [0.10, 0.40, -0.5 * log(4.0), log(0.55)]
    val0, g_an, b_base, ok = DRModels._phylo_mean_laplace_nuisance_fg(
        Val(:gamma_fixed), aux_from, s.n, s.Xμ, s.leaf_node, s.Q, s.logdetQ, θ;
        grad = true, b0 = zeros(s.q), newton_tol = NTOL, newton_maxiter = NMAX,
    )
    @test ok && isfinite(val0) && val0 < 1e17
    g_fd = _fd_nuisance(Val(:gamma_fixed), aux_from, s.n, s.Xμ, s.leaf_node, s.Q, s.logdetQ, θ, b_base)
    max_abs_diff = maximum(abs, g_an .- g_fd)
    @info "Gamma phylo gradient gate" max_abs_diff g_an g_fd
    @test max_abs_diff ≤ 1e-6
end

# ---- Binomial --------------------------------------------------------------
@testset "Binomial phylo Laplace gradient gate (#165): FD-vs-exact ≤ 1e-6" begin
    s = _phylo_setup_for_gate(2043)
    _logistic(z) = 1 / (1 + exp(-z))
    prob = _logistic.(-0.10 .+ 0.45 .* s.x .+ s.u[s.species])
    ntr = fill(10, s.n)
    sint = [rand(Distributions.Binomial(ntr[i], prob[i])) for i in 1:s.n]
    logchoose = [DRModels._logfactorial(ntr[i]) - DRModels._logfactorial(sint[i]) -
                 DRModels._logfactorial(ntr[i] - sint[i]) for i in 1:s.n]
    aux = (s = sint, ntr = ntr, logchoose = logchoose)
    # θ = [βμ(2); logσphylo], OFF the optimum.
    θ = [0.05, 0.55, log(0.55)]
    val0, g_an, b_base, ok = DRModels._phylo_mean_laplace_fg(
        Val(:binomial), aux, s.n, s.Xμ, s.leaf_node, s.Q, s.logdetQ, θ;
        grad = true, b0 = zeros(s.q), newton_tol = NTOL, newton_maxiter = NMAX,
    )
    @test ok && isfinite(val0) && val0 < 1e17
    g_fd = _fd_meanonly(Val(:binomial), aux, s.n, s.Xμ, s.leaf_node, s.Q, s.logdetQ, θ, b_base)
    max_abs_diff = maximum(abs, g_an .- g_fd)
    @info "Binomial phylo gradient gate" max_abs_diff g_an g_fd
    @test max_abs_diff ≤ 1e-6
end

# ---- Beta (honest, looser tolerance: data Hessian not sign-definite) -------
# Beta's d²ℓ/dη² is not guaranteed ≥ 0, so the inner solve keeps the safeguarded
# line search (no unsafe full Newton step). The mode is therefore converged only
# to the line search's reach near the minimum; we record the genuinely-achieved
# FD-vs-analytic error rather than forcing the 1e-6 gate. If a future inner-solve
# improvement (e.g. a trust region) tightens the beta mode, drop BETA_TOL to 1e-6.
const BETA_TOL = 1e-4
@testset "Beta phylo Laplace gradient gate (#165): achieved FD error ≤ $(BETA_TOL)" begin
    s = _phylo_setup_for_gate(2044; p = 10, m = 4)
    _logistic(z) = 1 / (1 + exp(-z))
    prec = 8.0
    μ = _logistic.(0.10 .+ 0.30 .* s.x .+ s.u[s.species])
    y = clamp.([rand(Distributions.Beta(μi * prec, (1 - μi) * prec)) for μi in μ], 1e-4, 1 - 1e-4)
    ylogit = log.(y) .- log1p.(-y)
    function aux_from(logsigma)
        φ = exp(clamp(-2 * logsigma, -8.0, 8.0))
        return (y = y, precision = φ, ylogit = ylogit,
                lgammaφ = loggamma(φ), digammaφ = digamma(φ))
    end
    θ = [0.05, 0.35, -0.5 * log(6.0), log(0.55)]
    val0, g_an, b_base, ok = DRModels._phylo_mean_laplace_nuisance_fg(
        Val(:beta_fixed), aux_from, s.n, s.Xμ, s.leaf_node, s.Q, s.logdetQ, θ;
        grad = true, b0 = zeros(s.q), newton_tol = NTOL, newton_maxiter = NMAX,
    )
    @test ok && isfinite(val0) && val0 < 1e17
    g_fd = _fd_nuisance(Val(:beta_fixed), aux_from, s.n, s.Xμ, s.leaf_node, s.Q, s.logdetQ, θ, b_base)
    max_abs_diff = maximum(abs, g_an .- g_fd)
    @info "Beta phylo gradient gate (honest, line-search inner mode)" max_abs_diff g_an g_fd
    @test max_abs_diff ≤ BETA_TOL
end

# ---- Beta-binomial (#166) ---------------------------------------------------
# Generalizes the `:beta_fixed` kernel above to beta-binomial's discrete
# known-trials data term (shifted digamma/trigamma/polygamma arguments; see
# docs/dev-log/plans/2026-08-02-166-betabinomial-kernel-design.md). Unlike Beta,
# the discrete data term keeps the inner mode's Newton basin well-behaved
# enough in practice to reach the tight ≤ 1e-6 gate (measured, not assumed).
@testset "Beta-binomial phylo Laplace gradient gate (#166): FD-vs-exact ≤ 1e-6" begin
    s = _phylo_setup_for_gate(2045; p = 10, m = 4)
    _logistic(z) = 1 / (1 + exp(-z))
    φtrue = 9.0
    μ = _logistic.(0.10 .+ 0.30 .* s.x .+ s.u[s.species])
    ntr = fill(8, s.n)
    sint = [rand(Distributions.BetaBinomial(ntr[i], μ[i] * φtrue, (1 - μ[i]) * φtrue)) for i in 1:s.n]
    logchoose = [DRModels._logfactorial(ntr[i]) - DRModels._logfactorial(sint[i]) -
                 DRModels._logfactorial(ntr[i] - sint[i]) for i in 1:s.n]
    function aux_from(logsigma)
        φ = exp(clamp(-2 * logsigma, -8.0, 8.0))
        lgamma_nphi = [loggamma(ntr[i] + φ) for i in 1:s.n]
        return (s = sint, ntr = ntr, logchoose = logchoose, precision = φ,
                lgamma_nphi = lgamma_nphi, lgammaφ = loggamma(φ), digammaφ = digamma(φ))
    end
    θ = [0.05, 0.35, -0.5 * log(6.0), log(0.55)]
    val0, g_an, b_base, ok = DRModels._phylo_mean_laplace_nuisance_fg(
        Val(:betabinomial_fixed), aux_from, s.n, s.Xμ, s.leaf_node, s.Q, s.logdetQ, θ;
        grad = true, b0 = zeros(s.q), newton_tol = NTOL, newton_maxiter = NMAX,
    )
    @test ok && isfinite(val0) && val0 < 1e17
    g_fd = _fd_nuisance(Val(:betabinomial_fixed), aux_from, s.n, s.Xμ, s.leaf_node, s.Q, s.logdetQ, θ, b_base)
    max_abs_diff = maximum(abs, g_an .- g_fd)
    @info "Beta-binomial phylo gradient gate" max_abs_diff g_an g_fd
    @test max_abs_diff ≤ 1e-6
end
