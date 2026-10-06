# test_relmat_counts_nb2.jl — general user-supplied PD-covariance random effect
# (relatedness / animal model / precomputed spatial) for the OVERDISPERSED-count
# NB2 family and the positive-continuous Gamma family (#167 follow-up).
#
# Companion to test_relmat_counts.jl (Poisson). The nuisance-parameter
# sparse-Laplace spine `_fit_general_mean_laplace_nuisance` /
# `_phylo_mean_laplace_nuisance_fg` is fully general in the prior precision Q:
# nothing downstream of `(Q, leaf_node)` requires Q to come from a tree. So NB2
# and Gamma with an arbitrary user-supplied PD covariance C — routed via
# `relmat(1 | id)` + `K = C` (and the `animal(1 | id)` / `spatial(1 | id)`
# aliases) — reuse the verified phylo nuisance fitter with the tree precision
# swapped for C⁻¹.
#
# This file checks, for NB2 and Gamma:
#   (a) parameter recovery of the fixed effects, the variance component σ_b, and
#       the family's nuisance parameter (NB2 dispersion / Gamma shape); and
#   (b) the exact analytic outer gradient matches a tightly-converged central
#       finite-difference gradient of the true marginal NLL to ≤ 1e-6 (the FD
#       gate the gradient path must pass), evaluated at a θ OFF the optimum so
#       the implicit db̂/dθ terms are exercised — driven through the SAME
#       Q-generic `_phylo_mean_laplace_nuisance_fg` as the phylo gate, with a
#       relmat-derived precision.
using DRModels
using Test, Random, LinearAlgebra, Statistics
import Distributions
using SpecialFunctions: loggamma

# A genuine, well-conditioned PD correlation among `G` groups from an exponential
# kernel over random latent positions (a relatedness/spatial-style matrix that is
# NOT a tree), rescaled to unit diagonal so the recovered `:resd` block is the
# random-effect SD σ_b directly.
function _random_corr_nb(rng, G; range = 0.8, jitter = 1e-8)
    pos = rand(rng, G, 2) .* 6.0
    D = [sqrt(sum(abs2, pos[k, :] .- pos[l, :])) for k in 1:G, l in 1:G]
    C = exp.(-D ./ range) + jitter * I
    d = sqrt.(diag(C))
    return Symmetric(C ./ (d * d'))
end

# Tight-mode FD reference for the Q-generic nuisance fg, warm-started from b_base.
const _NB_NTOL = 1e-10
const _NB_NMAX = 400
const _NB_FDH  = 1e-4
function _fd_nuisance_relmat(kind, aux_from, n, Xμ, leaf_node, Q, logdetQ, θ, b_base)
    function mnll(t)
        v = DRModels._phylo_mean_laplace_nuisance_fg(
            kind, aux_from, n, Xμ, leaf_node, Q, logdetQ, Vector{Float64}(t);
            grad = false, b0 = copy(b_base), newton_tol = _NB_NTOL, newton_maxiter = _NB_NMAX,
        )[1]
        @assert isfinite(v) && v < 1e17 "marginal NLL infeasible (sentinel) at θ = $t"
        return v
    end
    g = zeros(length(θ))
    for k in eachindex(θ)
        tp = copy(θ); tp[k] += _NB_FDH
        tm = copy(θ); tm[k] -= _NB_FDH
        g[k] = (mnll(tp) - mnll(tm)) / (2 * _NB_FDH)
    end
    return g
end

# ---- NB2 recovery ----------------------------------------------------------
@testset "NB2 relmat(1|id) — general-covariance sparse Laplace recovery" begin
    rng = MersenneTwister(20260613)
    G = 80
    m = 15
    C = _random_corr_nb(rng, G; range = 0.8)
    id = repeat(1:G, inner = m)
    n = length(id)
    x = randn(rng, n)
    β = [0.20, 0.35]
    σb = 0.45
    sizep = 3.0                                    # NB2 dispersion (size) θ
    u = σb .* (cholesky(C).L * randn(rng, G))
    u .-= mean(u)
    μ = exp.(β[1] .+ β[2] .* x .+ u[id])
    y = Float64.([rand(rng, Distributions.NegativeBinomial(sizep, sizep / (sizep + μi))) for μi in μ])

    fit = drm(bf(@formula(y ~ x + relmat(1 | id)), @formula(sigma ~ 1)),
              NegBinomial2(); data = (; y, x, id), K = Matrix(C), se = false)

    @test fit.converged
    @test coef(fit, :mu)[1] ≈ β[1] atol = 0.25       # intercept (centered latent draw)
    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.15       # slope (independent of the RE)
    @test re_sd(fit)[:id] ≈ σb atol = 0.20           # variance component recovered
    @test exp(-2 * coef(fit, :sigma)[1]) ≈ sizep rtol = 0.6   # dispersion in the right ballpark
    @test isfinite(loglik(fit))
    @test all(fitted(fit) .> 0)
end

# ---- NB2 FD gate ≤ 1e-6 ----------------------------------------------------
@testset "NB2 relmat — exact gradient vs finite differences (FD gate ≤ 1e-6)" begin
    rng = MersenneTwister(20260614)
    G = 12
    m = 5
    C = _random_corr_nb(rng, G; range = 1.2)
    id = repeat(1:G, inner = m)
    n = length(id)
    x = randn(rng, n)
    sizep = 4.0
    u = 0.45 .* (cholesky(C).L * randn(rng, G))
    μ = exp.(0.20 .+ 0.35 .* x .+ u[id])
    yint = [rand(rng, Distributions.NegativeBinomial(sizep, sizep / (sizep + μi))) for μi in μ]
    y = Float64.(yint)

    Q, leaf_node = DRModels._general_cov_setup(Matrix(C), id)
    q = size(Q, 1)
    logdetQ = logdet(cholesky(Symmetric(Q); check = false))
    Xμ = hcat(ones(n), x)
    function aux_from(logσ)
        r = exp(clamp(-2 * logσ, -8.0, 8.0))   # size r = 1/σ² = exp(−2ψ); ψ = log σ (drmTMB)
        lconst = [loggamma(yint[i] + r) - loggamma(r) - DRModels._logfactorial(yint[i]) for i in eachindex(yint)]
        return (y = y, size = r, lconst = lconst)
    end
    # θ = [βμ(2); logσ (= −0.5·log size); logσ_relmat], OFF the optimum.
    θ = [0.10, 0.45, -0.5 * log(3.0), log(0.55)]
    val0, g_an, b_base, ok = DRModels._phylo_mean_laplace_nuisance_fg(
        Val(:nb2_fixed), aux_from, n, Xμ, leaf_node, Q, logdetQ, θ;
        grad = true, b0 = zeros(q), newton_tol = _NB_NTOL, newton_maxiter = _NB_NMAX,
    )
    @test ok && isfinite(val0) && val0 < 1e17
    g_fd = _fd_nuisance_relmat(Val(:nb2_fixed), aux_from, n, Xμ, leaf_node, Q, logdetQ, θ, b_base)
    max_abs_diff = maximum(abs, g_an .- g_fd)
    @info "NB2 relmat gradient gate" max_abs_diff g_an g_fd
    @test max_abs_diff ≤ 1e-6
end

# ---- Gamma recovery --------------------------------------------------------
@testset "Gamma relmat(1|id) — general-covariance sparse Laplace recovery" begin
    rng = MersenneTwister(20260615)
    G = 80
    m = 8
    C = _random_corr_nb(rng, G; range = 0.8)
    id = repeat(1:G, inner = m)
    n = length(id)
    x = randn(rng, n)
    β = [0.30, 0.30]
    σb = 0.40
    shape = 6.0                                    # Gamma shape α = 1/σ²
    u = σb .* (cholesky(C).L * randn(rng, G))
    μ = exp.(β[1] .+ β[2] .* x .+ u[id])
    y = Float64.([rand(rng, Distributions.Gamma(shape, μi / shape)) for μi in μ])

    fit = drm(bf(@formula(y ~ x + relmat(1 | id)), @formula(sigma ~ 1)),
              Gamma(); data = (; y, x, id), K = Matrix(C), se = false)

    @test fit.converged
    @test coef(fit, :mu)[1] ≈ β[1] atol = 0.25       # intercept (mean(u) ≈ 0 here)
    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.15       # slope (independent of the RE)
    @test re_sd(fit)[:id] ≈ σb atol = 0.20           # variance component recovered
    @test exp(-2 * coef(fit, :sigma)[1]) ≈ shape rtol = 0.6   # shape in the right ballpark
    @test isfinite(loglik(fit))
    @test all(fitted(fit) .> 0)
end

# ---- Gamma FD gate ≤ 1e-6 --------------------------------------------------
@testset "Gamma relmat — exact gradient vs finite differences (FD gate ≤ 1e-6)" begin
    rng = MersenneTwister(20260616)
    G = 12
    m = 5
    C = _random_corr_nb(rng, G; range = 1.2)
    id = repeat(1:G, inner = m)
    n = length(id)
    x = randn(rng, n)
    shape = 6.0
    u = 0.40 .* (cholesky(C).L * randn(rng, G))
    μ = exp.(0.30 .+ 0.30 .* x .+ u[id])
    y = [rand(rng, Distributions.Gamma(shape, μi / shape)) for μi in μ]

    Q, leaf_node = DRModels._general_cov_setup(Matrix(C), id)
    q = size(Q, 1)
    logdetQ = logdet(cholesky(Symmetric(Q); check = false))
    Xμ = hcat(ones(n), x)
    function aux_from(logsigma)
        α = exp(clamp(-2 * logsigma, -8.0, 8.0))
        lconst = [α * log(α) - loggamma(α) + (α - 1) * log(y[i]) for i in eachindex(y)]
        return (y = y, shape = α, lconst = lconst)
    end
    # θσ stored as -0.5 log α; pick α ≈ 4.0; θ OFF the optimum.
    θ = [0.10, 0.40, -0.5 * log(4.0), log(0.55)]
    val0, g_an, b_base, ok = DRModels._phylo_mean_laplace_nuisance_fg(
        Val(:gamma_fixed), aux_from, n, Xμ, leaf_node, Q, logdetQ, θ;
        grad = true, b0 = zeros(q), newton_tol = _NB_NTOL, newton_maxiter = _NB_NMAX,
    )
    @test ok && isfinite(val0) && val0 < 1e17
    g_fd = _fd_nuisance_relmat(Val(:gamma_fixed), aux_from, n, Xμ, leaf_node, Q, logdetQ, θ, b_base)
    max_abs_diff = maximum(abs, g_an .- g_fd)
    @info "Gamma relmat gradient gate" max_abs_diff g_an g_fd
    @test max_abs_diff ≤ 1e-6
end

# ---- aliases + routing errors ----------------------------------------------
@testset "NB2/Gamma animal/spatial aliases + routing errors" begin
    rng = MersenneTwister(20260617)
    G = 30
    m = 6
    C = _random_corr_nb(rng, G; range = 1.5)
    id = repeat(1:G, inner = m)
    n = length(id)
    x = randn(rng, n)
    u = 0.40 .* (cholesky(C).L * randn(rng, G))
    μ = exp.(0.2 .+ 0.35 .* x .+ u[id])
    Cm = Matrix(C)

    yc = Float64.([rand(rng, Distributions.NegativeBinomial(4.0, 4.0 / (4.0 + μi))) for μi in μ])
    # NB2 animal(1|id) takes the relatedness matrix via `A = …`
    fit_a = drm(bf(@formula(y ~ x + animal(1 | id)), @formula(sigma ~ 1)),
                NegBinomial2(); data = (; y = yc, x, id), A = Cm, se = false)
    @test fit_a.converged
    @test haskey(re_sd(fit_a), :id)
    @test coef(fit_a, :mu)[2] ≈ 0.35 atol = 0.25

    yg = Float64.([rand(rng, Distributions.Gamma(6.0, μi / 6.0)) for μi in μ])
    # Gamma spatial(1|id) accepts a precomputed spatial covariance via `K = …`
    fit_s = drm(bf(@formula(y ~ x + spatial(1 | id)), @formula(sigma ~ 1)),
                Gamma(); data = (; y = yg, x, id), K = Cm, se = false)
    @test fit_s.converged
    @test haskey(re_sd(fit_s), :id)

    # relmat/animal without their matrix → clear error
    @test_throws ErrorException drm(bf(@formula(y ~ x + relmat(1 | id)), @formula(sigma ~ 1)),
                                    NegBinomial2(); data = (; y = yc, x, id), se = false)
    @test_throws ErrorException drm(bf(@formula(y ~ x + animal(1 | id)), @formula(sigma ~ 1)),
                                    Gamma(); data = (; y = yg, x, id), se = false)
    # coords-only spatial → error (coordinate-based range is Gaussian-only for now)
    @test_throws ErrorException drm(bf(@formula(y ~ x + spatial(1 | id)), @formula(sigma ~ 1)),
                                    NegBinomial2(); data = (; y = yc, x, id), coords = rand(G, 2), se = false)
    # non-constant sigma with a structured RE → clear error (constant-sigma only)
    @test_throws ErrorException drm(bf(@formula(y ~ x + relmat(1 | id)), @formula(sigma ~ 0 + x)),
                                    NegBinomial2(); data = (; y = yc, x, id), K = Cm, se = false)
end
