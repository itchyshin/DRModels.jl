# test_relmat_counts_beta.jl — general user-supplied PD-covariance random effect
# (relatedness / animal model / precomputed spatial) for the BETA family on the
# open interval (0,1) (#167 follow-up; the last family to get general-covariance
# RE support).
#
# Companion to test_relmat_counts_nb2.jl (NB2/Gamma). The nuisance-parameter
# sparse-Laplace spine `_fit_general_mean_laplace_nuisance` /
# `_phylo_mean_laplace_nuisance_fg` is fully general in the prior precision Q:
# nothing downstream of `(Q, leaf_node)` requires Q to come from a tree. So Beta
# with an arbitrary user-supplied PD covariance C — routed via `relmat(1 | id)` +
# `K = C` (and the `animal(1 | id)` / `spatial(1 | id)` aliases) — reuses the
# verified phylo nuisance fitter with the tree precision swapped for C⁻¹, driven
# through the SAME Q-generic `Val(:beta_fixed)` kernel as the phylo gate.
#
# This file checks, for Beta:
#   (a) parameter recovery of the fixed effects, the variance component σ_b, and
#       the Beta precision φ (the `sigma` slot, σ = 1/√φ); and
#   (b) the exact analytic outer gradient matches a tightly-converged central
#       finite-difference gradient of the true marginal NLL to ≤ 1e-6 (the FD
#       gate the gradient path must pass), evaluated at a θ OFF the optimum so
#       the implicit db̂/dθ terms are exercised — with a relmat-derived precision.
using DRModels
using Test, Random, LinearAlgebra
import Distributions
using SpecialFunctions: loggamma, digamma

_logistic_b(η) = 1 / (1 + exp(-η))

# A genuine, well-conditioned PD correlation among `G` groups from an exponential
# kernel over random latent positions (a relatedness/spatial-style matrix that is
# NOT a tree), rescaled to unit diagonal so the recovered `:resd` block is the
# random-effect SD σ_b directly.
function _random_corr_beta(rng, G; range = 0.8, jitter = 1e-8)
    pos = rand(rng, G, 2) .* 6.0
    D = [sqrt(sum(abs2, pos[k, :] .- pos[l, :])) for k in 1:G, l in 1:G]
    C = exp.(-D ./ range) + jitter * I
    d = sqrt.(diag(C))
    return Symmetric(C ./ (d * d'))
end

# Tight-mode FD reference for the Q-generic nuisance fg, warm-started from b_base.
const _BT_NTOL = 1e-10
const _BT_NMAX = 400
const _BT_FDH  = 1e-4
function _fd_nuisance_relmat_beta(kind, aux_from, n, Xμ, leaf_node, Q, logdetQ, θ, b_base)
    function mnll(t)
        v = DRModels._phylo_mean_laplace_nuisance_fg(
            kind, aux_from, n, Xμ, leaf_node, Q, logdetQ, Vector{Float64}(t);
            grad = false, b0 = copy(b_base), newton_tol = _BT_NTOL, newton_maxiter = _BT_NMAX,
        )[1]
        @assert isfinite(v) && v < 1e17 "marginal NLL infeasible (sentinel) at θ = $t"
        return v
    end
    g = zeros(length(θ))
    for k in eachindex(θ)
        tp = copy(θ); tp[k] += _BT_FDH
        tm = copy(θ); tm[k] -= _BT_FDH
        g[k] = (mnll(tp) - mnll(tm)) / (2 * _BT_FDH)
    end
    return g
end

# Beta NLL inverse-precision aux, mirroring DRModels's internal `_beta_laplace_setup`
# exactly (φ = 1/σ²; logit-transformed responses cached). θσ is stored as
# -0.5·log φ, so the recovered precision is exp(-2·coef(:sigma)).
function _beta_aux_from(y, ylogit)
    return function (logsigma)
        φ = exp(clamp(-2 * logsigma, -8.0, 8.0))
        return (y = y, precision = φ, ylogit = ylogit,
                lgammaφ = loggamma(φ), digammaφ = digamma(φ))
    end
end

# ---- Beta recovery ---------------------------------------------------------
@testset "Beta relmat(1|id) — general-covariance sparse Laplace recovery" begin
    rng = MersenneTwister(20260618)
    G = 80
    m = 8
    C = _random_corr_beta(rng, G; range = 0.8)
    id = repeat(1:G, inner = m)
    n = length(id)
    x = randn(rng, n)
    β = [0.30, 0.40]
    σb = 0.45
    φtrue = 12.0                                   # Beta precision (σ = 1/√φ)
    u = σb .* (cholesky(C).L * randn(rng, G))
    μ = _logistic_b.(β[1] .+ β[2] .* x .+ u[id])
    y = Float64.([rand(rng, Distributions.Beta(μi * φtrue, (1 - μi) * φtrue)) for μi in μ])

    fit = drm(bf(@formula(y ~ x + relmat(1 | id)), @formula(sigma ~ 1)),
              Beta(); data = (; y, x, id), K = Matrix(C), se = false)

    @test fit.converged
    @test coef(fit, :mu)[1] ≈ β[1] atol = 0.25       # intercept (mean(u) ≈ 0 here)
    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.15       # slope (independent of the RE)
    @test re_sd(fit)[:id] ≈ σb atol = 0.20           # variance component recovered
    @test exp(-2 * coef(fit, :sigma)[1]) ≈ φtrue rtol = 0.6   # precision in the right ballpark
    @test isfinite(loglik(fit))
    @test all(0 .< fitted(fit) .< 1)
end

# ---- Beta FD gate ≤ 1e-6 ---------------------------------------------------
@testset "Beta relmat — exact gradient vs finite differences (FD gate ≤ 1e-6)" begin
    rng = MersenneTwister(20260619)
    G = 12
    m = 5
    C = _random_corr_beta(rng, G; range = 1.2)
    id = repeat(1:G, inner = m)
    n = length(id)
    x = randn(rng, n)
    φtrue = 12.0
    u = 0.45 .* (cholesky(C).L * randn(rng, G))
    μ = _logistic_b.(0.30 .+ 0.40 .* x .+ u[id])
    y = Float64.([rand(rng, Distributions.Beta(μi * φtrue, (1 - μi) * φtrue)) for μi in μ])

    Q, leaf_node = DRModels._general_cov_setup(Matrix(C), id)
    q = size(Q, 1)
    logdetQ = logdet(cholesky(Symmetric(Q); check = false))
    Xμ = hcat(ones(n), x)
    ylogit = log.(y) .- log1p.(-y)
    aux_from = _beta_aux_from(y, ylogit)
    # θ = [βμ(2); -0.5·log φ; logσ_relmat], OFF the optimum (pick φ ≈ 8.0).
    θ = [0.15, 0.50, -0.5 * log(8.0), log(0.55)]
    val0, g_an, b_base, ok = DRModels._phylo_mean_laplace_nuisance_fg(
        Val(:beta_fixed), aux_from, n, Xμ, leaf_node, Q, logdetQ, θ;
        grad = true, b0 = zeros(q), newton_tol = _BT_NTOL, newton_maxiter = _BT_NMAX,
    )
    @test ok && isfinite(val0) && val0 < 1e17
    g_fd = _fd_nuisance_relmat_beta(Val(:beta_fixed), aux_from, n, Xμ, leaf_node, Q, logdetQ, θ, b_base)
    max_abs_diff = maximum(abs, g_an .- g_fd)
    @info "Beta relmat gradient gate" max_abs_diff g_an g_fd
    @test max_abs_diff ≤ 1e-6
end

# ---- aliases + routing errors ----------------------------------------------
@testset "Beta animal/spatial aliases + routing errors" begin
    rng = MersenneTwister(20260620)
    G = 30
    m = 6
    C = _random_corr_beta(rng, G; range = 1.5)
    id = repeat(1:G, inner = m)
    n = length(id)
    x = randn(rng, n)
    u = 0.40 .* (cholesky(C).L * randn(rng, G))
    μ = _logistic_b.(0.2 .+ 0.35 .* x .+ u[id])
    Cm = Matrix(C)
    y = Float64.([rand(rng, Distributions.Beta(μi * 12.0, (1 - μi) * 12.0)) for μi in μ])

    # Beta animal(1|id) takes the relatedness matrix via `A = …`
    fit_a = drm(bf(@formula(y ~ x + animal(1 | id)), @formula(sigma ~ 1)),
                Beta(); data = (; y, x, id), A = Cm, se = false)
    @test fit_a.converged
    @test haskey(re_sd(fit_a), :id)
    @test coef(fit_a, :mu)[2] ≈ 0.35 atol = 0.25

    # Beta spatial(1|id) accepts a precomputed spatial covariance via `K = …`
    fit_s = drm(bf(@formula(y ~ x + spatial(1 | id)), @formula(sigma ~ 1)),
                Beta(); data = (; y, x, id), K = Cm, se = false)
    @test fit_s.converged
    @test haskey(re_sd(fit_s), :id)

    # relmat/animal without their matrix → clear error
    @test_throws ErrorException drm(bf(@formula(y ~ x + relmat(1 | id)), @formula(sigma ~ 1)),
                                    Beta(); data = (; y, x, id), se = false)
    @test_throws ErrorException drm(bf(@formula(y ~ x + animal(1 | id)), @formula(sigma ~ 1)),
                                    Beta(); data = (; y, x, id), se = false)
    # coords-only spatial → error (coordinate-based range is Gaussian-only for now)
    @test_throws ErrorException drm(bf(@formula(y ~ x + spatial(1 | id)), @formula(sigma ~ 1)),
                                    Beta(); data = (; y, x, id), coords = rand(G, 2), se = false)
    # non-constant sigma with a structured RE → clear error (constant-sigma only)
    @test_throws ErrorException drm(bf(@formula(y ~ x + relmat(1 | id)), @formula(sigma ~ 0 + x)),
                                    Beta(); data = (; y, x, id), K = Cm, se = false)
end
