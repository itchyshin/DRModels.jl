# phylo_interaction.jl — bipartite two-tree interaction random effect on the
# Gaussian mean. An interaction effect between two phylogenies (e.g. host ×
# parasite) has covariance σ²·(C_A ⊗ C_B), the Kronecker product of the two
# trees' phylo correlations. With one observation per (host, parasite) cell the
# marginal stays exactly Gaussian:
#     y ~ N(Xβ, V),   V = σ²·(C_A ⊗ C_B) + σ_e²·I,
# fit by ML over θ = [βμ; log σ_e; log σ] (Cholesky of V each evaluation).
#
# FIRST SLICE: DENSE assembly + factorisation of the n×n V (n = n_A·n_B),
# correctness first, intended for modest n (n_A·n_B ≲ 100). The single
# interaction SD σ is reported as a named variance component via `re_sd`.
#
# Observation order is the standard Kronecker order: cell (a, b) → row
# (a−1)·n_B + b, so the residual-effect covariance is exactly kron(C_A, C_B).
# C_A (n_A×n_A) and C_B (n_B×n_B) are correlation matrices (unit diagonal); they
# come from a tree via `phylo_correlation` (reusing `sigma_phy_dense`) or are
# supplied directly.

using LinearAlgebra: cholesky, Symmetric, dot, logdet, kron, I, issuccess, diag
using Statistics: std

"""
    phylo_correlation(tree) -> Matrix

Leaf **correlation** matrix from a phylogeny. `tree` is an `AugmentedPhy`
(e.g. from `random_balanced_tree` / `augmented_phy`) or a Newick string. The
phylo covariance is built with `sigma_phy_dense` and rescaled to unit diagonal.
"""
function phylo_correlation(tree)
    phy = tree isa AbstractString ? augmented_phy(tree) : tree
    C = sigma_phy_dense(phy; σ²_phy = 1.0)
    d = sqrt.(diag(C))
    return C ./ (d * d')
end

"""
    phylo_interaction_nll(θ, y, X, CAB; pμ) -> Real

Marginal negative log-likelihood of the bipartite two-tree interaction model at
parameters `θ = [βμ(pμ); log σ_e; log σ]`, with `CAB = C_A ⊗ C_B` (the dense
n×n interaction correlation) precomputed. The marginal is exactly Gaussian:
`V = σ²·CAB + σ_e²·I`. A non-PD step returns a large finite penalty (so a
line-search probe never throws / returns Inf). AD-safe (`eltype(θ)` generic).
"""
function phylo_interaction_nll(θ, y, X, CAB; pμ::Int)
    n = length(y)
    βμ = θ[1:pμ]
    lσe = θ[pμ+1]
    lσ = θ[pμ+2]
    σ² = exp(2 * lσ)
    σe² = exp(2 * lσe)
    T = eltype(θ)
    V = σ² .* CAB .+ σe² .* Matrix{T}(I, n, n)
    # `check = false` + finite penalty: a line-search step into a numerically
    # non-PD V must be rejected by backtracking, not throw a PosDefException
    # (Julia 1.12's line search asserts the objective stays finite).
    Vfac = cholesky(Symmetric(V); check = false)
    issuccess(Vfac) || return convert(T, 1e18)
    r = y .- X * βμ
    quad = dot(r, Vfac \ r)
    return 0.5 * (logdet(Vfac) + quad) + 0.5 * n * log(2π)
end

"""
    fit_phylo_interaction(y, X, C_A, C_B; pμ = size(X, 2),
                          munames = ["(Intercept)"], group = :interaction,
                          g_tol = 1e-8) -> DrmFit

Fit the Gaussian bipartite two-tree interaction model by maximum likelihood.
`C_A` (n_A×n_A) and `C_B` (n_B×n_B) are the host/parasite phylo correlation
matrices; observations are in Kronecker order (cell `(a, b)` at row
`(a−1)·n_B + b`), so `length(y) == n_A·n_B`. Returns a [`DrmFit`](@ref) whose
`loglik` is the marginal log-likelihood; the interaction SD σ and the residual
SD σ_e are recovered via `re_sd(fit)[group]` and `exp(coef(fit, :sigma)[1])`.

A dense n×n covariance is assembled and factored each evaluation — O(n³),
intended for modest n.
"""
function fit_phylo_interaction(y::AbstractVector, X::AbstractMatrix,
                               C_A::AbstractMatrix, C_B::AbstractMatrix;
                               pμ::Int = size(X, 2),
                               munames::Vector{String} = ["(Intercept)"],
                               group::Symbol = :interaction,
                               g_tol::Real = 1e-8)
    nA = size(C_A, 1); nB = size(C_B, 1)
    n = nA * nB
    length(y) == n ||
        error("phylo_interaction: length(y) = $(length(y)) but n_A·n_B = $n " *
              "($nA × $nB); observations must cover the full host × parasite grid")
    size(X, 1) == n ||
        error("phylo_interaction: X has $(size(X, 1)) rows but needs n = $n")
    CAB = kron(Matrix{Float64}(C_A), Matrix{Float64}(C_B))   # C_A ⊗ C_B, dense n×n

    nll(θ) = phylo_interaction_nll(θ, y, X, CAB; pμ = pμ)

    # Start: OLS mean, residual SD split evenly between residual and interaction.
    βμ0 = X \ y
    s0 = std(y .- X * βμ0)
    θ0 = zeros(pμ + 2)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(s0 / sqrt(2) + eps())     # log σ_e
    θ0[pμ+2] = log(s0 / sqrt(2) + eps())     # log σ
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol);
                         autodiff = :forward)
    θ̂ = Optim.minimizer(res)
    V = try
        Matrix(inv(Symmetric(ForwardDiff.hessian(nll, θ̂))))
    catch
        fill(NaN, pμ + 2, pμ + 2)
    end

    # Blocks: μ; residual log σ_e under :sigma; interaction log σ under :resd so
    # `re_sd` reports it as a named variance component for `group`.
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+1), :resd => (pμ+2):(pμ+2)]
    names = [:mu => munames, :sigma => ["(Intercept)"], :resd => [String(group)]]
    means = Dict(:mu => X * θ̂[1:pμ])
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => fill(exp(θ̂[pμ+1]), n))
    return _withnll(DrmFit(Gaussian(), blocks, names, θ̂, V, -nll(θ̂), n,
                           drm_optim_converged(res), means, obs, scales), nll)
end
