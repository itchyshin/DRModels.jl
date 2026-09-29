# location_only.jl — CONJUGATE Gaussian phylogenetic mixed model (location-only).
#
# Promoted from src/experimental/location_only.jl (issue #12). This is the
# opt-in sparse paths for the Gaussian phylo-MEAN model — a single
# structured random intercept on the mean μ with a CONSTANT residual scale σ.
# The first path is closed-form conjugate EM with EXACT O(p) Takahashi traces.
# The second path is a GLLVM.jl-style sparse L-BFGS optimizer on the same
# all-node marginal likelihood, profiling β at every variance trial.
#
# Model:  y_i = X_i β + u_{s(i)} + ε_i
#         u  ~ N(0, σ²_phy · Σ_phy)   via the sparse augmented precision
#         ε  ~ N(0, σ² I)
#
# Augmented-state representation (Hadfield / sparse_phy.jl pattern):
#   Root-conditioned prior precision:  P = (1/σ²_phy) · Q_cond  (n_keep × n_keep, PD)
#   Leaf-observation map: S ∈ R^{n × n_keep}  (one entry per obs per kept node)
#
# The marginal p(y | β, σ²_phy, σ²) is CLOSED-FORM Gaussian:
#     y | β, θ  ~  N( X β,  V )    V = S P^{-1} S'  +  σ² I_n
#
# Woodbury on V^{-1}:
#     M    = P + (1/σ²) S'S         (n_keep × n_keep, sparse, PD)
#     V^{-1} z  = (z - S (M^{-1} (1/σ²) S' z)) / σ²
#     logdetV   = n log σ²  +  logdet M  −  logdet P
#
# EXACT O(p) TRACES via Takahashi selected inverse:
#     Tr(Q_cond M^{-1})  — needed for σ²_phy M-step
#     Tr(S M^{-1} S')    — needed for σ² M-step
# Both lie within the Takahashi pattern of M (same sparsity as Q_cond plus the
# leaf diagonal from S'S — but leaves ARE in Q_cond's pattern), so they are exact.
#
# Depends on `AugmentedPhy` / `sparse_phy.jl` and `takahashi_selinv` (already
# loaded by the verified core engine include chain in DRModels.jl — do NOT re-include).

using LinearAlgebra, SparseArrays, Statistics, Random

const _LOCONLY_PENALTY = 1e18

# ─────────────────────────────────────────────────────────────────────────────
# Problem struct (location-only)
# ─────────────────────────────────────────────────────────────────────────────

struct LocOnlyProblem
    phy::AugmentedPhy{Float64}
    n_keep::Int                           # 2p-2 root-conditioned nodes
    p::Int                                # species (leaves)
    n::Int                                # total observations
    Q_cond::SparseMatrixCSC{Float64,Int}  # (n_keep × n_keep) root-conditioned Q
    leaf_pos::Vector{Int}                 # leaf k -> index in kept nodes (1:n_keep)
    species::Vector{Int}                  # data row i -> species index (1:p)
    y::Vector{Float64}                    # responses (n,)
    X::Matrix{Float64}                    # design matrix (n × k)
    k::Int                                # number of fixed effects
    S::SparseMatrixCSC{Float64,Int}       # observation-node selection (n × n_keep)
    STS_diag::Vector{Float64}             # diagonal of S'S (n_keep,), sparse = at leaf_pos
end

# Callable full-vector objective for the sparse Gaussian phylogenetic
# location-only route. It carries the problem object so inference code can use
# the stronger specialised profile path for the variance component while generic
# objective consumers can still evaluate `fit.nll(theta)`.
struct LocOnlyObjective{P}
    prob::P
    pμ::Int
end

function (o::LocOnlyObjective)(θ)
    return _loconly_marginal_nll(o.prob, @view(θ[1:(o.pμ)]), θ[o.pμ + 1], θ[o.pμ + 2])
end

function _loconly_objective_grad!(g, o::LocOnlyObjective, θ)
    return _loconly_marginal_grad!(g, o.prob, θ, o.pμ)
end

function make_loc_problem(phy::AugmentedPhy, y, X; species=1:(phy.n_leaves))
    n_total = phy.n_total
    keep = setdiff(1:n_total, [phy.root_index])
    Q_cond = phy.Q_topology[keep, keep]
    pos = Dict(node => i for (i, node) in enumerate(keep))
    leaf_pos = [pos[phy.leaf_indices[k]] for k in 1:(phy.n_leaves)]
    n = length(y)
    sp = collect(Int, species)
    k = size(X, 2)

    # Build S (n × n_keep): obs i → node leaf_pos[species[i]]
    SI = collect(1:n)
    SJ = [leaf_pos[sp[i]] for i in 1:n]
    SV = ones(n)
    S = sparse(SI, SJ, SV, n, length(keep))

    # STS diagonal (= count of obs per kept-node position)
    STS_d = zeros(length(keep))
    for i in 1:n
        STS_d[leaf_pos[sp[i]]] += 1.0
    end

    return LocOnlyProblem(
        phy,
        length(keep),
        phy.n_leaves,
        n,
        Q_cond,
        leaf_pos,
        sp,
        Float64.(y),
        Float64.(X),
        k,
        S,
        STS_d,
    )
end

"""
    _phylo_mean_leaf_index(phy::AugmentedPhy, labels) -> Vector{Int}

Map per-row group `labels` to the tree's raw leaf index (`1:phy.n_leaves`) —
the convention `make_loc_problem`'s `species=` keyword expects. Mirrors the
three-tier match `_poisson_phylo_setup` already uses for the Laplace phylo
routes, most to least specific:

  1. leaf NAME (`phy.leaf_names`) — subset-tolerant: `labels` may name only
     SOME of the tree's leaves (e.g. after a caller drops missing-response
     rows upstream — #482). A leaf that never appears in `labels` simply gets
     no observation and stays in the phylo PRIOR only, exactly like the
     σ-phylo route's documented "species whose every row is missing stays in
     the prior with no likelihood term" (`gaussian_core.jl`, the σ-phylo
     missing-response block).
  2. integer tip index `1:phy.n_leaves`.
  3. positional (`_group_index`, first-seen order) — the only tier with no
     name information to anchor to, so it requires `labels` to visit every
     tip exactly once (`G == phy.n_leaves`); this is the pre-#482 behaviour,
     kept as a last-resort fallback.

Throws an `ArgumentError` naming the mismatch if none of the three tiers
apply (e.g. group labels that are neither tree tip names nor valid integer
indices, and do not even form a complete one-level-per-tip bijection).
"""
function _phylo_mean_leaf_index(phy::AugmentedPhy, labels)
    by_name = Dict(phy.leaf_names[i] => i for i in 1:phy.n_leaves)
    leaf = Vector{Int}(undef, length(labels))
    matched_names = true
    @inbounds for i in eachindex(labels)
        key = string(labels[i])
        if haskey(by_name, key)
            leaf[i] = by_name[key]
        else
            matched_names = false
            break
        end
    end
    matched_names && return leaf

    numeric_labels = true
    @inbounds for i in eachindex(labels)
        li = labels[i]
        if li isa Integer && 1 <= Int(li) <= phy.n_leaves
            leaf[i] = Int(li)
        else
            numeric_labels = false
            break
        end
    end
    numeric_labels && return leaf

    gidx, G = _group_index(labels)
    G == phy.n_leaves ||
        throw(ArgumentError("drm (Gaussian phylo mean): the group column has $(G) distinct " *
            "level(s) but the tree has $(phy.n_leaves) leaves, and the levels are not the " *
            "tree's leaf names or integer tip indices either — phylo group labels must match " *
            "tree tip names, integer tip indices, or (as a last resort) supply exactly one " *
            "level per tree tip."))
    return gidx
end

# ─────────────────────────────────────────────────────────────────────────────
# Core Woodbury helpers — all O(p) via sparse Cholesky
# ─────────────────────────────────────────────────────────────────────────────

"""
Build M = P + (1/σ²) S'S.  Return (P, M, chM, chP).
chP = chol(P + ridge) for logdet P; chM = chol(M) for the Woodbury solves.
"""
function build_M(prob::LocOnlyProblem, σ²_phy::Float64, σ²::Float64)
    n_keep = prob.n_keep
    P = (1 / σ²_phy) * prob.Q_cond
    M = copy(P)
    @inbounds for j in 1:n_keep
        if prob.STS_diag[j] > 0
            M[j, j] += prob.STS_diag[j] / σ²
        end
    end
    chM = cholesky(Symmetric(M); check=false)
    issuccess(chM) || (M += 1e-8 * I; chM = cholesky(Symmetric(M)))
    chP = cholesky(Symmetric(P + 1e-10 * I); check=false)
    issuccess(chP) || (chP = cholesky(Symmetric(P + 1e-6 * I)))
    return P, M, chM, chP
end

" V^{-1} z via Woodbury: (z - S (M^{-1} (1/σ²) S' z)) / σ² "
@inline function Vinv_mul(prob::LocOnlyProblem, chM, σ²::Float64, z)
    STz = prob.S' * z
    return (z .- prob.S * (chM \ (STz / σ²))) / σ²
end

" logdet V = n log σ² + logdet M − logdet P "
@inline function logdetV_val(prob::LocOnlyProblem, σ²::Float64, chM, chP)
    return prob.n * log(σ²) + logdet(chM) - logdet(chP)
end

# ─────────────────────────────────────────────────────────────────────────────
# Marginal log-likelihood
# ─────────────────────────────────────────────────────────────────────────────

function marginal_loglik(
    prob::LocOnlyProblem, β::AbstractVector, σ²_phy::Float64, σ²::Float64
)
    (σ²_phy <= 0 || σ² <= 0) && return -Inf
    P, M, chM, chP = build_M(prob, σ²_phy, σ²)
    e = prob.y .- prob.X * β
    Ve = Vinv_mul(prob, chM, σ², e)
    quad = dot(e, Ve)
    ldV = logdetV_val(prob, σ², chM, chP)
    return -0.5 * (prob.n * log(2π) + ldV + quad)
end

# The Woodbury representation subtracts two O(1/σ²) terms.  Once the
# residual variance is small relative to the phylogenetic variance, that
# subtraction no longer represents the marginal covariance: it can manufacture
# a large *negative* NLL and an apparently converged fit.  Treat that part of
# parameter space as numerically unavailable.  This is a representation limit,
# not a statistical lower bound; the dense small-model oracle remains available
# to investigate a genuine boundary estimate.
#
# WHERE THE BAR GOES (#574 follow-up, measured 2026-09-05).  The relative error
# of the subtraction grows as eps * (σ²_phy / σ²), so the ratio σ²/σ²_phy is
# the fraction of the mantissa that survives it.  #574 put the bar at
# σ²/σ²_phy >= eps -- exactly the point where NO digits survive, i.e. with
# zero margin -- and the objective is therefore still wrong just above it.
# Measured on the #461 fixture (G = 100, one row per species, replicate 38 of
# the B = 60 bootstrap), Woodbury vs a dense V = σ²I + σ²_phy·C oracle at
# lσ_phy = -26.9176:
#
#     lσ      log10(σ²/σ²_phy)   woodbury_ll     dense_ll     rel.diff
#    -30.0        -2.68          -6.27054e24  -6.27054e24    6.75e-14
#    -36.0        -7.89          -6.30213e24  -6.30213e24    1.93e-08
#    -40.0       -11.36          -6.30204e24  -6.30213e24    1.37e-05
#    -44.0       -14.84          +4.37745e25  -3.15149e25    2.39      <- sign flip
#
# The sign flip -- the whole #461 runaway -- happens at σ²/σ²_phy ~ 6 eps,
# INSIDE #574's bar, so the optimizer could still reach it and Optim reported
# convergence there.  It is a coin flip decided by rounding: the same fixture,
# seeds and source pass on x86_64 Linux CI and fail on aarch64 macOS.
#
# The bar is therefore the standard rule for a cancellation-limited difference:
# keep at least HALF the mantissa, σ²/σ²_phy >= sqrt(eps), which is
# 2(lσ - lσ_phy) >= 0.5 log(eps).  That is the -36.0 row above (rel.diff
# 1.93e-08 ~ sqrt(eps)), and it leaves ~8 nats of headroom over the observed
# break instead of 0.94.  In scale terms it refuses σ/σ_phy < 1.2e-4.
@inline function _loconly_resolvable_scales(lσ::Real, lσ_phy::Real)
    return lσ - lσ_phy >= 0.25 * log(eps(Float64))
end

function _loconly_marginal_nll(
    prob::LocOnlyProblem, β::AbstractVector, lσ::Real, lσ_phy::Real
)
    (isfinite(lσ) && isfinite(lσ_phy) && abs(lσ) < 50 && abs(lσ_phy) < 50 &&
     _loconly_resolvable_scales(lσ, lσ_phy)) ||
        return _LOCONLY_PENALTY
    σ²_phy = exp(2 * Float64(lσ_phy))
    σ² = exp(2 * Float64(lσ))
    ll = marginal_loglik(prob, β, σ²_phy, σ²)
    return isfinite(ll) ? -ll : _LOCONLY_PENALTY
end

function _loconly_profile_beta(prob::LocOnlyProblem, lσ::Real, lσ_phy::Real)
    (isfinite(lσ) && isfinite(lσ_phy) && abs(lσ) < 50 && abs(lσ_phy) < 50 &&
     _loconly_resolvable_scales(lσ, lσ_phy)) ||
        return nothing, _LOCONLY_PENALTY, nothing
    σ²_phy = exp(2 * Float64(lσ_phy))
    σ² = exp(2 * Float64(lσ))
    try
        _, _, chM, _ = build_M(prob, σ²_phy, σ²)
        VX = Vinv_mul(prob, chM, σ², prob.X)
        Vy = Vinv_mul(prob, chM, σ², prob.y)
        XtVX = prob.X' * VX
        β = XtVX \ (prob.X' * Vy)
        nll = _loconly_marginal_nll(prob, β, lσ, lσ_phy)
        return β, nll, chM
    catch
        return nothing, _LOCONLY_PENALTY, nothing
    end
end

_loconly_reml_constant_offset(prob::LocOnlyProblem) = 0.5 * prob.k * log(2π)

function _loconly_reml_components(prob::LocOnlyProblem, lσ::Real, lσ_phy::Real)
    β, ml_nll, chM = _loconly_profile_beta(prob, lσ, lσ_phy)
    if β === nothing || !isfinite(ml_nll)
        return (nll = _LOCONLY_PENALTY, ml_nll = ml_nll, penalty = NaN,
                beta = nothing, info = nothing, converged = false)
    end
    σ² = exp(2 * Float64(lσ))
    VX = Vinv_mul(prob, chM, σ², prob.X)
    info = Matrix(0.5 .* (prob.X' * VX .+ VX' * prob.X))
    ch = cholesky(Symmetric(info); check = false)
    if !issuccess(ch)
        return (nll = _LOCONLY_PENALTY, ml_nll = ml_nll, penalty = _LOCONLY_PENALTY,
                beta = β, info = info, converged = false)
    end
    penalty = sum(log, diag(ch.U))            # 0.5 * logdet(X'V^{-1}X)
    nll = ml_nll + penalty - _loconly_reml_constant_offset(prob)
    ok = isfinite(nll) && nll < _LOCONLY_PENALTY
    return (nll = ok ? nll : _LOCONLY_PENALTY, ml_nll = ml_nll, penalty = penalty,
            beta = β, info = info, converged = ok)
end

_loconly_reml_nll(prob::LocOnlyProblem, lσ::Real, lσ_phy::Real) =
    _loconly_reml_components(prob, lσ, lσ_phy).nll

function _loconly_dense_reml_components(prob::LocOnlyProblem, lσ::Real, lσ_phy::Real)
    (isfinite(lσ) && isfinite(lσ_phy) && abs(lσ) < 50 && abs(lσ_phy) < 50) ||
        return (nll = _LOCONLY_PENALTY, ml_nll = _LOCONLY_PENALTY,
                penalty = NaN, beta = nothing, info = nothing,
                converged = false, matrix_mode = :dense_developer)
    σ² = exp(2 * Float64(lσ))
    σ²_phy = exp(2 * Float64(lσ_phy))
    try
        S = Matrix(prob.S)
        Qinv = Matrix(inv(Symmetric(Matrix(prob.Q_cond))))
        V = σ² .* Matrix{Float64}(I, prob.n, prob.n) .+ σ²_phy .* (S * Qinv * S')
        chV = cholesky(Symmetric(V); check = false)
        if !issuccess(chV)
            return (nll = _LOCONLY_PENALTY, ml_nll = _LOCONLY_PENALTY,
                    penalty = _LOCONLY_PENALTY, beta = nothing, info = nothing,
                    converged = false, matrix_mode = :dense_developer)
        end
        VinvX = chV \ prob.X
        Vinvy = chV \ prob.y
        info = Matrix(0.5 .* (prob.X' * VinvX .+ VinvX' * prob.X))
        chX = cholesky(Symmetric(info); check = false)
        if !issuccess(chX)
            return (nll = _LOCONLY_PENALTY, ml_nll = _LOCONLY_PENALTY,
                    penalty = _LOCONLY_PENALTY, beta = nothing, info = info,
                    converged = false, matrix_mode = :dense_developer)
        end
        β = info \ (prob.X' * Vinvy)
        r = prob.y .- prob.X * β
        ml_nll = 0.5 * (prob.n * log(2π) + logdet(chV) + dot(r, chV \ r))
        penalty = sum(log, diag(chX.U))
        nll = ml_nll + penalty - _loconly_reml_constant_offset(prob)
        ok = isfinite(nll) && nll < _LOCONLY_PENALTY
        return (nll = ok ? nll : _LOCONLY_PENALTY, ml_nll = ml_nll,
                penalty = penalty, beta = β, info = info, converged = ok,
                matrix_mode = :dense_developer)
    catch err
        err isa InterruptException && rethrow(err)
        return (nll = _LOCONLY_PENALTY, ml_nll = _LOCONLY_PENALTY,
                penalty = NaN, beta = nothing, info = nothing,
                converged = false, matrix_mode = :dense_developer)
    end
end

_loconly_dense_reml_nll(prob::LocOnlyProblem, lσ::Real, lσ_phy::Real) =
    _loconly_dense_reml_components(prob, lσ, lσ_phy).nll

function _loconly_dense_comparator_diagnostic(prob::LocOnlyProblem, lσ::Real, lσ_phy::Real)
    sparse = _loconly_reml_components(prob, lσ, lσ_phy)
    dense = _loconly_dense_reml_components(prob, lσ, lσ_phy)
    nll_absdiff = abs(sparse.nll - dense.nll)
    beta_absdiff = (sparse.beta === nothing || dense.beta === nothing) ? NaN :
        maximum(abs.(sparse.beta .- dense.beta))
    info_absdiff = (sparse.info === nothing || dense.info === nothing) ? NaN :
        maximum(abs.(sparse.info .- dense.info))
    finite = sparse.converged && dense.converged &&
        isfinite(nll_absdiff) && isfinite(beta_absdiff) && isfinite(info_absdiff)
    return (
        target = :gaussian_loconly_reml,
        comparator = :dense_same_estimand_oracle,
        sparse_nll = sparse.nll,
        dense_nll = dense.nll,
        nll_absdiff = nll_absdiff,
        beta_absdiff = beta_absdiff,
        info_absdiff = info_absdiff,
        finite = finite,
        matrix_mode = :dense_developer,
    )
end

function _loconly_reml_dense_score_diagnostic(prob::LocOnlyProblem, lσ::Real, lσ_phy::Real;
                                             h::Real = 1e-5)
    (isfinite(lσ) && isfinite(lσ_phy) && abs(lσ) < 50 && abs(lσ_phy) < 50) ||
        return (target = :gaussian_loconly_reml, parameterization = :log_sd,
                score = fill(NaN, 2), fd_score = fill(NaN, 2),
                max_absdiff = NaN, finite = false, h = Float64(h),
                matrix_mode = :dense_developer)
    θ = [Float64(lσ), Float64(lσ_phy)]
    try
        σ² = exp(2 * θ[1])
        σ²_phy = exp(2 * θ[2])
        S = Matrix(prob.S)
        Qinv = Matrix(inv(Symmetric(Matrix(prob.Q_cond))))
        Cobs = S * Qinv * S'
        In = Matrix{Float64}(I, prob.n, prob.n)
        V = σ² .* In .+ σ²_phy .* Cobs
        chV = cholesky(Symmetric(V); check = false)
        if !issuccess(chV)
            return (target = :gaussian_loconly_reml, parameterization = :log_sd,
                    score = fill(NaN, 2), fd_score = fill(NaN, 2),
                    max_absdiff = NaN, finite = false, h = Float64(h),
                    matrix_mode = :dense_developer)
        end
        Vinv = chV \ In
        XtVX = prob.X' * Vinv * prob.X
        chX = cholesky(Symmetric(XtVX); check = false)
        if !issuccess(chX)
            return (target = :gaussian_loconly_reml, parameterization = :log_sd,
                    score = fill(NaN, 2), fd_score = fill(NaN, 2),
                    max_absdiff = NaN, finite = false, h = Float64(h),
                    matrix_mode = :dense_developer)
        end
        Pinv = chX \ Matrix{Float64}(I, size(XtVX, 1), size(XtVX, 2))
        P = Vinv .- Vinv * prob.X * Pinv * prob.X' * Vinv
        py = P * prob.y
        dV = (2 * σ² .* In, 2 * σ²_phy .* Cobs)
        score = [0.5 * (tr(P * dV[i]) - dot(py, dV[i] * py)) for i in 1:2]
        fd_score = _loconly_fd_gradient2(v -> _loconly_reml_nll(prob, v[1], v[2]), θ; h = h)
        max_absdiff = maximum(abs.(score .- fd_score))
        finite = all(isfinite, score) && all(isfinite, fd_score) && isfinite(max_absdiff)
        return (target = :gaussian_loconly_reml, parameterization = :log_sd,
                score = score, fd_score = fd_score, max_absdiff = max_absdiff,
                finite = finite, h = Float64(h), matrix_mode = :dense_developer)
    catch err
        err isa InterruptException && rethrow(err)
        return (target = :gaussian_loconly_reml, parameterization = :log_sd,
                score = fill(NaN, 2), fd_score = fill(NaN, 2),
                max_absdiff = NaN, finite = false, h = Float64(h),
                matrix_mode = :dense_developer)
    end
end

function _loconly_reml_sparse_score_diagnostic(prob::LocOnlyProblem, lσ::Real, lσ_phy::Real;
                                               h::Real = 1e-5)
    fail = (target = :gaussian_loconly_reml, parameterization = :log_sd,
            score = fill(NaN, 2), dense_score = fill(NaN, 2), fd_score = fill(NaN, 2),
            trace_terms = fill(NaN, 2), quadratic_terms = fill(NaN, 2),
            correction_terms = fill(NaN, 2), max_absdiff_dense = NaN,
            max_absdiff_fd = NaN, finite = false, h = Float64(h),
            matrix_mode = :sparse_woodbury_developer)
    (isfinite(lσ) && isfinite(lσ_phy) && abs(lσ) < 50 && abs(lσ_phy) < 50) ||
        return fail
    θ = [Float64(lσ), Float64(lσ_phy)]
    try
        σ² = exp(2 * θ[1])
        σ²_phy = exp(2 * θ[2])
        _, _, chM, _ = build_M(prob, σ²_phy, σ²)
        chQ = cholesky(Symmetric(prob.Q_cond); check = false)
        issuccess(chQ) || return fail
        VX = Vinv_mul(prob, chM, σ², prob.X)
        Vy = Vinv_mul(prob, chM, σ², prob.y)
        info = Matrix(0.5 .* (prob.X' * VX .+ VX' * prob.X))
        chX = cholesky(Symmetric(info); check = false)
        issuccess(chX) || return fail
        Pinv = chX \ Matrix{Float64}(I, size(info, 1), size(info, 2))

        py = Vy .- VX * (Pinv * (prob.X' * Vy))

        _, tr_SMS = exact_traces(prob, chM)
        trVinv = prob.n / σ² - tr_SMS / (σ²^2)
        residual_correction = 2 * σ² * tr(Pinv * (VX' * VX))
        residual_trace = 2 * σ² * trVinv - residual_correction
        residual_quad = 2 * σ² * dot(py, py)

        S_dense = Matrix(prob.S)
        VinvS = Vinv_mul(prob, chM, σ², S_dense)
        STVinvS = Matrix(prob.S' * VinvS)
        Q_STVinvS = chQ \ STVinvS
        STVX = Matrix(prob.S' * VX)
        Q_STVX = chQ \ STVX
        phylo_correction = 2 * σ²_phy * tr(Pinv * (STVX' * Q_STVX))
        phylo_trace = 2 * σ²_phy * tr(Q_STVinvS) - phylo_correction
        STpy = prob.S' * py
        phylo_quad = 2 * σ²_phy * dot(STpy, chQ \ STpy)

        trace_terms = [residual_trace, phylo_trace]
        quadratic_terms = [residual_quad, phylo_quad]
        correction_terms = [residual_correction, phylo_correction]
        score = 0.5 .* (trace_terms .- quadratic_terms)
        dense = _loconly_reml_dense_score_diagnostic(prob, lσ, lσ_phy; h = h)
        fd_score = _loconly_fd_gradient2(v -> _loconly_reml_nll(prob, v[1], v[2]), θ; h = h)
        max_absdiff_dense = maximum(abs.(score .- dense.score))
        max_absdiff_fd = maximum(abs.(score .- fd_score))
        finite = dense.finite && all(isfinite, score) && all(isfinite, fd_score) &&
            all(isfinite, trace_terms) && all(isfinite, quadratic_terms) &&
            isfinite(max_absdiff_dense) && isfinite(max_absdiff_fd)
        return (target = :gaussian_loconly_reml, parameterization = :log_sd,
                score = score, dense_score = dense.score, fd_score = fd_score,
                trace_terms = trace_terms, quadratic_terms = quadratic_terms,
                correction_terms = correction_terms,
                max_absdiff_dense = max_absdiff_dense,
                max_absdiff_fd = max_absdiff_fd, finite = finite, h = Float64(h),
                matrix_mode = :sparse_woodbury_developer)
    catch err
        err isa InterruptException && rethrow(err)
        return fail
    end
end

function _loconly_reml_fd_stability_diagnostic(prob::LocOnlyProblem, lσ::Real, lσ_phy::Real;
                                               steps = (1e-3, 1e-4, 1e-5))
    θ = [Float64(lσ), Float64(lσ_phy)]
    obj(v) = _loconly_reml_nll(prob, v[1], v[2])
    gradients = [_loconly_fd_gradient2(obj, θ; h = h) for h in steps]
    hessians = [_loconly_fd_hessian2(obj, θ; h = h) for h in steps]
    gradient_max_absdiff = length(gradients) <= 1 ? 0.0 :
        maximum(maximum(abs.(gradients[i] .- gradients[j]))
                for i in eachindex(gradients), j in eachindex(gradients) if i < j)
    hessian_max_absdiff = length(hessians) <= 1 ? 0.0 :
        maximum(maximum(abs.(hessians[i] .- hessians[j]))
                for i in eachindex(hessians), j in eachindex(hessians) if i < j)
    finite = all(g -> all(isfinite, g), gradients) && all(H -> all(isfinite, H), hessians)
    return (
        target = :gaussian_loconly_reml,
        parameterization = :log_sd,
        steps = Tuple(Float64.(steps)),
        gradients = gradients,
        hessians = hessians,
        gradient_max_absdiff = gradient_max_absdiff,
        hessian_max_absdiff = hessian_max_absdiff,
        finite = finite,
        matrix_mode = :dense_developer,
    )
end

function _loconly_reml_boundary_status(prob::LocOnlyProblem, lσ::Real, lσ_phy::Real;
                                       zero_tol::Real = 1e-7)
    if !(isfinite(lσ) && isfinite(lσ_phy) && abs(lσ) < 50 && abs(lσ_phy) < 50)
        return (boundary_status = :nonfinite_objective, finite = false,
                converged = false, sigma = NaN, sigma_phy = NaN)
    end
    σ = exp(Float64(lσ))
    σ_phy = exp(Float64(lσ_phy))
    rank(prob.X) < prob.k && return (
        boundary_status = :singular_fixed_effect_information,
        finite = false,
        converged = false,
        sigma = σ,
        sigma_phy = σ_phy,
    )
    comp = _loconly_reml_components(prob, lσ, lσ_phy)
    if !comp.converged
        status = comp.info === nothing || !all(isfinite, comp.info) ?
            :nonfinite_objective : :singular_fixed_effect_information
        return (boundary_status = status, finite = false, converged = false,
                sigma = σ, sigma_phy = σ_phy)
    end
    status = (σ <= zero_tol || σ_phy <= zero_tol) ? :near_zero_variance : :interior
    return (boundary_status = status, finite = true, converged = true,
            sigma = σ, sigma_phy = σ_phy)
end

function _loconly_reml_local_profile_diagnostic(prob::LocOnlyProblem, lσ::Real, lσ_phy::Real;
                                                step::Real = 0.05)
    θ = [Float64(lσ), Float64(lσ_phy)]
    h = Float64(step)
    obj(v) = _loconly_reml_nll(prob, v[1], v[2])
    center = obj(θ)
    residual_axis = [obj([θ[1] - h, θ[2]]), center, obj([θ[1] + h, θ[2]])]
    phylo_axis = [obj([θ[1], θ[2] - h]), center, obj([θ[1], θ[2] + h])]
    finite = isfinite(center) && all(isfinite, residual_axis) && all(isfinite, phylo_axis)
    axis_min = finite &&
        center <= minimum((residual_axis[1], residual_axis[3])) + 1e-8 &&
        center <= minimum((phylo_axis[1], phylo_axis[3])) + 1e-8
    return (
        target = :gaussian_loconly_reml,
        parameterization = :log_sd,
        center = center,
        step = h,
        residual_axis = residual_axis,
        phylo_axis = phylo_axis,
        finite = finite,
        center_is_axis_min = axis_min,
    )
end

# Guarded: a failed probe (non-finite, or the `_LOCONLY_PENALTY` sentinel) returns an
# all-NaN 2x2 matrix -- the convention every caller already uses for a failed
# diagnostic (`finite = false`) -- instead of differencing the sentinel into a
# finite garbage Hessian.
function _loconly_fd_hessian2(f, θ::AbstractVector{<:Real}; h::Real = 1e-4)
    H, ok = _fd_hessian_from_values(f, Float64.(θ); hstep = h, sentinel = _LOCONLY_PENALTY)
    return ok ? 0.5 .* (H .+ H') : fill(NaN, 2, 2)
end

function _loconly_fd_gradient2(f, θ::AbstractVector{<:Real}; h::Real = 1e-5)
    g = zeros(2)
    x = Float64.(θ)
    for i in 1:2
        ei = zeros(2)
        si = h * max(abs(x[i]), 1.0)
        ei[i] = si
        g[i] = (f(x .+ ei) - f(x .- ei)) / (2 * si)
    end
    return g
end

function _loconly_ai_information_diagnostic(prob::LocOnlyProblem, lσ::Real, lσ_phy::Real;
                                            h::Real = 1e-4)
    (isfinite(lσ) && isfinite(lσ_phy) && abs(lσ) < 50 && abs(lσ_phy) < 50) ||
        return (target = :gaussian_loconly_reml, parameterization = :log_sd,
                ai = fill(NaN, 2, 2), observed = fill(NaN, 2, 2),
                relative_error = NaN, finite = false, h = Float64(h),
                matrix_mode = :dense_developer)
    θ = [Float64(lσ), Float64(lσ_phy)]
    try
        σ² = exp(2 * θ[1])
        σ²_phy = exp(2 * θ[2])
        n = prob.n
        S = Matrix(prob.S)
        Qinv = Matrix(inv(Symmetric(Matrix(prob.Q_cond))))
        Cobs = S * Qinv * S'
        In = Matrix{Float64}(I, n, n)
        V = σ² .* In .+ σ²_phy .* Cobs
        chV = cholesky(Symmetric(V); check = false)
        if !issuccess(chV)
            return (target = :gaussian_loconly_reml, parameterization = :log_sd,
                    ai = fill(NaN, 2, 2), observed = fill(NaN, 2, 2),
                    relative_error = NaN, finite = false, h = Float64(h),
                    matrix_mode = :dense_developer)
        end
        Vinv = chV \ In
        XtVX = prob.X' * Vinv * prob.X
        chX = cholesky(Symmetric(XtVX); check = false)
        if !issuccess(chX)
            return (target = :gaussian_loconly_reml, parameterization = :log_sd,
                    ai = fill(NaN, 2, 2), observed = fill(NaN, 2, 2),
                    relative_error = NaN, finite = false, h = Float64(h),
                    matrix_mode = :dense_developer)
        end
        Pinv = chX \ Matrix{Float64}(I, size(XtVX, 1), size(XtVX, 2))
        P = Vinv .- Vinv * prob.X * Pinv * prob.X' * Vinv
        py = P * prob.y
        dV = (2 * σ² .* In, 2 * σ²_phy .* Cobs)
        ai = zeros(2, 2)
        for i in 1:2, j in 1:2
            ai[i, j] = 0.5 * dot(py, dV[i] * (P * (dV[j] * py)))
        end
        ai .= 0.5 .* (ai .+ ai')
        observed = _loconly_fd_hessian2(v -> _loconly_reml_nll(prob, v[1], v[2]), θ; h = h)
        rel = norm(ai .- observed) / max(norm(observed), eps(Float64))
        finite = all(isfinite, ai) && all(isfinite, observed) && isfinite(rel)
        return (target = :gaussian_loconly_reml, parameterization = :log_sd,
                ai = ai, observed = observed, relative_error = rel, finite = finite,
                h = Float64(h), matrix_mode = :dense_developer)
    catch err
        err isa InterruptException && rethrow(err)
        return (target = :gaussian_loconly_reml, parameterization = :log_sd,
                ai = fill(NaN, 2, 2), observed = fill(NaN, 2, 2),
                relative_error = NaN, finite = false, h = Float64(h),
                matrix_mode = :dense_developer)
    end
end

function _loconly_reml_sparse_ai_information_diagnostic(prob::LocOnlyProblem,
                                                        lσ::Real, lσ_phy::Real;
                                                        h::Real = 1e-4,
                                                        verify_dense::Bool = true)
    fail = (target = :gaussian_loconly_reml, parameterization = :log_sd,
            ai = fill(NaN, 2, 2), dense_ai = fill(NaN, 2, 2),
            observed = fill(NaN, 2, 2), max_absdiff_dense = NaN,
            relative_error_observed = NaN, finite = false, h = Float64(h),
            matrix_mode = :sparse_woodbury_developer)
    (isfinite(lσ) && isfinite(lσ_phy) && abs(lσ) < 50 && abs(lσ_phy) < 50) ||
        return fail
    θ = [Float64(lσ), Float64(lσ_phy)]
    try
        σ² = exp(2 * θ[1])
        σ²_phy = exp(2 * θ[2])
        _, _, chM, _ = build_M(prob, σ²_phy, σ²)
        chQ = cholesky(Symmetric(prob.Q_cond); check = false)
        issuccess(chQ) || return fail
        VX = Vinv_mul(prob, chM, σ², prob.X)
        info = Matrix(0.5 .* (prob.X' * VX .+ VX' * prob.X))
        chX = cholesky(Symmetric(info); check = false)
        issuccess(chX) || return fail
        Pinv = chX \ Matrix{Float64}(I, size(info, 1), size(info, 2))

        function P_mul(z)
            Vz = Vinv_mul(prob, chM, σ², z)
            return Vz .- VX * (Pinv * (prob.X' * Vz))
        end
        function dV_mul(axis::Int, z)
            if axis == 1
                return 2 * σ² .* z
            end
            return 2 * σ²_phy .* (prob.S * (chQ \ (prob.S' * z)))
        end

        py = P_mul(prob.y)
        ai = zeros(2, 2)
        for i in 1:2, j in 1:2
            ai[i, j] = 0.5 * dot(py, dV_mul(i, P_mul(dV_mul(j, py))))
        end
        ai .= 0.5 .* (ai .+ ai')

        dense = verify_dense ?
            _loconly_ai_information_diagnostic(prob, lσ, lσ_phy; h = h) :
            nothing
        dense_ai = dense === nothing ? fill(NaN, 2, 2) : dense.ai
        observed = dense === nothing ? fill(NaN, 2, 2) : dense.observed
        max_absdiff_dense = dense === nothing ? NaN : maximum(abs.(ai .- dense.ai))
        rel_obs = dense === nothing ? NaN :
            norm(ai .- observed) / max(norm(observed), eps(Float64))
        finite = all(isfinite, ai) && (dense === nothing ||
            (dense.finite && all(isfinite, observed) &&
             isfinite(max_absdiff_dense) && isfinite(rel_obs)))
        return (target = :gaussian_loconly_reml, parameterization = :log_sd,
                ai = ai, dense_ai = dense_ai, observed = observed,
                max_absdiff_dense = max_absdiff_dense,
                relative_error_observed = rel_obs, finite = finite,
                h = Float64(h), matrix_mode = :sparse_woodbury_developer)
    catch err
        err isa InterruptException && rethrow(err)
        return fail
    end
end

function _loconly_reml_optimizer_diagnostic(prob::LocOnlyProblem; starts = nothing,
                                            g_tol::Real = 1e-6,
                                            iterations::Integer = 80)
    β0 = prob.X \ prob.y
    s0 = std(prob.y .- prob.X * β0) + eps()
    start_list = starts === nothing ? [
        [log(s0 + eps()), log(s0 / 2 + eps())],
        [log(s0 / sqrt(2) + eps()), log(s0 / sqrt(2) + eps())],
        [log(s0 / 2 + eps()), log(s0 + eps())],
    ] : [Float64.(s) for s in starts]

    obj(v) = _loconly_reml_nll(prob, v[1], v[2])
    function fg!(F, G, v)
        val = obj(v)
        G !== nothing && copyto!(G, _loconly_fd_gradient2(obj, v))
        return F === nothing ? nothing : val
    end

    od = Optim.NLSolversBase.only_fg!(fg!)
    ls = Optim.LBFGS(; linesearch = Optim.LineSearches.BackTracking(; order = 3))
    opts = Optim.Options(; g_tol = Float64(g_tol), iterations = iterations, f_reltol = 1e-8)
    records = Any[]
    best = nothing
    best_nll = Inf
    for (i, start) in enumerate(start_list)
        start_nll = obj(start)
        try
            res = Optim.optimize(od, copy(start), ls, opts)
            v = Float64.(Optim.minimizer(res))
            nll = obj(v)   # NOT Optim.minimum(res): see optim_minimum_guard.jl
            grad = _loconly_fd_gradient2(obj, v)
            record = (
                start_index = i,
                start = copy(start),
                start_nll = start_nll,
                minimizer = v,
                nll = nll,
                converged = Optim.converged(res),
                iterations = Optim.iterations(res),
                g_norm = maximum(abs, grad),
                status = isfinite(nll) ? :finite : :nonfinite,
            )
            push!(records, record)
            if isfinite(nll) && nll < best_nll
                best = record
                best_nll = nll
            end
        catch err
            err isa InterruptException && rethrow(err)
            push!(records, (
                start_index = i,
                start = copy(start),
                start_nll = start_nll,
                minimizer = fill(NaN, 2),
                nll = NaN,
                converged = false,
                iterations = 0,
                g_norm = NaN,
                status = :optimizer_error,
            ))
        end
    end

    finite = best !== nothing && isfinite(best.nll)
    g_norm = finite ? best.g_norm : NaN
    accepted = finite && (best.converged || (isfinite(g_norm) && g_norm <= max(1e-3, 10 * g_tol)))
    observed = finite ?
        _loconly_fd_hessian2(v -> _loconly_reml_nll(prob, v[1], v[2]), best.minimizer) :
        fill(NaN, 2, 2)
    observed_eig = try
        eigvals(Symmetric(observed))
    catch err
        err isa InterruptException && rethrow(err)
        fill(NaN, 2)
    end
    observed_pd = all(isfinite, observed_eig) && minimum(observed_eig) > 0
    dense_cmp = finite ?
        _loconly_dense_comparator_diagnostic(prob, best.minimizer[1], best.minimizer[2]) :
        nothing
    boundary = finite ?
        _loconly_reml_boundary_status(prob, best.minimizer[1], best.minimizer[2]) :
        (boundary_status = :nonfinite_objective, finite = false,
         converged = false, sigma = NaN, sigma_phy = NaN)
    local_profile = finite ?
        _loconly_reml_local_profile_diagnostic(prob, best.minimizer[1], best.minimizer[2]) :
        nothing
    fd_stability = finite ?
        _loconly_reml_fd_stability_diagnostic(prob, best.minimizer[1], best.minimizer[2]) :
        nothing
    dense_score = finite ?
        _loconly_reml_dense_score_diagnostic(prob, best.minimizer[1], best.minimizer[2]) :
        nothing
    finite_records = filter(r -> r.status === :finite && isfinite(r.nll), records)
    n_finite_records = length(finite_records)
    n_accepted_records = count(r -> r.status === :finite && isfinite(r.g_norm) &&
                                  r.g_norm <= max(1e-3, 10 * g_tol), records)
    start_nll_best = isempty(records) ? NaN : minimum(r -> r.start_nll, records)
    best_improvement = finite && isfinite(start_nll_best) ? start_nll_best - best.nll : NaN
    return (
        target = :gaussian_loconly_reml,
        estimator = :fd_reml_optimizer_experiment,
        parameterization = :log_sd,
        optimizer = :lbfgs_fd_gradient,
        finite = finite,
        accepted = accepted,
        best_start_index = finite ? best.start_index : 0,
        best_minimizer = finite ? best.minimizer : fill(NaN, 2),
        best_nll = finite ? best.nll : NaN,
        best_g_norm = g_norm,
        observed_hessian = observed,
        observed_hessian_eig = observed_eig,
        observed_hessian_pd = observed_pd,
        dense_comparator = dense_cmp,
        boundary_status = boundary.boundary_status,
        local_profile = local_profile,
        fd_stability = fd_stability,
        dense_score = dense_score,
        best_score_norm = dense_score === nothing ? NaN : maximum(abs, dense_score.score),
        n_starts = length(start_list),
        n_finite_records = n_finite_records,
        n_accepted_records = n_accepted_records,
        best_improvement = best_improvement,
        records = records,
        claim_status = :optimizer_experiment,
        ai_reml_ready = false,
        reason_not_ai_reml = "finite-difference gradient experiment; no average-information update is implemented",
    )
end

function _loconly_reml_dense_score_optimizer_diagnostic(prob::LocOnlyProblem; starts = nothing,
                                                        g_tol::Real = 1e-6,
                                                        iterations::Integer = 80)
    β0 = prob.X \ prob.y
    s0 = std(prob.y .- prob.X * β0) + eps()
    start_list = starts === nothing ? [
        [log(s0 + eps()), log(s0 / 2 + eps())],
        [log(s0 / sqrt(2) + eps()), log(s0 / sqrt(2) + eps())],
        [log(s0 / 2 + eps()), log(s0 + eps())],
    ] : [Float64.(s) for s in starts]

    obj(v) = _loconly_reml_nll(prob, v[1], v[2])
    function fg!(F, G, v)
        val = obj(v)
        if G !== nothing
            score = _loconly_reml_dense_score_diagnostic(prob, v[1], v[2]).score
            copyto!(G, all(isfinite, score) ? score : zeros(2))
        end
        return F === nothing ? nothing : val
    end

    od = Optim.NLSolversBase.only_fg!(fg!)
    ls = Optim.LBFGS(; linesearch = Optim.LineSearches.BackTracking(; order = 3))
    opts = Optim.Options(; g_tol = Float64(g_tol), iterations = iterations, f_reltol = 1e-8)
    records = Any[]
    best = nothing
    best_nll = Inf
    for (i, start) in enumerate(start_list)
        start_nll = obj(start)
        try
            res = Optim.optimize(od, copy(start), ls, opts)
            v = Float64.(Optim.minimizer(res))
            nll = obj(v)   # NOT Optim.minimum(res): see optim_minimum_guard.jl
            score = _loconly_reml_dense_score_diagnostic(prob, v[1], v[2]).score
            record = (
                start_index = i,
                start = copy(start),
                start_nll = start_nll,
                minimizer = v,
                nll = nll,
                converged = Optim.converged(res),
                iterations = Optim.iterations(res),
                score_norm = maximum(abs, score),
                status = isfinite(nll) && all(isfinite, score) ? :finite : :nonfinite,
            )
            push!(records, record)
            if record.status === :finite && nll < best_nll
                best = record
                best_nll = nll
            end
        catch err
            err isa InterruptException && rethrow(err)
            push!(records, (
                start_index = i,
                start = copy(start),
                start_nll = start_nll,
                minimizer = fill(NaN, 2),
                nll = NaN,
                converged = false,
                iterations = 0,
                score_norm = NaN,
                status = :optimizer_error,
            ))
        end
    end

    finite = best !== nothing && isfinite(best.nll)
    dense_cmp = finite ?
        _loconly_dense_comparator_diagnostic(prob, best.minimizer[1], best.minimizer[2]) :
        nothing
    boundary = finite ?
        _loconly_reml_boundary_status(prob, best.minimizer[1], best.minimizer[2]) :
        (boundary_status = :nonfinite_objective, finite = false,
         converged = false, sigma = NaN, sigma_phy = NaN)
    return (
        target = :gaussian_loconly_reml,
        estimator = :dense_score_reml_optimizer_experiment,
        parameterization = :log_sd,
        optimizer = :lbfgs_dense_reml_score,
        finite = finite,
        accepted = finite && (best.converged || best.score_norm <= max(1e-3, 10 * g_tol)),
        best_start_index = finite ? best.start_index : 0,
        best_minimizer = finite ? best.minimizer : fill(NaN, 2),
        best_nll = finite ? best.nll : NaN,
        best_score_norm = finite ? best.score_norm : NaN,
        dense_comparator = dense_cmp,
        boundary_status = boundary.boundary_status,
        n_starts = length(start_list),
        n_finite_records = count(r -> r.status === :finite, records),
        records = records,
        claim_status = :optimizer_experiment,
        ai_reml_ready = false,
        reason_not_ai_reml = "dense analytic score optimizer experiment; no average-information update is implemented",
    )
end

function _loconly_reml_sparse_score_optimizer_diagnostic(prob::LocOnlyProblem; starts = nothing,
                                                         g_tol::Real = 1e-6,
                                                         iterations::Integer = 80)
    β0 = prob.X \ prob.y
    s0 = std(prob.y .- prob.X * β0) + eps()
    start_list = starts === nothing ? [
        [log(s0 + eps()), log(s0 / 2 + eps())],
        [log(s0 / sqrt(2) + eps()), log(s0 / sqrt(2) + eps())],
        [log(s0 / 2 + eps()), log(s0 + eps())],
    ] : [Float64.(s) for s in starts]

    obj(v) = _loconly_reml_nll(prob, v[1], v[2])
    function fg!(F, G, v)
        val = obj(v)
        if G !== nothing
            score = _loconly_reml_sparse_score_diagnostic(prob, v[1], v[2]).score
            copyto!(G, all(isfinite, score) ? score : zeros(2))
        end
        return F === nothing ? nothing : val
    end

    od = Optim.NLSolversBase.only_fg!(fg!)
    ls = Optim.LBFGS(; linesearch = Optim.LineSearches.BackTracking(; order = 3))
    opts = Optim.Options(; g_tol = Float64(g_tol), iterations = iterations, f_reltol = 1e-8)
    records = Any[]
    best = nothing
    best_nll = Inf
    for (i, start) in enumerate(start_list)
        start_nll = obj(start)
        try
            res = Optim.optimize(od, copy(start), ls, opts)
            v = Float64.(Optim.minimizer(res))
            nll = obj(v)   # NOT Optim.minimum(res): see optim_minimum_guard.jl
            sparse_score = _loconly_reml_sparse_score_diagnostic(prob, v[1], v[2])
            record = (
                start_index = i,
                start = copy(start),
                start_nll = start_nll,
                minimizer = v,
                nll = nll,
                converged = Optim.converged(res),
                iterations = Optim.iterations(res),
                score_norm = maximum(abs, sparse_score.score),
                max_absdiff_dense = sparse_score.max_absdiff_dense,
                status = isfinite(nll) && sparse_score.finite ? :finite : :nonfinite,
            )
            push!(records, record)
            if record.status === :finite && nll < best_nll
                best = record
                best_nll = nll
            end
        catch err
            err isa InterruptException && rethrow(err)
            push!(records, (
                start_index = i,
                start = copy(start),
                start_nll = start_nll,
                minimizer = fill(NaN, 2),
                nll = NaN,
                converged = false,
                iterations = 0,
                score_norm = NaN,
                max_absdiff_dense = NaN,
                status = :optimizer_error,
            ))
        end
    end

    finite = best !== nothing && isfinite(best.nll)
    dense_cmp = finite ?
        _loconly_dense_comparator_diagnostic(prob, best.minimizer[1], best.minimizer[2]) :
        nothing
    boundary = finite ?
        _loconly_reml_boundary_status(prob, best.minimizer[1], best.minimizer[2]) :
        (boundary_status = :nonfinite_objective, finite = false,
         converged = false, sigma = NaN, sigma_phy = NaN)
    sparse_score = finite ?
        _loconly_reml_sparse_score_diagnostic(prob, best.minimizer[1], best.minimizer[2]) :
        nothing
    return (
        target = :gaussian_loconly_reml,
        estimator = :sparse_score_reml_optimizer_experiment,
        parameterization = :log_sd,
        optimizer = :lbfgs_sparse_woodbury_reml_score,
        finite = finite,
        accepted = finite && (best.converged || best.score_norm <= max(1e-3, 10 * g_tol)),
        best_start_index = finite ? best.start_index : 0,
        best_minimizer = finite ? best.minimizer : fill(NaN, 2),
        best_nll = finite ? best.nll : NaN,
        best_score_norm = finite ? best.score_norm : NaN,
        best_max_absdiff_dense = finite ? best.max_absdiff_dense : NaN,
        dense_comparator = dense_cmp,
        boundary_status = boundary.boundary_status,
        sparse_score = sparse_score,
        n_starts = length(start_list),
        n_finite_records = count(r -> r.status === :finite, records),
        records = records,
        claim_status = :optimizer_experiment,
        ai_reml_ready = false,
        reason_not_ai_reml = "sparse Woodbury score optimizer experiment; no average-information update is implemented",
    )
end

function _loconly_reml_ai_update_optimizer_diagnostic(prob::LocOnlyProblem; starts = nothing,
                                                      g_tol::Real = 1e-6,
                                                      iterations::Integer = 30,
                                                      max_halvings::Integer = 12,
                                                      ridge::Real = 1e-8,
                                                      condition_limit::Real = 1e10)
    β0 = prob.X \ prob.y
    s0 = std(prob.y .- prob.X * β0) + eps()
    start_list = starts === nothing ? [
        [log(s0 + eps()), log(s0 / 2 + eps())],
        [log(s0 / sqrt(2) + eps()), log(s0 / sqrt(2) + eps())],
        [log(s0 / 2 + eps()), log(s0 + eps())],
    ] : [Float64.(s) for s in starts]

    obj(v) = _loconly_reml_nll(prob, v[1], v[2])
    records = Any[]
    best = nothing
    best_nll = Inf
    for (i, start) in enumerate(start_list)
        θ = copy(start)
        start_nll = obj(θ)
        iter_records = Any[]
        status = :max_iterations
        if !isfinite(start_nll)
            record = (
                start_index = i,
                start = copy(start),
                start_nll = start_nll,
                minimizer = fill(NaN, 2),
                nll = NaN,
                converged = false,
                iterations = 0,
                score_norm = NaN,
                status = :nonfinite_start,
                trace = iter_records,
            )
            push!(records, record)
            continue
        end
        for iter in 1:iterations
            nll_before = obj(θ)
            if !isfinite(nll_before)
                status = :nonfinite_objective
                break
            end
            score_diag = _loconly_reml_sparse_score_diagnostic(prob, θ[1], θ[2])
            if !score_diag.finite || !all(isfinite, score_diag.score)
                status = :nonfinite_score
                break
            end
            score = score_diag.score
            score_norm = maximum(abs, score)
            if score_norm <= g_tol
                status = :converged
                push!(iter_records, (
                    iteration = iter,
                    nll_before = nll_before,
                    nll_after = nll_before,
                    score_norm = score_norm,
                    step_norm = 0.0,
                    step_factor = 0.0,
                    halvings = 0,
                    ai_min_eig = NaN,
                    ai_condition = NaN,
                    ridge_added = 0.0,
                    status = :converged,
                ))
                break
            end
            info_diag = _loconly_reml_sparse_ai_information_diagnostic(
                prob, θ[1], θ[2]; verify_dense = false,
            )
            if !info_diag.finite || !all(isfinite, info_diag.ai)
                status = :nonfinite_information
                break
            end
            ai = Matrix(0.5 .* (info_diag.ai .+ info_diag.ai'))
            eig = try
                eigvals(Symmetric(ai))
            catch err
                err isa InterruptException && rethrow(err)
                fill(NaN, 2)
            end
            if !all(isfinite, eig)
                status = :singular_information
                break
            end
            max_eig = maximum(abs, eig)
            min_eig = minimum(eig)
            cond = max_eig / max(abs(min_eig), eps(Float64))
            if !isfinite(cond) || cond > condition_limit
                status = :ill_conditioned_information
                break
            end
            ridge_added = min_eig > ridge ? 0.0 : Float64(ridge) - min_eig + Float64(ridge)
            if ridge_added > 0
                ai .+= ridge_added .* Matrix{Float64}(I, 2, 2)
            end
            step = try
                ai \ score
            catch err
                err isa InterruptException && rethrow(err)
                fill(NaN, 2)
            end
            if !all(isfinite, step)
                status = :singular_information
                break
            end
            step_factor = 1.0
            halvings = 0
            candidate = θ .- step_factor .* step
            candidate_nll = obj(candidate)
            while (!(all(isfinite, candidate) && maximum(abs, candidate) < 50 &&
                     isfinite(candidate_nll) && candidate_nll < nll_before) &&
                   halvings < max_halvings)
                step_factor /= 2
                halvings += 1
                candidate = θ .- step_factor .* step
                candidate_nll = obj(candidate)
            end
            accepted_step = all(isfinite, candidate) && maximum(abs, candidate) < 50 &&
                isfinite(candidate_nll) && candidate_nll < nll_before
            push!(iter_records, (
                iteration = iter,
                nll_before = nll_before,
                nll_after = accepted_step ? candidate_nll : nll_before,
                score_norm = score_norm,
                step_norm = norm(step_factor .* step),
                step_factor = step_factor,
                halvings = halvings,
                ai_min_eig = min_eig,
                ai_condition = cond,
                ridge_added = ridge_added,
                status = accepted_step ? :accepted_step : :no_descent,
            ))
            if !accepted_step
                status = :no_descent
                break
            end
            θ .= candidate
        end

        final_nll = obj(θ)
        final_score = _loconly_reml_sparse_score_diagnostic(prob, θ[1], θ[2])
        final_score_norm = final_score.finite ? maximum(abs, final_score.score) : NaN
        if status === :max_iterations && isfinite(final_score_norm) && final_score_norm <= max(1e-3, 10 * g_tol)
            status = :accepted_by_score_tolerance
        end
        converged = status === :converged || status === :accepted_by_score_tolerance
        record = (
            start_index = i,
            start = copy(start),
            start_nll = start_nll,
            minimizer = copy(θ),
            nll = final_nll,
            converged = converged,
            iterations = length(iter_records),
            score_norm = final_score_norm,
            status = final_score.finite && isfinite(final_nll) ? status : :nonfinite_final,
            trace = iter_records,
        )
        push!(records, record)
        if final_score.finite && isfinite(final_nll) && final_nll < best_nll
            best = record
            best_nll = final_nll
        end
    end

    finite = best !== nothing && isfinite(best.nll)
    accepted = finite && (best.converged ||
        (isfinite(best.score_norm) && best.score_norm <= max(1e-3, 10 * g_tol)))
    dense_cmp = finite ?
        _loconly_dense_comparator_diagnostic(prob, best.minimizer[1], best.minimizer[2]) :
        nothing
    boundary = finite ?
        _loconly_reml_boundary_status(prob, best.minimizer[1], best.minimizer[2]) :
        (boundary_status = :nonfinite_objective, finite = false,
         converged = false, sigma = NaN, sigma_phy = NaN)
    sparse_score = finite ?
        _loconly_reml_sparse_score_diagnostic(prob, best.minimizer[1], best.minimizer[2]) :
        nothing
    sparse_information = finite ?
        _loconly_reml_sparse_ai_information_diagnostic(prob, best.minimizer[1], best.minimizer[2]) :
        nothing
    return (
        target = :gaussian_loconly_reml,
        estimator = :guarded_ai_update_reml_optimizer_experiment,
        parameterization = :log_sd,
        optimizer = :guarded_sparse_average_information_update,
        finite = finite,
        accepted = accepted,
        best_start_index = finite ? best.start_index : 0,
        best_minimizer = finite ? best.minimizer : fill(NaN, 2),
        best_nll = finite ? best.nll : NaN,
        best_score_norm = finite ? best.score_norm : NaN,
        dense_comparator = dense_cmp,
        boundary_status = boundary.boundary_status,
        sparse_score = sparse_score,
        sparse_information = sparse_information,
        n_starts = length(start_list),
        n_finite_records = count(r -> isfinite(r.nll), records),
        n_accepted_records = count(r -> r.converged || (isfinite(r.score_norm) &&
                                  r.score_norm <= max(1e-3, 10 * g_tol)), records),
        records = records,
        claim_status = :optimizer_experiment,
        ai_reml_ready = false,
        reason_not_ai_reml = "guarded average-information update experiment; no simulation, bridge, or coverage gate is implemented",
    )
end

function _loconly_reml_recovery_grid_diagnostic(; reps::Integer = 8,
                                                G::Integer = 8,
                                                n_per_species::Integer = 2,
                                                sigma::Real = 0.45,
                                                sigma_phy::Real = 0.7,
                                                beta = (0.25, -0.4),
                                                branch_length::Real = 0.25,
                                                seed::Integer = 20260624,
                                                iterations::Integer = 30)
    rng = MersenneTwister(seed)
    phy = random_balanced_tree(G; branch_length = branch_length)
    species = repeat(1:G, inner = n_per_species)
    n = length(species)
    x = range(-1.0, 1.0; length = n)
    X = hcat(ones(n), collect(x))
    β = Float64.(collect(beta))
    C = sigma_phy_dense(phy; σ²_phy = 1.0)
    L = cholesky(Symmetric(C)).L
    σ = Float64(sigma)
    σ_phy = Float64(sigma_phy)
    records = Any[]

    for rep in 1:reps
        u = σ_phy .* (L * randn(rng, G))
        y = X * β .+ u[species] .+ σ .* randn(rng, n)
        prob = make_loc_problem(phy, y, X; species = species)
        fit = _loconly_reml_ai_update_optimizer_diagnostic(prob; iterations = iterations)
        finite = fit.finite && all(isfinite, fit.best_minimizer)
        sigma_hat = finite ? exp(fit.best_minimizer[1]) : NaN
        sigma_phy_hat = finite ? exp(fit.best_minimizer[2]) : NaN
        push!(records, (
            rep = rep,
            accepted = fit.accepted,
            finite = finite,
            sigma_hat = sigma_hat,
            sigma_phy_hat = sigma_phy_hat,
            log_sigma_hat = finite ? fit.best_minimizer[1] : NaN,
            log_sigma_phy_hat = finite ? fit.best_minimizer[2] : NaN,
            sigma_error = sigma_hat - σ,
            sigma_phy_error = sigma_phy_hat - σ_phy,
            log_sigma_error = finite ? fit.best_minimizer[1] - log(σ) : NaN,
            log_sigma_phy_error = finite ? fit.best_minimizer[2] - log(σ_phy) : NaN,
            score_norm = fit.best_score_norm,
            nll = fit.best_nll,
            boundary_status = fit.boundary_status,
            optimizer_status = fit.accepted ? :accepted : :not_accepted,
            iterations = finite ? fit.records[fit.best_start_index].iterations : 0,
        ))
    end

    accepted = [r.accepted && r.finite for r in records]
    idx = findall(identity, accepted)
    n_accepted = length(idx)
    function metric(vals, f)
        n_accepted == 0 && return NaN
        return f([vals[i] for i in idx])
    end
    sigma_errors = [r.sigma_error for r in records]
    sigma_phy_errors = [r.sigma_phy_error for r in records]
    log_sigma_errors = [r.log_sigma_error for r in records]
    log_sigma_phy_errors = [r.log_sigma_phy_error for r in records]
    mcse(vals) = n_accepted <= 1 ? NaN : std([vals[i] for i in idx]) / sqrt(n_accepted)
    rmse(vals) = metric(vals, v -> sqrt(mean(abs2, v)))
    boundary_levels = (:interior, :near_zero_variance,
        :singular_fixed_effect_information, :nonfinite_objective)
    boundary_counts = NamedTuple{boundary_levels}(
        Tuple(count(r -> r.boundary_status === level, records) for level in boundary_levels),
    )
    return (
        target = :gaussian_loconly_phylo_reml,
        estimator = :guarded_ai_update_reml_optimizer_experiment,
        design = :tiny_deterministic_recovery_grid,
        ademp = (
            aim = "diagnose point-estimate recovery for the exact-Gaussian location-only phylogenetic mean cell",
            data_generating_mechanism = "y = X beta + u_species + epsilon, u ~ N(0, sigma_phy^2 Sigma_phy), epsilon ~ N(0, sigma^2 I)",
            estimands = ("sigma", "sigma_phy", "log_sigma", "log_sigma_phy"),
            methods = ("guarded_sparse_average_information_update",),
            performance = ("bias", "rmse", "mcse_bias", "convergence_rate", "boundary_counts"),
        ),
        conditions = (
            reps = reps,
            G = G,
            n_per_species = n_per_species,
            n = n,
            sigma = σ,
            sigma_phy = σ_phy,
            beta = Tuple(β),
            branch_length = Float64(branch_length),
            seed = seed,
        ),
        n_reps = reps,
        n_accepted = n_accepted,
        convergence_rate = reps == 0 ? NaN : n_accepted / reps,
        mean_sigma_hat = metric([r.sigma_hat for r in records], mean),
        mean_sigma_phy_hat = metric([r.sigma_phy_hat for r in records], mean),
        bias_sigma = metric(sigma_errors, mean),
        bias_sigma_phy = metric(sigma_phy_errors, mean),
        bias_log_sigma = metric(log_sigma_errors, mean),
        bias_log_sigma_phy = metric(log_sigma_phy_errors, mean),
        rmse_sigma = rmse(sigma_errors),
        rmse_sigma_phy = rmse(sigma_phy_errors),
        mcse_bias_sigma = mcse(sigma_errors),
        mcse_bias_sigma_phy = mcse(sigma_phy_errors),
        boundary_counts = boundary_counts,
        records = records,
        claim_status = :simulation_diagnostic,
        coverage_status = :not_evaluated,
        ai_reml_ready = false,
        reason_not_ai_reml = "tiny recovery grid only; no coverage, bridge, large-tree, or q4 validation gate is implemented",
    )
end

function _loconly_reml_recovery_condition_grid_diagnostic(; reps::Integer = 2,
                                                          iterations::Integer = 30,
                                                          cells = nothing)
    default_cells = (
        (name = :baseline_interior, G = 10, n_per_species = 3,
         sigma = 0.45, sigma_phy = 0.7, branch_length = 0.25, seed = 20260624),
        (name = :higher_phylo_interior, G = 10, n_per_species = 3,
         sigma = 0.45, sigma_phy = 1.0, branch_length = 0.25, seed = 20260724),
    )
    cell_list = cells === nothing ? default_cells : cells
    rows = Any[]
    for (i, cell) in enumerate(cell_list)
        name = hasproperty(cell, :name) ? cell.name : Symbol("cell_", i)
        diag = _loconly_reml_recovery_grid_diagnostic(
            ; reps = hasproperty(cell, :reps) ? cell.reps : reps,
            G = hasproperty(cell, :G) ? cell.G : 10,
            n_per_species = hasproperty(cell, :n_per_species) ? cell.n_per_species : 3,
            sigma = hasproperty(cell, :sigma) ? cell.sigma : 0.45,
            sigma_phy = hasproperty(cell, :sigma_phy) ? cell.sigma_phy : 0.7,
            beta = hasproperty(cell, :beta) ? cell.beta : (0.25, -0.4),
            branch_length = hasproperty(cell, :branch_length) ? cell.branch_length : 0.25,
            seed = hasproperty(cell, :seed) ? cell.seed : 20260624 + i,
            iterations = hasproperty(cell, :iterations) ? cell.iterations : iterations,
        )
        push!(rows, (
            cell = name,
            n_reps = diag.n_reps,
            n_accepted = diag.n_accepted,
            convergence_rate = diag.convergence_rate,
            bias_sigma = diag.bias_sigma,
            bias_sigma_phy = diag.bias_sigma_phy,
            rmse_sigma = diag.rmse_sigma,
            rmse_sigma_phy = diag.rmse_sigma_phy,
            mcse_bias_sigma = diag.mcse_bias_sigma,
            mcse_bias_sigma_phy = diag.mcse_bias_sigma_phy,
            boundary_counts = diag.boundary_counts,
            diagnostic = diag,
        ))
    end
    return (
        target = :gaussian_loconly_phylo_reml,
        estimator = :guarded_ai_update_reml_optimizer_experiment,
        design = :tiny_condition_recovery_grid,
        n_cells = length(rows),
        rows = Tuple(rows),
        min_convergence_rate = isempty(rows) ? NaN : minimum(r -> r.convergence_rate, rows),
        all_cells_accepted = all(r -> r.n_accepted == r.n_reps, rows),
        claim_status = :simulation_diagnostic,
        coverage_status = :not_evaluated,
        ai_reml_ready = false,
        reason_not_ai_reml = "condition grid is diagnostic only; no coverage, bridge, large-tree, or q4 validation gate is implemented",
    )
end

function _loconly_reml_weak_signal_recovery_probe(; reps::Integer = 2,
                                                  G::Integer = 8,
                                                  n_per_species::Integer = 2,
                                                  sigma::Real = 0.45,
                                                  sigma_phy::Real = 0.2,
                                                  branch_length::Real = 0.25,
                                                  seed::Integer = 20260630,
                                                  iterations::Integer = 25)
    diag = _loconly_reml_recovery_grid_diagnostic(
        ; reps = reps,
        G = G,
        n_per_species = n_per_species,
        sigma = sigma,
        sigma_phy = sigma_phy,
        branch_length = branch_length,
        seed = seed,
        iterations = iterations,
    )
    boundary_reps = diag.boundary_counts.near_zero_variance +
        diag.boundary_counts.nonfinite_objective +
        diag.boundary_counts.singular_fixed_effect_information
    return (
        target = :gaussian_loconly_phylo_reml,
        estimator = :guarded_ai_update_reml_optimizer_experiment,
        design = :weak_signal_boundary_probe,
        diagnostic = diag,
        boundary_reps = boundary_reps,
        boundary_rate = diag.n_reps == 0 ? NaN : boundary_reps / diag.n_reps,
        convergence_rate = diag.convergence_rate,
        expected_behavior = :boundary_states_allowed,
        claim_status = :simulation_diagnostic,
        coverage_status = :not_evaluated,
        ai_reml_ready = false,
        reason_not_ai_reml = "weak-signal boundary probe only; low convergence or boundary states are diagnostic outcomes",
    )
end

function _loconly_reml_weak_signal_condition_grid_diagnostic(; reps::Integer = 1,
                                                             iterations::Integer = 20,
                                                             cells = nothing)
    default_cells = (
        (name = :low_phylo_signal, sigma_phy = 0.2, seed = 20260630),
        (name = :near_zero_phylo_signal, sigma_phy = 0.05, seed = 20260631),
    )
    selected = cells === nothing ? default_cells : cells
    rows = Any[]
    for cell in selected
        diag = _loconly_reml_weak_signal_recovery_probe(
            ; reps = reps,
            sigma_phy = cell.sigma_phy,
            seed = cell.seed,
            iterations = iterations,
        )
        push!(rows, merge((cell = cell.name,), diag))
    end
    return (
        target = :gaussian_loconly_phylo_reml,
        estimator = :guarded_ai_update_reml_optimizer_experiment,
        design = :weak_signal_condition_grid,
        n_cells = length(rows),
        rows = Tuple(rows),
        expected_behavior = :boundary_states_allowed,
        claim_status = :simulation_diagnostic,
        coverage_status = :not_evaluated,
        ai_reml_ready = false,
        reason_not_ai_reml = "weak-signal condition grid is diagnostic only; boundary states are expected outcomes",
    )
end

function _loconly_reml_broader_recovery_grid_diagnostic(; reps::Integer = 1,
                                                        iterations::Integer = 25,
                                                        cells = nothing)
    default_cells = (
        (name = :baseline_interior, G = 10, n_per_species = 3,
         sigma = 0.45, sigma_phy = 0.7, branch_length = 0.25, seed = 20260624),
        (name = :higher_phylo_interior, G = 10, n_per_species = 3,
         sigma = 0.45, sigma_phy = 1.0, branch_length = 0.25, seed = 20260724),
        (name = :medium_interior_stress, G = 20, n_per_species = 3,
         sigma = 0.45, sigma_phy = 0.7, branch_length = 0.25, seed = 20260825),
    )
    selected = cells === nothing ? default_cells : cells
    rows = Any[]
    for cell in selected
        diag = _loconly_reml_recovery_grid_diagnostic(
            ; reps = reps,
            G = cell.G,
            n_per_species = cell.n_per_species,
            sigma = cell.sigma,
            sigma_phy = cell.sigma_phy,
            branch_length = cell.branch_length,
            seed = cell.seed,
            iterations = iterations,
        )
        push!(rows, merge((cell = cell.name,), diag))
    end
    return (
        target = :gaussian_loconly_phylo_reml,
        estimator = :guarded_ai_update_reml_optimizer_experiment,
        design = :broader_recovery_grid,
        n_cells = length(rows),
        rows = Tuple(rows),
        min_convergence_rate = isempty(rows) ? NaN : minimum(r -> r.convergence_rate, rows),
        all_cells_accepted = all(r -> r.n_accepted == r.n_reps, rows),
        expected_behavior = :stable_or_stress_recovery,
        claim_status = :simulation_diagnostic,
        coverage_status = :not_evaluated,
        ai_reml_ready = false,
        reason_not_ai_reml = "broader recovery grid is diagnostic only; no coverage, bridge, large-tree, or q4 validation gate is implemented",
    )
end

const _LOCONLY_REML_SIMULATION_STATUS_FIELDS = (
    :row_id, :target, :estimator, :design, :claim_status, :coverage_status,
    :expected_behavior, :n_reps, :n_accepted, :convergence_rate,
    :boundary_rate, :failure_reason_counts, :bias_sigma, :bias_sigma_phy,
    :rmse_sigma, :rmse_sigma_phy, :mcse_bias_sigma, :mcse_bias_sigma_phy,
    :mcse_status, :runtime_seconds, :runtime_budget_seconds, :seed,
    :seed_registry, :next_gate, :evidence,
)

function _loconly_reml_simulation_status_schema()
    return _LOCONLY_REML_SIMULATION_STATUS_FIELDS
end

function _loconly_reml_failure_reason_counts(counts)
    return (
        near_zero_variance = counts.near_zero_variance,
        nonfinite_objective = counts.nonfinite_objective,
        singular_fixed_effect_information = counts.singular_fixed_effect_information,
    )
end

function _loconly_reml_expected_behavior(row_id::Symbol)
    row_id === :weak_signal_boundary_probe && return :boundary_states_allowed
    row_id === :condition_grid && return :row_separated_stable_recovery
    row_id in (:larger_interior_stress, :medium_interior_stress,
        :large_interior_stress) && return :stress_smoke
    row_id === :large_interior_stress_skipped && return :skipped_runtime_guard
    return :stable_interior_recovery
end

function _loconly_reml_runtime_budget_seconds(row_id::Symbol)
    row_id === :large_interior_stress && return 30.0
    row_id === :medium_interior_stress && return 15.0
    row_id === :larger_interior_stress && return 10.0
    return 5.0
end

_loconly_reml_large_stress_min_budget_seconds() = 20.0

function _loconly_reml_skipped_runtime_stress_row(row_id::Symbol,
                                                 runtime_budget_seconds::Real,
                                                 seed::Integer,
                                                 evidence::AbstractString)
    return (
        row_id = row_id,
        target = :gaussian_loconly_phylo_reml,
        estimator = :guarded_ai_update_reml_optimizer_experiment,
        design = :large_interior_stress_grid,
        claim_status = :simulation_diagnostic,
        coverage_status = :not_evaluated,
        n_reps = 0,
        n_accepted = 0,
        convergence_rate = NaN,
        boundary_rate = NaN,
        failure_reason_counts = (
            near_zero_variance = 0,
            nonfinite_objective = 0,
            singular_fixed_effect_information = 0,
        ),
        bias_sigma = NaN,
        bias_sigma_phy = NaN,
        rmse_sigma = NaN,
        rmse_sigma_phy = NaN,
        mcse_bias_sigma = NaN,
        mcse_bias_sigma_phy = NaN,
        mcse_status = :diagnostic_only,
        runtime_seconds = 0.0,
        runtime_budget_seconds = Float64(runtime_budget_seconds),
        seed = seed,
        seed_registry = (primary = seed, deterministic = true),
        evidence = evidence,
        next_gate = :runtime_budget_review,
        expected_behavior = :skipped_runtime_guard,
    )
end

function _loconly_reml_simulation_status(; include_medium_stress::Bool = false,
                                         medium_stress_reps::Integer = 1,
                                         include_large_stress::Bool = false,
                                         large_stress_reps::Integer = 1,
                                         large_stress_budget_seconds::Real = 0.0)
    evidence = "test/test_location_only_reml_mme.jl"
    function boundary_rate(counts, n_reps)
        boundary_reps = counts.near_zero_variance + counts.nonfinite_objective +
            counts.singular_fixed_effect_information
        return n_reps == 0 ? NaN : boundary_reps / n_reps
    end
    function recovery_row(row_id::Symbol, design::Symbol, diag, runtime_seconds::Real,
                          next_gate::Symbol;
                          runtime_budget_seconds::Real =
                              _loconly_reml_runtime_budget_seconds(row_id))
        return (
            row_id = row_id,
            target = diag.target,
            estimator = diag.estimator,
            design = design,
            claim_status = diag.claim_status,
            coverage_status = diag.coverage_status,
            n_reps = diag.n_reps,
            n_accepted = diag.n_accepted,
            convergence_rate = diag.convergence_rate,
            boundary_rate = boundary_rate(diag.boundary_counts, diag.n_reps),
            failure_reason_counts = _loconly_reml_failure_reason_counts(diag.boundary_counts),
            bias_sigma = diag.bias_sigma,
            bias_sigma_phy = diag.bias_sigma_phy,
            rmse_sigma = diag.rmse_sigma,
            rmse_sigma_phy = diag.rmse_sigma_phy,
            mcse_bias_sigma = diag.mcse_bias_sigma,
            mcse_bias_sigma_phy = diag.mcse_bias_sigma_phy,
            mcse_status = :diagnostic_only,
            runtime_seconds = Float64(runtime_seconds),
            runtime_budget_seconds = Float64(runtime_budget_seconds),
            seed = diag.conditions.seed,
            seed_registry = (primary = diag.conditions.seed, deterministic = true),
            evidence = evidence,
            next_gate = next_gate,
            expected_behavior = _loconly_reml_expected_behavior(row_id),
        )
    end

    baseline = nothing
    baseline_time = @elapsed baseline = _loconly_reml_recovery_grid_diagnostic(
        ; reps = 3, G = 10, n_per_species = 3, sigma = 0.45, sigma_phy = 0.7,
        seed = 20260624, iterations = 30,
    )

    condition = nothing
    condition_time = @elapsed condition = _loconly_reml_recovery_condition_grid_diagnostic(
        ; reps = 2, iterations = 30,
    )
    condition_reps = sum(r -> r.n_reps, condition.rows)
    condition_accepted = sum(r -> r.n_accepted, condition.rows)
    condition_boundary_reps = sum(
        r -> r.boundary_counts.near_zero_variance +
             r.boundary_counts.nonfinite_objective +
             r.boundary_counts.singular_fixed_effect_information,
        condition.rows,
    )
    condition_row = (
        row_id = :condition_grid,
        target = condition.target,
        estimator = condition.estimator,
        design = condition.design,
        claim_status = condition.claim_status,
        coverage_status = condition.coverage_status,
        n_reps = condition_reps,
        n_accepted = condition_accepted,
        convergence_rate = condition_reps == 0 ? NaN : condition_accepted / condition_reps,
        boundary_rate = condition_reps == 0 ? NaN : condition_boundary_reps / condition_reps,
        failure_reason_counts = (
            near_zero_variance = sum(r -> r.boundary_counts.near_zero_variance, condition.rows),
            nonfinite_objective = sum(r -> r.boundary_counts.nonfinite_objective, condition.rows),
            singular_fixed_effect_information = sum(r -> r.boundary_counts.singular_fixed_effect_information, condition.rows),
        ),
        bias_sigma = mean([r.bias_sigma for r in condition.rows]),
        bias_sigma_phy = mean([r.bias_sigma_phy for r in condition.rows]),
        rmse_sigma = mean([r.rmse_sigma for r in condition.rows]),
        rmse_sigma_phy = mean([r.rmse_sigma_phy for r in condition.rows]),
        mcse_bias_sigma = mean([r.mcse_bias_sigma for r in condition.rows]),
        mcse_bias_sigma_phy = mean([r.mcse_bias_sigma_phy for r in condition.rows]),
        mcse_status = :diagnostic_only,
        runtime_seconds = Float64(condition_time),
        runtime_budget_seconds = _loconly_reml_runtime_budget_seconds(:condition_grid),
        seed = Tuple(r.diagnostic.conditions.seed for r in condition.rows),
        seed_registry = (primary = Tuple(r.diagnostic.conditions.seed for r in condition.rows),
                         deterministic = true),
        evidence = evidence,
        next_gate = :broader_condition_grid,
        expected_behavior = _loconly_reml_expected_behavior(:condition_grid),
    )

    weak = nothing
    weak_time = @elapsed weak = _loconly_reml_weak_signal_recovery_probe(
        ; reps = 2, seed = 20260630, iterations = 25,
    )
    weak_diag = weak.diagnostic
    weak_row = (
        row_id = :weak_signal_boundary_probe,
        target = weak.target,
        estimator = weak.estimator,
        design = weak.design,
        claim_status = weak.claim_status,
        coverage_status = weak.coverage_status,
        n_reps = weak_diag.n_reps,
        n_accepted = weak_diag.n_accepted,
        convergence_rate = weak.convergence_rate,
        boundary_rate = weak.boundary_rate,
        failure_reason_counts = _loconly_reml_failure_reason_counts(weak_diag.boundary_counts),
        bias_sigma = weak_diag.bias_sigma,
        bias_sigma_phy = weak_diag.bias_sigma_phy,
        rmse_sigma = weak_diag.rmse_sigma,
        rmse_sigma_phy = weak_diag.rmse_sigma_phy,
        mcse_bias_sigma = weak_diag.mcse_bias_sigma,
        mcse_bias_sigma_phy = weak_diag.mcse_bias_sigma_phy,
        mcse_status = :diagnostic_only,
        runtime_seconds = Float64(weak_time),
        runtime_budget_seconds = _loconly_reml_runtime_budget_seconds(:weak_signal_boundary_probe),
        seed = weak_diag.conditions.seed,
        seed_registry = (primary = weak_diag.conditions.seed, deterministic = true),
        evidence = evidence,
        next_gate = :boundary_diagnostics,
        expected_behavior = _loconly_reml_expected_behavior(:weak_signal_boundary_probe),
    )

    stress = nothing
    stress_time = @elapsed stress = _loconly_reml_recovery_grid_diagnostic(
        ; reps = 2, G = 16, n_per_species = 3, sigma = 0.45, sigma_phy = 0.7,
        seed = 20260824, iterations = 30,
    )
    rows = Any[
        recovery_row(:stable_recovery, baseline.design, baseline, baseline_time,
                     :broader_recovery_grid),
        condition_row,
        weak_row,
        recovery_row(:larger_interior_stress, :larger_interior_stress_grid,
                     stress, stress_time, :optional_runtime_stress),
    ]
    if include_medium_stress
        medium = nothing
        medium_time = @elapsed medium = _loconly_reml_recovery_grid_diagnostic(
            ; reps = medium_stress_reps, G = 20, n_per_species = 3,
            sigma = 0.45, sigma_phy = 0.7, seed = 20260825,
            iterations = 25,
        )
        push!(rows, recovery_row(:medium_interior_stress, :medium_interior_stress_grid,
                                 medium, medium_time, :optional_runtime_stress))
    end
    if include_large_stress
        if large_stress_budget_seconds < _loconly_reml_large_stress_min_budget_seconds()
            push!(rows, _loconly_reml_skipped_runtime_stress_row(
                :large_interior_stress_skipped,
                large_stress_budget_seconds,
                20260826,
                evidence,
            ))
        else
            large = nothing
            large_time = @elapsed large = _loconly_reml_recovery_grid_diagnostic(
                ; reps = large_stress_reps, G = 32, n_per_species = 3,
                sigma = 0.45, sigma_phy = 0.7, seed = 20260826,
                iterations = 25,
            )
            push!(rows, recovery_row(:large_interior_stress,
                                     :large_interior_stress_grid,
                                     large, large_time, :optional_runtime_stress;
                                     runtime_budget_seconds =
                                         large_stress_budget_seconds))
        end
    end
    return (
        target = :gaussian_loconly_phylo_reml,
        estimator = :guarded_ai_update_reml_optimizer_experiment,
        rows = Tuple(rows),
        n_rows = length(rows),
        claim_status = :simulation_diagnostic,
        coverage_status = :not_evaluated,
        ai_reml_ready = false,
        evidence = evidence,
        reason_not_ai_reml = "simulation-status rows are diagnostic only; no coverage, bridge, large-tree, or q4 validation gate is implemented",
    )
end

function _loconly_reml_validate_simulation_status(status)
    required = _loconly_reml_simulation_status_schema()
    errors = String[]
    expected_order = (:stable_recovery, :condition_grid,
        :weak_signal_boundary_probe, :larger_interior_stress)
    row_ids = Tuple(r.row_id for r in status.rows)
    if row_ids[1:min(length(row_ids), length(expected_order))] !=
            expected_order[1:min(length(row_ids), length(expected_order))]
        push!(errors, "default row order changed")
    end
    status.coverage_status === :not_evaluated ||
        push!(errors, "status has evaluated coverage")
    status.ai_reml_ready === false ||
        push!(errors, "status is marked AI-REML ready")
    for row in status.rows
        names = propertynames(row)
        for field in required
            field in names || push!(errors, "row $(row.row_id) missing field $(field)")
        end
        row.target === :gaussian_loconly_phylo_reml ||
            push!(errors, "row $(row.row_id) has wrong target")
        row.estimator === :guarded_ai_update_reml_optimizer_experiment ||
            push!(errors, "row $(row.row_id) has wrong estimator")
        row.claim_status === :simulation_diagnostic ||
            push!(errors, "row $(row.row_id) has wrong claim_status")
        row.coverage_status === :not_evaluated ||
            push!(errors, "row $(row.row_id) has evaluated coverage")
        row.expected_behavior in (:stable_interior_recovery,
            :row_separated_stable_recovery, :boundary_states_allowed,
            :stress_smoke, :skipped_runtime_guard) ||
            push!(errors, "row $(row.row_id) has unexpected behavior label")
        row.n_reps >= 0 || push!(errors, "row $(row.row_id) has negative n_reps")
        0 <= row.n_accepted <= row.n_reps ||
            push!(errors, "row $(row.row_id) has invalid accepted count")
        (isnan(row.boundary_rate) || 0 <= row.boundary_rate <= 1) ||
            push!(errors, "row $(row.row_id) has invalid boundary_rate")
        row.mcse_status === :diagnostic_only ||
            push!(errors, "row $(row.row_id) has non-diagnostic MCSE status")
        row.runtime_seconds >= 0 ||
            push!(errors, "row $(row.row_id) has negative runtime")
        if row.expected_behavior === :skipped_runtime_guard
            row.runtime_budget_seconds >= 0 ||
                push!(errors, "row $(row.row_id) has negative runtime budget")
        else
            row.runtime_budget_seconds > 0 ||
                push!(errors, "row $(row.row_id) has no runtime budget")
        end
        isempty(string(row.seed)) &&
            push!(errors, "row $(row.row_id) has empty seed")
        isempty(row.evidence) &&
            push!(errors, "row $(row.row_id) has empty evidence")
    end
    return (
        ok = isempty(errors),
        errors = Tuple(errors),
        required_fields = required,
        row_order = row_ids,
        claim_status = status.claim_status,
        coverage_status = status.coverage_status,
        ai_reml_ready = status.ai_reml_ready,
    )
end

function _loconly_reml_tsv_value(x)
    return replace(string(x), "\t" => " ", "\n" => " ")
end

function _loconly_reml_simulation_status_provenance(status =
        _loconly_reml_simulation_status())
    function helper(row_id)
        row_id === :stable_recovery &&
            return "_loconly_reml_recovery_grid_diagnostic"
        row_id === :condition_grid &&
            return "_loconly_reml_recovery_condition_grid_diagnostic"
        row_id === :weak_signal_boundary_probe &&
            return "_loconly_reml_weak_signal_recovery_probe"
        row_id === :larger_interior_stress &&
            return "_loconly_reml_recovery_grid_diagnostic"
        row_id === :medium_interior_stress &&
            return "_loconly_reml_recovery_grid_diagnostic"
        row_id in (:large_interior_stress, :large_interior_stress_skipped) &&
            return "_loconly_reml_recovery_grid_diagnostic"
        return "_loconly_reml_simulation_status"
    end
    rows = Tuple((
        row_id = row.row_id,
        helper = helper(row.row_id),
        test = status.evidence,
        artifact = "docs/dev-log/validation-status/2026-06-21-loconly-reml-simulation-status.tsv",
        claim_boundary = "exact-Gaussian diagnostic only; coverage not evaluated; ai_reml_ready=false",
    ) for row in status.rows)
    return (
        target = status.target,
        estimator = status.estimator,
        rows = rows,
        n_rows = length(rows),
        coverage_status = status.coverage_status,
        ai_reml_ready = status.ai_reml_ready,
    )
end

function _loconly_reml_write_simulation_status_tsv(path;
                                                   status = nothing,
                                                   include_medium_stress::Bool = false,
                                                   medium_stress_reps::Integer = 1,
                                                   include_large_stress::Bool = false,
                                                   large_stress_reps::Integer = 1,
                                                   large_stress_budget_seconds::Real = 0.0)
    if status === nothing
        status = _loconly_reml_simulation_status(
            ; include_medium_stress = include_medium_stress,
            medium_stress_reps = medium_stress_reps,
            include_large_stress = include_large_stress,
            large_stress_reps = large_stress_reps,
            large_stress_budget_seconds = large_stress_budget_seconds,
        )
    end
    validation = _loconly_reml_validate_simulation_status(status)
    if !validation.ok
        error("simulation-status validation failed: $(join(validation.errors, "; "))")
    end
    schema = _loconly_reml_simulation_status_schema()
    dir = dirname(String(path))
    !isempty(dir) && mkpath(dir)
    open(path, "w") do io
        println(io, join(string.(schema), "\t"))
        for row in status.rows
            println(io, join((_loconly_reml_tsv_value(getproperty(row, field))
                              for field in schema), "\t"))
        end
    end
    return (
        path = String(path),
        n_rows = length(status.rows),
        schema = schema,
        validation = validation,
        claim_status = status.claim_status,
        coverage_status = status.coverage_status,
        ai_reml_ready = status.ai_reml_ready,
    )
end

function _loconly_reml_validation_status()
    return (
        target = :gaussian_loconly_phylo_reml,
        estimator = :supplied_variance_reml,
        source_status = :partial,
        tests_status = :partial,
        comparator_status = :dense_same_estimand_oracle,
        external_comparator_status = :planned,
        optimizer_status = :experiment_only,
        r_bridge_status = :planned,
        claim_status = :internal_diagnostic,
        q4_status = :excluded,
        evidence = "test/test_location_only_reml_mme.jl",
    )
end

const _LOCONLY_REML_EXTERNAL_COMPARATOR_FIELDS = (
    :comparator_id, :target, :comparator, :same_estimand_status,
    :dependency_status, :artifact_status, :decision, :reason, :next_gate,
)

const _LOCONLY_REML_EXTERNAL_COMPARATOR_FIXTURE_VERSION =
    "loconly-gaussian-phylo-reml-v1"

const _LOCONLY_REML_EXTERNAL_COMPARATOR_FIXTURE_FIELDS = (
    :fixture_id, :version, :target, :estimator, :parameterization,
    :tree_shape, :branch_length, :n_species, :n_obs, :n_per_species,
    :species, :x, :X, :y, :Sigma_phy, :known_sigma, :known_sigma_phy,
    :reference, :seed_registry, :claim_status, :coverage_status,
    :ai_reml_ready,
)

const _LOCONLY_REML_EXTERNAL_COMPARATOR_VERSION_PROBE_FIELDS = (
    :comparator_id, :target, :candidate_package, :candidate_version,
    :fixture_id, :fixture_version, :same_estimand_status, :dependency_status,
    :artifact_status, :decision, :blocked_by, :required_evidence, :next_gate,
)

const _LOCONLY_REML_EXTERNAL_COMPARATOR_PROBE_RESULT_FIELDS = (
    :probe_id, :target, :fixture_id, :fixture_version, :comparator_id,
    :candidate_package, :candidate_version, :package_available,
    :rscript_status, :fit_status, :same_estimand_status, :dependency_status,
    :artifact_status, :claim_status, :coverage_status, :ai_reml_ready,
    :next_gate, :evidence,
)

const _LOCONLY_REML_EXTERNAL_COMPARATOR_FIT_FEASIBILITY_FIELDS = (
    :check_id, :target, :fixture_id, :fixture_version, :comparator_id,
    :candidate_package, :candidate_version, :n_obs, :n_species,
    :n_per_species, :within_species_x_varies, :fit_status,
    :same_estimand_status, :dependency_status, :blocked_by, :reason,
    :claim_status, :coverage_status, :ai_reml_ready, :next_gate,
)

const _LOCONLY_REML_DERIVATIVE_FD_STATUS_FIELDS = (
    :check_id, :target, :fixture_id, :fixture_version, :parameterization,
    :matrix_mode, :h, :score_max_absdiff_fd, :dense_sparse_max_absdiff,
    :finite, :claim_status, :coverage_status, :ai_reml_ready, :next_gate,
)

const _LOCONLY_REML_LINE_SEARCH_STATUS_FIELDS = (
    :check_id, :target, :fixture_id, :fixture_version, :estimator,
    :optimizer, :n_starts, :n_records, :n_accepted_records,
    :accepted, :best_score_norm, :boundary_status, :trace_status,
    :claim_status, :coverage_status, :ai_reml_ready, :reason_not_ai_reml,
    :next_gate,
)

const _LOCONLY_REML_BOUNDARY_GRID_STATUS_FIELDS = (
    :row_id, :target, :design, :condition, :n_reps, :n_accepted,
    :convergence_rate, :boundary_rate, :near_zero_variance,
    :nonfinite_objective, :singular_fixed_effect_information,
    :expected_behavior, :claim_status, :coverage_status, :ai_reml_ready,
    :next_gate,
)

const _LOCONLY_REML_PROFILE_STATUS_FIELDS = (
    :row_id, :target, :fixture_id, :fixture_version, :parameterization,
    :axis, :step, :finite, :center_is_axis_min, :center_nll, :left_nll,
    :right_nll, :claim_status, :coverage_status, :ai_reml_ready, :next_gate,
)

const _LOCONLY_REML_VARIANCE_COMPONENT_STATUS_FIELDS = (
    :row_id, :target, :fixture_id, :fixture_version, :estimator, :optimizer,
    :component, :logsd_parameter, :logsd_estimate, :sd_estimate,
    :variance_estimate, :point_status, :interval_status, :boundary_status,
    :claim_status, :coverage_status, :ai_reml_ready, :next_gate,
)

function _loconly_reml_external_comparator_schema()
    return _LOCONLY_REML_EXTERNAL_COMPARATOR_FIELDS
end

function _loconly_reml_external_comparator_fixture_schema()
    return _LOCONLY_REML_EXTERNAL_COMPARATOR_FIXTURE_FIELDS
end

function _loconly_reml_external_comparator_version_probe_schema()
    return _LOCONLY_REML_EXTERNAL_COMPARATOR_VERSION_PROBE_FIELDS
end

function _loconly_reml_external_comparator_probe_result_schema()
    return _LOCONLY_REML_EXTERNAL_COMPARATOR_PROBE_RESULT_FIELDS
end

function _loconly_reml_external_comparator_fit_feasibility_schema()
    return _LOCONLY_REML_EXTERNAL_COMPARATOR_FIT_FEASIBILITY_FIELDS
end

function _loconly_reml_derivative_fd_status_schema()
    return _LOCONLY_REML_DERIVATIVE_FD_STATUS_FIELDS
end

function _loconly_reml_line_search_status_schema()
    return _LOCONLY_REML_LINE_SEARCH_STATUS_FIELDS
end

function _loconly_reml_boundary_grid_status_schema()
    return _LOCONLY_REML_BOUNDARY_GRID_STATUS_FIELDS
end

function _loconly_reml_profile_status_schema()
    return _LOCONLY_REML_PROFILE_STATUS_FIELDS
end

function _loconly_reml_variance_component_status_schema()
    return _LOCONLY_REML_VARIANCE_COMPONENT_STATUS_FIELDS
end

function _loconly_reml_external_comparator_fixture()
    G = 6
    n_per_species = 2
    branch_length = 0.25
    phy = random_balanced_tree(G; branch_length = branch_length)
    species = repeat(1:G, inner = n_per_species)
    n = length(species)
    x = collect(range(-0.75, 0.75; length = n))
    X = hcat(ones(n), x)
    beta = [0.2, -0.35]
    known_sigma = 0.4
    known_sigma_phy = 0.55
    species_effect = known_sigma_phy .* collect(range(-0.4, 0.4; length = G))
    residual_pattern = known_sigma .* 0.15 .* sin.(collect(1:n))
    y = X * beta .+ species_effect[species] .+ residual_pattern
    prob = make_loc_problem(phy, y, X; species = species)
    lσ = log(known_sigma)
    lσ_phy = log(known_sigma_phy)
    dense = _loconly_dense_reml_components(prob, lσ, lσ_phy)
    boundary = _loconly_reml_boundary_status(prob, lσ, lσ_phy)
    return (
        fixture_id = :loconly_gaussian_phylo_reml_v1,
        version = _LOCONLY_REML_EXTERNAL_COMPARATOR_FIXTURE_VERSION,
        target = :gaussian_loconly_phylo_reml,
        estimator = :supplied_variance_reml,
        parameterization = :log_sd,
        tree_shape = :balanced,
        branch_length = branch_length,
        n_species = G,
        n_obs = n,
        n_per_species = n_per_species,
        species = Tuple(species),
        x = Tuple(x),
        X = Matrix(X),
        y = Tuple(y),
        Sigma_phy = Matrix(sigma_phy_dense(phy; σ²_phy = 1.0)),
        known_sigma = known_sigma,
        known_sigma_phy = known_sigma_phy,
        reference = (
            comparator = :internal_dense_gls_oracle,
            beta_hat = Tuple(dense.beta),
            reml_nll = dense.nll,
            ml_nll = dense.ml_nll,
            restricted_penalty = dense.penalty,
            fixed_effect_information = Matrix(dense.info),
            boundary_status = boundary.boundary_status,
            can_report_restricted_likelihood_directly = true,
            covariance_target = :sigma_phy_squared_times_Sigma_phy_plus_sigma_squared_I,
        ),
        seed_registry = (primary = 20260622, deterministic = true, rng_used = false),
        claim_status = :internal_diagnostic,
        coverage_status = :not_evaluated,
        ai_reml_ready = false,
    )
end

function _loconly_reml_validate_external_comparator_fixture(fixture)
    schema = _loconly_reml_external_comparator_fixture_schema()
    errors = String[]
    names = propertynames(fixture)
    for field in schema
        field in names || push!(errors, "fixture missing field $(field)")
    end
    fixture.target === :gaussian_loconly_phylo_reml ||
        push!(errors, "fixture has wrong target")
    fixture.version == _LOCONLY_REML_EXTERNAL_COMPARATOR_FIXTURE_VERSION ||
        push!(errors, "fixture has wrong version")
    fixture.estimator === :supplied_variance_reml ||
        push!(errors, "fixture has wrong estimator")
    fixture.parameterization === :log_sd ||
        push!(errors, "fixture has wrong parameterization")
    fixture.n_obs == length(fixture.y) == length(fixture.species) == size(fixture.X, 1) ||
        push!(errors, "fixture has inconsistent observation dimensions")
    fixture.n_species == size(fixture.Sigma_phy, 1) == size(fixture.Sigma_phy, 2) ||
        push!(errors, "fixture has inconsistent covariance dimensions")
    size(fixture.X, 2) == length(fixture.reference.beta_hat) ||
        push!(errors, "fixture beta reference does not match X")
    all(1 <= sp <= fixture.n_species for sp in fixture.species) ||
        push!(errors, "fixture species index outside covariance target")
    isfinite(fixture.reference.reml_nll) ||
        push!(errors, "fixture reference REML nll is not finite")
    isfinite(fixture.reference.ml_nll) ||
        push!(errors, "fixture reference ML nll is not finite")
    fixture.reference.boundary_status === :interior ||
        push!(errors, "fixture reference is not interior")
    fixture.claim_status === :internal_diagnostic ||
        push!(errors, "fixture has wrong claim_status")
    fixture.coverage_status === :not_evaluated ||
        push!(errors, "fixture has wrong coverage_status")
    !fixture.ai_reml_ready ||
        push!(errors, "fixture must not mark ai_reml_ready")
    return (
        ok = isempty(errors),
        errors = Tuple(errors),
        required_fields = schema,
        version = fixture.version,
        target = fixture.target,
        coverage_status = fixture.coverage_status,
        ai_reml_ready = fixture.ai_reml_ready,
    )
end

function _loconly_reml_external_comparator_version_probe_rows(fixture)
    return (
        (
            comparator_id = :phylolm_or_equivalent_reml,
            target = :gaussian_loconly_phylo_reml,
            candidate_package = :unselected_phylolm_style_reml,
            candidate_version = :unprobed,
            fixture_id = fixture.fixture_id,
            fixture_version = fixture.version,
            same_estimand_status = :requires_fixture_reproduction,
            dependency_status = :not_added,
            artifact_status = :probe_plan_defined,
            decision = :run_optional_developer_probe_before_dependency,
            blocked_by = :external_package_version_not_probed,
            required_evidence = (
                :package_name_and_version,
                :same_restricted_likelihood_target,
                :same_covariance_target,
                :variance_component_point_estimates,
                :restricted_loglikelihood_or_reconstructable_value,
                :boundary_status,
            ),
            next_gate = :optional_developer_comparator_script,
        ),
    )
end

function _loconly_reml_validate_external_comparator_version_probe(rows, fixture)
    schema = _loconly_reml_external_comparator_version_probe_schema()
    errors = String[]
    for row in rows
        names = propertynames(row)
        for field in schema
            field in names ||
                push!(errors, "version probe $(row.comparator_id) missing field $(field)")
        end
        row.target === :gaussian_loconly_phylo_reml ||
            push!(errors, "version probe $(row.comparator_id) has wrong target")
        row.fixture_id === fixture.fixture_id ||
            push!(errors, "version probe $(row.comparator_id) has wrong fixture_id")
        row.fixture_version == fixture.version ||
            push!(errors, "version probe $(row.comparator_id) has wrong fixture_version")
        row.candidate_version === :unprobed ||
            push!(errors, "version probe $(row.comparator_id) must remain unprobed")
        row.dependency_status === :not_added ||
            push!(errors, "version probe $(row.comparator_id) must not add a dependency")
        row.artifact_status === :probe_plan_defined ||
            push!(errors, "version probe $(row.comparator_id) has invalid artifact_status")
        :same_restricted_likelihood_target in row.required_evidence ||
            push!(errors, "version probe $(row.comparator_id) lacks restricted likelihood evidence gate")
        :same_covariance_target in row.required_evidence ||
            push!(errors, "version probe $(row.comparator_id) lacks covariance target evidence gate")
    end
    return (
        ok = isempty(errors),
        errors = Tuple(errors),
        required_fields = schema,
        n_rows = length(rows),
        fixture_id = fixture.fixture_id,
        fixture_version = fixture.version,
        dependency_status = :not_added,
    )
end

function _loconly_reml_external_comparator_probe_status(;
        candidate_package::AbstractString = "phylolm",
        candidate_version::AbstractString = "unprobed",
        package_available::Bool = false,
        rscript_status::Symbol = :not_checked,
        evidence::AbstractString =
            "tools/loconly-reml-external-comparator-probe.jl",
)
    comparator_status = _loconly_reml_external_comparator_status()
    fixture = comparator_status.fixture
    version_probe = only(comparator_status.version_probe_rows)
    row = (
        probe_id = :phylolm_style_version_probe,
        target = :gaussian_loconly_phylo_reml,
        fixture_id = fixture.fixture_id,
        fixture_version = fixture.version,
        comparator_id = version_probe.comparator_id,
        candidate_package = String(candidate_package),
        candidate_version = String(candidate_version),
        package_available = package_available,
        rscript_status = rscript_status,
        fit_status = :not_run,
        same_estimand_status = :requires_fixture_reproduction,
        dependency_status = :not_added,
        artifact_status = :optional_developer_probe_written,
        claim_status = :internal_diagnostic,
        coverage_status = :not_evaluated,
        ai_reml_ready = false,
        next_gate = :external_fixture_fit_comparison,
        evidence = String(evidence),
    )
    return (
        target = :gaussian_loconly_phylo_reml,
        fixture_id = fixture.fixture_id,
        fixture_version = fixture.version,
        rows = (row,),
        n_rows = 1,
        schema = _loconly_reml_external_comparator_probe_result_schema(),
        claim_status = :internal_diagnostic,
        coverage_status = :not_evaluated,
        ai_reml_ready = false,
    )
end

function _loconly_reml_validate_external_comparator_probe_status(status)
    required = _loconly_reml_external_comparator_probe_result_schema()
    errors = String[]
    status.target === :gaussian_loconly_phylo_reml ||
        push!(errors, "probe status has wrong target")
    status.claim_status === :internal_diagnostic ||
        push!(errors, "probe status has wrong claim_status")
    status.coverage_status === :not_evaluated ||
        push!(errors, "probe status has evaluated coverage")
    !status.ai_reml_ready ||
        push!(errors, "probe status must not mark ai_reml_ready")
    status.n_rows == length(status.rows) ||
        push!(errors, "probe status n_rows does not match rows")
    for row in status.rows
        names = propertynames(row)
        for field in required
            field in names || push!(errors, "probe row $(row.probe_id) missing field $(field)")
        end
        row.target === :gaussian_loconly_phylo_reml ||
            push!(errors, "probe row $(row.probe_id) has wrong target")
        row.fixture_id === status.fixture_id ||
            push!(errors, "probe row $(row.probe_id) has wrong fixture_id")
        row.fixture_version == status.fixture_version ||
            push!(errors, "probe row $(row.probe_id) has wrong fixture_version")
        row.fit_status === :not_run ||
            push!(errors, "probe row $(row.probe_id) must not mark a fit as run")
        row.same_estimand_status === :requires_fixture_reproduction ||
            push!(errors, "probe row $(row.probe_id) has invalid same_estimand_status")
        row.dependency_status === :not_added ||
            push!(errors, "probe row $(row.probe_id) must not add a dependency")
        row.claim_status === :internal_diagnostic ||
            push!(errors, "probe row $(row.probe_id) has wrong claim_status")
        row.coverage_status === :not_evaluated ||
            push!(errors, "probe row $(row.probe_id) has evaluated coverage")
        !row.ai_reml_ready ||
            push!(errors, "probe row $(row.probe_id) must not mark ai_reml_ready")
        isempty(row.candidate_package) &&
            push!(errors, "probe row $(row.probe_id) has empty candidate_package")
        isempty(row.candidate_version) &&
            push!(errors, "probe row $(row.probe_id) has empty candidate_version")
        isempty(row.evidence) &&
            push!(errors, "probe row $(row.probe_id) has empty evidence")
    end
    return (
        ok = isempty(errors),
        errors = Tuple(errors),
        required_fields = required,
        n_rows = length(status.rows),
        claim_status = status.claim_status,
        coverage_status = status.coverage_status,
        ai_reml_ready = status.ai_reml_ready,
    )
end

function _loconly_reml_write_external_comparator_probe_tsv(path; kwargs...)
    status = _loconly_reml_external_comparator_probe_status(; kwargs...)
    validation = _loconly_reml_validate_external_comparator_probe_status(status)
    if !validation.ok
        error("external-comparator probe validation failed: $(join(validation.errors, "; "))")
    end
    schema = _loconly_reml_external_comparator_probe_result_schema()
    dir = dirname(String(path))
    !isempty(dir) && mkpath(dir)
    open(path, "w") do io
        println(io, join(string.(schema), "\t"))
        for row in status.rows
            println(io, join((_loconly_reml_tsv_value(getproperty(row, field))
                              for field in schema), "\t"))
        end
    end
    return (
        path = String(path),
        n_rows = length(status.rows),
        schema = schema,
        validation = validation,
        claim_status = status.claim_status,
        coverage_status = status.coverage_status,
        ai_reml_ready = status.ai_reml_ready,
    )
end

function _loconly_reml_fixture_within_species_x_varies(fixture)
    for sp in unique(fixture.species)
        x_sp = [fixture.x[i] for i in eachindex(fixture.x)
                if fixture.species[i] == sp]
        length(unique(x_sp)) > 1 && return true
    end
    return false
end

function _loconly_reml_external_comparator_fit_feasibility_status(;
        candidate_package::AbstractString = "phylolm",
        candidate_version::AbstractString = "unprobed",
)
    comparator_status = _loconly_reml_external_comparator_status()
    fixture = comparator_status.fixture
    repeated_observations = fixture.n_obs != fixture.n_species
    within_species_x_varies =
        _loconly_reml_fixture_within_species_x_varies(fixture)
    status = repeated_observations || within_species_x_varies ?
        :not_same_estimand_current_fixture : :same_estimand_candidate
    row = (
        check_id = :phylolm_fixture_fit_feasibility,
        target = :gaussian_loconly_phylo_reml,
        fixture_id = fixture.fixture_id,
        fixture_version = fixture.version,
        comparator_id = :phylolm_or_equivalent_reml,
        candidate_package = String(candidate_package),
        candidate_version = String(candidate_version),
        n_obs = fixture.n_obs,
        n_species = fixture.n_species,
        n_per_species = fixture.n_per_species,
        within_species_x_varies = within_species_x_varies,
        fit_status = :not_run,
        same_estimand_status = status,
        dependency_status = :not_added,
        blocked_by = status === :not_same_estimand_current_fixture ?
            :repeated_observation_fixture_not_tip_level : :none,
        reason = status === :not_same_estimand_current_fixture ?
            "The current fixture has replicated species rows and observation-level covariate variation; a tip-level phylolm fit would not reproduce the same REML target." :
            "The fixture is tip-level and can advance to an external fit comparison.",
        claim_status = :internal_diagnostic,
        coverage_status = :not_evaluated,
        ai_reml_ready = false,
        next_gate = status === :not_same_estimand_current_fixture ?
            :choose_replicate_capable_comparator_or_tip_level_fixture :
            :run_external_fixture_fit_comparison,
    )
    return (
        target = :gaussian_loconly_phylo_reml,
        rows = (row,),
        n_rows = 1,
        schema = _loconly_reml_external_comparator_fit_feasibility_schema(),
        claim_status = :internal_diagnostic,
        coverage_status = :not_evaluated,
        ai_reml_ready = false,
    )
end

function _loconly_reml_validate_external_comparator_fit_feasibility(status)
    required = _loconly_reml_external_comparator_fit_feasibility_schema()
    errors = String[]
    status.target === :gaussian_loconly_phylo_reml ||
        push!(errors, "fit feasibility status has wrong target")
    status.claim_status === :internal_diagnostic ||
        push!(errors, "fit feasibility status has wrong claim_status")
    status.coverage_status === :not_evaluated ||
        push!(errors, "fit feasibility status has evaluated coverage")
    !status.ai_reml_ready ||
        push!(errors, "fit feasibility status must not mark ai_reml_ready")
    status.n_rows == length(status.rows) ||
        push!(errors, "fit feasibility status n_rows does not match rows")
    for row in status.rows
        names = propertynames(row)
        for field in required
            field in names ||
                push!(errors, "fit feasibility row $(row.check_id) missing field $(field)")
        end
        row.target === :gaussian_loconly_phylo_reml ||
            push!(errors, "fit feasibility row $(row.check_id) has wrong target")
        row.fit_status === :not_run ||
            push!(errors, "fit feasibility row $(row.check_id) must not mark a fit as run")
        row.same_estimand_status in
            (:not_same_estimand_current_fixture, :same_estimand_candidate) ||
            push!(errors, "fit feasibility row $(row.check_id) has invalid same_estimand_status")
        row.dependency_status === :not_added ||
            push!(errors, "fit feasibility row $(row.check_id) must not add a dependency")
        row.coverage_status === :not_evaluated ||
            push!(errors, "fit feasibility row $(row.check_id) has evaluated coverage")
        !row.ai_reml_ready ||
            push!(errors, "fit feasibility row $(row.check_id) must not mark ai_reml_ready")
        if row.same_estimand_status === :not_same_estimand_current_fixture
            row.blocked_by === :repeated_observation_fixture_not_tip_level ||
                push!(errors, "fit feasibility row $(row.check_id) has wrong blocker")
        end
    end
    return (
        ok = isempty(errors),
        errors = Tuple(errors),
        required_fields = required,
        n_rows = length(status.rows),
        claim_status = status.claim_status,
        coverage_status = status.coverage_status,
        ai_reml_ready = status.ai_reml_ready,
    )
end

function _loconly_reml_problem_from_external_fixture(fixture)
    phy = random_balanced_tree(fixture.n_species;
                               branch_length = fixture.branch_length)
    return make_loc_problem(
        phy, collect(fixture.y), fixture.X; species = collect(fixture.species),
    )
end

function _loconly_reml_derivative_fd_status(; h::Real = 1e-5)
    fixture = _loconly_reml_external_comparator_fixture()
    prob = _loconly_reml_problem_from_external_fixture(fixture)
    lσ = log(fixture.known_sigma)
    lσ_phy = log(fixture.known_sigma_phy)
    dense = _loconly_reml_dense_score_diagnostic(prob, lσ, lσ_phy; h = h)
    sparse = _loconly_reml_sparse_score_diagnostic(prob, lσ, lσ_phy; h = h)
    rows = (
        (
            check_id = :dense_score_fd_fixture,
            target = :gaussian_loconly_phylo_reml,
            fixture_id = fixture.fixture_id,
            fixture_version = fixture.version,
            parameterization = :log_sd,
            matrix_mode = dense.matrix_mode,
            h = dense.h,
            score_max_absdiff_fd = dense.max_absdiff,
            dense_sparse_max_absdiff = 0.0,
            finite = dense.finite,
            claim_status = :internal_diagnostic,
            coverage_status = :not_evaluated,
            ai_reml_ready = false,
            next_gate = :derivative_grid_status,
        ),
        (
            check_id = :sparse_score_fd_fixture,
            target = :gaussian_loconly_phylo_reml,
            fixture_id = fixture.fixture_id,
            fixture_version = fixture.version,
            parameterization = :log_sd,
            matrix_mode = sparse.matrix_mode,
            h = sparse.h,
            score_max_absdiff_fd = sparse.max_absdiff_fd,
            dense_sparse_max_absdiff = sparse.max_absdiff_dense,
            finite = sparse.finite,
            claim_status = :internal_diagnostic,
            coverage_status = :not_evaluated,
            ai_reml_ready = false,
            next_gate = :derivative_grid_status,
        ),
    )
    return (
        target = :gaussian_loconly_phylo_reml,
        fixture_id = fixture.fixture_id,
        fixture_version = fixture.version,
        rows = rows,
        n_rows = length(rows),
        schema = _loconly_reml_derivative_fd_status_schema(),
        claim_status = :internal_diagnostic,
        coverage_status = :not_evaluated,
        ai_reml_ready = false,
    )
end

function _loconly_reml_validate_derivative_fd_status(status;
        max_fd_absdiff::Real = 1e-5,
        max_dense_sparse_absdiff::Real = 1e-7)
    required = _loconly_reml_derivative_fd_status_schema()
    errors = String[]
    status.target === :gaussian_loconly_phylo_reml ||
        push!(errors, "derivative FD status has wrong target")
    status.claim_status === :internal_diagnostic ||
        push!(errors, "derivative FD status has wrong claim_status")
    status.coverage_status === :not_evaluated ||
        push!(errors, "derivative FD status has evaluated coverage")
    !status.ai_reml_ready ||
        push!(errors, "derivative FD status must not mark ai_reml_ready")
    status.n_rows == length(status.rows) ||
        push!(errors, "derivative FD status n_rows does not match rows")
    for row in status.rows
        names = propertynames(row)
        for field in required
            field in names ||
                push!(errors, "derivative FD row $(row.check_id) missing field $(field)")
        end
        row.target === :gaussian_loconly_phylo_reml ||
            push!(errors, "derivative FD row $(row.check_id) has wrong target")
        row.parameterization === :log_sd ||
            push!(errors, "derivative FD row $(row.check_id) has wrong parameterization")
        row.finite ||
            push!(errors, "derivative FD row $(row.check_id) is not finite")
        row.score_max_absdiff_fd <= max_fd_absdiff ||
            push!(errors, "derivative FD row $(row.check_id) exceeds FD tolerance")
        row.dense_sparse_max_absdiff <= max_dense_sparse_absdiff ||
            push!(errors, "derivative FD row $(row.check_id) exceeds dense-sparse tolerance")
        row.claim_status === :internal_diagnostic ||
            push!(errors, "derivative FD row $(row.check_id) has wrong claim_status")
        row.coverage_status === :not_evaluated ||
            push!(errors, "derivative FD row $(row.check_id) has evaluated coverage")
        !row.ai_reml_ready ||
            push!(errors, "derivative FD row $(row.check_id) must not mark ai_reml_ready")
    end
    return (
        ok = isempty(errors),
        errors = Tuple(errors),
        required_fields = required,
        n_rows = length(status.rows),
        claim_status = status.claim_status,
        coverage_status = status.coverage_status,
        ai_reml_ready = status.ai_reml_ready,
    )
end

function _loconly_reml_line_search_status(; iterations::Integer = 30)
    fixture = _loconly_reml_external_comparator_fixture()
    prob = _loconly_reml_problem_from_external_fixture(fixture)
    lσ = log(fixture.known_sigma)
    lσ_phy = log(fixture.known_sigma_phy)
    starts = (
        [lσ, lσ_phy],
        [log(fixture.known_sigma * 1.2), log(fixture.known_sigma_phy * 0.8)],
    )
    fit = _loconly_reml_ai_update_optimizer_diagnostic(
        prob; starts = starts, iterations = iterations,
    )
    trace_status = fit.accepted ? :accepted_guarded_line_search :
        :not_accepted_guarded_line_search
    row = (
        check_id = :guarded_ai_update_line_search_fixture,
        target = :gaussian_loconly_phylo_reml,
        fixture_id = fixture.fixture_id,
        fixture_version = fixture.version,
        estimator = fit.estimator,
        optimizer = fit.optimizer,
        n_starts = fit.n_starts,
        n_records = length(fit.records),
        n_accepted_records = fit.n_accepted_records,
        accepted = fit.accepted,
        best_score_norm = fit.best_score_norm,
        boundary_status = fit.boundary_status,
        trace_status = trace_status,
        claim_status = :optimizer_experiment,
        coverage_status = :not_evaluated,
        ai_reml_ready = false,
        reason_not_ai_reml = fit.reason_not_ai_reml,
        next_gate = :line_search_grid_status,
    )
    return (
        target = :gaussian_loconly_phylo_reml,
        fixture_id = fixture.fixture_id,
        fixture_version = fixture.version,
        rows = (row,),
        n_rows = 1,
        schema = _loconly_reml_line_search_status_schema(),
        claim_status = :optimizer_experiment,
        coverage_status = :not_evaluated,
        ai_reml_ready = false,
    )
end

function _loconly_reml_validate_line_search_status(status;
        max_score_norm::Real = 1e-3)
    required = _loconly_reml_line_search_status_schema()
    errors = String[]
    status.target === :gaussian_loconly_phylo_reml ||
        push!(errors, "line-search status has wrong target")
    status.claim_status === :optimizer_experiment ||
        push!(errors, "line-search status has wrong claim_status")
    status.coverage_status === :not_evaluated ||
        push!(errors, "line-search status has evaluated coverage")
    !status.ai_reml_ready ||
        push!(errors, "line-search status must not mark ai_reml_ready")
    status.n_rows == length(status.rows) ||
        push!(errors, "line-search status n_rows does not match rows")
    for row in status.rows
        names = propertynames(row)
        for field in required
            field in names ||
                push!(errors, "line-search row $(row.check_id) missing field $(field)")
        end
        row.target === :gaussian_loconly_phylo_reml ||
            push!(errors, "line-search row $(row.check_id) has wrong target")
        row.estimator === :guarded_ai_update_reml_optimizer_experiment ||
            push!(errors, "line-search row $(row.check_id) has wrong estimator")
        row.optimizer === :guarded_sparse_average_information_update ||
            push!(errors, "line-search row $(row.check_id) has wrong optimizer")
        row.accepted ||
            push!(errors, "line-search row $(row.check_id) was not accepted")
        row.n_records == row.n_starts ||
            push!(errors, "line-search row $(row.check_id) n_records does not match n_starts")
        row.n_accepted_records >= 1 ||
            push!(errors, "line-search row $(row.check_id) has no accepted records")
        row.best_score_norm <= max_score_norm ||
            push!(errors, "line-search row $(row.check_id) exceeds score tolerance")
        row.claim_status === :optimizer_experiment ||
            push!(errors, "line-search row $(row.check_id) has wrong claim_status")
        row.coverage_status === :not_evaluated ||
            push!(errors, "line-search row $(row.check_id) has evaluated coverage")
        !row.ai_reml_ready ||
            push!(errors, "line-search row $(row.check_id) must not mark ai_reml_ready")
        occursin("no simulation, bridge, or coverage gate", row.reason_not_ai_reml) ||
            push!(errors, "line-search row $(row.check_id) has wrong readiness reason")
    end
    return (
        ok = isempty(errors),
        errors = Tuple(errors),
        required_fields = required,
        n_rows = length(status.rows),
        claim_status = status.claim_status,
        coverage_status = status.coverage_status,
        ai_reml_ready = status.ai_reml_ready,
    )
end

function _loconly_reml_boundary_grid_status(; reps::Integer = 1,
                                            iterations::Integer = 20)
    grid = _loconly_reml_weak_signal_condition_grid_diagnostic(
        ; reps = reps, iterations = iterations,
    )
    rows = Tuple((
        row_id = row.cell,
        target = grid.target,
        design = grid.design,
        condition = row.diagnostic.conditions,
        n_reps = row.diagnostic.n_reps,
        n_accepted = row.diagnostic.n_accepted,
        convergence_rate = row.convergence_rate,
        boundary_rate = row.boundary_rate,
        near_zero_variance = row.diagnostic.boundary_counts.near_zero_variance,
        nonfinite_objective = row.diagnostic.boundary_counts.nonfinite_objective,
        singular_fixed_effect_information =
            row.diagnostic.boundary_counts.singular_fixed_effect_information,
        expected_behavior = grid.expected_behavior,
        claim_status = grid.claim_status,
        coverage_status = grid.coverage_status,
        ai_reml_ready = false,
        next_gate = :boundary_grid_expansion,
    ) for row in grid.rows)
    return (
        target = grid.target,
        estimator = grid.estimator,
        design = :near_zero_variance_boundary_grid,
        rows = rows,
        n_rows = length(rows),
        schema = _loconly_reml_boundary_grid_status_schema(),
        expected_behavior = grid.expected_behavior,
        claim_status = grid.claim_status,
        coverage_status = grid.coverage_status,
        ai_reml_ready = false,
        reason_not_ai_reml = grid.reason_not_ai_reml,
    )
end

function _loconly_reml_validate_boundary_grid_status(status)
    required = _loconly_reml_boundary_grid_status_schema()
    errors = String[]
    status.target === :gaussian_loconly_phylo_reml ||
        push!(errors, "boundary grid status has wrong target")
    status.claim_status === :simulation_diagnostic ||
        push!(errors, "boundary grid status has wrong claim_status")
    status.coverage_status === :not_evaluated ||
        push!(errors, "boundary grid status has evaluated coverage")
    !status.ai_reml_ready ||
        push!(errors, "boundary grid status must not mark ai_reml_ready")
    status.expected_behavior === :boundary_states_allowed ||
        push!(errors, "boundary grid status has wrong expected_behavior")
    status.n_rows == length(status.rows) ||
        push!(errors, "boundary grid status n_rows does not match rows")
    for row in status.rows
        names = propertynames(row)
        for field in required
            field in names ||
                push!(errors, "boundary grid row $(row.row_id) missing field $(field)")
        end
        row.target === :gaussian_loconly_phylo_reml ||
            push!(errors, "boundary grid row $(row.row_id) has wrong target")
        0 <= row.n_accepted <= row.n_reps ||
            push!(errors, "boundary grid row $(row.row_id) has invalid accepted count")
        0 <= row.boundary_rate <= 1 ||
            push!(errors, "boundary grid row $(row.row_id) has invalid boundary_rate")
        row.near_zero_variance + row.nonfinite_objective +
            row.singular_fixed_effect_information <= row.n_reps ||
            push!(errors, "boundary grid row $(row.row_id) has invalid boundary counts")
        row.expected_behavior === :boundary_states_allowed ||
            push!(errors, "boundary grid row $(row.row_id) has wrong expected_behavior")
        row.claim_status === :simulation_diagnostic ||
            push!(errors, "boundary grid row $(row.row_id) has wrong claim_status")
        row.coverage_status === :not_evaluated ||
            push!(errors, "boundary grid row $(row.row_id) has evaluated coverage")
        !row.ai_reml_ready ||
            push!(errors, "boundary grid row $(row.row_id) must not mark ai_reml_ready")
    end
    return (
        ok = isempty(errors),
        errors = Tuple(errors),
        required_fields = required,
        n_rows = length(status.rows),
        claim_status = status.claim_status,
        coverage_status = status.coverage_status,
        ai_reml_ready = status.ai_reml_ready,
    )
end

function _loconly_reml_profile_status(; iterations::Integer = 30,
                                      step::Real = 0.05)
    fixture = _loconly_reml_external_comparator_fixture()
    prob = _loconly_reml_problem_from_external_fixture(fixture)
    lσ = log(fixture.known_sigma)
    lσ_phy = log(fixture.known_sigma_phy)
    starts = (
        [lσ, lσ_phy],
        [log(fixture.known_sigma * 1.2), log(fixture.known_sigma_phy * 0.8)],
    )
    fit = _loconly_reml_ai_update_optimizer_diagnostic(
        prob; starts = starts, iterations = iterations,
    )
    prof = fit.finite ?
        _loconly_reml_local_profile_diagnostic(
            prob, fit.best_minimizer[1], fit.best_minimizer[2]; step = step,
        ) :
        (center = NaN, step = Float64(step), residual_axis = fill(NaN, 3),
         phylo_axis = fill(NaN, 3), finite = false, center_is_axis_min = false)
    function row(row_id::Symbol, axis::Symbol, values)
        return (
            row_id = row_id,
            target = :gaussian_loconly_phylo_reml,
            fixture_id = fixture.fixture_id,
            fixture_version = fixture.version,
            parameterization = :log_sd,
            axis = axis,
            step = prof.step,
            finite = prof.finite,
            center_is_axis_min = prof.center_is_axis_min,
            center_nll = prof.center,
            left_nll = values[1],
            right_nll = values[3],
            claim_status = :internal_diagnostic,
            coverage_status = :not_evaluated,
            ai_reml_ready = false,
            next_gate = :profile_grid_status,
        )
    end
    rows = (
        row(:residual_logsd_profile_axis, :log_sigma, prof.residual_axis),
        row(:phylogenetic_logsd_profile_axis, :log_sigma_phy, prof.phylo_axis),
    )
    return (
        target = :gaussian_loconly_phylo_reml,
        fixture_id = fixture.fixture_id,
        fixture_version = fixture.version,
        rows = rows,
        n_rows = length(rows),
        schema = _loconly_reml_profile_status_schema(),
        claim_status = :internal_diagnostic,
        coverage_status = :not_evaluated,
        ai_reml_ready = false,
    )
end

function _loconly_reml_validate_profile_status(status)
    required = _loconly_reml_profile_status_schema()
    errors = String[]
    status.target === :gaussian_loconly_phylo_reml ||
        push!(errors, "profile status has wrong target")
    status.claim_status === :internal_diagnostic ||
        push!(errors, "profile status has wrong claim_status")
    status.coverage_status === :not_evaluated ||
        push!(errors, "profile status has evaluated coverage")
    !status.ai_reml_ready ||
        push!(errors, "profile status must not mark ai_reml_ready")
    status.n_rows == length(status.rows) ||
        push!(errors, "profile status n_rows does not match rows")
    for row in status.rows
        names = propertynames(row)
        for field in required
            field in names ||
                push!(errors, "profile row $(row.row_id) missing field $(field)")
        end
        row.target === :gaussian_loconly_phylo_reml ||
            push!(errors, "profile row $(row.row_id) has wrong target")
        row.parameterization === :log_sd ||
            push!(errors, "profile row $(row.row_id) has wrong parameterization")
        row.finite ||
            push!(errors, "profile row $(row.row_id) is not finite")
        row.center_is_axis_min ||
            push!(errors, "profile row $(row.row_id) center is not axis minimum")
        row.center_nll <= min(row.left_nll, row.right_nll) + 1e-8 ||
            push!(errors, "profile row $(row.row_id) center exceeds neighboring axis points")
        row.claim_status === :internal_diagnostic ||
            push!(errors, "profile row $(row.row_id) has wrong claim_status")
        row.coverage_status === :not_evaluated ||
            push!(errors, "profile row $(row.row_id) has evaluated coverage")
        !row.ai_reml_ready ||
            push!(errors, "profile row $(row.row_id) must not mark ai_reml_ready")
    end
    return (
        ok = isempty(errors),
        errors = Tuple(errors),
        required_fields = required,
        n_rows = length(status.rows),
        claim_status = status.claim_status,
        coverage_status = status.coverage_status,
        ai_reml_ready = status.ai_reml_ready,
    )
end

function _loconly_reml_variance_component_status(; iterations::Integer = 30)
    fixture = _loconly_reml_external_comparator_fixture()
    prob = _loconly_reml_problem_from_external_fixture(fixture)
    lσ = log(fixture.known_sigma)
    lσ_phy = log(fixture.known_sigma_phy)
    starts = (
        [lσ, lσ_phy],
        [log(fixture.known_sigma * 1.2), log(fixture.known_sigma_phy * 0.8)],
    )
    fit = _loconly_reml_ai_update_optimizer_diagnostic(
        prob; starts = starts, iterations = iterations,
    )
    finite_fit = fit.finite && fit.accepted && all(isfinite, fit.best_minimizer)
    function row(row_id::Symbol, component::Symbol, logsd_parameter::Symbol,
                 logsd_estimate::Real)
        finite = finite_fit && isfinite(logsd_estimate)
        sd_estimate = finite ? exp(Float64(logsd_estimate)) : NaN
        variance_estimate = finite ? abs2(sd_estimate) : NaN
        return (
            row_id = row_id,
            target = :gaussian_loconly_phylo_reml,
            fixture_id = fixture.fixture_id,
            fixture_version = fixture.version,
            estimator = fit.estimator,
            optimizer = fit.optimizer,
            component = component,
            logsd_parameter = logsd_parameter,
            logsd_estimate = finite ? Float64(logsd_estimate) : NaN,
            sd_estimate = sd_estimate,
            variance_estimate = variance_estimate,
            point_status = finite ? :finite_optimizer_diagnostic : :not_available,
            interval_status = :not_evaluated,
            boundary_status = fit.boundary_status,
            claim_status = :internal_diagnostic,
            coverage_status = :not_evaluated,
            ai_reml_ready = false,
            next_gate = :variance_component_parity_status,
        )
    end
    rows = (
        row(:residual_variance_component, :residual, :log_sigma,
            fit.best_minimizer[1]),
        row(:phylogenetic_variance_component, :phylogenetic, :log_sigma_phy,
            fit.best_minimizer[2]),
    )
    return (
        target = :gaussian_loconly_phylo_reml,
        fixture_id = fixture.fixture_id,
        fixture_version = fixture.version,
        rows = rows,
        n_rows = length(rows),
        schema = _loconly_reml_variance_component_status_schema(),
        point_status = all(r -> r.point_status === :finite_optimizer_diagnostic,
                           rows) ? :finite_optimizer_diagnostic : :partial,
        interval_status = :not_evaluated,
        claim_status = :internal_diagnostic,
        coverage_status = :not_evaluated,
        ai_reml_ready = false,
    )
end

function _loconly_reml_validate_variance_component_status(status)
    required = _loconly_reml_variance_component_status_schema()
    errors = String[]
    status.target === :gaussian_loconly_phylo_reml ||
        push!(errors, "variance component status has wrong target")
    status.point_status === :finite_optimizer_diagnostic ||
        push!(errors, "variance component status has wrong point_status")
    status.interval_status === :not_evaluated ||
        push!(errors, "variance component status has evaluated intervals")
    status.claim_status === :internal_diagnostic ||
        push!(errors, "variance component status has wrong claim_status")
    status.coverage_status === :not_evaluated ||
        push!(errors, "variance component status has evaluated coverage")
    !status.ai_reml_ready ||
        push!(errors, "variance component status must not mark ai_reml_ready")
    status.n_rows == length(status.rows) ||
        push!(errors, "variance component status n_rows does not match rows")
    for row in status.rows
        names = propertynames(row)
        for field in required
            field in names ||
                push!(errors, "variance component row $(row.row_id) missing field $(field)")
        end
        row.target === :gaussian_loconly_phylo_reml ||
            push!(errors, "variance component row $(row.row_id) has wrong target")
        row.estimator === :guarded_ai_update_reml_optimizer_experiment ||
            push!(errors, "variance component row $(row.row_id) has wrong estimator")
        row.optimizer === :guarded_sparse_average_information_update ||
            push!(errors, "variance component row $(row.row_id) has wrong optimizer")
        row.component in (:residual, :phylogenetic) ||
            push!(errors, "variance component row $(row.row_id) has wrong component")
        row.logsd_parameter in (:log_sigma, :log_sigma_phy) ||
            push!(errors, "variance component row $(row.row_id) has wrong logsd_parameter")
        row.point_status === :finite_optimizer_diagnostic ||
            push!(errors, "variance component row $(row.row_id) has wrong point_status")
        isfinite(row.logsd_estimate) ||
            push!(errors, "variance component row $(row.row_id) has nonfinite logsd_estimate")
        isfinite(row.sd_estimate) && row.sd_estimate > 0 ||
            push!(errors, "variance component row $(row.row_id) has invalid sd_estimate")
        isfinite(row.variance_estimate) && row.variance_estimate > 0 ||
            push!(errors, "variance component row $(row.row_id) has invalid variance_estimate")
        isapprox(row.variance_estimate, abs2(row.sd_estimate);
                 rtol = 1e-10, atol = 1e-12) ||
            push!(errors, "variance component row $(row.row_id) has inconsistent variance_estimate")
        row.boundary_status === :interior ||
            push!(errors, "variance component row $(row.row_id) is not interior")
        row.interval_status === :not_evaluated ||
            push!(errors, "variance component row $(row.row_id) has evaluated intervals")
        row.claim_status === :internal_diagnostic ||
            push!(errors, "variance component row $(row.row_id) has wrong claim_status")
        row.coverage_status === :not_evaluated ||
            push!(errors, "variance component row $(row.row_id) has evaluated coverage")
        !row.ai_reml_ready ||
            push!(errors, "variance component row $(row.row_id) must not mark ai_reml_ready")
    end
    return (
        ok = isempty(errors),
        errors = Tuple(errors),
        required_fields = required,
        n_rows = length(status.rows),
        point_status = status.point_status,
        interval_status = status.interval_status,
        claim_status = status.claim_status,
        coverage_status = status.coverage_status,
        ai_reml_ready = status.ai_reml_ready,
    )
end

function _loconly_reml_external_comparator_candidates()
    return (
        (
            comparator_id = :internal_dense_gls_oracle,
            target = :gaussian_loconly_phylo_reml,
            comparator = "DRModels.jl dense GLS oracle",
            same_estimand_status = :same_estimand_internal,
            dependency_status = :internal,
            artifact_status = :covered_by_focused_test,
            decision = :retain_as_gate,
            reason = "Matches the exact Gaussian restricted objective and requires no new dependency.",
            next_gate = :external_package_selection,
        ),
        (
            comparator_id = :phylolm_or_equivalent_reml,
            target = :gaussian_loconly_phylo_reml,
            comparator = "phylolm-style Gaussian phylogenetic REML",
            same_estimand_status = :needs_fixture_confirmation,
            dependency_status = :not_added,
            artifact_status = :fixture_defined,
            decision = :scout_before_dependency,
            reason = "Candidate must match the versioned same-estimand fixture before any dependency is added.",
            next_gate = :external_package_version_probe,
        ),
        (
            comparator_id = :mixedmodels_or_generic_lmm,
            target = :gaussian_loconly_phylo_reml,
            comparator = "generic LMM package",
            same_estimand_status = :not_yet_same_estimand,
            dependency_status = :not_added,
            artifact_status = :not_applicable,
            decision = :do_not_use_without_covariance_target_match,
            reason = "A generic random-intercept LMM is not enough unless the supplied phylogenetic covariance or precision matches the target.",
            next_gate = :reject_or_specialize,
        ),
    )
end

function _loconly_reml_validate_external_comparator_rows(rows)
    schema = _loconly_reml_external_comparator_schema()
    errors = String[]
    for row in rows
        names = propertynames(row)
        for field in schema
            field in names || push!(errors, "comparator $(row.comparator_id) missing field $(field)")
        end
        row.target === :gaussian_loconly_phylo_reml ||
            push!(errors, "comparator $(row.comparator_id) has wrong target")
        row.dependency_status in (:internal, :not_added, :optional_developer_only) ||
            push!(errors, "comparator $(row.comparator_id) has invalid dependency_status")
        row.artifact_status in (:covered_by_focused_test, :planned, :not_applicable,
            :optional_developer_only, :fixture_defined) ||
            push!(errors, "comparator $(row.comparator_id) has invalid artifact_status")
    end
    return (
        ok = isempty(errors),
        errors = Tuple(errors),
        required_fields = schema,
        n_rows = length(rows),
    )
end

function _loconly_reml_external_comparator_status()
    rows = _loconly_reml_external_comparator_candidates()
    validation = _loconly_reml_validate_external_comparator_rows(rows)
    fixture = _loconly_reml_external_comparator_fixture()
    fixture_validation = _loconly_reml_validate_external_comparator_fixture(fixture)
    version_probe_rows =
        _loconly_reml_external_comparator_version_probe_rows(fixture)
    version_probe_validation =
        _loconly_reml_validate_external_comparator_version_probe(
            version_probe_rows, fixture)
    return (
        target = :gaussian_loconly_phylo_reml,
        external_comparator_status = :planned,
        dependency_status = :not_added,
        artifact_schema = _loconly_reml_external_comparator_schema(),
        fixture_status = :versioned_fixture_defined,
        fixture_schema = _loconly_reml_external_comparator_fixture_schema(),
        fixture = fixture,
        fixture_validation = fixture_validation,
        version_probe_status = :defined_not_run,
        version_probe_schema =
            _loconly_reml_external_comparator_version_probe_schema(),
        version_probe_rows = version_probe_rows,
        version_probe_validation = version_probe_validation,
        rows = rows,
        validation = validation,
        claim_status = :internal_diagnostic,
        coverage_status = :not_evaluated,
        ai_reml_ready = false,
        reason_not_added = "No external dependency is added until a same-estimand fixture and package/version are chosen.",
    )
end

function _loconly_reml_bridge_payload_schema()
    return (
        target = "gaussian_loconly_phylo_reml",
        estimator = "supplied_variance_reml",
        effective_REML = true,
        variance_components_source = "supplied_or_experimental_fd_optimized",
        trace_mode = "takahashi_selinv",
        score_mode = "dense_or_sparse_woodbury_diagnostic",
        information_mode = "ai_vs_observed_diagnostic",
        boundary_status_levels = (
            "interior",
            "near_zero_variance",
            "singular_fixed_effect_information",
            "nonfinite_objective",
        ),
        claim_status = "internal_diagnostic",
        r_bridge_status = "planned",
    )
end

function _loconly_reml_diagnostic_payload(prob::LocOnlyProblem, lσ::Real, lσ_phy::Real)
    return (
        target = :gaussian_loconly_phylo_reml,
        estimator = :supplied_variance_reml,
        parameterization = :log_sd,
        boundary = _loconly_reml_boundary_status(prob, lσ, lσ_phy),
        dense_comparator = _loconly_dense_comparator_diagnostic(prob, lσ, lσ_phy),
        score = _loconly_reml_dense_score_diagnostic(prob, lσ, lσ_phy),
        sparse_score = _loconly_reml_sparse_score_diagnostic(prob, lσ, lσ_phy),
        trace = _loconly_takahashi_trace_diagnostic(prob, lσ, lσ_phy),
        pev = _loconly_takahashi_pev_diagnostic(prob, lσ, lσ_phy),
        information = _loconly_ai_information_diagnostic(prob, lσ, lσ_phy),
        sparse_information = _loconly_reml_sparse_ai_information_diagnostic(prob, lσ, lσ_phy),
        fd_stability = _loconly_reml_fd_stability_diagnostic(prob, lσ, lσ_phy),
        local_profile = _loconly_reml_local_profile_diagnostic(prob, lσ, lσ_phy),
        validation_status = _loconly_reml_validation_status(),
        bridge_schema = _loconly_reml_bridge_payload_schema(),
        claim_status = :internal_diagnostic,
    )
end

function _loconly_profile_fg(prob::LocOnlyProblem, lσ::Real, lσ_phy::Real)
    β, nll, chM = _loconly_profile_beta(prob, lσ, lσ_phy)
    if β === nothing
        return _LOCONLY_PENALTY, zeros(2), β, chM
    end
    σ²_phy = exp(2 * Float64(lσ_phy))
    σ² = exp(2 * Float64(lσ))
    e = prob.y .- prob.X * β
    μ_post = chM \ (prob.S' * e / σ²)
    tr_QM, tr_SMS = exact_traces(prob, chM)
    qquad = dot(μ_post, prob.Q_cond * μ_post)
    e2 = e .- prob.S * μ_post
    rss = dot(e2, e2)
    grad = [prob.n - (rss + tr_SMS) / σ², prob.n_keep - (qquad + tr_QM) / σ²_phy]
    return nll, grad, β, chM
end

function _loconly_marginal_grad!(g, prob::LocOnlyProblem, θ::AbstractVector, pμ::Int)
    fill!(g, 0.0)
    length(θ) == pμ + 2 || throw(
        DimensionMismatch(
            "location-only gradient expected $(pμ + 2) parameters, got $(length(θ))"
        ),
    )
    lσ = θ[pμ + 1]
    lσ_phy = θ[pμ + 2]
    (isfinite(lσ) && isfinite(lσ_phy) && abs(lσ) < 50 && abs(lσ_phy) < 50 &&
     _loconly_resolvable_scales(lσ, lσ_phy)) || return g
    σ² = exp(2 * Float64(lσ))
    σ²_phy = exp(2 * Float64(lσ_phy))
    try
        _, _, chM, _ = build_M(prob, σ²_phy, σ²)
        β = @view θ[1:pμ]
        e = prob.y .- prob.X * β
        Ve = Vinv_mul(prob, chM, σ², e)
        g[1:pμ] .= -(prob.X' * Ve)

        μ_post = chM \ (prob.S' * e / σ²)
        tr_QM, tr_SMS = exact_traces(prob, chM)
        qquad = dot(μ_post, prob.Q_cond * μ_post)
        e2 = e .- prob.S * μ_post
        rss = dot(e2, e2)
        g[pμ + 1] = prob.n - (rss + tr_SMS) / σ²
        g[pμ + 2] = prob.n_keep - (qquad + tr_QM) / σ²_phy
    catch
        fill!(g, 0.0)
    end
    return g
end

# ─────────────────────────────────────────────────────────────────────────────
# EXACT TAKAHASHI TRACES for the EM M-step
#
# Tr(Q_cond M^{-1}): sum Q_cond[i,j] × M^{-1}[i,j] over the Q_cond pattern.
# Tr(S M^{-1} S') = Σ_j STS_diag[j] × M^{-1}[j, j]  (diagonal always in pattern).
# ─────────────────────────────────────────────────────────────────────────────

"""
EXACT Tr(Q_cond M^{-1}) and Tr(S M^{-1} S') via Takahashi selected inverse.
Returns (tr_QM, tr_SMS).
"""
function exact_traces(prob::LocOnlyProblem, chM)
    V_sel = takahashi_selinv(chM)

    rows = rowvals(prob.Q_cond)
    vals = nonzeros(prob.Q_cond)
    tr_QM = 0.0
    @inbounds for tcol in 1:(prob.n_keep)
        for idx in nzrange(prob.Q_cond, tcol)
            s = rows[idx]
            q = vals[idx]
            tr_QM += q * V_sel[s, tcol]
        end
    end

    tr_SMS = 0.0
    @inbounds for j in 1:(prob.n_keep)
        if prob.STS_diag[j] > 0
            tr_SMS += prob.STS_diag[j] * V_sel[j, j]
        end
    end

    return tr_QM, tr_SMS
end

function _loconly_takahashi_trace_diagnostic(prob::LocOnlyProblem, lσ::Real, lσ_phy::Real)
    (isfinite(lσ) && isfinite(lσ_phy) && abs(lσ) < 50 && abs(lσ_phy) < 50) ||
        return (trace_mode = :takahashi_selinv, tr_QM = NaN, tr_SMS = NaN,
                finite = false, n = prob.n, n_keep = prob.n_keep)
    σ²_phy = exp(2 * Float64(lσ_phy))
    σ² = exp(2 * Float64(lσ))
    try
        _, _, chM, _ = build_M(prob, σ²_phy, σ²)
        tr_QM, tr_SMS = exact_traces(prob, chM)
        finite = isfinite(tr_QM) && isfinite(tr_SMS)
        return (trace_mode = :takahashi_selinv, tr_QM = tr_QM, tr_SMS = tr_SMS,
                finite = finite, n = prob.n, n_keep = prob.n_keep)
    catch err
        err isa InterruptException && rethrow(err)
        return (trace_mode = :takahashi_selinv, tr_QM = NaN, tr_SMS = NaN,
                finite = false, n = prob.n, n_keep = prob.n_keep)
    end
end

function _loconly_takahashi_pev_diagnostic(prob::LocOnlyProblem, lσ::Real, lσ_phy::Real)
    (isfinite(lσ) && isfinite(lσ_phy) && abs(lσ) < 50 && abs(lσ_phy) < 50) ||
        return (trace_mode = :takahashi_selinv, posterior_variance = Float64[],
                leaf_posterior_variance = Float64[], finite = false, n_keep = prob.n_keep,
                n_leaves = prob.p)
    σ²_phy = exp(2 * Float64(lσ_phy))
    σ² = exp(2 * Float64(lσ))
    try
        _, _, chM, _ = build_M(prob, σ²_phy, σ²)
        V_sel = takahashi_selinv(chM)
        diag_all = [V_sel[j, j] for j in 1:(prob.n_keep)]
        leaf_diag = diag_all[prob.leaf_pos]
        finite = all(isfinite, diag_all)
        return (trace_mode = :takahashi_selinv, posterior_variance = diag_all,
                leaf_posterior_variance = leaf_diag, finite = finite,
                posterior_variance_min = minimum(diag_all),
                posterior_variance_max = maximum(diag_all),
                leaf_posterior_variance_mean = mean(leaf_diag),
                weighted_leaf_posterior_trace = sum(prob.STS_diag .* diag_all),
                n_keep = prob.n_keep, n_leaves = prob.p)
    catch err
        err isa InterruptException && rethrow(err)
        return (trace_mode = :takahashi_selinv, posterior_variance = Float64[],
                leaf_posterior_variance = Float64[], finite = false,
                n_keep = prob.n_keep, n_leaves = prob.p)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# EM FITTER — closed-form E-step + M-step with exact Takahashi traces
#
# E-step: μ_post = M^{-1} (1/σ²) S' (y - Xβ)  [posterior mean of u]
# M-step (β):      GLS = (X' V^{-1} X)^{-1} X' V^{-1} y
# M-step (σ²_phy): [μ' Q_cond μ + Tr(Q_cond M^{-1})] / n_keep
# M-step (σ²):     [||y - Xβ - Sμ||² + Tr(S M^{-1} S')] / n
# ─────────────────────────────────────────────────────────────────────────────

function em_fit(
    prob::LocOnlyProblem; β0=nothing, σ²_phy0=0.5, σ²0=0.5, max_iter=200, reltol=1e-8
)
    n = prob.n
    k = prob.k
    n_keep = prob.n_keep

    β = β0 === nothing ? zeros(k) : copy(Float64.(β0))
    σ²_phy = σ²_phy0
    σ² = σ²0

    ll_prev = -Inf
    iter = 0

    for it in 1:max_iter
        iter = it
        P, M, chM, chP = build_M(prob, σ²_phy, σ²)

        # E-step: posterior mean
        e = prob.y .- prob.X * β
        μ_post = chM \ (prob.S' * e / σ²)

        # Exact traces (Takahashi)
        tr_QM, tr_SMS = exact_traces(prob, chM)

        # M-step (β): GLS
        VX = Vinv_mul(prob, chM, σ², prob.X)
        Vy = Vinv_mul(prob, chM, σ², prob.y)
        β_new = (prob.X' * VX) \ (prob.X' * Vy)

        # M-step (σ²_phy): closed form
        qquad = dot(μ_post, prob.Q_cond * μ_post)
        σ²_phy_new = max(1e-8, (qquad + tr_QM) / n_keep)

        # M-step (σ²): closed form
        e2 = prob.y .- prob.X * β_new .- prob.S * μ_post
        σ²_new = max(1e-8, (dot(e2, e2) + tr_SMS) / n)

        β = β_new
        σ²_phy = σ²_phy_new
        σ² = σ²_new

        ll = marginal_loglik(prob, β, σ²_phy, σ²)
        if abs(ll - ll_prev) < reltol * (1 + abs(ll_prev))
            break
        end
        ll_prev = ll
    end

    _, _, chM, _ = build_M(prob, σ²_phy, σ²)
    e = prob.y .- prob.X * β
    u_post = chM \ (prob.S' * e / σ²)
    ll_final = marginal_loglik(prob, β, σ²_phy, σ²)
    return (β=β, σ²_phy=σ²_phy, σ²=σ², u=u_post, loglik=ll_final, iterations=iter)
end

# ─────────────────────────────────────────────────────────────────────────────
# Front-end adapter — build a DrmFit matching the structured-Gaussian shape.
#
# This is the `algorithm = :em` path for the Gaussian phylo-MEAN cell. It fits
# the SAME marginal as `_fit_structured_gaussian` (single structured mean RE,
# constant σ) but via conjugate EM. The structured-Gaussian fit parametrizes the
# phylo random effect as σ_s² · K with K the leaf CORRELATION matrix; the EM uses
# σ²_phy · Σ_phy with Σ_phy the (root-conditioned) Brownian covariance whose
# diagonal is not 1. The two therefore agree on β, the residual σ, and the
# marginal logLik, but their phylo-variance parameters live on different scales —
# so `re_sd` here reports the EM's σ_phy (documented), and the correctness anchor
# is logLik + β + residual σ.
#
# EM does not produce a coefficient vcov (the M-steps are closed-form, no outer
# Hessian); like other boundary cases we store NaNs in the vcov (documented).
# ─────────────────────────────────────────────────────────────────────────────

function _fit_structured_gaussian_em(
    fam::Gaussian, y, Xμ, Xσ, gidx, G, phy::AugmentedPhy, nmμ, nmσ, grp, g_tol
)
    pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    pσ == 1 || error(
        "algorithm = :em supports a CONSTANT residual scale only " *
        "(`sigma ~ 1`); got a $(pσ)-column `sigma` design",
    )
    G == phy.n_leaves || error(
        "algorithm = :em: the number of `$grp` levels ($G) must equal the tree's " *
        "leaf count ($(phy.n_leaves))",
    )
    n = length(y)

    prob = make_loc_problem(phy, y, Xμ; species=gidx)

    # Conjugate-EM init mirrors the structured fitter's OLS-residual heuristic.
    βμ0 = Xμ \ y
    res0 = y - Xμ * βμ0
    s0 = std(res0) + eps()
    r = em_fit(
        prob; β0=βμ0, σ²_phy0=(s0 / 2)^2, σ²0=s0^2, max_iter=500, reltol=max(1e-12, g_tol^2)
    )

    # θ matches the structured shape: [βμ; log σ (constant); log σ_phy].
    # σ here is the residual SD √σ²; σ_phy is √σ²_phy (the EM's phylo SD).
    θ̂ = vcat(r.β, log(sqrt(r.σ²)), log(sqrt(r.σ²_phy)))
    blocks = [
        :mu => 1:pμ, :sigma => (pμ + 1):(pμ + pσ), :resd => (pμ + pσ + 1):(pμ + pσ + 1)
    ]
    names = [:mu => nmμ, :sigma => nmσ, :resd => [String(grp)]]
    # EM yields no coefficient vcov → NaNs (documented), as for other boundary cases.
    V = fill(NaN, length(θ̂), length(θ̂))
    means = Dict(:mu => Xμ * r.β + prob.S * r.u)
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => fill(sqrt(r.σ²), n))
    converged = r.iterations < 500
    return DrmFit(fam, blocks, names, θ̂, V, r.loglik, n, converged, means, obs, scales)
end

function _fit_structured_gaussian_sparse_lbfgs(
    fam::Gaussian, y, Xμ, Xσ, gidx, G, phy::AugmentedPhy, nmμ, nmσ, grp, g_tol;
    penalty = nothing, reml::Bool = false
)
    pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    # #624 item (c). `reml = true` swaps the PROFILE-ML objective over
    # v = [log σ_e, log σ_phylo] for the Patterson–Thompson RESTRICTED one,
    #     nll_REML(v) = nll_ML(v, β̂(v)) + 0.5·logdet(Xμ′V⁻¹Xμ) − 0.5·pμ·log(2π),
    # built by `_loconly_reml_components` on this same sparse spine. β_μ is
    # profiled out EXACTLY by GLS at every v, so the restriction is exact — the
    # same integrated-out set (β_μ AND the phylo field u) and the same additive
    # constant `+0.5·pμ·log(2π)` that native drmTMB gets from TMB's Laplace fold
    # of `beta_mu` into `random=`. Nothing else about the route changes: ML stays
    # byte-for-byte the path it was.
    (!reml || penalty === nothing) || throw(ArgumentError(
        "drm: `penalty` and `method = :REML` cannot be combined on the sparse phylo-mean route"))
    pσ == 1 || error(
        "algorithm = :sparse_lbfgs supports a CONSTANT residual scale only " *
        "(`sigma ~ 1`); got a $(pσ)-column `sigma` design",
    )
    G == phy.n_leaves || error(
        "algorithm = :sparse_lbfgs: the number of `$grp` levels ($G) must equal " *
        "the tree's leaf count ($(phy.n_leaves))",
    )
    n = length(y)
    prob = make_loc_problem(phy, y, Xμ; species=gidx)

    βμ0 = Xμ \ y
    res0 = y - Xμ * βμ0
    s0 = std(res0) + eps()
    starts = [
        [log(s0 + eps()), log(s0 / 2 + eps())],
        [log(s0 / sqrt(2) + eps()), log(s0 / sqrt(2) + eps())],
        [log(s0 / 2 + eps()), log(s0 + eps())],
    ]

    # v = [log sd_residual, log sd_phylo]; the penalty is on the PHYLO SD, v[2].
    # It is added AFTER beta is profiled out, which is exact: the penalty does not
    # depend on beta, so the profiled beta-hat at fixed (v1, v2) is unchanged.
    # On a failed inner solve `_loconly_profile_fg` returns a sentinel with a zero
    # gradient — leave that alone rather than penalising a non-solution.
    # REML has no analytic score on this spine (the log-determinant term differentiates
    # through the sparse Cholesky), so it uses the central finite-difference gradient
    # `_loconly_fd_gradient2` — the same objective/gradient pair the validated
    # `_loconly_reml_optimizer_diagnostic` uses, over the same 2-parameter space.
    reml_obj(v) = _loconly_reml_nll(prob, v[1], v[2])
    function fg!(F, G, v)
        if reml
            val = reml_obj(v)
            G !== nothing && copyto!(G, _loconly_fd_gradient2(reml_obj, v))
            return F === nothing ? nothing : val
        end
        val, grad, β, _ = _loconly_profile_fg(prob, v[1], v[2])
        if penalty !== nothing && β !== nothing
            grad = copy(grad)
            val += _phylo_pen_apply_single!(grad, penalty, v, 2)
        end
        G !== nothing && copyto!(G, grad)
        return F === nothing ? nothing : val
    end

    ls = Optim.LBFGS(; linesearch=Optim.LineSearches.BackTracking(; order=3))
    opts = Optim.Options(; g_tol=g_tol, iterations=500, f_reltol=1e-9)
    od = Optim.NLSolversBase.only_fg!(fg!)
    best_res = nothing
    best_val = Inf
    for start in starts
        res = Optim.optimize(od, start, ls, opts)
        val = _objective_at_minimizer_fg(fg!, res)   # NOT Optim.minimum(res): see optim_minimum_guard.jl
        if isfinite(val) && val < best_val
            best_res = res
            best_val = val
        end
    end
    best_res === nothing &&
        error("algorithm = :sparse_lbfgs failed to find a finite sparse phylo optimum")

    v̂ = Optim.minimizer(best_res)
    β̂, nllhat, chM = _loconly_profile_beta(prob, v̂[1], v̂[2])
    β̂ === nothing && error("algorithm = :sparse_lbfgs optimum could not be evaluated")
    θ̂ = vcat(β̂, v̂[1], v̂[2])
    # REML log-likelihood at the REML optimum, and the plain ML value AT THAT SAME
    # point (`ml_loglik`, for AIC/BIC-style comparison — never a second optimum).
    reml_comp = reml ? _loconly_reml_components(prob, v̂[1], v̂[2]) : nothing
    if reml
        (reml_comp.converged && isfinite(reml_comp.nll)) ||
            error("drm (Gaussian phylo mean, method = :REML): the restricted objective is not " *
                  "finite at the optimum (Xμ′V⁻¹Xμ was not positive definite)")
    end

    loc_obj = LocOnlyObjective(prob, pμ)
    nllgrad! = (g, θ) -> _loconly_objective_grad!(g, loc_obj, θ)

    σ² = exp(2 * θ̂[pμ + 1])
    VX = Vinv_mul(prob, chM, σ², prob.X)
    V = zeros(length(θ̂), length(θ̂))
    V[1:pμ, 1:pμ] .= try
        inv(Symmetric((prob.X' * VX + VX' * prob.X) ./ 2))
    catch
        fill(NaN, pμ, pμ)
    end
    # #556: the variance-parameter block used to be NaN BY CONSTRUCTION, so
    # `sigma`/`resd` SEs were NA on every fit this route serves (Mizuno M2 —
    # caught by the acceptance matrix; drmTMB native reports finite SEs here).
    # β is profiled out EXACTLY, so the 2×2 curvature of the profiled objective
    # at v̂ is the correct observed information for (log σ_e, log σ_phylo), and
    # Gaussian mean/covariance orthogonality zeroes the cross block (left 0).
    let vhat = [θ̂[pμ + 1], θ̂[pμ + 2]]
        # Under REML the variance-component curvature must come from the RESTRICTED
        # objective (that is the whole point of REML SEs); the mean block above is
        # (Xμ′V̂⁻¹Xμ)⁻¹ either way, which is what TMB reports for `beta_mu` too.
        fv = function (v)
            reml && return reml_obj(v)
            val, _, βv, _ = _loconly_profile_fg(prob, v[1], v[2])
            if penalty !== nothing && βv !== nothing
                val += _phylo_pen_apply_single!(zeros(2), penalty, v, 2)
            end
            return val
        end
        Hv = _finite_hessian(fv, vhat; h = _fd_hessian_step(n))
        V[(pμ + 1):(pμ + 2), (pμ + 1):(pμ + 2)] .=
            _vcov_from_hessian(Hv; context = "sparse phylo-mean variance block")
    end
    e = prob.y .- prob.X * β̂
    u_post = chM \ (prob.S' * e / σ²)
    blocks = [
        :mu => 1:pμ, :sigma => (pμ + 1):(pμ + pσ), :resd => (pμ + pσ + 1):(pμ + pσ + 1)
    ]
    names = [:mu => nmμ, :sigma => nmσ, :resd => [String(grp)]]
    means = Dict(:mu => Xμ * β̂ + prob.S * u_post)
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => fill(exp(θ̂[pμ + 1]), n))
    # `nllhat` comes from `_loconly_profile_beta`, which never sees the penalty, so
    # `-nllhat` is the UNPENALIZED data log-likelihood — exactly what drmTMB keeps
    # in `fit$logLik`. The penalty at the optimum is recorded separately, giving
    # the drmTMB identity  -objective == loglik - phylo_penalty.
    fit = DrmFit(
        fam, blocks, names, θ̂, V, -nllhat, n, Optim.converged(best_res), means, obs, scales
    )
    # OBJECTIVE HONESTY. `loc_obj` / `nllgrad!` are the ML marginal and its analytic
    # score. Attaching them to a REML fit would hand every downstream consumer
    # (`fit.nll`, and the R bridge's `gradient` field) the WRONG objective: measured
    # on the drmTMB fixture, the ML score at the REML optimum is (1.01, 0.99) on the
    # two variance parameters, which reads as "not converged" for a fit that is
    # converged — the same class of silent mislabel as returning ML wearing a REML
    # label. So a REML fit carries the RESTRICTED objective as a function of the full
    # θ = [β; log σ_e; log σ_phylo] (the restriction term does not depend on β, so
    # this agrees with the profiled value at β̂), and NO gradient: there is no
    # analytic score for the log-determinant term on this sparse spine, and reporting
    # a finite-difference one as if it were exact would be its own small lie.
    reml_full_nll = function (θ)
        ml = _loconly_marginal_nll(prob, @view(θ[1:pμ]), θ[pμ + 1], θ[pμ + 2])
        isfinite(ml) || return _LOCONLY_PENALTY
        comp = _loconly_reml_components(prob, θ[pμ + 1], θ[pμ + 2])
        (comp.converged && isfinite(comp.penalty)) || return _LOCONLY_PENALTY
        val = ml + comp.penalty - _loconly_reml_constant_offset(prob)
        return isfinite(val) ? val : _LOCONLY_PENALTY
    end
    fit = _withranef(
        reml ? _withnll(fit, reml_full_nll) : _withnll(fit, loc_obj, nllgrad!),
        Dict(Symbol(grp) => u_post[prob.leaf_pos])
    )
    if penalty !== nothing
        fit = _withmap(fit, _phylo_pen_apply_single!(nothing, penalty, v̂, 2), penalty)
    end
    if reml
        fit = _withreml(fit, -reml_comp.nll, -reml_comp.ml_nll)
    end
    return fit
end
