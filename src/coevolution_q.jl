# coevolution_q.jl — GENERAL-q multivariate-Brownian coevolution on a tree (#188).
#
# WHAT THIS IS (and how it relates to the verified q=4 PLSM)
# ----------------------------------------------------------
# The verified `sparse_aug_plsm.jl` / `fit_q4_sparse_tmb.jl` engine fits the q=4
# bivariate location–scale PLSM: four axes (μ1, μ2, log σ1, log σ2) that share a
# 4×4 among-axis covariance Λ (= Σ_a). Two of its axes are *log-σ* axes and it
# carries a *residual correlation* ρ12 — so its per-leaf likelihood is
# intrinsically BIVARIATE (a 2×2 Gaussian with one ρ). That leaf model does not
# parameterise to "q traits"; see the report / HANDOVER for the q=4 baking.
#
# This file implements the *canonical general-q coevolution block*: the part of
# the model that DOES generalise to arbitrary q — the among-trait evolutionary
# covariance Λ (q×q) carried on the tree by the SAME sparse augmented-state
# precision P = kron(Q_topology, Λ⁻¹). The model here is multivariate Brownian
# motion of q traits with a diagonal residual (the among-trait dependence lives
# entirely in Λ, the standard coevolution identification, Hadfield–Nakagawa):
#
#   per node t:   b_t ∈ ℝ^q,   vec(B) ~ N(0, Q_topology⁻¹ ⊗ Λ)   (precision kron)
#   per data row i at tip s(i):  y_i = X_i β + b_{s(i)} + ε_i,  ε_i ~ N(0, D),
#                                D = diag(σ²_res,1 … σ²_res,q).
#
# Because the leaf likelihood is GAUSSIAN in u, this is a *conjugate* Laplace
# problem: H_uu = P + blockdiag(per-leaf D⁻¹) is CONSTANT in u, the inner mode is
# a single sparse solve (no Newton iteration), and the Laplace marginal is EXACT.
# That is the clean, well-posed object on which "recover the among-trait
# correlation + variances at q=6/q=8" is a meaningful test.
#
# It reuses the genuinely q-agnostic primitives from the verified engine —
# `augmented_tree_precision`, `prior_precision = kron(Q, Λ⁻¹)`, `random_*_tree`
# — and TOUCHES NONE of the q=4 leaf code, so the q=4 path is unaffected.
#
# θ layout: [β (k·q, trait-major); lc (q(q+1)/2, log-Cholesky of Λ);
#            log σ_res (q)].

using LinearAlgebra, SparseArrays, Statistics

# ---------------------------------------------------------------------------
# General q×q log-Cholesky  ⇆  covariance.  (q=4 ⇒ the 10-vector of fit_ml_q4.)
# Lower-triangular Cholesky L with exp() on the diagonal (so Λ = L L' is PD for
# any real vector). Column-major over the lower triangle: (i≥j), matching
# `lc_to_Λ`/`Λ_to_lc`'s q=4 convention exactly.
# ---------------------------------------------------------------------------

"Number of free log-Cholesky parameters for a q×q covariance: q(q+1)/2."
lc_len(q::Integer) = (q * (q + 1)) ÷ 2

"""
    lc_to_cov(v, q) -> Matrix

Map a length-`q(q+1)/2` log-Cholesky vector to a q×q SPD covariance `Λ = L L'`,
diagonal of `L` exponentiated. Column-major lower-triangle order (j outer, i≥j
inner) — identical to `lc_to_Λ` at q=4. Eltype follows `v` (AD-friendly).
"""
function lc_to_cov(v::AbstractVector{T}, q::Integer) where {T}
    length(v) == lc_len(q) || error("lc length $(length(v)) ≠ q(q+1)/2 = $(lc_len(q)) for q=$q")
    L = zeros(T, q, q)
    k = 0
    @inbounds for j in 1:q, i in j:q
        k += 1
        L[i, j] = i == j ? exp(v[k]) : v[k]
    end
    return L * L'
end

"""
    cov_to_lc(Λ) -> Vector{Float64}

Inverse of [`lc_to_cov`](@ref): log-Cholesky vector of a SPD `Λ` (q inferred
from `size`). Column-major lower-triangle order.
"""
function cov_to_lc(Λ::AbstractMatrix)
    q = LinearAlgebra.checksquare(Λ)
    L = cholesky(Symmetric(Λ)).L
    v = Float64[]
    @inbounds for j in 1:q, i in j:q
        push!(v, i == j ? log(L[i, j]) : L[i, j])
    end
    return v
end

# ---------------------------------------------------------------------------
# Problem container.
# ---------------------------------------------------------------------------

"""
    CoevoProblem

A general-q coevolution dataset bound to a tree. `Y[i, :]` is the q-trait vector
for data row `i`; `X` is the shared `n × k` fixed-effect design (same β columns
per trait — the common comparative-biology setup); `leaf_node[i]` maps row `i`
to its column in the root-conditioned augmented node set.
"""
struct CoevoProblem
    q::Int                       # number of traits / axes
    N::Int                       # n_keep = 2p - 2 augmented (non-root) nodes
    p::Int                       # number of tips
    k::Int                       # fixed-effect design width (per trait)
    leaf_node::Vector{Int}       # data row i -> node column in 1:N
    Y::Matrix{Float64}           # n × q observations
    X::Matrix{Float64}           # n × k fixed-effect design
end

"""
    make_coevo_problem(phy, Y, X; species) -> (prob::CoevoProblem, Q_cond)

Build a [`CoevoProblem`](@ref) and the root-conditioned sparse tree precision
`Q_cond` (q-agnostic; the SAME object the q=4 engine consumes). `species[i]`
maps data row `i` to a tip `1:phy.n_leaves`; default one row per tip.
"""
function make_coevo_problem(phy, Y::AbstractMatrix, X::AbstractMatrix;
                            species = 1:phy.n_leaves)
    Q_cond, leaf_pos, N = augmented_tree_precision(phy)
    n, q = size(Y)
    size(X, 1) == n || error("X has $(size(X,1)) rows but Y has $n")
    length(species) == n || error("species has $(length(species)) entries but Y has $n rows")
    leaf_node = [leaf_pos[species[i]] for i in 1:n]
    prob = CoevoProblem(q, N, phy.n_leaves, size(X, 2), leaf_node,
                        Matrix{Float64}(Y), Matrix{Float64}(X))
    return prob, Q_cond
end

"""
    make_coevo_problem_from_precision(Q, Y, X; group) -> (prob, Q)

Build a [`CoevoProblem`](@ref) from a known structured precision matrix over
observed levels rather than from an augmented phylogeny. `group[i]` maps row `i`
to a 1-based level in `Q`. This is the direct exact-Gaussian q2/q route for
known relatedness/covariance fixtures such as relmat and animal matrices.
"""
function make_coevo_problem_from_precision(Q_cond::AbstractMatrix,
                                           Y::AbstractMatrix,
                                           X::AbstractMatrix;
                                           group = 1:size(Q_cond, 1))
    G = LinearAlgebra.checksquare(Q_cond)
    n, q = size(Y)
    size(X, 1) == n || error("X has $(size(X,1)) rows but Y has $n")
    length(group) == n || error("group has $(length(group)) entries but Y has $n rows")
    leaf_node = Int.(collect(group))
    all(1 .<= leaf_node .<= G) ||
        error("group indices must be 1-based integers in 1:$G")
    Q = Matrix{Float64}(Q_cond)
    isposdef(Symmetric(Q)) || error("known structured precision must be positive definite")
    prob = CoevoProblem(q, G, G, size(X, 2), leaf_node,
                        Matrix{Float64}(Y), Matrix{Float64}(X))
    return prob, sparse(Q)
end

"""
    make_coevo_problem_from_covariance(K, Y, X; group) -> (prob, Q)

Build a [`CoevoProblem`](@ref) from a known structured covariance/relatedness
matrix over observed levels. The returned precision is `K^-1`, kept sparse for
the shared exact-Gaussian coevolution fitter.
"""
function make_coevo_problem_from_covariance(K::AbstractMatrix,
                                            Y::AbstractMatrix,
                                            X::AbstractMatrix;
                                            group = 1:size(K, 1))
    G = LinearAlgebra.checksquare(K)
    C = Matrix{Float64}(K)
    isposdef(Symmetric(C)) || error("known structured covariance must be positive definite")
    Q = Matrix(inv(cholesky(Symmetric(C))))
    size(Q) == (G, G) || error("internal covariance inversion returned the wrong size")
    return make_coevo_problem_from_precision(Q, Y, X; group = group)
end

# ---------------------------------------------------------------------------
# Conjugate Gaussian Laplace marginal (EXACT — the leaf model is Gaussian).
#
#   joint nll(u) = 0.5 u' P u + 0.5 Σ_i (r_i' D⁻¹ r_i) + 0.5 n logdet(2π D),
#     r_i = y_i − X_i β − u_block(s(i)).
# H_uu = P + Σ_i e_{s(i)} e_{s(i)}' ⊗ D⁻¹  (constant in u). The mode solves
# H_uu û = rhs, rhs_block(t) = Σ_{i: s(i)=t} D⁻¹ (y_i − X_i β).
# Marginal:  ℓ(θ) = −min_u joint nll − 0.5 logdet H_uu + 0.5 logdet P.
# ---------------------------------------------------------------------------

# Assemble H_uu = P + blockdiag over leaf nodes of (#rows at node)·D⁻¹.
function coevo_Huu(prob::CoevoProblem, P::SparseMatrixCSC, Dinv::AbstractMatrix)
    q = prob.q
    H = copy(P)
    counts = zeros(Int, prob.N)
    @inbounds for i in eachindex(prob.leaf_node)
        counts[prob.leaf_node[i]] += 1
    end
    @inbounds for t in 1:prob.N
        c = counts[t]
        c == 0 && continue
        base = q * (t - 1)
        for a in 1:q, b in 1:q
            H[base + a, base + b] += c * Dinv[a, b]
        end
    end
    return H
end

# RHS for the mode: rhs_block(t) = Σ_{i at t} D⁻¹ (y_i − X_i β).
function coevo_rhs(prob::CoevoProblem, β::AbstractMatrix, Dinv::AbstractMatrix)
    q = prob.q
    rhs = zeros(q * prob.N)
    resid = prob.Y .- prob.X * β              # n × q
    @inbounds for i in eachindex(prob.leaf_node)
        base = q * (prob.leaf_node[i] - 1)
        ri = @view resid[i, :]
        for a in 1:q
            s = 0.0
            for b in 1:q
                s += Dinv[a, b] * ri[b]
            end
            rhs[base + a] += s
        end
    end
    return rhs
end

"""
    coevo_marginal(prob, Q_cond, β, Λ, σ_res) -> (ℓ, û, ch_H, P)

EXACT Laplace (= Gaussian) marginal log-likelihood at the given parameters,
plus the inner mode `û`, the CHOLMOD factor of `H_uu`, and the sparse prior `P`.
`β` is `k × q` (trait-major columns), `Λ` is q×q SPD, `σ_res` a length-q vector
of residual SDs. Rejected q2 covariance states return `-Inf` and `nothing` for
the factor and prior, which have not been constructed.
"""
function coevo_marginal_cov(prob::CoevoProblem, Q_cond::SparseMatrixCSC,
                            β::AbstractMatrix, Λ::AbstractMatrix, D::AbstractMatrix)
    q = prob.q
    n = length(prob.leaf_node)
    size(D) == (q, q) || error("residual covariance has size $(size(D)); expected ($q, $q)")
    # Log-Cholesky is SPD only in exact arithmetic. Apply the same numerical
    # admissibility boundary as q2 REML before inversion, including on the ML
    # optimizer, final evaluation, and observed-information paths. A successful
    # factorization of H alone does not certify the covariance that produced it.
    if q == 2 && !(_q2_lambda_admissible(Λ) && isposdef(Symmetric(Λ)) &&
                   all(isfinite, D) && isposdef(Symmetric(D)))
        return (-Inf, zeros(prob.q * prob.N), nothing, nothing)
    end
    isposdef(Symmetric(D)) || error("residual covariance must be positive definite")
    Dinv = inv(Symmetric(D))
    P = prior_precision(Q_cond, inv(Λ))
    H = coevo_Huu(prob, P, Dinv)
    # #503 (follow-up): this was `cholesky(Symmetric(H))`, unguarded, behind the
    # comment "PD: P + PSD data term, root-conditioned". PD by construction in
    # exact arithmetic -- but H inherits inv(Λ), and at extreme Λ it lands on the
    # definiteness boundary, where LAPACK detects the indefiniteness on some
    # builds and not others. Observed throwing PosDefException on CI's Julia
    # 1.12.7 while passing on 1.12.6 and on macOS 1.10.
    #
    # Guarded HERE, at the source, rather than at each caller: three separate
    # call sites reached this one factorisation, and guarding them one at a time
    # took two CI rounds to find the third. Note the SAME pattern is already
    # applied to `chP` twelve lines below, with its own explanation -- that guard
    # was written and this one was missed, which is exactly the shape of #503.
    chH = cholesky(Symmetric(H); check = false)
    issuccess(chH) || return (-Inf, zeros(size(H, 1)), chH, P)
    rhs = coevo_rhs(prob, β, Dinv)
    û = chH \ rhs                                 # conjugate mode (one solve)

    # joint nll at û
    resid = prob.Y .- prob.X * β
    quad_data = 0.0
    @inbounds for i in 1:n
        base = q * (prob.leaf_node[i] - 1)
        for a in 1:q
            ra = resid[i, a] - û[base + a]
            for b in 1:q
                rb = resid[i, b] - û[base + b]
                quad_data += 0.5 * ra * Dinv[a, b] * rb
            end
        end
    end
    quad_prior = 0.5 * dot(û, P * û)
    logdetD = logdet(Symmetric(D))
    jn = quad_prior + quad_data + 0.5 * n * (q * log(2π) + logdetD)

    logdetH = logdet(chH)
    chP = cholesky(Symmetric(P) + 1e-10I; check = false)
    # A failed factorisation leaves an incomplete factor whose `logdet` returns a
    # finite-but-wrong value; that poisons ℓ and `fit_coevolution`'s isfinite
    # guard lets it through. Treat non-PD as an explicit barrier instead.
    issuccess(chP) || return (-Inf, û, chH, P)
    logdetP = logdet(chP)
    ℓ = -jn - 0.5 * logdetH + 0.5 * logdetP
    return ℓ, û, chH, P
end

function coevo_marginal(prob::CoevoProblem, Q_cond::SparseMatrixCSC,
                        β::AbstractMatrix, Λ::AbstractMatrix, σ_res::AbstractVector)
    D = Diagonal(σ_res .^ 2)
    return coevo_marginal_cov(prob, Q_cond, β, Λ, D)
end

# ---------------------------------------------------------------------------
# θ pack / unpack and the fit driver.
# ---------------------------------------------------------------------------

coevo_theta_len(prob::CoevoProblem) = prob.k * prob.q + lc_len(prob.q) + prob.q

"Unpack θ → (β::k×q, Λ::q×q, σ_res::q). Eltype follows θ."
function coevo_unpack(prob::CoevoProblem, θ::AbstractVector{T}) where {T}
    q = prob.q; k = prob.k
    nβ = k * q
    β = reshape(θ[1:nβ], k, q)
    lc = θ[(nβ + 1):(nβ + lc_len(q))]
    logσ = θ[(nβ + lc_len(q) + 1):end]
    return β, lc_to_cov(lc, q), exp.(logσ)
end

"Pack (β::k×q, Λ::q×q, σ_res::q) → θ (Float64)."
coevo_pack(β::AbstractMatrix, Λ::AbstractMatrix, σ_res::AbstractVector) =
    vcat(vec(Float64.(β)), cov_to_lc(Λ), log.(Float64.(σ_res)))

"""
    fit_coevolution(prob, Q_cond; β0, Λ0, σ0, g_tol, iterations) -> NamedTuple

Fit the general-q coevolution model by maximising the EXACT (conjugate-Gaussian)
Laplace marginal over (β, log-Cholesky Λ, log σ_res). Uses `LBFGS` with a
central finite-difference gradient: the marginal is a single sparse solve, so FD
is cheap, and a quasi-Newton method converges far faster than a simplex on the
`k·q + q(q+1)/2 + q`-dimensional θ. ForwardDiff cannot flow through the CHOLMOD
factor (the same constraint the q=4 engine hits), hence FD rather than AD here.
Returns `(; β, Λ, σ_res, loglik, converged, iterations, θ)`.
"""
function fit_coevolution(prob::CoevoProblem, Q_cond::SparseMatrixCSC;
                         β0 = nothing, Λ0 = nothing, σ0 = nothing,
                         g_tol::Float64 = 1e-6, iterations::Int = 1000,
                         fd_h::Float64 = 1e-6)
    q = prob.q
    if β0 === nothing
        β0 = prob.X \ prob.Y                      # k × q OLS
    end
    if Λ0 === nothing
        Λ0 = Matrix(0.2 * I(q))
    end
    if σ0 === nothing
        σ0 = fill(0.5, q)
    end
    θ0 = coevo_pack(β0, Matrix(Λ0), σ0)
    n = length(prob.leaf_node)

    function negℓ(θ)
        local β, Λ, σ
        try
            β, Λ, σ = coevo_unpack(prob, Vector{Float64}(θ))
        catch
            return Inf
        end
        ℓ, = coevo_marginal(prob, Q_cond, β, Λ, σ)
        return isfinite(ℓ) ? -ℓ / n : Inf         # mean objective (scale-invariant in p)
    end

    # Central FD gradient of the cheap conjugate marginal (no CHOLMOD-through-AD).
    function g!(G, θ)
        @inbounds for k in eachindex(θ)
            tp = copy(θ); tp[k] += fd_h
            tm = copy(θ); tm[k] -= fd_h
            fp = negℓ(tp); fm = negℓ(tm)
            G[k] = (isfinite(fp) && isfinite(fm)) ? (fp - fm) / (2fd_h) : 0.0
        end
        return G
    end

    res = Optim.optimize(negℓ, g!, θ0,
                         Optim.LBFGS(linesearch = Optim.LineSearches.MoreThuente()),
                         Optim.Options(g_tol = g_tol, iterations = iterations,
                                       f_reltol = 1e-10, successive_f_tol = 2))
    θ̂ = Optim.minimizer(res)
    β̂, Λ̂, σ̂ = coevo_unpack(prob, θ̂)
    return (; β = β̂, Λ = Λ̂, σ_res = σ̂,
            loglik = -Optim.minimum(res) * n,
            converged = Optim.converged(res),
            iterations = Optim.iterations(res),
            θ = θ̂)
end

coevo_q2_residual_theta_len(prob::CoevoProblem) =
    prob.k * 2 + lc_len(2) + 3

function coevo_q2_residual_unpack(prob::CoevoProblem, θ::AbstractVector{T}) where {T}
    prob.q == 2 || error("q2 residual-correlation route requires q = 2")
    k = prob.k
    nβ = 2k
    β = reshape(θ[1:nβ], k, 2)
    lc = θ[(nβ + 1):(nβ + lc_len(2))]
    logσ1 = θ[nβ + lc_len(2) + 1]
    logσ2 = θ[nβ + lc_len(2) + 2]
    ηρ = θ[nβ + lc_len(2) + 3]
    σ1 = exp(logσ1)
    σ2 = exp(logσ2)
    ρ = RHO_GUARD * tanh(ηρ)
    D = Matrix(Symmetric([
        σ1^2      ρ * σ1 * σ2
        ρ * σ1 * σ2  σ2^2
    ]))
    return β, lc_to_cov(lc, 2), D, [σ1, σ2], ρ
end

function coevo_q2_residual_pack(β::AbstractMatrix, Λ::AbstractMatrix,
                                σ_res::AbstractVector, rho12::Real)
    length(σ_res) == 2 || error("q2 residual pack requires two residual SDs")
    ρ = clamp(Float64(rho12), -0.95, 0.95)
    ηρ = atanh(ρ / RHO_GUARD)
    return vcat(vec(Float64.(β)), cov_to_lc(Λ), log.(Float64.(σ_res)), ηρ)
end

"""
    fit_coevolution_q2_residual(prob, Q_cond; ...) -> NamedTuple

Fit the q=2 phylogenetic coevolution model with a bivariate residual covariance.
This is the same target as drmTMB's bivariate Gaussian q2 location route:
two phylogenetic random-effect axes (`mu1`, `mu2`) plus residual `rho12`.
It is ML-only and complete-response only at the caller level.
"""
function fit_coevolution_q2_residual(prob::CoevoProblem, Q_cond::SparseMatrixCSC;
                                     β0 = nothing, Λ0 = nothing,
                                     σ0 = nothing, rho0 = 0.0,
                                     g_tol::Float64 = 1e-6,
                                     iterations::Int = 1000,
                                     fd_h::Float64 = 1e-6)
    prob.q == 2 || throw(ArgumentError("fit_coevolution_q2_residual requires q = 2"))
    if β0 === nothing
        β0 = prob.X \ prob.Y
    end
    if Λ0 === nothing
        Λ0 = Matrix(0.2 * I(2))
    end
    if σ0 === nothing
        resid = prob.Y .- prob.X * β0
        σ0 = [std(resid[:, 1]) + eps(), std(resid[:, 2]) + eps()]
    end
    θ0 = coevo_q2_residual_pack(β0, Matrix(Λ0), σ0, rho0)
    n = length(prob.leaf_node)

    function negℓ(θ)
        local β, Λ, D, ℓ
        try
            β, Λ, D, _, _ = coevo_q2_residual_unpack(prob, Vector{Float64}(θ))
            ℓ, = coevo_marginal_cov(prob, Q_cond, β, Λ, D)
        catch
            return Inf
        end
        return isfinite(ℓ) ? -ℓ / n : Inf
    end

    function g!(G, θ)
        @inbounds for k in eachindex(θ)
            tp = copy(θ); tp[k] += fd_h
            tm = copy(θ); tm[k] -= fd_h
            fp = negℓ(tp); fm = negℓ(tm)
            G[k] = (isfinite(fp) && isfinite(fm)) ? (fp - fm) / (2fd_h) : 0.0
        end
        return G
    end

    res = Optim.optimize(negℓ, g!, θ0,
                         Optim.LBFGS(linesearch = Optim.LineSearches.MoreThuente()),
                         Optim.Options(g_tol = g_tol, iterations = iterations,
                                       f_reltol = 1e-10, successive_f_tol = 2))
    θ̂ = Optim.minimizer(res)
    β̂, Λ̂, D̂, σ̂, ρ̂ = coevo_q2_residual_unpack(prob, θ̂)
    # #503 (follow-up): `negℓ` above already catches a throw from
    # `coevo_marginal_cov` and returns Inf, but this FINAL evaluation at θ̂ was
    # unprotected -- and it does throw on CI's Julia 1.12.7, where LAPACK
    # detects an indefiniteness other builds miss. A fit that cannot evaluate
    # its own objective at the point it reports is not converged in any useful
    # sense, so say so rather than returning a NaN likelihood beside
    # `converged = true`.
    local ℓ, û, cholesky_ok
    try
        ℓ, û, _, _ = coevo_marginal_cov(prob, Q_cond, β̂, Λ̂, D̂)
        cholesky_ok = isfinite(ℓ)
    catch e
        (e isa DomainError || e isa LinearAlgebra.PosDefException ||
         e isa LinearAlgebra.SingularException) || rethrow(e)
        ℓ = NaN; û = zeros(prob.q * prob.N); cholesky_ok = false
    end
    return (; β = β̂, Λ = Λ̂, residual_cov = D̂, σ_res = σ̂, rho12 = ρ̂,
            loglik = ℓ,
            converged = cholesky_ok && Optim.converged(res),
            iterations = Optim.iterations(res),
            θ = θ̂,
            u_hat = û)
end

# ---------------------------------------------------------------------------
# Simulator (used by tests and examples): draw q-trait Brownian tip values from
# the SAME sparse precision the sampler in bench/run_scaling.jl uses, add Xβ and
# diagonal residual noise.
# ---------------------------------------------------------------------------

"""
    simulate_coevolution(phy, β, Λ, σ_res; nrep, rng) -> (; Y, X, species)

Simulate `nrep` observations per tip from the general-q coevolution model on
`phy`: tip random effects `vec(B) ~ N(0, Q_cond⁻¹ ⊗ Λ)` drawn via the sparse
Cholesky, plus `Xβ` and `N(0, diag(σ_res²))` residuals. `β` is `k × q`.
`X = [1  x]` with a single standard-normal covariate (k = 2).
"""
function simulate_coevolution(phy, β::AbstractMatrix, Λ::AbstractMatrix,
                              σ_res::AbstractVector; nrep::Integer = 5,
                              rng = Random.default_rng())
    q = size(Λ, 1)
    Q_cond, leaf_pos, N = augmented_tree_precision(phy)
    P = prior_precision(Q_cond, inv(Λ))
    F = cholesky(Symmetric(P))
    u_aug = F.UP \ randn(rng, size(P, 1))         # vec(B) ~ N(0, P⁻¹), axis-inner
    p = phy.n_leaves
    species = repeat(1:p, inner = nrep)
    n = length(species)
    x = randn(rng, n)
    X = hcat(ones(n), x)
    Y = Matrix{Float64}(undef, n, q)
    @inbounds for i in 1:n
        base = q * (leaf_pos[species[i]] - 1)
        for a in 1:q
            Y[i, a] = X[i, :]' * β[:, a] + u_aug[base + a] + σ_res[a] * randn(rng)
        end
    end
    return (; Y, X, species)
end
