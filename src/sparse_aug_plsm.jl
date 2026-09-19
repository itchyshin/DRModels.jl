# sparse_aug_plsm.jl — SPARSE AUGMENTED-STATE Laplace-EM for the q=4 PLSM.
#
# Why: the dense routes failed on BOTH speed (400-dim ForwardDiff Hessian /
# repeated dense E-steps) and stability (dense inv(Σ_phy) of an ultrametric
# tree covariance is ill-conditioned → NaN). The augmented-state sparse
# precision Q_topology (Hadfield–Nakagawa, O(p) nnz, well-conditioned) fixes
# both: never forms dense Σ_phy⁻¹, and sparse Cholesky is O(p).
#
# Latent: node-major over the 2p-1 augmented nodes × 4 axes (mu1,mu2,
# log σ1, log σ2). u[(t-1)*4 + a] = axis a at augmented node t.
# Prior precision: P = kron(Q_topology, Λ_phy⁻¹)  [node outer, axis inner],
#   sparse, O(p) nnz. Data (bivariate, nonlinear scale) attaches only at the
#   p LEAF nodes. H_uu = P + blockdiag(4×4 data Hessian at each leaf node).
#
# Run the p=8 checkpoint:
#   cd /Users/z3437171/Dropbox/Github Local/drm-julia-poc/julia/drm_q4
#   /Users/z3437171/.juliaup/bin/julia --project=.. sparse_aug_plsm.jl

using LinearAlgebra, SparseArrays, ForwardDiff, Statistics, Printf
include(joinpath(@__DIR__, "sparse_phy.jl"))
include(joinpath(@__DIR__, "takahashi_selinv.jl"))

const RHO_GUARD = 0.99999999

# --- per-leaf bivariate-Gaussian data NLL as a function of the 4-vector u ----
# η_* carry the fixed-effect part (Xβ); u shifts mean (axes 1,2) and log-σ
# (axes 3,4). ρ has no random effect (rho12 ~ 1).
# Per-cell observed flags o1, o2 (Bool) support missing responses (issue #19): both
# observed ⇒ the bivariate term; exactly one observed ⇒ that response's univariate
# Gaussian marginal (the correct observed-data likelihood — drop the missing dim, ρ
# and the other σ leave the term); neither ⇒ 0 (the tip still couples through the
# tree prior, which is y-independent). The flags are CONSTANTS w.r.t. u, so leaf_grad
# / leaf_hess (and the exact gradient's ForwardDiff of leaf_hess) auto-propagate the
# mask. Defaulting o1=o2=true reproduces the original term BIT-FOR-BIT (all-observed
# reduction). The missing response value is NEVER touched in its dropped branch, so a
# NaN placeholder there is safe.
function leaf_nll(u, y1, y2, η1, η2, ηs1, ηs2, ηr, o1::Bool = true, o2::Bool = true)
    if o1 & o2
        mu1 = η1 + u[1]; mu2 = η2 + u[2]
        s1 = exp(ηs1 + u[3]); s2 = exp(ηs2 + u[4])
        ρ = RHO_GUARD * tanh(ηr)
        e1 = y1 - mu1; e2 = y2 - mu2
        omr2 = 1 - ρ^2
        quad = (e1^2 / s1^2 - 2ρ * e1 * e2 / (s1 * s2) + e2^2 / s2^2) / omr2
        return 0.5 * (log(s1^2 * s2^2 * omr2) + quad) + log(2π)
    elseif o1
        ls1 = ηs1 + u[3]; e1 = y1 - (η1 + u[1])      # univariate N(mu1, σ1²) marginal
        return 0.5 * (2 * ls1 + e1^2 / exp(2 * ls1)) + 0.5 * log(2π)
    elseif o2
        ls2 = ηs2 + u[4]; e2 = y2 - (η2 + u[2])      # univariate N(mu2, σ2²) marginal
        return 0.5 * (2 * ls2 + e2^2 / exp(2 * ls2)) + 0.5 * log(2π)
    else
        return zero(eltype(u))                        # neither observed
    end
end
leaf_grad(u, a...) = ForwardDiff.gradient(z -> leaf_nll(z, a...), u)
leaf_hess(u, a...) = ForwardDiff.hessian(z -> leaf_nll(z, a...), u)

# --- problem container -------------------------------------------------------
struct AugProblem
    phy::AugmentedPhy{Float64}
    n_total::Int                 # 2p-1
    p::Int                       # leaves
    leaf_node::Vector{Int}       # data row i -> augmented node index
    y1::Vector{Float64}; y2::Vector{Float64}
    X1::Matrix{Float64}; X2::Matrix{Float64}     # mu1, mu2 design (n×k)
    Xs1::Matrix{Float64}; Xs2::Matrix{Float64}; Xr::Matrix{Float64}
    obs1::Vector{Bool}; obs2::Vector{Bool}       # #19: per-row observed-response masks
end

# Backward-compatible constructor: no masks ⇒ all responses observed (the original
# all-leaf bivariate behaviour, bit-for-bit). Keeps every existing make_problem /
# AugProblem(...) caller working unchanged.
AugProblem(phy, n_total, p, leaf_node, y1, y2, X1, X2, Xs1, Xs2, Xr) =
    AugProblem(phy, n_total, p, leaf_node, y1, y2, X1, X2, Xs1, Xs2, Xr,
               trues(length(y1)), trues(length(y2)))

# Build the sparse prior precision P = kron(Q_topology, Λ⁻¹) (node-major), with a
# STRUCTURALLY FULL axis block.
#
# #577, and the REML half of it, #575. The obvious spelling,
# `kron(Q, sparse(Λinv))`, routes through `sparse`, which DROPS EXACT ZEROS. At an
# exactly diagonal Λ — including the engines' own default warm start `Λ0 = 0.3I` —
# that leaves the cross-axis entries of `H_uu` structurally ABSENT at non-leaf
# nodes, which carry no data block to supply them. The CHOLMOD pattern, and with
# it the Takahashi selected inverse, is then missing entries the logdet-H traces
# genuinely need: structurally absent, not zero in H⁻¹. Every exact-gradient path
# that feeds P into `takahashi_selinv` is silently wrong in those components.
#
# Measured on biv-q4-phylo-reml at Λ = 0.3I: two lc gradient components off by
# 0.751 and 12.1 on the ML path (`marginal_and_exact_grad`), and by 0.037 and
# 13.70 on the REML path; both collapse to the central-difference reference's own
# ~3e-5 noise floor the moment the block is stored in full.
#
# Storing every entry is numerically free — the matrix is identical, only its
# stored pattern changes — and it closes the whole class at the root instead of
# one call site at a time.
#
# #579 had applied the same fix locally on the REML path (`_reml_prior_precision`
# in reml_q4.jl). Once this root fix landed the helper was proven redundant —
# bit-identical pattern and values from both builders on the q4 fixture, nnz
# 1376 at a diagonal and at a fitted Λ — and collapsed (#563 follow-up); the
# REML path now calls this function directly.
function prior_precision(Q::SparseMatrixCSC, Λinv::AbstractMatrix)
    q = LinearAlgebra.checksquare(Λinv)
    rows = [a for _ in 1:q for a in 1:q]
    cols = [b for b in 1:q for _ in 1:q]
    vals = [Λinv[a, b] for b in 1:q for a in 1:q]
    return kron(Q, sparse(rows, cols, vals, q, q))
end

# η's from β (Float64) for the p leaves.
function leaf_etas(prob::AugProblem, β)
    (prob.X1 * β.mu1, prob.X2 * β.mu2, prob.Xs1 * β.s1, prob.Xs2 * β.s2, prob.Xr * β.rho)
end

# Assemble H_uu = P + blockdiag(leaf 4×4 data Hessians) at a given u (sparse).
function build_Huu(prob::AugProblem, P::SparseMatrixCSC, u::Vector{Float64}, β)
    η1, η2, ηs1, ηs2, ηr = leaf_etas(prob, β)
    H = copy(P)
    @inbounds for i in eachindex(prob.leaf_node)   # over DATA ROWS (≥1 per leaf)
        t = prob.leaf_node[i]; base = 4(t - 1)
        ublk = (u[base+1], u[base+2], u[base+3], u[base+4])
        Hb = leaf_hess([ublk...], prob.y1[i], prob.y2[i], η1[i], η2[i], ηs1[i], ηs2[i], ηr[i],
                       prob.obs1[i], prob.obs2[i])
        for a in 1:4, b in 1:4
            H[base+a, base+b] += Hb[a, b]
        end
    end
    return H
end

# Zero-allocation prior-coupling gradient term: g .= P*u via an in-place sparse
# matvec (no temporary). This is the pure-Julia arithmetic the inner Newton loop
# repeats every iteration; the engine-quality gate (#15) asserts it allocates
# nothing and stays flat across p (the CHOLMOD factor/solve is excluded from that
# gate as out-of-Julia-control). `g` must be a length-4·n_total preallocated buffer.
@inline function aug_prior_grad!(g::Vector{Float64}, P::SparseMatrixCSC, u::Vector{Float64})
    mul!(g, P, u)
    return g
end

# Gradient of the joint nll wrt u at u: P*u + leaf data gradients (at leaves).
function joint_grad(prob::AugProblem, P::SparseMatrixCSC, u::Vector{Float64}, β)
    η1, η2, ηs1, ηs2, ηr = leaf_etas(prob, β)
    g = similar(u)
    aug_prior_grad!(g, P, u)                        # g .= P*u (zero-alloc matvec)
    @inbounds for i in eachindex(prob.leaf_node)   # over DATA ROWS (≥1 per leaf)
        t = prob.leaf_node[i]; base = 4(t - 1)
        ublk = [u[base+1], u[base+2], u[base+3], u[base+4]]
        gb = leaf_grad(ublk, prob.y1[i], prob.y2[i], η1[i], η2[i], ηs1[i], ηs2[i], ηr[i],
                       prob.obs1[i], prob.obs2[i])
        for a in 1:4
            g[base+a] += gb[a]
        end
    end
    return g
end

# joint nll value at u (data at leaves + 0.5 u'Pu).
function joint_nll(prob::AugProblem, P::SparseMatrixCSC, u::Vector{Float64}, β)
    η1, η2, ηs1, ηs2, ηr = leaf_etas(prob, β)
    val = 0.5 * dot(u, P * u)
    @inbounds for i in eachindex(prob.leaf_node)   # over DATA ROWS (≥1 per leaf)
        t = prob.leaf_node[i]; base = 4(t - 1)
        val += leaf_nll((u[base+1], u[base+2], u[base+3], u[base+4]),
                        prob.y1[i], prob.y2[i], η1[i], η2[i], ηs1[i], ηs2[i], ηr[i],
                        prob.obs1[i], prob.obs2[i])
    end
    return val
end

# --- S5 change (b): opt-in cholesky! symbolic reuse ---------------------------
# Observational-only counters (never affect a result) read by the leaf-S5 G5.4
# gate (bench/profile_q4_sections.jl --gate fallback) and by
# test/test_q4_perf_identities.jl. `CHOL_FACTORIZATIONS` counts every
# factorisation sparse_pd_chol performs (reuse or fresh); `CHOL_REUSE_FALLBACKS`
# counts times the cholesky!-reuse path threw and a fresh cholesky was taken
# instead (a dropped structural zero would show up here as a nonzero count).
const CHOL_FACTORIZATIONS = Ref(0)
const CHOL_REUSE_FALLBACKS = Ref(0)
reset_chol_diagnostics!() = (CHOL_FACTORIZATIONS[] = 0; CHOL_REUSE_FALLBACKS[] = 0; nothing)

"""
    CholPatternCache()

Pattern carrier + factor cache for `sparse_pd_chol`'s opt-in cholesky!-reuse
path (S5 change (b)). One instance per FIT (created once by `fit_q4_sparse_tmb`
and threaded through every `marginal_and_exact_grad`/`estep_mode` call of that
fit): the sparsity pattern of `H_uu` is fixed for a given `(prob, Q_cond)` (see
`prior_precision`'s "structurally full axis block" — the pattern never depends
on Λ's or u's actual VALUES), so the CHOLMOD symbolic analysis is done ONCE on
first use and every later call is a numeric-only `cholesky!` refactorisation.
Mirrors the repo's own `chol_ref` idiom (gaussian_structured.jl's `eval_all`,
gaussian_sparse_lss.jl's `eval_core`): if the in-place update ever rejects the
pattern, `sparse_pd_chol` falls back to a fresh `cholesky` (still tree-sparse
O(p)) — correctness never depends on the reuse succeeding, only speed does.

`pattern_colptr`/`pattern_rowval` (S5d item 2) record the FIRST `Hr`'s
pattern and back [`_assert_chol_pattern_matches`](@ref)'s reuse-time check —
`hzero` does NOT serve this role despite its docstring (see
`_assert_chol_pattern_matches`'s own note: `0.0 .* Hr` is measured to be the
EMPTY sparse matrix, so it is not a usable pattern reference).
"""
mutable struct CholPatternCache
    factor::Any   # ::SparseArrays.CHOLMOD.Factor{Float64} once initialised
    hzero::Any    # ::SparseMatrixCSC{Float64,Int} pattern carrier (all-structural-zero)
    pattern_colptr::Any   # ::Vector{Int} the FIRST Hr's colptr, recorded at cache creation
    pattern_rowval::Any   # ::Vector{Int} the FIRST Hr's rowval, recorded at cache creation
end
CholPatternCache() = CholPatternCache(nothing, nothing, nothing, nothing)

"""
    CholPatternMismatch(msg)

Thrown by [`_assert_chol_pattern_matches`](@ref) when a candidate matrix's
sparsity pattern no longer matches the pattern a `CholPatternCache`'s factor
was built for (S9 audit, Q2: `cholesky!` does NOT throw on a pattern change —
it silently reuses the stale symbolic factorisation — so this assertion is
the only thing standing between a pattern change and a silently wrong
answer). `_chol_factorize`'s existing `catch` converts this into a fresh
`cholesky(Symmetric(...))` and counts it in `CHOL_REUSE_FALLBACKS`; the type
exists so that fallback is a PROVABLE consequence of a caught, named
condition (see `test/test_q4_perf_identities.jl`'s G5d.4), not an
unconditional catch-all guess.
"""
struct CholPatternMismatch <: Exception
    msg::String
end
Base.showerror(io::IO, e::CholPatternMismatch) = print(io, "CholPatternMismatch: ", e.msg)

"""
Assert `Hf`'s sparsity pattern (colptr, rowval -- nnz follows from `colptr`)
is IDENTICAL to the pattern recorded in `chol_ref` at first use
(`pattern_colptr`/`pattern_rowval`, set once in `_chol_factorize`'s
cache-creation branch). Throws `CholPatternMismatch` on any mismatch; called
on every `cholesky!`-reuse attempt in `_chol_factorize`, BEFORE `cholesky!`
itself, because `cholesky!` does not throw on a pattern change (S9 audit,
Q2).

Deliberately does NOT use `chol_ref.hzero` as the reference: `hzero = 0.0 .*
Hr` (the pre-existing S5b "pattern carrier") is, empirically, the EMPTY
sparse matrix -- Julia's sparse broadcast drops an all-exact-zero result
rather than keeping `Hr`'s structural pattern with zeroed values (MEASURED:
`nnz(0.0 .* Hr) == 0` for the real q4 `H_uu`). `Hr + chol_ref.hzero` is
therefore a no-op (`Hf === ` structurally `Hr`), so `hzero` cannot serve as a
stable reference pattern -- comparing against it would flag a mismatch on
EVERY reuse call, not just a genuine pattern change (confirmed: naively
wired this way, a real p=1000 fit showed 214/215 "mismatches"). This is a
pre-existing S5b latency, not introduced here; left as-is (out of this
leaf's scope) since `Hr`'s OWN pattern is independently stable by
construction (`prior_precision`'s fully-dense diagonal blocks -- MEASURED:
`build_Huu`/`build_Huu_expected`/`H+λ*I` all reproduce the SAME pattern as
`P` regardless of `u` or the ridge value), so `hzero`'s intended
belt-and-suspenders union was never load-bearing for correctness. Flagged
for a follow-up, not fixed here.
"""
function _assert_chol_pattern_matches(Hf::SparseMatrixCSC, chol_ref::CholPatternCache)
    if Hf.colptr != chol_ref.pattern_colptr || Hf.rowval != chol_ref.pattern_rowval
        throw(CholPatternMismatch(
            "sparse_pd_chol: cached sparsity pattern changed on reuse " *
            "(nnz $(length(chol_ref.pattern_rowval)) -> $(nnz(Hf))) -- falling back to a fresh cholesky"))
    end
    return nothing
end

"Add `ridge` to every diagonal entry of a COPY of `H`, via direct `nzval`
mutation at a freshly-scanned diagonal index map rather than the generic
sparse `H + ridge*I` (which would allocate a fresh union-of-patterns result).
`H`'s diagonal is always structurally present (see `prior_precision`'s
full-axis-block comment), so this never changes `H`'s sparsity pattern; `H`
itself is never mutated."
function _add_diag(H::SparseMatrixCSC, ridge::Real)
    Hr = SparseMatrixCSC(H.m, H.n, copy(H.colptr), copy(H.rowval), copy(H.nzval))
    @inbounds for j in 1:Hr.n
        found = false
        for k in nzrange(Hr, j)
            if Hr.rowval[k] == j
                Hr.nzval[k] += ridge
                found = true
                break
            end
        end
        found || error("sparse_pd_chol: diagonal entry ($j,$j) not structurally present in H -- the cholesky!-reuse pattern assumption is violated")
    end
    return Hr
end

# One factorisation attempt at a given ridge, either fresh (chol_ref===nothing,
# byte-identical to the pre-S5 code) or via the pattern cache.
function _chol_factorize(H::SparseMatrixCSC, ridge::Real, chol_ref::Nothing)
    Hs = ridge == 0.0 ? Symmetric(H) : Symmetric(H + ridge * I)
    return cholesky(Hs; check = false)
end
function _chol_factorize(H::SparseMatrixCSC, ridge::Real, chol_ref::CholPatternCache)
    Hr = ridge == 0.0 ? H : _add_diag(H, ridge)
    if chol_ref.factor === nothing
        chol_ref.hzero = 0.0 .* Hr
        chol_ref.pattern_colptr = copy(Hr.colptr)
        chol_ref.pattern_rowval = copy(Hr.rowval)
        ch = cholesky(Symmetric(Hr); check = false)
        chol_ref.factor = ch
        CHOL_FACTORIZATIONS[] += 1
        return ch
    end
    Hf = Hr + chol_ref.hzero
    try
        # S9 audit (Q2): a sparsity-pattern GROWTH into `cholesky!` does not
        # throw and returns a wrong-but-"successful" factorisation -- G5.4's
        # "0 fallbacks" alone cannot prove the pattern held. Assert it
        # explicitly, on every reuse attempt, before `cholesky!` ever runs.
        _assert_chol_pattern_matches(Hf, chol_ref)
        # `Hf` stores BOTH triangles explicitly (kron/leaf-block accumulation
        # never produces a one-triangle-only sparse matrix here, unlike the
        # ZtWZ-built templates in gaussian_structured.jl/gaussian_sparse_lss.jl,
        # whose bare-H `cholesky!` call this idiom otherwise mirrors). Passing
        # the bare two-triangle `Hf` to `cholesky!` does NOT throw but silently
        # double-counts off-diagonal contributions (measured: logdet inflated
        # ~2x, a real fit-corrupting bug caught only by G5.1/G5.5's numeric
        # identity checks, not by `issuccess`/G5.4's fallback counter -- CHOLMOD
        # reports success on the wrong answer). `Symmetric(Hf)` (upper triangle,
        # matching the ORIGINAL `cholesky(Symmetric(...))` analysis call) fixes
        # it exactly (verified to rtol 1e-10 against a fresh factorisation).
        cholesky!(chol_ref.factor, Symmetric(Hf); check = false)
        CHOL_FACTORIZATIONS[] += 1
        return chol_ref.factor
    catch
        CHOL_REUSE_FALLBACKS[] += 1
        ch = cholesky(Symmetric(Hf); check = false)
        CHOL_FACTORIZATIONS[] += 1
        return ch
    end
end

# Sparse PD-safe Cholesky: escalate a ridge until CHOLMOD succeeds. (H_uu is
# PD at the mode, but indefinite far from it — the log-σ axes have negative
# curvature when residuals are small, and Q_topology is rank-deficient.)
#
# `chol_ref`: nothing (default) reproduces the ORIGINAL fresh-cholesky-every-
# call behaviour byte-for-byte -- every caller that does not pass `chol_ref`
# (reml_q4.jl, src/experimental/*, and every route other than the q=4 ML
# phylo fit) is completely unaffected by S5 change (b). Passing a
# `CholPatternCache` opts a caller into the reuse path.
function sparse_pd_chol(H::SparseMatrixCSC; chol_ref::Union{Nothing,CholPatternCache} = nothing)
    # Non-finite guard. `cholesky(...; check=false)` suppresses the *not-PD*
    # exception but STILL throws `ArgumentError("matrix contains Infs or NaNs")`
    # on non-finite input. During optimisation a trial θ (e.g. the line search
    # probing the σ→0 collapse region) can hand `estep_mode` a Hessian with
    # Inf/NaN entries; without this guard that ArgumentError escapes `estep_mode`
    # BEFORE the caller's `any(!isfinite, g)` objective guard runs, crashing the
    # fit instead of the step being rejected. Substitute a finite PD surrogate so
    # the downstream marginal/gradient goes non-finite and the objective rejects
    # the step cleanly. (Tightening the fast-path acceptance gate in #317 routes
    # more warm E-steps through here, so this path must not throw.)
    if !all(isfinite, nonzeros(H))
        nu = size(H, 1)
        return cholesky(sparse(1.0I, nu, nu)), Inf   # finite surrogate; Inf flags failure
    end
    ch = _chol_factorize(H, 0.0, chol_ref)
    issuccess(ch) && return ch, 0.0
    λ = 1e-10
    for _ in 1:40                      # escalate up to ~1e30 — always succeeds
        ch = _chol_factorize(H, λ, chol_ref)
        issuccess(ch) && return ch, λ
        λ *= 10
    end
    # guaranteed-PD last resort: diagonally dominant ridge
    d = maximum(abs, diag(H)) + 1.0
    return _chol_factorize(H, d, chol_ref), d
end

# --- EXPECTED-information (Fisher) leaf 4×4 block (merged from the workflow's
# estep_lm). E[leaf_hess] over e ~ N(0,Σ(u)); leaf_hess is degree-≤2 in e so the
# 4-point symmetric sigma rule is EXACT. PD always (unlike the OBSERVED Hessian,
# whose log-σ curvature collapses at small residual → indefinite far from mode).
@inline function leaf_fisher(ublk, η1, η2, ηs1, ηs2, ηr)
    s1 = exp(ηs1 + ublk[3]); s2 = exp(ηs2 + ublk[4]); ρ = RHO_GUARD * tanh(ηr)
    L11 = s1; L21 = ρ * s2; L22 = s2 * sqrt(max(1 - ρ^2, 0.0)); c = sqrt(2.0)
    mu1 = η1 + ublk[1]; mu2 = η2 + ublk[2]
    HF = zeros(4, 4)
    for (a1, a2) in ((c*L11, c*L21), (-c*L11, -c*L21), (0.0, c*L22), (0.0, -c*L22))
        Hb = leaf_hess([ublk...], mu1 + a1, mu2 + a2, η1, η2, ηs1, ηs2, ηr)
        @inbounds for j in 1:4, i in 1:4; HF[i, j] += Hb[i, j]; end
    end
    HF ./= 4; return HF
end

# Expected-information joint Hessian H_E = P + blockdiag(Fisher leaf blocks); PD
# everywhere, used to STEER far from the mode. Fixed point unchanged (the step
# uses the TRUE gradient ∇J, so ∇J=0 at any solution — the same mode).
function build_Huu_expected(prob::AugProblem, P::SparseMatrixCSC, u::Vector{Float64}, β)
    η1, η2, ηs1, ηs2, ηr = leaf_etas(prob, β)
    H = copy(P)
    @inbounds for i in eachindex(prob.leaf_node)   # over DATA ROWS (≥1 per leaf)
        t = prob.leaf_node[i]; base = 4(t - 1)
        ublk = (u[base+1], u[base+2], u[base+3], u[base+4])
        Hb = leaf_fisher(ublk, η1[i], η2[i], ηs1[i], ηs2[i], ηr[i])
        for a in 1:4, b in 1:4; H[base+a, base+b] += Hb[a, b]; end
    end
    return H
end

# --- FAST PATH: cheap damped OBSERVED-Newton from a WARM start ----------------
# The old (pre-robust) mode-finder. For a warm u0 already near the mode this
# converges in 1–2 steps with NO Fisher overhead (observed build_Huu only) —
# the bulk of a fit's E-steps. Step = sparse_pd_chol(H)\g (the PD ridge guards
# the occasional indefinite observed H); backtracking line search on joint_nll.
#
# Acceptance (ok=true): the observed-Newton step direction hits a curvature floor
# near the mode and the line search can no longer strictly decrease joint_nll
# while ‖∇J‖ is at the mode. The FROZEN-mode Laplace marginal assumes ∇_u J = 0
# EXACTLY: a residual gradient ~ε leaves an O(ε) error in the mode that biases
# ½ logdet H and the joint term (H is evaluated off the mode). The phylo/crossed
# FG paths drive the inner mode to ~1e-8 for exactly this reason, so accepting a
# loose 1e-3 stall here would feed an off-mode b̂ into the marginal (#317).
# We therefore ACCEPT only a genuinely converged step: a clean convergence
# (‖∇J‖<ftol) OR a line-search stall once ‖∇J‖<stall_tol with stall_tol=1e-6.
# We FALL BACK (to the robust LM, which handles the indefinite log-σ curvature)
# on any of: non-finite f, a non-finite step, a |u| blow-up, or a stall while
# ‖∇J‖≥stall_tol — the hard surface the robust LM exists for.
function _estep_fast(prob::AugProblem, P::SparseMatrixCSC, β, u0::Vector{Float64};
                     n_newton=40, ftol=1e-6, stall_tol=1e-6, ucap=1e3,
                     chol_ref::Union{Nothing,CholPatternCache} = nothing)
    u = copy(u0)
    f = joint_nll(prob, P, u, β)
    isfinite(f) || return u, false
    g = joint_grad(prob, P, u, β); ng = norm(g)
    for _ in 1:n_newton
        ng < ftol && return u, true
        H = build_Huu(prob, P, u, β)
        ch, _ = sparse_pd_chol(H; chol_ref = chol_ref)
        step = ch \ g
        all(isfinite, step) || return u, false       # bad step → fall back
        α = 1.0
        unew = u .- α .* step; fnew = joint_nll(prob, P, unew, β); nbt = 0
        while !(isfinite(fnew) && fnew < f) && nbt < 30
            α *= 0.5; unew = u .- α .* step; fnew = joint_nll(prob, P, unew, β); nbt += 1
        end
        if !(isfinite(fnew) && fnew < f)             # line search stalled
            return u, ng < stall_tol                 # accept if at the curvature floor
        end
        u = unew; f = fnew
        maximum(abs, u) > ucap && return u, false          # |u| blew up
        g = joint_grad(prob, P, u, β); ng = norm(g)
    end
    return u, ng < stall_tol       # ran out of iters: accept only if near the mode
end

# --- ROBUST PATH: trust-region Levenberg–Marquardt on a HYBRID Hessian (merged
# from the workflow winner estep_lm — strictly dominates the old damped Newton:
# no NaN, no false convergence, robust to the indefinite observed Hessian far
# from the mode). EXPECTED-info Hessian steers while far (‖∇J‖≥gswitch, PD-
# everywhere, anti-runaway); OBSERVED build_Huu once close (<gswitch) for
# quadratic convergence. Trust region caps step ‖·‖∞ (anti σ→0 cascade);
# convergence on ‖∇J‖ ONLY (the old norm(α·step)<tol test masked the drift).
# `u0` warm-starts (converges in 1-2 steps); cold starts get ≥200 iters.
function _estep_robust(prob::AugProblem, P::SparseMatrixCSC, β;
                       u0=nothing, n_newton=40, tol=1e-8, trust=5.0, gswitch=1.0,
                       chol_ref::Union{Nothing,CholPatternCache} = nothing)
    nu = 4 * prob.n_total
    u = u0 === nothing ? zeros(nu) : copy(u0)
    nit = u0 === nothing ? max(n_newton, 200) : n_newton    # cold needs more iters
    f = joint_nll(prob, P, u, β)
    g = joint_grad(prob, P, u, β); ng = norm(g)
    H = ng < gswitch ? build_Huu(prob, P, u, β) : build_Huu_expected(prob, P, u, β)
    λ = 1e-2 * mean(abs.(diag(H))); λ = (isfinite(λ) && λ > 0) ? λ : 1.0
    λmax = 1e14
    for _ in 1:nit
        ng < tol && break
        ch_try, extra = sparse_pd_chol(H + λ * I; chol_ref = chol_ref)
        if extra > 0; λ = min(λmax, max(λ, λ + extra)); ch_try, _ = sparse_pd_chol(H + λ * I; chol_ref = chol_ref); end
        step = ch_try \ g
        sc = min(1.0, trust / max(maximum(abs, step), eps())); α = sc
        unew = u .- α .* step; fnew = joint_nll(prob, P, unew, β); nbt = 0
        while !(isfinite(fnew) && fnew < f) && nbt < 60
            α *= 0.5; unew = u .- α .* step; fnew = joint_nll(prob, P, unew, β); nbt += 1
        end
        if isfinite(fnew) && fnew < f
            u = unew; f = fnew
            g = joint_grad(prob, P, u, β); ng = norm(g)
            H = ng < gswitch ? build_Huu(prob, P, u, β) : build_Huu_expected(prob, P, u, β)
            λ = max(1e-12, λ * 0.5)
        else
            λ *= 4.0; λ > λmax && break
        end
    end
    Hobs = build_Huu(prob, P, u, β)
    ch, _ = sparse_pd_chol(Hobs; chol_ref = chol_ref)
    return u, ch, Hobs
end

# E-step dispatcher. WARM start (u0 given): try the cheap fast path first; only
# if it stalls/diverges fall back to the robust LM (warm-started from u0). COLD
# start (u0 === nothing): straight to the robust LM. SAME return contract
# (û, factor, H) with factor/H = OBSERVED Hessian at û (Laplace needs it).
#
# `chol_ref`: nothing (default) is IDENTICAL to pre-S5 behaviour for every
# existing caller (fit_ml_q4.jl, reml_q4.jl, sparse_em_fit.jl, src/experimental/*
# all call estep_mode without it). Only fit_q4_sparse_tmb's ML fg! loop passes a
# CholPatternCache, created once per fit (S5 change (b)).
function estep_mode(prob::AugProblem, P::SparseMatrixCSC, β;
                    u0=nothing, n_newton=40, tol=1e-8, trust=5.0, gswitch=1.0,
                    chol_ref::Union{Nothing,CholPatternCache} = nothing)
    if u0 !== nothing
        u, ok = _estep_fast(prob, P, β, Vector{Float64}(u0); n_newton=n_newton, chol_ref = chol_ref)
        if ok
            Hobs = build_Huu(prob, P, u, β)
            ch, _ = sparse_pd_chol(Hobs; chol_ref = chol_ref)
            return u, ch, Hobs
        end
    end
    return _estep_robust(prob, P, β; u0=u0, n_newton=n_newton, tol=tol,
                         trust=trust, gswitch=gswitch, chol_ref = chol_ref)
end

# Laplace marginal log-likelihood. Prior precision P = kron(Q_cond, Λ⁻¹) is
# positive DEFINITE by construction (make_problem drops the root row/col so Q_cond
# is PD and Λ⁻¹ is PD). The additive 1e-10 ridge is retained: it is load-bearing
# for optimiser stability — laplace_ll is evaluated at TRIAL Λ during the q=4 fit,
# some barely PD, and the ridge keeps logdetP finite there so the unguarded
# inv(Λ) in marginal_and_exact_grad (fit_q4_sparse_tmb.jl) never sees a poisoned
# step. Removing the ridge (issue #324.6) perturbs the optimiser trajectory enough
# to trip that pre-existing instability on tiny/ill-conditioned data; the bias it
# introduces is ~q·log(1+1e-10/λ_min) ≈ 1e-8, negligible. Left as-is; see PR note.
function laplace_ll(prob::AugProblem, P::SparseMatrixCSC, β, u, ch_H)
    nu = 4 * prob.n_total
    jn = joint_nll(prob, P, u, β)
    # Non-finite guard: at an extreme trial θ (huge/near-singular Λ ⇒ non-finite
    # prior precision, or an overflowing mode) `jn` is non-finite and the
    # `cholesky(Symmetric(P) + 1e-10I; check=false)` below would THROW
    # `ArgumentError("matrix contains Infs or NaNs")` (check=false suppresses only
    # the not-PD path). Return -Inf so the caller's `nll = -laplace_ll` is +Inf and
    # the optimiser rejects the step, instead of crashing the fit.
    (isfinite(jn) && all(isfinite, nonzeros(P))) || return -Inf
    logdetH = logdet(ch_H)
    # logdet of prior precision (ridge keeps the trial-Λ path finite; see above)
    chP = cholesky(Symmetric(P) + 1e-10I; check=false)
    logdetP = logdet(chP)
    # Laplace: ll = -jn - 0.5 logdetH + 0.5 logdetP + 0.5*nu*log(2π) - 0.5*nu*log(2π)
    #            = -jn - 0.5 logdetH + 0.5 logdetP
    return -jn - 0.5 * logdetH + 0.5 * logdetP
end
