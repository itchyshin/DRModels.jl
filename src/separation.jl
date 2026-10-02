# separation.jl — (quasi-)complete separation screen for Bernoulli / Binomial
# fixed effects (#731, #728; twin contract drmTMB #1268, R/separation.R).
#
# Policy (owner decision 2026-09-30, "detect and warn"): the fit is still
# returned, a warning names the affected coefficients, and their standard errors
# are reported as Inf (so `stderror`, `coeftable` and Wald `confint` all report
# Inf / (-Inf, Inf)). No refusal and no penalised (Firth / MSPL) estimator by
# default. The twins agree on WHICH coefficients are flagged and on the warning,
# not on the (non-existent) maximum-likelihood coefficients.
#
# Method (the separation criterion of Albert & Anderson 1984 / Konis 2007,
# implemented independently from the published mathematics). With signed design
# rows aᵢ = xᵢ for a success and −xᵢ for a failure, the likelihood has no finite
# maximiser iff the cone C = {d : A d ≥ 0} holds a direction with aᵢ·d > 0 for
# some row: moving along d never lowers the likelihood.
#
# Algorithm (#910 review: replaces a Bland-rule simplex that stalled on
# degenerate designs). By Farkas' lemma, for a row set U either −Σ_{i∈U} aᵢ lies
# in cone(rows), or some d ∈ C has Σ_U aᵢ·d > 0. A Lawson–Hanson non-negative
# least-squares solve of min_{y ≥ 0} ‖Aᵀy + Σ_U aᵢ‖ decides which: its residual
# r is zero in the first case, and otherwise r itself lies in C with
# Σ_U aᵢ·r = ‖r‖². Starting from U = all rows, every non-zero residual moves the
# rows it separates out of U; when the residual vanishes the remaining rows J
# are tied (aᵢ·d = 0 for every d ∈ C), and C spans the null space of A_J.
# Coefficient j is flagged iff that null space has a non-zero j-th coordinate,
# i.e. iff SOME direction in C moves βⱼ. (R's `detectseparation` reports the
# components of one particular LP direction instead, so the two can differ in
# which coefficients they list; no parity with it is claimed.) Typically one or
# two NNLS solves; each is bounded by an iteration budget, and an exhausted
# budget is reported (`conclusive = false`) and warned about, never read as "no
# separation".
#
# Scope: the fixed-effect design of a Binomial/Bernoulli mean (the FE-only
# fit). Random-intercept binomial routes are NOT screened here: a random
# intercept can absorb part of the separation, so the fixed-design check does
# not decide it.

# Numbers shared with the drmTMB twin (keep identical across both packages).
const _SEP_OBJ_TOL = 1e-9      # row margin (box-scaled direction) above this => row separated
const _SEP_COEF_TOL = 1e-6     # |null-space coordinate| above this => coefficient flagged
const _SEP_RANK_TOL = 1e-10    # relative |R_jj| / singular value below this => rank-deficient
const _SEP_RESID_TOL = 1e-8    # relative NNLS residual below this => no separating direction
_sep_max_iter(q) = 20q + 200   # NNLS budget (inner steps) per screen
const _SEP_NEAR_P = 1e-8       # near separation: a fitted probability within 1e-8 of 0/1 ...
const _SEP_NEAR_Z = 0.05       # ... and |βⱼ / SEⱼ| below this ...
# ... while βⱼ alone moves the linear predictor across the whole [p, 1 − p]
# probability range: |βⱼ|·range(xⱼ) > logit(1 − p) − logit(p) (= 36.84).
const _SEP_NEAR_SPAN = 2 * log((1 - _SEP_NEAR_P) / _SEP_NEAR_P)

# Signed, row-normalised constraint matrix on column-scaled, full-rank columns.
# Rows come first (every row with a success contributes +xᵢ, then every row with
# a failure −xᵢ — the drmTMB row order; rows of zeros are dropped), then the
# columns are scaled to unit max-abs and the rank is read from a pivoted QR of
# the signed rows, so aliased columns are not screened.
# Returns (A, keep) with `keep` the indices (into the columns of X) screened.
function _sep_constraints(X::AbstractMatrix, s, ntr)
    n, p = size(X)
    nz = [any(!iszero, view(X, i, :)) for i in 1:n]
    pos = findall(i -> nz[i] && s[i] > 0, 1:n)
    neg = findall(i -> nz[i] && ntr[i] - s[i] > 0, 1:n)
    isempty(pos) && isempty(neg) && return zeros(0, 0), Int[]
    M = vcat(X[pos, :], -X[neg, :])
    sc = [max(maximum(abs, view(M, :, j)), eps()) for j in 1:p]
    M = M ./ sc'
    F = qr(M, ColumnNorm())
    dR = abs.(diag(F.R))
    r = dR[1] == 0 ? 0 : count(>(_SEP_RANK_TOL * dR[1]), dR)
    keep = sort(F.p[1:r])
    A = M[:, keep]
    nrm = [norm(view(A, i, :)) for i in 1:size(A, 1)]
    ok = findall(>(0), nrm)
    A = A[ok, :] ./ nrm[ok]
    return A, keep
end

# Lawson–Hanson NNLS: minimise ‖Aᵀy + c‖ over y ≥ 0. Returns
# (r, converged, iter) with r = Aᵀy + c the residual; at a converged solution
# A r ≥ 0 (KKT), so r lies in the separation cone.
function _sep_nnls(A::Matrix{Float64}, c::Vector{Float64}, maxiter::Int)
    m = size(A, 1)
    y = zeros(m); P = falses(m); banned = falses(m)
    r = copy(c)
    tol = 1e-12 * max(1.0, norm(c))
    iter = 0
    while true
        w = -(A * r)
        blocked = false
        for i in 1:m
            if P[i]
                w[i] = -Inf
            elseif banned[i]
                w[i] > tol && (blocked = true)
                w[i] = -Inf
            end
        end
        jin = argmax(w)
        w[jin] <= tol && return (r = r, converged = !blocked, iter = iter)
        P[jin] = true
        first = true
        while true
            iter += 1
            iter > maxiter && return (r = r, converged = false, iter = iter)
            idx = findall(P)
            zP = _sep_lstsq(Matrix(transpose(A[idx, :])), -c)
            if all(>(0), zP)
                fill!(y, 0.0); y[idx] .= zP
                fill!(banned, false)
                break
            end
            if first && zP[findfirst(==(jin), idx)] <= 0
                # the entering variable cannot move (numerical tie): set it aside
                P[jin] = false; banned[jin] = true
                break
            end
            first = false
            neg = findall(<=(0), zP)
            ratio = [y[idx[k]] / (y[idx[k]] - zP[k]) for k in neg]
            kmin = argmin(ratio)
            α = ratio[kmin]
            for (k, i) in enumerate(idx)
                y[i] += α * (zP[k] - y[i])
            end
            y[idx[neg[kmin]]] = 0.0
            for i in idx
                y[i] <= 0 && (P[i] = false)
            end
            y[.!P] .= 0.0
        end
        r = c .+ transpose(A) * y
    end
end

# Least squares E z ≈ f with aliased columns of E set to zero (R's `qr.coef`
# convention, so the twins step identically on a rank-deficient passive set).
function _sep_lstsq(E::Matrix{Float64}, f::Vector{Float64})
    F = qr(E, ColumnNorm())
    dR = abs.(diag(F.R))
    k = length(dR)
    rk = (k == 0 || dR[1] == 0) ? 0 : count(>(1e-7 * dR[1]), dR)
    z = zeros(size(E, 2))
    rk == 0 && return z
    qtf = (F.Q' * f)[1:rk]
    z[F.p[1:rk]] = UpperTriangular(F.R[1:rk, 1:rk]) \ qtf
    return z
end

"""
    _detect_separation(X, s, ntr) -> NamedTuple

Separation screen for a Bernoulli/Binomial fixed-effect design `X` (successes
`s` out of `ntr` trials per row). Returns
`(separated::Bool, flagged::Vector{Int}, conclusive::Bool)` where `flagged` are
the column indices of `X` whose coefficients some separating direction moves,
and `conclusive = false` means the iteration budget ran out (the verdict is then
incomplete and is warned about). `maxiter` overrides the NNLS budget
(`20q + 200` inner steps, `q` the screened rank); tests only. Rank-deficient
(aliased) columns are not screened.
"""
function _detect_separation(X::AbstractMatrix, s, ntr; maxiter = nothing)
    none = (separated = false, flagged = Int[], conclusive = true)
    A, keep = _sep_constraints(Matrix{Float64}(X), Float64.(s), Float64.(ntr))
    q = length(keep)
    (q == 0 || size(A, 1) == 0) && return none
    m = size(A, 1)
    budget = maxiter === nothing ? _sep_max_iter(q) : maxiter
    pos = falses(m)          # rows some cone direction separates strictly
    conclusive = true
    while true
        U = .!pos
        any(U) || break
        cU = vec(sum(A[U, :], dims = 1))
        nn = _sep_nnls(A, cU, budget)
        budget -= nn.iter
        if !nn.converged
            conclusive = false; break
        end
        r = nn.r
        norm(r) <= _SEP_RESID_TOL * max(1.0, norm(cU)) && break
        marg = A * (r ./ maximum(abs, r))
        new = U .& (marg .> _SEP_OBJ_TOL)
        if minimum(marg) < -_SEP_RESID_TOL || !any(new)
            conclusive = false; break   # numerically unreliable direction: do not guess
        end
        pos .|= new
    end
    any(pos) || return (separated = false, flagged = Int[], conclusive = conclusive)
    J = .!pos
    flag = if !any(J)
        trues(q)
    else
        F = svd(A[J, :]; full = true)
        rk = count(>(_SEP_RANK_TOL * max(F.S[1], 1.0)), F.S)
        if rk >= q
            falses(q)
        else
            N = F.V[:, (rk + 1):q]
            vec(maximum(abs, N, dims = 2)) .> _SEP_COEF_TOL
        end
    end
    return (separated = true, flagged = keep[flag], conclusive = conclusive)
end

# Near separation (no exact separating direction, but a coefficient sits far out
# on a flat likelihood ridge). Scale-free: coefficient j is flagged when some
# fitted probability is within `_SEP_NEAR_P` of 0 or 1, βⱼ alone moves the
# logit across the whole [p, 1 − p] range (|βⱼ|·range(xⱼ) > `_SEP_NEAR_SPAN`),
# and its Wald |z| is below `_SEP_NEAR_Z`. A missing or non-finite SE (a failed
# Hessian) is never evidence of near separation. Returns the flagged indices.
function _near_separation(μ̂::AbstractVector, β::AbstractVector, se::AbstractVector,
                          xrange::AbstractVector)
    any(m -> min(m, 1 - m) < _SEP_NEAR_P, μ̂) || return Int[]
    return findall(j -> isfinite(se[j]) && se[j] > 0 && isfinite(β[j]) &&
                        abs(β[j]) * xrange[j] > _SEP_NEAR_SPAN &&
                        abs(β[j]) / se[j] < _SEP_NEAR_Z, eachindex(β))
end

# Detect-and-warn for a finished FE-only Binomial fit. Returns the vcov with the
# flagged coefficients' rows/columns set to NaN and their variances to Inf
# (`stderror` → Inf, Wald `confint` → (-Inf, Inf)), and warns. Internal refits
# (bootstrap replicates, run under `_without_boundary_warnings`) stay quiet.
function _separation_guard(X, s, ntr, θ̂, V, μ̂, names; maxiter = nothing)
    det = _detect_separation(X, s, ntr; maxiter = maxiter)
    kind = :none
    flagged = det.flagged
    if det.separated
        kind = :complete_or_quasi
    else
        se = [_boundary_se(V[j, j]) for j in 1:length(θ̂)]
        xrange = [maximum(view(X, :, j)) - minimum(view(X, :, j)) for j in 1:size(X, 2)]
        fl = _near_separation(μ̂, θ̂, se, xrange)
        if !isempty(fl)
            kind = :near; flagged = fl
        elseif !det.conclusive
            kind = :inconclusive
        end
    end
    kind === :none && return V
    quiet = get(task_local_storage(), :drm_quiet_boundary, false)
    if kind === :inconclusive
        quiet || @warn """
              The separation check for this Binomial fixed-effect fit was inconclusive
              (its iteration budget ran out). Separation is neither confirmed nor ruled
              out; inspect large coefficients and standard errors, or compare with a
              penalised (Firth / MSPL) fit.
              """ separation = kind flagged_coefficients = String[]
        return V
    end
    what = kind === :near ?
        "near-separation (a coefficient that moves the logit across the whole $(_SEP_NEAR_P) to 1 − $(_SEP_NEAR_P) probability range has Wald |z| below $(_SEP_NEAR_Z), with fitted probabilities within $(_SEP_NEAR_P) of 0/1)" :
        "(quasi-)complete separation"
    partial = det.conclusive ? "" :
        "\nThe separation check ran out of its iteration budget, so this list may be incomplete."
    quiet || @warn """
          Binomial fixed-effect fit shows $what. The maximum-likelihood estimate
          does not exist (or is not finite): the affected coefficients can grow
          without bound at little or no cost in likelihood, and their reported
          estimates are an arbitrary stopping point of the optimiser. Their standard
          errors are reported as Inf and their Wald intervals as (-Inf, Inf).
          Consider removing or collapsing the offending predictor, or a penalised
          fit (Firth / MSPL).$partial
          """ separation = kind flagged_coefficients = (isempty(flagged) ? String[] : names[flagged])
    V2 = copy(V)
    for j in flagged
        V2[j, :] .= NaN; V2[:, j] .= NaN; V2[j, j] = Inf
    end
    return V2
end
