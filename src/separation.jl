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
# Coefficient j is flagged iff that null space has a non-zero j-th coordinate
# (tested on the basis-free projector diagonal), i.e. iff SOME direction in C
# moves βⱼ; an aliased column whose combination uses a flagged column is flagged
# with it. (R's `detectseparation` reports the
# components of one particular LP direction instead, so the two can differ in
# which coefficients they list; no parity with it is claimed.) Typically one or
# two NNLS solves; each is bounded by an iteration budget, and an exhausted
# budget is reported (`conclusive = false`) and warned about, never read as "no
# separation". Columns are scaled by their median non-zero |entry| and rows to
# unit length, so the check is invariant to column units (no absolute floor)
# and robust to a single far outlier.
#
# Near separation: when the check PROVES there is no separation the MLE exists
# and is finite, so its Wald SE stands even if |z| is tiny (Hauck–Donner). The
# scale-free near rule is consulted only when the check is inconclusive.
#
# Scope: the fixed-effect design of a Binomial/Bernoulli mean (the FE-only
# fit). Random-intercept binomial routes are NOT screened here: a random
# intercept can absorb part of the separation, so the fixed-design check does
# not decide it.

# Numbers shared with the drmTMB twin (keep identical across both packages).
const _SEP_OBJ_TOL = 1e-9      # row margin (box-scaled direction) above this => row separated
const _SEP_COEF_TOL = 1e-6     # null-space projector sqrt(diag) above this => coefficient flagged
const _SEP_RANK_TOL = 1e-10    # relative |R_jj| / singular value below this => rank-deficient
const _SEP_RESID_TOL = 1e-8    # relative NNLS residual below this => no separating direction
_sep_max_iter(q) = 20q + 200   # NNLS budget (inner steps) per screen
const _SEP_NEAR_P = 1e-8       # near separation: a fitted probability within 1e-8 of 0/1 ...
const _SEP_NEAR_Z = 0.05       # ... and |βⱼ / SEⱼ| below this ...
# ... while βⱼ alone moves the linear predictor across the whole [p, 1 − p]
# probability range: |βⱼ|·range(xⱼ) > logit(1 − p) − logit(p) (= 36.84).
const _SEP_NEAR_SPAN = 2 * log((1 - _SEP_NEAR_P) / _SEP_NEAR_P)

# Signed, row-normalised constraint matrix on column-scaled, full-rank columns.
# Also returns the aliased columns and their coefficients on the kept ones.
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
    isempty(pos) && isempty(neg) && return zeros(0, 0), Int[], Int[], zeros(0, 0)
    M = vcat(X[pos, :], -X[neg, :])
    # Column scale: the median non-zero |entry| (1 for an all-zero column). Any
    # positive column scaling leaves separation unchanged; the median keeps a
    # single far outlier from shrinking every other entry towards the
    # tolerances, and has no absolute floor (a 1e-200-scaled column is screened).
    sc = map(1:p) do j
        nzv = [abs(v) for v in view(M, :, j) if v != 0]
        isempty(nzv) ? 1.0 : Statistics.median(nzv)
    end
    M = M ./ sc'
    M = M ./ [norm(view(M, i, :)) for i in 1:size(M, 1)]   # rows non-zero by construction
    F = qr(M, ColumnNorm())
    dR = abs.(diag(F.R))
    r = dR[1] == 0 ? 0 : count(>(_SEP_RANK_TOL * dR[1]), dR)
    keep = sort(F.p[1:r])
    A = M[:, keep]
    # Aliased columns (not screened) as combinations of the kept ones, so a
    # column riding on a flagged one (e.g. x2 = 2x) can be named too.
    alias = setdiff(1:p, keep)
    alias_coef = zeros(length(keep), length(alias))
    for (k, j) in enumerate(alias)
        alias_coef[:, k] = _sep_lstsq(A, M[:, j])
    end
    nrm = [norm(view(A, i, :)) for i in 1:size(A, 1)]
    ok = findall(>(0), nrm)
    A = A[ok, :] ./ nrm[ok]
    return A, keep, alias, alias_coef
end

# Lawson–Hanson NNLS: minimise ‖Aᵀy + c‖ over y ≥ 0. Returns
# (r, converged, iter) with r = Aᵀy + c the residual; at a converged solution
# A r ≥ 0 (KKT), so r lies in the separation cone.
function _sep_nnls(A::Matrix{Float64}, c::Vector{Float64}, maxiter::Int)
    m = size(A, 1)
    y = zeros(m); P = falses(m); banned = falses(m)
    r = copy(c)
    rtol = _SEP_RESID_TOL * max(1.0, norm(c))
    iter = 0
    while true
        # A residual this small certifies no separating direction (the caller
        # applies the same threshold); stop rather than chase rounding noise.
        norm(r) <= rtol && return (r = r, converged = true, iter = iter)
        w = -(A * r)
        # KKT tolerance on the box-scaled margin aᵢ·r / max|r| (rounding on
        # ill-conditioned designs is ~1e-10; the caller's feasibility check is 1e-8)
        tol = max(1e-9 * maximum(abs, r), 1e-12 * max(1.0, norm(c)))
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

# Least squares E z ≈ f by column-pivoted QR, with columns of E judged
# dependent (|R_kk| ≤ 1e-7·|R_11|) given a zero coefficient. R's `qr.coef` uses
# LINPACK dqrdc2 (limited pivoting, NA for aliased columns) instead, so on a
# rank-deficient passive set the twins can step differently; the verdicts are
# checked to agree, the iterates are not claimed to.
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
    A, keep, alias, alias_coef = _sep_constraints(Matrix{Float64}(X), Float64.(s), Float64.(ntr))
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
            # sqrt of the null-space projector diagonal: basis-invariant
            vec(sqrt.(sum(abs2, N, dims = 2))) .> _SEP_COEF_TOL
        end
    end
    flagged = keep[flag]
    if !isempty(alias) && any(flag)
        rides = vec(sum(abs, alias_coef[flag, :], dims = 1)) .> 1e-8
        flagged = sort(vcat(flagged, alias[rides]))
    end
    return (separated = true, flagged = flagged, conclusive = conclusive)
end

# Near separation (the exact check was inconclusive, and a coefficient sits far
# out on a flat likelihood ridge). Consulted only when `_detect_separation`
# returns `conclusive = false`. Scale-free: coefficient j is flagged when some
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
    elseif !det.conclusive
        # The near rule is consulted only when the exact check could not decide.
        # When it proves "not separated" the MLE exists and is finite, and its
        # Wald SE is the real curvature: a tiny |z| there is a legitimate
        # (Hauck–Donner) result, not separation (#914 review: a far x outlier
        # made a null slope look near-separated).
        se = [_boundary_se(V[j, j]) for j in 1:length(θ̂)]
        xrange = [maximum(view(X, :, j)) - minimum(view(X, :, j)) for j in 1:size(X, 2)]
        fl = _near_separation(μ̂, θ̂, se, xrange)
        if !isempty(fl)
            kind = :near; flagged = fl
        else
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
