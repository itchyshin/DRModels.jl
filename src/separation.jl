# separation.jl — (quasi-)complete separation screen for Bernoulli / Binomial
# fixed effects (#731, #728; twin contract drmTMB #1268).
#
# Policy (owner decision 2026-09-30, "detect and warn"): the fit is still
# returned, a warning names the affected coefficients, and their standard errors
# are reported as Inf. No refusal and no penalised (Firth / MSPL) estimator by
# default. The twins agree on WHICH coefficients are flagged and on the warning,
# not on the (non-existent) maximum-likelihood coefficients.
#
# Method (Konis 2007; the idea behind R's `detectseparation`, implemented here
# independently from the published method). With signed design rows
# x̃ᵢ = xᵢ for a success and −xᵢ for a failure, the Bernoulli/binomial
# likelihood has no finite maximiser iff some direction d satisfies x̃ᵢ·d ≥ 0
# for every row and > 0 for at least one: moving along d never lowers the
# likelihood and the coefficients diverge. This is a linear-programming
# feasibility question over the cone C = {d : X̃ d ≥ 0}. We solve
#
#     max  Σᵢ x̃ᵢ·d   s.t.  X̃ d ≥ 0,  −1 ≤ d ≤ 1
#
# (optimum > 0  <=>  separation), then, for every coefficient j, max/min dⱼ over
# the same polytope. Coefficient j is flagged when the cone contains a direction
# with dⱼ ≠ 0, i.e. when β̂ⱼ can drift without bound at no cost in likelihood
# (a generic recession direction lies in the relative interior of C, so it has
# dⱼ ≠ 0 exactly then). No LP package is added: the problem has p (small)
# variables and one constraint per row, so a small bounded simplex in
# "inequality form" (a basis of p active constraints, Bland's rule against the
# heavy degeneracy at d = 0) is used.
#
# Scope: the fixed-effect design of a Binomial/Bernoulli mean (the FE-only
# fit). Random-intercept binomial routes are NOT screened here: a random
# intercept can absorb part of the separation, so the fixed-design LP does not
# decide it.

# Numbers shared with the drmTMB twin (keep identical across both packages).
const _SEP_OBJ_TOL = 1e-9      # LP optimum above this => separation
const _SEP_COEF_TOL = 1e-6     # |dⱼ| (box-scaled) above this => coefficient flagged
const _SEP_RANK_TOL = 1e-10    # relative |R_jj| below this => aliased column, not screened
const _SEP_NEAR_P = 1e-8       # near separation: a fitted probability within 1e-8 of 0 or 1 ...
const _SEP_NEAR_SE = 1e4       # ... AND a Wald SE above 1e4 (or non-finite)

# Signed, row-normalised constraint matrix on column-scaled, full-rank columns.
# Rows come first (a success contributes +xᵢ, a failure −xᵢ; rows with no trials
# are dropped), then the columns are scaled to unit max-abs and the rank is read
# from a pivoted QR of the signed rows, so aliased columns are not screened.
# Returns (A, keep) with `keep` the indices (into the columns of X) screened.
function _sep_constraints(X::AbstractMatrix, s, ntr)
    n, p = size(X)
    rows = Vector{Float64}[]
    for i in 1:n
        any(!iszero, view(X, i, :)) || continue
        s[i] > 0 && push!(rows, X[i, :])
        ntr[i] - s[i] > 0 && push!(rows, -X[i, :])
    end
    isempty(rows) && return zeros(0, 0), Int[]
    M = reduce(vcat, (row' for row in rows))
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

# Maximise c·d over {A d ≥ 0, −1 ≤ d ≤ 1}. `basis0` holds p row indices of A that
# are linearly independent (d = 0 is then a — degenerate — vertex). Returns
# (value, d), or `nothing` if the iteration cap is hit.
function _sep_lp(A::Matrix{Float64}, c::Vector{Float64}, basis0::Vector{Int};
                 maxiter::Int = 50_000)
    m, p = size(A)
    basis = copy(basis0)
    gvec(i) = i <= m ? A[i, :] : i <= m + p ? (e = zeros(p); e[i-m] = 1.0; e) :
              (e = zeros(p); e[i-m-p] = -1.0; e)
    hval(i) = i <= m ? 0.0 : -1.0
    for _ in 1:maxiter
        Gb = reduce(vcat, (gvec(i)' for i in basis))
        hb = [hval(i) for i in basis]
        d = Gb \ hb
        λ = Gb' \ c                     # c = Σ λₖ gₖ; optimal for max iff all λ ≤ 0
        kpos = 0
        for k in 1:p
            if λ[k] > 1e-10 && (kpos == 0 || basis[k] < basis[kpos])
                kpos = k
            end
        end
        kpos == 0 && return (dot(c, d), d)
        eₖ = zeros(p); eₖ[kpos] = 1.0
        δ = Gb \ eₖ
        tmin = Inf; ileave = 0
        rate = A * δ
        sl = A * d
        for i in 1:m
            if rate[i] < -1e-10 && !(i in basis)
                t = max(sl[i], 0.0) / -rate[i]
                if t < tmin - 1e-12 || (abs(t - tmin) <= 1e-12 && i < ileave)
                    tmin = t; ileave = i
                end
            end
        end
        for j in 1:p
            if δ[j] < -1e-10 && !((m + j) in basis)
                t = max(d[j] + 1.0, 0.0) / -δ[j]
                i = m + j
                if t < tmin - 1e-12 || (abs(t - tmin) <= 1e-12 && i < ileave)
                    tmin = t; ileave = i
                end
            elseif δ[j] > 1e-10 && !((m + p + j) in basis)
                t = max(1.0 - d[j], 0.0) / δ[j]
                i = m + p + j
                if t < tmin - 1e-12 || (abs(t - tmin) <= 1e-12 && i < ileave)
                    tmin = t; ileave = i
                end
            end
        end
        ileave == 0 && return nothing   # cannot happen in a bounded polytope
        basis[kpos] = ileave
    end
    return nothing
end

"""
    _detect_separation(X, s, ntr) -> NamedTuple

Konis-style LP screen for (quasi-)complete separation of a Bernoulli/Binomial
fixed-effect design `X` (successes `s` out of `ntr` trials per row). Returns
`(separated::Bool, flagged::Vector{Int}, conclusive::Bool)` where `flagged` are
the column indices of `X` whose coefficients can diverge. Rank-deficient
(aliased) columns are not screened.
"""
function _detect_separation(X::AbstractMatrix, s, ntr)
    p = size(X, 2)
    A, keep = _sep_constraints(Matrix{Float64}(X), Float64.(s), Float64.(ntr))
    q = length(keep)
    (q == 0 || size(A, 1) == 0) && return (separated = false, flagged = Int[], conclusive = true)
    F = qr(Matrix(A'), ColumnNorm())
    basis0 = sort(F.p[1:q])
    res = _sep_lp(A, vec(sum(A, dims = 1)), basis0)
    res === nothing && return (separated = false, flagged = Int[], conclusive = false)
    res[1] > _SEP_OBJ_TOL || return (separated = false, flagged = Int[], conclusive = true)
    flagged = Int[]
    conclusive = true
    for (jj, j) in enumerate(keep)
        hit = false
        for sg in (1.0, -1.0)
            c = zeros(q); c[jj] = sg
            r = _sep_lp(A, c, basis0)
            r === nothing && (conclusive = false; continue)
            r[1] > _SEP_COEF_TOL && (hit = true)
        end
        hit && push!(flagged, j)
    end
    return (separated = true, flagged = sort(flagged), conclusive = conclusive)
end

# Near separation (the LP found no exact separating direction, but the fit is
# numerically degenerate): some fitted probability within 1e-8 of 0 or 1 AND a
# Wald SE above 1e4 or non-finite. Flags the coefficients with such an SE.
function _near_separation(μ̂::AbstractVector, se::AbstractVector)
    any(m -> min(m, 1 - m) < _SEP_NEAR_P, μ̂) || return Int[]
    return findall(v -> !isfinite(v) || v > _SEP_NEAR_SE, se)
end

# Detect-and-warn for a finished FE-only Binomial fit. Returns the vcov with the
# flagged coefficients' rows/columns set to NaN (stderror reports Inf) and warns.
function _separation_guard(X, s, ntr, θ̂, V, μ̂, names)
    det = _detect_separation(X, s, ntr)
    kind = :none
    flagged = det.flagged
    if det.separated
        kind = :complete_or_quasi
    else
        se = [_boundary_se(V[j, j]) for j in 1:length(θ̂)]
        fl = _near_separation(μ̂, se)
        if !isempty(fl)
            kind = :near; flagged = fl
        end
    end
    kind === :none && return V
    nm = isempty(flagged) ? "(none identified)" : join(names[flagged], ", ")
    what = kind === :near ? "near-separation (fitted probabilities within $(_SEP_NEAR_P) of 0/1 with a Wald SE above $(_SEP_NEAR_SE))" :
           "(quasi-)complete separation"
    @warn """
          Binomial fixed-effect fit shows $what. The maximum-likelihood estimate
          does not exist (or is not finite): the affected coefficients can grow
          without bound at no cost in likelihood, and their reported estimates are
          an arbitrary stopping point of the optimiser. Their standard errors are
          reported as Inf. Consider removing or collapsing the offending
          predictor, or a penalised fit (Firth / MSPL).
          """ separation = kind flagged_coefficients = (isempty(flagged) ? String[] : names[flagged])
    V2 = copy(V)
    for j in flagged
        V2[j, :] .= NaN; V2[:, j] .= NaN
    end
    return V2
end
