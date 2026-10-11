# Sparse kernel for the Gaussian multi-component scalar random-effect route
# `(1 | g) + (1 | h) + …` (`_fit_multi_ranef_gaussian`, gaussian_ranef.jl).
#
# #977: the route took a DENSE QR of the stacked (n + q) × q design
# [D^{-1/2}Z̃; I] on every objective and gradient call, and the gradient then
# formed a dense q × q inverse and an n × q leverage product. With crossed
# factors q = Σ G_k grows with n, so a fit at n = 2000 took 331 s against
# drmTMB's 0.5 s. This file computes the SAME quantities sparsely:
#
# * Value. A sparse QR (SuiteSparse SPQR) of that same stacked matrix. Its
#   triangular factor R satisfies RᵀR = P(I + Z̃ᵀD⁻¹Z̃)Pᵀ, so R is the
#   (column-permuted) sparse Cholesky factor of the q × q Woodbury capacitance
#   matrix. It is computed WITHOUT forming M, which keeps the #835/#837
#   cancellation guard (`test_cancellation_sweep.jl`) intact: forming M and
#   calling CHOLMOD loses the identity to rounding as σ_e → 0. logdet(M) =
#   2Σ log|Rᵢᵢ| and the quadratic is the same penalised RSS at the
#   least-squares mode b̂ as the dense route.
# * Gradient. The analytic gradient needs diag(M⁻¹) and the per-row leverage
#   z̃ᵢᵀM⁻¹z̃ᵢ, which touch M⁻¹ only on the pattern of M (and so of the factor).
#   `_multi_re_selinv` gets those entries with the Takahashi recursion (the
#   same recursion as `takahashi_selinv`), plus a dense trailing block. With
#   crossed factors, eliminating the larger factor fills the smaller one
#   (nearly) densely, so the trailing columns of the factor form a dense
#   triangle. That block is inverted with LAPACK (`potri`) instead of entry by
#   entry. The trailing block of (LLᵀ)⁻¹ depends only on the trailing block of
#   L, so this is exact, not an approximation.
#
# Dual-number calls (ForwardDiff through the objective, e.g. the small-q vcov
# Hessian) still use the dense `_multi_re_qr`, which is generic in the element
# type. Both routes evaluate the same expressions in exact arithmetic; the
# identity is pinned to ~1e-10 in test/test_multi_re_sparse_kernel.jl.

using SparseArrays
using LinearAlgebra

# Up to this q the vcov Hessian is ForwardDiff through the dense objective, as
# before #977, so SEs on small fits are bit-for-bit what they were. Above it the
# dense Dual QR costs O((n + q) q²) per Hessian column and we use central
# differences of the exact sparse analytic gradient instead.
const _MULTI_RE_DENSE_HESSIAN_MAXQ = 200

"""
    _MultiREDesign

Row-wise and sparse views of the multi-component scalar RE design `Z`
(n × q, one nonzero per component per row). `cols[i, k]` is the column of row
`i` in component `k` and `wts[i, k]` its weight; `Z` is the same matrix in CSC.
"""
struct _MultiREDesign
    Z::SparseMatrixCSC{Float64,Int}
    cols::Matrix{Int}
    wts::Matrix{Float64}
    colcomp::Vector{Int}
    offs::Vector{Int}
    Mpat::SparseMatrixCSC{Float64,Int}   # structural pattern of ZᵀZ (all ones)
end

function _multi_re_design(comps, n::Int)
    K = length(comps)
    Gks = [c[3] for c in comps]
    q = sum(Gks)
    offs = cumsum([0; Gks])
    cols = Matrix{Int}(undef, n, K)
    wts = Matrix{Float64}(undef, n, K)
    colcomp = Vector{Int}(undef, q)
    for (k, (w, gidx, Gk, _)) in enumerate(comps)
        for c in 1:Gk
            colcomp[offs[k]+c] = k
        end
        @inbounds for i in 1:n
            cols[i, k] = offs[k] + gidx[i]
            wts[i, k] = w[i]
        end
    end
    Z = sparse(repeat(1:n, K), vec(cols), vec(wts), n, q)
    Zp = sparse(repeat(1:n, K), vec(cols), ones(n * K), n, q)
    Mpat = Zp' * Zp
    fill!(nonzeros(Mpat), 1.0)
    return _MultiREDesign(Z, cols, wts, colcomp, offs, Mpat)
end

"""
    _multi_re_sparse_qr(Z̃, sdinv, r) -> (F, bhat, logdetM)

Sparse twin of `_multi_re_qr`: SPQR of `[sdinv .* Z̃; I]` (no rank dropping,
`tol = 0`, matching the dense Householder QR), the least-squares mode `b̂` and
`logdet(I + Z̃ᵀD⁻¹Z̃) = 2Σ log|Rᵢᵢ|`.
"""
function _multi_re_sparse_qr(Z̃::SparseMatrixCSC{Float64,Int}, sdinv::AbstractVector{Float64},
                             r::AbstractVector{Float64})
    q = size(Z̃, 2)
    A = vcat(Diagonal(sdinv) * Z̃, sparse(1.0I, q, q))
    F = qr(A; tol = 0.0)
    bhat = F \ vcat(sdinv .* r, zeros(q))
    logdetM = 2 * sum(log ∘ abs, diag(F.R))
    return F, bhat, logdetM
end

"""
    _MultiRESelInv

Selected inverse of `M = RᵀR` in the permuted basis of the factor: a sparse
lower part `Zvals` at the (closed) pattern of `L = Rᵀ` for the leading columns,
and a dense block `tail` for the trailing columns `t:q`. `ipos[c]` is the
permuted position of original column `c`.
"""
struct _MultiRESelInv
    colptr::Vector{Int}
    rowval::Vector{Int}
    Zvals::Vector{Float64}
    t::Int
    tail::Matrix{Float64}
    ipos::Vector{Int}
end

# M⁻¹[a, b] for ORIGINAL column indices a, b (must lie in the factor pattern).
@inline function _selinv_get(S::_MultiRESelInv, a::Int, b::Int)
    i = S.ipos[a]; j = S.ipos[b]
    return _selinv_getp(S, max(i, j), min(i, j))
end

# Permuted-basis lookup of the lower entry (r ≥ c).
@inline function _selinv_getp(S::_MultiRESelInv, r::Int, c::Int)
    t = S.t
    if c >= t
        return S.tail[r-t+1, c-t+1]
    end
    idx = _csc_rowidx(S.colptr, S.rowval, c, r)
    idx == -1 && throw(ErrorException(
        "multi-RE selected inverse: entry ($r, $c) is outside the factor pattern"))
    return S.Zvals[idx]
end

# Lower factor L = Rᵀ with positive diagonal (row signs of R flipped; LLᵀ is
# unchanged) and its pattern closed under the elimination tree, so every entry
# the Takahashi recursion reads is stored (fill positions hold explicit zeros).
# The pattern is seeded with the pattern of M itself (`Mperm`, permuted like
# R's columns), not only R's: a QR factor can be structurally sparser than the
# symbolic Cholesky factor of M (exact cancellation when the stacked design is
# not strong Hall), and the leverage needs M⁻¹ on every pattern entry of M.
function _closed_lower_factor(R::SparseMatrixCSC{Float64,Int}, Mperm::SparseMatrixCSC{Float64,Int})
    q = size(R, 2)
    L = sparse(transpose(R))
    rows = [Int[] for _ in 1:q]
    vals = [Float64[] for _ in 1:q]
    @inbounds for j in 1:q
        rng = L.colptr[j]:(L.colptr[j+1]-1)
        rows[j] = L.rowval[rng]
        vals[j] = L.nzval[rng]
        rows[j][1] == j || throw(ErrorException("multi-RE factor: missing diagonal in column $j"))
        if vals[j][1] < 0
            vals[j] .= .-vals[j]
        end
        extra = Int[]
        for idx in Mperm.colptr[j]:(Mperm.colptr[j+1]-1)
            i = Mperm.rowval[idx]
            i > j && !insorted(i, rows[j]) && push!(extra, i)
        end
        if !isempty(extra)
            newr = sort!(vcat(rows[j], extra))
            newv = zeros(length(newr))
            k = 1
            for (m, rr) in enumerate(newr)
                if k <= length(rows[j]) && rows[j][k] == rr
                    newv[m] = vals[j][k]; k += 1
                end
            end
            rows[j] = newr; vals[j] = newv
        end
    end
    @inbounds for j in 1:q
        length(rows[j]) > 1 || continue
        p = rows[j][2]                          # elimination-tree parent
        need = @view rows[j][3:end]
        isempty(need) && continue
        rp = rows[p]
        # sorted union of rows[p] and need (all > p)
        if !issubset(need, rp)
            newr = sort!(union(rp, need))
            newv = zeros(length(newr))
            k = 1
            for (m, rr) in enumerate(newr)
                if k <= length(rp) && rp[k] == rr
                    newv[m] = vals[p][k]; k += 1
                end
            end
            rows[p] = newr; vals[p] = newv
        end
    end
    colptr = Vector{Int}(undef, q + 1); colptr[1] = 1
    for j in 1:q
        colptr[j+1] = colptr[j] + length(rows[j])
    end
    return colptr, reduce(vcat, rows), reduce(vcat, vals)
end

"""
    _multi_re_selinv(F, Mpat) -> _MultiRESelInv

Entries of `M⁻¹` on the factor pattern, where `F` is the sparse QR from
`_multi_re_sparse_qr` (`RᵀR = M[pcol, pcol]`). Takahashi recursion on the
leading columns, dense LAPACK inverse on the trailing dense block.
"""
function _multi_re_selinv(F, Mpat::SparseMatrixCSC{Float64,Int})
    R = F.R
    q = size(R, 2)
    size(R, 1) == q || throw(ErrorException("multi-RE factor is rank deficient"))
    colptr, rowval, Lvals = _closed_lower_factor(R, Mpat[F.pcol, F.pcol])
    # Trailing dense block: the smallest t whose trailing triangle is at least
    # half full (then padding it to dense at most doubles its storage).
    t = q; cnt = 0
    for j in q:-1:1
        cnt += colptr[j+1] - colptr[j]
        m = q - j + 1
        if 2 * cnt >= m * (m + 1) ÷ 2
            t = j
        end
    end
    md = q - t + 1
    L22 = zeros(md, md)
    @inbounds for j in t:q, idx in colptr[j]:(colptr[j+1]-1)
        L22[rowval[idx]-t+1, j-t+1] = Lvals[idx]
    end
    LAPACK.potri!('L', L22)                     # lower triangle of (L22 L22ᵀ)⁻¹
    @inbounds for j in 1:md, i in 1:(j-1)
        L22[i, j] = L22[j, i]
    end
    S = _MultiRESelInv(colptr, rowval, zeros(length(Lvals)), t, L22, invperm(F.pcol))
    Zvals = S.Zvals
    @inbounds for j in (t-1):-1:1
        cs = colptr[j]; ce = colptr[j+1] - 1
        invLjj = 1.0 / Lvals[cs]
        for off_r in ce:-1:(cs+1)
            r = rowval[off_r]
            s = 0.0
            for off_k in (cs+1):ce
                k = rowval[off_k]
                s += Lvals[off_k] * _selinv_getp(S, max(k, r), min(k, r))
            end
            Zvals[off_r] = -s * invLjj
        end
        s = 0.0
        for off_k in (cs+1):ce
            s += Lvals[off_k] * Zvals[off_k]
        end
        Zvals[cs] = invLjj * invLjj - s * invLjj
    end
    return S
end

# diag(M⁻¹) in the original column order.
_selinv_diag(S::_MultiRESelInv, q::Int) = [_selinv_getp(S, S.ipos[c], S.ipos[c]) for c in 1:q]

# Leverage z̃ᵢᵀ M⁻¹ z̃ᵢ for every row, z̃ᵢ = (σ_k wᵢₖ at column cols[i, k]).
function _multi_re_leverage(S::_MultiRESelInv, des::_MultiREDesign, σk::AbstractVector{Float64})
    n, K = size(des.cols)
    lev = zeros(n)
    @inbounds for i in 1:n
        s = 0.0
        for a in 1:K
            za = σk[a] * des.wts[i, a]
            za == 0 && continue
            ca = des.cols[i, a]
            s += za * za * _selinv_get(S, ca, ca)
            for b in (a+1):K
                zb = σk[b] * des.wts[i, b]
                zb == 0 && continue
                s += 2 * za * zb * _selinv_get(S, ca, des.cols[i, b])
            end
        end
        lev[i] = s
    end
    return lev
end

# Central-difference Hessian of an exact analytic gradient (used for the vcov
# when q is too large for ForwardDiff through the dense objective). Step
# h = ∛eps · max(1, |θⱼ|) balances truncation O(h²) and rounding O(eps/h).
function _multi_re_fd_hessian(grad!, θ::Vector{Float64})
    p = length(θ)
    H = zeros(p, p)
    gp = zeros(p); gm = zeros(p)
    for j in 1:p
        h = cbrt(eps()) * max(1.0, abs(θ[j]))
        θp = copy(θ); θp[j] += h
        θm = copy(θ); θm[j] -= h
        grad!(gp, θp); grad!(gm, θm)
        H[:, j] .= (gp .- gm) ./ (θp[j] - θm[j])
    end
    return (H .+ H') ./ 2
end
