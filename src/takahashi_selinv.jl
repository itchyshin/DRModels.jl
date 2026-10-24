# Takahashi selected inverse for sparse positive-definite matrices.
#
# WHY THIS FILE EXISTS
# --------------------
# `src/sparse_phy_grad.jl`'s analytic gradient and `src/em_phylo.jl`'s E-step
# both need entries of `Q⁻¹` for a SPARSE precision `Q`. The Takahashi (1973) /
# Erisman–Tinney (1975) recursion gives the entries of `Q⁻¹` at the sparsity
# pattern of `L + Lᵀ` without forming the dense inverse.
#
# PERFORMANCE KERNEL (ported from HSquared.jl, 2026-09-24 lane C pilot)
# --------------------------------------------------------------------
# The hot recursion lives in `_selinv_zvals`. Cost is `Θ(Σⱼ|L[:,j]|²)`, not
# `O(nnz(L))`. HSquared.jl #361 replaced per-pair binary searches with a
# cap-free clique scatter (bit-identical). HSquared.jl #363 added an aligned-tail
# `@simd` path gated at a stated rtol (1.3e-15 relative at fill 471). Default
# here matches H²: scatter + SIMD. Use `_selinv_zvals(ch; strict_order=true)`
# for the bit-identical merge, or `per_pair=true` for the original recursion.
# Provenance: MIT code from itchyshin/HSquared.jl `src/takahashi_selinv.jl` on
# `origin/main` after #363; itself adapted earlier from this DRM file.
#
# IMPORTANT CAVEAT
# ----------------
# Exact only at entries in the `L + Lᵀ` pattern. The diagonal is always in
# pattern. Leaf-to-leaf covariances outside the elimination tree are not.
#
# THE MATH (column-oriented recursion)
# ------------------------------------
# Let `P Q Pᵀ = L Lᵀ`. Compute `Z = (P Q Pᵀ)⁻¹` at the symmetric `L + Lᵀ`
# pattern; recover `Q⁻¹` by `Z[invperm(ch.p), invperm(ch.p)]`.

# Binary search for row `i` in column `j` of a CSC sparse matrix; returns the
# nzval index if found, -1 otherwise. CSC row indices are sorted increasing.
@inline function _csc_rowidx(colptr::Vector{Int}, rowval::Vector{Int},
                              j::Int, i::Int)
    lo = colptr[j]; hi = colptr[j + 1] - 1
    @inbounds while lo <= hi
        m = (lo + hi) >>> 1
        rm = rowval[m]
        if rm == i
            return m
        elseif rm < i
            lo = m + 1
        else
            hi = m - 1
        end
    end
    return -1
end

# Shared Takahashi recursion, reused by takahashi_selinv and takahashi_diag.
# Returns the selected-inverse values `Zvals` aligned to the
# CSC structure of `L = sparse(ch.L)` (permuted ordering; column `j`'s diagonal is
# at `colptr[j]`, off-diagonals at the following nonzero offsets), plus the CSC
# arrays and permutation to map back to the original ordering. Cost is
# `Θ(Σⱼ|L[:,j]|²)` (see the WHY block above), NOT `O(nnz(L))`; the
# `L + Lᵀ` pattern entries are exact regardless of that cost.
function _selinv_zvals(ch::SparseArrays.CHOLMOD.Factor{Float64}; per_pair::Bool = false,
                       strict_order::Bool = false)
    L = sparse(ch.L)
    perm = ch.p
    n = size(L, 1)
    colptr = L.colptr
    rowval = L.rowval
    Lvals = L.nzval

    Zvals = zeros(Float64, length(Lvals))
    if per_pair
        _selinv_zvals_per_pair!(Zvals, colptr, rowval, Lvals, n)
        return Zvals, colptr, rowval, perm, n
    end

    # PERFORMANCE. Column `j` needs, for each of its clique rows `i_q` (its row pattern
    # below the diagonal, `i_1 < … < i_m`), the sum `s_q = Σ_p L[i_p, j] · Z[i_p, i_q]`
    # over ALL clique members `p`, accumulated in ascending `p`. `Z[i_p, i_q]` with
    # `p < q` is stored in column `i_p` at row `i_q` (entries live at column
    # `min(row, col)`), and by the Cholesky fill-path property every clique row after
    # `i_p` IS in column `i_p`'s pattern. So walk `p = 1…m` once: at step `p`, add the
    # diagonal term to `s_p`, then merge column `i_p`'s rows with the clique tail
    # `i_{p+1}…i_m`; each match `v = Z[i_p, i_q]` contributes `L[i_q, j]·v` to `s_p` and
    # `L[i_p, j]·v` to `s_q`. Every `s_q` therefore receives its terms in exactly
    # ascending `p` — the order of the original per-pair recursion — so the output is
    # BIT-IDENTICAL to it (`test/runtests.jl` pins this). Structurally absent entries
    # (the per-pair path adds `L·0.0`) are skipped: `s + ±0.0 == s` for every `s`
    # reachable here, because an accumulator that starts at `+0.0` can never become
    # `-0.0`. Scratch is one length-`maxm` vector; there is no clique-width cap.
    maxm = 0
    @inbounds for c in 1:n
        mc = colptr[c + 1] - colptr[c] - 1
        mc > maxm && (maxm = mc)
    end
    acc = zeros(Float64, maxm)

    @inbounds for j in n:-1:1
        cs = colptr[j]; ce = colptr[j + 1] - 1
        invLjj = 1.0 / Lvals[cs]
        m = ce - cs

        if m == 0
            Zvals[cs] = invLjj * invLjj
            continue
        end

        for q in 1:m
            acc[q] = 0.0
        end
        for p in 1:m
            ip = rowval[cs + p]
            lp = Lvals[cs + p]
            pcs = colptr[ip]; pce = colptr[ip + 1] - 1
            sp = acc[p] + lp * Zvals[pcs]               # p == q: Z[i_p, i_p]
            ntail = m - p
            if ntail > 0
                # ALIGNED TAIL. When column `i_p`'s first `ntail` off-diagonal rows ARE the
                # clique tail, the merge degenerates to a unit-stride walk. The three cheap
                # tests below are exact, not heuristic: the tail is a subset of column `i_p`'s
                # pattern (fill-path property), it has `ntail` elements, they lie in
                # `[i_{p+1}, i_m]`, and column `i_p` holds exactly `ntail` rows in that range
                # when its `ntail`-th row is `i_m` — so the two lists coincide. `@simd`
                # reassociates the `sp` reduction, which is why this path is gated at rtol
                # rather than bitwise (measured 1.3e-15 relative on a fill-471 factor);
                # `strict_order = true` keeps the bit-identical merge.
                if !strict_order && (pce - pcs) >= ntail &&
                   rowval[pcs + 1] == rowval[cs + p + 1] && rowval[pcs + ntail] == rowval[cs + m]
                    @simd for t in 1:ntail
                        v = Zvals[pcs + t]
                        sp += Lvals[cs + p + t] * v
                        acc[p + t] = muladd(lp, v, acc[p + t])
                    end
                elseif (pce - pcs) <= 11 * ntail
                    # linear merge of two ascending row lists
                    a = pcs + 1
                    q = p + 1
                    while q <= m && a <= pce
                        iq = rowval[cs + q]
                        ra = rowval[a]
                        if ra == iq
                            v = Zvals[a]
                            sp += Lvals[cs + q] * v
                            acc[q] += lp * v
                            a += 1; q += 1
                        elseif ra < iq
                            a += 1
                        else
                            q += 1
                        end
                    end
                else
                    # column i_p is long relative to the tail we want: search per entry
                    for q in (p + 1):m
                        idx = _csc_rowidx(colptr, rowval, ip, rowval[cs + q])
                        if idx != -1
                            v = Zvals[idx]
                            sp += Lvals[cs + q] * v
                            acc[q] += lp * v
                        end
                    end
                end
            end
            acc[p] = sp
        end
        for q in 1:m
            Zvals[cs + q] = -acc[q] * invLjj
        end

        s = 0.0
        for off_k in (cs + 1):ce
            s += Lvals[off_k] * Zvals[off_k]
        end
        Zvals[cs] = invLjj * invLjj - s * invLjj
    end

    return Zvals, colptr, rowval, perm, n
end

# The original per-pair recursion (one binary search per clique pair), kept as the
# bitwise reference `_selinv_zvals(ch; per_pair = true)` that the tests pin the
# scatter path against. Not used on any production path.
function _selinv_zvals_per_pair!(Zvals, colptr, rowval, Lvals, n)
    @inbounds for j in n:-1:1
        cs = colptr[j]; ce = colptr[j + 1] - 1
        invLjj = 1.0 / Lvals[cs]
        for off_r in ce:-1:(cs + 1)
            r = rowval[off_r]
            s = 0.0
            for off_k in (cs + 1):ce
                k = rowval[off_k]
                Lkj = Lvals[off_k]
                if k == r
                    z_kr = Zvals[colptr[r]]
                elseif k < r
                    idx = _csc_rowidx(colptr, rowval, k, r)
                    z_kr = idx == -1 ? 0.0 : Zvals[idx]
                else
                    idx = _csc_rowidx(colptr, rowval, r, k)
                    z_kr = idx == -1 ? 0.0 : Zvals[idx]
                end
                s += Lkj * z_kr
            end
            Zvals[off_r] = -s * invLjj
        end
        s = 0.0
        for off_k in (cs + 1):ce
            s += Lvals[off_k] * Zvals[off_k]
        end
        Zvals[cs] = invLjj * invLjj - s * invLjj
    end
    return Zvals
end

"""
    takahashi_selinv(ch::SparseArrays.CHOLMOD.Factor{Float64}) -> SparseMatrixCSC

Compute the Takahashi selected inverse of the matrix `Q` whose sparse Cholesky
factor is `ch` (`P · C · Pᵀ = L · Lᵀ`). Returns a `SparseMatrixCSC` holding
`Q⁻¹` (in the ORIGINAL un-permuted ordering) at the union sparsity of
`Pᵀ (L + Lᵀ) P`. Entries outside that pattern are NOT computed (and are NOT zero
in general). Kernel ported from HSquared.jl #361/#363 (MIT); API native to DRModels.jl.
"""
function takahashi_selinv(ch::SparseArrays.CHOLMOD.Factor{Float64})
    Zvals, colptr, rowval, perm, n = _selinv_zvals(ch)

    nnz_out = 2 * length(Zvals) - n
    I_out = Vector{Int}(undef, nnz_out)
    J_out = Vector{Int}(undef, nnz_out)
    V_out = Vector{Float64}(undef, nnz_out)
    idx = 0
    @inbounds for j in 1:n
        cs = colptr[j]; ce = colptr[j + 1] - 1
        idx += 1
        I_out[idx] = perm[j]; J_out[idx] = perm[j]; V_out[idx] = Zvals[cs]
        for off in (cs + 1):ce
            r = rowval[off]
            v = Zvals[off]
            idx += 1
            I_out[idx] = perm[r]; J_out[idx] = perm[j]; V_out[idx] = v
            idx += 1
            I_out[idx] = perm[j]; J_out[idx] = perm[r]; V_out[idx] = v
        end
    end
    return sparse(I_out, J_out, V_out, n, n)
end

"""
    takahashi_diag(ch::SparseArrays.CHOLMOD.Factor{Float64}) -> Vector{Float64}

Return ONLY `diag(Q⁻¹)` (length-n, in the ORIGINAL ordering) via the Takahashi
recursion, without materialising the full sparse output (a real memory saving;
the recursion's own FLOP cost is `Θ(Σⱼ|L[:,j]|²)`, comparable to the Cholesky
factorization itself, NOT `O(nnz(L))` — see the file header). The diagonal is
always in the `L + Lᵀ` pattern, so it is exact. Kernel ported from HSquared.jl #361/#363 (MIT); API native to DRModels.jl.
"""
function takahashi_diag(ch::SparseArrays.CHOLMOD.Factor{Float64})
    Zvals, colptr, _, perm, n = _selinv_zvals(ch)
    d = Vector{Float64}(undef, n)
    @inbounds for j in 1:n
        d[perm[j]] = Zvals[colptr[j]]
    end
    return d
end
