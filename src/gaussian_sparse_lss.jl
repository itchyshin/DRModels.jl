# Sparse exact marginal LSS engine for phylogenetic structured scale (#551).
# Fits the Mizuno et al. QQQ model / phylogenetic location–scale–scale:
#     y ~ Xμ βμ + a + e
#     e ~ N(0, diag(σ_e,i²)),      log σ_e,i = (Xσ βσ)ᵢ
#     a ~ N(0, D_a K D_a),          log σ_a,g = (Zg α)g,  D_a = diag(σ_a)
# using the augmented tree precision Q (O(p) nnz, root-conditioned) and
# Takahashi selected inverse.
#
# Exact GMRF sum-of-squares quadratic form avoids Woodbury cancellation.
# O(p) evaluation and exact analytic gradients for ML, and sparse Cholesky
# column solves for REML Xμ' V⁻¹ Xμ.

function _fit_phylo_gaussian_lss_sparse(fam::Gaussian, y, Xμ, Xσ, Zg, gidx, G, tree,
                                        nmμ, nmσ, nmsd, grp, g_tol;
                                        block::Symbol = :sd_phylo, reml::Bool = false)
    phy = tree isa AbstractString ? augmented_phy(tree) : tree
    phy.n_leaves == G ||
        error("phylo($grp): tree has $(phy.n_leaves) tips but `$grp` has $G levels")
    Q, leaf_pos, q = augmented_tree_precision(phy)
    Qs = dropzeros!(sparse(Symmetric(Q, :U)))
    chQ = cholesky(Symmetric(Qs); check = false)
    issuccess(chQ) ||
        error("phylo($grp): root-conditioned augmented precision is not PD")
    logdetCprior = -logdet(chQ)
    Qinv = takahashi_selinv(chQ)
    leaf_var = [Qinv[leaf_pos[t], leaf_pos[t]] for t in 1:G]
    inv_sd = [1.0 / sqrt(leaf_var[t]) for t in 1:G]

    n = length(y)
    pμ = size(Xμ, 2); pσ = size(Xσ, 2); psd = size(Zg, 2)
    np = pμ + pσ + psd
    iβμ = 1:pμ
    iβσ = (pμ+1):(pμ+pσ)
    iα = (pμ+pσ+1):np

    # Template sparsity pattern for H = Q + diag(Z'WZ)
    diag_template = zeros(q)
    for i in 1:n
        diag_template[leaf_pos[gidx[i]]] += 1.0
    end
    H_template = Qs + Diagonal(diag_template)
    Hzero = 0.0 .* H_template
    chol_ref = Ref(cholesky(Symmetric(H_template); check = false))
    issuccess(chol_ref[]) ||
        error("sparse LSS phylo: template Cholesky failed (non-PD pattern)")

    function eval_core(βμ, βσ, α; want_grad::Bool, use_ref::Bool = true)
        ημ = Xμ * βμ
        ησ = Xσ * βσ
        ησa = Zg * α
        w = exp.(-2 .* ησ)
        σa = exp.(ησa)
        wts_g = σa .* inv_sd
        wts = [wts_g[gidx[i]] for i in 1:n]
        r = y .- ημ

        diag_ZtWZ = zeros(q)
        for i in 1:n
            r_node = leaf_pos[gidx[i]]
            diag_ZtWZ[r_node] += w[i] * wts[i]^2
        end

        H = Qs + Diagonal(diag_ZtWZ) + Hzero
        ch = if use_ref
            try
                cholesky!(chol_ref[], H; check = false); chol_ref[]
            catch
                cholesky(Symmetric(H); check = false)
            end
        else
            cholesky(Symmetric(H); check = false)
        end
        issuccess(ch) || return (1e18, Float64[], r, zeros(q), zeros(n), w, wts, diag_ZtWZ, ch, false)

        b = zeros(q)
        for i in 1:n
            r_node = leaf_pos[gidx[i]]
            b[r_node] += wts[i] * w[i] * r[i]
        end

        â = ch \ b
        Zâ = [wts[i] * â[leaf_pos[gidx[i]]] for i in 1:n]
        res_lat = r .- Zâ

        logdetH = logdet(ch)
        logdetD = -sum(log, w)
        logdetV = logdetD + logdetCprior + logdetH

        quad_sos = dot(res_lat, w .* res_lat) + dot(â, Qs * â)
        nll_ml = 0.5 * (logdetV + quad_sos + n * log(2π))
        isfinite(nll_ml) || return (1e18, Float64[], r, â, Zâ, w, wts, diag_ZtWZ, ch, false)

        want_grad || return (nll_ml, Float64[], r, â, Zâ, w, wts, diag_ZtWZ, ch, true)

        # Takahashi selected inverse for exact O(p) analytic gradients
        Hinv = takahashi_selinv(ch)
        u = w .* res_lat

        g_βμ = -(Xμ' * u)

        g_βσ = zeros(pσ)
        for i in 1:n
            r_node = leaf_pos[gidx[i]]
            s_ii = Hinv[r_node, r_node]
            zHz_ii = wts[i]^2 * s_ii
            vinv_ii = w[i] - w[i]^2 * zHz_ii
            dη = vinv_ii / w[i] - (u[i]^2) / w[i]
            for k in 1:pσ
                g_βσ[k] += Xσ[i, k] * dη
            end
        end

        # DRModels.jl#627: accumulate the group quadratic term by a SCATTER over the n
        # observations instead of a GATHER that rescanned all n rows once per
        # group.  The old nested loop was O(G*n); it dominated every gradient
        # call at whole-tree scale (measured 380.0 ms per gradient at G = 16,384
        # against an 8.5 ms objective, and growing 4x per doubling of G) and so
        # made gradient-bound work -- profile confidence intervals above all --
        # impractical there while the objective itself remained O(p).  The
        # multi-component route already used this form
        # (`_lss_sparse_multi_objective_and_grad`).
        #
        # The rewrite is bit-for-bit identical, not merely equivalent, and both
        # halves of that matter: each `quad_g[g]` still accumulates from 0.0 over
        # exactly the same observations in the same ascending-`i` order the
        # gather used, and the subtraction is still the single expression
        # `trace - quad`, so no floating-point sum is reassociated.  Verified on
        # the fixtures in `test_lss_sparse_gradient_scaling.jl`: every profile CI
        # endpoint is unchanged in its last bit.
        quad_g = zeros(G)
        for i in 1:n
            quad_g[gidx[i]] += u[i] * Zâ[i]
        end
        d_g = zeros(G)
        for g in 1:G
            r_node = leaf_pos[g]
            s_node = Hinv[r_node, r_node]
            ztwz_node = diag_ZtWZ[r_node]
            trace_term = ztwz_node * s_node
            d_g[g] = trace_term - quad_g[g]
        end
        g_α = Zg' * d_g

        grad = vcat(g_βμ, g_βσ, g_α)
        all(isfinite, grad) || return (1e18, Float64[], r, â, Zâ, w, wts, diag_ZtWZ, ch, false)

        return (nll_ml, grad, r, â, Zâ, w, wts, diag_ZtWZ, ch, true)
    end

    function eval_reml(βμ, βσ, α; use_ref::Bool = true)
        nll_ml, _, r, â, Zâ, w, wts, diag_ZtWZ, ch, ok = eval_core(βμ, βσ, α; want_grad = false, use_ref = use_ref)
        ok || return 1e18
        XtVinvX = zeros(pμ, pμ)
        for k in 1:pμ
            Xk = @view Xμ[:, k]
            wXk = w .* Xk
            bX = zeros(q)
            for i in 1:n
                bX[leaf_pos[gidx[i]]] += wts[i] * wXk[i]
            end
            âX = ch \ bX
            ZâX = [wts[i] * âX[leaf_pos[gidx[i]]] for i in 1:n]
            VinvXk = w .* (Xk .- ZâX)
            for j in 1:pμ
                XtVinvX[j, k] = dot(@view(Xμ[:, j]), VinvXk)
            end
        end
        chX = cholesky(Symmetric(XtVinvX); check = false)
        issuccess(chX) || return nll_ml + REML_NONPD_PENALTY
        return nll_ml + 0.5 * logdet(chX) - 0.5 * pμ * log(2π)
    end

    unpack(θ) = (θ[iβμ], θ[iβσ], θ[iα])
    nll_ml_only(θ) = eval_core(unpack(θ)...; want_grad = false, use_ref = false)[1]
    # Profile calls can run concurrently for distinct coefficients.  Do not use
    # the optimiser's mutable `chol_ref`: each stored-gradient evaluation gets
    # its own factor and scratch arrays through `use_ref = false`.
    function nllgrad!(g, θ)
        _, grad, _, _, _, _, _, _, _, ok =
            eval_core(unpack(θ)...; want_grad = true, use_ref = false)
        if ok && length(grad) == length(g)
            copyto!(g, grad)
        else
            # A finite 1e18 objective sentinel plus a zero gradient can look
            # stationary to a stored-gradient profile optimizer.  Make the
            # callback failure explicit so the nuisance solve is rejected.
            fill!(g, NaN)
        end
        return g
    end
    nll_reml_only(θ) = eval_reml(unpack(θ)...; use_ref = false)

    βμ0 = Xμ \ y; res0 = y - Xμ * βμ0
    s0 = std(res0)
    θ0 = zeros(np)
    θ0[iβμ] .= βμ0
    θ0[first(iβσ)] = log(s0 / sqrt(2) + eps())
    θ0[iα] .= Zg \ fill(log(s0 / sqrt(2) + eps()), G)

    local res, θ̂, Vcov

    if !reml
        function fg!(F, Gout, θ)
            nll, grad, _, _, _, _, _, _, _, ok = eval_core(unpack(θ)...; want_grad = true, use_ref = true)
            if Gout !== nothing
                ok ? copyto!(Gout, grad) : fill!(Gout, 0.0)
            end
            return nll
        end
        od = Optim.NLSolversBase.only_fg!(fg!)
        res = Optim.optimize(od, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol))
        θ̂ = Optim.minimizer(res)

        # Finite differences on exact analytic gradient for variance-covariance matrix
        grad_at(θ) = eval_core(unpack(θ)...; want_grad = true, use_ref = false)[2]
        Hmat, hess_ok = _fd_hessian_from_grad(grad_at, θ̂)
        # A failed probe (empty gradient even after the retry) reports the NaN vcov, not a crash.
        Vcov = hess_ok ? _vcov_from_hessian(Hmat; context = "sparse LSS phylo") : fill(NaN, np, np)
    else
        res = Optim.optimize(nll_reml_only, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :finite)
        θ̂ = Optim.minimizer(res)

        # Finite difference Hessian for REML (value-based). A probe that is non-finite or hits
        # the 1e18 failure sentinel reports the NaN vcov, not a garbage finite Hessian.
        Hmat, hess_ok = _fd_hessian_from_values(nll_reml_only, θ̂)
        Vcov = hess_ok ? _vcov_from_hessian(Hmat; context = "sparse LSS phylo REML") : fill(NaN, np, np)
    end

    # Random effects (BLUPs)
    # #631: the destructured latent mean MUST NOT be named `â`.  `eval_core`
    # assigns `â` internally, so a second assignment to that name in THIS
    # enclosing scope makes it ONE shared `Core.Box` for the closure and the
    # function body -- the same defect class as #549's `V`.  `nll_ml_only` and
    # `nllgrad!` are stored on the fit and called CONCURRENTLY by a threaded
    # profile (`confint(...; method = :profile, threads = true)` runs the lower
    # and upper endpoint arms in parallel), so both arms wrote the same box:
    # measured on a 600-tip sparse LSS fit, 122/300 concurrent objective pairs
    # disagreed with the serial value by up to 2.0e-2 in the NLL and 7.4e-1 in
    # the gradient, which starved the nuisance solve of convergence and made the
    # endpoint search report a failed arm (returned as +/-Inf).  Bare CHOLMOD and
    # BLAS were exonerated on the same matrices; the box was the whole cause.
    βμ_hat, βσ_hat, α_hat = unpack(θ̂)
    _, _, _, â_hat, _, _, _, _, _, _ = eval_core(βμ_hat, βσ_hat, α_hat; want_grad = false, use_ref = false)
    σa_hat = exp.(Zg * α_hat)
    u_phylo = [σa_hat[t] * inv_sd[t] * â_hat[leaf_pos[t]] for t in 1:G]
    re_dict = Dict(Symbol(grp) => u_phylo)

    blocks = [:mu => iβμ, :sigma => iβσ, block => iα]
    names = [:mu => nmμ, :sigma => nmσ, block => nmsd]
    means = Dict(:mu => Xμ * θ̂[iβμ])
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xσ * θ̂[iβσ]))

    fit = _withranef(_withnll(DrmFit(fam, blocks, names, θ̂, Vcov, -nll_ml_only(θ̂), n,
                                     Optim.converged(res), means, obs, scales), nll_ml_only,
                                reml ? nothing : nllgrad!), re_dict)
    if reml
        return _withreml(fit, -nll_reml_only(θ̂), -nll_ml_only(θ̂))
    end
    return fit
end

# Alias for compatibility
const _fit_structured_gaussian_lss_sparse = _fit_phylo_gaussian_lss_sparse

# ---------------------------------------------------------------------------
# #563 S7b.1 — sparse MULTI-component block assembly + objective.
#
# Generalises the single-phylo-component engine above
# (`_fit_phylo_gaussian_lss_sparse`) to several `sd()` components in one fit,
# the way `#555`'s dense multi-component engine (`_fit_gaussian_lss_multi`,
# `gaussian_lss.jl:564`) already does — following the block-precision
# construction (`H = blockdiag(Q_k) + ZᵀWZ`) `gaussian_structured.jl`'s
# `_fit_two_structured_gaussian_sparse_spec` already uses for TWO fixed-
# variance structured MEAN components, generalised here to `m` components
# whose per-group SDs are themselves modelled (LSS, `sd(g) ~ …`, as #551
# does for one). See
# `docs/src/developer-notes/lss-sparse-multi-component.md` (§2/§3) for the
# reviewed design.
#
# Scope for THIS sub-slice (assembly + objective ONLY, evaluated at a fixed
# θ — no fit loop yet):
# - the block augmented precision H = blockdiag(Q_1,…,Q_m) + ZᵀWZ (§2),
#   including the genuine cross-component blocks (Z_k)ᵀ W Z_l the design
#   note introduces — `hcat`-ing the per-component sparse incidence columns
#   into one `Z = [Z_1|…|Z_m]` and forming `Zᵀ diag(W) Z` in one sparse
#   product assembles those cross blocks automatically, so no separate
#   cross-block loop is needed;
# - the per-component `b_c = Z_cᵀ W r` accumulation into a shared `b`
#   (`Z' * (w .* r)` sums every component's contribution at its own block
#   offset in one product, generalising the `b` loop at :75–79);
# - the ML objective at that fixed θ (logdet via the Cholesky factor, the
#   zero-cancellation quadratic form, and the per-component prior logdets,
#   generalising :85–90).
#
# NOT in scope here (design note §7, sub-slices S7b.2–S7b.4): the exact
# analytic gradient (the diagonal-only trace-term formula the note first
# drafted is WRONG for the cross-component case — §3/§8 finding 4 — so no
# gradient is attempted here at all), the REML correction (§4), and
# `drm()`/router wiring. `_lss_sparse_multi_objective` only evaluates a
# supplied θ; it does not fit. Because there is no repeated-evaluation loop
# yet in this sub-slice, the block Cholesky is (re)factorised fresh each
# call — the symbolic-factor-reuse-via-`cholesky!` idiom `#551` and
# `gaussian_structured.jl` use inside their optimiser closures only pays for
# itself once a fit loop calls this repeatedly (S7b.4); building that
# machinery here with no caller to benefit from it would be premature.

# One component of a sparse multi-component LSS fit. `Qs` is the component's
# sparse prior precision: the root-conditioned augmented tree precision for a
# phylogenetic component (generalising `Qs` at :20 above), or `I_G` for an
# iid component, since `K_c = I` (design note §1). `leaf_pos` maps a group
# level to its row within the component's own latent block; `inv_sd` is the
# Takahashi leaf-SD row-scaling (the `D_a T` row-scaling generalised, :53) —
# ≡ 1 for iid, since an iid group has no cross-group correlation to absorb.
struct _SparseLssComp
    gidx::Vector{Int}
    G::Int
    Zg::Matrix{Float64}
    q::Int
    leaf_pos::Vector{Int}
    Qs::SparseMatrixCSC{Float64,Int}
    inv_sd::Vector{Float64}
    logdetCprior::Float64
end

"""
    _sparse_lss_phylo_comp(gidx, G, Zg, tree) -> _SparseLssComp

The phylogenetic block of a sparse multi-component LSS fit: the same
root-conditioned augmented tree precision and Takahashi leaf-SD row-scaling
`_fit_phylo_gaussian_lss_sparse` builds for a single component (:16–27).
"""
function _sparse_lss_phylo_comp(gidx::AbstractVector{<:Integer}, G::Int, Zg::AbstractMatrix, tree)
    phy = tree isa AbstractString ? augmented_phy(tree) : tree
    phy.n_leaves == G ||
        error("phylo component: tree has $(phy.n_leaves) tips but the grouping has $G levels")
    Q, leaf_pos, q = augmented_tree_precision(phy)
    Qs = dropzeros!(sparse(Symmetric(Q, :U)))
    chQ = cholesky(Symmetric(Qs); check = false)
    issuccess(chQ) || error("sparse LSS multi (phylo): augmented precision is not PD")
    logdetCprior = -logdet(chQ)
    Qinv = takahashi_selinv(chQ)
    leaf_var = [Qinv[leaf_pos[t], leaf_pos[t]] for t in 1:G]
    inv_sd = [1.0 / sqrt(leaf_var[t]) for t in 1:G]
    return _SparseLssComp(collect(Int, gidx), G, Matrix{Float64}(Zg), q, leaf_pos, Qs, inv_sd, logdetCprior)
end

"""
    _sparse_lss_iid_comp(gidx, G, Zg) -> _SparseLssComp

An iid block: `K_c = I_G` (design note §1), so the latent state IS the group
level (`leaf_pos = 1:G`), prior precision `I_G`, and `inv_sd ≡ 1`.
"""
function _sparse_lss_iid_comp(gidx::AbstractVector{<:Integer}, G::Int, Zg::AbstractMatrix)
    Qs = sparse(1.0 * I, G, G)
    return _SparseLssComp(collect(Int, gidx), G, Matrix{Float64}(Zg), G, collect(1:G), Qs, ones(G), 0.0)
end

"""
    _lss_sparse_multi_assemble(θ, y, Xμ, Xσ, comps::Vector{_SparseLssComp})

Block assembly at a fixed `θ = [βμ; βσ; α_1; …; α_m]` (`comps` order):
`H = blockdiag(Q_1,…,Q_m) + ZᵀWZ` (design note §2), plus the joint sparse
incidence `Z = [Z_1|…|Z_m]`, the residual precision `w`, and the mean
residual `r`. Returns a `NamedTuple` `(H, Q_all, Z, w, r)`; `H` is what a
caller checks the O(p) fill claim (§2.1/§2.3) or PD-ness against.
"""
function _lss_sparse_multi_assemble(θ::AbstractVector, y::AbstractVector, Xμ::AbstractMatrix,
                                    Xσ::AbstractMatrix, comps::Vector{_SparseLssComp})
    n = length(y)
    pμ = size(Xμ, 2); pσ = size(Xσ, 2)
    m = length(comps)
    psds = [size(c.Zg, 2) for c in comps]
    offα = pμ + pσ .+ cumsum([0; psds])

    βμ = θ[1:pμ]; βσ = θ[(pμ+1):(pμ+pσ)]
    ημ = Xμ * βμ; ησ = Xσ * βσ
    w = exp.(-2 .* ησ)
    r = y .- ημ

    Zcols = Vector{SparseMatrixCSC{Float64,Int}}(undef, m)
    for (ci, c) in enumerate(comps)
        α = θ[(offα[ci]+1):offα[ci+1]]
        σa = exp.(c.Zg * α)
        wtsc = (σa .* c.inv_sd)[c.gidx]
        Zcols[ci] = _sparse_incidence(c.leaf_pos[c.gidx], n, c.q, wtsc)
    end
    Z = hcat(Zcols...)

    Q_all = blockdiag((c.Qs for c in comps)...)
    ZtWZ = dropzeros!(sparse(Symmetric(Z' * Diagonal(w) * Z)))
    H = Q_all + ZtWZ
    return (H = H, Q_all = Q_all, Z = Z, w = w, r = r)
end

"""
    _lss_sparse_multi_objective(θ, y, Xμ, Xσ, comps::Vector{_SparseLssComp};
                                reml::Bool = false) -> Float64

Negative Gaussian ML (or REML, `reml = true`) log-likelihood of a sparse
multi-component LSS model at a fixed `θ`, via the block assembly of
[`_lss_sparse_multi_assemble`](@ref): logdet through the Cholesky factor of
`H`, the zero-cancellation GMRF sum of squares (`:89` generalised:
`dot(â, Qs*â)` becomes `dot(â, Q_all*â) = Σ_c dot(â_c, Q_c*â_c)`, §3), and the
per-component prior logdets. Structural pattern (nested vs. crossed fill,
§2.1) and PD-ness are checked directly against `_lss_sparse_multi_assemble`'s
`H`, not through this scalar.

**REML (#563 S7b.3, design note §4).** `reml = true` adds the Patterson–
Thompson correction `+ 0.5*logdet(Xμ'V⁻¹Xμ) - 0.5*pμ*log(2π)` — the SAME
closed form and normalisation convention #558's single-component
`eval_reml` (`:135–156` above) and the dense multi-component route
(`gaussian_lss.jl:634–638`) both use. `Xμ'V⁻¹Xμ` is obtained via `pμ` sparse
Cholesky backsolves against the SAME shared factor `ch` this function already
computes (`_lss_sparse_multi_reml_pieces`, §4's block-accumulation
generalisation of `:138–152`'s per-column loop: `Z' * (w .* Xμ)` sums every
component's contribution into `bX` at its own block offset in one sparse
product, exactly as the ML `b` accumulation does at `:428` above — no
cross-component term, per the design note's own review, §8 finding 5). Cost:
`O(pμ · nnz(L))` on top of the ML evaluation, matching #558's order.
`reml = false` (the default) is the ORIGINAL S7b.1 code path, byte-for-byte
unchanged.

IN SCOPE (#563 S7b.1): the assembly is correct for genuinely crossed
component groupings too (it does not assume nesting) — but the design note's
O(p) fill argument (§2.1/§2.3) only covers one phylogenetic component plus
nested or small (`G_c ≪ p`) iid components; a crossed topology at comparable
scale is numerically correct here but NOT O(p) (§2.3 case (c)).

OUT OF SCOPE for this sub-slice: `drm()`/router wiring (S7b.4) — see the
block comment above this section.
"""
function _lss_sparse_multi_objective(θ::AbstractVector, y::AbstractVector, Xμ::AbstractMatrix,
                                     Xσ::AbstractMatrix, comps::Vector{_SparseLssComp};
                                     reml::Bool = false)
    n = length(y)
    asm = _lss_sparse_multi_assemble(θ, y, Xμ, Xσ, comps)
    H, Q_all, Z, w, r = asm.H, asm.Q_all, asm.Z, asm.w, asm.r
    ch = cholesky(Symmetric(H); check = false)
    issuccess(ch) || return 1e18

    b = Vector(Z' * (w .* r))
    â = ch \ b
    Zâ = Z * â
    res_lat = r .- Zâ

    logdetH = logdet(ch)
    logdetD = -sum(log, w)
    logdetCprior = sum(c.logdetCprior for c in comps)
    logdetV = logdetD + logdetCprior + logdetH

    quad_sos = dot(res_lat, w .* res_lat) + dot(â, Q_all * â)
    nll_ml = 0.5 * (logdetV + quad_sos + n * log(2π))
    isfinite(nll_ml) || return 1e18
    reml || return nll_ml

    corr, _, _, _, chX = _lss_sparse_multi_reml_pieces(Xμ, w, Z, ch)
    (chX !== nothing && isfinite(corr)) || return nll_ml + REML_NONPD_PENALTY
    return nll_ml + corr
end

"""
    _lss_sparse_multi_reml_pieces(Xμ, w, Z, ch) -> (corr, BX, ÂX, Xtilde, chX)

Shared REML machinery (#563 S7b.3, design note §4) for both
[`_lss_sparse_multi_objective`](@ref) and
[`_lss_sparse_multi_objective_and_grad`](@ref): the Patterson–Thompson
correction value `corr = 0.5*logdet(Xμ'V⁻¹Xμ) - 0.5*pμ*log(2π)` and the
intermediates its gradient (below) reuses, computed via the
Woodbury identity `V⁻¹x = w .* (x - Z*(H⁻¹*(Zᵀ*(w.*x))))` applied to every
column of `Xμ` AT ONCE (one `q_total × pμ` sparse backsolve against the
shared factor `ch`, rather than #558's `pμ` separate column solves — same
total cost, `O(pμ · nnz(L))`):

- `BX = Zᵀ(w .* Xμ)` (`q_total × pμ`): the multi-component generalisation of
  `:138–145`'s per-column `bX` accumulation — every component's contribution
  lands at its own block offset in one sparse product, no cross term (§8
  finding 5);
- `ÂX = ch \\ BX` (`q_total × pμ`): the joint BLUP of every `Xμ` column,
  generalising `:146`'s `âX = ch \\ bX`;
- `Xtilde = Xμ - Z*ÂX` (`n × pμ`): generalising `:147–148`'s
  `Xk .- ZâX`;
- `XtVinvX = Xμ'*(w .* Xtilde)` (`pμ × pμ`), `chX = cholesky(Symmetric(XtVinvX))`.

Returns `(NaN, BX, ÂX, Xtilde, nothing)` if `chX` is not PD (caller applies
`REML_NONPD_PENALTY`, `:154` above's convention).
"""
function _lss_sparse_multi_reml_pieces(Xμ::AbstractMatrix, w::AbstractVector,
                                       Z::AbstractMatrix, ch)
    pμ = size(Xμ, 2)
    BX = Matrix(Z' * (w .* Xμ))
    ÂX = ch \ BX
    Xtilde = Xμ .- Z * ÂX
    XtVinvX = Symmetric(Xμ' * (w .* Xtilde))
    chX = cholesky(XtVinvX; check = false)
    issuccess(chX) || return (NaN, BX, ÂX, Xtilde, nothing)
    corr = 0.5 * logdet(chX) - 0.5 * pμ * log(2π)
    return (corr, BX, ÂX, Xtilde, chX)
end

# ---------------------------------------------------------------------------
# #563 S7b.2 / S7b.2b — exact analytic gradient of the sparse multi-component
# LSS objective. See
# `docs/src/developer-notes/lss-sparse-multi-component.md` §3 (alignment
# table) and §8 finding 4 for the reviewed derivation this generalises.

"""
    _lss_sparse_multi_objective_and_grad(θ, y, Xμ, Xσ, comps::Vector{_SparseLssComp};
                                          cross_terms::Bool = true,
                                          reml::Bool = false) -> (nll, grad)

Exact analytic gradient of the sparse multi-component LSS objective
([`_lss_sparse_multi_objective`](@ref)), generalising #551's single-
component gradient (`:99–127` above) to `m` components sharing the block
augmented precision `H` (design note §2). `nll` is obtained by calling
[`_lss_sparse_multi_objective`](@ref) directly, so it is bit-identical to
that entry's own return value; `grad = [g_βμ; g_βσ; g_α_1; …; g_α_m]`.

**The cross-component correction (§3/§8 finding 4).** For a single
component, one observation touches exactly ONE latent node, so `Zᵀ W Z` is
diagonal and `∂H/∂α` only ever perturbs its own diagonal entry (#551's
`s_node = Hinv[r_node, r_node]`). With `m ≥ 2` components, one observation
touches `m` nodes (one per component), so `Zᵀ W Z` gains genuine cross-
component blocks, and `∂H[g,t]/∂α_c,g` is nonzero not only at `H[g,g]` but
at every `H[g,t]` for `t` a node of another component sharing an
observation with `g`. The design note's adversarial review (§8 finding 4)
showed the naive diagonal-only generalisation of #551's formula is WRONG:
on its 6-node nested toy, central-FD `d(logdet H)/dα_iid[1] = 1.400168`,
the diagonal-only formula gives `2.669901` (**91% error**), the corrected
formula (own-node term `Hinv[g,g]·∂H[g,g]/∂α_c,g` plus the cross-component
accumulation `Hinv[g,t]·∂H[g,t]/∂α_c,g` over every node `t` of every other
component sharing an observation with `g`) agrees to `3.4e-10`. Every
`Hinv[g,t]` this needs is already inside the Takahashi selected inverse's
pattern — sparse-Cholesky fill (`L+Lᵀ`) only ever grows a matrix's own
nonzero set, and `H[g,t] ≠ 0` (from the shared observation) implies `g,t`
are already coupled in `H`'s own pattern, hence in `L`'s (see
`takahashi_selinv`, `src/takahashi_selinv.jl:103`) — so the fix costs
nothing extra to fetch.

**A second, related correction not spelled out in the design note's own
alignment table (verified here by central-FD, not merely asserted): the
`quad_term` subtraction also needs a component-LOCAL fix.** The note's
table describes `quad_term` (the `u[i]*Zâ[i]` sum) as "unaffected" by
multi-component, apparently intending the JOINT
`Zâ_i = Σ_c' wts_c'[i]·â[node(i,c')]` `_lss_sparse_multi_objective` itself
computes as `Z*â`. Re-deriving `∂quad_sos/∂α_c,g` from scratch (using the
same GMRF zero-cancellation identity #551 relies on — `â` solves `H·â =
Zᵀ W r`, so `Q_all·â = Zᵀ W(r - Zâ)` and the `∂â/∂α` term vanishes
identically) shows the surviving term is
`-2·â[node_c(g)]·Σ_{i: node_c(i)=g} u_i·wts_c[i]`: it needs component `c`'s
OWN contribution `wts_c[i]·â[node_c(i)]` to row `i`'s fitted value, not the
joint sum over every component sharing that row. Using the joint `Zâ_i`
here silently collapses every component's `quad_term` to the SAME scalar
total whenever every component has a single-column `Zg` (confirmed on a
6-node nested toy mirroring the design note's: both components' `quad_term`
came out numerically identical under the joint formula — impossible, since
they must differ by component in general) — the component-local fix
matches central-FD to `~1e-9` relative at every point checked. Recorded
here because it is exactly the kind of load-bearing spec gap symbolic-
alignment exists to catch before code ships.

**`g_βσ`** needs the SAME generalisation as `g_α,c` for the same structural
reason: `Vinv_ii = w_i - w_i²·zᵢᵀH⁻¹zᵢ` where `zᵢ` (row `i` of the joint
`Z`) has one nonzero per component, so `zᵢᵀH⁻¹zᵢ` is the FULL bilinear form
`Σ_{c,c'} wts_c[i]·wts_c'[i]·Hinv[node_c(i), node_c'(i)]` over every
component pair touched by row `i` (own AND cross), not the single-node
read `Hinv[r_node, r_node]` #551 uses when `m = 1`.

`cross_terms = false` reproduces the WRONG diagonal-only `g_α` formula
(own-node term only, no cross-component accumulation) — a test-only hook
for the regression guard pinning finding 4 (design note §5 oracle 3): with
it, the returned `g_α` block must disagree with the FD gradient by a wide
margin, so a future regression to the pre-review #551-style formula is
caught rather than silently reintroduced. `g_βμ` and `g_βσ` are unaffected
by this flag.

On failure (non-PD `H` or a non-finite objective) returns `(1e18,
Float64[])`, matching #551's sentinel convention.

Complexity: one Takahashi selected inverse per evaluation (`O(nnz(L))`,
design note §2) plus `O(n)` per component for the own-node/quad
accumulation and `O(n)` per unordered component pair for the cross
accumulation — at most `m(m-1)/2` such pairs — so `O(p·n_components)`
overall for fixed component count `m`, the same order as the block
assembly itself (§2).

**REML (#563 S7b.3, design note §4).** `reml = true` adds the exact gradient
of the Patterson–Thompson correction `0.5*logdet(Xμ'V⁻¹Xμ) - 0.5*pμ*log(2π)`
([`_lss_sparse_multi_objective`](@ref)'s REML branch) to `g_βσ` and every
`g_α_c`; `g_βμ` is left UNTOUCHED. This is not a numerical convenience but an
exact identity: `Xμ'V⁻¹Xμ` depends on `θ` only through `V` (hence through
`βσ` and every `α_c`), never through `βμ`, so the correction's `β_μ` partial
is EXACTLY zero and **`β_μ` is profiled out of the REML gradient** — the
returned `g_βμ` block is bit-identical to the `reml = false` (ML) block at
the same `θ` (the same structural contract `reml_q4.jl`/
`gaussian_bivariate.jl` document for their own profiled-out mean
parameters), not merely close under finite differences.

Derivation, via the Schur-complement identity `logdet(Xμ'V⁻¹Xμ) = logdet(C) -
logdet(H)` for the bordered matrix `C = [Xμ'WXμ  Xμ'WZ; Zᵀ WXμ  H]` (so
`Xμ'V⁻¹Xμ` is `C`'s Schur complement w.r.t. `H`, standard REML/mixed-model-
equations fact): writing `ÂX = H⁻¹(Zᵀ W Xμ)` (`q_total × pμ`, the SAME
quantity [`_lss_sparse_multi_reml_pieces`](@ref) computes for the objective),
`M = Xμ'V⁻¹Xμ`, `Minv = M⁻¹` (`pμ × pμ`, cheap — `pμ` is small), and
`C2 = ÂX * Minv` (`q_total × pμ`), the two nonzero θ-derivative blocks of `C`
(`Xμ'WXμ` doesn't depend on θ at all) combine so that:

- `∂corr/∂βσ_k = -Σ_i Xσ[i,k]·w_i·(X̃ᵢ' Minv X̃ᵢ)`, where `X̃ = Xμ - Z*ÂX`
  (`n × pμ`, `Xtilde` in [`_lss_sparse_multi_reml_pieces`](@ref)) — no
  `Hinv`/Takahashi term survives here: the `H`-logdet and `C`-logdet
  contributions that both involve `Hinv[node,node]`-type bilinear forms
  cancel exactly (verified symbolically and against central-FD below);
- `∂corr/∂η_c,g` (per-group-level, chained through `Zg_c` exactly as `g_α_c`
  is): own-node term `diagZtWZ[c][node]·Ψ[ĝ,ĝ] - dot(BX[ĝ,:], C2[ĝ,:])` plus
  the SAME cross-component accumulation pattern the ML `g_α_c` uses (`:687–
  698` above) with `Hinv[gi,gj]` replaced by `Ψ[gi,gj] := dot(ÂX[gi,:],
  C2[gj,:])` (the `(gi,gj)` entry of `ÂX*Minv*ÂX'`, needed only at exactly
  the same sparse `(node, node')` pairs the ML gradient already visits — no
  new sparsity pattern to track), where `BX = Zᵀ(w .* Xμ)`
  (`_lss_sparse_multi_reml_pieces`).

Every quantity above (`ÂX`, `BX`, `Minv`, `Xtilde`) is already produced by a
single call to [`_lss_sparse_multi_reml_pieces`](@ref) — no extra Cholesky
factorisation beyond the one `chX` (`pμ × pμ`, trivial) it already forms.
"""
function _lss_sparse_multi_objective_and_grad(θ::AbstractVector, y::AbstractVector, Xμ::AbstractMatrix,
                                              Xσ::AbstractMatrix, comps::Vector{_SparseLssComp};
                                              cross_terms::Bool = true, reml::Bool = false)
    n = length(y)
    pμ = size(Xμ, 2); pσ = size(Xσ, 2)
    m = length(comps)
    psds = [size(c.Zg, 2) for c in comps]
    offα = pμ + pσ .+ cumsum([0; psds])

    nll_ml = _lss_sparse_multi_objective(θ, y, Xμ, Xσ, comps)
    (isfinite(nll_ml) && nll_ml < 1e18) || return (nll_ml, Float64[])

    βμ = θ[1:pμ]; βσ = θ[(pμ+1):(pμ+pσ)]
    ημ = Xμ * βμ; ησ = Xσ * βσ
    w = exp.(-2 .* ησ)
    r = y .- ημ

    # Node offset of each component's block within the shared H (cumsum of
    # component `q`, the same order `_lss_sparse_multi_assemble` lays `Z`
    # out in), the local latent-node touched by each row, and each row's
    # per-component `wts` scaling.
    offnode = Vector{Int}(undef, m + 1); offnode[1] = 0
    for (ci, c) in enumerate(comps)
        offnode[ci+1] = offnode[ci] + c.q
    end
    wtsC = Vector{Vector{Float64}}(undef, m)
    nodeC = Vector{Vector{Int}}(undef, m)
    for (ci, c) in enumerate(comps)
        α = θ[(offα[ci]+1):offα[ci+1]]
        σa = exp.(c.Zg * α)
        wtsC[ci] = (σa .* c.inv_sd)[c.gidx]
        nodeC[ci] = c.leaf_pos[c.gidx]
    end

    asm = _lss_sparse_multi_assemble(θ, y, Xμ, Xσ, comps)
    H, Q_all, Z = asm.H, asm.Q_all, asm.Z
    ch = cholesky(Symmetric(H); check = false)
    issuccess(ch) || return (1e18, Float64[])

    b = Vector(Z' * (w .* r))
    â = ch \ b
    Zâ = Z * â
    res_lat = r .- Zâ
    u = w .* res_lat

    Hinv = takahashi_selinv(ch)

    g_βμ = -(Xμ' * u)

    # Per component: diag(Zᵀ W Z) block (unchanged from #551) and this
    # component's OWN contribution to each row's fitted Zâ (needed for the
    # quad_term correction above — NOT the joint `Zâ`).
    diagZtWZ = Vector{Vector{Float64}}(undef, m)
    Zâ_own = Vector{Vector{Float64}}(undef, m)
    for (ci, c) in enumerate(comps)
        dz = zeros(c.q)
        zo = zeros(n)
        for i in 1:n
            node = nodeC[ci][i]
            dz[node] += w[i] * wtsC[ci][i]^2
            zo[i] = wtsC[ci][i] * â[offnode[ci] + node]
        end
        diagZtWZ[ci] = dz
        Zâ_own[ci] = zo
    end

    # g_βσ: full bilinear form zᵢᵀ Hinv zᵢ across every component pair (own
    # AND cross) touched by row i.
    g_βσ = zeros(pσ)
    for i in 1:n
        zHz = 0.0
        for ci in 1:m
            gi = offnode[ci] + nodeC[ci][i]
            zHz += wtsC[ci][i]^2 * Hinv[gi, gi]
            for cj in (ci+1):m
                gj = offnode[cj] + nodeC[cj][i]
                zHz += 2 * wtsC[ci][i] * wtsC[cj][i] * Hinv[gi, gj]
            end
        end
        vinv_ii = w[i] - w[i]^2 * zHz
        dη = vinv_ii / w[i] - u[i]^2 / w[i]
        for k in 1:pσ
            g_βσ[k] += Xσ[i, k] * dη
        end
    end

    # g_α_c: own-node trace term (#551) + cross-component trace term (§3/§8
    # finding 4) - quad_term (component-local Zâ, per the correction above).
    d_all = [zeros(c.G) for c in comps]
    for (ci, c) in enumerate(comps)
        for g in 1:c.G
            r_node = c.leaf_pos[g]
            ĝ = offnode[ci] + r_node
            d_all[ci][g] = diagZtWZ[ci][r_node] * Hinv[ĝ, ĝ]
        end
        for i in 1:n
            d_all[ci][c.gidx[i]] -= u[i] * Zâ_own[ci][i]
        end
    end
    if cross_terms
        for ci in 1:m, cj in (ci+1):m
            c_i = comps[ci]; c_j = comps[cj]
            for i in 1:n
                gi = offnode[ci] + nodeC[ci][i]
                gj = offnode[cj] + nodeC[cj][i]
                val = w[i] * wtsC[ci][i] * wtsC[cj][i] * Hinv[gi, gj]
                d_all[ci][c_i.gidx[i]] += val
                d_all[cj][c_j.gidx[i]] += val
            end
        end
    end
    g_αs = [comps[ci].Zg' * d_all[ci] for ci in 1:m]

    grad = vcat(g_βμ, g_βσ, g_αs...)
    all(isfinite, grad) || return (1e18, Float64[])
    reml || return (nll_ml, grad)

    # REML (#563 S7b.3, design note §4) — see the docstring's derivation.
    corr, BX, ÂX, Xtilde, chX = _lss_sparse_multi_reml_pieces(Xμ, w, Z, ch)
    (chX !== nothing && isfinite(corr)) || return (nll_ml + REML_NONPD_PENALTY, Float64[])

    Minv = chX \ Matrix{Float64}(I, pμ, pμ)
    C2 = ÂX * Minv                        # q_total × pμ
    XtildeM = Xtilde * Minv               # n × pμ

    g_βσ_corr = zeros(pσ)
    for i in 1:n
        qi = dot(view(Xtilde, i, :), view(XtildeM, i, :))
        coef = -w[i] * qi
        for k in 1:pσ
            g_βσ_corr[k] += Xσ[i, k] * coef
        end
    end

    d_all_corr = [zeros(c.G) for c in comps]
    for (ci, c) in enumerate(comps)
        for g in 1:c.G
            r_node = c.leaf_pos[g]
            ĝ = offnode[ci] + r_node
            psi_own = dot(view(ÂX, ĝ, :), view(C2, ĝ, :))
            term_a = -dot(view(BX, ĝ, :), view(C2, ĝ, :))
            d_all_corr[ci][g] = diagZtWZ[ci][r_node] * psi_own + term_a
        end
    end
    for ci in 1:m, cj in (ci+1):m
        c_i = comps[ci]; c_j = comps[cj]
        for i in 1:n
            gi = offnode[ci] + nodeC[ci][i]
            gj = offnode[cj] + nodeC[cj][i]
            psi_cross = dot(view(ÂX, gi, :), view(C2, gj, :))
            val = w[i] * wtsC[ci][i] * wtsC[cj][i] * psi_cross
            d_all_corr[ci][c_i.gidx[i]] += val
            d_all_corr[cj][c_j.gidx[i]] += val
        end
    end
    g_αs_corr = [comps[ci].Zg' * d_all_corr[ci] for ci in 1:m]

    grad_reml = copy(grad)
    grad_reml[(pμ+1):(pμ+pσ)] .+= g_βσ_corr
    for ci in 1:m
        grad_reml[(offα[ci]+1):offα[ci+1]] .+= g_αs_corr[ci]
    end
    all(isfinite, grad_reml) || return (1e18, Float64[])
    return (nll_ml + corr, grad_reml)
end

# ---------------------------------------------------------------------------
# #563 S7b.4 — public-route wiring: the sparse multi-component OPTIMISER loop
# (`_fit_gaussian_lss_sparse_multi`) and a route marker, called from the
# router in `gaussian_lss.jl` (`_drm_gaussian_lss_multi`, D-206). See
# `docs/src/developer-notes/lss-sparse-multi-component.md` §1 (router rule)
# and §5 (acceptance oracles 1/2/4).
#
# ROUTE MARKER. `DrmFit` is a plain positional struct with an 11-/19-/22-arg
# constructor ladder (`gaussian_core.jl:161-247`) specifically so that ~70
# call sites across ~20 family files never have to change when a new field is
# added; adding a `route` field there is out of scope for this sub-slice
# (edits here are confined to `gaussian_sparse_lss.jl` and the route-selection
# lines of `gaussian_lss.jl`). Record the router's decision in a small side
# table instead, keyed by `fit.theta` — NOT by `fit` itself: `DrmFit` is
# IMMUTABLE, and every `_with*` wrapper (`_withformula`, `_withnll`,
# `_withranef`, `_withreml`, `gaussian_core.jl:204-247`) RECONSTRUCTS a new
# `DrmFit` from the old one's fields rather than mutating it — `drm()` itself
# calls `_withformula` on the router's return value, so the object the router
# marks is never the object the caller receives (confirmed by running this
# file's own tests: marking `fit` directly always read back `:unknown`).
# `fit.theta` (`θ̂`, a `Vector{Float64}`) is the one field every wrapper
# forwards BY REFERENCE, never copies, so its identity survives the whole
# chain intact — and, being a genuinely mutable heap object, it also accepts
# a `WeakKeyDict` finalizer (unlike `DrmFit` itself), so a marked fit does
# not have to be kept alive by this table past its normal lifetime.
const _LSS_MULTI_ROUTE = WeakKeyDict{Vector{Float64},Symbol}()

"""
    _mark_lss_multi_route!(fit, route::Symbol) -> fit

Record which engine the sd() multi-component router chose for `fit` —
`:sparse_multi` or `:dense_multi` — keyed by `fit.theta`'s object identity
(see the block comment above for why) — and return `fit` unchanged.
Test-only bookkeeping (see `_lss_multi_route`); never read by any
fitting code.
"""
function _mark_lss_multi_route!(fit, route::Symbol)
    _LSS_MULTI_ROUTE[fit.theta] = route
    return fit
end

"""
    _lss_multi_route(fit) -> Symbol

Which engine the sd() multi-component router (#563 S7b.4, D-206) selected for
`fit`: `:sparse_multi` or `:dense_multi`. Returns `:unknown` for any fit the
multi-component router did not produce (the single-component `sd()`/
`sd_phylo()` routes, or any non-LSS fit) — those are not tracked here.
"""
_lss_multi_route(fit) = get(_LSS_MULTI_ROUTE, fit.theta, :unknown)

"""
    _fit_gaussian_lss_sparse_multi(fam, y, Xμ, Xσ, comps, comp_nm, comp_is_phylo,
                                   comp_label, nmμ, nmσ, g_tol; reml = false) -> DrmFit

The sparse multi-component analogue of `_fit_phylo_gaussian_lss_sparse`
(:13-263 above), generalising its LBFGS-on-the-exact-gradient optimiser loop
from one phylogenetic component to the `comps::Vector{_SparseLssComp}` block
this sub-slice's S7b.1-S7b.3 machinery already evaluates
([`_lss_sparse_multi_objective_and_grad`](@ref)). `comp_nm`/`comp_is_phylo`/
`comp_label` are PARALLEL metadata (one entry per `comps` element, same
order) carrying what `_SparseLssComp` itself does not — the coefficient
names, whether a component is the (at most one) phylogenetic block, and its
grouping-factor label — kept OUTSIDE `_SparseLssComp` deliberately: that
struct is the numeric machinery S7b.1-S7b.3 already exercise and pin by
tests, and adding label/name fields to it would be a change to code this
sub-slice does not need to touch.

Unlike the single-component route's REML branch (`eval_reml`, which falls
back to `autodiff = :finite` because #551 never built an analytic REML
gradient — design note §8 finding 5), S7b.3 DID build one
(`_lss_sparse_multi_objective_and_grad(...; reml = true)`), so both `reml =
false` and `reml = true` here use the SAME LBFGS-on-the-exact-gradient loop
and the SAME finite-difference-of-the-analytic-gradient Hessian for `vcov`
(mirroring :199-217's ML branch, not :219-240's REML branch) — there is no
`autodiff = :finite` fallback in this route at all.

Convergence contract, `DrmFit` fields, and the `nllgrad!`-only-for-ML /
`nothing`-for-REML convention (`test_lss_sparse.jl`'s
`_profile_autodiff_mode` check, `:stored` vs `:finite`) are exactly
`_fit_phylo_gaussian_lss_sparse`'s.
"""
function _fit_gaussian_lss_sparse_multi(fam::Gaussian, y, Xμ, Xσ, comps::Vector{_SparseLssComp},
                                        comp_nm::Vector{Vector{String}}, comp_is_phylo::Vector{Bool},
                                        comp_label::Vector{String}, nmμ, nmσ, g_tol; reml::Bool = false)
    n = length(y)
    pμ = size(Xμ, 2); pσ = size(Xσ, 2)
    m = length(comps)
    psds = [size(c.Zg, 2) for c in comps]
    offα = pμ + pσ .+ cumsum([0; psds])
    np = offα[end]

    nll_ml_only(θ) = _lss_sparse_multi_objective(θ, y, Xμ, Xσ, comps; reml = false)
    nll_reml_only(θ) = _lss_sparse_multi_objective(θ, y, Xμ, Xσ, comps; reml = true)
    grad_at(θ; use_reml::Bool = reml) =
        _lss_sparse_multi_objective_and_grad(θ, y, Xμ, Xσ, comps; reml = use_reml)[2]

    function nllgrad!(g, θ)
        _, grad = _lss_sparse_multi_objective_and_grad(θ, y, Xμ, Xσ, comps; reml = false)
        if length(grad) == length(g)
            copyto!(g, grad)
        else
            fill!(g, NaN)   # see :169-173's rationale: never look stationary on failure
        end
        return g
    end

    βμ0 = Xμ \ y; res0 = y - Xμ * βμ0
    s0 = std(res0)
    θ0 = zeros(np)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(s0 / sqrt(m + 1) + eps())
    for (ci, c) in enumerate(comps)
        θ0[offα[ci]+1:offα[ci+1]] .= c.Zg \ fill(log(s0 / sqrt(m + 1) + eps()), c.G)
    end

    function fg!(F, Gout, θ)
        nll, grad = _lss_sparse_multi_objective_and_grad(θ, y, Xμ, Xσ, comps; reml = reml)
        if Gout !== nothing
            if length(grad) == length(Gout)
                copyto!(Gout, grad)
            else
                fill!(Gout, 0.0)
            end
        end
        return nll
    end
    od = Optim.NLSolversBase.only_fg!(fg!)
    res = Optim.optimize(od, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol))
    θ̂ = Optim.minimizer(res)

    # Finite differences on the exact analytic gradient (ML or REML, per
    # `reml`) for the variance-covariance matrix — see the docstring for why
    # this differs from #551's REML branch.
    Hmat, hess_ok = _fd_hessian_from_grad(grad_at, θ̂)
    Vcov = hess_ok ?
        _vcov_from_hessian(Hmat; context = reml ? "sparse LSS multi REML" : "sparse LSS multi") :
        fill(NaN, np, np)

    # Random effects (BLUPs): joint â at θ̂, then each component's own
    # contribution at its own block offset — generalising :244-248's
    # `u_phylo` to every component, iid or phylogenetic (`inv_sd ≡ 1` for
    # iid, design note §1).
    asm = _lss_sparse_multi_assemble(θ̂, y, Xμ, Xσ, comps)
    ch_final = cholesky(Symmetric(asm.H); check = false)
    â = issuccess(ch_final) ? (ch_final \ Vector(asm.Z' * (asm.w .* asm.r))) : fill(NaN, size(asm.H, 1))
    offnode = Vector{Int}(undef, m + 1); offnode[1] = 0
    for (ci, c) in enumerate(comps)
        offnode[ci+1] = offnode[ci] + c.q
    end
    re_dict = Dict{Symbol,Vector{Float64}}()
    for (ci, c) in enumerate(comps)
        α = θ̂[offα[ci]+1:offα[ci+1]]
        σc = exp.(c.Zg * α)
        re_dict[Symbol(comp_label[ci])] =
            [σc[g] * c.inv_sd[g] * â[offnode[ci] + c.leaf_pos[g]] for g in 1:c.G]
    end

    iid_ci = [ci for ci in 1:m if !comp_is_phylo[ci]]
    phy_ci = [ci for ci in 1:m if comp_is_phylo[ci]]
    blocks = Pair{Symbol,UnitRange{Int}}[:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ)]
    names = Pair{Symbol,Vector{String}}[:mu => nmμ, :sigma => nmσ]
    # Contiguous-block assumption (mirrors the dense multi route,
    # `gaussian_lss.jl:659-669`): the ROUTER lays `comps` out iid-then-phylo,
    # so the iid α's occupy one contiguous slice of θ.
    if !isempty(iid_ci)
        lo = offα[first(iid_ci)] + 1; hi = offα[last(iid_ci)+1]
        nm = String[]
        for ci in iid_ci
            append!(nm, length(iid_ci) > 1 ? string.(comp_label[ci], ": ", comp_nm[ci]) : comp_nm[ci])
        end
        push!(blocks, :sd => lo:hi); push!(names, :sd => nm)
    end
    if !isempty(phy_ci)
        ci = only(phy_ci)
        push!(blocks, :sd_phylo => (offα[ci]+1):offα[ci+1])
        push!(names, :sd_phylo => comp_nm[ci])
    end
    means = Dict(:mu => Xμ * θ̂[1:pμ])
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]))

    fit = _withranef(_withnll(DrmFit(fam, blocks, names, θ̂, Vcov, -nll_ml_only(θ̂), n,
                                     Optim.converged(res), means, obs, scales), nll_ml_only,
                                reml ? nothing : nllgrad!), re_dict)
    if reml
        return _withreml(fit, -nll_reml_only(θ̂), -nll_ml_only(θ̂))
    end
    return fit
end
