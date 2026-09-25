# gaussian_structured.jl — Gaussian structured random effects on the mean with a
# KNOWN relatedness matrix. A structured intercept u ~ N(0, σ_s² K) leaves the
# marginal exactly Gaussian:
#     y ~ N(Xβ, D + σ_s² Z K Zᵀ),   D = diag(σ_i²),  Z the group indicator,
# fit in closed form (PGLS-style) via the matrix-determinant lemma + Woodbury:
#   logdet(V) = logdet(D) + G·log σ_s² + logdet(K) + logdet(M),
#   rᵀV⁻¹r    = rᵀD⁻¹r − Cᵀ M⁻¹ C,   M = (1/σ_s²)K⁻¹ + ZᵀD⁻¹Z (diagonal part).
# `relmat(1 | id)` supplies K directly; `animal()` / `phylo()` / `spatial()`
# reuse this engine with K from a pedigree / tree / coordinates.

using LinearAlgebra: cholesky, cholesky!, Symmetric, Diagonal, dot, logdet, inv, diag, I, issuccess,
    eigen, SymTridiagonal
using SparseArrays: SparseArrays, SparseMatrixCSC, sparse, nonzeros, rowvals,
    nzrange, blockdiag, dropzeros!

"""
    relmat(1 | id)

Structured random-intercept marker with a user-supplied relatedness matrix:

```julia
drm(bf(y ~ x + relmat(1 | id), sigma ~ 1), Gaussian(); data, K = K)
```

`K` is the correlation/relatedness matrix over the levels of `id`, ordered as
they first appear in `data`. The marginal stays Gaussian (closed-form).
"""
relmat(x) = x   # marker; intercepted during formula parsing

"""
    animal(1 | id)

Animal-model structured random intercept. Supply the additive-relatedness matrix
over the levels of `id` via `drm(...; A = A)`. Reuses the closed-form
structured-Gaussian engine (same as [`relmat`](@ref)).
"""
animal(x) = x

"""
    phylo(1 | species)

Phylogenetic structured random intercept on the Gaussian **mean**: pass the tree
via `drm(...; tree = tree)` (an `AugmentedPhy` from `random_balanced_tree` /
`augmented_phy`, or a Newick string). The phylogenetic correlation is built from
the tree (`sigma_phy_dense`) and the marginal is fit in closed form. (The q=4
phylogenetic *location-scale* model — a structured effect on `log σ` too — uses
the verified sparse-Laplace engine instead; see `HANDOVER.md`.)
"""
phylo(x) = x

# Phylogenetic correlation from a tree (AugmentedPhy or Newick string).
function _phylo_correlation(tree)
    phy = tree isa AbstractString ? augmented_phy(tree) : tree
    C = sigma_phy_dense(phy; σ²_phy = 1.0)
    d = sqrt.(diag(C))
    return C ./ (d * d')
end

"""
    spatial(1 | site)

Coordinate-spatial structured random intercept on the Gaussian mean. Pass site
coordinates via `drm(...; coords = coords)` (a `G×2` matrix, one row per `site`
level in first-seen order). The spatial correlation `K(ρ) = exp(-d / ρ)` is built
from pairwise distances and the range `ρ` is estimated jointly. Closed-form
Gaussian marginal (K is rebuilt each evaluation since it depends on `ρ`).
"""
spatial(x) = x

function _fit_structured_gaussian(fam::Gaussian, y, Xμ, Xσ, gidx, G, K, nmμ, nmσ, grp, g_tol)
    n = length(y)
    pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    # #548: with a NON-CONSTANT residual scale the Woodbury quadratic below
    # (`q1 - dot(C, Mfac \ C)`) loses every digit as σ_e,i → 0 — measured, it
    # returned logLik +1.1e105 with `converged = true` at a point whose true dense
    # value was −1873.48 (drmTMB: −70.38). Route that case through the stable
    # dense assembly added for #545, which reproduces drmTMB to 7 s.f. The
    # homoscedastic path (pσ == 1) is the verified one and is left untouched.
    if pσ > 1
        return _fit_structured_gaussian_lss(fam, y, Xμ, Xσ, ones(G, 1), gidx, G, K,
                                            nmμ, nmσ, [String(grp)], grp, g_tol;
                                            block = :resd)
    end
    Kfac = cholesky(Symmetric(K))
    Kinv = inv(Kfac)            # constant (K fixed)
    logdetK = logdet(Kfac)

    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; lσs = θ[pμ+pσ+1]
        ημ = Xμ * βμ; ησ = Xσ * βσ
        σs² = exp(2 * lσs)
        T = eltype(θ)
        S = zeros(T, G); C = zeros(T, G)
        q1 = zero(T); logdetD = zero(T)
        @inbounds for i in 1:n
            invD = exp(-2 * ησ[i]); r = y[i] - ημ[i]; a = r * invD; k = gidx[i]
            S[k] += invD; C[k] += a; q1 += r * a; logdetD += 2 * ησ[i]
        end
        M = Kinv ./ σs² + Diagonal(S)              # (1/σ_s²)K⁻¹ + ZᵀD⁻¹Z
        # `check = false` + a large FINITE penalty: a line-search step into a
        # non-PD region must not throw (PosDefException) nor return Inf — Julia
        # 1.12's HagerZhang line search asserts the objective is finite.
        Mfac = cholesky(Symmetric(M); check = false)
        issuccess(Mfac) || return convert(eltype(θ), 1e18)
        quad = q1 - dot(C, Mfac \ C)
        logdetV = logdetD + G * log(σs²) + logdetK + logdet(Mfac)
        return 0.5 * (logdetV + quad) + 0.5 * n * log(2π)
    end

    βμ0 = Xμ \ y; res0 = y - Xμ * βμ0
    θ0 = zeros(pμ + pσ + 1)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(std(res0) + eps())
    θ0[pμ+pσ+1] = log(std(res0) / 2 + eps())
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res)
    V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))

    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :resd => (pμ+pσ+1):(pμ+pσ+1)]
    names = [:mu => nmμ, :sigma => nmσ, :resd => [String(grp)]]
    means = Dict(:mu => Xμ * θ̂[1:pμ])
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]))
    return _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll)
end

# Row → level index for one structured marker on the dense Gaussian routes (the
# single-structured fallback and the two-structured route). A PHYLO marker maps
# rows to tree tips BY NAME / tip index (#482, `_phylo_mean_leaf_index`) with
# G = the number of tips, exactly as the sparse phylo-mean and meta_V routes do;
# first-seen order put rows on the wrong tips whenever the data's species order
# differed from the tree's (a different model from drmTMB's, and a different
# logLik from `algorithm = :auto` on the same data). relmat/animal/spatial levels
# are the user's matrix rows in first-seen order (`_group_index`), unchanged.
function _structured_group_index(kind::Symbol, grp::Symbol, labels, tree)
    kind === :phylo || return _group_index(labels)
    tree === nothing && error("phylo(1 | $grp) needs `tree = …`")
    phy = tree isa AbstractString ? augmented_phy(tree) : tree
    return _phylo_mean_leaf_index(phy, labels), phy.n_leaves
end

# Resolve one structured marker to its fixed G×G correlation/relatedness matrix
# from the keyword args. Used by the two-component path (relmat/animal/phylo;
# spatial estimates a range jointly and is not yet supported alongside a second
# component — tracked as a follow-up).
function _resolve_structured_matrix(kind::Symbol, grp::Symbol, G::Int; K, A, tree, coords)
    Cmat = if kind === :relmat
        K === nothing && error("relmat(1 | $grp) needs `K = …`")
        Matrix{Float64}(K)
    elseif kind === :animal
        A === nothing && error("animal(1 | $grp) needs the relatedness matrix `A = …`")
        Matrix{Float64}(A)
    elseif kind === :phylo
        tree === nothing && error("phylo(1 | $grp) needs `tree = …`")
        _phylo_correlation(tree)
    else  # :spatial
        error("spatial(1 | $grp) is not yet supported as one of two structured components " *
              "(it estimates a range jointly); use it as the only structured marker")
    end
    size(Cmat) == (G, G) ||
        error("structured matrix for `$grp` must be $(G)×$(G) (the number of `$grp` levels)")
    return Cmat
end

# Build the n×G group-indicator (one-hot) for a structured intercept.
function _structured_Z(gidx, G)
    n = length(gidx)
    Z = zeros(n, G)
    @inbounds for i in 1:n
        Z[i, gidx[i]] = 1.0
    end
    return Z
end

# Two structured intercepts on the Gaussian mean in ONE fit (e.g.
# `phylo(1|species) + relmat(1|id)`). The latent field is the SUM of two
# structured effects:
#     y = Xβ + Z₁a₁ + Z₂a₂ + ε,  a₁~N(0,σ₁²C₁),  a₂~N(0,σ₂²C₂),  ε~N(0,σ²I)
# so the marginal stays exactly Gaussian:
#     y ~ N(Xβ, V),  V = σ²I + σ₁² Z₁C₁Z₁ᵀ + σ₂² Z₂C₂Z₂ᵀ.
# FIRST CUT: DENSE assembly of V (correctness first); the residual scale is a
# single `sigma ~ 1` (homoscedastic), so D = σ²I. θ = [βμ; logσ (resid); logσ₁; logσ₂].
# Both σ₁ and σ₂ are reported as named variance components via `re_sd`/`vc`.
# Follow-up: sparse/Woodbury assembly for speed (tracked separately); a `sigma`
# predictor on the residual is a straightforward extension (D → diag).
function _fit_two_structured_gaussian(fam::Gaussian, y, Xμ, gidx1, G1, C1, gidx2, G2, C2,
                                      nmμ, grp1, grp2, g_tol)
    n = length(y)
    pμ = size(Xμ, 2)
    Z1 = _structured_Z(gidx1, G1)
    Z2 = _structured_Z(gidx2, G2)
    ZC1Zt = Z1 * C1 * Z1'        # constant building blocks (C₁, C₂ fixed)
    ZC2Zt = Z2 * C2 * Z2'
    Iₙ = Matrix{Float64}(I, n, n)

    function nll(θ)
        βμ = θ[1:pμ]; lσ = θ[pμ+1]; lσ1 = θ[pμ+2]; lσ2 = θ[pμ+3]
        σ² = exp(2 * lσ); σ1² = exp(2 * lσ1); σ2² = exp(2 * lσ2)
        V = σ² .* Iₙ .+ σ1² .* ZC1Zt .+ σ2² .* ZC2Zt
        # `check = false` so a line-search step that drives the residual scale to a
        # numerically non-PD V is rejected with a large finite penalty (the optimiser
        # then backtracks) instead of throwing a `PosDefException`.
        Vfac = cholesky(Symmetric(V); check = false)
        issuccess(Vfac) || return convert(eltype(θ), 1e18)
        r = y .- Xμ * βμ
        quad = dot(r, Vfac \ r)
        return 0.5 * (logdet(Vfac) + quad) + 0.5 * n * log(2π)
    end

    βμ0 = Xμ \ y; res0 = y - Xμ * βμ0
    s0 = std(res0)
    θ0 = zeros(pμ + 3)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(s0 / sqrt(3) + eps())     # balanced split: resid + 2 structured
    θ0[pμ+2] = log(s0 / sqrt(3) + eps())
    θ0[pμ+3] = log(s0 / sqrt(3) + eps())
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res)
    V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))

    # :resd carries BOTH structured SD parameters (logσ₁, logσ₂) so `re_sd` and
    # `vc` report them per grouping factor; :resid carries the residual logσ.
    blocks = [:mu => 1:pμ, :resid => (pμ+1):(pμ+1), :resd => (pμ+2):(pμ+3)]
    names = [:mu => nmμ, :resid => ["residual"], :resd => [String(grp1), String(grp2)]]
    means = Dict(:mu => Xμ * θ̂[1:pμ])
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => fill(exp(θ̂[pμ+1]), n))
    # Conditional RE estimates (BLUPs): â_j = σ_j² C_j Z_jᵀ V⁻¹ r at θ̂.
    blup = let
        βμ = θ̂[1:pμ]; σ1² = exp(2 * θ̂[pμ+2]); σ2² = exp(2 * θ̂[pμ+3])
        Vh = exp(2 * θ̂[pμ+1]) .* Iₙ .+ σ1² .* ZC1Zt .+ σ2² .* ZC2Zt
        Vinvr = cholesky(Symmetric(Vh)) \ (y .- Xμ * βμ)
        a1 = σ1² .* (C1 * (Z1' * Vinvr))
        a2 = σ2² .* (C2 * (Z2' * Vinvr))
        Dict(Symbol(grp1) => a1, Symbol(grp2) => a2)
    end
    return _withranef(_withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n,
        Optim.converged(res), means, obs, scales), nll), blup)
end

# Gaussian phylogenetic random INTERCEPT + SLOPE with two SDs (#620) — the
# model drmTMB fits for `phylo(1 + x | species, tree = tree)` on `mu`
# (`src/drmTMB.cpp` `has_phylo_mu`, q = 2, both dpars on mu):
#     yᵢ = xᵢᵀβ + a_{s(i)} + b_{s(i)} xᵢ + εᵢ,
#     a ~ N(0, σₐ² C),  b ~ N(0, σ_b² C),  a ⊥ b,  ε ~ N(0, D),  D = diag(σᵢ²),
# with C the tree correlation (tip covariance / tree height, drmTMB
# `drm_phylo_tip_covariance(correlation = TRUE)` ≡ `_phylo_correlation`) and
# σᵢ = exp(xσᵢᵀβσ) the fixed-effect residual scale. There is NO intercept–slope
# correlation: drmTMB's C++ adds one `exp(-2·log_sd_k)·uₖᵀQuₖ` term per field
# and never a cross term (the correlated branch there is the mu↔sigma q = 2
# cell only). The marginal stays exactly Gaussian, so the ML fit is the
# closed-form
#     y ~ N(Xβ, V),  V = D + σₐ² Z C Zᵀ + σ_b² Zₓ C Zₓᵀ,  Zₓ = diag(x) Z,
# which is drmTMB's Laplace objective evaluated exactly (Gaussian-linear).
# DENSE assembly (n×n), like `_fit_two_structured_gaussian`; a sparse/Woodbury
# spine is a follow-up. θ = [βμ; βσ; log σₐ; log σ_b].
function _fit_phylo_slope_gaussian(fam::Gaussian, y, Xμ, Xσ, gidx, G, C, xs, nmμ, nmσ,
                                   grp, var, g_tol)
    n = length(y)
    length(xs) == n || error("drm: slope variable `$(var)` has $(length(xs)) rows, expected $n")
    pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    Z = _structured_Z(gidx, G)          # n×G intercept indicator
    Zx = Z .* xs                        # n×G slope design: row i of Z scaled by xᵢ
    ZCZt_a = Z * C * Z'                 # constant building blocks (C fixed)
    ZCZt_b = Zx * C * Zx'

    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; lσa = θ[pμ+pσ+1]; lσb = θ[pμ+pσ+2]
        ημ = Xμ * βμ; ησ = Xσ * βσ
        σa² = exp(2 * lσa); σb² = exp(2 * lσb)
        V = σa² .* ZCZt_a .+ σb² .* ZCZt_b
        @inbounds for i in 1:n
            V[i, i] += exp(2 * ησ[i])
        end
        # `check = false` + a large FINITE penalty: a line-search step into a
        # non-PD region must not throw nor return Inf (HagerZhang asserts finite).
        Vfac = cholesky(Symmetric(V); check = false)
        issuccess(Vfac) || return convert(eltype(θ), 1e18)
        r = y .- ημ
        quad = dot(r, Vfac \ r)
        return 0.5 * (logdet(Vfac) + quad) + 0.5 * n * log(2π)
    end

    βμ0 = Xμ \ y; res0 = y - Xμ * βμ0
    s0 = std(res0)
    θ0 = zeros(pμ + pσ + 2)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(s0 / sqrt(3) + eps())              # balanced split: resid + 2 fields
    θ0[pμ+pσ+1] = log(s0 / sqrt(3) + eps())
    θ0[pμ+pσ+2] = log(s0 / sqrt(3) / (std(xs) + eps()) + eps())
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res)
    Vθ = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))

    # :resd carries the two field SDs (log σₐ, log σ_b): the intercept field keyed
    # by the bare group name (the single-structured convention), the slope field
    # by `<group>:<x>` — drmTMB's `phylo(1 | g)` / `phylo(0 + x | g)` rows.
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :resd => (pμ+pσ+1):(pμ+pσ+2)]
    names = [:mu => nmμ, :sigma => nmσ, :resd => [String(grp), "$(grp):$(var)"]]
    means = Dict(:mu => Xμ * θ̂[1:pμ])
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]))
    # Conditional field estimates (BLUPs) at θ̂: â = σₐ² C Zᵀ V⁻¹ r, b̂ = σ_b² C Zₓᵀ V⁻¹ r.
    blup = let
        βμ = θ̂[1:pμ]; βσ = θ̂[(pμ+1):(pμ+pσ)]
        σa² = exp(2 * θ̂[pμ+pσ+1]); σb² = exp(2 * θ̂[pμ+pσ+2])
        Vh = σa² .* ZCZt_a .+ σb² .* ZCZt_b
        ση² = exp.(2 .* (Xσ * βσ))
        @inbounds for i in 1:n
            Vh[i, i] += ση²[i]
        end
        Vinvr = cholesky(Symmetric(Vh)) \ (y .- Xμ * βμ)
        a = σa² .* (C * (Z' * Vinvr))
        b = σb² .* (C * (Zx' * Vinvr))
        Dict(Symbol(grp) => a, Symbol("$(grp):$(var)") => b)
    end
    fit = DrmFit(fam, blocks, names, θ̂, Vθ, -nll(θ̂), n, Optim.converged(res), means, obs, scales)
    return _withranef(_withnll(fit, nll), blup)
end

# Coordinate-spatial structured intercept: K(ρ) = exp(-d/ρ) from site distances,
# with the range ρ estimated jointly (θ gains log σ_s and log ρ). K depends on θ
# so it is rebuilt each evaluation; otherwise the closed-form marginal is as in
# `_fit_structured_gaussian`.
function _fit_spatial_gaussian(fam::Gaussian, y, Xμ, Xσ, gidx, G, coords, nmμ, nmσ, grp, g_tol)
    n = length(y)
    pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    # The spatial range is only identified with ≥2 distinct sites; a single level
    # (G=1) divides by G^2-G = 0, and coincident coordinates make every pairwise
    # distance 0 so meandist=0 seeds log(0) = -Inf, a non-finite optimiser start.
    G >= 2 || error("spatial(1 | $(grp)) needs at least 2 distinct sites; got G=$(G).")
    Ddist = [sqrt(sum(abs2, coords[k, :] .- coords[l, :])) for k in 1:G, l in 1:G]
    any(Ddist .> 0) ||
        error("spatial(1 | $(grp)): all site coordinates coincide; the spatial " *
              "range is not identified. Check for dropped coordinate variation.")
    meandist = sum(Ddist) / (G^2 - G)

    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; lσs = θ[pμ+pσ+1]; lρ = θ[pμ+pσ+2]
        ημ = Xμ * βμ; ησ = Xσ * βσ
        σs² = exp(2 * lσs); ρ = exp(lρ)
        K = exp.(-Ddist ./ ρ) + 1e-8 * I           # exponential spatial correlation (+ jitter)
        Kfac = cholesky(Symmetric(K); check = false)
        issuccess(Kfac) || return convert(eltype(θ), 1e18)
        T = eltype(θ)
        S = zeros(T, G); C = zeros(T, G)
        q1 = zero(T); logdetD = zero(T)
        @inbounds for i in 1:n
            invD = exp(-2 * ησ[i]); r = y[i] - ημ[i]; a = r * invD; k = gidx[i]
            S[k] += invD; C[k] += a; q1 += r * a; logdetD += 2 * ησ[i]
        end
        M = inv(Kfac) ./ σs² + Diagonal(S)
        Mfac = cholesky(Symmetric(M); check = false)
        issuccess(Mfac) || return convert(eltype(θ), 1e18)
        quad = q1 - dot(C, Mfac \ C)
        logdetV = logdetD + G * log(σs²) + logdet(Kfac) + logdet(Mfac)
        return 0.5 * (logdetV + quad) + 0.5 * n * log(2π)
    end

    βμ0 = Xμ \ y; res0 = y - Xμ * βμ0
    θ0 = zeros(pμ + pσ + 2)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(std(res0) + eps())
    θ0[pμ+pσ+1] = log(std(res0) / 2 + eps())
    θ0[pμ+pσ+2] = log(max(meandist, eps()))   # meandist>0 by the G/coincidence guards
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res)
    V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))

    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ),
        :resd => (pμ+pσ+1):(pμ+pσ+1), :range => (pμ+pσ+2):(pμ+pσ+2)]
    names = [:mu => nmμ, :sigma => nmσ, :resd => [String(grp)], :range => ["range"]]
    means = Dict(:mu => Xμ * θ̂[1:pμ])
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]))
    return _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll)
end

# ===========================================================================
# SPARSE O(p) path for the SAME two-structured Gaussian model (#225 follow-up)
# ===========================================================================
# Same model as `_fit_two_structured_gaussian`:
#     y = Xβ + Z₁a₁ + Z₂a₂ + ε,  aₖ ~ N(0, σₖ²Cₖ),  ε ~ N(0, σ²I),
#     y ~ N(Xβ, V),   V = σ²I + σ₁² Z₁C₁Z₁ᵀ + σ₂² Z₂C₂Z₂ᵀ.
#
# The dense path assembles and factors the n×n V each evaluation — O(n³). This
# path NEVER forms V. It works with the AUGMENTED LATENT a = [a₁; a₂] (length
# m = m₁ + m₂) whose prior precision is BLOCK-SPARSE,
#     P(θ) = blockdiag(σ₁⁻² Q₁, σ₂⁻² Q₂),   Qₖ = Cₖ⁻¹ (sparse for tree / Ainv),
# and integrates a out EXACTLY via one sparse Cholesky of the m×m
#     H(θ) = P(θ) + ZᵀZ / σ²,   Z = [Z₁ Z₂].
#
# Marginal NLL (matrix-determinant lemma + Woodbury, exact — the Gaussian
# Laplace is exact, so the "inner mode" is a single linear solve â = H⁻¹ b):
#     logdet V = n logσ² − logdet P + logdet H,
#     rᵀV⁻¹r   = rᵀr/σ² − bᵀâ,   b = Zᵀr/σ²,  â = H⁻¹ b,
#     NLL = ½(logdet V + rᵀV⁻¹r) + ½ n log 2π.
#
# Analytic gradient (mirrors the q4 engine's logdet/Takahashi recipe). With
# u = V⁻¹r = (r − Zâ)/σ² and â = [â₁; â₂] = H⁻¹b = Σ Zᵀu (BLUPs):
#   ∂NLL/∂β   = −Xᵀu
#   ∂NLL/∂lσ  = ½(2n − 2 tr(H⁻¹ ZᵀZ)/σ²) − σ²‖u‖²
#   ∂NLL/∂lσₖ = ½(2mₖ − 2 tr(σₖ⁻²Qₖ H⁻¹)) − wₖᵀâₖ,  wₖ = Zₖᵀu
# where the traces are read off the Takahashi SELECTED INVERSE of H at the
# (sparse) pattern of ZᵀZ / Qₖ — never a dense m×m inverse.

# Build the n×m sparse incidence Z mapping each observation to its latent row,
# with per-observation weight `wts` (defaults to 1). A non-unit weight folds a
# diagonal rescaling of the latent into Z — used to convert the augmented tree
# COVARIANCE into the leaf CORRELATION (the dense path's parameterisation) by
# scaling each leaf's contribution by 1/√(leaf variance), which keeps Z sparse.
function _sparse_incidence(rows::AbstractVector{<:Integer}, n::Int, m::Int,
                           wts::AbstractVector{<:Real} = ones(n))
    return sparse(collect(1:n), collect(rows), Float64.(wts), n, m)
end

# tr(Sblock · Hinv) for Sblock = Qk placed at offset `off`, Hinv the selected
# inverse. Only entries of Hinv at Qk's pattern (⊆ L+Lᵀ) are read.
function _trace_block_selinv(Hinv::SparseMatrixCSC, Qk::SparseMatrixCSC, off::Int)
    s = 0.0
    rv = rowvals(Qk); nzv = nonzeros(Qk)
    Hcol = Hinv.colptr; Hrow = rowvals(Hinv); Hval = nonzeros(Hinv)
    @inbounds for jk in 1:size(Qk, 2)
        jH = jk + off
        for p in nzrange(Qk, jk)
            iH = rv[p] + off
            idx = _csc_rowidx(Hcol, Hrow, jH, iH)
            idx == -1 && continue
            s += nzv[p] * Hval[idx]
        end
    end
    return s
end

# tr(H⁻¹ ZᵀZ): ZᵀZ sparse and ⊆ H's pattern. Read selected-inverse entries.
function _trace_selinv_full(Hinv::SparseMatrixCSC, ZtZ::SparseMatrixCSC)
    s = 0.0
    rv = rowvals(ZtZ); nzv = nonzeros(ZtZ)
    Hcol = Hinv.colptr; Hrow = rowvals(Hinv); Hval = nonzeros(Hinv)
    @inbounds for j in 1:size(ZtZ, 2)
        for p in nzrange(ZtZ, j)
            idx = _csc_rowidx(Hcol, Hrow, j, rv[p])
            idx == -1 && continue
            s += nzv[p] * Hval[idx]
        end
    end
    return s
end

# One structured component for the spec-based sparse engine. `Q` is the SPARSE
# prior precision over the component's latent (m rows); `rows[i]` maps obs i to a
# latent row and `wts[i]` weights that obs's incidence; `logdetCprior =
# logdet(Q⁻¹) = −logdet Q` is the constant in the Woodbury logdet. `leaf_pos[t]`
# maps level/leaf t → latent row and `blup_scale[t]` rescales that level's BLUP
# (so per-level estimates read out of an AUGMENTED, covariance-parameterised
# latent match the leaf-CORRELATION model). `grp` names the grouping factor.
struct _StructComp
    Q::SparseMatrixCSC{Float64,Int}
    rows::Vector{Int}
    wts::Vector{Float64}
    m::Int
    logdetCprior::Float64
    leaf_pos::Vector{Int}
    blup_scale::Vector{Float64}
    grp::Symbol
end

# DENSE-correlation component: Qₖ = Cₖ⁻¹ (inverts the dense leaf correlation —
# the original #231 behaviour). Latent rows ARE the levels, so leaf_pos = 1:G,
# unit weights / BLUP scales.
function _dense_comp(gidx, G, C, grp::Symbol)
    Q = dropzeros!(sparse(Symmetric(inv(Symmetric(Matrix(C))))))
    logdetC = logdet(Symmetric(Matrix(C)))
    rows = collect(Int, gidx)
    return _StructComp(Q, rows, ones(length(rows)), G, logdetC,
                       collect(1:G), ones(G), grp)
end

# END-TO-END SPARSE phylo component (#232): feed the root-conditioned augmented
# tree precision Q (O(p) nnz) DIRECTLY — no dense Ck, no dense inversion. The
# latent ã spans q = 2p−2 augmented nodes with prior precision Q (covariance
# Σ = Q⁻¹); obs attach at leaf nodes via leaf_pos.
#
# The dense `phylo(1|g)` path models the leaf CORRELATION C = D^{-1/2} Σ D^{-1/2}
# (D = diag of leaf variances). To match it EXACTLY while keeping the augmented
# precision sparse, reparametrise a_c = D^{-1/2} ã: the correlation effect on obs
# i becomes (1/√v_leaf(i)) · ã_leaf(i), i.e. a per-obs incidence WEIGHT
# 1/√v_leaf. The leaf variances v_t = (Q⁻¹)_{leaf,leaf} are read once from the
# Takahashi selected inverse of Q (O(p)). logdet(Σ) = −logdet(Q) is unchanged
# (the D-rescale lives in Z, not in the prior). Per-level BLUPs are scaled back
# by 1/√v_t.
function _phylo_aug_comp(gidx, G, tree, grp::Symbol)
    phy = tree isa AbstractString ? augmented_phy(tree) : tree
    phy.n_leaves == G ||
        error("phylo($grp): tree has $(phy.n_leaves) tips but `$grp` has $G levels")
    Q, leaf_pos, q = augmented_tree_precision(phy)
    Qs = dropzeros!(sparse(Symmetric(Q, :U)))   # symmetric, sparse, PD over kept nodes
    chQ = cholesky(Symmetric(Qs); check = false)
    issuccess(chQ) ||
        error("phylo($grp): root-conditioned augmented precision is not PD")
    logdetCprior = -logdet(chQ)                       # logdet(Σ) = logdet(Q⁻¹)
    # Leaf variances from the selected inverse of Q (diagonal entries are always
    # in the Takahashi pattern) — O(p), never forms a dense Q⁻¹.
    Qinv = takahashi_selinv(chQ)
    leaf_var = [Qinv[leaf_pos[t], leaf_pos[t]] for t in 1:G]
    inv_sd = [1.0 / sqrt(leaf_var[t]) for t in 1:G]   # D^{-1/2} per leaf
    rows = [leaf_pos[g] for g in gidx]                # obs → augmented leaf node
    wts = [inv_sd[g] for g in gidx]                   # incidence weight = 1/√v_leaf
    return _StructComp(Qs, rows, wts, q, logdetCprior, leaf_pos, inv_sd, grp)
end

# Spec-based sparse fitter for the two-structured Gaussian mean model. Integrates
# the augmented latent a = [a₁; a₂] via ONE sparse Cholesky of
#     H(θ) = blockdiag(σ₁⁻²Q₁, σ₂⁻²Q₂) + ZᵀWZ,   W = diag(σ⁻²) (the residual prec)
# reusing a SINGLE symbolic factorisation across all evaluations (cholesky! into a
# pre-analysed factor). `Xσ` carries the residual-scale design: `sigma ~ 1` ⇒
# homoscedastic (one logσ); `sigma ~ x` ⇒ D → diag (a logσ per row). logLik via
# det-lemma + Woodbury, variance-component gradient via Takahashi selected inverse.
function _fit_two_structured_gaussian_sparse_spec(fam::Gaussian, y, Xμ, Xσ,
                                                  comp1::_StructComp, comp2::_StructComp,
                                                  nmμ, nmσ, g_tol)
    n = length(y)
    pμ = size(Xμ, 2); pσ = size(Xσ, 2)
    Q1 = comp1.Q; Q2 = comp2.Q
    m1 = comp1.m; m2 = comp2.m; m = m1 + m2; off2 = m1
    Z1 = _sparse_incidence(comp1.rows, n, m1, comp1.wts)
    Z2 = _sparse_incidence(comp2.rows, n, m2, comp2.wts)
    Z = hcat(Z1, Z2)
    Xt = Matrix(Xμ')
    Xσt = Matrix(Xσ')
    logdetCprior1 = comp1.logdetCprior; logdetCprior2 = comp2.logdetCprior

    # ZᵀWZ at W = I gives the FIXED sparsity pattern of the data contribution; with
    # a heteroscedastic W only the VALUES move, so the union pattern of
    # H = blockdiag(Q1,Q2)+ZᵀWZ is CONSTANT across θ. We analyse it ONCE (the
    # `chol_ref` factor) and reuse that symbolic analysis via `cholesky!` (numeric
    # refactor only — no re-analysis). The `+ 0·H_template` term forces every
    # evaluation's H to carry the full analysed pattern; if a CHOLMOD pattern check
    # ever rejects the in-place update we fall back to a fresh `cholesky` (still
    # tree-sparse O(p)), so correctness never depends on the reuse succeeding.
    ZtZ_pat = dropzeros!(sparse(Symmetric(Z' * Z)))
    H_template = blockdiag(Q1, Q2) + ZtZ_pat
    Hzero = 0.0 .* H_template               # pattern carrier (all-structural-zero)
    chol_ref = Ref(cholesky(Symmetric(H_template); check = false))
    issuccess(chol_ref[]) ||
        error("sparse two-structured: template Cholesky failed (non-PD pattern)")

    # ZᵀWZ for a diagonal residual precision w (length n). Same nnz pattern as ZtZ_pat.
    function _ZtWZ(w)
        ZtW = Z' * Diagonal(w)
        return dropzeros!(sparse(Symmetric(ZtW * Z)))
    end

    function eval_all(βμ, βσ, lσ1, lσ2; want_grad::Bool)
        ησ = Xσ * βσ                       # log residual SD per row
        w = exp.(-2 .* ησ)                 # residual precision diag W
        σ1² = exp(2lσ1); σ2² = exp(2lσ2)
        r = y .- Xμ * βμ
        ZtWZ = _ZtWZ(w)
        H = blockdiag(Q1 ./ σ1², Q2 ./ σ2²) + ZtWZ + Hzero
        ch = try
            cholesky!(chol_ref[], H; check = false); chol_ref[]
        catch
            cholesky(Symmetric(H); check = false)
        end
        issuccess(ch) || return (1e18, Float64[], r, zeros(m), w, false)
        b = Z' * (w .* r)
        â = ch \ b
        logdetH = logdet(ch)
        logdetP = -2 * (m1 * lσ1 + m2 * lσ2) - logdetCprior1 - logdetCprior2
        logdetD = -sum(log, w)             # logdet(D) = Σ 2 ησ = −Σ log w
        logdetV = logdetD - logdetP + logdetH
        quad = dot(r, w .* r) - dot(b, â)
        nll = 0.5 * (logdetV + quad) + 0.5 * n * log(2π)
        isfinite(nll) || return (1e18, Float64[], r, â, w, false)
        want_grad || return (nll, Float64[], r, â, w, true)
        # u = V⁻¹ r = W(r − Zâ); ∂NLL/∂β = −Xᵀu.
        u = w .* (r .- Z * â)
        gβ = -(Xt * u)
        Hinv = takahashi_selinv(ch)
        # ∂NLL/∂βσ_j: D=diag(σ_i²), ∂logσ_i. ½(2·#rows·... ) with the residual trace.
        # tr(V⁻¹ ∂D/∂(2ησ_i)) − uᵢ² ∂(σ_i²). The residual log-det trace through H is
        # tr(H⁻¹ Zᵀ(∂W)Z) with ∂W/∂ησ_i = −2 w_i e_iᵀ. Collapsed per σ-covariate.
        # diag of V⁻¹ at row i: w_i − w_i (ZH⁻¹Zᵀ)_ii w_i = w_i − sᵢ, where
        # sᵢ = w_i² (Z Hinv Zᵀ)_ii. We need diag(Z Hinv Zᵀ).
        zHz = _diag_ZHinvZt(Hinv, comp1.rows, comp1.wts, comp2.rows, comp2.wts, off2, n)
        gβσ = zeros(pσ)
        @inbounds for i in 1:n
            vinv_ii = w[i] - w[i]^2 * zHz[i]          # (V⁻¹)_ii
            # ∂NLL/∂ησ_i = (V⁻¹)_ii σ_i² − u_i² σ_i², with σ_i² = 1/w_i ⇒
            #             = (V⁻¹)_ii / w_i − u_i² / w_i. Times design row.
            dη = vinv_ii / w[i] - (u[i]^2) / w[i]
            for k in 1:pσ
                gβσ[k] += Xσt[k, i] * dη
            end
        end
        trQ1 = _trace_block_selinv(Hinv, Q1, 0) / σ1²
        trQ2 = _trace_block_selinv(Hinv, Q2, off2) / σ2²
        w1pos = @view â[1:m1]; w2pos = @view â[off2+1:off2+m2]
        Zw = Z' * u                                   # = Σ Zᵀu (BLUP-conjugate)
        wv1 = @view Zw[1:m1]; wv2 = @view Zw[off2+1:off2+m2]
        glσ1 = 0.5 * (2 * m1 - 2 * trQ1) - dot(wv1, w1pos)
        glσ2 = 0.5 * (2 * m2 - 2 * trQ2) - dot(wv2, w2pos)
        grad = vcat(gβ, gβσ, glσ1, glσ2)
        # A finite nll can still pair with a non-finite gradient near a boundary
        # (overflowing residual precision, ill-conditioned selected-inverse).
        # LineSearches (HagerZhang) asserts both value and directional derivative
        # are finite, so fall back to the penalty region instead of emitting NaN/Inf.
        all(isfinite, grad) || return (1e18, Float64[], r, â, w, false)
        return (nll, grad, r, â, w, true)
    end

    # parameter layout: θ = [βμ(pμ); βσ(pσ); lσ1; lσ2]
    iβμ = 1:pμ; iβσ = (pμ+1):(pμ+pσ); il1 = pμ + pσ + 1; il2 = pμ + pσ + 2
    unpack(θ) = (θ[iβμ], θ[iβσ], θ[il1], θ[il2])
    nll_only(θ) = (p = unpack(θ); eval_all(p...; want_grad = false)[1])
    function fg!(F, Gout, θ)
        nll, grad, _, _, _, ok = eval_all(unpack(θ)...; want_grad = true)
        if Gout !== nothing
            ok ? copyto!(Gout, grad) : fill!(Gout, 0.0)
        end
        return nll
    end

    βμ0 = Xμ \ y; res0 = y - Xμ * βμ0
    s0 = std(res0)
    np = pμ + pσ + 2
    θ0 = zeros(np)
    θ0[iβμ] .= βμ0
    θ0[first(iβσ)] = log(s0 / sqrt(3) + eps())   # intercept of log σ; slopes start 0
    θ0[il1] = log(s0 / sqrt(3) + eps())
    θ0[il2] = log(s0 / sqrt(3) + eps())
    od = Optim.NLSolversBase.only_fg!(fg!)
    res = Optim.optimize(od, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol))
    θ̂ = Optim.minimizer(res)

    grad_at(θ) = eval_all(unpack(θ)...; want_grad = true)[2]
    Hmat = zeros(np, np)
    hstep = 1e-6
    for k in 1:np
        θp = copy(θ̂); θm = copy(θ̂)
        step = hstep * max(abs(θ̂[k]), 1.0)
        θp[k] += step; θm[k] -= step
        Hmat[:, k] .= (grad_at(θp) .- grad_at(θm)) ./ (2 * step)
    end
    Hmat .= 0.5 .* (Hmat .+ Hmat')
    V = try
        Matrix(inv(Symmetric(Hmat)))
    catch
        fill(NaN, np, np)
    end

    grp1 = comp1.grp; grp2 = comp2.grp
    blocks = [:mu => iβμ, :sigma => iβσ, :resd => (il1):(il2)]
    names = [:mu => nmμ, :sigma => nmσ, :resd => [String(grp1), String(grp2)]]
    means = Dict(:mu => Xμ * θ̂[iβμ])
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xσ * θ̂[iβσ]))
    blup = let v = eval_all(unpack(θ̂)...; want_grad = false)
        â = v[4]
        # Read per-LEVEL BLUPs out of the (possibly augmented) latent, rescaling
        # by blup_scale (= 1/√v_leaf for the correlation-matched phylo latent).
        a1 = Float64[comp1.blup_scale[t] * â[comp1.leaf_pos[t]] for t in eachindex(comp1.leaf_pos)]
        a2 = Float64[comp2.blup_scale[t] * â[off2+comp2.leaf_pos[t]] for t in eachindex(comp2.leaf_pos)]
        Dict(grp1 => a1, grp2 => a2)
    end
    return _withranef(_withnll(DrmFit(fam, blocks, names, θ̂, V, -nll_only(θ̂), n,
        Optim.converged(res), means, obs, scales), nll_only), blup)
end

# diag(Z H⁻¹ Zᵀ) where Z rows map obs → augmented latent with per-obs weights
# (wts1/wts2). Hinv is the Takahashi selected inverse (diagonal entries are always
# in its pattern). Used only by the `sigma ~ x` (heteroscedastic) gradient.
function _diag_ZHinvZt(Hinv::SparseMatrixCSC, rows1, wts1, rows2, wts2, off2, n)
    Hcol = Hinv.colptr; Hrow = rowvals(Hinv); Hval = nonzeros(Hinv)
    out = zeros(n)
    @inbounds for i in 1:n
        # obs i hits latent row r1 (comp1, weight w1) and r2+off2 (comp2, weight
        # w2); (ZH⁻¹Zᵀ)_ii = w1²(H⁻¹)_{r1,r1} + w2²(H⁻¹)_{r2,r2} + 2 w1 w2 (H⁻¹)_{r1,r2}.
        a = rows1[i]; bcol = rows2[i] + off2
        w1 = wts1[i]; w2 = wts2[i]
        ia = _csc_rowidx(Hcol, Hrow, a, a); out[i] += ia == -1 ? 0.0 : w1^2 * Hval[ia]
        ib = _csc_rowidx(Hcol, Hrow, bcol, bcol); out[i] += ib == -1 ? 0.0 : w2^2 * Hval[ib]
        iab = _csc_rowidx(Hcol, Hrow, max(a, bcol), min(a, bcol))
        out[i] += iab == -1 ? 0.0 : 2 * w1 * w2 * Hval[iab]
    end
    return out
end

# Resolve ONE structured marker to a sparse `_StructComp` for the end-to-end
# sparse engine. A phylo component feeds the augmented tree precision DIRECTLY
# (O(p), no dense Ck inversion, #232); relmat/animal use the user-supplied dense
# relatedness matrix with Qk = K⁻¹ (the matrix IS the input — no tree to exploit).
function _sparse_struct_comp(kind::Symbol, grp::Symbol, G::Int, gidx;
                             K, A, tree)
    if kind === :phylo
        tree === nothing && error("phylo(1 | $grp) needs `tree = …`")
        return _phylo_aug_comp(gidx, G, tree, grp)
    elseif kind === :relmat
        K === nothing && error("relmat(1 | $grp) needs `K = …`")
        return _dense_comp(gidx, G, Matrix{Float64}(K), grp)
    elseif kind === :animal
        A === nothing && error("animal(1 | $grp) needs the relatedness matrix `A = …`")
        return _dense_comp(gidx, G, Matrix{Float64}(A), grp)
    else
        error("the sparse two-structured path supports phylo / relmat / animal " *
              "components (got $kind for `$grp`)")
    end
end

# Backward-compatible entry (#231 signature): two DENSE leaf correlations,
# homoscedastic residual (`sigma ~ 1`). Builds dense-C specs and delegates.
function _fit_two_structured_gaussian_sparse(fam::Gaussian, y, Xμ, gidx1, G1, C1,
                                             gidx2, G2, C2, nmμ, grp1, grp2, g_tol)
    n = length(y)
    Xσ = ones(n, 1)
    comp1 = _dense_comp(gidx1, G1, C1, grp1)
    comp2 = _dense_comp(gidx2, G2, C2, grp2)
    fit = _fit_two_structured_gaussian_sparse_spec(fam, y, Xμ, Xσ, comp1, comp2,
                                                   nmμ, ["(Intercept)"], g_tol)
    # #231 reported the residual under :resid and used :sigma in scales; map the
    # homoscedastic intercept back to the historical block names for compatibility.
    return _remap_resid_block(fit)
end

# The #231 entry exposed blocks [:mu,:resid,:resd]; the spec engine exposes
# [:mu,:sigma,:resd]. For the homoscedastic wrapper, rename :sigma → :resid so the
# existing accessors/tests (which read `:resid => ["residual"]`) are unchanged.
function _remap_resid_block(fit::DrmFit)
    blocks = [k === :sigma ? (:resid => v) : (k => v) for (k, v) in fit.blocks]
    names = [k === :sigma ? (:resid => ["residual"]) : (k => v) for (k, v) in fit.coefnames]
    return DrmFit(fit.family, blocks, names, fit.theta, fit.vcov, fit.loglik,
                  fit.nobs, fit.converged, fit.means, fit.obs, fit.scales,
                  fit.formula, fit.nll, fit.nllgrad, fit.ranef,
                  fit.estim_method, fit.reml_loglik, fit.ml_loglik, fit.marginal)
end
