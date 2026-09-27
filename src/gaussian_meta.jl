# gaussian_meta.jl — Gaussian meta-analysis with known sampling (co)variances.
#
#   y_i ~ N(x_iᵀβ, v_i + σ_i²),   v_i supplied (known),  σ_i the residual
# heterogeneity SD (the σ formula; σ ~ 1 gives a single between-study
# heterogeneity SD σ, the classical meta-analytic τ). Marker: `meta_V(v)` inside
# the μ formula flags the data column `v` of known sampling variances. Mirrors
# drmTMB's `gaussian() + meta_V(V = V)`.

"""
    meta_V(v)

Formula marker for Gaussian meta-analysis: `v` is the data column of **known**
sampling variances. Use inside a `μ` formula, e.g.
`bf(y ~ x + meta_V(v), sigma ~ 1)`. The residual (between-study) heterogeneity SD
is the `σ` parameter (the between-study τ of classical meta-analysis).
(Diagonal known variances; dense/bivariate sampling covariance is planned.)

Random intercepts on the mean combine with `meta_V`, e.g.
`bf(y ~ x + meta_V(v) + (1 | study), sigma ~ 1)` or
`bf(y ~ x + meta_V(v) + phylo(1 | species) + (1 | study), sigma ~ 1)` (with
`tree = …`); `relmat` / `animal` take `K = …` / `A = …`. The marginal is
`N(Xβ, diag(v + σ²) + Σₖ sₖ² Zₖ Cₖ Zₖᵀ)`, drmTMB's model. Each random component
needs its own grouping column; slopes, `spatial`, and REML are not implemented.
"""
meta_V(v) = v     # identity stub; the marker is intercepted during formula parsing

function _fit_meta_gaussian(fam::Gaussian, y, Xμ, Xσ, vv, nmμ, nmσ, g_tol)
    n = length(y)
    pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]
        ημ = Xμ * βμ; ησ = Xσ * βσ                 # ησ = log σ_i
        s = zero(eltype(θ))
        @inbounds for i in 1:n
            var = vv[i] + exp(2 * ησ[i])           # known sampling var + heterogeneity σ²
            r = y[i] - ημ[i]
            s += log(var) + r * r / var
        end
        return 0.5 * s + 0.5 * n * log(2π)
    end
    βμ0 = Xμ \ y
    θ0 = zeros(pμ + pσ)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(std(y - Xμ * βμ0) / 2 + eps())   # σ below the total residual sd
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res)
    V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ)]
    names = [:mu => nmμ, :sigma => nmσ]
    means = Dict(:mu => Xμ * θ̂[1:pμ])
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => sqrt.(vv .+ exp.(2 .* (Xσ * θ̂[(pμ+1):(pμ+pσ)]))))  # √(v + σ²)
    return _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll)
end

# ─────────────────────────────────────────────────────────────────────────────
# Gaussian meta-analysis with known sampling variances PLUS random intercepts
# on the mean (Arc 2, drmTMB's `meta_V(V = v) + (1 | study)`, `+ phylo(1 | sp)`,
# `+ relmat(1 | id)`, `+ animal(1 | id)`, and sums of these).
#
# The model drmTMB fits (src/drmTMB.cpp, univariate Gaussian branch): given the
# random effects u, yᵢ ~ N(xᵢᵀβ + (Zu)ᵢ, vᵢ + σᵢ²) with vᵢ = `V_known`, σᵢ =
# exp(xσᵢᵀβσ) the residual heterogeneity, and one intercept field per component
# k, u_k ~ N(0, s_k² C_k) (C_k = I for `(1 | g)`, the tip correlation for
# `phylo`, K / A for `relmat` / `animal`). drmTMB integrates u by Laplace, which
# is exact here (Gaussian-linear), so the marginal it maximises is
#
#     y ~ N(Xβ, Ω),   Ω = D + Σ_k s_k² Z_k C_k Z_kᵀ,   D = diag(vᵢ + σᵢ²).
#
# Before this route existed the Gaussian router returned the `meta_V`-only fit
# for `meta_V + (1 | g)` (the random effect silently dropped) and the dense
# structured fit for `meta_V + phylo/relmat` (the known variances silently
# dropped). Both were a different model at the same formula.
#
# Evaluation: whitened Woodbury. With C_k = L_k L_kᵀ and Z̃ = Z·blockdiag(L_k)·S,
# S = diag(s_k per column), Ω = D + Z̃Z̃ᵀ and
#     logdet Ω = Σ log Dᵢ + logdet(I + Z̃ᵀD⁻¹Z̃),
#     rᵀΩ⁻¹r   = rᵀD⁻¹r − c̃ᵀ(I + Z̃ᵀD⁻¹Z̃)⁻¹c̃,   c̃ = Z̃ᵀD⁻¹r.
# The whitened capacitance stays well conditioned as s_k → 0 (a boundary
# variance), and Dᵢ ≥ vᵢ > 0 keeps D⁻¹ bounded. ZᵀD⁻¹Z is accumulated row by row
# (each row has one nonzero per component), so one evaluation costs
# O(n·K² + q³), q = Σ_k (levels of component k).
#
# `comps` is a vector of `(gidx, G, L, name)`: row → level index, the number of
# levels, the lower Cholesky factor of C (or `nothing` for C = I), and the
# grouping name reported in the `:resd` block (the same block layout as the
# single-intercept and structured routes, so the bridge labels it identically).
# ─────────────────────────────────────────────────────────────────────────────
function _fit_meta_gaussian_re(fam::Gaussian, y, Xμ, Xσ, vv, comps, nmμ, nmσ, g_tol)
    n = length(y)
    pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    K = length(comps)
    K ≥ 1 || error("drm: internal — _fit_meta_gaussian_re needs at least one random component")
    all(>(0), vv) ||
        throw(ArgumentError("drm: `meta_V(...)` sampling variances must be strictly positive"))
    Gks = [c[2] for c in comps]
    q = sum(Gks)
    offs = cumsum([0; Gks])
    colcomp = Vector{Int}(undef, q)
    for k in 1:K, c in 1:Gks[k]
        colcomp[offs[k]+c] = k
    end
    # Row i's column in the stacked design, per component.
    cols = [offs[k] .+ comps[k][1] for k in 1:K]
    # blockdiag(L_k): identity blocks for ordinary `(1 | g)`.
    Lbd = zeros(q, q)
    for k in 1:K
        r = (offs[k]+1):offs[k+1]
        L = comps[k][3]
        if L === nothing
            for c in r
                Lbd[c, c] = 1.0
            end
        else
            Lbd[r, r] .= L
        end
    end
    const_2pi = 0.5 * n * log(2π)

    # The pieces shared by the objective and the BLUPs, at any θ (dual-safe).
    function _pieces(θ)
        T = eltype(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]
        ημ = Xμ * βμ; ησ = Xσ * βσ
        A = zeros(T, q, q)                       # ZᵀD⁻¹Z
        b = zeros(T, q)                          # ZᵀD⁻¹r
        q1 = zero(T); logdetD = zero(T)
        @inbounds for i in 1:n
            Di = vv[i] + exp(2 * ησ[i])          # known sampling var + heterogeneity σ²
            invD = 1 / Di
            r = y[i] - ημ[i]
            q1 += r * r * invD
            logdetD += log(Di)
            for a in 1:K
                ca = cols[a][i]
                b[ca] += r * invD
                for bb in 1:K
                    A[ca, cols[bb][i]] += invD
                end
            end
        end
        s = [exp(θ[pμ+pσ+colcomp[c]]) for c in 1:q]
        W = Lbd .* s'                            # blockdiag(L_k)·S
        M = Symmetric(I + W' * A * W)
        c̃ = W' * b
        return M, c̃, q1, logdetD, W
    end

    function nll(θ)
        M, c̃, q1, logdetD, _ = _pieces(θ)
        Mfac = cholesky(M; check = false)
        # A line-search probe at an extreme scale must not throw, and must not
        # return Inf (HagerZhang asserts a finite objective).
        issuccess(Mfac) || return convert(eltype(θ), 1e18)
        return 0.5 * (logdetD + logdet(Mfac) + q1 - dot(c̃, Mfac \ c̃)) + const_2pi
    end

    βμ0 = Xμ \ y
    res0 = y - Xμ * βμ0
    s0 = std(res0) / sqrt(K + 1)                 # balanced split: heterogeneity + K fields
    θ0 = zeros(pμ + pσ + K)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(s0 + eps())
    θ0[(pμ+pσ+1):end] .= log(s0 + eps())
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res)
    V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))

    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :resd => (pμ+pσ+1):(pμ+pσ+K)]
    names = [:mu => nmμ, :sigma => nmσ, :resd => [String(c[4]) for c in comps]]
    means = Dict(:mu => Xμ * θ̂[1:pμ])
    obs = Dict(:mu => Vector{Float64}(y))
    # √(v + σ²): the conditional observation SD (random effects excluded), the
    # same `scales[:sigma]` convention as the `meta_V`-only route, from which
    # the bridge recovers σ and `V_known` (`_bridge_meta_parts`).
    scales = Dict(:sigma => sqrt.(vv .+ exp.(2 .* (Xσ * θ̂[(pμ+1):(pμ+pσ)]))))
    # BLUPs: E[u | y] = W (I + WᵀAW)⁻¹ c̃ at θ̂, split per component.
    blup = let
        M, c̃, _, _, W = _pieces(θ̂)
        u = W * (cholesky(M) \ c̃)
        Dict(Symbol(comps[k][4]) => u[(offs[k]+1):offs[k+1]] for k in 1:K)
    end
    # `converged` is the GRADIENT criterion (the #609 lesson: Optim's OR of the
    # x/f/g criteria can report a flat-but-unfinished run as converged).
    converged = Optim.converged(res) && Optim.g_converged(res)
    return _withranef(_withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, converged,
        means, obs, scales), nll), blup)
end
