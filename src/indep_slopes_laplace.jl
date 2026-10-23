# indep_slopes_laplace.jl — independent random intercepts and slopes on the mean
# for Gamma() and Student(), e.g. `(1 | g) + (0 + x | g)` (#960, drmTMB parity).
#
# drmTMB fits these layouts by TMB's Laplace approximation with one independent
# N(0, σ_k²) prior per term; it refuses the correlated `(1 + x | g)` form for
# these families (drmTMB #1457), so this route deliberately covers the
# independent form only. DRModels.jl's own correlated `(1 + x | g)` GHQ routes
# are unchanged.
#
# Model. K scalar components; component k has a per-observation weight w_k
# (1 for an intercept, the covariate x for a `(0 + x | g)` slope), a grouping
# index and G_k levels. With b = [b_1; …; b_K] (q = Σ G_k effects),
#
#     η_i = X_i′β_μ + Σ_k w_k[i] · b_k[g_k(i)],      b_k ~ N(0, σ_k² I),
#
# and the TMB-convention Laplace marginal (the (2π) constants cancel)
#
#     nll(θ) = Σ_i −log f(y_i | η_i(b̂)) + ½ Σ_k ‖b̂_k‖²/σ_k² + Σ_k G_k log σ_k + ½ log det 𝐇(b̂),
#
# 𝐇 the Hessian in b of the penalised negative log-likelihood. Same machinery
# as the Student crossed-intercept route (`_fit_student_crossed_laplace`, #725):
# Newton inner mode on Float64 primals (expected-information direction where
# the observed 𝐇 is indefinite), then two Newton steps in dual arithmetic so
# ForwardDiff carries db̂/dθ (implicit-function theorem). 𝐇 is dense q×q: fine
# for hundreds of levels.
#
# Per-family observation terms return (−log f, d/dη, d²/dη², expected d²/dη²):
#   Gamma (log link, shape α = σ⁻², the `_fit_gamma` convention):
#     −log f = −α log α + log Γ(α) − (α − 1) log y + α η + α y e^{−η}
#     d/dη = α − α y e^{−η},  d²/dη² = α y e^{−η},  E = α
#   Student (identity link, scale σ, ν = 2 + e^{η_ν}, t = (y − η)/σ, r = 1/ν):
#     as `_fit_student_crossed_laplace`.

function _indep_obs_terms(::Val{:gamma}, y, η, lσ, ην)
    α = exp(-2 * lσ)
    ye = y * exp(-η)
    val = -(α * log(α) - loggamma(α) + (α - 1) * log(y) - α * η - α * ye)
    return val, α - α * ye, α * ye, α
end

function _indep_obs_terms(::Val{:student}, y, η, lσ, ην)
    σ = exp(lσ); t = (y - η) / σ
    r = ην > 0 ? exp(-ην) / (1 + 2 * exp(-ην)) : 1 / (2 + exp(ην))
    d = 1 + r * t^2
    val = lσ - _student_logpdf_std(t, ην)
    g1 = -(1 + r) * t / (σ * d)
    w = (1 + r) * (1 - r * t^2) / (σ^2 * d^2)
    wE = (1 + r) / ((1 + 3r) * σ^2)
    return val, g1, w, wE
end

_indep_primal(x::ForwardDiff.Dual) = _indep_primal(ForwardDiff.value(x))
_indep_primal(x::Real) = Float64(x)

"""
    _fit_indep_re_laplace(fam, kind, y, Xμ, Xσ, Xν, comps, nmμ, nmσ, nmν, g_tol) -> DrmFit

Laplace fit of `Gamma()` (`kind = Val(:gamma)`, `Xν = nothing`) or `Student()`
(`kind = Val(:student)`) with independent scalar random-effect components on the
mean. `comps` is a vector of `(w, gidx, G, label)` (the Gaussian multi-RE
convention; `w` is the slope covariate or ones). θ = [β_μ; β_σ; (β_ν;) log σ_k].
Blocks `:mu`, `:sigma`, (`:nu`,) `:resd`. Tagged `marginal = :Laplace`.
"""
function _fit_indep_re_laplace(fam, kind::Val, y, Xμ, Xσ, Xν, comps, nmμ, nmσ, nmν, g_tol)
    n = length(y); yv = Float64.(y)
    has_nu = Xν !== nothing
    pμ, pσ = size(Xμ, 2), size(Xσ, 2); pν = has_nu ? size(Xν, 2) : 0
    K = length(comps)
    Gks = [c[3] for c in comps]
    offs = cumsum([0; Gks]); q = offs[end]
    W = [Float64.(c[1]) for c in comps]
    cols = [offs[k] .+ comps[k][2] for k in 1:K]       # column of b loaded by obs i, comp k
    compof = vcat([fill(k, Gks[k]) for k in 1:K]...)
    k0 = pμ + pσ + pν                                  # last fixed-effect index

    function joint_terms(b, η0, ησ, ην, invs; expected::Bool = false)
        T = promote_type(eltype(b), eltype(η0), eltype(ησ), eltype(ην), eltype(invs))
        H = zeros(T, q, q); grad = zeros(T, q); data = zero(T)
        @inbounds for i in 1:n
            η = η0[i]
            for k in 1:K
                η += W[k][i] * b[cols[k][i]]
            end
            v, g1, w, wE = _indep_obs_terms(kind, yv[i], η, ησ[i], ην[i])
            expected && (w = wE)
            data += v
            for k in 1:K
                wk = W[k][i]; iszero(wk) && continue
                ck = cols[k][i]
                grad[ck] += wk * g1
                for l in 1:K
                    wl = W[l][i]; iszero(wl) && continue
                    H[ck, cols[l][i]] += wk * wl * w
                end
            end
        end
        pen = zero(T)
        @inbounds for j in 1:q
            iv = invs[compof[j]]
            grad[j] += iv * b[j]; H[j, j] += iv
            pen += iv * b[j]^2
        end
        return data + 0.5 * pen, grad, H
    end
    function inner_mode(η0, ησ, ην, invs, b0; maxiter::Int = 200, tol::Real = 1e-10)
        b = copy(b0)
        for _ in 1:maxiter
            J0, grad, H = joint_terms(b, η0, ησ, ην, invs)
            isfinite(J0) || return b, false
            ch = cholesky(Symmetric(H); check = false)
            if !issuccess(ch)                          # indefinite: expected-information direction
                ch = cholesky(Symmetric(last(joint_terms(b, η0, ησ, ην, invs; expected = true)));
                              check = false)
                issuccess(ch) || return b, false
            end
            step = ch \ grad
            norm(step) <= tol * (1 + norm(b)) && return b, true
            α = 1.0; accepted = false
            while α >= 1e-10
                trial = b .- α .* step
                Jt = first(joint_terms(trial, η0, ησ, ην, invs))
                if isfinite(Jt) && Jt <= J0
                    b = trial; accepted = true; break
                end
                α /= 2
            end
            accepted || return b, norm(grad, Inf) <= 1e-6 * (1 + n)
        end
        return b, false
    end

    last_b = zeros(q)
    function nll(θ)
        T = eltype(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]
        ls = θ[k0+1:k0+K]
        any(l -> abs(_indep_primal(l)) > 30, ls) && return T(1e18)
        η0 = Xμ * βμ; ησ = Xσ * βσ
        ην = has_nu ? Xν * θ[pμ+pσ+1:k0] : zeros(T, n)
        any(l -> abs(_indep_primal(l)) > 30, ησ) && return T(1e18)
        invs = exp.(-2 .* ls)
        f0 = (_indep_primal.(η0), _indep_primal.(ησ), _indep_primal.(ην), _indep_primal.(invs))
        b̂, ok = inner_mode(f0..., last_b)
        ok || ((b̂, ok) = inner_mode(f0..., zeros(q)))
        ok || return T(1e18)
        T === Float64 && (last_b .= b̂)
        b = Vector{T}(b̂)
        for _ in 1:2                                   # lift b̂ to b̂(θ) in dual arithmetic
            _, grad, H = joint_terms(b, η0, ησ, ην, invs)
            ch = cholesky(Symmetric(H); check = false)
            issuccess(ch) || return T(1e18)
            b = b .- (ch \ grad)
        end
        J, _, H = joint_terms(b, η0, ησ, ην, invs)
        ch = cholesky(Symmetric(H); check = false)
        issuccess(ch) || return T(1e18)
        val = J + sum(Gks[k] * ls[k] for k in 1:K) + 0.5 * logdet(ch)
        isfinite(val) || return T(1e18)
        return val
    end

    # Interior start. Gamma: the fixed-effects fit (shape and mean are well behaved).
    # Student: OLS mean, sigma at half the residual spread, nu = 2 + e^eta at 12 — NOT the
    # fixed-effects Student fit, which sits at the nu -> 2 boundary when the random
    # effects are ignored and then stalls there (ChickWeight: 18 logLik short of drmTMB).
    θ0 = zeros(k0 + K)
    if has_nu
        βμ0 = Xμ \ yv
        θ0[1:pμ] .= βμ0
        θ0[pμ+1] = log(std(yv - Xμ * βμ0) + eps()) - log(2.0)
        θ0[pμ+pσ+1] = log(10.0)
    else
        fixed = _fit_gamma(fam, yv, Xμ, Xσ, nmμ, nmσ, g_tol)
        θ0[1:k0] .= fixed.theta[1:k0]
    end
    base = has_nu ? θ0[pμ+1] : log(0.3)
    for k in 1:K
        sw = all(==(1.0), W[k]) ? 1.0 : max(std(W[k]), 1e-8)
        θ0[k0+k] = base - log(sw)
    end
    # BFGS first: on the near-flat slope-variance ridge L-BFGS stalled at a 1e18 line-search
    # probe (simulated growth data) while BFGS reaches drmTMB's optimum in 5–25 iterations.
    opts = Optim.Options(g_tol = g_tol, iterations = 1000, f_reltol = NaN, x_abstol = NaN, x_reltol = NaN)
    res = Optim.optimize(nll, θ0, Optim.BFGS(), opts; autodiff = :forward)
    for k in 1:8                                       # bounded restarts (see student.jl #827)
        (Optim.converged(res) && nll(Optim.minimizer(res)) < 1e17) && break
        res2 = Optim.optimize(nll, Optim.minimizer(res), isodd(k) ? Optim.LBFGS() : Optim.BFGS(),
                              opts; autodiff = :forward)
        nll(Optim.minimizer(res2)) < nll(Optim.minimizer(res)) || break
        res = res2
    end
    θ̂ = Optim.minimizer(res)
    nllhat = nll(θ̂)
    gfinal = ForwardDiff.gradient(nll, θ̂)
    converged = nllhat < 1e17 && all(isfinite, gfinal) &&
                (Optim.converged(res) || _laplace_outer_converged(res, nllhat, gfinal, θ̂, n, g_tol))
    famname = has_nu ? "Student" : "Gamma"
    V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂);
                           context = "Laplace $famname (independent random intercepts/slopes)")
    labels = [c[4] for c in comps]
    blocks = Pair{Symbol,UnitRange{Int}}[:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ)]
    names = Pair{Symbol,Vector{String}}[:mu => nmμ, :sigma => nmσ]
    has_nu && (push!(blocks, :nu => (pμ+pσ+1):k0); push!(names, :nu => nmν))
    push!(blocks, :resd => (k0+1):(k0+K)); push!(names, :resd => labels)
    ημ̂ = Xμ * θ̂[1:pμ]
    means = Dict(:mu => has_nu ? ημ̂ : exp.(ημ̂)); obs = Dict(:mu => yv)   # population mean (b = 0)
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]))
    has_nu && (scales[:nu] = 2 .+ exp.(Xν * θ̂[(pμ+pσ+1):k0]))
    fit = DrmFit(fam, blocks, names, θ̂, V, -nllhat, n, converged, means, obs, scales)
    return _withiterations(_withmarginal(_withnll(fit, nll), :Laplace), Optim.iterations(res))
end

# Build `(w, gidx, G, label)` components for a mean made only of `(1 | g)` and
# `(0 + x | g)` terms with at least one slope; `nothing` otherwise (the caller
# keeps its existing routes). Slope labels are `"g:x"` so `vc`/`re_sd` keep the
# intercept and slope SDs of one factor apart.
function _has_indep_slopes(re)
    kinds = [_re_kind(r[1])[1] for r in re]
    return all(in((:intercept, :slope)), kinds) && any(==(:slope), kinds)
end

function _indep_slope_comps(re, data, n::Int)
    _has_indep_slopes(re) || return nothing
    kinds = [_re_kind(r[1]) for r in re]
    return map(zip(re, kinds)) do ((_, grp), (kind, var))
        gidx, G = _group_index(getproperty(data, grp))
        w = kind === :intercept ? ones(n) : Float64.(getproperty(data, var))
        (w, gidx, G, kind === :intercept ? String(grp) : "$(grp):$(var)")
    end
end
