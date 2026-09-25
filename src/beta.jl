# beta.jl — Beta family for responses on the open interval (0,1) (proportions,
# rates, probabilities). Logit link on the mean μ; the `sigma` slot carries σ
# with drmTMB's precision mapping φ = 1/σ² (so coef(:sigma) is log σ). The
# likelihood is Beta(μφ, (1-μ)φ): mean μ, variance μ(1-μ)/(1+φ). Fixed effects,
# ML. `Distributions.Beta` is used qualified — DRModels exports its own `Beta` family.

import Distributions

"""
    Beta()

Beta response family for proportions in `(0,1)`: logit link on the mean `μ`, and
the `sigma` slot carries `σ` with the precision mapping `φ = 1/σ²` (so
`coef(fit, :sigma)` is `log σ`; recover precision as `exp(-2·log σ)`). Likelihood
`Beta(μφ, (1-μ)φ)`. Mirrors `drmTMB`'s `beta_family`.
Crossed random intercepts on the mean, such as `(1 | g) + (1 | h)`, use the
sparse-Laplace engine when `sigma ~ 1`. A phylogenetic random intercept on the
mean, `phylo(1 | species)`, also uses the sparse-Laplace engine, as do general
user-supplied PD-covariance intercepts — `relmat(1 | id)` with `K = C`, the
`animal(1 | id)` (`A = C`) and `spatial(1 | id)` (`K = C`) aliases.

!!! note
    `DRModels.Beta` shadows `Distributions.Beta`; qualify the latter if you need it.

```julia
fit = drm(bf(y ~ x, sigma ~ 1), Beta(); data = dat)
fit = drm(bf(y ~ x + (1 | g) + (1 | h), sigma ~ 1), Beta(); data = dat)
fit_phy = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
              Beta(); data = dat, tree = tr, se = false)
exp(-2 * coef(fit, :sigma)[1])     # estimated precision φ

# Experimental (#136 Rung 1): Beta random-intercept variational (ELBO) marginal.
# Requires `sigma ~ 1`. Default remains Laplace (`marginal = :LA`).
fit_va = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Beta();
             data = dat, marginal = :VA)
```
"""
struct Beta end

_logistic(η) = 1 / (1 + exp(-η))

function drm(f::DrmFormula, fam::Beta; data, tree = nothing, K = nothing,
             A = nothing, coords = nothing, g_tol::Real = 1e-8, se::Bool = true,
             marginal::Symbol = :LA, method = nothing)
    _reject_method_as_marginal(fam, method)
    _scalar_laplace_requested(marginal) &&     # Arc 2: TMB-convention Laplace, ordinary (1 | g)
        return _drm_ordinary_laplace(f, fam; data = data, tree = tree, K = K, A = A,
                                     coords = coords, g_tol = g_tol, se = se, method = method)
    missing_fit = _fit_observed_response_rows(f, data) do data_observed
        drm(f, fam; data = data_observed, tree = tree, K = K, A = A,
            coords = coords, g_tol = g_tol, se = se, marginal = marginal, method = method)
    end
    missing_fit !== nothing && return missing_fit

    marg = _marginal_method(marginal)                     # :LA (default) or :VA (#136)
    marg isa AGHQ && _aghq_reject(fam, "this family")
    isva = marg isa Variational
    _lss_only_gaussian_guard(f, fam)   # #544: refuse, never silently drop, sd() parts
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    mv === nothing ||
        error("Beta() does not support meta_V markers")
    for (pname, r) in f.forms          # only the mean may carry a random effect
        pname === :mu && continue
        _, re2, mv2, st2 = _split_ranef(r)
        (isempty(re2) && mv2 === nothing && st2 === nothing) ||
            error("Beta(): only the mean formula may carry a random effect")
    end
    y, Xμ, nmμ = _design(f.response, fixed_mu, data)
    _, Xσ, nmσ = _design(f.response, get(rhs, :sigma, ConstantTerm(1)), data)
    all(yi -> 0 < yi < 1, y) ||
        error("Beta() requires responses strictly in the open interval (0, 1)")
    if st !== nothing
        isva && _va_reject(fam, "a phylogenetic/structured random effect")
        isempty(re) ||
            error("Beta() phylo structured effects cannot be combined with ordinary random effects yet")
        # `sigma ~ 1` keeps the scalar-dispersion spine; a covariate `sigma`
        # formula routes to the per-observation log-dispersion path (#164).
        kind, grp = st
        labels = getproperty(data, grp)
        if kind === :phylo
            tree === nothing && error("phylo(1 | $grp) needs `tree = ...`")
            return _withformula(_fit_beta_phylo_laplace(fam, y, Xμ, Xσ, labels, tree, nmμ, nmσ, grp, g_tol; se = se), f)
        elseif kind === :relmat || kind === :animal || kind === :spatial
            # General user-supplied PD covariance C on the mean (logit) intercept
            # (relatedness / animal model / precomputed spatial), with the Beta
            # precision a fixed nuisance. Reuses the phylo nuisance Laplace spine
            # with the tree precision swapped for C⁻¹ (#167).
            C = _poisson_structured_cov(kind, grp, K, A, coords)
            return _withformula(_fit_beta_relmat_laplace(fam, y, Xμ, Xσ, C, labels, nmμ, nmσ, grp, g_tol; se = se), f)
        else
            error("Beta() supports phylo/relmat/animal/spatial(1 | group) among structured markers")
        end
    end
    if !isempty(re)                    # random effect on the logit mean → GHQ/Laplace or VA (#136)
        if length(re) > 1
            isva && _va_reject(fam, "crossed/multiple random intercepts")
            all(_re_kind(r[1])[1] === :intercept for r in re) ||
                error("Beta() supports multiple random effects only as crossed/nested intercepts, e.g. `(1 | g) + (1 | h)`")
            comps = map(re) do r
                grp = r[2]; gidx, G = _group_index(getproperty(data, grp))
                (ones(length(y)), gidx, G, String(grp))
            end
            return _withformula(_fit_beta_crossed_laplace(fam, y, Xμ, Xσ, comps, nmμ, nmσ, g_tol), f)
        end
        (rk, var) = _re_kind(re[1][1]); grp = re[1][2]
        gidx, G = _group_index(getproperty(data, grp))
        if rk === :intercept            # (1 | g) → 1-D Gauss–Hermite (Laplace) or VA (#136)
            if isva
                size(Xσ, 2) == 1 || _va_reject(fam, "a non-constant `sigma` formula (VA kernels require `sigma ~ 1`)")
                return _withformula(_withmarginal(
                    _fit_beta_ranef_va(fam, y, Xμ, Xσ, gidx, G, nmμ, nmσ, grp, g_tol), :VA), f)
            end
            return _withformula(_fit_beta_ranef(fam, y, Xμ, Xσ, gidx, G, nmμ, nmσ, grp, g_tol), f)
        elseif rk === :corr             # (1 + x | g) → 2-D Gauss–Hermite marginal
            isva && _va_reject(fam, "a correlated random slope `(1 + x | g)`")
            return _withformula(_fit_beta_corr_ranef(fam, y, Xμ, Xσ, Float64.(getproperty(data, var)), gidx, G, nmμ, nmσ, grp, g_tol), f)
        else
            error("Beta() supports `(1 | g)` or `(1 + x | g)` on the mean, not `(0 + x | g)`")
        end
    end
    isva && _va_reject(fam, "no random intercept (fixed-effects-only)")
    return _withformula(_fit_beta(fam, y, Xμ, Xσ, nmμ, nmσ, g_tol), f)
end

# Beta GLMM with a random intercept (1|g) on the logit mean. b_g ~ N(0,σ_b²)
# integrated out per group by 32-node Gauss–Hermite quadrature; precision
# φ = 1/σ² is a fixed effect. Same scheme as the count GLMMs.
function _fit_beta_ranef(fam::Beta, y, Xμ, Xσ, gidx, G, nmμ, nmσ, grp, g_tol)
    n = length(y); pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    z, w = _gauss_hermite(32); logw = log.(w); K = length(z); rt2 = sqrt(2.0); lπ = log(π)
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; σb = exp(θ[pμ+pσ+1])
        η0 = Xμ * βμ; ησ = clamp.(Xσ * βσ, -15.0, 15.0)
        s = zero(eltype(θ))
        for idx in members
            isempty(idx) && continue
            terms = Vector{eltype(θ)}(undef, K)
            for k in 1:K
                δ = rt2 * σb * z[k]; gll = logw[k]
                for i in idx
                    μ = _logistic(clamp(η0[i] + δ, -30.0, 30.0)); φ = exp(-2 * ησ[i])
                    gll += Distributions.logpdf(Distributions.Beta(μ * φ, (1 - μ) * φ), y[i])
                end
                terms[k] = gll
            end
            mx = maximum(terms)
            s -= (-0.5 * lπ + mx + log(sum(exp.(terms .- mx))))
        end
        return s
    end
    ȳ = sum(y) / n; v = sum(abs2, y .- ȳ) / max(n - 1, 1)
    φ0 = max(ȳ * (1 - ȳ) / max(v, eps()) - 1, 0.5)
    θ0 = zeros(pμ + pσ + 1)
    θ0[1] = log(ȳ / (1 - ȳ)); θ0[pμ+1] = -0.5 * log(φ0); θ0[pμ+pσ+1] = log(0.5)
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :resd => (pμ+pσ+1):(pμ+pσ+1)]
    names = [:mu => nmμ, :sigma => nmσ, :resd => [String(grp)]]
    means = Dict(:mu => _logistic.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# Beta GLMM with a correlated random intercept+slope (1 + x | g) on the logit
# mean: per group (b0,b1) ~ N(0, Σ), Σ = L Lᵀ with L = [l11 0; cc l22] (log-
# Cholesky a=log l11, b=log l22, cc). The 2-D group integral is taken by a K×K
# Gauss–Hermite product rule (b = √2 L z), the same scheme as the count GLMMs;
# precision φ = 1/σ² is a fixed effect. θ = [β_μ; β_σ; a, b, cc]. O(G·K²).
function _fit_beta_corr_ranef(fam::Beta, y, Xμ, Xσ, xs, gidx, G, nmμ, nmσ, grp, g_tol)
    n = length(y); pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    z1, w1 = _gauss_hermite(12); lw = log.(w1); K = length(z1); rt2 = sqrt(2.0); lπ = log(π)
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]
        l11 = exp(θ[pμ+pσ+1]); l22 = exp(θ[pμ+pσ+2]); cc = θ[pμ+pσ+3]
        η0 = Xμ * βμ; ησ = clamp.(Xσ * βσ, -15.0, 15.0)
        s = zero(eltype(θ))
        for idx in members
            isempty(idx) && continue
            terms = Vector{eltype(θ)}(undef, K * K); t = 0
            for j in 1:K, k in 1:K
                t += 1
                b0 = rt2 * l11 * z1[j]; b1 = rt2 * (cc * z1[j] + l22 * z1[k])
                gll = lw[j] + lw[k]
                for i in idx
                    μ = _logistic(clamp(η0[i] + b0 + b1 * xs[i], -30.0, 30.0)); φ = exp(-2 * ησ[i])
                    gll += Distributions.logpdf(Distributions.Beta(μ * φ, (1 - μ) * φ), y[i])
                end
                terms[t] = gll
            end
            mx = maximum(terms)
            s -= (-lπ + mx + log(sum(exp.(terms .- mx))))   # 2-D: -log π
        end
        return s
    end
    ȳ = sum(y) / n; v = sum(abs2, y .- ȳ) / max(n - 1, 1)
    φ0 = max(ȳ * (1 - ȳ) / max(v, eps()) - 1, 0.5)
    θ0 = zeros(pμ + pσ + 3)
    θ0[1] = log(ȳ / (1 - ȳ)); θ0[pμ+1] = -0.5 * log(φ0)
    θ0[pμ+pσ+1] = log(0.4); θ0[pμ+pσ+2] = log(0.4); θ0[pμ+pσ+3] = 0.0
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :recov => (pμ+pσ+1):(pμ+pσ+3)]
    names = [:mu => nmμ, :sigma => nmσ, :recov => ["$(grp):L11", "$(grp):L22", "$(grp):L21"]]
    means = Dict(:mu => _logistic.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

function _fit_beta(fam::Beta, y, Xμ, Xσ, nmμ, nmσ, g_tol)
    n = length(y); pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]
        # Soft guards (see `_softclamp` in negbinomial.jl): identity across the
        # original hard band so no legitimate fit is distorted, smooth beyond so
        # the gradient stays live — restoring agreement with drmTMB's smooth guard.
        ημ = _softclamp.(Xμ * βμ, -30.0, 30.0, 3.0)   # μ ∈ (0,1) strictly
        ησ = _softclamp.(Xσ * βσ, -15.0, 15.0, 3.0)   # φ = exp(-2ησ) finite & > 0
        s = zero(eltype(θ))
        @inbounds for i in 1:n
            μ = _logistic(ημ[i]); φ = exp(-2 * ησ[i])
            s -= Distributions.logpdf(Distributions.Beta(μ * φ, (1 - μ) * φ), y[i])
        end
        return s
    end
    ȳ = sum(y) / n; v = sum(abs2, y .- ȳ) / max(n - 1, 1)
    φ0 = max(ȳ * (1 - ȳ) / max(v, eps()) - 1, 0.5)   # method-of-moments precision
    θ0 = zeros(pμ + pσ)
    θ0[1] = log(ȳ / (1 - ȳ))                          # logit mean
    θ0[pμ+1] = -0.5 * log(φ0)                          # σ = 1/√φ ⇒ log σ = -½ log φ
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ)]
    names = [:mu => nmμ, :sigma => nmσ]
    means = Dict(:mu => _logistic.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => Vector{Float64}(y))  # response-scale μ̂
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end
