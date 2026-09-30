# zeroonebeta.jl — Zero-one-inflated beta family for responses on the CLOSED
# interval [0,1] (proportions that can be exactly 0 or 1). A three-part mixture
# (drmTMB's `zero_one_beta`): with probability `zoi` the value is a boundary
# (P(1|boundary) = `coi`); otherwise a Beta(μ, φ) on (0,1).
#
#   P(y=0)        = zoi·(1-coi)
#   P(y=1)        = zoi·coi
#   f(y∈(0,1))    = (1-zoi)·Beta(y; μφ, (1-μ)φ)
#
# Parameters: mu (logit) / sigma (log, φ=1/σ²) / zoi (logit) / coi (logit). Fixed
# effects, ML. `Distributions.Beta` is used qualified.

import Distributions

"""
    ZeroOneBeta()

Zero-one-inflated beta family for proportions on the closed interval `[0,1]`
(values may be exactly 0 or 1). Parameters: `mu` (logit; beta mean on the
interior), `sigma` (log; precision `φ = 1/σ²`), `zoi` (logit; probability the
value is a boundary 0/1), `coi` (logit; probability of 1 given a boundary).
Mirrors `drmTMB`'s `zero_one_beta`. `fitted` returns the unconditional mean
`(1-zoi)·μ + zoi·coi`.

A random intercept on the mean, `(1 | g)`, is admitted the same way `Beta()`
admits it: `b_g ~ N(0, σ_b²)` on the logit mean is integrated out per group by
32-node Gauss–Hermite quadrature (mirrors drmTMB's ordinary RI on `zero_one_beta()`'s
`mu`, #723); `sigma`, `zoi`, `coi` stay fixed-effects-only.

A phylogenetic/relatedness random intercept on the mean, `phylo(1 | species)` /
`relmat(1 | id)` (with `K = C`) / `animal(1 | id)` (`A = C`) / `spatial(1 | id)`
(`K = C`), uses the sparse-Laplace mean-phylo engine (#739): `sigma`, `zoi ~ 1`,
`coi ~ 1` (intercept-only) stay fixed effects; `zoi`/`coi` are estimated by
their closed-form Bernoulli MLEs (the atom/coi mixture is completely separable
from the mean/dispersion/phylo likelihood — see `_zeroonebeta_laplace_setup`),
and the Beta interior likelihood on the mean uses the same verified
`:beta_fixed`-derived phylo/relmat spine as `Beta()`.

```julia
fit = drm(bf(y ~ x, sigma ~ 1, zoi ~ 1, coi ~ 1), ZeroOneBeta(); data = dat)
fit = drm(bf(y ~ x + (1 | g), sigma ~ 1, zoi ~ 1, coi ~ 1), ZeroOneBeta(); data = dat)
fit_phy = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1),
                 @formula(zoi ~ 1), @formula(coi ~ 1)),
              ZeroOneBeta(); data = dat, tree = tr, se = false)
```
"""
struct ZeroOneBeta end

function drm(f::DrmFormula, fam::ZeroOneBeta; data, tree = nothing, K = nothing,
             A = nothing, coords = nothing, g_tol::Real = 1e-8, se::Bool = true)
    missing_fit = _fit_observed_response_rows(f, data) do data_observed
        drm(f, fam; data = data_observed, tree = tree, K = K, A = A,
            coords = coords, g_tol = g_tol, se = se)
    end
    missing_fit !== nothing && return missing_fit

    _lss_only_gaussian_guard(f, fam)   # #544: refuse, never silently drop, sd() parts
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    mv === nothing ||
        error("ZeroOneBeta() does not support meta_V markers")
    for (pname, r) in f.forms          # only the mean may carry a random effect
        pname === :mu && continue
        _, re2, mv2, st2 = _split_ranef(r)
        (isempty(re2) && mv2 === nothing && st2 === nothing) ||
            error("ZeroOneBeta(): only the mean formula may carry a random effect")
    end
    y, Xμ, nmμ = _design(f.response, fixed_mu, data)
    _, Xσ, nmσ = _design(f.response, get(rhs, :sigma, ConstantTerm(1)), data)
    _, Xz, nmz = _design(f.response, get(rhs, :zoi, ConstantTerm(1)), data)
    _, Xc, nmc = _design(f.response, get(rhs, :coi, ConstantTerm(1)), data)
    all(yi -> 0 <= yi <= 1, y) ||
        error("ZeroOneBeta() requires responses in the closed interval [0, 1]")
    if st !== nothing
        isempty(re) ||
            error("ZeroOneBeta() phylo structured effects cannot be combined with ordinary random effects yet")
        size(Xσ, 2) == 1 && all(x -> x == 1.0, @view Xσ[:, 1]) ||
            error("ZeroOneBeta() phylo/relmat route currently supports a constant `sigma ~ 1` formula")
        size(Xz, 2) == 1 && all(x -> x == 1.0, @view Xz[:, 1]) ||
            error("ZeroOneBeta() phylo/relmat route currently supports a constant `zoi ~ 1` formula")
        size(Xc, 2) == 1 && all(x -> x == 1.0, @view Xc[:, 1]) ||
            error("ZeroOneBeta() phylo/relmat route currently supports a constant `coi ~ 1` formula")
        kind, grp = st
        labels = getproperty(data, grp)
        if kind === :phylo
            tree === nothing && error("phylo(1 | $grp) needs `tree = ...`")
            return _withformula(
                _fit_zeroonebeta_phylo_laplace(fam, y, Xμ, nmμ, nmσ, nmz, nmc, labels, tree, grp, g_tol; se = se), f)
        elseif kind === :relmat || kind === :animal || kind === :spatial
            C = _poisson_structured_cov(kind, grp, K, A, coords)
            return _withformula(
                _fit_zeroonebeta_relmat_laplace(fam, y, Xμ, nmμ, nmσ, nmz, nmc, C, labels, grp, g_tol; se = se), f)
        else
            error("ZeroOneBeta() supports phylo/relmat/animal/spatial(1 | group) among structured markers")
        end
    end
    if !isempty(re)                    # random effect on the logit mean → GHQ (#723)
        length(re) == 1 ||
            error("ZeroOneBeta() supports a single random intercept `(1 | g)` on the mean")
        (rk, _) = _re_kind(re[1][1]); grp = re[1][2]
        rk === :intercept ||
            error("ZeroOneBeta() supports `(1 | g)` on the mean, not `(0 + x | g)` or `(1 + x | g)`")
        gidx, G = _group_index(getproperty(data, grp))
        return _withformula(
            _fit_zeroonebeta_ranef(fam, y, Xμ, Xσ, Xz, Xc, gidx, G, nmμ, nmσ, nmz, nmc, grp, g_tol), f)
    end
    return _withformula(_fit_zeroonebeta(fam, y, Xμ, Xσ, Xz, Xc, nmμ, nmσ, nmz, nmc, g_tol), f)
end

function _fit_zeroonebeta(fam::ZeroOneBeta, y, Xμ, Xσ, Xz, Xc, nmμ, nmσ, nmz, nmc, g_tol)
    n = length(y)
    pμ, pσ, pz, pc = size(Xμ, 2), size(Xσ, 2), size(Xz, 2), size(Xc, 2)
    i1 = pμ + pσ; i2 = i1 + pz; i3 = i2 + pc
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:i1]; βz = θ[i1+1:i2]; βc = θ[i2+1:i3]
        ημ = clamp.(Xμ * βμ, -30.0, 30.0); ησ = clamp.(Xσ * βσ, -15.0, 15.0)
        ηz = clamp.(Xz * βz, -30.0, 30.0); ηc = clamp.(Xc * βc, -30.0, 30.0)
        s = zero(eltype(θ))
        @inbounds for i in 1:n
            lzoi = _log_logistic(ηz[i]); l1mzoi = _log1m_logistic(ηz[i])
            if y[i] == 0
                s -= lzoi + _log1m_logistic(ηc[i])      # log zoi + log(1-coi)
            elseif y[i] == 1
                s -= lzoi + _log_logistic(ηc[i])        # log zoi + log coi
            else
                μ = _logistic(ημ[i]); φ = exp(-2 * ησ[i])
                s -= l1mzoi + Distributions.logpdf(Distributions.Beta(μ * φ, (1 - μ) * φ), y[i])
            end
        end
        return s
    end
    # initialisations from the empirical mixture
    isb = (y .== 0) .| (y .== 1)
    cont = y[.!isb]; p̄ = isempty(cont) ? 0.5 : clamp(sum(cont) / length(cont), 1e-3, 1 - 1e-3)
    fb = clamp(sum(isb) / n, 1e-3, 1 - 1e-3)
    nb = sum(isb); co0 = nb == 0 ? 0.5 : clamp(sum(y .== 1) / nb, 1e-3, 1 - 1e-3)
    θ0 = zeros(i3)
    θ0[1] = log(p̄ / (1 - p̄))                 # logit μ
    θ0[pμ+1] = -0.5 * log(10.0)               # σ (φ ≈ 10)
    θ0[i1+1] = log(fb / (1 - fb))             # logit zoi
    θ0[i2+1] = log(co0 / (1 - co0))           # logit coi
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):i1, :zoi => (i1+1):i2, :coi => (i2+1):i3]
    names = [:mu => nmμ, :sigma => nmσ, :zoi => nmz, :coi => nmc]
    μ̂ = _logistic.(Xμ * θ̂[1:pμ])
    zoî = _logistic.(Xz * θ̂[(i1+1):i2]); coî = _logistic.(Xc * θ̂[(i2+1):i3])
    means = Dict(:mu => (1 .- zoî) .* μ̂ .+ zoî .* coî)   # unconditional mean (drmTMB fitted)
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:beta_mu => μ̂,
                  :sigma => exp.(Xσ * θ̂[(pμ+1):i1]),
                  :zoi => zoî,
                  :coi => coî)
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# ZeroOneBeta with a random intercept (1|g) on the logit mean (#723). Mirrors
# `_fit_beta_ranef` (beta.jl): b_g ~ N(0,σ_b²) integrated out per group by
# 32-node Gauss–Hermite quadrature; sigma/zoi/coi stay fixed effects (they do
# not depend on the group random effect, so only the Beta-interior term inside
# the quadrature sum changes with δ = √2·σ_b·z_k — the atom log-probabilities
# (log zoi + log(1-coi) / log zoi + log coi) are added at every node unchanged).
function _fit_zeroonebeta_ranef(fam::ZeroOneBeta, y, Xμ, Xσ, Xz, Xc, gidx, G, nmμ, nmσ, nmz, nmc, grp, g_tol)
    n = length(y)
    pμ, pσ, pz, pc = size(Xμ, 2), size(Xσ, 2), size(Xz, 2), size(Xc, 2)
    i1 = pμ + pσ; i2 = i1 + pz; i3 = i2 + pc
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    z, w = _gauss_hermite(32); logw = log.(w); K = length(z); rt2 = sqrt(2.0); lπ = log(π)
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:i1]; βz = θ[i1+1:i2]; βc = θ[i2+1:i3]
        σb = exp(θ[i3+1])
        η0 = Xμ * βμ; ησ = clamp.(Xσ * βσ, -15.0, 15.0)
        ηz = clamp.(Xz * βz, -30.0, 30.0); ηc = clamp.(Xc * βc, -30.0, 30.0)
        s = zero(eltype(θ))
        for idx in members
            isempty(idx) && continue
            terms = Vector{eltype(θ)}(undef, K)
            for k in 1:K
                δ = rt2 * σb * z[k]; gll = logw[k]
                for i in idx
                    lzoi = _log_logistic(ηz[i]); l1mzoi = _log1m_logistic(ηz[i])
                    if y[i] == 0
                        gll += lzoi + _log1m_logistic(ηc[i])
                    elseif y[i] == 1
                        gll += lzoi + _log_logistic(ηc[i])
                    else
                        μ = _logistic(clamp(η0[i] + δ, -30.0, 30.0)); φ = exp(-2 * ησ[i])
                        gll += l1mzoi + Distributions.logpdf(Distributions.Beta(μ * φ, (1 - μ) * φ), y[i])
                    end
                end
                terms[k] = gll
            end
            mx = maximum(terms)
            s -= (-0.5 * lπ + mx + log(sum(exp.(terms .- mx))))
        end
        return s
    end
    # initialisations from the empirical mixture (as in `_fit_zeroonebeta`), plus a
    # moderate random-intercept SD start.
    isb = (y .== 0) .| (y .== 1)
    cont = y[.!isb]; p̄ = isempty(cont) ? 0.5 : clamp(sum(cont) / length(cont), 1e-3, 1 - 1e-3)
    fb = clamp(sum(isb) / n, 1e-3, 1 - 1e-3)
    nb = sum(isb); co0 = nb == 0 ? 0.5 : clamp(sum(y .== 1) / nb, 1e-3, 1 - 1e-3)
    θ0 = zeros(i3 + 1)
    θ0[1] = log(p̄ / (1 - p̄))                 # logit μ
    θ0[pμ+1] = -0.5 * log(10.0)               # σ (φ ≈ 10)
    θ0[i1+1] = log(fb / (1 - fb))             # logit zoi
    θ0[i2+1] = log(co0 / (1 - co0))           # logit coi
    θ0[i3+1] = log(0.5)                        # log σ_b
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):i1, :zoi => (i1+1):i2, :coi => (i2+1):i3,
              :resd => (i3+1):(i3+1)]
    names = [:mu => nmμ, :sigma => nmσ, :zoi => nmz, :coi => nmc, :resd => [String(grp)]]
    μ̂ = _logistic.(Xμ * θ̂[1:pμ])   # fixed-effect (b=0) conditional mean, as `_fit_beta_ranef` reports
    zoî = _logistic.(Xz * θ̂[(i1+1):i2]); coî = _logistic.(Xc * θ̂[(i2+1):i3])
    means = Dict(:mu => (1 .- zoî) .* μ̂ .+ zoî .* coî)   # unconditional mean (drmTMB fitted)
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:beta_mu => μ̂,
                  :sigma => exp.(Xσ * θ̂[(pμ+1):i1]),
                  :zoi => zoî,
                  :coi => coî)
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# ---- Phylo/relmat structured random intercept on the mean (#739) -----------
# `zoi`/`coi` (intercept-only, `zoi ~ 1` / `coi ~ 1`) are completely separable
# from the mean/dispersion/phylo likelihood: their log-likelihood contribution
# (`log zoi`/`log(1-zoi)` for the atom/interior split, `log coi`/`log(1-coi)`
# for the 1-vs-0 split among atoms) does not depend on η, φ, or the phylo
# random effect at all, so their joint MLE is exactly the closed-form Bernoulli
# MLE computed here — no separate optimization axis is needed in the
# nuisance-Laplace spine (`Val(:zeroonebeta_fixed)`, sparse_laplace_glmm.jl,
# treats them as fixed `aux` fields). This mirrors `_beta_laplace_setup` for
# the mean/dispersion piece, restricted to the interior (non-atom) rows.
function _zeroonebeta_laplace_setup(y, Xμ)
    yv = Float64.(y)
    isatom = (yv .== 0) .| (yv .== 1)
    n = length(yv)
    nb = sum(isatom)
    zoi = clamp(nb / n, 1e-6, 1 - 1e-6)
    coi = nb == 0 ? 0.5 : clamp(sum(yv .== 1) / nb, 1e-6, 1 - 1e-6)
    ylogit = [isatom[i] ? 0.0 : (log(yv[i]) - log1p(-yv[i])) for i in 1:n]
    function aux_from(logsigma)
        φ = exp(clamp(-2 * logsigma, -8.0, 8.0))
        return (y = yv, precision = φ, ylogit = ylogit,
                lgammaφ = loggamma(φ), digammaφ = digamma(φ),
                zoi = zoi, coi = coi, isatom = isatom)
    end
    cont = yv[.!isatom]
    ȳ = isempty(cont) ? 0.5 : clamp(sum(cont) / length(cont), 1e-4, 1 - 1e-4)
    v = length(cont) > 1 ? sum(abs2, cont .- ȳ) / (length(cont) - 1) : 0.1
    φ0 = max(ȳ * (1 - ȳ) / max(v, eps()) - 1, 0.5)
    θβ0 = zeros(size(Xμ, 2))
    θβ0[1] = log(ȳ / (1 - ȳ))
    return aux_from, θβ0, -0.5 * log(φ0), zoi, coi
end

# Append the closed-form zoi/coi block (Bernoulli MLEs, asymptotic variance
# 1/(n·p·(1-p)) on the logit scale, independent of the phylo/dispersion block
# by the separability above) onto a `_fit_phylo_mean_laplace_nuisance` /
# `_fit_general_mean_laplace_nuisance` fit (`:mu`/`:sigma`/`:resd` blocks).
function _augment_zeroonebeta_fit(fit::DrmFit, zoi, coi, nmz, nmc, y, n, nb)
    p0 = length(fit.theta)
    θ̂ = vcat(fit.theta, log(zoi / (1 - zoi)), log(coi / (1 - coi)))
    V = zeros(p0 + 2, p0 + 2)
    V[1:p0, 1:p0] .= fit.vcov
    V[p0+1, p0+1] = 1 / (n * zoi * (1 - zoi))
    V[p0+2, p0+2] = nb == 0 ? NaN : 1 / (nb * coi * (1 - coi))
    blocks = vcat(fit.blocks, [:zoi => (p0+1):(p0+1), :coi => (p0+2):(p0+2)])
    names = vcat(fit.coefnames, [:zoi => nmz, :coi => nmc])
    μ̂ = fit.means[:mu]                              # `:beta_fixed`-kind fixed-effect (b=0) conditional mean
    means = merge(fit.means, Dict(:mu => (1 - zoi) .* μ̂ .+ zoi * coi))   # unconditional mean (drmTMB fitted)
    obs = merge(fit.obs, Dict(:mu => Vector{Float64}(y)))
    scales = merge(fit.scales, Dict(:beta_mu => μ̂, :zoi => fill(zoi, n), :coi => fill(coi, n)))
    return DrmFit(fit.family, blocks, names, θ̂, V, fit.loglik, fit.nobs, fit.converged,
                  means, obs, scales, fit.formula, fit.nll, fit.nllgrad, fit.ranef,
                  fit.estim_method, fit.reml_loglik, fit.ml_loglik, fit.marginal,
                  fit.phylo_penalty, fit.penalty, fit.iterations, fit.phylo_scale)
end

"""
    _fit_zeroonebeta_phylo_laplace(fam, y, Xμ, nmμ, nmσ, nmz, nmc, labels, tree, grp, g_tol; se)

`ZeroOneBeta()` sparse-Laplace fit with a phylogenetic random intercept
`phylo(1 | grp)` on the logit mean (#739). `zoi ~ 1` / `coi ~ 1` are fixed by
their closed-form Bernoulli MLEs (`_zeroonebeta_laplace_setup`); the interior
Beta likelihood on the mean reuses the verified `:beta_fixed`-derived
`_fit_phylo_mean_laplace_nuisance` spine via the additive `:zeroonebeta_fixed`
kernel (atom rows contribute zero η-derivatives; sparse_laplace_glmm.jl).
"""
function _fit_zeroonebeta_phylo_laplace(fam, y, Xμ, nmμ, nmσ, nmz, nmc, labels,
                                        tree, grp, g_tol; se::Bool = true,
                                        polish_iterations::Int = 0)
    n = length(y)
    nb = sum((Float64.(y) .== 0) .| (Float64.(y) .== 1))
    aux_from, θβ0, θσ0, zoi, coi = _zeroonebeta_laplace_setup(y, Xμ)
    fit = _fit_phylo_mean_laplace_nuisance(
        fam, Val(:zeroonebeta_fixed), aux_from, n, Xμ, labels, tree, nmμ, nmσ,
        grp, g_tol; θβ0 = θβ0, θσ0 = θσ0, sigma_scale = exp,
        se = se, polish_iterations = polish_iterations
    )
    return _augment_zeroonebeta_fit(fit, zoi, coi, nmz, nmc, y, n, nb)
end

"""
    _fit_zeroonebeta_relmat_laplace(fam, y, Xμ, nmμ, nmσ, nmz, nmc, C, labels, grp, g_tol; se)

`ZeroOneBeta()` sparse-Laplace fit with a general user-supplied PD covariance
`C` on the logit-mean random intercept (`relmat`/`animal`/`spatial(1 | grp)`;
#739). Reuses the phylo nuisance spine via `_general_cov_setup`, exactly as
`_fit_beta_relmat_laplace` does for `Beta()`.
"""
function _fit_zeroonebeta_relmat_laplace(fam, y, Xμ, nmμ, nmσ, nmz, nmc, C,
                                         labels, grp, g_tol; se::Bool = true,
                                         polish_iterations::Int = 0)
    n = length(y)
    nb = sum((Float64.(y) .== 0) .| (Float64.(y) .== 1))
    Q, leaf_node = _general_cov_setup(C, labels)
    aux_from, θβ0, θσ0, zoi, coi = _zeroonebeta_laplace_setup(y, Xμ)
    fit = _fit_general_mean_laplace_nuisance(
        fam, Val(:zeroonebeta_fixed), aux_from, n, Xμ, Q, leaf_node, nmμ, nmσ,
        grp, g_tol; θβ0 = θβ0, θσ0 = θσ0, sigma_scale = exp,
        se = se, polish_iterations = polish_iterations
    )
    return _augment_zeroonebeta_fit(fit, zoi, coi, nmz, nmc, y, n, nb)
end
