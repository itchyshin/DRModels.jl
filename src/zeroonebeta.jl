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

```julia
fit = drm(bf(y ~ x, sigma ~ 1, zoi ~ 1, coi ~ 1), ZeroOneBeta(); data = dat)
fit = drm(bf(y ~ x + (1 | g), sigma ~ 1, zoi ~ 1, coi ~ 1), ZeroOneBeta(); data = dat)
```
"""
struct ZeroOneBeta end

function drm(f::DrmFormula, fam::ZeroOneBeta; data, g_tol::Real = 1e-8)
    missing_fit = _fit_observed_response_rows(f, data) do data_observed
        drm(f, fam; data = data_observed, g_tol = g_tol)
    end
    missing_fit !== nothing && return missing_fit

    _lss_only_gaussian_guard(f, fam)   # #544: refuse, never silently drop, sd() parts
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    mv === nothing ||
        error("ZeroOneBeta() does not support meta_V markers")
    st === nothing ||
        error("ZeroOneBeta() does not support phylo/relmat structured random effects yet")
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
