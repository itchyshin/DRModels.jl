# lognormal.jl — LogNormal family for strictly-positive responses whose log is
# Gaussian. The mean formula μ is the mean of log y (identity link on the log
# scale); σ (log link) is the SD of log y. The log-density is the Gaussian
# log-density of log y plus the −log y change-of-variables Jacobian, so the fit
# reuses the Gaussian location–scale objective on log y. Fixed effects + random
# intercept/slope on μ, ML. Mirrors drmTMB's `lognormal`. `Distributions.Normal`
# is used qualified inside the GHQ random-effect paths.

import Distributions

"""
    LogNormal()

LogNormal response family for positive continuous data: the mean formula `μ` is
the **mean of `log y`** (identity link on the log scale), and `σ` (log link) is
the **SD of `log y`**. The response-scale median is `exp(μ)`. Mirrors `drmTMB`'s
`lognormal` family.

!!! note
    `DRModels.LogNormal` shadows `Distributions.LogNormal`; qualify the latter if needed.

A random intercept `(1 | g)` or a correlated random intercept+slope `(1 + x | g)`
may be placed on the log-mean `μ`; the group effect is integrated out by
Gauss–Hermite quadrature (`re_sd(fit)[:g]` / `vc(fit)[:g]`). Crossed/multiple
random intercepts, `(1 | g) + (1 | h)` (#736), delegate the same way as the
structured markers below: `log(y)` is exactly Gaussian, so the fit runs
WHOLESALE through `drm(f, Gaussian(); data = data-with-logged-response)`
(`_fit_multi_ranef_gaussian`, the closed-form exact crossed marginal, not a
Laplace approximation), with the reported log-likelihood shifted by
`-sum(log y)`.

## Structured markers (`phylo`/`relmat`)

`phylo(1 | group)` (needs `tree = ...`) and `relmat(1 | group)` (needs
`K = ...`) may be placed on the mean, exactly as drmTMB's `lognormal` family
supports. `log(y)` is exactly Gaussian, so a structured call delegates
WHOLESALE to `drm(f, Gaussian(); data = data-with-logged-response, tree = ...,
K = ...)` — the same identity `biv_lognormal()` already uses
(`src/bivariate_lognormal.jl`) — and shifts the reported log-likelihood by the
parameter-free Jacobian `-sum(log y)`. `theta`/`vcov`/`ranef` and the fitted
objective's gradient carry over untouched from the Gaussian-on-log(y) fit;
only `loglik` (and everything derived from it: aic/bic/deviance) shifts.
`animal`/`spatial` markers and `meta_V` are not implemented for `LogNormal()`
(no R parity cell) and stay refused, as does any marker on a formula other
than the mean.

```julia
fit = drm(bf(y ~ x, sigma ~ 1), LogNormal(); data = dat)
coef(fit, :mu)                 # on the log scale
exp(coef(fit, :sigma)[1])      # SD of log y

fit_phy = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
              LogNormal(); data = dat, tree = tr)
```
"""
struct LogNormal end

function drm(f::DrmFormula, fam::LogNormal; data, tree = nothing, K = nothing, g_tol::Real = 1e-8)
    missing_fit = _fit_observed_response_rows(f, data) do data_observed
        drm(f, fam; data = data_observed, tree = tree, K = K, g_tol = g_tol)
    end
    missing_fit !== nothing && return missing_fit

    _lss_only_gaussian_guard(f, fam)   # #544: refuse, never silently drop, sd() parts
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    mv === nothing ||
        error("LogNormal() does not support meta_V markers")
    for (pname, r) in f.forms          # only the mean may carry a random effect
        pname === :mu && continue
        _, re2, mv2, st2 = _split_ranef(r)
        (isempty(re2) && mv2 === nothing && st2 === nothing) ||
            error("LogNormal(): only the mean formula may carry a random effect")
    end
    if st !== nothing                  # structured marker on the mean (#563 S8)
        st[1] in (:phylo, :relmat) ||
            error("LogNormal() supports `phylo`/`relmat` structured markers on the mean " *
                  "(got `$(st[1])`); drmTMB's `lognormal` family does not implement " *
                  "`animal`/`spatial` markers")
        isempty(re) ||
            error("LogNormal() supports a single structured marker OR random-effect term " *
                  "on the mean, not both")
        return _withformula(_fit_lognormal_structured(fam, f, data, st[1], st[2]; tree = tree, K = K, g_tol = g_tol), f)
    end
    y, Xμ, nmμ = _design(f.response, fixed_mu, data)
    _, Xσ, nmσ = _design(f.response, get(rhs, :sigma, ConstantTerm(1)), data)
    all(yi -> yi > 0, y) || error("LogNormal() requires strictly positive responses")
    if !isempty(re)                    # random effect on the log-mean μ → GHQ
        if length(re) > 1              # crossed/multiple intercepts (#736)
            all(_re_kind(r[1])[1] === :intercept for r in re) ||
                error("LogNormal() supports multiple random effects only as crossed/nested " *
                      "intercepts, e.g. `(1 | g) + (1 | h)`")
            return _withformula(_fit_lognormal_crossed(fam, f, data, g_tol), f)
        end
        length(re) == 1 ||
            error("LogNormal() supports a single random-effect term on the mean")
        (rk, var) = _re_kind(re[1][1]); grp = re[1][2]
        gidx, G = _group_index(getproperty(data, grp))
        if rk === :intercept           # (1 | g) → 1-D Gauss–Hermite
            return _withformula(_fit_lognormal_ranef(fam, y, Xμ, Xσ, gidx, G, nmμ, nmσ, grp, g_tol), f)
        elseif rk === :corr            # (1 + x | g) → correlated 2-D Gauss–Hermite
            return _withformula(_fit_lognormal_corr_ranef(fam, y, Xμ, Xσ, Float64.(getproperty(data, var)), gidx, G, nmμ, nmσ, grp, g_tol), f)
        else
            error("LogNormal() supports `(1 | g)` or `(1 + x | g)` on the mean")
        end
    end
    return _withformula(_fit_lognormal(fam, y, Xμ, Xσ, nmμ, nmσ, g_tol), f)
end

# LogNormal GLMM with a random intercept (1|g) on the log-mean μ (the mean of
# log y). b_g ~ N(0,σ_b²) integrated out per group by 32-node Gauss–Hermite
# quadrature; σ = SD of log y (log link) is a fixed effect. Same scheme as the
# Gamma random intercept, with the per-observation contribution the Gaussian
# log-density of log y. (A closed-form Gaussian-LMM marginal on log y exists; GHQ
# is used here for consistency with the other non-Gaussian families.) The constant
# −Σ log y change-of-variables Jacobian is added to the reported loglik so it
# matches the fixed fitter's convention.
function _fit_lognormal_ranef(fam::LogNormal, y, Xμ, Xσ, gidx, G, nmμ, nmσ, grp, g_tol)
    n = length(y); pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    ly = log.(y); sumlogy = sum(ly)                  # Σ log y = the Jacobian offset
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    # The `(1|g)` marginal is linear-Gaussian in b_g (μ enters ly's mean by identity
    # link, no transform), so K = 1 (the mode-centred Laplace/AGHQ node) is exact for
    # this family — no larger rule can improve on it (#719).
    rule = _AGHQRule(1, 1); Zre = ones(n, 1); bcache = zeros(1, G)
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; σb = exp(θ[pμ+pσ+1])
        η0 = Xμ * βμ; ησ = clamp.(Xσ * βσ, -15.0, 15.0)
        ll = (i, η) -> Distributions.logpdf(Distributions.Normal(clamp(η, -30.0, 30.0), exp(ησ[i])), ly[i])
        L = reshape([σb], 1, 1)
        return -_aghq_marginal_loglik(ll, members, η0, Zre, L, rule, bcache) + sumlogy   # + Σ log y so loglik carries the Jacobian
    end
    βμ0 = Xμ \ ly
    θ0 = zeros(pμ + pσ + 1)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(std(ly - Xμ * βμ0) + eps()); θ0[pμ+pσ+1] = log(0.5)
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :resd => (pμ+pσ+1):(pμ+pσ+1)]
    names = [:mu => nmμ, :sigma => nmσ, :resd => [String(grp)]]
    means = Dict(:mu => exp.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# LogNormal GLMM with a CORRELATED random intercept+slope (1 + x | g) on the
# log-mean μ. Per group (b0,b1) ~ N(0, Σ_re), Σ_re = L Lᵀ with L = [l11 0; cc l22]
# (log-Cholesky parameters a, b, cc; l11=exp(a), l22=exp(b)). The 2-D group integral
# is taken by per-group ADAPTIVE Gauss–Hermite quadrature (`_aghq_marginal_loglik`,
# #834; log y is Gaussian in b, so this is exact for any nq). σ = SD of log y is a
# fixed effect. θ = [β_μ; β_σ; a, b, cc]. The constant −Σ log y
# Jacobian is added to the reported loglik to match the fixed fitter's convention.
function _fit_lognormal_corr_ranef(fam::LogNormal, y, Xμ, Xσ, xs, gidx, G, nmμ, nmσ, grp, g_tol; nq::Int = _CORR_RANEF_AGHQ_K)
    n = length(y); pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    ly = log.(y); sumlogy = sum(ly)                  # Σ log y = the Jacobian offset
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    rule = _AGHQRule(2, nq); Zre = hcat(ones(n), Float64.(xs)); bcache = zeros(2, G)   # #834: per-group AGHQ
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]
        L = _corr_ranef_L(θ[pμ+pσ+1], θ[pμ+pσ+2], θ[pμ+pσ+3])
        η0 = Xμ * βμ; ησ = clamp.(Xσ * βσ, -15.0, 15.0)
        ll = (i, η) -> Distributions.logpdf(Distributions.Normal(clamp(η, -30.0, 30.0), exp(ησ[i])), ly[i])
        return -_aghq_marginal_loglik(ll, members, η0, Zre, L, rule, bcache) + sumlogy   # + Σ log y: Jacobian
    end
    βμ0 = Xμ \ ly
    θ0 = zeros(pμ + pσ + 3)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(std(ly - Xμ * βμ0) + eps())
    θ0[pμ+pσ+1] = log(0.4); θ0[pμ+pσ+2] = log(0.4); θ0[pμ+pσ+3] = 0.0
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :recov => (pμ+pσ+1):(pμ+pσ+3)]
    names = [:mu => nmμ, :sigma => nmσ, :recov => ["$(grp):L11", "$(grp):L22", "$(grp):L21"]]
    means = Dict(:mu => exp.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

function _fit_lognormal(fam::LogNormal, y, Xμ, Xσ, nmμ, nmσ, g_tol)
    n = length(y); pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    ly = log.(y); sumlogy = sum(ly)                  # Σ log y = the Jacobian offset
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]
        ημ = Xμ * βμ; ησ = Xσ * βσ                   # ησ = log σ
        s = zero(eltype(θ))
        @inbounds for i in 1:n
            r = ly[i] - ημ[i]
            s += ησ[i] + 0.5 * r * r * exp(-2 * ησ[i])
        end
        return s + 0.5 * n * log(2π) + sumlogy        # Gaussian-on-log-y nll + Σ log y
    end
    βμ0 = Xμ \ ly
    θ0 = zeros(pμ + pσ)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(std(ly - Xμ * βμ0) + eps())
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ)]
    names = [:mu => nmμ, :sigma => nmσ]
    means = Dict(:mu => exp.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => Vector{Float64}(y))  # response-scale median
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# Crossed/multiple random intercepts `(1 | g) + (1 | h) + …` on the log-mean μ
# (#736). `log(y)` is exactly Gaussian, so — exactly like the phylo/relmat
# structured-marker delegation just below — this delegates the WHOLE fit to
# the public Gaussian dispatcher on a logged copy of the response
# (`_fit_multi_ranef_gaussian`, the closed-form exact crossed marginal
# likelihood; no Laplace approximation, so no new integrator), and shifts the
# reported log-likelihood by the parameter-free Jacobian `-sum(log y)`.
# Requires strictly positive responses (checked before logging).
function _fit_lognormal_crossed(fam::LogNormal, f::DrmFormula, data, g_tol)
    cols = NamedTuple(pairs(data))
    y = Vector{Float64}(getproperty(cols, f.response))
    all(yi -> yi > 0, y) || error("LogNormal() requires strictly positive responses")
    logdata = merge(cols, NamedTuple{(f.response,)}((log.(y),)))
    gfit = drm(f, Gaussian(); data = logdata, g_tol = g_tol)
    return _lognormal_jacobian_shift(fam, gfit, y)
end

# A `phylo(1 | group)` or `relmat(1 | group)` marker on the log-mean μ (#563
# S8). `log(y)` is exactly Gaussian, so this delegates the WHOLE fit to the
# public Gaussian dispatcher on a logged copy of the response — the same
# identity `src/bivariate_lognormal.jl` already uses for the bivariate family
# — rather than duplicating the phylo/relmat structured engine here. Requires
# strictly positive responses (checked before logging).
function _fit_lognormal_structured(fam::LogNormal, f::DrmFormula, data, kind::Symbol, grp::Symbol; tree, K, g_tol)
    if kind === :phylo
        tree === nothing && error("LogNormal(): phylo(1 | $grp) needs `tree = ...`")
    else # :relmat
        K === nothing && error("LogNormal(): relmat(1 | $grp) needs `K = ...`")
    end
    cols = NamedTuple(pairs(data))
    y = Vector{Float64}(getproperty(cols, f.response))
    all(yi -> yi > 0, y) || error("LogNormal() requires strictly positive responses")
    logdata = merge(cols, NamedTuple{(f.response,)}((log.(y),)))
    gfit = drm(f, Gaussian(); data = logdata, tree = tree, K = K, g_tol = g_tol)
    return _lognormal_jacobian_shift(fam, gfit, y)
end

# log f_Y(y) = log phi(log y; .) - sum(log y). Parameter-free, so theta-hat,
# vcov, and ranef carry over untouched from the Gaussian fit on log(y); only
# the likelihood VALUE (and everything derived from it: aic/bic/deviance)
# shifts by the Jacobian — mirrors `_lognormal_jacobian_shift` in
# src/bivariate_lognormal.jl for the univariate case.
function _lognormal_jacobian_shift(fam::LogNormal, gfit::DrmFit, y::Vector{Float64})
    jac = sum(log, y)
    gnll = gfit.nll
    lnll = gnll === nothing ? nothing : (θ -> gnll(θ) + jac)   # nll = -loglik, so the Jacobian ADDS
    reml_ll = isnan(gfit.reml_loglik) ? gfit.reml_loglik : gfit.reml_loglik - jac
    return DrmFit(fam, gfit.blocks, gfit.coefnames, gfit.theta, gfit.vcov,
                 gfit.loglik - jac, gfit.nobs, gfit.converged, gfit.means,
                 Dict(:mu => y), gfit.scales, gfit.formula, lnll, gfit.nllgrad, gfit.ranef,
                 gfit.estim_method, reml_ll, gfit.ml_loglik - jac, gfit.marginal,
                 gfit.phylo_penalty, gfit.penalty, gfit.iterations, gfit.phylo_scale)
end
