# tweedie.jl — Tweedie family (compound Poisson–Gamma, 1 < p < 2): semicontinuous
# responses with a point mass at 0 and a continuous positive part (biomass,
# rainfall, total insurance loss). Log link on the mean μ; `sigma` is the
# √dispersion (φ = σ²); `nu` is the power p on a logit-(1,2) link
# (p = 1 + logistic(η)). Var(y) = φ·μ^p. The density has no closed form — it is
# evaluated by the Dunn–Smyth compound Poisson–Gamma series. Mirrors drmTMB's
# `tweedie`. Fixed effects, ML.

using SpecialFunctions: loggamma

# log Σ_n W_n for the positive part: W_n = λⁿ y^{-nα} γ^{nα} / (n! Γ(-nα)), the
# compound Poisson–Gamma mixture (N ~ Poisson(λ); sum of N Gamma(-α, γ) jumps).
# The dominant n and the summation window are chosen in Float64 (detached, so the
# window doesn't break differentiability), then the window is summed in the
# parameter's number type so ForwardDiff differentiates through `loggamma(-nα)`.
function _tweedie_logW(y, λ, α, γ)
    z = log(λ) - α * log(y) + α * log(γ)
    zv = ForwardDiff.value(z); αv = ForwardDiff.value(α)
    logWn(nn) = nn * zv - loggamma(nn + 1.0) - loggamma(-nn * αv)
    nmax = 1; best = logWn(1.0)
    for nn in 2:400
        l = logWn(float(nn))
        l > best && (best = l; nmax = nn)
        (l < best - 45 && nn > nmax) && break
    end
    jl = max(1, nmax - 30); jh = nmax + 30
    terms = [j * z - loggamma(j + 1.0) - loggamma(-j * α) for j in jl:jh]
    m = maximum(terms)
    return m + log(sum(exp(t - m) for t in terms))
end

# Tweedie log-density at one point (1 < p < 2, y ≥ 0, μ,φ > 0).
function _logpdf_tweedie(y, μ, φ, p)
    λ = μ^(2 - p) / (φ * (2 - p))
    y == 0 && return -λ
    α = (2 - p) / (1 - p); γ = φ * (p - 1) * μ^(p - 1)
    return -λ - y / γ - log(y) + _tweedie_logW(y, λ, α, γ)
end

"""
    Tweedie()

Tweedie response family (compound Poisson–Gamma, power `1 < p < 2`) for
semicontinuous data — an exact-zero mass plus a positive continuous part. Log
link on the mean `μ`; `sigma` is the √dispersion (so `φ = σ²`, `coef(fit, :sigma)`
is `log σ`); `nu` is the power `p` on a logit-`(1,2)` link
(`p = 1 + logistic(coef(fit, :nu))`). `Var(y) = φ·μ^p`. Density via the Dunn–Smyth
series. Mirrors `drmTMB`'s `tweedie`. An ordinary `(1 | g)` random intercept
and an independent `(0 + x | g)` random slope on `mu` are both supported
(per-group adaptive Gauss–Hermite marginal); `sigma ~ 1` and `nu ~ 1` remain
fixed-effect sub-models on either route. Crossed/multiple random intercepts,
`(1 | g) + (1 | h)`, are also supported, via the sparse
augmented-state Laplace GLMM engine shared with Gamma/NB2/Beta's crossed
routes; `sigma ~ 1` and `nu ~ 1` are REQUIRED (not just default) on the
crossed route, and `p` is found by an outer 1-D profile search over the one
dispersion-like nuisance slot the shared engine does not also give to `p`.
The CORRELATED random slope `(1 + x | g)`, random effects on `sigma`/`nu`, and
structured (phylo/relmat/animal/spatial) markers on `mu` are not implemented
— matches `drmTMB` 0.7.0, which also restricts Tweedie mean random effects to
independent intercepts and slopes.

```julia
fit = drm(bf(y ~ x, sigma ~ 1, nu ~ 1), Tweedie(); data = dat)
exp(2 * coef(fit, :sigma)[1])               # dispersion φ
1 + 1 / (1 + exp(-coef(fit, :nu)[1]))       # power p ∈ (1,2)

fit_re = drm(bf(y ~ x + (1 | g), nu ~ 1), Tweedie(); data = dat)
exp(coef(fit_re, :resd)[1])                 # random-intercept SD

fit_slope = drm(bf(y ~ x + (0 + x | g), nu ~ 1), Tweedie(); data = dat)
exp(coef(fit_slope, :resd)[1])              # random-slope SD

fit_crossed = drm(bf(y ~ x + (1 | g) + (1 | h)), Tweedie(); data = dat)
re_sd(fit_crossed)                          # Dict(:g => ..., :h => ...)
```
"""
struct Tweedie end

_logit12(η) = 1 + 1 / (1 + exp(-η))            # (1,2) link for the power p

function drm(f::DrmFormula, fam::Tweedie; data, g_tol::Real = 1e-8)
    missing_fit = _fit_observed_response_rows(f, data) do data_observed
        drm(f, fam; data = data_observed, g_tol = g_tol)
    end
    missing_fit !== nothing && return missing_fit

    _lss_only_gaussian_guard(f, fam)   # #544: refuse, never silently drop, sd() parts
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    mv === nothing || error("Tweedie() does not support meta_V markers")
    for (pname, r) in f.forms          # only the mean may carry a random effect (#563)
        pname === :mu && continue
        _, re2, mv2, st2 = _split_ranef(r)
        (isempty(re2) && mv2 === nothing && st2 === nothing) ||
            error("Tweedie() random effects are not implemented for `sigma`/`nu`; " *
                  "only an ordinary `(1 | g)` random intercept on `mu` is supported " *
                  "(matches drmTMB, which rejects sigma/nu random effects for tweedie())")
    end
    y, Xμ, nmμ = _design(f.response, fixed_mu, data)
    _, Xσ, nmσ = _design(f.response, get(rhs, :sigma, ConstantTerm(1)), data)
    _, Xν, nmν = _design(f.response, get(rhs, :nu, ConstantTerm(1)), data)
    all(yi -> yi >= 0, y) || error("Tweedie() requires non-negative responses (zeros allowed)")

    st === nothing ||
        error("Tweedie() structured (phylo/relmat/animal/spatial) random effects on `mu` " *
              "are not implemented (matches drmTMB, which has no structured route for tweedie())")
    if !isempty(re)                    # random intercept/slope on mu → GHQ marginal (#563 S8)
        if length(re) > 1               # crossed/multiple intercepts (#737)
            all(_re_kind(r[1])[1] === :intercept for r in re) ||
                error("Tweedie() supports multiple random effects only as crossed/nested " *
                      "intercepts, e.g. `(1 | g) + (1 | h)`")
            size(Xσ, 2) == 1 ||
                error("Tweedie() crossed/multiple random intercepts currently require a " *
                      "constant `sigma` formula")
            size(Xν, 2) == 1 ||
                error("Tweedie() crossed/multiple random intercepts currently require a " *
                      "constant `nu` formula")
            comps = map(re) do r
                grp = r[2]; gidx, G = _group_index(getproperty(data, grp))
                (ones(length(y)), gidx, G, String(grp))
            end
            return _withformula(_fit_tweedie_crossed_laplace(fam, y, Xμ, comps, nmμ, nmσ, nmν, g_tol), f)
        end
        length(re) == 1 ||
            error("Tweedie() supports only a single `(1 | g)` random intercept or " *
                  "`(0 + x | g)` random slope on `mu`; crossed/multiple random effects " *
                  "are not implemented")
        (rk, var) = _re_kind(re[1][1]); grp = re[1][2]
        gidx, G = _group_index(getproperty(data, grp))
        if rk === :intercept
            return _withformula(
                _fit_tweedie_ranef(fam, y, Xμ, Xσ, Xν, gidx, G, nmμ, nmσ, nmν, grp, g_tol), f)
        elseif rk === :slope
            xs = Float64.(getproperty(data, var))
            return _withformula(
                _fit_tweedie_slope_ranef(fam, y, Xμ, Xσ, Xν, xs, gidx, G, nmμ, nmσ, nmν, grp, g_tol), f)
        else
            error("Tweedie() supports only an ordinary `(1 | g)` random intercept or an " *
                  "independent `(0 + x | g)` random slope on `mu`; correlated random " *
                  "slopes `(1 + x | g)` are not implemented (matches drmTMB 0.7.0: " *
                  "\"Only independent tweedie() mu random intercepts and slopes are " *
                  "implemented in this slice\")")
        end
    end
    return _withformula(_fit_tweedie(fam, y, Xμ, Xσ, Xν, nmμ, nmσ, nmν, g_tol), f)
end

function _fit_tweedie(fam::Tweedie, y, Xμ, Xσ, Xν, nmμ, nmσ, nmν, g_tol)
    n = length(y); pμ, pσ, pν = size(Xμ, 2), size(Xσ, 2), size(Xν, 2)
    i1 = pμ + pσ; i2 = i1 + pν
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:i1]; βν = θ[i1+1:i2]
        ημ = clamp.(Xμ * βμ, -30.0, 30.0); ησ = clamp.(Xσ * βσ, -15.0, 15.0)
        ην = clamp.(Xν * βν, -12.0, 12.0)        # keep p strictly inside (1,2) — avoids the boundary singularities
        s = zero(eltype(θ))
        @inbounds for i in 1:n
            μ = exp(ημ[i]); φ = exp(2 * ησ[i]); p = _logit12(ην[i])   # φ = σ²
            s -= _logpdf_tweedie(y[i], μ, φ, p)
        end
        return s
    end
    ȳ = sum(y) / n
    θ0 = zeros(i2)
    θ0[1] = log(ȳ + eps())            # log mean
    θ0[pμ+1] = 0.0                     # σ = 1 (φ = 1)
    θ0[i1+1] = 0.0                     # p = 1.5
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):i1, :nu => (i1+1):i2]
    names = [:mu => nmμ, :sigma => nmσ, :nu => nmν]
    means = Dict(:mu => exp.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => Vector{Float64}(y))   # response-scale μ̂
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):i1]),
                  :nu => _logit12.(Xν * θ̂[(i1+1):i2]))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, drm_optim_converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# Default nodes for Tweedie's two 1-D `(1|g)`/`(0+x|g)` mu-ranef routes.
# Adversarial review of the AGHQ switch (following #877's precedent of a
# family-specific K constant, `_BINOMIAL_RANEF_AGHQ_K`) found the generic
# `_RANEF1D_AGHQ_K = 5` default still 0.19 nat off on a zero-heavy,
# informative-group regime: swept K ∈ {5, 9, 15, 21} on the review's
# zero-heavy DGP (G=60, m=5, σ_b≈3.2, β0=−1.5, 203/300 zeros) and on an
# ordinary DGP (G=40, m=15, σ_b=3), logLik error at θ̂ vs an independent
# QuadGK reference: K=5 −0.187/−1.9e-4 nat, K=9 0.039/−2.5e-4, K=15
# −0.0037/1.1e-5, K=21 6.7e-4/1.3e-7. K=21 is the smallest of the four that
# lands within 1e-3 nat on BOTH; cost is 2.64× the pre-AGHQ 32-node grid on
# the ordinary DGP and 1.99× on the zero-heavy one (vs 2.2×/1.7× at K=5) —
# a reasonable increment for a route where fits already run in ~1-5 s. See
# test/test_tweedie_aghq_k.jl for the sweep.
const _TWEEDIE_RANEF_AGHQ_K = 21

# Tweedie compound Poisson–Gamma GLMM with a random intercept (1|g) on the log
# mean. b_g ~ N(0,σ_b²) integrated out per group by per-group adaptive
# Gauss–Hermite quadrature (`src/adaptive_ghq.jl`, #719/#834) — the same scheme
# as the Poisson/Gamma `(1 | g)` routes (src/poisson.jl `_fit_poisson_ranef`,
# src/gamma.jl `_fit_gamma_ranef`). `sigma`/`nu` stay ordinary fixed-effect
# sub-models (dispersion φ and power p are not group-varying); only `mu`'s
# intercept carries the random term. #563.
function _fit_tweedie_ranef(fam::Tweedie, y, Xμ, Xσ, Xν, gidx, G, nmμ, nmσ, nmν, grp, g_tol; K::Int = _TWEEDIE_RANEF_AGHQ_K)
    n = length(y); pμ, pσ, pν = size(Xμ, 2), size(Xσ, 2), size(Xν, 2)
    i1 = pμ + pσ; i2 = i1 + pν
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    rule = _AGHQRule(1, K); Zre = ones(n, 1); bcache = zeros(1, G)   # #719: per-group AGHQ
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:i1]; βν = θ[i1+1:i2]; σb = exp(θ[i2+1])
        η0 = clamp.(Xμ * βμ, -30.0, 30.0)
        ησ = clamp.(Xσ * βσ, -15.0, 15.0)
        ην = clamp.(Xν * βν, -12.0, 12.0)
        ll = (i, η) -> (μ = exp(clamp(η, -30.0, 30.0)); φ = exp(2 * ησ[i]); p = _logit12(ην[i]);
                        _logpdf_tweedie(y[i], μ, φ, p))
        L = reshape([σb], 1, 1)
        return -_aghq_marginal_loglik(ll, members, η0, Zre, L, rule, bcache)
    end
    ȳ = sum(y) / n
    θ0 = zeros(i2 + 1)
    θ0[1] = log(ȳ + eps())            # log mean
    θ0[pμ+1] = 0.0                     # σ = 1 (φ = 1)
    θ0[i1+1] = 0.0                     # p = 1.5
    θ0[i2+1] = log(0.5)                # σ_b
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):i1, :nu => (i1+1):i2, :resd => (i2+1):(i2+1)]
    names = [:mu => nmμ, :sigma => nmσ, :nu => nmν, :resd => [String(grp)]]
    means = Dict(:mu => exp.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):i1]),
                  :nu => _logit12.(Xν * θ̂[(i1+1):i2]))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, drm_optim_converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# Tweedie compound Poisson–Gamma GLMM with an INDEPENDENT random slope
# (0 + x | g) on the log mean: η_i = Xμβ_μ + b_g·x_i, b_g ~ N(0, σ_b²),
# integrated out per group by per-group adaptive Gauss–Hermite quadrature
# (`src/adaptive_ghq.jl`, #719/#834). Same scheme as `_fit_tweedie_ranef` (the
# ordinary random intercept), with the random-effect design column set to
# x_i (Zre) instead of the flat 1 used by the intercept route. Matches
# drmTMB's independent-slope route for tweedie() mu (`(0 + x | id)`, distinct
# from the still-unimplemented correlated `(1 + x | id)`). #563 S8.
function _fit_tweedie_slope_ranef(fam::Tweedie, y, Xμ, Xσ, Xν, xs, gidx, G, nmμ, nmσ, nmν, grp, g_tol; K::Int = _TWEEDIE_RANEF_AGHQ_K)
    n = length(y); pμ, pσ, pν = size(Xμ, 2), size(Xσ, 2), size(Xν, 2)
    i1 = pμ + pσ; i2 = i1 + pν
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    rule = _AGHQRule(1, K); Zre = reshape(xs, n, 1); bcache = zeros(1, G)   # #719: per-group AGHQ
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:i1]; βν = θ[i1+1:i2]; σb = exp(θ[i2+1])
        η0 = clamp.(Xμ * βμ, -30.0, 30.0)
        ησ = clamp.(Xσ * βσ, -15.0, 15.0)
        ην = clamp.(Xν * βν, -12.0, 12.0)
        ll = (i, η) -> (μ = exp(clamp(η, -30.0, 30.0)); φ = exp(2 * ησ[i]); p = _logit12(ην[i]);
                        _logpdf_tweedie(y[i], μ, φ, p))
        L = reshape([σb], 1, 1)
        return -_aghq_marginal_loglik(ll, members, η0, Zre, L, rule, bcache)
    end
    ȳ = sum(y) / n
    θ0 = zeros(i2 + 1)
    θ0[1] = log(ȳ + eps())            # log mean
    θ0[pμ+1] = 0.0                     # σ = 1 (φ = 1)
    θ0[i1+1] = 0.0                     # p = 1.5
    θ0[i2+1] = log(0.5)                # σ_b
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):i1, :nu => (i1+1):i2, :resd => (i2+1):(i2+1)]
    names = [:mu => nmμ, :sigma => nmσ, :nu => nmν, :resd => [String(grp)]]
    means = Dict(:mu => exp.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):i1]),
                  :nu => _logit12.(Xν * θ̂[(i1+1):i2]))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, drm_optim_converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# ---------------------------------------------------------------------------
# Crossed/multiple random intercepts (1|g)+(1|h) on the log mean (#737),
# reusing the sparse augmented-state Laplace GLMM engine already shared by
# Gamma/NB2/Beta's crossed routes (src/sparse_laplace_glmm.jl:
# `_fit_crossed_mean_laplace_nuisance`, `_crossed_mean_mode`) — NOT a new
# integrator. `sigma ~ 1` and `nu ~ 1` stay constant fixed-effect sub-models
# (mirrors Gamma's constant-`sigma` restriction on its crossed route).
#
# The shared engine's "nuisance" slot is exactly one scalar dispersion
# parameter (`_fit_crossed_mean_laplace_nuisance`'s `θσ`). The Tweedie power
# `p` is a SECOND nuisance scalar the shared engine has no room for (its θ
# layout is hardcoded to `[βμ; θσ; logσ_g; logσ_h]`), and widening that layout
# is a shared-file change out of this family file's ownership. Instead `p` is
# PROFILED OUT: for each candidate `p`, `inner_fit` refits (β, dispersion,
# σ_g, σ_h) to convergence with the existing engine, folding `p` into the
# dispersion-only `aux_from` closure; an outer 1-D Brent search over the
# logit-(1,2) parameterisation finds the `p` minimising that profile
# deviance. This is exact profile-likelihood optimisation (the profile MLE
# equals the joint MLE at the joint optimum) — no approximation is added
# beyond the crossed-Laplace machinery already in use.
#
# `_logpdf_tweedie`'s Dunn–Smyth series has no convenient closed-form
# derivative, so the per-observation value/derivatives the shared engine
# needs (`_laplace_value`/`_d1`/`_d2`/`_d3`, and the nuisance-parameter
# counterparts) are obtained by nested `ForwardDiff.derivative` on `η` (and on
# `logsigma` for the nuisance slot) rather than hand-derived analytically.
# Verified against central finite differences (interactive check, not part of
# the test suite): a single-observation d1/d2/d3 and nuisance-value/d1/d2
# probe matched 5-point-stencil finite differences to ~1e-4–1e-6 relative,
# including at `y = 0` (the exact-zero branch of `_logpdf_tweedie`, which is
# smooth in `η` for a FIXED `y`, so nested autodiff carries through it fine).

# Per-observation NLL contribution as an explicit function of BOTH the linear
# predictor η and the nuisance log-scale `logsigma` (dispersion φ = exp(2·logsigma)),
# so `_laplace_d3` (pure η) and the mixed/pure `logsigma` nuisance derivatives
# below can all be built by nesting `ForwardDiff.derivative` on this one
# closure, at whichever variable is being differentiated.
@inline function _tweedie_v_of(y::Real, η, logsigma, p::Real)
    μ = exp(clamp(η, -30.0, 30.0))
    φ = exp(2 * clamp(logsigma, -8.0, 8.0))
    return -_logpdf_tweedie(y, μ, φ, p)
end

_laplace_value(::Val{:tweedie_fixed}, aux, i, η) =
    _tweedie_v_of(aux.y[i], η, aux.logsigma, aux.p)

_laplace_d1(::Val{:tweedie_fixed}, aux, i, η) =
    ForwardDiff.derivative(x -> _tweedie_v_of(aux.y[i], x, aux.logsigma, aux.p), η)

_laplace_d2(::Val{:tweedie_fixed}, aux, i, η) =
    ForwardDiff.derivative(x -> _laplace_d1(Val(:tweedie_fixed), aux, i, x), η)

_laplace_d3(::Val{:tweedie_fixed}, aux, i, η) =
    ForwardDiff.derivative(x -> _laplace_d2(Val(:tweedie_fixed), aux, i, x), η)

_laplace_mean(::Val{:tweedie_fixed}, η) = exp(clamp(η, -30.0, 30.0))
_laplace_obs(::Val{:tweedie_fixed}, aux, i) = aux.y[i]

# d/d(logsigma) of the per-observation NLL contribution, at fixed η.
_laplace_nuisance_value(::Val{:tweedie_fixed}, aux, i, η) =
    ForwardDiff.derivative(ls -> _tweedie_v_of(aux.y[i], η, ls, aux.p), aux.logsigma)

# mixed partial d²/(dη d(logsigma)) — matches `_laplace_nuisance_d1` for
# `:gamma_fixed`/`:nb2_fixed`/`:beta_fixed` (verified against gamma_fixed's
# hand-derived `-2α·(1 - y/μ)` reading, which is `∂(nuisance_value)/∂η`).
_laplace_nuisance_d1(::Val{:tweedie_fixed}, aux, i, η) =
    ForwardDiff.derivative(x -> _laplace_nuisance_value(Val(:tweedie_fixed), aux, i, x), η)

# `_laplace_nuisance_d2` is NOT the pure d²v/d(logsigma)² — it is `∂(d2)/∂(logsigma)`,
# i.e. d³v/(dη² d(logsigma)): how the Hessian diagonal entry `w = d2` itself
# shifts with the nuisance parameter at FIXED η. This is what the crossed
# engine's `_crossed_mean_laplace_nuisance_fg` needs for the log-determinant's
# direct (not mode-implicit) dependence on the nuisance parameter — confirmed
# against `:gamma_fixed`'s hand-derived reading, `_laplace_nuisance_d2 =
# -2α·y/μ = -2·_laplace_d2` (since `α = exp(-2·logsigma)`, so
# `∂(αy/μ)/∂(logsigma) = -2α·y/μ`), i.e. `∂(d2)/∂(logsigma)`, not `∂²v/∂s²`.
# Getting this backwards (as a first pass here did) leaves every OTHER
# gradient component (β, RE-variance SDs) exactly right — because those flow
# entirely through η — while only the dispersion/nuisance component of the
# analytic gradient is wrong; caught by an off-optimum finite-difference gate
# on that one component (test/test_crossed_tweedie.jl).
_laplace_nuisance_d2(::Val{:tweedie_fixed}, aux, i, η) =
    ForwardDiff.derivative(ls -> _laplace_d2(Val(:tweedie_fixed), (y = aux.y, p = aux.p, logsigma = ls), i, η),
                           aux.logsigma)

function _fit_tweedie_crossed_laplace(fam::Tweedie, y, Xμ, comps, nmμ, nmσ, nmν, g_tol)
    length(comps) == 2 || error("_fit_tweedie_crossed_laplace requires two random-intercept components")
    yv = Float64.(y)
    n = length(yv)
    ȳ = sum(yv) / n
    θβ0 = zeros(size(Xμ, 2))
    θβ0[1] = log(ȳ + eps())

    function inner_fit(p::Real; polish_iterations::Int = 0)
        aux_from(logsigma) = (y = yv, p = p, logsigma = logsigma)
        return _fit_crossed_mean_laplace_nuisance(
            fam, Val(:tweedie_fixed), aux_from, n, Xμ,
            comps[1][2], comps[1][3], comps[2][2], comps[2][3], nmμ, nmσ,
            [comps[1][4], comps[2][4]], g_tol;
            θβ0 = θβ0, θσ0 = 0.0, sigma_scale = exp, se = false,
            polish_iterations = polish_iterations)
    end

    obj(ηp) = -inner_fit(_logit12(ηp)).loglik
    res = Optim.optimize(obj, -6.0, 6.0, Optim.Brent())
    p̂ = _logit12(Optim.minimizer(res))
    fit = inner_fit(p̂; polish_iterations = 25)

    pμ = size(Xμ, 2)
    nblk = pμ + 1 + 2   # mu + sigma(dispersion) + 2 crossed RE sds, the engine's own blocks
    blocks = vcat(fit.blocks, [:nu => (nblk + 1):(nblk + 1)])
    names = vcat(fit.coefnames, [:nu => nmν])
    theta = vcat(fit.theta, [Optim.minimizer(res)])
    V = fill(NaN, length(theta), length(theta))
    V[1:nblk, 1:nblk] .= fit.vcov
    scales = merge(fit.scales, Dict(:nu => fill(p̂, n)))
    return DrmFit(fam, blocks, names, theta, V, fit.loglik, fit.nobs, fit.converged,
                 fit.means, fit.obs, scales)
end
