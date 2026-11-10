# skewnormal.jl — Skew-normal family: location–scale–shape regression with an
# asymmetric (skewed) Gaussian. A formula per parameter — μ (location, identity),
# σ (scale, log link, the `sigma` slot), ν (slant / shape, identity, the `nu`
# slot). The density is Azzalini's skew-normal in internal (ξ, ω, α) form,
#   f(y) = 2·φ((y−ξ)/ω)·Φ(α(y−ξ)/ω) / ω,
# but the PUBLIC parameterisation is the moment form (mirroring drmTMB's
# `skew_normal`): μ is the MEAN of y, σ is the SD of y, and ν is the slant α.
# The internal location/scale are recovered from the moments via
#   δ = α/√(1+α²),  ω = σ/√(1 − 2δ²/π),  ξ = μ − ω·δ·√(2/π).
# As ν → 0 the family collapses to the symmetric Gaussian (and the slant becomes
# weakly identified — see the docstring note). Fixed effects, maximum likelihood.
# `Distributions.Normal` is used qualified for the stable log-pdf / log-cdf of φ/Φ.

import Distributions

"""
    SkewNormal()

Skew-normal response family for continuous, asymmetric data: identity link on
the location `μ`, log link on the scale `σ` (the `sigma` slot), and identity link
on the slant `ν` (the `nu` slot). The public parameterisation is the **moment**
form mirroring `drmTMB`'s `skew_normal`: `μ` is the **mean** of `y`, `σ` is the
**SD** of `y`, and `ν` is Azzalini's slant `α`. These are mapped internally to
the location–scale–slant `(ξ, ω, α)` of the density
`f(y) = 2·φ((y−ξ)/ω)·Φ(α(y−ξ)/ω)/ω`.

`ν = 0` recovers the symmetric [`Gaussian`](@ref); `ν > 0` skews right, `ν < 0`
skews left.

```julia
fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)), SkewNormal(); data = dat)
coef(fit, :mu)[1]           # mean of y (identity)
exp(coef(fit, :sigma)[1])   # SD of y
coef(fit, :nu)[1]           # estimated slant α (identity)
predict_parameters(fit, dat)[:nu]   # same slant (identity; not exp(η))

# Predictor-dependent SD for an ordinary random intercept
fit_sd = drm(bf(@formula(y ~ x + (1 | site)), @formula(sigma ~ 1),
                @formula(nu ~ 1), @formula(sd(site) ~ habitat)),
             SkewNormal(); data = dat)
predict_parameters(fit_sd, dat)[:sd]  # fitted random-intercept SD by site

# Predictor-dependent SD for one independent mean slope
fit_slope_sd = drm(bf(@formula(y ~ x + (0 + x | site)), @formula(sigma ~ 1),
                      @formula(nu ~ 1), @formula(sd(site, coef(x)) ~ habitat)),
                   SkewNormal(); data = dat, nAGQ = 9)
```

!!! note
    Near `ν = 0` the model is (locally) symmetric Gaussian and the slant is only
    weakly identified — its standard error inflates and recovery is loose. The
    fitter starts `ν` at a small nonzero value seeded by the sample-skewness sign
    to break this symmetry. The information matrix is singular at `ν = 0`, so the
    Wald SE for `nu` is unreliable there. [`check_drm`](@ref) reports
    `skew_normal_nu_wald` and warns when any fitted |ν̂| < 0.5. This cutoff is an
    uncalibrated heuristic of this package; no Wald test is run (#1060).
    Predictor-dependent SD supports one ordinary mean
    intercept or independent numeric slope and uses one-dimensional adaptive
    Gauss–Hermite quadrature. It requires fixed `sigma ~ 1`, `nu ~ 1`, complete
    responses, and a full-rank group-level SD design. Correlated and structured
    random effects are not supported on this route; interval coverage calibration
    is deferred.
"""
struct SkewNormal end

@drm_entry function drm(f::DrmFormula, fam::SkewNormal; data, g_tol::Real = 1e-8,
             nAGQ::Union{Nothing,Int} = nothing,
             marginal::Union{Nothing,Symbol} = nothing)
    _drm_entry_guard(f, fam, data)   # #1027 #1022
    # `marginal` exists only so that `update` / `bootstrap` can repeat a fit of the one-`(1 | g)`
    # Laplace route (its fit is tagged `:Laplace`); it is refused everywhere else, never ignored.
    marginal === nothing || marginal === :Laplace || throw(ArgumentError(
        "SkewNormal() accepts `marginal = :Laplace` only (its random-intercept route); " *
        "got `marginal = :$marginal`."))
    if !isempty(_sd_parts(f)) || !isempty(_sdphylo_parts(f))
        marginal === nothing ||
            _refuse_unsupported_kwargs((:marginal,), f, fam, (:control, :data, :g_tol, :nAGQ, :weights))
        nodes = nAGQ === nothing ? _RANEF1D_AGHQ_K : nAGQ
        nodes >= 1 || throw(ArgumentError("nAGQ must be ≥ 1; got $nodes"))
        return _drm_skewnormal_sd_regression(f, fam; data, g_tol, nAGQ = nodes)
    end
    nAGQ === nothing || throw(ArgumentError(
        "nAGQ is currently supported for SkewNormal random-effect SD regression only."))
    # #1123: drop missing/NaN-response rows (warned), like every other univariate fitter.
    missing_fit = _fit_observed_response_rows(f, data) do data_observed
        drm(f, fam; data = data_observed, g_tol = g_tol, marginal = marginal)
    end
    missing_fit !== nothing && return missing_fit

    _lss_only_gaussian_guard(f, fam)   # #544: refuse, never silently drop, sd() parts
    rhs = Dict(f.forms)
    _, re, mv, st = _split_ranef(rhs[:mu])
    (mv === nothing && st === nothing) ||
        error("SkewNormal() does not support meta_V / structured markers")
    # One ordinary `(1 | g)` on the mean: drmTMB's Laplace objective (`zi_re_laplace.jl`); every
    # other random-effect shape stays refused (the route re-checks and names it).
    isempty(re) || return _drm_zi_re_laplace(f, fam; data = data, g_tol = g_tol)
    marginal === nothing ||
        _refuse_unsupported_kwargs((:marginal,), f, fam, (:control, :data, :g_tol, :nAGQ, :weights))
    for (pname, r) in f.forms          # no parameter may carry a random effect
        pname === :mu && continue
        _, re2, mv2, st2 = _split_ranef(r)
        (isempty(re2) && mv2 === nothing && st2 === nothing) ||
            error("SkewNormal() supports fixed effects only")
    end
    # #1089: drop rows with a missing predictor or response (R's na.omit), with a count.
    data = _drop_missing_response_rows(f, _drop_missing_predictor_rows(f, data))
    mu_fixed, offset_term = _extract_offset(rhs[:mu])   # #938: offset(z), coefficient fixed at 1
    y, Xμ, nmμ = _design(f.response, mu_fixed, data)
    _, Xσ, nmσ = _design(f.response, get(rhs, :sigma, ConstantTerm(1)), data)
    _, Xν, nmν = _design(f.response, get(rhs, :nu, ConstantTerm(1)), data)
    return _withformula(_fit_skewnormal(fam, y, Xμ, Xσ, Xν, nmμ, nmσ, nmν, g_tol; offset = _offset_for_fit(fam, offset_term, data, size(Xμ, 1))), f)
end

# One ordinary mean intercept or independent mean slope with a
# predictor-dependent coefficient SD. The family uses moment parameters, so the
# latent coefficient shifts the conditional mean while leaving the response SD
# and slant unchanged.
function _drm_skewnormal_sd_regression(f::DrmFormula, fam::SkewNormal;
                                       data, g_tol::Real = 1e-8,
                                       nAGQ::Int = _RANEF1D_AGHQ_K)
    sdp = _sd_target_parts(f)
    length(sdp) == 1 || throw(ArgumentError(
        "SkewNormal random-effect SD regression requires exactly one `sd(group) ~ …` formula."))
    target = only(sdp)
    grp, sd_rhs = target.target.group, target.rhs
    (target.target.level === :iid && target.target.dpar === :mu &&
     target.target.block === nothing) || throw(ArgumentError(
        "SkewNormal SD regression currently targets one ordinary mean intercept or independent slope."))
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    mv === nothing && st === nothing && length(re) == 1 || throw(ArgumentError(
        "SkewNormal `sd($grp)` currently requires one ordinary intercept or independent numeric slope."))
    kind, re_term = _re_kind(re[1][1])
    member = kind === :intercept ? :intercept : kind === :slope ? re_term : nothing
    kind in (:intercept, :slope) || throw(ArgumentError(
        "SkewNormal SD regression supports `(1 | group)` or one independent numeric `(0 + x | group)` term."))
    target.target.member in (nothing, member) || throw(ArgumentError(
        "SkewNormal SD target `coef($(target.target.member))` does not match the fitted random-effect coefficient `$(member)`."))
    re[1][2] === grp || throw(ArgumentError(
        "SkewNormal `sd($grp)` must name the random-effect grouping variable `$(re[1][2])`."))
    for p in (:sigma, :nu)
        r = get(rhs, p, ConstantTerm(1))
        r isa ConstantTerm && r == ConstantTerm(1) || throw(ArgumentError(
            "SkewNormal `sd($grp)` currently requires fixed `$(p) ~ 1` submodels."))
    end
    for (pname, _) in f.forms
        (pname in (:mu, :sigma, :nu, Symbol("sd_", grp)) ||
         haskey(f.sd_targets, pname)) && continue
        throw(ArgumentError("SkewNormal `sd($grp)` supports only `mu`, fixed `sigma ~ 1`, " *
                            "fixed `nu ~ 1`, and one `sd($grp)` formula."))
    end
    fixed_mu, offset_term = _extract_offset(fixed_mu)
    offset_term === nothing || throw(ArgumentError(
        "SkewNormal random-effect SD regression does not yet support `offset(...)`."))
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "SkewNormal random-effect SD regression currently requires complete responses."))
    y, Xmu, nmmu = _design(f.response, fixed_mu, data)
    _, Xsigma, nmsigma = _design(f.response, rhs[:sigma], data)
    _, Xnu, nmnu = _design(f.response, rhs[:nu], data)
    gidx, G = _group_index(getproperty(data, grp))
    re_design = if kind === :intercept
        ones(Float64, length(gidx))
    else
        hasproperty(data, re_term) || throw(ArgumentError(
            "SkewNormal random-slope predictor `$re_term` is absent from the data."))
        raw = getproperty(data, re_term)
        raw isa AbstractVector{<:Real} && length(raw) == length(gidx) &&
            all(isfinite, raw) || throw(ArgumentError(
                "SkewNormal random-slope predictor `$re_term` must be numeric, finite, and have one value per response."))
        any(value -> !iszero(value), raw) || throw(ArgumentError(
            "SkewNormal random-slope predictor `$re_term` must contain at least one nonzero value."))
        Float64.(raw)
    end
    Zg, nmsd = _sd_group_design(f.response, sd_rhs, data, gidx, G, grp;
                                schema_cache = f.schema_cache, schema_key = target.key)
    size(Zg, 2) <= G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
        "SkewNormal `sd($grp)` group-level design is rank deficient."))
    return _withformula(_fit_skewnormal_sd_regression(fam, y, Xmu, Xsigma, Xnu,
        re_design, Zg, gidx, G, nmmu, nmsigma, nmnu, nmsd, grp, g_tol; nAGQ), f)
end

function _skewnormal_logpdf_moment(y, mu, log_sigma, alpha)
    sigma = exp(log_sigma)
    delta = alpha / sqrt(1 + alpha^2)
    omega = sigma / sqrt(1 - 2 * delta^2 / π)
    xi = mu - omega * delta * sqrt(2 / π)
    z = (y - xi) / omega
    return log(2.0) + Distributions.logpdf(Distributions.Normal(), z) +
           Distributions.logcdf(Distributions.Normal(), alpha * z) - log(omega)
end

function _fit_skewnormal_sd_regression(fam::SkewNormal, y, Xmu, Xsigma, Xnu,
        re_design, Zg, gidx, G, nmmu, nmsigma, nmnu, nmsd, grp, g_tol; nAGQ::Int)
    n = length(y)
    pmu, psigma, pnu, psd = size(Xmu, 2), size(Xsigma, 2), size(Xnu, 2), size(Zg, 2)
    members = [Int[] for _ in 1:G]
    for i in eachindex(gidx)
        push!(members[gidx[i]], i)
    end
    rule = _AGHQRule(1, nAGQ)
    # Intercepts use one; an independent slope uses its numeric x_i.
    Zre = reshape(Float64.(re_design), n, 1)
    bcache = [zeros(1, G) for _ in 1:Threads.maxthreadid()]
    function nll(theta)
        i_sigma = pmu + psigma
        i_nu = i_sigma + pnu
        beta_mu = @view theta[1:pmu]
        beta_sigma = @view theta[(pmu+1):i_sigma]
        beta_nu = @view theta[(i_sigma+1):i_nu]
        alpha_sd = @view theta[(i_nu+1):(i_nu+psd)]
        eta_mu = Xmu * beta_mu
        eta_sigma = Xsigma * beta_sigma
        eta_nu = Xnu * beta_nu
        sd_group = exp.(Zg * alpha_sd)
        cache = bcache[Threads.threadid()]
        ll = (i, eta) -> _skewnormal_logpdf_moment(y[i], eta, eta_sigma[i], eta_nu[i])
        total = zero(eltype(theta))
        for gi in 1:G
            Lg = reshape([sd_group[gi]], 1, 1)
            value, bhat = _aghq_group_logint(ll, members[gi], eta_mu, Zre,
                Lg, rule, @view(cache[:, gi]))
            cache[1, gi] = isfinite(bhat[1]) ? bhat[1] : 0.0
            total += value
        end
        return -total
    end
    beta_mu0 = Xmu \ y
    resid = y - Xmu * beta_mu0
    sd0 = max(std(resid) * 0.8, 1e-3)
    gamma1 = sum(((resid ./ (std(resid) + eps())) .^ 3)) / n
    alpha0 = clamp(2.0 * sign(gamma1 == 0 ? 1.0 : gamma1), -4.0, 4.0)
    theta0 = zeros(pmu + psigma + pnu + psd)
    theta0[1:pmu] .= beta_mu0
    theta0[pmu+1] = log(sd0)
    theta0[pmu+psigma+1] = alpha0
    theta0[(pmu+psigma+pnu+1):end] .= Zg \ fill(log(0.4), G)
    res = Optim.optimize(nll, theta0, Optim.LBFGS(),
        Optim.Options(g_tol = g_tol); autodiff = :forward)
    theta_hat = Optim.minimizer(res)
    V = _vcov_from_hessian(ForwardDiff.hessian(nll, theta_hat))
    i_sigma = pmu + psigma
    i_nu = i_sigma + pnu
    i_sd = i_nu + psd
    sd_group = exp.(Zg * theta_hat[(i_nu+1):i_sd])
    blocks = [:mu => 1:pmu, :sigma => (pmu+1):i_sigma,
              :nu => (i_sigma+1):i_nu, :sd => (i_nu+1):i_sd]
    names = [:mu => nmmu, :sigma => nmsigma, :nu => nmnu, :sd => nmsd]
    means = Dict(:mu => Xmu * theta_hat[1:pmu])
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xsigma * theta_hat[(pmu+1):i_sigma]),
                  :nu => Xnu * theta_hat[(i_sigma+1):i_nu],
                  :sd_group => sd_group, :sd_group_index => Float64.(gidx),
                  :aghq_nodes => Float64[nAGQ])
    fit = DrmFit(fam, blocks, names, theta_hat, V, -nll(theta_hat), n,
        drm_optim_converged(res), means, obs, scales)
    return _withiterations(_withranef(_withnll(fit, nll), Dict(grp => bcache[1][1, :])),
                           Optim.iterations(res))
end

# Skew-normal location–scale–shape fit by maximum likelihood. θ = [βμ; βσ; βν]:
# μ = Xμ βμ (mean, identity), log σ = Xσ βσ (SD, log link), ν = Xν βν (slant α,
# identity). Per observation the public (μ, σ, α) are mapped to internal (ξ, ω, α)
# and scored against Azzalini's log-density log 2 + logφ((y−ξ)/ω) + logΦ(α(y−ξ)/ω)
# − log ω. Stable in the tail via Distributions' logpdf/logcdf on Normal().
# O(n) per eval, fully ForwardDiff-differentiable. Mirrors the Student/LogNormal
# fixed fitters.
function _fit_skewnormal(fam::SkewNormal, y, Xμ, Xσ, Xν, nmμ, nmσ, nmν, g_tol; offset = nothing)
    n = length(y)
    pμ, pσ, pν = size(Xμ, 2), size(Xσ, 2), size(Xν, 2)
    rt2_π = sqrt(2.0 / π)
    w = _ctl_weights(n)                  # prior weights: row log-lik multipliers (#931)
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; βν = θ[pμ+pσ+1:pμ+pσ+pν]
        ημ = _eta_with_offset(Xμ, βμ, offset); ησ = Xσ * βσ; ην = Xν * βν      # ημ = mean, ησ = log σ, ην = slant α
        s = zero(eltype(θ))
        @inbounds for i in 1:n
            μ = ημ[i]; σ = exp(ησ[i]); α = ην[i]
            δ = α / sqrt(1 + α^2)
            ω = σ / sqrt(1 - 2 * δ^2 / π)              # δ²<1 ⇒ 1−2δ²/π ∈ (1−2/π, 1] > 0
            ξ = μ - ω * δ * rt2_π
            z = (y[i] - ξ) / ω
            # log f = log2 + logφ(z) + logΦ(α z) − log ω
            s -= w[i] * (log(2.0) + Distributions.logpdf(Distributions.Normal(), z) +
                         Distributions.logcdf(Distributions.Normal(), α * z) - log(ω))
        end
        return s
    end
    yo = _add_offset(y, offset === nothing ? nothing : -offset)   # #938: start from y - offset
    βμ0 = Xμ \ yo
    resid = yo - Xμ * βμ0
    sd0 = std(resid) + eps()
    # Seed the slant from the sample-skewness sign so the α=0 symmetry is broken.
    γ1 = sum(((resid ./ sd0) .^ 3)) / n               # moment skewness estimate
    α0 = clamp(2.0 * sign(γ1 == 0 ? 1.0 : γ1), -4.0, 4.0)   # small, signed, nonzero
    θ0 = zeros(pμ + pσ + pν)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(sd0)                                # σ init (SD scale)
    θ0[pμ+pσ+1] = α0                                   # slant init (identity)
    res = _ctl_optimize((x0, opts, method) -> Optim.optimize(nll, x0, method, opts; autodiff = :forward),
                       θ0; g_tol = g_tol)   # control: maxit / start / multi_start (#948 #994)
    θ̂ = Optim.minimizer(res); V = _ctl_se(() -> _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂)), θ̂)
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :nu => (pμ+pσ+1):(pμ+pσ+pν)]
    names = [:mu => nmμ, :sigma => nmσ, :nu => nmν]
    means = Dict(:mu => _eta_with_offset(Xμ, θ̂[1:pμ], offset)); obs = Dict(:mu => Vector{Float64}(y))   # μ = response-scale mean
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]),
                  :nu => Xν * θ̂[(pμ+pσ+1):(pμ+pσ+pν)])
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, drm_optim_converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# #1060: the skew-normal information matrix is singular at ν = 0 (the family
# collapses to the Gaussian, so the slant's score is identically zero there), so
# the Wald SE for ν is not trustworthy in a neighbourhood of 0 — drmTMB#1336
# measured type-I 0.236 for the nominal-0.05 Wald test. The rule is this package's
# own: the note fires when any fitted |ν̂ᵢ| < `cut` (0.5). The cutoff is an
# uncalibrated heuristic, not a drmTMB constant, and no Wald test is run.
# Returns `nothing` for any other family.
const _SKEWNORMAL_NU_WALD_CUT = 0.5

function _skew_normal_nu_wald(fit::DrmFit; cut::Real = _SKEWNORMAL_NU_WALD_CUT)
    fit.family isa SkewNormal || return nothing
    haskey(fit.scales, :nu) || return nothing
    ν = fit.scales[:nu]
    m = minimum(abs, ν)
    return (near_zero = m < cut, min_abs_nu = m, cut = float(cut))
end

_skew_normal_nu_wald_message(r) =
    "SkewNormal slant ν̂ is near 0 (smallest |ν̂| = $(round(r.min_abs_nu; digits = 3)) < " *
    "$(r.cut)): the information matrix is singular at ν = 0 (the model collapses to the " *
    "Gaussian), so the Wald SE / CI / test for `nu` is unreliable (anti-conservative; " *
    "drmTMB#1336). Prefer a likelihood-ratio test of ν = 0 or a profile / bootstrap interval."
