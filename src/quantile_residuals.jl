# =============================================================================
# Quantile residuals (Dunn–Smyth randomized quantile residuals, à la DHARMa /
# glmmTMB) — per-family conditional-distribution dispatch.
#
# This file is included from DRModels.jl *after* every family type is defined
# (gaussian, student, poisson, negbinomial, beta, betabinomial, binomial,
# gamma, lognormal, zeroonebeta, tweedie, cumulative). The `_conditional_dist`
# / `_is_continuous_family` methods dispatch on those family types, so they must
# be defined here rather than in gaussian_core.jl (which loads before the family
# files and would otherwise hit `UndefVarError: Student not defined`).
#
# Entry point: `_quantile_residuals(fit, rng)`, called by
# `residuals(fit; type = :quantile)` in gaussian_core.jl.
# =============================================================================

# ---- per-family conditional distribution (the parameter→Distributions map) ----
#
# `_conditional_dist(fam, i; μ, scales, obs)` returns the fitted per-observation
# response distribution `F_i` as a `Distributions.Distribution`. This is the one
# place the working-scale parameters (μ on the response scale; the `scales` Dict)
# are mapped to the `Distributions.jl` constructor, so it is reusable by
# `residuals(type=:quantile)` and any future `simulate`/PIT/predictive checks.
#
# Scale conventions (verified against the family kernels and `simulate`):
#   • Gaussian     Normal(μ, σ),                  σ = scales[:sigma]
#   • Student      μ + σ·TDist(ν)  (LocationScale), σ = scales[:sigma], ν = scales[:nu]
#   • LogNormal    LogNormal(meanlog, sdlog), meanlog = log(μ̂) (μ̂ stored = exp(η_μ),
#                  the response-scale median), sdlog = σ = scales[:sigma]
#   • Gamma        Gamma(α, μ/α),                 α = σ⁻²  (shape; scales[:sigma]⁻²)
#                  BUT for a coupled location–scale Gamma fit the sigma slot holds
#                  the SHAPE α directly (α = exp ψ), not σ — see the note below the
#                  Gamma method and `_gamma_sigma_is_shape`.
#   • Beta         Beta(μφ, (1−μ)φ),              φ = σ⁻²  (precision)
#   • Poisson      Poisson(μ)
#   • NegBinomial2 NegativeBinomial(φ, φ/(φ+μ)),  φ = σ⁻²  (size; scales[:sigma] = σ)
#   • TruncNB2     truncated(NegativeBinomial(φ, p); lower = 0)   (support ≥ 1)
#   • Binomial     Binomial(n, p),  p = μ (success prob), n = scales[:trials]
#   • BetaBinomial BetaBinomial(n, μφ, (1−μ)φ),  φ = σ⁻², n = scales[:trials]
# ZeroOneBeta and CumulativeLogit are mixtures / need cut intervals and are handled
# by the atomic / ordinal drivers below rather than returning a single Distribution.
function _conditional_dist(fam::Gaussian, i; μ, scales, obs, kwargs...)
    return Distributions.Normal(μ[i], scales[:sigma][i])
end
function _conditional_dist(fam::Student, i; μ, scales, obs, kwargs...)
    return μ[i] + scales[:sigma][i] * Distributions.TDist(scales[:nu][i])
end
function _conditional_dist(fam::LogNormal, i; μ, scales, obs, kwargs...)
    return Distributions.LogNormal(log(max(μ[i], eps())), scales[:sigma][i])
end
# Gamma sigma-slot convention differs by fit route (see `_gamma_sigma_is_shape`):
# the plain / ranef fits store scales[:sigma] = σ (shape α = σ⁻²); the coupled
# location–scale fit stores scales[:sigma] = α (the shape itself, α = exp ψ). The
# caller passes `gamma_sigma_is_shape` so the same Gamma(α, μ/α) is rebuilt either
# way. Getting this wrong makes the PIT use α = σ⁻² = 1/α², inverting the shape.
function _conditional_dist(fam::Gamma, i; μ, scales, obs, gamma_sigma_is_shape::Bool = false, kwargs...)
    α = gamma_sigma_is_shape ? scales[:sigma][i] : 1 / (scales[:sigma][i]^2)
    return Distributions.Gamma(α, μ[i] / α)
end
function _conditional_dist(fam::Beta, i; μ, scales, obs, kwargs...)
    φ = 1 / (scales[:sigma][i]^2)
    m = clamp(μ[i], eps(), 1 - eps())
    return Distributions.Beta(m * φ, (1 - m) * φ)
end
function _conditional_dist(fam::Poisson, i; μ, scales, obs, kwargs...)
    return Distributions.Poisson(max(μ[i], 0.0))
end
function _conditional_dist(fam::TruncatedPoisson, i; μ, scales, obs, kwargs...)
    return Distributions.Poisson(max(μ[i], 0.0))
end
function _conditional_dist(fam::NegBinomial2, i; μ, scales, obs, kwargs...)
    φ = 1 / (scales[:sigma][i]^2)               # scales[:sigma] = σ now; NB2 size = 1/σ²
    return Distributions.NegativeBinomial(φ, φ / (φ + μ[i]))
end
# TruncatedNegBinomial2 returns the *base* (untruncated) NB2; the zero-truncation
# F(k) = (NB.cdf(k) − NB.cdf(0)) / (1 − NB.cdf(0)) for k ≥ 1 is applied in the
# discrete driver (avoids the `truncated` discrete-lower-bound convention).
function _conditional_dist(fam::TruncatedNegBinomial2, i; μ, scales, obs, kwargs...)
    φ = 1 / (scales[:sigma][i]^2)               # scales[:sigma] = σ now; NB2 size = 1/σ²
    return Distributions.NegativeBinomial(φ, φ / (φ + μ[i]))
end
function _conditional_dist(fam::Binomial, i; μ, scales, obs, kwargs...)
    n = round(Int, scales[:trials][i])
    return Distributions.Binomial(n, clamp(μ[i], eps(), 1 - eps()))
end
function _conditional_dist(fam::BetaBinomial, i; μ, scales, obs, kwargs...)
    φ = 1 / (scales[:sigma][i]^2)
    m = clamp(μ[i], eps(), 1 - eps())
    n = round(Int, scales[:trials][i])
    return Distributions.BetaBinomial(n, m * φ, (1 - m) * φ)
end

# Does this fit store the Gamma SHAPE (α) in the sigma slot instead of σ? True only
# for the coupled location–scale Gamma route, whose kernel sets α = exp ψ and whose
# frontend stores scales[:sigma] = exp(Xψ β) = α (`locscale_frontend.jl`), unlike
# the plain / ranef Gamma fits that store scales[:sigma] = σ with α = σ⁻²
# (`gamma.jl`). Keyed off the `LocScaleObjective{Val{:gamma}}` the location–scale
# frontend attaches to the fit — the one structural marker that distinguishes the
# two routes. Any non-Gamma or non-location–scale fit returns `false`.
function _gamma_sigma_is_shape(fit::DrmFit)
    fit.family isa Gamma || return false
    obj = fit.nll
    return obj isa LocScaleObjective && obj.kind isa Val{:gamma}
end

# Families whose conditional distribution is continuous (PIT = F(y), no RNG) vs
# discrete (randomized PIT u ~ Uniform[F(y⁻), F(y)]).
_is_continuous_family(::Gaussian)   = true
_is_continuous_family(::Student)    = true
_is_continuous_family(::LogNormal)  = true
_is_continuous_family(::Gamma)      = true
_is_continuous_family(::Beta)       = true
_is_continuous_family(::Poisson)    = false
_is_continuous_family(::NegBinomial2) = false
_is_continuous_family(::TruncatedNegBinomial2) = false
_is_continuous_family(::TruncatedPoisson) = false
_is_continuous_family(::Binomial)   = false
_is_continuous_family(::BetaBinomial) = false

# The observed value the PIT is evaluated at. Binomial/BetaBinomial store the
# observed proportion in obs[:mu]; the count is proportion × trials.
_pit_obs(::Binomial, i; obs, scales)     = round(Int, obs[:mu][i] * scales[:trials][i])
_pit_obs(::BetaBinomial, i; obs, scales) = round(Int, obs[:mu][i] * scales[:trials][i])
_pit_obs(::Any, i; obs, scales)          = obs[:mu][i]

# ---- marginalising an ordinary random intercept `(1 | g)` on the mean (#760) ----
#
# `residuals(fit; type = :quantile)` on a mixed fit used to judge each row
# against `fitted(fit)` at the random intercept fixed at 0 (the population/
# fixed-effect mean stored in `means[:mu]` — see the `_fit_*_ranef` routes in
# beta.jl/binomial.jl/etc.), i.e. against the WRONG reference distribution.
# On a correctly specified GLMM that inflates the PIT residuals' spread — the
# fitted model is not misspecified, the reference distribution is. `ranef()`
# (drmTMB-style conditional modes) is not yet available for a non-Gaussian GLMM
# (`ranef()`'s docstring, gaussian_ranef.jl), so the reference this fixes on is
# the MARGINAL distribution: b_g ~ N(0, σ_b²) integrated out of the conditional
# response distribution by the same 32-node Gauss–Hermite quadrature the ranef
# fitters themselves use to integrate the random intercept out of the
# likelihood — the natural "what does the rest of the package already compute
# with this σ_b" answer, and what drmTMB/DHARMa call the population-level PIT.
# A future `type = :quantile, marginal = false` (or conditional-mode) variant is
# tracked separately once #759 wires non-Gaussian `ranef()`.
#
# Only the SINGLE ordinary random intercept case is marginalised (a lone
# `:resd` block with exactly one grouping name — the shape every `_fit_*_ranef`
# GLMM produces, and also an ultrametric-tree phylo/relmat random intercept
# whose per-tip marginal variance equals σ_b² when the tree height is 1, the
# convention this package documents elsewhere). Crossed `(1|g)+(1|h)`,
# correlated `(1+x|g)` (`:recov`), and families with no verified link mapping
# below keep the previous fixed-effect-only reference rather than risk a wrong
# marginalisation; `_ranef_link` returning `nothing` is exactly that guard.
_ranef_link(::Poisson) = (μ -> log(max(μ, eps())), exp)
_ranef_link(::NegBinomial2) = (μ -> log(max(μ, eps())), exp)
_ranef_link(::TruncatedNegBinomial2) = (μ -> log(max(μ, eps())), exp)
_ranef_link(::Gamma) = (μ -> log(max(μ, eps())), exp)
_ranef_link(::LogNormal) = (μ -> log(max(μ, eps())), exp)
_ranef_link(::Binomial) = (μ -> (m = clamp(μ, eps(), 1 - eps()); log(m / (1 - m))), _logistic)
_ranef_link(::Beta) = (μ -> (m = clamp(μ, eps(), 1 - eps()); log(m / (1 - m))), _logistic)
_ranef_link(::BetaBinomial) = (μ -> (m = clamp(μ, eps(), 1 - eps()); log(m / (1 - m))), _logistic)
_ranef_link(::ZeroOneBeta) = (μ -> (m = clamp(μ, eps(), 1 - eps()); log(m / (1 - m))), _logistic)
_ranef_link(::Gaussian) = (identity, identity)
_ranef_link(::Student) = (identity, identity)
_ranef_link(fam) = nothing

# A single length-n Dict sliced down to a length-1 Dict at row `i`, so the
# per-family `_conditional_dist(fam, 1; μ, scales, obs, …)` builder can be
# reused unchanged to construct one quadrature node's conditional distribution.
_at_index(d::Dict, i) = Dict(k => (v isa AbstractVector ? [v[i]] : v) for (k, v) in d)

# The single ordinary `(1 | g)` block on the mean, or `nothing`. Deliberately
# excludes crossed (two names) and correlated-slope (`:recov`) blocks.
function _ordinary_resd_range(fit::DrmFit)
    for (p, r) in fit.blocks
        p === :resd && length(r) == 1 && return r
    end
    return nothing
end

# Precomputed marginalisation context for `_cdf_value`, or `nothing` when the
# fit has no single ordinary random intercept on the mean, or the family has no
# verified link mapping above.
function _ranef_marginal_mix(fit::DrmFit, fam, μ)
    r = _ordinary_resd_range(fit)
    r === nothing && return nothing
    linkpair = _ranef_link(fam)
    linkpair === nothing && return nothing
    link, invlink = linkpair
    σb = exp(fit.theta[r[1]])
    eta0 = [link(μ[i]) for i in eachindex(μ)]
    z, w = _gauss_hermite(32)
    return (invlink = invlink, eta0 = eta0, rt2σb = sqrt(2.0) * σb, z = z, wk = w ./ sqrt(π))
end

# CDF at `yval` for observation `i`: the plain per-family conditional
# distribution (`mix === nothing`), or the σ_b-marginalised mixture over the
# random intercept's 32 Gauss–Hermite nodes (#760).
function _cdf_value(fam, i, yval; μ, scales, obs, gsis, mix)
    if mix === nothing
        d = _conditional_dist(fam, i; μ = μ, scales = scales, obs = obs, gamma_sigma_is_shape = gsis)
        return Distributions.cdf(d, yval)
    end
    acc = 0.0
    scales_i = _at_index(scales, i); obs_i = _at_index(obs, i)
    @inbounds for k in eachindex(mix.z)
        μk = mix.invlink(mix.eta0[i] + mix.rt2σb * mix.z[k])
        d = _conditional_dist(fam, 1; μ = [μk], scales = scales_i, obs = obs_i, gamma_sigma_is_shape = gsis)
        acc += mix.wk[k] * Distributions.cdf(d, yval)
    end
    return acc
end

# Zero-truncated Poisson PIT interval [F_t(y−1), F_t(y)], y ≥ 1, with
# F_t(k) = P(1 ≤ Y ≤ k) / (1 − P(0)), in log space (running `_logaddexp` of the
# pmf terms; `_log1mexp(-λ)` for the divisor) so a tiny λ cannot make it 0/0.
# Shared by `TruncatedPoisson()` and the positive part of a hurdle Poisson fit.
function _ztpois_pit_interval(λ, yi)
    log1mF0 = _log1mexp(-λ)
    lpmf(j) = j * log(λ) - λ - _logfactorial(j)
    loga = -Inf
    for j in 1:(yi - 1)
        loga = _logaddexp(loga, lpmf(j))
    end
    logb = _logaddexp(loga, lpmf(yi))
    return exp(loga - log1mF0), exp(logb - log1mF0)
end

# Zero-truncated NB2 PIT interval [F_t(y−1), F_t(y)], y ≥ 1, size r, mean μ.
# F_t(k) = (NB.cdf(k) − NB.cdf(0)) / (1 − NB.cdf(0)), k ≥ 1 — but built from
# `_nb2_logpmf` / `_log1mexp` (negbinomial.jl, poisson.jl) rather than
# `Distributions.cdf`. At extreme dispersion (r = 1/σ² ≫ μ, log σ ≲ -20) or
# extreme small μ (μ/r underflows to exactly 0),
# `Distributions.NegativeBinomial(r, r/(r+μ)).cdf` rounds to EXACTLY 1.0 for every
# k (r+μ rounds to r in float64, so p rounds to 1): both the numerator
# (NB.cdf(k) − F0) and the denominator (1 − F0) evaluate to 0.0, giving
# 0/0 = NaN (same cancellation #866/#874 fixed in the likelihood). Working
# entirely in log space avoids ever forming that degenerate p: log(1 − F0) via
# `_log1mexp(_nb2_logpmf(r, μ, 0))`, and log(NB.cdf(k) − F0) = log P(1 ≤ Y ≤ k)
# via a running `_logaddexp` sum of `_nb2_logpmf(r, μ, j)` terms, both finite for
# any r as long as μ > 0. Shared by `TruncatedNegBinomial2()` and the positive
# part of a hurdle NB2 fit.
function _ztnb2_pit_interval(r, μi, yi)
    log1mF0 = _log1mexp(_nb2_logpmf(r, μi, 0))
    if isinf(log1mF0)
        # μ underflowed to ~0 in float64: the untruncated model puts
        # (numerically) all its mass at 0, so the zero-truncated tail
        # probability for any observed y ≥ 1 is ~1 — saturate rather than
        # divide 0/0. The caller's `lo`/`hi` clamp still maps this to a finite
        # (large) residual, matching every other branch in this file.
        return 1.0, 1.0
    end
    loga = -Inf
    for j in 1:(yi - 1)
        loga = _logaddexp(loga, _nb2_logpmf(r, μi, j))
    end
    logb = yi >= 1 ? _logaddexp(loga, _nb2_logpmf(r, μi, yi)) : loga
    return exp(loga - log1mF0), exp(logb - log1mF0)
end

# PIT interval [F(y−1), F(y)] of a zero-inflated (`scales[:zi]`) or hurdle
# (`scales[:hu]`) Poisson / NB2 fit, from the FULL mixture CDF (#922). Both
# probabilities are stored on the probability scale (logit link) by
# `_fit_poisson_zi` / `_fit_negbin2_zi` / `_fit_poisson_hu` / `_fit_negbin2_hu`,
# and `μ` is the COUNT-component mean (`means[:mu]`):
#   zero-inflated: F(y) = π + (1 − π)·F_c(y),               y ≥ 0;
#   hurdle:        F(0) = p0, F(y) = p0 + (1 − p0)·F_t(y),   y ≥ 1,
# with F(−1) = 0, F_c the count CDF and F_t its zero-truncated CDF (the
# log-space helpers above, so a hurdle NB2 at extreme dispersion cannot go 0/0).
# `drm()` refuses `zi`/`hu` combined with any random or structured effect, so
# `mix` is always `nothing` here; the zero-inflated branch would marginalise a
# future `(1 | g)` through `_cdf_value`, and the hurdle branch refuses one rather
# than silently judging it at b = 0.
function _zi_hurdle_pit_interval(fit::DrmFit, fam, i, yi; μ, gsis, mix)
    if haskey(fit.scales, :zi)
        πi = fit.scales[:zi][i]
        Fc(k) = _cdf_value(fam, i, k; μ = μ, scales = fit.scales, obs = fit.obs, gsis = gsis, mix = mix)
        a = yi <= 0 ? 0.0 : πi + (1 - πi) * Fc(yi - 1)
        return a, πi + (1 - πi) * Fc(yi)
    end
    mix === nothing || throw(ArgumentError("residuals(type=:quantile): a hurdle fit " *
        "with a random effect has no σ_b-marginal PIT yet"))
    p0 = fit.scales[:hu][i]
    yi <= 0 && return 0.0, p0
    # Floor μ at eps() for BOTH count families, so a count mean that underflowed
    # to 0 gives the same zero-truncated limit (all mass at y = 1) for NB2 as for
    # Poisson, instead of the NB2 helper's saturated (1, 1).
    μi = max(μ[i], eps())
    at, bt = fam isa NegBinomial2 ?
        _ztnb2_pit_interval(1 / (fit.scales[:sigma][i]^2), μi, yi) :
        _ztpois_pit_interval(μi, yi)
    return p0 + (1 - p0) * at, p0 + (1 - p0) * bt
end

# Randomized quantile residuals r_i = Φ⁻¹(u_i) (Dunn & Smyth; DHARMa / glmmTMB).
# Continuous families use u = F(y); discrete families randomize within the jump
# interval [F(y⁻), F(y)]; ZeroOneBeta / CumulativeLogit use the atomic / ordinal
# drivers (point-mass mixtures); zero-inflated / hurdle count fits randomize
# within the full mixture CDF (`_zi_hurdle_pit_interval`, #922). The per-family
# parameter map lives in `_conditional_dist`. A fit with a single ordinary
# random intercept `(1 | g)` on the mean judges every row against the
# σ_b-MARGINAL distribution, not the fixed-effect-only (b = 0) distribution
# (#760) — see `_ranef_marginal_mix`.
function _quantile_residuals(fit::DrmFit, rng)
    haskey(fit.means, :mu) ||
        throw(ArgumentError("residuals(type=:quantile) is univariate-only"))
    fam = fit.family
    (fam isa Gaussian && haskey(fit.scales, :sigma1)) &&
        throw(ArgumentError("residuals(type=:quantile) is univariate-only"))
    y = fit.obs[:mu]
    μ = fit.means[:mu]
    n = length(y)
    lo = eps(); hi = 1 - eps()
    std_normal = Distributions.Normal()

    if fam isa ZeroOneBeta
        return _quantile_residuals_zeroonebeta(fit, rng, lo, hi)
    elseif fam isa CumulativeLogit
        return _quantile_residuals_cumulative(fit, rng, lo, hi)
    elseif fam isa Tweedie
        throw(ArgumentError("residuals(type=:quantile): Tweedie has no closed-form CDF " *
            "in Distributions.jl; a Tweedie compound Poisson–Gamma CDF is tracked as a " *
            "follow-up. All other DRModels.jl families are supported."))
    end

    # Single-distribution families via `_conditional_dist`.
    applicable(_is_continuous_family, fam) ||
        throw(ArgumentError("residuals(type=:quantile): $(nameof(typeof(fam))) has no " *
            "verified per-family CDF mapping yet"))
    # The Gamma sigma slot is σ (plain/ranef) or the shape α (location–scale); the
    # flag routes `_conditional_dist(::Gamma)` accordingly (non-Gamma ignores it).
    gsis = _gamma_sigma_is_shape(fit)
    # `mix` marginalises a single ordinary random intercept on the mean over its
    # fitted σ_b (#760); `nothing` for a fixed-effects-only fit (unchanged
    # behaviour) or a random-effect shape/family this fix does not cover.
    mix = _ranef_marginal_mix(fit, fam, μ)
    u = Vector{Float64}(undef, n)
    if _is_continuous_family(fam)
        @inbounds for i in 1:n
            F = _cdf_value(fam, i, y[i]; μ = μ, scales = fit.scales, obs = fit.obs,
                           gsis = gsis, mix = mix)
            u[i] = clamp(F, lo, hi)
        end
    elseif fam isa TruncatedPoisson
        @inbounds for i in 1:n
            a, b = _ztpois_pit_interval(max(μ[i], eps()), round(Int, y[i]))
            u[i] = clamp(a + (b - a) * rand(rng), lo, hi)
        end
    elseif fam isa TruncatedNegBinomial2
        @inbounds for i in 1:n
            if mix === nothing
                # Fixed-effect reference: the log-space zero-truncated NB2
                # CDF (#876), finite at extreme dispersion.
                r = 1 / (fit.scales[:sigma][i]^2)       # NB2 size; scales[:sigma] = σ
                a, b = _ztnb2_pit_interval(r, μ[i], round(Int, y[i]))
            else
                # σ_b-marginal reference (#760): truncate the Gauss–Hermite
                # mixture of untruncated NB2 CDFs.
                yi = round(Int, y[i])
                F0 = _cdf_value(fam, i, 0; μ = μ, scales = fit.scales, obs = fit.obs, gsis = gsis, mix = mix)
                denom = 1 - F0
                # zero-truncated CDF: F_t(k) = (NB.cdf(k) − F0)/(1 − F0), k ≥ 1
                a = (_cdf_value(fam, i, yi - 1; μ = μ, scales = fit.scales, obs = fit.obs, gsis = gsis, mix = mix) - F0) / denom
                b = (_cdf_value(fam, i, yi; μ = μ, scales = fit.scales, obs = fit.obs, gsis = gsis, mix = mix) - F0) / denom
            end
            u[i] = clamp(a + (b - a) * rand(rng), lo, hi)
        end
    elseif haskey(fit.scales, :zi) || haskey(fit.scales, :hu)
        # Zero-inflated / hurdle Poisson or NB2: the full mixture CDF (#922).
        @inbounds for i in 1:n
            a, b = _zi_hurdle_pit_interval(fit, fam, i, round(Int, y[i]); μ = μ, gsis = gsis, mix = mix)
            u[i] = clamp(a + (b - a) * rand(rng), lo, hi)
        end
    else
        @inbounds for i in 1:n
            yi = _pit_obs(fam, i; obs = fit.obs, scales = fit.scales)
            a = _cdf_value(fam, i, yi - 1; μ = μ, scales = fit.scales, obs = fit.obs, gsis = gsis, mix = mix)
            b = _cdf_value(fam, i, yi; μ = μ, scales = fit.scales, obs = fit.obs, gsis = gsis, mix = mix)
            u[i] = clamp(a + (b - a) * rand(rng), lo, hi)
        end
    end
    return Distributions.quantile.(std_normal, u)
end

# ZeroOneBeta atomic driver. Mixture: P(0) = zoi(1−coi) at the atom 0, P(1) = zoi·coi
# at the atom 1, and the interior is (1−zoi)·Beta(μβφ,(1−μβ)φ) on (0,1). The CDF is
#   F(0⁻)=0, F(0)=zoi(1−coi);
#   F(y∈(0,1)) = zoi(1−coi) + (1−zoi)·Beta.cdf(y);   (continuous interior)
#   F(1⁻)=zoi(1−coi)+(1−zoi), F(1)=1.
# A value AT an atom gets u ~ Uniform[F(atom⁻), F(atom)] (randomized across the mass);
# interior values get the plain PIT. Generalizes the discrete driver.
#
# The zoi/coi atom masses never depend on the mean's random effect, so their
# contribution is exactly the marginal one already; only the interior Beta
# term needs σ_b-marginalising (#760, extended to `ZeroOneBeta()`'s own `(1|g)`
# route added by #723) — same 32-node Gauss–Hermite mixture, logit link on
# `beta_mu`, via `_ranef_marginal_mix`/`_cdf_value`.
function _quantile_residuals_zeroonebeta(fit::DrmFit, rng, lo, hi)
    y = fit.obs[:mu]; n = length(y)
    μb = fit.scales[:beta_mu]; σ = fit.scales[:sigma]
    zoi = fit.scales[:zoi]; coi = fit.scales[:coi]
    mix = _ranef_marginal_mix(fit, fit.family, μb)
    std_normal = Distributions.Normal()
    u = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        p0 = zoi[i] * (1 - coi[i])              # mass at 0
        if y[i] == 0
            u[i] = clamp((0.0) + (p0 - 0.0) * rand(rng), lo, hi)
        elseif y[i] == 1
            a = p0 + (1 - zoi[i])               # F(1⁻)
            u[i] = clamp(a + (1.0 - a) * rand(rng), lo, hi)
        else
            φ = 1 / (σ[i]^2)
            if mix === nothing
                m = clamp(μb[i], eps(), 1 - eps())
                Fc = Distributions.cdf(Distributions.Beta(m * φ, (1 - m) * φ), y[i])
            else
                Fc = 0.0
                @inbounds for k in eachindex(mix.z)
                    m = clamp(mix.invlink(mix.eta0[i] + mix.rt2σb * mix.z[k]), eps(), 1 - eps())
                    Fc += mix.wk[k] * Distributions.cdf(Distributions.Beta(m * φ, (1 - m) * φ), y[i])
                end
            end
            u[i] = clamp(p0 + (1 - zoi[i]) * Fc, lo, hi)
        end
    end
    return Distributions.quantile.(std_normal, u)
end

# CumulativeLogit ordinal driver. For category k ∈ {1,…,K}, F(k) = logistic(cuts[k] − η)
# (F(K)=1, F(0)=0); the observed category gets a randomized PIT within its probability
# interval [F(k−1), F(k)] — the discrete driver applied to the cumulative cutpoints.
function _quantile_residuals_cumulative(fit::DrmFit, rng, lo, hi)
    y = round.(Int, fit.obs[:mu]); n = length(y)
    η = fit.scales[:ordinal_eta]; cuts = fit.scales[:ordinal_cuts]
    K = length(cuts) + 1
    std_normal = Distributions.Normal()
    Fcum(k, i) = k <= 0 ? 0.0 : k >= K ? 1.0 : _logistic(cuts[k] - η[i])
    u = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        k = y[i]
        a = Fcum(k - 1, i); b = Fcum(k, i)
        u[i] = clamp(a + (b - a) * rand(rng), lo, hi)
    end
    return Distributions.quantile.(std_normal, u)
end
