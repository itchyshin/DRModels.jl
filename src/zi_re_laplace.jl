# zi_re_laplace.jl: one ordinary random intercept `(1 | g)` on the mean of the
# zero-inflated / hurdle / zero-truncated count families and of SkewNormal, fitted by
# drmTMB's objective: the TMB-convention Laplace marginal.
#
# No new Laplace code. This file adds observation terms to the dense spine of
# `indep_slopes_laplace.jl` (`_fit_indep_re_laplace`), the route Gamma, Student,
# Poisson, NB2, Binomial, Beta and Tweedie already use for independent random terms. Each
# term returns (−log f, d/dη, d²/dη², expected d²/dη²) for one row. Only η, the linear
# predictor of the mean, carries a random effect, so the derivatives are taken in η alone;
# the zero-inflation / hurdle logit, the NB2 dispersion and the skew-normal slant are
# fixed-effect parameters and enter as plain numbers. The derivatives come from
# ForwardDiff on the row value (nested under the outer ForwardDiff over θ), so they cannot
# disagree with the density the fixed-effects fitters use. The "expected" second derivative
# is only the positive-definite fallback direction of the inner Newton solve; the Laplace
# value always uses the observed Hessian. It is `max(d², 0)`.
#
#   kind        family and parts                                        extra block
#   Val(:zip)   Poisson   mu + zi,  zi = logit P(structural zero)        :zi  (the ν slot)
#   Val(:zinb)  NB2       mu + sigma + zi                                :zi
#   Val(:tnb)   zero-truncated NB2   mu + sigma                          none
#   Val(:hnb)   hurdle NB2 (truncated NB2 + hu)  mu + sigma + hu         :hu
#   Val(:skew)  SkewNormal  mu (mean) + sigma (SD) + nu (slant)          :nu
#
# Scope. Exactly one `(1 | g)` on `mu`; maximum likelihood; no `offset(...)`; no random
# effect on `sigma`/`zi`/`hu`/`nu`; no structured marker; no `meta_V`. Everything else is
# refused by name, never rerouted. The fit is tagged `marginal = :Laplace`, because
# Laplace is the only integrator on this route (`:LA` is not a different integrator here, and
# `:AGHQ` / `:VA` are refused).

function _indep_ad_terms(v, η)
    d1 = h -> ForwardDiff.derivative(v, h)
    w = ForwardDiff.derivative(d1, η)
    return v(η), d1(η), w, max(w, zero(w))
end

function _indep_obs_terms(::Val{:zip}, y, η, lσ, ηz)
    ηzc = clamp(ηz, -30, 30)
    lπ = _log_logistic(ηzc); l1mπ = _log1m_logistic(ηzc); lf = loggamma(y + 1)
    v = h -> begin
        hc = _indep_clampη(h); μ = exp(hc)
        y == 0 ? -_logaddexp(lπ, l1mπ - μ) : -(l1mπ + y * hc - μ - lf)
    end
    return _indep_ad_terms(v, η)
end

function _indep_obs_terms(::Val{:zinb}, y, η, lσ, ηz)
    ηzc = clamp(ηz, -30, 30)
    lπ = _log_logistic(ηzc); l1mπ = _log1m_logistic(ηzc)
    r = exp(-2 * clamp(lσ, -20, 20)); k = round(Int, y)
    v = h -> begin
        μ = exp(clamp(h, -20, 20)); nb = _nb2_logpmf(r, μ, k)
        k == 0 ? -_logaddexp(lπ, l1mπ + nb) : -(l1mπ + nb)
    end
    return _indep_ad_terms(v, η)
end

function _indep_obs_terms(::Val{:tnb}, y, η, lσ, ην)
    r = exp(-2 * clamp(lσ, -20, 20)); k = round(Int, y)
    v = h -> begin
        μ = exp(clamp(h, -20, 20))
        -(_nb2_logpmf(r, μ, k) - _log1mexp(_nb2_logpmf(r, μ, 0)))
    end
    return _indep_ad_terms(v, η)
end

function _indep_obs_terms(::Val{:hnb}, y, η, lσ, ηh)
    ηhc = clamp(ηh, -30, 30)
    lπ = _log_logistic(ηhc); l1mπ = _log1m_logistic(ηhc)
    k = round(Int, y)
    if k == 0                                    # structural zero: no dependence on η
        z = zero(η) * 0
        return -lπ + z, z, z, z
    end
    r = exp(-2 * clamp(lσ, -20, 20))
    v = h -> begin
        μ = exp(clamp(h, -20, 20))
        -(l1mπ + _nb2_logpmf(r, μ, k) - _log1mexp(_nb2_logpmf(r, μ, 0)))
    end
    return _indep_ad_terms(v, η)
end

function _indep_obs_terms(::Val{:skew}, y, η, lσ, α)
    v = h -> -_skewnormal_logpdf_moment(y, h, lσ, α)
    return _indep_ad_terms(v, η)
end

_indep_mean(::Val{:zip}, η) = exp(η)
_indep_mean(::Val{:zinb}, η) = exp(η)
_indep_mean(::Val{:tnb}, η) = exp(η)
_indep_mean(::Val{:hnb}, η) = exp(η)
_indep_mean(::Val{:skew}, η) = η

_indep_nu_block(::Val{:zip}) = :zi
_indep_nu_block(::Val{:zinb}) = :zi
_indep_nu_block(::Val{:hnb}) = :hu
_indep_nu_scale(::Union{Val{:zip},Val{:zinb},Val{:hnb}}, ην) = _logistic.(ην)
_indep_nu_scale(::Val{:skew}, ην) = ην

_indep_famname(::Val{:zip}) = "zero-inflated Poisson"
_indep_famname(::Val{:zinb}) = "zero-inflated NegBinomial2"
_indep_famname(::Val{:tnb}) = "TruncatedNegBinomial2"
_indep_famname(::Val{:hnb}) = "hurdle NegBinomial2"
_indep_famname(::Val{:skew}) = "SkewNormal"

# Fixed-effects starts: the family's own fixed-effect fit, as the other kinds do.
_indep_fixed_start(::Val{:zip}, fam, yv, Xμ, Xσ, Xν, nmμ, nmσ, nmν, g_tol) =
    _fit_poisson_zi(fam, yv, Xμ, Xν, nmμ, nmν, g_tol).theta
_indep_fixed_start(::Val{:zinb}, fam, yv, Xμ, Xσ, Xν, nmμ, nmσ, nmν, g_tol) =
    _fit_negbin2_zi(fam, yv, Xμ, Xσ, Xν, nmμ, nmσ, nmν, g_tol).theta
_indep_fixed_start(::Val{:tnb}, fam, yv, Xμ, Xσ, Xν, nmμ, nmσ, nmν, g_tol) =
    _fit_truncated_negbin2(fam, yv, Xμ, Xσ, nmμ, nmσ, g_tol; se = false).theta
_indep_fixed_start(::Val{:hnb}, fam, yv, Xμ, Xσ, Xν, nmμ, nmσ, nmν, g_tol) =
    _fit_negbin2_hu(fam, yv, Xμ, Xσ, Xν, nmμ, nmσ, nmν, g_tol).theta
_indep_fixed_start(::Val{:skew}, fam, yv, Xμ, Xσ, Xν, nmμ, nmσ, nmν, g_tol) =
    _fit_skewnormal(fam, yv, Xμ, Xσ, Xν, nmμ, nmσ, nmν, g_tol).theta

# The dense Laplace spine builds a q x q Hessian over all random-effect levels. Same limit and
# wording as the other dense Laplace engines (`_GSL_MAX_LATENT`, gaussian_sigma_sd_laplace.jl).
function _dense_laplace_guard(fam, q::Integer)
    q <= _GSL_MAX_LATENT || throw(ArgumentError(
        "drm ($(_family_label(fam))): this random-effect model uses a dense Laplace " *
        "engine limited to $(_GSL_MAX_LATENT) latent coordinates (got $q)."))
    return nothing
end

function _zi_re_reject(fam, what)
    throw(ArgumentError(
        "drm ($(_family_label(fam))): a random effect on the mean with zero inflation, a hurdle, " *
        "zero truncation or a skew-normal response is fitted only for one ordinary random " *
        "intercept `(1 | g)` on `mu`, by the Laplace approximation (what drmTMB uses), " *
        "maximum likelihood. $what is not available on this route. Refused rather than " *
        "fitted as a different model."))
end

"""
    _drm_zi_re_laplace(f, fam; data, g_tol) -> DrmFit

Entry for `Poisson()` + `zi`, `NegBinomial2()` + `zi` / `hu`, `TruncatedNegBinomial2()`
(with or without `hu`) and `SkewNormal()` when `mu` carries one `(1 | g)`. Validates the
admitted shape and calls `_fit_indep_re_laplace` with the matching observation kind.
"""
function _drm_zi_re_laplace(f::DrmFormula, fam; data, g_tol::Real = 1e-8, start = nothing)
    rhs = Dict(f.forms)
    haskey(rhs, :zi) && haskey(rhs, :hu) &&
        error("`zi` and `hu` cannot both be specified (zero-inflation vs hurdle)")
    if fam isa Poisson
        haskey(rhs, :hu) && _zi_re_reject(fam, "a hurdle part `hu` (use `zi`, or `NegBinomial2()`)")
        allowed = (:mu, :zi)
    elseif fam isa NegBinomial2
        allowed = (:mu, :sigma, :zi, :hu)
    elseif fam isa TruncatedNegBinomial2
        allowed = (:mu, :sigma, :hu)
    else
        allowed = (:mu, :sigma, :nu)             # SkewNormal
    end
    for (pname, r) in f.forms
        pname === :sigma && fam isa Poisson && r === ConstantTerm(1) && continue   # `bf` fills `sigma ~ 1`
        pname in allowed || _zi_re_reject(fam, "a `$pname` formula")
        pname === :mu && continue
        _, re2, mv2, st2 = _split_ranef(r)
        (isempty(re2) && mv2 === nothing && st2 === nothing) ||
            _zi_re_reject(fam, "a random effect on `$pname`")
    end
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    mv === nothing || _zi_re_reject(fam, "a `meta_V` marker")
    st === nothing || _zi_re_reject(fam, "a structured (phylo/relmat/animal/spatial) random effect")
    fixed_mu, offset_term = _extract_offset(fixed_mu)
    offset_term === nothing ||
        _zi_re_reject(fam, "`offset(...)` (its combination with this random-effect route is not verified against drmTMB)")
    length(re) == 1 || _zi_re_reject(fam, "crossed or multiple random effects")
    _re_kind(re[1][1])[1] === :intercept ||
        _zi_re_reject(fam, "a random slope or a correlated block (only `(1 | g)`)")
    grp = re[1][2]
    gidx, G = _group_index(getproperty(data, grp); col = grp)
    y, Xμ, nmμ = _design(f.response, fixed_mu, data)
    n = length(y)
    _dense_laplace_guard(fam, G)
    comps = [(ones(n), gidx, G, String(grp))]
    has_sigma = !(fam isa Poisson)
    Xσ = nothing; nmσ = String[]
    if has_sigma
        _, Xσ, nmσ = _design(f.response, get(rhs, :sigma, ConstantTerm(1)), data)
    end
    if fam isa SkewNormal
        _, Xν, nmν = _design(f.response, get(rhs, :nu, ConstantTerm(1)), data)
        kind = Val(:skew)
        all(isfinite, y) || error("SkewNormal() requires finite responses")
    else
        all(yi -> yi ≥ 0 && isinteger(yi), y) ||
            error("$(_family_label(fam)) requires non-negative integer counts as the response")
        if fam isa TruncatedNegBinomial2 && !haskey(rhs, :hu)
            all(≥(1), y) ||
                error("TruncatedNegBinomial2() requires positive integer counts (≥ 1) as the response")
            kind = Val(:tnb); Xν = nothing; nmν = String[]
        else
            part = haskey(rhs, :zi) ? :zi : haskey(rhs, :hu) ? :hu : nothing
            part === nothing && _zi_re_reject(fam, "no `zi` or `hu` part (use the ordinary random-intercept route)")
            _, Xν, nmν = _design(f.response, rhs[part], data)
            kind = fam isa Poisson ? Val(:zip) : part === :zi ? Val(:zinb) : Val(:hnb)
        end
    end
    # The truncated-NB2 + hu spelling is the NegBinomial2 hurdle (see the fixed-effects entry).
    fitfam = (fam isa TruncatedNegBinomial2 && haskey(rhs, :hu)) ? NegBinomial2() : fam
    run = st -> _fit_indep_re_laplace(fitfam, kind, y, Xμ, Xσ, Xν, comps, nmμ, nmσ, nmν, g_tol; start = st)
    fit = run(start)
    start === nothing && kind isa Union{Val{:zip},Val{:zinb},Val{:tnb},Val{:hnb}} &&
        (fit = _zi_re_restarts(fit, run, kind, y, Xμ, Xσ, Xν))
    return _withformula(fit, f)
end

# Interior restarts. The start of these kinds is the family's fixed-effects fit (random effect
# ignored), and that fit can sit on a boundary: the zero-mass logit at about -30 (the likelihood is
# flat in that coordinate) or a dispersion at its limit (an overdispersed count explained by the
# dispersion alone). The Laplace objective is then flat or not finite at the start, and the fit
# stalls there with a log-likelihood up to 10 units below drmTMB (ZINB, two seeds of the review) or
# NaN. A fit that did not converge, has no finite log-likelihood, or whose zero-mass linear
# predictor reached 15 or more in absolute value is refitted from interior starts: mean block by
# least squares on log(y + 0.5), log sigma = log(0.5), log(1), log(0.3), zero-mass intercept
# 0, -1, 1, random-effect log SD log(0.3). At most three refits, and only these fits pay for them.
# A refit replaces the first fit only if its log-likelihood is higher (or equal and it converged).
# The converged flag is never changed: a fit that no start rescues stays `converged = false`.
const _ZI_RE_RESTART_STARTS = ((0.5, 0.0), (1.0, -1.0), (0.3, 1.0))   # (sigma, zero-mass intercept)
const _ZI_RE_BOUNDARY_ETA = 15.0

function _zi_re_restarts(fit::DrmFit, run, kind::Val, y, Xμ, Xσ, Xν)
    pμ = size(Xμ, 2); pσ = Xσ === nothing ? 0 : size(Xσ, 2); pν = Xν === nothing ? 0 : size(Xν, 2)
    νr = (pμ + pσ + 1):(pμ + pσ + pν)
    at_boundary(ft) = pν > 0 && maximum(abs, Xν * ft.theta[νr]) >= _ZI_RE_BOUNDARY_ETA
    ok(ft) = ft.converged && isfinite(ft.loglik) && ft.loglik > -1e17 && !at_boundary(ft)
    ok(fit) && return fit
    βμ = Xμ \ log.(Float64.(y) .+ 0.5)
    best = fit
    for (σ0, z0) in _ZI_RE_RESTART_STARTS
        st = zeros(length(fit.theta))
        st[1:pμ] .= βμ
        pσ > 0 && (st[pμ+1] = log(σ0))
        pν > 0 && (st[pμ+pσ+1] = z0)
        st[end] = log(0.3)
        cand = try run(st) catch err; err isa InterruptException && rethrow(); nothing end
        cand === nothing && continue
        lc = isfinite(cand.loglik) ? cand.loglik : -Inf; lb = isfinite(best.loglik) ? best.loglik : -Inf
        better = lc > lb + 1e-8 || (abs(lc - lb) <= 1e-8 && cand.converged && !best.converged)
        confirmed = ok(cand) || (cand.converged && abs(lc - lb) <= 1e-6)
        better && (best = cand)
        confirmed && break
    end
    return best
end

# ---------------------------------------------------------------------------
# Ordinal kind: CumulativeLogit with independent random terms `(1 | g) + (0 + x | g)`.
#
# Same spine; the cutpoints are the fixed-effect block that sits in the `sigma` slot
# (block `:cutpoints`, names `theta1…`, the increments δ of `_cumulative_cuts`: θ_1 = δ_1,
# θ_k = θ_{k-1} + e^{δ_k}). A row with category k depends only on the two cutpoints that bound
# it, so `_indep_aux_eta` hands each row its lower and upper cutpoint (ησ, ην) built from δ
# inside the objective, which keeps ForwardDiff's derivative through the ordering transform.
# The location intercept is absorbed by the cutpoints (the entry drops it, as every ordinal
# route does). Unlike the profile-over-cutpoints crossed route, the cutpoints are in θ, so
# their standard errors come out of the same Hessian as every other coefficient.
# ---------------------------------------------------------------------------
_indep_response(::Val{:cumlogit}, y) = (K = round(Int, maximum(y)); [(round(Int, k), K) for k in y])
_indep_obsval(::Val{:cumlogit}, yv) = Float64[k for (k, _) in yv]
_indep_mean(::Val{:cumlogit}, η) = η
_indep_famname(::Val{:cumlogit}) = "CumulativeLogit"
_indep_sigma_block(::Val{:cumlogit}) = :cutpoints

@inline function _cum_loglik_bounds(k::Int, K::Int, h, lo, up)
    k == 1 && return _log_logistic(up - h)
    k == K && return _log_logistic(h - lo)
    la = _log_logistic(up - h); lb = _log_logistic(lo - h)
    return la + _log1mexp(min(lb - la, zero(lb - la)))
end

function _indep_obs_terms(::Val{:cumlogit}, kd, η, lo, up)
    k, K = kd
    return _indep_ad_terms(h -> -_cum_loglik_bounds(k, K, h, lo, up), η)
end

function _indep_aux_eta(::Val{:cumlogit}, yv, Xσ, Xν, θ, pμ, pσ, k0, ::Type{T}) where {T}
    n = length(yv)
    cuts = _cumulative_cuts(θ[pμ+1:pμ+pσ])
    lo = zeros(T, n); up = zeros(T, n)
    @inbounds for i in 1:n
        k, K = yv[i]
        k > 1 && (lo[i] = cuts[k-1])
        k < K && (up[i] = cuts[k])
    end
    return lo, up
end

function _indep_fixed_start(::Val{:cumlogit}, fam, yv, Xμ, Xσ, Xν, nmμ, nmσ, nmν, g_tol)
    K = yv[1][2]; pμ = size(Xμ, 2)
    θ = zeros(pμ + size(Xσ, 2))
    θ[pμ+1:end] .= _cumulative_cut_init([k for (k, _) in yv], K, length(yv))
    return θ
end

function _indep_post(::Val{:cumlogit}, yv, means, scales, ημ̂, θ̂, pμ, pσ)
    K = yv[1][2]; cuts = _cumulative_cuts(θ̂[pμ+1:pμ+pσ])
    return Dict(:mu => _cumulative_score(ημ̂, cuts, K)),
           Dict(:ordinal_eta => ημ̂, :ordinal_cuts => Float64.(cuts))
end

# Independent intercept + slope terms `(1 | g) + (0 + x | g)` on the mean of a CumulativeLogit.
function _fit_cumulative_indep_laplace(fam, y::Vector{Int}, Xμ, K, comps, nmμ, g_tol)
    nc = K - 1
    _dense_laplace_guard(fam, sum(c[3] for c in comps))
    return _fit_indep_re_laplace(fam, Val(:cumlogit), y, Xμ, zeros(length(y), nc), nothing, comps,
                                 nmμ, ["theta$k" for k in 1:nc], String[], g_tol)
end

# ---------------------------------------------------------------------------
# Post-fit paths that are NOT verified for the multi-term fits opened here.
#
# CumulativeLogit with independent terms and Binomial with three or more crossed intercepts have
# no marginal (fresh random effect) simulator and no sigma_b-marginal PIT. Without a marginal
# simulator `bootstrap` falls back to `simulate(re = :zero)`, which draws without the random
# effects, and a quantile residual would be judged at b = 0. Both would be quietly wrong, so
# they are refused by name for exactly these fits (fits of the pre-existing two-term routes are
# not touched).
# ---------------------------------------------------------------------------
function _multi_re_laplace_unverified(fit::DrmFit)
    fit.marginal === :Laplace || return false
    i = findfirst(b -> first(b) === :resd, fit.blocks)
    i === nothing && return false
    k = length(last(fit.blocks[i]))
    return (fit.family isa CumulativeLogit && k >= 2) || (fit.family isa Binomial && k >= 3)
end

function _refuse_multi_re_laplace(fit::DrmFit, what)
    _multi_re_laplace_unverified(fit) || return nothing
    throw(ArgumentError(
        "$what is not available for this $(nameof(typeof(fit.family)))() fit with several random terms " *
        "(Laplace route): no fresh-random-effect simulator or marginal quantile residual has been " *
        "verified for it, and the fallback would draw without the random effects. Use `predict`, " *
        "`vcov` or `confint(fit)` instead" *
        (fit.family isa CumulativeLogit ? ", or `simulate(fit; re = :zero)` for population-level draws." : ".")))
end
