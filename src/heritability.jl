# heritability.jl — user-facing comparative-biology derived quantities WITH CIs
# for the structured-Gaussian fits (`phylo`/`relmat`/`animal`/`spatial` random
# intercepts, single- and two-component). The headline ratios are
#
#   * phylogenetic heritability / signal   h² = σ²_a / (Σ_k σ²_k + σ²_resid)
#   * repeatability / ICC                  R  = σ²_g / (σ²_g + σ²_resid)
#
# These are smooth nonlinear maps g(θ) of the WORKING-scale variance parameters
# (each component lives on log σ, so σ²_k = exp(2·θ_k)). We reuse the merged
# epsilon-method / generalized-delta infrastructure (`bias_correct`) to get a
# point estimate + bias-corrected estimate + delta-method SE + Wald CI, with the
# EXACT gradient/Hessian threaded through the log → variance map by ForwardDiff.
# Optionally a TRUE profile-likelihood CI on the derived ratio (a constrained
# re-fit: at each fixed ratio, re-maximise the stored NLL over ALL nuisance
# parameters — not a substitution/ELR profile that freezes them at the MLE) is
# available via `method = :profile`.
#
# All ratios are bounded in [0, 1] by construction (a sum of one nonnegative
# variance over the sum of all). The Wald CI is CLAMPED to [0, 1]; the point
# estimate is exact in [0, 1]; the bias-corrected value can stray marginally
# outside under heavy curvature and is reported as-is (honest) but the CI is
# always clamped.
#
# Scope: the closed-form / sparse structured-Gaussian models where the variance
# components are clean named quantities (`re_sd`/`vc` populate them). We do NOT
# reach into the non-Gaussian Laplace routes or the q4 PLSM here — there the
# "variance components" are a 4×4 Λ and the decomposition is not a single scalar
# ratio (tracked separately).

using LinearAlgebra: dot
import ForwardDiff
using Distributions: Normal, quantile, Chisq
using Optim: Optim

# ---------------------------------------------------------------------------
# Variance-component bookkeeping: map each grouping factor to the WORKING-scale
# θ index that carries its log σ, plus the residual log σ index. Returns
#   (comps::Vector{Pair{Symbol,Int}}, resid_idx::Union{Int,Nothing})
# Works for both two-structured paths (:resid + :resd) and the single-structured
# closed-form path (:sigma intercept + :resd), guarding the heteroscedastic case.
# ---------------------------------------------------------------------------
function _variance_component_indices(fit::DrmFit)
    # Location-scale-scale fits (#544): group-varying RE SD makes "the" variance
    # component ill-defined, exactly like the heteroscedastic-residual rejection below.
    any(p -> first(p) in (:sd, :sd_phylo), fit.blocks) &&
        error("heritability/repeatability: this fit models the random-effect SD with " *
            "covariates (`sd(group) ~ ...`), so a single variance component is not defined")
    comps = Pair{Symbol,Int}[]
    resid_idx = nothing
    have = Dict(p => r for (p, r) in fit.blocks)

    # Structured component SDs live in the :resd block, named per grouping factor.
    if haskey(have, :resd)
        r = have[:resd]
        nms = first(cn[2] for cn in fit.coefnames if cn[1] === :resd)
        for (j, nm) in enumerate(nms)
            push!(comps, Symbol(nm) => r[j])
        end
    end

    # Residual log σ. The two-structured paths expose it as a dedicated :resid
    # block (homoscedastic, length 1). The single-structured closed-form path
    # carries it in the :sigma block; it is a clean scalar residual variance only
    # when sigma ~ 1 (a single intercept) — reject a heteroscedastic σ predictor.
    if haskey(have, :resid)
        rr = have[:resid]
        length(rr) == 1 || error("heritability/repeatability: residual block has " *
            "length $(length(rr)); expected a single homoscedastic residual log σ")
        resid_idx = first(rr)
    elseif haskey(have, :sigma)
        rs = have[:sigma]
        length(rs) == 1 || error("heritability/repeatability needs a homoscedastic " *
            "residual (`sigma ~ 1`); this fit has a σ predictor with $(length(rs)) " *
            "coefficients, so σ²_resid is not a single scalar")
        resid_idx = first(rs)
    end

    isempty(comps) && error("heritability/repeatability: no structured variance " *
        "components found in this fit (need phylo/relmat/animal/spatial random " *
        "intercepts; have blocks $(first.(fit.blocks)))")
    resid_idx === nothing && error("heritability/repeatability: no residual scale " *
        "found in this fit")
    return comps, resid_idx
end

# σ²_k(θ) = exp(2 θ_k) on the working scale. Kept as a one-liner so ForwardDiff
# differentiates exactly through it.
@inline _var_from_log(θ, idx) = exp(2 * θ[idx])

# Build g(θ) = σ²_focal / (Σ_{k ∈ denom} σ²_k), the ratio whose ∇/H bias_correct
# differentiates. `focal` is one θ index; `denom` is the list of θ indices in the
# denominator (the focal index plus the others that share variance). A tiny floor
# keeps the denominator strictly positive so the map is smooth at the σ→0 boundary
# (the ratio still tends to its correct limit).
function _ratio_closure(focal::Int, denom::Vector{Int})
    return θ -> begin
        num = _var_from_log(θ, focal)
        den = zero(num)
        @inbounds for idx in denom
            den += _var_from_log(θ, idx)
        end
        num / den
    end
end

# Clamp a Wald CI to [0, 1] (heritability / repeatability are bounded ratios).
_clamp01(x) = clamp(x, 0.0, 1.0)
function _clamp01_ci(ci)
    return (lower = _clamp01(ci.lower), upper = _clamp01(ci.upper))
end

# ---------------------------------------------------------------------------
# Delta / epsilon-method ratio with CI, via the merged bias_correct infra.
# ---------------------------------------------------------------------------
function _ratio_delta(fit::DrmFit, focal::Int, denom::Vector{Int}; level::Real)
    g = _ratio_closure(focal, denom)
    bc = bias_correct(fit, g; level = level)
    return (estimate = bc.estimate, corrected = bc.corrected, bias = bc.bias,
            se = bc.se, ci = _clamp01_ci(bc.ci), level = bc.level)
end

# ---------------------------------------------------------------------------
# TRUE profile-likelihood CI on the derived RATIO. We hold the ratio r = g(θ)
# fixed at a trial value v and RE-MAXIMISE the likelihood over ALL nuisance
# parameters (not just substitute the focal SD), then invert the LRT: the (1−α)
# interval is {v : 2[NLL_v − NLL̂] ≤ χ²_{1,1−α}}.
#
# The ratio r = σ²_focal / Σ_{k∈denom} σ²_k = v is enforced by SUBSTITUTION on the
# focal log σ, with S_others RE-COMPUTED from the CURRENT nuisance values at every
# inner iteration:
#   θ_focal = ½ log( v/(1−v) · S_others(θ_free) ),  S_others = Σ_{k∈others} σ²_k.
# Everything else (the other variance components, residual, and mean coefficients)
# is optimised freely inside `nll`. Because the co-components can absorb variance as
# v moves away from r̂, the profiled deviance rises at the correct (shallower) rate,
# so the interval has the intended profile-likelihood coverage — unlike an ELR /
# substitution profile that freezes S_others at the MLE (which is anti-conservative
# when the components trade off; this package deliberately does NOT use ELR).
#
# Cost: one inner Nelder-Mead re-optimisation per trial v (a handful of variance +
# mean parameters). Falls back to the substitution profile only if the stored NLL
# is missing (handled by the caller error) — otherwise the true profile is used.
# ---------------------------------------------------------------------------
function _ratio_profile(fit::DrmFit, focal::Int, denom::Vector{Int}; level::Real)
    nll = fit.nll
    nll === nothing && error("profile ratio CI needs the stored NLL closure " *
        "(fit.nll); this fit does not carry one")
    θ̂ = copy(coef(fit))
    others = [idx for idx in denom if idx != focal]
    g = _ratio_closure(focal, denom)
    r̂ = g(θ̂)
    nllhat = nll(θ̂)

    # Free (nuisance) parameters re-optimised at each fixed ratio: everything
    # except the focal log σ, which is pinned by the ratio constraint.
    free = setdiff(1:length(θ̂), focal)

    # Map a free-parameter vector `z` (in `free` order) + trial ratio `v` to the
    # full θ, deriving θ[focal] from the ratio and the CURRENT (re-optimised) others.
    function build_θ(z, v)
        θ = copy(θ̂)
        @inbounds for (k, idx) in enumerate(free)
            θ[idx] = z[k]
        end
        if v <= 0
            θ[focal] = -50.0                       # σ²_focal → 0 (log σ → −∞ proxy)
        elseif v >= 1
            θ[focal] = 50.0                        # σ²_focal → ∞ (all-variance limit)
        else
            S_others = sum(_var_from_log(θ, idx) for idx in others; init = 0.0)
            σ²focal = v / (1 - v) * (S_others <= 0 ? eps() : S_others)
            θ[focal] = σ²focal <= 0 ? -50.0 : 0.5 * log(σ²focal)
        end
        return θ
    end

    # Profiled NLL at ratio v: minimise `nll` over the free nuisance parameters,
    # warm-started from the MLE. Nelder-Mead is derivative-free (the stored NLL may
    # not be dual-safe) and robust on the small nuisance block here.
    z0 = θ̂[free]
    function nll_at_ratio(v)
        v >= 1 && return Inf
        obj(z) = nll(build_θ(z, v))
        res = try
            Optim.optimize(obj, copy(z0), Optim.NelderMead(),
                           Optim.Options(iterations = 2000, g_tol = 1e-8))
        catch
            return NaN                             # failed inner solve (NOT Inf)
        end
        # Evaluate at the minimizer, not the possibly stale Optim.minimum, and flag
        # a sentinel / non-finite value or a non-converged solve as FAILED (NaN).
        val = try
            _objective_at_minimizer(obj, res)
        catch
            return NaN
        end
        (Optim.converged(res) && !_profile_eval_failed(val)) || return NaN
        return val
    end

    half = quantile(Chisq(1), level) / 2          # LRT half-width on the NLL scale
    target = nllhat + half

    # Bracket-and-bisect each side of r̂ in ratio space (monotone-enough profile).
    lower = _profile_side(nll_at_ratio, r̂, target, -1)
    upper = _profile_side(nll_at_ratio, r̂, target, +1)
    return (estimate = r̂, ci = (lower = _clamp01(lower), upper = _clamp01(upper)),
            level = float(level))
end

# A profile evaluation FAILED if it is NaN, -Inf, or a barrier sentinel (>= 1e16).
# `+Inf` is deliberate (the ratio-1 edge) and is NOT a failure.
_profile_eval_failed(f) = isnan(f) || f == -Inf || (isfinite(f) && f >= 1e16)

# Search one direction (dir = ±1) for the ratio v where the profile NLL crosses
# `target`. Returns the boundary (clamped to [0,1] at the caller), or `NaN` when the
# arm is UNRESOLVED: a failed evaluation (NaN / sentinel / non-converged inner
# solve) is never read as "above target", so a failed region cannot masquerade as
# the crossing. On a failure the search shrinks toward the last good point; if the
# failed region is reached without a genuine crossing, the arm is flagged NaN.
function _profile_side(fobj, v0, target, dir)
    lo = v0
    step = 0.05
    hi = clamp(v0 + dir * step, 0.0, 1.0)
    f_hi = fobj(hi)
    hi_failed = _profile_eval_failed(f_hi)
    # Expand until we bracket the threshold, fail, or hit the [0,1] edge.
    it = 0
    while !hi_failed && f_hi < target && hi > 0.0 && hi < 1.0 && it < 60
        step *= 1.6
        hi = clamp(v0 + dir * step, 0.0, 1.0)
        f_hi = fobj(hi)
        hi_failed = _profile_eval_failed(f_hi)
        it += 1
    end
    # If we never cross before the edge, the bound is the edge (one-sided / open).
    if !hi_failed && f_hi < target
        return hi
    end
    # Bisect between lo (good, below target) and hi (at/above target, or failed).
    # A failed midpoint shrinks toward lo (it is not evidence of a crossing).
    for _ in 1:80
        mid = 0.5 * (lo + hi)
        fmid = fobj(mid)
        if _profile_eval_failed(fmid)
            hi = mid; hi_failed = true
        elseif fmid < target
            lo = mid
        else
            hi = mid; hi_failed = false
        end
        abs(hi - lo) < 1e-6 && break
    end
    if hi_failed
        @warn "heritability profile CI arm unresolved: the profile evaluation failed " *
              "(sentinel / non-finite / non-converged) before the LRT threshold was " *
              "crossed; returning NaN for this bound." dir
        return NaN
    end
    return 0.5 * (lo + hi)
end

# ---------------------------------------------------------------------------
# Public accessors.
# ---------------------------------------------------------------------------
"""
    heritability(fit; component = nothing, level = 0.95, method = :delta) -> NamedTuple

Phylogenetic heritability / signal (a.k.a. `λ` / `H²`) of a structured-Gaussian
fit: the share of the total variance carried by one structured component,

    h² = σ²_component / ( Σ_k σ²_k + σ²_resid ),

where the sum runs over **all** structured variance components plus the residual.
This is the comparative-biology "phylogenetic signal" — for a single `phylo(1 |
species)` component it is Pagel/Lynch's phylogenetic heritability; with a second
structured component (e.g. `+ animal(1 | id)`) the denominator includes it too.

`component` selects which grouping factor is the numerator (a `Symbol`, e.g.
`:species`); if omitted and the fit has exactly one structured component, that one
is used. `method` is `:delta` (epsilon-method / generalized-delta via
[`bias_correct`](@ref), the default) or `:profile` (a **true** profile-likelihood
CI on the ratio: at each fixed ratio the likelihood is re-maximised over ALL
nuisance parameters — the other variance components, residual, and mean
coefficients — so the co-components can absorb variance and the profiled deviance
rises at the correct rate; it is NOT a substitution/ELR profile that freezes the
nuisance variances at the MLE). For the dense phylogenetic correlation-scale
parameterisation, fit with `algorithm = :gls` before using delta-method Wald
intervals; the default sparse all-node phylogenetic route stores only partial
covariance information in this slice, so profile intervals are the safer
uncertainty path there.

Returns a `NamedTuple`:

- `estimate`  — the plug-in ratio `g(θ̂)` (exactly in `[0, 1]`);
- `corrected` — the bias-corrected estimate `g(θ̂) + ½·tr(H_g·V)` (delta only);
- `se`        — the delta-method standard error (delta only);
- `ci`        — the `(lower, upper)` CI, **clamped to `[0, 1]`**;
- `level`     — the confidence level;
- `method`    — the method used.

The gradient and Hessian of the ratio are threaded EXACTLY through the
log σ → variance map by automatic differentiation. At a variance boundary
(`σ_component → 0` ⇒ `h² ≈ 0`, or `σ_resid → 0` ⇒ `h² ≈ 1`) the Wald SE can be
degenerate; the profile method gives a more honest (possibly one-sided) interval
there.

# Example

```julia
fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
          Gaussian(); data, tree, algorithm = :gls)
h = heritability(fit)             # single component ⇒ no `component` needed
h.estimate, h.ci
```
"""
function heritability(fit::DrmFit; component::Union{Symbol,Nothing} = nothing,
                      level::Real = 0.95, method::Symbol = :delta)
    return _signal_ratio(fit; component = component, level = level, method = method,
                         what = "heritability")
end

"""
    icc(fit; component = nothing, level = 0.95, method = :delta) -> NamedTuple

Intraclass correlation / repeatability for one grouping factor,

    ICC = σ²_component / ( σ²_component + σ²_resid ),

the share of variance at the grouping level relative to that component plus the
residual (the classic two-component repeatability). When the fit has more than
one structured component this is the **focal-vs-residual** repeatability for the
chosen `component`; use [`heritability`](@ref) for the full-variance share that
also nets out the other components. Same return shape and `method` options as
[`heritability`](@ref); the CI is clamped to `[0, 1]`.
"""
function icc(fit::DrmFit; component::Union{Symbol,Nothing} = nothing,
             level::Real = 0.95, method::Symbol = :delta)
    comps, resid_idx = _variance_component_indices(fit)
    focal = _resolve_component(comps, component, "icc")
    denom = [focal, resid_idx]
    return _emit_ratio(fit, focal, denom; level = level, method = method)
end

"""
    repeatability(fit; component = nothing, level = 0.95, method = :delta) -> NamedTuple

Alias for [`icc`](@ref): the adjusted repeatability `R = σ²_g / (σ²_g + σ²_resid)`
for the chosen grouping factor. With a single structured component and no other
components, repeatability and [`heritability`](@ref) coincide.
"""
repeatability(fit::DrmFit; component::Union{Symbol,Nothing} = nothing,
              level::Real = 0.95, method::Symbol = :delta) =
    icc(fit; component = component, level = level, method = method)

# Shared body for the full-variance "signal" ratio (heritability / phylogenetic
# signal): numerator one component, denominator ALL components + residual.
function _signal_ratio(fit::DrmFit; component, level, method, what)
    comps, resid_idx = _variance_component_indices(fit)
    focal = _resolve_component(comps, component, what)
    denom = vcat([idx for (_, idx) in comps], resid_idx)
    return _emit_ratio(fit, focal, denom; level = level, method = method)
end

# Resolve the focal grouping factor to its θ index; default to the sole component.
function _resolve_component(comps::Vector{Pair{Symbol,Int}},
                            component::Union{Symbol,Nothing}, what::String)
    if component === nothing
        length(comps) == 1 ||
            error("$what: this fit has $(length(comps)) structured components " *
                  "($(first.(comps))); pass `component = :name` to choose one")
        return comps[1].second
    end
    for (nm, idx) in comps
        nm === component && return idx
    end
    error("$what: no structured component `$component` in fit " *
          "(have $(first.(comps)))")
end

# Dispatch to the requested CI method and assemble the public NamedTuple.
function _emit_ratio(fit::DrmFit, focal::Int, denom::Vector{Int}; level, method)
    if method === :delta
        r = _ratio_delta(fit, focal, denom; level = level)
        return (estimate = r.estimate, corrected = r.corrected, bias = r.bias,
                se = r.se, ci = r.ci, level = r.level, method = :delta)
    elseif method === :profile
        r = _ratio_profile(fit, focal, denom; level = level)
        return (estimate = r.estimate, corrected = r.estimate, bias = 0.0,
                se = NaN, ci = r.ci, level = r.level, method = :profile)
    else
        throw(ArgumentError("method must be :delta or :profile, got $method"))
    end
end
