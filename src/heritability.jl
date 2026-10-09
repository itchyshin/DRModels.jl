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
#   (comps::Vector{Pair{Symbol,Int}}, resid_idx::Int, omega::Vector{Int})
# Works for both two-structured paths (:resid + :resd) and the single-structured
# closed-form path (:sigma intercept + :resd), guarding the heteroscedastic case.
#
# A random intercept on the SCALE (`sigma ~ (1 | g)`) also lands in :resd, under
# `<group>_logsigma`, but its SD ω lives on the log-σ axis: it is NOT a variance
# component of the response and never enters `comps`. Its θ indices are returned
# in `omega`; the residual entry of every denominator then becomes the marginal
# residual variance E[σ²] = exp(2 b₀ + 2 Σ_k ω_k²) (see `_vc_var`), as drmTMB's
# R/heritability.R does.
# ---------------------------------------------------------------------------
function _variance_component_indices(fit::DrmFit)
    # Location-scale-scale fits (#544): group-varying RE SD makes "the" variance
    # component ill-defined, exactly like the heteroscedastic-residual rejection below.
    any(p -> first(p) in (:sd, :sd_phylo), fit.blocks) &&
        error("heritability/repeatability: this fit models the random-effect SD with " *
            "covariates (`sd(group) ~ ...`), so a single variance component is not defined. " *
            "The estimand is covariate-CONDITIONAL, R(z) = σ_b(z)² / (σ_b(z)² + σ_e(z)²) " *
            "(h²(z) = σ_a(z)² / (σ_a(z)² + σ_e(z)²) for `sd(g, phylogenetic)`): pass the " *
            "covariate values you want, e.g. `repeatability(fit, (; sex = [0.0, 1.0]))`.")
    comps = Pair{Symbol,Int}[]
    omega = Int[]
    resid_idx = nothing
    have = Dict(p => r for (p, r) in fit.blocks)

    # Structured component SDs live in the :resd block, named per grouping factor.
    # `<group>_logsigma` entries are scale-axis random intercepts (log ω), kept apart.
    if haskey(have, :resd)
        r = have[:resd]
        nms = first(cn[2] for cn in fit.coefnames if cn[1] === :resd)
        for (j, nm) in enumerate(nms)
            if _is_logsigma_re(fit, nm)
                push!(omega, r[j])
            else
                push!(comps, Symbol(nm) => r[j])
            end
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

    isempty(comps) && !isempty(omega) && error("heritability/repeatability: this fit " *
        "has a random intercept only on the scale (`sigma ~ (1 | g)`). Its SD lives on " *
        "the log σ scale and is not a variance component of the response, so there is " *
        "no mean random-effect variance to put in the ratio. Add a mean random " *
        "intercept (e.g. `y ~ x + (1 | g)`), as drmTMB requires too.")
    isempty(comps) && error("heritability/repeatability: no structured variance " *
        "components found in this fit (need phylo/relmat/animal/spatial random " *
        "intercepts; have blocks $(first.(fit.blocks)))")
    resid_idx === nothing && error("heritability/repeatability: no residual scale " *
        "found in this fit")
    return comps, resid_idx, omega
end

# Is the `:resd` entry `nm` a random intercept on log σ? The routes name it
# `<group>_logsigma` (#322), but a mean grouping column could carry that suffix
# too, so the name only counts when the `sigma` formula really has a random term
# on `<group>`. A fit without a retained formula falls back to the suffix.
function _is_logsigma_re(fit::DrmFit, nm)
    s = String(nm)
    endswith(s, "_logsigma") || return false
    f = fit.formula
    f isa DrmFormula || return true
    i = findfirst(p -> first(p) === :sigma, f.forms)
    i === nothing && return false
    grp = s[1:end-length("_logsigma")]
    rhs = replace(string(last(f.forms[i])), r"\s+" => " ")
    return occursin("| $grp)", rhs)
end

# Variance contributed by θ index `idx` to a ratio. A component's variance is
# exp(2 θ_idx). The residual entry (`idx == resid`) is exp(2 b₀) for a constant
# scale and, when the scale carries random intercepts with log SDs θ[omega], the
# marginal residual variance E[σ²] = exp(2 b₀ + 2 Σ_k exp(2 θ_ωk)) (log σ ~
# N(b₀, Σ ω²) ⇒ E[exp(2 log σ)] = exp(2 b₀ + 2 Σ ω²)).
@inline function _vc_var(θ, idx::Int, resid::Int, omega::Vector{Int})
    (idx == resid && !isempty(omega)) || return _var_from_log(θ, idx)
    s = 2 * θ[idx]
    @inbounds for w in omega
        s += 2 * exp(2 * θ[w])
    end
    return exp(s)
end

# σ²_k(θ) = exp(2 θ_k) on the working scale. Kept as a one-liner so ForwardDiff
# differentiates exactly through it.
@inline _var_from_log(θ, idx) = exp(2 * θ[idx])

# Build g(θ) = σ²_focal / (Σ_{k ∈ denom} σ²_k), the ratio whose ∇/H bias_correct
# differentiates. `focal` is one θ index; `denom` is the list of θ indices in the
# denominator (the focal index plus the others that share variance). A tiny floor
# keeps the denominator strictly positive so the map is smooth at the σ→0 boundary
# (the ratio still tends to its correct limit).
function _ratio_closure(focal::Int, denom::Vector{Int}; resid::Int = 0,
                        omega::Vector{Int} = Int[])
    return θ -> begin
        num = _var_from_log(θ, focal)
        den = zero(num)
        @inbounds for idx in denom
            den += _vc_var(θ, idx, resid, omega)
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
function _ratio_delta(fit::DrmFit, focal::Int, denom::Vector{Int}; level::Real,
                      resid::Int = 0, omega::Vector{Int} = Int[])
    g = _ratio_closure(focal, denom; resid = resid, omega = omega)
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
function _ratio_profile(fit::DrmFit, focal::Int, denom::Vector{Int}; level::Real,
                        resid::Int = 0, omega::Vector{Int} = Int[])
    nll = fit.nll
    nll === nothing && error("profile ratio CI needs the stored NLL closure " *
        "(fit.nll); this fit does not carry one")
    θ̂ = copy(coef(fit))
    others = [idx for idx in denom if idx != focal]
    g = _ratio_closure(focal, denom; resid = resid, omega = omega)
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
            S_others = sum(_vc_var(θ, idx, resid, omega) for idx in others; init = 0.0)
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

# Location–scale–scale fits

With `sd(g) ~ z` / `sd(g, phylogenetic) ~ z` and `sigma ~ z` the ratio depends on the
covariates, so this form refuses. Use `heritability(fit, newdata; level = 0.95)`
(phylogenetic `sd`) or [`repeatability`](@ref)`(fit, newdata)` (iid `sd`), which return
the covariate-conditional `R(z) = σ_b(z)² / (σ_b(z)² + σ_e(z)²)` per row of `newdata`
with a Wald-on-logit interval; see [`repeatability`](@ref) for the definition.

A random intercept on `sigma` is not a component: it turns `σ²_resid` into the
marginal residual variance `E[σ²]`; see [`repeatability`](@ref).
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

For location–scale–scale fits (`sd(g) ~ z`) this form refuses; use
`icc(fit, newdata)` — the covariate-conditional repeatability documented under
[`repeatability`](@ref).

A random intercept on `sigma` is not a component: it turns `σ²_resid` into the
marginal residual variance `E[σ²]`; see [`repeatability`](@ref).
"""
function icc(fit::DrmFit; component::Union{Symbol,Nothing} = nothing,
             level::Real = 0.95, method::Symbol = :delta)
    comps, resid_idx, omega = _variance_component_indices(fit)
    focal = _resolve_component(comps, component, "icc")
    denom = [focal, resid_idx]
    return _emit_ratio(fit, focal, denom; level = level, method = method,
                       resid = resid_idx, omega = omega)
end

"""
    repeatability(fit; component = nothing, level = 0.95, method = :delta) -> NamedTuple
    repeatability(fit, newdata; level = 0.95) -> NamedTuple

Alias for [`icc`](@ref): the adjusted repeatability `R = σ²_g / (σ²_g + σ²_resid)`
for the chosen grouping factor. With a single structured component and no other
components, repeatability and [`heritability`](@ref) coincide.

# A random intercept on the scale

A random intercept on `sigma` (`sigma ~ (1 | g)`, SD reported as
`re_sd(fit)[:g_logsigma]`) lives on the log σ scale. It is not a variance
component of the response, so it is never a `component` and never enters the
numerator. It makes the residual variance vary by group, so the residual entry of
the denominator becomes the marginal residual variance

    E[σ²] = exp(2 b₀ + 2 Σ_k ω_k²),

where `b₀` is the `sigma` intercept and `ω_k` the log-σ random-intercept SDs (not
the squared median `exp(2 b₀)`); the delta-method gradient runs through `ω_k` too.
This is drmTMB's definition. A fit whose only random effect is on `sigma` has no
mean component and is refused.

# Location–scale–scale fits: the estimand is conditional on covariates

When the between-individual SD and the residual SD both depend on covariates
(`sd(id) ~ z`, `sigma ~ z`),

    y_ij ~ N(μ_ij, σ_e(z_i)²),   b_i ~ N(0, σ_b(z_i)²),
    log σ_b(z) = α'z,            log σ_e(z) = γ'z,

repeatability is a FUNCTION of the covariates, not a number:

    R(z) = σ_b(z)² / (σ_b(z)² + σ_e(z)²) = logistic( 2 (α'z − γ'z) ),

the correlation between two observations of an individual whose covariates are `z`.
No single scalar is the repeatability of such a model — any one number is a choice of
covariate distribution (and the ratio of average variances is NOT the average of the
ratios) — so `repeatability(fit)` / `icc(fit)` / `heritability(fit)` **refuse** these
fits rather than pick one silently. Use the two-argument form: `newdata` is a
column table holding every predictor of the `sd(g)` and `sigma` formulas
(categorical predictors must contain all levels, in the training coding, so the
design matches). It returns, per row of `newdata`,

- `estimate` — `R(z)`;
- `se_logit` — the Wald SE of `logit R(z) = 2(α'z − γ'z)`, which is LINEAR in the
  coefficients, so it is exact up to the usual Wald approximation (uses the joint
  `vcov` of the `sd` and `sigma` blocks, so the covariance between them counts);
- `lower`, `upper` — the Wald interval on the logit scale mapped back to `(0, 1)`;
- `level`, `method = :wald_logit`.

```julia
fit = drm(bf(@formula(y ~ sex + (1 | id)), @formula(sigma ~ sex),
             @formula(sd(id) ~ sex)), Gaussian(); data)
repeatability(fit, (; sex = [0.0, 1.0]))   # R for females, R for males
```

`sd(g, phylogenetic) ~ z` fits are handled by the same call to
[`heritability`](@ref): `heritability(fit, newdata)` returns
`h²(z) = σ_a(z)² / (σ_a(z)² + σ_e(z)²)`, the tip-level share of variance that is
phylogenetic.
"""
repeatability(fit::DrmFit; component::Union{Symbol,Nothing} = nothing,
              level::Real = 0.95, method::Symbol = :delta) =
    icc(fit; component = component, level = level, method = method)

# Covariate-conditional ratio for location-scale-scale fits (#694). `kind` = :iid
# (`sd(g) ~ z`, block :sd, ratio = repeatability) or :phylo (`sd(g, phylogenetic) ~ z`,
# block :sd_phylo, ratio = phylogenetic h²(z)). logit R(z) = 2(η_sd − η_σ) is linear in
# the coefficients, so the Wald SE on that scale is c'Vc with c = (2 x_sd, −2 x_σ).
function _conditional_ratio(fit::DrmFit, newdata; kind::Symbol, level::Real, what::String)
    0 < level < 1 || throw(ArgumentError("$what: `level` must be in (0, 1), got $level"))
    f = fit.formula
    f isa DrmFormula || error("$what: this fit did not retain its formula")
    numblk = kind === :iid ? :sd : :sd_phylo
    have = Dict(fit.blocks)
    (haskey(have, numblk) && haskey(have, :sigma)) ||
        error("$what(fit, newdata): this fit has no `" *
              (kind === :iid ? "sd(group) ~ …" : "sd(group, phylogenetic) ~ …") *
              "` submodel, so repeatability is a single number — call `$what(fit)` instead")
    pre = kind === :iid ? "sd_" : "sdphy_"
    parts = [k => r for (k, r) in f.forms if startswith(String(k), pre)]
    length(parts) == 1 ||
        error("$what(fit, newdata): found $(length(parts)) `sd()` submodels of this kind; " *
              "the conditional ratio is implemented for exactly one")
    any(p -> first(p) in (:sd, :sd_phylo) && first(p) !== numblk, fit.blocks) &&
        error("$what(fit, newdata): multi-component location–scale–scale fits have no single " *
              "two-component ratio")
    nd = NamedTuple(pairs(newdata))
    nrows = length(first(values(nd)))
    ndr = _predict_data(nd, f.response)
    _, Xsd, _ = _design(f.response, last(parts[1]), ndr)
    fixed_sigma, _, _, _ = _split_ranef(Dict(f.forms)[:sigma])
    _, Xσ, _ = _design(f.response, fixed_sigma, ndr; schema_cache = f.schema_cache, schema_key = :sigma)
    isd, isg = have[numblk], have[:sigma]
    (size(Xsd, 2) == length(isd) && size(Xσ, 2) == length(isg)) ||
        throw(DimensionMismatch("$what(fit, newdata): `newdata` builds a design with " *
            "$(size(Xsd, 2)) `sd` and $(size(Xσ, 2)) `sigma` columns but the fit has " *
            "$(length(isd)) and $(length(isg)); a categorical predictor in `newdata` must " *
            "contain all its training levels"))
    α = fit.theta[isd]; γ = fit.theta[isg]
    V = fit.vcov
    d = 2 .* (Xsd * α .- Xσ * γ)
    est = 1 ./ (1 .+ exp.(-d))
    se = similar(d)
    for i in 1:nrows
        c = zeros(length(fit.theta))
        c[isd] .= 2 .* Xsd[i, :]
        c[isg] .= -2 .* Xσ[i, :]
        se[i] = sqrt(max(dot(c, V * c), 0.0))
    end
    z = quantile(Normal(), 1 - (1 - level) / 2)
    lo = 1 ./ (1 .+ exp.(-(d .- z .* se)))
    hi = 1 ./ (1 .+ exp.(-(d .+ z .* se)))
    return (estimate = est, se_logit = se, lower = lo, upper = hi, level = level,
            method = :wald_logit)
end

repeatability(fit::DrmFit, newdata; level::Real = 0.95) =
    _conditional_ratio(fit, newdata; kind = :iid, level = level, what = "repeatability")
icc(fit::DrmFit, newdata; level::Real = 0.95) =
    _conditional_ratio(fit, newdata; kind = :iid, level = level, what = "icc")
heritability(fit::DrmFit, newdata; level::Real = 0.95) =
    _conditional_ratio(fit, newdata; kind = :phylo, level = level, what = "heritability")

# Shared body for the full-variance "signal" ratio (heritability / phylogenetic
# signal): numerator one component, denominator ALL components + residual.
function _signal_ratio(fit::DrmFit; component, level, method, what)
    comps, resid_idx, omega = _variance_component_indices(fit)
    focal = _resolve_component(comps, component, what)
    denom = vcat([idx for (_, idx) in comps], resid_idx)
    return _emit_ratio(fit, focal, denom; level = level, method = method,
                       resid = resid_idx, omega = omega)
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
function _emit_ratio(fit::DrmFit, focal::Int, denom::Vector{Int}; level, method,
                     resid::Int = 0, omega::Vector{Int} = Int[])
    if method === :delta
        r = _ratio_delta(fit, focal, denom; level = level, resid = resid, omega = omega)
        return (estimate = r.estimate, corrected = r.corrected, bias = r.bias,
                se = r.se, ci = r.ci, level = r.level, method = :delta)
    elseif method === :profile
        r = _ratio_profile(fit, focal, denom; level = level, resid = resid, omega = omega)
        return (estimate = r.estimate, corrected = r.estimate, bias = 0.0,
                se = NaN, ci = r.ci, level = r.level, method = :profile)
    else
        throw(ArgumentError("method must be :delta or :profile, got $method"))
    end
end
