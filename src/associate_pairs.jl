# associate_pairs.jl — staged (frozen-margin) association, the Julia twin of
# drmTMB's `associate_pairs()` / `latent_normal()` / `association()`.
#
# THIS IS A TWO-STAGE ESTIMATOR, NOT A JOINT MODEL. Two univariate models are
# fitted independently, then FROZEN, and a single latent-normal copula
# association is estimated conditional on them. The margins are never refitted.
#
# The consequence has to be stated wherever the uncertainty is reported: because
# the margins are treated as exact, the association's standard error is
# CONDITIONAL — it ignores margin estimation error. drmTMB bounds its own version
# the same way (alpha-scale Wald only, an experimental-interval warning, and no
# simultaneous eta bands or profiles), and this port must not offer intervals the
# twin refuses.
#
# First slice: the `gaussian_bernoulli` pair class, whose likelihood is CLOSED
# FORM. The Gaussian margin's latent is observed exactly, so the Bernoulli's
# contribution is a conditional univariate normal CDF — no rectangle probability,
# no quadrature, no new dependency. The four pair classes that censor BOTH
# latents need a 1-D adaptive integral with a retained error estimate and are
# deliberately deferred (design note
# `docs/dev-log/design/2026-08-15-a3c-design-staged-association.md`).

using Distributions: Normal, cdf, logcdf, logccdf, logpdf, pdf, quantile
import Distributions
import QuadGK

"""
    LatentNormal

Association kernel for [`associate_pairs`](@ref) — drmTMB's `latent_normal()`.
The two margins are coupled through a bivariate standard normal latent with
correlation `eta`.
"""
struct LatentNormal end

"""
    latent_normal()

The latent-normal association kernel. Must be passed explicitly: like drmTMB,
there is no implicit association model.
"""
latent_normal() = LatentNormal()

"""
    PairAssociation

Result of [`associate_pairs`](@ref): a staged, frozen-margin association fit.

Fields: `pair_class`, `coefficients` (association scale, `alpha`), `eta`
(`0.999999·tanh(alpha)`), `loglik`, `score` and `curvature` (finite-difference
diagnostics at the optimum), `near_boundary`, `multistart_disagreement`, and
`nobs`.

**The uncertainty is conditional on the frozen margins.** See
[`association`](@ref).
"""
struct PairAssociation
    pair_class::Symbol
    coefficients::Vector{Float64}
    coefnames::Vector{String}
    eta::Float64
    loglik::Float64
    score::Vector{Float64}
    curvature::Vector{Float64}
    near_boundary::Bool
    multistart_disagreement::Bool
    nobs::Int
    components::Any          # frozen margins, kept for integration diagnostics
end

const _ASSOC_ETA_GUARD = 0.999999
const _ASSOC_BOUND = 8.0

_assoc_eta(alpha) = _ASSOC_ETA_GUARD * tanh(alpha)

"""
    associate_pairs(fit_1, fit_2; kernel, association = nothing, marginal = :LA)

Estimate a staged latent-normal association between two **already-fitted**
univariate models — drmTMB's `associate_pairs()`.

`kernel` must be given explicitly as [`latent_normal()`](@ref). `association`
defaults to an intercept (`~ 1`); only the intercept-only form is implemented in
this slice, matching drmTMB's boundary for every pair class except its one
Bernoulli × NB2 exception.

The margins are **frozen**: `fit_1` and `fit_2` are used as fitted and are never
re-estimated. Reported uncertainty is therefore conditional on them.

Implemented pair class: **gaussian × binomial** (`gaussian_bernoulli`). The other
four reviewed classes need a rectangle probability and are not implemented yet;
they error rather than silently approximating.

```julia
fg = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = d)
fb = drm(bf(@formula(z ~ x)), Binomial(); data = d)
a  = associate_pairs(fg, fb; kernel = latent_normal())
a.eta
```
"""
function associate_pairs(fit_1::DrmFit, fit_2::DrmFit; kernel = nothing,
                         association = nothing, marginal::Symbol = :LA)
    # QuadGK rectangle integration (when used) is not Liu–Pierce AGHQ. Reject
    # the keyword rather than silently relabelling (#448).
    if Symbol(uppercase(String(marginal))) === :AGHQ
        throw(ArgumentError(
            "associate_pairs: `marginal = :AGHQ` is not available (#448). " *
            "Staged association uses a closed form or QuadGK rectangle integration — " *
            "neither is 1-D Liu–Pierce AGHQ. Omit `marginal` or pass `:LA`."))
    end
    kernel isa LatentNormal ||
        throw(ArgumentError("associate_pairs: supply an explicit " *
            "`kernel = latent_normal()` declaration — there is no implicit " *
            "association kernel."))
    association === nothing || _assoc_intercept_only(association)
    _assoc_require_converged(fit_1, "fit_1")
    _assoc_require_converged(fit_2, "fit_2")

    pc, comps = _assoc_components(fit_1, fit_2)
    loglik, n = _assoc_loglik_for(pc, comps)

    # Multistart on a bounded scalar: the profile is not assumed unimodal, and
    # drmTMB starts from -1, 0, 1 for the intercept-only case.
    starts = (-1.0, 0.0, 1.0)
    objective(a) = (v = loglik(a); isfinite(v) ? -v : prevfloat(Inf))
    best_a, best_o = 0.0, Inf
    objectives = Float64[]
    for s in starts
        a = _assoc_optimise_scalar(objective, s)
        o = objective(a)
        push!(objectives, o)
        if o < best_o
            best_o, best_a = o, a
        end
    end

    # Finite-difference score + curvature, exactly as drmTMB: no Hessian, and no
    # report at all when the optimum sits on the box boundary.
    h = 1e-4
    if best_a - h <= -_ASSOC_BOUND || best_a + h >= _ASSOC_BOUND
        score, curv = NaN, NaN
    else
        lo, hi = objective(best_a - h), objective(best_a + h)
        score = (lo - hi) / (2h)
        curv = -(hi - 2 * objective(best_a) + lo) / h^2
    end

    η = _assoc_eta(best_a)
    tol = 1e-7 * (1 + abs(best_o))
    finite = filter(isfinite, objectives)
    disagree = length(finite) < 2 || (maximum(finite) - minimum(finite)) > tol

    return PairAssociation(pc, [best_a], ["(Intercept)"], η, -best_o,
                           [score], [curv], abs(η) >= 0.995, disagree, n, comps)
end

# One dispatch point: two closed-form classes, three rectangle classes.
function _assoc_loglik_for(pc::Symbol, c)
    pc === :gaussian_bernoulli &&
        return (alpha -> _assoc_loglik_gaussian_bernoulli(alpha, c)), length(c.gaussian_y)
    pc === :gaussian_nbinom2 &&
        return (alpha -> _assoc_loglik_gaussian_nbinom2(alpha, c)), length(c.gaussian_y)
    pc in (:bernoulli_bernoulli, :bernoulli_nbinom2, :nbinom2_nbinom2) &&
        return (alpha -> _assoc_loglik_rectangle(alpha, c)), length(c.e1.lower)
    throw(ArgumentError("associate_pairs: no likelihood for pair class `$pc`."))
end

# Bounded scalar optimisation by golden-section on [-8, 8]: the association scale
# is one bounded coordinate, so this needs no gradient and cannot leave the box
# (mirroring drmTMB's `nlminb` bounds).
function _assoc_optimise_scalar(objective, start::Float64)
    lo, hi = -_ASSOC_BOUND, _ASSOC_BOUND
    # Bracket around the start so a multistart actually explores different basins.
    width = 4.0
    lo = max(lo, start - width)
    hi = min(hi, start + width)
    invphi = (sqrt(5.0) - 1) / 2
    c = hi - invphi * (hi - lo)
    d = lo + invphi * (hi - lo)
    fc, fd = objective(c), objective(d)
    for _ in 1:200
        if fc < fd
            hi, d, fd = d, c, fc
            c = hi - invphi * (hi - lo)
            fc = objective(c)
        else
            lo, c, fc = c, d, fd
            d = lo + invphi * (hi - lo)
            fd = objective(d)
        end
        (hi - lo) < 1e-10 && break
    end
    return (lo + hi) / 2
end

# Only `~ 1` in this slice. drmTMB allows a varying association formula for one
# pair class only (literal-Bernoulli × ordinary-NB2), which is not implemented
# here — so reject rather than silently fit an intercept.
function _assoc_intercept_only(association)
    ok = association isa FormulaTerm && association.rhs isa ConstantTerm
    ok || throw(ArgumentError("associate_pairs: only an intercept-only " *
        "`association = @formula(association ~ 1)` is implemented. drmTMB admits a " *
        "varying association formula for literal-Bernoulli × ordinary-NB2 only; " *
        "that cell is not ported yet."))
    return association
end

# ── Latent-interval machinery for DISCRETE margins ───────────────────────────
#
# A discrete margin does not pin its latent to a point; it censors it to an
# interval. For a count `y` with CDF F, the latent normal lies in
# `[Φ⁻¹(F(y−1)), Φ⁻¹(F(y))]` — the standard PIT representation. Everything is
# carried in LOG space, and the tail is chosen per row, because
# `log(exp(a) − exp(b))` loses all its digits when the two are close.

"""
    _assoc_logdiffexp(a, b)

`log(exp(a) − exp(b))` for `a ≥ b`, computed without forming either exponential.
Returns `-Inf` when the two are indistinguishable rather than a negative
argument to `log`.
"""
function _assoc_logdiffexp(a::Real, b::Real)
    (isfinite(a) && a > b) || return -Inf
    d = b - a
    d < -700 && return a                    # exp(d) underflows; the difference is a
    e = -expm1(d)                           # 1 − exp(b−a), accurate as d → 0⁻
    e > 0 || return -Inf
    return a + log(e)
end

# Latent interval endpoints (z scale) for an NB2 count margin.
# drmTMB's convention, shared with DRModels.jl: size = 1/sigma^2.
function _assoc_nb2_endpoints(y::AbstractVector, mu::AbstractVector,
                              sigma::AbstractVector)
    n = length(y)
    lower = Vector{Float64}(undef, n)
    upper = Vector{Float64}(undef, n)
    Z = Normal()
    @inbounds for i in 1:n
        r = 1 / (sigma[i]^2)                       # NB2 size = 1/sigma^2
        p = r / (r + mu[i])                        # Distributions' success prob
        d = Distributions.NegativeBinomial(r, p)
        yi = y[i]
        upper[i] = quantile(Z, clamp(cdf(d, yi), 0.0, 1.0))
        lower[i] = yi > 0 ? quantile(Z, clamp(cdf(d, yi - 1), 0.0, 1.0)) : -Inf
    end
    return (lower = lower, upper = upper)
end

# Latent interval endpoints for a literal Bernoulli margin: y = 1 ⇒ z above the
# threshold, y = 0 ⇒ below it.
function _assoc_bernoulli_endpoints(y::AbstractVector, p::AbstractVector)
    n = length(y)
    lower = Vector{Float64}(undef, n)
    upper = Vector{Float64}(undef, n)
    Z = Normal()
    @inbounds for i in 1:n
        thr = quantile(Z, 1 - p[i])                # qnorm(p, lower = FALSE)
        if y[i] == 1
            lower[i], upper[i] = thr, Inf
        else
            lower[i], upper[i] = -Inf, thr
        end
    end
    return (lower = lower, upper = upper)
end

# log P(z2 ∈ [lo, hi] | z1) for a bivariate standard normal with correlation eta,
# where z1 is OBSERVED. Closed form — a difference of conditional normal CDFs,
# with the tail chosen to avoid catastrophic cancellation (drmTMB's branch rule).
function _assoc_cond_interval_logprob(lo::Float64, hi::Float64, z1::Float64,
                                      η::Float64, s::Float64)
    a = (lo - η * z1) / s
    b = (hi - η * z1) / s
    Z = Normal()
    if b <= 0                       # both in the left tail
        return _assoc_logdiffexp(logcdf(Z, b), logcdf(Z, a))
    elseif a >= 0                   # both in the right tail — use survival
        return _assoc_logdiffexp(logccdf(Z, a), logccdf(Z, b))
    else                            # straddles 0; plain lower tail is accurate
        return _assoc_logdiffexp(logcdf(Z, b), logcdf(Z, a))
    end
end

# Rectangle probability for the BOTH-CENSORED classes, reduced from a 2-D
# integral to a 1-D adaptive one exactly as drmTMB does:
#
#     P = ∫ φ(z₁) · P(z₂ ∈ [lo₂, hi₂] | z₁) dz₁   over z₁ ∈ [lo₁, hi₁]
#
# QuadGK returns (value, abs_error); the error is KEPT so per-row integration
# quality is reportable rather than assumed.
function _assoc_rectangle_prob(lo1::Float64, hi1::Float64, lo2::Float64,
                               hi2::Float64, η::Float64)
    s = sqrt(1 - η * η)
    s > 0 || return (value = 0.0, abs_error = Inf)
    Z = Normal()
    f = z1 -> pdf(Z, z1) * exp(_assoc_cond_interval_logprob(lo2, hi2, z1, η, s))
    val, err = QuadGK.quadgk(f, lo1, hi1; rtol = 1e-10, maxevals = 10^5)
    return (value = val, abs_error = err)
end

# Freeze the margins and pull out exactly what the pair likelihood consumes.
function _assoc_components(fit_1::DrmFit, fit_2::DrmFit)
    g_idx = fit_1.family isa Gaussian ? 1 : (fit_2.family isa Gaussian ? 2 : 0)
    b_idx = fit_1.family isa Binomial ? 1 : (fit_2.family isa Binomial ? 2 : 0)
    n_idx = fit_1.family isa NegBinomial2 ? 1 : (fit_2.family isa NegBinomial2 ? 2 : 0)

    # gaussian × nbinom2 — closed form, like gaussian_bernoulli: the Gaussian
    # latent is observed, so the count's contribution is a conditional interval
    # probability, NOT a rectangle.
    if g_idx != 0 && n_idx != 0 && g_idx != n_idx
        gfit = g_idx == 1 ? fit_1 : fit_2
        nfit = n_idx == 1 ? fit_1 : fit_2
        gy, gmu, gsd = _assoc_gaussian_parts(gfit)
        ny, nmu, nsig = _assoc_nb2_parts(nfit)
        _assoc_same_rows(length(gy), length(ny))
        return :gaussian_nbinom2,
               (gaussian_y = gy, gaussian_mu = gmu, gaussian_sigma = gsd,
                nb_endpoints = _assoc_nb2_endpoints(ny, nmu, nsig))
    end

    # bernoulli × nbinom2 — BOTH censored ⇒ rectangle.
    if b_idx != 0 && n_idx != 0 && b_idx != n_idx
        bfit = b_idx == 1 ? fit_1 : fit_2
        nfit = n_idx == 1 ? fit_1 : fit_2
        by, bp = _assoc_bernoulli_parts(bfit)
        ny, nmu, nsig = _assoc_nb2_parts(nfit)
        _assoc_same_rows(length(by), length(ny))
        return :bernoulli_nbinom2,
               (e1 = _assoc_bernoulli_endpoints(by, bp),
                e2 = _assoc_nb2_endpoints(ny, nmu, nsig))
    end

    # nbinom2 × nbinom2 — BOTH censored ⇒ rectangle.
    if fit_1.family isa NegBinomial2 && fit_2.family isa NegBinomial2
        y1, mu1, s1 = _assoc_nb2_parts(fit_1)
        y2, mu2, s2 = _assoc_nb2_parts(fit_2)
        _assoc_same_rows(length(y1), length(y2))
        return :nbinom2_nbinom2,
               (e1 = _assoc_nb2_endpoints(y1, mu1, s1),
                e2 = _assoc_nb2_endpoints(y2, mu2, s2))
    end

    # bernoulli × bernoulli — BOTH censored ⇒ rectangle.
    if fit_1.family isa Binomial && fit_2.family isa Binomial
        by1, bp1 = _assoc_bernoulli_parts(fit_1)
        by2, bp2 = _assoc_bernoulli_parts(fit_2)
        _assoc_same_rows(length(by1), length(by2))
        return :bernoulli_bernoulli,
               (e1 = _assoc_bernoulli_endpoints(by1, bp1),
                e2 = _assoc_bernoulli_endpoints(by2, bp2))
    end

    (g_idx != 0 && b_idx != 0 && g_idx != b_idx) ||
        throw(ArgumentError("associate_pairs: unreviewed pair class. drmTMB admits " *
            "exactly five: gaussian×binomial, gaussian×nbinom2, binomial×nbinom2, " *
            "binomial×binomial and nbinom2×nbinom2. Anything else needs its own " *
            "Arc 6 review upstream before it can be ported."))

    gfit = g_idx == 1 ? fit_1 : fit_2
    bfit = b_idx == 1 ? fit_1 : fit_2

    gy = gfit.obs[:mu]
    gmu = gfit.means[:mu]
    gsd = gfit.scales[:sigma]
    gsd = length(gsd) == 1 ? fill(gsd[1], length(gy)) : gsd
    bp = bfit.means[:mu]                    # fitted success probability
    by = bfit.obs[:mu]                      # observed proportion; Bernoulli ⇒ 0/1

    length(gy) == length(bp) ||
        throw(ArgumentError("associate_pairs: the two fits must share the same rows " *
            "(got $(length(gy)) and $(length(bp)))."))
    all(v -> v == 0 || v == 1, by) ||
        throw(ArgumentError("associate_pairs: the binomial margin must be literal " *
            "Bernoulli (0/1 responses, one trial per row) for the " *
            "`gaussian_bernoulli` class."))

    return :gaussian_bernoulli,
           (gaussian_y = Vector{Float64}(gy), gaussian_mu = Vector{Float64}(gmu),
            gaussian_sigma = Vector{Float64}(gsd), binary_y = Vector{Float64}(by),
            binary_p = Vector{Float64}(bp))
end

# A staged estimator FREEZES its margins, so a margin that did not converge is
# not a smaller problem than usual — it is silently load-bearing. Its fitted
# values define the latent intervals, and a collapsed dispersion (say sigma driven
# to the boundary) narrows those intervals and biases the association with no
# visible failure anywhere downstream. Refuse rather than freeze it.
#
# Found by the A3c-3 parity harness: DRModels.jl's NB2 fitter failed to converge on a
# legitimately overdispersed sample (sigma-hat 5.6e-7, logLik -3200.8 vs drmTMB's
# -2909.5), and the association came back attenuated -- 0.42 against drmTMB's
# 0.58 -- with nothing in the result to indicate why.
function _assoc_require_converged(f::DrmFit, name::AbstractString)
    is_converged(f) && return nothing
    throw(ArgumentError("associate_pairs: `$name` did not converge, so it cannot be " *
        "frozen as a margin. A staged association conditions on the margin as if it " *
        "were exact: a non-converged fit (e.g. a dispersion collapsed to its " *
        "boundary) would bias the association with no visible failure. Refit the " *
        "margin first, or use `engine = \"tmb\"` for that margin."))
end

# ── shared margin extractors ─────────────────────────────────────────────────

function _assoc_same_rows(n1::Int, n2::Int)
    n1 == n2 || throw(ArgumentError("associate_pairs: the two fits must share the " *
        "same rows (got $n1 and $n2)."))
    return nothing
end

function _assoc_gaussian_parts(f::DrmFit)
    y = Vector{Float64}(f.obs[:mu])
    mu = Vector{Float64}(f.means[:mu])
    sd = f.scales[:sigma]
    sd = length(sd) == 1 ? fill(Float64(sd[1]), length(y)) : Vector{Float64}(sd)
    return y, mu, sd
end

function _assoc_bernoulli_parts(f::DrmFit)
    p = Vector{Float64}(f.means[:mu])
    y = Vector{Float64}(f.obs[:mu])
    all(v -> v == 0 || v == 1, y) ||
        throw(ArgumentError("associate_pairs: a binomial margin must be literal " *
            "Bernoulli (0/1 responses, one trial per row) for the staged classes."))
    return y, p
end

function _assoc_nb2_parts(f::DrmFit)
    y = Vector{Float64}(f.obs[:mu])
    mu = Vector{Float64}(f.means[:mu])
    sg = f.scales[:sigma]
    sg = length(sg) == 1 ? fill(Float64(sg[1]), length(y)) : Vector{Float64}(sg)
    all(v -> v >= 0 && v == floor(v), y) ||
        throw(ArgumentError("associate_pairs: an nbinom2 margin must be ordinary " *
            "non-negative integer counts."))
    return y, mu, sg
end

# ── pair likelihoods ─────────────────────────────────────────────────────────

# Closed form. The Gaussian latent is OBSERVED (z), so the Bernoulli's
# contribution is the conditional normal CDF at the shifted threshold — no
# bivariate CDF and no quadrature is required for this pair class.
function _assoc_loglik_gaussian_bernoulli(alpha, c)
    η = _assoc_eta(alpha)
    s = sqrt(1 - η * η)
    s > 0 || return -Inf
    Z = Normal()
    total = 0.0
    @inbounds for i in eachindex(c.gaussian_y)
        z = (c.gaussian_y[i] - c.gaussian_mu[i]) / c.gaussian_sigma[i]
        thr = quantile(Z, 1 - c.binary_p[i])          # qnorm(p, lower = FALSE)
        cz = (thr - η * z) / s
        lb = c.binary_y[i] == 1 ? logcdf(Z, -cz) : logcdf(Z, cz)
        total += logpdf(Normal(c.gaussian_mu[i], c.gaussian_sigma[i]), c.gaussian_y[i]) + lb
    end
    return total
end

# Closed form too — and this one corrects the A3c design note, which assumed all
# four remaining classes needed quadrature. The Gaussian latent is observed, so
# the count's censored latent contributes a conditional INTERVAL probability: a
# difference of conditional normal CDFs, tail chosen per row.
function _assoc_loglik_gaussian_nbinom2(alpha, c)
    η = _assoc_eta(alpha)
    s = sqrt(1 - η * η)
    s > 0 || return -Inf
    total = 0.0
    lo, hi = c.nb_endpoints.lower, c.nb_endpoints.upper
    @inbounds for i in eachindex(c.gaussian_y)
        z = (c.gaussian_y[i] - c.gaussian_mu[i]) / c.gaussian_sigma[i]
        lp = _assoc_cond_interval_logprob(lo[i], hi[i], z, η, s)
        isfinite(lp) || return -Inf
        total += logpdf(Normal(c.gaussian_mu[i], c.gaussian_sigma[i]), c.gaussian_y[i]) + lp
    end
    return total
end

# The three BOTH-CENSORED classes share one likelihood: a rectangle probability
# per row, via the 1-D adaptive integral. Identical code for
# bernoulli×bernoulli, bernoulli×nbinom2 and nbinom2×nbinom2 — only the endpoint
# construction differs, and that already happened in `_assoc_components`.
function _assoc_loglik_rectangle(alpha, c)
    η = _assoc_eta(alpha)
    abs(η) < 1 || return -Inf
    total = 0.0
    l1, u1 = c.e1.lower, c.e1.upper
    l2, u2 = c.e2.lower, c.e2.upper
    @inbounds for i in eachindex(l1)
        r = _assoc_rectangle_prob(l1[i], u1[i], l2[i], u2[i], η)
        (isfinite(r.value) && r.value > 0) || return -Inf
        total += log(r.value)
    end
    return total
end

"""
    integration_diagnostics(a::PairAssociation)

Per-row quadrature quality for a both-censored staged fit, or `nothing` for the
closed-form classes.

Returns the rectangle probability and QuadGK's absolute error estimate per row,
plus the worst relative error. drmTMB retains `abs.error` from its own adaptive
integration for exactly this reason: a rectangle probability that silently lost
precision would corrupt the association without any visible failure.
"""
function integration_diagnostics(a::PairAssociation)
    a.components === nothing && return nothing
    c = a.components
    (haskey(c, :e1) && haskey(c, :e2)) || return nothing
    η = a.eta
    n = length(c.e1.lower)
    val = Vector{Float64}(undef, n)
    err = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        r = _assoc_rectangle_prob(c.e1.lower[i], c.e1.upper[i],
                                  c.e2.lower[i], c.e2.upper[i], η)
        val[i], err[i] = r.value, r.abs_error
    end
    rel = [v > 0 ? e / v : Inf for (v, e) in zip(val, err)]
    return (probability = val, abs_error = err, relative_error = rel,
            worst_relative_error = maximum(rel))
end

"""
    association(a::PairAssociation)

Association summary for a staged fit — drmTMB's `association()`.

Returns a `NamedTuple` with `eta` (the latent correlation), `alpha` (the
association-scale coefficient), `score`/`curvature` diagnostics,
`near_boundary`, and `multistart_disagreement`.

**The uncertainty is conditional on the frozen margins** — it ignores margin
estimation error, so it is not a joint standard error. Matching drmTMB, no
simultaneous `eta` bands or profile intervals are offered.
"""
function association(a::PairAssociation)
    return (pair_class = a.pair_class, eta = a.eta, alpha = a.coefficients[1],
            score = a.score[1], curvature = a.curvature[1],
            near_boundary = a.near_boundary,
            multistart_disagreement = a.multistart_disagreement,
            nobs = a.nobs, loglik = a.loglik,
            uncertainty = "conditional on frozen margins; experimental")
end
