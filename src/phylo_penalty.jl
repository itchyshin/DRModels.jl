# phylo_penalty.jl — penalized (MAP) phylogenetic variance components, the Julia
# twin of drmTMB's `drm_phylo_penalty()` / `drm_phylo_penalty_sweep()`
# (drmTMB `R/penalty.R`, `src/drmTMB.cpp:93`).
#
# THIS CHANGES THE OBJECTIVE. A penalty is added to the negative log-likelihood,
# so the estimator stops being ML and becomes maximum-a-posteriori. That is why
# this is engine capability and not a post-fit accessor: every affected route's
# objective AND its analytic gradient have to carry the extra term.
#
# Two terms, matching drmTMB's C++ exactly:
#
#   SD   (per phylogenetic SD, PC-prior style)
#       pen += rate*sd - log(sd) - log(rate)
#   which is the negative log Exponential(rate) density on the SD scale plus the
#   log-Jacobian of the log_sd -> sd transform. Since the routes optimise on
#   v = log(sd), d(pen)/dv = rate*exp(v) - 1.
#
#   COR  (only when `cor_sd` is given and the route HAS a correlation parameter)
#       pen += 0.5*z^2/s^2 + log(s) + 0.5*log(2pi),   z = atanh(cor)
#   the full negative log Normal(0, s) density, normalising constant included, on
#   the UNCONSTRAINED correlation.
#
# THE PARAMETERISATION TRAP. drmTMB penalises `eta_cor_phylo`, its unconstrained
# correlation. DRModels.jl's coupled route optimises `L21`, a Cholesky off-diagonal.
# Penalising `L21` directly would be a DIFFERENT PRIOR, not a port. So the
# correlation is recovered first --- cor = L21 / sqrt(L21^2 + L22^2) --- and
# atanh(cor) is penalised. That stays closed-form differentiable, so the affected
# routes keep their analytic gradients and no finite differencing creeps in.
#
# `loglik` on the returned fit stays the UNPENALIZED data log-likelihood, exactly
# as drmTMB keeps `fit$logLik` unpenalized and reports `fit$phylo_penalty`
# separately. The reported vcov, by contrast, IS the penalized curvature --- that
# is what a MAP fit should report, and it is why drmTMB warns that the standard
# errors are credible-interval-shaped rather than frequentist.

"""
    PhyloPenalty

A penalty specification for phylogenetic variance components — drmTMB's
`drm_phylo_penalty()` object. Fields `sd_u`, `sd_alpha`, `rate`, `cor_sd`.

`rate` is derived, not supplied: `rate = -log(sd_alpha) / sd_u`, so that a priori
`P(sd > sd_u) = sd_alpha`.
"""
struct PhyloPenalty
    sd_u::Float64
    sd_alpha::Float64
    rate::Float64
    cor_sd::Union{Nothing,Float64}
end

"""
    PhyloCorPenaltyNeedsTwoSD(msg)

Raised when `cor_sd` is requested on a model that has no phylogenetic
correlation parameter to penalize. It is a distinct type, not a bare
`ArgumentError`, because [`drm_phylo_penalty_sweep`](@ref) must catch precisely
this condition — a `cor_sd` sweep over a model with no correlation would return
identical rows and look like a prior-sensitivity check while being a no-op.

Mirrors drmTMB's classed condition `drm_phylo_cor_penalty_needs_two_sd`.
"""
struct PhyloCorPenaltyNeedsTwoSD <: Exception
    msg::String
end
Base.showerror(io::IO, e::PhyloCorPenaltyNeedsTwoSD) = print(io, e.msg)

"""
    drm_phylo_penalty(; sd_u = 1.0, sd_alpha = 0.05, cor_sd = nothing)

A PC-prior-style penalty on phylogenetic variance components — the Julia twin of
drmTMB's `drm_phylo_penalty()`. Passing it to [`drm`](@ref) as `penalty = …`
turns the fit into a **MAP** estimate.

- `sd_u`, `sd_alpha`: an exponential prior on each phylogenetic SD, calibrated so
  that a priori `P(sd > sd_u) = sd_alpha`. Smaller `sd_u` shrinks harder.
- `cor_sd`: optional SD of a mean-zero normal prior on the *unconstrained*
  (Fisher-z) phylogenetic correlation. Only meaningful for a route that actually
  estimates a correlation — the coupled mean↔σ block. Requesting it elsewhere
  raises [`PhyloCorPenaltyNeedsTwoSD`](@ref) rather than silently doing nothing.

The penalized fit reports an **unpenalized** `loglik`; the penalty value is on
`fit.phylo_penalty`. Standard errors come from the penalized curvature and are
credible-interval-shaped — do not read them as frequentist, and do not compare
penalized fits with `aic` / `lrtest` (both refuse).

!!! warning "`sd_u` is measured on YOUR tree's scale"
    `sd_u` is a threshold on the phylogenetic SD, and that SD is only meaningful
    relative to the tree's scale. DRModels.jl builds its phylogenetic covariance from
    the **branch lengths as supplied**, so a tree of height `h` has tip variance
    `h` and the fitted SD carries a factor `sqrt(h)`. drmTMB instead standardises
    via `ape::vcv(tree, corr = TRUE)`, whose tips always have variance 1.

    So the same `sd_u` is the **same prior on both sides only when the tree has
    unit height**. Copying `sd_u = 1` from an R script onto a raw tree of height
    2 gives a prior roughly `sqrt(2)` tighter than intended. Rescale first:

    ```julia
    # ape: tree\$edge.length <- tree\$edge.length / max(diag(vcv(tree)))
    ```

    Measured: on an `ape::rcoal` tree of height 1.5285 the two SDs differed by
    exactly `sqrt(1.5285) = 1.2363` while the log-likelihoods agreed to five
    decimals — the same fit, two scales. See
    `tools/parity_phylo_penalty.R`.

# Example
```julia
pen = drm_phylo_penalty(sd_u = 0.5, sd_alpha = 0.05)
fit = drm(bf(mu = @formula(y ~ x + phylo(1 | species)), sigma = @formula(~1)),
          Gaussian(); data = dat, tree = tree, penalty = pen)
fit.phylo_penalty        # the penalty at the optimum
loglik(fit)              # UNPENALIZED data log-likelihood
```
"""
function drm_phylo_penalty(; sd_u::Real = 1.0, sd_alpha::Real = 0.05,
                           cor_sd::Union{Nothing,Real} = nothing)
    (isfinite(sd_u) && sd_u > 0) ||
        throw(ArgumentError("drm_phylo_penalty: `sd_u` must be a single positive number (got $sd_u)"))
    (isfinite(sd_alpha) && 0 < sd_alpha < 1) ||
        throw(ArgumentError("drm_phylo_penalty: `sd_alpha` must be a single number in (0, 1) (got $sd_alpha)"))
    if cor_sd !== nothing
        (isfinite(cor_sd) && cor_sd > 0) ||
            throw(ArgumentError("drm_phylo_penalty: `cor_sd` must be a single positive number or nothing (got $cor_sd)"))
    end
    rate = -log(Float64(sd_alpha)) / Float64(sd_u)
    return PhyloPenalty(Float64(sd_u), Float64(sd_alpha), rate,
                        cor_sd === nothing ? nothing : Float64(cor_sd))
end

# ---------------------------------------------------------------------------
# The two penalty terms, each returning (value, derivative).
# ---------------------------------------------------------------------------

# SD term on the LOG-SD scale: the routes all optimise v = log(sd).
#   value = rate*exp(v) - v - log(rate)
#   dv    = rate*exp(v) - 1
@inline function _phylo_pen_sd(rate::Float64, v::Real)
    sd = exp(v)
    return (rate * sd - v - log(rate), rate * sd - 1.0)
end

# COR term on the UNCONSTRAINED (Fisher-z) scale.
#   value = 0.5*(z/s)^2 + log(s) + 0.5*log(2pi)
#   dz    = z/s^2
@inline function _phylo_pen_cor(s::Float64, z::Real)
    return (0.5 * (z / s)^2 + log(s) + 0.5 * log(2π), z / s^2)
end

const _PHYLO_PEN_NO_COR_MSG =
    "penalty: `cor_sd` needs a phylogenetic correlation parameter to penalize, " *
    "and this model has none. Only the coupled mean↔σ block (`phylo_coupled = true`, " *
    "with matching `phylo(1 | g)` on both `mu` and `sigma`) estimates one. " *
    "Drop `cor_sd` to penalize the SDs alone."

_phylo_pen_refuse_cor() = throw(PhyloCorPenaltyNeedsTwoSD(_PHYLO_PEN_NO_COR_MSG))

# ---------------------------------------------------------------------------
# Route appliers. Each takes the route's parameter vector, adds the penalty to
# the objective value, and adds the matching derivatives into the gradient.
# `nothing` penalty is a hard no-op so a plain ML fit stays bit-identical --- the
# same guarantee drmTMB gives by always emitting the DATA fields with
# `penalize_phylo = 0L`.
# ---------------------------------------------------------------------------

# Single phylogenetic SD held at index `idx` on the log scale. Covers the
# mean-only sparse route (v = [log sd_resid, log sd_phylo]) and the asymmetric
# sigma-phylo block (theta = [beta_mu; beta_psi; logL22]).
function _phylo_pen_apply_single!(g, pen::PhyloPenalty, v, idx::Int)
    pen.cor_sd === nothing || _phylo_pen_refuse_cor()
    val, dv = _phylo_pen_sd(pen.rate, v[idx])
    g === nothing || (g[idx] += dv)
    return val
end

# Two INDEPENDENT phylogenetic SDs (the separate block, theta = [...; logL11; logL22]).
# There is no correlation parameter in this block --- it is constrained to zero ---
# so `cor_sd` is refused rather than quietly ignored.
function _phylo_pen_apply_separate!(g, pen::PhyloPenalty, θ, i1::Int, i2::Int)
    pen.cor_sd === nothing || _phylo_pen_refuse_cor()
    v1, d1 = _phylo_pen_sd(pen.rate, θ[i1])
    v2, d2 = _phylo_pen_sd(pen.rate, θ[i2])
    if g !== nothing
        g[i1] += d1
        g[i2] += d2
    end
    return v1 + v2
end

# The coupled block: lambda = theta[i0:i0+2] = [logL11, L21, logL22], and
# Lambda = L*L' is a COVARIANCE, so
#     sd_mu  = L11 = exp(logL11)
#     sd_psi = sqrt(L21^2 + L22^2) =: r
#     cor    = L21 / r
# The SD penalty applies to log(sd_mu) and log(r); the correlation penalty to
# z = atanh(cor). Both derivative chains are closed form:
#     dr/dL21      = L21/r          dr/dlogL22   = L22^2/r
#     dz/dL21      = 1/r            dz/dlogL22   = -L21/r
function _phylo_pen_apply_coupled!(g, pen::PhyloPenalty, θ, i0::Int)
    logL11 = θ[i0]
    a      = θ[i0 + 1]              # L21
    b      = exp(θ[i0 + 2])         # L22
    r      = sqrt(a * a + b * b)    # = sd_psi

    # --- SD on the mean axis (its own coordinate, no chain rule) ---
    val, d11 = _phylo_pen_sd(pen.rate, logL11)
    g === nothing || (g[i0] += d11)

    # --- SD on the sigma axis: penalise log(r), then push through r ---
    vr, dvr = _phylo_pen_sd(pen.rate, log(r))     # dvr = d(pen)/d(log r)
    val += vr
    dpen_dr = dvr / r                              # d(pen)/dr
    if g !== nothing
        g[i0 + 1] += dpen_dr * (a / r)             # dr/dL21
        g[i0 + 2] += dpen_dr * (b * b / r)         # dr/dlogL22
    end

    # --- correlation, only when asked for ---
    if pen.cor_sd !== nothing
        z = atanh(clamp(a / r, -0.999999999, 0.999999999))
        vc, dz = _phylo_pen_cor(pen.cor_sd, z)
        val += vc
        if g !== nothing
            g[i0 + 1] += dz * (1.0 / r)            # dz/dL21
            g[i0 + 2] += dz * (-a / r)             # dz/dlogL22
        end
    end
    return val
end

# ---------------------------------------------------------------------------
# The cor_sd sweep — drmTMB's `drm_phylo_penalty_sweep()`.
# ---------------------------------------------------------------------------

"""
    drm_phylo_penalty_sweep(f, fam; data, cor_sd = [0.25, 0.5, 1.0],
                            sd_u = 1.0, sd_alpha = 0.05,
                            phylo_coupled = true, kwargs...)

Prior-sensitivity sweep over the phylogenetic-correlation penalty — the Julia
twin of drmTMB's `drm_phylo_penalty_sweep()`. Refits the model once per `cor_sd`
value and reports whether the conclusion moves as the prior tightens.

Returns `(; summary, fits)`. `summary` is a `Vector{NamedTuple}` with fields
`cor_sd`, `converged`, `vcov_posdef`, `loglik`, `cor`, and `error` — a failed fit
contributes a row of `NaN`/`missing` plus its message rather than aborting the
sweep. `fits` is a `Dict` keyed `"cor_sd=<value>"`.

`phylo_coupled = true` is the default because only the coupled mean↔σ block
estimates a phylogenetic correlation at all; sweeping `cor_sd` over any other
block would produce identical rows and *look* like a sensitivity check while
being a no-op. That case raises [`PhyloCorPenaltyNeedsTwoSD`](@ref) from a probe
fit before any of the sweep runs, so the no-op is reported rather than sold.

Extra `kwargs...` are forwarded to [`drm`](@ref) (`tree`, `g_tol`, …).

# Example
```julia
sw = drm_phylo_penalty_sweep(
        bf(mu = @formula(y ~ x + phylo(1 | species)),
           sigma = @formula(~ 1 + phylo(1 | species))),
        Gaussian(); data = dat, tree = tree, cor_sd = [0.25, 0.5, 1.0])
sw.summary          # one row per cor_sd
```
"""
function drm_phylo_penalty_sweep(f, fam; data,
                                 cor_sd = [0.25, 0.5, 1.0],
                                 sd_u::Real = 1.0, sd_alpha::Real = 0.05,
                                 phylo_coupled::Bool = true,
                                 kwargs...)
    cs = collect(cor_sd)
    (!isempty(cs) && all(x -> isa(x, Real) && isfinite(x) && x > 0, cs)) ||
        throw(ArgumentError("drm_phylo_penalty_sweep: `cor_sd` must be a non-empty vector of " *
                            "positive finite numbers (got $cor_sd)"))

    # Probe first. A sweep whose every row is identical is worse than an error: it
    # reads as a passed sensitivity check. drmTMB probes for exactly this reason.
    probe = drm_phylo_penalty(sd_u = sd_u, sd_alpha = sd_alpha, cor_sd = cs[1])
    try
        drm(f, fam; data = data, phylo_coupled = phylo_coupled, penalty = probe, kwargs...)
    catch e
        e isa PhyloCorPenaltyNeedsTwoSD && throw(PhyloCorPenaltyNeedsTwoSD(
            "drm_phylo_penalty_sweep: this model has no phylogenetic correlation to penalize, " *
            "so every `cor_sd` would return an IDENTICAL row — a sweep here would look like a " *
            "prior-sensitivity check while being a no-op. " * _PHYLO_PEN_NO_COR_MSG))
        rethrow()
    end

    summary = NamedTuple[]
    fits = Dict{String,Any}()
    for s in cs
        pen = drm_phylo_penalty(sd_u = sd_u, sd_alpha = sd_alpha, cor_sd = s)
        key = "cor_sd=$(s)"
        try
            fit = drm(f, fam; data = data, phylo_coupled = phylo_coupled, penalty = pen, kwargs...)
            fits[key] = fit
            V = fit.vcov
            pd = all(isfinite, V) && isposdef(Symmetric(V))
            ρ = haskey(fit.scales, :lambda_cor) ? fit.scales[:lambda_cor][1] : NaN
            push!(summary, (cor_sd = Float64(s), converged = fit.converged, vcov_posdef = pd,
                            loglik = loglik(fit), cor = ρ, error = nothing))
        catch e
            fits[key] = e
            push!(summary, (cor_sd = Float64(s), converged = false, vcov_posdef = false,
                            loglik = NaN, cor = NaN, error = sprint(showerror, e)))
        end
    end
    return (; summary, fits)
end
