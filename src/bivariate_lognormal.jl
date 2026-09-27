# bivariate_lognormal.jl — bivariate lognormal model, the Julia twin of
# drmTMB's `biv_lognormal()`.
#
# Two positive responses that are jointly lognormal: log(Y) is bivariate normal.
# The likelihood is therefore CLOSED FORM — no integration, no rectangle
# probabilities, no quadrature — which makes this the easy member of the
# bivariate non-Gaussian family (Gustavsson's MLLN; see
# `docs/dev-log/design/2026-08-14-a3-rescope-bivariate-nongaussian.md`).
#
#     f_Y(y) = phi_2(log y; mu, Sigma) * prod(1 / y_k)
#     log f_Y(y) = log phi_2(log y; mu, Sigma) - sum(log y_k)
#
# The Jacobian `- sum(log y_k)` does **not** depend on any parameter, so the MLE
# and its covariance are *identical* to fitting the bivariate Gaussian model to
# `log y` — for EVERY route the Gaussian dispatcher can select, not only the
# residual one: the q=2 exact-Gaussian and q=4 sparse-Laplace PLSM routes carry
# the identity too (issue #471). Only the likelihood value shifts. That is why
# this family delegates the WHOLE fit to `drm(f, Gaussian(); data = log.(data))`
# rather than duplicating any of it: one kernel, one set of guards (`RHO_GUARD`,
# per-cell missingness, finite seeding, the phylo tree-height scale), no second
# copy to drift out of parity with the verified engine.
#
# Scale contract (drmTMB parity): `biv_lognormal()`'s dpars are
# `mu1, mu2, sigma1, sigma2, rho12` with links
# `identity, identity, log, log, atanh_guarded` — the SAME structure as
# `biv_gaussian()`. The identity links on `mu1`/`mu2` mean the location
# parameters live on the **log-response** scale, and `rho12` is the
# **log-residual** correlation, *not* the raw-scale Pearson correlation of the
# two response columns. drmTMB's vignette makes that scale part of the
# scientific interpretation; so does this docstring.

"""
    drm(f::BivariateDrmFormula, ::LogNormal; data, tree = nothing, K = nothing,
        A = nothing, coords = nothing, spatial_range = nothing, g_tol = 1e-8,
        q4_g_tol = 1e-3, q4_iterations = 300, q4_n_newton = 40, q4_vcov = true,
        method = :ML)

Fit a bivariate **lognormal** residual-correlation model — drmTMB's
`biv_lognormal()`.

Both responses must be strictly positive. `log(y1), log(y2)` are modelled as
bivariate normal, so `mu1`/`mu2` are means **on the log scale** (identity link)
and `rho12` is the **log-residual** correlation — not the Pearson correlation of
the raw responses. `sigma1`/`sigma2` are log-scale SDs (log link).

## Structured markers (`phylo`/`relmat`/`animal`/`spatial`)

`phylo(1 | group)`, `relmat(1 | group)`, `animal(1 | group)`, and
`spatial(1 | group)` markers are supported through the **same delegation**
that already gives this family its residual fit: `log(Y)` is exactly bivariate
Gaussian, so a structured call here literally runs
`drm(f, Gaussian(); data = log.(data), …)` and shifts the reported
log-likelihood by the (parameter-free) Jacobian. There is no separate
lognormal structured engine to build or verify — the q=2 exact-Gaussian route
(matching markers on `mu1`/`mu2` only) and the q=4 sparse-Laplace PLSM route
(matching markers on `mu1`, `mu2`, `sigma1`, and `sigma2`) are exactly the ones
documented under `drm(::BivariateDrmFormula, ::Gaussian)`, run on logged data.
Because the Jacobian does not depend on any parameter, `theta`/`vcov`/`ranef`
and the fitted objective's *gradient* carry over untouched; only the
log-likelihood value (and everything derived from it: aic/bic/deviance) shifts.

**Scale of a structured SD.** The group-level covariance (`fit.ranef.Sigma_a`
for q=4, the random-intercept variance for q=2) is on the **log-response
scale** for the `mu1`/`mu2` axes, and on the **log(SD of log-response) scale**
for the `sigma1`/`sigma2` axes — two log transforms deep for the scale axes,
mirroring the fixed-effect `sigma1`/`sigma2` convention above. `ranef(fit)` and
`vc(fit)` inherit this without change because they are the untouched Gaussian
output on `log(y)`.

Matching drmTMB's first slice, `method = :REML` is not implemented for this
family: the residual-only cell has no random effects to integrate out, and
extending REML to the structured cells is a later slice; use `method = :ML`
(the default).

```julia
d = (; y1 = exp.(0.4 .+ randn(200)), y2 = exp.(0.1 .+ randn(200)), x = randn(200))
fit = drm(bf(@formula(y1 ~ x), @formula(y2 ~ x),
             @formula(sigma1 ~ 1), @formula(sigma2 ~ 1), @formula(rho12 ~ 1)),
          LogNormal(); data = d)
coef(fit)      # mu1/mu2 on the log scale
corpairs(fit)  # log-residual rho12
```
"""
function drm(f::BivariateDrmFormula, fam::LogNormal; data, tree = nothing,
             K = nothing, A = nothing, coords = nothing, spatial_range = nothing,
             g_tol::Real = 1e-8, q4_g_tol::Real = 1e-3, q4_iterations::Int = 300,
             q4_n_newton::Int = 40, q4_vcov::Bool = true, method::Symbol = :ML)
    method === :ML ||
        throw(ArgumentError("drm: the bivariate lognormal route implements ML only " *
            "(got method = :$method). drmTMB's `biv_lognormal()` first slice has no " *
            "random effects to integrate out, and REML for the structured (phylo/" *
            "relmat/animal/spatial) cells is a later slice."))
    cols = NamedTuple(pairs(data))
    y1 = Vector{Float64}(getproperty(cols, f.response1))
    y2 = Vector{Float64}(getproperty(cols, f.response2))
    obs1 = _observed_response_mask(y1)
    obs2 = _observed_response_mask(y2)

    # Positivity is a modelling precondition, not a numerical accident: log(y)
    # of a non-positive observation is not a smaller likelihood, it is undefined.
    # Check only the OBSERVED cells — a missing cell in one response is legal
    # here exactly as in the Gaussian route.
    _check_positive_response(y1, obs1, f.response1)
    _check_positive_response(y2, obs2, f.response2)

    logdata = _with_logged_responses(data, f.response1, f.response2, y1, y2, obs1, obs2)
    # Delegate the WHOLE fit — residual, q=2 exact-Gaussian, or q=4 sparse-Laplace
    # PLSM — to the public Gaussian dispatcher on logged data. It alone decides
    # which of those three routes a formula's markers select
    # (`_bivariate_q4_marker`); re-parsing the marker grammar here would be
    # exactly the parallel design this family's delegation was built to avoid.
    gfit = drm(f, Gaussian(); data = logdata, tree = tree, K = K, A = A,
               coords = coords, spatial_range = spatial_range, g_tol = g_tol,
               q4_g_tol = q4_g_tol, q4_iterations = q4_iterations,
               q4_n_newton = q4_n_newton, q4_vcov = q4_vcov, method = method)
    return _lognormal_jacobian_shift(fam, gfit, y1, y2, obs1, obs2)
end

# log f_Y(y) = log phi_2(log y; .) - sum over OBSERVED cells of log y. Parameter-free,
# so theta-hat, vcov, and ranef carry over untouched from the Gaussian fit on
# log(y); only the likelihood VALUE (and everything derived from it: aic/bic/
# deviance) shifts by the Jacobian. This holds for every route the Gaussian
# dispatcher can return (residual, q=2, q=4 phylo/structured) because none of
# them make the Jacobian depend on theta — so the gradient callback (`nllgrad`)
# also carries over unchanged.
function _lognormal_jacobian_shift(fam::LogNormal, gfit::DrmFit, y1, y2, obs1, obs2)
    jac = zero(Float64)
    @inbounds for i in eachindex(y1)
        obs1[i] && (jac += log(y1[i]))
        obs2[i] && (jac += log(y2[i]))
    end
    gnll = gfit.nll
    lnll = gnll === nothing ? nothing : (θ -> gnll(θ) + jac)   # nll = -loglik, so the Jacobian ADDS
    reml_ll = isnan(gfit.reml_loglik) ? gfit.reml_loglik : gfit.reml_loglik - jac
    # `iterations` is passed through directly (rather than defaulted to -1 and
    # re-attached): this route borrows the Gaussian bivariate fit's optimiser run
    # wholesale — only the reported likelihood is Jacobian-shifted — so that run's
    # count is the honest count for THIS fit too.
    # `reml_ll` is Jacobian-shifted like loglik/ml_loglik. That matters now that
    # #470 makes REML reachable on the q=2 structured route this delegates to:
    # passing gfit.reml_loglik unshifted would report a REML value on the wrong
    # scale. NaN-safe, since reml_loglik is NaN on ML fits.
    return DrmFit(fam, gfit.blocks, gfit.coefnames, gfit.theta, gfit.vcov,
                 gfit.loglik - jac, gfit.nobs, gfit.converged, gfit.means,
                 Dict(:mu1 => Vector{Float64}(y1), :mu2 => Vector{Float64}(y2)),
                 gfit.scales, gfit.formula, lnll, gfit.nllgrad, gfit.ranef,
                 gfit.estim_method, reml_ll, gfit.ml_loglik - jac, gfit.marginal,
                 gfit.phylo_penalty, gfit.penalty, gfit.iterations, gfit.phylo_scale)
end

# Strictly-positive check on the observed cells of a bivariate lognormal response.
function _check_positive_response(y, obs, name::Symbol)
    @inbounds for i in eachindex(y)
        obs[i] || continue
        y[i] > 0 ||
            throw(ArgumentError("drm: the bivariate lognormal route requires strictly " *
                "positive responses; `$name` has a non-positive observed value at row $i " *
                "(got $(y[i])). Use `Gaussian()` on pre-logged data if a zero is a " *
                "genuine measurement rather than a lognormal draw."))
    end
    return nothing
end

# A copy of `data` with the two response columns replaced by their logs.
# Unobserved cells are carried through untouched so the Gaussian kernel's
# per-cell missingness mask still sees them as missing.
function _with_logged_responses(data, r1::Symbol, r2::Symbol, y1, y2, obs1, obs2)
    nt = NamedTuple(pairs(data))
    l1 = [obs1[i] ? log(y1[i]) : y1[i] for i in eachindex(y1)]
    l2 = [obs2[i] ? log(y2[i]) : y2[i] for i in eachindex(y2)]
    return merge(nt, NamedTuple{(r1, r2)}((l1, l2)))
end
