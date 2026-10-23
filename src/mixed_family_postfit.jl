# mixed_family_postfit.jl — post-fit accessors for `fit_mixed_family` (the
# cross-family bivariate shared-latent GHQ model).
#
# `fit_mixed_family` returns a plain `NamedTuple` (no `DrmFit` struct), so these
# helpers take that NamedTuple directly. They are deliberately thin: a tidy
# coefficient table, the two information criteria, per-axis fitted means, and a
# printed summary. Naming mirrors the `mf_` prefix already used for the model's
# internal kernels; the public ones are exported from `DRModels`.

"""
    _mf_nparams(fit) -> Int

Number of free parameters in a `fit_mixed_family` fit: the two fixed-effect
blocks `β1`/`β2`, the two latent loadings `λ1`/`λ2`, and the two dispersion
sub-model blocks `βσ1`/`βσ2` (empty for dispersionless axes). This equals the
length of the internal `θ` vector the optimiser minimised.
"""
_mf_nparams(fit) =
    length(fit.β1) + length(fit.β2) + 2 + length(fit.βσ1) + length(fit.βσ2)

"""
    mf_coef(fit) -> NamedTuple

Tidy table of the cross-family fit's point estimates as three equal-length
vectors `(; axis, term, estimate)`. Rows, in order:

- `β1`/`β2`: fixed-effect coefficients per axis, labelled `b1[1], b1[2], …`.
- `bσ1`/`bσ2`: dispersion sub-model coefficients (log-native scale), labelled
  `bsig1[…]` (omitted for dispersionless axes — Poisson/Binomial).
- `lambda`: the two latent loadings `λ1`/`λ2` (`axis = :shared`).
- `rho`: the latent-scale correlation `rho_latent` (`axis = :shared`).

`axis` is `:y1` / `:y2` for the per-axis rows and `:shared` for the loadings and
ρ. Estimates are on the model's native parameter scale (link scale for β,
log-native for βσ).
"""
function mf_coef(fit)
    axis = Symbol[]; term = String[]; est = Float64[]
    for (j, b) in enumerate(fit.β1)
        push!(axis, :y1); push!(term, "b1[$j]"); push!(est, b)
    end
    for (j, b) in enumerate(fit.β2)
        push!(axis, :y2); push!(term, "b2[$j]"); push!(est, b)
    end
    for (j, b) in enumerate(fit.βσ1)
        push!(axis, :y1); push!(term, "bsig1[$j]"); push!(est, b)
    end
    for (j, b) in enumerate(fit.βσ2)
        push!(axis, :y2); push!(term, "bsig2[$j]"); push!(est, b)
    end
    push!(axis, :shared); push!(term, "lambda1"); push!(est, fit.λ1)
    push!(axis, :shared); push!(term, "lambda2"); push!(est, fit.λ2)
    push!(axis, :shared); push!(term, "rho"); push!(est, fit.rho_latent)
    return (; axis, term, estimate = est)
end

"""
    mf_aic(fit) -> Float64

Akaike information criterion, `-2·loglik + 2·k`, where `k = _mf_nparams(fit)` is
the number of free parameters. `fit_mixed_family` fits by ML, so this is
directly comparable across mean/dispersion structures. Lower is better.
Returns `NaN` (with a warning) for a non-converged fit or one on the 1e10
objective sentinel (`loglik ≤ -1e9`).
"""
function mf_aic(fit)
    _mf_ic_ok(fit) || return NaN
    return -2 * fit.loglik + 2 * _mf_nparams(fit)
end

# AIC/BIC guard: a non-converged fit, or one on the 1e10 objective sentinel
# (loglik <= -1e9), has no meaningful likelihood — warn and let the caller return NaN.
function _mf_ic_ok(fit)
    ok = fit.converged && isfinite(fit.loglik) && fit.loglik > -1e9
    ok || @warn "cross-family fit is non-converged or on the objective sentinel; information criterion is NaN"
    return ok
end

"""
    mf_bic(fit; nobs) -> Float64

Bayesian (Schwarz) information criterion, `-2·loglik + k·log(nobs)`, with
`k = _mf_nparams(fit)`. The fit NamedTuple does not carry the sample size, so
`nobs` — the number of observation PAIRS `n` — is a required keyword. Lower is
better; comparable across structures (ML fit).
"""
function mf_bic(fit; nobs::Integer)
    nobs > 0 || throw(ArgumentError("nobs must be positive"))
    _mf_ic_ok(fit) || return NaN
    return -2 * fit.loglik + _mf_nparams(fit) * log(nobs)
end

"""
    mf_fitted(fit, X1, X2) -> NamedTuple

Per-axis fitted means on the RESPONSE scale, evaluated at the latent `u = 0`
(the population/marginal-mode convention): `μ_k = g_k⁻¹(X_k β_k)` with the
shared random effect set to zero, i.e. the fitted value for a "typical"
individual whose latent draw is at the mean. Returns `(; mu1, mu2)`, each a
length-`size(X_k, 1)` vector. The inverse link `g_k⁻¹` is the family's own
(identity for Gaussian, exp for Poisson/NB2/Gamma, logistic for
Binomial/Beta — see `_mf_mean`).

NOTE: this is the conditional mean at `u = 0`, NOT the marginal mean `E[y_k]`
(which for a non-identity link differs from `g_k⁻¹(Xβ)` by a Jensen term in the
shared latent). The `u = 0` convention matches the link-scale linear predictor
used everywhere else in the model.
"""
function mf_fitted(fit, X1, X2)
    size(X1, 2) == length(fit.β1) ||
        throw(DimensionMismatch("X1 has $(size(X1, 2)) columns, expected $(length(fit.β1))"))
    size(X2, 2) == length(fit.β2) ||
        throw(DimensionMismatch("X2 has $(size(X2, 2)) columns, expected $(length(fit.β2))"))
    mu1 = _mf_mean.(Ref(fit.fam1), X1 * fit.β1)
    mu2 = _mf_mean.(Ref(fit.fam2), X2 * fit.β2)
    return (; mu1, mu2)
end

"""
    mf_fitted(fit)

Fitted means for a cross-family fit produced by the **formula** route
(`drm(bf(...), (Fam1(), Fam2()); data = …)`), which carries its own design
matrices. A fit built from raw matrices via `fit_mixed_family` does not, so
there they must still be supplied: `mf_fitted(fit, X1, X2)`.

Same `u = 0` convention as the three-argument method above.
"""
function mf_fitted(fit)
    (haskey(fit, :X1) && haskey(fit, :X2)) || throw(ArgumentError(
        "mf_fitted: this fit does not carry its design matrices, so they must be " *
        "supplied: `mf_fitted(fit, X1, X2)`. Fits made through the formula route " *
        "`drm(bf(...), (Fam1(), Fam2()); data = …)` carry them and work with " *
        "`mf_fitted(fit)`."))
    return mf_fitted(fit, fit.X1, fit.X2)
end

"""
    mf_summary(fit; nobs = nothing, io = stdout)

Print a human-readable summary of a `fit_mixed_family` fit: the coefficient
table from [`mf_coef`](@ref), the latent correlation `ρ` with its available CIs
(Wald / profile / bootstrap, whichever are finite), the log-likelihood, the AIC
(and BIC when `nobs` is supplied), and the convergence flag. Returns `fit`
invisibly. Pass `nobs` (the number of observation pairs) to include BIC.
"""
function mf_summary(fit; nobs = nothing, io::IO = stdout)
    tbl = mf_coef(fit)
    println(io, "Cross-family bivariate fit (shared-latent GHQ)")
    println(io, "  converged: ", fit.converged, "   iterations: ", fit.iterations)
    println(io, "  Coefficients:")
    wterm = maximum(length.(tbl.term))
    for i in eachindex(tbl.term)
        println(io, "    ", rpad(tbl.term[i], wterm), "  (", tbl.axis[i], ")  ",
                _mf_fmt(tbl.estimate[i]))
    end
    println(io, "  Latent correlation rho = ", _mf_fmt(fit.rho_latent))
    _mf_print_ci(io, "    Wald     95% CI", fit.rho_ci_wald)
    _mf_print_ci(io, "    profile  95% CI", fit.rho_ci_profile)
    _mf_print_ci(io, "    bootstrap   95% CI", fit.rho_ci_boot)
    println(io, "  logLik = ", _mf_fmt(fit.loglik),
            "   AIC = ", _mf_fmt(mf_aic(fit)),
            nobs === nothing ? "" : "   BIC = " * _mf_fmt(mf_bic(fit; nobs = nobs)))
    return fit
end

# --- small formatting helpers (internal) ------------------------------------
_mf_fmt(x) = isfinite(x) ? string(round(x; digits = 4)) : string(x)
function _mf_print_ci(io, label, ci)
    (lo, hi) = ci
    if isfinite(lo) && isfinite(hi)
        println(io, label, ": (", _mf_fmt(lo), ", ", _mf_fmt(hi), ")")
    end
    return nothing
end
