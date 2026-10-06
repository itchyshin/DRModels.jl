# bias_correct.jl — generalized-delta / epsilon-method bias correction for
# smooth nonlinear DERIVED quantities of a fitted DRModels model, in the spirit of
# TMB's `sdreport(..., bias.correct = TRUE)` (Thorson & Kristensen 2016,
# Fisheries Research 175:66–74).
#
# For a smooth scalar map g : θ ↦ g(θ) of the fitted parameters (e.g. a variance
# σ² from log σ, a correlation ρ12 from atanh, a back-transformed mean exp(η)),
# the plug-in estimate g(θ̂) is biased for E[g(θ̂)] whenever g is curved, because
# θ̂ is (approximately) Gaussian with mean θ and covariance V = vcov(fit). A
# second-order Taylor expansion of g about θ̂ gives
#
#     E[g(θ̂)] ≈ g(θ̂) + ½ · tr(H_g(θ̂) · V),
#
# where H_g is the Hessian of g. The correction term ½·tr(H_g·V) is exactly the
# epsilon-method bias correction. The delta-method standard error uses the EXACT
# gradient ∇g and the stored covariance,
#
#     se = sqrt(∇g(θ̂)ᵀ · V · ∇g(θ̂)).
#
# Both the gradient and Hessian are computed by automatic differentiation
# (ForwardDiff) of the user's g, so they are exact for any smooth g — no
# finite-difference error. The Wald-style CI is built on the working scale of g
# (estimate ± z·se); see the docstring for the honest caveats.
#
# Scope / assumptions (be honest):
#   * This is a SECOND-ORDER correction. It is exact for quadratic g and for the
#     mean/variance of a Gaussian θ̂ up to O(‖V‖²); it helps most where the
#     transform is curved — precisely the dispersion / variance / correlation
#     parameters that the Laplace approximation biases.
#   * It assumes θ̂ ≈ N(θ, V) (the usual asymptotic Wald premise) and that V is
#     the observed-information covariance the fit already exposes. At a singular
#     variance boundary V is not PD; the correction is reported but the SE/CI
#     inherit the boundary behaviour of `vcov`/`stderror` (bootstrap/profile are
#     preferable there — see `confint`).
#   * It is a point-and-SE tool for DERIVED quantities; it is deliberately
#     distinct from the raw plug-in accessors (`coef`, `sigma`, `rho12`, …),
#     which report g(θ̂) without the correction.

using LinearAlgebra: dot
import ForwardDiff
using Distributions: Normal, quantile

"""
    bias_correct(fit, g::Function; level = 0.95) -> NamedTuple

Epsilon-method (generalized-delta) bias correction for a smooth scalar derived
quantity `g(θ)` of a fitted model, à la TMB `sdreport(..., bias.correct = TRUE)`
(Thorson & Kristensen 2016).

`g` receives the full coefficient vector `θ = coef(fit)` (working scale: μ on the
response scale, σ on `log σ`, ρ12 on `atanh ρ12`, random-effect SDs on `log σ_b`)
and returns a real scalar.

Returns a `NamedTuple` with

- `estimate`  — the raw plug-in value `g(θ̂)`;
- `corrected` — the bias-corrected value `g(θ̂) + ½·tr(H_g·V)`;
- `bias`      — the correction itself, `½·tr(H_g·V)` (so `corrected = estimate + bias`);
- `se`        — the delta-method standard error `√(∇gᵀ V ∇g)` using the EXACT
                gradient and the stored covariance `V = vcov(fit)`;
- `ci`        — the `(lower, upper)` Wald interval `corrected ± z·se` at `level`;
- `level`     — the confidence level used.

# Method

With `θ̂ ≈ N(θ, V)`, a second-order Taylor expansion of `g` about `θ̂` gives the
expectation `E[g(θ̂)] ≈ g(θ̂) + ½·tr(H_g(θ̂)·V)`, and the first-order delta method
gives the variance `∇g(θ̂)ᵀ V ∇g(θ̂)`. The gradient `∇g` and Hessian `H_g` are
obtained by automatic differentiation of `g`, so they are exact for any smooth
`g`.

# Anchors / what to trust

- **Linear `g` (identity anchor):** for `g(θ) = θ_k` (or any affine map) `H_g = 0`,
  so `corrected == estimate` and `se`, `ci` equal the plug-in Wald values exactly.
- **Curved `g` (curvature anchor):** for `g = exp` on a coordinate with Wald mean
  `m` and variance `v`, `corrected = exp(m)(1 + v/2)`, the second-order expansion
  of the analytic `E[exp(θ_k)] = exp(m + v/2)` (they agree to `O(v²)`).

# Caveats

This is a *second-order* correction under the asymptotic premise `θ̂ ≈ N(θ, V)`;
it does not fix non-normality of `θ̂` or a non-PD `V` at a variance boundary
(there, prefer `bootstrap_ci` / profile `confint`). It is kept distinct from the
raw plug-in accessors on purpose.

# Example

```julia
fit = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian(); data = (; y, x))
k   = only(findall(==(:sigma), first.(fit.blocks)))   # σ block
ki  = fit.blocks[k].second[1]                          # log σ intercept index
bc  = bias_correct(fit, θ -> exp(θ[ki]))               # back-transformed σ
bc.estimate, bc.corrected, bc.se, bc.ci
```
"""
function bias_correct(fit::DrmFit, g::Function; level::Real = 0.95)
    θ̂ = coef(fit)
    V = vcov(fit)
    return _bias_correct(g, θ̂, V; level = level)
end

"""
    bias_correct(θ̂::AbstractVector, V::AbstractMatrix, g::Function; level = 0.95)

Lower-level form operating directly on a point estimate `θ̂` and its covariance
`V` (e.g. for testing against closed forms, or for derived quantities not tied to
a `DrmFit`). Same return shape as the `DrmFit` method.
"""
function bias_correct(θ̂::AbstractVector, V::AbstractMatrix, g::Function;
                      level::Real = 0.95)
    return _bias_correct(g, θ̂, V; level = level)
end

function _bias_correct(g, θ̂::AbstractVector, V::AbstractMatrix; level::Real)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1), got $level"))
    θ = collect(float.(θ̂))
    n = length(θ)
    size(V) == (n, n) ||
        throw(DimensionMismatch("vcov is $(size(V)) but θ has length $n"))

    estimate = float(g(θ))

    # Exact gradient and Hessian of g by automatic differentiation.
    grad = ForwardDiff.gradient(g, θ)
    H = ForwardDiff.hessian(g, θ)

    # Delta-method variance: ∇gᵀ V ∇g. A valid ZERO variance (e.g. a constant g,
    # or a zero gradient) is a degenerate-but-well-defined point: se = 0 ⇒ the CI
    # collapses to the corrected value. Only a non-finite or NEGATIVE variance
    # (numerically broken / non-PD direction) is undefined ⇒ Inf (unbounded CI).
    var = dot(grad, V * grad)
    se = (!isfinite(var) || var < 0) ? Inf : sqrt(var)

    # Epsilon-method bias correction: ½·tr(H_g · V) = ½·Σ_ij H_ij V_ij.
    bias = 0.0
    @inbounds for j in 1:n, i in 1:n
        bias += H[i, j] * V[i, j]
    end
    bias *= 0.5

    corrected = estimate + bias
    z = quantile(Normal(), 1 - (1 - level) / 2)
    ci = (lower = corrected - z * se, upper = corrected + z * se)

    return (estimate = estimate, corrected = corrected, bias = bias,
            se = se, ci = ci, level = float(level))
end
