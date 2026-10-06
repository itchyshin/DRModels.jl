# chibar.jl — chi-bar-square boundary-corrected p-values for variance-component
# likelihood-ratio tests.
#
# Testing a variance component = 0 (e.g. dropping a random effect, σ_b² = 0) is a
# BOUNDARY problem: the null value sits on the edge of the parameter space, so the
# usual χ²(df) reference distribution for the LR statistic is WRONG. Under the
# regularity conditions of Self & Liang (1987) and Stram & Lee (1994), the LR
# statistic for q variance components held at 0 follows a chi-bar-square (χ̄²)
# mixture of χ² distributions, not a single χ²(q). Using the naive χ²(q) p-value
# makes the test CONSERVATIVE (overstated p-values, too few rejections).
#
# Special cases implemented here:
#   * q = 1 — one boundary parameter: χ̄² = 0.5·χ²(0) + 0.5·χ²(1), so the tail
#     p-value is p = 0.5·P(χ²₁ > stat). (The χ²(0) atom is a point mass at 0 and
#     contributes nothing to the upper tail for stat > 0.)
#   * q = 2 — two INDEPENDENT boundary parameters: χ̄² = 0.25·χ²(0) + 0.5·χ²(1)
#     + 0.25·χ²(2), so p = 0.5·P(χ²₁ > stat) + 0.25·P(χ²₂ > stat).
#
# Assumptions (documented per `chibar_pvalue` / `lrt_boundary`):
#   - The tested parameters lie on the boundary (variances ≥ 0 tested at 0).
#   - For q = 2 the two boundary parameters are mutually INDEPENDENT (zero
#     information correlation), which gives the clean 0.25/0.5/0.25 weights. With
#     correlated components the weights depend on the information matrix and this
#     simple mixture is only approximate.
#   - All OTHER parameters are interior (regularity away from the boundary), so a
#     standard Laplace/asymptotic expansion holds for them.
#   - ML fits (REML log-likelihoods are not comparable across mean structures —
#     same caveat as `lrtest`).
#
# References:
#   Self, S.G. & Liang, K.-Y. (1987) JASA 82:605–610.
#   Stram, D.O. & Lee, J.W. (1994) Biometrics 50:1171–1177.

using Distributions: Chisq, ccdf

"""
    chibar_pvalue(stat::Real, q::Integer = 1) -> Float64

Chi-bar-square (χ̄²) boundary-corrected upper-tail p-value for a likelihood-ratio
statistic `stat = 2·(ℓ_full − ℓ_reduced)` that tests `q` **variance components =
0**. Because a variance is constrained to be non-negative, testing it at zero is a
**boundary** problem and the naive `χ²(q)` reference distribution is wrong (and
conservative); the correct null is a mixture of `χ²` distributions (Self & Liang
1987; Stram & Lee 1994).

Supported `q`:

- `q = 1` (one boundary parameter): null is `0.5·χ²(0) + 0.5·χ²(1)`, so

      p = 0.5 · P(χ²₁ > stat).

  At `stat = 0` this returns `0.5`; for `stat > 0` it is exactly half the naive
  `χ²(1)` p-value.

- `q = 2` (two **independent** boundary parameters): null is
  `0.25·χ²(0) + 0.5·χ²(1) + 0.25·χ²(2)`, so

      p = 0.5 · P(χ²₁ > stat) + 0.25 · P(χ²₂ > stat).

  At `stat = 0` this returns `0.75`.

# Assumptions
- The `q` tested parameters are variances tested at the boundary `0`.
- For `q = 2`, the two components are **independent** (uncorrelated information);
  otherwise the 0.25/0.5/0.25 weights are only approximate.
- All other parameters are interior (regularity away from the boundary).
- ML fits (as with [`lrtest`](@ref); REML likelihoods are not cross-comparable).

A negative `stat` (the reduced model fit *better* — non-nesting or non-convergence)
is clamped to `0`, returning the boundary value (`0.5` for `q = 1`, `0.75` for
`q = 2`); inspect `stat` directly in that case. Only `q ∈ (1, 2)` are supported;
other values throw an `ArgumentError`.

# Example
```julia
stat = 3.5
chibar_pvalue(stat, 1)              # 0.5 * P(χ²₁ > 3.5)
chibar_pvalue(stat, 1) ≈ 0.5 * ccdf(Distributions.Chisq(1), stat)   # true
chibar_pvalue(0.0, 1) == 0.5        # boundary point mass
chibar_pvalue(0.0, 2) == 0.75
```
"""
function chibar_pvalue(stat::Real, q::Integer = 1)
    s = max(float(stat), 0.0)               # boundary clamp (see docstring)
    if q == 1
        return 0.5 * ccdf(Chisq(1), s)
    elseif q == 2
        return 0.5 * ccdf(Chisq(1), s) + 0.25 * ccdf(Chisq(2), s)
    else
        throw(ArgumentError(
            "chibar_pvalue: only q = 1 or q = 2 boundary parameters are supported " *
            "(got q = $q). For q = 1 the mixture is 0.5·χ²(0)+0.5·χ²(1); for q = 2 " *
            "(independent components) it is 0.25·χ²(0)+0.5·χ²(1)+0.25·χ²(2)."))
    end
end

"""
    lrt_boundary(fit_full::DrmFit, fit_reduced::DrmFit; q::Integer = 1) -> NamedTuple

Boundary-corrected likelihood-ratio test for `q` **variance components = 0**,
comparing two **nested**, **ML**-fitted models. `fit_reduced` must be `fit_full`
with the `q` variance component(s) removed (e.g. dropping a random effect
`(1 | g)`). Unlike [`lrtest`](@ref) — which uses the naive `χ²(q)` reference and is
**conservative** for variance-component tests — this uses the chi-bar-square
mixture appropriate to a parameter on the boundary of its space.

Returns a `NamedTuple` `(; statistic, q, pvalue, pvalue_naive)`:

- `statistic = 2 · (loglik(fit_full) − loglik(fit_reduced))` — the LR statistic.
- `q` — the number of boundary variance components tested (the mixture order).
- `pvalue = chibar_pvalue(statistic, q)` — the χ̄² boundary p-value (see
  [`chibar_pvalue`](@ref)).
- `pvalue_naive = ccdf(Chisq(q), max(statistic, 0))` — the naive `χ²(q)` p-value,
  for comparison. Always `pvalue ≤ pvalue_naive` (the correction makes the test
  less conservative, i.e. more powerful), so reporting both is informative.

# Assumptions
Same as [`chibar_pvalue`](@ref): the `q` dropped parameters are variances tested
at `0`; for `q = 2` they are independent; all other parameters are interior; ML
fits. A negative statistic clamps to the boundary p-value.

# Example
```julia
# Random intercept vs no random effect: dropping (1 | g) removes ONE variance.
full    = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data)
reduced = drm(bf(@formula(y ~ x),           @formula(sigma ~ 1)), Gaussian(); data)

t = lrt_boundary(full, reduced; q = 1)
t.statistic        # 2·(ℓ_full − ℓ_reduced)
t.pvalue           # χ̄²: 0.5·P(χ²₁ > stat)   — correct for the boundary
t.pvalue_naive     # χ²(1): P(χ²₁ > stat)     — conservative (≈ 2× too large)
```
"""
function lrt_boundary(fit_full::DrmFit, fit_reduced::DrmFit; q::Integer = 1)
    _reml_compare_guard(fit_reduced, fit_full, "lrt_boundary")
    _sentinel_compare_guard(fit_reduced, fit_full, "lrt_boundary")
    statistic = 2 * (loglik(fit_full) - loglik(fit_reduced))
    pvalue = chibar_pvalue(statistic, q)
    pvalue_naive = ccdf(Chisq(q), max(statistic, 0.0))
    return (; statistic, q, pvalue, pvalue_naive)
end
