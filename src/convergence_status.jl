# Twin contract with drmTMB (FX3b; drmTMB #1554, PR #1564): the three-valued
# `convergence_status`. The numeric cuts and their order MUST stay identical to
# drmTMB `R/check.R` (`drm_twin_convergence_status_from_parts`) and its
# `docs/design/278-twin-convergence-status.md`:
#
#   1  optimiser gate: `fit.converged`, finite loglik strictly above -1e15  else :degenerate
#   2  geometry: Poisson / NB2 cone (`DrmFit.geometry`), Gaussian all-sigma
#      collapse (`_nondegenerate_fit`)                                      :degenerate
#   3  covariance square, finite, positive definite                         else :degenerate
#   4  every diagonal SE finite                                             else :degenerate
#   5  inflated SE: SE >= 50 and SE >= 1000 x median(SEs < 50)              :degenerate
#   6  design rank: an aliased fixed-effect column (#1110) is caught in step 2
#      via `_nondegenerate_fit` (`_design_rank_deficient`), as :degenerate
#   7  any finite |rho12| > 0.98                                            :boundary
#   8  any finite random-effect SD < 1e-4                                   :boundary
#   9  otherwise                                                            :converged
#
# Steps 3-5 are not applied when standard errors were turned off
# (`DrmFit.uncertainty === :skipped`), as drmTMB does for
# `drm_control(se = FALSE)`: a skipped covariance is not a failed one.

const _TWIN_INFLATED_SE_FLOOR = 50.0
const _TWIN_INFLATED_SE_RATIO = 1000.0
const _TWIN_RE_SD_BOUNDARY = 1e-4
const _TWIN_RHO12_BOUNDARY = 0.98
const _TWIN_SENTINEL_LOGLIK = -1e15

# drmTMB `drm_standard_errors_from_vcov`: sqrt of each finite, non-negative
# variance; NaN for any other diagonal entry.
function _twin_standard_errors(V::AbstractMatrix)
    size(V, 1) == 0 && return Float64[]
    return [isfinite(v) && v >= 0 ? sqrt(v) : NaN for v in diag(V)]
end

# drmTMB `drm_twin_vcov_positive_definite`: every entry finite and the smallest
# eigenvalue of the symmetric matrix strictly positive. R's
# `eigen(symmetric = TRUE)` reads the lower triangle, hence `:L`.
function _twin_vcov_positive_definite(V::AbstractMatrix)
    (size(V, 1) == 0 || size(V, 1) != size(V, 2)) && return false
    all(isfinite, V) || return false
    ev = eigvals(Symmetric(Matrix{Float64}(V), :L))
    return all(isfinite, ev) && minimum(ev) > 0
end

# drmTMB `drm_twin_inflated_se_is_true`. The reference is the median of the
# finite SEs below the floor; when every finite SE is at or above the floor it
# is the smallest finite SE.
function _twin_inflated_se(se::AbstractVector)
    finite_se = filter(isfinite, se)
    isempty(finite_se) && return false
    reference = filter(s -> s < _TWIN_INFLATED_SE_FLOOR, finite_se)
    median_se = isempty(reference) ? minimum(finite_se) : median(reference)
    return any(s -> isfinite(s) && s >= _TWIN_INFLATED_SE_FLOOR &&
                    s >= _TWIN_INFLATED_SE_RATIO * median_se, se)
end

# The values drmTMB reads for a Julia-engine fit: `sdpars` is the exponentiated
# `resd_*` block (plus the decoded `recov` pair), which is `re_sd(fit)`, and the
# residual rho12 is `corpairs`. A fit `re_sd` refuses (an `sd(g) ~ x` model)
# contributes no SD, as it contributes no `resd_*` block to drmTMB.
function _twin_re_sd_values(fit::DrmFit)
    d = try
        re_sd(fit)
    catch err
        err isa ArgumentError || rethrow()
        return Float64[]
    end
    return collect(Float64, values(d))
end

function _twin_rho12_values(fit::DrmFit)
    r = corpairs(fit)
    return r isa AbstractVector ? collect(Float64, r) : Float64[]
end

"""
    convergence_status(fit) -> Symbol

Classify a fit as `:converged`, `:boundary` or `:degenerate`, by the same
numeric rules as drmTMB's `convergence_status()` (FX3b; drmTMB
`docs/design/278-twin-convergence-status.md`). The first matching rule wins:

1. `:degenerate` unless the optimiser stopped normally (`fit.converged`) at a
   finite log-likelihood strictly above the `-1e15` failed-objective sentinel.
2. `:degenerate` when a geometry screen fired: a Poisson or NegBinomial2
   fixed-effect MLE that does not exist (#1109), or a Gaussian residual scale
   collapsed onto the data.
3. `:degenerate` unless `vcov(fit)` is square, finite and positive definite.
4. `:degenerate` unless every standard error is finite.
5. `:degenerate` when a standard error is at least 50 **and** at least 1000
   times the median of the standard errors below 50 (the smallest standard
   error when none is below 50): a near-flat, weakly identified direction.
6. `:boundary` when a fitted residual correlation has `|rho12| > 0.98`, or a
   fitted random-effect SD ([`re_sd`](@ref)) is below `1e-4`.
7. `:converged` otherwise.

Rules 3 to 5 are skipped when standard errors were turned off (`se = false`, or
`q4_vcov = false`), as drmTMB skips them for `drm_control(se = FALSE)`: such a
fit is `:converged` unless rules 1 or 2 apply. A fit with an aliased fixed-effect
column (#1110) is `:degenerate` through rule 2, as the design-rank rule of drmTMB.

[`is_converged`](@ref) and [`converged`](@ref) are `true` only for
`:converged`. The raw optimiser flag is `fit.converged`.
"""
function convergence_status(fit::DrmFit)
    if !fit.converged || !isfinite(fit.loglik) || fit.loglik <= _TWIN_SENTINEL_LOGLIK
        return :degenerate
    end
    (fit.geometry === :ok && _nondegenerate_fit(fit)) || return :degenerate
    if fit.uncertainty !== :skipped
        se = _twin_standard_errors(fit.vcov)
        if !_twin_vcov_positive_definite(fit.vcov) || isempty(se) ||
           !all(isfinite, se) || _twin_inflated_se(se)
            return :degenerate
        end
    end
    at_boundary =
        any(s -> isfinite(s) && s < _TWIN_RE_SD_BOUNDARY, _twin_re_sd_values(fit)) ||
        any(r -> isfinite(r) && abs(r) > _TWIN_RHO12_BOUNDARY, _twin_rho12_values(fit))
    return at_boundary ? :boundary : :converged
end

"""
    is_converged(fit) -> Bool

`true` exactly when [`convergence_status`](@ref)`(fit) === :converged`, as in
drmTMB. A `false` here means the reported estimates or standard errors should
not be taken as an interior, identified optimum.

This is stricter than the raw `fit.converged` flag, which only says that the
optimiser's own test was met. The Gaussian log-likelihood is unbounded as the
residual scale goes to zero: with one row per group a structured random effect
can interpolate the data, so `sigma` collapses and the objective runs away.
A one-row-per-species phylogenetic fit produced `sd_phylo = 22980`,
`sigma = 7.5e-15`, `loglik = 6.8e13` with `fit.converged = true`, and 25% of
parametric-bootstrap replicates in that example reached such a point. Since
FX3b, a fit with a variance component or residual correlation at its boundary
(`:boundary`) is also `false` here.
"""
is_converged(fit::DrmFit) = convergence_status(fit) === :converged

"""
    converged(fit) -> Bool

The same predicate as [`is_converged`](@ref), under the name drmTMB also
exports. The raw optimiser flag is `fit.converged`.
"""
converged(fit) = is_converged(fit)   # untyped: the joint missing-data wrappers forward `is_converged`

# The optimiser flag AND the degeneracy bar: `is_converged` before FX3b. Kept for
# the callers where a `:boundary` optimum is still a usable fit (bootstrap and
# profile refits, frozen margins, heritability, the bridge's `converged` key).
_converged_nondegenerate(fit::DrmFit) = fit.converged && _nondegenerate_fit(fit)
