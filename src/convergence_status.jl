# Twin contract with drmTMB (FX3b; drmTMB #1554, PR #1564): the three-valued
# `convergence_status`. The numeric cuts and their order MUST stay identical to
# drmTMB `R/check.R` (`drm_twin_convergence_status_from_parts`) and its
# `docs/design/278-twin-convergence-status.md`:
#
#   1  optimiser gate: `fit.converged`, finite loglik strictly above -1e15  else :degenerate
#   2  geometry: Poisson / NB2 cone and fixed-effect binomial (quasi-)complete
#      separation (`DrmFit.geometry`), Gaussian all-sigma
#      collapse (`_nondegenerate_fit`)                                      :degenerate
#   3  covariance square, finite, positive definite                         else :degenerate
#   4  every diagonal SE finite                                             else :degenerate
#   5  inflated SE: with SE* = SE / sd(y) for identity-link mu coefficients (1 otherwise),
#      SE* >= 50 and SE* >= 1000 x median(SE* < 50)                          :degenerate
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

# drmTMB `drm_twin_inflated_se_is_true` (drmTMB #1781, unit-free). Each SE is first divided by
# its coefficient's reference scale `se_scale` (default 1), then the absolute floor and the
# median ratio apply, so rescaling y by c leaves the verdict unchanged. The reference is the
# median of the finite scaled SEs below the floor; when every finite scaled SE is at or above
# the floor it is the smallest finite one.
function _twin_inflated_se(se::AbstractVector, se_scale = 1.0)
    se = se ./ se_scale
    finite_se = filter(isfinite, se)
    isempty(finite_se) && return false
    reference = filter(s -> s < _TWIN_INFLATED_SE_FLOOR, finite_se)
    median_se = isempty(reference) ? minimum(finite_se) : median(reference)
    return any(s -> isfinite(s) && s >= _TWIN_INFLATED_SE_FLOOR &&
                    s >= _TWIN_INFLATED_SE_RATIO * median_se, se)
end

# drmTMB `drm_se_reference_scales`: sd(y) for the coefficients of an identity-link location
# block (`mu`, `mu1`, `mu2`; Gaussian, Student, SkewNormal), 1 for every link-scale one. The sd is
# over the finite observed responses; a non-finite or zero sd falls back to 1. Non-intercept
# `mu` slopes still carry y/x units: they are scaled by sd(y) only, as in drmTMB.
function _twin_se_reference_scales(fit::DrmFit)
    sc = ones(size(fit.vcov, 1))
    fit.family isa Union{Gaussian,Student,SkewNormal} || return sc
    for (blk, r) in fit.blocks
        blk in (:mu, :mu1, :mu2) || continue
        y = filter(isfinite, get(fit.obs, blk, Float64[]))
        length(y) > 1 || continue
        sy = std(y)
        (isfinite(sy) && sy > 0) || continue
        for k in r
            k <= length(sc) && (sc[k] = sy)
        end
    end
    return sc
end

# #1190: the sparse-Laplace routes clamp every random-effect log SD to [-8, 3] and, at the
# floor, report that coordinate with NaN variance and covariances (the Hessian of the other
# parameters is inverted, `_vcov_with_clamped_logsd_fixed`). drmTMB has no floor: the same
# collapsed SD runs on to ~1e-28 and is `:boundary` by rule 8. The Julia floor e^-8 = 3.4e-4
# can never fall under the 1e-4 cut, so a log SD within 1e-3 of the floor counts as at the
# boundary instead, and rules 3 to 5 read the covariance of the remaining coordinates.
# Returns the coordinates in a `:resd` block that sit at the floor with an all-NaN
# covariance row and column.
function _twin_floor_fixed_coords(fit::DrmFit)
    V = fit.vcov
    (size(V, 1) == size(V, 2) == length(fit.theta)) || return Int[]
    any(p -> first(p) === :resd, fit.blocks) || return Int[]
    return [k for k in _block_range(fit, :resd)
            if fit.theta[k] <= -8.0 + _CLAMP_ACTIVE_TOL &&
               all(isnan, @view(V[k, :])) && all(isnan, @view(V[:, k]))]
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

# SHINICHI-1753-1118: the converged rule shared by drmTMB, DRModels.jl, GLLVModels.jl
# and gllvmTMB. `converged` needs ALL of: the optimiser finished at a finite
# log-likelihood; a Newton step of at most 1e-3 standard errors; no parameter at a boundary; a positive
# definite Hessian with finite, non-inflated standard errors. The statuses, first match
# wins, are all warnings (never errors):
#
#   :not_converged  the optimiser did not finish, the log-likelihood is not finite (or is
#                   the failed-objective sentinel), or the Newton step in SE units exceeds
#                   1e-3, or the gradient is non-finite (tested BEFORE the boundary rule,
#                   SHINICHI-0724-1120 item 1, as drmTMB #1781)
#   :boundary       SD ~ 0, |cor| ~ 1, a Heywood case (a Gaussian residual scale collapsed
#                   onto the data) or a parameter at the end of its search interval. Tested
#                   BEFORE the Hessian, so a boundary fit never reads :degenerate
#   :degenerate     singular / non-PD Hessian, non-finite or inflated SE, an aliased design,
#                   (quasi-)complete separation, a non-existent count MLE
#   :converged      otherwise
#
# "Gradient near zero" is the Newton step in standard-error units (Shinichi, 19 Nov 2026,
# CONV-GRADIENT-CUT; the same rule as drmTMB): with g the gradient of the objective at the
# optimum and H its Hessian, V = H^-1 = `vcov(fit)` and SE_i = sqrt(V_ii),
#
#     max_i |(V g)_i| / SE_i  <=  1e-3.
#
# It is unit-free in the response, in the sample size and in the parameter scale: it says how
# far, in standard errors, one more Newton step would still move any parameter.
const _TWIN_NEWTON_STEP_TOL = 1e-3

# The Newton step in SE units, or `(value = NaN, reason)` when it cannot be scored:
#   :no_objective  the fit stores no objective
#   :reml          a REML fit stores the ML objective (its gradient there is not zero at the
#                  REML optimum), so it is not the objective `vcov` inverts: skipped
#   :no_vcov       standard errors were turned off, or `vcov` is not square, finite and
#                  positive definite over `fit.theta` (the covariance rules report that)
#   :no_gradient   no gradient could be produced
# Cost control: the gradient of a stored objective is up to 2p objective evaluations, and
# `convergence_status` is read by `show`, `coeftable` and bootstrap filters.
function _newton_step_se_units(fit::DrmFit)
    fit.nll === nothing && return (value = NaN, reason = :no_objective)
    fit.estim_method === :REML && return (value = NaN, reason = :reml)
    V = fit.vcov
    (fit.uncertainty === :skipped || size(V, 1) != size(V, 2) ||
     size(V, 1) != length(fit.theta) || !_twin_vcov_positive_definite(V)) &&
        return (value = NaN, reason = :no_vcov)
    g = _check_grad_vector(fit).grad
    g === nothing && return (value = NaN, reason = :no_gradient)
    step = Symmetric(Matrix{Float64}(V), :L) * g
    se = _twin_standard_errors(V)
    val = all(isfinite, se) && all(>(0), se) ? maximum(abs.(step) ./ se) : NaN
    return isfinite(val) ? (value = val, reason = :scored) : (value = NaN, reason = :no_gradient)
end

# The value the rule compares with the cut: a gradient that no producer could return as a
# finite vector at a fit with a finite log-likelihood is a NON-FINITE gradient, which is
# `Inf` (so `:not_converged`, SHINICHI-0724-1120 item 1, as drmTMB's `drm_newton_step_se`).
# Not scored for the other reasons (no objective, REML, no usable vcov): `NaN`.
function _newton_step_for_rule(fit::DrmFit)
    r = _newton_step_se_units(fit)
    return r.reason === :no_gradient ? Inf : r.value
end

# Boundary parameters as `(name, kind)`; kind is :re_sd, :rho12, :heywood or :bound.
function _twin_boundary_params(fit::DrmFit)
    out = NamedTuple{(:name, :kind),Tuple{String,Symbol}}[]
    haskey(fit.scales, :cond_logsd_wall) && push!(out, (name = "cond_logsd", kind = :re_sd))
    # A skipped covariance (`se = false`) is all-NaN, so every row would look like a floor row.
    for k in (fit.uncertainty === :skipped ? Int[] : _twin_floor_fixed_coords(fit))
        nm = "sd"
        for (blk, names) in fit.coefnames
            blk === :resd || continue
            r = _block_range(fit, :resd)
            k in r && (nm = string("sd:", names[k - first(r) + 1]))
        end
        push!(out, (name = nm, kind = :re_sd))
    end
    sdv = try re_sd(fit) catch err; err isa ArgumentError || rethrow(); nothing end
    if sdv !== nothing
        for (k, v) in sdv
            isfinite(v) && v < _TWIN_RE_SD_BOUNDARY &&
                push!(out, (name = string("sd:", k), kind = :re_sd))
        end
    end
    rho = _twin_rho12_values(fit)
    for (i, r) in enumerate(rho)
        isfinite(r) && abs(r) > _TWIN_RHO12_BOUNDARY &&
            push!(out, (name = length(rho) == 1 ? "rho12" : string("rho12[", i, "]"), kind = :rho12))
    end
    # Heywood: the Gaussian residual scale collapsed onto the data (`_nondegenerate_fit`),
    # which is not an aliased design (that one is :degenerate).
    !_nondegenerate_fit(fit) && !_design_rank_deficient(fit) &&
        push!(out, (name = "sigma", kind = :heywood))
    fit.geometry === :power_at_bound && push!(out, (name = "power", kind = :bound))
    return out
end

# Everything the rule reads, in one pass. `status` is decided in the order above.
function _convergence_report(fit::DrmFit)
    optok = fit.converged && isfinite(fit.loglik) && fit.loglik > _TWIN_SENTINEL_LOGLIK
    skipped = fit.uncertainty === :skipped
    floor_fixed = skipped ? Int[] : _twin_floor_fixed_coords(fit)
    pd = missing
    ses_ok = true
    if !skipped
        keep = setdiff(axes(fit.vcov, 1), floor_fixed)
        V = isempty(floor_fixed) ? fit.vcov : fit.vcov[keep, keep]
        se = _twin_standard_errors(V)
        pd = _twin_vcov_positive_definite(V)
        ses_ok = !isempty(se) && all(isfinite, se) &&
                 !_twin_inflated_se(se, _twin_se_reference_scales(fit)[keep])
    end
    # `fit.converged` already folds in a non-finite or sentinel log-likelihood (`_report_converged`).
    code = optok ? 0 : 1
    msg = optok ? "optimiser finished at a finite log-likelihood" :
          (isfinite(fit.loglik) && fit.loglik > _TWIN_SENTINEL_LOGLIK) ?
          "optimiser did not report convergence" :
          "log-likelihood is not finite or is the failed-objective sentinel"
    fit.iterations >= 0 && (msg = string(msg, " (iterations = ", fit.iterations, ")"))
    wall = haskey(fit.scales, :cond_logsd_wall) && isfinite(fit.loglik) &&
           fit.loglik > _TWIN_SENTINEL_LOGLIK
    # A fit that ended at the lower log-SD search bound of the independent-term Laplace route
    # (`_INDEP_LOGSD_GUARD`) has an SD of zero in practice and an optimiser flag of false.
    bparams = (optok || wall) ? _twin_boundary_params(fit) : _twin_boundary_params_none()
    if !optok && !wall
        return (status = :not_converged, boundary_params = bparams, pd_hessian = pd,
                max_gradient = NaN, optimizer_code = code, optimizer_message = msg)
    end
    # Order (SHINICHI-0724-1120 item 1, as drmTMB #1781): gradient first, then boundary.
    grad = _newton_step_for_rule(fit)
    if (isfinite(grad) && grad > _TWIN_NEWTON_STEP_TOL) || grad == Inf
        return (status = :not_converged, boundary_params = bparams, pd_hessian = pd,
                max_gradient = grad, optimizer_code = code, optimizer_message = msg)
    end
    if !isempty(bparams)
        return (status = :boundary, boundary_params = bparams, pd_hessian = pd,
                max_gradient = grad, optimizer_code = code, optimizer_message = msg)
    end
    status = if !(fit.geometry === :ok && !_design_rank_deficient(fit))
        :degenerate
    elseif !skipped && !(pd === true && ses_ok)
        :degenerate
    else
        :converged
    end
    return (status = status, boundary_params = bparams, pd_hessian = pd,
            max_gradient = grad, optimizer_code = code, optimizer_message = msg)
end

_twin_boundary_params_none() = NamedTuple{(:name, :kind),Tuple{String,Symbol}}[]

"""
    convergence_status(fit) -> Symbol

Classify a fit as `:converged`, `:not_converged`, `:boundary` or `:degenerate`
(Shinichi's converged rule, SHINICHI-1753-1118, shared with drmTMB). The first
matching rule wins and every status is a warning, never an error:

1. `:not_converged` unless the optimiser stopped normally (`fit.converged`) at a
   finite log-likelihood strictly above the `-1e15` failed-objective sentinel.
2. `:not_converged` when the Newton step in standard-error units,
   `max_i |(H⁻¹ g)_i| / SE_i` with `H⁻¹ = vcov(fit)`, `SE_i = sqrt((H⁻¹)_ii)` and `g` the
   gradient of the objective at the optimum, exceeds `1e-3`, or when no finite gradient can be produced (a non-finite gradient). This is tested BEFORE the boundary rule (SHINICHI-0724-1120, as drmTMB #1781). It is not scored (and the rule
   passes) when the fit stores no objective, when standard errors are off or `vcov` is not
   positive definite (rules 5 to 7 then speak), or for a REML fit, whose stored objective is
   the ML one and so is not the objective `vcov` inverts.
3. `:boundary` when a parameter is at a boundary, tested BEFORE the Hessian: a fitted
   residual correlation has `|rho12| > 0.98`; a random-effect SD ([`re_sd`](@ref)) is
   below `1e-4` (also a log SD within `1e-3` of the sparse-Laplace floor of `-8`, whose
   standard error is `NaN`, #1190); a Gaussian residual scale has collapsed onto the data
   (Heywood); or a Tweedie power ended at an end of its search interval.
4. `:degenerate` when a geometry screen fired: a Poisson or NegBinomial2 fixed-effect MLE
   that does not exist (#1109), a fixed-effect-only Binomial fit with (quasi-)complete
   separation (#1186), or an aliased fixed-effect column (#1110).
5. `:degenerate` unless `vcov(fit)` is square, finite and positive definite.
6. `:degenerate` unless every standard error is finite.
7. `:degenerate` when a scaled standard error `SE* = SE / s` is at least 50 **and** at least
   1000 times the median of the scaled standard errors below 50 (the smallest when none is
   below 50), with `s = sd(y)` for the coefficients of an identity-link location block
   (`mu`, `mu1`, `mu2` of Gaussian, Student, SkewNormal) and `s = 1` for link-scale ones, so
   the verdict does not change under `y -> c y` (drmTMB #1781). Non-intercept `mu` slopes
   still carry y/x units and are scaled by sd(y) only, as in drmTMB.
8. `:converged` otherwise.

Rules 5 to 7 are skipped when standard errors were turned off (`se = false`, or
`q4_vcov = false`), as drmTMB skips them for `drm_control(se = FALSE)`.

[`is_converged`](@ref) and [`converged`](@ref) are `true` only for `:converged`.
The raw optimiser flag is `fit.converged`. [`convergence_info`](@ref) returns the full
record: `converged`, `convergence_status`, `boundary_params` (name and kind),
`pd_hessian`, `max_gradient`, `optimizer_code` and `optimizer_message`.
"""
convergence_status(fit::DrmFit) = _convergence_report(fit).status

"""
    convergence_info(fit) -> NamedTuple

The converged rule's whole record, with the field names shared with drmTMB:

- `converged` — `convergence_status === :converged`.
- `convergence_status` — see [`convergence_status`](@ref).
- `boundary_params` — a vector of `(name, kind)`; `kind` is `:re_sd`, `:rho12`, `:heywood`
  or `:bound`. Empty unless the status is `:boundary`.
- `pd_hessian` — `true` / `false`, or `missing` when standard errors were turned off.
- `max_gradient` — the Newton step in standard-error units `max_i |(H⁻¹ g)_i| / SE_i` (not
  the raw gradient); `NaN` when it was not scored (no stored objective, a REML fit, no usable
  `vcov`, or the status was settled before the gradient).
- `optimizer_code` — `0` finished at a finite log-likelihood, `1` did not (including a non-finite
  or sentinel log-likelihood).
- `optimizer_message` — a short plain-language message. `DrmFit` stores only the optimiser's
  boolean flag and iteration count, so this is not the raw Optim message.
"""
function convergence_info(fit::DrmFit)
    r = _convergence_report(fit)
    return (converged = r.status === :converged, convergence_status = r.status,
            boundary_params = r.boundary_params, pd_hessian = r.pd_hessian,
            max_gradient = r.max_gradient, optimizer_code = r.optimizer_code,
            optimizer_message = r.optimizer_message)
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
