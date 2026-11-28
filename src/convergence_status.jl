# Twin convergence contract with drmTMB (#1554): optimizer, relative-gradient,
# boundary, and Hessian checks apply to prior and ordinary fits. Prior fits use
# the penalized objective; ordinary fits use the data likelihood.
#
#   1  optimiser not finished or unusable objective                         :not_converged
#   2  relative gradient unavailable or above 1e-4                          :not_converged
#   3  likelihood geometry failure                                          :degenerate
#   4  random-effect or residual correlation at its boundary                :boundary
#   5  covariance/Hessian non-positive-definite, invalid SE, inflated SE     :degenerate
#   6  otherwise                                                            :converged
# A skipped Hessian is not evidence of positive definiteness.

const _TWIN_INFLATED_SE_FLOOR = 50.0
const _TWIN_INFLATED_SE_RATIO = 1000.0
const _TWIN_RE_SD_BOUNDARY = 1e-4
const _TWIN_LOG_RE_SD_BOUNDARY = log(_TWIN_RE_SD_BOUNDARY)
const _TWIN_RHO12_BOUNDARY = 0.98
const _TWIN_SENTINEL_LOGLIK = -1e15
const _TWIN_RELATIVE_GRADIENT_TOLERANCE = 1e-4

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
    V isa AbstractMatrix || return Int[]
    (size(V, 1) == size(V, 2) == length(fit.theta)) || return Int[]
    any(p -> first(p) === :resd, fit.blocks) || return Int[]
    return [k for k in _block_range(fit, :resd)
            if fit.theta[k] <= -8.0 + _CLAMP_ACTIVE_TOL &&
               all(isnan, @view(V[k, :])) && all(isnan, @view(V[:, k]))]
end

function _twin_active_logsd_bounds(fit::DrmFit)
    lower = _twin_floor_fixed_coords(fit)
    upper = Int[]
    if any(p -> first(p) === :resd, fit.blocks)
        for i in _block_range(fit, :resd)
            if fit.theta[i] <= -8.0 + _CLAMP_ACTIVE_TOL
                push!(lower, i)
            elseif fit.theta[i] >= 3.0 - _CLAMP_ACTIVE_TOL
                push!(upper, i)
            end
        end
    end
    return (lower=unique(lower), upper=unique(upper))
end

# The values drmTMB reads for a Julia-engine fit: `sdpars` is the exponentiated
# `resd_*` block (plus the decoded `recov` pair), which is `re_sd(fit)`, and the
# residual rho12 is `corpairs`. A fit `re_sd` refuses (an `sd(g) ~ x` model)
# contributes no SD, as it contributes no `resd_*` block to drmTMB.
function _twin_re_sd_below_boundary(fit::DrmFit)
    d = try
        re_sd(fit)
    catch err
        err isa ArgumentError || rethrow()
        return false
    end
    direct_keys = Set{Symbol}()
    for (p, r) in fit.blocks
        p === :resd || continue
        nms = first(cn[2] for cn in fit.coefnames if cn[1] === :resd)
        for (j, nm) in enumerate(nms)
            push!(direct_keys, Symbol(nm))
            # Compare the fitted log-SD coordinate directly. Reconstructing
            # `exp(log(boundary))` can round one ulp below the strict public cut.
            fit.theta[r[j]] < _TWIN_LOG_RE_SD_BOUNDARY && return true
        end
    end
    return any(kv -> !(first(kv) in direct_keys) &&
                     isfinite(last(kv)) && last(kv) < _TWIN_RE_SD_BOUNDARY, pairs(d))
end

function _twin_rho12_values(fit::DrmFit)
    r = corpairs(fit)
    return r isa AbstractVector ? collect(Float64, r) : Float64[]
end

function _twin_re_correlation_values(fit::DrmFit)
    covariances = try
        vc(fit)
    catch err
        err isa InterruptException && rethrow()
        return Float64[]
    end
    correlations = Float64[]
    for Σ in values(covariances)
        Σ isa AbstractMatrix && size(Σ, 1) == size(Σ, 2) && size(Σ, 1) > 1 || continue
        sd = sqrt.(max.(diag(Σ), 0.0))
        for i in 1:(length(sd) - 1), j in (i + 1):length(sd)
            den = sd[i] * sd[j]
            push!(correlations, den > 0 ? Σ[i, j] / den : NaN)
        end
    end
    return correlations
end

"""
    convergence_status(fit) -> Symbol

Classify a fit as `:converged`, `:not_converged`, `:boundary`, or `:degenerate`.
The optimizer and relative-gradient checks use the objective actually fitted:
the penalized objective for MAP fits and the data objective otherwise.

1. `:not_converged` unless the optimizer finished and the objective is finite.
2. `:not_converged` when the (projected, at an identified active bound)
   relative gradient is unavailable or exceeds `1e-4`.
3. `:degenerate` when a geometry screen fires: a Poisson or NegBinomial2
   fixed-effect MLE that does not exist (#1109), a fixed-effect-only Binomial
   fit with (quasi-)complete separation (#1186; drmTMB `binomial_separation`),
   a Gaussian residual-scale collapse, or an aliased fixed-effect column.
4. `:boundary` only when a random-effect SD is below `1e-4`, a residual or
   random-effect correlation has absolute value above `0.98`, or a random-
   effect log SD reaches the sparse-Laplace floor near `-8`, and the fitted
   objective is stationary. At a true active optimizer bound, stationarity uses
   the projected score and lower/upper KKT sign condition. A wrong bound sign
   is `:not_converged`; an unresolvable active coordinate is `:degenerate`.
5. `:degenerate` unless the covariance/Hessian is positive definite, standard
   errors are finite, and no standard error exceeds 50 and 1000 times the
   median of standard errors below 50 (or the smallest standard error when
   none is below 50).
6. `:converged` otherwise.

A skipped or unavailable Hessian is not evidence of positive definiteness.
For a prior fit the penalized Hessian determines convergence. The
unpenalized Hessian is reported separately by [`check_drm`](@ref) to identify
parameters held by prior curvature.

[`is_converged`](@ref) and [`converged`](@ref) are `true` only for
`:converged`. The raw optimiser flag is `fit.converged`.
"""
function _fit_objective_gradient(fit::DrmFit)
    nll = fit.nll
    nll === nothing && return nothing
    θ = Vector{Float64}(fit.theta)

    if nll isa _FitObjectiveGradient
        g = zeros(length(θ))
        try
            nll.gradient!(g, θ)
            all(isfinite, g) && return g
        catch err
            err isa InterruptException && rethrow()
        end
        return nothing
    end

    if nll isa LocScaleObjective
        base = size(nll.Xμ, 2) + size(nll.Xψ, 2)
        perm = vcat(collect(1:base), [base + 1, base + 3, base + 2])
        g = zeros(length(θ))
        g[perm] = _ls_objective_gradient(nll, θ[perm])
        return all(isfinite, g) ? g : nothing
    end

    objective = if nll isa _AGHQDiagnosticObjective && fit.nllgrad === nothing
        snapshot = copy(nll.bcache)
        x -> nll.diagnostic(x, copy(snapshot))
    else
        nll
    end
    if fit.nllgrad !== nothing
        g = zeros(length(θ))
        try
            fit.nllgrad(g, θ)
            all(isfinite, g) && return g
        catch err
            err isa InterruptException && rethrow()
        end
    end
    try
        g = ForwardDiff.gradient(objective, θ)
        all(isfinite, g) && return Vector{Float64}(g)
    catch err
        err isa InterruptException && rethrow()
    end

    # Some exact Float64 objectives are not dual-number safe. A central
    # difference of the same objective keeps stationarity as a live gate.
    g = zeros(length(θ))
    for i in eachindex(θ)
        step = 1e-5 * max(abs(θ[i]), 1.0)
        xp = copy(θ); xm = copy(θ)
        xp[i] += step; xm[i] -= step
        fp = try Float64(objective(xp)) catch err
            err isa InterruptException && rethrow(); NaN
        end
        fm = try Float64(objective(xm)) catch err
            err isa InterruptException && rethrow(); NaN
        end
        g[i] = (fp - fm) / (2step)
    end
    return all(isfinite, g) ? g : nothing
end

function _relative_objective_gradient(fit::DrmFit)
    nll = fit.nll
    nll === nothing && return NaN
    θ = Vector{Float64}(fit.theta)
    g = fit.estim_method === :MAP ? _map_objective_gradient(fit) : _fit_objective_gradient(fit)
    g === nothing && return NaN
    # Some REML fitters keep the full ML objective in `fit.nll` for profile
    # inference, while the optimizer targeted the profiled restricted objective.
    # Their callback carries the actual fitting objective as well as its score.
    objective_fn = nll isa _FitObjectiveGradient ? nll.objective : nll
    objective = try
        fit.estim_method === :MAP ? _map_objective_nll(fit, θ) : Float64(objective_fn(θ))
    catch err
        err isa InterruptException && rethrow()
        return NaN
    end
    isfinite(objective) && all(isfinite, θ) || return NaN
    return maximum(abs.(g) .* max.(1.0, abs.(θ))) / max(1.0, abs(objective))
end

struct _FitObjectiveGradient{F,G,P}
    objective::F
    gradient!::G
    profile_objective::P
end

(fg::_FitObjectiveGradient)(θ) = fg.profile_objective(θ)

function _fit_convergence_status(;
    optimizer_ok::Bool,
    objective_ok::Bool,
    relative_gradient::Real,
    geometry_ok::Bool,
    at_boundary::Bool,
    boundary_kkt_valid::Union{Nothing,Bool}=nothing,
    hessian_pd::Bool,
    standard_errors_ok::Bool,
    inflated_se::Bool,
)
    (!optimizer_ok || !objective_ok) && return :not_converged
    if at_boundary
        boundary_kkt_valid === false && return :not_converged
        boundary_kkt_valid === nothing && return :degenerate
        (!isfinite(relative_gradient) ||
         relative_gradient > _TWIN_RELATIVE_GRADIENT_TOLERANCE) && return :not_converged
        return :boundary
    end
    (!isfinite(relative_gradient) ||
     relative_gradient > _TWIN_RELATIVE_GRADIENT_TOLERANCE) && return :not_converged
    !geometry_ok && return :degenerate
    (!hessian_pd || !standard_errors_ok || inflated_se) && return :degenerate
    return :converged
end

# KKT signs for a minimised objective: at a lower bound a valid score is
# non-negative; at an upper bound it is non-positive. Project valid active
# scores to zero so the remaining relative-gradient check applies to free
# coordinates only. `nothing` means the active constraint could not be mapped.
function _projected_kkt_gradient(g::AbstractVector, lower::AbstractVector{<:Integer},
                                 upper::AbstractVector{<:Integer}; tol::Real=1e-6)
    all(isfinite, g) || return nothing
    isempty(intersect(lower, upper)) || return nothing
    all(i -> 1 <= i <= length(g), lower) || return nothing
    all(i -> 1 <= i <= length(g), upper) || return nothing
    all(i -> g[i] >= -tol, lower) || return false
    all(i -> g[i] <= tol, upper) || return false
    pg = Float64.(g)
    for i in lower
        pg[i] = min(pg[i], 0.0)
    end
    for i in upper
        pg[i] = max(pg[i], 0.0)
    end
    return pg
end

function _boundary_constraint_indices(fit::DrmFit)
    bounds = _twin_active_logsd_bounds(fit)
    # `cond_logsd_wall` records an inner conditional-mode clamp. It is a true
    # active constraint, but it does not identify an outer theta coordinate;
    # do not mistake a separate :resd optimizer bound for a map of this wall.
    cond_wall_unmapped = haskey(fit.scales, :cond_logsd_wall)
    return (lower=bounds.lower, upper=bounds.upper,
            active=!isempty(bounds.lower) || !isempty(bounds.upper),
            known=!cond_wall_unmapped)
end

function _relative_projected_gradient(fit::DrmFit, g, constraints)
    constraints.known || return NaN
    pg = _projected_kkt_gradient(g, constraints.lower, constraints.upper)
    pg === nothing && return NaN
    pg === false && return Inf
    objective_fn = fit.nll isa _FitObjectiveGradient ? fit.nll.objective : fit.nll
    objective = try
        fit.estim_method === :MAP ? _map_objective_nll(fit, fit.theta) : Float64(objective_fn(fit.theta))
    catch
        NaN
    end
    isfinite(objective) || return NaN
    return maximum(abs.(pg) .* max.(1.0, abs.(fit.theta))) / max(1.0, abs(objective))
end

# Check curvature only when requested for a fit that deliberately skipped SEs.
# The covariance is temporary and is not attached to the returned fit.
function _twin_objective_curvature(fit::DrmFit)
    fit.nll === nothing && return nothing
    try
        objective = fit.estim_method === :MAP ? t -> _map_objective_nll(fit, t) : fit.nll
        H = ForwardDiff.hessian(objective, fit.theta)
        all(isfinite, H) || return nothing
        F = cholesky(Symmetric(Matrix{Float64}(H)); check=false)
        isposdef(F) || return (positive_definite=false, finite_se=false, inflated_se=false)
        V = inv(F)
        se = _twin_standard_errors(V)
        return (positive_definite=true, finite_se=!isempty(se) && all(isfinite, se),
                inflated_se=_twin_inflated_se(se))
    catch err
        err isa InterruptException && rethrow()
        return nothing
    end
end

"""
    convergence_status(fit::DrmFit) -> Symbol

Classify a fitted model as `:converged`, `:not_converged`, `:boundary`, or
`:degenerate` using optimizer completion, objective gradient, parameter
boundaries, fit geometry, and Hessian-based uncertainty checks. When a prior is
active, the status checks the penalized fit objective. Use [`is_converged`](@ref)
for the corresponding Boolean predicate. The raw optimizer flag is available
as `fit.converged`. If standard errors were skipped, this check evaluates the
objective Hessian when available; it does not add standard errors to the fit.
This curvature check can be expensive for models with many parameters.
"""
function convergence_status(fit::DrmFit)
    objective = NaN
    if fit.nll !== nothing
        objective = try
            fit.estim_method === :MAP ? _map_objective_nll(fit, fit.theta) : Float64(fit.nll(fit.theta))
        catch err
            err isa InterruptException && rethrow()
            NaN
        end
    end
    objective_ok = isfinite(objective) && isfinite(fit.loglik) &&
        fit.loglik > _TWIN_SENTINEL_LOGLIK
    active_bounds = _twin_active_logsd_bounds(fit)
    at_boundary = haskey(fit.scales, :cond_logsd_wall) ||
        !isempty(active_bounds.lower) || !isempty(active_bounds.upper) ||
        _twin_re_sd_below_boundary(fit) ||
        any(r -> isfinite(r) && abs(r) > _TWIN_RHO12_BOUNDARY,
            vcat(_twin_rho12_values(fit), _twin_re_correlation_values(fit)))
    penalized = fit.estim_method === :MAP
    constraints = at_boundary ? _boundary_constraint_indices(fit) : nothing
    gradient = at_boundary ? _map_objective_gradient(fit) : nothing
    relative_gradient = at_boundary ?
        (gradient === nothing ? NaN : _relative_projected_gradient(fit, gradient, constraints)) :
        _relative_objective_gradient(fit)
    geometry_ok = penalized || (fit.geometry === :ok && _nondegenerate_fit(fit))

    V = fit.vcov
    se = V isa AbstractMatrix ? _twin_standard_errors(V) : Float64[]
    hessian_pd = V isa AbstractMatrix && _twin_vcov_positive_definite(V)
    standard_errors_ok = !isempty(se) && all(isfinite, se)
    inflated_se = _twin_inflated_se(se)
    report = fit.control_info
    if report isa DRMControlReport && report.se in (:discarded, :skipped)
        curvature = report.se === :discarded ? report.discarded_curvature : _twin_objective_curvature(fit)
        if curvature !== nothing
            hessian_pd = curvature.positive_definite
            # No standard errors are reported for this fit, so the inflated-SE
            # diagnostic does not apply. Curvature itself must still be valid.
            standard_errors_ok = curvature.finite_se
            inflated_se = false
        end
    end
    return _fit_convergence_status(
        optimizer_ok=fit.converged,
        objective_ok=objective_ok,
        relative_gradient=relative_gradient,
        geometry_ok=geometry_ok,
        at_boundary=at_boundary,
        boundary_kkt_valid=at_boundary ? (gradient === nothing || constraints === nothing || !constraints.known ? nothing :
            (!constraints.active ? true :
             (_projected_kkt_gradient(gradient, constraints.lower, constraints.upper) === false ? false : true))) : nothing,
        hessian_pd=hessian_pd,
        standard_errors_ok=standard_errors_ok,
        inflated_se=inflated_se,
    )
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
