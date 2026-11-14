# inference.jl — Wald + profile-likelihood inference for fitted DRModels models.
# Wald: estimate ± z·se from the observed information stored on the fit, on each
# parameter's working scale (log σ, atanh ρ12, log σ_b). Profile: invert the
# likelihood-ratio statistic — the endpoints where 2(ℓ̂ − ℓ_profile) = χ²₁(level),
# re-optimising the nuisance parameters at each fixed value. Mirrors drmTMB's
# `confint(..., method = "wald" | "profile")`.

using LinearAlgebra: diag, isposdef, Symmetric, eigvals, BLAS
using Distributions: Normal, Chisq, quantile
using Optim: Optim
using Random: Random
using Statistics: Statistics
import StatsAPI: stderror, confint

"""
    stderror(fit) -> Vector{Float64}

Wald standard errors (√diag of the covariance), in the fit's coefficient order.

A coefficient's Wald SE is defined only where its estimated variance is finite
and positive. At a singular boundary the observed information is not
positive-definite, so the stored covariance carries a non-positive (or
non-finite) variance for the unidentified direction. Rather than return a silent
`NaN` there, that coefficient reports `Inf` — an undefined, infinitely wide
standard error — which propagates to an unbounded `(-Inf, Inf)` Wald interval.
[`check_drm`](@ref) flags the same situation via `vcov_posdef`; drmTMB returns
all-`NaN` from `sdreport` in this case.
"""
stderror(fit::DrmFit) = _wald_withheld(fit) ? _homtoep_refuse_wald("stderror") : _stderror(fit)

# Internal: the Hessian-based SEs regardless of the public Wald scope (profile
# step sizes, plots); `_display_se` is what `coeftable` / `show` print.
_stderror(fit::DrmFit) = _boundary_se.(diag(fit.vcov))
_display_se(fit::DrmFit) = (_wald_withheld(fit) || _wald_unreliable(fit) !== nothing) ?
    fill(NaN, length(fit.theta)) : _stderror(fit)

# #1193 (from #1127 Part B): `show` and `coeftable` do not print Wald SEs when the
# fit is not converged (`is_converged` false) or its Hessian is not positive
# definite; drmTMB refuses `vcov()` there. Returns the reason `show` prints, or
# `nothing` when the printed SEs stand. `stderror()` / `vcov()` are unchanged.
function _wald_unreliable(fit::DrmFit)
    if fit.uncertainty !== :skipped && !_twin_vcov_positive_definite(fit.vcov)
        return "the Hessian is not positive definite"
    end
    status = convergence_status(fit)
    status === :converged && return nothing
    return string("the fit is not converged (convergence_status = ", status, ")")
end

# √v where the variance is identified (finite, positive); Inf otherwise. Keeps a
# non-PD boundary direction from poisoning the whole SE vector with NaN.
_boundary_se(v::Real) = (isfinite(v) && v > 0) ? sqrt(v) : Inf

const _CIRow = NamedTuple{
    (:param, :coef, :estimate, :lower, :upper),Tuple{Symbol,String,Float64,Float64,Float64}
}

const _BootstrapSummaryRow = NamedTuple{
    (:param, :coef, :estimate, :std_error, :lower, :upper),
    Tuple{Symbol,String,Float64,Float64,Float64,Float64},
}

const _BootstrapFailureRow = NamedTuple{
    (:replicate, :seed, :message),Tuple{Int,UInt,String}
}

const _ProfileStatsRow = NamedTuple{
    (
        :param,
        :coef,
        :evaluations,
        :gradient_evaluations,
        :bracket_expansions,
        :root_iterations,
        :lower_unbounded,
        :upper_unbounded,
        :nonmonotone,
        # DRModels.jl#493: the refinement loop's `thi - tlo < 1e-8` bracket-collapse
        # exit was being treated identically to genuine convergence
        # (`abs(ht) < 1e-9`), which let a trapped, non-monotone profiled-nll
        # surface report a fabricated near-zero step as if it were a real
        # endpoint (measured: arm width 3.7e-09 vs a healthy ~0.2). These flag
        # that failure per arm, distinct from `unbounded` (a legitimate result:
        # the profile never crosses within the search range).
        :lower_endpoint_failed,
        :upper_endpoint_failed,
        # Generic-profile nuisance solves are auditable independently for each
        # endpoint.  These are intentionally status-only additions: CI rows
        # retain their established five-field public shape.
        :lower_nuisance_method,
        :upper_nuisance_method,
        :lower_nuisance_fallback,
        :upper_nuisance_fallback,
        :lower_nuisance_reason,
        :upper_nuisance_reason,
    ),
    Tuple{Symbol,String,Int,Int,Int,Int,Bool,Bool,Bool,Bool,Bool,
          Symbol,Symbol,Bool,Bool,Symbol,Symbol},
}

"""
Internal result of one fixed-coordinate nuisance solve.

`accepted` is deliberately stronger than a finite optimizer-reported minimum:
the optimizer must have terminated successfully, its minimizer must be finite,
and the objective is evaluated again at that minimizer.  We do not impose a
separate score threshold here; Optim termination is not a proof of stationarity.
"""
const _ProfileNuisanceResult = NamedTuple{
    (:value, :minimizer, :accepted, :method, :fallback, :reason),
    Tuple{Float64,Vector{Float64},Bool,Symbol,Bool,Symbol},
}

function _worker_threads(active::Bool, ntasks::Int)
    return active ? min(max(ntasks, 1), Threads.nthreads()) : 1
end
function _blas_oversubscribed(active::Bool)
    return active && Threads.nthreads() > 1 && BLAS.get_num_threads() > 1
end

# Run `f()` with BLAS pinned to one thread when a multi-threaded Julia region is
# about to make CONCURRENT dense-BLAS calls. BLAS thread count is process-global,
# so coordinated overlapping scopes retain the pin until the last scope exits.
# Concurrent callers into multi-threaded OpenBLAS contend on its internal locks:
# measured on the #545 dense route (64 tips, B = 99 bootstrap, 8 Julia threads),
# serial 5.03 s / threaded-unpinned 9.53 s / threaded-pinned 0.58 s — the pin is
# the difference between a 1.9x SLOWDOWN and an 8.7x speedup (#550).
#
# The lock covers only this state transition, never `f()`. Uncoordinated external
# calls to `BLAS.set_num_threads` while a scope is active remain outside this
# helper's contract; the last coordinated scope restores the setting it observed.
const _blas_pin_lock = ReentrantLock()
const _blas_pin_scopes = Ref{Int}(0)
const _blas_pin_restore = Ref{Int}(1)

function _with_pinned_blas(f, active::Bool)
    active || return f()
    lock(_blas_pin_lock)
    try
        if _blas_pin_scopes[] == 0
            old = BLAS.get_num_threads()
            old > 1 && BLAS.set_num_threads(1)
            _blas_pin_restore[] = old
        end
        _blas_pin_scopes[] += 1
    finally
        unlock(_blas_pin_lock)
    end
    try
        return f()
    finally
        lock(_blas_pin_lock)
        try
            _blas_pin_scopes[] > 0 || error("BLAS pin scope underflow")
            _blas_pin_scopes[] -= 1
            _blas_pin_scopes[] == 0 && BLAS.set_num_threads(_blas_pin_restore[])
        finally
            unlock(_blas_pin_lock)
        end
    end
end

function _refuse_selected_optimizer_refit(fit::DrmFit, entrypoint::AbstractString)
    info = fit.control_info
    if info isa DRMControlReport && info.requested_optimizer !== :default
        throw(ArgumentError("$entrypoint: this fit requested `control.optimizer = " *
                            "$(repr(info.requested_optimizer))`; inference refits do not " *
                            "yet forward the selected optimizer. Refit with " *
                            "`DRMControl(optimizer = :default)` for this inference route."))
    end
    return nothing
end

"""
    confint(fit; level = 0.95, method = :wald, threads = false, parm = nothing)

Confidence intervals for every coefficient, as a vector of
`(param, coef, estimate, lower, upper)` rows on each parameter's working scale
(μ on the response scale; σ on `log σ`; ρ12 on `atanh ρ12`; random-effect SDs on
`log σ_b`).

- `method = :wald` (default) — estimate ± z·se from the stored covariance.
- `method = :profile` — profile-likelihood interval: the endpoints where
  `2(ℓ̂ − ℓ_profile) = χ²₁(level)`, re-optimising the nuisance parameters at each
  fixed value (asymmetric, and exact under the LR statistic where Wald is only
  quadratic-approximate). Works on any fitted Gaussian model, and on the
  non-Gaussian canonical location–scale fit (`(1 | tag | group)` coupled RE),
  where it routes to a constrained L-BFGS nuisance solve with fresh objective
  and gradient acceptance checks on the variance boundary.
  The endpoint search uses warm-start continuation (each profiled solve starts
  from the previous point's optimum) and a guarded-Newton root-find driven by the
  envelope-theorem slope `∂nll/∂θ_k`, falling back to bisection. This guarded
  local search records explicit failed endpoint arms when it cannot certify an
  evaluated root. Endpoint validity assumes an accurate inner nuisance solve: the
  `:finite` autodiff path can leave the profiled NLL slightly non-monotone. When
  that is detected the bracket is reset to `[0, t]` and pure bisection is used.
  This is a guarded local search, not a guarantee of the globally first LR
  crossing; a `nonmonotone` flag is set on the profile stats row so callers can
  inspect that limitation.
  An endpoint arm that the search cannot certify is REFUSED, not returned: this
  method throws an `ArgumentError` naming the coefficient, the arm, and the
  nuisance-solve reason rather than reporting the failed side as a signed `Inf`.
  Use [`profile_result`](@ref) when you want the same rows plus
  the per-endpoint diagnostics instead of an exception.
  Pass `threads = true` to profile coefficients in parallel when the fitted
  objective is thread-safe; if only one coefficient is profiled, its lower and
  upper endpoint searches are run in parallel instead. Coupled location–scale
  random-effect fits (a shared `(1 | p | group)` term on `mu` and `sigma`) also
  profile coefficients in parallel, but each coefficient's lower and
  upper endpoint searches stay serial, so profiling a single coefficient there
  gains nothing from `threads = true`.
  A fit that explicitly selected `DRMControl(optimizer = :lbfgs | :bfgs)` is
  refused before profiling, because nuisance refits cannot yet forward it.
- `parm = :resd` or `parm = [:mu, :resd]` — restrict intervals to one or more
  parameter blocks. This is especially useful for profiling random-effect SDs
  without also profiling fixed-effect coefficients.
  A random-effect SD (`:resd`) whose lower arm reaches the numerical floor
  without an LR crossing is reported on the SD scale as `[0, u]` (drmTMB
  `profile.boundary`), not as a thrown error and not as `(-Inf, Inf)`.

Mirrors drmTMB's `confint(fit, method = "wald" | "profile")`.
"""
function confint(
    fit::DrmFit; level::Real=0.95, method::Symbol=:wald, threads::Bool=false, parm=nothing
)
    _is_fixed_prior_fit(fit) && fit.estim_method === :MAP && throw(ArgumentError("confint: intervals are not qualified for $(_prior_kind_label(fit)) MAP fits; covariance is local penalized curvature"))
    level = _validate_ci_level(level; what="confint")
    method === :wald && _wald_withheld(fit) && throw(ArgumentError("confint: Homogeneous Toeplitz " *
        "mean-coefficient Wald intervals are not yet qualified; mean-coefficient likelihood " *
        "profiles are qualified in drmTMB's retained primary panel cells, Wald covariance and " *
        "intervals remain deferred. Use `method = :profile`."))
    method === :wald && return _wald_ci(fit, level, parm)
    if method === :profile
        _refuse_selected_optimizer_refit(fit, "confint(method = :profile)")
        # drmTMB (#1448): the paired phylo() + OU route has point-recovery
        # evidence only; its profile intervals are not calibrated.
        _is_paired_phylo_temporal(fit) && @warn "confint: profile intervals from the paired " *
            "`phylo()` plus OU `temporal()` model are not calibrated. This development route has " *
            "point-recovery evidence only; no interval-calibration study has been run. Do not " *
            "report these intervals as confidence intervals."
        result = profile_result(fit; level, threads, parm)
        result.failed > 0 && _throw_profile_endpoint_failure(result)
        return result.ci
    end
    throw(ArgumentError("confint: method must be :wald or :profile (got :$method)"))
end

# DRModels.jl#631: a failed endpoint arm must NEVER leave the user-facing interval
# routine as a bound. `profile_result` is the AUDITABLE surface -- it keeps the
# ±Inf convention alongside `lower_endpoint_failed` / `upper_endpoint_failed` so
# a caller that asked for diagnostics can read them. `confint` is the surface a
# user reads a number off, and there `-Inf` is indistinguishable from a real
# confidence limit the moment it is printed, copied into a table, or written into
# a paper. Refuse instead, naming the coefficient, the arm, the nuisance-solve
# reason, and what to use in its place.
function _profile_failed_arm_descriptions(result)
    arms = String[]
    for s in result.stats
        (s.lower_endpoint_failed || s.upper_endpoint_failed) || continue
        sides = String[]
        s.lower_endpoint_failed &&
            push!(sides, "lower (nuisance solve: $(s.lower_nuisance_reason))")
        s.upper_endpoint_failed &&
            push!(sides, "upper (nuisance solve: $(s.upper_nuisance_reason))")
        push!(arms, "$(s.param):$(s.coef) — " * join(sides, ", "))
    end
    return arms
end

function _throw_profile_endpoint_failure(result)
    arms = _profile_failed_arm_descriptions(result)
    detail = isempty(arms) ? "$(result.failed) coefficient(s); per-arm diagnostics unavailable" :
        join(arms, "; ")
    throw(ArgumentError(
        "confint (method = :profile): the endpoint search did not converge for " *
        "$(result.failed) of $(result.attempted) coefficient(s) — $detail. " *
        "A non-converged endpoint is NOT a confidence limit, so it is refused " *
        "here rather than returned as a signed Inf. Use `method = :wald`, or " *
        "`bootstrap_ci` / `bootstrap_result`, for an interval on this " *
        "coefficient; call `profile_result(fit; ...)` if you need the same rows " *
        "with the per-endpoint diagnostics that explain the failure."))
end

# Build CI rows from the σ-phylo location-scale route's PRECOMPUTED boundary-aware profile CIs
# (fit.scales[:profile_ci_sd_mu] / [:profile_ci_sd_sigma], reported on the SD scale — an honest
# [0, x] at the variance boundary). Empty unless the fit was built with profile_ci=true. That
# route attaches no re-optimisable objective, so the generic profiler cannot recompute them; we
# surface the stored CIs instead. (Ayumi #2: σ-phylo SD boundary CI reachable from R.)
function _glsp_stored_profile_rows(fit::DrmFit)
    rows = _CIRow[]
    namemap = Dict(p => nms for (p, nms) in fit.coefnames)
    for (sdkey, param) in ((:profile_ci_sd_mu, :resd_mu), (:profile_ci_sd_sigma, :resd_sigma))
        (haskey(fit.scales, sdkey) && haskey(namemap, param)) || continue
        lo, hi = fit.scales[sdkey]
        est = exp(coef(fit, param)[1])                 # SD-scale point estimate
        push!(rows, (param=param, coef=namemap[param][1], estimate=Float64(est),
                     lower=Float64(lo), upper=Float64(hi)))
    end
    return rows
end

"""
    profile_result(fit; level = 0.95, threads = false, parm = nothing)

Auditable profile-likelihood confidence intervals. Returns a `NamedTuple` with:

- `ci` — the same rows `confint(fit; method = :profile)` returns, except that a
  FAILED endpoint arm is kept here as a signed `Inf` alongside its
  `lower_endpoint_failed` / `upper_endpoint_failed` flag. This is the auditable
  surface; `confint` refuses such a row rather than returning it;
- `stats` — per-coefficient endpoint work counts;
- `endpoint_diagnostics` — canonical location–scale endpoint reason, last
  evaluated candidate, and residual for each arm; other profile backends omit
  this additive diagnostic;
- `attempted`, `used`, `failed` — coefficient counts;
- `threaded`, `worker_threads`, `julia_threads`, `blas_threads`,
  `blas_oversubscribed`, `elapsed` — CPU context and wall-clock timing;
- `autodiff` — profile nuisance-gradient backend (`:stored`, `:forward`,
  `:finite`, or canonical location–scale `:locscale`).

For generic thread-safe objectives, set `threads = true` to parallelise
independent profile coefficients; when one coefficient is profiled, its endpoint
arms may run in parallel. Canonical location–scale profile jobs use only the
coefficient-level policy: each job owns its nuisance state, while its lower and
upper endpoint chains remain serial. A fit with an explicit `DRMControl.optimizer`
is refused before profiling. Threaded profiling is refused for
structured Binomial random-effect SD fits because their Laplace objectives use
mutable shared warm-start state; ordinary independent Binomial quadrature routes
are not subject to that restriction.
"""
function profile_result(fit::DrmFit; level::Real=0.95, threads::Bool=false, parm=nothing)
    _is_fixed_prior_fit(fit) && fit.estim_method === :MAP && throw(ArgumentError("profile_result: likelihood profiles are not qualified for $(_prior_kind_label(fit)) fits"))
    level = _validate_ci_level(level; what="profile_result")
    _refuse_selected_optimizer_refit(fit, "profile_result")
    threads && _binomial_sd_threaded_profile_unsupported(fit) &&
        throw(ArgumentError("profile_result: threaded profiling is not supported for structured Binomial random-effect SD fits; " *
                            "use `threads = false`."))
    _wald_withheld(fit) && (parm = _homtoep_profile_parm(fit, parm))
    fit.nll isa LocScaleObjective && return _ls_profile_result(
        fit; level=level, threads=threads, parm=parm
    )
    if fit.nll isa LocOnlyObjective
        jobs = _profile_jobs(fit, parm)
        if !isempty(jobs) && all(job.param === :resd for job in jobs)
            return _loconly_profile_result(fit; level=level, threads=threads, jobs=jobs)
        end
    end
    # σ-phylo location-scale: surface the PRECOMPUTED boundary-aware SD-scale profile CIs
    # (the route has no re-optimisable objective). Present only when profile_ci=true was set.
    let stored = _glsp_stored_profile_rows(fit)
        if !isempty(stored)
            sel = filter(r -> _ci_coef_selected(r.param, r.coef, parm), stored)
            return (ci=sel, stats=_ProfileStatsRow[], attempted=length(sel), used=length(sel),
                    failed=0, threaded=false, worker_threads=1, julia_threads=Threads.nthreads(),
                    blas_threads=BLAS.get_num_threads(), blas_oversubscribed=false,
                    elapsed=0.0, autodiff=:stored, level=float(level))
        end
    end
    fit.nll === nothing && throw(
        ArgumentError(
            "profile intervals require the fitted objective; this model was not built with one " *
            "(σ-phylo location-scale: fit with profile_ci=true to get boundary-aware SD CIs)",
        ),
    )
    nll = fit.nll
    nllgrad = fit.nllgrad
    θ̂ = copy(fit.theta)
    nllhat = nll(θ̂)
    isfinite(nllhat) || throw(ArgumentError("profile intervals require a finite fitted objective"))
    autodiff = _profile_autodiff_mode(nll, nllgrad, θ̂)
    half = quantile(Chisq(1), level) / 2
    se = _stderror(fit)
    jobs = _profile_jobs(fit, parm)
    rows = Vector{_CIRow}(undef, length(jobs))
    stats = Vector{_ProfileStatsRow}(undef, length(jobs))
    threaded = threads && Threads.nthreads() > 1
    coefficient_threaded = threaded && length(jobs) > 1
    endpoint_threaded = threaded && length(jobs) == 1
    elapsed = @elapsed begin
        if coefficient_threaded
            _with_pinned_blas(true) do
                Threads.@threads for i in eachindex(jobs)
                    rows[i], stats[i] = _profile_row_result(
                        jobs[i], nll, nllgrad, θ̂, nllhat, half, se, autodiff
                    )
                end
            end
        else
            _with_pinned_blas(endpoint_threaded) do
                for i in eachindex(jobs)
                    rows[i], stats[i] = _profile_row_result(
                        jobs[i],
                        nll,
                        nllgrad,
                        θ̂,
                        nllhat,
                        half,
                        se,
                        autodiff;
                        endpoint_threads=endpoint_threaded,
                    )
                end
            end
        end
    end
    # DRModels.jl#493: a coefficient counts as `failed` when either arm's endpoint
    # search hit the bracket-collapse exit without genuine convergence (the row
    # is still returned, with ±Inf on the failed side — see `_profile_endpoint_result`).
    failed = count(s -> s.lower_endpoint_failed || s.upper_endpoint_failed, stats)
    return (
        ci=rows,
        stats=stats,
        attempted=length(jobs),
        used=length(rows),
        failed=failed,
        threaded=threaded,
        worker_threads=_worker_threads(threaded, endpoint_threaded ? 2 : length(jobs)),
        julia_threads=Threads.nthreads(),
        blas_threads=BLAS.get_num_threads(),
        blas_oversubscribed=_blas_oversubscribed(threaded),
        elapsed=elapsed,
        autodiff=autodiff,
        level=float(level),
    )
end

# `parm` selection. Two granularities, because a block can be expensive:
#
#   nothing                      every coefficient
#   :mu                          every coefficient in the :mu block
#   [:mu, :sigma]                every coefficient in those blocks
#   :phylocov => "L44"           ONE coefficient           (#495)
#   [:phylocov => "L44", ...]    several named coefficients (#495)
#   [:mu, :phylocov => "L44"]    mixed block and coefficient selectors
#
# The per-coefficient form exists because profiling is priced per coefficient and
# some blocks are large: `:phylocov` on the q4 route is TEN entries, so a
# diagnostic that only needs three (a calibrated control, an over-coverer and an
# under-coverer) previously had to pay for all ten or none. Block-level callers
# are unaffected -- every pre-existing `parm` value means exactly what it did.
_ci_block_of(sel) = sel isa Pair ? first(sel) : sel

function _ci_param_selected(param::Symbol, parm)
    parm === nothing && return true
    parm isa Symbol && return param === parm
    parm isa Pair && return param === first(parm)
    if parm isa AbstractVector
        return any(sel -> param === _ci_block_of(sel), parm)
    end
    throw(ArgumentError(
        "confint: parm must be nothing, a Symbol, a `:block => \"coef\"` Pair, " *
        "or a Vector of those",
    ))
end

# A `parm` that names a coefficient which does not exist must THROW, not quietly
# return zero rows. A silent empty result from a typo is indistinguishable from a
# legitimate "nothing selected", and this codebase has repeatedly been bitten by
# measurements taken through apparatus that was not actually connected.
function _ci_validate_parm(fit::DrmFit, parm)
    parm === nothing && return nothing
    sels = parm isa AbstractVector ? collect(parm) : [parm]
    known = Set{Tuple{Symbol,String}}()
    blocks = Set{Symbol}()
    for ((p, _), (_, nms)) in zip(fit.blocks, fit.coefnames)
        push!(blocks, p)
        for nm in nms
            push!(known, (p, String(nm)))
        end
    end
    # A bare block Symbol naming a block this fit does not have is NOT an error:
    # callers legitimately pass a SUPERSET of block names covering several model
    # shapes (e.g. [:mu, :sigma, :resd, :resd_sigma]) and expect absent blocks to
    # be skipped. Rejecting that would break existing callers for no benefit.
    #
    # A Pair is different -- it names one specific coefficient -- but only when
    # its block is actually present. Absent block => same superset logic, skip it.
    # So the check fires exactly where a typo is the only plausible explanation:
    # the block exists, and the coefficient in it does not.
    for sel in sels
        sel isa Pair || continue
        blk = first(sel)
        blk in blocks || continue
        if !((blk, String(last(sel))) in known)
            avail = sort([c for (b, c) in known if b === blk])
            throw(ArgumentError(
                "confint: no coefficient `$(last(sel))` in block `:$(blk)`. " *
                "Available in that block: $(avail).",
            ))
        end
    end
    return nothing
end

# Full selection: block AND coefficient name. A bare block selector admits every
# coefficient in it, so this reduces to `_ci_param_selected` unless a Pair names
# the coefficient explicitly.
function _ci_coef_selected(param::Symbol, coef::AbstractString, parm)
    parm === nothing && return true
    parm isa Symbol && return param === parm
    parm isa Pair && return param === first(parm) && String(last(parm)) == String(coef)
    if parm isa AbstractVector
        return any(parm) do sel
            sel isa Pair ?
                (param === first(sel) && String(last(sel)) == String(coef)) :
                param === sel
        end
    end
    throw(ArgumentError(
        "confint: parm must be nothing, a Symbol, a `:block => \"coef\"` Pair, " *
        "or a Vector of those",
    ))
end

function _wald_ci(fit::DrmFit, level::Real, parm)
    level = _validate_ci_level(level; what="confint")
    _ci_validate_parm(fit, parm)
    se = stderror(fit)
    z = quantile(Normal(), 1 - (1 - level) / 2)
    rows = _CIRow[]
    for ((p, r), (_, nms)) in zip(fit.blocks, fit.coefnames)
        _ci_param_selected(p, parm) || continue
        for (j, idx) in enumerate(r)
            _ci_coef_selected(p, nms[j], parm) || continue
            est = fit.theta[idx]
            s = se[idx]
            push!(
                rows,
                (param=p, coef=nms[j], estimate=est, lower=est - z * s, upper=est + z * s),
            )
        end
    end
    return rows
end

function _profile_jobs(fit::DrmFit, parm)
    _ci_validate_parm(fit, parm)
    jobs = NamedTuple{(:param, :coef, :k),Tuple{Symbol,String,Int}}[]
    for ((pp, r), (_, nms)) in zip(fit.blocks, fit.coefnames)
        _ci_param_selected(pp, parm) || continue
        for (j, k) in enumerate(r)
            _ci_coef_selected(pp, nms[j], parm) || continue
            push!(jobs, (param=pp, coef=nms[j], k=k))
        end
    end
    return jobs
end

# Profile-likelihood CIs for the location–scale fit, routed to the robust
# L-BFGS constrained profiler `_ls_profile_ci_result` (locscale_profile.jl). The DrmFit packs
# the covariance block in `:recov` order [logL11, logL22, L21] while the engine
# packs [logL11, L21, logL22]; the permutation below (an involution that swaps
# the last two covariance entries) maps a DrmFit coefficient index to the engine
# index the profiler expects, and the engine θ̂ back from the stored `theta`.
function _ls_profile_result(fit::DrmFit; level::Real=0.95, threads::Bool=false, parm=nothing)
    obj = fit.nll::LocScaleObjective
    base = size(obj.Xμ, 2) + size(obj.Xψ, 2)
    perm = vcat(collect(1:base), [base + 1, base + 3, base + 2])  # involution
    θengine = fit.theta[perm]
    se = _stderror(fit)                                   # DrmFit (recov) order
    jobs = _profile_jobs(fit, parm)
    rows = Vector{_CIRow}(undef, length(jobs))
    stats = Vector{_ProfileStatsRow}(undef, length(jobs))
    endpoint_diagnostics = Vector{NamedTuple}(undef, length(jobs))
    # Only coefficient jobs are independent.  Each call below allocates its
    # own profile `lastsol`/typed-seed Refs, and retains the serial lower→upper
    # endpoint chain required by `_ls_profile_ci_result`.
    threaded = threads && Threads.nthreads() > 1 && length(jobs) > 1
    elapsed = @elapsed begin
        nmin = obj(θengine)                               # marginal NLL at θ̂ (once)
        function profile_one(i)
            job = jobs[i]
            s = se[job.k]
            se_arg = (isfinite(s) && s > 0) ? s : nothing
            ci = _ls_profile_ci_result(
                obj.kind,
                obj.y,
                obj.Xμ,
                obj.Xψ,
                obj.gidx,
                obj.G,
                obj.Q,
                θengine;
                idx=perm[job.k],
                level=level,
                nll_min=nmin,
                se=se_arg,
                whitened=obj.whitened,
            )
            row = (
                param=job.param,
                coef=job.coef,
                estimate=fit.theta[job.k],
                lower=ci.lower,
                upper=ci.upper,
            )
            lower_status = ci.lower_status
            upper_status = ci.upper_status
            lower_nuisance = lower_status.nuisance
            upper_nuisance = upper_status.nuisance
            stat = (
                param=job.param,
                coef=job.coef,
                evaluations=lower_status.evaluations + upper_status.evaluations,
                gradient_evaluations=lower_status.gradient_evaluations +
                                     upper_status.gradient_evaluations,
                bracket_expansions=lower_status.bracket_expansions +
                                   upper_status.bracket_expansions,
                root_iterations=lower_status.root_iterations + upper_status.root_iterations,
                lower_unbounded=lower_status.unbounded,
                upper_unbounded=upper_status.unbounded,
                nonmonotone=false,
                lower_endpoint_failed=lower_status.endpoint_failed,
                upper_endpoint_failed=upper_status.endpoint_failed,
                lower_nuisance_method=lower_nuisance === nothing ? :not_checked : lower_nuisance.method,
                upper_nuisance_method=upper_nuisance === nothing ? :not_checked : upper_nuisance.method,
                lower_nuisance_fallback=lower_nuisance === nothing ? false : lower_nuisance.fallback,
                upper_nuisance_fallback=upper_nuisance === nothing ? false : upper_nuisance.fallback,
                lower_nuisance_reason=lower_nuisance === nothing ? :not_checked : lower_nuisance.reason,
                upper_nuisance_reason=upper_nuisance === nothing ? :not_checked : upper_nuisance.reason,
            )
            diagnostic = (
                param=job.param,
                coef=job.coef,
                lower=(reason=lower_status.reason, candidate=lower_status.candidate,
                       residual=lower_status.residual, cancellation=lower_status.cancellation,
                       accepted=lower_status.accepted,
                       unbounded=lower_status.unbounded,
                       endpoint_failed=lower_status.endpoint_failed,
                       evaluations=lower_status.evaluations,
                       gradient_evaluations=lower_status.gradient_evaluations,
                       bracket_expansions=lower_status.bracket_expansions,
                       root_iterations=lower_status.root_iterations,
                       contractions=lower_status.contractions,
                       nuisance=lower_nuisance),
                upper=(reason=upper_status.reason, candidate=upper_status.candidate,
                       residual=upper_status.residual, cancellation=upper_status.cancellation,
                       accepted=upper_status.accepted,
                       unbounded=upper_status.unbounded,
                       endpoint_failed=upper_status.endpoint_failed,
                       evaluations=upper_status.evaluations,
                       gradient_evaluations=upper_status.gradient_evaluations,
                       bracket_expansions=upper_status.bracket_expansions,
                       root_iterations=upper_status.root_iterations,
                       contractions=upper_status.contractions,
                       nuisance=upper_nuisance),
            )
            return row, stat, diagnostic
        end
        if threaded
            _with_pinned_blas(true) do
                Threads.@threads for i in eachindex(jobs)
                    rows[i], stats[i], endpoint_diagnostics[i] = profile_one(i)
                end
            end
        else
            for i in eachindex(jobs)
                rows[i], stats[i], endpoint_diagnostics[i] = profile_one(i)
            end
        end
    end
    return (
        ci=rows,
        stats=stats,
        attempted=length(jobs),
        used=length(rows),
        failed=count(s -> s.lower_endpoint_failed || s.upper_endpoint_failed, stats),
        endpoint_diagnostics=endpoint_diagnostics,
        threaded=threaded,
        worker_threads=_worker_threads(threaded, length(jobs)),
        julia_threads=Threads.nthreads(),
        blas_threads=BLAS.get_num_threads(),
        blas_oversubscribed=_blas_oversubscribed(threaded),
        elapsed=elapsed,
        autodiff=:locscale,
        level=float(level),
    )
end

function _loconly_profile_result(
    fit::DrmFit; level::Real=0.95, threads::Bool=false, jobs=nothing
)
    obj = fit.nll::LocOnlyObjective
    θ̂ = copy(fit.theta)
    nllhat = _loconly_profile_fg(obj.prob, θ̂[obj.pμ + 1], θ̂[obj.pμ + 2])[1]
    half = quantile(Chisq(1), level) / 2
    jobs === nothing && (jobs = _profile_jobs(fit, :resd))
    rows = Vector{_CIRow}(undef, length(jobs))
    stats = Vector{_ProfileStatsRow}(undef, length(jobs))
    threaded = threads && Threads.nthreads() > 1
    endpoint_threaded = threaded && length(jobs) == 1
    elapsed = @elapsed begin
        if threaded && length(jobs) > 1
            Threads.@threads for i in eachindex(jobs)
                rows[i], stats[i] = _loconly_profile_row_result(
                    jobs[i], obj, θ̂, nllhat, half
                )
            end
        else
            for i in eachindex(jobs)
                rows[i], stats[i] = _loconly_profile_row_result(
                    jobs[i], obj, θ̂, nllhat, half; endpoint_threads=endpoint_threaded
                )
            end
        end
    end
    return (
        ci=rows,
        stats=stats,
        attempted=length(jobs),
        used=length(rows),
        # #1023: computed from the per-arm flags, as on the generic path.
        failed=count(s -> s.lower_endpoint_failed || s.upper_endpoint_failed, stats),
        threaded=threaded,
        worker_threads=_worker_threads(threaded, endpoint_threaded ? 2 : length(jobs)),
        julia_threads=Threads.nthreads(),
        blas_threads=BLAS.get_num_threads(),
        blas_oversubscribed=_blas_oversubscribed(threaded),
        elapsed=elapsed,
        autodiff=:loconly,
        level=float(level),
    )
end

function _loconly_profile_row_result(
    job, obj::LocOnlyObjective, θ̂, nllhat, half; endpoint_threads::Bool=false
)
    k = job.k
    est = θ̂[k]
    lσ0 = θ̂[obj.pμ + 1]
    if endpoint_threads && Threads.nthreads() > 1
        left = Threads.@spawn _loconly_profile_endpoint_result(
            obj, lσ0, est, nllhat, half, -1.0
        )
        right = Threads.@spawn _loconly_profile_endpoint_result(
            obj, lσ0, est, nllhat, half, +1.0
        )
        lo, lstats = fetch(left)
        hi, rstats = fetch(right)
    else
        lo, lstats = _loconly_profile_endpoint_result(obj, lσ0, est, nllhat, half, -1.0)
        hi, rstats = _loconly_profile_endpoint_result(obj, lσ0, est, nllhat, half, +1.0)
    end
    rewritten = _rewrite_log_sd_boundary_ci(job, est, lo, hi, lstats, rstats)
    rewritten === nothing || return rewritten
    row = (param=job.param, coef=job.coef, estimate=est, lower=lo, upper=hi)
    stats = (
        param=job.param,
        coef=job.coef,
        evaluations=lstats.evaluations + rstats.evaluations,
        gradient_evaluations=lstats.gradient_evaluations + rstats.gradient_evaluations,
        bracket_expansions=lstats.bracket_expansions + rstats.bracket_expansions,
        root_iterations=lstats.root_iterations + rstats.root_iterations,
        lower_unbounded=lstats.unbounded,
        upper_unbounded=rstats.unbounded,
        nonmonotone=lstats.nonmonotone || rstats.nonmonotone,
        # #1023: these are now measured by `_loconly_profile_endpoint_result`
        # (sentinel / non-finite objective and the nuisance solve's own exit),
        # not hard-coded. The bracket-collapse residual check of #493 is still
        # specific to the generic path.
        lower_endpoint_failed=lstats.endpoint_failed,
        upper_endpoint_failed=rstats.endpoint_failed,
        lower_nuisance_method=:specialized,
        upper_nuisance_method=:specialized,
        lower_nuisance_fallback=lstats.nuisance_fallback,
        upper_nuisance_fallback=rstats.nuisance_fallback,
        lower_nuisance_reason=lstats.nuisance_reason,
        upper_nuisance_reason=rstats.nuisance_reason,
    )
    return row, stats
end

function _loconly_profile_endpoint_result(
    obj::LocOnlyObjective, lσ0::Real, estimate::Real, nllhat::Real, half::Real, dir::Real
)
    target = nllhat + half
    lσ_start = Float64(lσ0)
    evaluations = 0
    gradient_evaluations = 0
    last_nuisance = (fallback=false, reason=:accepted)

    # #1023: an evaluation is usable only when the nuisance solve finished and the
    # objective is finite and below the sentinel. A sentinel or NaN is never read
    # as "above the threshold" (that turned the edge of the failure region into a
    # CI endpoint).
    function heval(t)
        f, lσ_opt, slope, nuisance = _loconly_profiled_resd_nll(obj, lσ_start, estimate + dir * t)
        nuisance.accepted && (lσ_start = lσ_opt)
        evaluations += 1
        gradient_evaluations += 1
        last_nuisance = nuisance
        return (h=f - target, hp=dir * slope, ok=nuisance.accepted)
    end
    # The log-SD parameterisation is bounded (|log σ_phy| < 50, `_loconly_profile_beta`).
    # A profile still below the threshold at that edge is unbounded within the
    # parameter space, not a failed search.
    at_domain_edge(t) = !(abs(estimate + dir * t) < 50)
    function arm(value; unbounded=false, endpoint_failed=false, iters=0, root_iterations=0,
                 nonmonotone=false)
        stats = (
            evaluations=evaluations,
            gradient_evaluations=gradient_evaluations,
            bracket_expansions=iters,
            root_iterations=root_iterations,
            unbounded=unbounded,
            nonmonotone=nonmonotone,
            endpoint_failed=endpoint_failed,
            nuisance_fallback=last_nuisance.fallback,
            nuisance_reason=last_nuisance.reason,
        )
        return value, stats
    end
    open_value = dir < 0 ? -Inf : Inf

    tlo = 0.0
    thi = 0.1
    at_domain_edge(thi) && return arm(open_value; unbounded=true)
    ev = heval(thi)
    ev.ok || return arm(open_value; endpoint_failed=true)
    hhi = ev.h
    hprev = hhi
    nonmonotone = false
    iters = 0
    t_edge = 49.999 - dir * estimate     # last probe just inside the domain bound
    while hhi < 0 && iters < 60
        tnext = thi * 1.6
        if at_domain_edge(tnext)
            # Probe once just inside the bound; still below the threshold there
            # means no crossing exists in the parameter space.
            t_edge > thi || return arm(open_value; unbounded=true, iters=iters,
                                       nonmonotone=nonmonotone)
            tnext = t_edge
        end
        tlo = thi
        thi = tnext
        ev = heval(thi)
        ev.ok || return arm(open_value; endpoint_failed=true, iters=iters, nonmonotone=nonmonotone)
        hhi = ev.h
        hhi < hprev - 1e-12 && (nonmonotone = true)
        hprev = hhi
        iters += 1
    end
    if hhi < 0
        return arm(open_value; unbounded=true, iters=iters, nonmonotone=nonmonotone)
    end
    nonmonotone && (tlo = 0.0)   # conservative bracket to the first crossing (see _profile_endpoint_result)

    t = (tlo + thi) / 2
    root_iterations = 0
    for _ in 1:80
        root_iterations += 1
        ev = heval(t)
        ev.ok || return arm(open_value; endpoint_failed=true, iters=iters,
                            root_iterations=root_iterations, nonmonotone=nonmonotone)
        ht, hp = ev.h, ev.hp
        abs(ht) < 1e-9 && break
        ht < 0 ? (tlo = t) : (thi = t)
        tn = (!nonmonotone && isfinite(hp) && hp > 0) ? t - ht / hp : (tlo + thi) / 2
        t = (tlo < tn < thi) ? tn : (tlo + thi) / 2
        thi - tlo < 1e-8 && break
    end
    return arm(estimate + dir * t; iters=iters, root_iterations=root_iterations,
               nonmonotone=nonmonotone)
end

function _loconly_profiled_resd_nll(obj::LocOnlyObjective, lσ0::Real, lσ_phy::Real)
    x0 = [Float64(lσ0)]
    function fg!(F, G, x)
        val, grad, _, _ = _loconly_profile_fg(obj.prob, x[1], lσ_phy)
        G !== nothing && (G[1] = grad[1])
        return F === nothing ? nothing : val
    end
    fallback = false
    res = try
        od = Optim.NLSolversBase.only_fg!(fg!)
        Optim.optimize(
            od,
            x0,
            Optim.LBFGS(; linesearch=Optim.LineSearches.BackTracking(; order=3)),
            Optim.Options(; iterations=80, g_tol=1e-8, x_abstol=1e-10),
        )
    catch err
        err isa InterruptException && rethrow(err)
        # #1023: the fallback is recorded (and its own exit checked below).
        fallback = true
        obj1(x) = _loconly_profile_fg(obj.prob, x[1], lσ_phy)[1]
        Optim.optimize(
            obj1,
            x0,
            Optim.NelderMead(),
            Optim.Options(; iterations=120, x_abstol=1e-10),
        )
    end
    lσ = Optim.minimizer(res)[1]
    val, grad, _, _ = _loconly_profile_fg(obj.prob, lσ, lσ_phy)
    # #1023: an unfinished nuisance minimisation gives a profile NLL that is too
    # high (an interval that is too narrow), and the 1e18 sentinel is not a value.
    reason = if !(isfinite(lσ) && isfinite(val)) || val >= _LOCONLY_PENALTY / 100
        :nonfinite_objective
    elseif !(Optim.converged(res) || abs(grad[1]) <= 1e-6 * max(1, obj.prob.n))
        # A backtracking line search can stop "unsuccessfully" at a rounding-level
        # optimum, so a scale-free score check (the residual-scale score is a sum
        # over n observations) also certifies the nuisance solve.
        fallback ? :fallback_not_converged : :not_converged
    else
        :accepted
    end
    nuisance = (accepted=reason === :accepted, fallback=fallback, reason=reason)
    return val, lσ, grad[2], nuisance
end

# Profiled objective: minimise nll over every component except k, with θ[k] = v.
# Returns (minimum, û_nuisance) so the caller can WARM-START the next solve from
# the previous point's nuisance optimum (continuation along the profile path) —
# the dominant speedup over cold-starting each solve from the joint MLE.
function _profile_autodiff_mode(nll, nllgrad, θ̂::Vector{Float64})
    nllgrad !== nothing && return :stored
    try
        ForwardDiff.gradient(nll, θ̂)
        return :forward
    catch err
        err isa InterruptException && rethrow()
        # Some fitted objectives are exact on Float64 but not dual-number safe
        # because they solve an inner sparse/Laplace mode with Float64 work
        # arrays. Keep profiling valid by using finite-difference nuisance
        # gradients when the Float64 objective itself is fine.
        try
            nll(θ̂)
        catch
            rethrow(err)
        end
        return :finite
    end
end

# Compare profile and reference NLLs at their represented scale.  We retain an
# eight-ULP cancellation allowance, but no relative-to-NLL tolerance: adding a
# huge constant to an objective must not make a real one-unit discrepancy pass.
# A profiled value at or beyond this while the reference NLL is ordinary is an
# optimiser-wall sentinel (e.g. 1e18), never a real objective; it must not be
# read as a profile crossing.  A uniformly shifted objective (reference also
# huge) is a precision problem, handled below, not a sentinel.
const _PROFILE_SENTINEL = 1e16

# drmTMB `profile_endpoint_crossing(..., allow_lower_boundary = TRUE)` for
# exp-transformed SD targets (R/profile.R). A lower arm that reaches
# `log(sqrt(eps))` without crossing the LR threshold is the SD-scale boundary 0,
# not a failed search and not the whole real line. Refs #1117.
const _PROFILE_SD_LOG_FLOOR = log(sqrt(eps(Float64)))
# Pseudo-inverse SEs at a collapsed variance can be ~1e-19 and then the
# geometric bracket never leaves the floor (seed 20261016902: 1.6^40 * 5e-19
# stays at 6e-11). Treat that as "no usable step".
const _PROFILE_MIN_STEP = 1e-6

_is_log_sd_profile_param(param::Symbol) =
    param === :resd || param === :resd_mu || param === :resd_sigma

function _profile_usable_step(se_k::Real, est::Real)
    (isfinite(se_k) && se_k > _PROFILE_MIN_STEP) && return Float64(se_k)
    return 1.0
end

function _profile_reference_difference(value::Real, reference::Real)
    (isfinite(value) && isfinite(reference) && abs(value) >= _PROFILE_SENTINEL &&
     abs(reference) < _PROFILE_SENTINEL) && return (
        status=:sentinel_objective, difference=NaN, cancellation=NaN,
    )
    (isfinite(value) && isfinite(reference)) || return (
        status=:nonfinite_objective, difference=NaN, cancellation=NaN,
    )
    cancellation = 8 * max(eps(abs(Float64(value))), eps(abs(Float64(reference))))
    difference = Float64(value) - Float64(reference)
    isfinite(difference) && isfinite(2 * difference) || return (
        status=:insufficient_precision, difference=difference, cancellation=cancellation,
    )
    difference < -max(1e-10, cancellation) && return (
        status=:below_reference, difference=difference, cancellation=cancellation,
    )
    return (status=:accepted, difference=difference, cancellation=cancellation)
end

function _profile_nuisance_result(
    nll,
    θ̂::Vector{Float64},
    k::Int,
    v::Real,
    u0::Vector{Float64};
    autodiff::Symbol=:forward,
    nllgrad=nothing,
    primary_iterations::Union{Nothing,Int}=nothing,
    fallback_iterations::Union{Nothing,Int}=nothing,
    primary_attempt=nothing,
)
    p = length(θ̂)
    idx = [i for i in 1:p if i != k]
    if isempty(idx)
        value = try
            Float64(nll([float(v)]))
        catch err
            err isa InterruptException && rethrow()
            NaN
        end
        return (value=value, minimizer=Float64[], accepted=isfinite(value),
                method=:direct, fallback=false,
                reason=isfinite(value) ? :accepted : :nonfinite_objective)
    end
    function obj(u)
        θ = Vector{eltype(u)}(undef, p)
        θ[k] = convert(eltype(u), v)
        @inbounds for (t, i) in enumerate(idx)
            θ[i] = u[t]
        end
        return nll(θ)
    end
    grad_u! = if autodiff === :stored && nllgrad !== nothing
        gfull = zeros(p)
        function (Gout, u)
            θ = Vector{Float64}(undef, p)
            θ[k] = float(v)
            @inbounds for (t, i) in enumerate(idx)
                θ[i] = u[t]
            end
            nllgrad(gfull, θ)
            @inbounds for (t, i) in enumerate(idx)
                Gout[t] = gfull[i]
            end
            return Gout
        end
    else
        nothing
    end
    return _profile_optimize_result(
        obj, u0, autodiff;
        (grad!)=grad_u!, primary_iterations, fallback_iterations,
        primary_attempt,
    )
end

function _profile_attempt(obj, u0, method::Symbol, autodiff::Symbol;
                          (grad!)=nothing, iterations::Union{Nothing,Int}=nothing,
                          fallback::Bool=false)
    res = try
        if method === :lbfgs_stored
            od = Optim.OnceDifferentiable(obj, grad!, u0)
            ls = Optim.LineSearches.BackTracking(; iterations=20)
            Optim.optimize(
                od, u0, Optim.LBFGS(; linesearch=ls),
                Optim.Options(; iterations=something(iterations, 40), g_tol=1e-6, x_abstol=1e-8),
            )
        elseif method === :nelder_mead
            iterations === nothing ? Optim.optimize(obj, u0, Optim.NelderMead()) :
                Optim.optimize(
                    obj, u0, Optim.NelderMead(), Optim.Options(; iterations, x_abstol=1e-8),
                )
        elseif method === :lbfgs_finite_backtracking
            ls = Optim.LineSearches.BackTracking(; iterations=20)
            iterations === nothing ?
                Optim.optimize(obj, u0, Optim.LBFGS(; linesearch=ls); autodiff) :
                Optim.optimize(
                    obj, u0, Optim.LBFGS(; linesearch=ls),
                    Optim.Options(; iterations, g_tol=1e-6, x_abstol=1e-8);
                    autodiff,
                )
        else
            # The pre-slice forward/finite path intentionally used Optim's
            # defaults (rather than the capped stored-gradient budget). Keep
            # that policy unless a private test override requests a budget.
            iterations === nothing ?
                Optim.optimize(obj, u0, Optim.LBFGS(); autodiff) :
                Optim.optimize(
                    obj, u0, Optim.LBFGS(),
                    Optim.Options(; iterations, g_tol=1e-6, x_abstol=1e-8);
                    autodiff,
                )
        end
    catch err
        err isa InterruptException && rethrow()
        return (value=NaN, minimizer=Float64[], accepted=false, method=method,
                fallback=fallback, reason=:exception)
    end
    minimizer = try
        Float64.(Optim.minimizer(res))
    catch err
        err isa InterruptException && rethrow()
        Float64[]
    end
    all(isfinite, minimizer) || return (
        value=NaN, minimizer=minimizer, accepted=false, method=method,
        fallback=fallback, reason=:nonfinite_minimizer,
    )
    value = try
        Float64(obj(minimizer))
    catch err
        err isa InterruptException && rethrow()
        NaN
    end
    isfinite(value) || return (
        value=value, minimizer=minimizer, accepted=false, method=method,
        fallback=fallback, reason=:nonfinite_objective,
    )
    Optim.converged(res) || return (
        value=value, minimizer=minimizer, accepted=false, method=method,
        fallback=fallback, reason=fallback ? :fallback_not_converged : :not_converged,
    )
    return (value=value, minimizer=minimizer, accepted=true, method=method,
            fallback=fallback, reason=:accepted)
end

function _profile_optimize_result(obj, u0::Vector{Float64}, autodiff::Symbol;
                                  (grad!)=nothing, primary_iterations::Union{Nothing,Int}=nothing,
                                  fallback_iterations::Union{Nothing,Int}=nothing,
                                  primary_attempt=nothing)
    primary_method = grad! === nothing ?
        (autodiff === :finite ? :lbfgs_finite : :lbfgs_forward) : :lbfgs_stored
    primary = primary_attempt === nothing ?
        _profile_attempt(obj, u0, primary_method, autodiff;
                         (grad!)=grad!, iterations=primary_iterations) :
        primary_attempt(obj, u0, primary_method, autodiff, grad!)
    primary.accepted && return primary

    # A finite stored-gradient attempt can hit its bounded outer-iteration
    # budget immediately after a reference-objective evaluation refreshes a
    # fitted workspace. Continue once from that finite candidate with the same
    # bounded method. Acceptance still requires Optim convergence; this is not
    # permission to accept the first non-converged result or switch algorithms.
    if grad! !== nothing && primary.reason === :not_converged &&
       isfinite(primary.value) && all(isfinite, primary.minimizer)
        continuation = _profile_attempt(
            obj, primary.minimizer, :lbfgs_stored, autodiff;
            (grad!)=grad!, iterations=something(primary_iterations, 40), fallback=true,
        )
        continuation.accepted && return continuation
        return continuation
    end

    # A finite-difference LBFGS endpoint can return a usable candidate without
    # converging, especially when its default line search stalls. Retry that
    # candidate with bounded BackTracking; only Optim convergence is accepted.
    if grad! === nothing && autodiff === :finite && primary.reason === :not_converged &&
       isfinite(primary.value) && all(isfinite, primary.minimizer)
        return _profile_attempt(
            obj, primary.minimizer, :lbfgs_finite_backtracking, :finite;
            iterations=something(primary_iterations, 3000), fallback=true,
        )
    end

    # Preserve the remaining fallback policy. A stored-gradient construction
    # exception retries finite-difference LBFGS; a finite-difference exception
    # can retry value-only Nelder--Mead.
    if grad! !== nothing && primary.reason === :exception
        return _profile_attempt(obj, u0, :lbfgs_finite, :finite;
                                iterations=something(primary_iterations, 40), fallback=true)
    elseif grad! === nothing && autodiff === :finite && primary.reason === :exception
        return _profile_attempt(obj, u0, :nelder_mead, :finite;
                                iterations=fallback_iterations, fallback=true)
    end
    return primary
end

# Backward-compatible pair helper for existing internal callers.  New generic
# profiling must use `_profile_nuisance_result` to retain a rejected-solve status.
function _profiled_nll(
    nll, θ̂::Vector{Float64}, k::Int, v::Real, u0::Vector{Float64};
    autodiff::Symbol=:forward, nllgrad=nothing,
)
    result = _profile_nuisance_result(nll, θ̂, k, v, u0; autodiff, nllgrad)
    result.accepted || throw(ArgumentError(
        "profile nuisance solve failed ($(result.method), $(result.reason))",
    ))
    return result.value, result.minimizer
end

# Historical internal entry point.  It remains for callers outside the generic
# profiler; the generic route uses `_profile_optimize_result` and validates the
# candidate before accepting it.
function _profile_optimize(obj, u0::Vector{Float64}, autodiff::Symbol; (grad!)=nothing)
    result = _profile_optimize_result(obj, u0, autodiff; (grad!)=grad!)
    result.accepted || throw(ArgumentError(
        "profile nuisance solve failed ($(result.method), $(result.reason))",
    ))
    return result
end

# Analytic slope of the PROFILED nll in θ[k], by the envelope theorem: at the
# profiled optimum the nuisance partials vanish, so d/dv [min_u nll] = ∂nll/∂θ_k
# evaluated at (v, û). One ForwardDiff gradient component — no extra solves.
function _profile_slope(
    nll, nllgrad, θ̂::Vector{Float64}, k::Int, v::Real, û::Vector{Float64}
)
    p = length(θ̂)
    idx = [i for i in 1:p if i != k]
    θ = Vector{Float64}(undef, p)
    θ[k] = float(v)
    @inbounds for (t, i) in enumerate(idx)
        θ[i] = û[t]
    end
    if nllgrad !== nothing
        g = zeros(p)
        nllgrad(g, θ)
        return g[k]
    end
    return ForwardDiff.gradient(nll, θ)[k]
end

# Endpoint in working coordinate where the profiled nll rises by `half` above the
# minimum, searching from θ̂[k] in direction dir ∈ {-1,+1}. h(t) = profiled_nll −
# target is increasing in t ≥ 0 with h(0) = −half < 0. We bracket by expansion,
# then root-find with a GUARDED NEWTON step (using the envelope-theorem slope,
# h'(t) = dir · ∂nll/∂θ_k): a Newton step that stays inside the maintained
# bracket is accepted, otherwise we fall back to bisection. Correctness is
# bracket-guaranteed; the analytic slope only buys faster (quadratic) convergence.
# Each evaluation warm-starts the nuisance optimisation from the previous û.
function _profile_endpoint(nll, nllgrad, θ̂, k, nllhat, half, s, dir, u0, autodiff;
                           allow_lower_boundary::Bool=false)
    value, _ = _profile_endpoint_result(
        nll, nllgrad, θ̂, k, nllhat, half, s, dir, u0, autodiff;
        allow_lower_boundary=allow_lower_boundary,
    )
    return value
end

function _profile_endpoint_result(nll, nllgrad, θ̂, k, nllhat, half, s, dir, u0, autodiff;
                                 allow_lower_boundary::Bool=false)
    target = nllhat + half
    p = length(θ̂)
    nidx = p - 1
    û = copy(u0)
    evaluations = 0
    gradient_evaluations = 0
    # h(t) and its derivative h'(t).  A failed solve deliberately does not
    # update `û`, so a bad arm cannot warm-start its next point from an invalid
    # nuisance state.
    function heval(t)
        nuisance = _profile_nuisance_result(
            nll, θ̂, k, θ̂[k] + dir * t, û; autodiff, nllgrad,
        )
        evaluations += 1
        nuisance.accepted || return (ok=false, h=NaN, hp=NaN, nuisance=nuisance)
        isempty(nuisance.minimizer) || (û = nuisance.minimizer)
        hp = if nidx == 0 || autodiff === :finite
            NaN
        else
            try
                gradient_evaluations += 1
                dir * _profile_slope(nll, nllgrad, θ̂, k, θ̂[k] + dir * t, û)
            catch err
                err isa InterruptException && rethrow()
                NaN
            end
        end
        reference = _profile_reference_difference(nuisance.value, nllhat)
        if reference.status !== :accepted
            nuisance = merge(nuisance, (accepted=false, reason=reference.status))
            return (ok=false, h=NaN, hp=NaN, cancellation=reference.cancellation, nuisance=nuisance)
        end
        h = reference.difference - half
        isfinite(h) || begin
            nuisance = merge(nuisance, (accepted=false, reason=:insufficient_precision))
            return (ok=false, h=NaN, hp=NaN, cancellation=reference.cancellation, nuisance=nuisance)
        end
        return (ok=true, h=h, hp=hp, cancellation=reference.cancellation, nuisance=nuisance)
    end
    function failed_arm(nuisance, root_iterations=0, bracket_expansions=0, nonmonotone=false)
        value = dir < 0 ? -Inf : Inf
        stats = (
            evaluations=evaluations,
            gradient_evaluations=gradient_evaluations,
            bracket_expansions=bracket_expansions,
            root_iterations=root_iterations,
            unbounded=false,
            nonmonotone=nonmonotone,
            endpoint_failed=true,
            nuisance_method=nuisance.method,
            nuisance_fallback=nuisance.fallback,
            nuisance_reason=nuisance.reason,
        )
        return value, stats
    end
    function boundary_arm(nuisance, root_iterations=0, bracket_expansions=0, nonmonotone=false)
        # #1117 / drmTMB allow_lower_boundary: the LR threshold was not crossed
        # before the log-SD floor. That is a one-sided [0, u] interval, not a
        # failed endpoint (confint must not throw) and not (-Inf, Inf).
        stats = (
            evaluations=evaluations,
            gradient_evaluations=gradient_evaluations,
            bracket_expansions=bracket_expansions,
            root_iterations=root_iterations,
            unbounded=true,
            nonmonotone=nonmonotone,
            endpoint_failed=false,
            nuisance_method=nuisance.method,
            nuisance_fallback=nuisance.fallback,
            nuisance_reason=:sd_boundary,
        )
        return -Inf, stats
    end
    coord(t) = θ̂[k] + dir * t
    at_sd_floor(t) = allow_lower_boundary && dir < 0 && coord(t) <= _PROFILE_SD_LOG_FLOOR
    t_floor = allow_lower_boundary && dir < 0 ? (θ̂[k] - _PROFILE_SD_LOG_FLOOR) : Inf
    dummy_boundary = (method=:not_checked, fallback=false, reason=:sd_boundary)
    if allow_lower_boundary && dir < 0 && θ̂[k] <= _PROFILE_SD_LOG_FLOOR
        return boundary_arm(dummy_boundary)
    end
    # Bracket: expand until h > 0. h(t) is assumed increasing (LR profile), but an
    # inexactly-solved inner nuisance optimisation can make it slightly non-monotone
    # (dip below `target` then rise). We track this and reset `tlo = 0` so
    # bisection works over the full observed bracket. This guarded local search
    # does not establish that the returned crossing is globally first.
    tlo = 0.0
    thi = s
    if at_sd_floor(thi)
        t_floor <= 0 && return boundary_arm(dummy_boundary)
        thi = t_floor
    end
    evalhi = heval(thi)
    if !evalhi.ok
        return at_sd_floor(thi) ? boundary_arm(evalhi.nuisance) : failed_arm(evalhi.nuisance)
    end
    hhi = evalhi.h
    hprev = hhi
    nonmonotone = false
    iters = 0
    while hhi < 0 && iters < 40
        tlo = thi
        tnext = thi * 1.6
        if at_sd_floor(tnext)
            t_floor <= thi && return boundary_arm(evalhi.nuisance, 0, iters, nonmonotone)
            tnext = t_floor
        end
        thi = tnext
        evalhi = heval(thi)
        if !evalhi.ok
            return at_sd_floor(thi) ? boundary_arm(evalhi.nuisance, 0, iters, nonmonotone) :
                failed_arm(evalhi.nuisance, 0, iters, nonmonotone)
        end
        hhi = evalhi.h
        hhi < hprev - 1e-12 && (nonmonotone = true)
        hprev = hhi
        iters += 1
        at_sd_floor(thi) && hhi < 0 && return boundary_arm(evalhi.nuisance, 0, iters, nonmonotone)
    end
    if hhi < 0
        if at_sd_floor(thi)
            return boundary_arm(evalhi.nuisance, 0, iters, nonmonotone)
        end
        # A searched-range limit is meaningful only when the profiled value is
        # separated from the LR target by more than its represented-NLL
        # cancellation.  Otherwise the sign itself is unresolved.
        if hhi + evalhi.cancellation >= 0
            uncertain = merge(evalhi.nuisance, (reason=:insufficient_precision,))
            return failed_arm(uncertain, 0, iters, nonmonotone)
        end
        value = dir < 0 ? -Inf : Inf
        stats = (
            evaluations=evaluations,
            gradient_evaluations=gradient_evaluations,
            bracket_expansions=iters,
            root_iterations=0,
            unbounded=true,
            nonmonotone=nonmonotone,
            endpoint_failed=false,
            nuisance_method=evalhi.nuisance.method,
            nuisance_fallback=evalhi.nuisance.fallback,
            nuisance_reason=evalhi.nuisance.reason,
        )
        return value, stats
    end
    nonmonotone && (tlo = 0.0)   # guarded local bisection over the full observed bracket
    # Guarded Newton on [tlo, thi]; if the profile is non-monotone fall back to pure
    # bisection (Newton can jump past the first crossing on a noisy slope).
    t = (tlo + thi) / 2
    root_iterations = 0
    # `converged` is set ONLY by the genuine convergence test `abs(ht) < 1e-9`
    # (DRModels.jl#493). The loop's OTHER exit, `thi - tlo < 1e-8`, is a bracket-
    # collapse safety valve, not a root: on a trapped, non-monotone profiled-nll
    # surface (the warm-started inner nuisance solve landing in a spurious local
    # optimum ~28 NLL units above the true profile, for every t on the affected
    # arm) `ht` stays pinned near its saturated value the entire time — measured
    # ht ≈ +25.85 at exit on seed 1's degenerate upper sigma arm — while `thi`
    # is repeatedly halved toward `tlo = 0`, exiting with a step of ~7e-9 that
    # looks like a converged endpoint but is not one.
    #
    # Convergence must be judged on the SCALE OF THE PROBLEM, not on an absolute
    # 1e-9 alone. `abs(ht) < 1e-9` is reachable only when the Newton branch is
    # live (`hp` finite and positive). On the BISECTION-ONLY path — `hp = NaN`,
    # i.e. `autodiff === :finite`, which the shipping q2 bivariate structured
    # route takes (`gaussian_bivariate.jl` attaches no gradient callback and
    # ForwardDiff genuinely throws on its inner sparse Cholesky) — pure halving
    # needs ~27 steps to drive an O(0.1) bracket under the 1e-8 floor, and that
    # floor is hit while `abs(ht)` is still ~1e-7. Judging that as failure
    # condemns a perfectly good endpoint: measured 10/12 arms on real q2 fits and
    # 29/30 on healthy Cell U fits forced onto `:finite`, every one of them with
    # a tightly clustered, plausible width.
    #
    # The two populations are separated by magnitude, not by exit route:
    #   legitimate bisection exit   abs(ht) ~ 1e-7 .. 1e-8
    #   genuinely trapped surface   abs(ht) ~ 2.0 .. 1e4   (>= `half` itself)
    # so endpoint certification compares its residual plus represented-NLL
    # cancellation against `max(1e-9, 1e-4 * half)`.  A profile whose NLL scale
    # cannot resolve that bound is reported as `:insufficient_precision`.
    converged = false
    ht = NaN
    final_nuisance = evalhi.nuisance
    final_cancellation = evalhi.cancellation
    endpoint_tolerance = max(1e-9, 1e-4 * abs(half))
    for _ in 1:60
        root_iterations += 1
        t_eval = t
        ev = heval(t_eval)
        ev.ok || return failed_arm(ev.nuisance, root_iterations, iters, nonmonotone)
        ht, hp = ev.h, ev.hp
        final_nuisance = ev.nuisance
        final_cancellation = ev.cancellation
        # Preserve the original accurate normal-root criterion.  The looser
        # cancellation-aware certificate is only for a bracket-collapse exit.
        if abs(ht) < 1e-9
            if final_cancellation > endpoint_tolerance
                uncertain = merge(final_nuisance, (reason=:insufficient_precision,))
                return failed_arm(uncertain, root_iterations, iters, nonmonotone)
            end
            converged = true
            t = t_eval
            break
        end
        ht < 0 ? (tlo = t_eval) : (thi = t_eval)
        tn = (!nonmonotone && isfinite(hp) && hp > 0) ? t_eval - ht / hp : (tlo + thi) / 2   # Newton, else bisect
        t = (tlo < tn < thi) ? tn : (tlo + thi) / 2                     # guard into bracket
        if thi - tlo < 1e-8
            # Bracket collapsed. That is a genuine root iff the residual is small
            # relative to the target; otherwise the surface never crossed and the
            # step is fabricated (DRModels.jl#493).
            converged = isfinite(ht) && abs(ht) + final_cancellation <= endpoint_tolerance
            converged && (t = t_eval)
            break
        end
    end
    # A failed endpoint mirrors the `unbounded` convention (±Inf toward `dir`,
    # never a fabricated finite value) rather than returning θ̂[k] + dir*t as if
    # it were a real root; `endpoint_failed` distinguishes it from a genuine
    # unbounded profile so callers can tell "doesn't cross" from "solver could
    # not tell where it crosses".
    # `t` is updated to the next trial after evaluating `ht`; return only the
    # coordinate that was actually evaluated and met the residual criterion.
    # A bracket-collapse acceptance retains the evaluated midpoint for the same
    # reason, rather than reporting an unevaluated next Newton step.
    if !converged && final_cancellation > endpoint_tolerance
        final_nuisance = merge(final_nuisance, (reason=:insufficient_precision,))
    end
    value = converged ? θ̂[k] + dir * t : (dir < 0 ? -Inf : Inf)
    stats = (
        evaluations=evaluations,
        gradient_evaluations=gradient_evaluations,
        bracket_expansions=iters,
        root_iterations=root_iterations,
        unbounded=false,
        nonmonotone=nonmonotone,
        endpoint_failed=!converged,
        nuisance_method=final_nuisance.method,
        nuisance_fallback=final_nuisance.fallback,
        nuisance_reason=final_nuisance.reason,
    )
    return value, stats
end

function _profile_row(job, nll, nllgrad, θ̂, nllhat, half, se, autodiff)
    row, _ = _profile_row_result(job, nll, nllgrad, θ̂, nllhat, half, se, autodiff)
    return row
end

function _binomial_sd_threaded_profile_unsupported(fit::DrmFit)
    fit.family isa Binomial || return false
    has_sd_block = any(block -> begin
        name = String(first(block))
        name == "sd" || startswith(name, "sd_") ||
            startswith(name, "sdtarget_")
    end, fit.blocks)
    has_sd_block || return false
    targets = _sd_target_parts(fit.formula)
    any(item -> item.target.level !== :iid, targets) && return true
    return !isempty(_sdphylo_parts(fit.formula))
end

function _profile_row_result(
    job, nll, nllgrad, θ̂, nllhat, half, se, autodiff;
    endpoint_threads::Bool=false,
)
    k = job.k
    est = θ̂[k]
    allow_lower = _is_log_sd_profile_param(job.param)
    s = _profile_usable_step(se[k], est)
    u0 = θ̂[[i for i in 1:length(θ̂) if i != k]]
    if endpoint_threads && Threads.nthreads() > 1
        left = Threads.@spawn _profile_endpoint_result(
            nll, nllgrad, θ̂, k, nllhat, half, s, -1, u0, autodiff;
            allow_lower_boundary=allow_lower,
        )
        right = Threads.@spawn _profile_endpoint_result(
            nll, nllgrad, θ̂, k, nllhat, half, s, +1, u0, autodiff
        )
        lo, lstats = fetch(left)
        hi, rstats = fetch(right)
    else
        lo, lstats = _profile_endpoint_result(
            nll, nllgrad, θ̂, k, nllhat, half, s, -1, u0, autodiff;
            allow_lower_boundary=allow_lower,
        )
        hi, rstats = _profile_endpoint_result(
            nll, nllgrad, θ̂, k, nllhat, half, s, +1, u0, autodiff
        )
    end
    rewritten = _rewrite_log_sd_boundary_ci(job, est, lo, hi, lstats, rstats)
    rewritten === nothing || return rewritten
    row = (param=job.param, coef=job.coef, estimate=est, lower=lo, upper=hi)
    stats = (
        param=job.param,
        coef=job.coef,
        evaluations=lstats.evaluations + rstats.evaluations,
        gradient_evaluations=lstats.gradient_evaluations + rstats.gradient_evaluations,
        bracket_expansions=lstats.bracket_expansions + rstats.bracket_expansions,
        root_iterations=lstats.root_iterations + rstats.root_iterations,
        lower_unbounded=lstats.unbounded,
        upper_unbounded=rstats.unbounded,
        nonmonotone=lstats.nonmonotone || rstats.nonmonotone,
        lower_endpoint_failed=lstats.endpoint_failed,
        upper_endpoint_failed=rstats.endpoint_failed,
        lower_nuisance_method=lstats.nuisance_method,
        upper_nuisance_method=rstats.nuisance_method,
        lower_nuisance_fallback=lstats.nuisance_fallback,
        upper_nuisance_fallback=rstats.nuisance_fallback,
        lower_nuisance_reason=lstats.nuisance_reason,
        upper_nuisance_reason=rstats.nuisance_reason,
    )
    return row, stats
end

# #1117: a log-SD lower arm that hit the floor (or failed there) is the
# SD-scale boundary 0. Report [0, exp(u)] so confint neither throws nor
# returns (-Inf, Inf). Interior (both-finite) resd rows stay on log σ_b.
function _rewrite_log_sd_boundary_ci(job, log_est, lo, hi, lstats, rstats)
    _is_log_sd_profile_param(job.param) || return nothing
    lower_boundary = (lo == -Inf) && (lstats.unbounded || lstats.endpoint_failed)
    lower_boundary || return nothing
    sd_est = log_est <= _PROFILE_SD_LOG_FLOOR ? 0.0 : exp(log_est)
    upper_ok = isfinite(hi) && !rstats.endpoint_failed && !rstats.unbounded
    sd_hi = upper_ok ? exp(hi) : Inf
    row = (param=job.param, coef=job.coef, estimate=sd_est, lower=0.0, upper=sd_hi)
    stats = (
        param=job.param,
        coef=job.coef,
        evaluations=lstats.evaluations + rstats.evaluations,
        gradient_evaluations=lstats.gradient_evaluations + rstats.gradient_evaluations,
        bracket_expansions=lstats.bracket_expansions + rstats.bracket_expansions,
        root_iterations=lstats.root_iterations + rstats.root_iterations,
        lower_unbounded=true,
        upper_unbounded=!upper_ok,
        nonmonotone=lstats.nonmonotone || rstats.nonmonotone,
        lower_endpoint_failed=false,
        upper_endpoint_failed=rstats.endpoint_failed,
        lower_nuisance_method=hasproperty(lstats, :nuisance_method) ? lstats.nuisance_method : :specialized,
        upper_nuisance_method=hasproperty(rstats, :nuisance_method) ? rstats.nuisance_method : :specialized,
        lower_nuisance_fallback=lstats.nuisance_fallback,
        upper_nuisance_fallback=rstats.nuisance_fallback,
        lower_nuisance_reason=:sd_boundary,
        upper_nuisance_reason=rstats.nuisance_reason,
    )
    return row, stats
end

# #1119: the percentile bootstrap interval for a sigma coefficient in a mixed model
# under-covers (measured 83 to 90% at a nominal 95%; profile 92 to 94%). The interval is
# kept as is; `bootstrap_ci` only warns, once per call, and points to the profile route.
_term_has_bar(t::FunctionTerm) = t.f === (|) || any(_term_has_bar, t.args)
_term_has_bar(t::Tuple) = any(_term_has_bar, t)
_term_has_bar(t::AbstractVector) = any(_term_has_bar, t)
_term_has_bar(_) = false

function _warn_bootstrap_sigma_percentile(formula, rows)
    hasproperty(formula, :forms) || return nothing
    any(p -> _term_has_bar(p.second), formula.forms) || return nothing
    any(r -> startswith(string(r.param), "sigma"), rows) || return nothing
    @warn "bootstrap_ci: the percentile interval for a `sigma` coefficient in a mixed model " *
          "(a model with random effects) is too narrow: in a simulation of a Gaussian " *
          "location-scale model with a mean random intercept it covered 83 to 90% at a " *
          "nominal 95%, against 92 to 94% for profile intervals. The percentile intervals " *
          "are returned unchanged; prefer `confint(fit, method = :profile)` for the " *
          "`sigma` terms."
    return nothing
end

"""
    bootstrap_ci(formula, family::Gaussian; data, B = 300, level = 0.95, rng = default_rng(), K = nothing, A = nothing, tree = nothing, coords = nothing, threads = false, failures = :error, check_converged = true, algorithm = :auto, g_tol = 1e-8)
    bootstrap_ci(formula, family; data, B = 300, level = 0.95, rng = default_rng(), K = nothing, A = nothing, tree = nothing, coords = nothing, threads = false, failures = :error, check_converged = true)
    bootstrap_ci(fit::DrmFit{<:Gaussian}; data, B = 300, level = 0.95, rng = default_rng(), K = nothing, A = nothing, tree = nothing, coords = nothing, threads = false, failures = :error, check_converged = true, algorithm = :auto, g_tol = 1e-8)
    bootstrap_ci(fit::DrmFit; data, B = 300, level = 0.95, rng = default_rng(), K = nothing, A = nothing, tree = nothing, coords = nothing, threads = false, failures = :error, check_converged = true)

Parametric bootstrap confidence intervals: fit the model, then `simulate` `B`
replicate responses, refit each, and take percentile intervals per coefficient.
Univariate-response models (fixed / random-effect / meta / structured). Same row
shape as [`confint`](@ref). Set `threads = true` to refit bootstrap replicates
in parallel. Pass through the structured-provider keywords (`K` / `A` / `tree` /
`tree1` / `tree2` / `coords`)
and, for Gaussian fits, the solver controls (`algorithm` / `g_tol`) exactly as
to [`drm`](@ref). Use `bootstrap_result` when you need attempted/used/failed
counts and per-replicate failure messages. A thrown refit is recorded; the
default `failures = :error` then aborts. Pass `failures = :skip` to keep the
successful draws.
If you already have
`fit = drm(...)`, pass the fit directly to avoid refitting the base model before
the bootstrap replicates. A homogeneous Toeplitz `temporal()` fit returns the
mean-coefficient rows only (drmTMB refuses the temporal bootstrap and defers
σ and lag-correlation intervals). `tree1` and `tree2` are used by the direct
Julia Binomial `phylo_interaction()` SD route.
The fit-based form refuses a fit with an explicit `DRMControl.optimizer`
before any bootstrap refit; the formula-based form keeps its default solver.

The intervals are plain percentile intervals. For a `sigma` coefficient in a mixed
model (a model with random effects) they are too narrow, and `bootstrap_ci` warns
and recommends `confint(fit, method = :profile)` (percentile covered 83 to 90% at a
nominal 95% in a simulated Gaussian location-scale model with a mean random
intercept; profile covered 92 to 94%).
"""
function bootstrap_ci(
    formula::DrmFormula,
    family::Gaussian;
    data,
    B::Real=300,
    level::Real=0.95,
    rng=default_rng(),
    K=nothing,
    A=nothing,
    tree=nothing,
    coords=nothing,
    threads::Bool=false,
    failures::Symbol=:error,
    check_converged::Bool=true,
    algorithm::Symbol=:auto,
    g_tol::Real=1e-8,
)
    rows = bootstrap_summary(
        formula,
        family;
        data,
        B,
        level,
        rng,
        K,
        A,
        tree,
        coords,
        threads,
        failures,
        check_converged,
        algorithm,
        g_tol,
    )
    _warn_bootstrap_sigma_percentile(formula, rows)
    return _bootstrap_ci_rows(rows)
end

# Family-agnostic parametric bootstrap — any family `simulate` supports.
# #480: K/A/tree/coords ARE forwardable here — #479 established that the non-Gaussian
# bootstrap can thread a structured covariance; the guard was a one-sided
# plumbing gap, not a real restriction. Same row shape as the Gaussian method
# and `confint`.
function bootstrap_ci(
    formula::DrmFormula,
    family;
    data,
    B::Real=300,
    level::Real=0.95,
    rng=default_rng(),
    K=nothing,
    A=nothing,
    tree=nothing,
    tree1=nothing,
    tree2=nothing,
    coords=nothing,
    threads::Bool=false,
    failures::Symbol=:error,
    check_converged::Bool=true,
)
    rows = bootstrap_summary(
        formula, family; data, B, level, rng, K, A, tree, tree1, tree2, coords, threads, failures,
        check_converged
    )
    _warn_bootstrap_sigma_percentile(formula, rows)
    return _bootstrap_ci_rows(rows)
end

function bootstrap_ci(
    fit::DrmFit;
    data,
    B::Real=300,
    level::Real=0.95,
    rng=default_rng(),
    K=nothing,
    A=nothing,
    tree=nothing,
    tree1=nothing,
    tree2=nothing,
    coords=nothing,
    threads::Bool=false,
    failures::Symbol=:error,
    check_converged::Bool=true,
)
    _refuse_selected_optimizer_refit(fit, "bootstrap_ci")
    rows = bootstrap_summary(
        fit; data, B, level, rng, K, A, tree, tree1, tree2, coords, threads, failures, check_converged
    )
    _warn_bootstrap_sigma_percentile(fit.formula, rows)
    return _bootstrap_ci_rows(rows)
end

function bootstrap_ci(
    fit::DrmFit{<:Gaussian};
    data,
    B::Real=300,
    level::Real=0.95,
    rng=default_rng(),
    K=nothing,
    A=nothing,
    tree=nothing,
    coords=nothing,
    threads::Bool=false,
    failures::Symbol=:error,
    check_converged::Bool=true,
    algorithm::Symbol=:auto,
    g_tol::Real=1e-8,
)
    _refuse_selected_optimizer_refit(fit, "bootstrap_ci")
    rows = bootstrap_summary(
        fit;
        data,
        B,
        level,
        rng,
        K,
        A,
        tree,
        coords,
        threads,
        failures,
        check_converged,
        algorithm,
        g_tol,
    )
    _warn_bootstrap_sigma_percentile(fit.formula, rows)
    return _bootstrap_ci_rows(rows)
end

"""
    bootstrap_summary(formula, family::Gaussian; data, B = 300, level = 0.95, rng = default_rng(), K = nothing, A = nothing, tree = nothing, coords = nothing, threads = false, failures = :error, check_converged = true, algorithm = :auto, g_tol = 1e-8)
    bootstrap_summary(formula, family; data, B = 300, level = 0.95, rng = default_rng(), K = nothing, A = nothing, tree = nothing, coords = nothing, threads = false, failures = :error, check_converged = true)
    bootstrap_summary(fit::DrmFit{<:Gaussian}; data, B = 300, level = 0.95, rng = default_rng(), K = nothing, A = nothing, tree = nothing, coords = nothing, threads = false, failures = :error, check_converged = true, algorithm = :auto, g_tol = 1e-8)
    bootstrap_summary(fit::DrmFit; data, B = 300, level = 0.95, rng = default_rng(), K = nothing, A = nothing, tree = nothing, coords = nothing, threads = false, failures = :error, check_converged = true)

Parametric bootstrap coefficient summaries in one pass: point estimate,
bootstrap standard error, and percentile confidence interval. This is the
bootstrap analogue of using `stderror(fit)` plus `confint(fit)`, avoiding a
second bootstrap run when both SEs and intervals are needed. Row fields are
`(param, coef, estimate, std_error, lower, upper)`. By default, a thrown refit
is recorded and then aborts (`failures = :error`). Set `failures = :skip` to
compute summaries from successful replicates; call `bootstrap_result` to
inspect the skipped failures. A homogeneous Toeplitz `temporal()` fit returns
the mean-coefficient rows only, as [`bootstrap_ci`](@ref). For the direct Julia
Binomial `phylo_interaction()` SD route, supply `tree1` and `tree2` to preserve
the fitted pair field during refits.
The fit-based form refuses a fit with an explicit `DRMControl.optimizer`
before any bootstrap refit.
"""
function bootstrap_summary(
    formula::DrmFormula,
    family::Gaussian;
    data,
    B::Real=300,
    level::Real=0.95,
    rng=default_rng(),
    K=nothing,
    A=nothing,
    tree=nothing,
    coords=nothing,
    threads::Bool=false,
    failures::Symbol=:error,
    check_converged::Bool=true,
    algorithm::Symbol=:auto,
    g_tol::Real=1e-8,
)
    result = bootstrap_result(
        formula,
        family;
        data,
        B,
        level,
        rng,
        K,
        A,
        tree,
        coords,
        threads,
        failures,
        check_converged,
        algorithm,
        g_tol,
    )
    return result.summary
end

# Family-agnostic summary method — any family `simulate` supports. #480: K/A/tree/coords
# thread through to `bootstrap_result`, which forwards them to `drm(...)` only
# when supplied — see the comment there for why they are not Gaussian-only.
function bootstrap_summary(
    formula::DrmFormula,
    family;
    data,
    B::Real=300,
    level::Real=0.95,
    rng=default_rng(),
    K=nothing,
    A=nothing,
    tree=nothing,
    tree1=nothing,
    tree2=nothing,
    coords=nothing,
    threads::Bool=false,
    failures::Symbol=:error,
    check_converged::Bool=true,
)
    result = bootstrap_result(
        formula, family; data, B, level, rng, K, A, tree, tree1, tree2, coords, threads, failures,
        check_converged
    )
    return result.summary
end

function bootstrap_summary(
    fit::DrmFit;
    data,
    B::Real=300,
    level::Real=0.95,
    rng=default_rng(),
    K=nothing,
    A=nothing,
    tree=nothing,
    tree1=nothing,
    tree2=nothing,
    coords=nothing,
    threads::Bool=false,
    failures::Symbol=:error,
    check_converged::Bool=true,
)
    _refuse_selected_optimizer_refit(fit, "bootstrap_summary")
    result = bootstrap_result(
        fit; data, B, level, rng, K, A, tree, tree1, tree2, coords, threads, failures, check_converged
    )
    return result.summary
end

function bootstrap_summary(
    fit::DrmFit{<:Gaussian};
    data,
    B::Real=300,
    level::Real=0.95,
    rng=default_rng(),
    K=nothing,
    A=nothing,
    tree=nothing,
    coords=nothing,
    threads::Bool=false,
    failures::Symbol=:error,
    check_converged::Bool=true,
    algorithm::Symbol=:auto,
    g_tol::Real=1e-8,
)
    _refuse_selected_optimizer_refit(fit, "bootstrap_summary")
    result = bootstrap_result(
        fit;
        data,
        B,
        level,
        rng,
        K,
        A,
        tree,
        coords,
        threads,
        failures,
        check_converged,
        algorithm,
        g_tol,
    )
    return result.summary
end

"""
    bootstrap_result(formula, family::Gaussian; data, B = 300, level = 0.95, rng = default_rng(), K = nothing, A = nothing, tree = nothing, coords = nothing, threads = false, failures = :error, check_converged = true, algorithm = :auto, g_tol = 1e-8, method = :ML, marginal = :LA, penalty = nothing)
    bootstrap_result(formula, family; data, B = 300, level = 0.95, rng = default_rng(), K = nothing, A = nothing, tree = nothing, coords = nothing, threads = false, failures = :error, check_converged = true, method = :ML, marginal = :LA, penalty = nothing)
    bootstrap_result(fit::DrmFit{<:Gaussian}; data, B = 300, level = 0.95, rng = default_rng(), K = nothing, A = nothing, tree = nothing, coords = nothing, threads = false, failures = :error, check_converged = true, algorithm = :auto, g_tol = 1e-8)
    bootstrap_result(fit; data, B = 300, level = 0.95, rng = default_rng(), K = nothing, A = nothing, tree = nothing, coords = nothing, threads = false, failures = :error, check_converged = true)

Auditable parametric bootstrap. Returns a `NamedTuple` with:

- `summary`: the same rows returned by `bootstrap_summary`;
- `failures`: rows `(replicate, seed, message)` for failed refits;
- `attempted`, `used`, `failed`: replicate counts;
- `status`: `"bootstrap"` when every replicate was kept, `"bootstrap_incomplete"`
  when any replicate was dropped and at least two remain, `"bootstrap_at_boundary"`
  when those retained draws pile up on a variance-component or correlation bound,
  or `"bootstrap_unavailable"` when exactly one replicate remains. If none
  remain, the call raises an error;
- `boundary_params`: `"param:coef"` names that triggered the boundary status;
- `seeds`: the per-replicate seeds used for reproducibility;
- `estimation_method`, `marginal`: the estimator of the seed fit (`:ML`, `:REML` or
  `:MAP`; `:LA`, `:Laplace`, `:VA` or `:AGHQ`);
- `refit_estimation_method`, `refit_marginal`: the estimator the retained
  replicates used (`:mixed` if their estimators differ). A warning marks a
  mismatch with a known seed estimator;
- `threaded`: whether threaded refits were actually used.
- `worker_threads`, `julia_threads`, `blas_threads`, `blas_oversubscribed`,
  `elapsed`: CPU context and wall-clock time for the simulated-refit phase.

`failures = :error` (default) records a thrown refit and then aborts. Pass
`failures = :skip` to keep the successful draws and return
`status = "bootstrap_incomplete"` with a warning. A refit that does
not converge, or that lands on a degenerate optimum, is not that kind of
error: with `check_converged = true` (the default) it is dropped, `status` is
`"bootstrap_incomplete"` (or `"bootstrap_at_boundary"` when the retained draws
also sit on a bound), and a warning names that status. A refit at a variance
boundary is kept: [`convergence_status`](@ref) calls it `:boundary`, but it is a
real draw. Set `check_converged = false` only to reproduce the old behaviour that counted
unconverged refits in the interval. Passing an existing `DrmFit`
reuses that point estimate as the bootstrap seed fit and starts directly with
the `B` simulated refits. `g_tol` controls the optimizer tolerance for the seed
fit and refits when a formula is supplied, and for refits when an existing fit
is supplied. The q4 Gaussian phylogenetic shortcut delegates to
`bootstrap_sigma_a` and does not use this `g_tol` argument; call that function
directly with `q4_g_tol` to change its optimizer tolerance. Other Gaussian
bootstrap refits also pass `algorithm`
through to `drm(...)`; this is useful for large structured models where `:auto`
selects a sparse route. For a homogeneous Toeplitz `temporal()` fit, `summary`
holds the mean-coefficient rows only, as [`bootstrap_ci`](@ref). For the direct
Julia Binomial `phylo_interaction()` SD route, pass the same `tree1` and `tree2`
to preserve the fitted product field in simulation and refits.
The fit-based form refuses a fit with an explicit `DRMControl.optimizer`
before any bootstrap refit. The fit-based form repeats the seed fit's REML,
marginal (`:Laplace`, `:VA`, `:AGHQ`) and MAP penalty in every replicate. The
formula-based form takes `method` (`:ML` or `:REML`), `marginal` and `penalty` and
uses them for the seed fit and the replicates alike (DRModels.jl#1025).
The selected family route must support any requested non-default estimator or
penalty. `algorithm` and `g_tol` are Gaussian-overload controls. A joint
`mi(x)` fit (`impute`/`missing` controls) is not bootstrapped: there is no
`bootstrap_result` method for `JointDrmFit`, and the call raises a `MethodError`
rather than refitting with default settings.

!!! warning "Silent conditional fallback (#997)"
    When the marginal simulator cannot be built, `bootstrap_result` draws replicates
    from the CONDITIONAL `simulate(fit)` (random effects held at `b = 0`), which
    collapses variance-component intervals (#459), and it does so **without a
    warning**. Known cases, kept as `@test_broken` in
    `test/test_marginal_simulator_fallback_997.jl`: a mean grouping column missing
    from `data`, a correlated slope `(1 + x | g)`, and several ordinary bars that
    repeat a group or mix intercepts with slopes. (Crossed ordinary intercepts on
    distinct groups, `(1 | g1) + (1 | g2)`, now get a proper marginal draw.) Only a `meta_V(...)` + random-effect fit
    throws. Pass the same `data` (and `tree` / `K` / `A`) used to fit, and treat
    variance-component intervals from these fits as unreliable.
"""
function bootstrap_result(
    formula::DrmFormula,
    family::Gaussian;
    data,
    B::Real=300,
    level::Real=0.95,
    rng=default_rng(),
    K=nothing,
    A=nothing,
    tree=nothing,
    coords=nothing,
    threads::Bool=false,
    failures::Symbol=:error,
    check_converged::Bool=true,
    algorithm::Symbol=:auto,
    g_tol::Real=1e-8,
    method::Symbol=:ML,
    marginal::Symbol=:LA,
    penalty=nothing,
)
    _check_bootstrap_failure_mode(failures)
    opts = _bootstrap_formula_options(method, marginal, penalty)
    fit0 = drm(formula, family; data, K, A, tree, coords, algorithm, g_tol, opts...)
    refit = datab -> drm(formula, family; data=datab, K, A, tree, coords, algorithm, g_tol, opts...)
    simulate_fn = _marginal_simulator(fit0, data; K, A, tree, coords)
    return _bootstrap_result(
        fit0, formula, data, B, level, rng, threads, refit;
        failures, check_converged, simulate_fn
    )
end

function bootstrap_result(
    fit::DrmFit{<:Gaussian};
    data,
    B::Real=300,
    level::Real=0.95,
    rng=default_rng(),
    K=nothing,
    A=nothing,
    tree=nothing,
    coords=nothing,
    threads::Bool=false,
    failures::Symbol=:error,
    check_converged::Bool=true,
    algorithm::Symbol=:auto,
    g_tol::Real=1e-8,
)
    _refuse_selected_optimizer_refit(fit, "bootstrap_result")
    # Bivariate q=4 phylogenetic fit (also a DrmFit{<:Gaussian}): no scalar SD
    # block to refit-and-recoef — the quantities of interest are the among-axis
    # SDs sqrt.(diag(Σ_a)). Route to the dedicated parametric bootstrap, which
    # gives boundary-honest percentile CIs. Covariance providers are carried by the fit.
    if fit.formula isa BivariateDrmFormula && fit.ranef isa NamedTuple &&
       haskey(fit.ranef, :Sigma_a)
        return bootstrap_sigma_a(fit; data = data, B = B, level = level, rng = rng,
                                 failures = (failures === :error ? :error : :warn),
                                 check_converged = check_converged)
    end
    _check_bootstrap_failure_mode(failures)
    formula = _bootstrap_fit_formula(fit)
    # LSS refits must preserve the seed fit's estimator. Other Gaussian routes
    # forward REML, a non-default marginal, and a stored phylo penalty (MAP).
    # On the univariate refit, a non-default marginal (:Laplace, :VA, :AGHQ)
    # and a REML seed are forwarded too. Otherwise the replicate is the default
    # :LA / ML fit and the interval describes a different estimator
    # (DRModels.jl#1038, #1025).
    refit_options = _bootstrap_refit_kwargs(fit)
    # `drm(::BivariateDrmFormula, ::Gaussian; ...)` accepts `method` in
    # (:ML, :REML) and does not accept `algorithm` or `marginal`
    # (src/gaussian_bivariate.jl). Either of those keywords throws a
    # MethodError on the first replicate. A bivariate formula is not a
    # univariate LSS `DrmFormula`, so `_is_gaussian_lss` is false, but a
    # REML seed still stores `method => :REML` in `refit_options`. Pass
    # `method` only when that keyword is present, and leave `marginal` off
    # this call.
    refit = if formula isa BivariateDrmFormula
        biv_kw = _bootstrap_bivariate_refit_kwargs(refit_options)
        datab -> drm(formula, fit.family; data=datab, K, A, tree, coords, g_tol, biv_kw...)
    else
        datab -> drm(formula, fit.family; data=datab, K, A, tree, coords, algorithm, g_tol, refit_options...)
    end
    # #459: redraw the random effects rather than conditioning on the fitted BLUPs.
    # For the residual-only bivariate fit this returns `nothing` (no random
    # effects to marginalise), so the conditional `simulate` is used — correct,
    # because that fit's only stochastic part IS the residual pair.
    simulate_fn = _marginal_simulator(fit, data; K=K, A=A, tree=tree, coords=coords)
    return _bootstrap_result(
        fit, formula, data, B, level, rng, threads, refit;
        failures, check_converged, simulate_fn
    )
end

function bootstrap_result(
    formula::DrmFormula,
    family;
    data,
    B::Real=300,
    level::Real=0.95,
    rng=default_rng(),
    K=nothing,
    A=nothing,
    Ainv=nothing,
    tree=nothing,
    tree1=nothing,
    tree2=nothing,
    coords=nothing,
    threads::Bool=false,
    failures::Symbol=:error,
    check_converged::Bool=true,
    g_tol::Real=1e-8,
    method::Symbol=:ML,
    marginal::Symbol=:LA,
    penalty=nothing,
)
    _check_bootstrap_failure_mode(failures)
    # #480: mirror #479's fit-based fix. Forward a structured-matrix keyword to
    # `drm(...)` only when the caller actually supplied it -- most non-Gaussian
    # `drm` methods (LogNormal, Tweedie, Student, SkewNormal, ZeroOneBeta,
    # CumulativeLogit, TruncatedNegBinomial2) do not declare `K`/`A`/`tree` at
    # all, so forwarding them unconditionally (even as `nothing`) would throw a
    # MethodError on every ordinary unstructured non-Gaussian fit. The
    # structured routes (Poisson, Binomial, Gamma, Beta, BetaBinomial,
    # NegBinomial2, Gaussian) accept and use them; `drm(...)`'s own per-family
    # checks (e.g. "phylo(1 | g) needs `tree = …`") catch a genuine mismatch
    # loudly instead of silently refitting an unstructured model.
    extra = Dict{Symbol,Any}()
    tree !== nothing && (extra[:tree] = tree)
    tree1 !== nothing && (extra[:tree1] = tree1)
    tree2 !== nothing && (extra[:tree2] = tree2)
    K !== nothing && (extra[:K] = K)
    A !== nothing && (extra[:A] = A)
    Ainv !== nothing && (extra[:Ainv] = Ainv)
    coords !== nothing && (extra[:coords] = coords)
    # #1025: only a non-default estimator is forwarded, as `drm` for some
    # families does not declare `method`, `marginal` or `penalty`.
    for (k, v) in pairs(_bootstrap_formula_options(method, marginal, penalty))
        extra[k] = v
    end
    fit0 = drm(formula, family; data, extra..., g_tol)
    refit = datab -> drm(formula, family; data=datab, extra..., g_tol)
    # #459/#479: redraw the random effects rather than conditioning on the
    # fitted BLUPs, so a variance-component bootstrap CI is not degenerate.
    simulate_fn = _marginal_simulator(fit0, data; K=K, A=A, Ainv=Ainv,
                                      tree=tree, tree1=tree1, tree2=tree2,
                                      coords=coords)
    return _bootstrap_result(
        fit0, formula, data, B, level, rng, threads, refit;
        failures, check_converged, simulate_fn
    )
end

function bootstrap_result(
    fit::DrmFit;
    data,
    B::Real=300,
    level::Real=0.95,
    rng=default_rng(),
    K=nothing,
    A=nothing,
    Ainv=nothing,
    tree=nothing,
    tree1=nothing,
    tree2=nothing,
    coords=nothing,
    threads::Bool=false,
    failures::Symbol=:error,
    check_converged::Bool=true,
    g_tol::Real=1e-8,
)
    _refuse_selected_optimizer_refit(fit, "bootstrap_result")
    # Bivariate q=4 phylogenetic fit: there is no scalar SD block to refit-and-
    # recoef; the quantities of interest are the among-axis SDs sqrt.(diag(Σ_a)).
    # Route to the dedicated parametric bootstrap (boundary-honest percentile CIs).
    if fit.formula isa BivariateDrmFormula && fit.ranef isa NamedTuple &&
       haskey(fit.ranef, :Sigma_a)
        # Covariance providers are carried by the fit (fit.ranef.phy) — accept and ignore them.
        return bootstrap_sigma_a(fit; data = data, B = B, level = level, rng = rng,
                                 failures = (failures === :error ? :error : :warn),
                                 check_converged = check_converged)
    end
    _check_bootstrap_failure_mode(failures)
    formula = _bootstrap_fit_formula(fit)
    # #479: the univariate non-Gaussian structured routes (phylo/relmat/animal/
    # spatial Laplace) do NOT stash the tree/K/A on the fit the way the bivariate
    # q=4 route stashes `fit.ranef.phy` -- `_fit_poisson_general_laplace` et al.
    # never call `_withranef`. So, exactly like the Gaussian method just above,
    # the caller re-supplies the same K/A/tree/coords used to produce `fit`; a mismatch
    # (or an unsupported family/route) is caught loudly by `drm(...)`'s own
    # per-family checks (e.g. "relmat(1 | g) needs K = …") rather than silently
    # refitting an unstructured model.
    extra = Dict{Symbol,Any}()
    tree !== nothing && (extra[:tree] = tree)
    tree1 !== nothing && (extra[:tree1] = tree1)
    tree2 !== nothing && (extra[:tree2] = tree2)
    K !== nothing && (extra[:K] = K)
    A !== nothing && (extra[:A] = A)
    coords !== nothing && (extra[:coords] = coords)
    # Repeat the seed fit's integrator and REML setting. :LA / ML are the
    # defaults, so only a different choice is forwarded (DRModels.jl#1038, #1025).
    for (k, v) in pairs(_bootstrap_refit_kwargs(fit))
        extra[k] = v
    end
    binomial_nq = pop!(extra, :nq, nothing)
    if fit.family isa Binomial && binomial_nq !== nothing
        # Binomial SD fits expose their quadrature order only through the
        # internal fit route. Preserve it on bootstrap refits rather than
        # silently switching to the public route's default order.
        refit = datab -> _drm_binomial_sd_regression(
            formula, fit.family; data=datab, K, A, Ainv, tree, tree1, tree2, coords,
            marginal=fit.marginal, nq=binomial_nq, g_tol,
        )
    else
        Ainv !== nothing && (extra[:Ainv] = Ainv)
        refit = datab -> drm(formula, fit.family; data=datab, extra..., g_tol)
    end
    simulate_fn = _marginal_simulator(fit, data; K=K, A=A, Ainv=Ainv, tree=tree,
                                      tree1=tree1, tree2=tree2, coords=coords)   # #459 / #479
    return _bootstrap_result(
        fit, formula, data, B, level, rng, threads, refit;
        failures, check_converged, simulate_fn
    )
end

# The fit-based bootstrap refits from the fit's own stored formula. A univariate
# `DrmFormula` has always been accepted; the RESIDUAL-ONLY bivariate Gaussian fit
# (`bf(mu1 = ..., mu2 = ..., sigma1 = ..., sigma2 = ..., rho12 = ...)` with no
# structured term) is accepted too — it stores a `BivariateDrmFormula`, has no
# random effects (`fit.ranef === nothing`), and so is a plain refit-and-recoef
# problem exactly like the univariate case. A STRUCTURED bivariate fit is NOT
# admitted here: every `bootstrap_result` method tests
# `fit.ranef isa NamedTuple && haskey(fit.ranef, :Sigma_a)` first and routes the
# q=4 phylogenetic fit to `bootstrap_sigma_a`, so it never reaches this function.
# A `DrmFit` with no `.formula` at all (an internal, formula-less fit) is still
# refused: there is nothing to refit from.
function _bootstrap_fit_formula(fit::DrmFit)
    (fit.formula isa DrmFormula || fit.formula isa BivariateDrmFormula) &&
        return fit.formula
    throw(
        ArgumentError(
            "fit-based bootstrap requires a univariate `DrmFit` created by `drm`; " *
            "bivariate and formula-less internal fits are not yet supported",
        ),
    )
end

# --- Marginal parametric bootstrap simulation (#459) -------------------------
#
# `simulate(fit; re = :zero)` (before #1092 the only `simulate`) is a CONDITIONAL
# simulator: for a Gaussian fit it returns
# `fit.means[:mu] .+ fit.scales[:sigma] .* randn(n)`, and `fit.means[:mu]` ALREADY
# CONTAINS the fitted BLUPs. Bootstrapping a fixed effect that way is defensible.
# Bootstrapping a VARIANCE COMPONENT that way is not: every replicate re-uses the
# same realised random effects, so the refitted SD barely moves and the percentile
# interval collapses toward the point estimate.
#
# Measured 2026-08-24 (#459): the phylo-SD bootstrap CI came out 1674x NARROWER
# than native TMB on identical data, B and seed -- implied replicate SD 6.45e-05
# against 0.108. The point estimate was right, which is why it looked plausible.
#
# The correct parametric bootstrap redraws the random effects from their fitted
# distribution and adds them to the FIXED-effect mean:
#
#     y* = Xb + Z u*  + e*,    u* ~ N(0, sd_u^2 K),   e* ~ N(0, sigma^2)
#
# which is exactly what `bootstrap_q4_phylo.jl` already does for the q=4 path
# ("redraws tip random effects from the fitted N(0, Q_cond^-1 (x) Sigma_a) ... adds
# them to the fitted fixed effects"). The univariate path simply never followed it.
#
# Returns a closure `rng -> ysim`, or `nothing` when the fit has NO random effects
# (there conditional and marginal simulation coincide and `simulate` is correct).
# Gaussian LSS bootstrap uses the full marginal model, not fitted random effects.
# Prepared arrays are read-only; each call allocates its own draws and response.
# Keywords a bootstrap replicate must repeat so it is the same estimator as
# the seed fit. `:LA` and `:ML` are the `drm` defaults and are omitted.
# Gaussian location-scale-scale passes the seed method explicitly, `:ML` or
# `:REML`. That route rejects any other method. A penalized MAP seed is
# repeated by forwarding the stored `penalty` (the fitter does not take
# `method = :MAP`).
function _bootstrap_refit_kwargs(fit::DrmFit)
    _is_fixed_prior_fit(fit) && fit.estim_method === :MAP && throw(ArgumentError("bootstrap: $(_prior_kind_label(fit)) refits are not yet qualified"))
    kw = Pair{Symbol,Any}[]
    if _is_gaussian_lss(fit)
        method = estimation_method(fit)
        method in (:ML, :REML) ||
            throw(ArgumentError("LSS bootstrap supports ML/REML seed fits only"))
        push!(kw, :method => method)
    elseif estimation_method(fit) === :REML
        push!(kw, :method => :REML)
    elseif estimation_method(fit) === :MAP
        fit.penalty === nothing && throw(ArgumentError(
            "bootstrap refit: this fit is :MAP but stores no penalty, so the " *
            "replicate cannot repeat the penalized estimator"))
    end
    # SD-regression routes with explicit public `nAGQ` support accept that
    # setting on bootstrap refits (currently Poisson, NB2, zero-truncated NB2,
    # and SkewNormal).
    # Binomial also records its private quadrature count, but its public `drm`
    # method does not accept that keyword, so passing it would break refits.
    if (fit.family isa Poisson || fit.family isa TruncatedPoisson ||
        fit.family isa NegBinomial2 ||
        fit.family isa TruncatedNegBinomial2 || fit.family isa SkewNormal) &&
       haskey(fit.scales, :sd_group) && haskey(fit.scales, :aghq_nodes)
        push!(kw, :nAGQ => round(Int, only(fit.scales[:aghq_nodes])))
    end
    if fit.family isa Binomial && haskey(fit.scales, :sd_group) &&
       haskey(fit.scales, :aghq_nodes)
        push!(kw, :nq => round(Int, only(fit.scales[:aghq_nodes])))
    end
    if _is_fixed_prior_fit(fit)
        push!(kw, :priors => fit.penalty.specs)
    elseif fit.penalty !== nothing
        push!(kw, :penalty => fit.penalty)
    end
    fit.marginal === :LA || push!(kw, :marginal => fit.marginal)
    weights(fit) === nothing || push!(kw, :weights => weights(fit))   # #931: same weighted likelihood
    return NamedTuple(kw)
end

# Estimator keywords the formula-based `bootstrap_result` forwards to `drm` for
# the seed fit and every replicate. Defaults (`:ML`, `:LA`, no penalty) are left
# off, matching `_bootstrap_refit_kwargs`. A MAP fit is requested with `penalty`,
# not `method = :MAP`.
function _bootstrap_formula_options(method::Symbol, marginal::Symbol, penalty)
    method in (:ML, :REML) || throw(ArgumentError(
        "bootstrap_result: `method` must be :ML or :REML (got :$method); " *
        "request a penalized MAP fit with `penalty`"))
    kw = Pair{Symbol,Any}[]
    method === :ML || push!(kw, :method => method)
    marginal === :LA || push!(kw, :marginal => marginal)
    penalty === nothing || push!(kw, :penalty => penalty)
    return NamedTuple(kw)
end

# The estimator a fit used, for the record on a bootstrap result. A fit type
# that carries no estimator (a joint-model fit) reports `:unknown`.
_bootstrap_estimator(fit::DrmFit) = (estimation_method(fit), fit.marginal)
_bootstrap_estimator(fit) = (:unknown, :unknown)

# Keywords the bivariate Gaussian bootstrap refit may pass to `drm`.
# `method` and `weights` are included only when the seed kwargs contain them. `marginal` is
# left off, because that method does not accept it.
function _bootstrap_bivariate_refit_kwargs(refit_options)
    kw = Pair{Symbol,Any}[]
    haskey(refit_options, :method) && push!(kw, :method => refit_options.method)
    # Prior weights are part of the likelihood: every replicate refit uses them, as the
    # univariate path does (`_bootstrap_refit_kwargs`).
    haskey(refit_options, :weights) && push!(kw, :weights => refit_options.weights)
    return NamedTuple(kw)
end

_is_gaussian_lss(fit::DrmFit) = fit.family isa Gaussian &&
    fit.formula isa DrmFormula &&
    (!isempty(_sd_parts(fit.formula)) || !isempty(_sdphylo_parts(fit.formula)))

function _lss_marginal_simulator(fit::DrmFit, data; tree=nothing)
    f = fit.formula
    rhs = Dict(f.forms)
    fixed_mu, re, metav, structured = _split_ranef(rhs[:mu])
    fixed_sigma, sigma_re, _, structured_sigma = _split_ranef(rhs[:sigma])
    (metav === nothing && isempty(sigma_re) && structured_sigma === nothing) ||
        throw(ArgumentError("LSS bootstrap requires ordinary mean effects and fixed residual-scale predictors"))
    re_kinds = [_re_kind(r[1]) for r in re]
    all(item -> first(item) in (:intercept, :slope), re_kinds) ||
        throw(ArgumentError("LSS bootstrap supports ordinary intercepts and independent numeric slopes only"))
    has_slope = any(item -> first(item) === :slope, re_kinds)
    has_slope && !(length(re) == 1 && structured === nothing) &&
        throw(ArgumentError("LSS bootstrap supports an independent random slope only as the sole ordinary mean effect"))
    raw_y = _table_column(data, f.response)
    _, observed = _coerce_response_column(raw_y)
    count(observed) == fit.nobs ||
        throw(ArgumentError("LSS bootstrap observed response count does not match the seed fit"))
    _, Xmu, nm_mu = _design(f.response, fixed_mu, data)
    _, Xsigma, nm_sigma = _design(f.response, fixed_sigma, data)
    names = Dict(fit.coefnames)
    get(names, :mu, String[]) == nm_mu && get(names, :sigma, String[]) == nm_sigma ||
        throw(ArgumentError("LSS bootstrap mean/scale design names do not match the seed fit"))
    mu = Xmu * coef(fit, :mu)
    sigma = exp.(Xsigma * coef(fit, :sigma))
    length(mu) == length(sigma) == length(observed) ||
        throw(ArgumentError("LSS bootstrap designs must preserve the full input rows"))
    all(isfinite, mu) && all(x -> isfinite(x) && x > 0, sigma) ||
        throw(ArgumentError("LSS bootstrap requires finite means and positive finite residual scales"))
    sdmap = Dict(_sd_parts(f))
    sdphy = _sdphylo_parts(f)
    # IID blocks precede phylogenetic blocks, matching the fitted parameter layout.
    components = NamedTuple[]
    aiid = isempty(re) ? Float64[] : coef(fit, :sd)
    offset = 0
    expected_names = String[]
    for (j, (_, grp)) in enumerate(re)
        labels = _table_column(data, grp)
        any(ismissing, labels) && throw(ArgumentError("LSS bootstrap grouping `$grp` contains missing labels"))
        gidx, G = _group_index(labels)
        kind, slope = re_kinds[j]
        loading = if kind === :intercept
            ones(Float64, length(gidx))
        else
            hasproperty(data, slope) || throw(ArgumentError(
                "LSS bootstrap data are missing random-slope predictor `$slope`"))
            raw = getproperty(data, slope)
            raw isa AbstractVector{<:Real} && length(raw) == length(gidx) && all(isfinite, raw) ||
                throw(ArgumentError("LSS bootstrap random-slope predictor `$slope` must be finite numeric values, one per response"))
            Float64.(raw)
        end
        Zg, nm = haskey(sdmap, grp) ?
            _sd_group_design(f.response, sdmap[grp], data, gidx, G, grp) :
            (ones(G, 1), ["(Intercept)"])
        width = size(Zg, 2)
        offset + width <= length(aiid) ||
            throw(ArgumentError("LSS bootstrap iid SD coefficients do not match the group designs"))
        scale = exp.(Zg * aiid[offset+1:offset+width])
        push!(components, (; gidx, G, scale, factor=nothing, loading))
        append!(expected_names, length(re)>1 ? string.(grp, ": ", nm) : nm)
        offset += width
    end
    offset == length(aiid) ||
        throw(ArgumentError("LSS bootstrap has unused iid SD coefficients"))
    isempty(re) || get(names, :sd, String[]) == expected_names ||
        throw(ArgumentError("LSS bootstrap iid SD coefficient names do not match the group designs"))
    if structured !== nothing
        kind, grp = structured
        kind === :phylo || throw(ArgumentError("LSS bootstrap supports the phylogenetic structured level only"))
        tree === nothing && throw(ArgumentError("LSS bootstrap with phylo(1 | $grp) requires `tree`"))
        phy, gidx, G = _lss_phylo_group_index(tree, _table_column(data, grp), grp)
        Zg, nm = if isempty(sdphy)
            (ones(G, 1), ["(Intercept)"])
        else
            sgrp, srhs = only(sdphy)
            sgrp === grp || throw(ArgumentError("LSS bootstrap phylogenetic SD grouping does not match the mean"))
            _sd_group_design(f.response, srhs, data, gidx, G, grp)
        end
        aphy = coef(fit, :sd_phylo)
        length(aphy) == size(Zg, 2) && get(names, :sd_phylo, String[]) == nm ||
            throw(ArgumentError("LSS bootstrap phylogenetic SD coefficients do not match the group design"))
        scale = exp.(Zg * aphy)
        # Every LSS phylogenetic route defines its SD against correlation,
        # including a scalar phylogenetic component in a multi-component fit.
        factor = cholesky(Symmetric(_phylo_correlation(phy))).L
        push!(components, (; gidx, G, scale, factor, loading=ones(Float64, length(gidx))))
    elseif !isempty(sdphy)
        throw(ArgumentError("LSS bootstrap phylogenetic SD needs a phylogenetic mean effect"))
    end
    isempty(components) && throw(ArgumentError("LSS bootstrap found no mean variance component"))
    all(c -> all(isfinite, c.scale), components) ||
        throw(ArgumentError("LSS bootstrap requires finite random-effect scales"))
    return function (rng)
        out = copy(mu)
        for c in components
            noise = randn(rng, c.G)
            u = c.scale .* (c.factor === nothing ? noise : c.factor * noise)
            out .+= c.loading .* u[c.gidx]
        end
        out .+= sigma .* randn(rng, length(out))
        all(isfinite, out) || throw(ArgumentError("LSS bootstrap produced a nonfinite response"))
        out[.!observed] .= NaN
        return out
    end
end

function _marginal_simulator(fit::DrmFit, data; K=nothing, A=nothing, tree=nothing,
                             tree1=nothing, tree2=nothing,
                             Ainv=nothing, coords=nothing)
    sim = _marginal_simulator_build(fit, data; K, A, Ainv, tree, tree1, tree2, coords)
    sim === nothing && _refuse_multi_re_laplace(fit, "a simulation or bootstrap with fresh random effects")
    # `nothing` sends `bootstrap` to the CONDITIONAL `simulate`. On the Gaussian
    # `meta_V(...)` + random-effect route (`_fit_meta_gaussian_re`) that fallback
    # is a different model: `means[:mu]` is Xβ and `scales[:sigma]` is √(v + σ²),
    # so the conditional draw has NO random field at all. Refuse instead.
    if sim === nothing && _is_meta_gaussian_re(fit)
        throw(ArgumentError("bootstrap: could not build the marginal simulator for this " *
            "`meta_V(...)` + random-effect fit (pass the same `data` and `tree` / `K` / `A` " *
            "used to fit it); the conditional fallback would drop the random effect"))
    end
    return sim
end

function _is_meta_gaussian_re(fit::DrmFit)
    (fit.family isa Gaussian && fit.formula isa DrmFormula) || return false
    rhs = Dict(fit.formula.forms)
    haskey(rhs, :mu) || return false
    _, re, metav, structured, _ = _split_ranef(rhs[:mu]; allow_phylo_slope = true)
    return metav !== nothing && (!isempty(re) || structured !== nothing)
end

function _binomial_mesh_sd_marginal_simulator(fit::DrmFit, data, mesh_metadata)
    f = fit.formula
    f isa DrmFormula || throw(ArgumentError(
        "bootstrap: Binomial mesh SD simulation requires the fitted formula."))
    group = mesh_metadata.group
    hasproperty(data, group) || throw(ArgumentError(
        "bootstrap: data are missing mesh spatial group `$group`."))
    projection, precision, row_ids = _binomial_mesh_inputs(
        mesh_metadata, data, f.response)
    fitted_ids = mesh_metadata.alignment_ids
    length(fitted_ids) == length(fit.scales[:sd]) || throw(ArgumentError(
        "bootstrap: fitted mesh SD values and alignment IDs do not match."))
    sd_by_id = Dict(id => value for (id, value) in zip(fitted_ids, fit.scales[:sd]))
    all(id -> haskey(sd_by_id, id), row_ids) || throw(ArgumentError(
        "bootstrap: mesh simulation rows do not match the fitted SD rows."))
    row_sd = Float64[sd_by_id[id] for id in row_ids]
    length(row_sd) == fit.nobs || throw(ArgumentError(
        "bootstrap: mesh simulation requires the fitted observation rows."))
    qchol = cholesky(Symmetric(precision); check = false)
    issuccess(qchol) || throw(ArgumentError(
        "bootstrap: stored mesh precision is not positive definite."))
    eta_fixed = predict(fit, data; type = :link)
    length(eta_fixed) == length(row_sd) || throw(ArgumentError(
        "bootstrap: mesh fixed predictions and SD rows do not align."))
    return function (rng)
        # `.UP` undoes CHOLMOD's fill-reducing permutation, so the field
        # has covariance precision^-1; `.U` alone would not.
        field = qchol.UP \ randn(rng, size(precision, 1))
        eta = eta_fixed + row_sd .* (projection * field)
        return _simulate_once(fit, rng; mu = _binomial_mean.(Ref(fit.family), eta))
    end
end

function _binomial_sd_marginal_simulator(fit::DrmFit, data, re;
                                         structured = nothing,
                                         phylo_slope = nothing,
                                         animal_slope = nothing,
                                         spatial_slope = nothing,
                                         K = nothing,
                                         A = nothing,
                                         Ainv = nothing,
                                         tree = nothing,
                                         coords = nothing)
    if fit.ranef isa NamedTuple && haskey(fit.ranef, :mesh)
        return _binomial_mesh_sd_marginal_simulator(fit, data, fit.ranef.mesh)
    end
    f = fit.formula
    targets = _sd_target_parts(f)
    related_targets = filter(item -> item.target.level in (:animal, :relmat, :spatial), targets)
    animal_targets = filter(item -> item.target.level === :animal, targets)
    iid_targets = filter(item -> item.target.level === :iid, targets)
    if length(animal_targets) == 1 && length(iid_targets) == 1 && length(targets) == 2
        target = only(animal_targets)
        descriptor = target.target
        group = descriptor.group
        descriptor.dpar === :mu && descriptor.member === nothing &&
            descriptor.block === nothing || throw(ArgumentError(
                "bootstrap: Binomial animal-plus-IID simulation requires an animal mean-intercept SD target"))
        structured == (:animal, group) || throw(ArgumentError(
            "bootstrap: Binomial animal-plus-IID simulation requires one matching animal effect"))
        length(re) == 1 && first(_re_kind(only(re)[1])) === :intercept ||
            throw(ArgumentError("bootstrap: Binomial animal-plus-IID simulation requires one IID random intercept"))
        iid_group = only(iid_targets).target.group
        last(only(re)) === iid_group || throw(ArgumentError(
            "bootstrap: the IID SD target must match random-intercept group `$iid_group`"))
        animal_loading = if animal_slope === nothing
            ones(Float64, fit.nobs)
        else
            hasproperty(data, animal_slope) || throw(ArgumentError(
                "bootstrap: data are missing animal slope predictor `$animal_slope`"))
            raw = getproperty(data, animal_slope)
            raw isa AbstractVector{<:Real} && length(raw) == fit.nobs &&
                all(isfinite, raw) || throw(ArgumentError(
                "bootstrap: animal slope predictor `$animal_slope` must be numeric, finite, and have one value per response"))
            Float64.(raw)
        end
        Ainv !== nothing && A !== nothing && throw(ArgumentError(
            "bootstrap: supply only one of `A` and `Ainv` for Binomial animal simulation"))
        matrix = Ainv === nothing ? A : Ainv
        matrix === nothing && throw(ArgumentError(
            "bootstrap: supply the same animal covariance `A` or precision `Ainv` used to fit the model"))
        hasproperty(data, group) && hasproperty(data, iid_group) || throw(ArgumentError(
            "bootstrap: data are missing an animal or IID grouping column"))
        animal_index, G = _group_index(getproperty(data, group))
        iid_index, H = _group_index(getproperty(data, iid_group))
        length(animal_index) == length(iid_index) == fit.nobs || throw(ArgumentError(
            "bootstrap: animal/IID group indexes do not match fitted response rows"))
        C = Matrix{Float64}(matrix)
        size(C) == (G, G) || throw(ArgumentError(
            "bootstrap: animal matrix must match first-seen animal group order"))
        factor = if Ainv === nothing
            _checked_relmat_chol(C, group)
        else
            Q = sparse(C)
            all(isfinite, nonzeros(Q)) || throw(ArgumentError(
                "bootstrap: animal `Ainv` must be finite"))
            norm(Q - transpose(Q), Inf) <= 1e-10 * max(norm(Q, Inf), 1.0) ||
                throw(ArgumentError("bootstrap: animal `Ainv` must be symmetric"))
            Q = sparse(Symmetric((Q + transpose(Q)) / 2))
            qchol = cholesky(Symmetric(Q); check = false)
            issuccess(qchol) || throw(ArgumentError(
                "bootstrap: animal `Ainv` must be positive definite"))
            qchol
        end
        iid_target = only(iid_targets)
        Wa, names_a = _sd_group_design(f.response, target.rhs, data,
            animal_index, G, group; schema_cache = f.schema_cache,
            schema_key = target.key, store_schema = false)
        Wi, names_i = _sd_group_design(f.response, iid_target.rhs, data,
            iid_index, H, iid_group; schema_cache = f.schema_cache,
            schema_key = iid_target.key, store_schema = false)
        blocks = _sd_target_blocks(targets)
        animal_block = _sd_target_block(target.target; disambiguate = true)
        iid_block = _sd_target_block(iid_target.target; disambiguate = true)
        names_a == _block_coefnames(fit, animal_block) || throw(ArgumentError(
            "bootstrap: animal SD design does not match fitted block `$animal_block`"))
        names_i == _block_coefnames(fit, iid_block) || throw(ArgumentError(
            "bootstrap: IID SD design does not match fitted block `$iid_block`"))
        animal_block in blocks && iid_block in blocks || throw(ArgumentError(
            "bootstrap: fitted SD target identities do not match the animal/IID formulas"))
        sd_animal = _binomial_sd_scale(Wa * coef(fit, animal_block))
        sd_iid = _binomial_sd_scale(Wi * coef(fit, iid_block))
        eta_fixed = Vector{Float64}(predict(fit, data; type = :link))
        length(eta_fixed) == fit.nobs || throw(ArgumentError(
            "bootstrap: fixed Binomial predictions do not match fitted response rows"))
        return function (rng)
            animal_noise = Ainv === nothing ? factor.L * randn(rng, G) :
                _binomial_animal_precision_draw(rng, factor, G)
            iid_noise = randn(rng, H)
            eta = eta_fixed .+ animal_loading .* sd_animal[animal_index] .*
                  animal_noise[animal_index] .+
                  sd_iid[iid_index] .* iid_noise[iid_index]
            return _simulate_once(fit, rng; mu = _mean_response(fit.family, eta))
        end
    end
    relmat_targets = filter(item -> item.target.level === :relmat, targets)
    if length(relmat_targets) == 1 && length(iid_targets) == 1 && length(targets) == 2
        target, iid_target = only(relmat_targets), only(iid_targets)
        descriptor = target.target
        group, iid_group = descriptor.group, only(iid_targets).target.group
        descriptor.dpar === :mu && descriptor.member === nothing &&
            descriptor.block === nothing || throw(ArgumentError(
                "bootstrap: Binomial related-matrix-plus-IID simulation requires a related-matrix mean-intercept SD target"))
        structured == (:relmat, group) || throw(ArgumentError(
            "bootstrap: related-matrix-plus-IID simulation requires one matching relmat effect"))
        length(re) == 1 && first(_re_kind(only(re)[1])) === :intercept &&
            last(only(re)) === iid_group || throw(ArgumentError(
                "bootstrap: related-matrix-plus-IID simulation requires one matching IID random intercept"))
        A === nothing && Ainv === nothing || throw(ArgumentError(
            "bootstrap: related-matrix-plus-IID simulation uses `K`, not `A` or `Ainv`"))
        K === nothing && throw(ArgumentError(
            "bootstrap: supply the same related-matrix covariance `K` used to fit the model"))
        hasproperty(data, group) && hasproperty(data, iid_group) || throw(ArgumentError(
            "bootstrap: data are missing a related-matrix or IID grouping column"))
        rel_index, G = _group_index(getproperty(data, group))
        iid_index, H = _group_index(getproperty(data, iid_group))
        length(rel_index) == length(iid_index) == fit.nobs || throw(ArgumentError(
            "bootstrap: related-matrix/IID group indexes do not match fitted response rows"))
        C = Matrix{Float64}(K)
        size(C) == (G, G) || throw(ArgumentError(
            "bootstrap: related-matrix covariance must match first-seen levels of `$group`"))
        factor = _checked_relmat_chol(C, group)
        Wr, names_r = _sd_group_design(f.response, target.rhs, data,
            rel_index, G, group; schema_cache = f.schema_cache,
            schema_key = target.key, store_schema = false)
        Wi, names_i = _sd_group_design(f.response, iid_target.rhs, data,
            iid_index, H, iid_group; schema_cache = f.schema_cache,
            schema_key = iid_target.key, store_schema = false)
        rel_block = _sd_target_block(target.target; disambiguate = true)
        iid_block = _sd_target_block(iid_target.target; disambiguate = true)
        names_r == _block_coefnames(fit, rel_block) || throw(ArgumentError(
            "bootstrap: related-matrix SD design does not match fitted block `$rel_block`"))
        names_i == _block_coefnames(fit, iid_block) || throw(ArgumentError(
            "bootstrap: IID SD design does not match fitted block `$iid_block`"))
        sd_rel = _binomial_sd_scale(Wr * coef(fit, rel_block))
        sd_iid = _binomial_sd_scale(Wi * coef(fit, iid_block))
        eta_fixed = Vector{Float64}(predict(fit, data; type = :link))
        length(eta_fixed) == fit.nobs || throw(ArgumentError(
            "bootstrap: fixed Binomial predictions do not match fitted response rows"))
        return function (rng)
            rel_noise = factor.L * randn(rng, G)
            iid_noise = randn(rng, H)
            eta = eta_fixed .+ sd_rel[rel_index] .* rel_noise[rel_index] .+
                  sd_iid[iid_index] .* iid_noise[iid_index]
            return _simulate_once(fit, rng; mu = _mean_response(fit.family, eta))
        end
    end
    if !isempty(related_targets)
        length(related_targets) == 1 && length(targets) == 1 || throw(ArgumentError(
            "bootstrap: Binomial structured SD simulation requires one structured SD target"))
        target = only(related_targets)
        descriptor = target.target
        provider = descriptor.level
        descriptor.dpar === :mu && descriptor.member === nothing &&
            descriptor.block === nothing || throw(ArgumentError(
                "bootstrap: Binomial structured SD simulation requires the sole structured mean coefficient"))
        isempty(re) && structured == (provider, descriptor.group) ||
            throw(ArgumentError("bootstrap: Binomial structured SD simulation requires one `$provider` effect and no other random effects"))
        matrix = provider === :animal ? A : provider === :spatial ? nothing : K
        matrix_name = provider === :animal && Ainv !== nothing ? "Ainv" : provider === :animal ? "A" : provider === :spatial ? "coords" : "K"
        provider === :animal && Ainv !== nothing && A !== nothing && throw(ArgumentError(
            "bootstrap: supply only one of `A` and `Ainv` for Binomial animal SD simulation"))
        provider === :animal && Ainv !== nothing && (matrix = Ainv)
        group = descriptor.group
        hasproperty(data, group) || throw(ArgumentError(
            "bootstrap: data are missing structured grouping column `$group`"))
        gidx, G = _group_index(getproperty(data, group))
        provider === :spatial && (matrix = _spatial_covariance_from_coords(
            group, G, coords, nothing; legacy_mean_range = true))   # mirrors the binomial fit
        matrix === nothing && throw(ArgumentError(
            "bootstrap: supply the same structured covariance `$matrix_name` used to fit the Binomial $provider SD model"))
        C = Matrix{Float64}(matrix)
        size(C) == (G, G) || throw(ArgumentError(
            "bootstrap: Binomial $provider matrix must match the number of observed group levels in first-seen order"))
        factor = if provider === :animal && Ainv !== nothing
            Q = sparse(C)
            all(isfinite, nonzeros(Q)) || throw(ArgumentError(
                "bootstrap: animal `Ainv` must contain only finite values"))
            asymmetry = norm(Q - transpose(Q), Inf)
            scale = max(norm(Q, Inf), 1.0)
            asymmetry <= 1e-10 * scale || throw(ArgumentError(
                "bootstrap: animal `Ainv` must be symmetric"))
            Q = sparse(Symmetric((Q + transpose(Q)) / 2))
            qchol = cholesky(Symmetric(Q); check = false)
            issuccess(qchol) || throw(ArgumentError(
                "bootstrap: animal `Ainv` must be positive definite"))
            qchol
        else
            _checked_relmat_chol(C, group)
        end
        slope = provider === :spatial ? spatial_slope : animal_slope
        re_design = if slope === nothing
            ones(Float64, length(gidx))
        else
            hasproperty(data, slope) || throw(ArgumentError(
                "bootstrap: data are missing $provider slope predictor `$slope`"))
            raw = getproperty(data, slope)
            raw isa AbstractVector{<:Real} && length(raw) == length(gidx) &&
                all(isfinite, raw) || throw(ArgumentError(
                "bootstrap: $provider slope predictor `$slope` must be numeric, finite, and have one value per response"))
            Float64.(raw)
        end
        Zg, names = _sd_group_design(f.response, target.rhs, data, gidx, G, group;
            schema_cache = f.schema_cache, schema_key = target.key, store_schema = false)
        names == _block_coefnames(fit, :sd) || throw(ArgumentError(
            "bootstrap: the structured SD design columns do not match fitted block `:sd`"))
        group_sd = _binomial_structured_sd_group_scale(Zg, coef(fit, :sd))
        eta_fixed = Vector{Float64}(predict(fit, data; type = :link))
        length(eta_fixed) == length(gidx) == fit.nobs || throw(ArgumentError(
            "bootstrap: Binomial structured prediction, grouping, and fitted response lengths do not match"))
        return function (rng)
            noise = provider === :animal && Ainv !== nothing ?
                _binomial_animal_precision_draw(rng, factor, G) :
                (factor.L * randn(rng, G))
            b = group_sd .* noise
            mu_star = _mean_response(fit.family, eta_fixed .+ re_design .* b[gidx])
            return _simulate_once(fit, rng; mu = mu_star)
        end
    end
    iid_targets = filter(item -> item.target.level === :iid, targets)
    phylo_targets = _sdphylo_parts(f)
    structured_sim = nothing
    if structured !== nothing
        first(structured) === :phylo || throw(ArgumentError(
            "bootstrap: marginal simulation for Binomial SD regression supports the phylogenetic structured provider only"))
        length(phylo_targets) == 1 && first(only(phylo_targets)) === last(structured) ||
            throw(ArgumentError("bootstrap: a Binomial phylogenetic effect needs one matching `sd(group, phylogenetic)` target"))
        phylo = tree isa AbstractString ? augmented_phy(tree) : tree
        phylo === nothing && throw(ArgumentError(
            "bootstrap: supply the same phylogenetic tree used to fit the Binomial SD model"))
        group = last(structured)
        hasproperty(data, group) || throw(ArgumentError(
            "bootstrap: data are missing phylogenetic grouping column `$group`"))
        labels = getproperty(data, group)
        row_tip = _phylo_mean_leaf_index(phylo, labels)
        group_index, n_group = _group_index(labels)
        n_group == phylo.n_leaves || throw(ArgumentError(
            "bootstrap: Binomial phylogenetic SD simulation requires data for every tree tip"))
        _, sd_rhs = only(phylo_targets)
        Zgroup, names = _sd_group_design(f.response, sd_rhs, data,
            group_index, n_group, group; schema_cache = f.schema_cache,
            schema_key = :sd_phylo, store_schema = false)
        names == _block_coefnames(fit, :sd_phylo) || throw(ArgumentError(
            "bootstrap: the phylogenetic SD design columns do not match fitted block `:sd_phylo`"))
        tip_group = zeros(Int, phylo.n_leaves)
        @inbounds for i in eachindex(row_tip)
            tip_group[row_tip[i]] = group_index[i]
        end
        all(!iszero, tip_group) || throw(ArgumentError(
            "bootstrap: Binomial phylogenetic SD simulation requires SD predictors for every tree tip"))
        Ztip = Zgroup[tip_group, :]
        covariance = fit.phylo_scale === :correlation ? _phylo_correlation(phylo) :
            sigma_phy_dense(phylo; σ²_phy = 1.0)
        Lphy = cholesky(Symmetric(Matrix{Float64}(covariance))).L
        phy_sd = _binomial_sd_scale(Ztip * coef(fit, :sd_phylo))
        re_design = if phylo_slope === nothing
            ones(Float64, fit.nobs)
        else
            hasproperty(data, phylo_slope) || throw(ArgumentError(
                "bootstrap: data are missing phylogenetic slope predictor `$phylo_slope`"))
            raw = getproperty(data, phylo_slope)
            raw isa AbstractVector{<:Real} && length(raw) == fit.nobs &&
                all(isfinite, raw) || throw(ArgumentError(
                "bootstrap: phylogenetic slope predictor `$phylo_slope` must be numeric, finite, and have one value per response"))
            any(!iszero, raw) || throw(ArgumentError(
                "bootstrap: phylogenetic slope predictor `$phylo_slope` is zero for every observation"))
            Float64.(raw)
        end
        structured_sim = (; row_tip, Lphy, phy_sd, re_design,
                          n_group = phylo.n_leaves)
    elseif !isempty(phylo_targets)
        throw(ArgumentError(
            "bootstrap: a Binomial phylogenetic SD target requires its matching phylogenetic mean effect"))
    end
    correlated_members(term) = if term isa FunctionTerm && term.f === (+) &&
                                  any(t -> t isa ConstantTerm && t.n == 1, term.args)
        vars = filter(t -> t isa Term, term.args)
        [:intercept; getproperty.(vars, :sym)]
    else
        nothing
    end
    correlated = length(re) == 1 && correlated_members(re[1][1]) !== nothing
    effects = if correlated
        members = correlated_members(only(re)[1])
        1 <= length(iid_targets) <= length(members) <= 3 || throw(ArgumentError(
            "bootstrap: correlated Binomial SD simulation supports blocks of up to three coefficients and needs at least one SD target"))
        term, group = only(re)
        map(members) do member
            kind = member === :intercept ? :intercept : :slope
            slope = kind === :slope ? member : nothing
            (; group, member, kind, slope)
        end
    else
        length(re) == length(iid_targets) || throw(ArgumentError(
            "bootstrap: cannot align every Binomial random-effect coefficient with one SD target"))
        map(re) do (term, group)
            kind, slope = _re_kind(term)
            member = kind === :intercept ? :intercept : kind === :slope ? slope : nothing
            member === nothing && throw(ArgumentError(
                "bootstrap: Binomial SD regression marginal simulation supports independent effects or one correlated intercept-slope block"))
            (; group, member, kind, slope)
        end
    end
    group_indices = map(effects) do effect
        group = effect.group
        hasproperty(data, group) || throw(ArgumentError(
            "bootstrap: data are missing the Binomial random-effect grouping column `$group`"))
        gidx, G = _group_index(getproperty(data, group))
        length(gidx) == fit.nobs || throw(ArgumentError(
            "bootstrap: grouping column `$group` does not match the fitted response rows"))
        (gidx, G)
    end
    correlated && length(unique(effect.group for effect in effects)) != 1 &&
        throw(ArgumentError(
            "bootstrap: correlated Binomial random-effect coefficients must share one grouping variable"))
    n = fit.nobs
    (!correlated && length(iid_targets) != length(effects)) && throw(ArgumentError(
        "bootstrap: each Binomial random-effect coefficient needs one SD target"))

    loadings = Vector{Vector{Float64}}(undef, length(effects))
    group_sds = Vector{Vector{Float64}}(undef, length(effects))
    used = Set{Symbol}()
    for (j, effect) in enumerate(effects)
        group = effect.group
        gidx, G = group_indices[j]
        matches = filter(iid_targets) do item
            t = item.target
            t.group === effect.group && t.level === :iid && t.dpar === :mu &&
                (t.member === nothing || t.member === effect.member)
        end
        length(matches) <= 1 || throw(ArgumentError(
            "bootstrap: Binomial SD target does not uniquely identify `$(effect.group)` coefficient `$(effect.member)`"))
        if isempty(matches)
            correlated || throw(ArgumentError(
                "bootstrap: a Binomial independent random effect is missing its SD target"))
            recov = findfirst(pair -> first(pair) === :recov, fit.blocks)
            recov === nothing && throw(ArgumentError(
                "bootstrap: omitted correlated coefficient SD is missing from the covariance block"))
            recov_range = last(recov)
            recov_names = _block_coefnames(fit, :recov)
            constant_index = findfirst(name -> endswith(name, ":log_sd_$(effect.member)"), recov_names)
            constant_index === nothing && throw(ArgumentError(
                "bootstrap: constant SD for correlated coefficient `$(effect.member)` is missing"))
            group_sds[j] = fill(
                _binomial_sd_scale(fit.theta[first(recov_range) + constant_index - 1]), G)
            if effect.kind === :intercept
                loadings[j] = ones(n)
            else
                hasproperty(data, effect.slope) || throw(ArgumentError(
                    "bootstrap: data are missing random-slope predictor `$(effect.slope)`"))
                raw = getproperty(data, effect.slope)
                raw isa AbstractVector{<:Real} && length(raw) == n &&
                    all(isfinite, raw) || throw(ArgumentError(
                    "bootstrap: random-slope predictor `$(effect.slope)` must be numeric, finite, and have one value per response"))
                loadings[j] = Float64.(raw)
            end
            continue
        end
        target = only(matches)
        target.key in used && throw(ArgumentError(
            "bootstrap: a Binomial SD target maps to more than one random-effect coefficient"))
        push!(used, target.key)
        loadings[j] = if effect.kind === :intercept
            ones(n)
        else
            hasproperty(data, effect.slope) || throw(ArgumentError(
                "bootstrap: data are missing random-slope predictor `$(effect.slope)`"))
            raw = getproperty(data, effect.slope)
            raw isa AbstractVector{<:Real} && length(raw) == n && all(isfinite, raw) ||
                throw(ArgumentError(
                    "bootstrap: random-slope predictor `$(effect.slope)` must be numeric, finite, and have one value per response"))
            Float64.(raw)
        end
        block = _sd_target_block(target.target; disambiguate = length(targets) > 1)
        Zg, names = _sd_group_design(f.response, target.rhs, data, gidx, G, group;
            schema_cache = f.schema_cache, schema_key = target.key, store_schema = false)
        names == _block_coefnames(fit, block) || throw(ArgumentError(
            "bootstrap: the SD design columns do not match fitted block `$block`"))
        group_sds[j] = _binomial_sd_scale(Zg * coef(fit, block))
    end

    eta_fixed = try
        Vector{Float64}(predict(fit, data; type = :link))
    catch err
        throw(ArgumentError("bootstrap: cannot build the fixed Binomial link prediction: $(sprint(showerror, err))"))
    end
    length(eta_fixed) == n == fit.nobs || throw(ArgumentError(
        "bootstrap: Binomial prediction, grouping, and fitted response lengths do not match"))
    recov_range = nothing
    correlation_eta = Float64[]
    correlation_names = String[]
    if correlated
        recov = findfirst(pair -> first(pair) === :recov, fit.blocks)
        recov === nothing && throw(ArgumentError(
            "bootstrap: correlated Binomial SD fit is missing its covariance parameter block"))
        recov_range = last(recov)
        recov_names = _block_coefnames(fit, :recov)
        correlation_names = if length(effects) == 2
            ["$(first(effects).group):atanh_rho"]
        elseif length(effects) == 3
            ["$(first(effects).group):atanh_partial_rho_21",
             "$(first(effects).group):atanh_partial_rho_31",
             "$(first(effects).group):atanh_partial_rho_32_given_1"]
        else
            throw(ArgumentError(
                "bootstrap: correlated Binomial simulation requires two or three coefficients"))
        end
        correlation_indices = map(correlation_names) do name
            findfirst(==(name), recov_names)
        end
        all(!isnothing, correlation_indices) || throw(ArgumentError(
            "bootstrap: correlated Binomial SD fit is missing its named correlation parameters"))
        correlation_eta = Float64[fit.theta[first(recov_range) + i - 1]
                                  for i in correlation_indices]
    end
    return function (rng)
        eta = copy(eta_fixed)
        if correlated
            gidx, G = group_indices[1]
            q = length(effects)
            u = randn(rng, q, G)
            if q == 2
                rho, innovation = _binomial_bounded_partial(only(correlation_eta))
                b0 = group_sds[1] .* u[1, :]
                b1 = group_sds[2] .* (rho .* u[1, :] .+ innovation .* u[2, :])
                eta .+= b0[gidx] .+ loadings[2] .* b1[gidx]
            else
                pairs = _binomial_bounded_partial.(correlation_eta)
                partials, innovations = first.(pairs), last.(pairs)
                p21, p31, p32 = partials
                s21, s31, s32 = innovations
                b0 = group_sds[1] .* u[1, :]
                b1 = group_sds[2] .* (p21 .* u[1, :] .+ s21 .* u[2, :])
                b2 = group_sds[3] .* (p31 .* u[1, :] .+ s31 .* p32 .* u[2, :] .+
                                       s31 .* s32 .* u[3, :])
                eta .+= b0[gidx] .+ loadings[2] .* b1[gidx] .+
                        loadings[3] .* b2[gidx]
            end
        else
            for (j, (gidx, G)) in enumerate(group_indices)
                random_coefficients = group_sds[j] .* randn(rng, G)
                eta .+= loadings[j] .* random_coefficients[gidx]
            end
        end
        if structured_sim !== nothing
            phy_effect = structured_sim.phy_sd .* (
                structured_sim.Lphy * randn(rng, structured_sim.n_group))
            eta .+= structured_sim.re_design .* phy_effect[structured_sim.row_tip]
        end
        mu_star = _mean_response(fit.family, eta)
        return _simulate_once(fit, rng; mu = mu_star)
    end
end

function _binomial_phylo_interaction_sd_marginal_simulator(fit::DrmFit, data,
                                                            tree1, tree2)
    tree1 === nothing && throw(ArgumentError(
        "bootstrap: supply the same `tree1` used to fit the phylogenetic-interaction model"))
    tree2 === nothing && throw(ArgumentError(
        "bootstrap: supply the same `tree2` used to fit the phylogenetic-interaction model"))
    f = fit.formula
    target = only(filter(item ->
        item.target.level === :phylogenetic_interaction, _sd_target_parts(f)))
    marker = _binomial_phylo_interaction_parts(Dict(f.forms)[:mu])
    marker === nothing && throw(ArgumentError(
        "bootstrap: fitted interaction SD formula has no phylo_interaction() marker"))
    host = getproperty(data, marker.group1)
    pollinator = getproperty(data, marker.group2)
    pair = getproperty(data, target.target.group)
    length(host) == length(pollinator) == length(pair) == fit.nobs ||
        throw(ArgumentError("bootstrap: host, pollinator, pair, and fitted rows do not align"))
    (any(ismissing, host) || any(ismissing, pollinator) || any(ismissing, pair)) &&
        throw(ArgumentError("bootstrap: interaction grouping columns cannot contain missing values"))
    pair_index, Gpair = _group_index(pair)
    pair_to_combo = Dict{Int,Tuple{String,String}}()
    combo_to_pair = Dict{Tuple{String,String},Int}()
    for i in eachindex(pair_index)
        combo = (string(host[i]), string(pollinator[i]))
        group = pair_index[i]
        haskey(pair_to_combo, group) && pair_to_combo[group] != combo &&
            throw(ArgumentError("bootstrap: each pair ID must identify one host-pollinator combination"))
        haskey(combo_to_pair, combo) && combo_to_pair[combo] != group &&
            throw(ArgumentError("bootstrap: each host-pollinator combination must have one pair ID"))
        pair_to_combo[group] = combo
        combo_to_pair[combo] = group
    end
    Q1, node1, _ = _poisson_phylo_setup(tree1, host)
    Q2, node2, _ = _poisson_phylo_setup(tree2, pollinator)
    Q = kron(Q2, Q1)
    leaf_node = (node2 .- 1) .* size(Q1, 1) .+ node1
    factor = cholesky(Symmetric(Q); check = false)
    issuccess(factor) || throw(ArgumentError(
        "bootstrap: the supplied tree pair gives a non-positive-definite interaction precision"))
    ndr = merge(NamedTuple(pairs(data)),
                NamedTuple{(f.response,)}((zeros(fit.nobs),)))
    Zpair, names = _sd_group_design(f.response, target.rhs, data,
        pair_index, Gpair, target.target.group; schema_cache = f.schema_cache,
        schema_key = target.key, store_schema = false)
    block = _sd_target_block(target.target; disambiguate = false)
    names == _block_coefnames(fit, block) || throw(ArgumentError(
        "bootstrap: phylogenetic-interaction SD design differs from the fitted coefficient schema"))
    sd_pair = _binomial_sd_scale(Zpair * coef(fit, block))
    eta_fixed = Vector{Float64}(predict(fit, data; type = :link))
    length(eta_fixed) == fit.nobs || throw(ArgumentError(
        "bootstrap: fixed Binomial predictions do not match fitted responses"))
    re_design = if marker.slope === nothing
        ones(Float64, fit.nobs)
    else
        hasproperty(data, marker.slope) || throw(ArgumentError(
            "bootstrap: phylogenetic-interaction slope predictor `$(marker.slope)` is missing from data"))
        raw_slope = getproperty(data, marker.slope)
        raw_slope isa AbstractVector{<:Real} && length(raw_slope) == fit.nobs &&
            all(isfinite, raw_slope) || throw(ArgumentError(
            "bootstrap: phylogenetic-interaction slope predictor `$(marker.slope)` must be numeric, finite, and have one value per fitted response"))
        Float64.(raw_slope)
    end
    return function (rng)
        latent = factor.UP \ randn(rng, size(Q, 1))
        eta = eta_fixed .+ sd_pair[pair_index] .* latent[leaf_node] .* re_design
        return _simulate_once(fit, rng; mu = _mean_response(fit.family, eta))
    end
end

function _poisson_structured_sd_marginal_simulator(
    fit::DrmFit, data, matrix; provider::Symbol, animal_slope = nothing)
    f = fit.formula
    targets = _sd_target_parts(f)
    length(targets) == 1 || throw(ArgumentError(
        "bootstrap: Poisson structured SD simulation requires one covariance SD target"))
    target = only(targets)
    descriptor = target.target
    (descriptor.level === provider && provider in (:relmat, :animal) &&
     descriptor.dpar === :mu && descriptor.block === nothing) ||
        throw(ArgumentError("bootstrap: Poisson structured SD simulation requires one mean structured target"))
    rhs = Dict(f.forms)
    _, re, metav, structured, _, _, fitted_animal_slope = _split_ranef(
        rhs[:mu]; allow_animal_slope = provider === :animal,
        allow_relmat_slope = provider === :relmat)
    expected_member = fitted_animal_slope === nothing ? :intercept : fitted_animal_slope
    descriptor.member in (nothing, expected_member) || throw(ArgumentError(
        "bootstrap: Poisson structured SD target does not match coefficient `$(expected_member)`."))
    metav === nothing && isempty(re) && structured == (provider, descriptor.group) ||
        throw(ArgumentError("bootstrap: Poisson structured SD simulation requires one matching structured mean coefficient and no other random effects"))
    fitted_animal_slope == animal_slope || throw(ArgumentError(
        "bootstrap: Poisson animal SD simulator slope does not match the fitted mean effect"))
    matrix === nothing && throw(ArgumentError(
        "bootstrap: supply the same covariance matrix used to fit the Poisson $(provider) SD model"))
    matrix_name = provider === :animal ? "animal" : "relmat"
    group = descriptor.group
    hasproperty(data, group) || throw(ArgumentError(
        "bootstrap: data are missing Poisson $(matrix_name) grouping column `$group`"))
    gidx, G = _group_index(getproperty(data, group))
    C = Matrix{Float64}(matrix)
    size(C) == (G, G) || throw(ArgumentError(
        "bootstrap: Poisson $(matrix_name) covariance matrix must match the number of observed group levels in first-seen order"))
    factor = _poisson_structured_sd_factor(C, G, matrix_name; normalize = false)
    Zg, names = _sd_group_design(f.response, target.rhs, data, gidx, G, group;
        schema_cache = f.schema_cache, schema_key = target.key, store_schema = false)
    names == _block_coefnames(fit, :sd) || throw(ArgumentError(
        "bootstrap: Poisson $(matrix_name) SD design columns do not match fitted block `:sd`"))
    sd_group = exp.(Zg * coef(fit, :sd))
    loading = fitted_animal_slope === nothing ? ones(Float64, length(gidx)) :
        _poisson_animal_slope_loading(data, fitted_animal_slope, length(gidx))
    eta_fixed = Vector{Float64}(predict(fit, data; type = :link))
    length(eta_fixed) == fit.nobs == length(gidx) || throw(ArgumentError(
        "bootstrap: Poisson $(matrix_name) prediction, grouping, and fitted response lengths do not match"))
    return function (rng)
        u = sd_group .* (factor.L * randn(rng, G))
        mu_star = _mean_response(fit.family, eta_fixed .+ loading .* u[gidx])
        return _simulate_once(fit, rng; mu = mu_star)
    end
end

function _poisson_structured_sd_factor(matrix, G::Int, provider::AbstractString;
                                        normalize::Bool = false)
    C = Matrix{Float64}(matrix)
    size(C) == (G, G) || throw(ArgumentError(
        "bootstrap: Poisson $(provider) covariance matrix must match first-seen group order"))
    d = sqrt.(diag(C))
    all(x -> isfinite(x) && x > 0, d) || throw(ArgumentError(
        "bootstrap: Poisson $(provider) covariance matrix must have a positive finite diagonal"))
    covariance = normalize ? C ./ (d * d') : C
    factor = cholesky(Symmetric(covariance); check = false)
    issuccess(factor) || throw(ArgumentError(
        "bootstrap: Poisson $(provider) covariance matrix must be positive definite"))
    return factor
end

function _poisson_animal_slope_marginal_simulator(fit::DrmFit, data, matrix,
                                                  slope::Symbol)
    f = fit.formula
    rhs = Dict(f.forms)
    fixed_mu, re, metav, structured, _, _, fitted_slope = _split_ranef(
        rhs[:mu]; allow_animal_slope = true)
    metav === nothing && isempty(re) && structured !== nothing &&
        first(structured) === :animal && fitted_slope === slope || throw(ArgumentError(
        "bootstrap: Poisson animal-slope simulation requires one matching animal mean slope"))
    fixed_mu, off = _extract_offset(fixed_mu)
    off === nothing || throw(ArgumentError(
        "bootstrap: Poisson animal-slope simulation does not support `offset(...)`"))
    group = last(structured)
    hasproperty(data, group) || throw(ArgumentError(
        "bootstrap: data are missing Poisson animal grouping column `$group`"))
    gidx, G = _group_index(getproperty(data, group))
    matrix === nothing && throw(ArgumentError(
        "bootstrap: supply the same covariance matrix used to fit the Poisson animal model"))
    factor = _poisson_structured_sd_factor(matrix, G, "animal"; normalize = false)
    loading = _poisson_animal_slope_loading(data, slope, length(gidx))
    _, Xmu, _ = _design(f.response, fixed_mu, data)
    eta_fixed = Xmu * coef(fit, :mu)
    sd_group = fill(exp(only(coef(fit, :resd))), G)
    return function (rng)
        u = sd_group .* (factor.L * randn(rng, G))
        mu_star = _mean_response(fit.family, eta_fixed .+ loading .* u[gidx])
        return _simulate_once(fit, rng; mu = mu_star)
    end
end

function _gamma_structured_sd_marginal_simulator(fit::DrmFit, data, matrix;
                                                 provider::Symbol,
                                                 animal_slope = nothing)
    f = fit.formula
    targets = _sd_target_parts(f)
    length(targets) == 1 || throw(ArgumentError(
        "bootstrap: Gamma structured SD simulation requires one structured SD target"))
    target = only(targets)
    descriptor = target.target
    descriptor.level === provider && provider in (:relmat, :animal, :spatial) &&
        descriptor.dpar === :mu && descriptor.block === nothing ||
        throw(ArgumentError(
            "bootstrap: Gamma structured SD simulation requires one mean structured target"))
    rhs = Dict(f.forms)
    _, re, metav, structured, _, _, fitted_animal_slope = _split_ranef(
        rhs[:mu]; allow_animal_slope = provider === :animal,
        allow_spatial_slope = provider === :spatial)
    expected_member = fitted_animal_slope === nothing ? :intercept : fitted_animal_slope
    descriptor.member in (nothing, expected_member) ||
        throw(ArgumentError(
            "bootstrap: Gamma structured SD target does not match its fitted coefficient"))
    metav === nothing && isempty(re) && structured == (provider, descriptor.group) ||
        throw(ArgumentError(
            "bootstrap: Gamma structured SD simulation requires one matching mean effect and no other random effects"))
    fitted_animal_slope == animal_slope || throw(ArgumentError(
        "bootstrap: Gamma animal SD simulator slope does not match the fitted mean effect"))
    matrix === nothing && throw(ArgumentError(
        "bootstrap: supply the same covariance matrix used to fit the Gamma $provider SD model"))
    group = descriptor.group
    hasproperty(data, group) || throw(ArgumentError(
        "bootstrap: data are missing Gamma structured grouping column `$group`"))
    gidx, G = _group_index(getproperty(data, group))
    C = Matrix{Float64}(matrix)
    size(C) == (G, G) || throw(ArgumentError(
        "bootstrap: Gamma structured covariance matrix must match first-seen group order"))
    d = sqrt.(diag(C))
    all(x -> isfinite(x) && x > 0, d) || throw(ArgumentError(
        "bootstrap: Gamma structured covariance matrix must have a positive finite diagonal"))
    covariance = Symmetric(C)   # K and A enter RAW, as in the fit
    factor = cholesky(covariance; check = false)
    issuccess(factor) || throw(ArgumentError(
        "bootstrap: Gamma structured covariance matrix must be positive definite"))
    Zg, names = _sd_group_design(f.response, target.rhs, data, gidx, G, group;
        schema_cache = f.schema_cache, schema_key = target.key, store_schema = false)
    names == _block_coefnames(fit, :sd) || throw(ArgumentError(
        "bootstrap: Gamma structured SD design columns do not match fitted block `:sd`"))
    sd_group = exp.(Zg * coef(fit, :sd))
    loading = if animal_slope === nothing
        ones(Float64, length(gidx))
    else
        hasproperty(data, animal_slope) || throw(ArgumentError(
            "bootstrap: data are missing Gamma animal slope predictor `$animal_slope`"))
        raw = getproperty(data, animal_slope)
        raw isa AbstractVector{<:Real} && length(raw) == length(gidx) &&
            all(isfinite, raw) || throw(ArgumentError(
                "bootstrap: Gamma animal slope predictor `$animal_slope` must be finite numeric values with one value per response"))
        Float64.(raw)
    end
    eta_fixed = Vector{Float64}(predict(fit, data; type = :link))
    length(eta_fixed) == fit.nobs == length(gidx) || throw(ArgumentError(
        "bootstrap: Gamma structured prediction, grouping, and fitted response lengths do not match"))
    return function (rng)
        u = sd_group .* (factor.L * randn(rng, G))
        mu_star = _mean_response(fit.family, eta_fixed .+ loading .* u[gidx])
        return _simulate_once(fit, rng; mu = mu_star)
    end
end

function _gamma_animal_intercept_slope_sd_marginal_simulator(fit::DrmFit,
        data, matrix; animal_block)
    f = fit.formula
    targets = _sd_target_parts(f)
    length(targets) == 2 || throw(ArgumentError(
        "bootstrap: Gamma animal intercept-and-slope simulation needs two SD targets"))
    group, slope = animal_block.group, animal_block.slope
    rhs = Dict(f.forms)
    parsed = _animal_intercept_slope(rhs[:mu])
    parsed == animal_block || throw(ArgumentError(
        "bootstrap: Gamma animal intercept-and-slope simulator does not match the fitted mean effect"))
    matrix === nothing && throw(ArgumentError(
        "bootstrap: supply the same relatedness matrix `A` used to fit the Gamma animal model"))
    hasproperty(data, group) && hasproperty(data, slope) || throw(ArgumentError(
        "bootstrap: data are missing Gamma animal grouping or slope columns"))
    gidx, G = _group_index(getproperty(data, group))
    C = Matrix{Float64}(matrix)
    size(C) == (G, G) || throw(ArgumentError(
        "bootstrap: Gamma animal covariance matrix must match first-seen group order"))
    d = sqrt.(diag(C))
    all(x -> isfinite(x) && x > 0, d) || throw(ArgumentError(
        "bootstrap: Gamma animal covariance matrix must have a positive finite diagonal"))
    factor = cholesky(Symmetric(C); check = false)
    issuccess(factor) || throw(ArgumentError(
        "bootstrap: Gamma animal covariance matrix must be positive definite"))
    ordered = [only(filter(item -> item.target.member === member, targets))
               for member in (:intercept, slope)]
    scales = Vector{Vector{Float64}}(undef, 2)
    blocks = Vector{Symbol}(undef, 2)
    for j in 1:2
        target = ordered[j]
        blocks[j] = _sd_target_block(target.target; disambiguate = true)
        Zg, names = _sd_group_design(f.response, target.rhs, data, gidx, G, group;
            schema_cache = f.schema_cache, schema_key = target.key,
            store_schema = false)
        names == _block_coefnames(fit, blocks[j]) || throw(ArgumentError(
            "bootstrap: Gamma animal SD design does not match fitted block `$(blocks[j])`"))
        scales[j] = exp.(Zg * coef(fit, blocks[j]))
    end
    x = getproperty(data, slope)
    x isa AbstractVector{<:Real} && length(x) == fit.nobs && all(isfinite, x) ||
        throw(ArgumentError("bootstrap: Gamma animal slope must be finite numeric values with one value per response"))
    eta_fixed = Vector{Float64}(predict(fit, data; type = :link))
    return function (rng)
        b0 = scales[1] .* (factor.L * randn(rng, G))
        b1 = scales[2] .* (factor.L * randn(rng, G))
        mu_star = _mean_response(fit.family,
            eta_fixed .+ b0[gidx] .+ Float64.(x) .* b1[gidx])
        _simulate_once(fit, rng; mu = mu_star)
    end
end

# Independent two-field structured intercept-plus-slope (`animal(1 + x | g)`,
# `relmat(1 + x | g)`, `phylo(1 + x | g)` or `spatial(1 + x | g)`) with one SD
# formula per coefficient, on the Poisson, NB2 and Gaussian routes. Draws
# b0, b1 ~ N(0, C) independently (C = A or K, RAW as in the fit; for phylo and
# spatial the fit's own kernel and row-to-level map, passed as `kernel =
# (C, gidx, G)`), then eta = eta_fixed + sd0[g] * b0[g] + x * sd1[g] * b1[g]
# and y from the family at the fitted dispersion (Gaussian: fitted sigma).
function _structured_q2_sd_marginal_simulator(fit::DrmFit, data, matrix;
                                              provider::Symbol, block,
                                              kernel = nothing)
    f = fit.formula
    label = String(provider)
    level = _q2_target_level(provider)
    targets = filter(item -> item.target.level === level, _sd_target_parts(f))
    length(targets) == 2 || throw(ArgumentError(
        "bootstrap: $(label) intercept-and-slope simulation needs two SD targets"))
    group, slope = block.group, block.slope
    ordered = [only(filter(item -> item.target.member === member, targets))
               for member in (:intercept, slope)]
    kernel === nothing && matrix === nothing && throw(ArgumentError(
        "bootstrap: supply the same covariance matrix (`" *
        (provider === :animal ? "A` or `Ainv" : "K") *
        "`) used to fit the $(label) intercept-and-slope model"))
    hasproperty(data, group) && hasproperty(data, slope) || throw(ArgumentError(
        "bootstrap: data are missing the $(label) grouping or slope column"))
    gidx, G = kernel === nothing ? _group_index(getproperty(data, group)) :
        (kernel[2], kernel[3])
    C = Matrix{Float64}(kernel === nothing ? matrix : kernel[1])
    size(C) == (G, G) || throw(ArgumentError(
        "bootstrap: $(label) covariance matrix must match first-seen group order"))
    all(x -> isfinite(x) && x > 0, diag(C)) || throw(ArgumentError(
        "bootstrap: $(label) covariance matrix must have a positive finite diagonal"))
    factor = cholesky(Symmetric(C); check = false)
    issuccess(factor) || throw(ArgumentError(
        "bootstrap: $(label) covariance matrix must be positive definite"))
    scales = Vector{Vector{Float64}}(undef, 2)
    for j in 1:2
        target = ordered[j]
        blk = _sd_target_block(target.target; disambiguate = true)
        Zg, names = _sd_group_design(f.response, target.rhs, data, gidx, G, group;
            schema_cache = f.schema_cache, schema_key = target.key,
            store_schema = false)
        names == _block_coefnames(fit, blk) || throw(ArgumentError(
            "bootstrap: $(label) SD design does not match fitted block `$(blk)`"))
        scales[j] = exp.(Zg * coef(fit, blk))
    end
    x = getproperty(data, slope)
    x isa AbstractVector{<:Real} && all(isfinite, x) || throw(ArgumentError(
        "bootstrap: $(label) slope must be finite numeric values with one value per response"))
    eta_fixed = Vector{Float64}(predict(fit, data; type = :link))
    length(eta_fixed) == length(gidx) == length(x) || throw(ArgumentError(
        "bootstrap: $(label) prediction, grouping and slope lengths do not match"))
    xf = Float64.(x)
    # Labelled `(1 + x | p | g)`: unit fields correlated with the fitted rho.
    rho = hasproperty(block, :label) && block.label !== nothing ?
        _Q2_RHO_BOUND * tanh(only(coef(fit, :recov))) : 0.0
    return function (rng)
        z0 = factor.L * randn(rng, G)
        z1 = factor.L * randn(rng, G)
        b0 = scales[1] .* z0
        b1 = scales[2] .* (rho .* z0 .+ sqrt(1 - rho^2) .* z1)
        eta = eta_fixed .+ b0[gidx] .+ xf .* b1[gidx]
        if fit.family isa Gaussian
            return eta .+ fit.scales[:sigma] .* randn(rng, length(eta))
        end
        return _simulate_once(fit, rng; mu = _mean_response(fit.family, eta))
    end
end

function _draw_phylo_raw_field(rng::Random.AbstractRNG, phy::AugmentedPhy)
    field = zeros(Float64, phy.n_total)
    seen = falses(phy.n_total)
    seen[phy.root_index] = true
    queue = [phy.root_index]
    Q = phy.Q_topology
    row_index = rowvals(Q)
    values = nonzeros(Q)
    cursor = 1
    while cursor <= length(queue)
        parent = queue[cursor]
        cursor += 1
        for k in nzrange(Q, parent)
            child = row_index[k]
            child == parent && continue
            seen[child] && continue
            edge_precision = -values[k]
            isfinite(edge_precision) && edge_precision > 0 || throw(ArgumentError(
                "bootstrap: phylogenetic tree has a non-positive or non-finite branch precision"))
            field[child] = field[parent] + randn(rng) / sqrt(edge_precision)
            seen[child] = true
            push!(queue, child)
        end
    end
    all(seen) || throw(ArgumentError(
        "bootstrap: phylogenetic tree topology is disconnected"))
    return field
end

function _draw_scaled_phylo_tip_field(rng::Random.AbstractRNG, phy::AugmentedPhy,
                                      tip_sd::AbstractVector{<:Real})
    length(tip_sd) == length(phy.leaf_indices) || throw(DimensionMismatch(
        "phylogenetic tip SDs must match the tree leaves"))
    raw_field = _draw_phylo_raw_field(rng, phy)
    return Float64.(tip_sd) .* raw_field[phy.leaf_indices]
end

function _gamma_phylo_sd_marginal_simulator(fit::DrmFit, data, tree)
    f = fit.formula
    targets = _gamma_sd_targets(f)
    length(targets) == 1 || throw(ArgumentError(
        "bootstrap: Gamma phylogenetic SD simulation requires one SD target"))
    target = only(targets)
    descriptor = target.target
    descriptor.level === :phylogenetic && descriptor.dpar === :mu &&
        descriptor.member === nothing && descriptor.block === nothing ||
        throw(ArgumentError(
            "bootstrap: Gamma phylogenetic SD simulation requires one mean intercept target"))
    rhs = Dict(f.forms)
    _, re, metav, structured, structured_slope =
        _split_ranef(rhs[:mu]; allow_phylo_slope = true)
    isempty(re) && metav === nothing && structured == (:phylo, descriptor.group) &&
        structured_slope === nothing || throw(ArgumentError(
            "bootstrap: Gamma phylogenetic SD simulation requires one matching phylogenetic intercept and no other random effects"))
    hasproperty(data, descriptor.group) || throw(ArgumentError(
        "bootstrap: data are missing phylogenetic grouping column `$(descriptor.group)`"))
    tree === nothing && throw(ArgumentError(
        "bootstrap: supply the same phylogenetic tree used to fit the Gamma SD model"))
    phy, row_tip, n_tip = _lss_phylo_group_index(
        tree, _table_column(data, descriptor.group), descriptor.group)
    length(phy.leaf_indices) == n_tip ||
        error("Gamma phylogenetic tip map changed unexpectedly")
    group_index, n_group = _group_index(_table_column(data, descriptor.group))
    tip_group = zeros(Int, n_tip)
    @inbounds for i in eachindex(row_tip)
        ti, gi = row_tip[i], group_index[i]
        tip_group[ti] == 0 || tip_group[ti] == gi || throw(ArgumentError(
            "bootstrap: Gamma phylogenetic SD predictors must be constant within each tree tip"))
        tip_group[ti] = gi
    end
    all(!iszero, tip_group) || throw(ArgumentError(
        "bootstrap: Gamma phylogenetic SD simulation requires SD predictors for every tree tip"))
    Zgroup, names = _sd_group_design(f.response, target.rhs, data,
        group_index, n_group, descriptor.group; schema_cache = f.schema_cache,
        schema_key = target.key, store_schema = false)
    names == _block_coefnames(fit, :sd_phylo) || throw(ArgumentError(
        "bootstrap: Gamma phylogenetic SD columns do not match fitted block `:sd_phylo`"))
    size(Zgroup, 1) == n_group || throw(ArgumentError(
        "bootstrap: Gamma phylogenetic SD design does not match observed groups"))
    Ztip = Zgroup[tip_group, :]
    tip_sd = exp.(Ztip * coef(fit, :sd_phylo))
    eta_fixed = Vector{Float64}(predict(fit, data; type = :link))
    length(eta_fixed) == fit.nobs == length(row_tip) || throw(ArgumentError(
        "bootstrap: Gamma phylogenetic prediction, grouping, and fitted response lengths do not match"))
    return function (rng)
        scaled_tip_field = _draw_scaled_phylo_tip_field(rng, phy, tip_sd)
        eta = eta_fixed .+ scaled_tip_field[row_tip]
        mu_star = _mean_response(fit.family, eta)
        return _simulate_once(fit, rng; mu = mu_star)
    end
end

function _negbin2_hu_sd_marginal_simulator(fit::DrmFit, data, re)
    f = fit.formula
    targets = _sd_target_parts(f)
    length(targets) == 1 || throw(ArgumentError(
        "bootstrap: hurdle NB2 SD simulation requires exactly one mean SD target"))
    target = only(targets)
    descriptor = target.target
    descriptor.level === :iid && descriptor.dpar === :mu && descriptor.block === nothing ||
        throw(ArgumentError("bootstrap: hurdle NB2 SD simulation supports one ordinary mean coefficient target"))
    length(re) == 1 || throw(ArgumentError(
        "bootstrap: hurdle NB2 SD simulation supports one ordinary mean coefficient"))
    term, re_group = only(re)
    kind, slope = _re_kind(term)
    kind in (:intercept, :slope) || throw(ArgumentError(
        "bootstrap: hurdle NB2 SD simulation supports an intercept or independent numeric slope"))
    member = kind === :intercept ? :intercept : slope
    descriptor.member in (nothing, member) || throw(ArgumentError(
        "bootstrap: hurdle NB2 SD target does not match random-effect coefficient `$member`"))
    group = descriptor.group
    re_group === group || throw(ArgumentError(
        "bootstrap: hurdle NB2 SD target does not match the fitted mean grouping variable `$re_group`"))
    only(re)[2] === group || throw(ArgumentError(
        "bootstrap: hurdle NB2 SD target does not match the fitted mean grouping variable"))
    hasproperty(data, group) || throw(ArgumentError(
        "bootstrap: data are missing hurdle NB2 random-effect grouping column `$group`"))
    gidx, G = _group_index(getproperty(data, group))
    Zg, names = _sd_group_design(f.response, target.rhs, data, gidx, G, group;
        schema_cache = f.schema_cache, schema_key = target.key, store_schema = false)
    names == _block_coefnames(fit, :sd) || throw(ArgumentError(
        "bootstrap: hurdle NB2 SD design columns do not match the fitted coefficient block"))
    sdgroup = exp.(Zg * coef(fit, :sd))
    loading = if kind === :intercept
        ones(Float64, length(gidx))
    else
        hasproperty(data, slope) || throw(ArgumentError(
            "bootstrap: data are missing hurdle NB2 random-slope predictor `$slope`"))
        raw = getproperty(data, slope)
        raw isa AbstractVector{<:Real} && length(raw) == length(gidx) &&
            all(isfinite, raw) || throw(ArgumentError(
                "bootstrap: hurdle NB2 random-slope predictor `$slope` must be finite numeric values with one value per response"))
        Float64.(raw)
    end
    eta_fixed = Vector{Float64}(predict(fit, data; type = :link))
    length(eta_fixed) == length(gidx) == fit.nobs || throw(ArgumentError(
        "bootstrap: hurdle NB2 prediction, grouping, and fitted response lengths do not match"))
    return function (rng)
        eta = eta_fixed .+ loading .* (sdgroup .* randn(rng, G))[gidx]
        mu_star = _mean_response(fit.family, eta)
        return _simulate_once(fit, rng; mu = mu_star)
    end
end

function _beta_animal_sd_marginal_simulator(fit::DrmFit, data, matrix)
    f = fit.formula
    targets = _sd_target_parts(f)
    length(targets) == 1 || throw(ArgumentError(
        "bootstrap: Beta animal SD simulation requires exactly one SD target"))
    target = only(targets)
    descriptor = target.target
    descriptor.level === :animal && descriptor.dpar === :mu &&
        descriptor.block === nothing && descriptor.member in (nothing, :intercept) ||
        throw(ArgumentError("bootstrap: Beta animal SD simulation requires one animal mean-intercept target"))
    rhs = Dict(f.forms)
    _, re, metav, structured = _split_ranef(rhs[:mu])
    metav === nothing && isempty(re) &&
        _collect_structured(rhs[:mu]) == [(:animal, descriptor.group)] ||
        throw(ArgumentError("bootstrap: Beta animal SD simulation requires exactly one matching animal mean intercept"))
    group = descriptor.group
    hasproperty(data, group) || throw(ArgumentError(
        "bootstrap: data are missing Beta animal group column `$group`"))
    gidx, G = _group_index(getproperty(data, group))
    matrix === nothing && throw(ArgumentError(
        "bootstrap: supply the same covariance matrix used to fit the Beta animal model"))
    size(matrix) == (G, G) || throw(ArgumentError(
        "bootstrap: Beta animal covariance matrix must match first-seen group order"))
    C = Matrix{Float64}(matrix)
    d = sqrt.(diag(C))
    all(x -> isfinite(x) && x > 0, d) || throw(ArgumentError(
        "bootstrap: Beta animal covariance matrix must have a positive finite diagonal"))
    R = Symmetric(C)   # A enters RAW, as in the fit
    factor = cholesky(R; check = false)
    issuccess(factor) || throw(ArgumentError(
        "bootstrap: Beta animal covariance matrix must be positive definite"))
    Wsd, names = _sd_group_design(f.response, target.rhs, data, gidx, G, group;
        schema_cache = f.schema_cache, schema_key = target.key)
    get(Dict(fit.coefnames), :sd, String[]) == names || throw(ArgumentError(
        "bootstrap: Beta animal SD design does not match the fitted SD block"))
    size(Wsd, 1) == G || throw(ArgumentError(
        "bootstrap: Beta animal SD design must have one row per animal group"))
    sd_group = exp.(Wsd * coef(fit, :sd))
    eta_fixed = Vector{Float64}(predict(fit, data; type = :link))
    return function (rng)
        u = sd_group .* (factor.L * randn(rng, G))
        mu_star = 1 ./ (1 .+ exp.(-clamp.(eta_fixed .+ u[gidx], -30.0, 30.0)))
        return _simulate_once(fit, rng; mu = mu_star)
    end
end

function _negbin2_animal_sd_precision_marginal_simulator(fit::DrmFit, data, Ainv)
    f = fit.formula
    targets = filter(item -> item.target.level === :animal, _sd_target_parts(f))
    length(targets) == 1 || throw(ArgumentError(
        "bootstrap: NB2 animal precision simulation requires exactly one animal SD target"))
    target = only(targets)
    descriptor = target.target
    descriptor.dpar === :mu && descriptor.block === nothing &&
        descriptor.member in (nothing, :intercept) || throw(ArgumentError(
        "bootstrap: NB2 animal precision simulation requires one animal mean-intercept SD target"))
    group = descriptor.group
    hasproperty(data, group) || throw(ArgumentError(
        "bootstrap: data are missing NB2 animal grouping column `$group`"))
    rhs = Dict(f.forms)
    _, re, metav, structured, structured_slope, _, _ =
        _split_ranef(rhs[:mu]; allow_phylo_slope = true)
    isempty(re) && metav === nothing && structured == (:animal, group) &&
        structured_slope === nothing || throw(ArgumentError(
        "bootstrap: NB2 animal precision simulation requires one animal random intercept and no other random effects"))
    Ainv === nothing && throw(ArgumentError(
        "bootstrap: supply the same animal precision `Ainv` used to fit the NB2 model"))
    labels = _table_column(data, group)
    gidx, G = _group_index(labels)
    Q, precision_index = _nb2_animal_precision_setup(Ainv, labels, group)
    precision_index == gidx || throw(ArgumentError(
        "bootstrap: NB2 animal precision labels do not match the fitted group order"))
    factor = cholesky(Symmetric(Q); check = false)
    issuccess(factor) || throw(ArgumentError(
        "bootstrap: NB2 animal precision must be positive definite"))
    Wsd, names = _sd_group_design(f.response, target.rhs, data, gidx, G, group;
        schema_cache = f.schema_cache, schema_key = target.key, store_schema = false)
    names == _block_coefnames(fit, :sd) || throw(ArgumentError(
        "bootstrap: NB2 animal SD design columns do not match the fitted `:sd` block"))
    size(Wsd, 1) == G || throw(ArgumentError(
        "bootstrap: NB2 animal SD design must have one row per animal group"))
    sd_group = exp.(Wsd * coef(fit, :sd))
    eta_fixed = Vector{Float64}(predict(fit, data; type = :link))
    length(eta_fixed) == fit.nobs == length(gidx) || throw(ArgumentError(
        "bootstrap: NB2 fitted prediction and animal grouping lengths do not match"))
    return function (rng)
        animal_field = factor.UP \ randn(rng, G)
        eta = eta_fixed .+ (sd_group .* animal_field)[gidx]
        return _simulate_once(fit, rng; mu = _mean_response(fit.family, eta))
    end
end

_binomial_animal_precision_draw(rng::Random.AbstractRNG, factor, n::Int) =
    factor.UP \ randn(rng, n)

function _betabinomial_sigma_sd_marginal_simulator(fit::DrmFit, data)
    f = fit.formula
    targets = filter(item -> item.target.dpar === :sigma, _sd_target_parts(f))
    length(targets) == 1 || throw(ArgumentError(
        "bootstrap: BetaBinomial sigma SD simulation requires exactly one sigma target"))
    target = only(targets)
    descriptor = target.target
    (descriptor.level === :iid && descriptor.block === nothing) ||
        throw(ArgumentError(
            "bootstrap: BetaBinomial sigma SD simulation requires one ordinary sigma coefficient"))
    rhs = Dict(f.forms)
    fixed_sigma, re, mv, st = _split_ranef(rhs[:sigma])
    valid_re = !isempty(re) && length(re) == 1 && mv === nothing && st === nothing
    kind, slope = valid_re ? _re_kind(re[1][1]) : (:none, nothing)
    member = kind === :intercept ? nothing : kind === :slope ? slope : :unsupported
    valid_re && kind in (:intercept, :slope) && re[1][2] === descriptor.group &&
        descriptor.member === member ||
        throw(ArgumentError(
            "bootstrap: BetaBinomial sigma SD simulation requires one matching sigma random coefficient"))
    hasproperty(data, descriptor.group) || throw(ArgumentError(
        "bootstrap: data are missing sigma random-effect group `$(descriptor.group)`"))
    gidx, G = _group_index(getproperty(data, descriptor.group))
    _, Xsigma, nmsigma = _design(f.response, fixed_sigma, data;
        schema_cache = f.schema_cache, schema_key = :sigma, store_schema = false)
    nmsigma == _block_coefnames(fit, :sigma) || throw(ArgumentError(
        "bootstrap: BetaBinomial sigma design does not match the fitted coefficient schema"))
    Wsd, nmsd = _sd_group_design(f.response, target.rhs, data, gidx, G,
        descriptor.group; schema_cache = f.schema_cache, schema_key = target.key,
        store_schema = false)
    nmsd == _block_coefnames(fit, :sd) || throw(ArgumentError(
        "bootstrap: BetaBinomial sigma SD design does not match the fitted coefficient schema"))
    eta_fixed = Xsigma * coef(fit, :sigma)
    sd_group = exp.(Wsd * coef(fit, :sd))
    re_design = if kind === :intercept
        ones(length(gidx))
    else
        hasproperty(data, slope) || throw(ArgumentError(
            "bootstrap: data are missing sigma random-slope predictor `$slope`"))
        raw = getproperty(data, slope)
        raw isa AbstractVector{<:Real} && length(raw) == length(gidx) &&
            all(isfinite, raw) || throw(ArgumentError(
                "bootstrap: sigma random-slope predictor `$slope` must be a finite numeric vector with one value per response"))
        Float64.(raw)
    end
    length(gidx) == length(eta_fixed) == fit.nobs || throw(ArgumentError(
        "bootstrap: BetaBinomial sigma predictions, groups, and fitted responses do not align"))
    return function (rng)
        eta_sigma = eta_fixed .+ re_design .* sd_group[gidx] .* randn(rng, G)[gidx]
        sigma_star = exp.(clamp.(eta_sigma, -15.0, 15.0))
        return _simulate_once(fit, rng; sigma = sigma_star)
    end
end

# A Gaussian `sigma ~ 1 + (1 | g)` (and the same term next to a mean RE)
# stores the population (b = 0) scale in `scales[:sigma]`. The log-σ
# intercept SD is `re_sd(fit)[Symbol("$(g)_logsigma")]`. Nothing to draw
# when the scale formula has no ordinary intercept RE (#926).
function _sigma_ranef_components(fit::DrmFit, data)
    (fit.family isa Gaussian && fit.formula isa DrmFormula) || return nothing
    rhs = Dict(fit.formula.forms)
    haskey(rhs, :sigma) || return nothing
    _, sigma_re, _, structured_sigma = _split_ranef(rhs[:sigma])
    (structured_sigma === nothing && length(sigma_re) == 1) || return nothing
    first(_re_kind(sigma_re[1][1])) === :intercept || return nothing
    grp = sigma_re[1][2]
    hasproperty(data, grp) || return nothing
    labels = getproperty(data, grp)
    any(ismissing, labels) && return nothing
    gidx, G = _group_index(labels; col = grp)
    sds = re_sd(fit)
    key = Symbol("$(grp)_logsigma")
    (sds isa AbstractDict && haskey(sds, key)) || return nothing
    τ = Float64(sds[key])
    (isfinite(τ) && τ > 0) || return nothing
    return (; gidx, G, τ)
end

# Sigma-only route: no mean RE, so the earlier builder returned `nothing`
# and bootstrap used the conditional `simulate` (constant σ). Draw the
# log-σ intercept and scale each row's residual SD.
function _sigma_only_marginal_simulator(fit::DrmFit, data)
    sig = _sigma_ranef_components(fit, data)
    sig === nothing && return nothing
    haskey(fit.means, :mu) || return nothing
    mu = Vector{Float64}(fit.means[:mu])
    sigma0 = Vector{Float64}(_scale_vector(fit, :sigma))
    n = length(mu)
    (length(sigma0) == n && length(sig.gidx) == n) || return nothing
    return function (rng)
        u = sig.τ .* randn(rng, sig.G)
        return mu .+ (sigma0 .* exp.(u[sig.gidx])) .* randn(rng, n)
    end
end

# `Ainv` -> covariance `A` for the simulators of routes whose fitter converts the
# precision at entry (same helper, so the simulated kernel is the fitted one).
function _simulate_ainv_to_A(fit::DrmFit, data, A, Ainv)
    Ainv === nothing && return A
    A === nothing || throw(ArgumentError(
        "bootstrap: supply only one of `A` and `Ainv`"))
    Acov, _ = _ainv_as_labelled_A(fit.formula, data, nothing, Ainv, nothing)
    return Acov
end

# Several ORDINARY random intercepts on the mean, `(1 | g) + (1 | g2) [+ ...]`: the
# fields are independent, so a fresh draw is one `N(0, sd_k^2)` vector per grouping
# factor, summed on the linear-predictor scale (response scale for Gaussian). `re_sd`
# is keyed by group, so only intercepts on DISTINCT groups are covered; a slope term, a
# repeated group, an `sd(g) ~ ...` part or a ZeroOneBeta mean is not, and returns
# `nothing` (the caller then refuses, as before).
# Binomial / BetaBinomial draws need the trials per row. A fit from a route that did not
# store them (`fit.scales[:trials]`, for example a fit saved by an earlier version) gets
# them from the response columns of `data`
# (0/1 response: all ones; `cbind(successes, failures)`: their sum). Returns the fit
# unchanged for other families or when the trials are stored, and `nothing` when the
# trials cannot be rebuilt from `data` (the caller then refuses, naming `data =`).
function _fit_with_simulate_trials(fit::DrmFit, data)
    (fit.family isa Binomial || fit.family isa BetaBinomial) || return fit
    haskey(fit.scales, :trials) && return fit
    fit.formula isa DrmFormula || return nothing
    # The fitter's own parse of the response (`_binomial_response`), so the rebuild accepts
    # exactly what the fit accepted: non-negative integer successes and failures (rows with
    # zero trials included) or a 0/1 vector, and refuses everything else.
    tr = try
        Vector{Float64}(last(_binomial_response(fit.formula, data)))
    catch err
        err isa InterruptException && rethrow(err)
        return nothing
    end
    (length(tr) == fit.nobs && all(isfinite, tr)) || return nothing
    scales = merge(fit.scales, Dict{Symbol,Vector{Float64}}(:trials => tr))
    return DrmFit(
        fit.family, fit.blocks, fit.coefnames, fit.theta, fit.vcov,
        fit.loglik, fit.nobs, fit.converged, fit.means, fit.obs, scales,
        fit.formula, fit.nll, fit.nllgrad, fit.ranef,
        fit.estim_method, fit.reml_loglik, fit.ml_loglik, fit.marginal,
        fit.phylo_penalty, fit.penalty, fit.iterations, fit.phylo_scale,
        fit.uncertainty, fit.geometry,
    )
end

function _ordinary_multi_marginal_simulator(fit::DrmFit, data, re)
    isempty(_sd_parts(fit.formula)) && isempty(_sdphylo_parts(fit.formula)) || return nothing
    fit.family isa ZeroOneBeta && return nothing
    fit = _fit_with_simulate_trials(fit, data)
    fit === nothing && return nothing
    groups = Symbol[g for (_, g) in re]
    length(unique(groups)) == length(groups) || return nothing
    all(t -> first(_re_kind(t[1])) === :intercept, re) || return nothing
    sds = re_sd(fit)
    sds isa AbstractDict || return nothing
    comps = Tuple{Vector{Int},Int,Float64}[]
    for g in groups
        (hasproperty(data, g) && haskey(sds, g)) || return nothing
        labels = getproperty(data, g)
        any(ismissing, labels) && return nothing
        gidx, G = _group_index(labels; col = g)
        length(gidx) == fit.nobs || return nothing
        sd = Float64(sds[g])
        isfinite(sd) || return nothing
        push!(comps, (gidx, G, sd))
    end
    draw_re(rng) = sum(sd .* randn(rng, G)[gidx] for (gidx, G, sd) in comps)
    if fit.family isa Gaussian
        mu_fixed = try
            v = Vector{Float64}(predict(fit, data))
            length(v) == length(fit.means[:mu]) ? v : nothing
        catch e
            e isa InterruptException && rethrow()
            nothing
        end
        mu_fixed === nothing && return nothing
        sigma = Vector{Float64}(_scale_vector(fit, :sigma))
        sig = _sigma_ranef_components(fit, data)
        n = length(mu_fixed)
        return function (rng)
            σ = sig === nothing ? sigma : sigma .* exp.(sig.τ .* randn(rng, sig.G)[sig.gidx])
            return mu_fixed .+ draw_re(rng) .+ σ .* randn(rng, n)
        end
    end
    eta_fixed = try
        Vector{Float64}(predict(fit, data; type = :link))
    catch e
        e isa InterruptException && rethrow()
        nothing
    end
    (eta_fixed === nothing || length(eta_fixed) != fit.nobs) && return nothing
    return function (rng)
        mu_star = _mean_response(fit.family, eta_fixed .+ draw_re(rng))
        return _simulate_once(fit, rng; mu = mu_star)
    end
end

function _marginal_simulator_build(fit::DrmFit, data; K=nothing, A=nothing,
                                   Ainv=nothing, tree=nothing, tree1=nothing,
                                   tree2=nothing, coords=nothing)
    # temporal(): `simulate` already draws the full marginal model (a fresh chain
    # per series and a fresh `(1 | id)` intercept), so it IS the marginal simulator.
    _is_temporal_fit(fit) && return rng -> _temporal_simulate(fit, rng)
    # sigma random effects with constant SDs (src/sigma_ranef_blocks_laplace.jl)
    let sim = _sigma_blocks_marginal_simulator(fit, data)
        sim === nothing || return sim
    end
    fit.nll isa LocScaleObjective &&
        return _ls_marginal_simulator(fit, data; K, A, tree, coords)
    # NB2/Gamma structured sigma fields (src/sigma_structured_laplace.jl)
    if fit isa DrmFit && fit.formula isa DrmFormula && fit.family isa Union{NegBinomial2,Gamma} &&
       (haskey(fit.scales, :sigma_random_effect_slope) || _ss_has_sigma_structured_sd(fit.formula))
        # Gamma and NB2 take an animal precision `Ainv` at fit time by converting it to
        # the labelled covariance `A = inv(Ainv)`; the simulator needs that same `A`.
        A = _simulate_ainv_to_A(fit, data, A, Ainv)
        return _sigma_struct_marginal_simulator(fit, data; K, A, tree, coords)
    end
    if fit isa DrmFit && fit.formula isa DrmFormula &&
       fit.family isa Union{Poisson,NegBinomial2,Gaussian} &&
       haskey(Dict(fit.formula.forms), :mu) &&
       length(_sd_target_parts(fit.formula)) == 2
        for (provider, block) in ((:animal, _animal_intercept_slope(Dict(fit.formula.forms)[:mu];
                                                                    allow_label = true)),
                                  (:relmat, _relmat_intercept_slope(Dict(fit.formula.forms)[:mu];
                                                                    allow_label = true)))
            block === nothing && continue
            matrix = if provider === :relmat
                K
            elseif Ainv !== nothing
                A === nothing || throw(ArgumentError(
                    "bootstrap: supply only one of `A` and `Ainv` for animal intercept-and-slope simulation"))
                labels = _table_column(data, block.group)
                Q, _ = fit.family isa NegBinomial2 ?
                    _nb2_animal_precision_setup(Ainv, labels, block.group) :
                    _poisson_animal_precision_setup(Ainv, labels, block.group)
                Matrix(inv(Symmetric(Matrix(Q))))
            else
                A
            end
            return _structured_q2_sd_marginal_simulator(
                fit, data, matrix; provider, block)
        end
        # phylo / coordinate-spatial q2: rebuild the fit's kernel and row map.
        q2ps = _phylo_spatial_q2_block(fit.formula)
        if q2ps !== nothing
            kind, block = q2ps
            hasproperty(data, block.group) || throw(ArgumentError(
                "bootstrap: data are missing the $(kind) grouping column `$(block.group)`"))
            # Same label resolution as `drm()` (#957): a Dict `coords` is
            # matched to the site levels by name.
            kcoords = kind === :spatial ?
                _resolve_coords_labels(fit.formula, data, coords, nothing) : nothing
            kernel = _structured_q2_kernel(kind, block.group, getproperty(data, block.group);
                tree, K = nothing, A = nothing, coords = kcoords,
                who = "bootstrap: `$(kind)(1 + $(block.slope) | $(block.group))` simulation")
            return _structured_q2_sd_marginal_simulator(
                fit, data, nothing; provider = kind, block, kernel)
        end
    end
    # Ordinary correlated `(1 + x | g)` / labelled `(1 + x | p | g)` q2 (Poisson, NB2):
    # the structured q2 simulator with the identity kernel and the fitted rho.
    if fit isa DrmFit && fit.formula isa DrmFormula && fit.family isa Union{Poisson,NegBinomial2} &&
       length(_sd_target_parts(fit.formula)) == 2
        ob = _ordinary_q2_sd_block(fit.formula, fit.family)
        if ob !== nothing
            hasproperty(data, ob.group) || throw(ArgumentError(
                "bootstrap: data are missing the grouping column `$(ob.group)`"))
            gidx, G = _group_index(getproperty(data, ob.group))
            return _structured_q2_sd_marginal_simulator(fit, data, nothing;
                provider = :iid, block = (; group = ob.group, slope = ob.slope,
                    label = something(ob.label, :correlated)),
                kernel = (Matrix{Float64}(I, G, G), gidx, G))
        end
    end
    if fit isa DrmFit && fit.formula isa DrmFormula &&
       fit.family isa Union{Gamma,NegBinomial2,Beta} &&
       count(item -> item.target.dpar === :mu, _sd_target_parts(fit.formula)) == 2 &&
       length(_sd_target_parts(fit.formula)) == 2
        A = _simulate_ainv_to_A(fit, data, A, Ainv)   # Gamma fit-time `Ainv` -> `A`
        kc = fit.family isa Union{Gamma,Beta} && any(item -> item.target.level === :spatial, _sd_target_parts(fit.formula)) ?
            _resolve_coords_labels(fit.formula, data, coords, nothing) : coords
        sim = _twofield_sd_marginal_simulator(fit, data; K, A, tree, coords = kc)
        sim === nothing || return sim
    end
    _is_gaussian_lss(fit) && return _lss_marginal_simulator(fit, data; tree)
    fit.formula isa DrmFormula || return nothing
    # Gaussian needs a residual scale; other families carry their dispersion in the
    # family object or in `scales`, and some (Poisson) have none at all.
    (fit.family isa Gaussian && !haskey(fit.scales, :sigma)) && return nothing
    rhs = Dict(fit.formula.forms)
    haskey(rhs, :mu) || return nothing
    if fit.family isa Binomial && any(item ->
        item.target.level === :phylogenetic_interaction,
        _sd_target_parts(fit.formula))
        return _binomial_phylo_interaction_sd_marginal_simulator(
            fit, data, tree1, tree2)
    end
    if fit.family isa BetaBinomial && any(
        item -> item.target.dpar === :sigma, _sd_target_parts(fit.formula))
        return _betabinomial_sigma_sd_marginal_simulator(fit, data)
    end
    if fit.family isa Beta && !isempty(_sd_target_parts(fit.formula)) &&
       any(item -> item.target.level === :animal, _sd_target_parts(fit.formula))
        return _beta_animal_sd_marginal_simulator(fit, data, A)
    end
    allow_binomial_animal_slope = fit.family isa Binomial &&
        !isempty(_sd_target_parts(fit.formula))
    allow_binomial_phylo_slope_only = fit.family isa Binomial &&
        !isempty(_sdphylo_parts(fit.formula))
    allow_binomial_relmat_slope = fit.family isa Union{Binomial,Poisson} &&
        any(item.target.level === :relmat for item in _sd_target_parts(fit.formula))
    allow_binomial_spatial_slope = fit.family isa Union{Binomial,Gamma} &&
        any(item.target.level === :spatial for item in _sd_target_parts(fit.formula))
    allow_gamma_animal_slope = fit.family isa Gamma &&
        !isempty(_sd_target_parts(fit.formula))
    allow_gamma_animal_intercept_slope = allow_gamma_animal_slope
    allow_poisson_animal_slope = fit.family isa Poisson ||
        (fit.family isa NegBinomial2 &&
         any(item -> item.target.level === :animal, _sd_target_parts(fit.formula)))
    _, re, metav_ms, structured, structured_slope, _, animal_slope =
        _split_ranef(rhs[:mu]; allow_phylo_slope = true,
                     allow_phylo_slope_only = allow_binomial_phylo_slope_only,
                     allow_animal_slope = allow_binomial_animal_slope ||
                                           allow_gamma_animal_slope ||
                                           allow_poisson_animal_slope,
                     allow_relmat_slope = allow_binomial_relmat_slope,
                     allow_spatial_slope = allow_binomial_spatial_slope,
                     allow_animal_intercept_slope =
                         allow_gamma_animal_intercept_slope)
    # A Gaussian fit with SEVERAL random fields on the mean where at least one is
    # structured or the fit has `meta_V(...)` (Arc 2: `_fit_meta_gaussian_re`, e.g.
    # `phylo(1 | sp) + (1 | study)`, and the two-structured route, e.g.
    # `phylo(1 | sp) + relmat(1 | id)`): the simulator below draws ONE field, so it
    # would silently bootstrap a model with the other field(s) missing. Refuse.
    # Fits whose fields are all ordinary bars keep their existing path (the
    # `length(re) == 1` check below). A single field is drawn below: `scales[:sigma]`
    # is √(v + σ²) under meta_V, and phylo rows go to tree leaves as the fit maps
    # them (`phylo_leaf` below).
    if fit.family isa Gaussian && (metav_ms !== nothing || structured !== nothing)
        nfields = length(re) + length(_collect_structured(rhs[:mu]))
        nfields <= 1 ||
            throw(ArgumentError("bootstrap: the marginal simulator for a Gaussian fit " *
                "with $(nfields) random fields on the mean is not implemented; use " *
                "`profile` intervals or the Wald `vcov`"))
    end
    # A Gaussian `phylo(1 + x | g)` fit (#620) carries TWO phylogenetic fields;
    # the simulator below draws a single structured intercept, so building it
    # would silently bootstrap the wrong (intercept-only) model. Refuse.
    !(fit.family isa Gaussian && structured_slope !== nothing) ||
        throw(ArgumentError("bootstrap: the marginal simulator for the Gaussian " *
            "`phylo(1 + $(structured_slope) | $(structured[2]))` two-SD random-slope fit is " *
            "not implemented (#620); use `profile` intervals or the Wald `vcov`"))
    # No random effect on the mean: conditional and marginal coincide UNLESS
    # there is a scale-axis random intercept (`sigma ~ (1 | g)`). That case
    # used to fall through to `simulate`, which draws at b = 0 (#926).
    if isempty(re) && structured === nothing
        return _sigma_only_marginal_simulator(fit, data)
    end
    if fit.family isa Poisson && structured !== nothing &&
       first(structured) === :animal && animal_slope !== nothing &&
       isempty(_sd_target_parts(fit.formula))
        return _poisson_animal_slope_marginal_simulator(
            fit, data, A, animal_slope)
    end
    if fit.family isa Binomial &&
       (!isempty(_sd_target_parts(fit.formula)) ||
        !isempty(_sdphylo_parts(fit.formula)))
        return _binomial_sd_marginal_simulator(fit, data, re;
            structured, phylo_slope = structured_slope, animal_slope,
            spatial_slope = structured !== nothing && first(structured) === :spatial ? animal_slope : nothing,
            K, A, Ainv, tree, coords)
    end
    # NB2 animal slope-only SD (`animal(0 + x | g)`): the family-generic structured
    # simulator draws u = sd_group .* (chol(A).L * z) on the RAW A (as the NB2 fit
    # uses), then eta = eta_fixed + x * u[g], y ~ NB2(exp(eta), sigma).
    if fit.family isa NegBinomial2 && animal_slope !== nothing &&
       structured !== nothing && first(structured) === :animal &&
       any(item -> item.target.level === :animal, _sd_target_parts(fit.formula))
        matrix = if Ainv !== nothing
            A === nothing || throw(ArgumentError(
                "bootstrap: supply only one of `A` and `Ainv` for NB2 animal simulation"))
            group = last(structured)
            Q, _ = _nb2_animal_precision_setup(Ainv, _table_column(data, group), group)
            Matrix(inv(Symmetric(Matrix(Q))))
        else
            A
        end
        return _poisson_structured_sd_marginal_simulator(
            fit, data, matrix; provider = :animal, animal_slope)
    end
    if fit.family isa NegBinomial2 && Ainv !== nothing &&
       any(item -> item.target.level === :animal, _sd_target_parts(fit.formula))
        A === nothing || throw(ArgumentError(
            "bootstrap: supply only one of `A` and `Ainv` for NB2 animal simulation"))
        return _negbin2_animal_sd_precision_marginal_simulator(fit, data, Ainv)
    end
    if fit.family isa Poisson && any(
        item -> item.target.level in (:relmat, :animal), _sd_target_parts(fit.formula))
        matching = filter(
            item -> item.target.level in (:relmat, :animal),
            _sd_target_parts(fit.formula))
        length(matching) == 1 || throw(ArgumentError(
            "bootstrap: Poisson structured SD simulation requires exactly one structured target"))
        provider = only(matching).target.level
        matrix = if provider === :animal && Ainv !== nothing
            A === nothing || throw(ArgumentError(
                "bootstrap: supply only one of `A` and `Ainv` for Poisson animal SD simulation"))
            group = only(matching).target.group
            labels = _table_column(data, group)
            Q, _ = _poisson_animal_precision_setup(Ainv, labels, group)
            Matrix(inv(Symmetric(Matrix(Q))))
        else
            provider === :animal ? A : K
        end
        return _poisson_structured_sd_marginal_simulator(
            fit, data, matrix; provider,
            animal_slope = animal_slope)
    end
    if fit.family isa Gamma && any(
        item -> item.target.level === :spatial, _sd_target_parts(fit.formula))
        length(_sd_target_parts(fit.formula)) == 1 || throw(ArgumentError(
            "bootstrap: Gamma structured SD simulation requires exactly one structured target"))
        group = only(_sd_target_parts(fit.formula)).target.group
        hasproperty(data, group) || throw(ArgumentError(
            "bootstrap: data are missing Gamma structured grouping column `$group`"))
        coords === nothing && throw(ArgumentError(
            "bootstrap: supply the same `coords` used to fit the Gamma spatial SD model"))
        # Same label resolution and the same shared kernel as the fit (`_drm_gamma_sd_regression`).
        kcoords = _gamma_coords_matrix(_resolve_coords_labels(fit.formula, data, coords, nothing))
        _, G = _group_index(getproperty(data, group))
        return _gamma_structured_sd_marginal_simulator(
            fit, data, _gaussian_spatial_sd_kernel(kcoords, G, group);
            provider = :spatial, animal_slope)
    end
    if fit.family isa Gamma && any(
        item -> item.target.level in (:relmat, :animal), _sd_target_parts(fit.formula))
        matching = filter(
            item -> item.target.level in (:relmat, :animal),
            _sd_target_parts(fit.formula))
        animal_block = _animal_intercept_slope(rhs[:mu])
        if animal_block !== nothing && length(matching) == 2 &&
           all(item -> item.target.level === :animal &&
                       item.target.group === animal_block.group, matching)
            return _gamma_animal_intercept_slope_sd_marginal_simulator(
                fit, data, A; animal_block)
        end
        length(matching) == 1 || throw(ArgumentError(
            "bootstrap: Gamma structured SD simulation requires exactly one structured target"))
        descriptor = only(matching).target
        provider = descriptor.level
        matrix = provider === :animal ? A : K
        return _gamma_structured_sd_marginal_simulator(
            fit, data, matrix; provider,
            animal_slope = provider === :animal ? animal_slope : nothing)
    end
    if fit.family isa Gamma && !isempty(_sdphylo_parts(fit.formula))
        return _gamma_phylo_sd_marginal_simulator(fit, data, tree)
    end
    if fit.family isa NegBinomial2 && haskey(fit.scales, :hu) &&
       !isempty(_sd_target_parts(fit.formula))
        return _negbin2_hu_sd_marginal_simulator(fit, data, re)
    end

    # Which grouping factor, and what covariance does its random effect have?
    phylo_leaf = nothing   # row → tree-leaf index, set for a phylo field
    re_weight = ones(Float64, fit.nobs)
    grp, Kg = if structured !== nothing
        g = structured[2]
        hasproperty(data, g) || return nothing
        _, G0 = _group_index(getproperty(data, g); col = g)
        # SCALE TRAP -- verified by round-trip, not assumed. `re_sd` for a PHYLO term
        # is defined against the RAW covariance `sigma_phy_dense(phy)`, whose diagonal
        # is the tree height, NOT the normalised correlation that
        # `_resolve_structured_matrix` returns. Drawing with the correlation matrix
        # under-disperses by sqrt(height): measured round-trip ratios across trees of
        # height 0.85/1.7/5.667 were 1.116/0.699/0.374 with the correlation matrix and
        # 1.032/0.917/0.917 with the raw one. relmat/animal are supplied by the user
        # and used as given, so there the resolver's matrix is already correct.
        if structured[1] === :phylo
            phy = tree isa AbstractString ? augmented_phy(tree) : tree
            phy === nothing && return nothing
            # Rows → tree LEAVES by name / tip index (#482, `_phylo_mean_leaf_index`),
            # exactly as every Gaussian and non-Gaussian phylo mean fit maps them,
            # NOT by first-seen order. First-seen order drew the field on the wrong
            # tips whenever the data's species order differed from the tree's
            # (measured: sister-tip covariance 0.294 in the model, -0.005 in 6000
            # draws), and a tree with tips absent from the data failed the size
            # check and fell back to `simulate`. The tip matrix follows the fit's
            # SD scale (`fit.phylo_scale`): the raw covariance on the sparse,
            # meta_V and Laplace routes; the tip CORRELATION on the dense Gaussian
            # fallback (`sigma ~ x`, `algorithm = :gls`/`:lbfgs`) and the
            # two-structured route, where a raw-covariance draw over-disperses the
            # field by the tree height (measured, height 1.5: drawn cov 0.284 vs
            # 0.186 in the model; also wrong on main).
            phylo_leaf = (_phylo_mean_leaf_index(phy, getproperty(data, g)), phy.n_leaves)
            (g, fit.phylo_scale === :correlation ? _phylo_correlation(phy) :
                                                   sigma_phy_dense(phy; σ²_phy = 1.0))
        elseif structured[1] === :spatial && K === nothing && coords !== nothing
            _require_finite_inputs(; coords=coords, context="spatial bootstrap")
            cmat = Matrix{Float64}(coords)
            size(cmat, 1) == G0 ||
                throw(ArgumentError("spatial bootstrap coords must have one row per `$g` level (G = $G0)"))
            size(cmat, 2) >= 1 ||
                throw(ArgumentError("spatial bootstrap coords need at least one coordinate column"))
            if any(b -> first(b) === :range, fit.blocks)
                ρ = exp(only(coef(fit, :range)))
                isfinite(ρ) && ρ > 0 ||
                    throw(ArgumentError("spatial bootstrap requires a positive finite fitted range"))
                D = [sqrt(sum(abs2, @view(cmat[k, :]) .- @view(cmat[l, :])))
                     for k in 1:G0, l in 1:G0]
                # Match the coordinate-spatial fitting routes exactly: the random
                # effect SD is defined against exp(-distance / fitted range), with
                # the same numerical diagonal jitter used during fitting.
                (g, exp.(-D ./ ρ) + 1e-8I)
            else
                # Fits with no `:range` block (Poisson, Gamma and the other count routes)
                # were fitted with drmTMB's fixed native kernel; draw from that one.
                (g, _gaussian_spatial_sd_kernel(cmat, G0, g))
            end
        else
            (g, _resolve_structured_matrix(structured[1], g, G0;
                                           K=K, A=A, tree=tree, coords=coords))
        end
    else
        # Ordinary independent scalar effect. A slope uses the supplied numeric
        # covariate as its observation-level loading on the group coefficient.
        length(re) > 1 && return _ordinary_multi_marginal_simulator(fit, data, re)
        length(re) == 1 || return nothing
        re_kind, re_slope = _re_kind(re[1][1])
        re_kind in (:intercept, :slope) || return nothing
        g = re[1][2]
        hasproperty(data, g) || return nothing
        if re_kind === :slope
            hasproperty(data, re_slope) || return nothing
            raw_weight = getproperty(data, re_slope)
            raw_weight isa AbstractVector{<:Real} || return nothing
            all(isfinite, raw_weight) || return nothing
            re_weight = Float64.(raw_weight)
        end
        _, G0 = _group_index(getproperty(data, g); col = g)
        (g, Matrix{Float64}(LinearAlgebra.I, G0, G0))
    end

    gidx, G = phylo_leaf === nothing ? _group_index(getproperty(data, grp); col = grp) : phylo_leaf
    size(Kg) == (G, G) || return nothing
    # Location-scale-scale fits (#544/#545): the RE SD is per group,
    # σ_g,k = exp(Z_k' α), so the draw scales each group's effect individually.
    # For the PHYLO lss fit the α coefficients are defined against the
    # NORMALISED correlation (that is what `_fit_structured_gaussian_lss`
    # receives via `_phylo_correlation`), so the draw must use the correlation
    # too — the raw-covariance SCALE TRAP note above applies to the scalar
    # `re_sd` definition, not to this route.
    lss_sd = _sd_parts(fit.formula); lss_phy = _sdphylo_parts(fit.formula)
    sd_g = if !isempty(lss_sd) || !isempty(lss_phy)
        (sgrp, srhs) = isempty(lss_phy) ? lss_sd[1] : lss_phy[1]
        sgrp === grp || return nothing
        if !isempty(lss_phy)
            phy = tree isa AbstractString ? augmented_phy(tree) : tree
            phy === nothing && return nothing
            Kg = _phylo_correlation(phy)
            size(Kg) == (G, G) || return nothing
        end
        Zg, _ = _sd_group_design(fit.formula.response, srhs, data, gidx, G, grp)
        exp.(Zg * coef(fit, isempty(lss_phy) ? :sd : :sd_phylo))
    else
        sds = re_sd(fit)
        (sds isa AbstractDict && haskey(sds, grp)) || return nothing
        fill(Float64(sds[grp]), G)
    end
    L = cholesky(Symmetric(Matrix{Float64}(Kg))).L

    # The FIXED-effect mean comes from `predict(fit, data)`, not from unpicking
    # `fit.means[:mu]`.
    #
    # Whether `means[:mu]` is conditional or marginal depends on the fitting route,
    # and there is NO usable rule -- measured 2026-08-24 across three routes:
    #
    #   route            ranef has BLUPs   means[:mu] is
    #   phylo(1|g)       yes               CONDITIONAL  (|means - Xb| = 1.53)
    #   relmat(1|g)      no                MARGINAL     (|means - Xb| = 0)
    #   ordinary (1|g)   yes               MARGINAL     (|means - Xb| = 0)
    #
    # So BLUP presence predicts nothing. Both plausible rules were tried and both
    # were wrong: subtracting whenever BLUPs exist double-REMOVES the random effect
    # on the ordinary route (round-trip ratio 1.46 ~ the sqrt(2) signature of
    # counting it twice), and never subtracting double-COUNTS it on the phylo route.
    #
    # `predict(fit, data)` returns the fixed-effect prediction on ALL THREE routes
    # (measured: |predict - Xb| = 0 exactly in each), so it answers the question
    # directly instead of inferring it. Fall back to `means[:mu]` only if `predict`
    # is unavailable, and refuse rather than guess if that is also unusable.
    mu_fixed = try
        v = Vector{Float64}(predict(fit, data))
        length(v) == length(fit.means[:mu]) ? v : nothing
    catch
        nothing
    end
    mu_fixed === nothing && return nothing

    if fit.family isa Gaussian
        sigma = Vector{Float64}(_scale_vector(fit, :sigma))
        n = length(mu_fixed)
        sig = _sigma_ranef_components(fit, data)
        return function (rng)
            u = sd_g .* (L * randn(rng, G))
            σ = sig === nothing ? sigma : sigma .* exp.(sig.τ .* randn(rng, sig.G)[sig.gidx])
            return mu_fixed .+ re_weight .* u[gidx] .+ σ .* randn(rng, n)
        end
    end

    # NON-GAUSSIAN (#462). The Gaussian branch above adds the random effect on the
    # RESPONSE scale because identity is the link. Everywhere else the random effect
    # lives on the LINK scale and has to pass through the inverse link before the
    # family draw:
    #
    #     eta* = Xb + Z u*        u* ~ N(0, sd_u^2 K)
    #     mu*  = linkinv(eta*)
    #     y*   ~ Family(mu*, aux)
    #
    # Measured before the fix: a Poisson `(1|g)` fit produced replicates with a
    # between-group SD of 0.696 against 2.690 in the observed data -- roughly a
    # quarter of the real group structure -- because the conditional simulator
    # reused the fitted mean.
    #
    # `predict(fit, data; type = :link)` is the marginal linear predictor (measured
    # exact for Poisson: |predict(link) - Xb| = 0), and `_simulate_once(fit, rng;
    # mu = ...)` reuses the verified per-family draw at the new mean rather than
    # duplicating it here.
    eta_fixed = try
        Vector{Float64}(predict(fit, data; type = :link))
    catch
        nothing
    end
    eta_fixed === nothing && return nothing
    length(eta_fixed) == fit.nobs || return nothing
    return function (rng)
        u = sd_g .* (L * randn(rng, G))
        eta_star = eta_fixed .+ re_weight .* u[gidx]
        if fit.family isa ZeroOneBeta
            # The beta component has its own mean parameter. `_simulate_once`
            # normally reads that fitted mean from the fit, so passing `mu_star`
            # through its generic `mu` keyword would discard the newly drawn
            # random effect. Draw the mixture directly using the updated mean.
            mu_beta = _logistic.(clamp.(eta_star, -30.0, 30.0))
            sigma = _scale_vector(fit, :sigma)
            phi = @. 1 / (sigma * sigma)
            zoi = _scale_vector(fit, :zoi)
            coi = _scale_vector(fit, :coi)
            return Float64[_rand_zeroonebeta(rng, mu_beta[i], phi[i], zoi[i], coi[i])
                           for i in eachindex(mu_beta)]
        end
        mu_star = _mean_response(fit.family, eta_star)
        return _simulate_once(fit, rng; mu = mu_star)
    end
end

function _bootstrap_result(
    fit0,
    formula::Union{DrmFormula,BivariateDrmFormula},
    data,
    B::Real,
    level::Real,
    rng,
    threads::Bool,
    refit;
    failures::Symbol=:error,
    check_converged::Bool=true,
    simulate_fn=nothing,
)
    _check_bootstrap_failure_mode(failures)
    B = _validate_bootstrap_B(B)
    level = _validate_ci_level(level; what="bootstrap")
    est = coef(fit0)
    p = length(est)
    draws = Matrix{Float64}(undef, B, p)
    # Distinct BitVector indices share a machine word: parallel writes can
    # silently lose a successful replicate. Byte-addressable flags are independent.
    ok = fill(false, B)
    messages = Vector{Union{Nothing,String}}(nothing, B)
    # A thrown refit is recorded. The default `failures = :error` then aborts
    # after every replicate is recorded. `failures = :skip` keeps the
    # successful draws and sets `bootstrap_incomplete`. A non-converged refit
    # is always a drop: it warns and does not abort under `failures = :error`.
    # `Vector{Bool}` is bit-packed, so parallel writes can drop a flag.
    refit_error = zeros(UInt8, B)
    refit_estimator = fill((:unknown, :unknown), B)
    seeds = rand(rng, UInt, B)

    function run_one!(b)
        rr = Random.MersenneTwister(seeds[b])
        try
            # Marginal draw when the fit has random effects (#459); the plain
            # `simulate(fit0; re = :zero)` is conditional and collapses a variance-component CI.
            # `re = :zero` keeps the fallback's previous draw (#1092 changed the
            # `simulate` default; this path is reached only without a simulator).
            ysim = simulate_fn === nothing ? simulate(fit0; rng=rr, re=:zero) : simulate_fn(rr)
            datab = _bootstrap_data(formula, data, ysim)
            fitb = _without_boundary_warnings(() -> refit(datab))
            # Not the raw `.converged` field: this gate also rejects a degenerate
            # optimum (sigma collapsed, likelihood runaway), which the optimiser's
            # own flag happily calls converged (#461). Not `is_converged` either
            # (FX3b): a refit whose variance component lands on its boundary is a
            # real draw, and dropping it would bias the interval away from zero.
            if check_converged && !_converged_nondegenerate(fitb)
                messages[b] = _BOOTSTRAP_UNCONVERGED
            else
                draws[b, :] = coef(fitb)
                refit_estimator[b] = _bootstrap_estimator(fitb)
                ok[b] = true
            end
        catch err
            messages[b] = sprint(showerror, err)
            refit_error[b] = 0x01
        end
        return nothing
    end

    threaded = threads && Threads.nthreads() > 1
    elapsed = @elapsed begin
        if threaded
            _with_pinned_blas(threaded) do
                Threads.@threads for b in 1:B
                    run_one!(b)
                end
            end
        else
            for b in 1:B
                run_one!(b)
            end
        end
    end

    failure_rows = _BootstrapFailureRow[]
    for b in 1:B
        messages[b] === nothing && continue
        push!(failure_rows, (replicate=b, seed=seeds[b], message=messages[b]::String))
    end
    if failures === :error && any(!iszero, refit_error)
        first_b = findfirst(!iszero, refit_error)
        throw(
            ErrorException(
                "bootstrap failed in $(count(!iszero, refit_error)) of $B replicates; first failure replicate $(first_b), seed $(seeds[first_b]): $(messages[first_b])",
            ),
        )
    end
    used = count(ok)
    if used == 0
        first_fail = isempty(failure_rows) ? nothing : failure_rows[1]
        extra = first_fail === nothing ? "" :
            "; first failure replicate $(first_fail.replicate), seed $(first_fail.seed): $(first_fail.message)"
        throw(ErrorException("all $B bootstrap replicates failed$extra"))
    end
    kept = draws[ok, :]
    summary = _homtoep_bootstrap_rows(fit0, _bootstrap_summary_rows(fit0, kept, est, level))
    boundary_params = _bootstrap_boundary_params(fit0, kept)
    failed_n = length(failure_rows)
    status = _finish_bootstrap_status(used, failed_n, boundary_params)
    seed_method, seed_marginal = _bootstrap_estimator(fit0)
    kept_est = unique(refit_estimator[ok])
    refit_method = length(kept_est) == 1 ? kept_est[1][1] : :mixed
    refit_marginal = length(kept_est) == 1 ? kept_est[1][2] : :mixed
    (refit_method, refit_marginal) == (seed_method, seed_marginal) ||
        seed_method === :unknown ||
        @warn "bootstrap replicates used a different estimator from the seed fit; the interval does not describe the point estimate's estimator" seed_method seed_marginal refit_method refit_marginal
    return (
        summary=summary,
        failures=failure_rows,
        attempted=B,
        used=used,
        failed=failed_n,
        seeds=seeds,
        estimation_method=seed_method,
        marginal=seed_marginal,
        refit_estimation_method=refit_method,
        refit_marginal=refit_marginal,
        threaded=threaded,
        worker_threads=_worker_threads(threaded, B),
        julia_threads=Threads.nthreads(),
        blas_threads=BLAS.get_num_threads(),
        blas_oversubscribed=_blas_oversubscribed(threaded),
        elapsed=elapsed,
        check_converged=check_converged,
        status=status,
        boundary_params=boundary_params,
    )
end

function _check_bootstrap_failure_mode(failures::Symbol)
    failures === :error && return nothing
    failures === :skip && return nothing
    throw(ArgumentError("bootstrap failures must be :error or :skip (got :$failures)"))
end

# Bivariate Gaussian replicate. `_simulate_once` returns a
# `Dict(:mu1 => y1*, :mu2 => y2*)` for a bivariate fit, so both response columns
# are replaced at once. The OBSERVATION PATTERN is part of the design, not part
# of the model: a cell that was unobserved in the data stays unobserved in the
# replicate, so every refit sees the same likelihood structure the seed fit did.
# (`_simulate_once` draws at every row, including rows the seed fit's
# `_observed_response_mask` excluded, so without this the replicates would fit a
# COMPLETE-DATA model and silently disagree with the seed fit.)
function _bootstrap_data(formula::BivariateDrmFormula, data, ysim)
    ysim isa AbstractDict || throw(ArgumentError(
        "bivariate bootstrap expects a `Dict(:mu1 => ..., :mu2 => ...)` draw " *
        "(got $(typeof(ysim)))"))
    y1 = _bootstrap_keep_unobserved(getproperty(data, formula.response1), ysim[:mu1])
    y2 = _bootstrap_keep_unobserved(getproperty(data, formula.response2), ysim[:mu2])
    return merge(data, NamedTuple{(formula.response1, formula.response2)}((y1, y2)))
end

# Copy `sim` where the original response was observed, keep the original cell
# (`missing` or `NaN`) where it was not. The output keeps the column's own
# eltype widened to hold a Float64 draw, so a `Vector{Union{Missing,Float64}}`
# stays one while an integer-typed response column (which the fit accepts) does
# not make every replicate fail with an `InexactError`. This is the
# bivariate sibling of `_restore_response_mask!` below: both decide "absent"
# with `_is_response_missing`, the predicate the fit itself uses, so the mask
# restored here is the mask the seed fit used. They differ only in shape:
# this one is non-mutating, keeps the column's own representation, and refuses
# a wrong-length draw; the univariate one normalises to Float64/NaN in place
# because the `cbind` route needs NaN arithmetic to carry the mask into the
# failure column.
function _bootstrap_keep_unobserved(orig::AbstractVector, sim::AbstractVector)
    length(orig) == length(sim) || throw(ArgumentError(
        "bivariate bootstrap draw has length $(length(sim)), expected $(length(orig))"))
    out = similar(orig, promote_type(eltype(orig), Float64))
    @inbounds for i in eachindex(out)
        o = orig[i]
        out[i] = _is_response_missing(o) ? o : sim[i]
    end
    return out
end

# Re-apply the seed fit's response mask to a simulated replicate draw
# (drmTMB #1188). `simulate` draws over the FULL design -- it must, because
# `means`/`scales` are rebuilt over the full design on the missing-response
# routes (#646) -- so merging the draw back wholesale hands every replicate
# MORE rows than the seed fit actually observed. A bootstrap whose replicates
# are richer than the original necessarily understates uncertainty, and the
# narrowing deepens with the missing fraction.
#
# MEASURED on the R side of the same defect (drmTMB `bootstrap_response_data`,
# same fixture family: `y = 0.3 + 0.5x + N(0,1) exp(0.1x)`, n = 60,
# `bf(y ~ x, sigma ~ x)`, Gaussian, `fixef:mu:x`, truth 0.5; S = 200 datasets,
# B = 99 replicates each, nominal 95% percentile interval):
#
#   masked   before   after    Wald reference
#   10%      0.895    0.910    0.920
#   30%      0.820    0.895    0.925
#   50%      0.720    0.910    0.915
#
# `_is_response_missing` treats both `missing` and NaN as absent, exactly as
# `_coerce_response_column` does when the fit reads the response, so the mask
# restored here is the mask the fit itself used.
#
# A draw whose length differs from the data is an error, not a no-op: merged
# back as-is, a short draw sits beside full-length covariates and every
# replicate refit fails (or misaligns rows).
function _restore_response_mask!(ysim::AbstractVector{Float64}, raw)
    length(raw) == length(ysim) || throw(ArgumentError(
        "bootstrap draw has length $(length(ysim)), expected $(length(raw)) (one value " *
        "per row of `data`); pass the same `data` the model was fitted to. If `drm` " *
        "dropped rows with a missing value when it fitted the model (#1089), pass " *
        "`drm_listwise(formula, data)` as `data`"))
    @inbounds for i in eachindex(ysim)
        if _is_response_missing(raw[i])
            ysim[i] = NaN
        end
    end
    return ysim
end

function _bootstrap_data(formula::DrmFormula, data, ysim)
    raw1 = getproperty(data, formula.response)
    if formula.response2 === nothing
        y = _restore_response_mask!(collect(Float64, ysim), raw1)
        return merge(data, NamedTuple{(formula.response,)}((y,)))
    end
    # A `cbind(successes, failures)` response is absent when EITHER cell is
    # absent, so coerce both columns through the same NaN convention and let
    # NaN arithmetic carry the mask into the failure column.
    y1, _ = _coerce_response_column(raw1)
    y2, _ = _coerce_response_column(getproperty(data, formula.response2))
    ntr = y1 .+ y2
    succ = _restore_response_mask!(collect(Float64, ysim), ntr)
    fail = ntr .- succ
    return merge(data, NamedTuple{(formula.response, formula.response2)}((succ, fail)))
end

function _bootstrap_ci_rows(rows)
    return _CIRow[
        (param=r.param, coef=r.coef, estimate=r.estimate, lower=r.lower, upper=r.upper) for
        r in rows
    ]
end

function _bootstrap_boundary_params(fit0, draws)
    names = String[]
    size(draws, 1) < _BOOTSTRAP_BOUNDARY_MIN_DRAWS && return names
    for ((pp, r), (_, nms)) in zip(fit0.blocks, fit0.coefnames)
        for (j, col) in enumerate(r)
            _bootstrap_param_at_boundary(pp, @view draws[:, col]) || continue
            push!(names, string(pp, ":", nms[j]))
        end
    end
    return names
end

function _bootstrap_summary_rows(fit0, draws, est, level)
    α = (1 - level) / 2
    rows = _BootstrapSummaryRow[]
    # Index by the stored UnitRange `r` of each block, NOT a private sequential
    # counter (issue #325.3): `coef`/`vcov`/`draws` columns are in θ order, so the
    # block's own range `r` is the correct column set. A sequential `col += 1` is
    # correct only when the blocks partition 1:p contiguously and in θ order — a
    # latent misalignment if a future/bivariate fit reorders or gaps its blocks.
    for ((pp, r), (_, nms)) in zip(fit0.blocks, fit0.coefnames)
        for (j, col) in enumerate(r)
            v = @view draws[:, col]
            push!(
                rows,
                (
                    param=pp,
                    coef=nms[j],
                    estimate=est[col],
                    std_error=Statistics.std(v),
                    lower=Statistics.quantile(v, α),
                    upper=Statistics.quantile(v, 1 - α),
                ),
            )
        end
    end
    return rows
end

# Max |d nll / d theta| at the estimate, AND where the number came from. A
# location-scale fit carries a `LocScaleObjective` whose inner mode-solve is
# Float64-only (not dual-number safe), so ForwardDiff can't differentiate it --
# use its exact analytic outer gradient instead (which also upgrades the
# diagnostic from NaN to a real gradient norm). Sparse q=4 fits can store a
# Float64-only gradient callback; use it before trying ForwardDiff through the
# objective. `nothing` means no objective -> (NaN, :none).
#
# WHY THE FALLBACK AND WHY THE SOURCE (measured 2026-09-05, DRModels.jl origin/main
# 109b6421c). FOUR shipping routes store a bare objective (no gradient callback)
# that is exact on Float64 but NOT dual-number safe, so the unguarded
# `ForwardDiff.gradient` on the last line THREW and the health check crashed on
# the very fits it exists to report on:
#
#   bivariate q=2 structured (`gaussian_bivariate.jl:626`; its objective calls
#     `coevo_marginal_cov`, which factorises a sparse H_uu through CHOLMOD at
#     `coevolution_q.jl:245`) and the sparse two-structured Gaussian mean route
#     (`gaussian_structured.jl:652`, same CHOLMOD factorisation at :529) both
#     raised `TypeError: in Sparse, in Tv, expected Tv<:Union{Float64,
#     ComplexF64}, got Type{ForwardDiff.Dual{...}}`;
#   sparse LSS under REML, single-component (`gaussian_sparse_lss.jl:285`) and
#     multi-component (`:1018`) -- both store `reml ? nothing : nllgrad!`, so
#     only the REML arm reaches this line -- raised the DIFFERENT
#     `MethodError: no method matching Float64(::ForwardDiff.Dual{...})` from a
#     Float64 work array. Two failure modes, hence a broad `catch`.
#
# The fix REPORTS instead of raising, which is what a diagnostic is for (the
# same argument the `vcov_complete` branch of `check_drm` below makes for a
# deliberately partial covariance). It does not return NaN and
# stop there: `check_drm`'s verdict is
# `ok = converged && (penalized || isnan(mag) || mag <= grad_tol) && pd`, whose
# `isnan(mag)` disjunct is a PASS-THROUGH written for a fit that stores no
# objective at all. A NaN here would silently borrow it, so a route whose
# stationarity was never tested would report `ok = true` and look identical to a
# clean fit -- the same defect as the throw, only quiet. A central finite
# difference of the SAME objective keeps the criterion live (the pattern
# `_profile_autodiff_mode` at :805-821 already uses for this exact situation),
# and `source` says which producer supplied the number, because a
# finite-difference magnitude is good to about 1e-6 relative, not to machine
# precision. `:unavailable` is the last resort -- a NaN that is explicitly
# labelled and warned about, so it can never be mistaken for `:none`.
const _CHECK_GRAD_FD_STEP = 1e-5

# NaN whenever a producer throws OR yields a non-finite magnitude: to the caller
# those are the same answer -- this producer cannot supply a usable number.
function _check_grad_attempt(producer)
    try
        m = producer()
        return isfinite(m) ? Float64(m) : NaN
    catch err
        err isa InterruptException && rethrow()
        return NaN
    end
end

# Central difference of the Float64 objective, at the same relative step as
# `_loconly_fd_gradient2` (src/location_only.jl:573). Costs 2p objective
# evaluations, with p the outer parameter count (tens on every route that needs
# this). Returns NaN as soon as any piece is non-finite: a fabricated magnitude
# would be worse than an honest "unavailable".
function _check_fd_max_abs_grad(nll, theta::Vector{Float64}; h::Real = _CHECK_GRAD_FD_STEP)
    isfinite(nll(theta)) || return NaN
    x = copy(theta)
    m = 0.0
    for i in eachindex(x)
        xi = x[i]
        s = h * max(abs(xi), 1.0)
        x[i] = xi + s; fp = nll(x)
        x[i] = xi - s; fm = nll(x)
        x[i] = xi
        gi = (fp - fm) / (2 * s)
        isfinite(gi) || return NaN
        m = max(m, abs(gi))
    end
    return m
end

function _check_grad_fallback(nll, theta::Vector{Float64})
    m = _check_grad_attempt(() -> _check_fd_max_abs_grad(nll, theta))
    isnan(m) && return (magnitude = NaN, source = :unavailable)
    return (magnitude = m, source = :finite)
end

function _check_max_abs_grad(fit::DrmFit)
    nll = fit.nll
    nll === nothing && return (magnitude = NaN, source = :none)
    if nll isa LocScaleObjective
        base = size(nll.Xμ, 2) + size(nll.Xψ, 2)
        perm = vcat(collect(1:base), [base + 1, base + 3, base + 2])  # recov->engine
        theta = fit.theta[perm]
        m = _check_grad_attempt(() -> maximum(abs, _ls_objective_gradient(nll, theta)))
        isnan(m) || return (magnitude = m, source = :locscale)
        return _check_grad_fallback(nll, theta)
    end
    if fit.nllgrad !== nothing
        m = _check_grad_attempt(function ()
            g = zeros(length(fit.theta))
            fit.nllgrad(g, fit.theta)
            return maximum(abs, g)
        end)
        isnan(m) || return (magnitude = m, source = :stored)
        return _check_grad_fallback(nll, fit.theta)
    end
    m = _check_grad_attempt(() -> maximum(abs, ForwardDiff.gradient(nll, fit.theta)))
    isnan(m) || return (magnitude = m, source = :forward)
    return _check_grad_fallback(nll, fit.theta)
end

"""
    check_drm(fit) -> NamedTuple

Post-fit convergence / identifiability diagnostics — drmTMB's `check_drm()`.
Returns a `NamedTuple` and logs a short report:

- `convergence_status` — [`convergence_status`](@ref)`(fit)`: `:converged`,
  `:boundary` or `:degenerate`, by the same rules as drmTMB (FX3b).
- `converged` — [`is_converged`](@ref)`(fit)`, i.e. `convergence_status ===
  :converged`. A Gaussian fit whose scale collapsed is reported `false` here even
  when `fit.converged` is `true` (#949), and so is a `:boundary` fit (FX3b).
- `max_abs_grad` — `max|∇nll|` at the optimum (≈ 0 at a clean interior optimum;
  `NaN` when no gradient could be produced: see `grad_source` for which of the
  two reasons applies).
- `grad_source` names which producer supplied `max_abs_grad`, so a caller can tell an
  exact gradient from an approximate one and either from no gradient at all.
  Reuses the `autodiff` vocabulary of [`profile_result`](@ref): `:locscale` (the
  canonical location-scale objective's exact analytic outer gradient), `:stored`
  (the fit's own gradient callback), `:forward` (ForwardDiff through the stored
  objective), `:finite` (a central finite difference of the stored objective,
  used when the objective is exact on `Float64` but not dual-number safe, and
  accurate to roughly `1e-6` relative, NOT to machine precision), `:none` (the
  fit stores no objective, so there is nothing to differentiate), and
  `:unavailable` (an objective IS stored but neither its derivative nor a finite
  difference of it produced a finite value). `:none` and `:unavailable` are the
  two `NaN` cases and they mean different things; both leave `ok` **unscored**
  against the gradient criterion, and `:unavailable` also warns.
- `vcov_complete` — whether the stored covariance is finite throughout. Some
  routes report a **partial** covariance by design: the sparse phylo fitter
  computes the fixed-effect block and leaves the variance-component block `NaN`.
  When this is `false` the three fields below cannot be computed and are reported
  as `false` / `NaN` / `Inf` rather than raising.
- `vcov_posdef` — whether the stored covariance is positive-definite (drmTMB's
  `sdreport` is all-`NaN` exactly when this fails).
- `min_eigval` / `cond` — smallest eigenvalue and condition number of the
  covariance; a near-zero `min_eigval` flags a singular / weakly identified
  direction (e.g. a variance pinned at the boundary).
- `penalized_map` — whether this is a penalized (MAP) fit, using a phylogenetic
  penalty or fixed-effect priors. Standard errors describe local penalized
  curvature, and `loglik` is the unpenalized data log-likelihood.
- `variance_boundary` — `nothing` unless the fit is Gaussian with a grouped / structured
  random effect and a homoscedastic residual (`sigma ~ 1`); then the NamedTuple of
  `_variance_boundary(fit)` (`residual_at_boundary`, `residual_ratio`,
  `structured_at_boundary`, `structured_ratios`, `one_obs_per_group`). Flags a variance
  component at its lower boundary (the residual σ̂ of a phylogenetic fit whose structured
  term absorbs all the variance is an optimiser-stopping artefact, not an estimate — #724)
  and says when σ_a and σ_e are separated only by the covariance structure (one
  observation per group — #697). Does not affect `ok`.
- `temporal_boundary` — `nothing` unless the fit has a `temporal()` term; then
  `(at_boundary, findings)`, drmTMB's `temporal_boundary` rules: residual σ̂ below
  1e-3 · sd(y), OU decay × longest series span below 1e-4 or × shortest gap above 30,
  AR1 |φ̂| above 0.999. drmTMB reports such a fit as `convergence_status() == "boundary"`
  (so its `is_converged` is false since FX3b). Does not affect `ok`.
- `skew_normal_nu_wald` — `nothing` unless the fit is [`SkewNormal`](@ref); then
  `(near_zero, min_abs_nu, cut)`. `near_zero` is `true` when any fitted slant
  |ν̂| < 0.5 (`cut`): the information matrix is singular at `ν = 0`, so the Wald SE
  for `nu` is unreliable there (#1060), and a warning is emitted. The cutoff is an
  uncalibrated heuristic of this package; no Wald test is run. Does not affect `ok`.
- `fixed_effect_rank_deficiency` — `nothing` unless a fixed-effect design (`mu` or
  `sigma`) had an aliased column (`sigma ~ 1 + x + z` with `z = 2x + 1`, a duplicated
  column); then `(deficient = true, blocks)`, one `(param, rank, ncol, aliased)` per
  rank-deficient block, naming the aliased coefficients (#1110; drmTMB's
  `fixed_effect_rank_deficiency`). Such a fit is not converged (`is_converged` is
  `false`), the aliased coordinates' SEs are `NaN`, and `ok` is `false`. Detected on the
  Gaussian location-scale route.
- `ok` — `true` when the optimiser converged at a non-degenerate optimum (the
  test `is_converged` made before FX3b), the gradient is small, the covariance is
  PD, and no fixed-effect design is rank-deficient. It does not read
  `convergence_status`, which is reported in its own field: a `:boundary` fit, or
  one the twin rules call `:degenerate` for an inflated standard error, can still
  be `ok`.
  On a phylogenetic-penalty fit the gradient criterion is **dropped** because
  its stored objective is the unpenalized likelihood. Fixed-prior fits retain
  the penalized objective and its gradient is checked. The criterion is also dropped
  whenever `max_abs_grad` is `NaN`, so read `grad_source` before reading `ok`:
  `ok = true` alongside `:none` or `:unavailable` means *converged and PD only*,
  with stationarity untested.

A non-`ok` result is informative, not an error: a model sitting on a variance
boundary (Watanabe-singular) can be the data's MLE, with valid Wald SEs on the
remaining directions — see [`confint`](@ref).
"""
function check_drm(fit::DrmFit; grad_tol::Real=1e-3)
    grad = _check_max_abs_grad(fit)
    mag = grad.magnitude
    V = fit.vcov
    # A diagnostic must REPORT trouble, not crash on it. Several routes return a
    # deliberately PARTIAL covariance — the sparse phylo fitter
    # (`_fit_structured_gaussian_sparse_lbfgs`) computes the β block and leaves the
    # variance-component block as NaN — and `isposdef`/`eigvals` throw outright on a
    # non-finite matrix. Running the documented health check on a perfectly good
    # phylo fit then raised `ArgumentError: matrix contains Infs or NaNs` instead of
    # returning a report, which is backwards. Report the incompleteness as a field.
    vcov_complete = all(isfinite, V)
    pd, mineig, cnd = if vcov_complete
        _pd = isposdef(Symmetric(V))
        ev = eigvals(Symmetric(V))
        _mineig = minimum(ev)
        _maxeig = maximum(ev)
        (_pd, _mineig, _mineig > 0 ? _maxeig / _mineig : Inf)
    else
        (false, NaN, Inf)
    end
    # Phylogenetic-penalty fits retain the data objective; fixed-prior fits retain
    # the penalized objective and can be checked for stationarity directly.
    penalized = fit.estim_method === :MAP
    # #1003: a failed diagnostic warns instead of reading as "not that kind of fit".
    vb = try _variance_boundary(fit) catch err; _boundary_diagnostic_failed("variance-boundary", err) end
    tb = try _temporal_boundary(fit) catch err; _boundary_diagnostic_failed("temporal-boundary", err) end
    status = convergence_status(fit)   # FX3b twin status; #949: not the raw flag
    snw = _skew_normal_nu_wald(fit)   # #1060: `nothing` unless SkewNormal
    frd = _fixed_effect_rank_deficiency(fit)   # #1110: `nothing` unless a design is aliased
    conv = status === :converged       # is_converged(fit)
    fixed_prior = _is_fixed_prior_fit(fit) && penalized
    ok = _converged_nondegenerate(fit) && ((penalized && !fixed_prior) || isnan(mag) || mag <= grad_tol) && pd && frd === nothing
    report = (
        convergence_status=status,
        converged=conv,
        max_abs_grad=mag,
        grad_source=grad.source,
        vcov_complete=vcov_complete,
        vcov_posdef=pd,
        min_eigval=mineig,
        cond=cnd,
        penalized_map=penalized,
        variance_boundary=vb,
        temporal_boundary=tb === nothing ? nothing : (at_boundary = !isempty(tb), findings = tb),
        skew_normal_nu_wald=snw,
        fixed_effect_rank_deficiency=frd,
        ok=ok,
    )
    @info "check_drm" convergence_status = report.convergence_status converged = report.converged max_abs_grad = report.max_abs_grad grad_source =
        report.grad_source vcov_complete = report.vcov_complete vcov_posdef = report.vcov_posdef min_eigval =
        report.min_eigval cond = report.cond penalized_map = report.penalized_map ok = report.ok
    # #1110: the NaN rows of an aliased design are explained by its own warning below.
    (vcov_complete || frd !== nothing) || @warn "check_drm: the stored covariance has non-finite entries, so " *
        "`vcov_posdef` / `min_eigval` / `cond` could not be computed and `ok` is false. This is " *
        "EXPECTED on routes that report a partial covariance — the sparse phylo fitter computes " *
        "the fixed-effect block only. Wald SEs on the finite directions are still usable; for the " *
        "variance components use `profile_ci = true` or `profile_result`."
    # A number that is not an exact gradient, and a NaN that is not `:none`, both
    # have to be SAID -- otherwise a route whose stationarity could not be tested
    # is indistinguishable from one that passed, which is the defect the
    # ForwardDiff guard above exists to remove.
    grad.source === :finite && @warn "check_drm: `max_abs_grad` is a CENTRAL FINITE DIFFERENCE of " *
        "the stored objective, not an exact gradient. This fit's objective is exact on `Float64` " *
        "but not dual-number safe (it factorises a sparse matrix through CHOLMOD, or keeps Float64 " *
        "work arrays), so ForwardDiff could not be run through it. Read the magnitude to about " *
        "1e-6 relative precision, not to machine precision; `grad_source` is `:finite`."
    grad.source === :unavailable && @warn "check_drm: the gradient at the estimate could not be " *
        "computed on this fit: neither the stored objective's derivative nor a central finite " *
        "difference of it produced a finite value, so `max_abs_grad` is NaN and `ok` was NOT " *
        "scored against the gradient criterion: `ok = true` here means converged and " *
        "positive-definite covariance ONLY, with stationarity untested. `grad_source` is " *
        "`:unavailable`, which is NOT `:none` (a fit that stores no objective at all)."
    vbmsg = vb === nothing ? nothing : _variance_boundary_message(vb)
    vbmsg === nothing || @warn "check_drm: " * vbmsg
    (tb === nothing || isempty(tb)) || @warn "check_drm: " * _temporal_boundary_message(tb)
    (snw === nothing || !snw.near_zero) || @warn "check_drm: " * _skew_normal_nu_wald_message(snw)
    frd === nothing || @warn "check_drm: " * _design_rank_message(frd.blocks)
    # drmTMB emits the equivalent advisory from `check_penalized_fit()`.
    fixed_prior && @warn "check_drm: $(_prior_kind_label(fit)) MAP fit — standard errors describe local penalized curvature; data log-likelihood and prior contribution are separate."
    penalized && !fixed_prior && @warn "check_drm: penalized (MAP) fit — standard errors come from the penalized " *
        "curvature and are credible-interval-shaped, not frequentist. `loglik` is the UNPENALIZED " *
        "data log-likelihood; the penalty is `fit.phylo_penalty`. `lrtest`/`anova` across " *
        "penalized fits are refused."
    return report
end
