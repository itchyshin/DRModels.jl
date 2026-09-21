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
stderror(fit::DrmFit) = _boundary_se.(diag(fit.vcov))

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
  upper endpoint searches are run in parallel instead. Canonical location–scale
  profiling remains serial in this release, including when `threads = true`.
- `parm = :resd` or `parm = [:mu, :resd]` — restrict intervals to one or more
  parameter blocks. This is especially useful for profiling random-effect SDs
  without also profiling fixed-effect coefficients.

Mirrors drmTMB's `confint(fit, method = "wald" | "profile")`.
"""
function confint(
    fit::DrmFit; level::Real=0.95, method::Symbol=:wald, threads::Bool=false, parm=nothing
)
    method === :wald && return _wald_ci(fit, level, parm)
    if method === :profile
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
upper endpoint chains remain serial.
"""
function profile_result(fit::DrmFit; level::Real=0.95, threads::Bool=false, parm=nothing)
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
    se = stderror(fit)
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
    se = stderror(fit)                                    # DrmFit (recov) order
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
        failed=0,
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
        # `_loconly_profile_endpoint_result` was not in scope for DRModels.jl#493 (the
        # reported degenerate fit routes through the generic path, not here) and
        # is not instrumented for the same bracket-collapse failure; default false
        # rather than claim a check that was not made.
        lower_endpoint_failed=false,
        upper_endpoint_failed=false,
        lower_nuisance_method=:specialized,
        upper_nuisance_method=:specialized,
        lower_nuisance_fallback=false,
        upper_nuisance_fallback=false,
        lower_nuisance_reason=:not_checked,
        upper_nuisance_reason=:not_checked,
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

    function heval(t)
        f, lσ_opt, slope = _loconly_profiled_resd_nll(obj, lσ_start, estimate + dir * t)
        lσ_start = lσ_opt
        evaluations += 1
        gradient_evaluations += 1
        return (f - target, dir * slope)
    end

    tlo = 0.0
    thi = 0.1
    hhi, _ = heval(thi)
    hprev = hhi
    nonmonotone = false
    iters = 0
    while isfinite(hhi) && hhi < 0 && iters < 60
        tlo = thi
        thi *= 1.6
        hhi, _ = heval(thi)
        isfinite(hhi) && hhi < hprev - 1e-12 && (nonmonotone = true)
        hprev = hhi
        iters += 1
    end
    if isfinite(hhi) && hhi < 0
        value = dir < 0 ? -Inf : Inf
        stats = (
            evaluations=evaluations,
            gradient_evaluations=gradient_evaluations,
            bracket_expansions=iters,
            root_iterations=0,
            unbounded=true,
            nonmonotone=nonmonotone,
        )
        return value, stats
    end
    nonmonotone && (tlo = 0.0)   # conservative bracket to the first crossing (see _profile_endpoint_result)

    t = (tlo + thi) / 2
    root_iterations = 0
    for _ in 1:80
        root_iterations += 1
        ht, hp = heval(t)
        abs(ht) < 1e-9 && break
        ht < 0 ? (tlo = t) : (thi = t)
        tn = (!nonmonotone && isfinite(hp) && hp > 0) ? t - ht / hp : (tlo + thi) / 2
        t = (tlo < tn < thi) ? tn : (tlo + thi) / 2
        thi - tlo < 1e-8 && break
    end
    value = estimate + dir * t
    stats = (
        evaluations=evaluations,
        gradient_evaluations=gradient_evaluations,
        bracket_expansions=iters,
        root_iterations=root_iterations,
        unbounded=false,
        nonmonotone=nonmonotone,
    )
    return value, stats
end

function _loconly_profiled_resd_nll(obj::LocOnlyObjective, lσ0::Real, lσ_phy::Real)
    x0 = [Float64(lσ0)]
    function fg!(F, G, x)
        val, grad, _, _ = _loconly_profile_fg(obj.prob, x[1], lσ_phy)
        G !== nothing && (G[1] = grad[1])
        return F === nothing ? nothing : val
    end
    res = try
        od = Optim.NLSolversBase.only_fg!(fg!)
        Optim.optimize(
            od,
            x0,
            Optim.LBFGS(; linesearch=Optim.LineSearches.BackTracking(; order=3)),
            Optim.Options(; iterations=80, g_tol=1e-8, x_abstol=1e-10),
        )
    catch
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
    return val, lσ, grad[2]
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
function _profile_reference_difference(value::Real, reference::Real)
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
function _profile_endpoint(nll, nllgrad, θ̂, k, nllhat, half, s, dir, u0, autodiff)
    value, _ = _profile_endpoint_result(
        nll, nllgrad, θ̂, k, nllhat, half, s, dir, u0, autodiff
    )
    return value
end

function _profile_endpoint_result(nll, nllgrad, θ̂, k, nllhat, half, s, dir, u0, autodiff)
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
    # Bracket: expand until h > 0. h(t) is assumed increasing (LR profile), but an
    # inexactly-solved inner nuisance optimisation can make it slightly non-monotone
    # (dip below `target` then rise). We track this and reset `tlo = 0` so
    # bisection works over the full observed bracket. This guarded local search
    # does not establish that the returned crossing is globally first.
    tlo = 0.0
    thi = s
    evalhi = heval(thi)
    evalhi.ok || return failed_arm(evalhi.nuisance)
    hhi = evalhi.h
    hprev = hhi
    nonmonotone = false
    iters = 0
    while hhi < 0 && iters < 40
        tlo = thi
        thi *= 1.6
        evalhi = heval(thi)
        evalhi.ok || return failed_arm(evalhi.nuisance, 0, iters, nonmonotone)
        hhi = evalhi.h
        hhi < hprev - 1e-12 && (nonmonotone = true)
        hprev = hhi
        iters += 1
    end
    if hhi < 0
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

function _profile_row_result(
    job, nll, nllgrad, θ̂, nllhat, half, se, autodiff; endpoint_threads::Bool=false
)
    k = job.k
    est = θ̂[k]
    s = (isfinite(se[k]) && se[k] > 0) ? se[k] : max(abs(est), 1.0)
    u0 = θ̂[[i for i in 1:length(θ̂) if i != k]]
    if endpoint_threads && Threads.nthreads() > 1
        left = Threads.@spawn _profile_endpoint_result(
            nll, nllgrad, θ̂, k, nllhat, half, s, -1, u0, autodiff
        )
        right = Threads.@spawn _profile_endpoint_result(
            nll, nllgrad, θ̂, k, nllhat, half, s, +1, u0, autodiff
        )
        lo, lstats = fetch(left)
        hi, rstats = fetch(right)
    else
        lo, lstats = _profile_endpoint_result(
            nll, nllgrad, θ̂, k, nllhat, half, s, -1, u0, autodiff
        )
        hi, rstats = _profile_endpoint_result(
            nll, nllgrad, θ̂, k, nllhat, half, s, +1, u0, autodiff
        )
    end
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

"""
    bootstrap_ci(formula, family; data, B = 300, level = 0.95, rng = default_rng(), threads = false, K =, A =, tree =, coords =, algorithm = :auto, g_tol = 1e-8)
    bootstrap_ci(fit; data, B = 300, level = 0.95, rng = default_rng(), threads = false, K =, A =, tree =, coords =, algorithm = :auto, g_tol = 1e-8)

Parametric bootstrap confidence intervals: fit the model, then `simulate` `B`
replicate responses, refit each, and take percentile intervals per coefficient.
Univariate-response models (fixed / random-effect / meta / structured). Same row
shape as [`confint`](@ref). Set `threads = true` to refit bootstrap replicates
in parallel. Pass through the structured-provider keywords (`K` / `A` / `tree` /
`coords`)
and, for Gaussian fits, the solver controls (`algorithm` / `g_tol`) exactly as
to [`drm`](@ref). Use `bootstrap_result` when you need attempted/used/failed
counts and per-replicate failure messages. If you already have
`fit = drm(...)`, pass the fit directly to avoid refitting the base model before
the bootstrap replicates.
"""
function bootstrap_ci(
    formula::DrmFormula,
    family::Gaussian;
    data,
    B::Int=300,
    level::Real=0.95,
    rng=default_rng(),
    K=nothing,
    A=nothing,
    tree=nothing,
    coords=nothing,
    threads::Bool=false,
    failures::Symbol=:error,
    check_converged::Bool=false,
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
    B::Int=300,
    level::Real=0.95,
    rng=default_rng(),
    K=nothing,
    A=nothing,
    tree=nothing,
    coords=nothing,
    threads::Bool=false,
    failures::Symbol=:error,
    check_converged::Bool=false,
)
    rows = bootstrap_summary(
        formula, family; data, B, level, rng, K, A, tree, coords, threads, failures,
        check_converged
    )
    return _bootstrap_ci_rows(rows)
end

function bootstrap_ci(
    fit::DrmFit;
    data,
    B::Int=300,
    level::Real=0.95,
    rng=default_rng(),
    K=nothing,
    A=nothing,
    tree=nothing,
    coords=nothing,
    threads::Bool=false,
    failures::Symbol=:error,
    check_converged::Bool=false,
)
    rows = bootstrap_summary(
        fit; data, B, level, rng, K, A, tree, coords, threads, failures, check_converged
    )
    return _bootstrap_ci_rows(rows)
end

function bootstrap_ci(
    fit::DrmFit{<:Gaussian};
    data,
    B::Int=300,
    level::Real=0.95,
    rng=default_rng(),
    K=nothing,
    A=nothing,
    tree=nothing,
    coords=nothing,
    threads::Bool=false,
    failures::Symbol=:error,
    check_converged::Bool=false,
    algorithm::Symbol=:auto,
    g_tol::Real=1e-8,
)
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
    return _bootstrap_ci_rows(rows)
end

"""
    bootstrap_summary(formula, family; data, B = 300, level = 0.95, rng = default_rng(), threads = false, K =, A =, tree =, coords =, algorithm = :auto, g_tol = 1e-8)
    bootstrap_summary(fit; data, B = 300, level = 0.95, rng = default_rng(), threads = false, K =, A =, tree =, coords =, algorithm = :auto, g_tol = 1e-8)

Parametric bootstrap coefficient summaries in one pass: point estimate,
bootstrap standard error, and percentile confidence interval. This is the
bootstrap analogue of using `stderror(fit)` plus `confint(fit)`, avoiding a
second bootstrap run when both SEs and intervals are needed. Row fields are
`(param, coef, estimate, std_error, lower, upper)`. By default, any failed
replicate errors after all failures are recorded. Set `failures = :skip` to
compute summaries from successful replicates; call `bootstrap_result` to
inspect the skipped failures.
"""
function bootstrap_summary(
    formula::DrmFormula,
    family::Gaussian;
    data,
    B::Int=300,
    level::Real=0.95,
    rng=default_rng(),
    K=nothing,
    A=nothing,
    tree=nothing,
    coords=nothing,
    threads::Bool=false,
    failures::Symbol=:error,
    check_converged::Bool=false,
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
    B::Int=300,
    level::Real=0.95,
    rng=default_rng(),
    K=nothing,
    A=nothing,
    tree=nothing,
    coords=nothing,
    threads::Bool=false,
    failures::Symbol=:error,
    check_converged::Bool=false,
)
    result = bootstrap_result(
        formula, family; data, B, level, rng, K, A, tree, coords, threads, failures,
        check_converged
    )
    return result.summary
end

function bootstrap_summary(
    fit::DrmFit;
    data,
    B::Int=300,
    level::Real=0.95,
    rng=default_rng(),
    K=nothing,
    A=nothing,
    tree=nothing,
    coords=nothing,
    threads::Bool=false,
    failures::Symbol=:error,
    check_converged::Bool=false,
)
    result = bootstrap_result(
        fit; data, B, level, rng, K, A, tree, coords, threads, failures, check_converged
    )
    return result.summary
end

function bootstrap_summary(
    fit::DrmFit{<:Gaussian};
    data,
    B::Int=300,
    level::Real=0.95,
    rng=default_rng(),
    K=nothing,
    A=nothing,
    tree=nothing,
    coords=nothing,
    threads::Bool=false,
    failures::Symbol=:error,
    check_converged::Bool=false,
    algorithm::Symbol=:auto,
    g_tol::Real=1e-8,
)
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
    bootstrap_result(formula, family; data, B = 300, level = 0.95, rng = default_rng(), threads = false, failures = :error, check_converged = false, K =, A =, tree =, coords =, algorithm = :auto, g_tol = 1e-8)
    bootstrap_result(fit; data, B = 300, level = 0.95, rng = default_rng(), threads = false, failures = :error, check_converged = false, K =, A =, tree =, coords =, algorithm = :auto, g_tol = 1e-8)

Auditable parametric bootstrap. Returns a `NamedTuple` with:

- `summary` — the same rows returned by `bootstrap_summary`;
- `failures` — rows `(replicate, seed, message)` for failed refits;
- `attempted`, `used`, `failed` — replicate counts;
- `seeds` — the per-replicate seeds used for reproducibility;
- `threaded` — whether threaded refits were actually used.
- `worker_threads`, `julia_threads`, `blas_threads`, `blas_oversubscribed`,
  `elapsed` — CPU context and wall-clock time for the simulated-refit phase.

`failures = :error` (default) records failures and then errors if any replicate
failed. `failures = :skip` computes summaries from successful replicates and
keeps the failure records in the return value. Set `check_converged = true` to
treat non-converged refits as failed replicates. Passing an existing `DrmFit`
reuses that point estimate as the bootstrap seed fit and starts directly with
the `B` simulated refits. Gaussian bootstrap refits pass `algorithm` and
`g_tol` through to `drm(...)`; this is useful for large structured models where
`:auto` selects a sparse route and the tolerance is part of the benchmarked
workflow.
"""
function bootstrap_result(
    formula::DrmFormula,
    family::Gaussian;
    data,
    B::Int=300,
    level::Real=0.95,
    rng=default_rng(),
    K=nothing,
    A=nothing,
    tree=nothing,
    coords=nothing,
    threads::Bool=false,
    failures::Symbol=:error,
    check_converged::Bool=false,
    algorithm::Symbol=:auto,
    g_tol::Real=1e-8,
)
    _check_bootstrap_failure_mode(failures)
    fit0 = drm(formula, family; data, K, A, tree, coords, algorithm, g_tol)
    refit = datab -> drm(formula, family; data=datab, K, A, tree, coords, algorithm, g_tol)
    simulate_fn = _marginal_simulator(fit0, data; K, A, tree, coords)
    return _bootstrap_result(
        fit0, formula, data, B, level, rng, threads, refit;
        failures, check_converged, simulate_fn
    )
end

function bootstrap_result(
    fit::DrmFit{<:Gaussian};
    data,
    B::Int=300,
    level::Real=0.95,
    rng=default_rng(),
    K=nothing,
    A=nothing,
    tree=nothing,
    coords=nothing,
    threads::Bool=false,
    failures::Symbol=:error,
    check_converged::Bool=false,
    algorithm::Symbol=:auto,
    g_tol::Real=1e-8,
)
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
    # retain their existing dispatch here; MAP needs its separate penalty contract.
    refit_options = if _is_gaussian_lss(fit)
        method = estimation_method(fit)
        method in (:ML, :REML) || throw(ArgumentError("LSS bootstrap supports ML/REML seed fits only"))
        (; method)
    else
        (;)
    end
    # `drm(::BivariateDrmFormula, ::Gaussian; ...)` declares no `algorithm`
    # keyword (src/gaussian_bivariate.jl), so forwarding it would throw a
    # `MethodError` on the first replicate. `refit_options` is empty on this
    # branch by construction — `_is_gaussian_lss` requires `fit.formula isa
    # DrmFormula` — so dropping it changes nothing here either.
    refit = if formula isa BivariateDrmFormula
        datab -> drm(formula, fit.family; data=datab, K, A, tree, coords, g_tol)
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
    B::Int=300,
    level::Real=0.95,
    rng=default_rng(),
    K=nothing,
    A=nothing,
    tree=nothing,
    coords=nothing,
    threads::Bool=false,
    failures::Symbol=:error,
    check_converged::Bool=false,
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
    K !== nothing && (extra[:K] = K)
    A !== nothing && (extra[:A] = A)
    coords !== nothing && (extra[:coords] = coords)
    fit0 = drm(formula, family; data, extra...)
    refit = datab -> drm(formula, family; data=datab, extra...)
    # #459/#479: redraw the random effects rather than conditioning on the
    # fitted BLUPs, so a variance-component bootstrap CI is not degenerate.
    simulate_fn = _marginal_simulator(fit0, data; K=K, A=A, tree=tree, coords=coords)
    return _bootstrap_result(
        fit0, formula, data, B, level, rng, threads, refit;
        failures, check_converged, simulate_fn
    )
end

function bootstrap_result(
    fit::DrmFit;
    data,
    B::Int=300,
    level::Real=0.95,
    rng=default_rng(),
    K=nothing,
    A=nothing,
    tree=nothing,
    coords=nothing,
    threads::Bool=false,
    failures::Symbol=:error,
    check_converged::Bool=false,
)
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
    K !== nothing && (extra[:K] = K)
    A !== nothing && (extra[:A] = A)
    coords !== nothing && (extra[:coords] = coords)
    refit = datab -> drm(formula, fit.family; data=datab, extra...)
    simulate_fn = _marginal_simulator(fit, data; K=K, A=A, tree=tree,
                                      coords=coords)   # #459 / #479
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
# `simulate(fit)` is a CONDITIONAL simulator: for a Gaussian fit it returns
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
_is_gaussian_lss(fit::DrmFit) = fit.family isa Gaussian &&
    fit.formula isa DrmFormula &&
    (!isempty(_sd_parts(fit.formula)) || !isempty(_sdphylo_parts(fit.formula)))

function _lss_marginal_simulator(fit::DrmFit, data; tree=nothing)
    f = fit.formula
    rhs = Dict(f.forms)
    fixed_mu, re, metav, structured = _split_ranef(rhs[:mu])
    fixed_sigma, sigma_re, _, structured_sigma = _split_ranef(rhs[:sigma])
    (metav === nothing && isempty(sigma_re) && structured_sigma === nothing) ||
        throw(ArgumentError("LSS bootstrap requires mean random intercepts and fixed residual-scale predictors"))
    all(r -> first(_re_kind(r[1])) === :intercept, re) ||
        throw(ArgumentError("LSS bootstrap supports random intercepts only"))
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
    for (_, grp) in re
        labels = _table_column(data, grp)
        any(ismissing, labels) && throw(ArgumentError("LSS bootstrap grouping `$grp` contains missing labels"))
        gidx, G = _group_index(labels)
        Zg, nm = haskey(sdmap, grp) ?
            _sd_group_design(f.response, sdmap[grp], data, gidx, G, grp) :
            (ones(G, 1), ["(Intercept)"])
        width = size(Zg, 2)
        offset + width <= length(aiid) ||
            throw(ArgumentError("LSS bootstrap iid SD coefficients do not match the group designs"))
        scale = exp.(Zg * aiid[offset+1:offset+width])
        push!(components, (; gidx, G, scale, factor=nothing))
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
        push!(components, (; gidx, G, scale, factor))
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
            out .+= u[c.gidx]
        end
        out .+= sigma .* randn(rng, length(out))
        all(isfinite, out) || throw(ArgumentError("LSS bootstrap produced a nonfinite response"))
        out[.!observed] .= NaN
        return out
    end
end

function _marginal_simulator(fit::DrmFit, data; K=nothing, A=nothing, tree=nothing,
                             coords=nothing)
    fit.nll isa LocScaleObjective &&
        return _ls_marginal_simulator(fit, data; K, A, tree, coords)
    _is_gaussian_lss(fit) && return _lss_marginal_simulator(fit, data; tree)
    fit.formula isa DrmFormula || return nothing
    # Gaussian needs a residual scale; other families carry their dispersion in the
    # family object or in `scales`, and some (Poisson) have none at all.
    (fit.family isa Gaussian && !haskey(fit.scales, :sigma)) && return nothing
    rhs = Dict(fit.formula.forms)
    haskey(rhs, :mu) || return nothing
    _, re, _, structured, structured_slope = _split_ranef(rhs[:mu]; allow_phylo_slope = true)
    # A Gaussian `phylo(1 + x | g)` fit (#620) carries TWO phylogenetic fields;
    # the simulator below draws a single structured intercept, so building it
    # would silently bootstrap the wrong (intercept-only) model. Refuse.
    structured_slope === nothing ||
        throw(ArgumentError("bootstrap: the marginal simulator for the Gaussian " *
            "`phylo(1 + $(structured_slope) | $(structured[2]))` two-SD random-slope fit is " *
            "not implemented (#620); use `profile` intervals or the Wald `vcov`"))
    # No random effect on the mean: conditional and marginal coincide, and the
    # plain `simulate` is already correct. Nothing to build.
    (isempty(re) && structured === nothing) && return nothing

    # Which grouping factor, and what covariance does its random effect have?
    grp, Kg = if structured !== nothing
        g = structured[2]
        hasproperty(data, g) || return nothing
        _, G0 = _group_index(getproperty(data, g))
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
            (g, sigma_phy_dense(phy; σ²_phy = 1.0))
        elseif structured[1] === :spatial && K === nothing && coords !== nothing
            cmat = Matrix{Float64}(coords)
            size(cmat, 1) == G0 ||
                throw(ArgumentError("spatial bootstrap coords must have one row per `$g` level (G = $G0)"))
            size(cmat, 2) >= 1 ||
                throw(ArgumentError("spatial bootstrap coords need at least one coordinate column"))
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
            (g, _resolve_structured_matrix(structured[1], g, G0;
                                           K=K, A=A, tree=tree, coords=coords))
        end
    else
        # Ordinary `(1 | g)`: independent random intercepts, so the covariance is I.
        length(re) == 1 || return nothing
        g = re[1][2]
        hasproperty(data, g) || return nothing
        _, G0 = _group_index(getproperty(data, g))
        (g, Matrix{Float64}(LinearAlgebra.I, G0, G0))
    end

    gidx, G = _group_index(getproperty(data, grp))
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
        return function (rng)
            u = sd_g .* (L * randn(rng, G))
            return mu_fixed .+ u[gidx] .+ sigma .* randn(rng, n)
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
        mu_star = _mean_response(fit.family, eta_fixed .+ u[gidx])
        return _simulate_once(fit, rng; mu = mu_star)
    end
end

function _bootstrap_result(
    fit0,
    formula::Union{DrmFormula,BivariateDrmFormula},
    data,
    B::Int,
    level::Real,
    rng,
    threads::Bool,
    refit;
    failures::Symbol=:error,
    check_converged::Bool=false,
    simulate_fn=nothing,
)
    _check_bootstrap_failure_mode(failures)
    B >= 1 || throw(ArgumentError("bootstrap requires B >= 1"))
    est = coef(fit0)
    p = length(est)
    draws = Matrix{Float64}(undef, B, p)
    # Distinct BitVector indices share a machine word: parallel writes can
    # silently lose a successful replicate. Byte-addressable flags are independent.
    ok = fill(false, B)
    messages = Vector{Union{Nothing,String}}(nothing, B)
    seeds = rand(rng, UInt, B)

    function run_one!(b)
        rr = Random.MersenneTwister(seeds[b])
        try
            # Marginal draw when the fit has random effects (#459); the plain
            # `simulate` is conditional and collapses a variance-component CI.
            ysim = simulate_fn === nothing ? simulate(fit0; rng=rr) : simulate_fn(rr)
            datab = _bootstrap_data(formula, data, ysim)
            fitb = refit(datab)
            # `is_converged`, not the raw `.converged` field: the accessor also
            # rejects a degenerate optimum (sigma collapsed, likelihood runaway),
            # which the optimiser's own flag happily calls converged (#461).
            if check_converged && !is_converged(fitb)
                error("refit did not converge or landed on a degenerate optimum")
            end
            draws[b, :] = coef(fitb)
            ok[b] = true
        catch err
            messages[b] = sprint(showerror, err)
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
    if !isempty(failure_rows) && failures === :error
        first_failure = first(failure_rows)
        throw(
            ErrorException(
                "bootstrap failed in $(length(failure_rows)) of $B replicates; first failure replicate $(first_failure.replicate), seed $(first_failure.seed): $(first_failure.message)",
            ),
        )
    end
    used = count(ok)
    used > 0 || throw(ErrorException("all $B bootstrap replicates failed"))
    summary = _bootstrap_summary_rows(fit0, draws[ok, :], est, level)
    return (
        summary=summary,
        failures=failure_rows,
        attempted=B,
        used=used,
        failed=length(failure_rows),
        seeds=seeds,
        threaded=threaded,
        worker_threads=_worker_threads(threaded, B),
        julia_threads=Threads.nthreads(),
        blas_threads=BLAS.get_num_threads(),
        blas_oversubscribed=_blas_oversubscribed(threaded),
        elapsed=elapsed,
        check_converged=check_converged,
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
function _restore_response_mask!(ysim::AbstractVector{Float64}, raw)
    length(raw) == length(ysim) || return ysim
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

- `converged` — the optimiser's convergence flag.
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
- `penalized_map` — whether this is a penalized (MAP) fit
  (`penalty = drm_phylo_penalty(...)`). Such a fit reports standard errors from
  the *penalized* curvature, which are credible-interval-shaped rather than
  frequentist, and `loglik` is the *unpenalized* data log-likelihood.
- `ok` — `true` when converged, the gradient is small, and the covariance is PD.
  On a penalized fit the gradient criterion is **dropped**: the stored objective
  is unpenalized, so its gradient is non-zero at the MAP optimum by construction
  and scoring it would report a correct fit as broken. It is also dropped
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
    # A4c: on a penalized (MAP) fit the stored objective is the UNPENALIZED
    # likelihood, whose gradient is deliberately NON-ZERO at the MAP optimum — the
    # penalty's gradient is what cancels it. Scoring that as non-convergence would
    # report a correct fit as broken, so the gradient criterion is dropped for MAP
    # fits and `max_abs_grad` is reported for information only.
    penalized = fit.estim_method === :MAP
    ok = fit.converged && (penalized || isnan(mag) || mag <= grad_tol) && pd
    report = (
        converged=fit.converged,
        max_abs_grad=mag,
        grad_source=grad.source,
        vcov_complete=vcov_complete,
        vcov_posdef=pd,
        min_eigval=mineig,
        cond=cnd,
        penalized_map=penalized,
        ok=ok,
    )
    @info "check_drm" converged = report.converged max_abs_grad = report.max_abs_grad grad_source =
        report.grad_source vcov_complete = report.vcov_complete vcov_posdef = report.vcov_posdef min_eigval =
        report.min_eigval cond = report.cond penalized_map = report.penalized_map ok = report.ok
    vcov_complete || @warn "check_drm: the stored covariance has non-finite entries, so " *
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
    # drmTMB emits the equivalent advisory from `check_penalized_fit()`.
    penalized && @warn "check_drm: penalized (MAP) fit — standard errors come from the penalized " *
        "curvature and are credible-interval-shaped, not frequentist. `loglik` is the UNPENALIZED " *
        "data log-likelihood; the penalty is `fit.phylo_penalty`. `lrtest`/`anova` across " *
        "penalized fits are refused."
    return report
end
