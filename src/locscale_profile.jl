# locscale_profile.jl — profile-likelihood confidence intervals for the q=2
# location–scale fit (#202). The right uncertainty tool where Wald is poor — the
# group-covariance (variance/correlation) parameters near weak identification.
#
# For a packed parameter `idx`, the profile NLL fixes θ[idx] and re-optimises the
# rest. Three design choices make this robust AND fast:
#  * the constrained solve uses L-BFGS with backtracking and the exact reduced
#    gradient, warm-started from the preceding profile point;
#  * the χ²₁ crossing is searched with a VENZON–MOOLGAVKAR-style guarded-Newton
#    method using the profile deviance AND its envelope-theorem slope
#    (∂nll/∂θ[idx] at the constrained optimum, from the exact gradient). The
#    maintained bracket guards local refinement; each arm records an explicit
#    accepted, no-crossing, or failed terminal status;
#  * everything is WARM-STARTED across profile points (free-parameter init + inner
#    mode), and the brackets are SEEDED from the Wald half-width.
#
# Variance/correlation robustness: profiling such a parameter toward its boundary
# can drive Λ near-singular, so the constrained solve returns the documented
# infeasible sentinel. Such a failed trial is reported as a failed signed-infinity
# endpoint; only a finite, resolved no-crossing search is labelled unbounded.

import Distributions

# Marginal-NLL sentinel: `_ls_fit_nll` / the profile inner objective return this
# when the inner mode (or the prior factorisation) is infeasible — e.g. a profiled
# variance/correlation parameter driven toward its boundary makes Λ near-singular.
# A constrained optimum whose value is at/above this is treated as INFEASIBLE.
const _LS_PROFILE_INFEASIBLE = 1e18

# Boundary-identified log-Cholesky diagonals (owner decision 15 (a), 2026-09-30).
# The packed vector ends with `(log L11, L21, log L22)`. When a log-diagonal has
# slid below this value the marginal NLL is flat along it (the variance is
# numerically zero; gradient decays only like exp(2 * coordinate)), so the strict
# 1e-7 stationarity test cannot be met by L-BFGS there. That coordinate alone is
# exempted from the free-gradient check; every other coordinate keeps the strict
# gate, and the exemption is reported as reason `:accepted_boundary_exempt`.
# The off-diagonal L21 of the SAME row is exempted too when log L22 is below the
# cutoff: with L22 ~ 0 the second variance is L21^2 + L22^2 ~ L21^2, so L21 is the
# same flat boundary direction (measured on the location-scale test fixture: the
# only coordinate above 1e-7 at a trial point was L21, 1.9e-7, all others < 5e-9).
const _LS_PROFILE_BOUNDARY_LOGCHOL = -8.0

const _LSProfileNuisanceResult = NamedTuple{
    (:value, :minimizer, :accepted, :method, :fallback, :reason, :converged, :gradient_maxabs),
    Tuple{Float64,Vector{Float64},Bool,Symbol,Bool,Symbol,Bool,Float64},
}

_ls_profile_nuisance_result(value, minimizer, accepted, reason;
                            converged=false, gradient_maxabs=NaN, fallback=false) =
    (value=Float64(value), minimizer=collect(float.(minimizer)), accepted=accepted,
     method=:lbfgs, fallback=fallback, reason=reason, converged=converged,
     gradient_maxabs=Float64(gradient_maxabs))::_LSProfileNuisanceResult

function _ls_profile_candidate_status(f, g!, xmin, converged::Bool; fallback=false,
                                      logchol_diag::AbstractVector{<:Integer} = Int[],
                                      l21_pos::Int = 0, l22_pos::Int = 0,
                                      l22_fixed::Float64 = NaN)
    all(isfinite, xmin) ||
        return _ls_profile_nuisance_result(Inf, xmin, false, :nonfinite_minimizer;
                                           converged=converged, fallback=fallback)
    converged || return _ls_profile_nuisance_result(Inf, xmin, false, :not_converged;
                                                    fallback=fallback)
    value = try
        f(xmin)
    catch err
        err isa InterruptException && rethrow()
        return _ls_profile_nuisance_result(Inf, xmin, false, :exception;
                                           converged=true, fallback=fallback)
    end
    (isfinite(value) && value < _LS_PROFILE_INFEASIBLE / 2) ||
        return _ls_profile_nuisance_result(value, xmin, false, :nonfinite_objective;
                                           converged=true, fallback=fallback)
    gradient = similar(xmin)
    try
        g!(gradient, xmin)
    catch err
        err isa InterruptException && rethrow()
        return _ls_profile_nuisance_result(value, xmin, false, :exception;
                                           converged=true, fallback=fallback)
    end
    all(isfinite, gradient) ||
        return _ls_profile_nuisance_result(value, xmin, false, :nonfinite_gradient;
                                           converged=true, fallback=fallback)
    gradient_maxabs = isempty(gradient) ? 0.0 : maximum(abs, gradient)
    exempt = [k for k in logchol_diag if xmin[k] < _LS_PROFILE_BOUNDARY_LOGCHOL]
    l22 = l22_pos > 0 ? xmin[l22_pos] : l22_fixed
    l21_pos > 0 && l22 < _LS_PROFILE_BOUNDARY_LOGCHOL && push!(exempt, l21_pos)
    if !isempty(exempt) && gradient_maxabs > 1e-7
        kept = [abs(gradient[k]) for k in eachindex(gradient) if !(k in exempt)]
        kept_maxabs = isempty(kept) ? 0.0 : maximum(kept)
        kept_maxabs <= 1e-7 &&
            return _ls_profile_nuisance_result(value, xmin, true, :accepted_boundary_exempt;
                                               converged=true, gradient_maxabs=kept_maxabs,
                                               fallback=fallback)
    end
    gradient_maxabs <= 1e-7 ||
        return _ls_profile_nuisance_result(value, xmin, false, :not_stationary;
                                           converged=true, gradient_maxabs=gradient_maxabs,
                                           fallback=fallback)
    return _ls_profile_nuisance_result(value, xmin, true, :accepted;
                                       converged=true, gradient_maxabs=gradient_maxabs,
                                       fallback=fallback)
end

# Profile NLL at θ[idx] = val: minimise the marginal over the other packed params.
# `x0` seeds the free-parameter optimisation (warm-start across profile points);
# `mwarm` is the shared inner-mode warm-start Ref. A finite optimizer-reported
# minimum is not sufficient: acceptance requires successful termination, a
# fresh finite objective evaluation, and a fresh free-gradient evaluation below
# the existing L-BFGS `g_tol`.
function _ls_profile_nll_result(kind, y, Xμ, Xψ, gidx, G, Q, θ̂, idx::Int, val::Real;
                                Zη = _ls_canonical_Zeta(length(y)),
                                Zψ = _ls_canonical_Zpsi(length(y)),
                                x0 = nothing,
                                whitened::Bool = false,
                                mwarm = whitened ? Ref{Union{Nothing,_LSWhitenedSeed}}(nothing) :
                                                   Ref{Union{Nothing,Vector{Float64}}}(nothing),
                                iterations::Int = 200)
    p = length(θ̂); pμ = size(Xμ, 2); pψ = size(Xψ, 2)
    free = [k for k in 1:p if k != idx]
    # Free-vector positions of the log-Cholesky diagonals (log L11, log L22).
    logchol_diag = [k - (k > idx) for k in (p - 2, p) if k != idx]
    l21_pos = idx == p - 1 ? 0 : (p - 1) - (p - 1 > idx)
    l22_pos = idx == p ? 0 : p - (p > idx)
    l22_fixed = idx == p ? Float64(val) : NaN
    build(θf) = (θ = collect(float.(θ̂)); θ[free] .= θf; θ[idx] = val; θ)
    function f(θf)
        θ = build(θf)
        if whitened
            result = _ls_whitened_eval(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ;
                                      seed=mwarm[], gradient=false)
            result.status.ok && (mwarm[] = result.seed)
            return result.status.ok ? result.value : _LS_PROFILE_INFEASIBLE
        end
        βμ = @view θ[1:pμ]; βψ = @view θ[pμ+1:pμ+pψ]
        P = prior_precision(Q, _ls_lc_inv2x2(θ[pμ+pψ+1:pμ+pψ+3]))   # stable: never forms Λ
        v, a, ok = _ls_marginal_nll(kind, y, Xμ * βμ, Xψ * βψ, gidx, G, P, Zη, Zψ; a0 = mwarm[])
        ok && (mwarm[] = copy(a))
        return ok ? v : _LS_PROFILE_INFEASIBLE
    end
    function g!(gf, θf)
        gfull = whitened ?
            _ls_whitened_eval(kind, y, Xμ, Xψ, gidx, G, Q, build(θf), Zη, Zψ; seed=mwarm[]).gradient :
            _ls_marginal_grad(kind, y, Xμ, Xψ, gidx, G, Q, build(θf), Zη, Zψ; a0=mwarm[])
        gf .= gfull[free]
        return gf
    end
    init = x0 === nothing ? float.(θ̂[free]) : float.(x0)
    # Hessian-free inner solve (LBFGS + backtracking). Each L-BFGS iteration uses
    # one exact-gradient call (`g!`); an observed-information central-difference
    # Jacobian would require 2·p_free `_ls_marginal_grad` calls per Newton step.
    # The
    # `_LS_PROFILE_INFEASIBLE` sentinel still fences the line search off the
    # near-singular-Λ boundary, so boundary robustness is preserved.
    # On the paired route, floating-point equality of successive objective
    # values must not stop a solve before its required gradient tolerance.
    # NaN disables only the x/f stopping comparisons in Optim; the existing
    # gradient threshold, iteration budget and fresh postcheck are unchanged.
    opts = whitened ? Optim.Options(g_tol=1e-7, iterations=iterations,
                                   x_abstol=NaN, x_reltol=NaN,
                                   f_abstol=NaN, f_reltol=NaN) :
                      Optim.Options(g_tol=1e-7, iterations=iterations)
    function solve(start; fallback=false)
        res = try
            Optim.optimize(f, g!, start,
                           Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking()),
                           opts)
        catch err
            err isa InterruptException && rethrow()
            return _ls_profile_nuisance_result(Inf, start, false, :exception;
                                               fallback=fallback)
        end
        xmin = Optim.minimizer(res)
        # NOT Optim.minimum(res): after a failed line search it can hold a
        # rejected trial's value while xmin has already moved on (see
        # optim_minimum_guard.jl). Fall back to it only if a fresh `f(xmin)`
        # itself errors.
        reported = try
            f(xmin)
        catch err
            err isa InterruptException && rethrow()
            Optim.minimum(res)
        end
        # Re-evaluate the objective and gradient even after `Optim.converged`: its
        # status can be driven by a non-gradient stopping condition.
        candidate = _ls_profile_candidate_status(
            f, g!, xmin, Optim.converged(res); fallback=fallback,
            logchol_diag=logchol_diag, l21_pos=l21_pos, l22_pos=l22_pos,
            l22_fixed=l22_fixed,
        )
        if !candidate.accepted && candidate.reason === :not_converged
            return _ls_profile_nuisance_result(reported, xmin, false, :not_converged;
                                               fallback=fallback)
        end
        return candidate
    end
    candidate = solve(init)
    if whitened && !candidate.accepted
        # A platform-sensitive line search can exhaust its budget near a useful
        # endpoint. Continue once from that endpoint with a fresh inner seed;
        # the replacement still has to earn the unchanged convergence and exact
        # gradient checks. This is distinct from accepting the exhausted solve.
        mwarm[] = nothing
        candidate = solve(candidate.minimizer; fallback=true)
    end
    if whitened && !candidate.accepted && x0 !== nothing
        # A damaged cross-point warm start may also strand the continuation.
        # Its final fallback is the fitted nuisance coordinates, again with a
        # fresh inner seed and the identical strict postchecks.
        mwarm[] = nothing
        candidate = solve(float.(θ̂[free]); fallback=true)
    end
    return candidate
end

# Historical three-tuple helper retained for internal callers that destructure it.
function _ls_profile_nll(kind, y, Xμ, Xψ, gidx, G, Q, θ̂, idx::Int, val::Real; kwargs...)
    result = _ls_profile_nll_result(kind, y, Xμ, Xψ, gidx, G, Q, θ̂, idx, val; kwargs...)
    return result.value, result.minimizer, result.accepted
end

# Venzon–Moolgavkar-style endpoint search. Find t ≥ 0 where the profile-deviance
# gap h(t) = profile_nll(x0 + dir·t) − thr crosses zero (h(0) < 0; h increasing in
# t). The work splits into a SHORT bracket expansion to straddle the crossing,
# then a GUARDED NEWTON refinement using the analytic envelope-theorem slope
# h'(t) = dir·∂nll/∂θ[idx], falling back to bisection whenever a Newton step would
# leave the maintained bracket or the slope is unusable. The search reports an
# explicit status instead of treating an unresolved candidate as an endpoint.
#
# `evalh(val)` returns the profile gap, its θ[idx]-slope (NOT yet multiplied by
# `dir`), and a feasibility flag. A failed trial is a numerical endpoint failure,
# never evidence of an unbounded interval. A finite negative gap after the bounded
# expansion budget is instead reported as `:no_crossing` in the searched range.
#
# A single unevaluable TRIAL, however, is not an unresolvable ENDPOINT (#651). The
# nuisance solve is a numerical procedure, so an individual profile point can fail
# to converge while its neighbours solve cleanly; on the canonical (whitened)
# location-scale route the accepted arms sit close enough to the stationarity
# threshold that a last-bit platform difference decides it. Both phases therefore
# CONTRACT toward a point already known to be feasible, on the bounded budget
# `maxcontract`, instead of abandoning the arm:
#   * during expansion, `thi` is halved toward the feasible `tlo`;
#   * during refinement, the TRIAL is halved toward `tlo`, leaving the maintained
#     bracket -- and its proof that a crossing exists -- untouched.
# Every loop stays bounded, so the search still terminates in at most
# `(maxexpand + maxnewton + 1) * (1 + maxcontract)` evaluations. Contraction only
# postpones failure: an arm that cannot be evaluated anywhere down to `xtol` is
# still reported as a failed, signed-infinity endpoint.
function _ls_profile_root_result(evalh, x0; dir::Float64, init::Float64,
                                 cancellation::Float64 = 0.0,
                                 maxexpand::Int = 40, maxnewton::Int = 30,
                                 maxcontract::Int = 8,
                                 ftol::Float64 = 1e-7, xtol::Float64 = 1e-8)
    evaluations = Ref(0)
    gradient_evaluations = Ref(0)
    contractions = Ref(0)
    terminal_nuisance = Ref{Any}(nothing)
    function evaluate(t)
        evaluations[] += 1
        candidate = x0 + dir * t
        isfinite(candidate) ||
            return (gap=NaN, slope=NaN, ok=false, reason=:nonfinite_candidate,
                    cancellation=cancellation)
        raw = try
            evalh(candidate)
        catch err
            err isa InterruptException && rethrow()
            return (gap=NaN, slope=NaN, ok=false, reason=:exception, cancellation=NaN)
        end
        gap, slope, ok = try
            raw
        catch err
            err isa InterruptException && rethrow()
            return (gap=NaN, slope=NaN, ok=false, reason=:invalid_evaluation,
                    cancellation=NaN)
        end
        raw isa NamedTuple && haskey(raw, :nuisance) &&
            (terminal_nuisance[] = raw.nuisance)
        reason = raw isa NamedTuple && haskey(raw, :reason) ? raw.reason : :evaluation_failed
        allowance = raw isa NamedTuple && haskey(raw, :cancellation) ?
                    raw.cancellation : cancellation
        (isfinite(allowance) && allowance >= 0) ||
            return (gap=NaN, slope=NaN, ok=false, reason=:nonfinite_cancellation,
                    cancellation=NaN)
        !ok && return (gap=NaN, slope=NaN, ok=false, reason=reason,
                       cancellation=Float64(allowance))
        !isfinite(gap) && return (gap=gap, slope=slope, ok=false,
                                  reason=:nonfinite_evaluation,
                                  cancellation=Float64(allowance))
        isfinite(slope) && (gradient_evaluations[] += 1)
        return (gap=Float64(gap), slope=Float64(slope), ok=true, reason=:accepted,
                cancellation=Float64(allowance))
    end
    make_result(value, accepted, unbounded, endpoint_failed, reason,
                bracket_expansions, root_iterations, candidate, residual, allowance=NaN) = (
        value=Float64(value), accepted=accepted, unbounded=unbounded,
        endpoint_failed=endpoint_failed, reason=reason,
        bracket_expansions=bracket_expansions, root_iterations=root_iterations,
        evaluations=evaluations[], gradient_evaluations=gradient_evaluations[],
        contractions=contractions[],
        candidate=Float64(candidate), residual=Float64(residual),
        cancellation=Float64(allowance),
        nuisance=terminal_nuisance[],
    )
    (isfinite(x0) && isfinite(dir) && isfinite(init)) ||
        return make_result(dir > 0 ? Inf : -Inf, false, false, true,
                           :nonfinite_initialization, 0, 0, NaN, NaN, cancellation)
    init > 0 ||
        return make_result(dir > 0 ? Inf : -Inf, false, false, true,
                           :invalid_initialization, 0, 0, NaN, NaN, cancellation)
    dir != 0 ||
        return make_result(Inf, false, false, true, :invalid_direction, 0, 0, NaN, NaN,
                           cancellation)
    (isfinite(cancellation) && cancellation >= 0) ||
        return make_result(dir > 0 ? Inf : -Inf, false, false, true,
                           :nonfinite_cancellation, 0, 0, NaN, NaN, cancellation)
    tlo = 0.0                                   # h(tlo) < 0 (feasible by construction)
    thi = init
    maxexpand > 0 ||
        return make_result(dir > 0 ? Inf : -Inf, false, false, true, :invalid_search_budget,
                           0, 0, NaN, NaN, cancellation)
    current = (gap=NaN, slope=NaN, ok=false, reason=:not_evaluated, cancellation=NaN)
    expansions = 0
    for _ in 1:maxexpand
        current = evaluate(thi)
        # #651: an unevaluable trial does not prove the endpoint is unreachable.
        # `tlo` is feasible -- h(0) < 0 holds by construction at the fitted
        # optimum, and every later `tlo` was itself evaluated -- so halve toward
        # it on a bounded budget rather than abandoning the arm.
        contracted = 0
        while !current.ok && contracted < maxcontract && thi - tlo > xtol
            thi = tlo + (thi - tlo) / 2
            contracted += 1
            contractions[] += 1
            current = evaluate(thi)
        end
        current.ok || return make_result(dir > 0 ? Inf : -Inf, false, false, true,
                                         current.reason, expansions, 0, x0 + dir * thi, current.gap,
                                         current.cancellation)
        abs(current.gap) < ftol && abs(current.gap) + current.cancellation <= ftol &&
            return make_result(x0 + dir * thi, true, false, false, :accepted,
                               expansions, 0, x0 + dir * thi, current.gap,
                               current.cancellation)
        current.gap > current.cancellation && break
        current.gap >= -current.cancellation &&
            return make_result(dir > 0 ? Inf : -Inf, false, false, true,
                               :insufficient_precision, expansions, 0,
                               x0 + dir * thi, current.gap, current.cancellation)
        tlo = thi
        thi *= 1.6
        expansions += 1
    end
    if !(current.gap > current.cancellation)
        contractions[] == 0 &&
            return make_result(dir > 0 ? Inf : -Inf, false, true, false, :no_crossing,
                               expansions, 0, x0 + dir * (thi / 1.6), current.gap,
                               current.cancellation)
        # The expansion budget ran out inside a region whose trials could not be
        # evaluated. Absence of a crossing was never established there, so this
        # is a failed endpoint, NOT an unbounded interval (#651, fail closed).
        return make_result(dir > 0 ? Inf : -Inf, false, false, true, :infeasible_region,
                           expansions, 0, x0 + dir * thi, current.gap,
                           current.cancellation)
    end

    # Guarded Newton on [tlo, thi]; h(tlo) < 0 ≤ h(thi). Every accepted
    # endpoint is an actually evaluated candidate with a small residual.
    t = thi
    for iteration in 0:maxnewton
        abs(current.gap) < ftol && abs(current.gap) + current.cancellation <= ftol &&
            return make_result(x0 + dir * t, true, false, false, :accepted,
                               expansions, iteration, x0 + dir * t, current.gap,
                               current.cancellation)
        abs(current.gap) <= current.cancellation &&
            return make_result(dir > 0 ? Inf : -Inf, false, false, true,
                               :insufficient_precision, expansions, iteration,
                               x0 + dir * t, current.gap, current.cancellation)
        iteration == maxnewton &&
            return make_result(dir > 0 ? Inf : -Inf, false, false, true, :max_iterations,
                               expansions, iteration, x0 + dir * t, current.gap,
                               current.cancellation)
        thi - tlo < xtol &&
            return make_result(dir > 0 ? Inf : -Inf, false, false, true, :bracket_collapse,
                               expansions, iteration, x0 + dir * t, current.gap,
                               current.cancellation)
        current.gap > current.cancellation ? (thi = t) : (tlo = t)
        sd = dir * current.slope                # h'(t) = dir · ∂nll/∂θ[idx]
        tn = (isfinite(sd) && sd > 0) ? t - current.gap / sd : (tlo + thi) / 2
        t = (tlo < tn < thi) ? tn : (tlo + thi) / 2
        current = evaluate(t)
        # #651: the maintained bracket still proves a crossing (h(tlo) < 0 <=
        # h(thi)); an unevaluable interior trial only means THIS point could not
        # be solved. Halve the trial toward the feasible `tlo` on a bounded
        # budget, leaving the bracket and its proof untouched.
        contracted = 0
        while !current.ok && contracted < maxcontract && t - tlo > xtol
            t = tlo + (t - tlo) / 2
            contracted += 1
            contractions[] += 1
            current = evaluate(t)
        end
        current.ok || return make_result(dir > 0 ? Inf : -Inf, false, false, true,
                                         current.reason, expansions, iteration + 1,
                                         x0 + dir * t, current.gap, current.cancellation)
    end
    error("unreachable location-scale profile root state")
end

# Historical scalar endpoint helper retained for direct internal callers.
function _ls_profile_root(evalh, x0; kwargs...)
    return _ls_profile_root_result(evalh, x0; kwargs...).value
end

"""
    _ls_profile_ci_result(kind, y, Xμ, Xψ, gidx, G, Q, θ̂; idx, level=0.95,
                          nll_min=nothing, se=nothing)

Profile-likelihood CI for packed parameter `idx`, inverting `2(ℓ̂ − ℓ_profile) =
χ²₁(level)` by a Venzon–Moolgavkar-style guarded-Newton root-find (the profile
deviance and its envelope-theorem slope, bracket-safeguarded). A directional
`±Inf` may mean either that no finite crossing was found in the bounded search
range or that an endpoint failed; `_ls_profile_ci_result` distinguishes them.
`se` (a Wald SE for `idx`) seeds the bracket width; if omitted it is derived from
the observed information.

`Zη`/`Zψ` are the mean/scale latent loadings and MUST match the model that was
fit: the profiler reconstructs the marginal from `(kind, y, Xμ, Xψ, gidx, G, Q)`
plus these loadings, so passing the wrong loadings profiles a different model
(#325.4). They default to the canonical loadings (`Zη=[1 0]`, `Zψ=[0 1]`) — the
only case wired through `LocScaleObjective` today; a non-canonical (slope-axis)
fit must pass its own `Zη`/`Zψ` here for the CI to be correct.
"""
function _ls_profile_ci_result(kind, y, Xμ, Xψ, gidx, G, Q, θ̂; idx::Int, level::Real = 0.95,
                               nll_min = nothing, se = nothing,
                               Zη = _ls_canonical_Zeta(length(y)),
                               Zψ = _ls_canonical_Zpsi(length(y)),
                               whitened::Bool = false)
    nmin = if nll_min !== nothing
        nll_min
    elseif whitened
        result = _ls_whitened_eval(kind, y, Xμ, Xψ, gidx, G, Q, θ̂, Zη, Zψ; gradient=false)
        result.status.ok ? result.value : _LS_PROFILE_INFEASIBLE
    else
        _ls_fit_nll(kind, y, Xμ, Xψ, gidx, G, Q, θ̂, Zη, Zψ)
    end
    half = Distributions.quantile(Distributions.Chisq(1), level) / 2
    z = Distributions.quantile(Distributions.Normal(), 1 - (1 - level) / 2)
    if se === nothing
        V = whitened ? _ls_whitened_vcov(kind, y, Xμ, Xψ, gidx, G, Q, θ̂, Zη, Zψ) :
                       _ls_vcov(kind, y, Xμ, Xψ, gidx, G, Q, θ̂, Zη, Zψ)
        se = (V === nothing || V[idx, idx] ≤ 0) ? abs(θ̂[idx]) + 1.0 : sqrt(V[idx, idx])
    end
    step0 = max(z * se, 1e-3)
    p = length(θ̂); free = [k for k in 1:p if k != idx]
    mwarm = whitened ? Ref{Union{Nothing,_LSWhitenedSeed}}(nothing) :
                      Ref{Union{Nothing,Vector{Float64}}}(nothing)
    lastsol = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    # Profile gap, its envelope-theorem slope, and feasibility at θ[idx] = val.
    function evalh(val)
        nuisance = _ls_profile_nll_result(kind, y, Xμ, Xψ, gidx, G, Q, θ̂, idx, val;
                                           Zη=Zη, Zψ=Zψ, x0=lastsol[], mwarm=mwarm,
                                           whitened=whitened)
        nuisance.accepted || return (gap=NaN, slope=NaN, ok=false,
                                     reason=nuisance.reason, nuisance=nuisance)
        lastsol[] = nuisance.minimizer
        # Slope ∂nll/∂θ[idx] at the constrained optimum: the idx-component of the
        # exact full gradient (free-parameter components ≈ 0 by stationarity).
        θ = collect(float.(θ̂)); θ[free] .= nuisance.minimizer; θ[idx] = val
        slope = try
            g = whitened ?
                _ls_whitened_eval(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ; seed=mwarm[]).gradient :
                _ls_marginal_grad(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ; a0=mwarm[])
            (length(g) == p && isfinite(g[idx])) ? g[idx] : NaN
        catch err
            err isa InterruptException && rethrow()
            NaN
        end
        reference = _profile_reference_difference(nuisance.value, nmin)
        reference.status === :accepted || return (
            gap=NaN, slope=NaN, ok=false, reason=reference.status,
            cancellation=reference.cancellation, nuisance=nuisance,
        )
        gap = reference.difference - half
        isfinite(gap) || return (
            gap=NaN, slope=NaN, ok=false, reason=:insufficient_precision,
            cancellation=reference.cancellation, nuisance=nuisance,
        )
        return (gap=gap, slope=slope, ok=true, cancellation=reference.cancellation,
                nuisance=nuisance)
    end
    lower = _ls_profile_root_result(evalh, θ̂[idx]; dir=-1.0, init=step0)
    lastsol[] = nothing                         # reset warm-start before the other side
    mwarm[] = nothing
    upper = _ls_profile_root_result(evalh, θ̂[idx]; dir=+1.0, init=step0)
    return (lower=lower.value, upper=upper.value, lower_status=lower, upper_status=upper)
end

# Historical two-bound CI helper retained for direct internal callers.
"""
    _ls_profile_ci(kind, y, Xμ, Xψ, gidx, G, Q, θ̂; kwargs...)

Compatibility wrapper returning only `(lower, upper)` from
[`_ls_profile_ci_result`](@ref). Use the structured result to inspect endpoint
failure, no-crossing, and nuisance diagnostics.
"""
function _ls_profile_ci(kind, y, Xμ, Xψ, gidx, G, Q, θ̂; kwargs...)
    result = _ls_profile_ci_result(kind, y, Xμ, Xψ, gidx, G, Q, θ̂; kwargs...)
    return (lower=result.lower, upper=result.upper)
end
