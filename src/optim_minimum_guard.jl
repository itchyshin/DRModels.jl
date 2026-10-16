# optim_minimum_guard.jl — evaluate the objective AT THE MINIMIZER, never trust
# a bare `Optim.minimum(res)` for a REPORTED value or a cross-candidate COMPARISON.
#
# Root cause (Optim.jl v1.13.3, this repo's Manifest; confirmed by a minimal
# reproduction, not inferred): when a line search fails, `perform_linesearch!`
# (src/utilities/perform_linesearch.jl) catches the `LineSearchException`, sets
# `state.alpha = ex.alpha` and returns `false`. `update_state!`
# (src/multivariate/solvers/first_order/l_bfgs.jl) STILL applies that alpha —
# `state.x .= state.x .+ state.alpha .* state.s` — moving the minimizer, but
# then returns `true` to signal failure. The outer loop
# (src/multivariate/optimize/optimize.jl) breaks on that signal BEFORE calling
# `update_g!`, which is what would otherwise refresh the objective/gradient
# cache at the new `state.x`. So `Optim.minimum(res)` can be the value from an
# earlier, REJECTED line-search trial, while `Optim.minimizer(res)` has already
# moved past it. Reproduced directly: an LBFGS fit with a flat (zero-gradient)
# barrier region and an oversized initial step forces "Status: failure (line
# search failed)" with `Optim.minimum(res) == 1e18` (the barrier sentinel)
# while `f(Optim.minimizer(res)) == 9.0` (the true value at the point actually
# returned) — the same pattern found in draft PR #827 (minimizer's loglik
# -296.2228 vs. `Optim.minimum` reading 1e18).
#
# This can ONLY happen when `Optim.converged(res) == false` (a line-search
# failure ends the `optimize` call immediately, so the run cannot also report
# converged), so a value already guarded behind `Optim.converged(res)` does not
# need this helper. Everywhere else that reports or compares the objective must
# re-evaluate it at the minimizer instead.

"""
    _objective_at_minimizer(f, res)

Evaluate `f` fresh at `Optim.minimizer(res)` instead of trusting
`Optim.minimum(res)`, which can hold the value of an earlier, rejected
line-search trial after a failed line search (see this file's header). Use
this wherever the objective is REPORTED (e.g. as a logLik) or COMPARED across
candidates/restarts. In the converged case the two are identical, so this
changes nothing there.
"""
_objective_at_minimizer(f, res) = f(Optim.minimizer(res))

"""
    _objective_at_minimizer_fg(fg!, res)

As [`_objective_at_minimizer`](@ref), for objectives built with
`Optim.only_fg!(fg!)` / `Optim.NLSolversBase.only_fg!(fg!)`, whose closure has
signature `fg!(F, G, x)`. Calls `fg!(true, nothing, x̂)` at
`x̂ = Optim.minimizer(res)` to get the objective value without touching `G`.
"""
_objective_at_minimizer_fg(fg!, res) = fg!(true, nothing, Optim.minimizer(res))

"""
    _better_restart(f, res, res2)

Return whichever of the incumbent `res` and the restart `res2` has the lower
objective, comparing `f` evaluated FRESH at each `Optim.minimizer` (never the
possibly-stale `Optim.minimum`). A tie keeps the incumbent `res`.
"""
_better_restart(f, res, res2) =
    _objective_at_minimizer(f, res2) < _objective_at_minimizer(f, res) ? res2 : res

"""
    drm_optim_converged(res) -> Bool

True only when Optim reports convergence and the gradient criterion fired.

`Optim.converged` is the OR of the x, f, and g criteria. `Optim.Options(g_tol = g_tol)`
leaves the f and x tolerances at 0, so a numerical plateau (two identical successive
values) sets `f_converged` or `x_converged` while the gradient is still large.
That is the defect in DRModels.jl#944. Routes that deliberately stop on `f_reltol`
near a variance boundary are not switched to this predicate.
"""
drm_optim_converged(res) = Optim.converged(res) && Optim.g_converged(res)

"""
    _NEWTON_DECREMENT_TOL

Per-observation bound on the Newton decrement `½ g'H⁻¹g` used by
[`_newton_decrement_small`](@ref). `1e-10` is a judgement threshold, not a derived
or calibrated one: the decrement approximates the nll still to be gained, so
`n * 1e-10` is far below any change in logLik a user could act on, yet far above
the floating-point noise of a fit at the optimum. Change it here only.
"""
const _NEWTON_DECREMENT_TOL = 1e-10

"""
    _newton_decrement_small(g, H, n) -> Bool

Unit-free stationarity test (DRModels.jl#983). The absolute `g_tol` of
[`drm_optim_converged`](@ref) is unit dependent: for a tiny residual SD the mean
gradient scales like `1/σ²` and coefficient noise alone exceeds `g_tol` at the exact
optimum. The Newton decrement `½ g'H⁻¹g` is invariant to affine reparametrisation
(so to rescaling `y`). True only when `H` is finite and positive definite, `g` is
finite, and the decrement is below `n * _NEWTON_DECREMENT_TOL`. A plateau with a
large gradient has a large decrement and is still rejected (#944).
"""
function _newton_decrement_small(g::AbstractVector, H::AbstractMatrix, n::Integer)
    (all(isfinite, g) && all(isfinite, H)) || return false
    F = cholesky(Symmetric(Matrix(H)); check = false)
    issuccess(F) || return false
    return 0.5 * dot(g, F \ g) <= n * _NEWTON_DECREMENT_TOL
end

"""
    _SCALE_RUNOFF_LOG_SPREAD

Unit-free bound on the spread of fitted `log σ_i` across observations used by
[`_scale_runoff`](@ref): `log(1e4)`, i.e. a 10⁴-fold range of fitted scales
(10⁸-fold in variance).

Why 10⁴: it is a judgement threshold, not a derived or calibrated one. It is the
`log(1e4)` spread that DRModels.jl#1059 proposes, taken from drmTMB's native
engine rule, so the two routes can agree. It is far larger than the scale range
of ordinary heteroscedastic data, yet far from floating-point limits, so a fit
beyond it is more often a collapsing scale than a real design. The cost is that a
genuinely well-identified fit with a wider range is also reported as not
converged. Change it here only.
"""
const _SCALE_RUNOFF_LOG_SPREAD = log(1e4)

"""
    _scale_runoff(scales) -> Bool

True when the fitted scale has run off toward zero (or infinity) for some
observations (DRModels.jl#1059). `scales` is the fit's `Dict(:sigma => σ_i)`.
The rule is the spread `maximum(log σ) - minimum(log σ)` across observations
exceeding `log(1e4)`; it is invariant to the units of the response (rescaling
`y` shifts every `log σ_i` by the same constant), unlike an absolute log-sigma
bound. A divergent `sigma ~ x` fit (a group whose response is constant, so its
log σ → -∞ while the gradient test still fires) is flagged; a non-finite or
non-positive scale counts as run-off. Callers AND the negation into
`converged`, as the Hessian-finite check already is.
"""
function _scale_runoff(scales)
    σ = get(scales, :sigma, nothing)
    σ === nothing && return false
    isempty(σ) && return false
    all(s -> isfinite(s) && s > 0, σ) || return true
    lo, hi = extrema(log, σ)
    return hi - lo > _SCALE_RUNOFF_LOG_SPREAD
end

"""
    _fd_gradient!(G, f, θ, h; one_sided = false) -> G

Central finite-difference gradient of `f` at `θ` with absolute step `h`, written
into `G`. A coordinate whose `θ ± h` probe is not finite is `NaN`, never `0`
(DRModels.jl#946): a zero there counts toward Optim's `g_tol` and can manufacture
a "converged" point exactly where the objective stops being finite, i.e. at the
variance boundary. When every probe is finite the result is the plain central
difference, unchanged from the inline loops this replaces.

`one_sided = true` is for an optimiser's search direction only: a coordinate with
exactly one finite probe then takes the one-sided difference against `f(θ)`
(evaluated once, only if needed), so the line search keeps a genuine slope instead
of a fabricated zero. It is still `NaN` when neither probe, or `f(θ)` itself, is
finite. A convergence verdict must use the strict default.
"""
function _fd_gradient!(G::AbstractVector, f, θ::AbstractVector, h::Real;
                       one_sided::Bool = false)
    f0 = nothing
    @inbounds for k in eachindex(θ)
        tp = copy(θ); tp[k] += h
        tm = copy(θ); tm[k] -= h
        fp = f(tp); fm = f(tm)
        if isfinite(fp) && isfinite(fm)
            G[k] = (fp - fm) / (2h)
        elseif one_sided && (isfinite(fp) || isfinite(fm))
            f0 === nothing && (f0 = f(θ))
            G[k] = !isfinite(f0) ? NaN : isfinite(fp) ? (fp - f0) / h : (f0 - fm) / h
        else
            G[k] = NaN
        end
    end
    return G
end

"""
    _gradient_verified(res, g, n, g_tol, hessian) -> Bool

Gradient-verified `converged` flag for the routes that let Optim stop on a relative
objective change (`f_reltol`): the q=4 ML, Fisher-z and REML fitters and the
coevolution q / q=2 residual / q=2 REML fitters (DRModels.jl#946). `Optim.converged`
is an OR over the x, f and g criteria, so on those routes it is `true` at an
f-plateau whatever the gradient. This adds no new criterion: it is the two-branch
rule of `_fit_fixed_gaussian` (#983), drm_optim_converged's gradient test OR the
unit-free [`_newton_decrement_small`](@ref), applied to a gradient re-evaluated
at the minimizer.

- `g`: gradient of the SUM-scale negative log-likelihood at the minimizer, over
  the free coordinates only (a pinned coordinate's gradient is not zero; its
  constraint is active). Re-evaluated, not Optim's cached one, which can be stale
  after a failed line search (see this file's header) or belong to a polish run
  judged at another tolerance. Finite-difference gradients must come from the
  strict [`_fd_gradient!`](@ref), so a non-finite probe is `NaN` and fails.
- `n`: number of observations (data rows), as in `_newton_decrement_small`.
- `g_tol`: the caller's tolerance on the MEAN-scale gradient these routes hand to
  Optim, so the first branch is `maximum(abs, g) / n <= g_tol`, the test
  `Optim.g_converged` applies.
- `hessian`: zero-argument function returning `(H, ok)`, the sum-scale Hessian
  over the same free coordinates (e.g. `_fd_hessian_from_grad` or
  `_fd_hessian_from_values`). It is called only when the first branch fails, so a
  fit that meets `g_tol` pays nothing extra. `nothing` skips the second branch
  (cheap, conservative verdict for internal warm starts).

True when `g` is finite and either Optim reports convergence with the absolute
gradient test met, or the stop was not a budget stop (iteration or time limit) and
the Newton decrement is below `n * _NEWTON_DECREMENT_TOL` with a finite positive
definite `H`. An f-plateau with a large gradient, typical at a variance boundary,
has a large decrement and is reported as not converged. There is no separate
boundary status: a boundary fit is `converged` only if it meets this gradient rule.
"""
function _gradient_verified(res, g::AbstractVector, n::Integer, g_tol::Real, hessian)
    all(isfinite, g) || return false
    Optim.converged(res) && maximum(abs, g; init = 0.0) / n <= g_tol && return true
    (Optim.iteration_limit_reached(res) || res.stopped_by.time_limit) && return false
    hessian === nothing && return false
    H, ok = hessian()
    ok || return false
    return _newton_decrement_small(g, H, n)
end
