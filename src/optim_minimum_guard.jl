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

import LinearAlgebra

# Unit-free stationarity, the drmTMB #1503 rule. An absolute gradient tolerance
# (Optim's `g_tol = 1e-8`) moves with the predictor units: the same model at
# `x` and at `x * 1000` can stall with `|g|∞` just above 1e-8 while the Newton
# step is a negligible fraction of a standard error. Converged means
#
#     max_i | (H⁻¹ g)_i | / SE_i  ≤  1e-3
#
# with `SE_i = sqrt((H⁻¹)_ii)`. A Hessian that is non-finite, singular, or not
# positive definite is not usable for that ratio. A saddle (a negative
# eigenvalue) is not a minimum. A singular Hessian falls back to the absolute
# rule `max |g| ≤ 1e-3`, which still rejects a runaway and accepts a stall
# that only missed the absolute 1e-8 bar.
const _UNITFREE_SE_TOL = 1e-3
const _UNITFREE_ABS_TOL = 1e-3

function _unitfree_converged(H::AbstractMatrix, g::AbstractVector)
    length(g) == size(H, 1) == size(H, 2) || return false
    all(isfinite, g) || return false
    abs_ok = maximum(abs, g) <= _UNITFREE_ABS_TOL
    all(isfinite, H) || return abs_ok
    Hs = Matrix{Float64}(H)
    Hs = LinearAlgebra.Symmetric((Hs .+ Hs') ./ 2)
    ev = LinearAlgebra.eigvals(Hs)
    all(isfinite, ev) || return abs_ok
    scale = maximum(abs, ev)
    scale == 0 && return abs_ok
    # A negative eigenvalue is a saddle, not a minimum (same bar as the vcov guard).
    minimum(ev) < -_VCOV_RTOL * scale && return false
    minimum(abs, ev) <= _VCOV_RTOL * scale && return abs_ok
    C = LinearAlgebra.cholesky(Hs; check = false)
    LinearAlgebra.issuccess(C) || return abs_ok
    δ = C \ collect(Float64, g)
    n = length(g)
    se2 = LinearAlgebra.diag(C \ Matrix{Float64}(LinearAlgebra.I, n, n))
    worst = 0.0
    for i in 1:n
        se = sqrt(max(se2[i], 0.0))
        se > 0 || return abs_ok
        worst = max(worst, abs(δ[i]) / se)
    end
    return worst <= _UNITFREE_SE_TOL
end
