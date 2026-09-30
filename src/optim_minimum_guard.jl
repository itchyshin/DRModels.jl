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
