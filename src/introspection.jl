# introspection.jl — A4d-2. Two post-fit inventories, the Julia twins of
# drmTMB's `profile_targets()` (R/profile.R) and `structured_effects()`
# (R/methods.R).
#
# Neither changes a fit. Both exist so downstream code does not have to re-parse
# formula text or guess what `confint(..., method = :profile)` will accept — and
# both are careful to report what is ACTUALLY available on THIS fit rather than
# what the package can do in general. A readiness column that always says "ready"
# would be worse than no column at all.

"""
    profile_targets(fit::DrmFit; ready_only = false) -> Vector{NamedTuple}

Every parameter [`profile_result`](@ref) / `confint(..., method = :profile)` can
be asked for on **this** fit, with an honest readiness flag — drmTMB's
`profile_targets()`.

Runs no optimisation: it walks the fitted object. One row per coefficient with

- `parm` — the coefficient name;
- `param` — its block (`:mu`, `:sigma`, `:resd`, …);
- `index` — its position in `fit.theta`;
- `estimate` — the fitted value **on the estimation scale**;
- `scale` — `:log` for a variance-component / scale coefficient (and the
  temporal OU decay), `:atanh` for the temporal AR1 persistence and the Toeplitz partial
  autocorrelations, `:identity` otherwise;
- `profile_ready` — whether a profile interval can actually be computed here;
- `profile_note` — why, when it cannot.

Pass `ready_only = true` to drop the unavailable rows.

# Example
```julia
tg = profile_targets(fit)
filter(r -> !r.profile_ready, tg)        # what will refuse, and why
```
"""
function profile_targets(fit::DrmFit; ready_only::Bool = false)
    jobs = _profile_jobs(fit, nothing)
    stored = _glsp_stored_profile_rows(fit)
    stored_params = Set(r.param for r in stored)

    # Mirror `profile_result`'s dispatch exactly — if the readiness rule and the
    # dispatch rule ever disagree, this inventory becomes a liar.
    route, note = if fit.nll isa LocScaleObjective
        (:locscale, "")
    elseif fit.nll isa LocOnlyObjective && !isempty(jobs) && all(j -> j.param === :resd, jobs)
        (:loconly, "")
    elseif !isempty(stored)
        (:stored, "profile CI precomputed at fit time (σ-phylo route has no re-optimisable objective)")
    elseif fit.nll === nothing
        (:none, "the fitted objective was not stored on this fit; for the σ-phylo " *
                "location-scale route refit with `profile_ci = true`")
    else
        (:generic, "")
    end

    rows = NamedTuple[]
    for j in jobs
        ready, why = if route === :none
            (false, note)
        elseif route === :stored
            j.param in stored_params ? (true, note) :
                (false, "no precomputed profile row for block `$(j.param)` on this fit")
        else
            (true, note)
        end
        # homtoep (drmTMB #1449): mean-coefficient profiles only; σ and the
        # lag-correlation (PAC) intervals are deferred, as in drmTMB.
        if _wald_withheld(fit) && j.param !== :mu
            ready, why = (false, "temporal_homtoep_nonmean_intervals_deferred")
        end
        ready_only && !ready && continue
        push!(rows, (parm = j.coef, param = j.param, index = j.k,
                     estimate = fit.theta[j.k],
                     scale = _profile_target_scale(j.param),
                     profile_ready = ready, profile_note = why))
    end
    return rows
end

# Which coefficients live on a log scale in `theta`. The variance-component and
# residual-scale blocks are stored as logs; mean coefficients are not.
_profile_target_scale(param::Symbol) =
    param in (:sigma, :resd, :resd_mu, :resd_sigma, :recov, :sd, :sd_phylo, :temporal_decay) ? :log :
    param in (:temporal_phi, :temporal_pac) ? :atanh : :identity

"""
    structured_effects(fit::DrmFit) -> Vector{NamedTuple}

One row per **structured marker** in the fitted formula — drmTMB's
`structured_effects()`. Fields `dpar`, `kind`, `grouping`.

`kind` is the marker (`:phylo`, `:relmat`, `:animal`, `:spatial`, `:temporal`), `grouping` the
factor it wraps, and `dpar` the distributional parameter whose formula carried
it. Exists so downstream code never has to grep or re-parse formula text.

Returns an empty vector for a model with no structured markers, and for a fit
whose formula was not retained.

# Example
```julia
structured_effects(fit)
# 2-element Vector{NamedTuple}:
#  (dpar = :mu,    kind = :phylo, grouping = :species)
#  (dpar = :sigma, kind = :phylo, grouping = :species)
```
"""
function structured_effects(fit::DrmFit)
    rows = NamedTuple[]
    f = fit.formula
    f === nothing && return rows
    forms = _structured_effects_forms(f)
    forms === nothing && return rows
    for (dpar, rhs) in forms
        for (kind, grp) in _collect_structured(rhs)
            push!(rows, (dpar = dpar, kind = kind, grouping = grp))
        end
        for tt in _collect_temporal(rhs)
            push!(rows, (dpar = dpar, kind = :temporal, grouping = tt.group))
        end
    end
    return rows
end

# `DrmFormula` stores `forms::Vector{Pair{Symbol,Any}}`. Anything else (e.g. a
# bivariate formula shape) returns `nothing` rather than guessing at a layout —
# an introspection helper that invents structure is worse than one that declines.
function _structured_effects_forms(f)
    hasproperty(f, :forms) || return nothing
    fs = getproperty(f, :forms)
    fs isa AbstractVector || return nothing
    return fs
end

"""
    bridge_diagnostics(fit::DrmFit) -> NamedTuple

Route-aware convergence diagnostics for the `engine = "julia"` R bridge (#569)
— the Julia twin of [`check_drm`](@ref), reshaped for `drm_bridge`'s payload
plus the two quantities `check_drm` does not itself report: which internal
route produced this fit and which integrator/optimiser it used.

Every field is read straight off `fit`, or computed by the SAME logic
[`check_drm`](@ref) uses (`_check_max_abs_grad`, the covariance
finiteness/positive-definiteness check) — deliberately NOT by calling
`check_drm(fit)` itself, which additionally `@info`/`@warn`-logs a report on
every call. `drm_bridge` calls this for every bridged fit, and a bridge
boundary that writes to stderr on every ordinary fit is a regression in its
own right (an R caller doing `@test_nowarn drm_bridge(...)`-equivalent
checking, or just tailing its own logs, would see one `check_drm` report per
`engine = "julia"` fit that nobody asked to be told about). Nothing here is
fabricated. A quantity a route does not record is `missing`, never a
fabricated zero or `NaN` standing in for it:

- `route` — the fitted objective's Julia type name (`fit.nll === nothing`
  reports `"none"`); the honest, ungeneralised answer to "which internal
  objective fitted this model".
- `integrator` — `fit.marginal` (`:LA`, `:Laplace`, `:VA`, `:AGHQ`).
- `optimizer` — `"Optim.LBFGS"` when [`niterations`](@ref) recorded an achieved
  iteration count (see its docstring for exactly which routes that covers);
  `missing` on a route with no single outer optimiser call to attribute one to.
- `converged` — `fit.converged`.
- `iterations` — `niterations(fit)`; `missing` when unrecorded (`niterations`
  returns `-1`).
- `max_abs_grad`, `grad_source` — the same `(magnitude, source)` pair
  `check_drm` reports as `max_abs_grad`/`grad_source` (`_check_max_abs_grad`);
  `max_abs_grad` is `missing` (not `NaN`) when `grad_source` is `:none` or
  `:unavailable`, i.e. no gradient was actually produced.
- `vcov_complete` — whether `fit.vcov` is finite throughout (`check_drm`'s
  `vcov_complete`).
- `vcov_posdef`, `min_eigval`, `cond` — `check_drm`'s positive-definiteness /
  eigenvalue / condition-number trio, but `missing` (not `check_drm`'s
  documented `false`/`NaN`/`Inf` placeholders) whenever `vcov_complete` is
  `false`, since those three cannot actually be computed then.
- `penalized_map` — `fit.estim_method === :MAP` (`check_drm`'s
  `penalized_map`).
- `boundary` — 1-based indices into `fit.theta`/`diag(fit.vcov)` whose stored
  variance is non-finite or negative — the same condition [`stderror`](@ref)
  already reports as an infinite standard error. Empty (not `missing`) when
  none are.

# Example
```julia
d = bridge_diagnostics(fit)
d.route        # e.g. "LocScaleObjective"
d.grad_source  # e.g. :stored
```
"""
function bridge_diagnostics(fit::DrmFit)
    grad = _check_max_abs_grad(fit)
    iters = niterations(fit)
    V = fit.vcov
    vcov_complete = all(isfinite, V)
    vcov_posdef, min_eigval, cond_num = if vcov_complete
        S = Symmetric(V)
        ev = eigvals(S)
        mineig = minimum(ev)
        (isposdef(S), mineig, mineig > 0 ? maximum(ev) / mineig : Inf)
    else
        (false, NaN, Inf)
    end
    d = diag(V)
    boundary = findall(i -> !isfinite(d[i]) || d[i] < 0, eachindex(d))
    no_gradient = grad.source in (:none, :unavailable)
    return (
        route = fit.nll === nothing ? "none" : String(nameof(typeof(fit.nll))),
        integrator = fit.marginal,
        optimizer = iters >= 0 ? "Optim.LBFGS" : missing,
        converged = fit.converged,
        iterations = iters >= 0 ? iters : missing,
        max_abs_grad = no_gradient ? missing : grad.magnitude,
        grad_source = grad.source,
        vcov_complete = vcov_complete,
        vcov_posdef = vcov_complete ? vcov_posdef : missing,
        min_eigval = vcov_complete ? min_eigval : missing,
        cond = vcov_complete ? cond_num : missing,
        penalized_map = fit.estim_method === :MAP,
        boundary = boundary,
    )
end
