# control.jl — `DRMControl`, the one `control =` keyword of `drm()`, and prior
# `weights =` on the fixed-effect routes (owner decisions 8 and 9, 11 Oct 2026;
# #942 #948 #994 #971 #931).
#
# How it reaches every family route without a keyword on every fitter:
#
#   * `@drm_entry` wraps each public `drm(::Formula, ::Family; ...)` method. It
#     adds the `control` and `weights` keywords and runs the method body inside
#     a task-local *control scope*.
#   * A fitter that honours a field reads the scope through one helper and marks
#     the field as consumed: `_ctl_optimize` (maxit / start / multi_start),
#     `_ctl_se` (se) and `_ctl_weights` (weights).
#   * When the route returns, `_ctl_finish` checks that every field the user set
#     was consumed. A field that no fitter consumed is a named refusal
#     (`ArgumentError`), never a silent no-op. The one exception is `se = false`:
#     a route with no skip path still computes its Hessian, and the wrapper then
#     drops the standard errors and records `se = :discarded` in the report.
#
# The names follow drmTMB's `drm_control()` (`maxit`, `start`, `multi_start`,
# `se`). No drmTMB source is used; only the documented argument names.

using Random: Xoshiro, randn

"""
    DRMControl(; maxit = nothing, start = nothing, multi_start = 1, se = true,
                 seed = 1, jitter = 0.5)

Optimiser and inference settings for [`drm`](@ref), passed as one keyword:
`drm(f, family; data, control = DRMControl(...))`. The field names follow
drmTMB's `drm_control()`.

| Field | Meaning |
|:--|:--|
| `maxit` | Iteration cap of the outer optimiser (`nothing` = the route's default cap). |
| `start` | Starting values: a vector on the internal parameter scale, in `coef(fit)` order (see [`coefnames`](@ref) of a default fit). `nothing` = the route's own start. |
| `multi_start` | Number of starts. `1` = a single start. With `k > 1`, the first start is `start` (or the route's own start), and starts `2:k` add reproducible perturbations to it. The fit with the lowest objective is kept. |
| `se` | `false` skips the standard errors. `vcov(fit)` is then all `NaN`. |
| `seed` | Seed of the perturbations for `multi_start` (a DRModels.jl extension). |
| `jitter` | Perturbation size for `multi_start`. Start `j > 1` is `x0 .+ jitter .* max.(abs.(x0), 1) .* z`, where `z` is standard normal (a DRModels.jl extension). |

# Which routes honour which field

* `se = false` works on **every** route. Fixed-effect fits of every univariate
  family skip the Hessian. Routes that already had an `se` keyword skip it too.
  On any other route, the Hessian is still computed and then discarded.
  [`control_info`](@ref)`(fit).se` tells you which case applied (`:skipped` or
  `:discarded`).
* `maxit`, `start` and `multi_start` are honoured on the **fixed-effect** routes
  of every univariate family. This includes the main Gaussian location-scale
  route. A model with random, structured (`phylo`, `relmat`, `spatial`, …) or
  `meta_V` terms, or a bivariate model, refuses these fields with an
  `ArgumentError` before fitting. A route that does not honour a field it was
  given never ignores it silently: it throws an `ArgumentError` that names the
  field.

The default `DRMControl()` changes nothing: a fit with it is identical to a fit
without `control`.

See [`control_info`](@ref) for what a controlled fit did (per-start
log-likelihoods, the kept start, the `se` outcome).

# Example
```julia
f = bf(@formula(dist ~ 1 + speed), @formula(sigma ~ 1 + speed))
fit = drm(f, Gaussian(); data = cars, control = DRMControl(se = false))
all(isnan, vcov(fit))                                  # no standard errors

fit5 = drm(f, Gamma(); data = dat, control = DRMControl(multi_start = 5, maxit = 2000))
control_info(fit5).start_logliks                       # one log-likelihood per start
```
"""
struct DRMControl
    maxit::Union{Nothing,Int}
    start::Union{Nothing,Vector{Float64}}
    multi_start::Int
    se::Bool
    seed::Int
    jitter::Float64
    function DRMControl(maxit, start, multi_start, se, seed, jitter)
        maxit === nothing || maxit >= 1 ||
            throw(ArgumentError("DRMControl: `maxit` must be a positive integer or `nothing`; got $maxit"))
        multi_start >= 1 ||
            throw(ArgumentError("DRMControl: `multi_start` must be at least 1; got $multi_start"))
        if start !== nothing
            isempty(start) && throw(ArgumentError("DRMControl: `start` is empty"))
            all(isfinite, start) ||
                throw(ArgumentError("DRMControl: `start` must contain only finite values"))
        end
        (isfinite(jitter) && jitter > 0) ||
            throw(ArgumentError("DRMControl: `jitter` must be a positive finite number; got $jitter"))
        new(maxit, start, multi_start, se, seed, jitter)
    end
end

function DRMControl(; maxit = nothing, start = nothing, multi_start::Integer = 1,
                    se::Bool = true, seed::Integer = 1, jitter::Real = 0.5)
    maxit === nothing || maxit isa Integer ||
        throw(ArgumentError("DRMControl: `maxit` must be a positive integer or `nothing`; got $(repr(maxit))"))
    st = start === nothing ? nothing :
        start isa AbstractVector{<:Real} ? Vector{Float64}(start) :
        throw(ArgumentError("DRMControl: `start` must be a real vector on the internal " *
                            "parameter scale, in `coef(fit)` order; got $(typeof(start))"))
    return DRMControl(maxit === nothing ? nothing : Int(maxit), st, Int(multi_start), se,
                      Int(seed), Float64(jitter))
end

function Base.show(io::IO, c::DRMControl)
    print(io, "DRMControl(maxit = ", repr(c.maxit), ", start = ",
          c.start === nothing ? "nothing" : "[$(length(c.start)) values]",
          ", multi_start = ", c.multi_start, ", se = ", c.se, ")")
end

_ctl_optim_requested(c::DRMControl) =
    c.maxit !== nothing || c.start !== nothing || c.multi_start > 1

"""
    DRMControlReport

What `drm(...; control, weights)` did, returned by [`control_info`](@ref).

Fields:
- `control::DRMControl`: the settings used.
- `start_logliks::Vector{Float64}`: the final log-likelihood reached from each
  start, in start order (`NaN` for a start that failed). Empty when the
  optimiser settings were not used.
- `kept_start::Int`: index of the start that was kept (`0` when not used).
- `starts_agree::Bool`: `true` when every successful start reached the kept
  log-likelihood to within `1e-6 * (1 + |loglik|)`.
- `se::Symbol`: `:computed`, `:skipped` (the Hessian was not computed) or
  `:discarded` (the route computed it, and the wrapper dropped it because
  `se = false`).
- `weights`: the prior weights, or `nothing`.
"""
struct DRMControlReport
    control::DRMControl
    start_logliks::Vector{Float64}
    kept_start::Int
    starts_agree::Bool
    se::Symbol
    weights::Union{Nothing,Vector{Float64}}
end

# ---------------------------------------------------------------------------
# The task-local control scope
# ---------------------------------------------------------------------------

mutable struct _ControlScope
    control::DRMControl
    weights::Union{Nothing,Vector{Float64}}
    formula::Any                 # the formula object the scope was opened for
    used_optim::Bool
    used_se::Bool
    used_weights::Bool
    start_nll::Vector{Float64}
    kept::Int
end

const _CTL_KEY = :DRModels_drm_control_scope

_active_control_scope() = get(task_local_storage(), _CTL_KEY, nothing)

function _check_prior_weights(w)
    w === nothing && return nothing
    w isa AbstractVector{<:Real} ||
        throw(ArgumentError("drm: `weights` must be a vector of non-negative real numbers, " *
                            "one per row of `data`; got $(typeof(w))"))
    wv = Vector{Float64}(w)
    isempty(wv) && throw(ArgumentError("drm: `weights` is empty"))
    all(x -> isfinite(x) && x >= 0, wv) ||
        throw(ArgumentError("drm: `weights` must be finite and non-negative"))
    any(>(0), wv) || throw(ArgumentError("drm: `weights` are all zero"))
    return wv
end

# Does any formula of `f` carry a random, structured, meta_V or sd() term? On
# such a model the optimiser fields and `weights` are refused before fitting.
function _ctl_has_latent_terms(f)
    f isa DrmFormula || return true          # bivariate / other bundles: not a fixed-effect route
    try
        isempty(_sd_parts(f)) || return true
        isempty(_sdphylo_parts(f)) || return true
        for (_, rhs) in f.forms
            _, re, mv, st = _split_ranef(rhs; allow_phylo_slope = true, allow_temporal = true)
            (isempty(re) && mv === nothing && st === nothing) || return true
        end
    catch
        return false     # unparseable here: let the route decide, `_ctl_finish` still checks
    end
    return false
end

function _ctl_preflight(scope::_ControlScope, f, fam)
    c = scope.control
    wants_optim = _ctl_optim_requested(c)
    wants_w = scope.weights !== nothing
    (wants_optim || wants_w) || return nothing
    _ctl_has_latent_terms(f) || return nothing
    what = String[]
    c.maxit !== nothing && push!(what, "maxit")
    c.start !== nothing && push!(what, "start")
    c.multi_start > 1 && push!(what, "multi_start")
    wants_optim && (what = ["`control = DRMControl(" * join(what, ", ") * " = …)`"])
    wants_w && push!(what, "`weights`")
    kind = f isa DrmFormula ? "a model with random, structured, `meta_V` or `sd()` terms" :
                              "a bivariate or multi-response model"
    throw(ArgumentError("drm: " * join(what, " and ") * " is supported only on the " *
        "fixed-effect routes of univariate families for now (#948, #994, #931). This is " *
        kind * " ($(typeof(fam))). `control = DRMControl(se = false)` works on every route."))
end

# Run a `drm` method body under the control scope. Called by `@drm_entry`.
function _drm_run_controlled(body::Function, f, fam, control, weights)
    outer = _active_control_scope()
    if control === nothing && weights === nothing
        outer === nothing && return body()
        # The missing-response wrappers re-enter `drm` on the SAME formula object
        # with the observed rows: that call belongs to the scope. Any other nested
        # `drm` call (an internal sub-fit) runs without the user's control.
        outer.formula === f && return body()
        return task_local_storage(body, _CTL_KEY, nothing)
    end
    control === nothing && (control = DRMControl())
    control isa DRMControl ||
        throw(ArgumentError("drm: `control` must be a `DRMControl(...)`; got $(typeof(control))"))
    scope = _ControlScope(control, _check_prior_weights(weights), f,
                          false, false, false, Float64[], 0)
    _ctl_preflight(scope, f, fam)
    fit = task_local_storage(body, _CTL_KEY, scope)
    return _ctl_finish(fit, scope, fam)
end

function _ctl_finish(fit, scope::_ControlScope, fam)
    c = scope.control
    route = "$(typeof(fam)) route"
    if _ctl_optim_requested(c) && !scope.used_optim
        throw(ArgumentError("drm: this $route does not honour `control = DRMControl(maxit, " *
            "start, multi_start)` yet (#948, #994); the fit was not returned. Drop those " *
            "fields or use a fixed-effect model."))
    end
    if scope.weights !== nothing && !scope.used_weights
        throw(ArgumentError("drm: this $route does not support prior `weights` yet (#931); " *
            "the fit was not returned. Weights are implemented on the fixed-effect ML " *
            "routes of univariate families."))
    end
    fit isa DrmFit || throw(ArgumentError("drm: `control` / `weights` are not supported " *
        "on this $route, which returns a $(typeof(fit))"))
    se = :computed
    V = fit.vcov
    if !c.se
        if scope.used_se || all(isnan, V)
            se = :skipped
        else
            se = :discarded
            V = fill(NaN, size(V))
        end
    end
    lls = -scope.start_nll
    agree = true
    if scope.kept > 0
        best = lls[scope.kept]
        tol = 1e-6 * (1 + abs(best))
        agree = all(ll -> !isfinite(ll) || abs(ll - best) <= tol, lls)
        nfail = count(!isfinite, lls)
        if !agree || nfail > 0
            @warn "drm: the $(length(lls)) starts of `multi_start` did not all agree. " *
                  "Kept start $(scope.kept) (logLik $(best)); the log-likelihoods per start " *
                  "are $(lls)$(nfail > 0 ? "; $nfail start(s) failed" : ""). See `control_info(fit)`."
        end
    end
    report = DRMControlReport(c, lls, scope.kept, agree, se, scope.weights)
    return _withcontrolinfo(fit, V, report)
end

_withcontrolinfo(fit, V, report) = DrmFit(fit.family, fit.blocks, fit.coefnames, fit.theta,
    V, fit.loglik, fit.nobs, fit.converged, fit.means, fit.obs, fit.scales, fit.formula, fit.nll,
    fit.nllgrad, fit.ranef, fit.estim_method, fit.reml_loglik, fit.ml_loglik, fit.marginal,
    fit.phylo_penalty, fit.penalty, fit.iterations, fit.phylo_scale, report,
    fit.uncertainty, fit.geometry)

# ---------------------------------------------------------------------------
# Helpers the fitters call
# ---------------------------------------------------------------------------

"""
    _ctl_optimize(run, θ0; g_tol) -> Optim result

Run the outer optimiser of a fixed-effect route. `run(x0, options)` must call
`Optim.optimize` from `x0` with `options`. Without an active `maxit` / `start` /
`multi_start`, this is exactly `run(θ0, Optim.Options(g_tol = g_tol))`.
"""
function _ctl_optimize(run::Function, θ0::AbstractVector; g_tol::Real)
    s = _active_control_scope()
    if s === nothing || !_ctl_optim_requested(s.control)
        return run(θ0, Optim.Options(g_tol = g_tol))
    end
    c = s.control
    s.used_optim = true
    x0 = Vector{Float64}(θ0)
    if c.start !== nothing
        length(c.start) == length(x0) ||
            throw(ArgumentError("DRMControl: `start` has $(length(c.start)) values but this " *
                "model has $(length(x0)) parameters (in `coef(fit)` order; see " *
                "`coefnames(fit)` of a default fit)"))
        x0 = copy(c.start)
    end
    opts = c.maxit === nothing ? Optim.Options(g_tol = g_tol) :
                                 Optim.Options(g_tol = g_tol, iterations = c.maxit)
    rng = Xoshiro(c.seed)
    nll = fill(NaN, c.multi_start)
    best = nothing; kept = 0
    for k in 1:c.multi_start
        xk = k == 1 ? x0 : x0 .+ c.jitter .* max.(abs.(x0), 1.0) .* randn(rng, length(x0))
        res = try
            run(xk, opts)
        catch e
            (e isa InterruptException || c.multi_start == 1) && rethrow()
            nothing
        end
        res === nothing && continue
        v = Optim.minimum(res)
        isfinite(v) || continue
        nll[k] = v
        if best === nothing || v < Optim.minimum(best)
            best = res; kept = k
        end
    end
    best === nothing &&
        error("drm: none of the $(c.multi_start) starts reached a finite objective")
    s.start_nll = nll; s.kept = kept
    return best
end

"""
    _ctl_se(compute, θ̂) -> Matrix

Return `compute()` (the route's covariance), unless the active control says
`se = false`. In that case return an all-`NaN` matrix without calling `compute`.
"""
function _ctl_se(compute::Function, θ̂::AbstractVector)
    s = _active_control_scope()
    if s !== nothing && !s.control.se
        s.used_se = true
        return fill(NaN, length(θ̂), length(θ̂))
    end
    return compute()
end

# Merge a route's own `se` keyword with the active control (used by `@drm_entry`).
function _ctl_merge_se(se::Bool)
    s = _active_control_scope()
    return s === nothing ? se : (se && s.control.se)
end

"""
    _ctl_weights(n; route) -> Vector{Float64}

Prior weights for an `n`-row fixed-effect likelihood: the user's `weights`, or
`ones(n)` when none were given. A fitter that calls this multiplies row `i`'s
log-likelihood by `w[i]` (drmTMB semantics). Pass `supported = false` from a
fitter whose estimator cannot take weights. It then refuses them by name.
"""
function _ctl_weights(n::Integer; supported::Bool = true, route::AbstractString = "this route")
    s = _active_control_scope()
    (s === nothing || s.weights === nothing) && return ones(n)
    supported || throw(ArgumentError("drm: prior `weights` are not supported on $route (#931)"))
    w = s.weights
    length(w) == n || throw(DimensionMismatch("drm: `weights` has $(length(w)) entries but " *
        "the model uses $n rows. Give one weight per row of `data`; rows with a missing " *
        "response are not yet supported together with `weights`."))
    s.used_weights = true
    return w
end

# ---------------------------------------------------------------------------
# @drm_entry: add `control` / `weights` to a public `drm` method
# ---------------------------------------------------------------------------

_kwname(p::Symbol) = p
function _kwname(p::Expr)
    p.head === :kw && return _kwname(p.args[1])
    p.head === :(::) && return _kwname(p.args[1])
    p.head === :... && return :...
    error("@drm_entry: unexpected keyword form $p")
end

"""
    @drm_entry function drm(f::Formula, fam::Family; kw...) ... end

Add the `control` and `weights` keywords to a public `drm` method, and run its
body under the control scope (see src/control.jl). A method with an `se`
keyword also has `se` merged with `control.se`.
"""
macro drm_entry(fdef)
    (fdef isa Expr && fdef.head === :function) || error("@drm_entry needs a function definition")
    sig = fdef.args[1]
    call = sig.head === :where ? sig.args[1] : sig
    (call.head === :call && length(call.args) >= 4 && call.args[2] isa Expr &&
     call.args[2].head === :parameters) ||
        error("@drm_entry needs `drm(formula, family; keywords...)`")
    params = call.args[2]
    names = map(_kwname, params.args)
    splat = findfirst(==(:...), names)
    insert_at = splat === nothing ? length(params.args) + 1 : splat
    insert!(params.args, insert_at, Expr(:kw, :weights, nothing))
    insert!(params.args, insert_at, Expr(:kw, :control, nothing))
    fname = _kwname(call.args[3])
    famname = _kwname(call.args[4])
    body = fdef.args[2]
    if :se in names
        # A route's own `se = false` keyword is the same request as
        # `control = DRMControl(se = false)`: route it through the scope so the
        # fixed-effect fitters skip the Hessian too (#942).
        body = :(let se = _ctl_merge_se(se); $body; end)
        pre = :(se || control !== nothing || _active_control_scope() !== nothing || (control = DRMControl(se = false)))
    else
        pre = nothing
    end
    fdef.args[2] = quote
        $pre
        return _drm_run_controlled($fname, $famname, control, weights) do
            $body
        end
    end
    return esc(:(Base.@__doc__ $fdef))
end
