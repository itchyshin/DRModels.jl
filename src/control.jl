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
                 seed = 1, jitter = 0.5, optimizer = :default)

Optimiser and inference settings for [`drm`](@ref), passed as one keyword:
`drm(f, family; data, control = DRMControl(...))`. The field names follow
drmTMB's `drm_control()`.

| Field | Meaning |
|:--|:--|
| `maxit` | Iteration cap of the outer optimiser (`nothing` = the route's default cap). |
| `start` | Starting values: a vector on the internal parameter scale, in `coef(fit)` order of a default fit. `fit.coefnames` groups its labels by parameter block. `nothing` = the route's own start. |
| `multi_start` | Number of starts. `1` = a single start. With `k > 1`, the first start is `start` (or the route's own start), and starts `2:k` add reproducible perturbations to it. The fit with the lowest objective is kept. |
| `se` | `false` skips the standard errors. `vcov(fit)` is then all `NaN`. |
| `seed` | Seed of the perturbations for `multi_start` (a DRModels.jl extension). |
| `jitter` | Perturbation size for `multi_start`. Start `j > 1` is `x0 .+ jitter .* max.(abs.(x0), 1) .* z`, where `z` is standard normal (a DRModels.jl extension). |
| `optimizer` | `:default` preserves the route's current method and restart policy. `:lbfgs` or `:bfgs` runs that Optim.jl method directly on qualified routes. |

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
* `optimizer = :lbfgs` and `optimizer = :bfgs` work on the fixed-effect ML
  routes above and Gaussian ML with one mean random intercept `(1 | g)`.
  That random-intercept route uses its own conditional two-start restart;
  `maxit`, `start`, `multi_start`, `seed` and `jitter` remain unavailable there. Other latent,
  structured, bivariate and REML routes refuse an explicit optimizer. Gaussian
  selection also requires `algorithm = :auto` and no alternate integration,
  imputation or profiling options.
  This control selects the primary fit only. Fit-based `profile_result`,
  `confint(...; method = :profile)` and `bootstrap_*` refuse an explicitly
  selected fit until their refits can forward the method. Wald intervals remain
  available when standard errors were computed.

The default `DRMControl()` changes nothing: a fit with it is identical to a fit
without `control`.

See [`control_info`](@ref) for the requested and actual method, attempted
stages, native stopping settings, the kept start and the `se` outcome.

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
    optimizer::Symbol
    function DRMControl(maxit, start, multi_start, se, seed, jitter, optimizer)
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
        optimizer in (:default, :lbfgs, :bfgs) ||
            throw(ArgumentError("DRMControl: `optimizer` must be :default, :lbfgs or :bfgs; got $(repr(optimizer))"))
        new(maxit, start, multi_start, se, seed, jitter, optimizer)
    end
end

# Preserve the six-argument constructor supplied by the original control API.
DRMControl(maxit, start, multi_start, se, seed, jitter) =
    DRMControl(maxit, start, multi_start, se, seed, jitter, :default)

function DRMControl(; maxit = nothing, start = nothing, multi_start::Integer = 1,
                    se::Bool = true, seed::Integer = 1, jitter::Real = 0.5,
                    optimizer::Symbol = :default)
    maxit === nothing || maxit isa Integer ||
        throw(ArgumentError("DRMControl: `maxit` must be a positive integer or `nothing`; got $(repr(maxit))"))
    st = start === nothing ? nothing :
        start isa AbstractVector{<:Real} ? Vector{Float64}(start) :
        throw(ArgumentError("DRMControl: `start` must be a real vector on the internal " *
                            "parameter scale, in `coef(fit)` order; got $(typeof(start))"))
    return DRMControl(maxit === nothing ? nothing : Int(maxit), st, Int(multi_start), se,
                      Int(seed), Float64(jitter), optimizer)
end

function Base.show(io::IO, c::DRMControl)
    print(io, "DRMControl(maxit = ", repr(c.maxit), ", start = ",
          c.start === nothing ? "nothing" : "[$(length(c.start)) values]",
          ", multi_start = ", c.multi_start, ", se = ", c.se,
          ", optimizer = ", repr(c.optimizer), ")")
end

_ctl_optim_requested(c::DRMControl) =
    c.maxit !== nothing || c.start !== nothing || c.multi_start > 1 || c.optimizer !== :default

"""
    DRMControlReport

What `drm(...; control, weights)` did, returned by [`control_info`](@ref).

Fields:
- `control::DRMControl`: the settings used.
- `start_logliks::Vector{Float64}`: Optim.jl's reported log-likelihood from each
  start, in start order (`NaN` for a start that failed). Empty when the
  optimiser settings were not used.
- `kept_start::Int`: index of the start that was kept (`0` when not used).
- `starts_agree::Bool`: `true` when every successful start reached the kept
  log-likelihood to within `1e-6 * (1 + |loglik|)`.
- `se::Symbol`: `:computed`, `:skipped` (the Hessian was not computed) or
  `:discarded` (the route computed it, and the wrapper dropped it because
  `se = false`).
- `weights`: the prior weights, or `nothing`.
- `requested_optimizer` / `actual_optimizer`: the request and the method run
  on the selected stage (`:unrecorded` when this route does not expose its
  optimizer to this report).
- `optimizer_runs`: immutable per-attempt records in execution order with
  method, stage, start, native iterations/convergence/objective/gradient
  residual, stopping settings and finite/error status. The objective and
  gradient residual on fixed-effect routes are Optim.jl's reported values;
  Gaussian random-intercept entries recompute both at the returned minimizer.
- `selected_optimizer_stage`: the stage that supplied the returned fit.
- `native_controls`: a per-fit snapshot of `maxit`, `start`, `multi_start`,
  `seed`, `jitter` and `se`.
"""
struct DRMControlReport
    control::DRMControl
    start_logliks::Vector{Float64}
    kept_start::Int
    starts_agree::Bool
    se::Symbol
    weights::Union{Nothing,Vector{Float64}}
    requested_optimizer::Symbol
    actual_optimizer::Symbol
    optimizer_runs::Tuple
    selected_optimizer_stage::Symbol
    native_controls::NamedTuple
end

_ctl_control_snapshot(c::DRMControl) =
    DRMControl(; maxit = c.maxit, start = c.start === nothing ? nothing : copy(c.start),
               multi_start = c.multi_start, se = c.se, seed = c.seed,
               jitter = c.jitter, optimizer = c.optimizer)

_ctl_native_controls(c::DRMControl) =
    (maxit = c.maxit, start = c.start === nothing ? nothing : Tuple(c.start),
     multi_start = c.multi_start, seed = c.seed, jitter = c.jitter, se = c.se)

# Retain the report constructor exported by the original six-field control API.
# It has no solver trace, so the actual method must remain unrecorded.
function DRMControlReport(c::DRMControl, lls::Vector{Float64}, kept::Int,
                          agree::Bool, se::Symbol, weights::Union{Nothing,Vector{Float64}})
    return DRMControlReport(_ctl_control_snapshot(c), copy(lls), kept, agree, se,
                            weights === nothing ? nothing : copy(weights),
                            c.optimizer, :unrecorded, (), :none, _ctl_native_controls(c))
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
    optimizer_runs::Vector{NamedTuple}
    selected_optimizer_stage::Symbol
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

# The only latent route qualified for explicit solver choice is Gaussian ML
# with one mean random intercept. Its own two-start safeguard stays in charge.
function _ctl_simple_gaussian_re(f, fam)
    f isa DrmFormula && fam isa Gaussian || return false
    try
        isempty(_sd_parts(f)) && isempty(_sdphylo_parts(f)) || return false
        for (axis, rhs) in f.forms
            _, re, mv, st = _split_ranef(rhs; allow_phylo_slope = true, allow_temporal = true)
            (mv === nothing && st === nothing) || return false
            if axis === :mu
                length(re) == 1 && _re_kind(re[1][1])[1] === :intercept || return false
            else
                isempty(re) || return false
            end
        end
        return true
    catch
        return false
    end
end

# Latent-term shapes whose fitter reads prior `weights` (#1219, drmTMB row log-likelihood
# multipliers): a Gaussian or Poisson mean with ONE ordinary random intercept `(1 | g)`, and a
# Gaussian mean with `meta_V(v)`; every other axis fixed-effect only. Anything else keeps the
# refusal below. Each admitted fitter calls `_ctl_weights`, and `_ctl_finish` still refuses a
# fit that did not read them.
function _ctl_weights_latent_ok(f, fam)
    if f isa BivariateDrmFormula          # residual-only bivariate Gaussian: no latent term anywhere
        fam isa Gaussian || return false
        try
            for (_, rhs) in f.forms
                _, re, mv, st = _split_ranef(rhs; allow_phylo_slope = true, allow_temporal = true)
                (isempty(re) && mv === nothing && st === nothing) || return false
            end
            return true
        catch
            return false
        end
    end
    f isa DrmFormula && (fam isa Gaussian || fam isa Poisson) || return false
    try
        isempty(_sd_parts(f)) && isempty(_sdphylo_parts(f)) || return false
        for (axis, rhs) in f.forms
            _, re, mv, st = _split_ranef(rhs; allow_phylo_slope = true, allow_temporal = true)
            st === nothing || return false
            if axis === :mu
                if mv !== nothing
                    fam isa Gaussian && isempty(re) || return false
                else
                    length(re) == 1 && _re_kind(re[1][1])[1] === :intercept || return false
                end
            else
                (isempty(re) && mv === nothing) || return false
            end
        end
        return true
    catch
        return false
    end
end

function _ctl_preflight(scope::_ControlScope, f, fam)
    c = scope.control
    wants_optim = _ctl_optim_requested(c)
    wants_w = scope.weights !== nothing
    (wants_optim || wants_w) || return nothing
    _ctl_has_latent_terms(f) || return nothing
    wants_w && !wants_optim && _ctl_weights_latent_ok(f, fam) && return nothing
    if c.optimizer !== :default && !wants_w && c.maxit === nothing &&
       c.start === nothing && c.multi_start == 1 && c.seed == 1 && c.jitter == 0.5 &&
       _ctl_simple_gaussian_re(f, fam)
        return nothing
    end
    what = String[]
    c.maxit !== nothing && push!(what, "maxit")
    c.start !== nothing && push!(what, "start")
    c.multi_start > 1 && push!(what, "multi_start")
    c.seed != 1 && push!(what, "seed")
    c.jitter != 0.5 && push!(what, "jitter")
    c.optimizer !== :default && push!(what, "optimizer")
    wants_optim && (what = ["`control = DRMControl(" * join(what, ", ") * " = …)`"])
    wants_w && push!(what, "`weights`")
    kind = f isa DrmFormula ? "a model with random, structured, `meta_V` or `sd()` terms" :
                              "a bivariate or multi-response model"
    throw(ArgumentError("drm: " * join(what, " and ") * " is supported only on the " *
        "fixed-effect routes of univariate families for now (#948, #994, #931). This is " *
        kind * " ($(typeof(fam))), so the fit was refused before it started, not run " *
        "with the field ignored. drmTMB's `drm_control(multi_start =)` does accept these " *
        "fields on mixed models; the gap here is tracked in #1219. " *
        "`control = DRMControl(se = false)` works on every route."))
end

# ---------------------------------------------------------------------------
# Unsupported keywords: a readable refusal instead of a raw MethodError
# ---------------------------------------------------------------------------
# A keyword a family route does not take (`method = :REML` on LogNormal, Student or
# Tweedie; `tree`/`K`/`A`/`ids`/`coords` on a route with no structured provider)
# used to reach the user as `MethodError: no method matching drm(...; method=...)`,
# while the same call without the keyword gave a model-support message. Two entry
# points close this, neither of which changes a call that works:
#   * `@drm_entry` gives every public `drm` method a trailing `kwargs...` and checks
#     it first (methods that already had one are left alone);
#   * `_drm_run_controlled` turns the keyword `MethodError` a route's inner fitter
#     raises for a keyword it does not declare into the same refusal.

# What to try, per keyword. `R` is what drmTMB 0.8.1 does for the same request, from
# the refusal-gap map (docs/dev-log/parity-crosswalk-evidence-20261107/fuzz-gap-map).
function _kw_advice(key::Symbol, fam, accepted = Symbol[])
    famn = string(nameof(typeof(fam)))
    if key === :ids && (:A in accepted || :coords in accepted)
        return "`ids` labels the rows of `A` or `coords`, and this route reads them in the order " *
               "the group levels first appear in the data. Drop `ids` and pass `A` (or `coords`) " *
               "in that order." * _kw_r_note(:A, fam)
    elseif key === :method
        r = fam isa Union{LogNormal,Student,Tweedie} ?
            " drmTMB 0.8.1 does not fit REML = TRUE for this family either." : ""
        return "this route is ML-only: drop `method` (ML is the default). REML exists only on " *
               "the routes that document it (for example Gaussian)." * r
    elseif key === :tree
        return "`tree` is read by a `phylo(...)` term, and $famn has no phylogenetic route for " *
               "this model. Drop `tree` and the `phylo(...)` term, or use a family with a " *
               "phylogenetic route." * _kw_r_note(key, fam)
    elseif key === :K
        return "`K` is read by a `relmat(...)` term, and $famn has no relmat route for this " *
               "model. Drop `K` and the `relmat(...)` term, or use a family with a relmat route." *
               _kw_r_note(key, fam)
    elseif key === :A || key === :Ainv || key === :ids
        return "`A`/`Ainv`/`ids` are read by an `animal(...)` term, and $famn has no animal route " *
               "for this model. Drop them and the `animal(...)` term, or use a family with an " *
               "animal route." * _kw_r_note(key, fam)
    elseif key === :coords
        return "`coords` is read by a `spatial(...)` term, and $famn has no spatial route for " *
               "this model. Drop `coords` and the `spatial(...)` term, or use a family with a " *
               "spatial route." * _kw_r_note(key, fam)
    end
    if key === :se
        return "`se` is not a keyword of this route. Use `control = DRMControl(se = false)`, " *
               "which works on every route."
    end
    return "check the spelling, or drop it."
end

# What the refusal-gap map says drmTMB 0.8.1 does for a structured term of these families:
# (family, structured kind) => (cells drmTMB fitted, cells run). The 50 cells per entry are the
# random-effect SD-regression grid of docs/dev-log/parity-crosswalk-evidence-20261107/fuzz-gap-map/
# gap_map.tsv; the grid does NOT cover every model (drmTMB fits a Student `nu ~ phylo(1 | id,
# tree = tree)`, which no cell exercises), so a refusal outside it is "not verified against drmTMB".
const _KW_R_GRID = Dict{Tuple{Symbol,Symbol},Tuple{Int,Int}}(
    (:Student, :phylo) => (0, 50), (:Student, :relmat) => (0, 50),
    (:Student, :animal) => (0, 50), (:Student, :spatial) => (2, 50),
    (:LogNormal, :phylo) => (1, 50), (:LogNormal, :relmat) => (1, 50),
    (:LogNormal, :animal) => (0, 50), (:LogNormal, :spatial) => (0, 50),
    (:Tweedie, :phylo) => (3, 50), (:Tweedie, :relmat) => (3, 50),
    (:Tweedie, :animal) => (0, 50), (:Tweedie, :spatial) => (0, 50),
    (:Binomial, :phylo) => (8, 50), (:Binomial, :relmat) => (6, 50),
    (:Binomial, :animal) => (7, 50), (:Binomial, :spatial) => (6, 50),
)

_kw_structured_kind(key::Symbol) =
    key === :tree ? :phylo : key === :K ? :relmat :
    key in (:A, :Ainv, :ids) ? :animal : key === :coords ? :spatial : nothing

# Only claim what the grid shows; anything else says "not verified against drmTMB".
function _kw_r_note(key::Symbol, fam)
    kind = _kw_structured_kind(key)
    fn = nameof(typeof(fam))
    (kind === nothing || !haskey(_KW_R_GRID, (fn, kind))) &&
        return " Not verified against drmTMB for this model."
    fits, cells = _KW_R_GRID[(fn, kind)]
    what = "`$(kind)(...)` $(fn) cells of the refusal-gap map (a random-effect SD-regression grid)"
    return fits == 0 ?
        " drmTMB 0.8.1 refused all $cells $what; the grid does not cover every model " *
        "(drmTMB does fit a Student `nu ~ phylo(1 | id, tree = tree)`), so for other models " *
        "this is not verified against drmTMB." :
        " drmTMB 0.8.1 fitted $fits of the $cells $what, so for those models this is a gap " *
        "on the Julia side; for other models it is not verified against drmTMB."
end

function _refuse_unsupported_kwargs(unknown, f, fam, accepted)
    keys_ = collect(Symbol, unknown)
    famn = string(nameof(typeof(fam)))
    kind = f isa DrmFormula ? "" : " (multi-response route)"
    accepted = collect(Symbol, accepted)
    acc = join(("`$k`" for k in sort(accepted)), ", ")
    lines = [string("`", k, "`: ", _kw_advice(k, fam, accepted)) for k in keys_]
    throw(ArgumentError("drm: the $famn route$kind does not accept the keyword" *
        (length(keys_) > 1 ? "s " : " ") * join(("`$k`" for k in keys_), ", ") *
        ". It was refused before any fitting, not run with the keyword ignored. " *
        join(lines, " ") * " Keywords this route accepts: $acc."))
end

# Keywords any method of `fn` accepts for `(f, fam)` (without the splat marker).
function _accepted_kwargs(fn, f, fam)
    ks = Symbol[]
    for m in methods(fn, Tuple{typeof(f),typeof(fam)})
        for k in Base.kwarg_decl(m)
            endswith(string(k), "...") || push!(ks, k)
        end
    end
    # `control` and `weights` are inserted into every public `drm` method by `@drm_entry`;
    # an inner fitter's own signature does not list them, so read them off the public method.
    for m in methods(drm, Tuple{typeof(f),typeof(fam)})
        for k in Base.kwarg_decl(m)
            k in (:control, :weights) && push!(ks, k)
        end
    end
    return unique!(ks)
end

# Re-raise a keyword `MethodError` from a route's fitter as the refusal above. Anything
# else (another MethodError, a different function, a keyword the fitter does declare) is
# left alone.
function _refuse_kwarg_methoderror(e::MethodError, f, fam)
    e.f === Core.kwcall && length(e.args) >= 4 || return nothing
    kw, fn, pf, pfam = e.args[1], e.args[2], e.args[3], e.args[4]
    (kw isa NamedTuple && fn isa Function && pf === f && typeof(pfam) === typeof(fam)) || return nothing
    nm = string(nameof(fn))
    (nm == "drm" || startswith(nm, "_drm_")) || return nothing
    accepted = _accepted_kwargs(fn, f, fam)
    isempty(accepted) && return nothing
    unknown = [k for k in keys(kw) if !(k in accepted)]
    isempty(unknown) && return nothing
    _refuse_unsupported_kwargs(unknown, f, fam, accepted)
end

# Run a `drm` method body under the control scope. Called by `@drm_entry`.
function _drm_run_controlled_inner(body::Function, f, fam, control, weights)
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
                          false, false, false, Float64[], 0, NamedTuple[], :none)
    _ctl_preflight(scope, f, fam)
    fit = task_local_storage(body, _CTL_KEY, scope)
    return _ctl_finish(fit, scope, fam)
end

function _drm_run_controlled(body::Function, f, fam, control, weights)
    try
        return _drm_run_controlled_inner(body, f, fam, control, weights)
    catch e
        e isa MethodError && _refuse_kwarg_methoderror(e, f, fam)
        rethrow()
    end
end

function _ctl_finish(fit, scope::_ControlScope, fam)
    c = scope.control
    route = "$(typeof(fam)) route"
    if _ctl_optim_requested(c) && !scope.used_optim
        throw(ArgumentError("drm: this $route does not honour `control = DRMControl(maxit, " *
            "start, multi_start, optimizer)` yet (#948, #994); the fit was not returned. Drop those " *
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
    selected_run = scope.kept > 0 ? scope.kept : length(scope.optimizer_runs)
    actual = isempty(scope.optimizer_runs) ? :unrecorded : scope.optimizer_runs[selected_run].method
    report = DRMControlReport(_ctl_control_snapshot(c), copy(lls), scope.kept, agree, se,
                              scope.weights === nothing ? nothing : copy(scope.weights),
                              c.optimizer, actual, Tuple(scope.optimizer_runs),
                              scope.selected_optimizer_stage, _ctl_native_controls(c))
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

Run the outer optimiser of a fixed-effect route. `run(x0, options, method)` must call
`Optim.optimize` from `x0` with `options`. Without an active `maxit` / `start` /
`multi_start` or explicit selector, this uses the route's original LBFGS
method and `Optim.Options(g_tol = g_tol)`.
"""
function _ctl_optimize(run::Function, θ0::AbstractVector; g_tol::Real)
    s = _active_control_scope()
    if s === nothing || !_ctl_optim_requested(s.control)
        res = run(θ0, Optim.Options(g_tol = g_tol), Optim.LBFGS())
        if s !== nothing
            push!(s.optimizer_runs, _ctl_run_record(:lbfgs, :primary, 1, θ0, res, g_tol, nothing))
            s.selected_optimizer_stage = :primary
        end
        return res
    end
    c = s.control
    s.used_optim = true
    selected = c.optimizer === :bfgs ? Optim.BFGS() : Optim.LBFGS()
    method = c.optimizer === :bfgs ? :bfgs : :lbfgs
    x0 = Vector{Float64}(θ0)
    if c.start !== nothing
        length(c.start) == length(x0) ||
            throw(ArgumentError("DRMControl: `start` has $(length(c.start)) values but this " *
                "model has $(length(x0)) parameters (in `coef(fit)` order; see " *
                "`fit.coefnames` of a default fit)"))
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
            run(xk, opts, selected)
        catch e
            (e isa InterruptException || c.multi_start == 1) && rethrow()
            nothing
        end
        if res === nothing
            push!(s.optimizer_runs, (method = method, stage = :start, start_index = k,
                                     start = Tuple(xk), iterations = missing,
                                     converged = false, objective = NaN, gradient_inf = missing,
                                     g_tol = Float64(g_tol), maxit = c.maxit, status = :error))
            continue
        end
        push!(s.optimizer_runs, _ctl_run_record(method, :start, k, xk, res, g_tol, c.maxit))
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
    s.selected_optimizer_stage = :start
    return best
end

_ctl_run_record(method, stage, k, start, res, g_tol, maxit) =
    (method = method, stage = stage, start_index = k, start = Tuple(Float64.(start)),
     iterations = Optim.iterations(res), converged = Optim.converged(res),
     objective = Optim.minimum(res), gradient_inf = Optim.g_residual(res),
     g_tol = Float64(g_tol), maxit = maxit,
     status = isfinite(Optim.minimum(res)) && all(isfinite, Optim.minimizer(res)) ?
              :finite : :nonfinite)

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
        "the model uses $n rows. Give one weight per row of `data`."))
    s.used_weights = true
    return w
end

# `_ctl_weights`, but `nothing` (not `ones(n)`) when the user gave no weights, for the
# routes whose unweighted arithmetic must stay byte-for-byte what it was.
function _ctl_weights_or_nothing(n::Integer)
    s = _active_control_scope()
    (s === nothing || s.weights === nothing) && return nothing
    return _ctl_weights(n)
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
    # A method with no `kwargs...` of its own gets one, so an unsupported keyword
    # reaches `_refuse_unsupported_kwargs` instead of failing dispatch with a MethodError.
    extra_check = nothing
    if splat === nothing
        accepted = Tuple(vcat(names, [:control, :weights]))
        push!(params.args, Expr(:..., :_drm_extra_kw))
        extra_check = :(isempty(_drm_extra_kw) ||
            _refuse_unsupported_kwargs(keys(_drm_extra_kw), $fname, $famname, $accepted))
    end
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
    # `data` is a named keyword on some routes and travels inside `kwargs...` on others.
    record = if :data in names
        :(_record_factor_levels!($fname, data))
    elseif splat !== nothing
        # `insert!` above moved the splat, so find it again.
        kwsplat = params.args[findfirst(a -> a isa Expr && a.head === :..., params.args)].args[1]
        :(haskey($kwsplat, :data) && _record_factor_levels!($fname, $kwsplat[:data]))
    else
        # the extra-keyword splat added above is named `_drm_extra_kw`
        :(haskey(_drm_extra_kw, :data) && _record_factor_levels!($fname, _drm_extra_kw[:data]))
    end
    # The fixed-effect design rank check (#1110) for every route but the Gaussian one, which
    # makes its own call; `_apply_design_rank` flags the finished fit when the check found an alias.
    precheck = if :data in names
        :(_precheck_design_rank($fname, $famname, data))
    elseif splat !== nothing
        kwsplat2 = params.args[findfirst(a -> a isa Expr && a.head === :..., params.args)].args[1]
        :(haskey($kwsplat2, :data) && _precheck_design_rank($fname, $famname, $kwsplat2[:data]))
    else
        :(haskey(_drm_extra_kw, :data) && _precheck_design_rank($fname, $famname, _drm_extra_kw[:data]))
    end
    # Each fit works on its own copy of the formula bundle (`_fit_formula`): the schema cache, the
    # factor levels and the `poly` bases written at fit time belong to the fit, never to the
    # user's `bf(...)` object, so a formula reused on other data (a second fit, a cross-validation
    # loop, two threads) behaves like a fresh one.
    fdef.args[2] = quote
        $extra_check
        $pre
        let $fname = _fit_formula($fname)
            $record
            $precheck
            return _apply_design_rank(task_local_storage(_FIT_FORMULA_KEY, $fname) do
                _drm_run_controlled($fname, $famname, control, weights) do
                    $body
                end
            end)
        end
    end
    return esc(:(Base.@__doc__ $fdef))
end
