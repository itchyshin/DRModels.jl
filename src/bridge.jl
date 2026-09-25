# bridge.jl — primitive R-facing boundary for `drmTMB(..., engine = "julia")`.
#
# The public R glue lives in the drmTMB repository. This file keeps the Julia
# side deliberately boring for JuliaCall: strings, column tables, plain arrays,
# and dictionaries cross the boundary; DRModels.jl objects stay on the Julia side.

import StatsModels
using Printf: @sprintf

# `nu` is deliberately NOT here (#1090): univariate Student owns a keyed `nu`
# formula too, so it cannot discriminate bivariate — mu1/mu2/sigma1/sigma2/
# rho12 do. The bivariate branch still threads a keyed `nu` through to
# biv_student when mu1/mu2 route it there; the univariate keyed branch already
# lists `:nu` in its parameter order.
const _BRIDGE_BIVARIATE_KEYS = Set((:mu1, :mu2, :sigma1, :sigma2, :rho12))
const _BRIDGE_TREE_CACHE = Dict{UInt64,Tuple{String,Any}}()
const _BRIDGE_TREE_CACHE_MAX = 4

"""
    drm_bridge(; formula, family, data, tree = nothing, K = nothing,
               A = nothing, coords = nothing, newdata = nothing,
               options = Dict())

Fit a DRModels.jl model through a marshalling-friendly boundary for R callers.
`formula` may be a semicolon-separated string such as
`"y ~ x; sigma ~ x"` or a dictionary / named tuple whose values are formula
strings. `family` is a string such as `"gaussian"`, `"student"`, `"nbinom2"`,
or `"biv_gaussian"`. `data` is a column table, dictionary, or named tuple.

`formula` accepts R syntax beyond plain `@formula`: `:` interactions, `*`
crossing, `- 1`/general `- term` removal, `(...)^k` crossing, and
`scale(x)`/`I(expr)`/`factor(x)`/`poly(x, k)` (materialised into real columns —
`I(...)` only over a safe `+ - * / ^` grammar, never arbitrary code).
`poly(x, k)` is R's ORTHOGONAL basis (`raw = FALSE`, the default) and expands to
`k` columns; only `poly(x, k)` itself is accepted — `raw = TRUE` is spelled
`I(x^k)`, and an explicit `coefs =` or the multivariate `poly(x, y, degree)`
are rejected rather than approximated (#492). These materialised columns are NOT
(yet) reconstructed for `newdata` — a formula using them combined with `newdata`
fails loudly (missing column) rather than silently mismodelling; see #467.

The return value is a `Dict{String,Any}` made of primitive R-reconstructable
pieces: named coefficients, covariance matrix, likelihood summaries, fitted
values, residuals, scales, and residual correlations when present.

It also carries what drmTMB's post-fit surface consumes: `"dpars"`, the
per-observation distributional parameters on the response scale keyed by dpar
name, plus `"trials"` for binomial-type families. Pass `newdata` (a column
table) to add `"dpars_newdata"`, the same parameters evaluated on fresh rows —
what `fitted_distribution(object, newdata = ...)` needs.
"""
function drm_bridge(; formula, family::AbstractString, data, tree = nothing,
        K = nothing, A = nothing, coords = nothing, newdata = nothing,
        options = Dict{String,Any}())
    dat = _bridge_data(data)
    bundle, dat, labels = _bridge_formula(formula, family, dat; labels = true)
    fam = _bridge_family(family)
    opts = _bridge_options(options)
    coef_labels = get(opts, :coef_labels, nothing)
    fit = _bridge_fit(bundle, fam, dat; tree = tree, K = K, A = A,
                      coords = coords, options = opts)
    return _bridge_flatten(fit; family = String(family), newdata = newdata,
                           labels = labels, coef_labels = coef_labels)
end

"""
    drm_bridge_objective_at(formula, family, data, tree, options = Dict();
                            beta, Lambda, rho12)

The bivariate q=4 phylogenetic REML counterpart to [`reml_objective_at`](@ref)
(#575) reached through the SAME marshalling-friendly boundary [`drm_bridge`](@ref)
uses — one SUPPORTED entry point for the drmTMB R shim
(`drm_julia_reml_objective_at()`, `R/julia-bridge.R`), replacing its previous
dependency on five private DRModels.jl names (`_bridge_data`, `_bridge_formula`,
`_bivariate_q4_marker`, `_design`, `_phylo_species_index`) reached by qualified
name. `formula`, `family`, `data`, `tree`, `options` are exactly the payload
`drm_bridge` takes for a bivariate q=4 phylogenetic model (a formula with
`phylo(...)` markers shared by `mu1`, `mu2`, `sigma1`, `sigma2`); `options` is
accepted for positional parity with that payload but is not otherwise used —
this is a read-only diagnostic, not a fitting route.

`beta`, `Lambda`, `rho12` are the outer evaluation POINT, in the SAME
(fixed-effect, among-axis-covariance, residual-correlation) parameterisation
`drm_bridge`'s q4 phylo REML fit reports: `beta` is a `NamedTuple` or `Dict`
with `mu1`/`mu2`/`sigma1`/`sigma2` numeric vectors (inner-Newton warm starts
for the profiled-out fixed effects — `reml_objective_at` reprofiles them at
`phi` regardless of the warm start supplied), `Lambda` is the 4×4 symmetric
among-axis covariance matrix (axis order mu1, mu2, sigma1, sigma2), and
`rho12` is the residual correlation. Internally: `Lambda`/`rho12` are packed
into DRModels.jl's own `phi = (beta_rho, lc)` via `pack_phi` and
[`reml_objective_at`](@ref) evaluates the q=4 REML objective there.

Returns a `Dict{String,Any}` with `"objective"` and `"reml_loglik"` (the
normalised Patterson–Thompson restricted log-likelihood `reml_objective_at`
reports — the two keys carry the same value; `"objective"` is the
route-agnostic name, `"reml_loglik"` names the DRModels.jl convention explicitly),
`"raw_reml_ll"` (the pre-normalisation value), `"converged_inner"` (the inner
conditional-Newton alternation's own convergence flag — a barrier hit
surfaces as `-Inf`/`false` rather than an error), and `"contract" =>
"bridge_objective_at_v1"` so R callers can assert the return shape.

This is a DIAGNOSTIC: it selects nothing, fits nothing, and promotes no
capability-ledger row. See #575 for the cross-engine mode-finder-vs-
objective-translation question it exists to answer.
"""
function drm_bridge_objective_at(formula, family::AbstractString, data, tree,
        options = Dict{String,Any}(); beta, Lambda, rho12)
    tree === nothing && throw(ArgumentError(
        "drm_bridge_objective_at: `tree` is required for the bivariate q=4 " *
        "phylogenetic REML objective-at diagnostic"))
    dat = _bridge_data(data)
    bundle, dat2, _ = _bridge_formula(formula, family, dat; labels = true)
    bundle isa BivariateDrmFormula || throw(ArgumentError(
        "drm_bridge_objective_at: only the bivariate q=4 phylogenetic REML " *
        "route is supported (got a univariate formula)"))
    rhs = Dict(bundle.forms)
    fixed, marker = _bivariate_q4_marker(rhs)
    (marker !== nothing && marker[1] === :phylo_q4) || throw(ArgumentError(
        "drm_bridge_objective_at: only the bivariate q=4 phylogenetic REML " *
        "route is supported (formula has no shared `phylo(...)` term on " *
        "mu1, mu2, sigma1, and sigma2)"))
    grp = marker[2]
    phy = _as_augmented_phy(_bridge_tree(tree))
    species = _phylo_species_index(phy, getproperty(dat2, grp))

    y1, X1, _ = _design(bundle.response1, fixed[:mu1], dat2)
    y2, X2, _ = _design(bundle.response2, fixed[:mu2], dat2)
    _, Xs1, _ = _design(bundle.response1, fixed[:sigma1], dat2)
    _, Xs2, _ = _design(bundle.response1, fixed[:sigma2], dat2)
    _, Xr, _  = _design(bundle.response1, fixed[:rho12], dat2)
    prob, Q_cond = make_problem(phy, y1, y2, X1, X2, Xs1, Xs2, Xr; species = species)

    Lam = Matrix{Float64}(Lambda)
    size(Lam) == (4, 4) || throw(ArgumentError(
        "drm_bridge_objective_at: `Lambda` must be a 4x4 symmetric numeric " *
        "matrix (the phylo q4 among-axis covariance, axis order mu1, mu2, " *
        "sigma1, sigma2); got size $(size(Lam))"))
    rho_vec = [Float64(rho12)]
    phi = pack_phi(prob, rho_vec, Lam)

    beta_mu1 = _bridge_objective_at_beta_field(beta, :mu1, size(X1, 2))
    beta_mu2 = _bridge_objective_at_beta_field(beta, :mu2, size(X2, 2))
    beta_s1  = _bridge_objective_at_beta_field(beta, :sigma1, size(Xs1, 2))
    beta_s2  = _bridge_objective_at_beta_field(beta, :sigma2, size(Xs2, 2))
    beta0 = (mu1 = beta_mu1, mu2 = beta_mu2, s1 = beta_s1, s2 = beta_s2, rho = rho_vec)

    rr = reml_objective_at(prob, Q_cond, phi; beta0 = beta0, n_newton = 60)
    return Dict{String,Any}(
        "objective" => rr.reml_loglik,
        "reml_loglik" => rr.reml_loglik,
        "raw_reml_ll" => rr.raw_reml_ll,
        "converged_inner" => rr.converged,
        "contract" => "bridge_objective_at_v1",
    )
end

# Fetch `beta[field]` from a NamedTuple or Dict (String or Symbol keyed), as a
# `Vector{Float64}`, and check it against the design width `reml_objective_at`
# expects — naming both the field and the expected/actual lengths so a
# mismatched R-side `beta` fails loudly rather than silently misaligning
# coefficients to the wrong design columns.
function _bridge_objective_at_beta_field(beta, field::Symbol, expected_len::Int)
    value = if beta isa AbstractDict
        haskey(beta, field) ? beta[field] :
            haskey(beta, String(field)) ? beta[String(field)] :
            throw(ArgumentError("drm_bridge_objective_at: `beta` is missing field `$(field)`"))
    elseif beta isa NamedTuple
        hasproperty(beta, field) ||
            throw(ArgumentError("drm_bridge_objective_at: `beta` is missing field `$(field)`"))
        getproperty(beta, field)
    else
        throw(ArgumentError("drm_bridge_objective_at: `beta` must be a NamedTuple or " *
            "Dict with mu1/mu2/sigma1/sigma2 fields"))
    end
    v = value isa Number ? Float64[value] : Vector{Float64}(vec(value))
    length(v) == expected_len || throw(ArgumentError(
        "drm_bridge_objective_at: `beta[$(field)]` must have length $(expected_len) " *
        "(the $(field) design width); got length $(length(v))"))
    return v
end

"""
    drm_bridge_q2_phylo(; Y, X, species, tree, options = Dict())

Private diagnostic boundary for the restricted q2 phylogenetic coevolution
point export. This intentionally bypasses the public formula bridge because the
general-q coevolution model is diagonal-residual Gaussian evidence, not the full
bivariate `rho12` q2 route. The return payload is the same primitive dictionary
shape consumed by the row-contract tests.
"""
function drm_bridge_q2_phylo(; Y, X, species, tree, options = Dict{String,Any}())
    y = Matrix{Float64}(Y)
    x = Matrix{Float64}(X)
    sp = Int.(vec(species))
    size(y, 2) == 2 ||
        throw(ArgumentError("drm_bridge_q2_phylo: `Y` must have exactly two columns"))
    size(x, 1) == size(y, 1) ||
        throw(ArgumentError("drm_bridge_q2_phylo: `X` and `Y` must have the same number of rows"))
    length(sp) == size(y, 1) ||
        throw(ArgumentError("drm_bridge_q2_phylo: `species` length must match the number of rows in `Y`"))
    phy = _bridge_tree(tree)
    all(1 .<= sp .<= phy.n_leaves) ||
        throw(ArgumentError("drm_bridge_q2_phylo: `species` must contain 1-based tip indices"))
    opts = _bridge_options(options)
    prob, Q_cond = make_coevo_problem(phy, y, x; species = sp)
    fit = fit_coevolution(
        prob,
        Q_cond;
        iterations = Int(get(opts, :iterations, 180)),
        g_tol = Float64(get(opts, :g_tol, 1e-4)),
        fd_h = Float64(get(opts, :fd_h, 1e-6)),
    )
    return _bridge_q2_point_export(fit; family = "biv_gaussian",
                                   structured_type = "phylo")
end

"""
    drm_bridge_q2_known_precision(; Y, X, group, Q,
                                  structured_type = "relmat",
                                  precision_source = nothing,
                                  options = Dict())

Private diagnostic boundary for a restricted q2 known-precision provider payload.
This consumes `Q` as a precision matrix directly through
`make_coevo_problem_from_precision`; it does not invert or relabel `Q` as a
covariance matrix, and it does not imply formula, slope, REML, or interval
support. Provider identity is deliberately narrow: `structured_type = "relmat"`
records `precision_source = "Q"`, while `structured_type = "animal"` records
`precision_source = "Ainv"`.
"""
function drm_bridge_q2_known_precision(; Y, X, group, Q,
        structured_type = "relmat", precision_source = nothing,
        options = Dict{String,Any}())
    y = Matrix{Float64}(Y)
    x = Matrix{Float64}(X)
    g = Int.(vec(group))
    qmat = Matrix{Float64}(Q)
    st, source = _bridge_q2_known_precision_provider(structured_type,
                                                     precision_source)
    size(y, 2) == 2 ||
        throw(ArgumentError("drm_bridge_q2_known_precision: `Y` must have exactly two columns"))
    size(x, 1) == size(y, 1) ||
        throw(ArgumentError("drm_bridge_q2_known_precision: `X` and `Y` must have the same number of rows"))
    length(g) == size(y, 1) ||
        throw(ArgumentError("drm_bridge_q2_known_precision: `group` length must match the number of rows in `Y`"))
    opts = _bridge_options(options)
    prob, Q_cond = make_coevo_problem_from_precision(qmat, y, x; group = g)
    fit = fit_coevolution_q2_residual(
        prob,
        Q_cond;
        iterations = Int(get(opts, :iterations, 180)),
        g_tol = Float64(get(opts, :g_tol, 1e-4)),
        fd_h = Float64(get(opts, :fd_h, 1e-6)),
    )
    out = _bridge_q2_point_export(
        fit;
        family = "biv_gaussian",
        structured_type = st,
    )
    out["input_scale"] = "precision"
    out["precision_source"] = source
    out["precision_matrix"] = qmat
    out["claim_boundary"] = join((
        "Direct q2 $st known-precision point export only for complete-response",
        "exact-Gaussian ML fixtures; `$source` is consumed as a precision",
        "matrix without implicit precision-to-covariance conversion. No",
        "R-via-Julia formula support, structured slope support, broad q2",
        "bridge support, q2 REML, q4, AI-REML, interval reliability, or",
        "interval coverage is promoted.",
    ), " ")
    return out
end

"""
    drm_bridge_inference(; formula, family, data, tree = nothing,
                         K = nothing, A = nothing, coords = nothing,
                         options = Dict(), method = "profile",
                         level = 0.95, B = 199, seed = nothing,
                         threads = false, parm = nothing)

Run a narrow inference primitive for the R bridge.

With `parm = nothing` (the default) this targets the Gaussian phylogenetic SD
block (`param = :resd` / `:resd_mu` / `:resd_sigma`). Pass `parm = "sd:mu"`
or `"sd:sigma"` to select the coupled location- or scale-axis row explicitly,
or `"sd:resd"` / `"sd:resd_mu"` / `"sd:resd_sigma"` when an R bridge caller
must preserve the fitted parameter block exactly.

Pass `parm = "fixef:<dpar>:<coef>"` (e.g. `"fixef:mu:x"`) to instead profile
or bootstrap a single ordinary fixed-effect coefficient, on its link scale —
the same primitive `DRModels.profile_result` / `DRModels.bootstrap_result` calls the R
bridge previously had to reach by calling DRModels.jl's underscore-prefixed
marshalling internals directly (see #475); this kwarg is the supported route
that replaces that qualified-internal call. Returns the same payload shape
either way. For an explicit structured fixed-effect target, the supplied
covariance provider (`tree`, `K`, `A`, or `coords`) is reused for the initial
fit, marginal simulation, and every bootstrap refit.
"""
function drm_bridge_inference(; formula, family::AbstractString, data,
        tree = nothing, K = nothing, A = nothing, coords = nothing,
        options = Dict{String,Any}(), method::AbstractString = "profile",
        level::Real = 0.95, B::Integer = 199, seed = nothing,
        threads::Bool = false, parm = nothing)
    dat = _bridge_data(data)
    bundle, dat, labels = _bridge_formula(formula, family, dat; labels = true)
    fam = _bridge_family(family)
    opts = _bridge_options(options)
    bridge_method = lowercase(strip(String(method)))
    is_biv = bundle isa BivariateDrmFormula
    target = parm === nothing ? nothing : _bridge_parse_inference_parm(parm)
    # The univariate σ-phylo location-scale route precomputes its boundary-aware
    # profile CIs into the fit (it has no re-optimisable objective), so request them
    # at fit time for the profile method. The bivariate q=4 route has no such flag
    # (its drm method rejects `profile_ci`) — skip it there. An explicit fixed-effect
    # `parm` target profiles the fit's own re-optimisable objective and never reads
    # that precomputed stash, so it is skipped there too — matching the R bridge's
    # previous qualified-internal call, which never set this option either.
    (!is_biv && (target === nothing || target.kind === :sd) && bridge_method == "profile") &&
        (opts[:profile_ci] = true)
    tree_obj = tree === nothing ? nothing : _bridge_tree(tree)
    fit = _bridge_fit(bundle, fam, dat; tree = tree_obj, K = K,
                      A = A, coords = coords, options = opts)
    rawtarget = target === nothing || target.kind !== :fixef ? nothing :
        _bridge_raw_fixef_target(fit, labels, target)

    # Bivariate q=4 phylogenetic fit: the uncertainty target is the four among-axis
    # SDs sqrt.(diag(Σ_a)), not a single SD row. The boundary makes the q4 profile
    # singular, so the route is the parametric bootstrap; return all four rows.
    # Skipped when an explicit fixed-effect `parm` target was given (below).
    if target === nothing && is_biv && fit.ranef isa NamedTuple && haskey(fit.ranef, :Sigma_a)
        return _bridge_bivariate_inference(fit, dat, bridge_method;
                                           B = B, level = level, seed = seed)
    end

    if bridge_method == "profile"
        # Preserve the implicit SD target set, but an explicit fixed-effect
        # request must profile only that coefficient, not its entire block.
        profile_parm = target === nothing ? [:resd_sigma, :resd, :resd_mu] :
            target.kind === :fixef ? rawtarget.param => rawtarget.coef : target.param
        result = profile_result(fit; level = level, threads = threads, parm = profile_parm)
        row = target === nothing ? _bridge_pick_sd_row(result.ci) :
            target.kind === :fixef ? _bridge_pick_fixef_row(result.ci, rawtarget) :
            _bridge_pick_sd_row(result.ci, target.param)
        outcome = _bridge_profile_outcome(result, row)
        target !== nothing && target.kind === :fixef && (row = merge(row, (coef = target.coef,)))
        # DRModels.jl#631: `profile_failed` means the endpoint search could not certify a
        # root, and the row carries the ±Inf placeholder for the failed arm. R reads
        # this payload straight into `confint()`'s `lower`/`upper` columns, where an
        # infinite bound is indistinguishable from a real confidence limit (the
        # `conf.status` column that says otherwise is columns away, and easy to miss).
        # Raise instead: an R caller sees an error, never a bound it can quote.
        # A genuinely UNBOUNDED profile (the likelihood never crosses the LR
        # threshold in the searched range) keeps status "profile" and is unaffected.
        outcome.status == "profile_failed" && throw(ArgumentError(
            "drm_bridge_inference: profile endpoint search did not converge for " *
            "`$(row.param):$(row.coef)` — $(outcome.message). A non-converged " *
            "endpoint is NOT a confidence limit, so it is refused here rather " *
            "than returned as an infinite bound. Use `method = \"wald\"` or " *
            "`method = \"bootstrap\"` for this target."))
        return _bridge_inference_flatten(
            row;
            method = "profile",
            status = outcome.status,
            attempted = result.attempted,
            used = result.used,
            failed = result.failed,
            elapsed = result.elapsed,
            threaded = result.threaded,
            worker_threads = result.worker_threads,
            julia_threads = result.julia_threads,
            blas_threads = result.blas_threads,
            message = outcome.message,
        )
    elseif bridge_method == "bootstrap"
        rng = seed === nothing ? Random.default_rng() :
              Random.MersenneTwister(Int(seed))
        result = if target !== nothing && !(fit isa DrmFit{<:Gaussian})
            # The generic method accepts the covariance provider but not the
            # Gaussian-specific algorithm/g_tol controls. Preserve the original
            # tree for the marginal sampler and every non-Gaussian refit.
            bootstrap_result(
                fit; data = dat, B = Int(B), level = level, rng = rng,
                tree = tree_obj, K = K, A = A, coords = coords, threads = threads,
                failures = :skip, check_converged = true,
            )
        else
            bootstrap_result(
                fit; data = dat, B = Int(B), level = level, rng = rng,
                tree = tree_obj, K = K, A = A, coords = coords,
                threads = threads, failures = :skip,
                # #459: a percentile CI must not be computed over refits that did not
                # converge. This was `false`, which was harmless only while the
                # simulator was conditional -- every replicate then re-used the fitted
                # BLUPs, so every refit converged trivially and the interval was
                # degenerate anyway. With a correct marginal simulator some replicates
                # are genuinely hard, and admitting their diverged estimates put the
                # upper percentile at 179 against a point estimate of 1.30.
                # `failures = :skip` drops them and `used`/`failed` report how many.
                check_converged = true,
                algorithm = Symbol(get(opts, :algorithm, :auto)),
                g_tol = Float64(get(opts, :g_tol, 1e-8)),
            )
        end
        row = target === nothing ? _bridge_pick_sd_row(result.summary) :
            target.kind === :fixef ? _bridge_pick_fixef_row(result.summary, rawtarget) :
            _bridge_pick_sd_row(result.summary, target.param)
        target !== nothing && target.kind === :fixef && (row = merge(row, (coef = target.coef,)))
        return _bridge_inference_flatten(
            row;
            method = "bootstrap",
            status = result.used >= 2 ? "bootstrap" : "bootstrap_unavailable",
            attempted = result.attempted,
            used = result.used,
            failed = result.failed,
            elapsed = result.elapsed,
            threaded = result.threaded,
            worker_threads = result.worker_threads,
            julia_threads = result.julia_threads,
            blas_threads = result.blas_threads,
            message = "$(result.used)/$(result.attempted) successful refits",
        )
    end
    throw(ArgumentError("drm_bridge_inference: unsupported method `$method`"))
end

# Parse the R bridge's `"fixef:<dpar>:<coef>"` target string (e.g. `"fixef:mu:x"`)
# into the `(param, coef)` pair `_ci_param_selected`/`_bridge_pick_fixef_row` need.
# `limit = 3` keeps a `:`-bearing coefficient name (e.g. an interaction `"x:z"`)
# intact in the third part rather than splitting it further.
function _bridge_parse_fixef_parm(parm)
    parm isa AbstractString || throw(ArgumentError(
        "drm_bridge_inference: `parm` must be a string of the form " *
        "`\"fixef:<dpar>:<coef>\"` (e.g. `\"fixef:mu:x\"`)"))
    parts = split(String(parm), ':'; limit = 3)
    (length(parts) == 3 && lowercase(parts[1]) == "fixef") || throw(ArgumentError(
        "drm_bridge_inference: unsupported `parm` target `$(repr(parm))`; expected " *
        "`\"fixef:<dpar>:<coef>\"` (e.g. `\"fixef:mu:x\"`)"))
    return (param = Symbol(parts[2]), coef = String(parts[3]))
end

# Parse the bridge's explicit fixed-effect or phylogenetic-SD target.
function _bridge_parse_inference_parm(parm)
    parm isa AbstractString || throw(ArgumentError(
        "drm_bridge_inference: `parm` must be a `fixef:<dpar>:<coef>` or `sd:<dpar>` string"))
    text = String(parm)
    startswith(lowercase(text), "fixef:") && return merge((kind = :fixef,), _bridge_parse_fixef_parm(text))
    parts = split(text, ':'; limit = 2)
    (length(parts) == 2 && lowercase(parts[1]) == "sd") || throw(ArgumentError(
        "drm_bridge_inference: unsupported `parm` target `$(repr(parm))`; expected " *
        "`fixef:<dpar>:<coef>` or `sd:mu` / `sd:sigma`"))
    target = Symbol(parts[2])
    target === :mu && return (kind = :sd, param = :resd_mu)
    target === :sigma && return (kind = :sd, param = :resd_sigma)
    target in (:resd, :resd_mu, :resd_sigma) && return (kind = :sd, param = target)
    throw(ArgumentError(
        "drm_bridge_inference: unsupported SD target `$(repr(parm))`; expected `sd:mu`, " *
        "`sd:sigma`, `sd:resd`, `sd:resd_mu`, or `sd:resd_sigma`"))
end

_bridge_parse_sd_parm(parm) = _bridge_parse_inference_parm(parm).param

# Pick the single fixed-effect row named by an explicit `parm` target. There is NO
# silent fall-back: if the (param, coef) pair is not present the row would be some
# other coefficient mislabelled as the requested one, so throw an explicit error
# naming what WAS available (mirrors `_bridge_pick_sd_row`'s discipline).
function _bridge_pick_fixef_row(rows, target)
    for row in rows
        row.param === target.param && row.coef == target.coef && return row
    end
    got = join(("$(r.param):$(r.coef)" for r in rows), ", ")
    throw(ArgumentError("drm_bridge_inference: no row for target " *
        "`fixef:$(target.param):$(target.coef)` in the result; got [$(got)]."))
end

# The complete set of `options` keys `_bridge_fit` knows how to forward, PLUS
# any key `drm_bridge` (or a sibling entry point) legitimately reads off the
# SAME `options` dict before it reaches `_bridge_fit` — `:coef_labels`
# (design 258 §7.1) is extracted and consumed by `drm_bridge` itself (the
# public coefficient-name echo), never forwarded as a `drm(...)` kwarg, but it
# is still a real key on the dict `_bridge_fit` sees, so it must not trip the
# fail-closed check below. Kept as a literal set (not derived from `kwargs`
# below) so that check is a simple, auditable diff against this list rather
# than depending on which branches happened to fire for a given fit.
const _BRIDGE_KNOWN_OPTION_KEYS = Set((
    :g_tol, :algorithm, :method, :marginal, :se, :profile_ci, :phylo_coupled, :sparse,
    :q4_g_tol, :q4_iterations, :q4_n_newton, :q4_vcov, :coef_labels,
))

function _bridge_fit(bundle, fam, data; tree, K, A, coords, options)
    # Fail closed on an unknown option key (#527-adjacent): a typo'd or
    # not-yet-wired key must error loudly and NAME the key, rather than being
    # silently dropped and the caller left believing it took effect.
    unknown = setdiff(keys(options), _BRIDGE_KNOWN_OPTION_KEYS)
    isempty(unknown) || throw(ArgumentError(
        "drm_bridge: unknown option key(s) `$(join(sort(String.(unknown)), "`, `"))`; " *
        "known keys are: $(join(sort(String.(collect(_BRIDGE_KNOWN_OPTION_KEYS))), ", "))."))
    kwargs = Dict{Symbol,Any}()
    tree !== nothing && (kwargs[:tree] = _bridge_tree(tree))
    K !== nothing && (kwargs[:K] = K)
    A !== nothing && (kwargs[:A] = A)
    coords !== nothing && (kwargs[:coords] = coords)
    if haskey(options, :g_tol)
        kwargs[:g_tol] = Float64(options[:g_tol])
    end
    if haskey(options, :algorithm)
        kwargs[:algorithm] = Symbol(options[:algorithm])
    end
    if haskey(options, :method)
        kwargs[:method] = Symbol(options[:method])
    end
    if haskey(options, :marginal)
        # Arc 2: `"Laplace"` selects the TMB-convention Laplace route on an
        # ordinary `(1 | g)` (ordinary_laplace.jl). Forwarded verbatim: a family
        # without a `marginal` keyword, or a model outside the route, errors.
        kwargs[:marginal] = Symbol(options[:marginal])
    end
    if haskey(options, :se)
        kwargs[:se] = Bool(options[:se])
    end
    if haskey(options, :profile_ci)
        kwargs[:profile_ci] = Bool(options[:profile_ci])
    end
    if haskey(options, :phylo_coupled)
        kwargs[:phylo_coupled] = Bool(options[:phylo_coupled])
    end
    if haskey(options, :sparse)
        # Native `drm(...)` keyword alias for `algorithm = :sparse`/`:sparse_lbfgs`
        # (forces the O(p) sparse whole-tree engine on large phylogenetic LSS
        # fits). Previously reachable natively but silently dropped by the
        # bridge — the exact control an R caller needs on a many-species fit.
        kwargs[:sparse] = Bool(options[:sparse])
    end
    if _bridge_is_bivariate_phylo_q4(bundle, fam, tree) && !haskey(options, :q4_vcov)
        # The bridge's q4 uncertainty route is profile/bootstrap over among-axis
        # SDs. Avoid the auxiliary finite-difference Wald covariance by default:
        # it is expensive at large q4 phylogenetic fits and can fail after a
        # usable fit has been found.
        kwargs[:q4_vcov] = false
    end
    if haskey(options, :q4_g_tol)
        kwargs[:q4_g_tol] = Float64(options[:q4_g_tol])
    end
    if haskey(options, :q4_iterations)
        kwargs[:q4_iterations] = Int(options[:q4_iterations])
    end
    if haskey(options, :q4_n_newton)
        kwargs[:q4_n_newton] = Int(options[:q4_n_newton])
    end
    if haskey(options, :q4_vcov)
        kwargs[:q4_vcov] = Bool(options[:q4_vcov])
    end
    return drm(bundle, fam; data = data, kwargs...)
end

function _bridge_is_bivariate_phylo_q4(bundle, fam, tree)
    return bundle isa BivariateDrmFormula && fam isa Gaussian && tree !== nothing
end

# Pick the variance-component SD row from a profile/bootstrap result for the bridge: prefer the
# σ-phylo location-scale σ-axis SD (:resd_sigma), then the legacy phylo SD block (:resd), then
# the μ-axis SD (:resd_mu). (Routes the bridge inference to the SD that matters for the σ-phylo
# cell Ayumi needs.) There is NO silent fall-back to the first row: if none of the expected SD
# params is present the row would be a fixed-effect coefficient mislabelled as the SD CI on the R
# side, so throw an explicit error naming the params that WERE returned.
function _bridge_pick_sd_row(rows)
    for want in (:resd_sigma, :resd, :resd_mu)
        for row in rows
            row.param === want && return row
        end
    end
    isempty(rows) && throw(ArgumentError("drm_bridge_inference: no SD row in the result"))
    got = join(unique(String(r.param) for r in rows), ", ")
    throw(ArgumentError("drm_bridge_inference: no variance-component SD row " *
        "(:resd_sigma, :resd, or :resd_mu) in the result; got params [$(got)]. " *
        "Refusing to mislabel a fixed-effect row as the SD confidence interval."))
end

function _bridge_pick_sd_row(rows, want::Symbol)
    for row in rows
        row.param === want && return row
    end
    got = join(unique(String(r.param) for r in rows), ", ")
    throw(ArgumentError("drm_bridge_inference: no requested SD row `$want`; got params [$got]"))
end

function _bridge_tree(tree)
    tree isa AbstractString || return tree
    key = hash(String(tree))
    cached = get(_BRIDGE_TREE_CACHE, key, nothing)
    if cached !== nothing && cached[1] == tree
        return cached[2]
    end
    parsed = augmented_phy(tree)
    if length(_BRIDGE_TREE_CACHE) >= _BRIDGE_TREE_CACHE_MAX
        empty!(_BRIDGE_TREE_CACHE)
    end
    _BRIDGE_TREE_CACHE[key] = (String(tree), parsed)
    return parsed
end

function _bridge_options(options)
    options === nothing && return Dict{Symbol,Any}()
    if options isa NamedTuple
        return Dict{Symbol,Any}(Symbol(k) => v for (k, v) in pairs(options))
    elseif options isa AbstractDict
        return Dict{Symbol,Any}(Symbol(String(k)) => v for (k, v) in pairs(options))
    end
    throw(ArgumentError("drm_bridge: `options` must be a dictionary or named tuple"))
end

function _bridge_data(data)
    if data isa NamedTuple
        return NamedTuple(Symbol(k) => _bridge_column(v) for (k, v) in pairs(data))
    elseif data isa AbstractDict
        return NamedTuple(Symbol(String(k)) => _bridge_column(v) for (k, v) in pairs(data))
    end
    return data
end

_bridge_column(v::AbstractVector) = collect(v)
_bridge_column(v) = v

function _bridge_family(family::AbstractString)
    fam = lowercase(strip(String(family)))
    fam in ("gaussian", "normal") && return Gaussian()
    fam in ("biv_gaussian", "gaussian_bivariate", "bivariate_gaussian") && return Gaussian()
    fam in ("student", "student_t", "student-t") && return Student()
    # drmTMB's `skew_normal()` (dpars mu, sigma, nu — the slant rides in the
    # same `nu` slot Student uses). Public moment parameterisation on BOTH
    # sides (mu = E[y], sigma = SD[y], nu = Azzalini's alpha), so the bridged
    # coefficients are the native ones, not a re-parameterisation.
    fam in ("skew_normal", "skewnormal") && return SkewNormal()
    # drmTMB's `biv_student()` — bivariate-ness is a FORMULA property here.
    fam in ("biv_student", "student_bivariate", "bivariate_student") && return Student()
    fam == "poisson" && return Poisson()
    fam in ("nbinom2", "negbinomial2", "negative_binomial_2") && return NegBinomial2()
    fam in ("truncated_nbinom2", "truncated_negbinomial2") && return TruncatedNegBinomial2()
    fam == "beta" && return Beta()
    fam in ("beta_binomial", "betabinomial") && return BetaBinomial()
    fam == "binomial" && return Binomial()
    fam == "gamma" && return Gamma()
    fam == "lognormal" && return LogNormal()
    # drmTMB's `biv_lognormal()`. Bivariate-ness is a property of the FORMULA in
    # DRModels.jl (a `BivariateDrmFormula`), not of the family type — exactly as
    # `biv_gaussian` maps to `Gaussian()` above.
    fam in ("biv_lognormal", "lognormal_bivariate", "bivariate_lognormal") && return LogNormal()
    fam in ("zero_one_beta", "zeroonebeta") && return ZeroOneBeta()
    fam == "tweedie" && return Tweedie()
    fam in ("cumulative_logit", "ordinal") && return CumulativeLogit()
    hint = _bridge_mixture_family_hint(fam)
    hint === nothing ||
        throw(ArgumentError("drm_bridge: unsupported family `$family`. " * hint))
    throw(ArgumentError("drm_bridge: unsupported family `$family`"))
end

"""Actionable refusal text for a zero-inflated / hurdle family SPELLING.

Zero-inflation (`zi`) and hurdle (`hu`) are FORMULA parts here, not families:
the count family stays `poisson` / `nbinom2` and a keyed `zi ~ …` or `hu ~ …`
entry adds the mixture (see `_bridge_formula` above, `test_zi.jl`,
`test_hurdle.jl`). drmTMB spells them the same way -- its `drm_family_type()`
returns `"poisson"` for a zero-inflated Poisson, and `"zi_poisson"` is only its
post-fit `model_type` -- so the R bridge never sends these tags. A caller
writing `drm_bridge` by hand reasonably might, and a bare "unsupported family"
does not tell them the spelling that works.

Deliberately NOT an alias. Mapping `"zi_poisson"` to `Poisson()` would fit a
PLAIN Poisson, with no error, whenever the caller omitted the `zi ~` part --
a silent wrong answer in place of a loud refusal.

Returns `nothing` (so the caller falls through to the plain message) unless the
tag is a recognised mixture prefix on a count family this bridge accepts.
"""
function _bridge_mixture_family_hint(fam::AbstractString)
    m = match(r"^(?:zi|zero_inflated)_(.+)$", fam)
    part = "zi"
    if m === nothing
        m = match(r"^hurdle_(.+)$", fam)
        part = "hu"
    end
    m === nothing && return nothing
    # Only ever point at a COUNT family: `zi`/`hu` are count-mixture parts, and
    # this explicit list (rather than a recursive `_bridge_family` probe) keeps
    # the lookup terminating and the suggestion one this bridge can honour.
    base = m.captures[1]
    base in ("nbinom2", "negbinomial2", "negative_binomial_2") && (base = "nbinom2")
    base in ("poisson", "nbinom2") || return nothing
    return "Zero-inflation (`zi`) and hurdle (`hu`) are formula parts here, not " *
           "families: pass `family = \"$base\"` and add a `$part` entry to " *
           "`formula`, e.g. `Dict(\"mu\" => \"y ~ x\", \"$part\" => \"$part ~ 1\")`."
end

"""Internal, typed provenance for bridge-only public coefficient labels.

`atoms` maps collision-safe materialised Julia symbols to their exact R term
spellings. `function_labels` retains source spelling for every admitted scalar
function term, including redundant parentheses. Neither changes a formula,
matrix, or fit.
"""
struct _BridgeFormulaLabels
    data
    atoms::Dict{Symbol,String}
    # A translated scalar AST can be identical in two distributional parts
    # while the R source deliberately differs only by redundant parentheses.
    # Keep those labels per fitted formula parameter, never in a global map.
    function_labels::Dict{Symbol,Dict{String,String}}
end

function _bridge_formula(formula, family::AbstractString, data; labels::Bool = false)
    ctx = _BridgeXlateCtx(data)
    parts = _bridge_formula_parts(formula)
    parsed = map(p -> _bridge_parse_formula_part(p, ctx), parts)
    any(isnothing, parsed) &&
        throw(ArgumentError("drm_bridge: could not parse formula specification"))

    keyed = Dict{Symbol,Any}()
    keyed_labels = Dict{Symbol,Dict{String,String}}()
    positional = Any[]
    positional_labels = Dict{String,String}[]
    lss = Any[]                # `sd(g) ~ …` / `sd_phylo(s) ~ …` parts (#546):
    lss_labels = Dict{String,String}[]
                               # marker-keyed, so they belong to NEITHER bucket and
                               # must not trip the keyed-vs-positional guard below.
    for item in parsed
        key, (form, function_labels) = item
        if key === nothing
            if form.lhs isa FunctionTerm && (form.lhs.f === sd || form.lhs.f === sd_phylo)
                push!(lss, form)
                push!(lss_labels, function_labels)
            else
                push!(positional, form)
                push!(positional_labels, function_labels)
            end
        else
            keyed[key] = form
            keyed_labels[key] = function_labels
        end
    end

    bundle = if any(k -> k in _BRIDGE_BIVARIATE_KEYS, keys(keyed))
        (isempty(positional) && haskey(keyed, :mu1) && haskey(keyed, :mu2)) ||
            throw(ArgumentError("drm_bridge: bivariate formulas need keyed `mu1` and `mu2` entries"))
        bf(; mu1 = keyed[:mu1], mu2 = keyed[:mu2],
             sigma1 = get(keyed, :sigma1, nothing), sigma2 = get(keyed, :sigma2, nothing),
             nu = get(keyed, :nu, nothing), rho12 = get(keyed, :rho12, nothing))
    elseif !isempty(keyed)
        isempty(positional) ||
            throw(ArgumentError("drm_bridge: do not mix keyed and positional univariate formulas"))
        haskey(keyed, :mu) ||
            throw(ArgumentError("drm_bridge: keyed univariate formulas need a `mu` entry"))
        ordered = Any[keyed[:mu]]
        for p in (:sigma, :nu, :zi, :hu, :zoi, :coi)
            haskey(keyed, p) && push!(ordered, keyed[p])
        end
        append!(ordered, lss)
        for k in keys(keyed)
            k in (:mu, :sigma, :nu, :zi, :hu, :zoi, :coi) ||
                throw(ArgumentError("drm_bridge: unknown univariate formula part `$k`. " *
                    "Supported: mu, sigma, nu, zi, hu, zoi, coi, sd(group), sd_phylo(group)."))
        end
        bf(ordered...)
    else
        isempty(positional) &&
            throw(ArgumentError("drm_bridge: at least one formula is required"))
        bf(positional..., lss...)
    end

    # Bind source provenance after `bf` assigns the canonical parameter keys.
    # Positional and LSS formula parts preserve their established bundle order.
    by_param = Dict{Symbol,Dict{String,String}}()
    positional_i = 1
    lss_i = 1
    for (param, _) in bundle.forms
        if haskey(keyed_labels, param)
            by_param[param] = keyed_labels[param]
        elseif _bridge_lss_form_key(param) !== nothing
            if lss_i <= length(lss_labels)
                by_param[param] = lss_labels[lss_i]
                lss_i += 1
            else
                # Family-internal coordinates such as ordinal cutpoints have
                # no user RHS and therefore no formula-source provenance.
                by_param[param] = Dict{String,String}()
            end
        elseif positional_i <= length(positional_labels)
            by_param[param] = positional_labels[positional_i]
            positional_i += 1
        else
            # Do not invent a label for a non-formula coordinate. The export
            # path keeps it raw/identity unless a family has its own mapping.
            by_param[param] = Dict{String,String}()
        end
    end
    positional_i == length(positional_labels) + 1 || error("drm_bridge: unused positional label provenance")
    lss_i == length(lss_labels) + 1 || error("drm_bridge: unused LSS label provenance")

    augmented = isempty(ctx.extra) ? data : merge(_bridge_ctx_cols!(ctx), NamedTuple(ctx.extra))
    return labels ? (bundle, augmented,
                     _BridgeFormulaLabels(augmented, copy(ctx.labels), by_param)) :
                    (bundle, augmented)
end

# Render one keyed part as a parsable string. Location-scale-scale entries
# (#546) arrive keyed by the marker call itself -- `sd_phylo(species)` -- and
# the R side already writes the full `sd_phylo(species) ~ rhs` into the VALUE.
# Emitting `key = value` there would make Julia read `f(x) = body` as a
# short-form FUNCTION DEFINITION (body wrapped in a block), which the formula
# parser cannot see through. So pass the value straight through: it is already
# the positional spelling `bf` understands.
function _bridge_keyed_part(k::AbstractString, v)
    occursin(r"^sd(_phylo)?\([^()]+\)$", k) && return String(v)
    return "$k = $v"
end

function _bridge_formula_parts(formula)
    if formula isa AbstractString
        return filter(!isempty, strip.(split(String(formula), ';')))
    elseif formula isa NamedTuple
        return [_bridge_keyed_part(String(k), v) for (k, v) in pairs(formula)]
    elseif formula isa AbstractDict
        return [_bridge_keyed_part(String(k), v) for (k, v) in pairs(formula)]
    elseif formula isa AbstractVector
        return String.(formula)
    end
    throw(ArgumentError("drm_bridge: `formula` must be a string, vector of strings, dictionary, or named tuple"))
end

# R model formulas write interactions with `:`, but Julia parses `:` as the
# RANGE operator, which has LOWER precedence than `+`. So `a + b + a:b` parses in
# Julia as `(a + b + a) : b` — mis-associating the `+` chain and, worse, pulling
# a trailing `phylo(1|g)` term inside a `FunctionTerm{Colon}` the engine can't
# read (Ayumi LS#2: `MethodError: |(::Int64, ::String)`). Julia's `&` has
# interaction-matching precedence (tighter than `+`), so rewrite `:` → `&` at the
# STRING level, before `Meta.parse`. A model-formula string never contains `::`.
function _bridge_translate_r_ops(part::AbstractString)
    occursin("::", part) && return part        # defensive: leave qualified names alone
    # R accepts whitespace between `I` and its call parenthesis; Julia parses
    # that spelling as implicit multiplication. Normalize only that admitted
    # materializer before `Meta.parse`, retaining the original text for labels.
    translated = replace(String(part), r"\bI\s+\(" => "I(")
    return replace(translated, ':' => '&')
end

# R formula constructs that `@formula` cannot evaluate as the R user intends,
# and so need a materialised column or an expanded term list to match R exactly
# (`I`, `scale`, `factor`, `poly`, `(...)^k`, and general `- term` removal —
# see `_bridge_xlate` below). All are implemented as faithful rewrites before
# `@formula`, each with an R-parity fixture on byte-identical data.
#
# `poly` was in this table as a blanket rejection until 2026-08-25, on the
# grounds that R's default `raw = FALSE` basis is "highest-risk to fake" and a
# raw-power stand-in would silently disagree. The premise was right and the
# conclusion was too broad: R's algorithm is deterministic, and transcribing it
# reproduces `stats::poly(x, 3)` to 9.99e-16 (#492). What remains rejected is
# the part that genuinely cannot be faked — `raw = TRUE` (write `I(x^k)`), an
# explicit `coefs =`, and multivariate `poly(x, y, degree)` — and that rejection
# now lives at the call site, where it can name the specific unsupported form.
const _BRIDGE_REJECT_CALLS = Dict{Symbol,String}(
    :^ => "R crossing `(...)^k` is unsupported via engine=\"julia\" for this shape (need a literal positive integer power over a `+`-only expression, with no `*` inside); expand it explicitly (e.g. `a + b + a&b`).",
)

# Mutable per-formula-bridge context: materialises `I(...)`, `scale(...)`, and
# `factor(...)` calls into real data columns (never `eval`s user code — see
# `_bridge_eval_I`), reusing the same synthesised column for repeated
# occurrences of the identical call across formula parts (e.g. `scale(x)` in
# both `mu` and `sigma`). `cols` is computed lazily so a formula that uses
# none of these constructs never touches `data` at all.
mutable struct _BridgeXlateCtx
    data
    cols::Union{Nothing,NamedTuple}
    extra::Dict{Symbol,Any}
    cache::Dict{String,Symbol}
    labels::Dict{Symbol,String}
    i_labels::Dict{String,String}
    part_function_labels::Dict{String,String}
    function_labels::Dict{String,String}
    n::Int
end
_BridgeXlateCtx(data) = _BridgeXlateCtx(data, nothing, Dict{Symbol,Any}(),
    Dict{String,Symbol}(), Dict{Symbol,String}(), Dict{String,String}(),
    Dict{String,String}(), Dict{String,String}(), 0)

function _bridge_ctx_cols!(ctx::_BridgeXlateCtx)
    ctx.cols === nothing && (ctx.cols = Tables.columntable(ctx.data))
    return ctx.cols
end

function _bridge_lookup_column(ctx::_BridgeXlateCtx, sym::Symbol)
    haskey(ctx.extra, sym) && return ctx.extra[sym]
    cols = _bridge_ctx_cols!(ctx)
    haskey(cols, sym) &&
        return cols[sym]
    throw(ArgumentError("drmTMB(engine=\"julia\"): column `$(sym)` referenced in the formula is not present in `data`."))
end

# Materialise (and cache) a new data column, returning the `Symbol` that now
# refers to it. Identical `kind`+source-expression pairs reuse the same
# synthesised column instead of recomputing it.
function _bridge_materialize!(ctx::_BridgeXlateCtx, kind::AbstractString, key_expr,
        compute::Function, label::AbstractString)
    key = kind * "::" * repr(key_expr)
    cached = get(ctx.cache, key, nothing)
    cached === nothing || return cached
    cols = _bridge_ctx_cols!(ctx)
    name = Symbol("")
    while true
        ctx.n += 1
        candidate = Symbol("__bridge_", kind, "_", ctx.n)
        # A user column can legitimately have the old synthetic spelling.  Do
        # not overwrite it: retain the real design column and allocate another
        # private name whose exact raw spelling is recorded below.
        if !(haskey(cols, candidate) || haskey(ctx.extra, candidate))
            name = candidate
            break
        end
    end
    ctx.extra[name] = compute()
    ctx.cache[key] = name
    ctx.labels[name] = String(label)
    return name
end

_bridge_r_label_piece(x::Symbol) = (String(x), 5, :atom)
_bridge_r_label_piece(x::Number) = (string(x), 5, :atom)
function _bridge_r_label_piece(e::Expr)
    e.head === :call || throw(ArgumentError("drm_bridge: cannot render formula expression $(repr(e)) as a public coefficient label"))
    f = e.args[1]
    f isa Symbol || throw(ArgumentError("drm_bridge: cannot render formula expression $(repr(e)) as a public coefficient label"))
    if f === :+ && length(e.args) == 2
        return _bridge_r_label_piece(e.args[2])
    elseif f === :- && length(e.args) == 2
        inner, iprec, _ = _bridge_r_label_piece(e.args[2])
        return (string("-", iprec < 3 ? "(" * inner * ")" : inner), 3, :unary)
    elseif f in (:+, :-, :*, :/, :^) && length(e.args) >= 3
        prec = f in (:+, :-) ? 1 : f in (:*, :/) ? 2 : 4
        op = f === :* ? " * " : f in (:+, :-) ? " $(f) " : string(f)
        left, lprec, _ = _bridge_r_label_piece(e.args[2])
        for arg in e.args[3:end]
            right, rprec, _ = _bridge_r_label_piece(arg)
            # Parenthesize precisely where omitting a group changes the
            # arithmetic expression.  This is label-only; the materialised
            # vector was already evaluated from the same safe AST.
            (lprec < prec || (f === :^ && lprec == prec)) && (left = "(" * left * ")")
            (rprec < prec || (f in (:-, :/, :^) && rprec == prec)) && (right = "(" * right * ")")
            left = left * op * right
            lprec = prec
        end
        return (left, prec, f)
    end
    return (string(f, "(", join((first(_bridge_r_label_piece(a)) for a in e.args[2:end]), ", "), ")"), 5, :atom)
end

_bridge_r_label(x) = first(_bridge_r_label_piece(x))

function _bridge_r_float_label(value::AbstractFloat)
    v = Float64(value)
    isfinite(v) || throw(ArgumentError("drm_bridge: non-finite numeric value cannot be rendered as an R coefficient label"))
    v == 0 && return "0"       # R does not retain a signed zero literal here.
    # R's model-matrix labels retain 15 significant digits, then choose the
    # shorter fixed or scientific spelling (fixed on ties).  A magnitude
    # threshold is insufficient: 100000 becomes 1e+05, but 100001 and
    # 1000000.1 remain fixed decimal.  Derive both candidates from ONE rounded
    # scientific representation so the choice never changes the numeric value.
    mantissa, exponent_text = split(@sprintf("%.14e", v), 'e'; limit = 2)
    exponent = parse(Int, exponent_text)
    sign = startswith(mantissa, "-") ? "-" : ""
    mantissa = sign == "-" ? mantissa[nextind(mantissa, firstindex(mantissa)):end] : mantissa
    scientific = sign * _bridge_trim_fractional_zeros(mantissa) *
                 "e" * @sprintf("%+03d", exponent)
    fixed = sign * _bridge_fixed_from_scientific(mantissa, exponent)
    return ncodeunits(fixed) <= ncodeunits(scientific) ? fixed : scientific
end

function _bridge_trim_fractional_zeros(text::AbstractString)
    value = String(text)
    occursin('.', value) || return value
    while endswith(value, "0")
        value = value[1:prevind(value, lastindex(value))]
    end
    endswith(value, ".") && (value = value[1:prevind(value, lastindex(value))])
    return value
end

function _bridge_fixed_from_scientific(mantissa::AbstractString, exponent::Integer)
    digits = replace(String(mantissa), "." => "")
    decimal = Int(exponent) + 1
    value = if decimal <= 0
        "0." * repeat("0", -decimal) * digits
    elseif decimal >= ncodeunits(digits)
        digits * repeat("0", decimal - ncodeunits(digits))
    else
        digits[1:decimal] * "." * digits[decimal+1:end]
    end
    return _bridge_trim_fractional_zeros(value)
end

function _bridge_r_number_label(token::AbstractString)
    value = tryparse(Float64, token)
    value === nothing && throw(ArgumentError("drm_bridge: cannot render numeric literal `$token` as an R coefficient label"))
    return _bridge_r_float_label(value)
end

# Canonicalise only the small arithmetic grammar admitted by `I(...)`.  Unlike
# reparsing/pretty-printing an AST, this retains explicit parentheses and unary
# `+`, which R's model-matrix labels retain even when algebra could remove them.
function _bridge_r_scalar_source_label(text::AbstractString)
    s = String(text)
    out = IOBuffer()
    i = firstindex(s)
    previous_atom = false
    while i <= lastindex(s)
        c = s[i]
        if isspace(c)
            i = nextind(s, i)
        elseif isletter(c) || c == '_'
            j = nextind(s, i)
            while j <= lastindex(s) && (isletter(s[j]) || isdigit(s[j]) || s[j] == '_')
                j = nextind(s, j)
            end
            print(out, s[i:prevind(s, j)])
            previous_atom = true
            i = j
        elseif isdigit(c) || c == '.'
            j = i
            while j <= lastindex(s) && (isdigit(s[j]) || s[j] in ('.', 'e', 'E', '+', '-'))
                # A sign belongs to an exponent only; otherwise it begins the
                # next arithmetic token.
                if s[j] in ('+', '-') && j != i && s[prevind(s, j)] ∉ ('e', 'E')
                    break
                end
                j = nextind(s, j)
            end
            print(out, _bridge_r_number_label(s[i:prevind(s, j)]))
            previous_atom = true
            i = j
        elseif c == '('
            print(out, c)
            previous_atom = false
            i = nextind(s, i)
        elseif c == ')'
            print(out, c)
            previous_atom = true
            i = nextind(s, i)
        elseif c == ','
            print(out, ", ")
            previous_atom = false
            i = nextind(s, i)
        elseif c in ('<', '>', '!', '=', '&', '|')
            # Generic scalar transforms already admit R-style comparisons and
            # boolean combinations.  Canonicalize their spacing only; this
            # scanner never evaluates or rewrites their formula semantics.
            j = nextind(s, i)
            if j <= lastindex(s) && s[j] == '=' && c in ('<', '>', '!', '=')
                op = string(c, '=')
                j = nextind(s, j)
                print(out, " ", op, " ")
            elseif j <= lastindex(s) && s[j] == c && c in ('&', '|')
                op = string(c, c)
                j = nextind(s, j)
                print(out, " ", op, " ")
            elseif c == '!'
                # `!` is prefix in R's deparsed formula labels. Keep `!=`
                # above binary and leave this form adjacent to its operand.
                print(out, '!')
            else
                print(out, " ", c, " ")
            end
            previous_atom = false
            i = j
        elseif c in ('+', '-', '*', '/', '^')
            binary = previous_atom
            if c == '*' || (c in ('+', '-') && binary)
                print(out, " ", c, " ")
            else
                print(out, c)
            end
            previous_atom = false
            i = nextind(s, i)
        else
            throw(ArgumentError("drm_bridge: cannot render scalar source `$text` as an R coefficient label"))
        end
    end
    return String(take!(out))
end

function _bridge_call_end(s::AbstractString, open::Integer)
    s[open] == '(' || throw(ArgumentError("drm_bridge: internal call scanner expected `(`"))
    depth = 1
    i = nextind(s, open)
    while i <= lastindex(s) && depth > 0
        s[i] == '(' && (depth += 1)
        s[i] == ')' && (depth -= 1)
        i = nextind(s, i)
    end
    depth == 0 || throw(ArgumentError("drm_bridge: unclosed scalar formula function"))
    return prevind(s, i)
end

# Scan the original R-side source before translation.  Parsed Julia ASTs do
# not retain redundant groups, unary `+`, or R's literal spelling, so public
# names must come from this restricted source provenance rather than pretty
# printing a transformed expression.
function _bridge_register_source_labels!(ctx::_BridgeXlateCtx, part::AbstractString)
    s = String(part)
    i = firstindex(s)
    while i <= lastindex(s)
        if isletter(s[i]) || s[i] == '_'
            start = i
            i = nextind(s, i)
            while i <= lastindex(s) && (isletter(s[i]) || isdigit(s[i]) || s[i] == '_')
                i = nextind(s, i)
            end
            name = s[start:prevind(s, i)]
            open = i
            while open <= lastindex(s) && isspace(s[open])
                open = nextind(s, open)
            end
            open <= lastindex(s) && s[open] == '(' || continue
            close = _bridge_call_end(s, open)
            source = s[start:close]
            parsed = Meta.parse(_bridge_translate_r_ops(source))
            if name == "I"
                length(parsed.args) == 2 || throw(ArgumentError("drm_bridge: `I(...)` takes exactly one expression."))
                key = repr(parsed.args[2])
                label = "I(" * _bridge_r_scalar_source_label(s[nextind(s, open):prevind(s, close)]) * ")"
                old = get(ctx.i_labels, key, nothing)
                (old === nothing || old == label) || throw(ArgumentError(
                    "drm_bridge: repeated `I(...)` expressions with different public spellings are ambiguous"))
                ctx.i_labels[key] = label
            else
                f = Symbol(name)
                # Formula operators and bridge DSL markers retain their own
                # grammar; only genuine scalar calls get source provenance.
                if !(f in _BRIDGE_DSL_CALLS || f in _BRIDGE_TERM_OPS || f in (:scale, :factor, :poly)) &&
                   !_bridge_contains_poly(parsed)
                    # The established xlate guard gives `poly()` under a
                    # scalar call its precise model-shape error. Do not let
                    # label provenance preempt that rejection.
                    key = repr(parsed)
                    label = _bridge_r_scalar_source_label(source)
                    old = get(ctx.part_function_labels, key, nothing)
                    (old === nothing || old == label) || throw(ArgumentError(
                        "drm_bridge: repeated scalar transforms with different public spellings are ambiguous"))
                    ctx.part_function_labels[key] = label
                end
            end
        else
            i = nextind(s, i)
        end
    end
    return ctx
end

function _bridge_record_scalar_label!(ctx::_BridgeXlateCtx, translated, label)
    label === nothing && return translated
    key = repr(translated)
    old = get(ctx.function_labels, key, nothing)
    (old === nothing || old == label) || throw(ArgumentError(
        "drm_bridge: scalar transform public labels collide within one formula part"))
    ctx.function_labels[key] = label
    return translated
end

# `I(expr)`: evaluate `expr` against the data through a SAFE, restricted
# arithmetic grammar (`+ - * / ^`, data columns, numeric literals) — never
# `Base.eval` on user-supplied text, which would run arbitrary code.
function _bridge_eval_I(expr, ctx::_BridgeXlateCtx)
    expr isa Symbol && return _bridge_lookup_column(ctx, expr)
    expr isa Number && return expr
    if expr isa Expr && expr.head === :call && expr.args[1] isa Symbol
        op = expr.args[1]
        fn = op === :+ ? (+) :
             op === :- ? (-) :
             op === :* ? (*) :
             op === :/ ? (/) :
             op === :^ ? (^) :
             throw(ArgumentError("drmTMB(engine=\"julia\"): `I(...)` only supports the arithmetic operators +, -, *, /, ^ over data columns and numeric literals; got `$(op)`."))
        args = Any[_bridge_eval_I(a, ctx) for a in expr.args[2:end]]
        return broadcast(fn, args...)
    end
    throw(ArgumentError("drmTMB(engine=\"julia\"): `I(...)` only supports arithmetic (+, -, *, /, ^) over data columns and numeric literals; unsupported expression `$(expr)`."))
end

# `scale(x)`: center and scale by the sample mean/SD (R's `scale()` default —
# `sd()` uses the n-1 denominator, matching `Statistics.std`'s default).
function _bridge_eval_scale(sym::Symbol, ctx::_BridgeXlateCtx)
    col = float.(_bridge_lookup_column(ctx, sym))
    mu = Statistics.mean(col)
    sdv = Statistics.std(col)
    sdv == 0 && throw(ArgumentError("drmTMB(engine=\"julia\"): `scale($(sym))` has zero standard deviation; cannot standardize a constant column."))
    return (col .- mu) ./ sdv
end

# `poly(x, k)`: R's ORTHOGONAL polynomial basis — `stats::poly()`'s default
# (`raw = FALSE`), not raw powers. R's algorithm is deterministic: centre `x`,
# build the Vandermonde `[1, xc, xc^2, …, xc^k]`, take its QR, rescale each
# column by its own norm, and drop the constant column. Transcribing it
# reproduces `stats::poly(x, 3)` to 9.99e-16 on byte-identical data (#492), and
# the QR sign convention already matches R's, so no sign-fixing pass is needed.
#
# Returns the n×k matrix; the caller materialises one data column per degree
# because Tables.jl does not accept a matrix-valued column in a column table.
#
# NOT supported, and rejected at the call site rather than approximated:
# `raw = TRUE` (write the powers with `I(x^k)`), an explicit `coefs =`, and the
# multivariate `poly(x, y, degree)`, which is a different construction.
function _bridge_eval_poly(sym::Symbol, degree::Int, ctx::_BridgeXlateCtx)
    col = float.(_bridge_lookup_column(ctx, sym))
    n = length(col)
    nuniq = length(unique(col))
    # R: "'degree' must be less than number of unique points".
    degree < nuniq || throw(ArgumentError(
        "drmTMB(engine=\"julia\"): `poly($(sym), $(degree))` needs the degree to be less than the " *
        "number of unique values in `$(sym)` (found $(nuniq)); R's `stats::poly()` errors here too."))
    xc = col .- (sum(col) / n)
    X = hcat((xc .^ k for k in 0:degree)...)
    F = LinearAlgebra.qr(X)
    Z = Matrix(F.Q) * LinearAlgebra.Diagonal(LinearAlgebra.diag(F.R))
    nrm2 = vec(sum(abs2, Z; dims = 1))
    any(v -> v <= 0 || !isfinite(v), nrm2) && throw(ArgumentError(
        "drmTMB(engine=\"julia\"): `poly($(sym), $(degree))` produced a degenerate basis; the column is " *
        "probably collinear at this degree. Lower the degree or precompute the basis in R."))
    Z = Z ./ sqrt.(nrm2)'
    return Z[:, 2:end]
end

# Does `e` contain a `*` (R/StatsModels crossing) anywhere? `-` (general term
# removal) and `^` (crossing power) below do their own term-list algebra by
# flattening `+`/`&` only; an unexpanded `*` inside that algebra could hide a
# term from a `-` removal or a `^` combination, silently disagreeing with R.
# Reject those combinations explicitly instead of guessing.
# Does `e` contain a `poly(...)` call anywhere? `poly` is the only construct
# here that rewrites to a GROUP of terms (`+`), so it is only meaningful where a
# group is meaningful. Under a scalar function — `log1p(poly(x, 2))` — R applies
# the function elementwise to a k-column MATRIX, giving k columns, while this
# rewrite would apply it to the SUM of the k columns, giving one. That is a
# silent disagreement of exactly the kind the blanket `poly` rejection existed to
# prevent, so it is rejected explicitly instead of being inherited by accident.
_bridge_contains_poly(e) = false
function _bridge_contains_poly(e::Expr)
    e.head === :call && e.args[1] === :poly && return true
    return any(_bridge_contains_poly, e.args)
end

# Formula operators under which a `+` group of terms means what R means.
# `-` and `(...)^k` are absent because they have their own branches above and
# flatten `+` themselves before doing term algebra.
const _BRIDGE_TERM_OPS = (:+, :&, :*, :~)
const _BRIDGE_DSL_CALLS = Set((:phylo, :relmat, :animal, :spatial, :sd,
    :sd_phylo, :meta_V, :cbind, :corpair, :|))

_bridge_contains_star(e) = false
function _bridge_contains_star(e::Expr)
    e.head === :call && e.args[1] === :* && return true
    return any(_bridge_contains_star, e.args)
end

# Flatten the top-level `+` chain of an already-`_bridge_xlate`d expression
# into its additive terms (each term itself opaque — a symbol, an `&`
# interaction, a function call, …). Used by both `-` (term removal) and `^`
# (crossing power) term algebra.
_bridge_formula_terms(e) = Any[e]
function _bridge_formula_terms(e::Expr)
    if e.head === :call && e.args[1] === :+
        return vcat(_bridge_formula_terms.(e.args[2:end])...)
    end
    return Any[e]
end

# Canonicalise a term for structural-equality matching: `&` (R's `:`) is an
# unordered interaction, so `a&b` must match `b&a` when removing a term.
_bridge_canon(t) = t
function _bridge_canon(e::Expr)
    if e.head === :call && e.args[1] === :&
        operands = sort(_bridge_canon.(e.args[2:end]); by = repr)
        return Expr(:call, :&, operands...)
    end
    return Expr(e.head, _bridge_canon.(e.args)...)
end

# `lhs - rhs`: remove every term of `rhs` (itself possibly a `+`-sum, e.g.
# `- (a + b)`) from the term list of `lhs`, by canonical structural match. A
# term absent from `lhs` is silently a no-op — this matches R's own
# `terms()` behaviour for a `-` naming a term that was never present.
function _bridge_remove_terms(lhs, rhs)
    lhs_terms = _bridge_formula_terms(lhs)
    rhs_canon = Set(_bridge_canon.(_bridge_formula_terms(rhs)))
    return filter(t -> !(_bridge_canon(t) in rhs_canon), lhs_terms)
end

function _bridge_terms_to_sum(terms::Vector)
    isempty(terms) && return 1
    length(terms) == 1 && return terms[1]
    return Expr(:call, :+, terms...)
end

# All `k`-subsets of `items`, preserving relative order (order doesn't affect
# the numerics — `&` is commutative in `@formula` — only readability).
function _bridge_combinations(items::Vector, k::Int)
    k <= 0 && return [Any[]]
    isempty(items) && return Vector{Any}[]
    first, rest = items[1], items[2:end]
    with_first = [vcat(Any[first], c) for c in _bridge_combinations(rest, k - 1)]
    without_first = _bridge_combinations(rest, k)
    return vcat(with_first, without_first)
end

# `(inner)^k`: R expands to every combination of `inner`'s order-1 terms taken
# 1..k at a time, joined by `:` (our `&`) — main effects through order-`k`
# interactions, e.g. `(a+b+c)^2` = `a+b+c+a&b+a&c+b&c`.
function _bridge_expand_power(base_terms::Vector, k::Int)
    out = Any[]
    for j in 1:k, combo in _bridge_combinations(base_terms, j)
        push!(out, length(combo) == 1 ? combo[1] : Expr(:call, :&, combo...))
    end
    return out
end

# Translate / validate the parsed formula tree before `@formula`. `:` is already
# `&` (handled at the string level); here we translate R's `- 1`/`- 0` intercept
# control, expand `(...)^k` crossing and general `- term` removal into the
# `+`/`&` terms `@formula` already understands faithfully (confirmed: `*`
# crossing already matches R via `@formula` natively), materialise `I`/`scale`/
# `factor` into real columns, and reject the constructs that cannot be made
# faithful. Markers (phylo/relmat/animal/spatial/meta_V/cbind) and StatsModels
# transforms (log/exp/…) pass through unchanged.
_bridge_xlate(x, ctx::_BridgeXlateCtx; scalar_context::Bool = false,
    atom_scope::Union{Nothing,String} = nothing) = x
function _bridge_xlate(e::Expr, ctx::_BridgeXlateCtx;
        scalar_context::Bool = false, atom_scope::Union{Nothing,String} = nothing)
    e.head === :call || return e
    f = e.args[1]
    if f === :-
        if scalar_context
            return Expr(:call, f, (_bridge_xlate(a, ctx;
                scalar_context = true, atom_scope = atom_scope) for a in e.args[2:end])...)
        elseif length(e.args) == 3 && e.args[3] === 1
            return Expr(:call, :+, 0, _bridge_xlate(e.args[2], ctx;
                atom_scope = atom_scope))   # `… - 1` → drop intercept
        elseif length(e.args) == 3 && e.args[3] === 0
            return _bridge_xlate(e.args[2], ctx; atom_scope = atom_scope) # `… - 0` → keep intercept
        elseif length(e.args) == 3
            lhs = _bridge_xlate(e.args[2], ctx; atom_scope = atom_scope)
            rhs = _bridge_xlate(e.args[3], ctx; atom_scope = atom_scope)
            (_bridge_contains_star(lhs) || _bridge_contains_star(rhs)) &&
                throw(ArgumentError("drmTMB(engine=\"julia\"): R term removal `-` combined with unexpanded `*` crossing is unsupported (the removed term could be hiding inside the `*`); expand the crossing explicitly (e.g. `a + b + a&b`) before removing a term."))
            return _bridge_terms_to_sum(_bridge_remove_terms(lhs, rhs))
        end
        throw(ArgumentError("drmTMB(engine=\"julia\"): R term removal with `-` is unsupported; list the terms you want explicitly."))
    elseif f === :^
        if scalar_context
            return Expr(:call, f, (_bridge_xlate(a, ctx;
                scalar_context = true, atom_scope = atom_scope) for a in e.args[2:end])...)
        elseif length(e.args) == 3 && e.args[3] isa Integer && e.args[3] >= 1
            # MEASURED 2026-08-25: `poly()` is NOT safe under `(...)^k`, and this is
            # the one place the `+`-group rewrite breaks. R treats `poly(x, 2)` as a
            # SINGLE term, so `(x + poly(x, 2))^2` crosses two terms and never forms
            # `poly1:poly2` — R gives 6 model-matrix columns. Flattening poly into
            # `p1 + p2` makes it two terms, so the same expansion yields 7, the extra
            # column being `p1 & p2`. Checked against `model.matrix()` on both sides.
            # Rejected rather than special-cased: keeping the group intact through the
            # power algebra would need a term-grouping concept the rewrite does not have.
            _bridge_contains_poly(e.args[2]) && throw(ArgumentError(
                "drmTMB(engine=\"julia\"): `poly(x, k)` inside `(...)^k` crossing is unsupported. R treats " *
                "`poly(x, k)` as ONE term, so it never crosses a poly column with another poly column; this " *
                "rewrite expands poly into k separate terms, which would add those extra interactions " *
                "(measured: R 6 columns vs 7 here for `(x + poly(x, 2))^2`). Expand the crossing explicitly, " *
                "or precompute the basis in R and pass the columns as covariates."))
            inner = _bridge_xlate(e.args[2], ctx; atom_scope = atom_scope)
            _bridge_contains_star(inner) &&
                throw(ArgumentError("drmTMB(engine=\"julia\"): R crossing `(...)^k` over an expression that already contains unexpanded `*` crossing is unsupported; expand explicitly."))
            return _bridge_terms_to_sum(_bridge_expand_power(_bridge_formula_terms(inner), Int(e.args[3])))
        end
        throw(ArgumentError("drmTMB(engine=\"julia\"): " * _BRIDGE_REJECT_CALLS[:^]))
    elseif f === :I
        length(e.args) == 2 ||
            throw(ArgumentError("drmTMB(engine=\"julia\"): `I(...)` takes exactly one expression."))
        label = get(ctx.i_labels, repr(e.args[2]), "I($(_bridge_r_label(e.args[2])))")
        # The same evaluated I() expression can legitimately appear with
        # distinct R spellings in different dpar formulas, e.g. `I(x^2)` for
        # mu and `I((x^2))` for sigma.  It therefore receives distinct bridge
        # atoms unless its exact public spelling also matches.  Within one
        # formula part `_bridge_register_i_labels!` still rejects ambiguous
        # repeated spellings before a collinear duplicate can be created.
        return _bridge_materialize!(ctx, "I", (e.args[2], label, atom_scope),
            () -> _bridge_eval_I(e.args[2], ctx),
            label)
    elseif f === :poly
        # `poly(x, k)` expands to k model-matrix columns, so unlike `scale()` this
        # returns a `+` GROUP of materialised symbols rather than one symbol. That
        # composes safely with the `-` and `(...)^k` algebra below because both
        # flatten `+` before operating on the term list.
        length(e.args) == 3 || throw(ArgumentError(
            "drmTMB(engine=\"julia\"): only `poly(x, k)` is supported via engine=\"julia\". " *
            "`raw = TRUE` is spelled `I(x^k)` term by term; an explicit `coefs =` and the multivariate " *
            "`poly(x, y, degree)` are unsupported — precompute those columns in R and pass them as covariates."))
        arg = e.args[2]
        arg isa Symbol || throw(ArgumentError(
            "drmTMB(engine=\"julia\"): `poly(...)` only supports a bare column reference (e.g. `poly(x, 3)`), " *
            "not a general expression; precompute the basis and pass it as covariates."))
        deg = e.args[3]
        (deg isa Integer && deg >= 1) || throw(ArgumentError(
            "drmTMB(engine=\"julia\"): `poly($(arg), …)` needs an integer degree >= 1 written as a literal " *
            "(got `$(deg)`)."))
        degi = Int(deg)
        basis = _bridge_eval_poly(arg, degi, ctx)
        syms = [_bridge_materialize!(ctx, "poly$(degi)c$(j)", arg,
            () -> basis[:, j], "poly($(_bridge_r_label(arg)), $(degi))$(j)") for j in 1:degi]
        return Expr(:call, :+, syms...)
    elseif f === :scale
        length(e.args) == 2 ||
            throw(ArgumentError("drmTMB(engine=\"julia\"): `scale(...)` with explicit `center`/`scale` arguments is unsupported via engine=\"julia\"; precompute the standardized column and pass it as a covariate."))
        arg = e.args[2]
        arg isa Symbol ||
            throw(ArgumentError("drmTMB(engine=\"julia\"): `scale(...)` only supports a bare column reference (e.g. `scale(x)`), not a general expression; precompute the standardized column and pass it as a covariate."))
        return _bridge_materialize!(ctx, "scale", (arg, atom_scope),
            () -> _bridge_eval_scale(arg, ctx), "scale($(_bridge_r_label(arg)))")
    elseif f === :factor
        length(e.args) == 2 ||
            throw(ArgumentError("drmTMB(engine=\"julia\"): `factor(...)` with extra arguments is unsupported via engine=\"julia\"; precompute the factor column and pass it as a covariate."))
        arg = e.args[2]
        arg isa Symbol ||
            throw(ArgumentError("drmTMB(engine=\"julia\"): `factor(...)` only supports a bare column reference (e.g. `factor(g)`), not a general expression; precompute the factor column and pass it as a covariate."))
        # A plain (non-`<:Real`) `Vector{Any}` copy flips StatsModels onto its
        # categorical dispatch, with levels ordered by `sort(unique(...))` on
        # the ORIGINAL values (numeric order preserved, not string order) —
        # exactly R's `factor()` default levels, giving `contr.treatment`
        # dummy coding against the same (lowest) baseline level.
        return _bridge_materialize!(ctx, "factor", (arg, atom_scope),
            () -> Any[v for v in _bridge_lookup_column(ctx, arg)],
            "factor($(_bridge_r_label(arg)))")
    elseif !(f isa Symbol)
        throw(ArgumentError("drmTMB(engine=\"julia\"): unsupported formula function `$(f)`; precompute it as a covariate column."))
    elseif haskey(_BRIDGE_REJECT_CALLS, f)
        throw(ArgumentError("drmTMB(engine=\"julia\"): " * _BRIDGE_REJECT_CALLS[f]))
    end
    # `poly()` may only sit where a GROUP of terms is meaningful. Checked BEFORE
    # recursing, while the argument expressions still say `poly(...)`.
    if !(f in _BRIDGE_TERM_OPS) && any(_bridge_contains_poly, e.args[2:end])
        throw(ArgumentError("drmTMB(engine=\"julia\"): `poly(x, k)` may only appear as a model term " *
            "(under `~`, `+`, `&`, `*`, or the `-` / `(...)^k` term algebra), not inside `$(f)(...)`. " *
            "`poly()` expands to k columns; R would apply `$(f)` elementwise to all k, while this rewrite " *
            "would apply it to their sum — one column instead of k, silently. Precompute the transformed " *
            "basis in R and pass the columns as covariates."))
    end
    # Recurse into EVERY remaining call's arguments. Formula operators retain
    # their existing term algebra; true scalar functions keep arithmetic `-`
    # and `^` rather than being mistaken for term removal/crossing. DSL marker
    # calls deliberately remain formula context, never arbitrary scalar calls.
    scalar_child = scalar_context || !(f in _BRIDGE_TERM_OPS || f in _BRIDGE_DSL_CALLS)
    source_label = scalar_child ? get(ctx.part_function_labels, repr(e), nothing) : nothing
    nested_scope = atom_scope === nothing ? source_label : atom_scope
    translated = Expr(:call, f, (_bridge_xlate(a, ctx;
        scalar_context = scalar_child, atom_scope = nested_scope) for a in e.args[2:end])...)
    return _bridge_record_scalar_label!(ctx, translated, source_label)
end

function _bridge_parse_formula_part(part::AbstractString, ctx::_BridgeXlateCtx)
    # I() spelling provenance is scoped to a single R formula part.  A single
    # context still owns all materialised data, but dpar formulas must not
    # reject each other merely because their same-valued transform is written
    # differently for a public coefficient selector.
    empty!(ctx.i_labels)
    empty!(ctx.part_function_labels)
    empty!(ctx.function_labels)
    _bridge_register_source_labels!(ctx, part)
    expr = Meta.parse(_bridge_translate_r_ops(part))
    if expr isa Expr && expr.head === :(=)
        length(expr.args) == 2 || return nothing
        key = expr.args[1]
        key isa Symbol || return nothing
        form = _bridge_formula_from_expr(expr.args[2], ctx)
        form === nothing && return nothing
        return key => (form, copy(ctx.function_labels))
    end
    form = _bridge_formula_from_expr(expr, ctx)
    form === nothing && return nothing
    return nothing => (form, copy(ctx.function_labels))
end

function _bridge_formula_from_expr(expr, ctx::_BridgeXlateCtx)
    (expr isa Expr && expr.head === :call && expr.args[1] === :~) || return nothing
    expr = _bridge_xlate(expr, ctx)
    return eval(Expr(:macrocall, Symbol("@formula"), LineNumberNode(0), expr))
end

function _bridge_flatten(fit; family::AbstractString, newdata = nothing,
        labels::Union{Nothing,_BridgeFormulaLabels} = nothing, coef_labels = nothing)
    cnames, cvals, raw_cnames, public_to_raw =
        _bridge_coef_vector(fit; labels = labels, coef_labels = coef_labels)
    V = Matrix{Float64}(vcov(fit))
    _bridge_validate_coordinate_axes(fit.blocks, fit.coefnames, length(cvals), V)
    fitted_vals, residual_vals = _bridge_fitted_marginal(fit)
    out = Dict{String,Any}(
        "family" => String(family),
        "coef_names" => cnames,
        "coefficients" => cvals,
        "coef" => Dict(cnames .=> cvals),
        "vcov" => V,
        "vcov_names" => cnames,
        "loglik" => loglik(fit),
        "aic" => aic(fit),
        "bic" => bic(fit),
        "df" => dof(fit),
        "nobs" => nobs(fit),
        "converged" => is_converged(fit),
        # estim_method / ml_loglik / infocrit_basis are unconditional (#624): every
        # DrmFit carries `estim_method` and an always-set `ml_loglik` (the plain ML
        # log-likelihood, equal to `loglik` for an ML fit). Before this, an R caller
        # reading `fit$bridge$loglik`/`$aic`/`$bic` off a `method = "REML"` bridge
        # fit had no signal that these were REML-restricted values — only a
        # Julia-side `@warn` that never crosses the boundary.
        "estim_method" => String(fit.estim_method),
        "ml_loglik" => fit.ml_loglik,
        "infocrit_basis" => fit.estim_method === :REML ? "reml" : "ml",
        # Optimiser iterations actually taken. -1 means the fitter does not record
        # it yet, which the R side must read as "unknown" -- NOT as zero. Before
        # 2026-08-24 this key did not exist at all, so `fit$bridge$iterations` was
        # NA everywhere and no bridge-side comparison of optimiser effort was
        # possible: a speed difference could be measured but never attributed.
        "iterations" => niterations(fit),
        # The integrator the fit actually used (`:LA` default, `:Laplace`,
        # `:VA`, `:AGHQ`), so the R side can refuse to claim same-target Laplace
        # parity unless the engine reports it (Arc 2).
        "marginal" => String(fit.marginal),
        # `fitted()`/`residuals()` AS drmTMB DEFINES THEM. Identical to DRModels.jl's
        # own for every fit except a zero-inflated count fit -- see
        # `_bridge_fitted_marginal`.
        "fitted" => _bridge_plain(fitted_vals),
        "residuals" => _bridge_plain(residual_vals),
        "sigma" => _bridge_plain(sigma(fit)),
        "corpairs" => _bridge_plain(corpairs(fit)),
        "dpars" => _bridge_dpars(fit),
    )
    # `reml_loglik` is omitted rather than reported as NaN for an ML fit (matches
    # `trials`/`meta`/`q4_point_export` below: absent, not a sentinel, when the
    # fit genuinely does not carry it). When it IS present, also echo the same
    # AIC/BIC cross-mean-structure caveat `_reml_infocrit_warn` already prints to
    # the Julia log — as `"warnings"`, since that log line never reaches R (#624).
    if fit.estim_method === :REML
        isnan(fit.reml_loglik) ||
            (out["reml_loglik"] = fit.reml_loglik)
        out["warnings"] = String[
            _reml_infocrit_warning_text("aic"),
            _reml_infocrit_warning_text("bic"),
        ]
    end
    # `gradient` is the Julia-side objective gradient at θ̂ — the diagnostic an R
    # caller wants when a fit looks off (e.g. non-convergence, a boundary
    # estimate). Only some routes attach `fit.nllgrad` (an in-place callback
    # `(g, θ) -> g`, not a precomputed vector), so — matching `reml_loglik`
    # above (#625) — the key is OMITTED, never zeros/NaN, when the route
    # carries none. `_bridge_coef_vector` already asserts (with an error) that
    # `cnames` is in exact 1:1 covariance order with `coef(fit)` (every raw
    # coordinate covered, no gaps), so `gradient_names` can reuse `cnames`
    # unchanged and stay index-aligned with `gradient`.
    if fit.nllgrad !== nothing
        θ̂ = coef(fit)
        g = zeros(length(θ̂))
        fit.nllgrad(g, θ̂)
        length(g) == length(cnames) || error(
            "drm_bridge: internal error — gradient length $(length(g)) does not match " *
            "the $(length(cnames)) named coefficients; refusing to mislabel gradient entries.")
        out["gradient"] = g
        out["gradient_names"] = cnames
    end
    if labels !== nothing || coef_labels !== nothing
        # `coef_names`/`vcov_names` are the public R spelling.  Retain the exact
        # Julia coordinate names and a bijection for the bridge inference route;
        # no numeric coordinate, value, or covariance entry is rewritten. When
        # `options["coef_labels"]` (design 258 §7.1) is supplied, its per-dpar
        # names are echoed verbatim (dpar-prefixed) as `coef_names`/
        # `vcov_names` in place of Julia's self-rendered names — `raw_coef_names`
        # still carries Julia's own spelling either way (#563).
        out["coef_label_contract"] = "bridge_formula_labels_v1"
        out["raw_coef_names"] = raw_cnames
        out["coef_name_map"] = public_to_raw
    end
    trials = _bridge_trials(fit)
    trials === nothing || (out["trials"] = trials)
    meta = _bridge_meta_parts(fit)
    if meta !== nothing
        # drmTMB's meta `sigma` dpar is the heterogeneity, NOT the total SD that
        # `scales[:sigma]` holds for simulation. Correct the dpar and ship
        # V_known alongside it, exactly as fitted_distribution_params() expects.
        out["dpars"]["sigma"] = meta.tau
        out["V_known"] = meta.v_known
    end
    newdata === nothing || (out["dpars_newdata"] = _bridge_dpars_newdata(fit, newdata))
    q4_point_export = _bridge_q4_point_export(fit; family = family)
    if !isempty(q4_point_export)
        out["q4_point_export"] = q4_point_export
    end
    q2_point_export = _bridge_q2_point_export(fit; family = family)
    if !isempty(q2_point_export)
        out["q2_point_export"] = q2_point_export
    end
    return out
end

function _bridge_coef_vector(fit; labels::Union{Nothing,_BridgeFormulaLabels} = nothing,
        coef_labels = nothing)
    θ = coef(fit)
    namemap = Dict(p => ns for (p, ns) in fit.coefnames)
    raw_names = String[]
    vals = Float64[]
    indices = Int[]
    block_ranges = Pair{Symbol,UnitRange{Int}}[]
    for (param, r) in fit.blocks
        haskey(namemap, param) || error("drm_bridge: no coefficient names for parameter block `$param`")
        pnames = namemap[param]
        length(pnames) == length(r) ||
            error("drm_bridge: coefficient-name mismatch for `$param`")
        block_start = length(raw_names) + 1
        for (nm, idx) in zip(pnames, r)
            push!(raw_names, "$(param)_$(nm)")
            push!(vals, θ[idx])
            push!(indices, idx)
        end
        push!(block_ranges, param => block_start:length(raw_names))
    end
    _bridge_validate_coordinate_axes(fit.blocks, fit.coefnames, length(θ), nothing)
    length(raw_names) == length(θ) || error("drm_bridge: coefficient blocks do not cover every raw coordinate")
    indices == collect(1:length(θ)) ||
        error("drm_bridge: coefficient blocks must cover raw coordinates in covariance order")
    length(unique(raw_names)) == length(raw_names) ||
        error("drm_bridge: raw coefficient labels are not unique")
    if coef_labels !== nothing
        names = _bridge_echo_coef_labels(coef_labels, block_ranges, raw_names)
        length(unique(names)) == length(names) ||
            error("drm_bridge: echoed coef_labels public names are not unique")
        # The echo is positional: it pastes R's names onto whatever columns
        # this fit built. Before trusting it, compare every regression block
        # DRModels.jl can render itself against the supplied spelling, so a design
        # that disagrees (factor level order, contrasts, term order) is
        # refused BY NAME rather than reported silently under R's names.
        labels === nothing ||
            _bridge_check_coef_labels_fidelity(fit, labels, block_ranges, raw_names, names)
        labels === nothing ||
            _bridge_check_lss_coef_labels_fidelity(fit, labels, raw_names, names)
        public_to_raw = Dict{String,String}(pub => raw for (pub, raw) in zip(names, raw_names))
        return names, vals, raw_names, public_to_raw
    end
    if labels === nothing
        return raw_names, vals, raw_names, Dict{String,String}(n => n for n in raw_names)
    end
    public_to_raw = _bridge_public_to_raw_coef_map(fit, labels, raw_names)
    raw_to_public = Dict(raw => public for (public, raw) in public_to_raw)
    names = String[raw_to_public[raw] for raw in raw_names]
    length(unique(names)) == length(names) || error("drm_bridge: public coefficient labels are not unique")
    Set(values(public_to_raw)) == Set(raw_names) ||
        error("drm_bridge: public coefficient map does not cover every raw coordinate")
    return names, vals, raw_names, public_to_raw
end

# Echo an R-supplied `options["coef_labels"] :: Dict{String,Vector{String}}`
# payload (design 258 §7.1: base-R `model.matrix()` column names per dpar, in
# order, fixed-effect part only) as the public `bridge_formula_labels_v1`
# names, prefixed `"<dpar>_"` to match the self-rendered convention. Fails
# closed: any count mismatch, unknown dpar key, or a dpar with fixed-effect
# columns but no supplied labels aborts naming the dpar and both counts —
# never guesses, pads, or reorders (#563).
#
# EVERY block reported by `fit.blocks` needs an entry, not just `mu`/`sigma`/
# the regression dpars — this includes `phylocov` and `sd`/`sd_phylo` random-
# effect blocks (R now labels those too, design 258 §7.5); a block present in
# `fit.blocks` with no matching key in `coef_labels` fails closed the same as
# any other missing dpar.
#
# A value may be a `Vector{String}` (the general case) or a bare `String`
# (accepted only for a length-1 block): JuliaCall unboxes a length-1 R
# character vector to a scalar Julia `String` rather than `[String]`, so an
# intercept-only dpar or a scalar `phylocov`/`sd` block can arrive unwrapped.
# A scalar String is treated exactly like `[String]` — same count check
# against the block's coefficient count, same fail-closed behaviour on a
# mismatch. Any other non-String, non-vector-of-String value (e.g. an `Int`)
# fails closed rather than being coerced (#563 follow-up to #594).
# Name the construct behind the commonest count mismatch (DRModels.jl #467/#609).
# R's `model.matrix()` gives every DECLARED factor level a column, including
# an all-zero one for a level no row uses; DRModels.jl codes only the levels it
# OBSERVES. R then supplies more names than this fit has columns, and the bare
# count message names neither the column nor the fix. Measured through drmTMB
# on 2026-09-05: `y ~ gempty` with `levels = c("a", "b", "c", "zz")` produced
# exactly this mismatch (4 supplied, 3 built) with no hint of the cause.
# Categorical columns are recognisable from the schema's own raw spelling,
# `"<dpar>_<column>: <level>"`.
function _bridge_count_mismatch_hint(n_supplied::Integer, n_actual::Integer,
        block::AbstractVector{String})
    n_supplied > n_actual || return ""
    coded = [name for name in block if occursin(": ", name)]
    isempty(coded) && return ""
    return ". Supplying MORE names than this fit has columns usually means a factor " *
        "level with no rows in the data reaching DRModels.jl: R's `model.matrix()` gives " *
        "such a level an all-zero column, DRModels.jl codes only the levels it observes. " *
        "The coded columns DRModels.jl built here are $(coded). Drop the unused levels " *
        "before fitting (`droplevels()` in R), or fit with `engine = \"tmb\"`"
end

function _bridge_echo_coef_labels(coef_labels, block_ranges::Vector{Pair{Symbol,UnitRange{Int}}},
        raw_names::Vector{String})
    supplied = Dict{String,Any}(String(k) => v for (k, v) in pairs(coef_labels))
    known_dpars = [String(param) for (param, _) in block_ranges]
    for dpar in keys(supplied)
        dpar in known_dpars ||
            error("drm_bridge: coef_labels supplies names for unknown dpar \"$dpar\"; " *
                  "the model has dpars: $(join(known_dpars, ", "))")
    end
    names = Vector{String}(undef, length(raw_names))
    for (param, range) in block_ranges
        dpar = String(param)
        n_actual = length(range)
        haskey(supplied, dpar) ||
            error("drm_bridge: coef_labels is missing an entry for dpar \"$dpar\" " *
                  "($n_actual fixed-effect columns; Julia names: $(raw_names[range])); " *
                  "the R side must supply names for every dpar when sending coef_labels")
        raw_dpar_labels = supplied[dpar]
        # JuliaCall unboxes a length-1 R character vector to a bare Julia
        # `String` rather than `[String]` (R list-wraps as a workaround
        # today); treat a scalar String exactly like a length-1 vector.
        # Anything else that is not a vector of strings fails closed instead
        # of silently coercing (e.g. an `Int` must not stringify into a
        # plausible-looking label) (#563 follow-up to #594).
        dpar_labels = raw_dpar_labels isa AbstractString ?
            String[raw_dpar_labels] : raw_dpar_labels
        dpar_labels isa AbstractVector && all(x -> x isa AbstractString, dpar_labels) ||
            error("drm_bridge: coef_labels[\"$dpar\"] must be a String (for one " *
                  "column) or a vector of Strings; got $(typeof(raw_dpar_labels))")
        n_supplied = length(dpar_labels)
        n_supplied == n_actual ||
            error("drm_bridge: coef_labels[\"$dpar\"] supplies $n_supplied names but the " *
                  "$dpar formula part has $n_actual fixed-effect columns " *
                  "(Julia names: $(raw_names[range])); the R side must send exactly one " *
                  "name per column" *
                  _bridge_count_mismatch_hint(n_supplied, n_actual, raw_names[range]))
        for (i, nm) in zip(range, dpar_labels)
            names[i] = "$(dpar)_$(nm)"
        end
    end
    return names
end

function _bridge_validate_coordinate_axes(blocks, names, p::Integer,
        V::Union{Nothing,AbstractMatrix})
    length(blocks) == length(names) ||
        error("drm_bridge: coefficient blocks and coefficient-name blocks differ in length")
    seen = Int[]
    for ((param, r), (name_param, ns)) in zip(blocks, names)
        param === name_param || error("drm_bridge: coefficient blocks and names disagree on parameter identity")
        length(r) == length(ns) || error("drm_bridge: coefficient-name mismatch for `$param`")
        append!(seen, r)
    end
    seen == collect(1:Int(p)) ||
        error("drm_bridge: coefficient blocks must be the ordered raw-coordinate partition 1:$p")
    V === nothing || size(V) == (p, p) ||
        error("drm_bridge: covariance matrix must be $p×$p in coefficient-coordinate order")
    return nothing
end

# Render every REGRESSION block (mu/sigma/…, mu1/mu2/…; not the `sd`/`sd_phylo`
# location-scale-scale blocks, which `_bridge_lss_public_to_raw!` handles) from
# the fit's own typed schema. Returns `(param, raw, public, vouched)` tuples in
# `form.forms` order, where `raw` equals the fitted block's `coefnames` (checked
# here, so a caller never labels columns the schema did not build) and
# `public` is the base-R spelling. `vouched` is `false` on the one fallback
# path in `_bridge_render_formula_block` where no public spelling could be
# derived and `raw` is returned in its place — a caller comparing spellings
# must skip such a block rather than compare Julia-internal names to R's.
function _bridge_rendered_regression_blocks(fit, labels::_BridgeFormulaLabels)
    out = Tuple{Symbol,Vector{String},Vector{String},Bool}[]
    form = hasproperty(fit, :formula) ? fit.formula : nothing
    form === nothing && return out
    forms = hasproperty(form, :forms) ? form.forms : Pair{Symbol,Any}[]
    block_names = Dict(p => ns for (p, ns) in fit.coefnames)
    for (param, rhs) in forms
        _bridge_lss_form_key(param) === nothing || continue
        haskey(block_names, param) || continue
        # Coupled location-scale fits build their fixed columns after removing
        # `(1 | tag | group)`. The ordinary random-effect splitter does not
        # handle that nested tag syntax; use the fitter's own projection so
        # neither the tag nor the structured group becomes a data column.
        if fit isa DrmFit && fit.nll isa LocScaleObjective && param in (:mu, :sigma)
            rhs = first(_ls_parse_coupled(rhs))
        end
        rendered = _bridge_render_formula_block(form, param, rhs, labels)
        rendered === nothing && continue
        raw, public = rendered
        vouched = !(public === raw)
        if fit isa DrmFit{<:CumulativeLogit} && param === :mu
            # Proportional-odds cutpoints absorb the location intercept.  The
            # ordinal fitter drops it after `_design` but deliberately retains
            # the user formula for prediction, so bridge label rendering must
            # make the same known family-specific projection.
            intercept = findfirst(==("(Intercept)"), raw)
            intercept === nothing || (deleteat!(raw, intercept); vouched && deleteat!(public, intercept))
        end
        expected = block_names[param]
        raw == expected || error("drm_bridge: typed schema labels do not match fitted raw columns for `$param`")
        length(public) == length(raw) || error("drm_bridge: public label width mismatch for `$param`")
        push!(out, (param, raw, public, vouched))
    end
    return out
end

# Fidelity check for an R-supplied `options["coef_labels"]` echo. The echo
# itself is positional — `_bridge_echo_coef_labels` only counts columns — so
# on its own it will happily print R's names over a design DRModels.jl built
# differently: a factor whose levels reached Julia in another order (a
# different baseline), an ordered factor R codes with `contr.poly`, a user
# `contr.sum`, or a term order the two engines disagree on. Every one of
# those has the SAME column count on both sides, so the count check passes
# and the coefficients are silently wrong under the right names (measured
# through drmTMB on 2026-09-04: an ordered factor differed by 1.25 in the
# baseline coefficient, name-identical). Here every regression block DRModels.jl
# can render itself (`_bridge_rendered_regression_blocks`) must render to
# exactly the supplied base-R names, in order; otherwise refuse, naming the
# dpar and BOTH spellings. Blocks DRModels.jl cannot render (`vouched == false`)
# and blocks with no formula counterpart (`phylocov`, `resd`, `sd`) are not
# compared — the R side names those itself and there is nothing on this side
# to compare them to.
function _bridge_check_coef_labels_fidelity(fit, labels::_BridgeFormulaLabels,
        block_ranges::Vector{Pair{Symbol,UnitRange{Int}}}, raw_names::Vector{String},
        echoed::Vector{String})
    ranges = Dict(param => range for (param, range) in block_ranges)
    for (param, raw, public, vouched) in _bridge_rendered_regression_blocks(fit, labels)
        vouched || continue
        haskey(ranges, param) || continue
        range = ranges[param]
        raw_names[range] == ["$(param)_$(r)" for r in raw] || error(
            "drm_bridge: internal error — rendered raw columns for `$param` do not align with the coefficient block")
        prefix = "$(param)_"
        supplied = String[name[nextind(name, firstindex(name), length(prefix)):end] for name in echoed[range]]
        supplied == public || error(
            "drm_bridge: coef_labels[\"$param\"] does not match the design DRModels.jl built for `$param`: " *
            "R supplied $(supplied) but DRModels.jl renders $(public) from its own model matrix " *
            "(Julia raw columns: $(raw)). The two engines disagree on the design columns — " *
            "usually a factor whose level order or contrasts differ between the R data and " *
            "what reached Julia (DRModels.jl codes every factor with treatment contrasts against " *
            "its first level, in the level order it received). Refusing to report DRModels.jl's " *
            "coefficients under R's names. Give the column an explicit, treatment-coded " *
            "level order in R before fitting (`factor(x, levels = c(...))`, not an ordered " *
            "factor or a `contrasts` attribute), or use `engine = \"tmb\"`.")
    end
    return nothing
end

# The location-scale-scale `sd_<group>` / `sdphy_<group>` formula blocks are
# rendered by `_bridge_lss_public_to_raw!`, NOT by
# `_bridge_rendered_regression_blocks`, which skips them by construction
# (`_bridge_lss_form_key(param) === nothing || continue`). So on the echo
# path above they were pasted with R's names and never compared against the
# design this side actually built, and the fidelity check that closes that
# hole for `mu`/`sigma` said nothing about them.
#
# Measured 2026-09-05, drmTMB origin/main 2fcbb0fbf against DRModels.jl aee371cc9,
# before this function existed: `bf(y ~ x + (1 | study), sigma ~ z,
# sd(study) ~ s_chr)`, where `s_chr` is a character column whose R
# locale-collated level order ("alpha", "Beta", "gamma") is not the
# code-point order Julia sorts a bare `Vector{String}` into ("Beta", "alpha",
# "gamma"). Both engines converged, both reported logLik -69.917488 (diff
# 2.98e-13) and identical coefficient names, `mu` and `sigma` agreed to
# 2.1e-11 -- and the `sd` block was off by 1.3853, with `s_chrBeta` reported
# as +0.692648 by `engine = "tmb"` and -0.692648 by `engine = "julia"`,
# the baseline having moved from "alpha" to "Beta". Declaring the column as
# a factor in R (`s_fac`) made the same fit faithful to 1.5e-10, which is
# what identifies the level order as the mechanism.
#
# Only the SINGLE-COMPONENT route is compared. The multi-IID route qualifies
# each public spelling with a `"<group>: "` prefix that the R side does not
# supply (see `_bridge_lss_public_to_raw!`), so comparing it here would
# refuse a design that is in fact faithful; it stays uncompared, as it was.
function _bridge_check_lss_coef_labels_fidelity(fit, labels::_BridgeFormulaLabels,
        raw_names::Vector{String}, echoed::Vector{String})
    form = hasproperty(fit, :formula) ? fit.formula : nothing
    form === nothing && return nothing
    forms = hasproperty(form, :forms) ? form.forms : Pair{Symbol,Any}[]
    block_names = Dict(p => ns for (p, ns) in fit.coefnames)
    position = Dict(name => i for (i, name) in enumerate(raw_names))
    for (key, rhs) in forms
        block = _bridge_lss_form_key(key)
        block === nothing && continue
        haskey(block_names, block) || continue
        rendered = _bridge_render_formula_block(form, block, rhs, labels; label_scope = key)
        rendered === nothing && continue
        raw, public = rendered
        # `public === raw` is the "nothing to vouch for" signal used by
        # `_bridge_rendered_regression_blocks`; and a block whose fitted
        # columns are not exactly this formula's is the multi-group route.
        public === raw && continue
        block_names[block] == raw || continue
        prefix = "$(block)_"
        supplied = String[]
        aligned = true
        for rawname in raw
            idx = get(position, "$(block)_$(rawname)", 0)
            if idx == 0 || !startswith(echoed[idx], prefix)
                aligned = false
                break
            end
            name = echoed[idx]
            push!(supplied, name[nextind(name, firstindex(name), length(prefix)):end])
        end
        aligned || continue
        supplied == public || error(
            "drm_bridge: coef_labels[\"$block\"] does not match the design DRModels.jl built " *
            "for the `$key` formula: R supplied $(supplied) but DRModels.jl renders $(public) " *
            "from its own model matrix (Julia raw columns: $(raw)). The two engines " *
            "disagree on the design columns of a group-level SD formula -- usually a " *
            "factor or character column whose level order differs between the R data and " *
            "what reached Julia (DRModels.jl codes every factor with treatment contrasts " *
            "against its first level, in the level order it received). Refusing to report " *
            "DRModels.jl's coefficients under R's names. Give the column an explicit, " *
            "treatment-coded level order in R before fitting (`factor(x, levels = c(...))`), " *
            "or use `engine = \"tmb\"`.")
    end
    return nothing
end

# `coefnames` records raw matrix columns.  Render public R spellings from the
# typed, schema-applied RHS instead of parsing those strings: factor levels may
# themselves contain `:`/spaces, so text substitution would corrupt them.
function _bridge_public_to_raw_coef_map(fit, labels::_BridgeFormulaLabels,
        raw_names::Vector{String})
    raw_to_public = Dict{String,String}(raw => raw for raw in raw_names)
    form = hasproperty(fit, :formula) ? fit.formula : nothing
    form === nothing && return Dict{String,String}(raw => raw for raw in raw_names)
    block_names = Dict(p => ns for (p, ns) in fit.coefnames)
    for (param, raw, public, _) in _bridge_rendered_regression_blocks(fit, labels)
        for (rawname, publicname) in zip(raw, public)
            raw_to_public["$(param)_$(rawname)"] = "$(param)_$(publicname)"
        end
    end
    _bridge_lss_public_to_raw!(raw_to_public, form, block_names, labels)
    public_to_raw = Dict{String,String}()
    for raw in raw_names
        public = raw_to_public[raw]
        haskey(public_to_raw, public) && error("drm_bridge: ambiguous public coefficient label `$public`")
        public_to_raw[public] = raw
    end
    return public_to_raw
end

_bridge_lss_form_key(param::Symbol) = startswith(String(param), "sdphy_") ? :sd_phylo :
                                      startswith(String(param), "sd_") ? :sd : nothing

function _bridge_lss_public_to_raw!(raw_to_public, form, block_names,
        labels::_BridgeFormulaLabels)
    for (key, rhs) in form.forms
        block = _bridge_lss_form_key(key)
        block === nothing && continue
        haskey(block_names, block) || continue
        group = String(key)[block === :sd ? length("sd_") + 1 : length("sdphy_") + 1:end]
        rendered = _bridge_render_formula_block(form, block, rhs, labels; label_scope = key)
        rendered === nothing && continue
        raw, public = rendered
        pnames = block_names[block]
        prefix = group * ": "
        qualified = filter(name -> startswith(name, prefix), pnames)
        if pnames == raw
            # The single-component routes store only their local sd formula
            # columns.  Multi-IID routes prefix each represented component by
            # group; unmapped scalar components intentionally remain identity.
            for (rawname, publicname) in zip(raw, public)
                raw_to_public["$(block)_$(rawname)"] = "$(block)_$(publicname)"
            end
        else
            isempty(qualified) && error(
                "drm_bridge: cannot align the `$block` group-level formula labels for `$group`")
            local_raw = String[name[nextind(name, firstindex(name), length(prefix)):end] for name in qualified]
            local_raw == raw || error("drm_bridge: grouped `$block` labels for `$group` do not match the typed schema")
            for (rawname, publicname) in zip(qualified, public)
                raw_to_public["$(block)_$(rawname)"] = "$(block)_$(prefix)$(publicname)"
            end
        end
    end
    return raw_to_public
end

function _bridge_render_formula_block(form, param::Symbol, rhs,
        labels::_BridgeFormulaLabels; label_scope::Symbol = param)
    response = if form isa DrmFormula
        form.response
    elseif form isa BivariateDrmFormula
        param in (:mu1, :sigma1, :rho12, :nu) ? form.response1 : form.response2
    else
        return nothing
    end
    # Random/structured pieces never become ordinary coefficient columns.  The
    # fixed part is what `_design` used for their associated fixed-effect block.
    fixed_rhs = try
        first(_split_ranef(rhs; allow_phylo_slope = true))   # #620: a Gaussian slope formula still labels
    catch
        rhs
    end
    schema_data = _bridge_label_schema_data(response, labels.data)
    ft = FormulaTerm(Term(response), fixed_rhs)
    ft = apply_schema(ft, schema(ft, schema_data), StatisticalModel)
    raw = String.(vec(coefnames(ft.rhs)))
    public = _bridge_public_term_labels(
        ft.rhs, _bridge_bool_column_atoms(labels.atoms, fixed_rhs, schema_data),
        _bridge_formula_symbol_order(fixed_rhs),
        get(labels.function_labels, label_scope, Dict{String,String}()))
    if length(public) != length(raw)
        _bridge_term_uses_atoms(ft.rhs, labels.atoms) && error(
            "drm_bridge: cannot render the public coefficient labels for a formula term containing a bridge materialisation")
        return raw, raw
    end
    return raw, public
end

# R codes a logical covariate as a two-level factor (levels FALSE, TRUE) under
# treatment contrasts, so its single design column is named `<sym>TRUE`;
# StatsModels keeps a `Vector{Bool}` continuous -- the SAME 0/1 column -- and
# names it `<sym>`. Render R's spelling for such columns (plain and inside
# interactions), so the coef_labels fidelity check compares like with like
# rather than refusing a design the two engines actually agree on (measured
# through drmTMB 2026-09-05: `y ~ x + flag` was refused as "flagTRUE" vs
# "flag" with max|coef diff| 2.6e-14). Scalar function terms keep their
# recorded source spelling and are not affected. Materialised atoms win.
function _bridge_bool_column_atoms(atoms::Dict{Symbol,String}, rhs, data)
    out = copy(atoms)
    for sym in keys(_bridge_formula_symbol_order(rhs))
        haskey(out, sym) && continue
        col = try
            _table_column(data, sym)
        catch
            nothing
        end
        col isa AbstractVector{Bool} || continue
        out[sym] = String(sym) * "TRUE"
    end
    return out
end

function _bridge_formula_symbol_order(rhs)
    order = Dict{Symbol,Int}()
    function visit(t)
        if hasproperty(t, :sym) && getproperty(t, :sym) isa Symbol
            sym = getproperty(t, :sym)
            haskey(order, sym) || (order[sym] = length(order) + 1)
        end
        hasproperty(t, :terms) && foreach(visit, getproperty(t, :terms))
        hasproperty(t, :args) && foreach(visit, getproperty(t, :args))
        return nothing
    end
    rhs isa Tuple ? foreach(visit, rhs) : visit(rhs)
    return order
end

function _bridge_label_schema_data(response::Symbol, data)
    raw_response = _table_column(data, response)
    y, observed = _coerce_response_column(raw_response)
    return all(observed) ? data :
           _replace_table_column(data, response, ifelse.(observed, y, 0.0))
end

function _bridge_term_uses_atoms(t, atoms::Dict{Symbol,String})
    hasproperty(t, :sym) && getproperty(t, :sym) isa Symbol &&
        haskey(atoms, getproperty(t, :sym)) && return true
    hasproperty(t, :terms) && return any(tt -> _bridge_term_uses_atoms(tt, atoms), getproperty(t, :terms))
    hasproperty(t, :args) && return any(tt -> _bridge_term_uses_atoms(tt, atoms), getproperty(t, :args))
    return false
end

function _bridge_raw_term_labels(t)
    raw = coefnames(t)
    return raw isa AbstractString ? [String(raw)] : String.(vec(raw))
end

function _bridge_one_public_term_label(t, atoms::Dict{Symbol,String},
        order::Dict{Symbol,Int}, function_labels::Dict{String,String})
    labels = _bridge_public_term_labels(t, atoms, order, function_labels)
    length(labels) == 1 || error(
        "drm_bridge: a scalar formula function cannot safely render a multi-column argument")
    return only(labels)
end

function _bridge_public_term_labels(t::StatsModels.ConstantTerm,
        atoms::Dict{Symbol,String}, order::Dict{Symbol,Int},
        function_labels::Dict{String,String} = Dict{String,String}())
    n = t.n
    return [n isa AbstractFloat ? _bridge_r_float_label(n) : string(n)]
end

function _bridge_public_term_labels(t::StatsModels.Term,
        atoms::Dict{Symbol,String}, order::Dict{Symbol,Int},
        function_labels::Dict{String,String} = Dict{String,String}())
    return [get(atoms, t.sym, String(t.sym))]
end

function _bridge_public_term_labels(t::StatsModels.FunctionTerm,
        atoms::Dict{Symbol,String}, order::Dict{Symbol,Int},
        function_labels::Dict{String,String} = Dict{String,String}())
    # Do not rewrite generic StatsModels terms that contain no bridge atom.
    # Their native raw label remains the established public spelling.  When a
    # materialised atom occurs below a scalar function, recursively render the
    # typed argument tree so `log1p(1 + __bridge_I_1)` becomes
    # `log1p(1 + I(x^2))` without changing its evaluated column.
    source_label = get(function_labels, repr(t.exorig), nothing)
    source_label === nothing || return [source_label]
    _bridge_term_uses_atoms(t, atoms) || return _bridge_raw_term_labels(t)
    args = [_bridge_one_public_term_label(arg, atoms, order, function_labels) for arg in t.args]
    f = t.f
    if f === (+)
        return [length(args) == 1 ? "+" * only(args) : join(args, " + ")]
    elseif f === (-)
        return [length(args) == 1 ? "-" * only(args) : join(args, " - ")]
    elseif f === (*)
        return [join(args, " * ")]
    elseif f === (/)
        return [join(args, "/")]
    elseif f === (^)
        return [join(args, "^")]
    end
    fname = try
        String(nameof(f))
    catch
        throw(ArgumentError(
            "drm_bridge: cannot render scalar formula function containing a bridge materialisation"))
    end
    return [fname * "(" * join(args, ", ") * ")"]
end

function _bridge_public_term_labels(t, atoms::Dict{Symbol,String},
        order::Dict{Symbol,Int} = Dict{Symbol,Int}(),
        function_labels::Dict{String,String} = Dict{String,String}())
    if t isa StatsModels.InterceptTerm
        return String.(vec(coefnames(t)))
    elseif t isa StatsModels.ContinuousTerm
        return [get(atoms, t.sym, String(t.sym))]
    elseif t isa StatsModels.CategoricalTerm
        raw = String.(vec(coefnames(t)))
        base = get(atoms, t.sym, String(t.sym))
        prefix = "$(t.sym): "
        all(startswith(name, prefix) for name in raw) || error(
            "drm_bridge: categorical schema labels for `$(t.sym)` have an unexpected spelling")
        typed = t.contrasts.coefnames
        length(typed) == length(raw) || error("drm_bridge: categorical contrast labels have an unexpected width")
        return [base * _bridge_r_factor_level_label(level) for level in typed]
    elseif t isa StatsModels.InteractionTerm
        terms = collect(t.terms)
        # R composes an interaction in the order each variable first appeared
        # in the formula.  The parser's local `x & g` order alone is therefore
        # insufficient for `g + x:g` and declared-level factor interactions.
        term_rank(tt) = hasproperty(tt, :sym) &&
                        haskey(order, getproperty(tt, :sym)) ?
                        order[getproperty(tt, :sym)] : typemax(Int)
        # `kron_insideout` must retain the original component order: it is the
        # raw model-matrix coordinate order.  Reorder only the strings inside
        # each displayed interaction tuple, otherwise two multi-column factors
        # would receive each other's public labels.
        orderperm = sortperm(term_rank.(terms))
        pieces = (_bridge_public_term_labels(tt, atoms, order, function_labels) for tt in terms)
        return String.(StatsModels.kron_insideout(
            (args...) -> join(args[orderperm], ":"), pieces...))
    elseif t isa StatsModels.MatrixTerm
        return reduce(vcat, (_bridge_public_term_labels(tt, atoms, order, function_labels) for tt in t.terms); init = String[])
    end
    raw = _bridge_raw_term_labels(t)
    _bridge_term_uses_atoms(t, atoms) && error(
        "drm_bridge: unsupported schema term containing a bridge materialisation; cannot safely render public labels")
    return raw
end

_bridge_r_factor_level_label(level::Bool) = level ? "TRUE" : "FALSE"
_bridge_r_factor_level_label(level::Integer) = string(level)
function _bridge_r_factor_level_label(level::AbstractFloat)
    # R uses the same decimal/scientific boundary for numeric factor levels as
    # for literals inside `I(...)`.  Avoid `Int(level)`: it overflows for
    # admitted finite levels such as `2e20`.
    return _bridge_r_float_label(level)
end
_bridge_r_factor_level_label(level) = String(level)

function _bridge_raw_fixef_target(fit, labels::_BridgeFormulaLabels, target)
    _, _, _, public_to_raw = _bridge_coef_vector(fit; labels = labels)
    public_full = "$(target.param)_$(target.coef)"
    raw_full = get(public_to_raw, public_full, nothing)
    raw_full === nothing && throw(ArgumentError(
        "drm_bridge_inference: no public coefficient named `fixef:$(target.param):$(target.coef)`"))
    prefix = "$(target.param)_"
    startswith(raw_full, prefix) || error("drm_bridge_inference: malformed public-to-raw coefficient map")
    raw_coef = raw_full[nextind(raw_full, firstindex(raw_full), length(prefix)):end]
    return (param = target.param, coef = raw_coef)
end

function _bridge_q4_point_export(fit; family::AbstractString)
    if !(fit.ranef isa NamedTuple) || !haskey(fit.ranef, :Sigma_a)
        return Dict{String,Any}()
    end
    Σ = Matrix{Float64}(fit.ranef.Sigma_a)
    size(Σ) == (4, 4) || return Dict{String,Any}()
    axes = haskey(fit.ranef, :axes) ? Tuple(fit.ranef.axes) :
        (:mu1, :mu2, :sigma1, :sigma2)
    length(axes) == 4 || return Dict{String,Any}()
    d = sqrt.(max.(diag(Σ), 0.0))
    R = Σ ./ (d * d')
    return Dict{String,Any}(
        "target" => "gaussian_q4_phylo",
        "dimension" => "q4",
        "family" => String(family),
        "estimator" => String(fit.estim_method),
        "axes" => String[String(axis) for axis in axes],
        "sigma_a_source" => "fit.ranef.Sigma_a",
        "sigma_a" => Σ,
        "sd" => Dict{String,Float64}(
            String(axes[i]) => Float64(d[i]) for i in eachindex(axes)
        ),
        "correlation" => R,
        "claim_boundary" => "Direct q4 point export only; no R-via-Julia q4 bridge parity, q4 REML, AI-REML, interval reliability, or interval coverage is promoted.",
    )
end

function _bridge_q2_point_export(fit; family::AbstractString = "biv_gaussian",
                                 structured_type::AbstractString = "phylo")
    export_type = fit isa DrmFit &&
                  fit.ranef isa NamedTuple &&
                  haskey(fit.ranef, :structured_type) ?
                  String(fit.ranef.structured_type) : String(structured_type)
    if fit isa DrmFit &&
       fit.ranef isa NamedTuple &&
       haskey(fit.ranef, :Sigma_a) &&
       size(fit.ranef.Sigma_a) == (2, 2)
        Σ = Matrix{Float64}(fit.ranef.Sigma_a)
        d = sqrt.(max.(diag(Σ), 0.0))
        R = Σ ./ (d * d')
        axes = haskey(fit.ranef, :axes) ? Tuple(fit.ranef.axes) : (:mu1, :mu2)
        residual_sd = Dict{String,Float64}(
            "mu1" => Float64(first(fit.scales[:sigma1])),
            "mu2" => Float64(first(fit.scales[:sigma2])),
        )
        boundary = "Direct q2 $(export_type) residual-correlation point export only for complete-response exact-Gaussian ML fixtures; R-via-Julia support is limited to route-specific q2 fixtures; no broad q2 bridge support, q2 REML, q4, AI-REML, interval reliability, or interval coverage is promoted."
        return Dict{String,Any}(
            "target" => "gaussian_q2_mu1_mu2_$(export_type)_residual_correlation",
            "dimension" => "q2",
            "family" => String(family),
            "structured_type" => export_type,
            "estimator" => String(fit.estim_method),
            "axes" => String[String(axis) for axis in axes],
            "sigma_a_source" => "fit.ranef.Sigma_a",
            "sigma_a" => Σ,
            "sd" => Dict{String,Float64}(
                String(axes[i]) => Float64(d[i]) for i in eachindex(axes)
            ),
            "correlation" => R,
            "residual_sd" => residual_sd,
            "residual_correlation" => Float64(first(fit.scales[:rho12])),
            "loglik" => Float64(loglik(fit)),
            "converged" => Bool(is_converged(fit)),
            "claim_boundary" => boundary,
        )
    end
    if !(fit isa NamedTuple) || !haskey(fit, :Λ)
        return Dict{String,Any}()
    end
    Σ = Matrix{Float64}(fit.Λ)
    size(Σ) == (2, 2) || return Dict{String,Any}()
    d = sqrt.(max.(diag(Σ), 0.0))
    R = Σ ./ (d * d')
    has_residual_correlation = haskey(fit, :residual_cov) && haskey(fit, :rho12)
    target_suffix = has_residual_correlation ?
        "residual_correlation" : "restricted_diagonal_residual"
    source = has_residual_correlation ?
        "fit_coevolution_q2_residual.Λ" : "fit_coevolution.Λ"
    boundary = has_residual_correlation ?
        "Direct q2 $(export_type) residual-correlation point export only for known-matrix complete-response exact-Gaussian ML fixtures; R-via-Julia support is limited to route-specific q2 fixtures; no broad q2 bridge support, q2 REML, q4, AI-REML, interval reliability, or interval coverage is promoted." :
        "Direct q2 $(export_type) restricted point export only for a diagonal-residual coevolution fixture; no R-via-Julia q2 bridge support, full q2 residual-correlation route, q2 REML, q4, interval reliability, or interval coverage is promoted."
    out = Dict{String,Any}(
        "target" => "gaussian_q2_mu1_mu2_$(export_type)_$(target_suffix)",
        "dimension" => "q2",
        "family" => String(family),
        "structured_type" => export_type,
        "estimator" => "ML",
        "axes" => ["mu1", "mu2"],
        "sigma_a_source" => source,
        "sigma_a" => Σ,
        "sd" => Dict{String,Float64}(
            "mu1" => Float64(d[1]),
            "mu2" => Float64(d[2]),
        ),
        "correlation" => R,
        "converged" => haskey(fit, :converged) ? Bool(fit.converged) : false,
        "claim_boundary" => boundary,
    )
    if haskey(fit, :σ_res)
        out["residual_sd"] = Dict{String,Float64}(
            "mu1" => Float64(fit.σ_res[1]),
            "mu2" => Float64(fit.σ_res[2]),
        )
    end
    if has_residual_correlation
        out["residual_correlation"] = Float64(fit.rho12)
    end
    if haskey(fit, :loglik)
        out["loglik"] = Float64(fit.loglik)
    end
    return out
end

"""
    _bridge_dpars(fit)

Per-observation distributional parameters on the **response** scale, keyed by
drmTMB dpar name (`"mu"`, `"sigma"`, …).

This is what drmTMB's post-fit surface actually consumes. `fitted_distribution()`
— the hub for `qq_plot()`, `worm_plot()`, `centile_chart()` and `exceedance()` —
builds its `d`/`p`/`q` closures from `fitted_distribution_params()`, which calls
`predict_parameters(object, dpar = <all dpars>, type = "response")` and needs one
column per dpar. The R side supplies the density machinery from its own family
tables; the only thing it cannot derive is the fitted parameter values.

Covers the in-sample case (R's `newdata = NULL`). Fresh-data prediction goes
through `predict_parameters(fit, newdata)`, which is a separate payload.

**A dpar is not `fitted()`.** For a mixture family the two differ, and feeding
the wrong one produces a wrong density *silently* because both are in range.
drmTMB's `mu` dpar for `zero_one_beta` is the **interior beta component** mean
`plogis(eta_mu)`, which it feeds to `drm_beta_shapes(mu, sigma)`; DRModels.jl stores
that as `beta_mu` and puts the *unconditional* mean
`(1 - zoi) * mu + zoi * coi` — the right answer for `fitted()` — in `means[:mu]`.
The override below repairs that one family.

Checked against drmTMB's full dpar table (`R/family-dpq.R`): every other family
DRModels.jl implements already agrees, including truncated NB2, whose `means[:mu]`
is the **untruncated** mean and so is already the correct dpar.

The other place this trap lives is zero-inflation, and it lives here in the
OPPOSITE direction. `zi ~ ...` is a MODIFIER rather than a family in DRModels.jl
(`Poisson()`/`NegBinomial2()` plus a `zi` formula part; `src/poisson.jl`,
`src/negbinomial.jl`), so the sentence that stood here -- "DRModels.jl has no
zi/hurdle families" -- was false, and false in the direction that hides a bug.
For a zero-inflated count fit `means[:mu]` holds the COUNT-COMPONENT mean
`exp(Xmu*betahat)`, which is exactly the `mu` dpar drmTMB wants, so `out["mu"]`
below needs no repair. What needs repair is the other half -- `fitted()` -- and
that is done in [`_bridge_fitted_marginal`](@ref), not here.
"""
function _bridge_dpars(fit::DrmFit)
    out = Dict{String,Vector{Float64}}()
    for (k, v) in pairs(fit.means)
        out[String(k)] = collect(float.(v))
    end
    for (k, v) in pairs(fit.scales)
        out[String(k)] = collect(float.(v))
    end
    if fit.family isa ZeroOneBeta && haskey(out, "beta_mu")
        out["mu"] = out["beta_mu"]        # interior beta mean is drmTMB's `mu`
        delete!(out, "beta_mu")           # not a drmTMB dpar name
    end
    # `trials` is per-row CONTEXT, not a dpar: drmTMB's binomial dpar set is
    # `mu` alone and beta_binomial's is `mu`/`sigma`, with
    # `fitted_distribution_params()` attaching `params\$trials` itself. It ships
    # as its own payload key (see `_bridge_trials`), not inside `dpars`.
    delete!(out, "trials")
    return out
end

"""
    _bridge_fitted_marginal(fit) -> (fitted, residuals)

`fitted()` and `residuals()` **as drmTMB defines them**, for the bridge payload
only. Returns DRModels.jl's own values unchanged for every fit except a
zero-inflated count fit.

DRModels.jl's `fitted(fit)` is `means[:mu]`, and for a zero-inflated Poisson or NB2
fit that slot deliberately holds the COUNT-COMPONENT mean `exp(Xmu*betahat)`,
because `simulate`, `marginal_parameters` and [`_bridge_dpars`](@ref) all read
the component mean from it. drmTMB's `fitted()` for `zi_poisson`/`zi_nbinom2`
is the UNCONDITIONAL mean `(1 - pi) * mu`, and its `residuals()` is
`y - fitted` (drmTMB `R/methods.R`, `drm_fitted_response`). So the same
converged model reported two different means across the two engines, silently.

Measured 2026-09-05 on drmTMB's own `tests/testthat/test-zi-nbinom2.R` fixture
(`new_zi_nbinom2_data()`, n = 1800, seed 20260613,
`bf(count ~ x + habitat, sigma ~ z, zi ~ w + habitat)`, `nbinom2()`; drmTMB
0.7.0, DRModels.jl at this pin): the two engines' coefficients agree to
4.56745752330789e-13 and their logLik to 1.72803993336856e-11, yet native
`fitted()[1:3]` was `1.313104216, 0.712847092, 2.252471245` against the
bridge's `1.45376964, 1.21616037, 2.56362484` -- max absolute disagreement
1.3665229755584 over the 1800 observations, a factor-of-`(1 - pi)` gap that no
coefficient or likelihood check can see.

Repaired HERE, on the bridge boundary, rather than in `means[:mu]`: DRModels.jl's
own `fitted`, `residuals`, `simulate`, `marginal_parameters` and dpar table
keep the component mean they are built on, and only drmTMB's contract surface
changes. Same shape as the `zero_one_beta` override in
[`_bridge_dpars`](@ref).

Selection is `haskey(fit.scales, :zi)`, which is true for exactly the two
zero-inflated count fits (`src/poisson.jl` `_fit_poisson_zi` stores
`scales[:zi]`; `src/negbinomial.jl` `_fit_negbin2_zi` stores `scales[:sigma]`
and `scales[:zi]`). Hurdle fits store `scales[:hu]` instead and are
deliberately NOT repaired here: drmTMB's hurdle mean divides by `1 - P(0)` as
well as scaling, and that route carries no bridge parity receipt yet.
"""
function _bridge_fitted_marginal(fit::DrmFit)
    fit_vals = fitted(fit)
    fit_vals isa AbstractVector || return (fit_vals, residuals(fit))
    (haskey(fit.scales, :zi) && haskey(fit.obs, :mu)) ||
        return (fit_vals, residuals(fit))
    pz = collect(float.(fit.scales[:zi]))
    mu = collect(float.(fit_vals))
    length(pz) == length(mu) || return (fit_vals, residuals(fit))
    marginal = (1 .- pz) .* mu
    return (marginal, collect(float.(fit.obs[:mu])) .- marginal)
end

"""
    _bridge_trials(fit)

Per-row binomial denominator, or `nothing` when the family has none.

`fitted_distribution_params()` attaches `params\$trials` for the `binomial` and
`beta_binomial` model types; without it the R side cannot evaluate those
densities on a Julia fit.
"""
_bridge_trials(fit::DrmFit) =
    haskey(fit.scales, :trials) ? collect(float.(fit.scales[:trials])) : nothing

"""
    _bridge_meta_parts(fit)

For a meta-analysis fit (`meta_V(v)` on `mu`), the between-study heterogeneity
`tau` and the known sampling variances `V_known`; `nothing` otherwise.

**Why this is a recovery rather than a stored field.** `gaussian_meta.jl` stores
`scales[:sigma] = sqrt(V + tau^2)` — the TOTAL per-study SD, which is what
`simulate()` needs. But drmTMB's meta `sigma` dpar is the heterogeneity ALONE,
with `V_known` supplied separately and the density forming `sqrt(V_known + sigma^2)`
itself. Emitting the total as `sigma` *and* a `V_known` would double-count the
sampling variance.

Both are recoverable exactly from what the fit already holds — `tau` from the
sigma coefficient (already the right dpar) and `V = total^2 - tau^2`, verified to
1.1e-16 — so no field, no struct change, and no change to `sigma()`'s public
contract is required.

Returns `nothing` when the sigma block carries predictors, because per-row `tau`
then needs the design matrix and `DrmFit` does not retain the data. That is a
declared boundary, not a silent approximation.
"""
function _bridge_meta_parts(fit::DrmFit)
    fit.formula isa DrmFormula || return nothing
    forms = Dict(fit.formula.forms)
    haskey(forms, :mu) || return nothing
    metav = try
        _, _, mv, _ = _split_ranef(forms[:mu]); mv
    catch
        nothing
    end
    metav === nothing && return nothing
    haskey(fit.scales, :sigma) || return nothing

    σblock = findfirst(p -> first(p) === :sigma, fit.blocks)
    σblock === nothing && return nothing
    rng = last(fit.blocks[σblock])
    length(rng) == 1 || return nothing        # sigma ~ 1 only; see docstring

    τ = exp(fit.theta[first(rng)])
    total = collect(float.(fit.scales[:sigma]))
    vknown = max.(total .^ 2 .- τ^2, 0.0)     # clamp: floating point can dip below 0
    return (tau = fill(τ, length(total)), v_known = vknown)
end

"""
    _bridge_dpars_newdata(fit, newdata)

Distributional parameters on the **response** scale for fresh rows.

R's `predict_parameters(object, newdata = ..., type = "response")` is what
`fitted_distribution(object, newdata = ...)` calls, and the `julia-engine`
vignette records the gap this closes: *fresh-data Julia prediction is currently
limited to location parameters*.

Unlike the in-sample block this reads the FORMULA rather than the stored
`means`/`scales`, so each parameter comes back as its own linear predictor
pushed through its link — which is already the dpar drmTMB wants (for
`zero_one_beta`, `mu` here is `plogis(eta_mu)`, the interior beta mean, with no
override needed).
"""
function _bridge_dpars_newdata(fit::DrmFit, newdata)
    nd = _bridge_data(newdata)
    pred = predict_parameters(fit, nd; type = :response)
    return Dict(String(k) => collect(float.(v)) for (k, v) in pairs(pred))
end

_bridge_plain(x::AbstractVector) = collect(x)
_bridge_plain(x::AbstractMatrix) = Matrix(x)
function _bridge_plain(x::AbstractDict)
    return Dict(String(k) => _bridge_plain(v) for (k, v) in pairs(x))
end
_bridge_plain(x) = x

const _BRIDGE_Q2_DIRECT_STRUCTURED_TYPES = ("phylo", "spatial", "animal", "relmat")
const _BRIDGE_Q2_KNOWN_PRECISION_PROVIDERS = (
    (structured_type = "animal", precision_source = "Ainv"),
    (structured_type = "relmat", precision_source = "Q"),
)
const _BRIDGE_Q2_DIRECT_COEFFICIENT_ORDER = (
    "mu1:(Intercept)",
    "mu1:x",
    "mu2:(Intercept)",
    "mu2:x",
    "sd_mu1:structured(group)",
    "sd_mu2:structured(group)",
    "cor_mu1_mu2:structured(group)",
)

function _bridge_q2_known_precision_provider(structured_type, precision_source)
    st = lowercase(strip(String(structured_type)))
    expected_sources = Dict(
        "animal" => "Ainv",
        "relmat" => "Q",
    )
    haskey(expected_sources, st) ||
        throw(ArgumentError(
            "drm_bridge_q2_known_precision: `structured_type` must be `animal` or `relmat`",
        ))
    source = precision_source === nothing ? expected_sources[st] :
             String(precision_source)
    source == expected_sources[st] ||
        throw(ArgumentError(
            "drm_bridge_q2_known_precision: `precision_source` for `$st` must be `$(expected_sources[st])`",
        ))
    return st, source
end

function _bridge_q2_direct_export_schema()
    return (
        :target,
        :structured_type,
        :dimension,
        :route,
        :estimator,
        :coefficient_order,
        :direct_status,
        :bridge_status,
        :unavailable_reason,
        :claim_boundary,
        :next_gate,
    )
end

function _bridge_q2_known_precision_schema()
    return (
        :target,
        :structured_type,
        :dimension,
        :route,
        :estimator,
        :input_scale,
        :precision_source,
        :direct_status,
        :bridge_status,
        :claim_boundary,
        :next_gate,
    )
end

function _bridge_q2_direct_export_status()
    coefficient_order = join(_BRIDGE_Q2_DIRECT_COEFFICIENT_ORDER, ";")
    return Tuple(
        begin
            direct_status = structured_type == "phylo" ?
                "available_residual_correlation_point_export" :
                structured_type in ("animal", "relmat") ?
                    "available_known_covariance_residual_correlation_point_export" :
                    "available_fixed_covariance_residual_correlation_fixture"
            bridge_status = "experimental"
            unavailable_reason = if structured_type == "phylo"
                "Same-target q2 phylo residual-correlation direct export and narrow R-via-Julia bridge parity fixture exist for complete-response exact-Gaussian ML."
            elseif structured_type == "spatial"
                "Direct q2 spatial evidence and the R-via-Julia bridge are limited to a fixed-covariance fixture; the range-estimating spatial route remains unsupported."
            else
                "Direct q2 $(structured_type) residual-correlation export and narrow R-via-Julia bridge parity fixture exist for known-covariance exact-Gaussian ML."
            end
            claim_boundary = if structured_type == "phylo"
                "Direct q2 phylo residual-correlation point export is fixture evidence only; R-via-Julia bridge support is narrow fixture support; no broad q2 bridge support, q2 REML, q4, AI-REML, interval reliability, or interval coverage is promoted."
            elseif structured_type == "spatial"
                "Direct q2 spatial fixed-covariance fixture evidence is not a range-estimating spatial route; R-via-Julia bridge support is narrow fixture support; no broad q2 bridge support, q2 REML, q4, AI-REML, interval reliability, or interval coverage is promoted."
            else
                "Direct q2 $(structured_type) known-covariance residual-correlation point export is fixture evidence only; R-via-Julia bridge support is narrow fixture support; no broad q2 bridge support, q2 REML, q4, AI-REML, interval reliability, or interval coverage is promoted."
            end
            next_gate = structured_type == "spatial" ?
                "Keep aggregate q2 acceptance scoped to fixed-covariance spatial fixtures; range-estimating spatial remains outside this bridge." :
                "Keep aggregate q2 acceptance scoped to complete-response exact-Gaussian ML fixtures before widening to q2 REML, q4, or interval claims."
            (
            target = "gaussian_q2_mu1_mu2_$structured_type",
            structured_type = structured_type,
            dimension = "q2",
            route = "direct_drmjl",
            estimator = "ML",
            coefficient_order = coefficient_order,
            direct_status = direct_status,
            bridge_status = bridge_status,
            unavailable_reason = unavailable_reason,
            claim_boundary = claim_boundary,
            next_gate = next_gate,
            )
        end
        for structured_type in _BRIDGE_Q2_DIRECT_STRUCTURED_TYPES
    )
end

function _bridge_q2_known_precision_status()
    return Tuple(
        begin
            st = spec.structured_type
            source = spec.precision_source
            claim_boundary = join((
                "Direct q2 $st known-precision point export is private",
                "complete-response exact-Gaussian ML fixture evidence only;",
                "`$source` is consumed as a precision matrix without implicit",
                "precision-to-covariance conversion. No R-via-Julia formula",
                "support, structured slope support, broad q2 bridge support,",
                "q2 REML, q4, AI-REML, interval reliability, or interval",
                "coverage is promoted.",
            ), " ")
            next_gate = join((
                "Use only as a Julia-side precision payload target until",
                "formula routing, structured slope support, and row-specific",
                "R-via-Julia parity evidence exist.",
            ), " ")
            (
            target = "gaussian_q2_mu1_mu2_$(st)_known_precision",
            structured_type = st,
            dimension = "q2",
            route = "direct_drmjl_private",
            estimator = "ML",
            input_scale = "precision",
            precision_source = source,
            direct_status = "available_known_precision_residual_correlation_point_export",
            bridge_status = "private_diagnostic",
            claim_boundary = claim_boundary,
            next_gate = next_gate,
            )
        end
        for spec in _BRIDGE_Q2_KNOWN_PRECISION_PROVIDERS
    )
end

function _bridge_q2_validate_direct_export_status(rows)
    schema = _bridge_q2_direct_export_schema()
    expected_targets = Set(
        "gaussian_q2_mu1_mu2_$structured_type"
        for structured_type in _BRIDGE_Q2_DIRECT_STRUCTURED_TYPES
    )
    expected_order = join(_BRIDGE_Q2_DIRECT_COEFFICIENT_ORDER, ";")
    errors = String[]
    seen = Set{String}()
    for (i, row) in enumerate(rows)
        propertynames(row) == schema ||
            push!(errors, "row $i schema does not match q2 direct export schema")
        target = String(getproperty(row, :target))
        push!(seen, target)
        target in expected_targets ||
            push!(errors, "row $i target is not registered: $target")
        getproperty(row, :dimension) == "q2" ||
            push!(errors, "row $i dimension must be q2")
        getproperty(row, :route) == "direct_drmjl" ||
            push!(errors, "row $i route must be direct_drmjl")
        getproperty(row, :estimator) == "ML" ||
            push!(errors, "row $i estimator must be ML")
        getproperty(row, :coefficient_order) == expected_order ||
            push!(errors, "row $i coefficient order does not match the q2 contract")
        if getproperty(row, :structured_type) == "phylo"
            getproperty(row, :direct_status) == "available_residual_correlation_point_export" ||
                push!(errors, "row $i phylo direct_status must record the residual-correlation point export")
        elseif getproperty(row, :structured_type) == "spatial"
            getproperty(row, :direct_status) == "available_fixed_covariance_residual_correlation_fixture" ||
                push!(errors, "row $i spatial direct_status must record the fixed-covariance fixture boundary")
            occursin("not a range-estimating spatial route", getproperty(row, :claim_boundary)) ||
                push!(errors, "row $i spatial claim boundary must reject range-estimating route support")
        else
            getproperty(row, :direct_status) == "available_known_covariance_residual_correlation_point_export" ||
                push!(errors, "row $i direct_status must record known-covariance residual-correlation point export")
            occursin("known-covariance", getproperty(row, :claim_boundary)) ||
                push!(errors, "row $i claim boundary must name known-covariance fixture evidence")
        end
        getproperty(row, :bridge_status) == "experimental" ||
            push!(errors, "row $i bridge_status must remain experimental")
        occursin("no broad q2 bridge support", getproperty(row, :claim_boundary)) ||
            push!(errors, "row $i claim boundary must reject broad q2 bridge support")
    end
    missing = setdiff(expected_targets, seen)
    isempty(missing) ||
        push!(errors, "missing q2 direct targets: $(join(sort(collect(missing)), ","))")
    return (
        ok = isempty(errors),
        errors = Tuple(errors),
        n_rows = length(rows),
        schema = schema,
    )
end

function _bridge_q2_validate_known_precision_status(rows)
    schema = _bridge_q2_known_precision_schema()
    expected = Dict(
        spec.structured_type => spec.precision_source
        for spec in _BRIDGE_Q2_KNOWN_PRECISION_PROVIDERS
    )
    expected_targets = Set(
        "gaussian_q2_mu1_mu2_$(st)_known_precision"
        for st in keys(expected)
    )
    errors = String[]
    seen = Set{String}()
    for (i, row) in enumerate(rows)
        propertynames(row) == schema ||
            push!(errors, "row $i schema does not match q2 known-precision schema")
        target = String(getproperty(row, :target))
        push!(seen, target)
        target in expected_targets ||
            push!(errors, "row $i target is not registered: $target")
        st = String(getproperty(row, :structured_type))
        haskey(expected, st) ||
            push!(errors, "row $i structured_type is not a known-precision provider")
        getproperty(row, :dimension) == "q2" ||
            push!(errors, "row $i dimension must be q2")
        getproperty(row, :route) == "direct_drmjl_private" ||
            push!(errors, "row $i route must be direct_drmjl_private")
        getproperty(row, :estimator) == "ML" ||
            push!(errors, "row $i estimator must be ML")
        getproperty(row, :input_scale) == "precision" ||
            push!(errors, "row $i input_scale must be precision")
        if haskey(expected, st)
            getproperty(row, :precision_source) == expected[st] ||
                push!(errors, "row $i precision_source does not match $st")
        end
        getproperty(row, :direct_status) == "available_known_precision_residual_correlation_point_export" ||
            push!(errors, "row $i direct_status must record known-precision residual-correlation point export")
        getproperty(row, :bridge_status) == "private_diagnostic" ||
            push!(errors, "row $i bridge_status must remain private_diagnostic")
        occursin("No R-via-Julia formula support", getproperty(row, :claim_boundary)) ||
            push!(errors, "row $i claim boundary must reject formula support")
        occursin("structured slope support", getproperty(row, :claim_boundary)) ||
            push!(errors, "row $i claim boundary must reject structured slope support")
        occursin("broad q2 bridge support", getproperty(row, :claim_boundary)) ||
            push!(errors, "row $i claim boundary must reject broad q2 bridge support")
    end
    missing = setdiff(expected_targets, seen)
    isempty(missing) ||
        push!(errors, "missing q2 known-precision targets: $(join(sort(collect(missing)), ","))")
    return (
        ok = isempty(errors),
        errors = Tuple(errors),
        n_rows = length(rows),
        schema = schema,
    )
end

const _BRIDGE_Q4_DIRECT_AXES = ("mu1", "mu2", "sigma1", "sigma2")

function _bridge_q4_direct_export_schema()
    return (
        :target,
        :axis,
        :dimension,
        :route,
        :estimator,
        :direct_sd_target,
        :sigma_a_source,
        :direct_status,
        :bridge_status,
        :inference_status,
        :claim_boundary,
        :next_gate,
    )
end

function _bridge_q4_direct_export_status()
    return Tuple(
        (
            target = "gaussian_q4_phylo_sd_$axis",
            axis = axis,
            dimension = "q4",
            route = "direct_drmjl",
            estimator = "ML",
            direct_sd_target = "sd_$axis",
            sigma_a_source = "fit.ranef.Sigma_a",
            direct_status = "available_point_target",
            bridge_status = "experimental",
            inference_status = "point_target_only",
            claim_boundary = "Direct q4 export is a status contract for point SD targets only; no R-via-Julia q4 bridge parity, q4 REML, AI-REML, interval reliability, or interval coverage is promoted.",
            next_gate = "Compare same-target native R/TMB, direct DRModels.jl, and R-via-Julia q4 point outputs before bridge parity.",
        )
        for axis in _BRIDGE_Q4_DIRECT_AXES
    )
end

function _bridge_q4_validate_direct_export_status(rows)
    schema = _bridge_q4_direct_export_schema()
    expected_targets = Set("gaussian_q4_phylo_sd_$axis" for axis in _BRIDGE_Q4_DIRECT_AXES)
    expected_sd_targets = Dict(axis => "sd_$axis" for axis in _BRIDGE_Q4_DIRECT_AXES)
    errors = String[]
    seen = Set{String}()
    for (i, row) in enumerate(rows)
        propertynames(row) == schema ||
            push!(errors, "row $i schema does not match q4 direct export schema")
        target = String(getproperty(row, :target))
        axis = String(getproperty(row, :axis))
        push!(seen, target)
        target in expected_targets ||
            push!(errors, "row $i target is not registered: $target")
        haskey(expected_sd_targets, axis) ||
            push!(errors, "row $i axis is not registered: $axis")
        getproperty(row, :dimension) == "q4" ||
            push!(errors, "row $i dimension must be q4")
        getproperty(row, :route) == "direct_drmjl" ||
            push!(errors, "row $i route must be direct_drmjl")
        getproperty(row, :estimator) == "ML" ||
            push!(errors, "row $i estimator must be ML")
        getproperty(row, :direct_sd_target) == get(expected_sd_targets, axis, "") ||
            push!(errors, "row $i direct_sd_target does not match axis")
        getproperty(row, :sigma_a_source) == "fit.ranef.Sigma_a" ||
            push!(errors, "row $i sigma_a_source must be fit.ranef.Sigma_a")
        getproperty(row, :direct_status) == "available_point_target" ||
            push!(errors, "row $i direct_status must be available_point_target")
        getproperty(row, :bridge_status) == "experimental" ||
            push!(errors, "row $i bridge_status must remain experimental")
        getproperty(row, :inference_status) == "point_target_only" ||
            push!(errors, "row $i inference_status must remain point_target_only")
        occursin("no R-via-Julia q4 bridge parity", getproperty(row, :claim_boundary)) ||
            push!(errors, "row $i claim boundary must reject bridge parity")
        occursin("interval coverage", getproperty(row, :claim_boundary)) ||
            push!(errors, "row $i claim boundary must reject interval coverage")
    end
    missing = setdiff(expected_targets, seen)
    isempty(missing) ||
        push!(errors, "missing q4 direct targets: $(join(sort(collect(missing)), ","))")
    return (
        ok = isempty(errors),
        errors = Tuple(errors),
        n_rows = length(rows),
        schema = schema,
    )
end

function _bridge_first_param_row(rows, param::Symbol)
    for row in rows
        row.param === param && return row
    end
    throw(ArgumentError("drm_bridge_inference: result has no `$param` row"))
end

# The bridge returns one selected row even when a profile call produced several.
# Describe that row's endpoint diagnostics, not an unrelated row's aggregate
# failure. Older/stored results without per-row stats retain a conservative
# aggregate fallback. Infinite endpoints alone do not imply optimization failure.
function _bridge_profile_outcome(result, row)
    selected = filter(s -> s.param === row.param && s.coef == row.coef, result.stats)
    if isempty(selected)
        return result.failed > 0 ?
            (status="profile_failed", message="profile solve failed; per-row diagnostics unavailable") :
            (status="profile", message="profile_result completed")
    end
    s = only(selected)
    endpoint = nothing
    if hasproperty(result, :endpoint_diagnostics)
        matching = filter(d -> d.param === row.param && d.coef == row.coef,
                          result.endpoint_diagnostics)
        isempty(matching) || (endpoint = only(matching))
    end
    if s.lower_endpoint_failed || s.upper_endpoint_failed
        arms = String[]
        for arm in (:lower, :upper)
            getproperty(s, Symbol(arm, :_endpoint_failed)) || continue
            reason = Symbol(arm, :_nuisance_reason)
            method = Symbol(arm, :_nuisance_method)
            fallback = Symbol(arm, :_nuisance_fallback)
            detail = String(arm)
            # Location-scale profiles distinguish root-search failure from the
            # nuisance optimizer's state. Keep this selected-row detail in the
            # message that the R bridge exposes as `profile.message`.
            if endpoint !== nothing && hasproperty(endpoint, arm)
                diagnostic = getproperty(endpoint, arm)
                detail *= " (endpoint=" * string(diagnostic.reason)
                detail *= "; candidate=" * string(diagnostic.candidate)
                detail *= "; residual=" * string(diagnostic.residual) * ")"
            end
            if hasproperty(s, reason)
                detail *= " (nuisance=" * string(getproperty(s, reason))
                hasproperty(s, method) && (detail *= "; " * string(getproperty(s, method)))
                hasproperty(s, fallback) && (detail *= "; fallback=" * string(getproperty(s, fallback)))
                detail *= ")"
            end
            push!(arms, detail)
        end
        return (status="profile_failed",
            message="profile endpoint solve failed: " * join(arms, " and "))
    end
    if s.lower_unbounded || s.upper_unbounded
        return (status="profile", message="profile did not cross threshold within searched range")
    end
    return (status="profile", message="profile_result completed")
end

function _bridge_inference_flatten(row; method::AbstractString,
        status::AbstractString, attempted::Integer, used::Integer,
        failed::Integer, elapsed::Real, threaded::Bool,
        worker_threads::Integer, julia_threads::Integer,
        blas_threads::Integer, message::AbstractString)
    # DRModels.jl#631 backstop: a failed-status row must never carry a bound at all.
    # The profile branch above raises before reaching here; this catches any
    # future status that pairs an infinite endpoint with a non-"profile" status.
    (status == "profile_failed" && !(isfinite(row.lower) && isfinite(row.upper))) &&
        throw(ArgumentError(
            "drm_bridge_inference: refusing to return an infinite bound for " *
            "`$(row.param):$(row.coef)` under status `$status` — $message"))
    return Dict{String,Any}(
        "method" => String(method),
        "param" => String(row.param),
        "coef" => String(row.coef),
        "estimate" => row.estimate,
        "lower" => row.lower,
        "upper" => row.upper,
        "status" => String(status),
        "message" => String(message),
        "attempted" => Int(attempted),
        "used" => Int(used),
        "failed" => Int(failed),
        "elapsed" => Float64(elapsed),
        "threaded" => Bool(threaded),
        "worker_threads" => Int(worker_threads),
        "julia_threads" => Int(julia_threads),
        "blas_threads" => Int(blas_threads),
    )
end

# Bivariate q=4 inference for the bridge: confidence intervals for the among-axis SDs
# (sd_mu1, sd_mu2, sd_sigma1, sd_sigma2) — these are boundary variance components, so the
# right tools are profile (default) and bootstrap, NOT Wald:
#   method = "profile"   -> profile_sigma_a  (hessian-free profile-likelihood CIs; a
#                           collapsed axis returns lower = 0, the honest no-signal interval)
#   method = "bootstrap" -> bootstrap_sigma_a (parametric percentile CIs + correlations)
#   method = "wald"      -> unavailable (the among-axis boundary Hessian is singular)
function _bridge_bivariate_inference(fit, dat, method::AbstractString;
                                     B::Integer, level::Real, seed)
    if method == "bootstrap"
        rng = seed === nothing ? Random.default_rng() :
              Random.MersenneTwister(Int(seed))
        result = bootstrap_result(fit; data = dat, B = Int(B), level = level,
                                  rng = rng, failures = :warn, check_converged = false)
        return _bridge_inference_flatten_multi(
            result.summary;
            method = "bootstrap",
            status = result.used >= 2 ? "bootstrap" : "bootstrap_unavailable",
            attempted = result.attempted, used = result.used, failed = result.failed,
            elapsed = result.elapsed,
            message = "$(result.used)/$(result.attempted) successful refits")
    elseif method == "profile"
        # PROFILE-likelihood CIs for the among-axis SDs — hessian-free, so valid exactly
        # where the boundary Hessian is singular (a collapsed axis returns lower = 0).
        # `fit` already carries the profile-ready stash (re.prob / re.Sigma_a / re.Q_cond
        # from gaussian_bivariate.jl), so no re-fit is needed.
        elapsed = @elapsed result = profile_sigma_a(fit; level = level)
        rows = result.summary
        return _bridge_inference_flatten_multi_profile(
            rows;
            method = "profile",
            status = "profile",
            attempted = length(rows), used = length(rows), failed = 0,
            elapsed = elapsed,
            message = "profile_sigma_a (hessian-free profile-likelihood CIs)")
    elseif method == "wald"
        throw(ArgumentError("drm_bridge_inference: `wald` CIs are not available for the " *
            "bivariate q=4 phylogenetic fit's among-axis SDs (boundary variance components — " *
            "the Hessian is singular at a collapsed axis); use method = \"profile\" (default) " *
            "or method = \"bootstrap\""))
    end
    throw(ArgumentError("drm_bridge_inference: unsupported method `$method`"))
end

# Multi-row payload for the bivariate route: param/coef/estimate/std_error/lower/
# upper come back as equal-length vectors so the R side reads them as a data.frame.
function _bridge_inference_flatten_multi(rows; method::AbstractString,
        status::AbstractString, attempted::Integer, used::Integer,
        failed::Integer, elapsed::Real, message::AbstractString)
    return Dict{String,Any}(
        "method" => String(method),
        "multi" => true,
        "param" => String[String(r.param) for r in rows],
        "coef" => String[String(r.coef) for r in rows],
        "estimate" => Float64[Float64(r.estimate) for r in rows],
        "std_error" => Float64[Float64(r.std_error) for r in rows],
        "lower" => Float64[Float64(r.lower) for r in rows],
        "upper" => Float64[Float64(r.upper) for r in rows],
        "status" => String(status),
        "message" => String(message),
        "attempted" => Int(attempted),
        "used" => Int(used),
        "failed" => Int(failed),
        "elapsed" => Float64(elapsed),
    )
end

# Profile rows carry (param, coef, estimate, lower, upper, deviance_floor, bounded) — NO
# std_error (a likelihood-ratio interval, not a Wald one), and `upper` may be Inf on a
# flat/collapsed axis. Emit std_error => NaN and carry the honest `bounded` flag so the R
# data.frame keeps the same columns as the bootstrap path.
function _bridge_inference_flatten_multi_profile(rows; method::AbstractString,
        status::AbstractString, attempted::Integer, used::Integer,
        failed::Integer, elapsed::Real, message::AbstractString)
    return Dict{String,Any}(
        "method" => String(method),
        "multi" => true,
        "param" => String[String(r.param) for r in rows],
        "coef" => String[String(r.coef) for r in rows],
        "estimate" => Float64[Float64(r.estimate) for r in rows],
        "std_error" => Float64[NaN for _ in rows],
        "lower" => Float64[Float64(r.lower) for r in rows],
        "upper" => Float64[Float64(r.upper) for r in rows],
        "bounded" => Bool[Bool(r.bounded) for r in rows],
        "status" => String(status),
        "message" => String(message),
        "attempted" => Int(attempted),
        "used" => Int(used),
        "failed" => Int(failed),
        "elapsed" => Float64(elapsed),
    )
end
