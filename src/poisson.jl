# poisson.jl — Poisson family for count responses. Log link on the mean
# (λ = exp(Xμβ)); no residual dispersion parameter (see the negative-binomial
# family for overdispersed counts). Fixed and random effects, maximum likelihood.
# Mirrors drmTMB's
# `poisson`. The log-pmf is written out explicitly (y·log λ − λ − log y!) so the
# fit needs no `Distributions.Poisson` — that name is left to Distributions.

"""
    Poisson()

Poisson response family for counts: log link on the mean `μ` (so `μ` coefficients
act on `log λ`). No residual scale parameter. Mirrors `drmTMB`'s `poisson` family.

!!! note
    `DRModels.Poisson` shadows `Distributions.Poisson`; if you need the distribution
    too (e.g. to simulate), qualify it as `Distributions.Poisson`.

```julia
fit = drm(bf(@formula(y ~ x)), Poisson(); data = dat)
fitted(fit)        # fitted counts λ = exp(Xβ̂), on the response scale

# Regress the SD of an ordinary Poisson random intercept on a group-level
# predictor. Each value of z must be constant within its group g.
fit_sd = drm(bf(@formula(y ~ x + (1 | g)), @formula(sd(g) ~ z)),
             Poisson(); data = dat)
length(predict_parameters(fit_sd, dat)[:sd])  # one value per observation

# A known-exposure offset (#727, drmTMB twin gap): `offset(log_exposure)`'s
# coefficient is fixed at 1, so λ = exposure · exp(Xβ̂). Fixed-effects-only mean, or
# (#941) with one ordinary random intercept — the standard rate GLMM:
fit_off = drm(bf(@formula(y ~ x + offset(log_exposure))), Poisson(); data = dat)
fit_rate = drm(bf(@formula(y ~ x + offset(log_exposure) + (1 | g))), Poisson(); data = dat)
fit_rate_tmb = drm(bf(@formula(y ~ x + offset(log_exposure) + (1 | g))), Poisson();
                   data = dat, marginal = :Laplace)   # drmTMB's Laplace objective

fit_phy = drm(bf(@formula(y ~ x + phylo(1 | species))), Poisson();
              data = dat, tree = tr, se = false)

# A predictor-dependent SD for one animal slope-only mean coefficient.
# `sd(animal, animal) ~ habitat` models the coefficient SD by animal;
# the row-level random contribution is `x * b_animal`.
fit_animal_sd = drm(
    bf(@formula(y ~ x + animal(0 + x | animal)),
       @formula(sd(animal, animal) ~ habitat)),
    Poisson(); data = dat, A = A)

# Experimental: Poisson random-intercept variational (ELBO) marginal.
# Default remains `marginal = :LA` (adaptive GHQ, 5 nodes per group, on an ordinary `(1 | g)`). `loglik` on a VA fit is an ELBO.
fit_va = drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = dat, marginal = :VA)

# TMB-convention Laplace on an ordinary `(1 | g)`, as native drmTMB fits it (ML only).
fit_lap = drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = dat, marginal = :Laplace)

# Opt-in Cox–Reid restricted estimation. ML is the default.
fit_reml = drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = dat, method = :REML)
estimation_method(fit_reml)   # :REML

# Opt-in 1-D Liu–Pierce AGHQ for `(1 | g)` only, with `nAGQ` nodes. The default
# `:LA` is already per-group adaptive GHQ with 5 nodes. k=1 ≡ 1-point Laplace; this
# identity does not establish parameter recovery. `:REML` is not available with `:AGHQ`.
fit_aghq = drm(bf(@formula(y ~ x + (1 | g))), Poisson();
               data = dat, marginal = :AGHQ, nAGQ = 5)
fit_aghq.marginal   # :AGHQ

fit_phy_reml = drm(bf(@formula(y ~ x + phylo(1 | species))), Poisson();
                   data = dat, tree = tr, se = false, method = :REML)
```

For an ordinary random-intercept route, the group SD follows
log(s_g) = Z_g α and uses adaptive Gauss–Hermite quadrature. One independent
numeric random slope is also supported. Sparse-Laplace routes support a
predictor-dependent SD for one phylogenetic or related-matrix mean intercept,
and an animal mean intercept or slope-only coefficient. For an animal slope,
`sd(animal, animal) ~ z` models the SD of the coefficient multiplying `x`;
each row contributes `xᵢ b_g` to the log mean. The SD predictors must be
constant within animal. The animal SD multiplies the supplied covariance `A`:
the marginal animal effect covariance is `D A D`, so non-unit diagonal values
in `A` affect the marginal SD. Ordinary IID intercept and slope SD routes,
and structured phylogenetic, related-matrix, and animal SD routes, accept a
fixed `offset(...)` in the mean formula. For an animal intercept, callers may instead
supply a positive-definite precision matrix with `Ainv`; Julia uses it directly
as `Q` in the latent covariance `D Q⁻¹ D`. Its rows and columns must follow the
first-seen animal order. The non-SD Poisson animal-slope route uses the same
covariance scaling. Correlated blocks, zero-inflation/hurdle terms, and REML
for animal slopes are not supported with these routes. Profile and
bootstrap methods are available as fit-level plumbing; their interval coverage
has not been calibrated.

# Restricted (Cox–Reid) estimation

`method = :REML` is **opt-in — ML remains the default**. It maximises the Cox–Reid
adjusted profile likelihood `ℓ_ML − ½·log|I_ββ|` on two Poisson routes:

- a scalar random intercept `(1 | g)` integrated by per-group adaptive GHQ (5 nodes)
- phylogenetic / `relmat` / `animal` / precomputed-spatial Laplace
  (`_fit_poisson_general_laplace`)

On the Gaussian route this correction is exactly Patterson–Thompson REML, so the
mechanism is anchored rather than ad hoc.

!!! warning "Limited evidence for phylogenetic models"
    A 16-tip / 12-seed Poisson phylo cell over-corrected under Cox–Reid
    (ML +8.18%, CR +17.41%). Do not read those percentages as a bias-sign
    headline or a reason to prefer `:REML` on trees. Recovery on larger trees
    has not been established. This is a small exploratory result, not a general
    performance claim.

!!! warning "It over-corrects when clusters are plentiful"
    On a Poisson `(1 | g)` cell with true `σ_b = 0.6` and 6 observations per cluster,
    ML was **−12.4%** at `G = 10` where Cox–Reid was **−1.8%** — but by `G = 40` ML was
    **+1.4%** and Cox–Reid **+4.4%**. Reach for it when clusters are few, not by habit.
    That asymmetry is why ML is the default.

Crossed intercepts, correlated slopes `(1 + x | g)`, coordinate-spatial with
estimated range `ρ`, fixed-effects-only, `zi`/`hu`, `marginal = :VA`, and
`marginal = :AGHQ` still error on `method = :REML` rather than silently
returning an ML fit. REML
log-likelihoods are not comparable across different fixed-effect structures, so
do not use them for model selection over `μ`.
"""
struct Poisson end

using SpecialFunctions: loggamma

_logfactorial(k::Integer) = k < 2 ? 0.0 : loggamma(Float64(k) + 1.0)   # log k!, O(1) (#1246); 0 for k = 0, 1

# ---- offset() on the Poisson mean (#727, drmTMB twin gap) ------------------
# `offset(x)` is a marker function, exactly like `phylo`/`relmat`/`meta_V`: the
# `@formula` macro captures `offset(log_exposure)` as a `FunctionTerm` and
# `_extract_offset` (below) intercepts it before the mean design matrix is
# built, so it never becomes an ordinary estimated covariate. Its coefficient
# is fixed at 1 (an offset, not a regressor): `η = Xμβ + offset`.
"""
    offset(x)

Formula marker for a known offset in the mean formula, for example
`y ~ x + offset(log_exposure)`. The term enters the mean's linear predictor with
its coefficient fixed at 1 (`η = Xμβ + offset`); it is not estimated, as in
drmTMB. Every univariate family applies it on the fixed-effects mean (for
`Gaussian()` that is exactly the fit of `y - offset`). Some families also apply it
with random effects: `Poisson()` with one ordinary `(1 | g)` (`marginal = :LA`,
`:AGHQ` or `:Laplace`); `NegBinomial2()` with `zi`/`hu` and with one ordinary
`(1 | g)` (`:LA` or `:Laplace`); `Gamma()` and `Beta()` with one `(1 | g)` under
`marginal = :Laplace`. Every other route raises an `ArgumentError` that names
it. An offset in any formula other than the mean (`sigma`, `nu`, `zi`, ...) is
refused, as drmTMB refuses it. At most one `offset(...)` term is allowed.
`predict(fit, newdata)` needs the offset column in `newdata`. Outside a formula,
`offset(x)` returns `x` unchanged.
"""
offset(x) = x

# Split an `offset(...)` marker off a (already `_split_ranef`-stripped) rhs,
# returning `(rhs_without_offset, offset_term_or_nothing)`. At most one
# `offset(...)` term is admitted, matching drmTMB (`y ~ x + offset(log_exposure)`).
function _extract_offset(rhs)
    terms = rhs isa Tuple ? collect(rhs) : Any[rhs]
    fixed = Any[]
    offset_term = nothing
    for t in terms
        if t isa FunctionTerm && t.f === offset
            offset_term === nothing ||
                error("drm: only one `offset(...)` term is supported")
            length(t.args) == 1 ||
                error("drm: `offset(...)` takes exactly one term, e.g. `offset(log_exposure)`")
            offset_term = t.args[1]
        else
            push!(fixed, t)
        end
    end
    fixed_rhs = isempty(fixed) ? ConstantTerm(1) :
                length(fixed) == 1 ? fixed[1] : Tuple(fixed)
    return fixed_rhs, offset_term
end

# Evaluate an `offset(...)` term (a plain column reference or an arbitrary
# numeric transform, e.g. `offset(log(exposure))`) to a `Float64` vector, reusing
# the same schema/modelcols machinery `_design` uses for ordinary terms — but with
# NO `StatisticalModel` context, so no implicit intercept column is inserted.
function _offset_vector(offset_term, data)
    sch = schema(offset_term, data)
    t = apply_schema(offset_term, sch)
    v = modelcols(t, data)
    v isa AbstractMatrix && size(v, 2) == 1 && (v = vec(v))
    v isa AbstractVector ||
        error("drm: `offset(...)` must resolve to a single numeric column")
    _require_finite_inputs(; offset=v)
    return Float64.(v)
end

# #938: linear predictor with an optional known offset (coefficient fixed at 1).
# `nothing` is the offset-free path, the same arithmetic as before (`X * β`).
_eta_with_offset(X, β, ::Nothing) = X * β
_eta_with_offset(X, β, offset) = X * β .+ offset
_add_offset(η, ::Nothing) = η
_add_offset(η, offset) = η .+ offset

# #938: move an intercept start value by the mean offset, so a fit with a large
# offset (a log exposure of 5, say) does not start far from its optimum. Only a
# start value: the objective is unchanged. Applied only when column 1 of `X` is
# an intercept.
function _offset_start(θβ0, X, offset)
    θ = collect(Float64, θβ0)
    if offset !== nothing && !isempty(θ) && size(X, 2) ≥ 1 && all(==(1.0), @view X[:, 1])
        θ[1] -= sum(offset) / length(offset)
    end
    return θ
end

# #938: the offset vector of a route that applies one (`nothing` without an
# offset), with the one-value-per-row and finiteness checks.
function _offset_for_fit(fam, offset_term, data, n::Integer)
    offset_term === nothing && return nothing
    off = _offset_vector(offset_term, data)
    length(off) == n ||
        throw(ArgumentError("$(_family_label(fam)): `offset(...)` must have one value per observation"))
    return off
end

# #938: a route that cannot hold the offset at coefficient 1 refuses it by name,
# instead of letting `_design` build it as an estimated column.
_refuse_offset_route(fam, route) = throw(ArgumentError(
    "$(_family_label(fam)): `offset(...)` in the mean formula is not supported with $route; " *
    "the offset (coefficient fixed at 1) is applied on the fixed-effects mean" *
    (fam isa NegBinomial2 ? ", with `zi`/`hu`, and with one ordinary `(1 | g)` " *
                            "(`marginal = :LA` or `:Laplace`)" :
     (fam isa Gamma || fam isa Beta) ? " and with one ordinary `(1 | g)` under " *
                                       "`marginal = :Laplace`" : "") *
    ". Fit without the offset, or add the term as an ordinary covariate if a free " *
    "slope is intended. See issue #938."))

# FX3b: `se = false` is recorded on the fit (`DrmFit.uncertainty`), so
# `convergence_status` does not read the skipped (NaN) covariance as a failure.
@drm_entry function drm(f::DrmFormula, fam::Poisson; se::Bool = true, kwargs...)
    return _withuncertainty(_drm_poisson_fit(f, fam; se = se, kwargs...), se)
end

function _drm_poisson_fit(f::DrmFormula, fam::Poisson; data, tree = nothing, K = nothing,
             A = nothing, Ainv = nothing, coords = nothing,
             g_tol::Real = 1e-8, se::Bool = true,
             marginal::Symbol = :LA, method = nothing,
             nAGQ::Union{Nothing,Int} = nothing, ids = nothing)
    f = _default_structured_q2_sd(f, fam)   # bare structured (1 + x | g): default `sd(…, coef(·)) ~ 1`
    _drm_entry_guard(f, fam, data; tree = tree, K = K, A = A, Ainv = Ainv, coords = coords)   # #1027 #1022 #1010 #938
    # `method` is the ML/REML selector. LA/VA/AGHQ is `marginal` (Q1 / #136 / #448).
    # `:REML` is the opt-in Cox–Reid restricted objective, admitted on Poisson
    # `(1 | g)` GHQ under `:LA` (#443) and on `_fit_poisson_general_laplace`
    # callers (phylo / relmat / animal / precomputed spatial; #450). `:REML` ×
    # `:AGHQ` is not wired. Every other route calls `_reject_reml_route` rather
    # than ignoring the request.
    meth = _reject_method_as_marginal(fam, method; allow_reml = true)
    reml = meth === :REML
    if !isempty(_sd_parts(f)) || !isempty(_sdphylo_parts(f))
        A = _resolve_relatedness_labels(f, data, A, ids; coords = coords)   # #957: match `A` by label (SD routes too)
        coords = _resolve_coords_labels(f, data, coords, ids)   # #957
        return _drm_poisson_sd_regression(f, fam; data, tree, K, A, Ainv, coords,
            g_tol, se, marginal, reml, nAGQ)   # `nothing` -> route default
    end
    # `nAGQ = nothing` resolves to 5 nodes on the plain `(1 | g)` routes; the
    # predictor-dependent-SD routes above choose their own default.
    nAGQ = nAGQ === nothing ? 5 : nAGQ
    Ainv === nothing || throw(ArgumentError(
        "Poisson `Ainv` is currently supported for predictor-dependent animal SD regression only."))
    _scalar_laplace_requested(marginal) &&     # Arc 2: TMB-convention Laplace, ordinary (1 | g)
        return _drm_ordinary_laplace(f, fam; data = data, tree = tree, K = K, A = A,
                                     coords = coords, g_tol = g_tol, se = se, method = method)

    missing_fit = _fit_observed_response_rows(f, data) do data_observed
        drm(f, fam; data = data_observed, tree = tree, K = K, A = A,
            coords = coords, g_tol = g_tol, se = se, marginal = marginal,
            method = method, nAGQ = nAGQ, ids = ids)
    end
    missing_fit !== nothing && return missing_fit
    A = _resolve_relatedness_labels(f, data, A, ids; coords = coords)   # #957: match `A` by label
    coords = _resolve_coords_labels(f, data, coords, ids)   # #957: match `coords` by site label

    marg = _marginal_method(marginal)                     # :LA (default), :VA (#136), :AGHQ (#448)
    isva = marg isa Variational
    isaghq = marg isa AGHQ
    _lss_only_gaussian_guard(f, fam)   # #544: refuse, never silently drop, sd() parts
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st, _, _, animal_slope =
        _split_ranef(rhs[:mu]; allow_animal_slope = true)
    mv === nothing ||
        error("Poisson() does not support meta_V markers")
    fixed_mu, offset_term = _extract_offset(fixed_mu)     # #727: offset(log_exposure), coefficient fixed at 1
    y, Xμ, nmμ = _design(f.response, fixed_mu, data)
    off = offset_term === nothing ? zeros(length(y)) : _offset_vector(offset_term, data)
    length(off) == length(y) ||
        error("Poisson(): `offset(...)` must have one value per observation")
    all(yi -> yi ≥ 0 && isinteger(yi), y) ||
        error("Poisson() requires non-negative integer counts as the response")
    # #941: an offset is admitted on the fixed-effects-only mean (#727) and on ONE
    # ordinary random intercept `(1 | g)` under `:LA` (GHQ), `:AGHQ` and `:Laplace`
    # (the standard rate model, drmTMB `poisson()` + offset + `(1 | g)`). Every other
    # route still refuses it rather than dropping it or estimating its coefficient.
    if offset_term !== nothing
        offset_re_ok = st === nothing && length(re) == 1 &&
                       _re_kind(re[1][1])[1] === :intercept && !isva
        if st !== nothing || haskey(rhs, :zi) || haskey(rhs, :hu) || (!isempty(re) && !offset_re_ok)
            error("Poisson(): `offset(...)` is currently supported on the fixed-effects-only " *
                  "mean and with a single ordinary random intercept `(1 | g)` " *
                  "(`marginal = :LA`, `:AGHQ` or `:Laplace`); not with a structured marker, " *
                  "a random slope, crossed/multiple random effects, `zi`, `hu`, or `marginal = :VA`")
        end
    end
    if st !== nothing
        isva && _va_reject(fam, "a phylogenetic/structured random effect")
        isaghq && _aghq_reject(fam, "a phylogenetic/structured random effect")
        # Phylo / relmat / animal / precomputed spatial share `_fit_poisson_general_laplace`
        # and admit opt-in Cox–Reid (#450). Probe Cell D is not a recovery headline —
        # the docstring warning is the honesty ledger. Coordinate spatial uses the fixed
        # native kernel and takes the same route (opt-in Cox–Reid included). AGHQ stays
        # rejected here (#448): 1-D Liu–Pierce is Poisson `(1 | g)` only.
        isempty(re) ||
            error("Poisson() structured effects cannot be combined with ordinary random effects yet")
        (haskey(rhs, :zi) || haskey(rhs, :hu)) &&
            error("Poisson() structured effects cannot be combined with `zi`/`hu` yet")
        kind, grp = st
        labels = getproperty(data, grp)
        if animal_slope !== nothing
            kind === :animal || throw(ArgumentError(
                "Poisson() supports a structured random slope only for `animal(0 + x | group)`."))
            isempty(re) || throw(ArgumentError(
                "Poisson() animal random slopes cannot yet be combined with ordinary random effects."))
            (haskey(rhs, :zi) || haskey(rhs, :hu)) && throw(ArgumentError(
                "Poisson() animal random slopes cannot yet be combined with `zi` or `hu`."))
            fixed_mu, slope_offset = _extract_offset(fixed_mu)
            slope_offset === nothing || throw(ArgumentError(
                "Poisson() animal random slopes do not yet support `offset(...)`."))
            _, Xslope_mu, nmslope_mu = _design(f.response, fixed_mu, data)
            slope_loading = _poisson_animal_slope_loading(data, animal_slope, length(y))
            C = _poisson_structured_cov(kind, grp, K, A, coords)
            fit = _fit_poisson_animal_slope_laplace(
                fam, y, Xslope_mu, slope_loading, C, labels, nmslope_mu,
                grp, g_tol; se, reml, normalize = false)
            return _withformula(fit, f)
        end
        if kind === :phylo
            tree === nothing && error("phylo(1 | $grp) needs `tree = …`")
            return _withformula(_fit_poisson_phylo_laplace(fam, y, Xμ, labels, tree, nmμ, grp, g_tol;
                                                          se = se, reml = reml), f)
        elseif kind === :spatial && K === nothing && coords !== nothing
            # Coordinate spatial intercept: drmTMB's FIXED-range kernel, exp(-d / r) + 1e-6 I with
            # r the median positive pairwise distance (`drm_spatial_coords_precision`), exactly as
            # the Gamma route (wave 8). The range is not a parameter, so the fit has the same
            # parameters as drmTMB (beta, log sd) and the same logLik at the optimum. It used to
            # estimate the range (a sixth parameter), which gained 1.23 logLik on the wave-10
            # Poisson cell but is a different model from the one drmTMB fits.
            _, Gsp = _group_index(labels)
            Csp = _spatial_covariance_from_coords(grp, Gsp, coords, nothing)
            return _withformula(_fit_poisson_relmat_laplace(fam, y, Xμ, Csp, labels, nmμ, grp, g_tol;
                                                           se = se, reml = reml,
                                                           normalize = true), f)
        elseif kind === :relmat || kind === :animal || kind === :spatial
            # General user-supplied PD covariance C on the mean intercept
            # (relatedness / animal model / PRECOMPUTED spatial). Reuses the phylo
            # sparse-Laplace spine with the tree precision swapped for C⁻¹ (#167).
            C = _poisson_structured_cov(kind, grp, K, A, coords)
            return _withformula(_fit_poisson_relmat_laplace(fam, y, Xμ, C, labels, nmμ, grp, g_tol;
                                                           se = se, reml = reml,
                                                           normalize = kind === :spatial), f)
        else
            error("Poisson() supports phylo/relmat/animal/spatial(1 | group) among structured markers")
        end
    end
    if !isempty(re)                                       # random intercept (1|g) → GHQ marginal
        isva && (haskey(rhs, :zi) || haskey(rhs, :hu)) &&
            _va_reject(fam, "`zi`/`hu` combined with a random effect")
        (haskey(rhs, :zi) || haskey(rhs, :hu)) &&
            error("Poisson() random effects cannot be combined with `zi`/`hu` yet")
        if length(re) > 1                                 # (1|g)+(1|h)+… crossed/multiple intercepts → sparse Laplace
            isva && _va_reject(fam, "crossed/multiple random intercepts")
            reml && _reject_reml_route(fam, "crossed/multiple random intercepts")
            # Independent intercepts + slopes `(1 | g) + (0 + x | g)`: TMB-convention Laplace
            # (drmTMB's layout), the shared dense route of `indep_slopes_laplace.jl`.
            if _has_indep_slopes(re)
                isaghq && _aghq_reject(fam, "independent random slopes `(0 + x | g)` (Laplace only)")
                return _withformula(_fit_indep_re_laplace(fam, Val(:poisson), y, Xμ, nothing, nothing,
                                                          _indep_slope_comps(re, data, length(y)),
                                                          nmμ, String[], String[], g_tol), f)
            end
            all(_re_kind(r[1])[1] === :intercept for r in re) ||
                error("Poisson() supports multiple random effects only as crossed/nested intercepts " *
                      "`(1 | g) + (1 | h)` or independent intercepts and slopes `(1 | g) + (0 + x | g)`; " *
                      "a correlated `(1 + x | g)` must be the only random-effect term")
            comps = map(re) do r
                grp = r[2]; gidx, G = _group_index(getproperty(data, grp); col = grp)
                (ones(length(y)), gidx, G, String(grp))
            end
            isaghq && return _withformula(_fit_poisson_crossed_aghq(fam, y, Xμ, comps, nmμ, g_tol; se = se), f)   # #761
            return _withformula(_fit_poisson_crossed_laplace(fam, y, Xμ, comps, nmμ, g_tol; se = se), f)
        end
        (rk, var) = _re_kind(re[1][1]); grp = re[1][2]; gidx, G = _group_index(getproperty(data, grp); col = grp)
        if rk === :intercept                              # (1 | g) → GHQ-32 :LA, VA (#136), or 1-D AGHQ (#448)
            # The one route where `method = :REML` punches through (#443). VA is an ELBO,
            # not a likelihood, so a restricted correction of it is not defined.
            # `:REML` × `:AGHQ` is not wired this slice (Cox–Reid on the AGHQ
            # marginal is deferred B).
            reml && isva && _reject_reml_route(fam, "`marginal = :VA` (the ELBO is not a likelihood)")
            reml && isaghq && throw(ArgumentError(
                "drm (Poisson): `method = :REML` is not available with `marginal = :AGHQ`. " *
                "The restricted (Cox–Reid) correction is not wired to the AGHQ marginal this slice. " *
                "Use `method = :ML` (the default) with `marginal = :AGHQ`, or `method = :REML` with `marginal = :LA`."))
            isva && return _withformula(_withmarginal(
                _fit_poisson_ranef_va(fam, y, Xμ, gidx, G, nmμ, grp, g_tol), :VA), f)
            isaghq && return _withformula(_withmarginal(
                _fit_poisson_ranef_aghq(fam, y, Xμ, gidx, G, nmμ, grp, g_tol; nAGQ = nAGQ,
                                        offset = off), :AGHQ), f)
            return _withformula(
                _fit_poisson_ranef(fam, y, Xμ, gidx, G, nmμ, grp, g_tol; reml = reml, offset = off), f)
        elseif rk === :corr                               # (1 + x | g) → 2-D GHQ (NOT AGHQ)
            isva && _va_reject(fam, "a correlated random slope `(1 + x | g)`")
            isaghq && _aghq_reject(fam, "a correlated random slope `(1 + x | g)` (12² tensor GHQ is not AGHQ)")
            # A 2-D per-cluster effect is not scalar-per-cluster: out of the certified cell.
            reml && _reject_reml_route(fam, "a correlated random slope `(1 + x | g)`")
            xs = Float64.(getproperty(data, var))
            return _withformula(_fit_poisson_corr_ranef(fam, y, Xμ, xs, gidx, G, nmμ, grp, g_tol), f)
        elseif rk === :slope                              # (0 + x | g) → TMB-convention Laplace (independent route)
            isva && _va_reject(fam, "an independent random slope `(0 + x | g)`")
            isaghq && _aghq_reject(fam, "an independent random slope `(0 + x | g)` (Laplace only)")
            reml && _reject_reml_route(fam, "an independent random slope `(0 + x | g)`")
            _require_finite_data_column(data, var)
            return _withformula(_fit_indep_re_laplace(fam, Val(:poisson), y, Xμ, nothing, nothing,
                                                      _indep_slope_comps(re, data, length(y)),
                                                      nmμ, String[], String[], g_tol), f)
        else
            error("Poisson() supports `(1 | g)`, `(0 + x | g)` or `(1 + x | g)` random effects on the mean")
        end
    end
    isva && _va_reject(fam, "no random intercept (fixed-effects-only / zi / hu)")
    isaghq && _aghq_reject(fam, "no random intercept (fixed-effects-only / zi / hu)")
    # No variance component ⇒ nothing for a restricted objective to correct.
    reml && _reject_reml_route(fam, "no random effect (fixed-effects-only / `zi` / `hu`)")
    haskey(rhs, :zi) && haskey(rhs, :hu) &&
        error("`zi` and `hu` cannot both be specified (zero-inflation vs hurdle)")
    if haskey(rhs, :zi)                                   # zero-inflated Poisson
        _, Xzi, nmzi = _design(f.response, rhs[:zi], data)
        return _withformula(_fit_poisson_zi(fam, y, Xμ, Xzi, nmμ, nmzi, g_tol), f)
    end
    if haskey(rhs, :hu)                                   # hurdle Poisson
        _, Xhu, nmhu = _design(f.response, rhs[:hu], data)
        return _withformula(_fit_poisson_hu(fam, y, Xμ, Xhu, nmμ, nmhu, g_tol), f)
    end
    return _withformula(_fit_poisson(fam, y, Xμ, nmμ, g_tol; offset = off), f)
end

# Poisson random-effect SD regression. For group g,
#   log(λ_i) = X_i β + w_i b_g,  b_g ~ Normal(0, exp(2 Z_g α)),
# where w_i is 1 for an intercept or the numeric slope predictor.
# The fixed-effect SD design is evaluated once per group, as in the Binomial
# route. This Poisson slice admits one ordinary iid intercept or independent slope;
# structured and correlated-block routes must not fall through to this fitter.
function _drm_poisson_sd_regression(f::DrmFormula, fam::Poisson; data,
                                    tree = nothing, K = nothing, A = nothing,
                                    Ainv = nothing, coords = nothing,
                                    g_tol::Real = 1e-8,
                                    se::Bool = true, marginal::Symbol = :LA,
                                    reml::Bool = false,
                                    nAGQ::Union{Nothing,Int} = nothing)
    reml && throw(ArgumentError(
        "Poisson `sd(group) ~ …` currently supports maximum likelihood only; " *
        "the Cox–Reid correction is not implemented for SD-regression coefficients."))
    laplace_fit = _laplace_sd_route(f, fam, Val(:poisson), marginal; data, g_tol)
    laplace_fit === nothing || return laplace_fit
    marginal === :LA || throw(ArgumentError(
        "Poisson `sd(group) ~ …` currently supports `marginal = :LA` only."))
    nAGQ === nothing || nAGQ >= 1 || throw(ArgumentError("nAGQ must be ≥ 1; got $nAGQ"))
    targets = _sd_target_parts(f)
    q2ps = _phylo_spatial_q2_block(f)
    q2ps === nothing || return _drm_count_phylo_spatial_q2_sd(f, fam, q2ps...;
        data, tree, K, A, Ainv, coords, g_tol, se, nAGQ)
    ordq2 = _ordinary_q2_sd_block(f, fam)
    ordq2 === nothing || return _drm_count_ordinary_q2_sd(f, fam, ordq2;
        data, tree, K, A, Ainv, coords, g_tol, se, nAGQ)
    if any(item -> item.target.level === :relmat, targets)
        Ainv === nothing || throw(ArgumentError(
            "Poisson related-matrix SD regression uses `K = …`, not animal precision `Ainv = …`."))
        return _drm_poisson_relmat_sd_regression(
            f, fam; data, tree, K, A, coords, g_tol, se)
    end
    if any(item -> item.target.level === :spatial, targets)
        return _drm_poisson_spatial_sd_regression(
            f, fam; data, tree, K, A, Ainv, coords, g_tol, se)
    end
    if any(item -> item.target.level === :animal, targets)
        return _drm_poisson_animal_sd_regression(
            f, fam; data, tree, K, A, Ainv, coords, g_tol, se)
    end
    Ainv === nothing || throw(ArgumentError(
        "Poisson `Ainv` is currently supported for predictor-dependent animal SD regression only."))
    !isempty(_sdphylo_parts(f)) && return _drm_poisson_phylo_sd_regression(
        f, fam; data, tree, K, A, coords, g_tol, se)
    # Independent terms / crossed intercepts with `sd()` formulas: TMB-convention Laplace with a
    # group-level log-SD regression per term (`indep_slopes_laplace.jl`).
    indep_shape = _indep_sd_shape(f)
    indep_shape === nothing ||
        return _drm_indep_sd_laplace(f, fam, Val(:poisson), indep_shape; data, g_tol)
    length(targets) == 1 || throw(ArgumentError(
        "Poisson random-effect SD regression requires exactly one SD target."))
    target = only(targets)
    (target.target.level === :iid && target.target.dpar === :mu &&
     target.target.block === nothing) || throw(ArgumentError(
        "Poisson SD regression currently targets one ordinary mean random-effect coefficient."))
    (tree === nothing && K === nothing && A === nothing && Ainv === nothing && coords === nothing) ||
        throw(ArgumentError("Poisson `sd(group) ~ …` currently supports one ordinary " *
            "independent random intercept or numeric slope; structured covariance inputs are not used."))
    isempty(_sdphylo_parts(f)) || throw(ArgumentError(
        "Poisson `sd(group, phylogenetic) ~ …` is not implemented by the ordinary random-intercept route."))
    sgrp, srhs = target.target.group, target.rhs
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    mv === nothing || throw(ArgumentError("Poisson `sd($sgrp)` cannot use meta_V()."))
    st === nothing || throw(ArgumentError(
        "Poisson `sd($sgrp)` currently requires one ordinary `(1 | $sgrp)` or `(0 + x | $sgrp)` effect."))
    length(re) == 1 || throw(ArgumentError(
        "Poisson `sd($sgrp)` currently requires exactly one ordinary independent random effect."))
    kind, slope = _re_kind(re[1][1])
    kind in (:intercept, :slope) || throw(ArgumentError(
        "Poisson `sd($sgrp)` supports `(1 | $sgrp)` or one independent numeric slope `(0 + x | $sgrp)`; correlated blocks are not implemented on this route."))
    member = kind === :intercept ? :intercept : slope
    target.target.member in (nothing, member) || throw(ArgumentError(
        "Poisson SD target `coef($(target.target.member))` does not match fitted coefficient `$(member)`."))
    grp = re[1][2]
    grp === sgrp || throw(ArgumentError(
        "Poisson `sd($sgrp)` must name the random-effect grouping variable `$grp`."))
    haskey(rhs, :hu) && throw(ArgumentError(
        "Poisson random-effect SD regression cannot yet be combined with `hu`."))
    fixed_mu, offset_term = _extract_offset(fixed_mu)
    for (pname, r) in f.forms
        (pname in (:mu, Symbol("sd_", sgrp)) || haskey(f.sd_targets, pname) ||
         (pname === :zi && haskey(rhs, :zi))) && continue
        pname === :sigma && r === ConstantTerm(1) && continue
        throw(ArgumentError("Poisson `sd($sgrp)` is a mean-only model with no residual sigma formula."))
    end
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "Poisson random-effect SD regression does not yet support missing responses; " *
        "drop incomplete rows before fitting so the group SD design remains aligned."))

    y, Xmu, nmmu = _design(f.response, fixed_mu, data)
    off = offset_term === nothing ? zeros(Float64, length(y)) :
          _offset_vector(offset_term, data)
    length(off) == length(y) || throw(ArgumentError(
        "Poisson `offset(...)` must have one value per observation."))
    all(yi -> yi >= 0 && isinteger(yi), y) || throw(ArgumentError(
        "Poisson() requires non-negative integer counts as the response"))
    gidx, G = _group_index(getproperty(data, grp))
    Zg, nmsd = _sd_group_design(f.response, srhs, data, gidx, G, grp;
                                schema_cache = f.schema_cache, schema_key = :sd)
    size(Zg, 2) <= G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
        "Poisson `sd($grp)` group-level design is rank deficient; use predictors that vary independently across groups."))
    w = if kind === :intercept
        ones(length(y))
    else
        hasproperty(data, slope) || throw(ArgumentError(
            "Poisson random-slope predictor `$slope` is absent from data."))
        raw = getproperty(data, slope)
        raw isa AbstractVector{<:Real} || throw(ArgumentError(
            "Poisson random-slope predictor `$slope` must be numeric."))
        all(isfinite, raw) || throw(ArgumentError(
            "Poisson random-slope predictor `$slope` must contain finite values."))
        Float64.(raw)
    end
    fit = if haskey(rhs, :zi)
        offset_term === nothing || throw(ArgumentError(
            "Poisson ZIP SD regression does not yet support `offset(...)`."))
        kind === :slope && target.target.member !== member && throw(ArgumentError(
            "Poisson ZIP random-slope SD regression requires an explicit `coef($slope)` selector, for example `sd($grp, coef($slope)) ~ predictors`."))
        kind === :slope && !any(!iszero, w) && throw(ArgumentError(
            "Poisson ZIP random-slope predictor `$slope` must contain at least one non-zero value."))
        _, Xzi, nmzi = _design(f.response, rhs[:zi], data)
        _fit_poisson_zi_sd_regression(fam, y, Xmu, Xzi, Zg, w, gidx, G,
            nmmu, nmzi, nmsd, grp, g_tol;
            nAGQ = nAGQ === nothing ? _ZIP_SD_AGHQ_K : nAGQ, se = se)
    else
        _fit_poisson_sd_regression(fam, y, Xmu, Zg, gidx, G,
            nmmu, nmsd, grp, g_tol;
            nAGQ = nAGQ === nothing ? _SD_IID_AGHQ_K : nAGQ, se = se, w = w, offset = off)
    end
    return _withformula(fit, f)
end

function _drm_poisson_relmat_sd_regression(f::DrmFormula, fam::Poisson; data,
                                           tree = nothing, K = nothing, A = nothing,
                                           coords = nothing, g_tol::Real = 1e-8,
                                           se::Bool = true)
    targets = _sd_target_parts(f)
    relmat_block = _relmat_intercept_slope(get(Dict(f.forms), :mu, nothing);
                                           allow_label = true)
    relmat_block !== nothing && return _drm_poisson_relmat_intercept_slope_sd(
        f, fam, targets, relmat_block; data, tree, K, A, coords, g_tol, se)
    length(targets) == 1 || throw(ArgumentError(
        "Poisson relmat SD regression requires exactly one random-effect SD formula."))
    target = only(targets)
    descriptor = target.target
    (descriptor.level === :relmat && descriptor.dpar === :mu &&
     descriptor.block === nothing) ||
        throw(ArgumentError(
            "Poisson relmat SD regression targets the sole structured mean coefficient."))
    tree === nothing && A === nothing && coords === nothing || throw(ArgumentError(
        "Poisson relmat SD regression uses `K = …` only; do not also supply `tree`, `A`, or `coords`."))
    K === nothing && throw(ArgumentError(
        "relmat(1 | $(descriptor.group)) needs the relatedness/covariance matrix `K = …`"))

    rhs = Dict(f.forms)
    # `relmat(0 + x | g)` is a slope-only coefficient: the SD formula models the
    # SD of the coefficient multiplying `x`, exactly as for the animal route.
    fixed_mu, re, metav, structured, _, _, relmat_slope =
        _split_ranef(rhs[:mu]; allow_relmat_slope = true)
    metav === nothing || throw(ArgumentError("Poisson relmat SD regression cannot use meta_V()."))
    structured !== nothing && first(structured) === :relmat || throw(ArgumentError(
        "Poisson `sd($(descriptor.group), relmat)` requires `relmat(1 | $(descriptor.group))` " *
        "or the slope-only `relmat(0 + x | $(descriptor.group))`."))
    isempty(re) || throw(ArgumentError(
        "Poisson relmat SD regression currently requires one structured coefficient without additional random effects."))
    expected_member = relmat_slope === nothing ? :intercept : relmat_slope
    descriptor.member in (nothing, expected_member) || throw(ArgumentError(
        "Poisson relmat SD target does not match fitted coefficient `$(expected_member)`."))
    last(structured) === descriptor.group || throw(ArgumentError(
        "Poisson relmat SD target must match the group in `relmat(1 | $(last(structured)))`."))
    (haskey(rhs, :zi) || haskey(rhs, :hu)) && throw(ArgumentError(
        "Poisson relmat SD regression cannot yet be combined with `zi` or `hu`."))
    fixed_mu, offset_term = _extract_offset(fixed_mu)
    for (pname, form) in f.forms
        (pname === :mu || pname === target.key) && continue
        pname === :sigma && form === ConstantTerm(1) && continue
        throw(ArgumentError(
            "Poisson relmat SD regression supports only `mu` and one related-matrix SD formula."))
    end
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "Poisson relmat SD regression requires observed responses; remove incomplete rows before fitting."))

    y, Xmu, nmmu = _design(f.response, fixed_mu, data)
    off = _poisson_sd_offset(offset_term, data, length(y))
    all(yi -> yi >= 0 && isinteger(yi), y) || throw(ArgumentError(
        "Poisson() requires non-negative integer counts as the response"))
    group = descriptor.group
    labels = getproperty(data, group)
    # relmat K enters RAW (native drmTMB never rescales K).
    Q, gidx = _general_cov_setup(K, labels; normalize = false)
    G = size(Q, 1)
    Zg, nmsd = _sd_group_design(f.response, target.rhs, data, gidx, G, group;
        schema_cache = f.schema_cache, schema_key = target.key)
    size(Zg, 2) <= G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
        "Poisson relmat SD target has a rank-deficient group-level design."))
    Zobs = Zg[gidx, :]
    loading = relmat_slope === nothing ? ones(Float64, length(y)) :
        _poisson_animal_slope_loading(data, relmat_slope, length(y); label = "relmat")
    fit = _fit_poisson_covariance_sd_laplace(
        fam, y, Xmu, Zobs, Zg, gidx, Q, nmmu, nmsd, group, g_tol;
        se, weight = loading, offset = off,
        context = "sparse-Laplace Poisson relmat SD regression")
    return _withformula(fit, f)
end

# Poisson intercept SD regression on a coordinate spatial covariance:
# `y ~ … + spatial(1 | g)` with `sd(g, spatial) ~ z` and `coords = …`. The covariance is
# the native fixed kernel exp(-d / median positive distance) + 1e-6 I (RAW, no
# rescaling), then the relmat SD Laplace objective is reused unchanged.
function _drm_poisson_spatial_sd_regression(f::DrmFormula, fam::Poisson; data,
                                            tree = nothing, K = nothing, A = nothing,
                                            Ainv = nothing, coords = nothing,
                                            g_tol::Real = 1e-8, se::Bool = true)
    who = "Poisson spatial SD regression"
    targets = _sd_target_parts(f)
    length(targets) == 1 || throw(ArgumentError(
        "$who requires exactly one random-effect SD formula."))
    target = only(targets)
    descriptor = target.target
    (descriptor.level === :spatial && descriptor.dpar === :mu &&
     descriptor.block === nothing && descriptor.member in (nothing, :intercept)) ||
        throw(ArgumentError("$who targets the structured mean intercept only."))
    (tree === nothing && K === nothing && A === nothing && Ainv === nothing) || throw(ArgumentError(
        "$who uses `coords = …` only; do not also supply `tree`, `K`, `A` or `Ainv`."))
    group = descriptor.group
    coords === nothing && throw(ArgumentError("spatial(1 | $group) needs `coords = …`"))
    rhs = Dict(f.forms)
    fixed_mu, re, metav, structured = _split_ranef(rhs[:mu])
    metav === nothing || throw(ArgumentError("$who cannot use meta_V()."))
    structured !== nothing && first(structured) === :spatial && last(structured) === group ||
        throw(ArgumentError("Poisson `sd($group, spatial)` requires `spatial(1 | $group)` in the location formula."))
    isempty(re) || throw(ArgumentError(
        "$who currently requires one spatial intercept without additional random effects."))
    (haskey(rhs, :zi) || haskey(rhs, :hu)) && throw(ArgumentError(
        "$who cannot yet be combined with `zi` or `hu`."))
    fixed_mu, offset_term = _extract_offset(fixed_mu)
    for (pname, form) in f.forms
        (pname === :mu || pname === target.key) && continue
        pname === :sigma && form === ConstantTerm(1) && continue
        throw(ArgumentError("$who supports only `mu` and one spatial SD formula."))
    end
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "$who requires observed responses; remove incomplete rows before fitting."))
    y, Xmu, nmmu = _design(f.response, fixed_mu, data)
    off = _poisson_sd_offset(offset_term, data, length(y))
    all(yi -> yi >= 0 && isinteger(yi), y) || throw(ArgumentError(
        "Poisson() requires non-negative integer counts as the response"))
    labels = getproperty(data, group)
    gidx0, G0 = _group_index(labels)
    C = _gaussian_spatial_sd_kernel(coords, G0, group)
    Q, gidx = _general_cov_setup(C, labels; normalize = false)
    G = size(Q, 1)
    Zg, nmsd = _sd_group_design(f.response, target.rhs, data, gidx, G, group;
        schema_cache = f.schema_cache, schema_key = target.key)
    size(Zg, 2) <= G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
        "$who: the SD group-level design is rank deficient; use predictors that vary independently across groups."))
    fit = _fit_poisson_covariance_sd_laplace(
        fam, y, Xmu, Zg[gidx, :], Zg, gidx, Q, nmmu, nmsd, group, g_tol;
        se, offset = off, context = "sparse-Laplace Poisson spatial SD regression")
    return _withformula(fit, f)
end

function _drm_poisson_animal_sd_regression(f::DrmFormula, fam::Poisson; data,
                                           tree = nothing, K = nothing, A = nothing,
                                           Ainv = nothing, coords = nothing,
                                           g_tol::Real = 1e-8,
                                           se::Bool = true)
    targets = _sd_target_parts(f)
    animal_block = _animal_intercept_slope(get(Dict(f.forms), :mu, nothing);
                                           allow_label = true)
    if animal_block !== nothing
        return _drm_poisson_animal_intercept_slope_sd(
            f, fam, targets, animal_block; data, tree, K, A, Ainv, coords,
            g_tol, se)
    end
    length(targets) == 1 || throw(ArgumentError(
        "Poisson animal SD regression requires exactly one random-effect SD formula."))
    target = only(targets)
    descriptor = target.target
    (descriptor.level === :animal && descriptor.dpar === :mu &&
     descriptor.block === nothing) ||
        throw(ArgumentError(
            "Poisson animal SD regression targets one structured mean coefficient."))
    tree === nothing && K === nothing && coords === nothing || throw(ArgumentError(
        "Poisson animal SD regression uses `A = …` or `Ainv = …`; do not also supply `tree`, `K`, or `coords`."))
    (A === nothing) == (Ainv === nothing) && throw(ArgumentError(
        "Poisson animal SD regression requires exactly one of `A = …` or `Ainv = …`."))

    rhs = Dict(f.forms)
    fixed_mu, re, metav, structured, _, _, animal_slope =
        _split_ranef(rhs[:mu]; allow_animal_slope = true)
    metav === nothing || throw(ArgumentError("Poisson animal SD regression cannot use meta_V()."))
    structured !== nothing && first(structured) === :animal || throw(ArgumentError(
        "Poisson `sd($(descriptor.group), animal)` requires an animal random effect for `$(descriptor.group)`."))
    isempty(re) || throw(ArgumentError(
        "Poisson animal SD regression currently requires one structured coefficient without additional random effects."))
    expected_member = animal_slope === nothing ? :intercept : animal_slope
    descriptor.member in (nothing, expected_member) || throw(ArgumentError(
        "Poisson animal SD target does not match fitted coefficient `$(expected_member)`."))
    last(structured) === descriptor.group || throw(ArgumentError(
        "Poisson animal SD target must match the group in `animal(... | $(last(structured)))`."))
    (haskey(rhs, :zi) || haskey(rhs, :hu)) && throw(ArgumentError(
        "Poisson animal SD regression cannot yet be combined with `zi` or `hu`."))
    fixed_mu, offset_term = _extract_offset(fixed_mu)
    for (pname, form) in f.forms
        (pname === :mu || pname === target.key) && continue
        pname === :sigma && form === ConstantTerm(1) && continue
        throw(ArgumentError(
            "Poisson animal SD regression supports only `mu` and one animal SD formula."))
    end
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "Poisson animal SD regression requires observed responses; remove incomplete rows before fitting."))

    y, Xmu, nmmu = _design(f.response, fixed_mu, data)
    off = _poisson_sd_offset(offset_term, data, length(y))
    all(yi -> yi >= 0 && isinteger(yi), y) || throw(ArgumentError(
        "Poisson() requires non-negative integer counts as the response"))
    group = descriptor.group
    labels = getproperty(data, group)
    Q, gidx = Ainv === nothing ? _general_cov_setup(A, labels; normalize = false) :
        _poisson_animal_precision_setup(Ainv, labels, group)
    G = size(Q, 1)
    Zg, nmsd = _sd_group_design(f.response, target.rhs, data, gidx, G, group;
        schema_cache = f.schema_cache, schema_key = target.key)
    size(Zg, 2) <= G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
        "Poisson animal SD target has a rank-deficient group-level design."))
    Zobs = Zg[gidx, :]
    loading = animal_slope === nothing ? ones(Float64, length(y)) :
        _poisson_animal_slope_loading(data, animal_slope, length(y))
    fit = _fit_poisson_covariance_sd_laplace(
        fam, y, Xmu, Zobs, Zg, gidx, Q, nmmu, nmsd, group, g_tol;
        se, weight = loading, offset = off,
        context = "sparse-Laplace Poisson animal SD regression")
    return _withformula(fit, f)
end

function _poisson_animal_precision_setup(Ainv::AbstractMatrix, labels, group::Symbol)
    gidx, G = _group_index(labels)
    size(Ainv, 1) == size(Ainv, 2) == G || throw(ArgumentError(
        "animal `Ainv` must be a $(G)×$(G) precision matrix ordered by the first-seen levels of `$(group)`."))
    Q = sparse(Float64.(Ainv))
    all(isfinite, nonzeros(Q)) || throw(ArgumentError(
        "animal `Ainv` must contain only finite values."))
    asymmetry = norm(Q - transpose(Q), Inf)
    scale = max(norm(Q, Inf), 1.0)
    asymmetry <= 1e-10 * scale || throw(ArgumentError(
        "animal `Ainv` must be symmetric."))
    Q = sparse(Symmetric((Q + transpose(Q)) / 2))
    qchol = cholesky(Symmetric(Q); check = false)
    issuccess(qchol) || throw(ArgumentError(
        "animal `Ainv` must be positive definite."))
    return Q, gidx
end

function _drm_poisson_phylo_sd_regression(f::DrmFormula, fam::Poisson; data,
                                          tree, K = nothing, A = nothing,
                                          coords = nothing, g_tol::Real = 1e-8,
                                          se::Bool = true)
    (K === nothing && A === nothing && coords === nothing) || throw(ArgumentError(
        "Poisson phylogenetic SD regression takes its covariance from `tree`; do not also supply `K`, `A`, or `coords`."))
    sdpp = _sdphylo_parts(f)
    length(sdpp) == 1 || throw(ArgumentError(
        "Poisson phylogenetic SD regression requires exactly one `sd(group, phylogenetic) ~ …` formula."))
    isempty(_sd_parts(f)) || throw(ArgumentError(
        "Poisson phylogenetic SD regression cannot yet be combined with ordinary `sd(group)` formulas."))
    sgrp, srhs = only(sdpp)
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    mv === nothing || throw(ArgumentError("Poisson phylogenetic SD regression cannot use meta_V()."))
    st !== nothing && first(st) === :phylo || throw(ArgumentError(
        "Poisson `sd($sgrp, phylogenetic)` requires `phylo(1 | $sgrp)` in the location formula."))
    tree === nothing && throw(ArgumentError("phylo(1 | $sgrp) needs `tree = …`"))
    isempty(re) || throw(ArgumentError(
        "Poisson `sd($sgrp, phylogenetic)` currently requires a phylogenetic random intercept without additional random effects."))
    last(st) === sgrp || throw(ArgumentError(
        "Poisson `sd($sgrp, phylogenetic)` must name the group in `phylo(1 | $(last(st)))`."))
    (haskey(rhs, :zi) || haskey(rhs, :hu)) && throw(ArgumentError(
        "Poisson phylogenetic SD regression cannot yet be combined with `zi` or `hu`."))
    fixed_mu, offset_term = _extract_offset(fixed_mu)
    for (pname, r) in f.forms
        (pname === :mu || pname === Symbol("sdphy_", sgrp)) && continue
        pname === :sigma && r === ConstantTerm(1) && continue
        throw(ArgumentError(
            "Poisson phylogenetic SD regression supports only `mu` and one phylogenetic SD formula."))
    end
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "Poisson phylogenetic SD regression requires observed responses; remove incomplete rows before fitting."))

    y, Xmu, nmmu = _design(f.response, fixed_mu, data)
    off = _poisson_sd_offset(offset_term, data, length(y))
    all(yi -> yi >= 0 && isinteger(yi), y) || throw(ArgumentError(
        "Poisson() requires non-negative integer counts as the response"))
    labels = getproperty(data, sgrp)
    _, tip_index, G = _lss_phylo_group_index(tree, labels, sgrp)
    group_index, Gdata = _group_index(labels)
    Gdata == G || error("phylogenetic group indexing changed unexpectedly")
    Zg, nmsd = _sd_group_design(f.response, srhs, data, group_index, Gdata, sgrp;
                                schema_cache = f.schema_cache,
                                schema_key = :sd_phylo)
    tip_group = zeros(Int, G)
    @inbounds for i in eachindex(tip_index)
        ti, gi = tip_index[i], group_index[i]
        tip_group[ti] == 0 || tip_group[ti] == gi || throw(ArgumentError(
            "Poisson phylogenetic SD predictors must be constant within each tree tip."))
        tip_group[ti] = gi
    end
    all(!iszero, tip_group) || throw(ArgumentError(
        "Poisson phylogenetic SD regression requires SD predictors for every tree tip."))
    Ztip = Matrix{Float64}(Zg[tip_group, :])
    Zobs = Ztip[tip_index, :]
    return _withformula(_fit_poisson_phylo_sd_laplace(
        fam, y, Xmu, Zobs, labels, tree, nmmu, nmsd, sgrp, g_tol;
        se, offset = off), f)
end

function _poisson_sd_offset(offset_term, data, n::Int)
    off = offset_term === nothing ? zeros(Float64, n) :
          _offset_vector(offset_term, data)
    length(off) == n || throw(ArgumentError(
        "Poisson `offset(...)` must have one value per observation."))
    return off
end

function _poisson_phylo_sd_mode(y, Xmu, Zobs, leaf_node, Q, beta, alpha;
                                offset = zeros(length(y)),
                                weight = ones(length(y)),
                                b0 = nothing, maxiter::Int = 60,
                                tol::Real = 1e-10)
    q = size(Q, 1)
    b = b0 === nothing ? zeros(q) : copy(b0)
    length(offset) == length(y) || throw(ArgumentError(
        "Poisson `offset(...)` must have one value per observation."))
    eta0 = Xmu * beta .+ offset
    length(weight) == length(y) || throw(ArgumentError(
        "Poisson structured random-effect loading must have one value per response."))
    all(isfinite, weight) || throw(ArgumentError(
        "Poisson structured random-effect loading must contain only finite values."))
    loading = Float64.(weight) .* exp.(Zobs * alpha)
    all(isfinite, loading) || return b, nothing, false
    function joint(bb)
        total = 0.5 * dot(bb, Q * bb)
        @inbounds for i in eachindex(eta0)
            eta = eta0[i] + loading[i] * bb[leaf_node[i]]
            total += exp(eta) - y[i] * eta + _logfactorial(round(Int, y[i]))
        end
        total
    end
    for _ in 1:maxiter
        grad = Q * b
        diagH = zeros(Float64, q)
        @inbounds for i in eachindex(eta0)
            li = leaf_node[i]
            eta = eta0[i] + loading[i] * b[li]
            mu = exp(eta)
            grad[li] += loading[i] * (mu - y[i])
            diagH[li] += loading[i]^2 * mu
        end
        H = Q + spdiagm(0 => diagH)
        ch = cholesky(Symmetric(H); check = false)
        issuccess(ch) || return b, ch, false
        step = ch \ grad
        norm(step) <= tol * (1 + norm(b)) && return b, ch, true
        if norm(step) <= 1e-3 * (1 + norm(b))
            b .-= step
            continue
        end
        f0 = joint(b)
        a = 1.0
        accepted = false
        while a >= 1e-5
            trial = b .- a .* step
            if joint(trial) <= f0
                b = trial
                accepted = true
                break
            end
            a *= 0.5
        end
        accepted || return b, ch, false
    end
    return b, nothing, false
end

function _poisson_phylo_sd_laplace(y, Xmu, Zobs, leaf_node, Q, logdetQ,
                                   theta; offset = zeros(length(y)),
                                   weight = ones(length(y)), b0 = nothing)
    pmu = size(Xmu, 2)
    beta = @view theta[1:pmu]
    alpha = @view theta[pmu+1:end]
    b, ch, ok = _poisson_phylo_sd_mode(y, Xmu, Zobs, leaf_node, Q,
                                        beta, alpha; offset, weight, b0 = b0)
    ok || return 1e18, b, false
    length(weight) == length(y) || throw(ArgumentError(
        "Poisson structured random-effect loading must have one value per response."))
    all(isfinite, weight) || throw(ArgumentError(
        "Poisson structured random-effect loading must contain only finite values."))
    loading = Float64.(weight) .* exp.(Zobs * alpha)
    length(offset) == length(y) || throw(ArgumentError(
        "Poisson `offset(...)` must have one value per observation."))
    eta0 = Xmu * beta .+ offset
    total = 0.5 * dot(b, Q * b)
    @inbounds for i in eachindex(y)
        eta = eta0[i] + loading[i] * b[leaf_node[i]]
        total += exp(eta) - y[i] * eta + _logfactorial(round(Int, y[i]))
    end
    return total - 0.5 * logdetQ + 0.5 * logdet(ch), b, true
end

function _fit_poisson_phylo_sd_laplace(fam, y, Xmu, Zobs, labels, tree,
                                       nmmu, nmsd, grp, g_tol;
                                       se::Bool = true, maxiter::Int = 250,
                                       offset = zeros(length(y)))
    Q, leaf_node, _ = _poisson_phylo_setup(tree, labels)
    size(Zobs, 1) == length(y) || throw(ArgumentError(
        "phylogenetic SD design must have one row per response"))
    size(Zobs, 2) == length(nmsd) || throw(ArgumentError(
        "phylogenetic SD names do not match its design"))
    qchol = cholesky(Symmetric(Q); check = false)
    issuccess(qchol) || error(
        "phylo(1 | $grp) tree precision is not positive definite after root conditioning")
    logdetQ = logdet(qchol)
    tip_index = _lss_phylo_group_index(tree, labels, grp)[2]
    return _fit_poisson_structured_sd_laplace(
        fam, y, Xmu, Zobs, leaf_node, Q, logdetQ, nmmu, nmsd, grp, g_tol;
        se, maxiter, sd_block = :sd_phylo, scale_key = :sd_phylo,
        extra_scales = Dict(:tree_tip_index => Float64.(tip_index)),
        offset,
        context = "sparse-Laplace Poisson phylogenetic SD regression")
end

function _fit_poisson_covariance_sd_laplace(fam, y, Xmu, Zobs, Zg, gidx, Q,
                                            nmmu, nmsd, grp, g_tol;
                                            se::Bool = true, maxiter::Int = 250,
                                            weight = ones(length(y)),
                                            offset = zeros(length(y)),
                                            context::String)
    qchol = cholesky(Symmetric(Q); check = false)
    issuccess(qchol) || throw(ArgumentError(
        "$context: the covariance matrix for `$grp` is not positive definite (identical " *
        "rows and columns, e.g. duplicated individuals, make it singular). Merge or remove " *
        "the duplicated individuals, or supply a positive-definite matrix."))
    return _fit_poisson_structured_sd_laplace(
        fam, y, Xmu, Zobs, gidx, Q, logdet(qchol), nmmu, nmsd, grp, g_tol;
        se, maxiter, sd_block = :sd, scale_key = :sd,
        group_design = Zg, group_index = gidx,
        weight, offset,
        context)
end

function _fit_poisson_structured_sd_laplace(fam, y, Xmu, Zobs, leaf_node, Q,
                                            logdetQ, nmmu, nmsd, grp, g_tol;
                                            se::Bool, maxiter::Int,
                                            sd_block::Symbol, scale_key::Symbol,
                                            group_design = nothing,
                                            group_index = nothing,
                                            extra_scales = Dict{Symbol,Vector{Float64}}(),
                                            weight = ones(length(y)),
                                            offset = zeros(length(y)),
                                            report_scale::Bool = true,
                                            context::String)
    n = length(y)
    length(offset) == n || throw(ArgumentError(
        "Poisson `offset(...)` must have one value per observation."))
    off = Float64.(offset)
    pmu, psd = size(Xmu, 2), size(Zobs, 2)
    last_b = zeros(size(Q, 1))
    function nll(theta)
        value, b, ok = _poisson_phylo_sd_laplace(
            y, Xmu, Zobs, leaf_node, Q, logdetQ, theta;
            offset = off, weight, b0 = last_b)
        if !ok
            value, b, ok = _poisson_phylo_sd_laplace(
                y, Xmu, Zobs, leaf_node, Q, logdetQ, theta;
                offset = off, weight, b0 = zeros(length(last_b)))
        end
        ok || return 1e18
        last_b .= b
        return value
    end
    function grad!(g, theta)
        for j in eachindex(theta)
            h = 2e-4 * (1 + abs(theta[j]))
            e = zeros(length(theta)); e[j] = h
            g[j] = (nll(theta .+ e) - nll(theta .- e)) / (2h)
        end
        return g
    end
    theta0 = zeros(pmu + psd)
    theta0[1:pmu] .= _poisson_fixed_start(y, Xmu)
    theta0[pmu+1] = log(0.4)
    od = Optim.OnceDifferentiable(nll, grad!, theta0)
    res = Optim.optimize(od, theta0,
        Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking()),
        Optim.Options(g_tol = g_tol, iterations = maxiter))
    theta_hat = Optim.minimizer(res)
    nll_hat = nll(theta_hat)
    gfinal = zeros(length(theta_hat)); grad!(gfinal, theta_hat)
    converged = _laplace_outer_converged(res, nll_hat, gfinal, theta_hat, n, g_tol)
    V = se ? _vcov_from_fd_hessian(
        _finite_hessian(nll, theta_hat; h = _fd_hessian_step(n));
        context = context) :
        fill(NaN, length(theta_hat), length(theta_hat))
    blocks = Pair{Symbol,UnitRange{Int}}[
        :mu => (1:pmu), sd_block => (pmu+1:pmu+psd)]
    names = Pair{Symbol,Vector{String}}[
        :mu => nmmu, sd_block => nmsd]
    means = Dict(:mu => exp.(Xmu * theta_hat[1:pmu] .+ off))
    obs = Dict(:mu => Vector{Float64}(y))
    sdobs = exp.(Zobs * theta_hat[pmu+1:end])
    scales = Dict{Symbol,Vector{Float64}}()
    report_scale && (scales[scale_key] = sdobs)
    if group_design !== nothing
        group_index === nothing && throw(ArgumentError(
            "structured SD group indices are required with a group-level design"))
        sdgroup = exp.(group_design * theta_hat[pmu+1:end])
        scales[:sd_group] = sdgroup
        scales[:sd] = sdgroup[group_index]
        scales[:sd_group_index] = Float64.(group_index)
    end
    merge!(scales, extra_scales)
    # b34: record the conditional field at theta_hat (fresh inner solve; `last_b`
    # was moved by the finite-difference stencils above). Phylogenetic fields are
    # not exported (only relmat / animal / spatial are covered and tested).
    if sd_block !== :sd_phylo
        _, b_hat, ok_hat = _poisson_phylo_sd_laplace(
            y, Xmu, Zobs, leaf_node, Q, logdetQ, theta_hat;
            offset = off, weight, b0 = zeros(length(last_b)))
        if ok_hat
            loading_hat = Float64.(weight) .* sdobs
            _attach_conditional_structured!(
                scales, loading_hat .* b_hat[leaf_node], b_hat)
        else
            _mark_cond_failed!(scales)
        end
    end
    fit = DrmFit(fam, blocks, names, theta_hat, Matrix(V), -nll_hat, n,
                 converged, means, obs, scales)
    return _withiterations(_withnll(fit, _nll_pinned(nll, theta_hat, nll_hat)), Optim.iterations(res))
end

function _poisson_animal_slope_loading(data, slope::Symbol, n::Int;
                                       label::AbstractString = "animal")
    hasproperty(data, slope) || throw(ArgumentError(
        "Poisson $label random-slope predictor `$slope` is absent from data."))
    raw = getproperty(data, slope)
    raw isa AbstractVector{<:Real} && length(raw) == n && all(isfinite, raw) ||
        throw(ArgumentError(
            "Poisson $label random-slope predictor `$slope` must be finite numeric values, one per response."))
    loading = Float64.(raw)
    any(!iszero, loading) || throw(ArgumentError(
        "Poisson $label random-slope predictor `$slope` is zero for every observation, so its SD is not identifiable."))
    return loading
end

function _fit_poisson_animal_slope_laplace(fam::Poisson, y, Xmu, weight, C,
                                            labels, nmmu, grp, g_tol;
                                            se::Bool = true, reml::Bool = false,
                                            normalize::Bool = false)
    reml && _reject_reml_route(fam, "an animal random slope `(0 + x | group)`")
    Q, gidx = _general_cov_setup(C, labels; normalize)
    qchol = cholesky(Symmetric(Q); check = false)
    issuccess(qchol) || throw(ArgumentError(
        "animal(0 + x | $grp) covariance must be positive definite"))
    return _fit_poisson_structured_sd_laplace(
        fam, y, Xmu, ones(Float64, length(y), 1), gidx, Q, logdet(qchol),
        nmmu, [String(grp)], grp, g_tol; se, maxiter = 250,
        sd_block = :resd, scale_key = :resd, weight,
        report_scale = false,
        context = "sparse-Laplace Poisson animal random-slope fit")
end

function _fit_poisson_sd_regression(fam::Poisson, y, Xmu, Zg, gidx, G,
                                     nmmu, nmsd, grp, g_tol; nAGQ::Int = _SD_IID_AGHQ_K,
                                     se::Bool = true,
                                     w::AbstractVector{<:Real} = ones(length(y)),
                                     offset::AbstractVector{<:Real} = zeros(length(y)))
    nAGQ >= 1 || throw(ArgumentError("nAGQ must be ≥ 1; got $nAGQ"))
    n = length(y); pmu = size(Xmu, 2); psd = size(Zg, 2)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    lf = [_logfactorial(round(Int, yi)) for yi in y]
    rule = _AGHQRule(1, nAGQ)
    length(w) == n || throw(ArgumentError(
        "Poisson random-effect loading must have one value per observation."))
    length(offset) == n || throw(ArgumentError(
        "Poisson offset must have one value per observation."))
    Zre = reshape(Float64.(w), n, 1)
    off = Float64.(offset)
    bcache = zeros(1, G)
    function nll(theta)
        beta = @view theta[1:pmu]
        alpha = @view theta[(pmu+1):(pmu+psd)]
        eta0 = Xmu * beta .+ off
        sdgroup = exp.(Zg * alpha)
        ll = (i, eta) -> begin
            etac = clamp(eta, -30.0, 30.0)
            y[i] * etac - exp(etac) - lf[i]
        end
        total = zero(eltype(theta))
        for gi in 1:G
            Lg = reshape([sdgroup[gi]], 1, 1)
            value, bhat = _aghq_group_logint(ll, members[gi], eta0, Zre, Lg,
                                               rule, @view(bcache[:, gi]))
            bcache[1, gi] = isfinite(bhat[1]) ? bhat[1] : 0.0
            total += value
        end
        return -total
    end
    theta0 = zeros(pmu + psd)
    theta0[1] = log(sum(y) / n + eps())
    theta0[(pmu+1):end] .= Zg \ fill(log(0.5), G)
    res = Optim.optimize(nll, theta0, Optim.LBFGS(),
                         Optim.Options(g_tol = g_tol); autodiff = :forward)
    thetahat = Optim.minimizer(res)
    V = se ? _vcov_from_hessian(ForwardDiff.hessian(nll, thetahat)) :
             fill(NaN, length(thetahat), length(thetahat))
    blocks = [:mu => 1:pmu, :sd => (pmu+1):(pmu+psd)]
    names = [:mu => nmmu, :sd => nmsd]
    sdgroup = exp.(Zg * thetahat[(pmu+1):end])
    means = Dict(:mu => exp.(Xmu * thetahat[1:pmu] .+ off))
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sd_group => sdgroup, :sd => sdgroup[gidx],
                  :sd_group_index => Float64.(gidx),
                  :aghq_nodes => [Float64(nAGQ)])
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, thetahat, V, -nll(thetahat), n,
                        drm_optim_converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# Zero-inflated Poisson with a predictor-dependent SD on one ordinary mean
# random intercept. The `zi` linear predictor is fixed-effect-only; for each
# group, AGHQ integrates the product of the ZIP observation likelihoods over
# b_g ~ Normal(0, exp(2 * (Zg * alpha)[g])).
function _fit_poisson_zi_sd_regression(fam::Poisson, y, Xmu, Xzi, Zg, w,
                                       gidx, G,
                                       nmmu, nmzi, nmsd, grp, g_tol;
                                       nAGQ::Int = _ZIP_SD_AGHQ_K, se::Bool = true)
    nAGQ >= 1 || throw(ArgumentError("nAGQ must be ≥ 1; got $nAGQ"))
    n = length(y); pmu = size(Xmu, 2); pzi = size(Xzi, 2); psd = size(Zg, 2)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    lf = [_logfactorial(round(Int, yi)) for yi in y]
    iszero_y = y .== 0
    rule = _AGHQRule(1, nAGQ)
    length(w) == n || throw(ArgumentError(
        "Poisson ZIP random-effect loading must have one value per observation."))
    all(isfinite, w) || throw(ArgumentError(
        "Poisson ZIP random-effect loading must contain finite values."))
    Zre = reshape(Float64.(w), n, 1)
    bcache = zeros(1, G)
    function nll(theta)
        beta_mu = @view theta[1:pmu]
        beta_zi = @view theta[(pmu+1):(pmu+pzi)]
        alpha = @view theta[(pmu+pzi+1):(pmu+pzi+psd)]
        eta_mu = Xmu * beta_mu
        eta_zi = clamp.(Xzi * beta_zi, -30.0, 30.0)
        sd_group = exp.(Zg * alpha)
        ll = (i, eta) -> begin
            eta_clamped = clamp(eta, -30.0, 30.0)
            lambda = exp(eta_clamped)
            log_pi = _log_logistic(eta_zi[i])
            log_one_minus_pi = _log1m_logistic(eta_zi[i])
            if iszero_y[i]
                _logaddexp(log_pi, log_one_minus_pi - lambda)
            else
                log_one_minus_pi + y[i] * eta_clamped - lambda - lf[i]
            end
        end
        total = zero(eltype(theta))
        for gi in 1:G
            Lg = reshape([sd_group[gi]], 1, 1)
            value, bhat = _aghq_group_logint(ll, members[gi], eta_mu, Zre, Lg,
                                               rule, @view(bcache[:, gi]))
            bcache[1, gi] = isfinite(bhat[1]) ? bhat[1] : 0.0
            total += value
        end
        return -total
    end
    theta0 = zeros(pmu + pzi + psd)
    theta0[1] = log(sum(y) / n + eps())
    theta0[pmu+1] = -1.5
    theta0[(pmu+pzi+1):end] .= Zg \ fill(log(0.5), G)
    res = Optim.optimize(nll, theta0, Optim.LBFGS(),
                         Optim.Options(g_tol = g_tol); autodiff = :forward)
    theta_hat = Optim.minimizer(res)
    V = se ? _vcov_from_hessian(ForwardDiff.hessian(nll, theta_hat)) :
             fill(NaN, length(theta_hat), length(theta_hat))
    sd_range = (pmu+pzi+1):(pmu+pzi+psd)
    blocks = [:mu => 1:pmu, :zi => (pmu+1):(pmu+pzi), :sd => sd_range]
    names = [:mu => nmmu, :zi => nmzi, :sd => nmsd]
    sd_group = exp.(Zg * theta_hat[sd_range])
    means = Dict(:mu => exp.(Xmu * theta_hat[1:pmu]))
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:zi => _logistic.(Xzi * theta_hat[(pmu+1):(pmu+pzi)]),
                  :sd_group => sd_group, :sd => sd_group[gidx],
                  :sd_group_index => Float64.(gidx),
                  :aghq_nodes => [Float64(nAGQ)])
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, theta_hat, V, -nll(theta_hat), n,
                        drm_optim_converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# Resolve a structured marker (relmat/animal/spatial) for a count family to its
# user-supplied G×G PD covariance from the keyword args. `relmat`/`animal` take
# the matrix directly (`K`/`A`); `spatial` accepts a precomputed covariance via
# `K`, or — handled upstream in `drm(...)` before this is reached — site `coords`
# for the fixed-range native kernel. Reaching the `:spatial`
# branch here means neither was supplied. Mirrors `_resolve_structured_matrix`
# (gaussian_structured.jl) but for the count Laplace route.
function _poisson_structured_cov(kind::Symbol, grp::Symbol, K, A, coords)
    if kind === :relmat
        K === nothing && error("relmat(1 | $grp) needs `K = …` (the relatedness/covariance matrix)")
        return _finite_user_matrix(K, "K")
    elseif kind === :animal
        A === nothing && error("animal(1 | $grp) needs the relatedness matrix `A = …`")
        return _finite_user_matrix(A, "A")
    else  # :spatial
        K !== nothing && return _finite_user_matrix(K, "K")
        error("spatial(1 | $grp) for counts needs either a precomputed spatial covariance " *
              "via `K = …`, or site `coords = …` (a G×2 coordinate matrix) for coordinate-based " *
              "exponential-kernel covariance with the range fixed at the median pairwise distance")
    end
end

# Poisson count GLMM with a random intercept (1|g) on log λ. b_g ~ N(0,σ_b²) is
# integrated out per group by K-node Gauss–Hermite quadrature (b = √2 σ_b z), the
# same scheme as the Gaussian σ-RE. O(n·K) per evaluation, fully differentiable.
#
# `reml = true` (public `method = :REML`, #443) switches to the opt-in Cox–Reid adjusted
# profile likelihood ℓ_CR = ℓ_ML − ½·log|I_ββ|, which reduces the ML finite-cluster
# downward bias in σ̂_b. The 32-node quadrature means the integral error is already paid,
# so σ̂_b's residual bias is the ML variance-component bias and nothing else — which is
# what makes this the clean first cell.
#
# Two things are deliberately reused rather than re-derived: `_glsp_reml_penalty` forms
# I_ββ (its FD-of-analytic-gradient construction reduces EXACTLY to Patterson–Thompson
# REML on the Gaussian route — probe Cell B matched the #440 Woodbury path to 2.9e-06),
# and `_glsp_reml_refit_clean` carries the guards production needs: the non-PD-I_ββ
# sentinel (2/60 seeds hit it at G=10) and the line-search fallback. This route has no
# analytic gradient closure, so the exact ForwardDiff gradient supplies `grad_fn`.
#
# ML stays the default: Cox–Reid OVER-corrects once clusters are plentiful (+4.38% at
# G=40 against −12.37% for ML at G=10).
function _fit_poisson_ranef(fam::Poisson, y, Xμ, gidx, G, nmμ, grp, g_tol; reml::Bool = false, K::Int = _RANEF1D_AGHQ_K,
                           offset::AbstractVector{<:Real} = zeros(length(y)))
    n = length(y); pμ = size(Xμ, 2)
    off = Float64.(offset)                               # #941: offset() coefficient fixed at 1
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    lf = [_logfactorial(round(Int, yi)) for yi in y]
    rule = _AGHQRule(1, K); Zre = ones(n, 1); bcache = zeros(1, G)   # #719: per-group AGHQ
    function nll(θ)
        βμ = θ[1:pμ]; σb = exp(θ[pμ+1])
        η0 = Xμ * βμ .+ off
        ll = (i, η) -> (ηc = clamp(η, -30.0, 30.0); y[i] * ηc - exp(ηc) - lf[i])
        L = reshape([σb], 1, 1)
        return -_aghq_marginal_loglik(ll, members, η0, Zre, L, rule, bcache)
    end
    θ0 = zeros(pμ + 1)
    θ0[1] = log(sum(y) / n + eps()) - sum(off) / n
    θ0[pμ+1] = log(0.5)
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); conv = drm_optim_converged(res)
    blocks = [:mu => 1:pμ, :resd => (pμ+1):(pμ+1)]
    names = [:mu => nmμ, :resd => [String(grp)]]
    scales = Dict{Symbol,Vector{Float64}}()
    if reml
        grad_fn = θ -> ForwardDiff.gradient(nll, θ)
        ml_nll = nll(θ̂)
        θ̂, conv, ml_nll, reml_nll, _ =
            _glsp_reml_refit_clean(nll, grad_fn, θ̂, pμ; ml_converged = conv)
        # Wald vcov under a restricted objective must be the inverse Hessian OF THAT
        # objective (#310), not the ML observed information.
        V = _glsp_reml_vcov(grad_fn, θ̂, pμ)
        means = Dict(:mu => exp.(Xμ * θ̂[1:pμ] .+ off)); obs = Dict(:mu => Vector{Float64}(y))
        fit = _withnll(DrmFit(fam, blocks, names, θ̂, V, -ml_nll, n, conv, means, obs, scales), nll)
        return _withreml(fit, -reml_nll, -ml_nll)
    end
    V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    means = Dict(:mu => exp.(Xμ * θ̂[1:pμ] .+ off)); obs = Dict(:mu => Vector{Float64}(y))   # population λ (b=0), × exposure
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, conv, means, obs, scales), nll),
        Optim.iterations(res))
end

# Opt-in 1-D Liu–Pierce AGHQ for Poisson `(1 | g)` (#448). Same model as
# `_fit_poisson_ranef`, but each group's integral uses the adaptive map around
# the group posterior mode (`_poisson_group_aghq_logint`) instead of
# prior-scaled GHQ-32. Default `:LA` is unchanged. `nAGQ=1` is 1-point Laplace
# plumbing, not a recovery claim. `:REML` is rejected upstream.
function _fit_poisson_ranef_aghq(fam::Poisson, y, Xμ, gidx, G, nmμ, grp, g_tol; nAGQ::Int = 5,
                                 offset::AbstractVector{<:Real} = zeros(length(y)))
    nAGQ >= 1 || throw(ArgumentError("nAGQ must be ≥ 1; got $nAGQ"))
    n = length(y); pμ = size(Xμ, 2)
    off = Float64.(offset)                               # #941: offset() coefficient fixed at 1
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    lf = [_logfactorial(round(Int, yi)) for yi in y]
    function nll(θ)
        βμ = θ[1:pμ]; σb = exp(θ[pμ+1])
        η0 = Xμ * βμ .+ off
        s = zero(eltype(θ))
        for idx in members
            isempty(idx) && continue
            s -= _poisson_group_aghq_logint(y, η0, lf, idx, σb, nAGQ)
        end
        return s
    end
    θ0 = zeros(pμ + 1)
    θ0[1] = log(sum(y) / n + eps()) - sum(off) / n
    θ0[pμ+1] = log(0.5)
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); conv = drm_optim_converged(res)
    # A penalised objective at the optimum means some group posterior could not be
    # evaluated there (#1063): fail cleanly rather than report a meaningless fit.
    nll̂ = nll(θ̂)
    (isfinite(nll̂) && nll̂ < -_AGHQ_PENALTY_LOGINT) || error(
        "marginal = :AGHQ could not evaluate the Poisson (1 | $grp) group integrals at the optimum " *
        "(non-finite or non-log-concave group posterior; #1063). " *
        "Use the default marginal = :LA or marginal = :Laplace for this model.")
    blocks = [:mu => 1:pμ, :resd => (pμ+1):(pμ+1)]
    names = [:mu => nmμ, :resd => [String(grp)]]
    scales = Dict{Symbol,Vector{Float64}}()
    V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    means = Dict(:mu => exp.(Xμ * θ̂[1:pμ] .+ off)); obs = Dict(:mu => Vector{Float64}(y))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, conv, means, obs, scales), nll),
        Optim.iterations(res))
end

# Poisson count GLMM with a correlated random intercept+slope (1 + x | g) on log λ.
# Per group (b0,b1) ~ N(0, Σ); because groups are disjoint the 2-D integral
# factorises; each group is integrated by per-group ADAPTIVE Gauss–Hermite quadrature
# (`_aghq_marginal_loglik`, #834: nodes b̂_g + √2 C z at each group's mode),
# `nq` nodes per axis (nq = 1 is Laplace). Σ is the log-Cholesky parameterisation
# L = [exp(a) 0; cc exp(b)] (the `vc` convention), so vc(fit) reconstructs Σ = L Lᵀ.
function _fit_poisson_corr_ranef(fam::Poisson, y, Xμ, xs, gidx, G, nmμ, grp, g_tol; nq::Int = _CORR_RANEF_AGHQ_K)
    n = length(y); pμ = size(Xμ, 2)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    lf = [_logfactorial(round(Int, yi)) for yi in y]
    rule = _AGHQRule(2, nq); Zre = hcat(ones(n), Float64.(xs)); bcache = zeros(2, G)   # #834: per-group AGHQ
    function nll(θ)
        βμ = θ[1:pμ]; L = _corr_ranef_L(θ[pμ+1], θ[pμ+2], θ[pμ+3]); η0 = Xμ * βμ
        ll = (i, η) -> (ηc = clamp(η, -30.0, 30.0); y[i] * ηc - exp(ηc) - lf[i])
        return -_aghq_marginal_loglik(ll, members, η0, Zre, L, rule, bcache)
    end
    θ0 = zeros(pμ + 3)
    θ0[1] = log(sum(y) / n + eps())
    θ0[pμ+1] = log(0.4); θ0[pμ+2] = log(0.4); θ0[pμ+3] = 0.0
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :recov => (pμ+1):(pμ+3)]
    names = [:mu => nmμ, :recov => ["$(grp):L11", "$(grp):L22", "$(grp):L21"]]
    means = Dict(:mu => exp.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict{Symbol,Vector{Float64}}()
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, drm_optim_converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# log-logistic helpers (stable): log π and log(1-π) for π = logistic(η).
_log_logistic(η) = -log1p(exp(-η))
_log1m_logistic(η) = -log1p(exp(η))
# log(exp(a) + exp(b)), numerically stable.
_logaddexp(a, b) = (m = max(a, b); m + log1p(exp(-abs(a - b))))
# log(1 - exp(x)) for x ≤ 0, numerically stable (hurdle zero-truncation term).
_log1mexp(x) = x < -log(2) ? log1p(-exp(x)) : log(-expm1(x))

# Zero-inflated Poisson: P(0) = π + (1-π)e^{-λ}, P(k>0) = (1-π)·Poisson(k; λ),
# with π = logistic(Xziᵀβ) (logit link) and λ = exp(Xμᵀβ).
function _fit_poisson_zi(fam::Poisson, y, Xμ, Xzi, nmμ, nmzi, g_tol)
    n = length(y); pμ, pz = size(Xμ, 2), size(Xzi, 2)
    lf = [_logfactorial(round(Int, yi)) for yi in y]
    iszero_y = y .== 0
    w = _ctl_weights(n)                  # prior weights: row log-lik multipliers (#931)
    function nll(θ)
        βμ = θ[1:pμ]; βz = θ[pμ+1:pμ+pz]
        ημ = clamp.(Xμ * βμ, -30.0, 30.0); ηz = clamp.(Xzi * βz, -30.0, 30.0)
        s = zero(eltype(θ))
        @inbounds for i in 1:n
            λ = exp(ημ[i]); lπ = _log_logistic(ηz[i]); l1mπ = _log1m_logistic(ηz[i])
            if iszero_y[i]
                s -= w[i] * (_logaddexp(lπ, l1mπ - λ))              # log(π + (1-π)e^{-λ})
            else
                s -= w[i] * (l1mπ + (y[i] * ημ[i] - λ - lf[i]))
            end
        end
        return s
    end
    pos = y[y.>0]
    θ0 = zeros(pμ + pz)
    θ0[1] = log((isempty(pos) ? sum(y) / n : sum(pos) / length(pos)) + eps())   # λ from non-zeros
    res = _ctl_optimize((x0, opts, method) -> Optim.optimize(nll, x0, method, opts; autodiff = :forward),
                       θ0; g_tol = g_tol)   # control: maxit / start / multi_start (#948 #994)
    θ̂ = Optim.minimizer(res); V = _ctl_se(() -> _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂)), θ̂)
    blocks = [:mu => 1:pμ, :zi => (pμ+1):(pμ+pz)]
    names = [:mu => nmμ, :zi => nmzi]
    means = Dict(:mu => exp.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:zi => _logistic.(Xzi * θ̂[(pμ+1):(pμ+pz)]))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, drm_optim_converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# Hurdle Poisson: P(0) = π, P(k>0) = (1-π)·Poisson(k; λ)/(1-e^{-λ}) [zero-truncated],
# with π = logistic(Xhuᵀβ) the hurdle (zero) probability and λ = exp(Xμᵀβ). All
# zeros are structural; the positive part is the zero-truncated Poisson.
function _fit_poisson_hu(fam::Poisson, y, Xμ, Xhu, nmμ, nmhu, g_tol)
    n = length(y); pμ, ph = size(Xμ, 2), size(Xhu, 2)
    lf = [_logfactorial(round(Int, yi)) for yi in y]
    iszero_y = y .== 0
    w = _ctl_weights(n)                  # prior weights: row log-lik multipliers (#931)
    function nll(θ)
        βμ = θ[1:pμ]; βh = θ[pμ+1:pμ+ph]
        ημ = clamp.(Xμ * βμ, -30.0, 30.0); ηh = clamp.(Xhu * βh, -30.0, 30.0)
        s = zero(eltype(θ))
        @inbounds for i in 1:n
            lπ = _log_logistic(ηh[i]); l1mπ = _log1m_logistic(ηh[i])
            if iszero_y[i]
                s -= w[i] * (lπ)                                        # log π
            else
                λ = exp(ημ[i])
                lpos = y[i] * ημ[i] - λ - lf[i]                # log Poisson(k; λ)
                s -= w[i] * (l1mπ + lpos - _log1mexp(-λ))               # − log(1 − e^{-λ})
            end
        end
        return s
    end
    pos = y[y.>0]
    θ0 = zeros(pμ + ph)
    θ0[1] = log((isempty(pos) ? sum(y) / n : sum(pos) / length(pos)) + eps())   # λ from non-zeros
    res = _ctl_optimize((x0, opts, method) -> Optim.optimize(nll, x0, method, opts; autodiff = :forward),
                       θ0; g_tol = g_tol)   # control: maxit / start / multi_start (#948 #994)
    θ̂ = Optim.minimizer(res); V = _ctl_se(() -> _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂)), θ̂)
    blocks = [:mu => 1:pμ, :hu => (pμ+1):(pμ+ph)]
    names = [:mu => nmμ, :hu => nmhu]
    means = Dict(:mu => exp.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:hu => _logistic.(Xhu * θ̂[(pμ+1):(pμ+ph)]))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, drm_optim_converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

function _fit_poisson(fam::Poisson, y, Xμ, nmμ, g_tol; offset::AbstractVector{<:Real} = zeros(length(y)))
    n = length(y); pμ = size(Xμ, 2)
    lf = [_logfactorial(round(Int, yi)) for yi in y]    # constant log y! term
    off = Float64.(offset)
    w = _ctl_weights(n)                  # prior weights: row log-lik multipliers (#931)
    function nll(θ)
        ημ = Xμ * θ .+ off                              # log λ; offset() coefficient fixed at 1
        s = zero(eltype(θ))
        @inbounds for i in 1:n
            s -= w[i] * (y[i] * ημ[i] - exp(ημ[i]) - lf[i])
        end
        return s
    end
    θ0 = zeros(pμ); θ0[1] = log(sum(y) / n + eps()) - (isempty(off) ? 0.0 : sum(off) / n)
    res = _ctl_optimize((x0, opts, method) -> Optim.optimize(nll, x0, method, opts; autodiff = :forward),
                       θ0; g_tol = g_tol)   # control: maxit / start / multi_start (#948 #994)
    θ̂ = Optim.minimizer(res); V = _ctl_se(() -> _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂)), θ̂)
    blocks = [:mu => 1:pμ]; names = [:mu => nmμ]
    means = Dict(:mu => exp.(Xμ * θ̂ .+ off)); obs = Dict(:mu => Vector{Float64}(y))   # response-scale λ
    V, nonexistent = _count_mle_existence_guard(Xμ, y, V, nmμ)   # #1109: detect + warn, SE → Inf
    scales = Dict{Symbol,Vector{Float64}}()
    fit = _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, drm_optim_converged(res), means, obs, scales), nll),
        Optim.iterations(res))
    return nonexistent ? _withgeometry(fit, :count_mle_nonexistent) : fit
end

# Independent animal intercept-and-slope q2 block with one SD formula per
# coefficient: `y ~ x + animal(1 + x | g)`, `sd(g, animal, coef(intercept)) ~ z0`,
# `sd(g, animal, coef(x)) ~ z1`. Mirrors the Gamma route; Poisson has no
# `sigma`, so the parameter vector is (beta, alpha0, alpha1).
function _drm_poisson_animal_intercept_slope_sd(f::DrmFormula, fam::Poisson,
        targets, animal_block; data, tree, K, A, Ainv, coords, g_tol, se)
    length(targets) == 2 || throw(ArgumentError(
        "Poisson `animal(1 + x | group)` SD regression needs one SD formula for the intercept and one for the slope."))
    group, slope = animal_block.group, animal_block.slope
    all(item -> item.target.level === :animal && item.target.group === group &&
        item.target.dpar === :mu && item.target.block === nothing, targets) ||
        throw(ArgumentError("Poisson independent animal intercept-and-slope SD regression requires two matching `animal` mean targets."))
    Set(item.target.member for item in targets) == Set((:intercept, slope)) ||
        throw(ArgumentError(
            "Poisson animal SD formulas must select `coef(intercept)` and `coef($slope)` exactly once."))
    (tree === nothing && K === nothing && coords === nothing) || throw(ArgumentError(
        "Poisson animal SD regression uses `A = …` or `Ainv = …`; do not also supply `tree`, `K`, or `coords`."))
    (A === nothing) == (Ainv === nothing) && throw(ArgumentError(
        "Poisson animal SD regression requires exactly one of `A = …` or `Ainv = …`."))
    rhs = Dict(f.forms)
    correlated = animal_block.label !== nothing
    fixed_mu, re, metav, structured, _, _, parsed_slope = _split_ranef(
        rhs[:mu]; allow_animal_intercept_slope = true,
        allow_labelled_intercept_slope = correlated)
    metav === nothing && isempty(re) && structured == (:animal, group) &&
        parsed_slope === slope || throw(ArgumentError(
        "Poisson animal SD regression requires one matching `animal(1 + $slope | $group)` " *
        "or labelled `animal(1 + $slope | p | $group)` term."))
    _collect_structured(rhs[:mu]) == [(:animal, group)] || throw(ArgumentError(
        "Poisson animal intercept-and-slope SD regression requires exactly one structured term, with no additional structured effects."))
    (haskey(rhs, :zi) || haskey(rhs, :hu)) && throw(ArgumentError(
        "Poisson animal SD regression cannot yet be combined with `zi` or `hu`."))
    fixed_mu, offset_term = _extract_offset(fixed_mu)
    allowed = Set(vcat([:mu, :sigma], [item.key for item in targets]))
    all(first(pair) in allowed for pair in f.forms) || throw(ArgumentError(
        "Poisson animal intercept-and-slope SD regression supports only `mu` and its two SD formulas."))
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "Poisson animal SD regression requires observed responses; remove incomplete rows before fitting."))
    y, Xmu, nmmu = _design(f.response, fixed_mu, data)
    off = _poisson_sd_offset(offset_term, data, length(y))
    all(yi -> yi >= 0 && isinteger(yi), y) || throw(ArgumentError(
        "Poisson() requires non-negative integer counts as the response"))
    labels = getproperty(data, group)
    Q, gidx = Ainv === nothing ? _general_cov_setup(A, labels; normalize = false) :
        _poisson_animal_precision_setup(Ainv, labels, group)
    G = size(Q, 1)
    ordered = map((:intercept, slope)) do member
        only(filter(item -> item.target.member === member, targets))
    end
    Zgs = Matrix{Float64}[]
    nmsds = Vector{String}[]
    for item in ordered
        Zg, names = _sd_group_design(f.response, item.rhs, data, gidx, G, group;
            schema_cache = f.schema_cache, schema_key = item.key)
        size(Zg, 2) <= G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
            "Poisson animal SD target `coef($(item.target.member))` has a rank-deficient group-level design."))
        push!(Zgs, Matrix{Float64}(Zg))
        push!(nmsds, names)
    end
    x = _poisson_animal_slope_loading(data, slope, length(y))
    if all(==(first(x)), x) && isequal(Zgs[1], Zgs[2])
        throw(ArgumentError(
            "Poisson animal intercept-and-slope SD regression cannot separate the intercept and slope variances when `$slope` is constant and both SD formulas have identical group-level designs."))
    end
    blocks = Symbol[_sd_target_block(item.target; disambiguate = true)
                    for item in ordered]
    return _withformula(_fit_poisson_animal_intercept_slope_laplace(
        fam, y, Xmu, Zgs, gidx, Q, nmmu, nmsds, blocks, group, x, off,
        g_tol; se, correlated), f)
end

# Joint prior precision of the two unit-scale structured fields (b0, b1):
# kron(R^-1, Q) with R = [1 rho; rho 1]. `rho = 0` is the independent block.
function _q2_joint_precision(Q, rho::Real)
    G = size(Q, 1)
    rho == 0 && return [Q spzeros(G, G); spzeros(G, G) Q]
    c = 1 / (1 - rho^2)
    return [c .* Q (-rho * c) .* Q; (-rho * c) .* Q c .* Q]
end

function _poisson_animal_intercept_slope_mode(y, eta0, load0, load1, gidx, Q;
                                              b0 = nothing, maxiter::Int = 80,
                                              tol::Real = 1e-10, rho::Real = 0.0)
    G = size(Q, 1)
    Qjoint = _q2_joint_precision(Q, rho)
    b = b0 === nothing ? zeros(2G) : copy(b0)
    function joint(bb)
        value = 0.5 * dot(bb, Qjoint * bb)
        b0v, b1v = @view(bb[1:G]), @view(bb[(G + 1):(2G)])
        @inbounds for i in eachindex(y)
            eta = eta0[i] + load0[i] * b0v[gidx[i]] + load1[i] * b1v[gidx[i]]
            value += exp(eta) - y[i] * eta
        end
        value
    end
    for _ in 1:maxiter
        b0v, b1v = @view(b[1:G]), @view(b[(G + 1):(2G)])
        grad = Qjoint * b
        h00, h11, h01 = zeros(G), zeros(G), zeros(G)
        @inbounds for i in eachindex(y)
            gi = gidx[i]
            eta = eta0[i] + load0[i] * b0v[gi] + load1[i] * b1v[gi]
            mu = exp(eta)
            d1 = mu - y[i]
            grad[gi] += load0[i] * d1
            grad[G + gi] += load1[i] * d1
            h00[gi] += load0[i]^2 * mu
            h11[gi] += load1[i]^2 * mu
            h01[gi] += load0[i] * load1[i] * mu
        end
        hessian = Qjoint + [spdiagm(0 => h00) spdiagm(0 => h01);
                            spdiagm(0 => h01) spdiagm(0 => h11)]
        ch = cholesky(Symmetric(hessian); check = false)
        issuccess(ch) || return b, ch, false
        step = ch \ grad
        norm(step) <= tol * (1 + norm(b)) && return b, ch, true
        if norm(step) <= 1e-3 * (1 + norm(b))
            b .-= step
            continue
        end
        f0 = joint(b)
        amount, accepted = 1.0, false
        while amount >= 1e-5
            trial = b .- amount .* step
            if joint(trial) <= f0
                b = trial
                accepted = true
                break
            end
            amount *= 0.5
        end
        accepted || return b, ch, false
    end
    return b, nothing, false
end

# `correlated = true` appends drmTMB's `eta_cor_phylo` to theta:
# rho = 0.999999 tanh(eta), and the unit-scale fields have prior N(0, R (x) C).
function _poisson_animal_intercept_slope_laplace(y, lf, Xmu, Z0, Z1, gidx, Q,
        logdetQ, theta, x, off; b0 = nothing, correlated::Bool = false)
    pmu = size(Xmu, 2)
    p0, p1 = size(Z0, 2), size(Z1, 2)
    beta = @view theta[1:pmu]
    a0 = @view theta[(pmu + 1):(pmu + p0)]
    a1 = @view theta[(pmu + p0 + 1):(pmu + p0 + p1)]
    rho = correlated ? _Q2_RHO_BOUND * tanh(theta[pmu + p0 + p1 + 1]) : 0.0
    eta0 = Xmu * beta .+ off
    load0 = exp.(Z0[gidx, :] * a0)
    load1 = x .* exp.(Z1[gidx, :] * a1)
    b, ch, ok = _poisson_animal_intercept_slope_mode(
        y, eta0, load0, load1, gidx, Q; b0, rho)
    ok || return 1e18, b, false
    G = size(Q, 1)
    b0v, b1v = @view(b[1:G]), @view(b[(G + 1):(2G)])
    value = 0.5 * dot(b, _q2_joint_precision(Q, rho) * b) + 0.5 * G * log1p(-rho^2)
    @inbounds for i in eachindex(y)
        eta = eta0[i] + load0[i] * b0v[gidx[i]] + load1[i] * b1v[gidx[i]]
        value += exp(eta) - y[i] * eta + lf[i]
    end
    return value - logdetQ + 0.5 * logdet(ch), b, true
end

# drmTMB bounds the structured correlation as 0.999999 * tanh(eta_cor_phylo).
const _Q2_RHO_BOUND = 0.999999

function _fit_poisson_animal_intercept_slope_laplace(fam, y, Xmu, Zgs, gidx, Q,
        nmmu, nmsds, blocks, group, x, off, g_tol;
        se::Bool = true, maxiter::Int = 250, correlated::Bool = false,
        second_start::Bool = false)
    length(Zgs) == 2 || throw(ArgumentError("expected two Poisson animal SD designs"))
    qchol = cholesky(Symmetric(Q); check = false)
    issuccess(qchol) || throw(ArgumentError(
        "Poisson animal covariance matrix must be positive definite"))
    logdetQ = logdet(qchol)
    lf = [_logfactorial(round(Int, yi)) for yi in y]
    pmu = size(Xmu, 2)
    psd = size.(Zgs, 2)
    ranges = (pmu + 1):(pmu + psd[1]), (pmu + psd[1] + 1):(pmu + sum(psd))
    theta0 = zeros(pmu + sum(psd) + (correlated ? 1 : 0))
    theta0[1:pmu] .= _poisson_fixed_start(y, Xmu)
    theta0[ranges[1]] .= Zgs[1] \ fill(log(0.4), size(Zgs[1], 1))
    theta0[ranges[2]] .= Zgs[2] \ fill(log(0.25), size(Zgs[2], 1))
    last_b = zeros(2 * size(Q, 1))
    function nll(theta)
        value, b, ok = _poisson_animal_intercept_slope_laplace(
            y, lf, Xmu, Zgs[1], Zgs[2], gidx, Q, logdetQ, theta, x, off;
            b0 = last_b, correlated)
        if !ok
            value, b, ok = _poisson_animal_intercept_slope_laplace(
                y, lf, Xmu, Zgs[1], Zgs[2], gidx, Q, logdetQ, theta, x, off;
                b0 = zeros(length(last_b)), correlated)
        end
        ok || return 1e18
        last_b .= b
        value
    end
    function grad!(g, theta)
        for j in eachindex(theta)
            h = 2e-4 * (1 + abs(theta[j]))
            e = zeros(length(theta)); e[j] = h
            g[j] = (nll(theta .+ e) - nll(theta .- e)) / (2h)
        end
        return g
    end
    run_from(th0) = Optim.optimize(Optim.OnceDifferentiable(nll, grad!, th0), th0,
        Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking()),
        Optim.Options(g_tol = g_tol, iterations = maxiter))
    result = run_from(theta0)
    if second_start
        # Opt-in second start for routes whose likelihood is multimodal in the slope
        # fixed effect (coordinate-spatial kernel: scale-300 campaign seed 147 stopped at
        # a stationary point 1.02 log-lik below the best, found from a larger SD start).
        # Keep the first optimum unless the second is better by more than 1e-6 in the
        # objective, so fits whose first start was already the best are unchanged.
        theta_first = Optim.minimizer(result)
        nll_first = nll(theta_first)
        b_first = copy(last_b)
        theta1 = copy(theta0)
        theta1[ranges[1]] .= Zgs[1] \ fill(log(0.7), size(Zgs[1], 1))
        theta1[ranges[2]] .= Zgs[2] \ fill(log(0.7), size(Zgs[2], 1))
        result2 = run_from(theta1)
        nll_second = nll(Optim.minimizer(result2))
        if nll_second < nll_first - 1e-6
            result = result2
        else
            last_b .= b_first
        end
    end
    # Boundary restarts (see `_q2_boundary_restarts`): the default start can stop
    # short of the interior optimum or on a flat ridge when SD coefficients
    # collapse (same helper as Binomial/Gamma q2 SD-z routes; issue pending).
    restart_info = Ref{Any}((ran = false, winner = 0, gain = 0.0, flipped = false))
    result = _q2_boundary_restarts(run_from, last_b, result,
        (sd0, sd1, het) -> _q2_alt_theta(theta0, ranges, Zgs, sd0, sd1, het),
        th -> min(minimum(exp.(Zgs[1] * th[ranges[1]])),
                  minimum(exp.(Zgs[2] * th[ranges[2]]))),
        r -> begin
            th = Optim.minimizer(r); g = zeros(length(th)); grad!(g, th)
            _laplace_outer_converged(r, Optim.minimum(r), g, th, length(y), g_tol)
        end; info = restart_info)
    ri = restart_info[]
    (ri.gain > _Q2_RESTART_GAIN || ri.flipped) &&
        @info "drm: Poisson animal q2 boundary restart changed the accepted optimum " *
              "(objective gain $(round(ri.gain; sigdigits = 3)), converged flipped = $(ri.flipped), " *
              "winning start $(ri.winner) of $(length(_Q2_ALT_STARTS)); 0 = default start). " *
              "See `fit.scales[:q2_restarts]`."
    theta = Optim.minimizer(result)
    nll_hat = nll(theta)
    gf = zeros(length(theta)); grad!(gf, theta)
    V = se ? _vcov_from_fd_hessian(
        _finite_hessian(nll, theta; h = _fd_hessian_step(length(y)));
        context = "Poisson animal intercept-and-slope SD regression") :
        fill(NaN, length(theta), length(theta))
    block_list = Pair{Symbol,UnitRange{Int}}[
        :mu => 1:pmu, blocks[1] => ranges[1], blocks[2] => ranges[2]]
    name_list = Pair{Symbol,Vector{String}}[
        :mu => nmmu, blocks[1] => nmsds[1], blocks[2] => nmsds[2]]
    if correlated
        push!(block_list, :recov => length(theta):length(theta))
        push!(name_list, :recov => ["$(group):atanh_rho"])
    end
    sd0 = exp.(Zgs[1] * theta[ranges[1]])
    sd1 = exp.(Zgs[2] * theta[ranges[2]])
    scales = Dict{Symbol,Vector{Float64}}()
    scales[blocks[1]] = sd0[gidx]
    scales[blocks[2]] = sd1[gidx]
    scales[Symbol(string(blocks[1]), "_group")] = sd0
    scales[Symbol(string(blocks[2]), "_group")] = sd1
    scales[:q2_restarts] = [ri.ran ? 1.0 : 0.0, Float64(ri.winner), Float64(ri.gain), ri.flipped ? 1.0 : 0.0]
    # Conditional modes at theta_hat (fresh inner solve), so `residuals(type = :quantile)`
    # can condition on them as drmTMB's does. `fitted()` stays population level.
    _, bhat, okhat = _poisson_animal_intercept_slope_laplace(
        y, lf, Xmu, Zgs[1], Zgs[2], gidx, Q, logdetQ, theta, x, off;
        b0 = zeros(2 * size(Q, 1)), correlated)
    if okhat
        Gq = size(Q, 1)
        eta_re = (exp.(Zgs[1][gidx, :] * theta[ranges[1]]) .* bhat[1:Gq][gidx]) .+
                 (x .* exp.(Zgs[2][gidx, :] * theta[ranges[2]]) .* bhat[(Gq + 1):(2Gq)][gidx])
        _attach_conditional_structured!(scales, eta_re, bhat[1:Gq], bhat[(Gq + 1):(2Gq)])
    else
        _mark_cond_failed!(scales)
    end
    fit = DrmFit(fam, block_list, name_list, theta, Matrix(V), -nll_hat,
        length(y), _laplace_outer_converged(result, nll_hat, gf, theta,
        length(y), g_tol), Dict(:mu => exp.(Xmu * theta[1:pmu] .+ off)),
        Dict(:mu => Vector{Float64}(y)), scales)
    return _withiterations(_withnll(fit, _nll_pinned(nll, theta, nll_hat)), Optim.iterations(result))
end

# Independent relmat intercept-and-slope q2 block, `relmat(1 + x | g)` with one SD
# formula per coefficient. Same fit helper as the animal route; K enters RAW
# (as in `_drm_poisson_relmat_sd_regression`).
function _drm_poisson_relmat_intercept_slope_sd(f::DrmFormula, fam::Poisson,
        targets, relmat_block; data, tree, K, A, coords, g_tol, se)
    length(targets) == 2 || throw(ArgumentError(
        "Poisson `relmat(1 + x | group)` SD regression needs one SD formula for the intercept and one for the slope."))
    group, slope = relmat_block.group, relmat_block.slope
    all(item -> item.target.level === :relmat && item.target.group === group &&
        item.target.dpar === :mu && item.target.block === nothing, targets) ||
        throw(ArgumentError("Poisson independent relmat intercept-and-slope SD regression requires two matching `relmat` mean targets."))
    Set(item.target.member for item in targets) == Set((:intercept, slope)) ||
        throw(ArgumentError(
            "Poisson relmat SD formulas must select `coef(intercept)` and `coef($slope)` exactly once."))
    (tree === nothing && A === nothing && coords === nothing) || throw(ArgumentError(
        "Poisson relmat SD regression uses `K = …` only; do not also supply `tree`, `A`, or `coords`."))
    K === nothing && throw(ArgumentError(
        "relmat(1 + $slope | $group) needs the relatedness/covariance matrix `K = …`"))
    rhs = Dict(f.forms)
    correlated = relmat_block.label !== nothing
    fixed_mu, re, metav, structured, _, _, parsed_slope = _split_ranef(
        rhs[:mu]; allow_relmat_intercept_slope = true,
        allow_labelled_intercept_slope = correlated)
    metav === nothing && isempty(re) && structured == (:relmat, group) &&
        parsed_slope === slope || throw(ArgumentError(
        "Poisson relmat SD regression requires one matching `relmat(1 + $slope | $group)` " *
        "or labelled `relmat(1 + $slope | p | $group)` term."))
    _collect_structured(rhs[:mu]) == [(:relmat, group)] || throw(ArgumentError(
        "Poisson relmat intercept-and-slope SD regression requires exactly one structured term, with no additional structured effects."))
    (haskey(rhs, :zi) || haskey(rhs, :hu)) && throw(ArgumentError(
        "Poisson relmat SD regression cannot yet be combined with `zi` or `hu`."))
    fixed_mu, offset_term = _extract_offset(fixed_mu)
    allowed = Set(vcat([:mu, :sigma], [item.key for item in targets]))
    all(first(pair) in allowed for pair in f.forms) || throw(ArgumentError(
        "Poisson relmat intercept-and-slope SD regression supports only `mu` and its two SD formulas."))
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "Poisson relmat SD regression requires observed responses; remove incomplete rows before fitting."))
    y, Xmu, nmmu = _design(f.response, fixed_mu, data)
    off = _poisson_sd_offset(offset_term, data, length(y))
    all(yi -> yi >= 0 && isinteger(yi), y) || throw(ArgumentError(
        "Poisson() requires non-negative integer counts as the response"))
    labels = getproperty(data, group)
    Q, gidx = _general_cov_setup(K, labels; normalize = false)
    G = size(Q, 1)
    ordered = map((:intercept, slope)) do member
        only(filter(item -> item.target.member === member, targets))
    end
    Zgs = Matrix{Float64}[]
    nmsds = Vector{String}[]
    for item in ordered
        Zg, names = _sd_group_design(f.response, item.rhs, data, gidx, G, group;
            schema_cache = f.schema_cache, schema_key = item.key)
        size(Zg, 2) <= G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
            "Poisson relmat SD target `coef($(item.target.member))` has a rank-deficient group-level design."))
        push!(Zgs, Matrix{Float64}(Zg))
        push!(nmsds, names)
    end
    x = _poisson_animal_slope_loading(data, slope, length(y); label = "relmat")
    if all(==(first(x)), x) && isequal(Zgs[1], Zgs[2])
        throw(ArgumentError(
            "Poisson relmat intercept-and-slope SD regression cannot separate the intercept and slope variances when `$slope` is constant and both SD formulas have identical group-level designs."))
    end
    blocks = Symbol[_sd_target_block(item.target; disambiguate = true)
                    for item in ordered]
    return _withformula(_fit_poisson_animal_intercept_slope_laplace(
        fam, y, Xmu, Zgs, gidx, Q, nmmu, nmsds, blocks, group, x, off,
        g_tol; se, correlated), f)
end
