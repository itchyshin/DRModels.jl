# negbinomial.jl — Negative-binomial (NB2) family for overdispersed counts.
# Log link on the mean μ. The `sigma` slot carries `log σ` (the family's secondary
# parameter, matching Gaussian/Beta/Gamma), and the NB2 size is θ = 1/σ² =
# exp(-2·coef(:sigma)) — the locked twin convention (drmTMB: size = 1/σ²). Every
# fitter here computes `r = exp(-2·ησ)` accordingly. Variance = μ + μ²/θ, so θ → ∞
# recovers Poisson. The likelihood is the log1p-space NB2 log-pmf `_nb2_logpmf`
# (below; it never forms p = θ/(θ+μ), which quantises at near-Poisson size,
# #866/#982) — ForwardDiff-safe. Fixed effects, ML. Mirrors drmTMB's `nbinom2`.

using Distributions: NegativeBinomial, logpdf
using SpecialFunctions: loggamma, logbeta

# Distributions' `zero(p) < p <= one(p)` rejects ForwardDiff Duals whose *value*
# is 1.0 but whose partials are nonzero — `one(p)` has zero partials, so Dual
# ordering fails `p <= one(p)` and FE `sigma ~ x` L-BFGS aborts mid-step
# (#385 / nbinom2-dispersion). Soft-clamped η keep r>0 and p∈(0,1] in exact
# arithmetic; disable constructor checks for AD.
@inline _nb2(r, p) = NegativeBinomial(r, p; check_args = false)

# Smooth clamp of a linear predictor, in the spirit of drmTMB's
# `drm_softclamp_log_sigma` (src/drmTMB.cpp): EXACTLY the identity inside the band
# `[lo, hi]` and C1-smoothly saturating within `margin` beyond each bound (overall
# range `[lo - margin, hi + margin]`). It replaces the fixed-effect paths' old hard
# `clamp`, whose gradient was a flat zero beyond the bound — a divergence drmTMB
# never has, since drmTMB does not clamp the mean predictor and soft-clamps only
# the log-scale. We keep each family's identity band equal to its original hard
# bound (rather than drmTMB's numeric [-12,12] log-σ band, which is calibrated for
# the Gaussian `sigma = exp(eta_sigma)` and is too tight for these families'
# `size/shape/precision = exp(-2·eta_sigma)` parameterisation — a legitimate
# near-Poisson NB2 wants eta_sigma ≈ -15, which that band would distort into a
# singular Hessian). So every fit the hard clamp accepted is unchanged to the bit;
# only the flat-gradient runaway tail becomes smooth. Branchless / ForwardDiff-safe.
# See docs/design/03-likelihoods.md ("Numerical guard on the scale linear
# predictor") and #324.
_softclamp(x, lo, hi, margin) =
    ifelse(x > hi, hi + margin * tanh((x - hi) / margin),
           ifelse(x < lo, lo - margin * tanh((lo - x) / margin), x))

"""
    NegBinomial2()

Negative-binomial (NB2) family for overdispersed counts: log link on the mean
`μ`, and log link on the scale `σ` (the `sigma` formula slot, so
`coef(fit, :sigma)` is `log σ`). The NB2 size is `θ = 1/σ² = exp(-2·coef(:sigma))`,
matching the Gamma/Beta convention and `drmTMB` (`size = 1/σ²`). Var = `μ + μ²/θ`;
as `θ → ∞` it tends to [`Poisson`](@ref). Mirrors `drmTMB`'s `nbinom2` family.
Crossed random intercepts on the mean, such as `(1 | g) + (1 | h)`, use the
sparse-Laplace engine when `sigma ~ 1`. A phylogenetic random intercept on the
mean, `phylo(1 | species)`, also uses the sparse-Laplace engine; here a covariate
dispersion formula `sigma ~ x` is supported (a per-observation log σ, so a
per-observation size θ = 1/σ²), while the crossed-intercept route still
requires `sigma ~ 1`.

```julia
fit = drm(bf(y ~ x, sigma ~ 1), NegBinomial2(); data = dat)
fit = drm(bf(y ~ x + (1 | g) + (1 | h), sigma ~ 1), NegBinomial2(); data = dat)
fit_phy = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
              NegBinomial2(); data = dat, tree = tr, se = false)
fit_disp = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ x)),
               NegBinomial2(); data = dat, tree = tr, se = false)  # log σ ~ x
# Coupled phylogenetic location–scale: shared tag + phylo group on both axes.
fit_ls = drm(bf(@formula(y ~ x + (1 | p | phylo(species))),
                @formula(sigma ~ 1 + (1 | p | phylo(species)))),
             NegBinomial2(); data = dat, tree = tr, se = false)
exp(-2 * coef(fit, :sigma)[1])  # estimated size θ = 1/σ²

# Experimental: NB2 random-intercept variational (ELBO) marginal.
# Requires `sigma ~ 1`. Default remains `marginal = :LA` (adaptive GHQ with
# `_RANEF1D_AGHQ_K` nodes on an ordinary `(1 | g)`).
fit_va = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), NegBinomial2();
             data = dat, marginal = :VA)

# TMB-convention Laplace on an ordinary `(1 | g)`, as native drmTMB fits it.
# Requires `sigma ~ 1`; ML only.
fit_lap = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), NegBinomial2();
              data = dat, marginal = :Laplace)

# Predictor-dependent SDs are available for ordinary intercepts and
# independent numeric slopes, and for an animal-model mean intercept with a
# supplied covariance matrix `A` or precision matrix `Ainv`. The animal route
# uses sparse Laplace and requires `sigma ~ 1`. `A` and `Ainv` are used as
# supplied (raw kernel, as in native drmTMB and the Poisson and Gamma animal
# routes); only a `relmat` `K` is rescaled to a correlation matrix.
fit_animal_sd = drm(bf(@formula(y ~ x + animal(1 | id)),
                       @formula(sigma ~ 1),
                       @formula(sd(id, animal) ~ habitat)),
                    NegBinomial2(); data = dat, A = A, se = false)
# `coef(fit_animal_sd, :sd)` models the log SD multiplier by habitat. The
# reported `:sd_group` values multiply the raw supplied `A` kernel. Interval coverage for predictor-dependent SDs is not yet calibrated.

# A predictor-dependent SD can also describe one ordinary random coefficient
# on the log-`sigma` predictor. The mean is fixed-effect-only, and the `sd`
# predictors are fixed and constant within group:
fit_sigma_sd = drm(bf(@formula(y ~ x),
                     @formula(sigma ~ 1 + (1 | g)),
                     @formula(sd(g, sigma) ~ habitat)),
                   NegBinomial2(); data = dat)
fit_sigma_slope_sd = drm(bf(@formula(y ~ x),
                            @formula(sigma ~ 1 + (0 + z | g)),
                            @formula(sd(g, sigma, coef(z)) ~ habitat)),
                          NegBinomial2(); data = dat)
# The sigma-side route uses 11-node adaptive GHQ by default; `nAGQ` overrides it.
# `sigma(fit_sigma_sd)` reports conditional scales at group modes.
# Interval coverage is not calibrated.
```
"""
struct NegBinomial2 end

function drm(f::DrmFormula, fam::NegBinomial2; data, tree = nothing, K = nothing,
             A = nothing, Ainv = nothing, coords = nothing,
             g_tol::Real = 1e-8, se::Bool = true,
             marginal::Symbol = :LA, method = nothing,
             nAGQ::Union{Nothing,Int} = nothing)
    _reject_method_as_marginal(fam, method)
    if !isempty(_sd_parts(f)) || !isempty(_sdphylo_parts(f))
        nAGQ === nothing || nAGQ >= 1 || throw(ArgumentError(
            "nAGQ must be ≥ 1; got $nAGQ"))
        return _drm_negbin2_sd_regression(f, fam; data, tree, K, A, Ainv,
                                           coords, g_tol, se, marginal, nAGQ)
    end
    Ainv === nothing || throw(ArgumentError(
        "NegBinomial2 `Ainv` is currently supported for animal random-effect SD regression only."))
    nAGQ === nothing || throw(ArgumentError(
        "nAGQ is currently supported for NegBinomial2 random-intercept SD regression only."))
    _scalar_laplace_requested(marginal) &&     # Arc 2: TMB-convention Laplace, ordinary (1 | g)
        return _drm_ordinary_laplace(f, fam; data = data, tree = tree, K = K, A = A,
                                     coords = coords, g_tol = g_tol, se = se, method = method)
    missing_fit = _fit_observed_response_rows(f, data) do data_observed
        drm(f, fam; data = data_observed, tree = tree, K = K, A = A,
            coords = coords, g_tol = g_tol, se = se, marginal = marginal, method = method)
    end
    missing_fit !== nothing && return missing_fit

    marg = _marginal_method(marginal)                     # :LA (default), :VA (#136), or :AGHQ (#761)
    isaghq = marg isa AGHQ
    isva = marg isa Variational
    _lss_only_gaussian_guard(f, fam)   # #544: refuse, never silently drop, sd() parts
    rhs = Dict(f.forms)
    # Location–scale: a coupled `(1 | tag | group)` shared by the mean and sigma
    # formulas → one 2×2 group-level covariance fit by the augmented-state engine.
    lc = _ls_coupled_re(rhs[:mu], get(rhs, :sigma, ConstantTerm(1)))
    if lc !== nothing
        isva && _va_reject(fam, "a coupled location–scale random effect")
        isaghq && _aghq_reject(fam, "a coupled location–scale random effect")
        return _withformula(_fit_locscale_frontend(Val(:nb2), fam, f, rhs, lc, data;
                                                    g_tol = g_tol, se = se,
                                                    tree = tree, K = K, A = A,
                                                    coords = coords), f)
    end
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    isaghq && !(length(re) > 1 && st === nothing) &&
        _aghq_reject(fam, "this model (NegBinomial2 `marginal = :AGHQ` covers crossed random " *
                          "intercepts `(1 | g) + (1 | h)` only, #761)")
    mv === nothing ||
        error("NegBinomial2() does not support meta_V markers")
    for (pname, r) in f.forms          # only the mean may carry a random effect
        pname === :mu && continue
        _, re2, mv2, st2 = _split_ranef(r)
        (isempty(re2) && mv2 === nothing && st2 === nothing) ||
            error("NegBinomial2(): only the mean formula may carry a random effect")
    end
    y, Xμ, nmμ = _design(f.response, fixed_mu, data)
    _, Xσ, nmσ = _design(f.response, get(rhs, :sigma, ConstantTerm(1)), data)
    all(yi -> yi ≥ 0 && isinteger(yi), y) ||
        error("NegBinomial2() requires non-negative integer counts as the response")
    if st !== nothing
        isva && _va_reject(fam, "a phylogenetic/structured random effect")
        isempty(re) ||
            error("NegBinomial2() phylo structured effects cannot be combined with ordinary random effects yet")
        (haskey(rhs, :zi) || haskey(rhs, :hu)) &&
            error("NegBinomial2() phylo structured effects cannot be combined with `zi`/`hu` yet")
        # `sigma ~ 1` keeps the scalar-dispersion spine; a covariate `sigma`
        # formula routes to the per-observation log-dispersion path (#164).
        kind, grp = st
        labels = getproperty(data, grp)
        if kind === :phylo
            tree === nothing && error("phylo(1 | $grp) needs `tree = …`")
            return _withformula(_fit_nb2_phylo_laplace(fam, y, Xμ, Xσ, labels, tree, nmμ, nmσ, grp, g_tol; se = se), f)
        elseif kind === :relmat || kind === :animal || kind === :spatial
            # General user-supplied PD covariance C on the mean intercept
            # (relatedness / animal model / precomputed spatial), with the NB2
            # dispersion a fixed nuisance. Reuses the phylo nuisance Laplace spine
            # with the tree precision swapped for C⁻¹ (#167).
            C = _poisson_structured_cov(kind, grp, K, A, coords)
            return _withformula(_fit_nb2_relmat_laplace(fam, y, Xμ, Xσ, C, labels, nmμ, nmσ, grp, g_tol;
                                                          se = se, normalize = kind !== :animal), f)
        else
            error("NegBinomial2() supports phylo/relmat/animal/spatial(1 | group) among structured markers")
        end
    end
    if !isempty(re)                                       # random effect on the mean → GHQ/Laplace or VA (#136)
        isva && (haskey(rhs, :zi) || haskey(rhs, :hu)) &&
            _va_reject(fam, "`zi`/`hu` combined with a random effect")
        (haskey(rhs, :zi) || haskey(rhs, :hu)) &&
            error("NegBinomial2() random effects cannot be combined with `zi`/`hu` yet")
        if length(re) > 1
            isva && _va_reject(fam, "crossed/multiple random intercepts")
            all(_re_kind(r[1])[1] === :intercept for r in re) ||
                error("NegBinomial2() supports multiple random effects only as crossed/nested intercepts, e.g. `(1 | g) + (1 | h)`")
            comps = map(re) do r
                grp = r[2]; gidx, G = _group_index(getproperty(data, grp))
                (ones(length(y)), gidx, G, String(grp))
            end
            isaghq && return _withformula(_fit_nb2_crossed_aghq(fam, y, Xμ, Xσ, comps, nmμ, nmσ, g_tol; se = se), f)   # #761
            return _withformula(_fit_nb2_crossed_laplace(fam, y, Xμ, Xσ, comps, nmμ, nmσ, g_tol; se = se), f)
        end
        (rk, var) = _re_kind(re[1][1]); grp = re[1][2]
        gidx, G = _group_index(getproperty(data, grp))
        if rk === :intercept                              # (1 | g) → 1-D GHQ (Laplace) or VA (#136)
            if isva
                size(Xσ, 2) == 1 || _va_reject(fam, "a non-constant `sigma` formula (VA kernels require `sigma ~ 1`)")
                return _withformula(_withmarginal(
                    _fit_nb2_ranef_va(fam, y, Xμ, Xσ, gidx, G, nmμ, nmσ, grp, g_tol), :VA), f)
            end
            return _withformula(_fit_negbin2_ranef(fam, y, Xμ, Xσ, gidx, G,
                nmμ, nmσ, grp, g_tol), f)
        elseif rk === :corr                               # (1 + x | g) → 2-D GHQ
            isva && _va_reject(fam, "a correlated random slope `(1 + x | g)`")
            return _withformula(_fit_negbin2_corr_ranef(fam, y, Xμ, Xσ, Float64.(getproperty(data, var)), gidx, G, nmμ, nmσ, grp, g_tol), f)
        else
            error("NegBinomial2() supports (1|g) or (1+x|g) on the mean")
        end
    end
    isva && _va_reject(fam, "no random intercept (fixed-effects-only / zi / hu)")
    haskey(rhs, :zi) && haskey(rhs, :hu) &&
        error("`zi` and `hu` cannot both be specified (zero-inflation vs hurdle)")
    if haskey(rhs, :zi)                                   # zero-inflated NB (ZINB)
        _, Xzi, nmzi = _design(f.response, rhs[:zi], data)
        return _withformula(_fit_negbin2_zi(fam, y, Xμ, Xσ, Xzi, nmμ, nmσ, nmzi, g_tol), f)
    end
    if haskey(rhs, :hu)                                   # hurdle NB
        _, Xhu, nmhu = _design(f.response, rhs[:hu], data)
        return _withformula(_fit_negbin2_hu(fam, y, Xμ, Xσ, Xhu, nmμ, nmσ, nmhu, g_tol), f)
    end
    return _withformula(_fit_negbin2(fam, y, Xμ, Xσ, nmμ, nmσ, g_tol), f)
end

# NB2 random-intercept or independent random-slope SD regression. For group k,
#   b_k ~ N(0, exp(2 Z_k' alpha)),
# while the NB2 dispersion remains a fixed-effects-only `sigma ~ 1` parameter.
# A slope-only term multiplies `b_k` by its observation-level predictor. Each
# group integral uses the existing adaptive one-dimensional GHQ machinery.
function _drm_negbin2_sd_regression(f::DrmFormula, fam::NegBinomial2; data,
                                    tree = nothing, K = nothing, A = nothing,
                                    Ainv = nothing, coords = nothing,
                                    g_tol::Real = 1e-8,
                                    se::Bool = true, marginal::Symbol = :LA,
                                    nAGQ::Union{Nothing,Int} = nothing)
    marginal === :LA || throw(ArgumentError(
        "NegBinomial2 `sd(group) ~ …` currently supports `marginal = :LA` only."))
    spatial_sd_target = any(t.target.level === :spatial for t in _sd_target_parts(f))
    (tree === nothing && (coords === nothing || spatial_sd_target)) ||
        throw(ArgumentError("NegBinomial2 `sd(group) ~ …` supports ordinary effects, " *
            "an animal intercept with `A` or `Ainv`, a relmat intercept with `K`, or a coordinate `spatial(0 + x | g)` slope with `coords`; other structured covariance routes are not implemented."))
    isempty(_sdphylo_parts(f)) || throw(ArgumentError(
        "NegBinomial2 `sd(group, phylogenetic) ~ …` is not implemented by the ordinary random-intercept route."))
    targets = _sd_target_parts(f)
    length(targets) == 1 || throw(ArgumentError(
        "NegBinomial2 random-effect SD regression requires exactly one `sd(group) ~ …` formula."))
    target = only(targets)
    descriptor, sgrp, srhs = target.target, target.target.group, target.rhs
    descriptor.dpar === :sigma && return _drm_negbin2_sigma_sd_regression(
        f, fam, target; data, tree, K, A, Ainv, coords, g_tol, se, marginal,
        nAGQ)
    (descriptor.level in (:iid, :animal, :relmat, :spatial) && descriptor.dpar === :mu &&
     descriptor.block === nothing) || throw(ArgumentError(
        "NegBinomial2 SD regression currently targets one ordinary, animal, relmat or coordinate-spatial slope mean random-effect coefficient."))
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st, _, _, animal_slope = _split_ranef(
        rhs[:mu]; allow_animal_slope = descriptor.level === :animal,
        allow_relmat_slope = descriptor.level === :relmat,
        allow_spatial_slope = descriptor.level === :spatial)
    mv === nothing || throw(ArgumentError("NegBinomial2 `sd($sgrp)` cannot use meta_V()."))
    structured_animal = st !== nothing
    if structured_animal
        kind, grp = st
        kind in (:animal, :relmat, :spatial) || throw(ArgumentError(
            "NegBinomial2 predictor-dependent SD currently supports only an animal or relmat intercept, or a coordinate spatial slope, among structured effects."))
        kind === :spatial && animal_slope === nothing && throw(ArgumentError(
            "NegBinomial2 spatial SD regression supports only the slope-only `spatial(0 + x | $grp)` term with `coords = …`."))
        isempty(re) || throw(ArgumentError(
            "NegBinomial2 $kind SD regression cannot yet be combined with ordinary random effects."))
        if kind === :animal
            K === nothing || throw(ArgumentError(
                "NegBinomial2 animal SD regression uses `A` or `Ainv`, not `K`."))
            (A === nothing) == (Ainv === nothing) && throw(ArgumentError(
                "animal(1 | $grp) with predictor-dependent SD requires exactly one of `A` or `Ainv`."))
        elseif kind === :spatial
            (A === nothing && Ainv === nothing && K === nothing) || throw(ArgumentError(
                "NegBinomial2 spatial slope SD regression uses `coords = …` only, not `K`, `A` or `Ainv`."))
            coords === nothing && throw(ArgumentError(
                "spatial(0 + $animal_slope | $grp) needs `coords = …`"))
        else
            (A === nothing && Ainv === nothing) || throw(ArgumentError(
                "NegBinomial2 relmat SD regression uses `K = …` only; do not also supply `A` or `Ainv`."))
            K === nothing && throw(ArgumentError(
                "relmat(1 | $grp) needs the relatedness/covariance matrix `K = …`"))
        end
        nAGQ === nothing || throw(ArgumentError(
            "nAGQ is not used by the sparse-Laplace $kind SD route; omit it."))
        descriptor.level === kind || throw(ArgumentError(
            "$kind($grp) requires a $kind-level SD target."))
        slope = animal_slope
    else
        (A === nothing && Ainv === nothing) || throw(ArgumentError(
            "NegBinomial2 `A` and `Ainv` are supported only for animal random-effect SD regression."))
        K === nothing || throw(ArgumentError(
            "NegBinomial2 `K` is supported only for relmat random-effect SD regression."))
        length(re) == 1 || throw(ArgumentError(
            "NegBinomial2 `sd($sgrp)` currently requires exactly one ordinary random intercept or one independent slope."))
        kind, slope = _re_kind(re[1][1])
        kind in (:intercept, :slope) || throw(ArgumentError(
            "NegBinomial2 `sd($sgrp)` supports `(1 | $sgrp)` or one independent numeric slope `(0 + x | $sgrp)`; correlated blocks are not implemented on this route."))
        grp = re[1][2]
        descriptor.level === :iid || throw(ArgumentError(
            "ordinary NegBinomial2 random effects require an ordinary SD target."))
    end
    nodes = nAGQ === nothing ? _RANEF1D_AGHQ_K : nAGQ
    grp === sgrp || throw(ArgumentError(
        "NegBinomial2 `sd($sgrp)` must name the random-effect grouping variable `$grp`."))
    member = structured_animal ? (slope === nothing ? :intercept : slope) :
             kind === :intercept ? :intercept : slope
    descriptor.member in (nothing, member) || throw(ArgumentError(
        "NegBinomial2 SD target `coef($(descriptor.member))` does not match the fitted random-effect coefficient `$member`."))
    sigma_rhs = get(rhs, :sigma, nothing)
    sigma_rhs isa ConstantTerm && sigma_rhs == ConstantTerm(1) || throw(ArgumentError(
        "NegBinomial2 `sd($grp)` requires a fixed `sigma ~ 1` submodel."))
    haskey(rhs, :zi) && throw(ArgumentError(
        "NegBinomial2 random-effect SD regression cannot yet be combined with `zi`."))
    if haskey(rhs, :hu)
        structured_animal && throw(ArgumentError(
            "Hurdle NegBinomial2 SD regression does not yet support animal or relmat structured effects."))
        fixed_mu, offset_term = _extract_offset(fixed_mu)
        offset_term === nothing || throw(ArgumentError(
            "Hurdle NegBinomial2 SD regression does not yet support `offset(...)`."))
        all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
            "Hurdle NegBinomial2 SD regression does not yet support missing responses."))
        y, Xmu, nmmu = _design(f.response, fixed_mu, data)
        _, Xsigma, nmsigma = _design(f.response, sigma_rhs, data)
        _, Xhu, nmhu = _design(f.response, rhs[:hu], data)
        all(yi -> yi ≥ 0 && isinteger(yi), y) || throw(ArgumentError(
            "NegBinomial2() requires non-negative integer counts as the response"))
        gidx, G = _group_index(getproperty(data, grp))
        Zg, nmsd = _sd_group_design(f.response, srhs, data, gidx, G, grp;
                                    schema_cache = f.schema_cache, schema_key = :sd)
        informative_groups = unique(gidx[y .> 0])
        size(Zg, 2) <= G && !isempty(informative_groups) &&
            rank(Zg[informative_groups, :]) == size(Zg, 2) || throw(ArgumentError(
            "Hurdle NegBinomial2 `sd($grp)` design is rank deficient across positive-count groups; all-zero groups inform `hu` but not the count-side random-effect SD."))
        w = if kind === :intercept
            ones(length(y))
        else
            hasproperty(data, slope) || throw(ArgumentError(
                "Hurdle NegBinomial2 random-slope predictor `$slope` is absent from data."))
            raw = getproperty(data, slope)
            raw isa AbstractVector{<:Real} && length(raw) == length(y) || throw(ArgumentError(
                "Hurdle NegBinomial2 random-slope predictor `$slope` must be numeric with one value per response."))
            all(isfinite, raw) || throw(ArgumentError(
                "Hurdle NegBinomial2 random-slope predictor `$slope` must contain finite values."))
            Float64.(raw)
        end
        for (pname, _) in f.forms
            (pname in (:mu, :sigma, :hu, Symbol("sd_", sgrp)) || haskey(f.sd_targets, pname)) && continue
            throw(ArgumentError("Hurdle NegBinomial2 SD regression supports only `mu`, fixed `sigma ~ 1`, fixed `hu`, and one `sd($grp)` formula."))
        end
        return _withformula(_fit_negbin2_hu_sd_regression(fam, y, Xmu, Xsigma,
            Xhu, Zg, gidx, G, nmmu, nmsigma, nmhu, nmsd, grp, g_tol;
            K = nodes, se, w), f)
    end
    fixed_mu, offset_term = _extract_offset(fixed_mu)
    offset_term === nothing || throw(ArgumentError(
        "NegBinomial2 random-effect SD regression does not yet support `offset(...)`."))
    for (pname, _) in f.forms
        (pname in (:mu, :sigma, Symbol("sd_", sgrp)) || haskey(f.sd_targets, pname)) && continue
        throw(ArgumentError("NegBinomial2 `sd($sgrp)` supports only `mu`, fixed `sigma ~ 1`, and one `sd($sgrp)` formula."))
    end
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "NegBinomial2 random-effect SD regression does not yet support missing responses; " *
        "drop incomplete rows before fitting so the group SD design remains aligned."))

    y, Xmu, nmmu = _design(f.response, fixed_mu, data)
    _, Xsigma, nmsigma = _design(f.response, sigma_rhs, data)
    all(yi -> yi ≥ 0 && isinteger(yi), y) || throw(ArgumentError(
        "NegBinomial2() requires non-negative integer counts as the response"))
    gidx, G = _group_index(getproperty(data, grp))
    Zg, nmsd = _sd_group_design(f.response, srhs, data, gidx, G, grp;
                                schema_cache = f.schema_cache, schema_key = :sd)
    size(Zg, 2) <= G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
        "NegBinomial2 `sd($grp)` group-level design is rank deficient; use predictors that vary independently across groups."))
    w = if structured_animal && slope !== nothing
        _poisson_animal_slope_loading(data, slope, length(y);
                                      label = "NegBinomial2 $kind")
    elseif structured_animal || kind === :intercept
        ones(length(y))
    else
        hasproperty(data, slope) || throw(ArgumentError(
            "NegBinomial2 random-slope predictor `$slope` is absent from data."))
        raw = getproperty(data, slope)
        raw isa AbstractVector{<:Real} || throw(ArgumentError(
            "NegBinomial2 random-slope predictor `$slope` must be numeric."))
        all(isfinite, raw) || throw(ArgumentError(
            "NegBinomial2 random-slope predictor `$slope` must contain finite values."))
        Float64.(raw)
    end
    if structured_animal
        C = if Ainv !== nothing
            nothing
        elseif kind === :spatial
            # Native fixed kernel exp(-d / median positive distance) + 1e-6 I, RAW.
            _gaussian_spatial_sd_kernel(coords, _group_index(getproperty(data, grp))[2], grp)
        else
            _poisson_structured_cov(kind, grp, K, A, coords)
        end
        fit = _fit_nb2_animal_sd_laplace(fam, y, Xmu, Xsigma, Zg, gidx,
            C, getproperty(data, grp), nmmu, nmsigma, nmsd, grp, g_tol; se, Ainv,
            # Animal A and slope-only relmat K enter RAW (native drmTMB does not
            # rescale them); the relmat intercept route keeps its normalised K.
            normalize = kind !== :animal && slope === nothing, weight = w)
    else
        fit = _fit_negbin2_sd_regression(fam, y, Xmu, Xsigma, Zg,
            gidx, G, nmmu, nmsigma, nmsd, grp, g_tol; K = nodes, se, w)
    end
    return _withformula(fit, f)
end

# Predictor-dependent SD for one ordinary NB2 random coefficient on log sigma.
# The mean remains fixed-effect-only; sigma has a fixed-effects component plus
# one independent intercept or numeric slope. For group g,
#   b_g ~ Normal(0, exp(2 W_g alpha)),
# and eta_sigma_i = X_sigma_i beta_sigma + z_i b_g.
function _drm_negbin2_sigma_sd_regression(f::DrmFormula, fam::NegBinomial2,
        target; data, tree, K, A, Ainv, coords, g_tol::Real, se::Bool,
        marginal::Symbol, nAGQ::Union{Nothing,Int})
    marginal === :LA || throw(ArgumentError(
        "NegBinomial2 sigma SD regression currently supports `marginal = :LA` only."))
    (tree === nothing && K === nothing && A === nothing && Ainv === nothing &&
     coords === nothing) || throw(ArgumentError(
        "NegBinomial2 sigma SD regression currently supports ordinary IID effects only."))
    descriptor = target.target
    (descriptor.level === :iid && descriptor.dpar === :sigma &&
     descriptor.block === nothing) || throw(ArgumentError(
        "NegBinomial2 sigma SD regression targets one ordinary sigma random coefficient."))
    nAGQ === nothing || nAGQ >= 1 || throw(ArgumentError(
        "nAGQ must be ≥ 1; got $nAGQ"))
    rhs = Dict(f.forms)
    fixed_mu, re_mu, mv_mu, st_mu = _split_ranef(rhs[:mu])
    isempty(re_mu) && mv_mu === nothing && st_mu === nothing || throw(ArgumentError(
        "NegBinomial2 sigma SD regression requires fixed-effect-only `mu`."))
    sigma_rhs = get(rhs, :sigma, nothing)
    sigma_rhs === nothing && throw(ArgumentError(
        "NegBinomial2 sigma SD regression requires one random coefficient in `sigma`."))
    fixed_sigma, re_sigma, mv_sigma, st_sigma = _split_ranef(sigma_rhs)
    mv_sigma === nothing && st_sigma === nothing && length(re_sigma) == 1 ||
        throw(ArgumentError(
        "NegBinomial2 sigma SD regression currently supports one ordinary sigma random coefficient."))
    kind, slope = _re_kind(re_sigma[1][1])
    kind in (:intercept, :slope) || throw(ArgumentError(
        "NegBinomial2 sigma SD regression supports one independent intercept or numeric slope."))
    group = re_sigma[1][2]
    group === descriptor.group || throw(ArgumentError(
        "`sd($(descriptor.group), sigma)` must target the sigma random-effect group `$group`."))
    member = kind === :intercept ? nothing : slope
    descriptor.member === member || throw(ArgumentError(
        "NegBinomial2 sigma SD target coefficient `$(descriptor.member)` does not match the fitted coefficient `$member`."))
    zi_rhs = get(rhs, :zi, nothing)
    if zi_rhs !== nothing
        kind === :intercept || throw(ArgumentError(
            "NegBinomial2 sigma SD regression with `zi` currently supports only a sigma random intercept."))
        fixed_sigma isa ConstantTerm && fixed_sigma == ConstantTerm(1) ||
            throw(ArgumentError(
            "NegBinomial2 sigma SD regression with `zi` currently requires fixed `sigma ~ 1`."))
    end
    allowed = Set((:mu, :sigma, target.key))
    zi_rhs === nothing || push!(allowed, :zi)
    all(p -> first(p) in allowed || haskey(f.sd_targets, first(p)), f.forms) ||
        throw(ArgumentError(
        "NegBinomial2 sigma SD regression supports fixed `mu`, one sigma random coefficient, and one sigma-targeted SD formula."))
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "NegBinomial2 sigma SD regression requires observed responses; remove incomplete rows before fitting."))
    Xzi = nothing
    nmzi = String[]
    if zi_rhs !== nothing
        fixed_zi, re_zi, mv_zi, st_zi = _split_ranef(zi_rhs)
        isempty(re_zi) && mv_zi === nothing && st_zi === nothing &&
            fixed_zi isa ConstantTerm && fixed_zi == ConstantTerm(1) ||
            throw(ArgumentError(
            "NegBinomial2 sigma SD regression with `zi` currently requires fixed `zi ~ 1`."))
        _, Xzi, nmzi = _design(f.response, fixed_zi, data)
    end
    y, Xmu, nmmu = _design(f.response, fixed_mu, data)
    _, Xsigma, nmsigma = _design(f.response, fixed_sigma, data)
    all(yi -> isfinite(yi) && yi >= 0 && isinteger(yi), y) || throw(ArgumentError(
        "NegBinomial2() requires finite, non-negative integer counts as the response"))
    gidx, G = _group_index(getproperty(data, group))
    Wsd, nmsd = _sd_group_design(f.response, target.rhs, data, gidx, G, group;
        schema_cache = f.schema_cache, schema_key = target.key)
    size(Wsd, 2) <= G && rank(Wsd) == size(Wsd, 2) || throw(ArgumentError(
        "NegBinomial2 `sd($group, sigma)` design is rank deficient across groups; use predictors that vary independently across groups."))
    loading = if kind === :intercept
        ones(length(y))
    else
        hasproperty(data, slope) || throw(ArgumentError(
            "NegBinomial2 sigma random-slope predictor `$slope` is absent from data."))
        raw = getproperty(data, slope)
        raw isa AbstractVector{<:Real} && length(raw) == length(y) &&
            all(isfinite, raw) || throw(ArgumentError(
            "NegBinomial2 sigma random-slope predictor `$slope` must be numeric and finite."))
        Float64.(raw)
    end
    return _withformula(_fit_negbin2_sigma_sd_regression(fam, y, Xmu,
        Xsigma, Xzi, Wsd, gidx, G, nmmu, nmsigma, nmzi, nmsd, group, g_tol;
        # Sigma-side SDs use 11 nodes by default: the five-node NB2 mean-side
        # rule left a 0.19 log-likelihood gap against the independent integral
        # on the initial slope fixture.
        K = nAGQ === nothing ? 11 : nAGQ, se, loading,
        ranef_name = kind === :intercept ? Symbol(group, "_logsigma") :
            Symbol(group, "_", slope, "_logsigma")), f)
end

function _fit_negbin2_sigma_sd_regression(fam::NegBinomial2, y, Xmu,
        Xsigma, Xzi, Wsd, gidx, G, nmmu, nmsigma, nmzi, nmsd, group, g_tol;
        K::Int = _RANEF1D_AGHQ_K, se::Bool = true,
        loading::AbstractVector{<:Real} = ones(length(y)), ranef_name::Symbol)
    n = length(y)
    length(loading) == n || throw(ArgumentError(
        "NB2 sigma random-effect loading must have one value per observation."))
    pmu, psigma, pzi, psd = size(Xmu, 2), size(Xsigma, 2),
        (Xzi === nothing ? 0 : size(Xzi, 2)), size(Wsd, 2)
    has_zi = pzi > 0
    yint = round.(Int, y)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    rule = _AGHQRule(1, K)
    Zre = reshape(Float64.(loading), n, 1)
    bcache = [zeros(1, G) for _ in 1:Threads.nthreads()]
    function nll(theta)
        beta_mu = @view theta[1:pmu]
        beta_sigma = @view theta[(pmu+1):(pmu+psigma)]
        zi_range = (pmu + psigma + 1):(pmu + psigma + pzi)
        beta_zi = has_zi ? @view(theta[zi_range]) : nothing
        sd_range = (pmu + psigma + pzi + 1):(pmu + psigma + pzi + psd)
        alpha = @view theta[sd_range]
        eta_mu = _softclamp.(Xmu * beta_mu, -20.0, 20.0, 3.0)
        eta_sigma = Xsigma * beta_sigma
        eta_zi = has_zi ? clamp.(Xzi * beta_zi, -30.0, 30.0) : nothing
        tau = exp.(Wsd * alpha)
        cache = bcache[Threads.threadid()]
        ll = (i, eta) -> begin
            log_sigma = eta_sigma[i] + eta
            shape = exp(clamp(-2 * log_sigma, -8.0, 8.0))
            log_nb = _nb2_logpmf(shape, exp(eta_mu[i]), yint[i])
            if has_zi
                log_zi = _log_logistic(eta_zi[i])
                log_one_minus_zi = _log1m_logistic(eta_zi[i])
                return yint[i] == 0 ?
                    _logaddexp(log_zi, log_one_minus_zi + log_nb) :
                    log_one_minus_zi + log_nb
            end
            return log_nb
        end
        total = zero(eltype(theta))
        for gi in 1:G
            value, bhat = _aghq_group_logint(ll, members[gi], zeros(n), Zre,
                reshape([tau[gi]], 1, 1), rule, @view(cache[:, gi]))
            cache[1, gi] = isfinite(bhat[1]) ? bhat[1] : 0.0
            total += value
        end
        return -total
    end
    theta0 = zeros(pmu + psigma + pzi + psd)
    positive_mean = max(mean(y), 0.1)
    theta0[1:pmu] .= Xmu \ fill(log(positive_mean), n)
    theta0[(pmu+1):(pmu+psigma)] .= Xsigma \ fill(-0.5, n)
    theta0[(pmu+psigma+pzi+1):end] .= Wsd \ fill(log(0.4), G)
    result = Optim.optimize(nll, theta0, Optim.LBFGS(),
                            Optim.Options(g_tol = g_tol); autodiff = :forward)
    theta_hat = Optim.minimizer(result)
    V = se ? _vcov_from_hessian(ForwardDiff.hessian(nll, theta_hat)) :
             fill(NaN, length(theta_hat), length(theta_hat))
    zi_range = (pmu + psigma + 1):(pmu + psigma + pzi)
    sd_range = (pmu + psigma + pzi + 1):(pmu + psigma + pzi + psd)
    tau = exp.(Wsd * theta_hat[sd_range])
    # Re-evaluate each conditional mode at the optimum for `sigma(fit)` and
    # `ranef(fit)`; these are the same conditional summaries used by Gamma.
    modes = zeros(G)
    eta_mu = _softclamp.(Xmu * theta_hat[1:pmu], -20.0, 20.0, 3.0)
    eta_sigma = Xsigma * theta_hat[(pmu+1):(pmu+psigma)]
    eta_zi = has_zi ? clamp.(Xzi * theta_hat[zi_range], -30.0, 30.0) : nothing
    ll = (i, eta) -> begin
        shape = exp(clamp(-2 * (eta_sigma[i] + eta), -8.0, 8.0))
        log_nb = _nb2_logpmf(shape, exp(eta_mu[i]), yint[i])
        if has_zi
            log_zi = _log_logistic(eta_zi[i])
            log_one_minus_zi = _log1m_logistic(eta_zi[i])
            return yint[i] == 0 ?
                _logaddexp(log_zi, log_one_minus_zi + log_nb) :
                log_one_minus_zi + log_nb
        end
        return log_nb
    end
    for gi in 1:G
        _, mode = _aghq_group_logint(ll, members[gi], zeros(n), Zre,
            reshape([tau[gi]], 1, 1), rule, [0.0])
        modes[gi] = mode[1]
    end
    sigma = exp.(eta_sigma .+ loading .* modes[gidx])
    scales = Dict{Symbol,Vector{Float64}}(:sigma => sigma,
        :sigma_random_effect => modes, :sd_group => tau, :sd => tau[gidx],
        :sd_group_index => Float64.(gidx), :aghq_nodes => [Float64(K)])
    blocks = [:mu => (1:pmu), :sigma => ((pmu+1):(pmu+psigma))]
    names = [:mu => nmmu, :sigma => nmsigma]
    if has_zi
        push!(blocks, :zi => zi_range)
        push!(names, :zi => nmzi)
        scales[:zi] = _logistic.(eta_zi)
    end
    push!(blocks, :sd => sd_range)
    push!(names, :sd => nmsd)
    fit = DrmFit(fam, blocks, names, theta_hat, Matrix(V), -nll(theta_hat), n,
        Optim.converged(result), Dict(:mu => exp.(eta_mu)),
        Dict(:mu => Vector{Float64}(y)), scales)
    fit = _withranef(fit, Dict(ranef_name => modes))
    return _withiterations(_withnll(fit, nll), Optim.iterations(result))
end

# Hurdle NB2 with a predictor-dependent SD on one ordinary mean intercept or
# one independently selected numeric mean slope.
# The hurdle probability and NB2 dispersion remain fixed-effect components.
# Since `hu` is independent of the latent intercept, zero rows carry no
# information about the mean SD; groups with no positive counts reduce to the
# hurdle likelihood alone.
function _fit_negbin2_hu_sd_regression(fam::NegBinomial2, y, Xmu, Xsigma,
        Xhu, Zg, gidx, G, nmmu, nmsigma, nmhu, nmsd, grp, g_tol;
        K::Int = _RANEF1D_AGHQ_K, se::Bool = true,
        w::AbstractVector{<:Real} = ones(length(y)))
    n = length(y)
    length(w) == n || throw(ArgumentError(
        "Hurdle NB2 random-effect loading must have one value per observation."))
    pmu, psigma, phu, psd = size(Xmu, 2), size(Xsigma, 2), size(Xhu, 2), size(Zg, 2)
    yint = round.(Int, y)
    zero_y = iszero.(yint)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    rule = _AGHQRule(1, K)
    Zre = reshape(Float64.(w), n, 1)
    bcache = [zeros(1, G) for _ in 1:Threads.nthreads()]
    function nll(theta)
        beta_mu = @view theta[1:pmu]
        beta_sigma = @view theta[(pmu+1):(pmu+psigma)]
        beta_hu = @view theta[(pmu+psigma+1):(pmu+psigma+phu)]
        alpha = @view theta[(pmu+psigma+phu+1):(pmu+psigma+phu+psd)]
        eta_mu = Xmu * beta_mu
        eta_sigma = clamp.(Xsigma * beta_sigma, -20.0, 20.0)
        eta_hu = clamp.(Xhu * beta_hu, -30.0, 30.0)
        sdgroup = exp.(Zg * alpha)
        cache = bcache[Threads.threadid()]
        ll = function (i, eta)
            log_pi = _log_logistic(eta_hu[i])
            zero_y[i] && return log_pi
            log_one_minus_pi = _log1m_logistic(eta_hu[i])
            mu = exp(clamp(eta, -20.0, 20.0))
            r = exp(-2 * eta_sigma[i])
            log_nb = _nb2_logpmf(r, mu, yint[i])
            return log_one_minus_pi + log_nb - _log1mexp(_nb2_logpmf(r, mu, 0))
        end
        total = zero(eltype(theta))
        for gi in 1:G
            rows = members[gi]
            if all(i -> zero_y[i], rows)
                total -= sum(i -> _log_logistic(eta_hu[i]), rows)
                cache[1, gi] = 0.0
                continue
            end
            value, bhat = _aghq_group_logint(ll, rows, eta_mu, Zre,
                reshape([sdgroup[gi]], 1, 1), rule, @view(cache[:, gi]))
            cache[1, gi] = isfinite(bhat[1]) ? bhat[1] : 0.0
            total -= value
        end
        return total
    end
    positive = y[y .> 0]
    m = isempty(positive) ? 1.0 : mean(positive)
    variance = sum(abs2, y .- mean(y)) / max(n - 1, 1)
    theta0 = zeros(pmu + psigma + phu + psd)
    theta0[1] = log(m + eps())
    theta0[pmu+1] = -0.5 * log(max(m^2 / max(variance - m, 0.1m + eps()), 0.5))
    theta0[(pmu+psigma+phu+1):end] .= Zg \ fill(log(0.5), G)
    zero_rate = clamp(mean(zero_y), 1e-5, 1 - 1e-5)
    theta0[(pmu+psigma+1):(pmu+psigma+phu)] .=
        Xhu \ fill(log(zero_rate / (1 - zero_rate)), n)
    result = Optim.optimize(nll, theta0, Optim.LBFGS(),
                            Optim.Options(g_tol = g_tol); autodiff = :forward)
    theta_hat = Optim.minimizer(result)
    V = se ? _vcov_from_hessian(ForwardDiff.hessian(nll, theta_hat)) :
             fill(NaN, length(theta_hat), length(theta_hat))
    hu_range = (pmu + psigma + 1):(pmu + psigma + phu)
    sd_range = (pmu + psigma + phu + 1):length(theta_hat)
    blocks = [:mu => 1:pmu, :sigma => (pmu+1):(pmu+psigma),
              :hu => hu_range, :sd => sd_range]
    names = [:mu => nmmu, :sigma => nmsigma, :hu => nmhu, :sd => nmsd]
    sdgroup = exp.(Zg * theta_hat[sd_range])
    scales = Dict(:sigma => exp.(Xsigma * theta_hat[(pmu+1):(pmu+psigma)]),
                  :hu => _logistic.(Xhu * theta_hat[hu_range]),
                  :sd_group => sdgroup, :sd => sdgroup[gidx],
                  :sd_group_index => Float64.(gidx), :aghq_nodes => [Float64(K)])
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, theta_hat, V, -nll(theta_hat), n,
            Optim.converged(result), Dict(:mu => exp.(Xmu * theta_hat[1:pmu])),
            Dict(:mu => Vector{Float64}(y)), scales), nll),
        Optim.iterations(result))
end

# NB2 animal-intercept SD regression. The supplied animal covariance is first
# mapped to the correlation convention used by the existing structured NB2
# route. The group multiplier is tau_g = exp(z_g' alpha), and NB2 dispersion
# remains a separate fixed sigma parameter. Since sparse Cholesky operations
# are not ForwardDiff-generic, the converged inner Laplace objective is
# optimized with finite-difference outer derivatives.
function _nb2_animal_precision_setup(Ainv::AbstractMatrix, labels, group::Symbol)
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

function _fit_nb2_animal_sd_laplace(fam::NegBinomial2, y, Xmu, Xsigma, Zg,
        gidx, C, labels, nmmu, nmsigma, nmsd, grp, g_tol;
        se::Bool = true, maxiter::Int = 250, Ainv = nothing,
        normalize::Bool = true, weight = ones(length(y)))
    n = length(y)
    length(weight) == n || throw(ArgumentError(
        "NB2 animal random-effect loading must have one value per response."))
    pmu, psd = size(Xmu, 2), size(Zg, 2)
    Q, group_index = Ainv === nothing ? _general_cov_setup(C, labels; normalize) :
        _nb2_animal_precision_setup(Ainv, labels, grp)
    group_index == gidx || throw(ArgumentError(
        "animal covariance labels do not match the fitted observation group order."))
    qchol = cholesky(Symmetric(Q); check = false)
    issuccess(qchol) || throw(ArgumentError(
        "animal covariance precision must be positive definite"))
    logdetQ = logdet(qchol)
    aux_from, beta_start, sigma_start = _nb2_laplace_setup(y, Xmu)
    last_u = zeros(size(Q, 1))

    function mode(beta, log_sigma, alpha; u0 = nothing, max_mode_iter::Int = 60)
        aux = aux_from(log_sigma)
        loading = exp.(Zg * alpha)
        all(isfinite, loading) || return zeros(size(Q, 1)), nothing, false
        u = u0 === nothing ? zeros(size(Q, 1)) : copy(u0)
        eta0 = Xmu * beta
        function joint(uu)
            value = 0.5 * dot(uu, Q * uu)
            @inbounds for i in 1:n
                eta = eta0[i] + weight[i] * loading[gidx[i]] * uu[gidx[i]]
                value += _laplace_v123(Val(:nb2_fixed), aux, i, eta)[1]
            end
            return value
        end
        for _ in 1:max_mode_iter
            grad = Q * u
            curvature = zeros(Float64, size(Q, 1))
            @inbounds for i in 1:n
                gi = gidx[i]
                li = weight[i] * loading[gi]
                eta = eta0[i] + li * u[gi]
                _, d1, d2, _ = _laplace_v123(Val(:nb2_fixed), aux, i, eta)
                grad[gi] += li * d1
                curvature[gi] += li^2 * d2
            end
            H = Q + spdiagm(0 => curvature)
            ch = cholesky(Symmetric(H); check = false)
            issuccess(ch) || return u, ch, false
            step = ch \ grad
            norm(step) <= 1e-10 * (1 + norm(u)) && return u, ch, true
            # Near the mode, objective changes can fall below floating-point
            # resolution even while the Newton step is still larger than tol.
            # Accept this small step as in the Poisson structured-SD mode
            # finder instead of incorrectly reporting a failed line search.
            if norm(step) <= 1e-3 * (1 + norm(u))
                u .-= step
                continue
            end
            f0 = joint(u)
            rate = 1.0
            accepted = false
            while rate >= 1e-5
                candidate = u .- rate .* step
                if joint(candidate) <= f0
                    u = candidate
                    accepted = true
                    break
                end
                rate *= 0.5
            end
            accepted || return u, ch, false
        end
        return u, nothing, false
    end

    function laplace(theta; u0 = nothing)
        beta = @view theta[1:pmu]
        log_sigma = theta[pmu + 1]
        alpha = @view theta[(pmu + 2):(pmu + 1 + psd)]
        u, ch, ok = mode(beta, log_sigma, alpha; u0)
        ok || return 1e18, u, false
        aux = aux_from(log_sigma)
        loading = exp.(Zg * alpha)
        eta0 = Xmu * beta
        value = 0.5 * dot(u, Q * u)
        @inbounds for i in 1:n
            eta = eta0[i] + weight[i] * loading[gidx[i]] * u[gidx[i]]
            value += _laplace_v123(Val(:nb2_fixed), aux, i, eta)[1]
        end
        return value - 0.5 * logdetQ + 0.5 * logdet(ch), u, true
    end

    function nll(theta)
        value, u, ok = laplace(theta; u0 = last_u)
        if !ok
            value, u, ok = laplace(theta; u0 = zeros(length(last_u)))
        end
        ok || return 1e18
        last_u .= u
        return value
    end
    function grad!(gradient, theta)
        for j in eachindex(theta)
            h = 2e-4 * (1 + abs(theta[j]))
            e = zeros(length(theta)); e[j] = h
            gradient[j] = (nll(theta .+ e) - nll(theta .- e)) / (2h)
        end
        return gradient
    end

    theta0 = zeros(pmu + 1 + psd)
    theta0[1:pmu] .= beta_start
    theta0[pmu + 1] = sigma_start
    theta0[(pmu + 2):end] .= Zg \ fill(log(0.4), size(Zg, 1))
    od = Optim.OnceDifferentiable(nll, grad!, theta0)
    result = Optim.optimize(od, theta0,
        Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking()),
        Optim.Options(g_tol = g_tol, iterations = maxiter))
    theta_hat = Optim.minimizer(result)
    value = nll(theta_hat)
    final_gradient = zeros(length(theta_hat)); grad!(final_gradient, theta_hat)
    converged = _laplace_outer_converged(
        result, value, final_gradient, theta_hat, n, g_tol)
    V = se ? _vcov_from_fd_hessian(
        _finite_hessian(nll, theta_hat; h = _fd_hessian_step(n));
        context = "sparse-Laplace NB2 animal SD regression") :
        fill(NaN, length(theta_hat), length(theta_hat))
    sd_range = (pmu + 2):(pmu + 1 + psd)
    sd_group = exp.(Zg * theta_hat[sd_range])
    sigma = exp.(Xsigma * [theta_hat[pmu + 1]])
    scales = Dict{Symbol,Vector{Float64}}(:sigma => sigma,
                  :sd_group => sd_group, :sd => sd_group[gidx],
                  :sd_group_index => Float64.(gidx))
    blocks = [:mu => (1:pmu), :sigma => ((pmu + 1):(pmu + 1)),
              :sd => sd_range]
    names = [:mu => nmmu, :sigma => nmsigma, :sd => nmsd]
    fit = DrmFit(fam, blocks, names, theta_hat, Matrix(V), -value, n,
        converged, Dict(:mu => exp.(Xmu * theta_hat[1:pmu])),
        Dict(:mu => Vector{Float64}(y)), scales)
    return _withiterations(_withnll(fit, nll), Optim.iterations(result))
end

function _fit_negbin2_sd_regression(fam::NegBinomial2, y, Xmu, Xsigma, Zg,
                                     gidx, G, nmmu, nmsigma, nmsd, grp, g_tol;
                                     K::Int = _RANEF1D_AGHQ_K, se::Bool = true,
                                     w::AbstractVector{<:Real} = ones(length(y)))
    n = length(y); pmu, psigma, psd = size(Xmu, 2), size(Xsigma, 2), size(Zg, 2)
    length(w) == n || throw(ArgumentError("NB2 random-effect loading must have one value per observation."))
    yint = round.(Int, y)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    rule = _AGHQRule(1, K)
    Zre = reshape(Float64.(w), n, 1)
    # A profile/bootstrap caller may evaluate the stored objective concurrently.
    # Each Julia thread therefore gets its own warm-start cache; the cache affects
    # convergence speed only, not the objective value.
    bcache = [zeros(1, G) for _ in 1:Threads.nthreads()]
    function nll(theta)
        beta_mu = @view theta[1:pmu]
        beta_sigma = @view theta[(pmu+1):(pmu+psigma)]
        alpha = @view theta[(pmu+psigma+1):(pmu+psigma+psd)]
        # Match `_fit_negbin2_ranef`: add the random intercept first and clamp
        # the combined predictor inside the observation log-pmf. Clamping the
        # fixed predictor here would change the likelihood when b offsets a
        # fixed predictor outside the identity band.
        eta_mu = Xmu * beta_mu
        eta_sigma = _softclamp.(Xsigma * beta_sigma, -20.0, 20.0, 3.0)
        sdgroup = exp.(Zg * alpha)
        cache = bcache[Threads.threadid()]
        ll = (i, eta) -> _nb2_logpmf(exp(-2 * eta_sigma[i]),
                                      exp(clamp(eta, -20.0, 20.0)), yint[i])
        total = zero(eltype(theta))
        for gi in 1:G
            Lg = reshape([sdgroup[gi]], 1, 1)
            value, bhat = _aghq_group_logint(ll, members[gi], eta_mu, Zre,
                                               Lg, rule, @view(cache[:, gi]))
            cache[1, gi] = isfinite(bhat[1]) ? bhat[1] : 0.0
            total += value
        end
        return -total
    end
    m = sum(y) / n
    v = sum(abs2, y .- m) / max(n - 1, 1)
    theta0 = zeros(pmu + psigma + psd)
    theta0[1] = log(m + eps())
    theta0[pmu+1] = -0.5 * log(max(m^2 / max(v - m, 0.1 * m + eps()), 0.5))
    theta0[(pmu+psigma+1):end] .= Zg \ fill(log(0.5), G)
    res = Optim.optimize(nll, theta0, Optim.LBFGS(),
                         Optim.Options(g_tol = g_tol); autodiff = :forward)
    theta_hat = Optim.minimizer(res)
    V = se ? _vcov_from_hessian(ForwardDiff.hessian(nll, theta_hat)) :
             fill(NaN, length(theta_hat), length(theta_hat))
    blocks = [:mu => 1:pmu, :sigma => (pmu+1):(pmu+psigma),
              :sd => (pmu+psigma+1):(pmu+psigma+psd)]
    names = [:mu => nmmu, :sigma => nmsigma, :sd => nmsd]
    sdgroup = exp.(Zg * theta_hat[(pmu+psigma+1):end])
    means = Dict(:mu => exp.(Xmu * theta_hat[1:pmu]))
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xsigma * theta_hat[(pmu+1):(pmu+psigma)]),
                  :sd_group => sdgroup, :sd => sdgroup[gidx],
                  :sd_group_index => Float64.(gidx), :aghq_nodes => [Float64(K)])
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, theta_hat, V, -nll(theta_hat), n,
                        Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# NB2 count GLMM with a random intercept (1|g) on log μ. b_g ~ N(0,σ_b²) is
# integrated out per group by `_RANEF1D_AGHQ_K`-node adaptive Gauss–Hermite
# quadrature; the scale `log σ`
# (the `sigma` slot, size θ = 1/σ²) is a fixed effect. Same scheme as the Poisson
# GLMM.
function _fit_negbin2_ranef(fam::NegBinomial2, y, Xμ, Xσ, gidx, G, nmμ, nmσ, grp, g_tol; K::Int = _RANEF1D_AGHQ_K)
    n = length(y); pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    yint = round.(Int, y)
    rule = _AGHQRule(1, K); Zre = ones(n, 1); bcache = zeros(1, G)   # #719: per-group AGHQ
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; σb = exp(θ[pμ+pσ+1])
        # #982 / #1039: `_nb2_logpmf` never forms p = r/(r+μ), which quantises
        # at near-Poisson size (r ≳ 1e12) and made this objective collapse and
        # turn irreproducible there; the soft clamp (identity inside ±20, as
        # the fixed-effect path) keeps the dispersion gradient alive past the
        # band instead of the hard clamp's exactly-zero AD derivative (#959).
        η0 = Xμ * βμ; ησ = _softclamp.(Xσ * βσ, -20.0, 20.0, 3.0)
        ll = (i, η) -> (μ = exp(clamp(η, -20.0, 20.0)); r = exp(-2 * ησ[i]);
                        _nb2_logpmf(r, μ, yint[i]))
        L = reshape([σb], 1, 1)
        return -_aghq_marginal_loglik(ll, members, η0, Zre, L, rule, bcache)
    end
    m = sum(y) / n; v = sum(abs2, y .- m) / max(n - 1, 1)
    θ0 = zeros(pμ + pσ + 1)
    θ0[1] = log(m + eps())
    θ0[pμ+1] = -0.5 * log(max(m^2 / max(v - m, 0.1 * m + eps()), 0.5))
    θ0[pμ+pσ+1] = log(0.5)
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :resd => (pμ+pσ+1):(pμ+pσ+1)]
    names = [:mu => nmμ, :sigma => nmσ, :resd => [String(grp)]]
    means = Dict(:mu => exp.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => Vector{Float64}(y))   # population μ (b=0)
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# NB2 count GLMM with a CORRELATED random intercept+slope (1 + x | g) on log μ:
# per group (b0,b1) ~ N(0, Σ_re), Σ_re a 2×2 covariance (log-Cholesky a, b, c).
# Unlike the Gaussian case there is no closed-form marginal (b enters μ through
# exp), so each group's 2-D integral is done by per-group ADAPTIVE Gauss–Hermite
# quadrature (`_aghq_marginal_loglik`, #834: nodes b̂_g + √2 C z at each group's
# mode), `nq` nodes per axis. Mirrors the Poisson /
# NB2 random-intercept route extended to two dimensions; the `recov` block and
# names follow the Gaussian correlated fit so `vc(fit)` reconstructs Σ.
function _fit_negbin2_corr_ranef(fam::NegBinomial2, y, Xμ, Xσ, xs, gidx, G, nmμ, nmσ, grp, g_tol; nq::Int = _CORR_RANEF_AGHQ_K)
    n = length(y); pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    yint = round.(Int, y)
    rule = _AGHQRule(2, nq); Zre = hcat(ones(n), Float64.(xs)); bcache = zeros(2, G)   # #834: per-group AGHQ
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]
        L = _corr_ranef_L(θ[pμ+pσ+1], θ[pμ+pσ+2], θ[pμ+pσ+3])   # Σ_re = L Lᵀ
        η0 = Xμ * βμ; ησ = _softclamp.(Xσ * βσ, -20.0, 20.0, 3.0)   # as `_fit_negbin2_ranef` (#982/#1039)
        ll = (i, η) -> (μ = exp(clamp(η, -20.0, 20.0)); r = exp(-2 * ησ[i]);
                        _nb2_logpmf(r, μ, yint[i]))
        return -_aghq_marginal_loglik(ll, members, η0, Zre, L, rule, bcache)
    end
    m = sum(y) / n; v = sum(abs2, y .- m) / max(n - 1, 1)
    θ0 = zeros(pμ + pσ + 3)
    θ0[1] = log(m + eps())
    θ0[pμ+1] = -0.5 * log(max(m^2 / max(v - m, 0.1 * m + eps()), 0.5))   # MoM dispersion init (as in _fit_negbin2_ranef)
    θ0[pμ+pσ+1] = log(0.4); θ0[pμ+pσ+2] = log(0.4); θ0[pμ+pσ+3] = 0.0
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :recov => (pμ+pσ+1):(pμ+pσ+3)]
    names = [:mu => nmμ, :sigma => nmσ, :recov => ["$(grp):L11", "$(grp):L22", "$(grp):L21"]]
    means = Dict(:mu => exp.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => Vector{Float64}(y))   # population μ (b=0)
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# Zero-inflated NB2: P(0) = π + (1-π)·NB(0), P(k>0) = (1-π)·NB(k), with
# π = logistic(Xziᵀβ). Reuses the log-logistic / logaddexp helpers (poisson.jl).
function _fit_negbin2_zi(fam::NegBinomial2, y, Xμ, Xσ, Xzi, nmμ, nmσ, nmzi, g_tol)
    n = length(y); pμ, pσ, pz = size(Xμ, 2), size(Xσ, 2), size(Xzi, 2)
    yint = round.(Int, y); iszero_y = y .== 0
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; βz = θ[pμ+pσ+1:pμ+pσ+pz]
        ημ = clamp.(Xμ * βμ, -20.0, 20.0); ησ = clamp.(Xσ * βσ, -20.0, 20.0)
        ηz = clamp.(Xzi * βz, -30.0, 30.0)
        s = zero(eltype(θ))
        @inbounds for i in 1:n
            μ = exp(ημ[i]); r = exp(-2 * ησ[i])
            lπ = _log_logistic(ηz[i]); l1mπ = _log1m_logistic(ηz[i])
            nb = _nb2_logpmf(r, μ, yint[i])                # #998: no quantised p
            if iszero_y[i]
                s -= _logaddexp(lπ, l1mπ + nb)             # log(π + (1-π)·NB(0))
            else
                s -= l1mπ + nb
            end
        end
        return s
    end
    pos = y[y.>0]; m = isempty(pos) ? sum(y) / n : sum(pos) / length(pos)
    v = sum(abs2, y .- sum(y) / n) / max(n - 1, 1)
    θ0 = zeros(pμ + pσ + pz)
    θ0[1] = log(m + eps())
    θ0[pμ+1] = -0.5 * log(max(m^2 / max(v - m, 0.1 * m + eps()), 0.5))
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :zi => (pμ+pσ+1):(pμ+pσ+pz)]
    names = [:mu => nmμ, :sigma => nmσ, :zi => nmzi]
    means = Dict(:mu => exp.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]),
                  :zi => _logistic.(Xzi * θ̂[(pμ+pσ+1):(pμ+pσ+pz)]))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# Numerically stable NB2 log-pmf log f(k; r, μ) — used in place of
# `logpdf(NegativeBinomial(r, r/(r+μ)), k)` wherever a zero-truncation term
# divides out P(0). Distributions' NegativeBinomial(r, p) forms p = r/(r+μ) and
# r + μ directly: once the dispersion (`sigma`) linear predictor is extreme
# enough that size r = exp(-2·ησ) ≫ μ (near-Poisson dispersion, r ~ 1e17-1e43 at
# log σ ≈ -20..-50, #866), μ is smaller than r's ULP and r + μ rounds to EXACTLY
# r: p rounds to EXACTLY 1.0, so log(1 - p) = log(0) = -Inf for every k > 0 while
# logpdf(·, 0) still comes out finite (it needs no log(1 - p) term). The
# hurdle's `_log1mexp(logpdf(d, 0))` — dividing out P(0) for the zero-truncated
# positive part — then computes -Inf - (-Inf) = NaN (#866; #846 fixed the
# analogous cancellation on the `(1|g)` AGHQ path). Working entirely in log1p
# space avoids ever forming r + μ: log p = -log1p(μ/r) and log(1-p) =
# log μ - log r - log1p(μ/r), both finite as μ/r → 0 (the r → ∞ limit is
# Poisson, Var → μ).
#
# loggamma(k+r) - loggamma(r) — O(1) per call (#883, review of #871/#874/#876):
# the original Σ_{j=0}^{k-1} log(r+j) is exact but O(k), which made every
# hurdle/TNB2 fit and quantile residual on ordinary large-count data (k in the
# thousands) 55×-2500× slower (measured). Three regimes, chosen so every one
# stays within 1e-10 relative error of a 512-bit BigFloat reference over
# k ∈ {0,…,1e6}, r ∈ {1e-3,…,1e40}:
#   - k ≤ 32: keep the exact O(k) sum (bounded cost, and safest at small k).
#   - r ≫ k (r > 1e6·(k+1)): direct loggamma(k+r) - loggamma(r) cancels
#     catastrophically (both are ~r·log(r), and their O(k·log r) difference is
#     swamped by float64's ~eps·r·log(r) rounding floor once r/k is large).
#     Use the asymptotic expansion in x = j/r instead: log(r+j) = log(r) +
#     log1p(j/r), so Σ log(r+j) = k·log(r) + Σ log1p(j/r), and for j ≤ k-1 ≪ r
#     the Taylor series of log1p(j/r) in powers of 1/r can be summed in closed
#     form via the power sums Σj, Σj², Σj³, Σj⁴ (O(1), no loop over j).
#   - otherwise: loggamma(k) - logbeta(r, k) (= log Γ(k+r)/Γ(r); see the
#     branch for why not the direct loggamma difference, #998).
function _nb2_loggammadiff(r, k::Integer)
    k == 0 && return 0.0
    if k <= 32
        s = 0.0
        for j in 0:(k-1)
            s += log(r + j)
        end
        return s
    elseif r > 1e6 * (k + 1)
        n = k - 1                      # j runs 0:n
        S1 = n * (n + 1) / 2
        S2 = n * (n + 1) * (2n + 1) / 6
        S3 = (n * (n + 1) / 2)^2
        S4 = n * (n + 1) * (2n + 1) * (3.0n^2 + 3.0n - 1) / 30
        return k * log(r) + S1 / r - S2 / (2r^2) + S3 / (3r^3) - S4 / (4r^4)
    else
        # Γ(k+r)/Γ(r) = Γ(k)/B(r, k): `logbeta` (Rmath's lbeta algorithm, with
        # the Stirling corrections for large arguments) does not form the two
        # O(r log r) loggammas whose difference this is. The direct
        # loggamma(k + r) - loggamma(r) left an absolute error of ~eps·r·log r,
        # ~4e-7 at r = 1e8, k = 160 (#998).
        return loggamma(k) - logbeta(r, k)
    end
end

function _nb2_logpmf(r, μ, k::Integer)
    l1p = log1p(μ / r)                          # log(1+μ/r), for -r·log(1+μ/r)
    s = -r * l1p - loggamma(k + 1) + _nb2_loggammadiff(r, k)
    if k != 0
        # k·(log μ - log r - l1p): computing this directly cancels
        # catastrophically once μ ≫ r (l1p ≈ log(μ/r) to many digits there,
        # #883's 1e-10 accuracy bar at k ~ 1e5-1e6 needs the stable form). The
        # stable rewrite is -k·log1p(r/μ) — but only take it when μ ≥ r, so
        # r/μ ≤ 1 and can never overflow; on the other side (r ≫ μ, e.g. μ
        # underflowing towards 0 at extreme dispersion) r/μ can itself
        # overflow to Inf while the direct subtraction is perfectly
        # well-conditioned there (no cancellation: the true value is large in
        # magnitude, matching the operands), so keep the direct form.
        s += μ >= r ? -k * log1p(r / μ) : k * (log(μ) - log(r) - l1p)
    end
    return s
end

# Hurdle NB2: P(0) = π, P(k>0) = (1-π)·NB(k)/(1-NB(0)) [zero-truncated], with
# π = logistic(Xhuᵀβ) the hurdle (zero) probability. Uses `_log1mexp` (poisson.jl).
function _fit_negbin2_hu(fam::NegBinomial2, y, Xμ, Xσ, Xhu, nmμ, nmσ, nmhu, g_tol)
    n = length(y); pμ, pσ, ph = size(Xμ, 2), size(Xσ, 2), size(Xhu, 2)
    yint = round.(Int, y); iszero_y = y .== 0
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; βh = θ[pμ+pσ+1:pμ+pσ+ph]
        ημ = clamp.(Xμ * βμ, -20.0, 20.0); ησ = clamp.(Xσ * βσ, -20.0, 20.0)
        ηh = clamp.(Xhu * βh, -30.0, 30.0)
        s = zero(eltype(θ))
        @inbounds for i in 1:n
            lπ = _log_logistic(ηh[i]); l1mπ = _log1m_logistic(ηh[i])
            if iszero_y[i]
                s -= lπ
            else
                μ = exp(ημ[i]); r = exp(-2 * ησ[i])
                s -= l1mπ + _nb2_logpmf(r, μ, yint[i]) - _log1mexp(_nb2_logpmf(r, μ, 0))
            end
        end
        return s
    end
    pos = y[y.>0]; m = isempty(pos) ? sum(y) / n : sum(pos) / length(pos)
    v = sum(abs2, y .- sum(y) / n) / max(n - 1, 1)
    θ0 = zeros(pμ + pσ + ph)
    θ0[1] = log(m + eps())
    θ0[pμ+1] = -0.5 * log(max(m^2 / max(v - m, 0.1 * m + eps()), 0.5))
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :hu => (pμ+pσ+1):(pμ+pσ+ph)]
    names = [:mu => nmμ, :sigma => nmσ, :hu => nmhu]
    means = Dict(:mu => exp.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]),
                  :hu => _logistic.(Xhu * θ̂[(pμ+pσ+1):(pμ+pσ+ph)]))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

function _fit_negbin2(fam::NegBinomial2, y, Xμ, Xσ, nmμ, nmσ, g_tol)
    n = length(y); pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    yint = round.(Int, y)
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]
        # Soft guards (see `_softclamp` above): identity across the original hard
        # band so no legitimate fit is distorted, smooth beyond so the gradient
        # stays live — restoring agreement with drmTMB's smooth guard.
        ημ = _softclamp.(Xμ * βμ, -20.0, 20.0, 3.0)  # keep p ∈ (0,1) on a runaway
        ησ = _softclamp.(Xσ * βσ, -20.0, 20.0, 3.0)  # NegativeBinomial rejects p ≤ 0 / size ≤ 0
        s = zero(eltype(θ))
        @inbounds for i in 1:n
            μ = exp(ημ[i]); r = exp(-2 * ησ[i])           # r = θ (size)
            # #982: `_nb2_logpmf`, not `logpdf(NegativeBinomial(r, r/(r+μ)))`.
            # At near-Poisson size (log σ ≈ -15, r ≈ 1e13) p = r/(r+μ) keeps
            # only ~3 significant digits of 1 − p, the log-likelihood picks up
            # O(0.01-0.2) rounding noise, and the optimiser found logLik values
            # ABOVE the Poisson maximum that NB2 can only approach from below.
            s -= _nb2_logpmf(r, μ, yint[i])
        end
        return s
    end
    m = sum(y) / n; v = sum(abs2, y .- m) / max(n - 1, 1)
    θ0 = zeros(pμ + pσ)
    θ0[1] = log(m + eps())                                  # log-mean intercept
    θ0[pμ+1] = -0.5 * log(max(m^2 / max(v - m, 0.1 * m + eps()), 0.5))  # MoM dispersion init
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ)]
    names = [:mu => nmμ, :sigma => nmσ]
    means = Dict(:mu => exp.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => Vector{Float64}(y))  # response-scale μ̂
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

"""
    TruncatedNegBinomial2()

Zero-truncated negative-binomial (NB2) family for strictly-positive counts (≥ 1)
— litter sizes, group sizes given presence. Same parameterisation as
[`NegBinomial2`](@ref) (log-link mean `μ`, `log σ` in the `sigma` slot, size
`θ = 1/σ²`)
but conditioned on `y ≥ 1`: `P(k) = NB(k) / (1 − NB(0))`. Mirrors `drmTMB`'s
`truncated_nbinom2`.

```julia
fit = drm(bf(y ~ x, sigma ~ 1), TruncatedNegBinomial2(); data = dat)
```

Adding an `hu` part fits the HURDLE NB2 instead: `P(0) = logistic(X_huᵀβ)` and the
positive counts follow this zero-truncated NB2. Zeros are then allowed in the
response. This is `drmTMB`'s spelling of the hurdle model — it has no
`hurdle_nbinom2()` constructor — and it fits the same likelihood as
[`NegBinomial2`](@ref) with `hu`, which is what the call delegates to (so the
returned fit reports `NegBinomial2` as its family).

```julia
fit = drm(bf(y ~ x, sigma ~ 1, hu ~ w), TruncatedNegBinomial2(); data = dat)
```
"""
struct TruncatedNegBinomial2 end

function drm(f::DrmFormula, fam::TruncatedNegBinomial2; data,
             g_tol::Real = 1e-8, se::Bool = true,
             nAGQ::Union{Nothing,Int} = nothing)
    if !isempty(_sd_parts(f)) || !isempty(_sdphylo_parts(f))
        nodes = nAGQ === nothing ? _RANEF1D_AGHQ_K : nAGQ
        nodes >= 1 || throw(ArgumentError("nAGQ must be ≥ 1; got $nodes"))
        return _drm_truncated_negbin2_sd_regression(f, fam; data, g_tol, se,
                                                    nAGQ = nodes)
    end
    missing_fit = _fit_observed_response_rows(f, data) do data_observed
        drm(f, fam; data = data_observed, g_tol = g_tol)
    end
    missing_fit !== nothing && return missing_fit

    nAGQ === nothing || throw(ArgumentError(
        "nAGQ is currently supported for TruncatedNegBinomial2 random-effect SD regression only."))
    _lss_only_gaussian_guard(f, fam)   # #544: refuse, never silently drop, sd() parts
    rhs = Dict(f.forms)
    for (_, r) in f.forms
        _, re, mv, st = _split_ranef(r)
        (isempty(re) && mv === nothing && st === nothing) ||
            error("TruncatedNegBinomial2() currently supports fixed effects only")
    end
    # Refuse, never silently drop, a formula part this family does not consume.
    # Before this guard `drm(bf(y ~ x, sigma ~ 1, zi ~ w), TruncatedNegBinomial2())`
    # dropped `zi` from the likelihood without a word and fitted the plain
    # zero-truncated model — a different model than the caller asked for. Only the
    # `hu` hurdle part is consumed (below); everything else is an error here.
    for (pname, _) in f.forms
        pname in (:mu, :sigma, :hu) ||
            error("TruncatedNegBinomial2() supports `mu`, `sigma`, and an optional `hu` " *
                  "hurdle part; got an unsupported formula part `$pname`. A `zi` " *
                  "zero-inflation part belongs to NegBinomial2(), whose count " *
                  "component is untruncated -- a different model, not a spelling.")
    end
    # Hurdle NB2, spelled drmTMB's way. drmTMB has no `hurdle_nbinom2()`
    # constructor: `truncated_nbinom2()` plus an `hu ~ ...` part IS its hurdle
    # model (R/family.R, R/drmTMB.R `model_type = if (has_hu) "hurdle_nbinom2"`),
    # and the truncated NB2 is exactly the positive component of that hurdle. The
    # likelihood is the one `NegBinomial2()` + `hu` already fits — P(0) = logistic(
    # X_huᵀβ), positive counts zero-truncated NB2 — so delegate to it rather than
    # duplicate a second copy of the same nll. The returned fit therefore reports
    # `NegBinomial2` as its family; the blocks, coefficients, and log-likelihood
    # are identical either way.
    haskey(rhs, :hu) && return drm(f, NegBinomial2(); data = data, g_tol = g_tol, se = se)
    y, Xμ, nmμ = _design(f.response, rhs[:mu], data)
    _, Xσ, nmσ = _design(f.response, get(rhs, :sigma, ConstantTerm(1)), data)
    all(yi -> yi ≥ 1 && isinteger(yi), y) ||
        error("TruncatedNegBinomial2() requires positive integer counts (≥ 1) as the response")
    return _withformula(_fit_truncated_negbin2(fam, y, Xμ, Xσ, nmμ, nmσ, g_tol;
                                               se = se), f)
end

# Predictor-dependent latent-coefficient SD for zero-truncated NB2. The group
# integral uses the same adaptive one-dimensional GHQ rule as NB2, with the
# conditional log likelihood adjusted by the stable zero-truncation normalizer.
function _drm_truncated_negbin2_sd_regression(f::DrmFormula,
                                               fam::TruncatedNegBinomial2;
                                               data, g_tol::Real = 1e-8,
                                               se::Bool = true,
                                               nAGQ::Int = _RANEF1D_AGHQ_K)
    targets = _sd_target_parts(f)
    length(targets) == 1 || throw(ArgumentError(
        "TruncatedNegBinomial2 random-effect SD regression requires exactly one `sd(group) ~ …` formula."))
    target = only(targets)
    descriptor, grp, sd_rhs = target.target, target.target.group, target.rhs
    (descriptor.level === :iid && descriptor.dpar === :mu &&
     descriptor.block === nothing) || throw(ArgumentError(
        "TruncatedNegBinomial2 SD regression targets one ordinary mean intercept or independent numeric slope."))
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    mv === nothing && st === nothing && length(re) == 1 || throw(ArgumentError(
        "TruncatedNegBinomial2 `sd($grp)` currently requires one ordinary independent random-effect coefficient."))
    kind, slope = _re_kind(re[1][1])
    member = kind === :intercept ? :intercept : kind === :slope ? slope : nothing
    kind in (:intercept, :slope) || throw(ArgumentError(
        "TruncatedNegBinomial2 `sd($grp)` supports `(1 | $grp)` or one independent numeric `(0 + x | $grp)` slope."))
    re[1][2] === grp || throw(ArgumentError(
        "TruncatedNegBinomial2 `sd($grp)` must target the grouping variable `$(re[1][2])`."))
    descriptor.member in (nothing, member) || throw(ArgumentError(
        "TruncatedNegBinomial2 SD target `coef($(descriptor.member))` does not match random-effect coefficient `$member`."))
    sigma_rhs = get(rhs, :sigma, ConstantTerm(1))
    sigma_rhs isa ConstantTerm && sigma_rhs == ConstantTerm(1) || throw(ArgumentError(
        "TruncatedNegBinomial2 `sd($grp)` currently requires a fixed `sigma ~ 1` submodel."))
    # `truncated_nbinom2() + hu` is drmTMB's Hurdle NB2 spelling. The
    # NegBinomial2 route already implements this exact Hurdle likelihood with
    # an ordinary mean random-effect SD model, so share that fitter instead of
    # rejecting the wrapper form.
    haskey(rhs, :hu) && return drm(f, NegBinomial2(); data = data,
                                  g_tol = g_tol, se = se, nAGQ = nAGQ)
    for (pname, _) in f.forms
        (pname in (:mu, :sigma, target.key) || haskey(f.sd_targets, pname)) && continue
        throw(ArgumentError("TruncatedNegBinomial2 `sd($grp)` supports only `mu`, fixed `sigma ~ 1`, and one `sd($grp)` formula."))
    end
    fixed_mu, offset_term = _extract_offset(fixed_mu)
    offset_term === nothing || throw(ArgumentError(
        "TruncatedNegBinomial2 random-effect SD regression does not yet support `offset(...)`."))
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "TruncatedNegBinomial2 random-effect SD regression does not yet support missing responses."))
    y, Xmu, nmmu = _design(f.response, fixed_mu, data)
    _, Xsigma, nmsigma = _design(f.response, sigma_rhs, data)
    all(yi -> yi >= 1 && isinteger(yi), y) || throw(ArgumentError(
        "TruncatedNegBinomial2() requires positive integer counts (≥ 1) as the response"))
    gidx, G = _group_index(getproperty(data, grp))
    Zg, nmsd = _sd_group_design(f.response, sd_rhs, data, gidx, G, grp;
                                schema_cache = f.schema_cache, schema_key = :sd)
    size(Zg, 2) <= G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
        "TruncatedNegBinomial2 `sd($grp)` group-level design is rank deficient."))
    w = if kind === :intercept
        ones(length(y))
    else
        hasproperty(data, slope) || throw(ArgumentError(
            "TruncatedNegBinomial2 random-slope predictor `$slope` is absent from data."))
        raw = getproperty(data, slope)
        raw isa AbstractVector{<:Real} && length(raw) == length(y) &&
            all(isfinite, raw) || throw(ArgumentError(
                "TruncatedNegBinomial2 random-slope predictor `$slope` must be a finite numeric vector."))
        Float64.(raw)
    end
    return _withformula(_fit_truncated_negbin2_sd_regression(fam, y, Xmu,
        Xsigma, Zg, gidx, G, nmmu, nmsigma, nmsd, grp, g_tol; nAGQ, se, w), f)
end

function _fit_truncated_negbin2_sd_regression(fam::TruncatedNegBinomial2,
        y, Xmu, Xsigma, Zg, gidx, G, nmmu, nmsigma, nmsd, grp, g_tol;
        nAGQ::Int = _RANEF1D_AGHQ_K, se::Bool = true,
        w::AbstractVector{<:Real} = ones(length(y)))
    n = length(y); pmu, psigma, psd = size(Xmu, 2), size(Xsigma, 2), size(Zg, 2)
    length(w) == n || throw(DimensionMismatch(
        "TruncatedNegBinomial2 random-effect loading must have one value per observation."))
    yint = round.(Int, y)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    rule = _AGHQRule(1, nAGQ)
    Zre = reshape(Float64.(w), n, 1)
    bcache = [zeros(1, G) for _ in 1:Threads.nthreads()]
    function nll(theta)
        beta_mu = @view theta[1:pmu]
        beta_sigma = @view theta[(pmu+1):(pmu+psigma)]
        alpha = @view theta[(pmu+psigma+1):(pmu+psigma+psd)]
        eta_mu = Xmu * beta_mu
        eta_sigma = _softclamp.(Xsigma * beta_sigma, -20.0, 20.0, 3.0)
        sdgroup = exp.(Zg * alpha)
        cache = bcache[Threads.threadid()]
        ll = (i, eta) -> begin
            mu = exp(clamp(eta, -20.0, 20.0))
            size = exp(-2 * eta_sigma[i])
            _nb2_logpmf(size, mu, yint[i]) -
                _log1mexp(_nb2_logpmf(size, mu, 0))
        end
        total = zero(eltype(theta))
        for gi in 1:G
            Lg = reshape([sdgroup[gi]], 1, 1)
            value, bhat = _aghq_group_logint(ll, members[gi], eta_mu,
                Zre, Lg, rule, @view(cache[:, gi]))
            cache[1, gi] = isfinite(bhat[1]) ? bhat[1] : 0.0
            total += value
        end
        return -total
    end
    theta0 = zeros(pmu + psigma + psd)
    theta0[1] = log(max(mean(y) * 0.65, 0.1))
    theta0[pmu+1] = -0.5 * log(0.5)
    theta0[(pmu+psigma+1):end] .= Zg \ fill(log(0.4), G)
    res = Optim.optimize(nll, theta0, Optim.LBFGS(),
        Optim.Options(g_tol = g_tol); autodiff = :forward)
    theta_hat = Optim.minimizer(res)
    V = se ? _vcov_from_hessian(ForwardDiff.hessian(nll, theta_hat)) :
             fill(NaN, length(theta_hat), length(theta_hat))
    blocks = [:mu => 1:pmu, :sigma => (pmu+1):(pmu+psigma),
              :sd => (pmu+psigma+1):(pmu+psigma+psd)]
    names = [:mu => nmmu, :sigma => nmsigma, :sd => nmsd]
    sdgroup = exp.(Zg * theta_hat[(pmu+psigma+1):end])
    means = Dict(:mu => exp.(Xmu * theta_hat[1:pmu]))
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xsigma * theta_hat[(pmu+1):(pmu+psigma)]),
                  :sd_group => sdgroup, :sd => sdgroup[gidx],
                  :sd_group_index => Float64.(gidx), :aghq_nodes => [Float64(nAGQ)])
    return _withiterations(_withnll(DrmFit(fam, blocks, names, theta_hat,
        V, -nll(theta_hat), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

function _fit_truncated_negbin2(fam::TruncatedNegBinomial2, y, Xμ, Xσ, nmμ, nmσ, g_tol;
                                se::Bool = true)
    n = length(y); pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    yint = round.(Int, y)
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]
        ημ = clamp.(Xμ * βμ, -20.0, 20.0); ησ = clamp.(Xσ * βσ, -20.0, 20.0)
        s = zero(eltype(θ))
        @inbounds for i in 1:n
            μ = exp(ημ[i]); r = exp(-2 * ησ[i])
            # `_nb2_logpmf` (not `logpdf(_nb2(r,p), ·)`): at extreme dispersion
            # (log σ ≈ -20..-50, r ~ 1e17-1e43, #866) r + μ rounds to EXACTLY r
            # and p rounds to EXACTLY 1.0, so `logpdf(·, 0)` is finite while
            # `logpdf(·, k>0)` is -Inf; the zero-truncation divisor
            # `_log1mexp(logpdf(d, 0))` then computed -Inf - (-Inf) = NaN. Same
            # cancellation `_fit_negbin2_hu` fixed (#866; #871).
            s -= _nb2_logpmf(r, μ, yint[i]) - _log1mexp(_nb2_logpmf(r, μ, 0))   # divide out P(0): zero-truncated
        end
        return s
    end
    m = sum(y) / n; v = sum(abs2, y .- m) / max(n - 1, 1)
    θ0 = zeros(pμ + pσ)
    θ0[1] = log(m + eps())
    θ0[pμ+1] = -0.5 * log(max(m^2 / max(v - m, 0.1 * m + eps()), 0.5))
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res)
    V = se ? _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂)) :
             fill(NaN, length(θ̂), length(θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ)]
    names = [:mu => nmμ, :sigma => nmσ]
    means = Dict(:mu => exp.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => Vector{Float64}(y))  # untruncated NB mean μ̂
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end
