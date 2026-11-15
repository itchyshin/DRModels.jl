# beta.jl — Beta family for responses on the open interval (0,1) (proportions,
# rates, probabilities). Logit link on the mean μ; the `sigma` slot carries σ
# with drmTMB's precision mapping φ = 1/σ² (so coef(:sigma) is log σ). The
# likelihood is Beta(μφ, (1-μ)φ): mean μ, variance μ(1-μ)/(1+φ). Fixed effects,
# ML. `Distributions.Beta` is used qualified — DRModels exports its own `Beta` family.

import Distributions

"""
    Beta()

Beta response family for proportions in `(0,1)`: logit link on the mean `μ`, and
the `sigma` slot carries `σ` with the precision mapping `φ = 1/σ²` (so
`coef(fit, :sigma)` is `log σ`; recover precision as `exp(-2·log σ)`). Likelihood
`Beta(μφ, (1-μ)φ)`. Mirrors `drmTMB`'s `beta_family`.
An ordinary mean random intercept or one independent numeric mean slope also
supports a predictor-dependent SD through `sd(group) ~ predictors`. A slope
targets its coefficient SD with `sd(group, coef(x)) ~ predictor`; the slope
loading is applied at each row. A supplied-A animal random intercept also
supports `sd(group, animal) ~ predictors` with `sigma ~ 1`; the animal covariance
is used as supplied (raw), as native drmTMB does. These SD routes
require complete responses and fixed `sigma ~ 1`. `sd(group, phylogenetic|relmat|spatial) ~ predictors`
is also supported on the matching structured intercept, and an independent
`animal|relmat|phylo|spatial(1 + x | group)` block takes one `sd(group, level, coef(·))`
formula per coefficient. Correlated (labelled) SD blocks and calibrated interval coverage
remain unsupported.
Crossed random intercepts on the mean, such as `(1 | g) + (1 | h)`, use the
sparse-Laplace engine when `sigma ~ 1`. A phylogenetic random intercept on the
mean, `phylo(1 | species)`, also uses the sparse-Laplace engine, as do general
user-supplied PD-covariance intercepts (used as supplied, not rescaled; the precomputed
`spatial` kernel keeps its rescaling) — `relmat(1 | id)` with `K = C`, the
`animal(1 | id)` (`A = C`) and `spatial(1 | id)` (`K = C`) aliases.

!!! note
    `DRModels.Beta` shadows `Distributions.Beta`; qualify the latter if you need it.

```julia
fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Beta(); data = dat)
fit = drm(bf(@formula(y ~ x + (1 | g) + (1 | h)), @formula(sigma ~ 1)), Beta(); data = dat)
fit_phy = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
              Beta(); data = dat, tree = tr, se = false)
fit_animal_sd = drm(bf(@formula(y ~ x + animal(1 | id)),
                       @formula(sigma ~ 1),
                       @formula(sd(id, animal) ~ z)),
                    Beta(); data = dat, A = A)
exp(-2 * coef(fit, :sigma)[1])     # estimated precision φ

# Experimental: Beta random-intercept variational (ELBO) marginal.
# Requires `sigma ~ 1`. Default remains `marginal = :LA` (adaptive GHQ, 5 nodes per group, on an ordinary `(1 | g)`).
fit_va = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Beta();
             data = dat, marginal = :VA)

# TMB-convention Laplace on an ordinary `(1 | g)`, as native drmTMB fits it.
# Requires `sigma ~ 1`; ML only.
fit_lap = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Beta();
              data = dat, marginal = :Laplace)
```
"""
struct Beta end

_logistic(η) = 1 / (1 + exp(-η))

# FX3b: `se = false` is recorded on the fit (`DrmFit.uncertainty`), so
# `convergence_status` does not read the skipped (NaN) covariance as a failure.
@drm_entry function drm(f::DrmFormula, fam::Beta; se::Bool = true, kwargs...)
    return _withuncertainty(_drm_beta_fit(f, fam; se = se, kwargs...), se)
end

function _drm_beta_fit(f::DrmFormula, fam::Beta; data, tree = nothing, K = nothing,
             A = nothing, coords = nothing, g_tol::Real = 1e-8, se::Bool = true,
             marginal::Symbol = :LA, method = nothing, ids = nothing)
    f = _default_structured_q2_sd(f, fam; method)   # bare structured (1 + x | g): default `sd(…, coef(·)) ~ 1`
    f = _default_structured_q1_sd(f, fam; method, K, coords)   # bare structured (1 | g): default `sd(g, kind) ~ 1`
    _drm_entry_guard(f, fam, data; tree = tree, K = K, A = A, coords = coords)   # #1027 #1022 #1010
    _reject_method_as_marginal(fam, method)
    if !isempty(_sd_parts(f)) || !isempty(_sdphylo_parts(f))
        A = _resolve_relatedness_labels(f, data, A, ids; coords = coords)   # #957: match `A` by label (SD routes too)
        coords = _resolve_coords_labels(f, data, coords, ids)   # #957
        return _drm_beta_sd_regression(f, fam; data, tree, K, A, coords,
                                       g_tol, se, marginal)
    end
    _scalar_laplace_requested(marginal) &&     # Arc 2: TMB-convention Laplace, ordinary (1 | g)
        return _drm_ordinary_laplace(f, fam; data = data, tree = tree, K = K, A = A,
                                     coords = coords, g_tol = g_tol, se = se, method = method)
    missing_fit = _fit_observed_response_rows(f, data) do data_observed
        drm(f, fam; data = data_observed, tree = tree, K = K, A = A,
            coords = coords, g_tol = g_tol, se = se, marginal = marginal, method = method, ids = ids)
    end
    missing_fit !== nothing && return missing_fit
    A = _resolve_relatedness_labels(f, data, A, ids; coords = coords)   # #957: match `A` by label
    coords = _resolve_coords_labels(f, data, coords, ids)   # #957: match `coords` by site label

    marg = _marginal_method(marginal)                     # :LA (default), :VA (#136), or :AGHQ (#761)
    isaghq = marg isa AGHQ
    isva = marg isa Variational
    _lss_only_gaussian_guard(f, fam)   # #544: refuse, never silently drop, sd() parts
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    fixed_mu, offset_term = _extract_offset(fixed_mu)   # #938: offset(z), coefficient fixed at 1
    offset_term === nothing || (isempty(re) && mv === nothing && st === nothing) ||
        _refuse_offset_route(fam, "a random, structured or `meta_V` term on the mean")
    isaghq && !(length(re) > 1 && st === nothing) &&
        _aghq_reject(fam, "this model (Beta `marginal = :AGHQ` covers crossed random " *
                          "intercepts `(1 | g) + (1 | h)` only, #761)")
    mv === nothing ||
        error("Beta() does not support meta_V markers")
    for (pname, r) in f.forms          # only the mean may carry a random effect
        pname === :mu && continue
        _, re2, mv2, st2 = _split_ranef(r)
        (isempty(re2) && mv2 === nothing && st2 === nothing) ||
            error("Beta(): only the mean formula may carry a random effect")
    end
    y, Xμ, nmμ = _design(f.response, fixed_mu, data)
    _, Xσ, nmσ = _design(f.response, get(rhs, :sigma, ConstantTerm(1)), data)
    all(yi -> 0 < yi < 1, y) ||
        error("Beta() requires responses strictly in the open interval (0, 1)")
    if st !== nothing
        isva && _va_reject(fam, "a phylogenetic/structured random effect")
        isempty(re) ||
            error("Beta() phylo structured effects cannot be combined with ordinary random effects yet")
        # `sigma ~ 1` keeps the scalar-dispersion spine; a covariate `sigma`
        # formula routes to the per-observation log-dispersion path (#164).
        kind, grp = st
        labels = getproperty(data, grp)
        if kind === :phylo
            tree === nothing && error("phylo(1 | $grp) needs `tree = ...`")
            return _withformula(_fit_beta_phylo_laplace(fam, y, Xμ, Xσ, labels, tree, nmμ, nmσ, grp, g_tol; se = se), f)
        elseif kind === :relmat || kind === :animal || kind === :spatial
            # General user-supplied PD covariance C on the mean (logit) intercept
            # (relatedness / animal model / precomputed spatial), with the Beta
            # precision a fixed nuisance. Reuses the phylo nuisance Laplace spine
            # with the tree precision swapped for C⁻¹ (#167).
            C = _poisson_structured_cov(kind, grp, K, A, coords)
            return _withformula(_fit_beta_relmat_laplace(fam, y, Xμ, Xσ, C, labels, nmμ, nmσ, grp, g_tol; se = se,
                normalize = kind === :spatial), f)
        else
            error("Beta() supports phylo/relmat/animal/spatial(1 | group) among structured markers")
        end
    end
    if !isempty(re)                    # random effect on the logit mean → GHQ/Laplace or VA (#136)
        if length(re) > 1
            isva && _va_reject(fam, "crossed/multiple random intercepts")
            if _has_indep_slopes(re)
                isaghq && _aghq_reject(fam, "independent random slopes `(0 + x | g)` (Laplace only)")
                return _withformula(_fit_indep_re_laplace(fam, Val(:beta), y, Xμ, Xσ, nothing,
                                                          _indep_slope_comps(re, data, length(y)),
                                                          nmμ, nmσ, String[], g_tol), f)
            end
            all(_re_kind(r[1])[1] === :intercept for r in re) ||
                error("Beta() supports multiple random effects only as crossed/nested intercepts " *
                      "`(1 | g) + (1 | h)` or independent intercepts and slopes `(1 | g) + (0 + x | g)`; " *
                      "a correlated `(1 + x | g)` must be the only random-effect term")
            comps = map(re) do r
                grp = r[2]; gidx, G = _group_index(getproperty(data, grp); col = grp)
                (ones(length(y)), gidx, G, String(grp))
            end
            isaghq && return _withformula(_fit_beta_crossed_aghq(fam, y, Xμ, Xσ, comps, nmμ, nmσ, g_tol; se = se), f)   # #761
            return _withformula(_fit_beta_crossed_laplace(fam, y, Xμ, Xσ, comps, nmμ, nmσ, g_tol; se = se), f)
        end
        (rk, var) = _re_kind(re[1][1]); grp = re[1][2]
        gidx, G = _group_index(getproperty(data, grp); col = grp)
        if rk === :intercept            # (1 | g) → 1-D Gauss–Hermite (Laplace) or VA (#136)
            if isva
                size(Xσ, 2) == 1 || _va_reject(fam, "a non-constant `sigma` formula (VA kernels require `sigma ~ 1`)")
                return _withformula(_withmarginal(
                    _fit_beta_ranef_va(fam, y, Xμ, Xσ, gidx, G, nmμ, nmσ, grp, g_tol), :VA), f)
            end
            return _withformula(_fit_beta_ranef(fam, y, Xμ, Xσ, gidx, G, nmμ, nmσ, grp, g_tol), f)
        elseif rk === :corr             # (1 + x | g) → 2-D Gauss–Hermite marginal
            isva && _va_reject(fam, "a correlated random slope `(1 + x | g)`")
            _require_finite_data_column(data, var)
            return _withformula(_fit_beta_corr_ranef(fam, y, Xμ, Xσ, Float64.(getproperty(data, var)), gidx, G, nmμ, nmσ, grp, g_tol), f)
        elseif rk === :slope            # (0 + x | g) → TMB-convention Laplace (independent route)
            isva && _va_reject(fam, "an independent random slope `(0 + x | g)`")
            isaghq && _aghq_reject(fam, "an independent random slope `(0 + x | g)` (Laplace only)")
            _require_finite_data_column(data, var)
            return _withformula(_fit_indep_re_laplace(fam, Val(:beta), y, Xμ, Xσ, nothing,
                                                      _indep_slope_comps(re, data, length(y)),
                                                      nmμ, nmσ, String[], g_tol), f)
        else
            error("Beta() supports `(1 | g)`, `(0 + x | g)` or `(1 + x | g)` on the mean")
        end
    end
    isva && _va_reject(fam, "no random intercept (fixed-effects-only)")
    return _withformula(_fit_beta(fam, y, Xμ, Xσ, nmμ, nmσ, g_tol; offset = _offset_for_fit(fam, offset_term, data, size(Xμ, 1))), f)
end

function _drm_beta_sd_regression(f::DrmFormula, fam::Beta; data,
                                tree = nothing, K = nothing, A = nothing,
                                coords = nothing, g_tol::Real = 1e-8,
                                se::Bool = true, marginal::Symbol = :LA,
                                nAGQ::Int = _SD_IID_AGHQ_K)
    laplace_fit = _laplace_sd_route(f, fam, Val(:beta), marginal; data, g_tol)
    laplace_fit === nothing || return laplace_fit
    marginal === :LA || throw(ArgumentError(
        "Beta `sd(group) ~ …` currently supports `marginal = :LA` only."))
    twofield = _drm_twofield_sd_route(f, fam; data, tree, K, A, coords, g_tol, se)
    twofield === nothing || return twofield   # structured `kind(1 + x | g)` with two coef SD formulas
    structured_targets = [t for t in _gamma_sd_targets(f)
                          if t.target.level in (:relmat, :phylogenetic, :spatial)]
    isempty(structured_targets) || return _drm_beta_structured_sd_regression(
        f, fam, structured_targets; data, tree, K, A, coords, g_tol, se)
    (tree === nothing && K === nothing && coords === nothing) || throw(ArgumentError(
        "Beta ordinary `sd(group) ~ …` and `sd(group, animal) ~ …` take no `tree`, `K`, or `coords` input; " *
        "use `sd(group, phylogenetic|relmat|spatial) ~ …` with the matching structured term."))
    isempty(_sdphylo_parts(f)) || throw(ArgumentError(
        "Beta `sd(group, phylogenetic) ~ …` is not implemented by the ordinary random-intercept route."))
    targets = _sd_target_parts(f)
    length(targets) == 1 || throw(ArgumentError(
        "Beta random-effect SD regression requires exactly one `sd(group) ~ …` formula (ordinary, " *
        "`animal`, or `phylogenetic`/`relmat`/`spatial`), or two coefficient formulas " *
        "`sd(group, level, coef(intercept)) ~ …` and `sd(group, level, coef(x)) ~ …` for an independent " *
        "`animal`/`relmat`/`phylo`/`spatial(1 + x | group)` block; correlated (labelled), three-or-more-coefficient, " *
        "ordinary q2 and crossed SD shapes are not implemented."))
    target = only(targets)
    (target.target.level in (:iid, :animal) && target.target.dpar === :mu &&
     target.target.block === nothing) || throw(ArgumentError(
        "Beta SD regression currently targets one ordinary or animal mean intercept, or an ordinary independent slope."))
    sgrp, srhs = target.target.group, target.rhs
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    structured_markers = _collect_structured(rhs[:mu])
    mv === nothing || throw(ArgumentError("Beta `sd($sgrp)` cannot use meta_V()."))
    is_animal = target.target.level === :animal
    if is_animal
        isempty(re) && length(structured_markers) == 1 &&
            only(structured_markers) == (:animal, sgrp) || throw(ArgumentError(
            "Beta `sd($sgrp, animal)` requires exactly one `animal(1 | $sgrp)` mean term."))
        last(st) === sgrp || throw(ArgumentError(
            "Beta `sd($sgrp, animal)` must target the group in `animal(1 | $sgrp)`."))
        A === nothing && throw(ArgumentError(
            "Beta `animal(1 | $sgrp)` SD regression requires `A = …`."))
        target.target.member in (nothing, :intercept) || throw(ArgumentError(
            "Beta animal SD regression supports only the intercept coefficient."))
    else
        A === nothing || throw(ArgumentError(
            "Beta ordinary `sd($sgrp)` does not use `A`; use `sd($sgrp, animal)` for a structured animal intercept."))
        isempty(structured_markers) || throw(ArgumentError(
            "Beta ordinary `sd($sgrp)` currently requires one ordinary random-effect term."))
        length(re) == 1 || throw(ArgumentError(
            "Beta `sd($sgrp)` currently requires exactly one ordinary random-effect term."))
    end
    kind, slope = is_animal ? (:intercept, nothing) : _re_kind(re[1][1])
    member = kind === :intercept ? :intercept : kind === :slope ? slope : nothing
    kind in (:intercept, :slope) || throw(ArgumentError(
        "Beta `sd($sgrp)` supports `(1 | $sgrp)` or one independent numeric slope `(0 + x | $sgrp)`; correlated blocks are not implemented on this route."))
    target.target.member in (nothing, member) || throw(ArgumentError(
        "Beta SD target `coef($(target.target.member))` does not match the fitted random-effect coefficient `$(member)`."))
    grp = is_animal ? last(st) : re[1][2]
    grp === sgrp || throw(ArgumentError(
        "Beta `sd($sgrp)` must name the random-effect grouping variable `$grp`."))
    sigma_rhs = get(rhs, :sigma, nothing)
    sigma_rhs isa ConstantTerm && sigma_rhs == ConstantTerm(1) || throw(ArgumentError(
        "Beta `sd($grp)` requires a fixed `sigma ~ 1` submodel."))
    for (pname, _) in f.forms
        (pname in (:mu, :sigma, Symbol("sd_", sgrp)) || haskey(f.sd_targets, pname)) && continue
        throw(ArgumentError("Beta `sd($grp)` supports only `mu`, fixed `sigma ~ 1`, and one `sd($grp)` formula."))
    end
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "Beta random-effect SD regression requires observed responses; remove incomplete rows before fitting."))
    y, Xmu, nmmu = _design(f.response, fixed_mu, data)
    _, Xsigma, nmsigma = _design(f.response, sigma_rhs, data)
    all(yi -> 0 < yi < 1, y) || throw(ArgumentError(
        "Beta() requires responses strictly in the open interval (0, 1)"))
    gidx, G = _group_index(getproperty(data, grp))
    Zg, nmsd = _sd_group_design(f.response, srhs, data, gidx, G, grp;
                                schema_cache = f.schema_cache,
                                schema_key = target.key)
    size(Zg, 2) <= G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
        "Beta `sd($grp)` group-level design is rank deficient; use predictors that vary independently across groups."))
    if is_animal
        C = _poisson_structured_cov(:animal, grp, K, A, coords)
        return _withformula(_fit_beta_animal_sd_regression(
            fam, y, Xmu, Xsigma, Zg, gidx, C, nmmu, nmsigma, nmsd, grp,
            g_tol; se), f)
    end
    w = if kind === :intercept
        ones(length(y))
    else
        hasproperty(data, slope) || throw(ArgumentError(
            "Beta random-slope predictor `$slope` is absent from data."))
        raw = getproperty(data, slope)
        raw isa AbstractVector{<:Real} || throw(ArgumentError(
            "Beta random-slope predictor `$slope` must be numeric."))
        length(raw) == length(y) && all(isfinite, raw) || throw(ArgumentError(
            "Beta random-slope predictor `$slope` must contain one finite value per retained response."))
        Float64.(raw)
    end
    return _withformula(_fit_beta_sd_regression(fam, y, Xmu, Xsigma, Zg,
        gidx, G, nmmu, nmsigma, nmsd, grp, g_tol; K = nAGQ, se, w), f)
end

# Beta intercept SD regression on a phylogenetic, related-matrix or coordinate
# spatial covariance: `y ~ … + phylo|relmat|spatial(1 | g)` with
# `sd(g, phylogenetic|relmat|spatial) ~ z`. The animal q1 Laplace objective is reused
# with the kernel swapped (`K` enters RAW, as for animal; phylo/spatial use the
# `_structured_q2_kernel` kernels). Kernel rows are re-ordered to the first-seen
# order of the data groups, which is what `_general_cov_setup` assumes.
function _drm_beta_structured_sd_regression(f::DrmFormula, fam::Beta, targets;
                                            data, tree, K, A, coords,
                                            g_tol::Real, se::Bool)
    length(targets) == 1 && length(_gamma_sd_targets(f)) == 1 || throw(ArgumentError(
        "Beta structured random-effect SD regression requires exactly one `sd(group, level) ~ …` formula."))
    target = only(targets)
    level = target.target.level
    kind = level === :phylogenetic ? :phylo : level
    sgrp, srhs = target.target.group, target.rhs
    who = "Beta `sd($sgrp, $level)` SD regression"
    (target.target.dpar === :mu && target.target.block === nothing &&
     target.target.member in (nothing, :intercept)) || throw(ArgumentError(
        "$who targets the structured mean intercept only."))
    A === nothing || throw(ArgumentError("$who does not use `A`."))
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    mv === nothing || throw(ArgumentError("$who cannot use meta_V()."))
    isempty(re) && _collect_structured(rhs[:mu]) == [(kind, sgrp)] && st == (kind, sgrp) ||
        throw(ArgumentError("$who requires exactly one matching `$kind(1 | $sgrp)` mean term and no other random effects."))
    sigma_rhs = get(rhs, :sigma, nothing)
    sigma_rhs isa ConstantTerm && sigma_rhs == ConstantTerm(1) || throw(ArgumentError(
        "$who requires a fixed `sigma ~ 1` submodel."))
    for (pname, _) in f.forms
        pname in (:mu, :sigma, target.key) && continue
        throw(ArgumentError("$who supports only `mu`, fixed `sigma ~ 1`, and one SD formula."))
    end
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "$who requires observed responses; remove incomplete rows before fitting."))
    fixed_mu, offset_term = _extract_offset(fixed_mu)
    offset_term === nothing || throw(ArgumentError("$who does not yet support `offset(...)`."))
    y, Xmu, nmmu = _design(f.response, fixed_mu, data)
    _, Xsigma, nmsigma = _design(f.response, sigma_rhs, data)
    all(yi -> 0 < yi < 1, y) || throw(ArgumentError(
        "Beta() requires responses strictly in the open interval (0, 1)"))
    labels = getproperty(data, sgrp)
    C, gidx, G = if kind === :relmat
        (tree === nothing && coords === nothing) || throw(ArgumentError(
            "$who uses `K = …` only; do not also supply `tree` or `coords`."))
        Kc = _poisson_structured_cov(:relmat, sgrp, K, nothing, nothing)
        gi, Gk = _group_index(labels)
        size(Kc) == (Gk, Gk) || throw(ArgumentError(
            "$who: `K` must be $(Gk)×$(Gk), one row per `$sgrp` level in order of first appearance."))
        Kc, gi, Gk
    else
        _structured_q2_kernel(kind, sgrp, labels; tree, K, A = nothing, coords, who)
    end
    Cchol = cholesky(Symmetric(Matrix{Float64}(C)); check = false)
    issuccess(Cchol) || throw(ArgumentError("$who: the $kind kernel is not positive definite."))
    Zg, nmsd = _sd_group_design(f.response, srhs, data, gidx, G, sgrp;
                                schema_cache = f.schema_cache, schema_key = target.key)
    size(Zg, 2) <= G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
        "$who: the SD group-level design is rank deficient; use predictors that vary independently across groups."))
    order = unique(gidx)
    length(order) == G || throw(ArgumentError("$who needs at least one row for every group level."))
    position = Dict(g => k for (k, g) in enumerate(order))
    return _withformula(_fit_beta_animal_sd_regression(fam, y, Xmu, Xsigma,
        Matrix{Float64}(Zg)[order, :], [position[g] for g in gidx],
        Matrix{Float64}(C)[order, order], nmmu, nmsigma, nmsd, sgrp, g_tol; se), f)
end

# LBFGS can stop on its relative-objective test while the mode gradient is still
# about 1e-6, which then fails the mode check below and turns one finite-difference
# stencil point of the outer optimizer into the 1e18 sentinel; the outer gradient
# is then ~1e21 and the outer search stops after a step or two while reporting
# convergence. The Hessian of the joint objective in u is exactly
# Q + diag(sum loading^2 * d2), so a few damped Newton steps are cheap. At this
# gradient size the objective decrease (~g^2/lambda) is at the floating-point
# resolution of the summed objective, so a step is accepted when the objective
# does not increase beyond that resolution AND the gradient norm falls. The
# mode tolerance used by the caller is unchanged.
function _beta_animal_sd_newton_polish(joint, mode, y, eta0, loading, gidx, Q,
                                       aux; tol::Real = 1e-8,
                                       maxiter::Int = 25)
    q = length(mode)
    u = copy(mode)
    fu = joint(u)
    isfinite(fu) || return mode
    grad = ForwardDiff.gradient(joint, u)
    for _ in 1:maxiter
        gnorm = norm(grad)
        gnorm <= tol && break
        diag_h = zeros(Float64, q)
        @inbounds for i in eachindex(y)
            gi = gidx[i]
            eta = eta0[i] + loading[i] * u[gi]
            abs(eta) < 30.0 || return u
            diag_h[gi] += loading[i]^2 * _laplace_d2(Val(:beta_fixed), aux, i, eta)
        end
        ch = cholesky(Symmetric(Q + spdiagm(0 => diag_h)); check = false)
        issuccess(ch) || break
        step = ch \ grad
        all(isfinite, step) || break
        accepted = false
        t = 1.0
        for _ in 1:30
            cand = u .- t .* step
            fc = joint(cand)
            if isfinite(fc) && fc <= fu + 1e-12 * (1 + abs(fu))
                gc = ForwardDiff.gradient(joint, cand)
                if norm(gc) < gnorm
                    u, fu, grad, accepted = cand, fc, gc, true
                    break
                end
            end
            t /= 2
        end
        accepted || break
    end
    return u
end

# Dense structured Beta animal-intercept Laplace objective used by the
# `sd(group, animal) ~ ...` route. The supplied A enters RAW (as native drmTMB
# does), matching the constant-SD Beta animal and relmat routes.
# The latent coordinate is u ~ N(0, A), while b = diag(exp(W*alpha)) * u;
# because integration stays in u-space, there is no separate Jacobian term.
function _beta_animal_sd_mode(y, eta0, loading, gidx, Q, aux; b0 = nothing,
                             maxiter::Int = 500, tol::Real = 1e-8)
    q = size(Q, 1)
    b = b0 === nothing ? zeros(q) : copy(b0)
    all(isfinite, loading) || return b, nothing, false
    function joint(u)
        value = 0.5 * dot(u, Q * u)
        @inbounds for i in eachindex(y)
            eta = eta0[i] + loading[i] * u[gidx[i]]
            value += _laplace_value(Val(:beta_fixed), aux, i, eta)
        end
        value
    end
    fit = Optim.optimize(joint, b, Optim.LBFGS(),
        Optim.Options(iterations = maxiter, g_abstol = tol); autodiff = :forward)
    mode = Optim.minimizer(fit)
    all(isfinite, mode) && isfinite(Optim.minimum(fit)) ||
        return mode, nothing, false
    mode = _beta_animal_sd_newton_polish(joint, mode, y, eta0, loading, gidx, Q,
                                         aux; tol)
    grad = ForwardDiff.gradient(joint, mode)
    # Optim's LBFGS termination can leave a small AD-gradient floor for the
    # dense animal objective. Keep the requested optimizer tolerance, but avoid
    # turning an otherwise regular stencil point into the failed-fit sentinel
    # solely because its final gradient lies between 1e-8 and 1e-7. Other mode
    # or curvature failures still return the sentinel.
    mode_tol = max(tol, 1e-7)
    norm(grad) <= mode_tol * (1 + norm(mode)) || return mode, nothing, false
    diag_h = zeros(Float64, q)
    @inbounds for i in eachindex(y)
        gi = gidx[i]
        eta = eta0[i] + loading[i] * mode[gi]
        abs(eta) < 30.0 || return mode, nothing, false
        diag_h[gi] += loading[i]^2 * _laplace_d2(Val(:beta_fixed), aux, i, eta)
    end
    hessian = Q + spdiagm(0 => diag_h)
    ch = cholesky(Symmetric(hessian); check = false)
    issuccess(ch) || return mode, ch, false
    return mode, ch, Optim.converged(fit)
end

function _beta_animal_sd_laplace_nll(theta, y, Xmu, Wsd, gidx, A; b0 = nothing)
    pmu = size(Xmu, 2)
    psd = size(Wsd, 2)
    length(theta) == pmu + 1 + psd || throw(ArgumentError(
        "Beta animal SD parameter vector must contain mu, sigma, and SD coefficients."))
    length(gidx) == length(y) == size(Xmu, 1) || throw(ArgumentError(
        "Beta animal SD data, design, and group indices must have matching rows."))
    Q, leaf_node = _general_cov_setup(A, gidx; normalize = false)
    size(Wsd, 1) == size(Q, 1) || throw(ArgumentError(
        "Beta animal SD design must have one row per animal level."))
    qchol = cholesky(Symmetric(Q); check = false)
    issuccess(qchol) || throw(ArgumentError(
        "Beta animal covariance precision is not positive definite."))
    logdetQ = logdet(qchol)
    beta = @view theta[1:pmu]
    sigma = theta[pmu + 1]
    -4.0 <= sigma <= 4.0 || return 1e18
    alpha = @view theta[(pmu + 2):end]
    aux_from, _, _ = _beta_laplace_setup(y, Xmu)
    aux = aux_from(sigma)
    eta0 = Xmu * beta
    group_sd = exp.(Wsd * alpha)
    loading = group_sd[leaf_node]
    u, ch, ok = _beta_animal_sd_mode(y, eta0, loading, leaf_node, Q, aux; b0)
    ok || return 1e18
    value = 0.5 * dot(u, Q * u)
    @inbounds for i in eachindex(y)
        eta = eta0[i] + loading[i] * u[leaf_node[i]]
        value += _laplace_value(Val(:beta_fixed), aux, i, eta)
    end
    return value - 0.5 * logdetQ + 0.5 * logdet(ch)
end

# Joint Beta animal-intercept SD regression for the supplied-A q1 slice. The
# latent coordinate is standardized u ~ N(0, R), with b_g = exp(W_g*alpha)u_g;
# the ordinary Beta location and log-dispersion coefficients are optimized
# jointly with alpha under the same Laplace objective as the dense oracle.
function _fit_beta_animal_sd_regression(fam::Beta, y, Xmu, Xsigma, Wsd, gidx,
                                        A, nm_mu, nm_sigma, nm_sd, grp, g_tol;
                                        se::Bool = true)
    size(Xsigma, 2) == 1 && all(==(1.0), @view Xsigma[:, 1]) ||
        throw(ArgumentError("Beta `animal(1 | $grp)` SD regression requires `sigma ~ 1`."))
    _, beta0, sigma0 = _beta_laplace_setup(y, Xmu)
    sigma0 = clamp(sigma0, -3.9, 3.9)
    pmu, psd = size(Xmu, 2), size(Wsd, 2)
    theta0 = vcat(beta0, sigma0, log(0.4), zeros(psd - 1))
    nll(theta) = _beta_animal_sd_laplace_nll(theta, y, Xmu, Wsd, gidx, A)
    function grad!(g, theta)
        for j in eachindex(theta)
            h = 2e-4 * (1 + abs(theta[j]))
            e = zeros(length(theta)); e[j] = h
            g[j] = (nll(theta .+ e) - nll(theta .- e)) / (2h)
        end
        return g
    end
    lower, upper = fill(-Inf, length(theta0)), fill(Inf, length(theta0))
    lower[pmu + 1], upper[pmu + 1] = -4.0, 4.0
    od = Optim.OnceDifferentiable(nll, grad!, theta0)
    result = Optim.optimize(od, lower, upper, theta0,
        Optim.Fminbox(Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking())),
        Optim.Options(g_tol = g_tol, iterations = 800))
    theta_hat = Optim.minimizer(result)
    nll_hat = nll(theta_hat)
    isfinite(nll_hat) && nll_hat < 1e18 || throw(ErrorException(
        "Beta animal SD optimization failed: no finite Laplace objective was found."))
    near_sigma_bound = abs(theta_hat[pmu + 1]) >= 4.0 -
        _fd_hessian_step(length(y)) * (1 + abs(theta_hat[pmu + 1]))
    V = se && !near_sigma_bound ? _vcov_from_fd_hessian(_finite_hessian(nll, theta_hat;
        h = _fd_hessian_step(length(y)))) :
        fill(NaN, length(theta_hat), length(theta_hat))
    blocks = [:mu => 1:pmu, :sigma => (pmu + 1):(pmu + 1),
              :sd => (pmu + 2):length(theta_hat)]
    names = [:mu => nm_mu, :sigma => nm_sigma, :sd => nm_sd]
    alpha = @view theta_hat[(pmu + 2):end]
    sd_group = exp.(Wsd * alpha)
    means = Dict(:mu => _logistic.(Xmu * theta_hat[1:pmu]))
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => fill(exp(theta_hat[pmu + 1]), length(y)),
                  :sd_group => sd_group, :sd => sd_group[gidx],
                  :sd_group_index => Float64.(gidx))
    fit = DrmFit(fam, blocks, names, theta_hat, Matrix(V), -nll_hat,
                 length(y), Optim.converged(result), means, obs, scales)
    return _withiterations(_withnll(fit, nll), Optim.iterations(result))
end

function _fit_beta_sd_regression(fam::Beta, y, Xmu, Xsigma, Zg, gidx, G,
                                 nmmu, nmsigma, nmsd, grp, g_tol;
                                 K::Int = _SD_IID_AGHQ_K, se::Bool = true,
                                 w::AbstractVector{<:Real} = ones(length(y)))
    K >= 1 || throw(ArgumentError("nAGQ must be ≥ 1; got $K"))
    n = length(y); pmu, psigma, psd = size(Xmu, 2), size(Xsigma, 2), size(Zg, 2)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    rule = _AGHQRule(1, K)
    length(w) == n || throw(ArgumentError(
        "Beta random-effect loading must have one value per observation."))
    Zre = reshape(Float64.(w), n, 1)
    bcache = [zeros(1, G) for _ in 1:Threads.maxthreadid()]
    function nll(theta)
        beta_mu = @view theta[1:pmu]
        beta_sigma = @view theta[(pmu+1):(pmu+psigma)]
        alpha = @view theta[(pmu+psigma+1):(pmu+psigma+psd)]
        eta_mu = Xmu * beta_mu
        eta_sigma = clamp.(Xsigma * beta_sigma, -15.0, 15.0)
        sdgroup = exp.(Zg * alpha)
        ll = (i, eta) -> begin
            mu = _logistic(clamp(eta, -30.0, 30.0))
            precision = exp(-2 * eta_sigma[i])
            Distributions.logpdf(Distributions.Beta(mu * precision,
                (1 - mu) * precision), y[i])
        end
        cache = bcache[Threads.threadid()]
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
    ybar = sum(y) / n; v = sum(abs2, y .- ybar) / max(n - 1, 1)
    precision0 = max(ybar * (1 - ybar) / max(v, eps()) - 1, 0.5)
    theta0 = zeros(pmu + psigma + psd)
    if all(isone, Zre)
        theta0[1] = log(ybar / (1 - ybar))
    else
        ylogit = log.(clamp.(Float64.(y), 1e-4, 1 - 1e-4) ./
                       (1 .- clamp.(Float64.(y), 1e-4, 1 - 1e-4)))
        theta0[1:pmu] .= Xmu \ ylogit
    end
    theta0[pmu+1] = -0.5 * log(precision0)
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
    means = Dict(:mu => _logistic.(Xmu * theta_hat[1:pmu]))
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xsigma * theta_hat[(pmu+1):(pmu+psigma)]),
                  :sd_group => sdgroup, :sd => sdgroup[gidx],
                  :sd_group_index => Float64.(gidx), :aghq_nodes => [Float64(K)])
    return _withiterations(_withnll(DrmFit(fam, blocks, names, theta_hat, V,
        -nll(theta_hat), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# Beta GLMM with a random intercept (1|g) on the logit mean. b_g ~ N(0,σ_b²)
# integrated out per group by 32-node Gauss–Hermite quadrature; precision
# φ = 1/σ² is a fixed effect. Same scheme as the count GLMMs.
function _fit_beta_ranef(fam::Beta, y, Xμ, Xσ, gidx, G, nmμ, nmσ, grp, g_tol; K::Int = _RANEF1D_AGHQ_K)
    n = length(y); pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    rule = _AGHQRule(1, K); Zre = ones(n, 1); bcache = zeros(1, G)   # #719: per-group AGHQ
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; σb = exp(θ[pμ+pσ+1])
        η0 = Xμ * βμ; ησ = clamp.(Xσ * βσ, -15.0, 15.0)
        ll = (i, η) -> (μ = _logistic(clamp(η, -30.0, 30.0)); φ = exp(-2 * ησ[i]);
                        Distributions.logpdf(Distributions.Beta(μ * φ, (1 - μ) * φ), y[i]))
        L = reshape([σb], 1, 1)
        return -_aghq_marginal_loglik(ll, members, η0, Zre, L, rule, bcache)
    end
    ȳ = sum(y) / n; v = sum(abs2, y .- ȳ) / max(n - 1, 1)
    φ0 = max(ȳ * (1 - ȳ) / max(v, eps()) - 1, 0.5)
    θ0 = zeros(pμ + pσ + 1)
    θ0[1] = log(ȳ / (1 - ȳ)); θ0[pμ+1] = -0.5 * log(φ0); θ0[pμ+pσ+1] = log(0.5)
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :resd => (pμ+pσ+1):(pμ+pσ+1)]
    names = [:mu => nmμ, :sigma => nmσ, :resd => [String(grp)]]
    means = Dict(:mu => _logistic.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => _reported_sigma(Xσ * θ̂[(pμ+1):(pμ+pσ)], -15.0, 15.0))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# Beta GLMM with a correlated random intercept+slope (1 + x | g) on the logit
# mean: per group (b0,b1) ~ N(0, Σ), Σ = L Lᵀ with L = [l11 0; cc l22] (log-
# Cholesky a=log l11, b=log l22, cc). The 2-D group integral is taken by
# per-group ADAPTIVE Gauss–Hermite quadrature
# (`_aghq_marginal_loglik`, #834: nodes b̂_g + √2 C z at each group's mode),
# `nq` nodes per axis, the same scheme as the count GLMMs; precision φ = 1/σ² is a
# fixed effect. θ = [β_μ; β_σ; a, b, cc].
function _fit_beta_corr_ranef(fam::Beta, y, Xμ, Xσ, xs, gidx, G, nmμ, nmσ, grp, g_tol; nq::Int = _CORR_RANEF_AGHQ_K)
    n = length(y); pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    rule = _AGHQRule(2, nq); Zre = hcat(ones(n), Float64.(xs)); bcache = zeros(2, G)   # #834: per-group AGHQ
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]
        L = _corr_ranef_L(θ[pμ+pσ+1], θ[pμ+pσ+2], θ[pμ+pσ+3])
        η0 = Xμ * βμ; ησ = clamp.(Xσ * βσ, -15.0, 15.0)
        ll = (i, η) -> (μ = _logistic(clamp(η, -30.0, 30.0)); φ = exp(-2 * ησ[i]);
                        Distributions.logpdf(Distributions.Beta(μ * φ, (1 - μ) * φ), y[i]))
        return -_aghq_marginal_loglik(ll, members, η0, Zre, L, rule, bcache)
    end
    ȳ = sum(y) / n; v = sum(abs2, y .- ȳ) / max(n - 1, 1)
    φ0 = max(ȳ * (1 - ȳ) / max(v, eps()) - 1, 0.5)
    θ0 = zeros(pμ + pσ + 3)
    θ0[1] = log(ȳ / (1 - ȳ)); θ0[pμ+1] = -0.5 * log(φ0)
    θ0[pμ+pσ+1] = log(0.4); θ0[pμ+pσ+2] = log(0.4); θ0[pμ+pσ+3] = 0.0
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :recov => (pμ+pσ+1):(pμ+pσ+3)]
    names = [:mu => nmμ, :sigma => nmσ, :recov => ["$(grp):L11", "$(grp):L22", "$(grp):L21"]]
    means = Dict(:mu => _logistic.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => _reported_sigma(Xσ * θ̂[(pμ+1):(pμ+pσ)], -15.0, 15.0))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# Newton polish for the fixed-effect Beta location-scale fit (#1127). LBFGS on a raw
# design (e.g. `temp` near 300 beside a factor `sigma` block with 2-4 rows per level)
# can exhaust its iterations, or stop on a plateau, with the gradient still far above
# `g_tol`. From that stop point, Newton with the ForwardDiff gradient and Hessian
# converges in a few steps. The polished run is kept ONLY if the objective is finite and
# no worse AND it reaches `g_tol`, so a fit that already met `g_tol` is returned untouched
# and nothing can get worse. Skipped when the user took control of the optimiser (`maxit` / `start` /
# `multi_start` / `optimizer`), so a requested cap or start is honoured.
function _beta_newton_polish(res, nll, g_tol)
    s = _active_control_scope()
    (s !== nothing && _ctl_optim_requested(s.control)) && return res
    Optim.g_residual(res) <= g_tol && return res
    θ = Optim.minimizer(res)
    all(isfinite, θ) || return res
    pol = try
        Optim.optimize(nll, θ, Optim.Newton(), Optim.Options(g_tol = g_tol, iterations = 200);
                       autodiff = :forward)
    catch e
        e isa InterruptException && rethrow()
        return res
    end
    v = Optim.minimum(pol)
    # Accept only a genuine stationary point: with 2 rows per level the scale can run off
    # to a boundary where Newton keeps "improving" nll at a huge gradient. Such a fit is
    # left exactly as the primary run reported it.
    return (isfinite(v) && v <= Optim.minimum(res) && all(isfinite, Optim.minimizer(pol)) &&
            Optim.g_residual(pol) <= g_tol) ? pol : res
end

function _fit_beta(fam::Beta, y, Xμ, Xσ, nmμ, nmσ, g_tol; offset = nothing)
    n = length(y); pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    w = _ctl_weights(n)                  # prior weights: row log-lik multipliers (#931)
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]
        # Soft guards (see `_softclamp` in negbinomial.jl): identity across the
        # original hard band so no legitimate fit is distorted, smooth beyond so
        # the gradient stays live — restoring agreement with drmTMB's smooth guard.
        ημ = _softclamp.(_eta_with_offset(Xμ, βμ, offset), -30.0, 30.0, 3.0)   # μ ∈ (0,1) strictly
        ησ = _softclamp.(Xσ * βσ, -15.0, 15.0, 3.0)   # φ = exp(-2ησ) finite & > 0
        s = zero(eltype(θ))
        @inbounds for i in 1:n
            μ = _logistic(ημ[i]); φ = exp(-2 * ησ[i])
            s -= w[i] * (Distributions.logpdf(Distributions.Beta(μ * φ, (1 - μ) * φ), y[i]))
        end
        return s
    end
    ȳ = sum(y) / n; v = sum(abs2, y .- ȳ) / max(n - 1, 1)
    φ0 = max(ȳ * (1 - ȳ) / max(v, eps()) - 1, 0.5)   # method-of-moments precision
    θ0 = zeros(pμ + pσ)
    θ0[1] = log(ȳ / (1 - ȳ))                          # logit mean
    θ0[pμ+1] = -0.5 * log(φ0)                          # σ = 1/√φ ⇒ log σ = -½ log φ
    θ0[1:pμ] .= _offset_start(θ0[1:pμ], Xμ, offset)   # #938
    res = _ctl_optimize((x0, opts, method) -> Optim.optimize(nll, x0, method, opts; autodiff = :forward),
                       θ0; g_tol = g_tol)   # control: maxit / start / multi_start (#948 #994)
    res = _beta_newton_polish(res, nll, g_tol)         # #1127
    θ̂ = Optim.minimizer(res); V = _ctl_se(() -> _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂)), θ̂)
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ)]
    names = [:mu => nmμ, :sigma => nmσ]
    means = Dict(:mu => _logistic.(_eta_with_offset(Xμ, θ̂[1:pμ], offset))); obs = Dict(:mu => Vector{Float64}(y))  # response-scale μ̂
    scales = Dict(:sigma => _reported_sigma(Xσ * θ̂[(pμ+1):(pμ+pσ)], -15.0, 15.0, 3.0))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end
