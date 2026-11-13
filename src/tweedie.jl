# tweedie.jl — Tweedie family (compound Poisson–Gamma, 1 < p < 2): semicontinuous
# responses with a point mass at 0 and a continuous positive part (biomass,
# rainfall, total insurance loss). Log link on the mean μ; `sigma` is the
# √dispersion (φ = σ²); `nu` is the power p on a logit-(1,2) link
# (p = 1 + logistic(η)). Var(y) = φ·μ^p. The density has no closed form — it is
# evaluated by the Dunn–Smyth compound Poisson–Gamma series. Mirrors drmTMB's
# `tweedie`. Fixed effects, ML.

using SpecialFunctions: loggamma

# log Σ_n W_n for the positive part: W_n = λⁿ y^{-nα} γ^{nα} / (n! Γ(-nα)), the
# compound Poisson–Gamma mixture (N ~ Poisson(λ); sum of N Gamma(-α, γ) jumps).
# The dominant n and the summation window are chosen in Float64 (detached, so the
# window doesn't break differentiability), then the window is summed in the
# parameter's number type so ForwardDiff differentiates through `loggamma(-nα)`.
#
# Window (#1028). The terms are log-concave in n and peak near
# n* ≈ y^{2-p} / (φ(2-p)), with a width that grows like √n*. The old fixed search
# (n ≤ 400) and fixed ±30-term window missed most of the mass once n* passed a
# few hundred — finite log-densities wrong by up to thousands of nats. Now the
# peak starts from the Stirling solution of d logW_n / dn = 0,
#   z − log n + α log(−α n) = 0  ⇒  log n* = (z + α log(−α)) / (1 − α),
# is polished to the exact integer maximum by a local climb, and the window grows
# from the peak in both directions until a term falls `_TWEEDIE_DROP` nats below
# the peak (each dropped term then contributes < e^-40 relative). If the window
# would need more than `_TWEEDIE_MAX_HALFWIDTH` terms on a side the density is
# returned as NaN, never as a finite truncated value.
const _TWEEDIE_DROP = 40.0
const _TWEEDIE_MAX_HALFWIDTH = 1_000_000

function _tweedie_window(zv::Float64, αv::Float64)
    logWn(nn) = nn * zv - loggamma(nn + 1.0) - loggamma(-nn * αv)
    lnpk = (zv + αv * log(-αv)) / (1 - αv)
    n0 = isfinite(lnpk) ? clamp(exp(lnpk), 1.0, 1e15) : 1.0
    nmax = max(1, round(Int, n0))
    best = logWn(float(nmax))
    isfinite(best) || return nothing
    while true                                   # climb to the integer maximum
        up = logWn(float(nmax + 1))
        if up > best
            nmax += 1; best = up; continue
        end
        nmax > 1 || break
        dn = logWn(float(nmax - 1))
        dn > best || break
        nmax -= 1; best = dn
    end
    cut = best - _TWEEDIE_DROP
    jl = nmax
    while jl > 1 && logWn(float(jl - 1)) >= cut
        jl -= 1
        nmax - jl > _TWEEDIE_MAX_HALFWIDTH && return nothing
    end
    jh = nmax
    while logWn(float(jh + 1)) >= cut
        jh += 1
        jh - nmax > _TWEEDIE_MAX_HALFWIDTH && return nothing
    end
    return jl, jh
end

# Plain Float64 value of a (possibly nested, e.g. inside `ForwardDiff.hessian`) Dual.
_tw_value(x::ForwardDiff.Dual) = _tw_value(ForwardDiff.value(x))
_tw_value(x::Real) = Float64(x)

function _tweedie_logW(y, λ, α, γ)
    z = log(λ) - α * log(y) + α * log(γ)
    win = _tweedie_window(_tw_value(z), _tw_value(α))
    win === nothing && return z * NaN
    jl, jh = win
    terms = [j * z - loggamma(j + 1.0) - loggamma(-j * α) for j in jl:jh]
    m = maximum(terms)
    return m + log(sum(exp(t - m) for t in terms))
end

# Tweedie log-density at one point (1 < p < 2, y ≥ 0, μ,φ > 0).
function _logpdf_tweedie(y, μ, φ, p)
    λ = μ^(2 - p) / (φ * (2 - p))
    y == 0 && return -λ
    α = (2 - p) / (1 - p); γ = φ * (p - 1) * μ^(p - 1)
    return -λ - y / γ - log(y) + _tweedie_logW(y, λ, α, γ)
end

"""
    Tweedie()

Tweedie response family (compound Poisson–Gamma, power `1 < p < 2`) for
semicontinuous data — an exact-zero mass plus a positive continuous part. Log
link on the mean `μ`; `sigma` is the √dispersion (so `φ = σ²`, `coef(fit, :sigma)`
is `log σ`); `nu` is the power `p` on a logit-`(1,2)` link
(`p = 1 + logistic(coef(fit, :nu))`). `Var(y) = φ·μ^p`. Density via the Dunn–Smyth
series. Mirrors `drmTMB`'s `tweedie`. An ordinary `(1 | g)` random intercept
and an independent `(0 + x | g)` random slope on `mu` are both supported
(per-group adaptive Gauss–Hermite marginal); `sigma ~ 1` and `nu ~ 1` remain
fixed-effect sub-models on either route. Crossed/multiple random intercepts,
`(1 | g) + (1 | h)`, are also supported, via the sparse
augmented-state Laplace GLMM engine shared with Gamma/NB2/Beta's crossed
routes; `sigma ~ 1` and `nu ~ 1` are REQUIRED (not just default) on the
crossed route, and `p` is found by an outer 1-D profile search over the one
dispersion-like nuisance slot the shared engine does not also give to `p`.
The CORRELATED random slope `(1 + x | g)`, random effects on `sigma`/`nu`, and
animal or spatial markers, structured slopes, and structured terms combined with
other random effects on `mu` are not implemented. A single phylogenetic
intercept `phylo(1 | g)` (with `tree = …`) or related-matrix intercept
`relmat(1 | g)` (with `K = …`) is supported, with fixed `sigma ~ 1` and
`nu ~ 1`; the bare term is the `sd(g, …) ~ 1` model. The phylogenetic scale is
the tip correlation of an ultrametric tree and `K` enters raw, as in drmTMB.
Both use a Laplace approximation over the group-level vector, the same
approximation drmTMB uses, so estimates (not log-likelihoods across engines)
can be compared at the optimum.

Predictor-dependent SD regression is available for one ordinary mean random
intercept or one independent numeric mean slope with a group-constant SD
design, for example `sd(group) ~ z` or
`sd(group, coef(x)) ~ z`. A single related-matrix mean intercept is also
available with `relmat(1 | group)`, `sd(group, relmat) ~ z`, and `K = K`, and a
single phylogenetic mean intercept with `phylo(1 | group)`,
`sd(group, phylogenetic) ~ z`, and `tree = tree` (`z` constant within a group).
For ordinary effects, the mean, dispersion, and power retain their existing
submodels. The structured routes currently require `sigma ~ 1` and
`nu ~ 1`. Ordinary effects use adaptive Gauss–Hermite quadrature; the
related-matrix and phylogenetic routes use a Laplace approximation. Correlated ordinary blocks
and other structured providers are not implemented, and interval coverage is
not calibrated.

```julia
fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)), Tweedie(); data = dat)
exp(2 * coef(fit, :sigma)[1])               # dispersion φ
1 + 1 / (1 + exp(-coef(fit, :nu)[1]))       # power p ∈ (1,2)

fit_re = drm(bf(@formula(y ~ x + (1 | g)), @formula(nu ~ 1)), Tweedie(); data = dat)
exp(coef(fit_re, :resd)[1])                 # random-intercept SD

fit_slope = drm(bf(@formula(y ~ x + (0 + x | g)), @formula(nu ~ 1)), Tweedie(); data = dat)
exp(coef(fit_slope, :resd)[1])              # random-slope SD

fit_crossed = drm(bf(@formula(y ~ x + (1 | g) + (1 | h))), Tweedie(); data = dat)
re_sd(fit_crossed)                          # Dict(:g => ..., :h => ...)
```
"""
struct Tweedie end

_logit12(η) = 1 + 1 / (1 + exp(-η))            # (1,2) link for the power p

@drm_entry function drm(f::DrmFormula, fam::Tweedie; data, g_tol::Real = 1e-8,
                         tree = nothing, K = nothing, A = nothing,
                         coords = nothing, marginal::Symbol = :LA)
    _drm_entry_guard(f, fam, data; tree = tree, K = K, A = A, coords = coords)   # #1027 #1022
    laplace = _scalar_laplace_requested(marginal)
    laplace || marginal === :LA || throw(ArgumentError(
        "drm (Tweedie): marginal = :$marginal is not available; use `:LA` (default) " *
        "or `:Laplace` (independent `(1 | g)` / `(0 + x | g)` terms on the mean, drmTMB's integrator)."))
    f = _default_tweedie_structured_sd(f)   # bare phylo/relmat(1 | g): default `sd(g, …) ~ 1`
    if isempty(_sd_parts(f)) && isempty(_sdphylo_parts(f)) &&
       (tree !== nothing || K !== nothing || A !== nothing || coords !== nothing)
        throw(ArgumentError(
            "Tweedie uses `tree` only for one `phylo(1 | g)` mean intercept and `K` only for one `relmat(1 | g)` mean intercept " *
            "(each alone on `mu`, with fixed `sigma ~ 1` and `nu ~ 1`, optionally with `sd(g, …) ~ …`); this formula has no such term, so `tree`, `K`, `A` or `coords` would be ignored. " *
            "Structured slopes, animal and spatial terms, and structured terms combined with ordinary random effects are not implemented; drmTMB 0.8.1 refuses structured Tweedie slopes as well."))
    end
    missing_fit = _fit_observed_response_rows(f, data) do data_observed
        _tweedie_validate_observed_relmat_sd_design(f, data, data_observed)
        K_observed = _tweedie_observed_relmat_K(f, data, data_observed, K)
        drm(f, fam; data = data_observed, g_tol, tree, K = K_observed, A, coords, marginal)
    end
    missing_fit !== nothing &&
        return _tweedie_restore_full_relmat_sd_outputs!(missing_fit, f, data, K)

    if !isempty(_sd_parts(f)) || !isempty(_sdphylo_parts(f))
        laplace && return _laplace_sd_route(f, fam, Val(:tweedie), marginal; data, g_tol)
        return _withformula(_drm_tweedie_sd_regression(
            f, fam; data, g_tol, tree, K, A, coords), f)
    end
    _lss_only_gaussian_guard(f, fam)   # #544: refuse, never silently drop, sd() parts
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    fixed_mu, offset_term = _extract_offset(fixed_mu)   # #938: offset(z), coefficient fixed at 1
    offset_term === nothing || (isempty(re) && mv === nothing && st === nothing) ||
        _refuse_offset_route(fam, "a random, structured or `meta_V` term on the mean")
    mv === nothing || error("Tweedie() does not support meta_V markers")
    for (pname, r) in f.forms          # only the mean may carry a random effect (#563)
        pname === :mu && continue
        _, re2, mv2, st2 = _split_ranef(r)
        (isempty(re2) && mv2 === nothing && st2 === nothing) ||
            error("Tweedie() random effects are not implemented for `sigma`/`nu`; " *
                  "only an ordinary `(1 | g)` random intercept on `mu` is supported " *
                  "(matches drmTMB, which rejects sigma/nu random effects for tweedie())")
    end
    y, Xμ, nmμ = _design(f.response, fixed_mu, data)
    _, Xσ, nmσ = _design(f.response, get(rhs, :sigma, ConstantTerm(1)), data)
    _, Xν, nmν = _design(f.response, get(rhs, :nu, ConstantTerm(1)), data)
    all(yi -> yi >= 0, y) || error("Tweedie() requires non-negative responses (zeros allowed)")

    st === nothing ||
        error("Tweedie() supports a structured effect on `mu` only as one `phylo(1 | g, tree = …)` or " *
              "`relmat(1 | g, K = …)` intercept; animal and spatial terms are not implemented " *
              "(drmTMB 0.8.1 refuses them for tweedie() too)")
    laplace && length(re) == 1 && _re_kind(re[1][1])[1] in (:intercept, :slope) &&
        mv === nothing && st === nothing && offset_term === nothing &&
        return _withformula(_fit_indep_re_laplace(fam, Val(:tweedie), y, Xμ, Xσ, Xν,
                                                  _indep_term_comps(re, data, length(y)),
                                                  nmμ, nmσ, nmν, g_tol), f)
    laplace && length(re) <= 1 && throw(ArgumentError(
        "drm (Tweedie): `marginal = :Laplace` needs one `(1 | g)` or `(0 + x | g)` term, or several " *
        "independent terms, on the mean (no correlated block, structured or `meta_V` term, or offset)."))
    if !isempty(re)                    # random intercept/slope on mu → GHQ marginal (#563 S8)
        if length(re) > 1               # crossed/multiple intercepts (#737)
            # Independent intercepts + slopes `(1 | g) + (0 + x | g)`: TMB-convention Laplace with the
            # power `p` estimated jointly (drmTMB's layout), the shared dense route of
            # `indep_slopes_laplace.jl`. Any `sigma` / `nu` formula is admitted.
            _has_indep_slopes(re) &&
                return _withformula(_fit_indep_re_laplace(fam, Val(:tweedie), y, Xμ, Xσ, Xν,
                                                          _indep_slope_comps(re, data, length(y)),
                                                          nmμ, nmσ, nmν, g_tol), f)
            all(_re_kind(r[1])[1] === :intercept for r in re) ||
                error("Tweedie() supports multiple random effects only as crossed/nested " *
                      "intercepts `(1 | g) + (1 | h)` or independent intercepts and slopes " *
                      "`(1 | g) + (0 + x | g)`; a correlated `(1 + x | g)` must be the only " *
                      "random-effect term")
            size(Xσ, 2) == 1 ||
                error("Tweedie() crossed/multiple random intercepts currently require a " *
                      "constant `sigma` formula")
            size(Xν, 2) == 1 ||
                error("Tweedie() crossed/multiple random intercepts currently require a " *
                      "constant `nu` formula")
            comps = map(re) do r
                grp = r[2]; gidx, G = _group_index(getproperty(data, grp); col = grp)
                (ones(length(y)), gidx, G, String(grp))
            end
            return _withformula(_fit_tweedie_crossed_laplace(fam, y, Xμ, comps, nmμ, nmσ, nmν, g_tol), f)
        end
        length(re) == 1 ||
            error("Tweedie() supports only a single `(1 | g)` random intercept or " *
                  "`(0 + x | g)` random slope on `mu`; crossed/multiple random effects " *
                  "are not implemented")
        (rk, var) = _re_kind(re[1][1]); grp = re[1][2]
        gidx, G = _group_index(getproperty(data, grp); col = grp)
        if rk === :intercept
            return _withformula(
                _fit_tweedie_ranef(fam, y, Xμ, Xσ, Xν, gidx, G, nmμ, nmσ, nmν, grp, g_tol), f)
        elseif rk === :slope
            xs = Float64.(getproperty(data, var))
            return _withformula(
                _fit_tweedie_slope_ranef(fam, y, Xμ, Xσ, Xν, xs, gidx, G, nmμ, nmσ, nmν, grp, g_tol), f)
        else
            error("Tweedie() supports only an ordinary `(1 | g)` random intercept or an " *
                  "independent `(0 + x | g)` random slope on `mu`; correlated random " *
                  "slopes `(1 + x | g)` are not implemented (matches drmTMB 0.7.0: " *
                  "\"Only independent tweedie() mu random intercepts and slopes are " *
                  "implemented in this slice\")")
        end
    end
    return _withformula(_fit_tweedie(fam, y, Xμ, Xσ, Xν, nmμ, nmσ, nmν, g_tol; offset = _offset_for_fit(fam, offset_term, data, size(Xμ, 1))), f)
end

# Predictor-dependent SD for one ordinary IID mean random intercept or
# independent slope. The Tweedie dispersion and power retain their existing
# submodels.
function _drm_tweedie_sd_regression(f::DrmFormula, fam::Tweedie; data,
                                    g_tol::Real = 1e-8, se::Bool = true,
                                    tree = nothing, K = nothing, A = nothing,
                                    coords = nothing,
                                    nAGQ::Int = _TWEEDIE_RANEF_AGHQ_K)
    targets = _gamma_sd_targets(f)   # also lifts `sd(g, phylogenetic) ~ …` to a phylogenetic target
    length(targets) == 1 || throw(ArgumentError(
        "Tweedie random-effect SD regression requires exactly one `sd(group) ~ …` formula."))
    target = only(targets)
    descriptor = target.target
    (descriptor.level in (:iid, :relmat, :phylogenetic) && descriptor.dpar === :mu &&
     descriptor.block === nothing) || throw(ArgumentError(
        "Tweedie SD regression targets one ordinary mean coefficient, one related-matrix mean intercept, or one phylogenetic mean intercept."))
    if descriptor.level in (:relmat, :phylogenetic)
        return _drm_tweedie_structured_sd_regression(f, fam; data, target, g_tol,
                                                      tree, K, A, coords, se)
    end
    (tree === nothing && K === nothing && A === nothing && coords === nothing) ||
        throw(ArgumentError("Tweedie ordinary SD regression does not use `tree`, `K`, `A`, or `coords`."))
    isempty(_sdphylo_parts(f)) || throw(ArgumentError(
        "Tweedie SD regression does not support structured random-effect SD formulas."))
    grp, sd_rhs = descriptor.group, target.rhs
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    mv === nothing || throw(ArgumentError("Tweedie SD regression cannot use meta_V()."))
    st === nothing || throw(ArgumentError("Tweedie SD regression does not support structured random effects."))
    (length(re) == 1) || throw(ArgumentError(
        "Tweedie SD regression requires exactly one ordinary mean random-effect term."))
    kind, slope = _re_kind(only(re)[1])
    kind in (:intercept, :slope) || throw(ArgumentError(
        "Tweedie SD regression supports one ordinary intercept or one independent numeric slope; correlated and structured effects are not implemented."))
    member = kind === :intercept ? :intercept : slope
    descriptor.member in (nothing, member) || throw(ArgumentError(
        "Tweedie SD target `coef($(descriptor.member))` does not match random-effect coefficient `$(member)`."))
    (kind === :intercept || descriptor.member === member) || throw(ArgumentError(
        "Tweedie random slopes require an explicit `coef($slope)` SD target."))
    only(re)[2] === grp || throw(ArgumentError(
        "Tweedie `sd($grp)` must name the random-effect grouping variable `$(only(re)[2])`."))
    for (pname, form_rhs) in f.forms
        (pname in (:mu, :sigma, :nu) || pname === target.key) && continue
        throw(ArgumentError(
            "Tweedie SD regression supports only `mu`, `sigma`, `nu`, and one `sd($grp)` formula."))
    end
    for pname in (:sigma, :nu)
        haskey(rhs, pname) || continue
        _, re_p, mv_p, st_p = _split_ranef(rhs[pname])
        (isempty(re_p) && mv_p === nothing && st_p === nothing) || throw(ArgumentError(
            "Tweedie() random effects are not implemented for `$pname`; only a mean random-effect coefficient is supported."))
    end
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "Tweedie random-effect SD regression requires observed responses; remove incomplete rows before fitting."))
    y, Xmu, nmmu = _design(f.response, fixed_mu, data)
    _, Xsigma, nmsigma = _design(f.response, get(rhs, :sigma, ConstantTerm(1)), data)
    _, Xnu, nmnu = _design(f.response, get(rhs, :nu, ConstantTerm(1)), data)
    all(yi -> isfinite(yi) && yi >= 0, y) || throw(ArgumentError(
        "Tweedie() requires finite non-negative responses (zeros allowed)"))
    gidx, G = _group_index(getproperty(data, grp))
    loading = if kind === :intercept
        ones(Float64, length(y))
    else
        hasproperty(data, slope) || throw(ArgumentError(
            "Tweedie random-slope predictor `$slope` is absent from data."))
        raw = getproperty(data, slope)
        raw isa AbstractVector{<:Real} && length(raw) == length(y) &&
            all(isfinite, raw) || throw(ArgumentError(
            "Tweedie random-slope predictor `$slope` must contain one finite numeric value per response."))
        any(x -> !iszero(x), raw) || throw(ArgumentError(
            "Tweedie random-slope predictor `$slope` must contain at least one nonzero value."))
        Float64.(raw)
    end
    Wsd, nmsd = _sd_group_design(f.response, sd_rhs, data, gidx, G, grp;
                                 schema_cache = f.schema_cache,
                                 schema_key = target.key)
    size(Wsd, 2) <= G && rank(Wsd) == size(Wsd, 2) || throw(ArgumentError(
        "Tweedie `sd($grp)` group-level design is rank deficient; use predictors that vary independently across groups."))
    return _fit_tweedie_sd_regression(fam, y, Xmu, Xsigma, Xnu, loading, Wsd, gidx,
        G, nmmu, nmsigma, nmnu, nmsd, grp, g_tol; K = nAGQ, se)
end

# One Laplace route for a single structured mean intercept: `relmat(1 | g)` with
# the raw covariance `K`, or `phylo(1 | g)` with the tip correlation of `tree`
# (the scale native drmTMB uses). Both share `_fit_tweedie_relmat_sd_laplace`.
function _drm_tweedie_structured_sd_regression(f::DrmFormula, fam::Tweedie;
        data, target, g_tol::Real, tree, K, A, coords, se::Bool)
    descriptor = target.target
    grp = descriptor.group
    phylogenetic = descriptor.level === :phylogenetic
    kind = phylogenetic ? :phylo : :relmat
    descriptor.member in (nothing, :intercept) || throw(ArgumentError(
        "Tweedie structured SD regression currently supports one mean intercept."))
    if phylogenetic
        (K === nothing && A === nothing && coords === nothing) || throw(ArgumentError(
            "Tweedie phylo SD regression uses `tree = …` only; do not also supply `K`, `A`, or `coords`."))
        tree === nothing && throw(ArgumentError("phylo(1 | $grp) needs `tree = …`."))
    else
        (tree === nothing && A === nothing && coords === nothing) || throw(ArgumentError(
            "Tweedie relmat SD regression uses `K = …` only; do not also supply `tree`, `A`, or `coords`."))
        K === nothing && throw(ArgumentError(
            "relmat(1 | $grp) with predictor-dependent SD requires `K = …`."))
    end

    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    mv === nothing || throw(ArgumentError("Tweedie $kind SD regression cannot use meta_V()."))
    st == (kind, grp) || throw(ArgumentError(
        "Tweedie `sd($grp, $(phylogenetic ? "phylogenetic" : "relmat"))` requires the matching `$kind(1 | $grp)` mean term."))
    structured = _collect_structured(rhs[:mu])
    length(structured) == 1 && only(structured) == (kind, grp) ||
        throw(ArgumentError(
            "Tweedie $kind SD regression requires exactly one `$kind(1 | $grp)` term."))
    isempty(re) || throw(ArgumentError(
        "Tweedie $kind SD regression currently requires one structured intercept without ordinary random effects."))
    for (pname, _) in f.forms
        (pname === :mu || pname === :sigma || pname === :nu || pname === target.key) && continue
        throw(ArgumentError("Tweedie $kind SD regression supports only `mu`, fixed `sigma ~ 1`, fixed `nu ~ 1`, and one $(phylogenetic ? "phylogenetic" : "related-matrix") SD formula."))
    end
    for pname in (:sigma, :nu)
        haskey(rhs, pname) || continue
        rhs[pname] == ConstantTerm(1) || throw(ArgumentError(
            "Tweedie $kind SD regression requires fixed `$(pname) ~ 1`; predictor-dependent `sigma` and `nu` are not supported on this route."))
        _, re_p, mv_p, st_p = _split_ranef(rhs[pname])
        (isempty(re_p) && mv_p === nothing && st_p === nothing) || throw(ArgumentError(
            "Tweedie $kind random effects are supported only for `mu`."))
    end
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "Tweedie $kind SD regression requires observed responses; remove incomplete rows before fitting."))
    y, Xmu, nmmu = _design(f.response, fixed_mu, data)
    _, Xsigma, nmsigma = _design(f.response, get(rhs, :sigma, ConstantTerm(1)), data)
    _, Xnu, nmnu = _design(f.response, get(rhs, :nu, ConstantTerm(1)), data)
    size(Xsigma, 2) == 1 && size(Xnu, 2) == 1 || throw(ArgumentError(
        "Tweedie $kind SD regression requires fixed `sigma ~ 1` and `nu ~ 1`."))
    all(yi -> isfinite(yi) && yi >= 0, y) || throw(ArgumentError(
        "Tweedie() requires finite non-negative responses (zeros allowed)"))
    labels = getproperty(data, grp)
    Q, gidx, kernel_variance = phylogenetic ?
        _tweedie_phylo_cov_setup(tree, labels, grp) :
        _tweedie_relmat_cov_setup(K, labels)
    G = size(Q, 1)
    Zg, nmsd = _sd_group_design(f.response, target.rhs, data, gidx, G, grp;
        schema_cache = f.schema_cache, schema_key = target.key)
    size(Zg, 2) <= G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
        "Tweedie `sd($grp, $(phylogenetic ? "phylogenetic" : "relmat"))` design is rank deficient; use predictors that vary independently across groups."))
    return _fit_tweedie_relmat_sd_laplace(fam, y, Xmu, Xsigma, Xnu,
        Zg, gidx, Q, kernel_variance, nmmu, nmsigma, nmnu, nmsd, grp, g_tol; se)
end

# The shared missing-response wrapper rebuilds rowwise predictions over the
# original design, but retains these structured-effect summaries from the
# observed-row fit. Rebuild them in the original group order as well, including
# groups with no observed response, using the original kernel scale.
function _tweedie_restore_full_relmat_sd_outputs!(fit::DrmFit{Tweedie},
        f::DrmFormula, data, K)
    targets = _sd_target_parts(f)
    length(targets) == 1 || return fit
    target = only(targets)
    target.target.level === :relmat || return fit
    K === nothing && return fit

    group = target.target.group
    labels = getproperty(data, group)
    _, group_index, kernel_variance = _tweedie_relmat_cov_setup(K, labels)
    sd_rows = predict_parameters(fit, data)[:sd]
    sd_group = Float64[
        sd_rows[findfirst(==(g), group_index)] for g in 1:length(kernel_variance)
    ]
    marginal_sd_group = sd_group .* sqrt.(kernel_variance)
    fit.scales[:sd_group] = sd_group
    fit.scales[:sd] = sd_rows
    fit.scales[:sd_group_index] = Float64.(group_index)
    fit.scales[:marginal_sd_group] = marginal_sd_group
    fit.scales[:marginal_sd] = marginal_sd_group[group_index]
    return fit
end

function _tweedie_observed_relmat_K(f::DrmFormula, data, data_observed, K)
    K === nothing && return nothing
    targets = _sd_target_parts(f)
    length(targets) == 1 || return K
    descriptor = only(targets).target
    descriptor.level === :relmat || return K
    group = descriptor.group
    original_levels = unique(getproperty(data, group))
    retained_levels = unique(getproperty(data_observed, group))
    length(original_levels) == size(K, 1) == size(K, 2) || throw(ArgumentError(
        "Tweedie relmat covariance matrix must match the original first-seen levels of `$(group)`."))
    keep = [findfirst(==(level), original_levels) for level in retained_levels]
    any(isnothing, keep) && throw(ArgumentError(
        "Observed `$(group)` levels must be a subset of the original levels."))
    keep_index = Int[index for index in keep]
    return K[keep_index, keep_index]
end

function _tweedie_validate_observed_relmat_sd_design(f::DrmFormula, data,
                                                       data_observed)
    targets = _sd_target_parts(f)
    length(targets) == 1 || return nothing
    target = only(targets)
    descriptor = target.target
    descriptor.level === :relmat || return nothing

    group = descriptor.group
    full_index, full_groups = _group_index(getproperty(data, group))
    _, full_names = _sd_group_design(f.response, target.rhs, data, full_index,
        full_groups, group; schema_cache = f.schema_cache,
        schema_key = target.key)

    observed_index, observed_groups =
        _group_index(getproperty(data_observed, group))
    observed_design, observed_names = _sd_group_design(f.response, target.rhs,
        data_observed, observed_index, observed_groups, group;
        schema_cache = f.schema_cache, schema_key = target.key)
    observed_names == full_names || throw(ArgumentError(
        "Tweedie `sd($group)` design changed after missing-response rows were removed."))

    unsupported = [full_names[j] for j in axes(observed_design, 2)
                   if all(iszero, @view observed_design[:, j])]
    isempty(unsupported) || throw(ArgumentError(
        "Tweedie `sd($group)` design coefficient(s) `$(join(unsupported, ", "))` " *
        "occur only in groups with no observed responses and cannot be estimated. " *
        "Remove those levels or provide observed responses for them."))
    return nothing
end

function _tweedie_relmat_covariance(C, labels)
    group_index, G = _group_index(labels)
    size(C) == (G, G) || throw(ArgumentError(
        "Tweedie relmat covariance matrix must be $(G)×$(G) in first-seen group order."))
    Cfloat = Matrix{Float64}(C)
    all(isfinite, Cfloat) || throw(ArgumentError(
        "Tweedie relmat covariance matrix must contain only finite values."))
    isapprox(Cfloat, Cfloat'; rtol = 1e-10, atol = 1e-12) || throw(ArgumentError(
        "Tweedie relmat covariance matrix must be symmetric."))
    Csym = Symmetric((Cfloat + Cfloat') / 2)
    Cchol = cholesky(Csym; check = false)
    issuccess(Cchol) || throw(ArgumentError(
        "Tweedie relmat covariance matrix must be positive definite."))
    return Csym, group_index, diag(Csym)
end

function _tweedie_relmat_cov_setup(C, labels)
    Csym, group_index, kernel_variance =
        _tweedie_relmat_covariance(C, labels)
    return sparse(Symmetric(inv(cholesky(Csym)))), group_index, kernel_variance
end

# Tip correlation of `tree` (ultrametric, every tip observed) as a precision, with
# rows indexed by tree leaf as in the phylo q2 routes.
function _tweedie_phylo_cov_setup(tree, labels, grp)
    C, gidx, _ = _structured_q2_kernel(:phylo, grp, labels; tree, K = nothing,
        A = nothing, coords = nothing, who = "Tweedie phylo SD regression")
    Csym = Symmetric(Matrix{Float64}(C))
    return sparse(Symmetric(inv(cholesky(Csym)))), gidx, diag(Csym)
end

# A bare `phylo(1 | g)` or `relmat(1 | g)` mean intercept is the SD route with
# `sd(g, …) ~ 1`, as the Gamma and count q2 routes do for bare structured blocks.
function _default_tweedie_structured_sd(f::DrmFormula)
    (isempty(f.sd_targets) && isempty(_sd_target_parts(f)) &&
     isempty(_sdphylo_parts(f))) || return f
    rhs = Dict(f.forms)
    mu = get(rhs, :mu, nothing)
    mu === nothing && return f
    st = try
        _, re, metav, structured = _split_ranef(mu)
        isempty(re) && metav === nothing ? structured : nothing
    catch
        nothing
    end
    (st isa Tuple && first(st) in (:phylo, :relmat)) || return f
    forms = copy(f.forms)
    targets = copy(f.sd_targets)
    key = Symbol("__sd_target_1")
    push!(forms, key => ConstantTerm(1))
    targets[key] = (group = last(st),
                    level = first(st) === :phylo ? :phylogenetic : :relmat,
                    member = nothing, dpar = :mu, block = nothing)
    return DrmFormula(f.response, forms, f.response2, f.schema_cache, targets)
end

function _tweedie_relmat_scaled_precision(Q, log_sd)
    inv_sd = exp.(-log_sd)
    return Symmetric(Diagonal(inv_sd) * Q * Diagonal(inv_sd))
end

function _tweedie_relmat_polish_mode(objective, initial; g_tol::Real,
                                     maxiter::Int = 8)
    mode = Float64.(initial)
    all(isfinite, mode) || return nothing
    function certified_mode()
        isfinite(objective(mode)) || return nothing
        gradient = ForwardDiff.gradient(objective, mode)
        all(isfinite, gradient) && norm(gradient, Inf) <= g_tol || return nothing
        hessian = ForwardDiff.hessian(objective, mode)
        all(isfinite, hessian) || return nothing
        issuccess(cholesky(Symmetric(hessian); check = false)) || return nothing
        return mode
    end
    for _ in 1:maxiter
        gradient = ForwardDiff.gradient(objective, mode)
        all(isfinite, gradient) || return nothing
        gradient_norm = norm(gradient, Inf)
        gradient_norm <= g_tol && return certified_mode()

        hessian = ForwardDiff.hessian(objective, mode)
        all(isfinite, hessian) || return nothing
        hchol = cholesky(Symmetric(hessian); check = false)
        issuccess(hchol) || return nothing
        step = hchol \ gradient
        all(isfinite, step) || return nothing

        value = objective(mode)
        roundoff = 64eps(Float64) * max(1.0, abs(value))
        polished = false
        for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625)
            candidate = mode .- fraction .* step
            candidate_value = objective(candidate)
            if isfinite(candidate_value) && candidate_value <= value + roundoff
                candidate_gradient = ForwardDiff.gradient(objective, candidate)
                if all(isfinite, candidate_gradient) &&
                   norm(candidate_gradient, Inf) < gradient_norm
                    mode = candidate
                    polished = true
                    break
                end
            end
        end
        polished || return nothing
    end
    return certified_mode()
end

function _tweedie_relmat_inner_mode(objective, initial; g_tol::Real = 1e-7,
                                    maxiter::Int = 150)
    result = Optim.optimize(objective, initial,
        Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking()),
        Optim.Options(g_tol = g_tol, iterations = maxiter); autodiff = :forward)
    mode = Optim.minimizer(result)
    all(isfinite, mode) || return nothing
    drm_optim_converged(result) && return mode
    # LBFGS can stop on a floating-point objective plateau just above g_tol.
    # Accept a mode only after a positive-definite Newton polish produces a
    # fresh gradient certificate at the same tolerance.
    return _tweedie_relmat_polish_mode(objective, mode; g_tol)
end

# d/deta and d^2/deta^2 of the Tweedie negative log-density at log-mean `eta`.
function _tweedie_eta_derivs(y, eta, phi, power)
    nll(e) = -_logpdf_tweedie(y, exp(e), phi, power)
    dnll(e) = ForwardDiff.derivative(nll, e)
    return dnll(eta), ForwardDiff.derivative(dnll, eta)
end

# Damped Newton for the latent mode of the structured Laplace objective, using
# the exact structured gradient and Hessian. Returns `nothing` unless the
# gradient certificate `g_tol` holds, so the caller can fall back to LBFGS.
function _tweedie_structured_newton_mode(objective, grad_hess, initial;
                                          g_tol::Real = 5e-7, maxiter::Int = 60)
    b = Float64.(initial)
    all(isfinite, b) || return nothing
    value = objective(b)
    isfinite(value) || return nothing
    for _ in 1:maxiter
        gradient, H = grad_hess(b)
        all(isfinite, gradient) || return nothing
        norm(gradient, Inf) <= g_tol && return b
        hchol = cholesky(Symmetric(H); check = false)
        step = issuccess(hchol) ? hchol \ gradient : gradient
        slope = dot(gradient, step)
        t, accepted = 1.0, false
        while t > 1e-10
            candidate = b .- t .* step
            cv = objective(candidate)
            # a roundoff allowance keeps the search alive near the optimum,
            # where the true decrease is below the objective's floating-point noise
            if isfinite(cv) && cv <= value - 1e-4 * t * slope + 1e-12 * (1 + abs(value))
                b, value, accepted = candidate, cv, true
                break
            end
            t /= 2
        end
        accepted || return nothing
    end
    return nothing
end

function _fit_tweedie_relmat_sd_laplace(fam, y, Xmu, Xsigma, Xnu, Zg,
        gidx, Q, kernel_variance, nmmu, nmsigma, nmnu, nmsd, grp, g_tol; se::Bool = true,
        maxiter::Int = 250)
    n, G = length(y), size(Q, 1)
    pmu, psigma, pnu, psd = size(Xmu, 2), size(Xsigma, 2), size(Xnu, 2), size(Zg, 2)
    psigma == 1 && pnu == 1 || throw(ArgumentError(
        "Tweedie related-matrix SD regression requires fixed `sigma ~ 1` and `nu ~ 1`."))
    Qdense = Matrix(Symmetric(Matrix(Q)))
    qchol = cholesky(Symmetric(Qdense); check = false)
    issuccess(qchol) || throw(ArgumentError("Tweedie related-matrix precision must be positive definite."))
    logdetQ = logdet(qchol)
    # Keep the outer objective deterministic.  A warm-start cache makes the
    # selected inner mode depend on the order in which Optim and its finite-
    # difference gradient probe theta, so the outer optimizer would not see a
    # single well-defined Laplace objective.
    mode_start = zeros(G)

    function nll_from_start(theta, initial_b)
        beta_mu = @view theta[1:pmu]
        eta_sigma = clamp(theta[pmu + 1], -15.0, 15.0)
        eta_nu = clamp(theta[pmu + 2], -12.0, 12.0)
        alpha = @view theta[(pmu + 3):end]
        eta0 = clamp.(Xmu * beta_mu, -30.0, 30.0)
        phi, power = exp(2eta_sigma), _logit12(eta_nu)
        log_sd = Zg * alpha
        Qsd = _tweedie_relmat_scaled_precision(Qdense, log_sd)
        logdetQsd = logdetQ - 2sum(log_sd)

        function joint(b)
            value = 0.5 * dot(b, Qsd * b)
            @inbounds for i in eachindex(y)
                eta = clamp(eta0[i] + b[gidx[i]], -30.0, 30.0)
                value -= _logpdf_tweedie(y[i], exp(eta), phi, power)
            end
            return value
        end

        # Observation i depends on b only through b[gidx[i]], so the exact
        # gradient and Hessian are Qsd * b plus per-observation scalar eta
        # derivatives; no G-dimensional ForwardDiff pass is needed.
        function grad_hess(b)
            gradient = Qsd * b
            H = Matrix(Qsd)
            @inbounds for i in eachindex(y)
                k = gidx[i]
                eta_raw = eta0[i] + b[k]
                (-30.0 < eta_raw < 30.0) || continue
                d1, d2 = _tweedie_eta_derivs(y[i], eta_raw, phi, power)
                gradient[k] += d1
                H[k, k] += d2
            end
            return gradient, H
        end
        b0 = all(isfinite, initial_b) ? copy(initial_b) : zeros(G)
        b_hat = _tweedie_structured_newton_mode(joint, grad_hess, b0)
        b_hat === nothing &&
            (b_hat = _tweedie_structured_newton_mode(joint, grad_hess, zeros(G)))
        b_hat === nothing &&
            (b_hat = _tweedie_relmat_inner_mode(joint, b0; g_tol = 5e-7, maxiter = 150))
        b_hat === nothing &&
            (b_hat = _tweedie_relmat_inner_mode(joint, zeros(G);
                                                g_tol = 5e-7, maxiter = 300))
        b_hat === nothing && return 1e18, nothing
        H = last(grad_hess(b_hat))
        hchol = cholesky(Symmetric(H); check = false)
        issuccess(hchol) || return 1e18, nothing
        value = joint(b_hat) - 0.5 * logdetQsd + 0.5 * logdet(hchol)
        return value, b_hat
    end

    function nll(theta)
        return first(nll_from_start(theta, mode_start))
    end

    function grad_at!(gradient, theta, objective)
        for j in eachindex(theta)
            h = 2e-4 * (1 + abs(theta[j]))
            direction = zeros(length(theta)); direction[j] = h
            gradient[j] = (objective(theta .+ direction) -
                           objective(theta .- direction)) / (2h)
        end
        return gradient
    end

    grad!(gradient, theta) = grad_at!(gradient, theta, nll)

    theta0 = zeros(pmu + psigma + pnu + psd)
    theta0[1] = log(sum(y) / n + eps())
    theta0[pmu + 1] = 0.0
    theta0[pmu + 2] = 0.0
    theta0[(pmu + 3):end] .= Zg \ fill(log(0.4), G)
    # A fixed start at zero for every outer evaluation is deterministic but
    # can be far from the fitted conditional mode.  Seed the immutable start
    # once at the deterministic initial outer parameters, then use that same
    # mode for every subsequent evaluation.
    _, initial_mode = nll_from_start(theta0, mode_start)
    initial_mode === nothing && throw(ErrorException(
        "Tweedie related-matrix SD regression could not initialize the latent mode"))
    mode_start .= initial_mode
    od = Optim.OnceDifferentiable(nll, grad!, theta0)
    # Match the ordinary Tweedie SD route: a unit first step can move the
    # power predictor close to 2, where the exact density series becomes much
    # wider. A scaled initial step avoids that unnecessary excursion.
    optimizer = Optim.LBFGS(alphaguess = Optim.LineSearches.InitialStatic(
        alpha = 0.05, scaled = true))
    result = Optim.optimize(od, theta0, optimizer,
        Optim.Options(g_tol = g_tol, iterations = maxiter))
    theta_hat = Optim.minimizer(result)
    nll_hat, final_b = nll_from_start(theta_hat, mode_start)
    final_b === nothing && throw(ErrorException(
        "Tweedie related-matrix SD regression could not recover a final latent mode"))
    # Refit derivatives and the retained likelihood from the same immutable
    # start used by every outer objective evaluation.
    nll_final(theta) = first(nll_from_start(theta, mode_start))
    final_gradient = zeros(length(theta_hat))
    grad_at!(final_gradient, theta_hat, nll_final)
    converged = _laplace_outer_converged(result, nll_hat, final_gradient,
                                          theta_hat, n, g_tol)
    V = se ? _vcov_from_fd_hessian(
        _finite_hessian(nll_final, theta_hat; h = _fd_hessian_step(n));
        context = "sparse-Laplace Tweedie related-matrix SD regression") :
        fill(NaN, length(theta_hat), length(theta_hat))
    sd_group = exp.(Zg * theta_hat[(pmu + psigma + pnu + 1):end])
    marginal_sd_group = sd_group .* sqrt.(kernel_variance)
    means = Dict(:mu => exp.(Xmu * theta_hat[1:pmu]))
    observed = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => fill(exp(theta_hat[pmu + 1]), n),
        :nu => fill(_logit12(theta_hat[pmu + 2]), n),
        :sd_group => sd_group, :sd => sd_group[gidx],
        :marginal_sd_group => marginal_sd_group,
        :marginal_sd => marginal_sd_group[gidx],
        :sd_group_index => Float64.(gidx))
    _warn_at_clamp("Tweedie", ["log sigma" => ([theta_hat[pmu + 1]], -15.0, 15.0),
        "power (logit scale)" => ([theta_hat[pmu + 2]], -12.0, 12.0)])
    blocks = Pair{Symbol,UnitRange{Int}}[:mu => 1:pmu,
        :sigma => (pmu + 1):(pmu + 1), :nu => (pmu + 2):(pmu + 2),
        :sd => (pmu + 3):length(theta_hat)]
    names = Pair{Symbol,Vector{String}}[:mu => nmmu, :sigma => nmsigma,
        :nu => nmnu, :sd => nmsd]
    fit = DrmFit(fam, blocks, names, theta_hat, Matrix(V), -nll_final(theta_hat), n,
        converged, means, observed, scales)
    fit = _withranef(fit, Dict(grp => Float64.(final_b)))
    return _withiterations(_withnll(fit, nll_final), Optim.iterations(result))
end

"""
    predict_parameters(fit::DrmFit{Tweedie}, newdata;
                       type = :response, se = false, K = nothing)

Predict fitted distributional parameters for new observations. For a
related-matrix SD target, `:sd` is the multiplier of the supplied kernel. Pass
`K` to also return `:marginal_sd`, using the covariance matrix ordered by the
first-seen group labels in `newdata`. With `se = true`, both entries include
delta-method standard errors. Without `K`, only the SD multiplier is returned.
"""
function predict_parameters(fit::DrmFit{Tweedie}, newdata;
                            type::Symbol = :response, se::Bool = false,
                            K = nothing)
    f = fit.formula
    f === nothing && throw(ArgumentError(
        "predict_parameters: this fit did not retain its formula"))
    type in (:response, :link) || throw(ArgumentError(
        "predict_parameters: `type` must be :response or :link"))
    forms = Dict(f.forms)
    if K !== nothing
        targets = _gamma_sd_targets(f)
        length(targets) == 1 && only(targets).target.level === :relmat ||
            throw(ArgumentError(
                "predict_parameters: `K` is used only for one Tweedie related-matrix SD target"))
    end
    nd = NamedTuple(pairs(newdata))
    nrows = length(first(values(nd)))
    ytrain = get(fit.obs, :mu, Float64[])
    response_placeholder = isempty(ytrain) ? 0.0 : first(ytrain)
    ndr = merge(nd, NamedTuple{(f.response,)}((fill(response_placeholder, nrows),)))
    V = se ? vcov(fit) : nothing
    out = se ? Dict{Symbol,NamedTuple}() : Dict{Symbol,Vector{Float64}}()

    for (p, r) in fit.blocks
        # The fitted SD block is named `:sd`, while the public formula stores
        # its target under a group-specific key such as `:sd_group`. Resolve
        # SD targets from the retained formula below instead of silently
        # skipping them at this generic parameter-presence check.
        (p === :sd || haskey(forms, p)) || continue
        if p === :sd
            targets = _gamma_sd_targets(f)
            length(targets) == 1 || throw(ArgumentError(
                "predict_parameters: Tweedie SD prediction requires one random-effect target"))
            target = only(targets)
            grp = target.target.group
            hasproperty(nd, grp) || throw(ArgumentError(
                "predict_parameters: newdata must contain the random-effect grouping column `$grp`"))
            gidx, G = _group_index(getproperty(nd, grp))
            Zg, nmsd = _sd_group_design(f.response, target.rhs, ndr, gidx, G, grp;
                schema_cache = f.schema_cache, schema_key = target.key,
                store_schema = false)
            nmsd == _block_coefnames(fit, p) || throw(ArgumentError(
                "predict_parameters: the SD design columns do not match the fitted coefficient schema"))
            eta = Zg * coef(fit, p)
            link_value = eta[gidx]
            value = type === :link ? link_value : exp.(link_value)
            if se
                Zrow = Zg[gidx, :]
                Vsd = view(V, r, r)
                var_eta = vec(sum((Zrow * Vsd) .* Zrow; dims = 2))
                se_value = sqrt.(max.(var_eta, 0.0))
                type === :response && (se_value .*= value)
                out[p] = (; value, se = se_value)
            else
                out[p] = value
            end
            if K !== nothing
                labels = getproperty(nd, grp)
                _, kernel_group_index, kernel_variance =
                    _tweedie_relmat_covariance(K, labels)
                kernel_group_index == gidx || throw(ArgumentError(
                    "predict_parameters: `K` must follow the first-seen group order in newdata"))
                marginal_link = eta[gidx] .+
                    0.5 .* log.(kernel_variance[gidx])
                marginal_value = type === :link ? marginal_link : exp.(marginal_link)
                if se
                    marginal_se = sqrt.(max.(var_eta, 0.0))
                    type === :response && (marginal_se .*= marginal_value)
                    out[:marginal_sd] = (; value = marginal_value, se = marginal_se)
                else
                    out[:marginal_sd] = marginal_value
                end
            end
            continue
        end

        fixed, _, _, _ = _split_ranef(forms[p]; allow_phylo_slope = true)
        Xp, eta = _predict_design(fit, f.response, fixed, ndr, p,
                                  f.schema_cache, nothing)
        value = type === :link ? eta : _param_response(fit.family, p, eta)
        if se
            se_value = _delta_se(Xp, view(V, r, r), eta, type, fit.family, p)
            out[p] = (; value, se = se_value)
        else
            out[p] = value
        end
    end
    return out
end

function _fit_tweedie_sd_regression(fam::Tweedie, y, Xmu, Xsigma, Xnu, loading, Wsd,
                                    gidx, G, nmmu, nmsigma, nmnu, nmsd, grp,
                                    g_tol; K::Int = _TWEEDIE_RANEF_AGHQ_K,
                                    se::Bool = true)
    n = length(y)
    pmu, psigma, pnu, psd = size(Xmu, 2), size(Xsigma, 2), size(Xnu, 2), size(Wsd, 2)
    i1, i2 = pmu + psigma, pmu + psigma + pnu
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    rule = _AGHQRule(1, K)
    Zre = reshape(Float64.(loading), n, 1)
    bcache = zeros(1, G)
    function nll(theta)
        beta_mu = @view theta[1:pmu]
        beta_sigma = @view theta[(pmu+1):i1]
        beta_nu = @view theta[(i1+1):i2]
        alpha = @view theta[(i2+1):(i2+psd)]
        eta_mu = clamp.(Xmu * beta_mu, -30.0, 30.0)
        eta_sigma = clamp.(Xsigma * beta_sigma, -15.0, 15.0)
        eta_nu = clamp.(Xnu * beta_nu, -12.0, 12.0)
        sd_group = exp.(Wsd * alpha)
        ll = (i, eta) -> (mu = exp(clamp(eta, -30.0, 30.0));
                           phi = exp(2 * eta_sigma[i]);
                           power = _logit12(eta_nu[i]);
                           _logpdf_tweedie(y[i], mu, phi, power))
        total = zero(eltype(theta))
        for gi in 1:G
            Lg = reshape([sd_group[gi]], 1, 1)
            value, bhat = _aghq_group_logint(ll, members[gi], eta_mu,
                Zre, Lg, rule, @view(bcache[:, gi]))
            bcache[1, gi] = isfinite(bhat[1]) ? bhat[1] : 0.0
            total += value
        end
        return -total
    end
    theta0 = zeros(i2 + psd)
    theta0[1] = log(sum(y) / n + eps())
    theta0[pmu+1] = 0.0
    theta0[i1+1] = 0.0
    theta0[(i2+1):end] .= Wsd \ fill(log(0.5), G)
    # The default unit initial step can send the power predictor from its
    # midpoint to p ≈ 2 in one line-search trial. The exact Tweedie series is
    # valid there, but its summation window becomes very wide. Scale the first
    # step by the direction norm so line search explores this boundary
    # gradually without imposing a parameter bound.
    optimizer = Optim.LBFGS(alphaguess = Optim.LineSearches.InitialStatic(
        alpha = 0.05, scaled = true))
    result = Optim.optimize(nll, theta0, optimizer,
        Optim.Options(g_tol = g_tol); autodiff = :forward)
    theta_hat = Optim.minimizer(result)
    V = se ? _vcov_from_hessian(ForwardDiff.hessian(nll, theta_hat)) :
        fill(NaN, length(theta_hat), length(theta_hat))
    blocks = [:mu => 1:pmu, :sigma => (pmu+1):i1, :nu => (i1+1):i2,
              :sd => (i2+1):(i2+psd)]
    names = [:mu => nmmu, :sigma => nmsigma, :nu => nmnu, :sd => nmsd]
    sd_group = exp.(Wsd * theta_hat[(i2+1):(i2+psd)])
    means = Dict(:mu => exp.(Xmu * theta_hat[1:pmu]))
    observed = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xsigma * theta_hat[(pmu+1):i1]),
                  :nu => _logit12.(Xnu * theta_hat[(i1+1):i2]),
                  :sd_group => sd_group, :sd => sd_group[gidx],
                  :sd_group_index => Float64.(gidx), :aghq_nodes => [Float64(K)])
    _warn_at_clamp("Tweedie", ["log sigma" => (Xsigma * theta_hat[(pmu+1):i1], -15.0, 15.0),
                               "power (logit scale)" => (Xnu * theta_hat[(i1+1):i2], -12.0, 12.0)])
    return _withiterations(_withnll(DrmFit(fam, blocks, names, theta_hat, V,
        -nll(theta_hat), n, drm_optim_converged(result), means, observed, scales), nll),
        Optim.iterations(result))
end

function _fit_tweedie(fam::Tweedie, y, Xμ, Xσ, Xν, nmμ, nmσ, nmν, g_tol; offset = nothing)
    n = length(y); pμ, pσ, pν = size(Xμ, 2), size(Xσ, 2), size(Xν, 2)
    i1 = pμ + pσ; i2 = i1 + pν
    w = _ctl_weights(n)                  # prior weights: row log-lik multipliers (#931)
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:i1]; βν = θ[i1+1:i2]
        ημ = clamp.(_eta_with_offset(Xμ, βμ, offset), -30.0, 30.0); ησ = clamp.(Xσ * βσ, -15.0, 15.0)
        ην = clamp.(Xν * βν, -12.0, 12.0)        # keep p strictly inside (1,2) — avoids the boundary singularities
        s = zero(eltype(θ))
        @inbounds for i in 1:n
            μ = exp(ημ[i]); φ = exp(2 * ησ[i]); p = _logit12(ην[i])   # φ = σ²
            s -= w[i] * (_logpdf_tweedie(y[i], μ, φ, p))
        end
        return s
    end
    ȳ = sum(y) / n
    θ0 = zeros(i2)
    θ0[1] = log(ȳ + eps())            # log mean
    θ0[pμ+1] = 0.0                     # σ = 1 (φ = 1)
    θ0[i1+1] = 0.0                     # p = 1.5
    θ0[1:pμ] .= _offset_start(θ0[1:pμ], Xμ, offset)   # #938
    res = _ctl_optimize((x0, opts, method) -> Optim.optimize(nll, x0, method, opts; autodiff = :forward),
                       θ0; g_tol = g_tol)   # control: maxit / start / multi_start (#948 #994)
    θ̂ = Optim.minimizer(res); V = _ctl_se(() -> _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂)), θ̂)
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):i1, :nu => (i1+1):i2]
    names = [:mu => nmμ, :sigma => nmσ, :nu => nmν]
    means = Dict(:mu => exp.(_eta_with_offset(Xμ, θ̂[1:pμ], offset))); obs = Dict(:mu => Vector{Float64}(y))   # response-scale μ̂
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):i1]),
                  :nu => _logit12.(Xν * θ̂[(i1+1):i2]))
    _warn_at_clamp("Tweedie", ["log sigma" => (Xσ * θ̂[(pμ+1):i1], -15.0, 15.0),   # #959
                               "power (logit scale)" => (Xν * θ̂[(i1+1):i2], -12.0, 12.0)])
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, drm_optim_converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# Default nodes for Tweedie's two 1-D `(1|g)`/`(0+x|g)` mu-ranef routes.
# Adversarial review of the AGHQ switch (following #877's precedent of a
# family-specific K constant, `_BINOMIAL_RANEF_AGHQ_K`) found the generic
# `_RANEF1D_AGHQ_K = 5` default still 0.19 nat off on a zero-heavy,
# informative-group regime: swept K ∈ {5, 9, 15, 21} on the review's
# zero-heavy DGP (G=60, m=5, σ_b≈3.2, β0=−1.5, 203/300 zeros) and on an
# ordinary DGP (G=40, m=15, σ_b=3), logLik error at θ̂ vs an independent
# QuadGK reference: K=5 −0.187/−1.9e-4 nat, K=9 0.039/−2.5e-4, K=15
# −0.0037/1.1e-5, K=21 6.7e-4/1.3e-7. K=21 is the smallest of the four that
# lands within 1e-3 nat on BOTH; cost is 2.64× the pre-AGHQ 32-node grid on
# the ordinary DGP and 1.99× on the zero-heavy one (vs 2.2×/1.7× at K=5) —
# a reasonable increment for a route where fits already run in ~1-5 s. See
# test/test_tweedie_aghq_k.jl for the sweep.
const _TWEEDIE_RANEF_AGHQ_K = 21

# Tweedie compound Poisson–Gamma GLMM with a random intercept (1|g) on the log
# mean. b_g ~ N(0,σ_b²) integrated out per group by per-group adaptive
# Gauss–Hermite quadrature (`src/adaptive_ghq.jl`, #719/#834) — the same scheme
# as the Poisson/Gamma `(1 | g)` routes (src/poisson.jl `_fit_poisson_ranef`,
# src/gamma.jl `_fit_gamma_ranef`). `sigma`/`nu` stay ordinary fixed-effect
# sub-models (dispersion φ and power p are not group-varying); only `mu`'s
# intercept carries the random term. #563.
function _fit_tweedie_ranef(fam::Tweedie, y, Xμ, Xσ, Xν, gidx, G, nmμ, nmσ, nmν, grp, g_tol; K::Int = _TWEEDIE_RANEF_AGHQ_K)
    n = length(y); pμ, pσ, pν = size(Xμ, 2), size(Xσ, 2), size(Xν, 2)
    i1 = pμ + pσ; i2 = i1 + pν
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    rule = _AGHQRule(1, K); Zre = ones(n, 1); bcache = zeros(1, G)   # #719: per-group AGHQ
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:i1]; βν = θ[i1+1:i2]; σb = exp(θ[i2+1])
        η0 = clamp.(Xμ * βμ, -30.0, 30.0)
        ησ = clamp.(Xσ * βσ, -15.0, 15.0)
        ην = clamp.(Xν * βν, -12.0, 12.0)
        ll = (i, η) -> (μ = exp(clamp(η, -30.0, 30.0)); φ = exp(2 * ησ[i]); p = _logit12(ην[i]);
                        _logpdf_tweedie(y[i], μ, φ, p))
        L = reshape([σb], 1, 1)
        return -_aghq_marginal_loglik(ll, members, η0, Zre, L, rule, bcache)
    end
    ȳ = sum(y) / n
    θ0 = zeros(i2 + 1)
    θ0[1] = log(ȳ + eps())            # log mean
    θ0[pμ+1] = 0.0                     # σ = 1 (φ = 1)
    θ0[i1+1] = 0.0                     # p = 1.5
    θ0[i2+1] = log(0.5)                # σ_b
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):i1, :nu => (i1+1):i2, :resd => (i2+1):(i2+1)]
    names = [:mu => nmμ, :sigma => nmσ, :nu => nmν, :resd => [String(grp)]]
    means = Dict(:mu => exp.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):i1]),
                  :nu => _logit12.(Xν * θ̂[(i1+1):i2]))
    _warn_at_clamp("Tweedie", ["log sigma" => (Xσ * θ̂[(pμ+1):i1], -15.0, 15.0),   # #959
                               "power (logit scale)" => (Xν * θ̂[(i1+1):i2], -12.0, 12.0)])
    _record_cond_fit_aghq!(scales, nll, θ̂, bcache, gidx, Zre)   # fitted-row modes for `fitted(fit)`
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, drm_optim_converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# Tweedie compound Poisson–Gamma GLMM with an INDEPENDENT random slope
# (0 + x | g) on the log mean: η_i = Xμβ_μ + b_g·x_i, b_g ~ N(0, σ_b²),
# integrated out per group by per-group adaptive Gauss–Hermite quadrature
# (`src/adaptive_ghq.jl`, #719/#834). Same scheme as `_fit_tweedie_ranef` (the
# ordinary random intercept), with the random-effect design column set to
# x_i (Zre) instead of the flat 1 used by the intercept route. Matches
# drmTMB's independent-slope route for tweedie() mu (`(0 + x | id)`, distinct
# from the still-unimplemented correlated `(1 + x | id)`). #563 S8.
function _fit_tweedie_slope_ranef(fam::Tweedie, y, Xμ, Xσ, Xν, xs, gidx, G, nmμ, nmσ, nmν, grp, g_tol; K::Int = _TWEEDIE_RANEF_AGHQ_K)
    n = length(y); pμ, pσ, pν = size(Xμ, 2), size(Xσ, 2), size(Xν, 2)
    i1 = pμ + pσ; i2 = i1 + pν
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    rule = _AGHQRule(1, K); Zre = reshape(xs, n, 1); bcache = zeros(1, G)   # #719: per-group AGHQ
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:i1]; βν = θ[i1+1:i2]; σb = exp(θ[i2+1])
        η0 = clamp.(Xμ * βμ, -30.0, 30.0)
        ησ = clamp.(Xσ * βσ, -15.0, 15.0)
        ην = clamp.(Xν * βν, -12.0, 12.0)
        ll = (i, η) -> (μ = exp(clamp(η, -30.0, 30.0)); φ = exp(2 * ησ[i]); p = _logit12(ην[i]);
                        _logpdf_tweedie(y[i], μ, φ, p))
        L = reshape([σb], 1, 1)
        return -_aghq_marginal_loglik(ll, members, η0, Zre, L, rule, bcache)
    end
    ȳ = sum(y) / n
    θ0 = zeros(i2 + 1)
    θ0[1] = log(ȳ + eps())            # log mean
    θ0[pμ+1] = 0.0                     # σ = 1 (φ = 1)
    θ0[i1+1] = 0.0                     # p = 1.5
    θ0[i2+1] = log(0.5)                # σ_b
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):i1, :nu => (i1+1):i2, :resd => (i2+1):(i2+1)]
    names = [:mu => nmμ, :sigma => nmσ, :nu => nmν, :resd => [String(grp)]]
    means = Dict(:mu => exp.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):i1]),
                  :nu => _logit12.(Xν * θ̂[(i1+1):i2]))
    _warn_at_clamp("Tweedie", ["log sigma" => (Xσ * θ̂[(pμ+1):i1], -15.0, 15.0),   # #959
                               "power (logit scale)" => (Xν * θ̂[(i1+1):i2], -12.0, 12.0)])
    _record_cond_fit_aghq!(scales, nll, θ̂, bcache, gidx, Zre)   # fitted-row modes for `fitted(fit)`
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, drm_optim_converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# ---------------------------------------------------------------------------
# Crossed/multiple random intercepts (1|g)+(1|h) on the log mean (#737),
# reusing the sparse augmented-state Laplace GLMM engine already shared by
# Gamma/NB2/Beta's crossed routes (src/sparse_laplace_glmm.jl:
# `_fit_crossed_mean_laplace_nuisance`, `_crossed_mean_mode`) — NOT a new
# integrator. `sigma ~ 1` and `nu ~ 1` stay constant fixed-effect sub-models
# (mirrors Gamma's constant-`sigma` restriction on its crossed route).
#
# The shared engine's "nuisance" slot is exactly one scalar dispersion
# parameter (`_fit_crossed_mean_laplace_nuisance`'s `θσ`). The Tweedie power
# `p` is a SECOND nuisance scalar the shared engine has no room for (its θ
# layout is hardcoded to `[βμ; θσ; logσ_g; logσ_h]`), and widening that layout
# is a shared-file change out of this family file's ownership. Instead `p` is
# PROFILED OUT: for each candidate `p`, `inner_fit` refits (β, dispersion,
# σ_g, σ_h) to convergence with the existing engine, folding `p` into the
# dispersion-only `aux_from` closure; an outer 1-D Brent search over the
# logit-(1,2) parameterisation finds the `p` minimising that profile
# deviance. This is exact profile-likelihood optimisation (the profile MLE
# equals the joint MLE at the joint optimum) — no approximation is added
# beyond the crossed-Laplace machinery already in use.
#
# `_logpdf_tweedie`'s Dunn–Smyth series has no convenient closed-form
# derivative, so the per-observation value/derivatives the shared engine
# needs (`_laplace_value`/`_d1`/`_d2`/`_d3`, and the nuisance-parameter
# counterparts) are obtained by nested `ForwardDiff.derivative` on `η` (and on
# `logsigma` for the nuisance slot) rather than hand-derived analytically.
# Verified against central finite differences (interactive check, not part of
# the test suite): a single-observation d1/d2/d3 and nuisance-value/d1/d2
# probe matched 5-point-stencil finite differences to ~1e-4–1e-6 relative,
# including at `y = 0` (the exact-zero branch of `_logpdf_tweedie`, which is
# smooth in `η` for a FIXED `y`, so nested autodiff carries through it fine).

# Per-observation NLL contribution as an explicit function of BOTH the linear
# predictor η and the nuisance log-scale `logsigma` (dispersion φ = exp(2·logsigma)),
# so `_laplace_d3` (pure η) and the mixed/pure `logsigma` nuisance derivatives
# below can all be built by nesting `ForwardDiff.derivative` on this one
# closure, at whichever variable is being differentiated.
@inline function _tweedie_v_of(y::Real, η, logsigma, p::Real)
    μ = exp(clamp(η, -30.0, 30.0))
    φ = exp(2 * clamp(logsigma, -8.0, 8.0))
    return -_logpdf_tweedie(y, μ, φ, p)
end

_laplace_value(::Val{:tweedie_fixed}, aux, i, η) =
    _tweedie_v_of(aux.y[i], η, aux.logsigma, aux.p)

_laplace_d1(::Val{:tweedie_fixed}, aux, i, η) =
    ForwardDiff.derivative(x -> _tweedie_v_of(aux.y[i], x, aux.logsigma, aux.p), η)

_laplace_d2(::Val{:tweedie_fixed}, aux, i, η) =
    ForwardDiff.derivative(x -> _laplace_d1(Val(:tweedie_fixed), aux, i, x), η)

_laplace_d3(::Val{:tweedie_fixed}, aux, i, η) =
    ForwardDiff.derivative(x -> _laplace_d2(Val(:tweedie_fixed), aux, i, x), η)

_laplace_mean(::Val{:tweedie_fixed}, η) = exp(clamp(η, -30.0, 30.0))
_laplace_obs(::Val{:tweedie_fixed}, aux, i) = aux.y[i]

# d/d(logsigma) of the per-observation NLL contribution, at fixed η.
_laplace_nuisance_value(::Val{:tweedie_fixed}, aux, i, η) =
    ForwardDiff.derivative(ls -> _tweedie_v_of(aux.y[i], η, ls, aux.p), aux.logsigma)

# mixed partial d²/(dη d(logsigma)) — matches `_laplace_nuisance_d1` for
# `:gamma_fixed`/`:nb2_fixed`/`:beta_fixed` (verified against gamma_fixed's
# hand-derived `-2α·(1 - y/μ)` reading, which is `∂(nuisance_value)/∂η`).
_laplace_nuisance_d1(::Val{:tweedie_fixed}, aux, i, η) =
    ForwardDiff.derivative(x -> _laplace_nuisance_value(Val(:tweedie_fixed), aux, i, x), η)

# `_laplace_nuisance_d2` is NOT the pure d²v/d(logsigma)² — it is `∂(d2)/∂(logsigma)`,
# i.e. d³v/(dη² d(logsigma)): how the Hessian diagonal entry `w = d2` itself
# shifts with the nuisance parameter at FIXED η. This is what the crossed
# engine's `_crossed_mean_laplace_nuisance_fg` needs for the log-determinant's
# direct (not mode-implicit) dependence on the nuisance parameter — confirmed
# against `:gamma_fixed`'s hand-derived reading, `_laplace_nuisance_d2 =
# -2α·y/μ = -2·_laplace_d2` (since `α = exp(-2·logsigma)`, so
# `∂(αy/μ)/∂(logsigma) = -2α·y/μ`), i.e. `∂(d2)/∂(logsigma)`, not `∂²v/∂s²`.
# Getting this backwards (as a first pass here did) leaves every OTHER
# gradient component (β, RE-variance SDs) exactly right — because those flow
# entirely through η — while only the dispersion/nuisance component of the
# analytic gradient is wrong; caught by an off-optimum finite-difference gate
# on that one component (test/test_crossed_tweedie.jl).
_laplace_nuisance_d2(::Val{:tweedie_fixed}, aux, i, η) =
    ForwardDiff.derivative(ls -> _laplace_d2(Val(:tweedie_fixed), (y = aux.y, p = aux.p, logsigma = ls), i, η),
                           aux.logsigma)

function _fit_tweedie_crossed_laplace(fam::Tweedie, y, Xμ, comps, nmμ, nmσ, nmν, g_tol)
    length(comps) == 2 || error("_fit_tweedie_crossed_laplace requires two random-intercept components")
    yv = Float64.(y)
    n = length(yv)
    ȳ = sum(yv) / n
    θβ0 = zeros(size(Xμ, 2))
    θβ0[1] = log(ȳ + eps())

    # The default first LBFGS trial step is the full, unscaled -gradient. From the
    # cold start the gradient is O(100-300) in the dispersion and log-SD
    # coordinates, so the first line-search probes land at wild parameters
    # (dispersion ~e^16, SDs at their clamps) where the Dunn-Smyth window runs to
    # ~1e5 terms and the inner Laplace mode solve burns its 60 iterations and
    # fails (~65 s each when measured). Every one of the ~25-40 Brent evaluations
    # repeated this. Capping the trial step norm at 1 removes the blow-up and
    # leaves the optimum unchanged (same LBFGS, same tolerance, same objective).
    function inner_fit(p::Real; polish_iterations::Int = 0)
        aux_from(logsigma) = (y = yv, p = p, logsigma = logsigma)
        return _fit_crossed_mean_laplace_nuisance(
            fam, Val(:tweedie_fixed), aux_from, n, Xμ,
            comps[1][2], comps[1][3], comps[2][2], comps[2][3], nmμ, nmσ,
            [comps[1][4], comps[2][4]], g_tol;
            θβ0 = θβ0, θσ0 = 0.0, sigma_scale = exp, se = false,
            alphaguess = Optim.LineSearches.InitialStatic(alpha = 1.0, scaled = true),
            polish_iterations = polish_iterations)
    end

    obj(ηp) = -inner_fit(_logit12(ηp)).loglik
    res = Optim.optimize(obj, -6.0, 6.0, Optim.Brent())
    p̂ = _logit12(Optim.minimizer(res))
    fit = inner_fit(p̂; polish_iterations = 25)

    pμ = size(Xμ, 2)
    nblk = pμ + 1 + 2   # mu + sigma(dispersion) + 2 crossed RE sds, the engine's own blocks
    blocks = vcat(fit.blocks, [:nu => (nblk + 1):(nblk + 1)])
    names = vcat(fit.coefnames, [:nu => nmν])
    theta = vcat(fit.theta, [Optim.minimizer(res)])
    V = fill(NaN, length(theta), length(theta))
    V[1:nblk, 1:nblk] .= fit.vcov
    scales = merge(fit.scales, Dict(:nu => fill(p̂, n)))
    return DrmFit(fam, blocks, names, theta, V, fit.loglik, fit.nobs, fit.converged,
                 fit.means, fit.obs, scales)
end
