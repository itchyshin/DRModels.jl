# student.jl — Student-t family: robust location–scale–shape regression.
# A formula per parameter: μ (location, identity), σ (scale, log link), ν
# (degrees of freedom, ν = 2 + exp(η) → ν > 2, finite variance). Location-scale t:
# logpdf = logpdf(TDist(ν), (y-μ)/σ) − log σ. Heavy tails downweight outliers;
# ν → ∞ recovers Gaussian. Fixed effects (ML), plus a random intercept `(1|g)` or
# a correlated random intercept+slope `(1+x|g)` on the mean μ (integrated out by
# Gauss–Hermite quadrature; σ and ν stay fixed). Mirrors drmTMB's `student`.

using Distributions: TDist, logpdf

# Standardised Student-t log-density log f_t(z; ν) with ν = 2 + exp(ην), stable for
# all ην (#721). `Distributions.TDist` evaluates loggamma((ν+1)/2) − loggamma(ν/2)
# − ½log(νπ) as a difference of numbers of size ~ν·log ν, which cancels
# catastrophically for large ν (at ν ≈ 1e16 it is off by ~31 nats), so an optimiser
# on the flat near-Gaussian ν ridge could walk into a spurious region and report a
# garbage loglik. For ν > ~1100 (ην > 7) we use the asymptotic expansion
#   loggamma(x+½) − loggamma(x) = ½log x − 1/(8x) + 1/(192x³) + O(x⁻⁵),  x = ν/2,
# written in r = 1/ν (computed without overflow), so the constant is
# −½log(2π) − r/4 + r³/24 (truncation error < 1e-16 here), and the kernel
# (ν+1)/2·log1p(z²/ν) = (1+r)/2 · z² · log1p(u)/u with u = z²r. As ην → ∞ this
# tends exactly to the Normal logpdf. Below the switch, TDist is accurate (< 1e-12).
function _student_logpdf_std(z, ην)
    ην > 7 || return logpdf(TDist(2 + exp(ην)), z)
    e = exp(-ην); r = e / (1 + 2e)                  # r = 1/ν, → 0 without overflow
    u = z^2 * r
    L = u < 1e-4 ? 1 - u / 2 + u^2 / 3 - u^3 / 4 : log1p(u) / u   # log1p(u)/u
    return -0.5 * log(2π) - r / 4 + r^3 / 24 - (1 + r) / 2 * z^2 * L
end

"""
    Student()

Student-t response family: identity link on the location `μ`, log link on the
scale `σ`, and the degrees of freedom as `ν = 2 + exp(η)` (so `ν > 2` and the
variance is always finite; `ν` coefficients act on `log(ν − 2)`). Note that `σ` is
the scale, not the standard deviation: for `ν > 2`, `SD[y] = σ·sqrt(ν/(ν − 2))`
(mirroring `drmTMB`). Robust sibling of [`Gaussian`](@ref) — heavy tails downweight
outliers, and `ν → ∞` tends to Gaussian. Mirrors `drmTMB`'s `student` family.

```julia
fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)), Student(); data = dat)
2 + exp(coef(fit, :nu)[1])  # estimated degrees of freedom (ν = 2 + exp(η))
predict_parameters(fit, dat)[:nu]   # same quantity (response-scale ν, not exp(η))

# Opt-in TMB-convention Laplace for one ordinary `(1 | g)` on the mean (what drmTMB fits):
fit_lap = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Student(); data = dat,
              marginal = :Laplace)

# A group-level covariate can model an ordinary mean intercept's SD:
# `dat` must include numeric `z` values that are constant within each `g` group.
fit_sd = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1),
                   @formula(nu ~ 1), @formula(sd(g) ~ z)), Student(); data = dat)
# It can also model the coefficient SD of one independent numeric mean slope:
fit_slope_sd = drm(bf(@formula(y ~ x + (0 + x | g)), @formula(sigma ~ 1),
                           @formula(nu ~ 1), @formula(sd(g, coef(x)) ~ z)),
                       Student(); data = dat)

# Independent random intercept and slope on one factor (#960, as drmTMB fits it; Laplace):
fit_ind = drm(bf(@formula(y ~ x + (1 | g) + (0 + x | g)), @formula(sigma ~ 1)), Student(); data = dat)
exp.(coef(fit_ind, :resd))         # SDs of the intercept and slope effects
```

Student-t SD regression currently supports one ordinary mean intercept or one
independent numeric mean slope, after the family's existing observed-response
filtering, with predictors that are constant within each retained group. For
multiple possible coefficients, use the matching `coef(intercept)` or
`coef(x)` selector. A slope formula models the SD of its random coefficient;
the observation-level contribution also depends on the slope predictor. This
route requires fixed `sigma ~ 1` and `nu ~ 1` submodels. It uses adaptive
Gauss–Hermite quadrature (`marginal = :LA`); correlated blocks, structured
effects, and `marginal = :Laplace` are not admitted on this route yet.
"""
struct Student end

# FX3b: `se = false` is recorded on the fit (`DrmFit.uncertainty`), so
# `convergence_status` does not read the skipped (NaN) covariance as a failure.
@drm_entry function drm(f::DrmFormula, fam::Student; se::Bool = true, kwargs...)
    return _warn_student_nu_boundary(_withuncertainty(_drm_student_fit(f, fam; se = se, kwargs...), se))
end

# #969: Student ν at the Gaussian limit. When the data have no heavy tails the ML
# estimate runs to ν → ∞ (1/ν → 0): the likelihood is flat in the ν linear predictor,
# the Hessian is singular there, and the pseudo-inverse can hand back a variance of
# ~0, which prints as SE(ν) = 0 and a zero-width Wald interval. FLAG ONLY: `vcov`,
# `stderror`, `converged` and `convergence_status` are NOT changed (how the ν SE
# should be reported, and whether such a fit counts as `:boundary`, is an owner
# decision; see #969). `1/ν < 1e-3` is the cut the issue's harness uses.
const _STUDENT_NU_BOUNDARY_INV = 1e-3

# Largest fitted ν over the rows, or `nothing` if the fit has no finite ν.
function _student_nu_boundary(fit::DrmFit)
    fit.family isa Student || return nothing
    nu = get(fit.scales, :nu, nothing)
    nu === nothing && return nothing
    v = filter(isfinite, collect(Float64, nu))
    isempty(v) && return nothing
    νmax = maximum(v)
    return 1 / νmax < _STUDENT_NU_BOUNDARY_INV ? νmax : nothing
end

function _warn_student_nu_boundary(fit)
    fit isa DrmFit || return fit
    νmax = _student_nu_boundary(fit)
    νmax === nothing && return fit
    get(task_local_storage(), :drm_quiet_boundary, false) === true && return fit
    @warn "drm: Student() degrees of freedom ν reached the Gaussian limit (fitted ν up to " *
          "$(round(νmax; sigdigits = 3)), 1/ν < $(_STUDENT_NU_BOUNDARY_INV)): boundary (limit). " *
          "The likelihood is flat in the `nu` coefficients, so their estimates are an arbitrary " *
          "optimiser stop and their Wald SE / interval are NOT valid (a reported SE of 0 means " *
          "'not computable', not 'precise'). The Gaussian family is the appropriate model; " *
          "compare it by likelihood ratio with a 0.5χ²₀ + 0.5χ²₁ boundary mixture (DRModels.jl #969). " *
          "`stderror`, `vcov` and `convergence_status` are unchanged."
    return fit
end

function _drm_student_fit(f::DrmFormula, fam::Student; data, g_tol::Real = 1e-8,
             se::Bool = true, marginal::Symbol = :LA)
    _drm_entry_guard(f, fam, data)   # #1027 #1022
    # `marginal = :Laplace` (opt-in, ordinary `(1 | g)` on the mean): the TMB-convention
    # Laplace integrator drmTMB uses; the default `:LA` is unchanged (adaptive GHQ).
    laplace = _scalar_laplace_requested(marginal)
    laplace || marginal === :LA || throw(ArgumentError(
        "drm (Student): marginal = :$marginal is not available; use `:LA` (default) " *
        "or `:Laplace` (ordinary `(1 | g)` on the mean)."))
    missing_fit = _fit_observed_response_rows(f, data) do data_observed
        drm(f, fam; data = data_observed, g_tol = g_tol, se = se, marginal = marginal)
    end
    missing_fit !== nothing && return missing_fit

    if !isempty(_sd_target_parts(f)) || !isempty(_sdphylo_parts(f))
        return _drm_student_sd_regression(f, fam; data, g_tol, se, marginal)
    end
    _lss_only_gaussian_guard(f, fam)   # #544: refuse, never silently drop, sd() parts
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    fixed_mu, offset_term = _extract_offset(fixed_mu)   # #938: offset(z), coefficient fixed at 1
    offset_term === nothing || (isempty(re) && mv === nothing && st === nothing) ||
        _refuse_offset_route(fam, "a random, structured or `meta_V` term on the mean")
    (mv === nothing && st === nothing) ||
        error("Student() does not support meta_V / structured markers")
    for (pname, r) in f.forms          # only the mean may carry a random effect
        pname === :mu && continue
        _, re2, mv2, st2 = _split_ranef(r)
        (isempty(re2) && mv2 === nothing && st2 === nothing) ||
            error("Student(): only the mean formula may carry a random effect")
    end
    y, Xμ, nmμ = _design(f.response, fixed_mu, data)
    _, Xσ, nmσ = _design(f.response, get(rhs, :sigma, ConstantTerm(1)), data)
    _, Xν, nmν = _design(f.response, get(rhs, :nu, ConstantTerm(1)), data)
    if laplace && !(length(re) == 1 && _re_kind(re[1][1])[1] in (:intercept, :slope))
        throw(ArgumentError("marginal = :Laplace is not available for Student() with " *
            (isempty(re) ? "no random effect (fixed-effects-only)" :
             length(re) > 1 ? "crossed/multiple random effects (already fitted by Laplace by default)" :
             "a correlated block `(1 + x | g)`") *
            ". This route covers exactly one ordinary random intercept `(1 | g)` or independent slope `(0 + x | g)` on the mean; " *
            "omit `marginal` (the default `:LA`) for other models."))
    end
    if length(re) > 1                                     # crossed intercepts → Laplace (#725)
        # Independent intercepts + slopes `(1 | g) + (0 + x | g)` (#960): TMB-convention
        # Laplace, as drmTMB fits them.
        slope_comps = _indep_slope_comps(re, data, length(y))
        slope_comps === nothing ||
            return _withformula(_fit_indep_re_laplace(fam, Val(:student), y, Xμ, Xσ, Xν,
                                                      slope_comps, nmμ, nmσ, nmν, g_tol), f)
        (length(re) == 2 && all(_re_kind(r[1])[1] === :intercept for r in re)) ||
            error("Student() supports multiple random effects on the mean only as two crossed/nested " *
                  "intercepts `(1 | g) + (1 | h)` or independent intercepts and slopes " *
                  "`(1 | g) + (0 + x | g)`")
        comps = map(re) do r
            gidx, G = _group_index(getproperty(data, r[2]); col = r[2])
            (gidx, G, String(r[2]))
        end
        return _withformula(_fit_student_crossed_laplace(fam, y, Xμ, Xσ, Xν, comps, nmμ, nmσ, nmν, g_tol), f)
    end
    if !isempty(re)                                       # random effect on the mean → GHQ
        length(re) == 1 || error("Student() supports a single random-effect term on the mean")
        (rk, var) = _re_kind(re[1][1]); grp = re[1][2]; gidx, G = _group_index(getproperty(data, grp); col = grp)
        if rk === :intercept                              # (1 | g) → 1-D GHQ (or opt-in Laplace)
            laplace && return _withformula(_withmarginal(_fit_student_ordinary_laplace(
                fam, y, Xμ, Xσ, Xν, gidx, G, nmμ, nmσ, nmν, grp, g_tol; se = se), :Laplace), f)
            return _withformula(_fit_student_ranef(fam, y, Xμ, Xσ, Xν, gidx, G, nmμ, nmσ, nmν, grp, g_tol), f)
        elseif rk === :corr                               # (1 + x | g) → 2-D GHQ
            xs = Float64.(getproperty(data, var))
            return _withformula(_fit_student_corr_ranef(fam, y, Xμ, Xσ, Xν, xs, gidx, G, nmμ, nmσ, nmν, grp, g_tol), f)
        elseif rk === :slope                              # (0 + x | g) → TMB-convention Laplace (independent route)
            _require_finite_data_column(data, var)
            return _withformula(_fit_indep_re_laplace(fam, Val(:student), y, Xμ, Xσ, Xν,
                                                      _indep_slope_comps(re, data, length(y)),
                                                      nmμ, nmσ, nmν, g_tol), f)
        else
            error("Student() supports `(1 | g)`, `(0 + x | g)` or `(1 + x | g)` random effects on the mean")
        end
    end
    return _withformula(_fit_student(fam, y, Xμ, Xσ, Xν, nmμ, nmσ, nmν, g_tol; offset = _offset_for_fit(fam, offset_term, data, size(Xμ, 1))), f)
end

# Crossed intercepts `(1 | g) + (1 | h)` with one `sd(g) ~ …` and one `sd(h) ~ …` formula: the
# sparse-Laplace crossed Student fit with a log-SD regression per factor.
function _drm_student_crossed_sd_regression(f::DrmFormula, fam::Student, targets; data, g_tol)
    who = "Student crossed SD regression"
    all(t -> t.target.level === :iid && t.target.dpar === :mu && t.target.block === nothing &&
             t.target.member in (nothing, :intercept), targets) || throw(ArgumentError(
        "$who targets the ordinary mean intercepts only."))
    isempty(_sdphylo_parts(f)) || throw(ArgumentError("$who does not support structured SD formulas."))
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    (mv === nothing && st === nothing && length(re) == 2 &&
     all(r -> _re_kind(r[1])[1] === :intercept, re) && re[1][2] !== re[2][2]) || throw(ArgumentError(
        "$who requires exactly two crossed intercepts `(1 | g) + (1 | h)` on the mean, " *
        "each with its own `sd(group) ~ …` formula."))
    ts = map(re) do r
        ms = filter(t -> t.target.group === r[2], targets)
        length(ms) == 1 || throw(ArgumentError(
            "$who needs exactly one `sd($(r[2])) ~ …` formula per grouping factor."))
        only(ms)
    end
    for (pname, form) in f.forms
        (pname === :mu || pname in (ts[1].key, ts[2].key)) && continue
        pname in (:sigma, :nu) && form isa ConstantTerm && form == ConstantTerm(1) && continue
        throw(ArgumentError("$who supports only `mu`, fixed `sigma ~ 1`, fixed `nu ~ 1` and its two SD formulas."))
    end
    fixed_mu, offset_term = _extract_offset(fixed_mu)
    offset_term === nothing || throw(ArgumentError("$who does not support `offset(...)`."))
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "$who requires observed responses; remove incomplete rows before fitting."))
    y, Xmu, nmmu = _design(f.response, fixed_mu, data)
    _, Xsigma, nmsigma = _design(f.response, ConstantTerm(1), data)
    _, Xnu, nmnu = _design(f.response, ConstantTerm(1), data)
    comps = map(re) do r
        gidx, G = _group_index(getproperty(data, r[2]); col = r[2])
        (gidx, G, String(r[2]))
    end
    designs = map(zip(re, ts, comps)) do (r, t, c)
        Zg, nm = _sd_group_design(f.response, t.rhs, data, c[1], c[2], r[2];
                                  schema_cache = f.schema_cache, schema_key = t.key)
        size(Zg, 2) <= c[2] && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
            "$who: the `sd($(r[2]))` group-level design is rank deficient; use predictors that vary independently across groups."))
        (Zg, nm)
    end
    blocks = [_sd_target_block(t.target; disambiguate = true) for t in ts]
    fit = _fit_student_crossed_laplace(fam, y, Xmu, Xsigma, Xnu, Tuple(comps), nmmu, nmsigma, nmnu, g_tol;
        sd_designs = (designs[1][1], designs[2][1], [designs[1][2], designs[2][2]], blocks))
    return _withformula(fit, f)
end

# Student SD regression: one ordinary intercept or one independent numeric
# slope on μ, with a group-constant fixed-effect design for log(SD). The groupwise AGHQ integral
# uses the same Student scale σ and ν = 2 + exp(η) parameterizations as the
# ordinary Student random-effect fit. Correlated and structured effects remain
# explicit refusals until their Student likelihoods are reviewed.
function _drm_student_sd_regression(f::DrmFormula, fam::Student; data,
                                    g_tol::Real, se::Bool, marginal::Symbol)
    laplace_fit = _laplace_sd_route(f, fam, Val(:student), marginal; data, g_tol)
    laplace_fit === nothing || return laplace_fit
    marginal === :LA || throw(ArgumentError(
        "Student `sd(group) ~ …` currently supports `marginal = :LA` only."))
    isempty(_sdphylo_parts(f)) || throw(ArgumentError(
        "Student `sd(group, phylogenetic) ~ …` is not implemented by the ordinary random-intercept route."))
    targets = _sd_target_parts(f)
    # Crossed random intercepts `(1 | g) + (1 | h)` with one SD formula each keep the route
    # checked against drmTMB (`test_struct_sd_counts.jl`); every other independent-term shape
    # goes to the shared independent-term Laplace route.
    indep_shape = _indep_sd_shape(f)
    crossed_pair = indep_shape !== nothing && length(targets) == 2 &&
        length(indep_shape.re) == 2 && all(k -> k[1] === :intercept, indep_shape.kinds)
    crossed_pair && return _drm_student_crossed_sd_regression(f, fam, targets; data, g_tol)
    indep_shape === nothing ||
        return _drm_indep_sd_laplace(f, fam, Val(:student), indep_shape; data, g_tol)
    length(targets) == 2 && return _drm_student_crossed_sd_regression(f, fam, targets; data, g_tol)
    length(targets) == 1 || throw(ArgumentError(
        "Student random-effect SD regression currently requires exactly one `sd(group) ~ …` formula, " *
        "or one per factor of crossed intercepts `(1 | g) + (1 | h)`."))
    target = only(targets)
    (target.target.level === :iid && target.target.dpar === :mu &&
     target.target.block === nothing) ||
        throw(ArgumentError("Student random-effect SD regression currently targets only an ordinary mean intercept or independent slope on `mu`."))
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    (mv === nothing && st === nothing) || throw(ArgumentError(
        "Student `sd($(target.target.group))` currently supports one ordinary intercept or independent numeric slope only."))
    length(re) == 1 || throw(ArgumentError(
        "Student random-effect SD regression currently supports exactly one ordinary random-effect coefficient."))
    effect_kind, effect_term = _re_kind(re[1][1])
    effect_member = if effect_kind === :intercept
        :intercept
    elseif effect_kind === :slope
        effect_term
    else
        throw(ArgumentError(
            "Student SD regression supports `(1 | group)` or one independent numeric `(0 + x | group)` term; correlated blocks are not admitted."))
    end
    target.target.member in (nothing, effect_member) || throw(ArgumentError(
        "Student SD target `coef($(target.target.member))` does not match the fitted random-effect coefficient `$(effect_member)`."))
    grp = re[1][2]
    target.target.group === grp || throw(ArgumentError(
        "Student `sd($(target.target.group))` must name the random-effect grouping variable `$grp`."))
    sigma_rhs = get(rhs, :sigma, ConstantTerm(1))
    sigma_rhs isa ConstantTerm && sigma_rhs == ConstantTerm(1) || throw(ArgumentError(
        "Student `sd($grp)` currently requires a fixed `sigma ~ 1` submodel."))
    nu_rhs = get(rhs, :nu, ConstantTerm(1))
    nu_rhs isa ConstantTerm && nu_rhs == ConstantTerm(1) || throw(ArgumentError(
        "Student `sd($grp)` currently requires a fixed `nu ~ 1` submodel."))
    for (pname, form) in f.forms
        pname == target.key && continue
        pname === :mu && continue
        _, re2, mv2, st2 = _split_ranef(form)
        (isempty(re2) && mv2 === nothing && st2 === nothing) || throw(ArgumentError(
            "Student(): only the mean formula may carry a random effect."))
        pname in (:sigma, :nu) || throw(ArgumentError(
            "Student `sd($grp)` supports only `mu`, `sigma`, `nu`, and one `sd($grp)` formula."))
    end
    y, Xmu, nmmu = _design(f.response, fixed_mu, data)
    _, Xsigma, nmsigma = _design(f.response, sigma_rhs, data)
    _, Xnu, nmnu = _design(f.response, nu_rhs, data)
    gidx, G = _group_index(getproperty(data, grp))
    Zre = if effect_kind === :intercept
        ones(length(y), 1)
    else
        hasproperty(data, effect_term) || throw(ArgumentError(
            "Student random-slope predictor `$(effect_term)` is absent from data."))
        values = getproperty(data, effect_term)
        values isa AbstractVector{<:Real} || throw(ArgumentError(
            "Student random-slope predictor `$(effect_term)` must be numeric."))
        length(values) == length(y) && all(isfinite, values) || throw(ArgumentError(
            "Student random-slope predictor `$(effect_term)` must contain one finite value per retained response."))
        reshape(Float64.(values), :, 1)
    end
    Zg, nmsd = _sd_group_design(f.response, target.rhs, data, gidx, G, grp;
                                schema_cache = f.schema_cache, schema_key = target.key)
    size(Zg, 2) <= G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
        "Student `sd($grp)` group-level design is rank deficient; use predictors that vary independently across groups."))
    return _withformula(_fit_student_sd_regression(fam, y, Xmu, Xsigma, Xnu, Zre, Zg,
        gidx, G, nmmu, nmsigma, nmnu, nmsd, grp, g_tol; se), f)
end

function _fit_student_sd_regression(fam::Student, y, Xmu, Xsigma, Xnu, Zre, Zg,
                                    gidx, G, nmmu, nmsigma, nmnu, nmsd,
                                    grp, g_tol; se::Bool = true,
                                    K::Int = _RANEF1D_AGHQ_K)
    K >= 1 || throw(ArgumentError("nAGQ must be ≥ 1; got $K"))
    n = length(y); pmu, psigma, pnu, psd = size(Xmu, 2), size(Xsigma, 2), size(Xnu, 2), size(Zg, 2)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    rule = _AGHQRule(1, K)
    size(Zre) == (n, 1) || throw(ArgumentError(
        "Student SD regression currently requires exactly one scalar random-effect coefficient."))
    bcache = [zeros(1, G) for _ in 1:Threads.maxthreadid()]
    function nll(theta)
        beta_mu = @view theta[1:pmu]
        beta_sigma = @view theta[(pmu+1):(pmu+psigma)]
        beta_nu = @view theta[(pmu+psigma+1):(pmu+psigma+pnu)]
        alpha = @view theta[(pmu+psigma+pnu+1):(pmu+psigma+pnu+psd)]
        eta_mu = Xmu * beta_mu
        eta_sigma = Xsigma * beta_sigma
        eta_nu = Xnu * beta_nu
        sdgroup = exp.(Zg * alpha)
        cache = bcache[Threads.threadid()]
        ll = (i, eta) -> begin
            z = (y[i] - eta) * exp(-eta_sigma[i])
            _student_logpdf_std(z, eta_nu[i]) - eta_sigma[i]
        end
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
    beta_mu0 = Xmu \ Float64.(y)
    theta0 = zeros(pmu + psigma + pnu + psd)
    theta0[1:pmu] .= beta_mu0
    theta0[pmu+1] = log(std(y - Xmu * beta_mu0) + eps())
    theta0[pmu+psigma+1] = log(10.0)
    theta0[(pmu+psigma+pnu+1):end] .= Zg \ fill(log(0.5), G)
    res = Optim.optimize(nll, theta0, Optim.LBFGS(),
                         Optim.Options(g_tol = g_tol); autodiff = :forward)
    theta_hat = Optim.minimizer(res)
    V, hess_finite = _vcov_from_hessian_checked(ForwardDiff.hessian(nll, theta_hat),
        _all_finite(y, Xmu, Xsigma, Xnu, Zg); context = "Student SD regression")
    blocks = [:mu => 1:pmu, :sigma => (pmu+1):(pmu+psigma),
              :nu => (pmu+psigma+1):(pmu+psigma+pnu),
              :sd => (pmu+psigma+pnu+1):(pmu+psigma+pnu+psd)]
    names = [:mu => nmmu, :sigma => nmsigma, :nu => nmnu, :sd => nmsd]
    sdgroup = exp.(Zg * theta_hat[(pmu+psigma+pnu+1):end])
    means = Dict(:mu => Xmu * theta_hat[1:pmu])
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xsigma * theta_hat[(pmu+1):(pmu+psigma)]),
                  :nu => 2 .+ exp.(Xnu * theta_hat[(pmu+psigma+1):(pmu+psigma+pnu)]),
                  :sd_group => sdgroup, :sd => sdgroup[gidx],
                  :sd_group_index => Float64.(gidx), :aghq_nodes => [Float64(K)])
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, theta_hat, V, -nll(theta_hat), n,
                        drm_optim_converged(res) && hess_finite,
                        means, obs, scales), nll), Optim.iterations(res))
end

# Student-t GLMM with a random intercept (1|g) on the mean μ (identity link).
# b_g ~ N(0,σ_b²) is integrated out per group by 32-node Gauss–Hermite quadrature
# (substitution b = √2 σ_b z, logsumexp over K nodes, normaliser −½ logπ); the
# scale σ and degrees of freedom ν stay fixed effects. Same scheme as the NB2/Gamma
# random-intercept GLMMs. θ = [βμ; βσ; βν; log σ_b]. O(n·K) per eval, differentiable.
function _fit_student_ranef(fam::Student, y, Xμ, Xσ, Xν, gidx, G, nmμ, nmσ, nmν, grp, g_tol; K::Int = _RANEF1D_AGHQ_K)
    n = length(y); pμ, pσ, pν = size(Xμ, 2), size(Xσ, 2), size(Xν, 2)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    rule = _AGHQRule(1, K); Zre = ones(n, 1); bcache = zeros(1, G)   # #719: per-group AGHQ
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; βν = θ[pμ+pσ+1:pμ+pσ+pν]; σb = exp(θ[pμ+pσ+pν+1])
        η0 = Xμ * βμ; ησ = Xσ * βσ; ην = Xν * βν     # μ identity → no exp clamp on the mean
        ll = (i, η) -> (zt = (y[i] - η) * exp(-ησ[i]); _student_logpdf_std(zt, ην[i]) - ησ[i])
        L = reshape([σb], 1, 1)
        return -_aghq_marginal_loglik(ll, members, η0, Zre, L, rule, bcache)
    end
    βμ0 = Xμ \ y
    θ0 = zeros(pμ + pσ + pν + 1)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(std(y - Xμ * βμ0) + eps())        # σ init
    θ0[pμ+pσ+1] = log(10.0)                           # ν init (mildly heavy-tailed)
    θ0[pμ+pσ+pν+1] = log(0.5)                         # σ_b init
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res)
    # #961: a non-finite Hessian on finite data is a degenerate fit, not a crash.
    V, hess_finite = _vcov_from_hessian_checked(ForwardDiff.hessian(nll, θ̂),
                                                _all_finite(y, Xμ, Xσ, Xν); context = "Student")
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :nu => (pμ+pσ+1):(pμ+pσ+pν), :resd => (pμ+pσ+pν+1):(pμ+pσ+pν+1)]
    names = [:mu => nmμ, :sigma => nmσ, :nu => nmν, :resd => [String(grp)]]
    means = Dict(:mu => Xμ * θ̂[1:pμ]); obs = Dict(:mu => Vector{Float64}(y))   # population μ (b=0)
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]),
                  :nu => 2 .+ exp.(Xν * θ̂[(pμ+pσ+1):(pμ+pσ+pν)]))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, drm_optim_converged(res) && hess_finite, means, obs, scales), nll),
        Optim.iterations(res))
end

# Student-t GLMM with a correlated random intercept+slope (1 + x | g) on the mean μ.
# Per group (b0,b1) ~ N(0, Σ); because groups are disjoint the 2-D integral factorises,
# and each group is integrated by per-group ADAPTIVE Gauss–Hermite quadrature
# (`_aghq_marginal_loglik`, #834: nodes b̂_g + √2 C z at each group's mode),
# `nq` nodes per axis. Σ is the log-Cholesky parameterisation L = [exp(a) 0; cc exp(b)]
# (the `vc` convention), so vc(fit) reconstructs Σ = L Lᵀ. The scale σ and df ν stay
# fixed effects. θ = [βμ; βσ; βν; a, b, cc].
function _fit_student_corr_ranef(fam::Student, y, Xμ, Xσ, Xν, xs, gidx, G, nmμ, nmσ, nmν, grp, g_tol; nq::Int = _CORR_RANEF_AGHQ_K)
    n = length(y); pμ, pσ, pν = size(Xμ, 2), size(Xσ, 2), size(Xν, 2)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    rule = _AGHQRule(2, nq); Zre = hcat(ones(n), Float64.(xs)); bcache = zeros(2, G)   # #834: per-group AGHQ
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; βν = θ[pμ+pσ+1:pμ+pσ+pν]
        L = _corr_ranef_L(θ[pμ+pσ+pν+1], θ[pμ+pσ+pν+2], θ[pμ+pσ+pν+3])
        η0 = Xμ * βμ; ησ = Xσ * βσ; ην = Xν * βν
        ll = (i, η) -> (zt = (y[i] - η) * exp(-ησ[i]); _student_logpdf_std(zt, ην[i]) - ησ[i])
        return -_aghq_marginal_loglik(ll, members, η0, Zre, L, rule, bcache)
    end
    βμ0 = Xμ \ y
    θ0 = zeros(pμ + pσ + pν + 3)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(std(y - Xμ * βμ0) + eps())        # σ init
    θ0[pμ+pσ+1] = log(10.0)                           # ν init
    θ0[pμ+pσ+pν+1] = log(0.4); θ0[pμ+pσ+pν+2] = log(0.4); θ0[pμ+pσ+pν+3] = 0.0   # log-Cholesky init
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res)
    # #961: a non-finite Hessian on finite data is a degenerate fit, not a crash.
    V, hess_finite = _vcov_from_hessian_checked(ForwardDiff.hessian(nll, θ̂),
                                                _all_finite(y, Xμ, Xσ, Xν); context = "Student")
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :nu => (pμ+pσ+1):(pμ+pσ+pν), :recov => (pμ+pσ+pν+1):(pμ+pσ+pν+3)]
    names = [:mu => nmμ, :sigma => nmσ, :nu => nmν, :recov => ["$(grp):L11", "$(grp):L22", "$(grp):L21"]]
    means = Dict(:mu => Xμ * θ̂[1:pμ]); obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]),
                  :nu => 2 .+ exp.(Xν * θ̂[(pμ+pσ+1):(pμ+pσ+pν)]))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, drm_optim_converged(res) && hess_finite, means, obs, scales), nll),
        Optim.iterations(res))
end

function _fit_student(fam::Student, y, Xμ, Xσ, Xν, nmμ, nmσ, nmν, g_tol; offset = nothing)
    n = length(y)
    pμ, pσ, pν = size(Xμ, 2), size(Xσ, 2), size(Xν, 2)
    w = _ctl_weights(n)                  # prior weights: row log-lik multipliers (#931)
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; βν = θ[pμ+pσ+1:pμ+pσ+pν]
        ημ = _eta_with_offset(Xμ, βμ, offset); ησ = Xσ * βσ; ην = Xν * βν
        s = zero(eltype(θ))
        @inbounds for i in 1:n
            z = (y[i] - ημ[i]) * exp(-ησ[i])
            s -= w[i] * (_student_logpdf_std(z, ην[i]) - ησ[i])       # − log σ Jacobian
        end
        return s
    end
    yo = _add_offset(y, offset === nothing ? nothing : -offset)   # #938: start from y - offset
    βμ0 = Xμ \ yo
    θ0 = zeros(pμ + pσ + pν)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(std(yo - Xμ * βμ0) + eps())       # σ init
    θ0[pμ+pσ+1] = log(10.0)                          # ν init (mildly heavy-tailed)
    res = _ctl_optimize((x0, opts, method) -> Optim.optimize(nll, x0, method, opts; autodiff = :forward),
                       θ0; g_tol = g_tol)   # control: maxit / start / multi_start (#948 #994)
    θ̂ = Optim.minimizer(res)
    # #961: a non-finite Hessian on finite data is a degenerate fit, not a crash.
    # The Hessian is always formed: the unit-free converged flag needs it even when
    # control says `se = false` and the stored covariance is left as NaN (#1050).
    H = ForwardDiff.hessian(nll, θ̂)
    Vh, hess_finite = _vcov_from_hessian_checked(H, _all_finite(y, Xμ, Xσ, Xν); context = "Student")
    V = _ctl_se(θ̂) do
        Vh
    end
    # Not the iteration cap, and the Newton step in SE units (#1050). Neither
    # `g_converged` (absolute 1e-8) nor the x/f/g OR is part of this flag (#983).
    converged = hess_finite && !Optim.iteration_limit_reached(res) &&
        _unitfree_converged(H, ForwardDiff.gradient(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :nu => (pμ+pσ+1):(pμ+pσ+pν)]
    names = [:mu => nmμ, :sigma => nmσ, :nu => nmν]
    means = Dict(:mu => _eta_with_offset(Xμ, θ̂[1:pμ], offset)); obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]),
                  :nu => 2 .+ exp.(Xν * θ̂[(pμ+pσ+1):(pμ+pσ+pν)]))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, converged, means, obs, scales), nll),
        Optim.iterations(res))
end

# Strip (possibly nested) ForwardDiff duals to Float64.
_student_fval(x::ForwardDiff.Dual) = _student_fval(ForwardDiff.value(x))
_student_fval(x::Real) = Float64(x)

# Student-t GLMM with two crossed (or nested) random intercepts on the mean,
# `(1 | g) + (1 | h)` (#725; drmTMB `student()` fits these by TMB's Laplace).
# b = [b_g; b_h] ~ N(0, diag(σ_g² I, σ_h² I)); the marginal likelihood is the
# Laplace approximation
#   nll(θ) = Σᵢ −log f(yᵢ | b̂) + ½ b̂ᵀΛb̂ + G log σ_g + H log σ_h + ½ logdet 𝐇(b̂),
# 𝐇 = ZᵀWZ + Λ, W = the observed (not expected) second derivative of −log f in μ.
# σ and ν may carry their own fixed-effect formulas. The Student data term is
# not log-concave, so the inner Newton falls back to the always-positive expected
# information (ν+1)/((ν+3)σ²) for its direction when 𝐇 is indefinite, with a
# line search; the Laplace term uses the observed 𝐇 and the evaluation fails
# closed (nll = 1e18) if 𝐇 is not positive definite at the mode. Derivatives are
# exact ForwardDiff: the Float64 mode b̂ is lifted to the dual numbers by two
# Newton steps at θ, which carry db̂/dθ (implicit-function theorem) accurately
# to second order, so the Hessian for vcov is exact as well.
# 𝐇 is dense (G+H)²: fine for hundreds of levels, slow for many thousands.
# θ = [βμ; βσ; βν; log σ_g; log σ_h].
# `sd_designs = nothing` is the constant-SD crossed route (one log SD per factor, block `:resd`).
# With `sd_designs = (Zg, Zh, nmsds, blocks)` each factor's log SD is a regression on its
# group-level design (`sd(g) ~ …`, `sd(h) ~ …`), so the prior precision and the log-determinant
# term are per level; the constant route is the special case Zg = 1, Zh = 1.
function _fit_student_crossed_laplace(fam::Student, y, Xμ, Xσ, Xν, comps, nmμ, nmσ, nmν, g_tol;
                                      sd_designs = nothing)
    n = length(y); pμ, pσ, pν = size(Xμ, 2), size(Xσ, 2), size(Xν, 2)
    (gidx, G, lg), (hidx, Hh, lh) = comps
    Zg = sd_designs === nothing ? ones(G, 1) : Matrix{Float64}(sd_designs[1])
    Zh = sd_designs === nothing ? ones(Hh, 1) : Matrix{Float64}(sd_designs[2])
    pg, ph = size(Zg, 2), size(Zh, 2)
    q = G + Hh
    yv = Float64.(y)
    # −log f and its μ-derivatives for observation i at mean μ. With t = (y−μ)/σ
    # and r = 1/ν (overflow-free, as in `_student_logpdf_std`):
    #   d/dμ = −(1+r) t / (σ(1 + r t²)),   d²/dμ² = (1+r)(1 − r t²) / (σ²(1 + r t²)²).
    function obs_terms(i, μ, lσ, ην)
        σ = exp(lσ); t = (yv[i] - μ) / σ
        r = ην > 0 ? exp(-ην) / (1 + 2 * exp(-ην)) : 1 / (2 + exp(ην))
        d = 1 + r * t^2
        val = lσ - _student_logpdf_std(t, ην)
        g1 = -(1 + r) * t / (σ * d)
        w = (1 + r) * (1 - r * t^2) / (σ^2 * d^2)
        wE = (1 + r) / ((1 + 3r) * σ^2)
        return val, g1, w, wE
    end
    function joint_terms(b, η0, ησ, ην, invg, invh; expected::Bool = false)
        T = promote_type(eltype(b), eltype(η0), eltype(ησ), eltype(ην), eltype(invg))
        H = zeros(T, q, q); grad = zeros(T, q); data = zero(T)
        @inbounds for i in 1:n
            a = gidx[i]; c = G + hidx[i]
            v, g1, w, wE = obs_terms(i, η0[i] + b[a] + b[c], ησ[i], ην[i])
            expected && (w = wE)
            data += v; grad[a] += g1; grad[c] += g1
            H[a, a] += w; H[c, c] += w; H[a, c] += w; H[c, a] += w
        end
        @inbounds for j in 1:G
            grad[j] += invg[j] * b[j]; H[j, j] += invg[j]
        end
        @inbounds for j in (G+1):q
            grad[j] += invh[j-G] * b[j]; H[j, j] += invh[j-G]
        end
        joint = data + 0.5 * sum(invg .* abs2.(@view b[1:G])) + 0.5 * sum(invh .* abs2.(@view b[G+1:q]))
        return joint, grad, H
    end
    function inner_mode(η0, ησ, ην, invg, invh, b0; maxiter::Int = 100, tol::Real = 1e-10)
        b = copy(b0)
        for _ in 1:maxiter
            J0, grad, H = joint_terms(b, η0, ησ, ην, invg, invh)
            ch = cholesky(Symmetric(H); check = false)
            if !issuccess(ch)                     # indefinite: expected-information direction
                ch = cholesky(Symmetric(last(joint_terms(b, η0, ησ, ην, invg, invh; expected = true)));
                              check = false)
                issuccess(ch) || return b, false  # fail closed: even the expected info is not PD here
            end
            step = ch \ grad
            norm(step) <= tol * (1 + norm(b)) && return b, true
            α = 1.0; accepted = false
            while α >= 1e-8
                trial = b .- α .* step
                if first(joint_terms(trial, η0, ησ, ην, invg, invh)) <= J0
                    b = trial; accepted = true; break
                end
                α /= 2
            end
            # A near-zero crossed variance makes invg/invh (and so H's diagonal) large,
            # which raises the floating-point floor the Newton gradient can reach before
            # line-search steps underflow -- confirmed on #827's sdh0 data: the gradient
            # plateaus at ~4e-6 (well converged in relative terms) but never crosses the
            # old 1e-8*(1+n) floor, so the line-search failure reads as non-convergence
            # and poisons the outer nll with a spurious 1e18. Loosened by 100x (still a
            # tight absolute tolerance relative to a Hessian diagonal of order invh).
            accepted || return b, norm(grad, Inf) <= 1e-6 * (1 + n)
        end
        return b, false
    end
    last_b = zeros(q)
    function nll(θ)
        T = eltype(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; βν = θ[pμ+pσ+1:pμ+pσ+pν]
        k0 = pμ + pσ + pν
        lsg = Zg * θ[k0+1:k0+pg]; lsh = Zh * θ[k0+pg+1:k0+pg+ph]
        # Wide guard, not a tight wall (#827 review): a crossed variance genuinely at
        # its zero boundary reports as log σ ≈ −12 to −13 (drmTMB's sdh0 case puts it
        # at −12.57), so a wall at 12 rejects a legitimate MLE outright. invg/invh stay
        # finite (no Float64 overflow) out to |log σ| ≈ 350, so 30 is generous headroom
        # while still catching a runaway line-search probe before invg/invh overflow.
        (any(x -> abs(_student_fval(x)) > 30, lsg) || any(x -> abs(_student_fval(x)) > 30, lsh)) && return T(1e18)
        η0 = Xμ * βμ; ησ = Xσ * βσ; ην = Xν * βν
        invg = exp.(-2 .* lsg); invh = exp.(-2 .* lsh)
        (all(x -> isfinite(_student_fval(x)), invg) && all(x -> isfinite(_student_fval(x)), invh)) ||
            return T(1e18)
        f0 = (_student_fval.(η0), _student_fval.(ησ), _student_fval.(ην),
              _student_fval.(invg), _student_fval.(invh))
        b̂, ok = inner_mode(f0..., last_b)
        ok || ((b̂, ok) = inner_mode(f0..., zeros(q)))
        ok || return T(1e18)
        T === Float64 && (last_b .= b̂)
        b = b̂
        for _ in 1:2                              # lift b̂ to b̂(θ) in dual arithmetic
            _, grad, H = joint_terms(b, η0, ησ, ην, invg, invh)
            ch = cholesky(Symmetric(H); check = false)
            issuccess(ch) || return T(1e18)
            b = b .- (ch \ grad)
        end
        J, _, H = joint_terms(b, η0, ησ, ην, invg, invh)
        ch = cholesky(Symmetric(H); check = false)
        issuccess(ch) || return T(1e18)
        return J + sum(lsg) + sum(lsh) + 0.5 * logdet(ch)
    end
    βμ0 = Xμ \ yv
    k = pμ + pσ + pν
    θ0 = zeros(k + pg + ph)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(std(yv - Xμ * βμ0) + eps()) - log(2.0)   # σ init (REs take a share)
    θ0[pμ+pσ+1] = log(10.0)                                  # ν init (mildly heavy-tailed)
    θ0[k+1:k+pg] .= Zg \ fill(θ0[pμ+1], G)                   # σ_g, σ_h init
    θ0[k+pg+1:k+pg+ph] .= Zh \ fill(θ0[pμ+1], Hh)
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol, iterations = 1000);
                         autodiff = :forward)
    # Repeated restart from θ̂ on non-convergence (#827 review; mirrors #837): a non-converged
    # LBFGS run can stop early after its line search crosses the fail-closed 1e18 region even
    # on ordinary, well-conditioned data (confirmed: restarting from θ̂ walks the reviewer's
    # "gauss" data to drmTMB's optimum). Each restart's own line search can fail again a few
    # steps later (still short of the outer g_tol), so this restarts a bounded number of times
    # rather than once, keeping whichever minimizer is best each time. `Optim.minimum` is NOT
    # trustworthy for that comparison: on a "line search failed" result it reports the value of
    # the LAST (rejected) trial point, not of `Optim.minimizer` (confirmed: the restart's own
    # minimizer was a genuine improvement, nll 296.22 vs 296.37, while its reported `minimum`
    # read 1e18) -- so `nll` is re-evaluated at each minimizer directly instead.
    for _ in 1:8
        Optim.converged(res) && break
        res2 = Optim.optimize(nll, Optim.minimizer(res), Optim.LBFGS(),
                              Optim.Options(g_tol = g_tol, iterations = 1000); autodiff = :forward)
        improved = nll(Optim.minimizer(res2)) < nll(Optim.minimizer(res))
        improved || break
        res = res2
    end
    θ̂ = Optim.minimizer(res)
    nllhat = nll(θ̂)
    gfinal = ForwardDiff.gradient(nll, θ̂)
    converged = nllhat < 1e17 && all(isfinite, gfinal) &&
                (Optim.converged(res) || _laplace_outer_converged(res, nllhat, gfinal, θ̂, n, g_tol))
    V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂);
                           context = "sparse-Laplace Student (crossed intercepts)")
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :nu => (pμ+pσ+1):k, :resd => (k+1):(k+2)]
    names = [:mu => nmμ, :sigma => nmσ, :nu => nmν, :resd => [lg, lh]]
    if sd_designs !== nothing
        blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :nu => (pμ+pσ+1):k,
                  sd_designs[4][1] => (k+1):(k+pg), sd_designs[4][2] => (k+pg+1):(k+pg+ph)]
        names = [:mu => nmμ, :sigma => nmσ, :nu => nmν,
                 sd_designs[4][1] => sd_designs[3][1], sd_designs[4][2] => sd_designs[3][2]]
    end
    means = Dict(:mu => Xμ * θ̂[1:pμ]); obs = Dict(:mu => yv)      # population μ (b = 0)
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]),
                  :nu => 2 .+ exp.(Xν * θ̂[(pμ+pσ+1):k]))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nllhat, n, converged, means, obs, scales), nll),
        Optim.iterations(res))
end
