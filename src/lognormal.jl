# lognormal.jl — LogNormal family for strictly-positive responses whose log is
# Gaussian. The mean formula μ is the mean of log y (identity link on the log
# scale); σ (log link) is the SD of log y. The log-density is the Gaussian
# log-density of log y plus the −log y change-of-variables Jacobian, so the fit
# reuses the Gaussian location–scale objective on log y. Fixed effects + random
# intercept/slope on μ, ML. Mirrors drmTMB's `lognormal`. `Distributions.Normal`
# is used qualified inside the GHQ random-effect paths.

import Distributions

"""
    LogNormal()

LogNormal response family for positive continuous data: the mean formula `μ` is
the **mean of `log y`** (identity link on the log scale), and `σ` (log link) is
the **SD of `log y`**. The response-scale median is `exp(μ)`. Mirrors `drmTMB`'s
`lognormal` family.

!!! note
    `DRModels.LogNormal` shadows `Distributions.LogNormal`; qualify the latter if needed.

A random intercept `(1 | g)` or a correlated random intercept+slope `(1 + x | g)`
may be placed on the log-mean `μ`; the group effect is integrated out by
Gauss–Hermite quadrature (`re_sd(fit)[:g]` / `vc(fit)[:g]`). Crossed/multiple
random intercepts, `(1 | g) + (1 | h)` (#736), delegate the same way as the
structured markers below: `log(y)` is exactly Gaussian, so the fit runs
WHOLESALE through `drm(f, Gaussian(); data = data-with-logged-response)`
(`_fit_multi_ranef_gaussian`, the closed-form exact crossed marginal, not a
Laplace approximation), with the reported log-likelihood shifted by
`-sum(log y)`.

## Structured markers (`phylo`/`relmat`)

`phylo(1 | group)` (needs `tree = ...`) and `relmat(1 | group)` (needs
`K = ...`) may be placed on the mean, exactly as drmTMB's `lognormal` family
supports. `log(y)` is exactly Gaussian, so a structured call delegates
WHOLESALE to `drm(f, Gaussian(); data = data-with-logged-response, tree = ...,
K = ...)` — the same identity `biv_lognormal()` already uses
(`src/bivariate_lognormal.jl`) — and shifts the reported log-likelihood by the
parameter-free Jacobian `-sum(log y)`. `theta`/`vcov`/`ranef` and the fitted
objective's gradient carry over untouched from the Gaussian-on-log(y) fit;
only `loglik` (and everything derived from it: aic/bic/deviance) shifts.
`animal`/`spatial` markers and `meta_V` are not implemented for `LogNormal()`
(no R parity cell) and stay refused, as does any marker on a formula other
than the mean.

```julia
fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), LogNormal(); data = dat)
coef(fit, :mu)                 # on the log scale
exp(coef(fit, :sigma)[1])      # SD of log y

fit_phy = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
              LogNormal(); data = dat, tree = tr)
```
"""
struct LogNormal end

@drm_entry function drm(f::DrmFormula, fam::LogNormal; data, tree = nothing, K = nothing, g_tol::Real = 1e-8,
             marginal::Symbol = :LA)
    _drm_entry_guard(f, fam, data; tree = tree, K = K)   # #1027 #1022 #1010
    # #1126: same `marginal` keyword as the other families. log y is exactly Gaussian
    # given the random intercept, so the Laplace approximation is exact: `:Laplace`
    # is the same fit as the default `:LA` (the mode-centred 1-node rule), tagged.
    laplace = _scalar_laplace_requested(marginal)
    laplace || _marginal_method(marginal) isa Laplace ||
        throw(ArgumentError("marginal = :$marginal is not available for LogNormal(): log y is " *
            "Gaussian given the random effect, so the default `:LA` (exact Gauss–Hermite / " *
            "Laplace) is used; `:Laplace` is accepted for an ordinary `(1 | g)` and gives the " *
            "same fit. `:VA` and `:AGHQ` have no LogNormal route."))
    sd_targets = _sd_target_parts(f)
    if !isempty(sd_targets) && !all(_formula_response_observed_mask(f, data))
        throw(ArgumentError("LogNormal random-effect SD regression requires observed responses; remove incomplete rows before fitting."))
    end
    missing_fit = _fit_observed_response_rows(f, data) do data_observed
        drm(f, fam; data = data_observed, tree = tree, K = K, g_tol = g_tol, marginal = marginal)
    end
    missing_fit !== nothing && return missing_fit
    if laplace
        _, re_l, mv_l, st_l = _split_ranef(Dict(f.forms)[:mu])
        (length(re_l) == 1 && mv_l === nothing && st_l === nothing &&
         _re_kind(re_l[1][1])[1] === :intercept) ||
            throw(ArgumentError("marginal = :Laplace is not available for LogNormal() with this " *
                "model. It covers exactly one ordinary random intercept `(1 | g)` on the mean; " *
                "omit `marginal` (the default `:LA`) for other models."))
        fit = drm(f, fam; data = data, tree = tree, K = K, g_tol = g_tol, marginal = :LA)
        return _withmarginal(fit, :Laplace)
    end

    if !isempty(sd_targets)
        corr_sd = _lognormal_corr_sd_block(f)
        corr_sd === nothing || return _withformula(
            _drm_lognormal_corr_sd(f, fam, data, g_tol, corr_sd), f)
        # More than one SD formula (crossed `(1 | g) + (1 | h)` with `sd(g)` and `sd(h)`, or the
        # labelled coefficient pair): log y is exactly Gaussian, so the whole fit is delegated
        # to the public Gaussian dispatcher on log(y), which refuses what it does not cover.
        length(sd_targets) == 1 || return _withformula(
            _fit_lognormal_multi_sd(fam, f, data, g_tol), f)
        return _withformula(_drm_lognormal_sd_regression(f, fam, data, g_tol, only(sd_targets)), f)
    end
    _lss_only_gaussian_guard(f, fam)   # #544: refuse, never silently drop, sd() parts
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    fixed_mu, offset_term = _extract_offset(fixed_mu)   # #938: offset(z), coefficient fixed at 1
    offset_term === nothing || (isempty(re) && mv === nothing && st === nothing) ||
        _refuse_offset_route(fam, "a random, structured or `meta_V` term on the mean")
    mv === nothing ||
        error("LogNormal() does not support meta_V markers")
    let sigre = _drm_sigma_ranef_laplace_family(f, fam; data, g_tol, se = true, fixed_mu, re, mv, st,
                                                offset_term = offset_term, isva = false, isaghq = marginal !== :LA)
        sigre === nothing || return sigre
    end
    for (pname, r) in f.forms          # only the mean may carry a random effect
        pname === :mu && continue
        _, re2, mv2, st2 = _split_ranef(r)
        (isempty(re2) && mv2 === nothing && st2 === nothing) ||
            error("LogNormal(): only the mean formula may carry a random effect")
    end
    if st !== nothing                  # structured marker on the mean (#563 S8)
        st[1] in (:phylo, :relmat) ||
            error("LogNormal() supports `phylo`/`relmat` structured markers on the mean " *
                  "(got `$(st[1])`); drmTMB's `lognormal` family does not implement " *
                  "`animal`/`spatial` markers")
        isempty(re) ||
            error("LogNormal() supports a single structured marker OR random-effect term " *
                  "on the mean, not both")
        return _withformula(_fit_lognormal_structured(fam, f, data, st[1], st[2]; tree = tree, K = K, g_tol = g_tol), f)
    end
    y, Xμ, nmμ = _design(f.response, fixed_mu, data)
    _, Xσ, nmσ = _design(f.response, get(rhs, :sigma, ConstantTerm(1)), data)
    all(yi -> yi > 0, y) || error("LogNormal() requires strictly positive responses")
    all(isfinite, y) || error("LogNormal() requires finite, strictly positive responses")
    if !isempty(re)                    # random effect on the log-mean μ → GHQ
        if length(re) > 1              # crossed/multiple intercepts (#736)
            # Independent intercepts + slopes `(1 | g) + (0 + x | g)`: `log(y)` is Gaussian, so the
            # exact Gaussian multi-component marginal applies (same delegation as crossed intercepts).
            all(_re_kind(r[1])[1] in (:intercept, :slope) for r in re) ||
                error("LogNormal() supports multiple random effects only as crossed/nested " *
                      "intercepts `(1 | g) + (1 | h)` or independent intercepts and slopes " *
                      "`(1 | g) + (0 + x | g)`; a correlated `(1 + x | g)` must be the only " *
                      "random-effect term")
            return _withformula(_fit_lognormal_crossed(fam, f, data, g_tol), f)
        end
        length(re) == 1 ||
            error("LogNormal() supports a single random-effect term on the mean")
        (rk, var) = _re_kind(re[1][1]); grp = re[1][2]
        gidx, G = _group_index(getproperty(data, grp); col = grp)
        if rk === :intercept           # (1 | g) → 1-D Gauss–Hermite
            return _withformula(_fit_lognormal_ranef(fam, y, Xμ, Xσ, gidx, G, nmμ, nmσ, grp, g_tol), f)
        elseif rk === :slope            # (0 + x | g) → exact weighted Gaussian marginal
            return _withformula(_fit_lognormal_independent_slope(
                fam, y, Xμ, Xσ, getproperty(data, var), gidx, G, nmμ, nmσ, grp, g_tol), f)
        elseif rk === :corr            # (1 + x | g) → correlated 2-D Gauss–Hermite
            _require_finite_data_column(data, var)
            return _withformula(_fit_lognormal_corr_ranef(fam, y, Xμ, Xσ, Float64.(getproperty(data, var)), gidx, G, nmμ, nmσ, grp, g_tol), f)
        else
            error("LogNormal() supports `(1 | g)`, `(0 + x | g)`, or `(1 + x | g)` on the mean")
        end
    end
    return _withformula(_fit_lognormal(fam, y, Xμ, Xσ, nmμ, nmσ, g_tol; offset = _offset_for_fit(fam, offset_term, data, size(Xμ, 1))), f)
end

# `log(y)` is Gaussian with the same location, residual scale, and random
# intercept as the LogNormal model. Reuse the exact Gaussian LSS marginal
# likelihood, then add the LogNormal change-of-variables Jacobian to the
# objective and keep the public fit family and response-scale summaries.
function _drm_lognormal_sd_regression(f::DrmFormula, fam::LogNormal, data, g_tol, target)
    descriptor = target.target
    (descriptor.level === :iid && descriptor.dpar === :mu && descriptor.block === nothing) ||
        throw(ArgumentError("LogNormal SD regression targets one ordinary mean random-effect SD."))
    grp = descriptor.group
    isempty(_sdphylo_parts(f)) || throw(ArgumentError(
        "LogNormal SD regression does not support structured SD formulas."))
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    mv === nothing || throw(ArgumentError("LogNormal SD regression does not support meta_V()."))
    st === nothing || throw(ArgumentError("LogNormal SD regression currently supports ordinary independent mean intercepts or slopes only."))
    length(re) == 1 && only(re)[2] === grp || throw(ArgumentError(
        "LogNormal `sd($grp)` requires one ordinary random-effect coefficient on the mean."))
    kind, slope = _re_kind(only(re)[1])
    kind in (:intercept, :slope) || throw(ArgumentError(
        "LogNormal `sd($grp)` supports `(1 | $grp)` or one independent numeric `(0 + x | $grp)` term; correlated blocks are not implemented on this route."))
    member = kind === :intercept ? :intercept : slope
    descriptor.member in (nothing, member) || throw(ArgumentError(
        "LogNormal SD target `coef($(descriptor.member))` does not match the fitted random-effect coefficient `$(member)`."))
    sigma_rhs = get(rhs, :sigma, nothing)
    sigma_rhs isa ConstantTerm && sigma_rhs == ConstantTerm(1) || throw(ArgumentError(
        "LogNormal `sd($grp)` currently requires fixed `sigma ~ 1`."))
    for (pname, _) in f.forms
        (pname === :mu || pname === :sigma || pname === target.key) && continue
        throw(ArgumentError("LogNormal `sd($grp)` supports only `mu`, fixed `sigma ~ 1`, and one SD formula."))
    end
    y, Xμ, nmμ = _design(f.response, fixed_mu, data)
    _, Xσ, nmσ = _design(f.response, sigma_rhs, data)
    all(yi -> isfinite(yi) && yi > 0, y) || throw(ArgumentError(
        "LogNormal() requires finite, strictly positive responses"))
    gidx, G = _group_index(getproperty(data, grp))
    Zg, nmsd = _sd_group_design(f.response, target.rhs, data, gidx, G, grp;
                                schema_cache = f.schema_cache, schema_key = target.key)
    size(Zg, 2) <= G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
        "LogNormal `sd($grp)` group-level design is rank deficient; use predictors that vary independently across groups."))
    w = if kind === :intercept
        nothing
    else
        hasproperty(data, slope) || throw(ArgumentError(
            "LogNormal random-slope predictor `$slope` is absent from data."))
        raw = getproperty(data, slope)
        raw isa AbstractVector{<:Real} && length(raw) == length(y) && all(isfinite, raw) ||
            throw(ArgumentError("LogNormal random-slope predictor `$slope` must contain one finite numeric value per response."))
        any(x -> !iszero(x), raw) || throw(ArgumentError(
            "LogNormal random-slope predictor `$slope` must contain at least one non-zero value."))
        Float64.(raw)
    end
    return _fit_lognormal_sd_regression(fam, y, Xμ, Xσ, Zg, w, gidx, G,
        nmμ, nmσ, nmsd, grp, g_tol)
end

function _fit_lognormal_sd_regression(fam::LogNormal, y, Xμ, Xσ, Zg, w, gidx, G,
                                      nmμ, nmσ, nmsd, grp, g_tol)
    logy = log.(y)
    fit_gaussian = _fit_ranef_gaussian_lss(Gaussian(), logy, Xμ, Xσ, Zg,
        gidx, G, nmμ, nmσ, nmsd, grp, g_tol; reml = false, w)
    jacobian = sum(logy)
    sd_group = exp.(Zg * coef(fit_gaussian, :sd))
    scales = copy(fit_gaussian.scales)
    scales[:sd_group] = sd_group
    scales[:sd] = sd_group[gidx]
    scales[:sd_group_index] = Float64.(gidx)
    fit = DrmFit(fam, fit_gaussian.blocks, fit_gaussian.coefnames,
        fit_gaussian.theta, fit_gaussian.vcov, fit_gaussian.loglik - jacobian,
        fit_gaussian.nobs, fit_gaussian.converged,
        Dict(:mu => exp.(fit_gaussian.means[:mu])), Dict(:mu => Float64.(y)), scales)
    fit = _withnll(fit, θ -> fit_gaussian.nll(θ) + jacobian)
    return _withiterations(_withranef(fit, fit_gaussian.ranef), fit_gaussian.iterations)
end

# Ordinary correlated `(1 + x | g)` with `sd(g, coef(·)) ~ …` formulas on one or both
# coefficients. log(y) is Gaussian given the random effects, so this is the Gaussian
# correlated-block SD fit on log(y) plus the LogNormal Jacobian, as for the independent
# routes above. Two coefficients only: drmTMB refuses the q = 3, 4 LogNormal blocks.
function _lognormal_corr_sd_block(f::DrmFormula)
    mu = get(Dict(f.forms), :mu, nothing)
    mu === nothing && return nothing
    try
        _, re, mv, st = _split_ranef(mu)
        (mv === nothing && st === nothing && length(re) == 1) || return nothing
        vars = _re_multi_corr_vars(re[1][1])
        (vars !== nothing && length(vars) == 1) || return nothing
        return (; re, vars)
    catch
        return nothing
    end
end

function _drm_lognormal_corr_sd(f::DrmFormula, fam::LogNormal, data, g_tol, block)
    rhs = Dict(f.forms)
    sigma_rhs = get(rhs, :sigma, nothing)
    sigma_rhs isa ConstantTerm && sigma_rhs == ConstantTerm(1) || throw(ArgumentError(
        "LogNormal correlated `(1 + x | g)` SD regression currently requires fixed `sigma ~ 1`."))
    isempty(_sdphylo_parts(f)) || throw(ArgumentError(
        "LogNormal SD regression does not support structured SD formulas."))
    fixed_mu, _, _, _ = _split_ranef(rhs[:mu])
    y, Xμ, nmμ = _design(f.response, fixed_mu, data)
    _, Xσ, nmσ = _design(f.response, sigma_rhs, data)
    all(yi -> isfinite(yi) && yi > 0, y) || throw(ArgumentError(
        "LogNormal() requires finite, strictly positive responses"))
    logy = log.(y)
    fit_gaussian = _drm_gaussian_ordinary_q34_sd(f, Gaussian(), Pair{Symbol,Any}[], block.re,
        [block.vars], nothing, nothing, Symbol[], nothing, false, logy, Xμ, Xσ, nmμ, nmσ,
        data, g_tol, :ML, :auto)
    jacobian = sum(logy)
    fit = DrmFit(fam, fit_gaussian.blocks, fit_gaussian.coefnames,
        fit_gaussian.theta, fit_gaussian.vcov, fit_gaussian.loglik - jacobian,
        fit_gaussian.nobs, fit_gaussian.converged,
        Dict(:mu => exp.(fit_gaussian.means[:mu])), Dict(:mu => Float64.(y)),
        copy(fit_gaussian.scales))
    return _withiterations(_withnll(fit, θ -> fit_gaussian.nll(θ) + jacobian),
                           fit_gaussian.iterations)
end

# Independent random slope on log(y) is an exact Gaussian LMM with loading x.
# Reuse the weighted Woodbury route and add the LogNormal Jacobian, preserving
# coefficient-scale random-effect summaries.
function _fit_lognormal_independent_slope(fam::LogNormal, y, Xμ, Xσ, raw_w,
                                          gidx, G, nmμ, nmσ, grp, g_tol)
    raw_w isa AbstractVector{<:Real} && length(raw_w) == length(y) && all(isfinite, raw_w) ||
        throw(ArgumentError("LogNormal random-slope predictor must contain one finite numeric value per response."))
    any(x -> !iszero(x), raw_w) || throw(ArgumentError(
        "LogNormal random-slope predictor must contain at least one non-zero value."))
    w = Float64.(raw_w)
    logy = log.(y)
    fit_gaussian = _fit_ranef_gaussian(Gaussian(), logy, Xμ, Xσ, gidx, G, w,
                                       nmμ, nmσ, grp, g_tol; reml = false)
    jacobian = sum(logy)
    fit = DrmFit(fam, fit_gaussian.blocks, fit_gaussian.coefnames,
        fit_gaussian.theta, fit_gaussian.vcov, fit_gaussian.loglik - jacobian,
        fit_gaussian.nobs, fit_gaussian.converged,
        Dict(:mu => exp.(fit_gaussian.means[:mu])), Dict(:mu => Float64.(y)),
        copy(fit_gaussian.scales))
    return _withiterations(_withranef(_withnll(fit, θ -> fit_gaussian.nll(θ) + jacobian),
                                      fit_gaussian.ranef), fit_gaussian.iterations)
end

# LogNormal GLMM with a random intercept (1|g) on the log-mean μ (the mean of
# log y). b_g ~ N(0,σ_b²) is integrated per group by a single mode-centred
# Gauss–Hermite node, exact for this linear-Gaussian log-response model.
# σ = SD of log y (log link) is a fixed effect. Each observation contributes
# the Gaussian log-density of log y. The constant
# −Σ log y change-of-variables Jacobian is added to the reported loglik so it
# matches the fixed fitter's convention.
function _fit_lognormal_ranef(fam::LogNormal, y, Xμ, Xσ, gidx, G, nmμ, nmσ, grp, g_tol)
    n = length(y); pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    ly = log.(y); sumlogy = sum(ly)                  # Σ log y = the Jacobian offset
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    # The `(1|g)` marginal is linear-Gaussian in b_g (μ enters ly's mean by identity
    # link, no transform), so K = 1 (the mode-centred Laplace/AGHQ node) is exact for
    # this family — no larger rule can improve on it (#719).
    rule = _AGHQRule(1, 1); Zre = ones(n, 1); bcache = zeros(1, G)
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; σb = exp(θ[pμ+pσ+1])
        η0 = Xμ * βμ; ησ = clamp.(Xσ * βσ, -15.0, 15.0)
        ll = (i, η) -> Distributions.logpdf(Distributions.Normal(clamp(η, -30.0, 30.0), exp(ησ[i])), ly[i])
        L = reshape([σb], 1, 1)
        return -_aghq_marginal_loglik(ll, members, η0, Zre, L, rule, bcache) + sumlogy   # + Σ log y so loglik carries the Jacobian
    end
    βμ0 = Xμ \ ly
    θ0 = zeros(pμ + pσ + 1)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(std(ly - Xμ * βμ0) + eps()); θ0[pμ+pσ+1] = log(0.5)
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :resd => (pμ+pσ+1):(pμ+pσ+1)]
    names = [:mu => nmμ, :sigma => nmσ, :resd => [String(grp)]]
    means = Dict(:mu => exp.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]))
    _record_cond_fit_aghq!(scales, nll, θ̂, bcache, gidx, Zre)   # fitted-row modes for `fitted(fit)`
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, drm_optim_converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# LogNormal GLMM with a CORRELATED random intercept+slope (1 + x | g) on the
# log-mean μ. Per group (b0,b1) ~ N(0, Σ_re), Σ_re = L Lᵀ with L = [l11 0; cc l22]
# (log-Cholesky parameters a, b, cc; l11=exp(a), l22=exp(b)). The 2-D group integral
# is taken by per-group ADAPTIVE Gauss–Hermite quadrature (`_aghq_marginal_loglik`,
# #834; log y is Gaussian in b, so this is exact for any nq). σ = SD of log y is a
# fixed effect. θ = [β_μ; β_σ; a, b, cc]. The constant −Σ log y
# Jacobian is added to the reported loglik to match the fixed fitter's convention.
function _fit_lognormal_corr_ranef(fam::LogNormal, y, Xμ, Xσ, xs, gidx, G, nmμ, nmσ, grp, g_tol; nq::Int = _CORR_RANEF_AGHQ_K)
    n = length(y); pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    ly = log.(y); sumlogy = sum(ly)                  # Σ log y = the Jacobian offset
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    rule = _AGHQRule(2, nq); Zre = hcat(ones(n), Float64.(xs)); bcache = zeros(2, G)   # #834: per-group AGHQ
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]
        L = _corr_ranef_L(θ[pμ+pσ+1], θ[pμ+pσ+2], θ[pμ+pσ+3])
        η0 = Xμ * βμ; ησ = clamp.(Xσ * βσ, -15.0, 15.0)
        ll = (i, η) -> Distributions.logpdf(Distributions.Normal(clamp(η, -30.0, 30.0), exp(ησ[i])), ly[i])
        return -_aghq_marginal_loglik(ll, members, η0, Zre, L, rule, bcache) + sumlogy   # + Σ log y: Jacobian
    end
    βμ0 = Xμ \ ly
    θ0 = zeros(pμ + pσ + 3)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(std(ly - Xμ * βμ0) + eps())
    θ0[pμ+pσ+1] = log(0.4); θ0[pμ+pσ+2] = log(0.4); θ0[pμ+pσ+3] = 0.0
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :recov => (pμ+pσ+1):(pμ+pσ+3)]
    names = [:mu => nmμ, :sigma => nmσ, :recov => ["$(grp):L11", "$(grp):L22", "$(grp):L21"]]
    means = Dict(:mu => exp.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]))
    _record_cond_fit_aghq!(scales, nll, θ̂, bcache, gidx, Zre)   # fitted-row modes for `fitted(fit)`
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, drm_optim_converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

function _fit_lognormal(fam::LogNormal, y, Xμ, Xσ, nmμ, nmσ, g_tol; offset = nothing)
    n = length(y); pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    w = _ctl_weights(n)                  # prior weights: row log-lik multipliers (#931)
    ly = log.(y); sumlogy = sum(w .* ly)             # Σ wᵢ log yᵢ = the (weighted) Jacobian offset
    sw = sum(w)                                      # == n exactly when unweighted
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]
        ημ = _eta_with_offset(Xμ, βμ, offset); ησ = Xσ * βσ                   # ησ = log σ
        s = zero(eltype(θ))
        @inbounds for i in 1:n
            r = ly[i] - ημ[i]
            s += w[i] * (ησ[i] + 0.5 * r * r * exp(-2 * ησ[i]))
        end
        return s + 0.5 * sw * log(2π) + sumlogy        # Gaussian-on-log-y nll + Σ log y
    end
    lyo = _add_offset(ly, offset === nothing ? nothing : -offset)   # #938: start from log y - offset
    βμ0 = Xμ \ lyo
    θ0 = zeros(pμ + pσ)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(std(lyo - Xμ * βμ0) + eps())
    res = _ctl_optimize((x0, opts, method) -> Optim.optimize(nll, x0, method, opts; autodiff = :forward),
                       θ0; g_tol = g_tol)   # control: maxit / start / multi_start (#948 #994)
    θ̂ = Optim.minimizer(res); V = _ctl_se(() -> _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂)), θ̂)
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ)]
    names = [:mu => nmμ, :sigma => nmσ]
    means = Dict(:mu => exp.(_eta_with_offset(Xμ, θ̂[1:pμ], offset))); obs = Dict(:mu => Vector{Float64}(y))  # response-scale median
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, drm_optim_converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# Crossed/multiple random intercepts `(1 | g) + (1 | h) + …` on the log-mean μ
# (#736). `log(y)` is exactly Gaussian, so — exactly like the phylo/relmat
# structured-marker delegation just below — this delegates the WHOLE fit to
# the public Gaussian dispatcher on a logged copy of the response
# (`_fit_multi_ranef_gaussian`, the closed-form exact crossed marginal
# likelihood; no Laplace approximation, so no new integrator), and shifts the
# reported log-likelihood by the parameter-free Jacobian `-sum(log y)`.
# Requires strictly positive responses (checked before logging).
function _fit_lognormal_crossed(fam::LogNormal, f::DrmFormula, data, g_tol)
    cols = NamedTuple(pairs(data))
    y = Vector{Float64}(getproperty(cols, f.response))
    all(yi -> yi > 0, y) || error("LogNormal() requires strictly positive responses")
    logdata = merge(cols, NamedTuple{(f.response,)}((log.(y),)))
    gfit = drm(f, Gaussian(); data = logdata, g_tol = g_tol)
    return _lognormal_jacobian_shift(fam, gfit, y)
end

function _fit_lognormal_multi_sd(fam::LogNormal, f::DrmFormula, data, g_tol)
    cols = NamedTuple(pairs(data))
    y = Vector{Float64}(getproperty(cols, f.response))
    all(yi -> isfinite(yi) && yi > 0, y) || throw(ArgumentError(
        "LogNormal() requires finite, strictly positive responses"))
    logdata = merge(cols, NamedTuple{(f.response,)}((log.(y),)))
    gfit = drm(f, Gaussian(); data = logdata, g_tol = g_tol)
    return _lognormal_jacobian_shift(fam, gfit, y)
end

# A `phylo(1 | group)` or `relmat(1 | group)` marker on the log-mean μ (#563
# S8). `log(y)` is exactly Gaussian, so this delegates the WHOLE fit to the
# public Gaussian dispatcher on a logged copy of the response — the same
# identity `src/bivariate_lognormal.jl` already uses for the bivariate family
# — rather than duplicating the phylo/relmat structured engine here. Requires
# strictly positive responses (checked before logging).
function _fit_lognormal_structured(fam::LogNormal, f::DrmFormula, data, kind::Symbol, grp::Symbol; tree, K, g_tol)
    if kind === :phylo
        tree === nothing && error("LogNormal(): phylo(1 | $grp) needs `tree = ...`")
    else # :relmat
        K === nothing && error("LogNormal(): relmat(1 | $grp) needs `K = ...`")
    end
    cols = NamedTuple(pairs(data))
    y = Vector{Float64}(getproperty(cols, f.response))
    all(yi -> yi > 0, y) || error("LogNormal() requires strictly positive responses")
    logdata = merge(cols, NamedTuple{(f.response,)}((log.(y),)))
    gfit = drm(f, Gaussian(); data = logdata, tree = tree, K = K, g_tol = g_tol)
    return _lognormal_jacobian_shift(fam, gfit, y)
end

# log f_Y(y) = log phi(log y; .) - sum(log y). Parameter-free, so theta-hat,
# vcov, and ranef carry over untouched from the Gaussian fit on log(y); only
# the likelihood VALUE (and everything derived from it: aic/bic/deviance)
# shifts by the Jacobian — mirrors `_lognormal_jacobian_shift` in
# src/bivariate_lognormal.jl for the univariate case. The Gaussian fit's `means`
# are on the LOG scale; `simulate` expects the response-scale
# median `exp(.)` like the plain LogNormal route (#1102); since #1282 `fitted` and
# `residuals` use E[Y] = exp(mu + sigma^2 / 2) (`means[:mu]` still holds the median).
function _lognormal_jacobian_shift(fam::LogNormal, gfit::DrmFit, y::Vector{Float64})
    jac = sum(log, y)
    gnll = gfit.nll
    lnll = gnll === nothing ? nothing : (θ -> gnll(θ) + jac)   # nll = -loglik, so the Jacobian ADDS
    reml_ll = isnan(gfit.reml_loglik) ? gfit.reml_loglik : gfit.reml_loglik - jac
    return DrmFit(fam, gfit.blocks, gfit.coefnames, gfit.theta, gfit.vcov,
                 gfit.loglik - jac, gfit.nobs, gfit.converged,
                 Dict(:mu => exp.(gfit.means[:mu])), Dict(:mu => y), gfit.scales,
                 gfit.formula, lnll, gfit.nllgrad, gfit.ranef,
                 gfit.estim_method, reml_ll, gfit.ml_loglik - jac, gfit.marginal,
                 gfit.phylo_penalty, gfit.penalty, gfit.iterations, gfit.phylo_scale,
                 gfit.uncertainty, gfit.geometry)
end
