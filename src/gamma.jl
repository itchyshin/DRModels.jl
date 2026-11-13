# gamma.jl — Gamma family for strictly-positive continuous responses (durations,
# sizes, concentrations). Log link on the mean μ; the `sigma` slot carries σ =
# the coefficient of variation, mapped to the shape α = 1/σ² (so var = μ²σ² and
# coef(:sigma) is log σ). Likelihood Gamma(α, μ/α) (shape–scale, mean μ). Fixed
# effects, ML. `Distributions.Gamma` is used qualified — DRModels exports its own
# `Gamma` family.

import Distributions

"""
    Gamma()

Gamma response family for positive continuous data: log link on the mean `μ`, and
the `sigma` slot carries `σ` = the coefficient of variation, mapped to the shape
`α = 1/σ²` (`coef(fit, :sigma)` is `log σ`; recover the shape as `exp(-2·log σ)`).
Likelihood `Gamma(α, μ/α)`; variance `μ²σ²`. Mirrors `drmTMB`'s `Gamma` family.
Crossed random intercepts on the mean, such as `(1 | g) + (1 | h)`, use the
sparse-Laplace engine when `sigma ~ 1`. A phylogenetic random intercept on the
mean, `phylo(1 | species)`, also uses the sparse-Laplace engine.
One ordinary random intercept or independent numeric slope on `sigma` can have
a predictor-dependent SD, while `mu` stays fixed-effect only. For a slope, use
`sigma ~ 1 + (0 + x | g)` and `sd(g, sigma, coef(x)) ~ z`; the SD describes the
random coefficient on `log(sigma)`, and each row's `x` multiplies that effect.
Here `sigma` is the Gamma coefficient of variation. These routes use per-group
adaptive Gauss-Hermite quadrature and require complete responses and
group-constant SD predictors.
`sigma(fit)` includes the fitted conditional group modes; `predict_parameters`
reports the population-level sigma predictor alongside the separate SD
prediction.

!!! note
    `DRModels.Gamma` shadows `Distributions.Gamma`; qualify the latter if you need it.

```julia
fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gamma(); data = dat)
fit = drm(bf(@formula(y ~ x + (1 | g) + (1 | h)), @formula(sigma ~ 1)), Gamma(); data = dat)
fit_phy = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
              Gamma(); data = dat, tree = tr, se = false)
exp(-2 * coef(fit, :sigma)[1])     # estimated shape α
fit_sigma_sd = drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + (1 | g)), @formula(sd(g, sigma) ~ z)),
                   Gamma(); data = dat)
fit_sigma_slope_sd = drm(
    bf(@formula(y ~ x), @formula(sigma ~ 1 + (0 + x | g)), @formula(sd(g, sigma, coef(x)) ~ z)),
    Gamma(); data = dat)

# Experimental: Gamma random-intercept variational (ELBO) marginal.
# Requires `sigma ~ 1`. Default remains `marginal = :LA` (adaptive GHQ, 5 nodes per group, on an ordinary `(1 | g)`).
fit_va = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gamma();
             data = dat, marginal = :VA)

# TMB-convention Laplace on an ordinary `(1 | g)`, as native drmTMB fits it.
# Requires `sigma ~ 1`; ML only.
fit_lap = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gamma();
              data = dat, marginal = :Laplace)

# Independent random intercept and slope on one factor, `(1 | g) + (0 + x | g)` (#960,
# what drmTMB fits; Laplace). A correlated `(1 + x | g)` is DRModels.jl's own GHQ route
# (drmTMB refuses it for Gamma). `sigma` may carry covariates here.
fit_ind = drm(bf(@formula(y ~ x + (1 | g) + (0 + x | g)), @formula(sigma ~ 1)), Gamma(); data = dat)
exp.(coef(fit_ind, :resd))         # SDs of the intercept and slope effects
```
"""
struct Gamma end

# FX3b: `se = false` is recorded on the fit (`DrmFit.uncertainty`), so
# `convergence_status` does not read the skipped (NaN) covariance as a failure.
@drm_entry function drm(f::DrmFormula, fam::Gamma; se::Bool = true, kwargs...)
    return _withuncertainty(_drm_gamma_fit(f, fam; se = se, kwargs...), se)
end

function _drm_gamma_fit(f::DrmFormula, fam::Gamma; data, tree = nothing, K = nothing,
             A = nothing, Ainv = nothing, coords = nothing, g_tol::Real = 1e-8, se::Bool = true,
             marginal::Symbol = :LA, method = nothing, ids = nothing)
    f = _default_structured_q2_sd(f, fam)   # bare structured (1 + x | g): default `sd(…, coef(·)) ~ 1`
    A, ids = _ainv_as_labelled_A(f, data, A, Ainv, ids)   # `Ainv` (precision) -> labelled covariance `A`
    _drm_entry_guard(f, fam, data; tree = tree, K = K, A = A, coords = coords)   # #1027 #1022 #1010
    _reject_method_as_marginal(fam, method)
    if !isempty(_sd_parts(f)) || !isempty(_sdphylo_parts(f))
        A = _resolve_relatedness_labels(f, data, A, ids; coords = coords)   # #957: match `A` by label (SD routes too)
        coords = _resolve_coords_labels(f, data, coords, ids)   # #957
        return _drm_gamma_sd_regression(f, fam; data, tree, K, A, coords,
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
    # Location–scale: a coupled `(1 | tag | group)` shared by the mean and sigma
    # formulas → one 2×2 group-level covariance fit by the augmented-state engine.
    lc = _ls_coupled_re(rhs[:mu], get(rhs, :sigma, ConstantTerm(1)))
    if lc !== nothing
        _has_offset_term(rhs[:mu]) && _refuse_offset_route(fam, "a coupled location–scale random effect")
        isva && _va_reject(fam, "a coupled location–scale random effect")
        isaghq && _aghq_reject(fam, "a coupled location–scale random effect")
        return _withformula(_fit_locscale_frontend(Val(:gamma), fam, f, rhs, lc, data;
                                                    g_tol = g_tol, se = se,
                                                    tree = tree, K = K, A = A,
                                                    coords = coords), f)
    end
    fixed_mu, re, mv, st, _, _, animal_slope =
        _split_ranef(rhs[:mu]; allow_animal_slope = true)
    fixed_mu, offset_term = _extract_offset(fixed_mu)   # #938: offset(z), coefficient fixed at 1
    offset_term === nothing || (isempty(re) && mv === nothing && st === nothing) ||
        _refuse_offset_route(fam, "a random, structured or `meta_V` term on the mean")
    indep_slopes = st === nothing && length(re) > 1 && _has_indep_slopes(re)
    isaghq && indep_slopes &&
        _aghq_reject(fam, "independent random slopes `(0 + x | g)` (Laplace only, #960)")
    isaghq && !(length(re) > 1 && st === nothing) &&
        _aghq_reject(fam, "this model (Gamma `marginal = :AGHQ` covers crossed random " *
                          "intercepts `(1 | g) + (1 | h)` only, #761)")
    mv === nothing ||
        error("Gamma() does not support meta_V markers")
    for (pname, r) in f.forms          # only the mean may carry a random effect
        pname === :mu && continue
        _, re2, mv2, st2 = _split_ranef(r)
        (isempty(re2) && mv2 === nothing && st2 === nothing) ||
            error("Gamma(): only the mean formula may carry a random effect")
    end
    y, Xμ, nmμ = _design(f.response, fixed_mu, data)
    _, Xσ, nmσ = _design(f.response, get(rhs, :sigma, ConstantTerm(1)), data)
    all(yi -> yi > 0, y) || error("Gamma() requires strictly positive responses")
    if st !== nothing
        isva && _va_reject(fam, "a phylogenetic/structured random effect")
        isempty(re) ||
            error("Gamma() phylo structured effects cannot be combined with ordinary random effects yet")
        # `sigma ~ 1` keeps the scalar-dispersion spine; a covariate `sigma`
        # formula routes to the per-observation log-dispersion path (#164).
        kind, grp = st
        labels = getproperty(data, grp)
        if kind === :animal && animal_slope !== nothing
            matrix = _poisson_structured_cov(:animal, grp, K, A, coords)
            hasproperty(data, animal_slope) || throw(ArgumentError(
                "Gamma animal random-slope predictor `$animal_slope` is absent from data."))
            raw = getproperty(data, animal_slope)
            raw isa AbstractVector{<:Real} && length(raw) == length(y) &&
                all(isfinite, raw) || throw(ArgumentError(
                "Gamma animal random-slope predictor `$animal_slope` must be numeric, finite, and have one value per response."))
            fit = _fit_gamma_animal_slope_laplace(
                fam, y, Xμ, Xσ, Float64.(raw), matrix, labels, nmμ, nmσ,
                grp, g_tol; se)
            return _withformula(fit, f)
        end
        if kind === :phylo
            tree === nothing && error("phylo(1 | $grp) needs `tree = ...`")
            return _withformula(_fit_gamma_phylo_laplace(fam, y, Xμ, Xσ, labels, tree, nmμ, nmσ, grp, g_tol; se = se), f)
        elseif kind === :relmat || kind === :animal || kind === :spatial
            # General user-supplied PD covariance C on the mean intercept
            # (relatedness / animal model / precomputed spatial), with the Gamma
            # shape a fixed nuisance. Reuses the phylo nuisance Laplace spine with
            # the tree precision swapped for C⁻¹ (#167).
            C = if kind === :spatial && coords !== nothing
                (K === nothing && A === nothing) || throw(ArgumentError(
                    "Gamma spatial effects use either `coords = …` or a precomputed `K = …`, not both."))
                _, G = _group_index(labels)
                _spatial_covariance_from_coords(grp, G, coords, nothing)   # shared coords validation + native kernel (default range)
            else
                _poisson_structured_cov(kind, grp, K, A, coords)
            end
            return _withformula(_fit_gamma_relmat_laplace(
                fam, y, Xμ, Xσ, C, labels, nmμ, nmσ, grp, g_tol;
                se = se, normalize = kind === :spatial), f)
        else
            error("Gamma() supports phylo/relmat/animal/spatial(1 | group) among structured markers")
        end
    end
    if !isempty(re)                    # random effect on the log mean → GHQ/Laplace or VA (#136)
        if length(re) > 1
            isva && _va_reject(fam, "crossed/multiple random intercepts")
            # Independent intercepts + slopes `(1 | g) + (0 + x | g)` (#960): the
            # TMB-convention Laplace drmTMB fits. Any sigma formula is admitted.
            indep_slopes &&
                return _withformula(_fit_indep_re_laplace(fam, Val(:gamma), y, Xμ, Xσ, nothing,
                                                          _indep_slope_comps(re, data, length(y)),
                                                          nmμ, nmσ, String[], g_tol), f)
            all(_re_kind(r[1])[1] === :intercept for r in re) ||
                error("Gamma() supports multiple random effects only as crossed/nested intercepts " *
                      "`(1 | g) + (1 | h)` or independent intercepts and slopes `(1 | g) + (0 + x | g)`; " *
                      "a correlated `(1 + x | g)` must be the only random-effect term")
            comps = map(re) do r
                grp = r[2]; gidx, G = _group_index(getproperty(data, grp); col = grp)
                (ones(length(y)), gidx, G, String(grp))
            end
            isaghq && return _withformula(_fit_gamma_crossed_aghq(fam, y, Xμ, Xσ, comps, nmμ, nmσ, g_tol; se = se), f)   # #761
            return _withformula(_fit_gamma_crossed_laplace(fam, y, Xμ, Xσ, comps, nmμ, nmσ, g_tol; se = se), f)
        end
        (rk, var) = _re_kind(re[1][1]); grp = re[1][2]
        gidx, G = _group_index(getproperty(data, grp); col = grp)
        if rk === :intercept           # (1 | g) → 1-D Gauss–Hermite (Laplace) or VA (#136)
            if isva
                size(Xσ, 2) == 1 || _va_reject(fam, "a non-constant `sigma` formula (VA kernels require `sigma ~ 1`)")
                return _withformula(_withmarginal(
                    _fit_gamma_ranef_va(fam, y, Xμ, Xσ, gidx, G, nmμ, nmσ, grp, g_tol), :VA), f)
            end
            return _withformula(_fit_gamma_ranef(fam, y, Xμ, Xσ, gidx, G, nmμ, nmσ, grp, g_tol), f)
        elseif rk === :corr            # (1 + x | g) → correlated 2-D Gauss–Hermite
            isva && _va_reject(fam, "a correlated random slope `(1 + x | g)`")
            _require_finite_data_column(data, var)
            return _withformula(_fit_gamma_corr_ranef(fam, y, Xμ, Xσ, Float64.(getproperty(data, var)), gidx, G, nmμ, nmσ, grp, g_tol), f)
        elseif rk === :slope           # (0 + x | g) → TMB-convention Laplace (independent route)
            isva && _va_reject(fam, "an independent random slope `(0 + x | g)`")
            _require_finite_data_column(data, var)
            return _withformula(_fit_indep_re_laplace(fam, Val(:gamma), y, Xμ, Xσ, nothing,
                                                      _indep_slope_comps(re, data, length(y)),
                                                      nmμ, nmσ, String[], g_tol), f)
        else
            error("Gamma() supports `(1 | g)`, `(0 + x | g)` or `(1 + x | g)` on the mean")
        end
    end
    isva && _va_reject(fam, "no random intercept (fixed-effects-only)")
    return _withformula(_fit_gamma(fam, y, Xμ, Xσ, nmμ, nmσ, g_tol; offset = _offset_for_fit(fam, offset_term, data, size(Xμ, 1))), f)
end

# Predictor-dependent ordinary Gamma random-effect SDs. The Gamma response
# retains its existing log-mean and log-dispersion parameterizations. The
# scalar intercept or slope coefficient has group SD exp(Z_g α); its
# observation-level contribution also uses the intercept/slope loading.
function _gamma_sd_targets(f::DrmFormula)
    targets = _sd_target_parts(f)
    isempty(targets) && !isempty(_sdphylo_parts(f)) || return targets
    return [
        (; key = Symbol("sdphy_", first(item)),
           target = (group = first(item), level = :phylogenetic,
                     member = nothing, dpar = :mu, block = nothing),
           rhs = last(item))
        for item in _sdphylo_parts(f)
    ]
end

function _drm_gamma_animal_intercept_slope_sd(f::DrmFormula, fam::Gamma,
        targets, animal_block; data, tree, K, A, coords, g_tol, se)
    length(targets) == 2 || throw(ArgumentError(
        "Gamma `animal(1 + x | group)` SD regression needs one SD formula for the intercept and one for the slope."))
    group, slope = animal_block.group, animal_block.slope
    all(item -> item.target.level === :animal && item.target.group === group &&
        item.target.dpar === :mu && item.target.block === nothing, targets) ||
        throw(ArgumentError("Gamma independent animal intercept-and-slope SD regression requires two matching `animal` mean targets."))
    members = [item.target.member for item in targets]
    Set(members) == Set((:intercept, slope)) || throw(ArgumentError(
        "Gamma animal SD formulas must select `coef(intercept)` and `coef($slope)` exactly once."))
    (tree === nothing && K === nothing && coords === nothing) || throw(ArgumentError(
        "Gamma animal SD regression uses `A = …` only."))
    A === nothing && throw(ArgumentError(
        "animal(1 + $slope | $group) needs the covariance matrix `A = …`"))
    rhs = Dict(f.forms)
    fixed_mu, re, metav, structured, _, _, parsed_slope = _split_ranef(
        rhs[:mu]; allow_animal_intercept_slope = true)
    metav === nothing && isempty(re) && structured == (:animal, group) &&
        parsed_slope === slope || throw(ArgumentError(
        "Gamma animal SD regression requires one matching independent `animal(1 + $slope | $group)` term."))
    sigma_rhs = get(rhs, :sigma, nothing)
    sigma_rhs isa ConstantTerm && sigma_rhs == ConstantTerm(1) || throw(ArgumentError(
        "Gamma animal SD regression requires a fixed `sigma ~ 1` submodel."))
    allowed = Set(vcat([:mu, :sigma], [item.key for item in targets]))
    all(first(pair) in allowed for pair in f.forms) || throw(ArgumentError(
        "Gamma animal intercept-and-slope SD regression supports only `mu`, fixed `sigma ~ 1`, and its two SD formulas."))
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "Gamma animal SD regression requires observed responses; remove incomplete rows before fitting."))
    structured_markers = _collect_structured(rhs[:mu])
    structured_markers == [(:animal, group)] || throw(ArgumentError(
        "Gamma animal intercept-and-slope SD regression requires exactly one structured term, `animal(1 + $slope | $group)`, with no additional structured effects."))
    y, Xmu, nmmu = _design(f.response, fixed_mu, data)
    _, Xsigma, nmsigma = _design(f.response, sigma_rhs, data)
    all(yi -> yi > 0, y) || throw(ArgumentError("Gamma() requires strictly positive responses"))
    labels = getproperty(data, group)
    Q, gidx = _general_cov_setup(A, labels; normalize = false)
    G = size(Q, 1)
    ordered = map((:intercept, slope)) do member
        matches = filter(item -> item.target.member === member, targets)
        length(matches) == 1 || throw(ArgumentError(
            "Gamma animal SD target `coef($member)` must occur exactly once."))
        only(matches)
    end
    Zgs = Matrix{Float64}[]
    nmsds = Vector{String}[]
    for item in ordered
        Zg, names = _sd_group_design(f.response, item.rhs, data, gidx, G, group;
            schema_cache = f.schema_cache, schema_key = item.key)
        size(Zg, 2) <= G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
            "Gamma animal SD target `coef($(item.target.member))` has a rank-deficient group-level design."))
        push!(Zgs, Matrix{Float64}(Zg))
        push!(nmsds, names)
    end
    hasproperty(data, slope) || throw(ArgumentError(
        "Gamma animal random-slope predictor `$slope` is absent from data."))
    x = getproperty(data, slope)
    x isa AbstractVector{<:Real} && length(x) == length(y) && all(isfinite, x) ||
        throw(ArgumentError("Gamma animal random-slope predictor `$slope` must be numeric, finite, and have one value per response."))
    any(!iszero, x) || throw(ArgumentError(
        "Gamma animal intercept-and-slope SD regression requires `$slope` to load the random slope (all-zero values make its SD unidentified)."))
    if all(==(first(x)), x) && isequal(Zgs[1], Zgs[2])
        throw(ArgumentError(
            "Gamma animal intercept-and-slope SD regression cannot separate the intercept and slope variances when `$slope` is constant and both SD formulas have identical group-level designs."))
    end
    blocks = Symbol[_sd_target_block(item.target; disambiguate = true)
                    for item in ordered]
    return _withformula(_fit_gamma_animal_intercept_slope_laplace(
        fam, y, Xmu, Xsigma, Zgs, gidx, Q, nmmu, nmsigma, nmsds,
        blocks, group, Float64.(x), g_tol; se), f)
end

function _gamma_animal_intercept_slope_mode(y, eta0, load0, load1, gidx, Q,
                                            aux; b0 = nothing,
                                            maxiter::Int = 80,
                                            tol::Real = 1e-10)
    G = size(Q, 1)
    Qjoint = [Q spzeros(G, G); spzeros(G, G) Q]
    b = b0 === nothing ? zeros(2G) : copy(b0)
    function joint(bb)
        b0v, b1v = @view(bb[1:G]), @view(bb[(G + 1):(2G)])
        value = 0.5 * dot(b0v, Q * b0v) + 0.5 * dot(b1v, Q * b1v)
        @inbounds for i in eachindex(y)
            eta = eta0[i] + load0[i] * b0v[gidx[i]] + load1[i] * b1v[gidx[i]]
            value += first(_laplace_v123(Val(:gamma_fixed), aux, i, eta))
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
            _, d1, d2, _ = _laplace_v123(Val(:gamma_fixed), aux, i, eta)
            grad[gi] += load0[i] * d1
            grad[G + gi] += load1[i] * d1
            h00[gi] += load0[i]^2 * d2
            h11[gi] += load1[i]^2 * d2
            h01[gi] += load0[i] * load1[i] * d2
        end
        Hdata = [spdiagm(0 => h00) spdiagm(0 => h01);
                 spdiagm(0 => h01) spdiagm(0 => h11)]
        hessian = Qjoint + Hdata
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

function _gamma_animal_intercept_slope_laplace(y, Xmu, Z0, Z1, gidx, Q,
        logdetQ, theta, aux_from, x; b0 = nothing)
    pmu = size(Xmu, 2)
    beta = @view theta[1:pmu]
    aux = aux_from(clamp(theta[pmu + 1], -4.0, 4.0))
    p0, p1 = size(Z0, 2), size(Z1, 2)
    a0 = @view theta[(pmu + 2):(pmu + 1 + p0)]
    a1 = @view theta[(pmu + 2 + p0):end]
    eta0 = Xmu * beta
    load0 = exp.(Z0[gidx, :] * a0)
    load1 = x .* exp.(Z1[gidx, :] * a1)
    b, ch, ok = _gamma_animal_intercept_slope_mode(
        y, eta0, load0, load1, gidx, Q, aux; b0)
    ok || return 1e18, b, false
    G = size(Q, 1)
    b0v, b1v = @view(b[1:G]), @view(b[(G + 1):(2G)])
    value = 0.5 * dot(b0v, Q * b0v) + 0.5 * dot(b1v, Q * b1v)
    @inbounds for i in eachindex(y)
        eta = eta0[i] + load0[i] * b0v[gidx[i]] + load1[i] * b1v[gidx[i]]
        value += first(_laplace_v123(Val(:gamma_fixed), aux, i, eta))
    end
    return value - logdetQ + 0.5 * logdet(ch), b, true
end

function _fit_gamma_animal_intercept_slope_laplace(fam, y, Xmu, Xsigma,
        Zgs, gidx, Q, nmmu, nmsigma, nmsds, blocks, group, x, g_tol;
        se::Bool = true, maxiter::Int = 250)
    length(Zgs) == 2 || throw(ArgumentError("expected two Gamma animal SD designs"))
    qchol = cholesky(Symmetric(Q); check = false)
    issuccess(qchol) || throw(ArgumentError("Gamma animal covariance matrix must be positive definite"))
    logdetQ = logdet(qchol)
    aux_from, beta0, logsigma0 = _gamma_laplace_setup(y, Xmu)
    pmu = size(Xmu, 2)
    size(Xsigma, 2) == 1 || throw(ArgumentError("Gamma animal SD regression requires `sigma ~ 1`."))
    psd = size.(Zgs, 2)
    ranges = (pmu + 2):(pmu + 1 + psd[1]), (pmu + 2 + psd[1]):(pmu + 1 + sum(psd))
    theta0 = zeros(pmu + 1 + sum(psd))
    theta0[1:pmu] .= beta0
    theta0[pmu + 1] = clamp(logsigma0, -3.9, 3.9)
    theta0[ranges[1]] .= Zgs[1] \ fill(log(0.4), size(Zgs[1], 1))
    theta0[ranges[2]] .= Zgs[2] \ fill(log(0.25), size(Zgs[2], 1))
    last_b = zeros(2 * size(Q, 1))
    function nll(theta)
        value, b, ok = _gamma_animal_intercept_slope_laplace(
            y, Xmu, Zgs[1], Zgs[2], gidx, Q, logdetQ, theta, aux_from, x; b0 = last_b)
        if !ok
            value, b, ok = _gamma_animal_intercept_slope_laplace(
                y, Xmu, Zgs[1], Zgs[2], gidx, Q, logdetQ, theta, aux_from, x;
                b0 = zeros(length(last_b)))
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
    lower, upper = fill(-Inf, length(theta0)), fill(Inf, length(theta0))
    lower[pmu + 1], upper[pmu + 1] = -4.0, 4.0
    od = Optim.OnceDifferentiable(nll, grad!, theta0)
    result = Optim.optimize(od, lower, upper, theta0,
        Optim.Fminbox(Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking())),
        Optim.Options(g_tol = g_tol, iterations = maxiter))
    theta = Optim.minimizer(result)
    nll_hat = nll(theta)
    gf = zeros(length(theta)); grad!(gf, theta)
    near_bound = abs(theta[pmu + 1]) >= 4 -
        _fd_hessian_step(length(y)) * (1 + abs(theta[pmu + 1]))
    V = se && !near_bound ? _vcov_from_fd_hessian(
        _finite_hessian(nll, theta; h = _fd_hessian_step(length(y)));
        context = "Gamma independent animal intercept-and-slope SD regression") :
        fill(NaN, length(theta), length(theta))
    p0, p1 = psd
    block_list = Pair{Symbol,UnitRange{Int}}[
        :mu => 1:pmu, :sigma => (pmu + 1):(pmu + 1),
        blocks[1] => ranges[1], blocks[2] => ranges[2]]
    name_list = Pair{Symbol,Vector{String}}[
        :mu => nmmu, :sigma => nmsigma,
        blocks[1] => nmsds[1], blocks[2] => nmsds[2]]
    sd0 = exp.(Zgs[1] * theta[ranges[1]])
    sd1 = exp.(Zgs[2] * theta[ranges[2]])
    scales = Dict{Symbol,Vector{Float64}}(:sigma => fill(exp(theta[pmu + 1]), length(y)))
    scales[blocks[1]] = sd0[gidx]
    scales[blocks[2]] = sd1[gidx]
    scales[Symbol(string(blocks[1]), "_group")] = sd0
    scales[Symbol(string(blocks[2]), "_group")] = sd1
    fit = DrmFit(fam, block_list, name_list, theta, Matrix(V), -nll_hat,
        length(y), _laplace_outer_converged(result, nll_hat, gf, theta,
        length(y), g_tol), Dict(:mu => exp.(Xmu * theta[1:pmu])),
        Dict(:mu => Vector{Float64}(y)), scales)
    return _withiterations(_withnll(fit, _nll_pinned(nll, theta, nll_hat)), Optim.iterations(result))
end

# A 1-D transect may be given as a plain vector; the shared kernel wants an n-by-1 matrix.
_gamma_coords_matrix(coords) = coords isa AbstractVector{<:Real} ? reshape(Float64.(coords), :, 1) : coords

function _drm_gamma_sd_regression(f::DrmFormula, fam::Gamma; data,
                                 tree = nothing, K = nothing, A = nothing,
                                 coords = nothing, g_tol::Real = 1e-8,
                                 se::Bool = true, marginal::Symbol = :LA,
                                 nAGQ::Int = _SD_IID_AGHQ_K)
    laplace_fit = _laplace_sd_route(f, fam, Val(:gamma), marginal; data, g_tol)
    laplace_fit === nothing || return laplace_fit
    marginal === :LA || throw(ArgumentError(
        "Gamma `sd(group) ~ …` currently supports `marginal = :LA` only."))
    nAGQ >= 1 || throw(ArgumentError("nAGQ must be ≥ 1; got $nAGQ"))
    _ss_has_sigma_structured_sd(f) &&
        return _drm_structured_sigma_sd_laplace(f, fam; data, tree, K, A, coords, g_tol, se)
    targets = _gamma_sd_targets(f)
    twofield = _drm_twofield_sd_route(f, fam; data, tree, K, A, coords, g_tol, se)
    twofield === nothing || return twofield
    if any(item -> item.target.level in (:relmat, :animal, :phylogenetic, :spatial), targets)
        return _drm_gamma_structured_sd_regression(f, fam; data, tree, K, A,
                                                coords, g_tol, se)
    end
    if length(targets) == 2 && count(item -> item.target.dpar === :sigma, targets) == 1
        # Joint mean + sigma ordinary intercepts, each with an SD formula (dense Laplace).
        return _drm_joint_sigma_sd_laplace(f, fam; data, tree, K, A, coords, g_tol, se)
    end
    indep_shape = _indep_sd_shape(f)
    indep_shape === nothing ||
        return _drm_indep_sd_laplace(f, fam, Val(:gamma), indep_shape; data, g_tol)
    length(targets) == 1 || throw(ArgumentError(
        "Gamma random-effect SD regression requires exactly one SD target."))
    target = only(targets)
    if target.target.dpar === :sigma
        return _drm_gamma_sigma_sd_regression(f, fam, target; data, tree, K, A,
            coords, g_tol, se, marginal, nAGQ)
    end
    (target.target.level === :iid && target.target.dpar === :mu &&
     target.target.block === nothing) || throw(ArgumentError(
        "Gamma SD regression currently targets one ordinary mean random-effect coefficient."))
    (tree === nothing && K === nothing && A === nothing && coords === nothing) ||
        throw(ArgumentError("Gamma `sd(group) ~ …` currently supports one ordinary independent random intercept or numeric slope only; structured covariance inputs are not used."))
    isempty(_sdphylo_parts(f)) || throw(ArgumentError(
        "Gamma `sd(group, phylogenetic) ~ …` is not implemented by the ordinary random-intercept route."))
    sgrp, srhs = target.target.group, target.rhs
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    mv === nothing || throw(ArgumentError("Gamma `sd($sgrp)` cannot use meta_V()."))
    st === nothing || throw(ArgumentError(
        "Gamma `sd($sgrp)` currently requires one ordinary `(1 | $sgrp)` or `(0 + x | $sgrp)` effect."))
    length(re) == 1 || throw(ArgumentError(
        "Gamma `sd($sgrp)` currently requires exactly one ordinary independent random effect."))
    kind, slope = _re_kind(re[1][1])
    kind in (:intercept, :slope) || throw(ArgumentError(
        "Gamma `sd($sgrp)` supports `(1 | $sgrp)` or one independent numeric slope `(0 + x | $sgrp)`; correlated blocks are not implemented on this route."))
    member = kind === :intercept ? :intercept : slope
    target.target.member in (nothing, member) || throw(ArgumentError(
        "Gamma SD target `coef($(target.target.member))` does not match fitted coefficient `$(member)`."))
    grp = re[1][2]
    grp === sgrp || throw(ArgumentError(
        "Gamma `sd($sgrp)` must name the random-effect grouping variable `$grp`."))
    sigma_rhs = get(rhs, :sigma, nothing)
    sigma_rhs isa ConstantTerm && sigma_rhs == ConstantTerm(1) || throw(ArgumentError(
        "Gamma `sd($grp)` requires a fixed `sigma ~ 1` submodel."))
    for (pname, _) in f.forms
        (pname in (:mu, :sigma, Symbol("sd_", sgrp)) || haskey(f.sd_targets, pname)) && continue
        throw(ArgumentError("Gamma `sd($grp)` supports only `mu`, fixed `sigma ~ 1`, and one `sd($grp)` formula."))
    end
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "Gamma random-effect SD regression requires observed responses; remove incomplete rows before fitting."))

    y, Xmu, nmmu = _design(f.response, fixed_mu, data)
    _, Xsigma, nmsigma = _design(f.response, sigma_rhs, data)
    all(yi -> yi > 0, y) || throw(ArgumentError("Gamma() requires strictly positive responses"))
    gidx, G = _group_index(getproperty(data, grp))
    Zg, nmsd = _sd_group_design(f.response, srhs, data, gidx, G, grp;
                                schema_cache = f.schema_cache, schema_key = :sd)
    size(Zg, 2) <= G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
        "Gamma `sd($grp)` group-level design is rank deficient; use predictors that vary independently across groups."))
    w = if kind === :intercept
        ones(length(y))
    else
        hasproperty(data, slope) || throw(ArgumentError(
            "Gamma random-slope predictor `$slope` is absent from data."))
        raw = getproperty(data, slope)
        raw isa AbstractVector{<:Real} || throw(ArgumentError(
            "Gamma random-slope predictor `$slope` must be numeric."))
        all(isfinite, raw) || throw(ArgumentError(
            "Gamma random-slope predictor `$slope` must contain finite values."))
        Float64.(raw)
    end
    return _withformula(_fit_gamma_sd_regression(fam, y, Xmu, Xsigma, Zg,
        gidx, G, nmmu, nmsigma, nmsd, grp, g_tol; K = nAGQ, se, w), f)
end

# Predictor-dependent SD for one ordinary random coefficient on Gamma's
# log-CV (sigma) predictor. The response mean remains fixed-effect only.
function _drm_gamma_sigma_sd_regression(f::DrmFormula, fam::Gamma, target;
        data, tree, K, A, coords, g_tol::Real, se::Bool, marginal::Symbol,
        nAGQ::Int)
    marginal === :LA || throw(ArgumentError(
        "Gamma sigma SD regression currently supports `marginal = :LA` only."))
    (tree === nothing && K === nothing && A === nothing && coords === nothing) ||
        throw(ArgumentError(
            "Gamma sigma SD regression uses an ordinary IID coefficient; tree, K, A, and coords are not used."))
    descriptor = target.target
    (descriptor.level === :iid && descriptor.dpar === :sigma &&
     descriptor.block === nothing) ||
        throw(ArgumentError(
            "Gamma sigma SD regression targets one ordinary sigma random coefficient."))
    group, sd_rhs = descriptor.group, target.rhs
    rhs = Dict(f.forms)
    fixed_mu, re_mu, mv_mu, st_mu = _split_ranef(rhs[:mu])
    isempty(re_mu) && mv_mu === nothing && st_mu === nothing ||
        throw(ArgumentError("Gamma sigma SD regression requires fixed-effect-only `mu`."))
    sigma_rhs = get(rhs, :sigma, nothing)
    sigma_rhs === nothing && throw(ArgumentError(
        "Gamma sigma SD regression requires `sigma ~ fixed effects + a random coefficient`."))
    fixed_sigma, re_sigma, mv_sigma, st_sigma = _split_ranef(sigma_rhs)
    mv_sigma === nothing && st_sigma === nothing && length(re_sigma) == 1 ||
        throw(ArgumentError(
            "Gamma sigma SD regression currently supports one ordinary sigma random coefficient."))
    kind, slope = _re_kind(re_sigma[1][1])
    kind in (:intercept, :slope) || throw(ArgumentError(
        "Gamma sigma SD regression supports one independent sigma intercept or numeric slope."))
    sigma_group = re_sigma[1][2]
    sigma_group === group || throw(ArgumentError(
        "`sd($group, sigma)` must target the sigma random-effect group `$sigma_group`."))
    member = kind === :intercept ? nothing : slope
    descriptor.member === member || throw(ArgumentError(
        "Gamma sigma SD target coefficient `$(descriptor.member)` does not match the fitted sigma coefficient `$member`."))
    allowed = Set((:mu, :sigma, target.key))
    all(p -> first(p) in allowed, f.forms) || throw(ArgumentError(
        "Gamma sigma SD regression supports fixed `mu`, one sigma random coefficient, and one sigma-targeted SD formula."))
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "Gamma sigma SD regression requires observed responses; remove incomplete rows before fitting."))

    y, Xmu, nmmu = _design(f.response, fixed_mu, data)
    _, Xsigma, nmsigma = _design(f.response, fixed_sigma, data)
    all(yi -> isfinite(yi) && yi > 0, y) || throw(ArgumentError(
        "Gamma() requires finite, strictly positive responses"))
    gidx, G = _group_index(getproperty(data, group))
    re_design = if kind === :intercept
        ones(length(y))
    else
        hasproperty(data, slope) || throw(ArgumentError(
            "Gamma sigma random-slope predictor `$slope` is absent from data."))
        raw = getproperty(data, slope)
        raw isa AbstractVector{<:Real} && all(isfinite, raw) || throw(ArgumentError(
            "Gamma sigma random-slope predictor `$slope` must be numeric and finite."))
        Float64.(raw)
    end
    Wsd, nmsd = _sd_group_design(f.response, sd_rhs, data, gidx, G, group;
                                 schema_cache = f.schema_cache, schema_key = target.key)
    size(Wsd, 2) <= G && rank(Wsd) == size(Wsd, 2) || throw(ArgumentError(
        "Gamma `sd($group, sigma)` group-level design is rank deficient; use predictors that vary independently across groups."))
    return _withformula(_fit_gamma_sigma_sd_regression(fam, y, Xmu, Xsigma,
        Wsd, gidx, G, nmmu, nmsigma, nmsd, group, g_tol;
        K = nAGQ, se, re_design,
        ranef_name = kind === :intercept ? Symbol(group, "_logsigma") :
            Symbol(group, "_", slope, "_logsigma")), f)
end

function _drm_gamma_structured_sd_regression(f::DrmFormula, fam::Gamma; data,
                                             tree = nothing, K = nothing, A = nothing,
                                             coords = nothing, g_tol::Real = 1e-8,
                                             se::Bool = true)
    targets = _gamma_sd_targets(f)
    rhs = Dict(f.forms)
    animal_block = _animal_intercept_slope(get(rhs, :mu, nothing))
    if animal_block !== nothing && any(item -> item.target.level === :animal, targets)
        return _drm_gamma_animal_intercept_slope_sd(f, fam, targets, animal_block;
            data, tree, K, A, coords, g_tol, se)
    end
    length(targets) == 1 || throw(ArgumentError(
        "Gamma structured SD regression requires exactly one random-effect SD formula."))
    target = only(targets)
    descriptor = target.target
    (descriptor.level in (:relmat, :animal, :phylogenetic, :spatial) && descriptor.dpar === :mu &&
     descriptor.block === nothing) ||
        throw(ArgumentError(
            "Gamma structured SD regression targets one phylogenetic, animal, related-matrix, or spatial mean intercept."))
    provider = descriptor.level
    phylogenetic = provider === :phylogenetic
    structured_kind = phylogenetic ? :phylo : provider
    matrix = provider === :animal ? A : provider === :spatial ? nothing : K
    matrix_name = provider === :animal ? "A" : provider === :spatial ? "coords" : "K"
    # Slope-only `phylo/relmat/spatial(0 + x | g)` is admitted with its matching
    # `sd(g, level) ~ …` target. The slope variable is returned in the 5th slot for
    # phylo and in the 7th slot for animal/relmat/spatial.
    fixed_mu, re, metav, structured, phylo_slope, _, animal_slope =
        _split_ranef(rhs[:mu]; allow_animal_slope = provider === :animal,
                     allow_animal_intercept_slope = provider === :animal,
                     allow_relmat_slope = provider === :relmat,
                     allow_spatial_slope = provider === :spatial,
                     allow_phylo_slope_only = phylogenetic)
    phylo_slope === nothing || (animal_slope = phylo_slope)
    if phylogenetic
        (K === nothing && A === nothing && coords === nothing) || throw(ArgumentError(
            "Gamma phylogenetic SD regression uses `tree = …` only; do not also supply `K`, `A`, or `coords`."))
        tree === nothing && throw(ArgumentError(
            "phylo(1 | $(descriptor.group)) needs `tree = …`"))
    elseif provider === :spatial
        (tree === nothing && A === nothing) || throw(ArgumentError(
            "Gamma spatial SD regression does not use `tree = …` or `A = …`."))
        if animal_slope !== nothing
            # Slope-only spatial: the native fixed kernel exp(-d / median distance)
            # + 1e-6 I enters RAW (not correlation-rescaled), as in native drmTMB.
            coords === nothing && throw(ArgumentError(
                "spatial(0 + $animal_slope | $(descriptor.group)) needs `coords = …`"))
            K === nothing || throw(ArgumentError(
                "Gamma spatial slope SD regression uses `coords = …` only, not a precomputed `K = …`."))
            _, Gs = _group_index(getproperty(data, descriptor.group))
            matrix = _gaussian_spatial_sd_kernel(_gamma_coords_matrix(coords), Gs, descriptor.group)
        elseif coords === nothing
            matrix = _poisson_structured_cov(:spatial, descriptor.group, K, A, nothing)
        else
            K === nothing || throw(ArgumentError(
                "Gamma spatial SD regression uses either `coords = …` or a precomputed `K = …`, not both."))
            # Same fixed native kernel as the slope branch and the q2 routes
            # (`_structured_q2_kernel`): exp(-d / median positive distance) + 1e-6 I.
            _, G = _group_index(getproperty(data, descriptor.group))
            matrix = _gaussian_spatial_sd_kernel(_gamma_coords_matrix(coords), G, descriptor.group)
        end
    else
        tree === nothing && coords === nothing &&
            (provider === :animal ? K === nothing : A === nothing) || throw(ArgumentError(
            "Gamma $(provider) SD regression uses `$matrix_name = …` only; do not supply another covariance provider."))
        matrix === nothing && throw(ArgumentError(
            "$(provider)(1 | $(descriptor.group)) needs the covariance matrix `$matrix_name = …`"))
    end

    metav === nothing || throw(ArgumentError("Gamma structured SD regression cannot use meta_V()."))
    structured == (structured_kind, descriptor.group) || throw(ArgumentError(
        "Gamma `sd($(descriptor.group), $(provider))` requires the matching structured mean effect."))
    if animal_slope === nothing
        descriptor.member in (nothing, :intercept) || throw(ArgumentError(
            "Gamma animal SD target does not match the fitted intercept coefficient."))
    else
        descriptor.member in (nothing, animal_slope) || throw(ArgumentError(
            "Gamma animal SD target does not match fitted slope coefficient `$(animal_slope)`."))
    end
    isempty(re) || throw(ArgumentError(
        "Gamma structured SD regression requires one structured coefficient without additional random effects."))
    (haskey(rhs, :zi) || haskey(rhs, :hu)) && throw(ArgumentError(
        "Gamma structured SD regression cannot be combined with `zi` or `hu`."))
    sigma_rhs = get(rhs, :sigma, nothing)
    sigma_rhs isa ConstantTerm && sigma_rhs == ConstantTerm(1) || throw(ArgumentError(
        "Gamma structured `sd($(descriptor.group))` requires a fixed `sigma ~ 1` submodel."))
    for (pname, _) in f.forms
        (pname === :mu || pname === :sigma || pname === target.key) && continue
        throw(ArgumentError(
            "Gamma structured SD regression supports only `mu`, fixed `sigma ~ 1`, and one structured SD formula."))
    end
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "Gamma structured SD regression requires observed responses; remove incomplete rows before fitting."))

    y, Xmu, nmmu = _design(f.response, fixed_mu, data)
    _, Xsigma, nmsigma = _design(f.response, sigma_rhs, data)
    all(yi -> yi > 0, y) || throw(ArgumentError("Gamma() requires strictly positive responses"))
    group = descriptor.group
    labels = getproperty(data, group)
    if phylogenetic
        phy, tip_index, G = _lss_phylo_group_index(tree, labels, group)
        _, latent_index, checked_phy = _poisson_phylo_setup(phy, labels)
        checked_phy === phy || error("Gamma phylogenetic tree resolution changed unexpectedly")
        Q, leaf_pos, _ = augmented_tree_precision(phy)
        group_index, Gdata = _group_index(labels)
        Gdata == G || error("Gamma phylogenetic tip indexing changed unexpectedly")
        Zgroup, nmsd = _sd_group_design(f.response, target.rhs, data,
            group_index, G, group; schema_cache = f.schema_cache,
            schema_key = target.key)
        tip_group = zeros(Int, G)
        @inbounds for i in eachindex(tip_index)
            ti, gi = tip_index[i], group_index[i]
            tip_group[ti] == 0 || tip_group[ti] == gi || throw(ArgumentError(
                "Gamma phylogenetic SD predictors must be constant within each tree tip."))
            tip_group[ti] = gi
        end
        all(!iszero, tip_group) || throw(ArgumentError(
            "Gamma phylogenetic SD regression requires SD predictors for every tree tip."))
        Zg = Matrix{Float64}(Zgroup[tip_group, :])
        Zobs = Zg[tip_index, :]
        sd_group_index = tip_index
        latent_index == leaf_pos[tip_index] || error(
            "Gamma phylogenetic response rows do not match the augmented-tree leaf ordering")
        gidx = latent_index
        sd_block = :sd_phylo
        scale_key = :sd_phylo
    else
        # relmat/animal/spatial K enters RAW (native drmTMB never rescales K).
        Q, gidx = _general_cov_setup(matrix, labels; normalize = false)
        G = size(Q, 1)
        Zg, nmsd = _sd_group_design(f.response, target.rhs, data, gidx, G, group;
            schema_cache = f.schema_cache, schema_key = target.key)
        Zobs = Zg[gidx, :]
        sd_group_index = gidx
        sd_block = :sd
        scale_key = :sd
    end
    size(Zg, 2) <= G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
        "Gamma structured SD target has a rank-deficient group-level design."))
    loading = if animal_slope === nothing
        ones(Float64, length(y))
    else
        hasproperty(data, animal_slope) || throw(ArgumentError(
            "Gamma animal random-slope predictor `$animal_slope` is absent from data."))
        raw = getproperty(data, animal_slope)
        raw isa AbstractVector{<:Real} && length(raw) == length(y) &&
            all(isfinite, raw) || throw(ArgumentError(
            "Gamma animal random-slope predictor `$animal_slope` must be numeric, finite, and have one value per response."))
        Float64.(raw)
    end
    fit = _fit_gamma_structured_sd_laplace(
        fam, y, Xmu, Xsigma, Zobs, Zg, gidx, Q, nmmu, nmsigma, nmsd,
        group, g_tol; weight = loading, se, sd_group_index, sd_block,
        scale_key)
    return _withformula(fit, f)
end

function _gamma_structured_sd_mode(y, eta0, loading, gidx, Q, aux; b0 = nothing,
                               maxiter::Int = 60, tol::Real = 1e-10)
    q = size(Q, 1)
    b = b0 === nothing ? zeros(q) : copy(b0)
    all(isfinite, loading) || return b, nothing, false
    function joint(bb)
        value = 0.5 * dot(bb, Q * bb)
        @inbounds for i in eachindex(y)
            eta = eta0[i] + loading[i] * bb[gidx[i]]
            value += first(_laplace_v123(Val(:gamma_fixed), aux, i, eta))
        end
        value
    end
    for _ in 1:maxiter
        grad = Q * b
        diag_h = zeros(Float64, q)
        @inbounds for i in eachindex(y)
            gi = gidx[i]
            eta = eta0[i] + loading[i] * b[gi]
            _, d1, d2, _ = _laplace_v123(Val(:gamma_fixed), aux, i, eta)
            grad[gi] += loading[i] * d1
            diag_h[gi] += loading[i]^2 * d2
        end
        hessian = Q + spdiagm(0 => diag_h)
        ch = cholesky(Symmetric(hessian); check = false)
        issuccess(ch) || return b, ch, false
        step = ch \ grad
        norm(step) <= tol * (1 + norm(b)) && return b, ch, true
        # Gamma's data contribution is convex in the latent linear predictor.
        # Near its mode the backtracking objective can stall at rounding error
        # before the strict tolerance is met; take the full Newton step inside
        # the quadratic basin, matching the established structured-Gamma path.
        if norm(step) <= 1e-3 * (1 + norm(b))
            b .-= step
            continue
        end
        f0 = joint(b)
        amount = 1.0
        accepted = false
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

function _gamma_structured_sd_laplace(y, Xmu, Zobs, gidx, Q, logdetQ, theta,
                                  aux_from; weight = ones(length(y)), b0 = nothing)
    pmu = size(Xmu, 2)
    beta = @view theta[1:pmu]
    logsigma = clamp(theta[pmu + 1], -4.0, 4.0)
    alpha = @view theta[(pmu + 2):end]
    aux = aux_from(logsigma)
    eta0 = Xmu * beta
    loading = weight .* exp.(Zobs * alpha)
    b, ch, ok = _gamma_structured_sd_mode(y, eta0, loading, gidx, Q, aux; b0)
    ok || return 1e18, b, false
    value = 0.5 * dot(b, Q * b)
    @inbounds for i in eachindex(y)
        eta = eta0[i] + loading[i] * b[gidx[i]]
        value += first(_laplace_v123(Val(:gamma_fixed), aux, i, eta))
    end
    return value - 0.5 * logdetQ + 0.5 * logdet(ch), b, true
end

function _fit_gamma_structured_sd_laplace(fam, y, Xmu, Xsigma, Zobs, Zg, gidx,
                                      Q, nmmu, nmsigma, nmsd, grp, g_tol;
                                      weight = ones(length(y)), se::Bool = true,
                                      sd_group_index = gidx, sd_block::Symbol = :sd,
                                      scale_key::Symbol = :sd,
                                      maxiter::Int = 250)
    qchol = cholesky(Symmetric(Q); check = false)
    issuccess(qchol) || throw(ArgumentError(
        "Gamma structured covariance matrix must be positive definite"))
    logdetQ = logdet(qchol)
    aux_from, beta0, logsigma0 = _gamma_laplace_setup(y, Xmu)
    pmu, psigma, psd = size(Xmu, 2), size(Xsigma, 2), size(Zobs, 2)
    psigma == 1 || throw(ArgumentError("Gamma structured SD regression requires `sigma ~ 1`."))
    last_b = zeros(size(Q, 1))
    function nll(theta)
            value, b, ok = _gamma_structured_sd_laplace(
            y, Xmu, Zobs, gidx, Q, logdetQ, theta, aux_from; weight, b0 = last_b)
        if !ok
            value, b, ok = _gamma_structured_sd_laplace(
                y, Xmu, Zobs, gidx, Q, logdetQ, theta, aux_from;
                weight, b0 = zeros(length(last_b)))
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
    theta0 = zeros(pmu + psigma + psd)
    theta0[1:pmu] .= beta0
    theta0[pmu + 1] = clamp(logsigma0, -3.9, 3.9)
    theta0[(pmu + 2):end] .= Zg \ fill(log(0.4), size(Zg, 1))
    lower = fill(-Inf, length(theta0))
    upper = fill(Inf, length(theta0))
    lower[pmu + 1] = -4.0
    upper[pmu + 1] = 4.0
    od = Optim.OnceDifferentiable(nll, grad!, theta0)
    result = Optim.optimize(od, lower, upper, theta0,
        Optim.Fminbox(Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking())),
        Optim.Options(g_tol = g_tol, iterations = maxiter))
    theta_hat = Optim.minimizer(result)
    nll_hat = nll(theta_hat)
    gfinal = zeros(length(theta_hat)); grad!(gfinal, theta_hat)
    sigma_index = pmu + 1
    sigma_bound_tol = 1e-6
    # Polish. The Fminbox L-BFGS can stop with a zero step (line search unable to move on
    # the finite-difference gradient) just outside the reporting test of
    # `_laplace_outer_converged`: Gamma phylo `sd-z` of the wave-10 gap map sat at R's
    # log-likelihood to 1e-10 with |grad|/n = 1.01e-6 against the 1e-6 limit. When the test
    # fails and no bound is active, take up to two Newton steps on the finite-difference
    # Hessian and keep each only if the objective does not rise. The test is unchanged.
    if !_laplace_outer_converged(result, nll_hat, gfinal, theta_hat, length(y), g_tol) &&
       isfinite(nll_hat) && nll_hat < 1e17 &&
       abs(theta_hat[sigma_index]) < 4 - 1e-3
        for _ in 1:2
            Hn = _finite_hessian(nll, theta_hat; h = _fd_hessian_step(length(y)))
            chn = cholesky(Symmetric(Matrix(Hn)); check = false)
            issuccess(chn) || break
            trial = theta_hat .- (chn \ gfinal)
            abs(trial[sigma_index]) < 4 - 1e-3 || break
            f_trial = nll(trial)
            (isfinite(f_trial) && f_trial <= nll_hat) || break
            theta_hat, nll_hat = trial, f_trial
            grad!(gfinal, theta_hat)
            _laplace_outer_converged(result, nll_hat, gfinal, theta_hat, length(y), g_tol) && break
        end
    end
    at_lower_sigma_bound = theta_hat[sigma_index] <= -4 + sigma_bound_tol
    at_upper_sigma_bound = theta_hat[sigma_index] >= 4 - sigma_bound_tol
    gcheck = copy(gfinal)
    at_lower_sigma_bound && gcheck[sigma_index] > 0 && (gcheck[sigma_index] = 0.0)
    at_upper_sigma_bound && gcheck[sigma_index] < 0 && (gcheck[sigma_index] = 0.0)
    converged = _laplace_outer_converged(
        result, nll_hat, gcheck, theta_hat, length(y), g_tol)
    h_sigma = _fd_hessian_step(length(y)) * (1 + abs(theta_hat[sigma_index]))
    near_sigma_bound = abs(theta_hat[sigma_index]) >= 4 - h_sigma
    se && near_sigma_bound && @warn(
        "Wald standard errors are withheld because Gamma sigma is at or within one Hessian step of the numerical shape bound")
    V = se && !near_sigma_bound ? _vcov_from_fd_hessian(
        _finite_hessian(nll, theta_hat; h = _fd_hessian_step(length(y)));
        context = "sparse-Laplace Gamma structured SD regression") :
        fill(NaN, length(theta_hat), length(theta_hat))
    blocks = Pair{Symbol,UnitRange{Int}}[
        :mu => (1:pmu), :sigma => (pmu + 1:pmu + 1),
        sd_block => (pmu + 2:pmu + 1 + psd)]
    names = Pair{Symbol,Vector{String}}[
        :mu => nmmu, :sigma => nmsigma, sd_block => nmsd]
    sdgroup = exp.(Zg * theta_hat[(pmu + 2):end])
    means = Dict(:mu => exp.(Xmu * theta_hat[1:pmu]))
    obs = Dict(:mu => Vector{Float64}(y))
    group_key = scale_key === :sd_phylo ? :sd_phylo_group : :sd_group
    index_key = scale_key === :sd_phylo ? :sd_phylo_group_index : :sd_group_index
    scales = Dict(:sigma => fill(exp(theta_hat[pmu + 1]), length(y)),
                  group_key => sdgroup, scale_key => sdgroup[sd_group_index],
                  index_key => Float64.(sd_group_index))
    scale_key === :sd_phylo && (scales[:tree_tip_index] = Float64.(sd_group_index))
    # b34: record the conditional field at theta_hat (fresh inner solve; `last_b`
    # was moved by the finite-difference stencils above). Phylogenetic fields are
    # not exported.
    if scale_key !== :sd_phylo
        _, b_hat, ok_hat = _gamma_structured_sd_laplace(
            y, Xmu, Zobs, gidx, Q, logdetQ, theta_hat, aux_from;
            weight, b0 = zeros(length(last_b)))
        if ok_hat
            loading_hat = Float64.(weight) .* exp.(Zobs * theta_hat[(pmu + 2):end])
            _attach_conditional_structured!(
                scales, loading_hat .* b_hat[gidx], b_hat)
        else
            _mark_cond_failed!(scales)
        end
    end
    fit = DrmFit(fam, blocks, names, theta_hat, Matrix(V), -nll_hat, length(y),
                 converged, means, obs, scales)
    return _withiterations(_withnll(fit, _nll_pinned(nll, theta_hat, nll_hat)), Optim.iterations(result))
end

function _fit_gamma_animal_slope_laplace(fam, y, Xmu, Xsigma, weight, A,
                                         labels, nmmu, nmsigma, group, g_tol;
                                         se::Bool = true)
    Q, gidx = _general_cov_setup(A, labels; normalize = false)
    G = size(Q, 1)
    Zg = ones(Float64, G, 1)
    Zobs = ones(Float64, length(y), 1)
    fit = _fit_gamma_structured_sd_laplace(
        fam, y, Xmu, Xsigma, Zobs, Zg, gidx, Q, nmmu, nmsigma,
        [String(group)], group, g_tol; weight, se)
    blocks = Pair{Symbol,UnitRange{Int}}[
        first(block) === :sd ? :resd => last(block) : block
        for block in fit.blocks]
    names = Pair{Symbol,Vector{String}}[
        first(block) === :sd ? :resd => [String(group)] : block
        for block in fit.coefnames]
    scales = copy(fit.scales)
    sd_range = only(last(block) for block in fit.blocks if first(block) === :sd)
    scales[:resd] = [exp(fit.theta[first(sd_range)])]
    delete!(scales, :sd_group)
    delete!(scales, :sd)
    delete!(scales, :sd_group_index)
    return DrmFit(fit.family, blocks, names, fit.theta, fit.vcov, fit.loglik,
        fit.nobs, fit.converged, fit.means, fit.obs, scales, fit.formula,
        fit.nll, fit.nllgrad, fit.ranef, fit.estim_method, fit.reml_loglik,
        fit.ml_loglik, fit.marginal, fit.phylo_penalty, fit.penalty,
        fit.iterations, fit.phylo_scale)
end

function _fit_gamma_sd_regression(fam::Gamma, y, Xmu, Xsigma, Zg, gidx, G,
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
    Zre = reshape(Float64.(w), n, 1)
    bcache = [zeros(1, G) for _ in 1:Threads.maxthreadid()]
    function nll(theta)
        beta_mu = @view theta[1:pmu]
        beta_sigma = @view theta[(pmu+1):(pmu+psigma)]
        alpha = @view theta[(pmu+psigma+1):(pmu+psigma+psd)]
        eta_mu = Xmu * beta_mu
        eta_sigma = clamp.(Xsigma * beta_sigma, -15.0, 15.0)
        sdgroup = exp.(Zg * alpha)
        cache = bcache[Threads.threadid()]
        ll = (i, eta) -> begin
            mu = exp(clamp(eta, -30.0, 30.0))
            shape = exp(-2 * eta_sigma[i])
            Distributions.logpdf(Distributions.Gamma(shape, mu / shape), y[i])
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
    ybar = sum(y) / n
    variance = sum(abs2, y .- ybar) / max(n - 1, 1)
    shape0 = max(ybar^2 / max(variance, eps()), 0.5)
    theta0 = zeros(pmu + psigma + psd)
    theta0[1] = log(ybar + eps())
    theta0[pmu+1] = -0.5 * log(shape0)
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

function _fit_gamma_sigma_sd_regression(fam::Gamma, y, Xmu, Xsigma, Wsd,
        gidx, G, nmmu, nmsigma, nmsd, grp, g_tol;
        K::Int = _RANEF1D_AGHQ_K, se::Bool = true, re_design = ones(length(y)),
        ranef_name::Symbol = Symbol(grp, "_logsigma"))
    K >= 1 || throw(ArgumentError("nAGQ must be ≥ 1; got $K"))
    n = length(y)
    pmu, psigma, psd = size(Xmu, 2), size(Xsigma, 2), size(Wsd, 2)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    rule = _AGHQRule(1, K)
    Zre = reshape(Float64.(re_design), n, 1)
    mode_cache = [zeros(1, G) for _ in 1:Threads.maxthreadid()]
    function evaluate(theta; return_modes::Bool = false)
        beta_mu = @view theta[1:pmu]
        beta_sigma = @view theta[(pmu+1):(pmu+psigma)]
        alpha = @view theta[(pmu+psigma+1):(pmu+psigma+psd)]
        eta_mu = Xmu * beta_mu
        eta_sigma = clamp.(Xsigma * beta_sigma, -15.0, 15.0)
        sd_group = exp.(Wsd * alpha)
        modes = mode_cache[Threads.threadid()]
        ll = (i, eta) -> begin
            mu = exp(clamp(eta_mu[i], -30.0, 30.0))
            log_sigma = clamp(eta, -15.0, 15.0)
            shape = exp(-2 * log_sigma)
            Distributions.logpdf(Distributions.Gamma(shape, mu / shape), y[i])
        end
        total = zero(eltype(theta))
        for gi in 1:G
            Lg = reshape([sd_group[gi]], 1, 1)
            value, bhat = _aghq_group_logint(ll, members[gi], eta_sigma,
                Zre, Lg, rule, @view(modes[:, gi]))
            modes[1, gi] = isfinite(_aghq_primal(bhat[1])) ? bhat[1] : 0.0
            total += value
        end
        return return_modes ? (total, vec(copy(modes))) : total
    end
    nll(theta) = -first(evaluate(theta))
    ybar = sum(y) / n
    variance = sum(abs2, y .- ybar) / max(n - 1, 1)
    shape0 = max(ybar^2 / max(variance, eps()), 0.5)
    theta0 = zeros(pmu + psigma + psd)
    theta0[1] = log(ybar + eps())
    theta0[pmu+1] = -0.5 * log(shape0)
    theta0[(pmu+psigma+1):end] .= Wsd \ fill(log(0.3), G)
    result = Optim.optimize(nll, theta0, Optim.LBFGS(),
        Optim.Options(g_tol = g_tol); autodiff = :forward)
    theta_hat = Optim.minimizer(result)
    V = se ? _vcov_from_hessian(ForwardDiff.hessian(nll, theta_hat)) :
             fill(NaN, length(theta_hat), length(theta_hat))
    _, mode_group = evaluate(theta_hat; return_modes = true)
    blocks = [:mu => 1:pmu, :sigma => (pmu+1):(pmu+psigma),
              :sd => (pmu+psigma+1):(pmu+psigma+psd)]
    names = [:mu => nmmu, :sigma => nmsigma, :sd => nmsd]
    eta_mu = Xmu * theta_hat[1:pmu]
    eta_sigma_fixed = Xsigma * theta_hat[(pmu+1):(pmu+psigma)]
    sd_group = exp.(Wsd * theta_hat[(pmu+psigma+1):end])
    means = Dict(:mu => exp.(eta_mu))
    observed = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(eta_sigma_fixed .+
                                  Zre[:, 1] .* mode_group[gidx]),
                  :sigma_fixed => exp.(eta_sigma_fixed),
                  :sigma_random_effect => mode_group,
                  :sigma_group_index => Float64.(gidx),
                  :sd_group => sd_group, :sd => sd_group[gidx],
                  :sd_group_index => Float64.(gidx),
                  :aghq_nodes => [Float64(K)])
    fit = _withiterations(_withnll(DrmFit(fam, blocks, names, theta_hat, V,
        -nll(theta_hat), n, Optim.converged(result), means, observed, scales), nll),
        Optim.iterations(result))
    return _withranef(fit, Dict(ranef_name => mode_group))
end

# Gamma GLMM with a random intercept (1|g) on the log mean. b_g ~ N(0,σ_b²)
# integrated out per group by 32-node Gauss–Hermite quadrature; shape α = 1/σ²
# is a fixed effect. Same scheme as the count GLMMs.
function _fit_gamma_ranef(fam::Gamma, y, Xμ, Xσ, gidx, G, nmμ, nmσ, grp, g_tol; K::Int = _RANEF1D_AGHQ_K)
    n = length(y); pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    rule = _AGHQRule(1, K); Zre = ones(n, 1); bcache = zeros(1, G)   # #719: per-group AGHQ
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; σb = exp(θ[pμ+pσ+1])
        η0 = Xμ * βμ; ησ = clamp.(Xσ * βσ, -15.0, 15.0)
        ll = (i, η) -> (μ = exp(clamp(η, -30.0, 30.0)); α = exp(-2 * ησ[i]);
                        Distributions.logpdf(Distributions.Gamma(α, μ / α), y[i]))
        L = reshape([σb], 1, 1)
        return -_aghq_marginal_loglik(ll, members, η0, Zre, L, rule, bcache)
    end
    ȳ = sum(y) / n; v = sum(abs2, y .- ȳ) / max(n - 1, 1)
    α0 = max(ȳ^2 / max(v, eps()), 0.5)
    θ0 = zeros(pμ + pσ + 1)
    θ0[1] = log(ȳ + eps()); θ0[pμ+1] = -0.5 * log(α0); θ0[pμ+pσ+1] = log(0.5)
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :resd => (pμ+pσ+1):(pμ+pσ+1)]
    names = [:mu => nmμ, :sigma => nmσ, :resd => [String(grp)]]
    means = Dict(:mu => exp.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => _reported_sigma(Xσ * θ̂[(pμ+1):(pμ+pσ)], -15.0, 15.0))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# Gamma GLMM with a CORRELATED random intercept+slope (1 + x | g) on the log mean.
# Per group (b0,b1) ~ N(0, Σ_re), Σ_re = L Lᵀ with L = [l11 0; cc l22] (log-Cholesky
# parameters a, b, cc; l11=exp(a), l22=exp(b)). The Gamma marginal has no closed form,
# so the 2-D group integral is taken by per-group ADAPTIVE Gauss–Hermite quadrature
# (`_aghq_marginal_loglik`, #834: nodes b̂_g + √2 C z at each group's mode),
# `nq` nodes per axis. Shape α = 1/σ² is a fixed effect. θ = [β_μ; β_σ; a, b, cc].
function _fit_gamma_corr_ranef(fam::Gamma, y, Xμ, Xσ, xs, gidx, G, nmμ, nmσ, grp, g_tol; nq::Int = _CORR_RANEF_AGHQ_K)
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
        ll = (i, η) -> (μ = exp(clamp(η, -30.0, 30.0)); α = exp(-2 * ησ[i]);
                        Distributions.logpdf(Distributions.Gamma(α, μ / α), y[i]))
        return -_aghq_marginal_loglik(ll, members, η0, Zre, L, rule, bcache)
    end
    ȳ = sum(y) / n; v = sum(abs2, y .- ȳ) / max(n - 1, 1)
    α0 = max(ȳ^2 / max(v, eps()), 0.5)
    θ0 = zeros(pμ + pσ + 3)
    θ0[1] = log(ȳ + eps()); θ0[pμ+1] = -0.5 * log(α0)
    θ0[pμ+pσ+1] = log(0.4); θ0[pμ+pσ+2] = log(0.4); θ0[pμ+pσ+3] = 0.0
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :recov => (pμ+pσ+1):(pμ+pσ+3)]
    names = [:mu => nmμ, :sigma => nmσ, :recov => ["$(grp):L11", "$(grp):L22", "$(grp):L21"]]
    means = Dict(:mu => exp.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => _reported_sigma(Xσ * θ̂[(pμ+1):(pμ+pσ)], -15.0, 15.0))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

function _fit_gamma(fam::Gamma, y, Xμ, Xσ, nmμ, nmσ, g_tol; offset = nothing)
    n = length(y); pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    w = _ctl_weights(n)                  # prior weights: row log-lik multipliers (#931)
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]
        # Soft guards (see `_softclamp` in negbinomial.jl): identity across the
        # original hard band so no legitimate fit is distorted, smooth beyond so
        # the gradient stays live — restoring agreement with drmTMB's smooth guard.
        ημ = _softclamp.(_eta_with_offset(Xμ, βμ, offset), -30.0, 30.0, 3.0)   # μ > 0 finite
        ησ = _softclamp.(Xσ * βσ, -15.0, 15.0, 3.0)   # α = exp(-2ησ) > 0 finite
        s = zero(eltype(θ))
        @inbounds for i in 1:n
            μ = exp(ημ[i]); α = exp(-2 * ησ[i])  # shape = 1/σ²
            s -= w[i] * (Distributions.logpdf(Distributions.Gamma(α, μ / α), y[i]))
        end
        return s
    end
    ȳ = sum(y) / n; v = sum(abs2, y .- ȳ) / max(n - 1, 1)
    α0 = max(ȳ^2 / max(v, eps()), 0.5)            # method-of-moments shape
    θ0 = zeros(pμ + pσ)
    θ0[1] = log(ȳ + eps())                        # log mean
    θ0[pμ+1] = -0.5 * log(α0)                      # σ = 1/√α ⇒ log σ = -½ log α
    θ0[1:pμ] .= _offset_start(θ0[1:pμ], Xμ, offset)   # #938
    res = _ctl_optimize((x0, opts, method) -> Optim.optimize(nll, x0, method, opts; autodiff = :forward),
                       θ0; g_tol = g_tol)   # control: maxit / start / multi_start (#948 #994)
    θ̂ = Optim.minimizer(res); V = _ctl_se(() -> _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂)), θ̂)
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ)]
    names = [:mu => nmμ, :sigma => nmσ]
    means = Dict(:mu => exp.(_eta_with_offset(Xμ, θ̂[1:pμ], offset))); obs = Dict(:mu => Vector{Float64}(y))  # response-scale μ̂
    scales = Dict(:sigma => _reported_sigma(Xσ * θ̂[(pμ+1):(pμ+pσ)], -15.0, 15.0, 3.0))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end
