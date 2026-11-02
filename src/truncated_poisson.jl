# truncated_poisson.jl — zero-truncated Poisson family and the shared hurdle
# Poisson spelling. Twin of drmTMB's `truncated_poisson()`. Mirrors
# `TruncatedNegBinomial2` (negbinomial.jl): without an `hu` part it fits the plain
# zero-truncated Poisson to strictly-positive counts; with an `hu` part it IS the
# hurdle Poisson and delegates to the existing `Poisson()` + `hu` route.

"""
    TruncatedPoisson()

Zero-truncated Poisson family for strictly-positive counts (≥ 1) — litter sizes,
group sizes given presence. Log link on the mean `λ`, no dispersion parameter,
conditioned on `y ≥ 1`: `P(k) = Poisson(k; λ) / (1 − e^{-λ})`. Mirrors
`drmTMB`'s `truncated_poisson()`.

```julia
fit = drm(bf(y ~ x), TruncatedPoisson(); data = dat)
```

For one ordinary random intercept or independent numeric mean slope,
`sd(group) ~ z` models its log SD using predictors constant within group. Select
the random-slope coefficient explicitly with `coef(x)`:

```julia
fit = drm(bf(y ~ x + (1 | site), sd(site) ~ habitat),
          TruncatedPoisson(); data = dat)

fit_slope = drm(bf(y ~ x + (0 + x | site),
                   sd(site, coef(x)) ~ habitat),
                TruncatedPoisson(); data = dat)
```

This route uses adaptive one-dimensional quadrature. Correlated, structured and
other random-effect blocks are not yet supported for TruncatedPoisson SD
regression. For a slope, the returned SD describes the random coefficient
multiplied by the row-level value of `x` in the linear predictor.
Missing or `NaN` responses are dropped with a warning, following the package's
observed-row policy; missing predictors still error.

Adding an `hu` part fits the HURDLE Poisson instead: `P(0) = logistic(X_huᵀβ)` and
the positive counts follow this zero-truncated Poisson. Zeros are then allowed in
the response. This is the shared `drmTMB` / `DRModels.jl` spelling of the hurdle
Poisson, and it fits the same likelihood as [`Poisson`](@ref) with `hu`, which is
what the call delegates to (so the returned fit reports `Poisson` as its family).
`Poisson()` + `hu` keeps working.

```julia
fit = drm(bf(y ~ x, hu ~ w), TruncatedPoisson(); data = dat)
```
"""
struct TruncatedPoisson end

function drm(f::DrmFormula, fam::TruncatedPoisson; data, g_tol::Real = 1e-8,
             se::Bool = true, nAGQ::Union{Nothing,Int} = nothing)
    missing_fit = _fit_observed_response_rows(f, data) do data_observed
        drm(f, fam; data = data_observed, g_tol = g_tol, se = se, nAGQ = nAGQ)
    end
    missing_fit !== nothing && return missing_fit

    if !isempty(_sd_parts(f)) || !isempty(_sdphylo_parts(f))
        nodes = nAGQ === nothing ? _RANEF1D_AGHQ_K : nAGQ
        nodes >= 1 || throw(ArgumentError("nAGQ must be ≥ 1; got $nodes"))
        return _drm_truncated_poisson_sd_regression(f, fam; data, g_tol, se,
                                                    nAGQ = nodes)
    end

    nAGQ === nothing || throw(ArgumentError(
        "nAGQ is currently supported for TruncatedPoisson random-effect SD regression only."))

    _lss_only_gaussian_guard(f, fam)   # #544: refuse, never silently drop, sd() parts
    rhs = Dict(f.forms)
    for (_, r) in f.forms
        _, re, mv, st = _split_ranef(r)
        (isempty(re) && mv === nothing && st === nothing) ||
            error("TruncatedPoisson() currently supports fixed effects only")
    end
    # Refuse, never silently drop, a formula part this family does not consume.
    # `bf()` fills in a default `sigma ~ 1`; Poisson has no dispersion, so that
    # intercept-only default is accepted (and ignored, as `Poisson()` does) but any
    # real `sigma` model is refused.
    for (pname, rhs_p) in f.forms
        pname === :sigma && rhs_p isa ConstantTerm && continue
        pname in (:mu, :hu) ||
            error("TruncatedPoisson() supports `mu` and an optional `hu` hurdle " *
                  "part (no dispersion parameter); got an unsupported formula part `$pname`. A `zi` " *
                  "zero-inflation part belongs to Poisson(), whose count " *
                  "component is untruncated -- a different model, not a spelling.")
    end
    # Hurdle Poisson: the likelihood `Poisson()` + `hu` already fits (P(0) =
    # logistic(X_huᵀβ), positive counts zero-truncated Poisson); delegate rather
    # than duplicate it. The fit reports `Poisson` as its family.
    haskey(rhs, :hu) && return drm(f, Poisson(); data = data, g_tol = g_tol, se = se)
    y, Xμ, nmμ = _design(f.response, rhs[:mu], data)
    all(yi -> yi ≥ 1 && isinteger(yi), y) ||
        error("TruncatedPoisson() requires positive integer counts (≥ 1) as the response")
    return _withformula(_fit_truncated_poisson(fam, y, Xμ, nmμ, g_tol; se), f)
end

# Predictor-dependent SD for one ordinary random intercept or independent mean
# slope. The positive-count likelihood, including its zero-truncation
# normalizer, stays inside each row's contribution to the group integral.
function _drm_truncated_poisson_sd_regression(f::DrmFormula,
                                               fam::TruncatedPoisson;
                                               data, g_tol::Real = 1e-8,
                                               se::Bool = true,
                                               nAGQ::Int = _RANEF1D_AGHQ_K)
    targets = _sd_target_parts(f)
    length(targets) == 1 || throw(ArgumentError(
        "TruncatedPoisson random-effect SD regression requires exactly one `sd(group) ~ …` formula."))
    target = only(targets)
    descriptor, grp, sd_rhs = target.target, target.target.group, target.rhs
    (descriptor.level === :iid && descriptor.dpar === :mu &&
     descriptor.block === nothing &&
     (descriptor.member === nothing || descriptor.member isa Symbol)) ||
        throw(ArgumentError(
            "TruncatedPoisson SD regression targets one ordinary mean random-effect coefficient."))
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    mv === nothing && st === nothing && length(re) == 1 || throw(ArgumentError(
        "TruncatedPoisson `sd($grp)` currently requires one ordinary random intercept."))
    kind, slope = _re_kind(re[1][1])
    kind in (:intercept, :slope) || throw(ArgumentError(
        "TruncatedPoisson `sd($grp)` supports one ordinary intercept or one independent numeric slope; correlated and structured effects are not implemented."))
    member = kind === :intercept ? :intercept : slope
    descriptor.member in (nothing, member) || throw(ArgumentError(
        "TruncatedPoisson SD target `coef($(descriptor.member))` does not match random-effect coefficient `$(member)`."))
    (kind === :intercept || descriptor.member === member) || throw(ArgumentError(
        "TruncatedPoisson random slopes require an explicit `coef($slope)` SD target."))
    re[1][2] === grp || throw(ArgumentError(
        "TruncatedPoisson `sd($grp)` must target random-effect grouping variable `$(re[1][2])`."))
    sigma_rhs = get(rhs, :sigma, ConstantTerm(1))
    sigma_rhs isa ConstantTerm && sigma_rhs == ConstantTerm(1) || throw(ArgumentError(
        "TruncatedPoisson `sd($grp)` has no residual dispersion model and requires `sigma ~ 1`."))
    haskey(rhs, :hu) && throw(ArgumentError(
        "TruncatedPoisson random-effect SD regression cannot yet be combined with `hu`."))
    for (pname, _) in f.forms
        (pname in (:mu, :sigma, target.key) || haskey(f.sd_targets, pname)) && continue
        throw(ArgumentError(
            "TruncatedPoisson `sd($grp)` supports `mu`, fixed `sigma ~ 1`, and one `sd($grp)` formula."))
    end
    fixed_mu, offset_term = _extract_offset(fixed_mu)
    offset_term === nothing || throw(ArgumentError(
        "TruncatedPoisson random-effect SD regression does not yet support `offset(...)`."))
    y, Xmu, nmmu = _design(f.response, fixed_mu, data)
    all(yi -> yi >= 1 && isinteger(yi), y) || throw(ArgumentError(
        "TruncatedPoisson() requires positive integer counts (≥ 1) as the response"))
    gidx, G = _group_index(getproperty(data, grp))
    Zg, nmsd = _sd_group_design(f.response, sd_rhs, data, gidx, G, grp;
                                schema_cache = f.schema_cache, schema_key = target.key)
    size(Zg, 2) <= G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
        "TruncatedPoisson `sd($grp)` group-level design is rank deficient; use predictors that vary independently across groups."))
    loading = if kind === :intercept
        nothing
    else
        hasproperty(data, slope) || throw(ArgumentError(
            "TruncatedPoisson random-slope predictor `$slope` is absent from data."))
        raw = getproperty(data, slope)
        raw isa AbstractVector{<:Real} && length(raw) == length(y) && all(isfinite, raw) ||
            throw(ArgumentError(
                "TruncatedPoisson random-slope predictor `$slope` must contain one finite numeric value per response."))
        any(x -> !iszero(x), raw) || throw(ArgumentError(
            "TruncatedPoisson random-slope predictor `$slope` must contain at least one non-zero value."))
        Float64.(raw)
    end
    return _withformula(_fit_truncated_poisson_sd_regression(
        fam, y, Xmu, Zg, loading, gidx, G, nmmu, nmsd, grp, g_tol; se, nAGQ), f)
end

function _fit_truncated_poisson_sd_regression(fam::TruncatedPoisson, y, Xmu,
        Zg, loading, gidx, G, nmmu, nmsd, grp, g_tol; se::Bool = true,
        nAGQ::Int = _RANEF1D_AGHQ_K)
    nAGQ >= 1 || throw(ArgumentError("nAGQ must be ≥ 1; got $nAGQ"))
    n = length(y); pmu, psd = size(Xmu, 2), size(Zg, 2)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    lf = [_logfactorial(round(Int, yi)) for yi in y]
    rule = _AGHQRule(1, nAGQ)
    Zre = loading === nothing ? ones(Float64, n, 1) : reshape(loading, n, 1)
    bcache = zeros(1, G)
    function nll(theta)
        beta = @view theta[1:pmu]
        alpha = @view theta[(pmu+1):(pmu+psd)]
        eta0 = Xmu * beta
        sdgroup = exp.(Zg * alpha)
        ll = (i, eta) -> begin
            etac = clamp(eta, -30.0, 30.0)
            lambda = exp(etac)
            y[i] * etac - lambda - lf[i] - _log1mexp(-lambda)
        end
        total = zero(eltype(theta))
        for gi in 1:G
            Lg = reshape([sdgroup[gi]], 1, 1)
            value, bhat = _aghq_group_logint(ll, members[gi], eta0,
                Zre, Lg, rule, @view(bcache[:, gi]))
            bcache[1, gi] = isfinite(bhat[1]) ? bhat[1] : 0.0
            total += value
        end
        return -total
    end
    theta0 = zeros(pmu + psd)
    theta0[1] = log(max(mean(y) * 0.6, 0.1))
    theta0[(pmu+1):end] .= Zg \ fill(log(0.5), G)
    res = Optim.optimize(nll, theta0, Optim.LBFGS(),
        Optim.Options(g_tol = g_tol); autodiff = :forward)
    theta_hat = Optim.minimizer(res)
    V = se ? _vcov_from_hessian(ForwardDiff.hessian(nll, theta_hat)) :
             fill(NaN, length(theta_hat), length(theta_hat))
    blocks = [:mu => 1:pmu, :sd => (pmu+1):(pmu+psd)]
    names = [:mu => nmmu, :sd => nmsd]
    sdgroup = exp.(Zg * theta_hat[(pmu+1):end])
    means = Dict(:mu => exp.(Xmu * theta_hat[1:pmu]))
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sd_group => sdgroup, :sd => sdgroup[gidx],
                  :sd_group_index => Float64.(gidx),
                  :aghq_nodes => [Float64(nAGQ)])
    return _withiterations(_withnll(DrmFit(fam, blocks, names, theta_hat,
        V, -nll(theta_hat), n, drm_optim_converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

function _fit_truncated_poisson(fam::TruncatedPoisson, y, Xμ, nmμ, g_tol;
                                se::Bool = true)
    n = length(y); pμ = size(Xμ, 2)
    lf = [_logfactorial(round(Int, yi)) for yi in y]
    function nll(θ)
        ημ = clamp.(Xμ * θ, -30.0, 30.0)
        s = zero(eltype(θ))
        @inbounds for i in 1:n
            λ = exp(ημ[i])
            s -= (y[i] * ημ[i] - λ - lf[i]) - _log1mexp(-λ)   # divide out P(0): zero-truncated
        end
        return s
    end
    θ0 = zeros(pμ)
    θ0[1] = log(sum(y) / n + eps())
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res)
    V = se ? _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂)) :
             fill(NaN, length(θ̂), length(θ̂))
    blocks = [:mu => 1:pμ]
    names = [:mu => nmμ]
    means = Dict(:mu => exp.(Xμ * θ̂)); obs = Dict(:mu => Vector{Float64}(y))   # untruncated Poisson mean λ̂
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, drm_optim_converged(res), means, obs, Dict{Symbol,Vector{Float64}}()), nll),
        Optim.iterations(res))
end
