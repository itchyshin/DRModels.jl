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

function drm(f::DrmFormula, fam::TruncatedPoisson; data, g_tol::Real = 1e-8)
    missing_fit = _fit_observed_response_rows(f, data) do data_observed
        drm(f, fam; data = data_observed, g_tol = g_tol)
    end
    missing_fit !== nothing && return missing_fit

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
    haskey(rhs, :hu) && return drm(f, Poisson(); data = data, g_tol = g_tol)
    y, Xμ, nmμ = _design(f.response, rhs[:mu], data)
    all(yi -> yi ≥ 1 && isinteger(yi), y) ||
        error("TruncatedPoisson() requires positive integer counts (≥ 1) as the response")
    return _withformula(_fit_truncated_poisson(fam, y, Xμ, nmμ, g_tol), f)
end

function _fit_truncated_poisson(fam::TruncatedPoisson, y, Xμ, nmμ, g_tol)
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
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ]
    names = [:mu => nmμ]
    means = Dict(:mu => exp.(Xμ * θ̂)); obs = Dict(:mu => Vector{Float64}(y))   # untruncated Poisson mean λ̂
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, drm_optim_converged(res), means, obs, Dict{Symbol,Vector{Float64}}()), nll),
        Optim.iterations(res))
end
