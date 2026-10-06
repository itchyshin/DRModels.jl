# skewnormal.jl — Skew-normal family: location–scale–shape regression with an
# asymmetric (skewed) Gaussian. A formula per parameter — μ (location, identity),
# σ (scale, log link, the `sigma` slot), ν (slant / shape, identity, the `nu`
# slot). The density is Azzalini's skew-normal in internal (ξ, ω, α) form,
#   f(y) = 2·φ((y−ξ)/ω)·Φ(α(y−ξ)/ω) / ω,
# but the PUBLIC parameterisation is the moment form (mirroring drmTMB's
# `skew_normal`): μ is the MEAN of y, σ is the SD of y, and ν is the slant α.
# The internal location/scale are recovered from the moments via
#   δ = α/√(1+α²),  ω = σ/√(1 − 2δ²/π),  ξ = μ − ω·δ·√(2/π).
# As ν → 0 the family collapses to the symmetric Gaussian (and the slant becomes
# weakly identified — see the docstring note). Fixed effects, maximum likelihood.
# `Distributions.Normal` is used qualified for the stable log-pdf / log-cdf of φ/Φ.

import Distributions

"""
    SkewNormal()

Skew-normal response family for continuous, asymmetric data: identity link on
the location `μ`, log link on the scale `σ` (the `sigma` slot), and identity link
on the slant `ν` (the `nu` slot). The public parameterisation is the **moment**
form mirroring `drmTMB`'s `skew_normal`: `μ` is the **mean** of `y`, `σ` is the
**SD** of `y`, and `ν` is Azzalini's slant `α`. These are mapped internally to
the location–scale–slant `(ξ, ω, α)` of the density
`f(y) = 2·φ((y−ξ)/ω)·Φ(α(y−ξ)/ω)/ω`.

`ν = 0` recovers the symmetric [`Gaussian`](@ref); `ν > 0` skews right, `ν < 0`
skews left.

```julia
fit = drm(bf(y ~ x, sigma ~ 1, nu ~ 1), SkewNormal(); data = dat)
coef(fit, :mu)[1]           # mean of y (identity)
exp(coef(fit, :sigma)[1])   # SD of y
coef(fit, :nu)[1]           # estimated slant α (identity)
```

!!! note
    Near `ν = 0` the model is (locally) symmetric Gaussian and the slant is only
    weakly identified — its standard error inflates and recovery is loose. The
    fitter starts `ν` at a small nonzero value seeded by the sample-skewness sign
    to break this symmetry.
"""
struct SkewNormal end

function drm(f::DrmFormula, fam::SkewNormal; data, g_tol::Real = 1e-8)
    _lss_only_gaussian_guard(f, fam)   # #544: refuse, never silently drop, sd() parts
    rhs = Dict(f.forms)
    _, re, mv, st = _split_ranef(rhs[:mu])
    (mv === nothing && st === nothing) ||
        error("SkewNormal() does not support meta_V / structured markers")
    isempty(re) ||
        error("SkewNormal() supports fixed effects only (no random effect on the mean)")
    for (pname, r) in f.forms          # no parameter may carry a random effect
        pname === :mu && continue
        _, re2, mv2, st2 = _split_ranef(r)
        (isempty(re2) && mv2 === nothing && st2 === nothing) ||
            error("SkewNormal() supports fixed effects only")
    end
    y, Xμ, nmμ = _design(f.response, rhs[:mu], data)
    _, Xσ, nmσ = _design(f.response, get(rhs, :sigma, ConstantTerm(1)), data)
    _, Xν, nmν = _design(f.response, get(rhs, :nu, ConstantTerm(1)), data)
    return _withformula(_fit_skewnormal(fam, y, Xμ, Xσ, Xν, nmμ, nmσ, nmν, g_tol), f)
end

# Skew-normal location–scale–shape fit by maximum likelihood. θ = [βμ; βσ; βν]:
# μ = Xμ βμ (mean, identity), log σ = Xσ βσ (SD, log link), ν = Xν βν (slant α,
# identity). Per observation the public (μ, σ, α) are mapped to internal (ξ, ω, α)
# and scored against Azzalini's log-density log 2 + logφ((y−ξ)/ω) + logΦ(α(y−ξ)/ω)
# − log ω. Stable in the tail via Distributions' logpdf/logcdf on Normal().
# O(n) per eval, fully ForwardDiff-differentiable. Mirrors the Student/LogNormal
# fixed fitters.
function _fit_skewnormal(fam::SkewNormal, y, Xμ, Xσ, Xν, nmμ, nmσ, nmν, g_tol)
    n = length(y)
    pμ, pσ, pν = size(Xμ, 2), size(Xσ, 2), size(Xν, 2)
    rt2_π = sqrt(2.0 / π)
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; βν = θ[pμ+pσ+1:pμ+pσ+pν]
        ημ = Xμ * βμ; ησ = Xσ * βσ; ην = Xν * βν      # ημ = mean, ησ = log σ, ην = slant α
        s = zero(eltype(θ))
        @inbounds for i in 1:n
            μ = ημ[i]; σ = exp(ησ[i]); α = ην[i]
            δ = α / sqrt(1 + α^2)
            ω = σ / sqrt(1 - 2 * δ^2 / π)              # δ²<1 ⇒ 1−2δ²/π ∈ (1−2/π, 1] > 0
            ξ = μ - ω * δ * rt2_π
            z = (y[i] - ξ) / ω
            # log f = log2 + logφ(z) + logΦ(α z) − log ω
            s -= log(2.0) + Distributions.logpdf(Distributions.Normal(), z) +
                 Distributions.logcdf(Distributions.Normal(), α * z) - log(ω)
        end
        return s
    end
    βμ0 = Xμ \ y
    resid = y - Xμ * βμ0
    sd0 = std(resid) + eps()
    # Seed the slant from the sample-skewness sign so the α=0 symmetry is broken.
    γ1 = sum(((resid ./ sd0) .^ 3)) / n               # moment skewness estimate
    α0 = clamp(2.0 * sign(γ1 == 0 ? 1.0 : γ1), -4.0, 4.0)   # small, signed, nonzero
    θ0 = zeros(pμ + pσ + pν)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(sd0)                                # σ init (SD scale)
    θ0[pμ+pσ+1] = α0                                   # slant init (identity)
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :nu => (pμ+pσ+1):(pμ+pσ+pν)]
    names = [:mu => nmμ, :sigma => nmσ, :nu => nmν]
    means = Dict(:mu => Xμ * θ̂[1:pμ]); obs = Dict(:mu => Vector{Float64}(y))   # μ = response-scale mean
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]),
                  :nu => Xν * θ̂[(pμ+pσ+1):(pμ+pσ+pν)])
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, drm_optim_converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end
