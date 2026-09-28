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
fit = drm(bf(y ~ x, sigma ~ 1, nu ~ 1), Student(); data = dat)
2 + exp(coef(fit, :nu)[1])  # estimated degrees of freedom (ν = 2 + exp(η))
```
"""
struct Student end

function drm(f::DrmFormula, fam::Student; data, g_tol::Real = 1e-8)
    missing_fit = _fit_observed_response_rows(f, data) do data_observed
        drm(f, fam; data = data_observed, g_tol = g_tol)
    end
    missing_fit !== nothing && return missing_fit

    _lss_only_gaussian_guard(f, fam)   # #544: refuse, never silently drop, sd() parts
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
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
    if !isempty(re)                                       # random effect on the mean → GHQ
        length(re) == 1 || error("Student() supports a single random-effect term on the mean")
        (rk, var) = _re_kind(re[1][1]); grp = re[1][2]; gidx, G = _group_index(getproperty(data, grp))
        if rk === :intercept                              # (1 | g) → 1-D GHQ
            return _withformula(_fit_student_ranef(fam, y, Xμ, Xσ, Xν, gidx, G, nmμ, nmσ, nmν, grp, g_tol), f)
        elseif rk === :corr                               # (1 + x | g) → 2-D GHQ
            xs = Float64.(getproperty(data, var))
            return _withformula(_fit_student_corr_ranef(fam, y, Xμ, Xσ, Xν, xs, gidx, G, nmμ, nmσ, nmν, grp, g_tol), f)
        else
            error("Student() supports `(1 | g)` or `(1 + x | g)` random effects on the mean")
        end
    end
    return _withformula(_fit_student(fam, y, Xμ, Xσ, Xν, nmμ, nmσ, nmν, g_tol), f)
end

# Student-t GLMM with a random intercept (1|g) on the mean μ (identity link).
# b_g ~ N(0,σ_b²) is integrated out per group by 32-node Gauss–Hermite quadrature
# (substitution b = √2 σ_b z, logsumexp over K nodes, normaliser −½ logπ); the
# scale σ and degrees of freedom ν stay fixed effects. Same scheme as the NB2/Gamma
# random-intercept GLMMs. θ = [βμ; βσ; βν; log σ_b]. O(n·K) per eval, differentiable.
function _fit_student_ranef(fam::Student, y, Xμ, Xσ, Xν, gidx, G, nmμ, nmσ, nmν, grp, g_tol)
    n = length(y); pμ, pσ, pν = size(Xμ, 2), size(Xσ, 2), size(Xν, 2)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    z, w = _gauss_hermite(32); logw = log.(w); K = length(z); rt2 = sqrt(2.0); lπ = log(π)
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; βν = θ[pμ+pσ+1:pμ+pσ+pν]; σb = exp(θ[pμ+pσ+pν+1])
        η0 = Xμ * βμ; ησ = Xσ * βσ; ην = Xν * βν     # μ identity → no exp clamp on the mean
        s = zero(eltype(θ))
        for idx in members
            isempty(idx) && continue
            terms = Vector{eltype(θ)}(undef, K)
            for k in 1:K
                δ = rt2 * σb * z[k]; gll = logw[k]
                for i in idx
                    μ = η0[i] + δ; zt = (y[i] - μ) * exp(-ησ[i])
                    gll += _student_logpdf_std(zt, ην[i]) - ησ[i]   # location-scale t, − log σ Jacobian
                end
                terms[k] = gll
            end
            mx = maximum(terms)
            s -= (-0.5 * lπ + mx + log(sum(exp.(terms .- mx))))
        end
        return s
    end
    βμ0 = Xμ \ y
    θ0 = zeros(pμ + pσ + pν + 1)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(std(y - Xμ * βμ0) + eps())        # σ init
    θ0[pμ+pσ+1] = log(10.0)                           # ν init (mildly heavy-tailed)
    θ0[pμ+pσ+pν+1] = log(0.5)                         # σ_b init
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :nu => (pμ+pσ+1):(pμ+pσ+pν), :resd => (pμ+pσ+pν+1):(pμ+pσ+pν+1)]
    names = [:mu => nmμ, :sigma => nmσ, :nu => nmν, :resd => [String(grp)]]
    means = Dict(:mu => Xμ * θ̂[1:pμ]); obs = Dict(:mu => Vector{Float64}(y))   # population μ (b=0)
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]),
                  :nu => 2 .+ exp.(Xν * θ̂[(pμ+pσ+1):(pμ+pσ+pν)]))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# Student-t GLMM with a correlated random intercept+slope (1 + x | g) on the mean μ.
# Per group (b0,b1) ~ N(0, Σ); because groups are disjoint the 2-D integral factorises,
# so it is done by a 2-D Gauss–Hermite tensor grid (K² nodes). Σ is the log-Cholesky
# parameterisation L = [exp(a) 0; cc exp(b)] (the `vc` convention), so vc(fit)
# reconstructs Σ = L Lᵀ; (b0,b1) = √2 L z. The scale σ and df ν stay fixed effects.
# θ = [βμ; βσ; βν; a, b, cc]. O(G·K²·group) per eval, fully differentiable.
function _fit_student_corr_ranef(fam::Student, y, Xμ, Xσ, Xν, xs, gidx, G, nmμ, nmσ, nmν, grp, g_tol)
    n = length(y); pμ, pσ, pν = size(Xμ, 2), size(Xσ, 2), size(Xν, 2)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    z1, w1 = _gauss_hermite(12); lw = log.(w1); K = length(z1); rt2 = sqrt(2.0); lπ = log(π)
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; βν = θ[pμ+pσ+1:pμ+pσ+pν]
        a = θ[pμ+pσ+pν+1]; b = θ[pμ+pσ+pν+2]; cc = θ[pμ+pσ+pν+3]
        l11 = exp(a); l22 = exp(b)
        η0 = Xμ * βμ; ησ = Xσ * βσ; ην = Xν * βν
        s = zero(eltype(θ))
        for idx in members
            isempty(idx) && continue
            terms = Vector{eltype(θ)}(undef, K * K)
            t = 0
            for j in 1:K, k in 1:K
                t += 1
                b0 = rt2 * l11 * z1[j]; b1 = rt2 * (cc * z1[j] + l22 * z1[k])   # √2 L z
                gll = lw[j] + lw[k]
                for i in idx
                    μ = η0[i] + b0 + b1 * xs[i]; zt = (y[i] - μ) * exp(-ησ[i])
                    gll += _student_logpdf_std(zt, ην[i]) - ησ[i]
                end
                terms[t] = gll
            end
            mx = maximum(terms)
            s -= (-lπ + mx + log(sum(exp.(terms .- mx))))      # 2-D: -0.5·2·logπ = -logπ
        end
        return s
    end
    βμ0 = Xμ \ y
    θ0 = zeros(pμ + pσ + pν + 3)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(std(y - Xμ * βμ0) + eps())        # σ init
    θ0[pμ+pσ+1] = log(10.0)                           # ν init
    θ0[pμ+pσ+pν+1] = log(0.4); θ0[pμ+pσ+pν+2] = log(0.4); θ0[pμ+pσ+pν+3] = 0.0   # log-Cholesky init
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :nu => (pμ+pσ+1):(pμ+pσ+pν), :recov => (pμ+pσ+pν+1):(pμ+pσ+pν+3)]
    names = [:mu => nmμ, :sigma => nmσ, :nu => nmν, :recov => ["$(grp):L11", "$(grp):L22", "$(grp):L21"]]
    means = Dict(:mu => Xμ * θ̂[1:pμ]); obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]),
                  :nu => 2 .+ exp.(Xν * θ̂[(pμ+pσ+1):(pμ+pσ+pν)]))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

function _fit_student(fam::Student, y, Xμ, Xσ, Xν, nmμ, nmσ, nmν, g_tol)
    n = length(y)
    pμ, pσ, pν = size(Xμ, 2), size(Xσ, 2), size(Xν, 2)
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; βν = θ[pμ+pσ+1:pμ+pσ+pν]
        ημ = Xμ * βμ; ησ = Xσ * βσ; ην = Xν * βν
        s = zero(eltype(θ))
        @inbounds for i in 1:n
            z = (y[i] - ημ[i]) * exp(-ησ[i])
            s -= _student_logpdf_std(z, ην[i]) - ησ[i]       # − log σ Jacobian
        end
        return s
    end
    βμ0 = Xμ \ y
    θ0 = zeros(pμ + pσ + pν)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(std(y - Xμ * βμ0) + eps())       # σ init
    θ0[pμ+pσ+1] = log(10.0)                          # ν init (mildly heavy-tailed)
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :nu => (pμ+pσ+1):(pμ+pσ+pν)]
    names = [:mu => nmμ, :sigma => nmσ, :nu => nmν]
    means = Dict(:mu => Xμ * θ̂[1:pμ]); obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]),
                  :nu => 2 .+ exp.(Xν * θ̂[(pμ+pσ+1):(pμ+pσ+pν)]))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end
