# betabinomial.jl — Beta-binomial family: successes out of known trials, with
# extra-binomial overdispersion. Two-column response `cbind(successes, failures)`
# (trials = successes + failures), exactly as drmTMB. Logit link on the mean
# success probability μ; the `sigma` slot is σ with precision φ = 1/σ² (same
# mapping as Beta) — likelihood BetaBinomial(n, μφ, (1-μ)φ). `Distributions.
# BetaBinomial` is used qualified — DRModels exports its own `BetaBinomial` family.

import Distributions

"""
    cbind(successes, failures)

Formula marker for a two-column count response, e.g.
`bf(cbind(successes, failures) ~ x, sigma ~ z)` with [`BetaBinomial`](@ref).
Trials are `successes + failures`. Mirrors drmTMB's `cbind(...)` response.
Only inspected structurally on a formula left-hand side.
"""
cbind(a, b) = hcat(a, b)

"""
    BetaBinomial()

Beta-binomial response family — successes out of known trials with
extra-binomial overdispersion. Logit link on the mean success probability `μ`;
the `sigma` slot carries `σ` with precision `φ = 1/σ²` (so `coef(fit, :sigma)`
is `log σ`). Likelihood `BetaBinomial(n, μφ, (1-μ)φ)`. Requires a two-column
response via [`cbind`](@ref). Mirrors `drmTMB`'s `beta_binomial`.
Crossed random intercepts on the mean, such as `(1 | g) + (1 | h)`, use the
sparse-Laplace engine when `sigma ~ 1`. A phylogenetic random intercept on the
mean, `phylo(1 | species)`, also uses the sparse-Laplace engine; both
routes require constant σ (overdispersion).

```julia
fit = drm(bf(cbind(successes, failures) ~ x, sigma ~ 1), BetaBinomial(); data = dat)
fit = drm(bf(cbind(successes, failures) ~ x + (1 | g) + (1 | h), sigma ~ 1), BetaBinomial(); data = dat)
fit_phy = drm(bf(@formula(cbind(successes, failures) ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
              BetaBinomial(); data = dat, tree = tr, se = false)
```
"""
struct BetaBinomial end

function drm(f::DrmFormula, fam::BetaBinomial; data, tree = nothing, g_tol::Real = 1e-8,
             se::Bool = true)
    missing_fit = _fit_observed_response_rows(f, data) do data_observed
        drm(f, fam; data = data_observed, tree = tree, g_tol = g_tol, se = se)
    end
    missing_fit !== nothing && return missing_fit

    f.response2 === nothing &&
        error("BetaBinomial() needs a two-column response: bf(cbind(successes, failures) ~ …)")
    _lss_only_gaussian_guard(f, fam)   # #544: refuse, never silently drop, sd() parts
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    mv === nothing ||
        error("BetaBinomial() does not support meta_V markers")
    for (pname, r) in f.forms          # only the mean may carry a random effect
        pname === :mu && continue
        _, re2, mv2, st2 = _split_ranef(r)
        (isempty(re2) && mv2 === nothing && st2 === nothing) ||
            error("BetaBinomial(): only the mean formula may carry a random effect")
    end
    s = Float64.(getproperty(data, f.response))          # successes
    fl = Float64.(getproperty(data, f.response2))        # failures
    (all(si -> si ≥ 0 && isinteger(si), s) && all(fi -> fi ≥ 0 && isinteger(fi), fl)) ||
        error("BetaBinomial() requires non-negative integer successes and failures")
    ntr = s .+ fl                                        # trials
    _, Xμ, nmμ = _design(f.response, fixed_mu, data)     # successes column is a dummy LHS
    _, Xσ, nmσ = _design(f.response, get(rhs, :sigma, ConstantTerm(1)), data)
    if st !== nothing
        isempty(re) ||
            error("BetaBinomial() phylo structured effects cannot be combined with ordinary random effects yet")
        kind, grp = st
        kind === :phylo ||
            error("BetaBinomial() currently supports only phylo(1 | group) among structured markers")
        tree === nothing && error("phylo(1 | $grp) needs `tree = ...`")
        (size(Xσ, 2) == 1 && all(x -> x == 1.0, @view Xσ[:, 1])) ||
            error("BetaBinomial() phylo(1 | group) currently supports only a constant sigma formula")
        labels = getproperty(data, grp)
        return _withformula(_fit_betabinomial_phylo_laplace(fam, s, ntr, Xμ, labels, tree, nmμ, nmσ, grp, g_tol; se = se), f)
    end
    if !isempty(re)                    # random effect on the logit mean → GHQ/Laplace
        if length(re) > 1
            all(_re_kind(r[1])[1] === :intercept for r in re) ||
                error("BetaBinomial() supports multiple random effects only as crossed/nested intercepts, e.g. `(1 | g) + (1 | h)`")
            (size(Xσ, 2) == 1 && all(x -> x == 1.0, @view Xσ[:, 1])) ||
                error("BetaBinomial() crossed random intercepts currently support only a constant sigma formula")
            comps = map(re) do r
                grp = r[2]; gidx, G = _group_index(getproperty(data, grp))
                (ones(length(s)), gidx, G, String(grp))
            end
            return _withformula(_fit_betabinomial_crossed_laplace(fam, s, ntr, Xμ, comps, nmμ, nmσ, g_tol), f)
        end
        (rk, var) = _re_kind(re[1][1]); grp = re[1][2]; gidx, G = _group_index(getproperty(data, grp))
        if rk === :intercept                              # (1 | g) → 1-D GHQ
            return _withformula(_fit_betabinomial_ranef(fam, s, ntr, Xμ, Xσ, gidx, G, nmμ, nmσ, grp, g_tol), f)
        elseif rk === :corr                               # (1 + x | g) → 2-D GHQ
            xs = Float64.(getproperty(data, var))
            return _withformula(_fit_betabinomial_corr_ranef(fam, s, ntr, Xμ, Xσ, xs, gidx, G, nmμ, nmσ, grp, g_tol), f)
        else
            error("BetaBinomial() supports `(1 | g)` or `(1 + x | g)` random effects on the mean")
        end
    end
    return _withformula(_fit_betabinomial(fam, s, ntr, Xμ, Xσ, nmμ, nmσ, g_tol), f)
end

# Beta-binomial GLMM with a random intercept (1|g) on the logit mean. b_g ~ N(0,σ_b²)
# integrated out per group by 32-node Gauss–Hermite quadrature (b = √2 σ_b z); the
# precision φ = 1/σ² stays a fixed effect. Same scheme as the Gamma/count GLMMs.
function _fit_betabinomial_ranef(fam::BetaBinomial, s, ntr, Xμ, Xσ, gidx, G, nmμ, nmσ, grp, g_tol)
    n = length(s); pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    sint = round.(Int, s); nint = round.(Int, ntr)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    z, w = _gauss_hermite(32); logw = log.(w); K = length(z); rt2 = sqrt(2.0); lπ = log(π)
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; σb = exp(θ[pμ+pσ+1])
        η0 = Xμ * βμ; ησ = clamp.(Xσ * βσ, -15.0, 15.0)
        v = zero(eltype(θ))
        for idx in members
            isempty(idx) && continue
            terms = Vector{eltype(θ)}(undef, K)
            for k in 1:K
                δ = rt2 * σb * z[k]; gll = logw[k]
                for i in idx
                    μ = _logistic(clamp(η0[i] + δ, -15.0, 15.0)); φ = exp(-2 * ησ[i])
                    gll += Distributions.logpdf(Distributions.BetaBinomial(nint[i], μ * φ, (1 - μ) * φ), sint[i])
                end
                terms[k] = gll
            end
            mx = maximum(terms)
            v -= (-0.5 * lπ + mx + log(sum(exp.(terms .- mx))))
        end
        return v
    end
    p̄ = clamp(sum(s) / max(sum(ntr), 1), 1e-3, 1 - 1e-3)
    θ0 = zeros(pμ + pσ + 1)
    θ0[1] = log(p̄ / (1 - p̄))                                # logit p̄
    θ0[pμ+1] = -0.5 * log(10.0)                             # moderate precision init (φ ≈ 10)
    θ0[pμ+pσ+1] = log(0.5)
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :resd => (pμ+pσ+1):(pμ+pσ+1)]
    names = [:mu => nmμ, :sigma => nmσ, :resd => [String(grp)]]
    means = Dict(:mu => _logistic.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => s ./ ntr)   # population μ (b=0)
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]), :trials => Float64.(nint))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# Beta-binomial GLMM with a correlated random intercept+slope (1 + x | g) on the
# logit mean. Per group (b0,b1) ~ N(0, Σ); logit μ_i = Xμ_iᵀβ + b0_g + b1_g·x_i.
# Because groups are disjoint the per-group 2-D integral factorises, so it is done
# by a 2-D Gauss–Hermite tensor grid (K² nodes). Σ is the log-Cholesky
# parameterisation L = [exp(a) 0; cc exp(b)] (the `vc` convention), so vc(fit)
# reconstructs Σ = L Lᵀ. The precision φ = 1/σ² stays a fixed effect.
function _fit_betabinomial_corr_ranef(fam::BetaBinomial, s, ntr, Xμ, Xσ, xs, gidx, G, nmμ, nmσ, grp, g_tol)
    n = length(s); pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    sint = round.(Int, s); nint = round.(Int, ntr)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    z1, w1 = _gauss_hermite(12); lw = log.(w1); K = length(z1); rt2 = sqrt(2.0); lπ = log(π)
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; a = θ[pμ+pσ+1]; b = θ[pμ+pσ+2]; cc = θ[pμ+pσ+3]
        l11 = exp(a); l22 = exp(b); η0 = Xμ * βμ; ησ = clamp.(Xσ * βσ, -15.0, 15.0)
        v = zero(eltype(θ))
        for idx in members
            isempty(idx) && continue
            terms = Vector{eltype(θ)}(undef, K * K)
            t = 0
            for j in 1:K, k in 1:K
                t += 1
                b0 = rt2 * l11 * z1[j]; b1 = rt2 * (cc * z1[j] + l22 * z1[k])   # √2 L z
                gll = lw[j] + lw[k]
                for i in idx
                    μ = _logistic(clamp(η0[i] + b0 + b1 * xs[i], -15.0, 15.0)); φ = exp(-2 * ησ[i])
                    gll += Distributions.logpdf(Distributions.BetaBinomial(nint[i], μ * φ, (1 - μ) * φ), sint[i])
                end
                terms[t] = gll
            end
            mx = maximum(terms)
            v -= (-lπ + mx + log(sum(exp.(terms .- mx))))      # 2-D: -0.5·2·logπ = -logπ
        end
        return v
    end
    p̄ = clamp(sum(s) / max(sum(ntr), 1), 1e-3, 1 - 1e-3)
    θ0 = zeros(pμ + pσ + 3)
    θ0[1] = log(p̄ / (1 - p̄))                                # logit p̄
    θ0[pμ+1] = -0.5 * log(10.0)                             # moderate precision init (φ ≈ 10)
    θ0[pμ+pσ+1] = log(0.4); θ0[pμ+pσ+2] = log(0.4); θ0[pμ+pσ+3] = 0.0
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :recov => (pμ+pσ+1):(pμ+pσ+3)]
    names = [:mu => nmμ, :sigma => nmσ, :recov => ["$(grp):L11", "$(grp):L22", "$(grp):L21"]]
    means = Dict(:mu => _logistic.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => s ./ ntr)
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]), :trials => Float64.(nint))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

function _fit_betabinomial(fam::BetaBinomial, s, ntr, Xμ, Xσ, nmμ, nmσ, g_tol)
    n = length(s); pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    sint = round.(Int, s); nint = round.(Int, ntr)
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]
        ημ = clamp.(Xμ * βμ, -30.0, 30.0)        # μ ∈ (0,1) strictly
        ησ = clamp.(Xσ * βσ, -15.0, 15.0)        # φ = exp(-2ησ) > 0 finite
        v = zero(eltype(θ))
        @inbounds for i in 1:n
            μ = _logistic(ημ[i]); φ = exp(-2 * ησ[i])
            v -= Distributions.logpdf(Distributions.BetaBinomial(nint[i], μ * φ, (1 - μ) * φ), sint[i])
        end
        return v
    end
    p̄ = clamp(sum(s) / max(sum(ntr), 1), 1e-3, 1 - 1e-3)   # overall success rate
    θ0 = zeros(pμ + pσ)
    θ0[1] = log(p̄ / (1 - p̄))                                # logit p̄
    θ0[pμ+1] = -0.5 * log(10.0)                             # moderate precision init (φ ≈ 10)
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ)]
    names = [:mu => nmμ, :sigma => nmσ]
    means = Dict(:mu => _logistic.(Xμ * θ̂[1:pμ]))           # fitted mean success probability
    obs = Dict(:mu => s ./ ntr)                             # observed proportion (for residuals)
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]), :trials => Float64.(nint))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end
