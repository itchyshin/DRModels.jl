# binomial.jl — plain Binomial / Bernoulli family: the classic logistic
# regression / logistic GLMM. Logit link on the mean success probability μ; no
# dispersion parameter (mean-only, like Poisson — see BetaBinomial for the
# overdispersed version). Two response forms: `cbind(successes, failures) ~ x`
# (trials = successes + failures, exactly as drmTMB) or a plain 0/1 Bernoulli
# vector. Fixed effects and a random intercept `(1 | g)` on the mean (logistic
# GLMM). `Distributions.Binomial` is used qualified — DRModels exports its own
# `Binomial` family.

import Distributions

"""
    Binomial()

Binomial response family — successes out of known trials (logistic regression).
Logit link on the mean success probability `μ`; no scale/dispersion parameter
(mean-only, like [`Poisson`](@ref)). Accepts either a two-column response via
[`cbind`](@ref) (`trials = successes + failures`) or a plain `0/1` Bernoulli
vector. Likelihood `Binomial(n, μ)` with `μ = logistic(η)`. Mirrors `drmTMB`'s
`binomial` family. A random intercept `(1 | g)` on the mean fits a logistic
GLMM; crossed intercepts such as `(1 | g) + (1 | h)` use the sparse-Laplace
engine. A phylogenetic random intercept on the mean, `phylo(1 | species)`, also
uses the sparse-Laplace engine.

!!! note
    `DRModels.Binomial` shadows `Distributions.Binomial`; if you need the
    distribution too (e.g. to simulate), qualify it as `Distributions.Binomial`.

```julia
fit = drm(bf(cbind(successes, failures) ~ x), Binomial(); data = dat)   # logistic regression
fit = drm(bf(y ~ x + (1 | g)), Binomial(); data = dat)                  # 0/1 logistic GLMM
fit = drm(bf(cbind(successes, failures) ~ x + (1 | g) + (1 | h)), Binomial(); data = dat)
fit_phy = drm(bf(@formula(cbind(successes, failures) ~ x + phylo(1 | species))),
              Binomial(); data = dat, tree = tr, se = false)
fitted(fit)        # fitted success probabilities μ̂ = logistic(Xβ̂)

# Experimental (#136 Rung 1): Binomial random-intercept variational (ELBO) marginal.
fit_va = drm(bf(@formula(y ~ x + (1 | g))), Binomial(); data = dat, marginal = :VA)

# TMB-convention Laplace on an ordinary `(1 | g)`, as native drmTMB fits it (ML only).
fit_lap = drm(bf(@formula(y ~ x + (1 | g))), Binomial(); data = dat, marginal = :Laplace)
```
"""
struct Binomial end

# `K` / `A` / `coords` are ACCEPTED BUT NOT SUPPORTED: Binomial admits `phylo`
# alone among the structured markers. Without them in the signature a user
# writing `relmat(1 | g)` with `K = K` got a bare
#     MethodError: no method matching drm(::DrmFormula, ::Binomial; K=…)
# — a dispatch failure instead of the explanation this method already carries a
# few lines below. Taking the arguments lets that refusal actually be reached.
function drm(f::DrmFormula, fam::Binomial; data, tree = nothing, K = nothing,
             A = nothing, coords = nothing, g_tol::Real = 1e-8,
             se::Bool = true, marginal::Symbol = :LA, method = nothing)
    _reject_method_as_marginal(fam, method)
    _scalar_laplace_requested(marginal) &&     # Arc 2: TMB-convention Laplace, ordinary (1 | g)
        return _drm_ordinary_laplace(f, fam; data = data, tree = tree, K = K, A = A,
                                     coords = coords, g_tol = g_tol, se = se, method = method)
    missing_fit = _fit_observed_response_rows(f, data) do data_observed
        drm(f, fam; data = data_observed, tree = tree, K = K, A = A, coords = coords,
            g_tol = g_tol, se = se, marginal = marginal, method = method)
    end
    missing_fit !== nothing && return missing_fit

    marg = _marginal_method(marginal)                     # :LA (default) or :VA (#136)
    marg isa AGHQ && _aghq_reject(fam, "this family")
    isva = marg isa Variational
    _lss_only_gaussian_guard(f, fam)   # #544: refuse, never silently drop, sd() parts
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    mv === nothing ||
        error("Binomial() does not support meta_V markers")
    for (pname, r) in f.forms          # Binomial is mean-only — reject any other parameter formula
        pname === :mu && continue
        pname === :sigma || error("Binomial() is mean-only; no sigma/dispersion parameter")
        r === ConstantTerm(1) ||
            error("Binomial() is mean-only; no sigma/dispersion parameter")
    end
    s, ntr = _binomial_response(f, data)
    y, Xμ, nmμ = _design(f.response, fixed_mu, data)      # successes column is a dummy LHS
    if st !== nothing
        isva && _va_reject(fam, "a phylogenetic/structured random effect")
        isempty(re) ||
            error("Binomial() phylo structured effects cannot be combined with ordinary random effects yet")
        kind, grp = st
        kind === :phylo ||
            error("Binomial() currently supports only phylo(1 | group) among structured markers")
        tree === nothing && error("phylo(1 | $grp) needs `tree = ...`")
        labels = getproperty(data, grp)
        return _withformula(_fit_binomial_phylo_laplace(fam, s, ntr, Xμ, labels, tree, nmμ, grp, g_tol; se = se), f)
    end
    if !isempty(re)                                       # random intercept (1|g) → GHQ/Laplace or VA (#136)
        if length(re) > 1
            isva && _va_reject(fam, "crossed/multiple random intercepts")
            all(_re_kind(r[1])[1] === :intercept for r in re) ||
                error("Binomial() supports multiple random effects only as crossed/nested intercepts, e.g. `(1 | g) + (1 | h)`")
            comps = map(re) do r
                grp = r[2]; gidx, G = _group_index(getproperty(data, grp))
                (ones(length(s)), gidx, G, String(grp))
            end
            return _withformula(_fit_binomial_crossed_laplace(fam, s, ntr, Xμ, comps, nmμ, g_tol), f)
        end
        (rk, var) = _re_kind(re[1][1]); grp = re[1][2]; gidx, G = _group_index(getproperty(data, grp))
        if rk === :intercept                              # (1 | g) → 1-D GHQ (Laplace) or VA (#136)
            isva && return _withformula(_withmarginal(
                _fit_binomial_ranef_va(fam, s, ntr, Xμ, gidx, G, nmμ, grp, g_tol), :VA), f)
            return _withformula(_fit_binomial_ranef(fam, s, ntr, Xμ, gidx, G, nmμ, grp, g_tol), f)
        else
            isva && _va_reject(fam, "a correlated random slope `(1 + x | g)`")
            error("Binomial() supports `(1 | g)` on the mean")
        end
    end
    isva && _va_reject(fam, "no random intercept (fixed-effects-only)")
    return _withformula(_fit_binomial(fam, s, ntr, Xμ, nmμ, g_tol), f)
end

# Successes and trials from a 0/1 Bernoulli response or `cbind(successes, failures)`.
# Shared by the default route and the `marginal = :Laplace` route (ordinary_laplace.jl).
function _binomial_response(f::DrmFormula, data)
    if f.response2 === nothing                            # plain 0/1 Bernoulli vector
        s = Float64.(getproperty(data, f.response))
        all(yi -> yi == 0 || yi == 1, s) ||
            error("Binomial() with a single-column response requires a 0/1 (Bernoulli) vector; use cbind(successes, failures) for trial counts")
        ntr = ones(length(s))
    else                                                  # cbind(successes, failures): n = s + f
        s = Float64.(getproperty(data, f.response))       # successes
        fl = Float64.(getproperty(data, f.response2))     # failures
        (all(si -> si ≥ 0 && isinteger(si), s) && all(fi -> fi ≥ 0 && isinteger(fi), fl)) ||
            error("Binomial() requires non-negative integer successes and failures")
        ntr = s .+ fl                                     # trials
    end
    return s, ntr
end

function _fit_binomial(fam::Binomial, s, ntr, Xμ, nmμ, g_tol)
    n = length(s); pμ = size(Xμ, 2)
    sint = round.(Int, s); nint = round.(Int, ntr)
    function nll(θ)
        ημ = clamp.(Xμ * θ, -15.0, 15.0)                  # μ = logistic(η) ∈ (0,1)
        v = zero(eltype(θ))
        @inbounds for i in 1:n
            μ = _logistic(ημ[i])
            v -= Distributions.logpdf(Distributions.Binomial(nint[i], μ), sint[i])
        end
        return v
    end
    p̄ = clamp(sum(s) / max(sum(ntr), 1), 1e-3, 1 - 1e-3)   # overall success rate
    θ0 = zeros(pμ); θ0[1] = log(p̄ / (1 - p̄))               # logit p̄
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ]; names = [:mu => nmμ]
    means = Dict(:mu => _logistic.(Xμ * θ̂))                # fitted success probability
    obs = Dict(:mu => s ./ ntr)                            # observed proportion (for residuals)
    scales = Dict(:trials => Float64.(nint))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# Binomial logistic GLMM with a random intercept (1|g) on the logit mean.
# b_g ~ N(0,σ_b²) is integrated out per group by 32-node Gauss–Hermite
# quadrature (b = √2 σ_b z), the same scheme as the Poisson/Beta GLMMs.
# O(n·K) per evaluation, fully differentiable. θ = [β_μ; log σ_b].
function _fit_binomial_ranef(fam::Binomial, s, ntr, Xμ, gidx, G, nmμ, grp, g_tol)
    n = length(s); pμ = size(Xμ, 2)
    sint = round.(Int, s); nint = round.(Int, ntr)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    z, w = _gauss_hermite(32); logw = log.(w); K = length(z); rt2 = sqrt(2.0); lπ = log(π)
    function nll(θ)
        βμ = θ[1:pμ]; σb = exp(θ[pμ+1])
        η0 = Xμ * βμ
        s_ = zero(eltype(θ))
        for idx in members
            isempty(idx) && continue
            terms = Vector{eltype(θ)}(undef, K)
            for k in 1:K
                δ = rt2 * σb * z[k]; gll = logw[k]
                for i in idx
                    μ = _logistic(clamp(η0[i] + δ, -15.0, 15.0))
                    gll += Distributions.logpdf(Distributions.Binomial(nint[i], μ), sint[i])
                end
                terms[k] = gll
            end
            mx = maximum(terms)
            s_ -= (-0.5 * lπ + mx + log(sum(exp.(terms .- mx))))
        end
        return s_
    end
    p̄ = clamp(sum(s) / max(sum(ntr), 1), 1e-3, 1 - 1e-3)
    θ0 = zeros(pμ + 1)
    θ0[1] = log(p̄ / (1 - p̄)); θ0[pμ+1] = log(0.5)
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :resd => (pμ+1):(pμ+1)]
    names = [:mu => nmμ, :resd => [String(grp)]]
    means = Dict(:mu => _logistic.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => s ./ ntr)   # population μ (b=0)
    scales = Dict(:trials => Float64.(nint))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end
