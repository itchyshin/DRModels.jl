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
GLMM by per-group ADAPTIVE Gauss–Hermite quadrature (`nq` nodes on the mode-
centred scale, #712/#713); crossed intercepts such as `(1 | g) + (1 | h)` use
the sparse-Laplace engine. A phylogenetic random intercept on the mean,
`phylo(1 | species)`, also uses the sparse-Laplace engine. A correlated random
intercept + slope `(1 + x | g)` is fit by the same per-group adaptive
quadrature helper, `nq` nodes per axis (the same scheme as
[`BetaBinomial`](@ref)); it needs within-group variation in `x` — a slope
predictor that is constant within any level of `g` leaves the slope SD and
group-level correlation unidentified, and is refused with an informative
error (mirrors drmTMB's `drm_validate_q2_slope_variation`). An independent
random slope `(0 + x | g)` and `marginal = :VA` on `(1 + x | g)` remain out of
scope.

!!! note "Separation: detect and warn (fixed-effects-only fits)"
    A fixed-effects-only Bernoulli/Binomial fit is screened for (quasi-)complete
    separation by a linear-programming check (Konis 2007, the idea behind R's
    `detectseparation`), at fit time. When some direction `d` makes the signed
    design `X̃ d ≥ 0` on every row, the maximum-likelihood estimate does not
    exist: the fit is still returned, a warning names the affected coefficients,
    and their standard errors are reported as `Inf`. Near separation (a fitted
    probability within 1e-8 of 0 or 1 together with a Wald SE above 1e4) is
    flagged the same way. No refusal and no penalised estimator by default. drmTMB
    runs the same check with the same constants, so the two packages flag the same
    coefficients (the coefficient values themselves are arbitrary stopping points
    and are not expected to agree). Random-intercept routes are not screened.

!!! warning "Crossed intercepts: Laplace is biased low for Bernoulli data"
    The default crossed fit is the Laplace approximation drmTMB and lme4 use. For
    Bernoulli / small-`n` Binomial data with a large random-intercept SD and few
    observations per group it underestimates the log-likelihood (≈9.4 nat on a
    300 × 4 Bernoulli design with σ_g = 2.5) and shrinks σ. When one grouping has
    at most 8 levels, `marginal = :AGHQ` integrates the same model by nested
    adaptive Gauss–Hermite quadrature and corrects this (#761). Its `loglik` is
    not comparable with a Laplace fit's, so `lrtest` refuses mixed `marginal`s.

!!! note
    `DRModels.Binomial` shadows `Distributions.Binomial`; if you need the
    distribution too (e.g. to simulate), qualify it as `Distributions.Binomial`.

```julia
fit = drm(bf(cbind(successes, failures) ~ x), Binomial(); data = dat)   # logistic regression
fit = drm(bf(y ~ x + (1 | g)), Binomial(); data = dat)                  # 0/1 logistic GLMM
fit = drm(bf(cbind(successes, failures) ~ x + (1 | g) + (1 | h)), Binomial(); data = dat)
fit_q = drm(bf(cbind(successes, failures) ~ x + (1 | g) + (1 | h)), Binomial();
            data = dat, marginal = :AGHQ)                   # accurate crossed integral, h ≤ 8 levels
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
    isaghq = marg isa AGHQ                                # :AGHQ: crossed intercepts only (#761)
    isva = marg isa Variational
    _lss_only_gaussian_guard(f, fam)   # #544: refuse, never silently drop, sd() parts
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    isaghq && !(length(re) > 1 && st === nothing) &&
        _aghq_reject(fam, "this model (Binomial `marginal = :AGHQ` covers crossed random intercepts `(1 | g) + (1 | h)` only, #761)")
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
            isaghq && return _withformula(_fit_binomial_crossed_aghq(fam, s, ntr, Xμ, comps, nmμ, g_tol; se = se), f)   # #761
            return _withformula(_fit_binomial_crossed_laplace(fam, s, ntr, Xμ, comps, nmμ, g_tol), f)
        end
        (rk, var) = _re_kind(re[1][1]); grp = re[1][2]; gidx, G = _group_index(getproperty(data, grp))
        if rk === :intercept                              # (1 | g) → 1-D GHQ (Laplace) or VA (#136)
            isva && return _withformula(_withmarginal(
                _fit_binomial_ranef_va(fam, s, ntr, Xμ, gidx, G, nmμ, grp, g_tol), :VA), f)
            return _withformula(_fit_binomial_ranef(fam, s, ntr, Xμ, gidx, G, nmμ, grp, g_tol), f)
        elseif rk === :corr                                # (1 + x | g) → 2-D GHQ tensor (#753)
            isva && _va_reject(fam, "a correlated random slope `(1 + x | g)`")
            xs = Float64.(getproperty(data, var))
            _binomial_check_slope_identifiable(xs, gidx, G, grp)
            return _withformula(_fit_binomial_corr_ranef(fam, s, ntr, Xμ, xs, gidx, G, nmμ, grp, g_tol), f)
        else
            isva && _va_reject(fam, "an independent random slope `(0 + x | g)`")
            error("Binomial() supports `(1 | g)` or `(1 + x | g)` on the mean")
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
    V = _separation_guard(Xμ, s, ntr, θ̂, V, means[:mu], nmμ)   # #731: detect + warn, SE → Inf
    obs = Dict(:mu => s ./ ntr)                            # observed proportion (for residuals)
    scales = Dict(:trials => Float64.(nint))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end

# Default nodes for the Binomial 1-D `(1 | g)` route (#712/#713 grouped-trial
# gap). Was a fixed 32-node grid on the PRIOR scale; with grouped trials (n
# trials/row > 1) and an informative group SD the posterior is a narrow spike
# and the grid missed it by up to 9.73 nat. Swept vs an independent AGHQ-40 on
# the grouped-trial DGP (G=100, 30 obs/group, 20 trials/row, RE SD 0.5–0.8):
# K=1 (Laplace) is the pre-existing multi-nat error, K=3 lands inside 0.01 nat
# with margin. See test/test_binomial_aghq.jl for the sweep.
const _BINOMIAL_RANEF_AGHQ_K = 3

# Binomial logistic GLMM with a random intercept (1|g) on the logit mean.
# b_g ~ N(0,σ_b²) is integrated out per group by adaptive Gauss–Hermite
# quadrature (`_aghq_marginal_loglik`, #834/#712/#713: nodes b̂_g + √2 c z at
# each group's mode), q = 1, `nq` nodes — replaces the old fixed 32-node
# prior-scale grid, which under-covered grouped trials (n trials/row > 1) with
# an informative group SD. θ = [β_μ; log σ_b].
function _fit_binomial_ranef(fam::Binomial, s, ntr, Xμ, gidx, G, nmμ, grp, g_tol; nq::Int = _BINOMIAL_RANEF_AGHQ_K)
    n = length(s); pμ = size(Xμ, 2)
    sint = round.(Int, s); nint = round.(Int, ntr)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    rule = _AGHQRule(1, nq); Zre = ones(n, 1); bcache = zeros(1, G)   # #834/#712/#713: per-group AGHQ
    function nll(θ)
        βμ = θ[1:pμ]; σb = exp(θ[pμ+1])
        L = reshape([σb], 1, 1)
        η0 = Xμ * βμ
        ll = (i, η) -> (μ = _logistic(clamp(η, -15.0, 15.0)); Distributions.logpdf(Distributions.Binomial(nint[i], μ), sint[i]))
        return -_aghq_marginal_loglik(ll, members, η0, Zre, L, rule, bcache)
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

# Identifiability guard for the correlated random slope (1 + x | g) (#753): the
# slope RE variance and the group-level (intercept, slope) correlation are
# unidentified when `xs` is constant within any level of `grp` — mirrors
# drmTMB's `drm_validate_q2_slope_variation` (R/mspl-estimator.R), reimplemented
# here (never vendored: drmTMB is GPL, DRModels.jl is MIT).
function _binomial_check_slope_identifiable(xs, gidx, G, grp)
    seen = [Set{Float64}() for _ in 1:G]
    for i in eachindex(xs)
        push!(seen[gidx[i]], xs[i])
    end
    all(length(s) ≥ 2 for s in seen) ||
        error("Binomial() correlated random slope `(1 + x | g)` needs within-group variation in the " *
              "slope predictor — it is constant within at least one level of `$grp`, so the slope SD " *
              "and the group-level intercept–slope correlation are unidentified. " *
              "Use a predictor that varies within `$grp`.")
end

# Binomial logistic GLMM with a correlated random intercept + slope (1 + x | g)
# on the logit mean (#753). Per group (b0,b1) ~ N(0, Σ); logit μ_i =
# Xμ_iᵀβ + b0_g + b1_g·x_i. Because groups are disjoint the per-group 2-D
# integral factorises; it is done by per-group ADAPTIVE Gauss–Hermite
# quadrature (`_aghq_marginal_loglik`, #834: nodes b̂_g + √2 C z at each
# group's mode), `nq` nodes per axis — the same scheme as
# `_fit_betabinomial_corr_ranef` (betabinomial.jl) minus the precision φ
# (Binomial has no dispersion parameter). Σ is the log-Cholesky
# parameterisation L = [exp(a) 0; cc exp(b)] (the `vc` convention), so vc(fit)
# reconstructs Σ = L Lᵀ.
function _fit_binomial_corr_ranef(fam::Binomial, s, ntr, Xμ, xs, gidx, G, nmμ, grp, g_tol; nq::Int = _CORR_RANEF_AGHQ_K)
    n = length(s); pμ = size(Xμ, 2)
    sint = round.(Int, s); nint = round.(Int, ntr)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    rule = _AGHQRule(2, nq); Zre = hcat(ones(n), Float64.(xs)); bcache = zeros(2, G)   # #834: per-group AGHQ
    function nll(θ)
        βμ = θ[1:pμ]
        L = _corr_ranef_L(θ[pμ+1], θ[pμ+2], θ[pμ+3])
        η0 = Xμ * βμ
        ll = (i, η) -> (μ = _logistic(clamp(η, -15.0, 15.0)); Distributions.logpdf(Distributions.Binomial(nint[i], μ), sint[i]))
        return -_aghq_marginal_loglik(ll, members, η0, Zre, L, rule, bcache)
    end
    p̄ = clamp(sum(s) / max(sum(ntr), 1), 1e-3, 1 - 1e-3)
    θ0 = zeros(pμ + 3)
    θ0[1] = log(p̄ / (1 - p̄))                                # logit p̄
    θ0[pμ+1] = log(0.4); θ0[pμ+2] = log(0.4); θ0[pμ+3] = 0.0
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :recov => (pμ+1):(pμ+3)]
    names = [:mu => nmμ, :recov => ["$(grp):L11", "$(grp):L22", "$(grp):L21"]]
    means = Dict(:mu => _logistic.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => s ./ ntr)
    scales = Dict(:trials => Float64.(nint))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end
