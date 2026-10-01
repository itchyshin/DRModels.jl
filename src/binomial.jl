# binomial.jl — plain Binomial / Bernoulli family: the classic logistic
# regression / logistic GLMM. Logit link on the mean success probability μ; no
# dispersion parameter (mean-only, like Poisson — see BetaBinomial for the
# overdispersed version). Two response forms: `cbind(successes, failures) ~ x`
# (trials = successes + failures, exactly as drmTMB) or a plain 0/1 Bernoulli
# vector. Fixed effects and a random intercept `(1 | g)` on the mean (logistic
# GLMM). `Distributions.Binomial` is used qualified — DRModels exports its own
# `Binomial` family.

import Distributions

# Numerically stable, UNCLAMPED Binomial(n, logistic(η)) log-likelihood. A clamp of
# η inside the objective makes it exactly flat (zero gradient under AD) wherever any
# |η_i| exceeds the clamp, so an L-BFGS step that overshoots there parks on the
# plateau and reports success with a garbage answer. `k·η − n·log1pexp(η)` is exact
# for any finite η (no `logistic` saturation to 0/1, no log(0)) and AD-safe.
@inline _log1pexp(x) = x > 0 ? x + log1p(exp(-x)) : log1p(exp(x))
@inline _logchoose(n, k) = loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1)
@inline _binomial_logit_ll(n, k, η, lc) = k * η - n * _log1pexp(η) + lc

# Without the clamp a wild line-search probe can reach a point where the AGHQ
# marginal (or its AD gradient) is NaN/Inf; Optim's line search asserts finiteness
# and throws. Map such probes to the repo's 1e18 failed-fit sentinel (zero
# gradient, hugely worse than any real fit) so the line search backtracks instead.
# Used only inside the optimiser call, never for the reported value / Hessian.
@inline _finite_or_sentinel(v::Real) = isfinite(v) ? v : oftype(v, 1e18)
@inline function _finite_or_sentinel(v::ForwardDiff.Dual)
    ok = isfinite(ForwardDiff.value(v)) && all(isfinite, ForwardDiff.partials(v))
    return ok ? v : oftype(v, 1e18)
end
_safe_objective(f) = θ -> _finite_or_sentinel(f(θ))

# L-BFGS with the default HagerZhang line search asserts (`B > A`, `isfinite(phi_c)`)
# on pathological brackets near a runaway optimum and throws, losing the fit. Retry
# from the start with BackTracking (no such assertions) and report whatever it
# reaches; `converged` stays honest (false if the gradient tolerance is not met).
function _optimize_with_fallback(nll, θ0, g_tol)
    f = _safe_objective(nll)
    opts = Optim.Options(g_tol = g_tol)
    try
        return Optim.optimize(f, θ0, Optim.LBFGS(), opts; autodiff = :forward)
    catch e
        e isa AssertionError || rethrow()
        return Optim.optimize(f, θ0, Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking()),
                              opts; autodiff = :forward)
    end
end

# Covariance for the quadrature routes at a runaway optimum (e.g. separated data):
# the AD Hessian can be non-finite, and `_vcov_from_hessian` (deliberately) throws on
# that. Report the repo's NaN-covariance convention (standard errors Inf) with a
# warning so the fit itself is not lost.
function _vcov_or_nan(H::AbstractMatrix)
    if !all(isfinite, H)
        @warn "Hessian at the optimum is not finite (runaway or separated fit): covariance is NaN, standard errors Inf."
        return fill(NaN, size(H))
    end
    return _vcov_from_hessian(H)
end

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

!!! note
    Unlike drmTMB, DRModels.jl does not run a `detectseparation`-style
    separation screen before fitting; a (quasi-)separated logistic fit may
    converge to a diverging boundary estimate without warning.

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
    lc = [_logchoose(nint[i], sint[i]) for i in 1:n]
    function nll(θ)
        ημ = Xμ * θ                                       # unclamped: see `_binomial_logit_ll`
        v = zero(eltype(θ))
        @inbounds for i in 1:n
            v -= _binomial_logit_ll(nint[i], sint[i], ημ[i], lc[i])
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
    lc = [_logchoose(nint[i], sint[i]) for i in 1:n]
    rule = _AGHQRule(1, nq); Zre = ones(n, 1); bcache = zeros(1, G)   # #834/#712/#713: per-group AGHQ
    function nll(θ)
        βμ = θ[1:pμ]; σb = exp(θ[pμ+1])
        L = reshape([σb], 1, 1)
        η0 = Xμ * βμ
        ll = (i, η) -> _binomial_logit_ll(nint[i], sint[i], η, lc[i])
        return -_aghq_marginal_loglik(ll, members, η0, Zre, L, rule, bcache)
    end
    p̄ = clamp(sum(s) / max(sum(ntr), 1), 1e-3, 1 - 1e-3)
    θ0 = zeros(pμ + 1)
    θ0[1] = log(p̄ / (1 - p̄)); θ0[pμ+1] = log(0.5)
    res = _optimize_with_fallback(nll, θ0, g_tol)
    θ̂ = Optim.minimizer(res); V = _vcov_or_nan(ForwardDiff.hessian(nll, θ̂))
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
    lc = [_logchoose(nint[i], sint[i]) for i in 1:n]
    rule = _AGHQRule(2, nq); Zre = hcat(ones(n), Float64.(xs)); bcache = zeros(2, G)   # #834: per-group AGHQ
    function nll(θ)
        βμ = θ[1:pμ]
        L = _corr_ranef_L(θ[pμ+1], θ[pμ+2], θ[pμ+3])
        η0 = Xμ * βμ
        ll = (i, η) -> _binomial_logit_ll(nint[i], sint[i], η, lc[i])
        return -_aghq_marginal_loglik(ll, members, η0, Zre, L, rule, bcache)
    end
    p̄ = clamp(sum(s) / max(sum(ntr), 1), 1e-3, 1 - 1e-3)
    θ0 = zeros(pμ + 3)
    θ0[1] = log(p̄ / (1 - p̄))                                # logit p̄
    θ0[pμ+1] = log(0.4); θ0[pμ+2] = log(0.4); θ0[pμ+3] = 0.0
    res = _optimize_with_fallback(nll, θ0, g_tol)
    θ̂ = Optim.minimizer(res); V = _vcov_or_nan(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :recov => (pμ+1):(pμ+3)]
    names = [:mu => nmμ, :recov => ["$(grp):L11", "$(grp):L22", "$(grp):L21"]]
    means = Dict(:mu => _logistic.(Xμ * θ̂[1:pμ])); obs = Dict(:mu => s ./ ntr)
    scales = Dict(:trials => Float64.(nint))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll),
        Optim.iterations(res))
end
