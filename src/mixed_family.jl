# mixed_family.jl — CROSS-FAMILY bivariate via a shared per-observation latent.
#
#   y1_i ~ fam1(η1_i),  y2_i ~ fam2(η2_i),   η_k = X_k β_k + λ_k u_i,  u_i ~ N(0,1)
#
# The shared scalar latent u_i induces dependence between two responses that may be
# from DIFFERENT families (e.g. Gaussian × Poisson). The 1-D integral over u is done
# by Gauss–Hermite quadrature (K nodes), so the marginal is smooth and
# ForwardDiff-friendly (LBFGS, autodiff=:forward). The dependence is reported on the
# link/latent scale via `link_residual` (Nakagawa & Schielzeth 2010):
#     ρ = λ1 λ2 / sqrt((λ1² + v1) (λ2² + v2)).
# For Gaussian × Gaussian the marginal is exactly bivariate normal, so logLik and ρ
# reduce to the residual-correlation (rho12) model (identifiability note below).
#
# Identifiability: λ1 is fixed positive (λ1 = exp(lλ1)) to remove the u → -u sign
# flip. When BOTH axes are Gaussian the individual loadings are not separately
# identified (a flat λ1 ridge), but the marginal covariance — hence logLik and ρ —
# are; for Gaussian × non-Gaussian (no free residual variance on the non-Gaussian
# axis) all parameters are identified.

# Which families carry a dispersion parameter in the θ vector. The dispersionless
# families (Poisson, Binomial) are fully pinned by η alone; every other family adds
# one slot per axis that is UNIFORMLY `log σ` — Gaussian/Beta/Gamma AND NB2 (the
# locked twin convention, #315/#316: the `sigma` slot is `log σ` everywhere, and the
# NB2 size is θ = 1/σ²). The scale value `d = exp(slot) = σ` is what `_mf_obs_ll`
# receives; each family then maps σ to its own dispersion (Beta φ = 1/σ², Gamma
# α = 1/σ², NB2 θ = 1/σ²).
_mf_has_disp(fam) = !(fam isa Poisson || fam isa Binomial)

# Per-observation conditional log-density at a latent node, dispatched on family.
# `d` is the family's scale σ = exp(slot) for ALL dispersion-carrying families
# (Gaussian/Beta/Gamma/NB2); dispersionless families ignore it (it is passed as 1).
_mf_obs_ll(::Gaussian, η, y, trials, d) =
    -0.5 * ((y - η) / d)^2 - log(d) - 0.9189385332046727  # 0.5*log(2π)
function _mf_obs_ll(::Poisson, η, y, trials, d)
    ηc = clamp(η, -30.0, 30.0)
    return y * ηc - exp(ηc) - loggamma(y + 1)
end
function _mf_obs_ll(::Binomial, η, y, trials, d)
    p = _logistic(clamp(η, -30.0, 30.0))
    return y * log(p) + (trials - y) * log1p(-p) +
           loggamma(trials + 1) - loggamma(y + 1) - loggamma(trials - y + 1)
end
# NB2 (mean μ=exp(η), scale σ=d, size θ=1/σ²): hand-coded log-pmf with loggamma
# (ForwardDiff-safe in θ; no Distributions.NegativeBinomial domain checks). Var =
# μ + μ²/θ. σ is clamped like Gamma/Beta (ησ∈[-15,15]) so wild line-search steps
# cannot push the size to 0/∞. The slot is `log σ` (θ = 1/σ²), matching
# negbinomial.jl's `r = exp(-2·ησ)` and the univariate fitter (#315/#316).
function _mf_obs_ll(::NegBinomial2, η, y, trials, d)
    μ = exp(clamp(η, -30.0, 30.0)); θ = inv(clamp(d, 3.1e-7, 3.3e6)^2)   # θ = 1/σ²
    return loggamma(y + θ) - loggamma(θ) - loggamma(y + 1) +
           θ * (log(θ) - log(θ + μ)) + y * (log(μ) - log(θ + μ))
end
# Beta (mean μ=logistic(η), precision φ=1/σ², σ=d): Beta(μφ, (1-μ)φ). The
# existing beta.jl uses exactly this Distributions.logpdf path under ForwardDiff;
# σ is clamped to beta.jl's range (ησ∈[-15,15]) to keep φ finite & positive.
function _mf_obs_ll(::Beta, η, y, trials, d)
    μ = _logistic(clamp(η, -30.0, 30.0)); φ = inv(clamp(d, 3.1e-7, 3.3e6)^2)
    return Distributions.logpdf(Distributions.Beta(μ * φ, (1 - μ) * φ), y)
end
# Gamma (mean μ=exp(η), shape α=1/σ², σ=d): Gamma(α, μ/α). Matches gamma.jl,
# including the σ clamp (ησ∈[-15,15]) that guards the shape/scale positivity.
function _mf_obs_ll(::Gamma, η, y, trials, d)
    μ = exp(clamp(η, -30.0, 30.0)); α = inv(clamp(d, 3.1e-7, 3.3e6)^2)
    return Distributions.logpdf(Distributions.Gamma(α, μ / α), y)
end

# Fitted mean on the response scale, for the link-residual evaluation.
_mf_mean(::Gaussian, η) = η
_mf_mean(::Poisson, η) = exp(η)
_mf_mean(::Binomial, η) = _logistic(η)
_mf_mean(::NegBinomial2, η) = exp(η)
_mf_mean(::Beta, η) = _logistic(η)
_mf_mean(::Gamma, η) = exp(η)

# Sensible fixed-effect starting values per family (link scale).
_mf_init(::Gaussian, X, y) = X \ y
_mf_init(::Poisson, X, y) = X \ log.(max.(y, 0.5))
_mf_init(::Binomial, X, y) = zeros(size(X, 2))  # logit scale; 0 → p = 0.5
_mf_init(::NegBinomial2, X, y) = X \ log.(max.(y, 0.5))            # log link
_mf_init(::Beta, X, y) = X \ (log.(clamp.(y, 1e-3, 1 - 1e-3)) .-   # logit link
                              log1p.(-clamp.(y, 1e-3, 1 - 1e-3)))
_mf_init(::Gamma, X, y) = X \ log.(max.(y, 1e-3))                  # log link

# Starting value for the `log σ` dispersion slot, per disp-carrying family
# (uniform `log σ` for Gaussian/Beta/Gamma/NB2; #315/#316).
function _mf_disp_init(::Gaussian, y)
    log(max(Statistics.std(y) / 2, 1e-2))                          # log σ
end
function _mf_disp_init(::NegBinomial2, y)                          # log σ = -½ log θ (MoM)
    m = Statistics.mean(y); v = Statistics.var(y)
    θ_MoM = max(m^2 / max(v - m, 0.1 * m + eps()), 0.5)
    -0.5 * log(θ_MoM)
end
function _mf_disp_init(::Beta, y)                                  # log σ = -½ log φ
    ȳ = Statistics.mean(y); v = Statistics.var(y)
    φ0 = max(ȳ * (1 - ȳ) / max(v, eps()) - 1, 0.5)
    -0.5 * log(φ0)
end
function _mf_disp_init(::Gamma, y)                                 # log σ = -½ log α
    ȳ = Statistics.mean(y); v = Statistics.var(y)
    α0 = max(ȳ^2 / max(v, eps()), 0.5)
    -0.5 * log(α0)
end

# Link-scale residual variance v_k for the latent correlation, given the raw
# dispersion slot value (or `nothing` for dispersionless families) and a
# representative fitted mean. Maps the log-native slot to each family's own
# dispersion convention before calling `link_residual`.
_mf_disp_v(fam::Gaussian, slot, μ̂) = link_residual(fam; dispersion = exp(2 * slot))      # σ²
_mf_disp_v(fam::Gamma, slot, μ̂) = link_residual(fam; dispersion = exp(2 * slot))         # σ² → trigamma(1/σ²)
_mf_disp_v(fam::Beta, slot, μ̂) = link_residual(fam, μ̂; dispersion = exp(-2 * slot))      # φ = 1/σ²
_mf_disp_v(fam::NegBinomial2, slot, μ̂) = link_residual(fam; dispersion = exp(-2 * slot)) # θ = 1/σ²
_mf_disp_v(fam::Poisson, slot, μ̂) = link_residual(fam, μ̂)
_mf_disp_v(fam::Binomial, slot, μ̂) = link_residual(fam, μ̂)

# Per-observation response sampler (for the parametric bootstrap CI). `d` is the
# natural dispersion as in `_mf_obs_ll`.
_mf_rand(::Gaussian, η, trials, d, rng) = η + d * randn(rng)
function _mf_rand(::Poisson, η, trials, d, rng)
    λ = exp(clamp(η, -20.0, 20.0))
    L = exp(-λ); k = 0; p = 1.0
    while true
        k += 1
        p *= rand(rng)
        p <= L && return Float64(k - 1)
    end
end
function _mf_rand(::Binomial, η, trials, d, rng)
    p = _logistic(clamp(η, -30.0, 30.0))
    s = 0
    for _ in 1:Int(trials)
        s += rand(rng) < p
    end
    return Float64(s)
end
function _mf_rand(::NegBinomial2, η, trials, d, rng)
    μ = exp(clamp(η, -20.0, 20.0)); θ = inv(d^2); p = θ / (θ + μ)   # d is σ, θ = 1/σ²
    return Float64(rand(rng, Distributions.NegativeBinomial(θ, p)))
end
function _mf_rand(::Beta, η, trials, d, rng)
    μ = _logistic(clamp(η, -30.0, 30.0)); φ = inv(d^2)
    return rand(rng, Distributions.Beta(μ * φ, (1 - μ) * φ))
end
function _mf_rand(::Gamma, η, trials, d, rng)
    μ = exp(clamp(η, -20.0, 20.0)); α = inv(d^2)
    return rand(rng, Distributions.Gamma(α, μ / α))
end

# Profile-likelihood CI on ρ (extracted from `fit_mixed_family` so it can be
# tested against a synthetic objective). `objfn` is the negative log-likelihood,
# `rho_of` maps θ to ρ, `θ̂` the optimum and `ρ` its ρ.
function _mf_profile_ci(objfn, rho_of, θ̂, ρ, level)
    nllhat = objfn(θ̂)
    q = Distributions.quantile(Distributions.Chisq(1), level)
    prof_dev = function (ρ0)
        zr0 = atanh(clamp(ρ0, -0.999999, 0.999999))
        pobj(θ) = objfn(θ) + 1.0e4 * (atanh(clamp(rho_of(θ), -0.999999, 0.999999)) - zr0)^2
        r = Optim.optimize(pobj, copy(θ̂), Optim.LBFGS(),
                           Optim.Options(g_tol = 1e-9); autodiff = :forward)
        # The 1e4 penalty makes this solve stiff. From a far start (the first
        # bisection point, dev >> q) the default backtracking line search can
        # stall within ~1e-9 of stationarity because trial objectives differ by
        # less than the ~1e-12 noise floor at nll ~ 600, and whether it stalls is
        # decided by last-ulp differences (start point, CPU, BLAS kernel). Such a
        # stalled solve was reported NaN, which dropped a whole CI arm. Retry once
        # from the stall point with a Hager-Zhang line search, then accept a solve
        # that is stationary to 1e-6 (an objective change of ~1e-12 at the
        # curvature here) as converged. Solves that already converged are
        # untouched (bit-identical).
        stalled(r) = !Optim.converged(r) && isfinite(Optim.minimum(r)) && Optim.minimum(r) < 1e9
        if stalled(r)
            r = Optim.optimize(pobj, copy(Optim.minimizer(r)),
                               Optim.LBFGS(linesearch = Optim.LineSearches.HagerZhang()),
                               Optim.Options(g_tol = 1e-9); autodiff = :forward)
        end
        inner_ok = Optim.converged(r) || (stalled(r) && Optim.g_residual(r) <= 1e-6)
        # A non-converged, non-finite or sentinel (>= 1e9) inner solve is
        # UNRESOLVED (NaN), never "crossed": the 1e10 plateau would read as a huge
        # deviance and pin the endpoint to the cliff edge.
        # A solve that stopped short of ρ0 (penalty constraint unmet, e.g. at a
        # cliff edge) while showing a deviance below q is also unresolved: it cannot
        # certify "not crossed". A large deviance is still a valid "crossed".
        θm = Optim.minimizer(r)
        fmin = objfn(θm)
        (inner_ok && isfinite(fmin) && fmin < 1e9) || return NaN
        dev = 2 * (fmin - nllhat)
        met = abs(atanh(clamp(rho_of(θm), -0.999999, 0.999999)) - zr0) < 0.05
        (met || dev >= q) || return NaN
        return dev
    end
    endpoint = function (a, b)   # dev(a), dev(b) straddle q → bisect for dev = q
        fa = prof_dev(a) - q
        isnan(fa) && return NaN
        for _ in 1:40
            mid = (a + b) / 2
            fm = prof_dev(mid) - q
            isnan(fm) && return NaN
            if fa * fm <= 0
                b = mid
            else
                a = mid; fa = fm
            end
            abs(b - a) < 1e-3 && break
        end
        (a + b) / 2
    end
    # An unresolved arm is reported NaN (the same "no interval" convention as
    # the Wald / bootstrap CIs); the other arm is unaffected.
    dlo = prof_dev(-0.999)
    dhi = prof_dev(0.999)
    lobnd = isnan(dlo) ? NaN : dlo >= q ? endpoint(-0.999, ρ) : -0.999
    hibnd = isnan(dhi) ? NaN : dhi >= q ? endpoint(ρ, 0.999) : 0.999
    return (lobnd, hibnd)
end

# Percentile bootstrap CI from a vector of refits (`nothing` = threw). Refits that
# did not converge or sit on the 1e10 sentinel (loglik <= -1e9) are skipped.
# Returns `(ci, n_kept)`.
function _mf_boot_ci(fits, B, level)
    ρs = Float64[]
    for fb in fits
        (fb === nothing || !fb.converged || !(fb.loglik > -1e9)) && continue
        push!(ρs, fb.rho_latent)
    end
    filter!(isfinite, ρs)
    if length(ρs) >= max(10, B ÷ 2)
        α = (1 - level) / 2
        return (Statistics.quantile(ρs, α), Statistics.quantile(ρs, 1 - α)), length(ρs)
    end
    return (NaN, NaN), length(ρs)
end

"""
    fit_mixed_family(; y1, X1, fam1, y2, X2, fam2,
                       trials1=ones(n), trials2=ones(n),
                       Xsigma1=ones(n,1), Xsigma2=ones(n,1), K=32, g_tol=1e-6)

Fit the cross-family bivariate model (shared per-observation latent) and return a
`NamedTuple` with fixed effects `β1`/`β2`, loadings `λ1`/`λ2`, per-axis scale
`σ1`/`σ2` (`NaN` for dispersionless axes), dispersion sub-model coefficients
`βσ1`/`βσ2` (`log σ` scale, empty for dispersionless axes), link-scale variances
`v1`/`v2`, the latent-scale correlation `rho_latent`, `loglik`, `converged`,
`iterations`, and the two family instances `fam1`/`fam2` (carried so the post-fit
accessors in `mixed_family_postfit.jl` can recover each axis's inverse link).

`fam1`/`fam2` are DRModels family instances. Supported: `Gaussian`, `Poisson`,
`Binomial`, `NegBinomial2`, `Beta`, `Gamma`. Dispersion-carrying families
(Gaussian/Beta/Gamma/NB2) all carry the scale `σ` in a per-observation `log σ`
SUB-MODEL `log σ_{k,i} = (Xσ_k · β_σk)_i` with `size(Xσ_k, 2)` coefficients;
`Poisson`/`Binomial` are dispersionless. Each family maps σ to its own dispersion
(Beta φ = 1/σ², Gamma α = 1/σ², NB2 size θ = 1/σ²), matching the univariate
fitters and `drmTMB` (#315/#316). `Xsigma1`/`Xsigma2` default to a single intercept
column (`ones(n,1)`), reproducing the intercept-only scalar dispersion. For
dispersion-carrying families `σ1`/`σ2` returns a representative scale
`exp(mean_i log σ_{k,i})` (recover the NB2 size as `θ = 1/σ1²`); the full sub-model
is in `βσ1`/`βσ2`. `trials*` are Binomial denominators (ignored otherwise).
"""
function fit_mixed_family(; y1, X1, fam1, y2, X2, fam2,
        trials1 = ones(length(y1)), trials2 = ones(length(y2)),
        Xsigma1 = ones(length(y1), 1), Xsigma2 = ones(length(y2), 1),
        K::Int = 32, g_tol::Float64 = 1e-6,
        confint::Bool = true, level::Float64 = 0.95,
        profile::Bool = false, B::Int = 0, rng = Random.default_rng(),
        aghq::Bool = false, aghq_K::Int = _RANEF1D_AGHQ_K)
    n = length(y1)
    n == length(y2) || throw(ArgumentError("y1 and y2 must have equal length"))
    p1 = size(X1, 2); p2 = size(X2, 2)
    # s1/s2: does each axis carry a `log σ` dispersion SUB-MODEL? `log σ` for all of
    # Gaussian/Beta/Gamma/NB2 (#315/#316); none for Poisson/Binomial. The scale is a
    # per-observation linear predictor on the `log σ` scale,
    #     log σ_{k,i} = (Xσ_k · β_σk)_i,
    # with q_k = size(Xσ_k, 2) coefficients (default Xσ_k = ones(n,1) ⇒ q_k = 1,
    # i.e. the single-scalar intercept-only model — byte-identical to the previous
    # scalar slot). Dispersionless axes carry no σ coefficients (q_k effectively 0).
    s1 = _mf_has_disp(fam1); s2 = _mf_has_disp(fam2)
    q1 = s1 ? size(Xsigma1, 2) : 0      # # dispersion coefficients on axis 1
    q2 = s2 ? size(Xsigma2, 2) : 0      # # dispersion coefficients on axis 2
    s1 && size(Xsigma1, 1) == n ||
        (s1 && throw(ArgumentError("Xsigma1 must have $n rows")))
    s2 && size(Xsigma2, 1) == n ||
        (s2 && throw(ArgumentError("Xsigma2 must have $n rows")))
    iλ1 = p1 + p2 + 1
    iλ2 = p1 + p2 + 2
    # is1/is2 are now RANGES of length q1/q2 (the β_σ coefficient blocks), or empty.
    is1 = s1 ? ((p1 + p2 + 3):(p1 + p2 + 2 + q1)) : (1:0)
    is2 = s2 ? ((p1 + p2 + 3 + q1):(p1 + p2 + 2 + q1 + q2)) : (1:0)
    ntheta = p1 + p2 + 2 + q1 + q2

    z, w = _gauss_hermite(K)
    rt2 = sqrt(2.0)
    logw = log.(w)
    half_log_pi = 0.5 * log(π)

    # NB: nll's locals are deliberately NOT named β1/λ1/σ1 etc. Sharing names with
    # the post-fit outer variables would box them (closure capture), and the
    # ForwardDiff.hessian below would then write Duals into the returned β1/λ1/σ1.
    function nll(θ)
        bb1 = θ[1:p1]
        bb2 = θ[p1+1:p1+p2]
        ll1 = exp(θ[iλ1]); ll2 = θ[iλ2]
        T = eltype(θ)
        # Per-observation scale σ from the `log σ` sub-model
        # σ_{k,i} = exp((Xσ_k·β_σk)_i). With the default ones-column Xσ this is the
        # constant exp(θ[isk[1]]) for every i — byte-identical to the old scalar.
        sd1v = s1 ? exp.(Xsigma1 * θ[is1]) : nothing
        sd2v = s2 ? exp.(Xsigma2 * θ[is2]) : nothing
        one_T = one(T)
        η1f = X1 * bb1
        η2f = X2 * bb2
        acc = Vector{T}(undef, K)
        total = zero(T)
        @inbounds for i in 1:n
            sd1 = s1 ? sd1v[i] : one_T   # scale σ at obs i (θ = 1/σ² for NB2)
            sd2 = s2 ? sd2v[i] : one_T
            for k in 1:K
                u = rt2 * z[k]
                ll = _mf_obs_ll(fam1, η1f[i] + ll1 * u, y1[i], trials1[i], sd1) +
                     _mf_obs_ll(fam2, η2f[i] + ll2 * u, y2[i], trials2[i], sd2)
                acc[k] = logw[k] + ll
            end
            mx = maximum(acc)
            total -= mx + log(sum(exp.(acc .- mx))) - half_log_pi
        end
        # Finite-penalty guard: aggressive line-search steps (e.g. large Poisson
        # loadings) can make the GHQ objective overflow to NaN/Inf mid-search. Map
        # a non-finite total to a large finite penalty of the SAME Dual type so
        # LBFGS backtracks instead of aborting. The finite/well-conditioned path is
        # untouched, so converged estimates and CIs are unchanged.
        return isfinite(ForwardDiff.value(total)) ? total : oftype(total, 1e10)
    end

    # `aghq = true`: the same model, integrated by per-"group" ADAPTIVE
    # Gauss-Hermite quadrature (`_aghq_marginal_loglik`, #834/#719) instead of
    # the fixed prior-scale grid above. Each observation i is its own "group"
    # of two virtual members — family 1's contribution at member 2i-1, family
    # 2's at member 2i — sharing the one latent u_i ~ N(0, 1) (a FIXED prior,
    # L = [1]; the loadings λ1/λ2, not a group SD, carry all θ-dependence, via
    # the random-effect "design" Zre = [λ1; λ2] rebuilt each nll call). Opt-in
    # and off by default: the prior-scale grid above is unchanged unless
    # `aghq = true` is requested (signature-stable; #719's finding names this
    # route but does not require flipping its default).
    aghq_rule = aghq ? _AGHQRule(1, aghq_K) : nothing
    aghq_bcache = aghq ? zeros(1, n) : zeros(0, 0)
    aghq_members = aghq ? [[2i - 1, 2i] for i in 1:n] : Vector{Vector{Int}}()
    function nll_aghq(θ)
        bb1 = θ[1:p1]
        bb2 = θ[p1+1:p1+p2]
        ll1 = exp(θ[iλ1]); ll2 = θ[iλ2]
        T = eltype(θ)
        sd1v = s1 ? exp.(Xsigma1 * θ[is1]) : nothing
        sd2v = s2 ? exp.(Xsigma2 * θ[is2]) : nothing
        one_T = one(T)
        η1f = X1 * bb1
        η2f = X2 * bb2
        η0 = Vector{T}(undef, 2n)
        Zre = Matrix{T}(undef, 2n, 1)
        @inbounds for i in 1:n
            η0[2i-1] = η1f[i]; η0[2i] = η2f[i]
            Zre[2i-1, 1] = ll1; Zre[2i, 1] = ll2
        end
        ll = function (m, η)
            if isodd(m)
                oi = (m + 1) ÷ 2
                sd1 = s1 ? sd1v[oi] : one_T
                return _mf_obs_ll(fam1, η, y1[oi], trials1[oi], sd1)
            else
                oi = m ÷ 2
                sd2 = s2 ? sd2v[oi] : one_T
                return _mf_obs_ll(fam2, η, y2[oi], trials2[oi], sd2)
            end
        end
        L = reshape([1.0], 1, 1)   # u ~ N(0,1) fixed prior; loadings live in Zre
        total = -_aghq_marginal_loglik(ll, aghq_members, η0, Zre, L, aghq_rule, aghq_bcache)
        return isfinite(ForwardDiff.value(total)) ? total : oftype(total, 1e10)
    end
    objfn = aghq ? nll_aghq : nll

    θ0 = zeros(ntheta)
    θ0[1:p1] = _mf_init(fam1, X1, y1)
    θ0[p1+1:p1+p2] = _mf_init(fam2, X2, y2)
    θ0[iλ1] = log(0.3)
    θ0[iλ2] = 0.1
    # Dispersion sub-model start: intercept = the family's scalar log-native init,
    # covariate slopes (if any) = 0. With the default ones-column this is exactly the
    # old single-slot start θ0[is1] = _mf_disp_init(...).
    s1 && (θ0[is1[1]] = _mf_disp_init(fam1, y1))
    s2 && (θ0[is2[1]] = _mf_disp_init(fam2, y2))

    res = Optim.optimize(objfn, θ0, Optim.LBFGS(),
                         Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Float64.(Optim.minimizer(res))   # concrete Float64; the ForwardDiff calls below get copies
    β1 = θ̂[1:p1]; β2 = θ̂[p1+1:p1+p2]
    λ1 = exp(θ̂[iλ1]); λ2 = θ̂[iλ2]
    # Dispersion sub-model coefficients (`log σ` scale) and a representative scale
    # σ_k = exp(mean_i log σ_{k,i}). The mean log σ is
    # computed as (column-mean of Xσ)·β_σ — algebraically mean(Xσ·β_σ) but, for the
    # default ones-column, exactly the single intercept (no summation round-off), so
    # σ_k = exp(θ̂[isk[1]]), byte-identical to the old scalar slot.
    βσ1 = s1 ? θ̂[is1] : Float64[]
    βσ2 = s2 ? θ̂[is2] : Float64[]
    σ1 = s1 ? exp(dot(vec(mean(Xsigma1, dims = 1)), βσ1)) : NaN
    σ2 = s2 ? exp(dot(vec(mean(Xsigma2, dims = 1)), βσ2)) : NaN

    # ρ(θ): loadings + link-scale residual variances. Reused by the point estimate
    # and the Fisher-z delta-method CI.
    function rho_of(θ)
        b1 = @view θ[1:p1]
        b2 = @view θ[p1+1:p1+p2]
        l1 = exp(θ[iλ1]); l2 = θ[iλ2]
        μ̄1 = mean(_mf_mean.(Ref(fam1), X1 * b1))
        μ̄2 = mean(_mf_mean.(Ref(fam2), X2 * b2))
        # Representative log-native dispersion = mean per-observation linear
        # predictor, formed as (column-mean of Xσ)·β_σ so the default ones-column
        # collapses exactly to the single intercept (no summation round-off).
        ls1 = s1 ? dot(vec(mean(Xsigma1, dims = 1)), θ[is1]) : nothing
        ls2 = s2 ? dot(vec(mean(Xsigma2, dims = 1)), θ[is2]) : nothing
        vv1 = _mf_disp_v(fam1, ls1, μ̄1)
        vv2 = _mf_disp_v(fam2, ls2, μ̄2)
        return l1 * l2 / sqrt((l1^2 + vv1) * (l2^2 + vv2))
    end
    ρ = rho_of(θ̂)
    v1 = _mf_disp_v(fam1, s1 ? dot(vec(mean(Xsigma1, dims = 1)), βσ1) : nothing,
                    mean(_mf_mean.(Ref(fam1), X1 * β1)))
    v2 = _mf_disp_v(fam2, s2 ? dot(vec(mean(Xsigma2, dims = 1)), βσ2) : nothing,
                    mean(_mf_mean.(Ref(fam2), X2 * β2)))

    # Fisher-z Wald CI: delta method on atanh(ρ(θ)) with the observed-information
    # vcov. NaN interval if the Hessian is not invertible / variance non-positive.
    rho_ci_wald = (NaN, NaN)
    if confint
        H = ForwardDiff.hessian(objfn, copy(θ̂))
        V = try
            inv(H)
        catch
            fill(NaN, size(H))
        end
        g = ForwardDiff.gradient(t -> atanh(clamp(rho_of(t), -0.999999, 0.999999)), copy(θ̂))
        var_z = dot(g, V * g)
        if isfinite(var_z) && var_z > 0
            zc = Distributions.quantile(Distributions.Normal(), 1 - (1 - level) / 2)
            zr = atanh(clamp(ρ, -0.999999, 0.999999))
            se = sqrt(var_z)
            rho_ci_wald = (tanh(zr - zc * se), tanh(zr + zc * se))
        end
    end

    # Profile-likelihood CI on ρ (the recommended interval): penalty-constrained
    # re-optimisation fixing ρ(θ)=ρ0 on the atanh scale, bisected to a χ²(1, level)
    # deviance drop. Better-calibrated than Wald near the boundary, cheaper than the
    # bootstrap. (objfn/rho_of are reused; nll's/nll_aghq's locals are renamed so
    # no Dual leaks.)
    rho_ci_profile = profile ? _mf_profile_ci(objfn, rho_of, θ̂, ρ, level) : (NaN, NaN)

    # Parametric bootstrap CI: resample the shared latent + per-family draws at θ̂,
    # refit, take percentile interval. (β1/λ1/σ1 are concrete Float64 — see the nll
    # local-naming note — so the resampled data is clean and the refits converge.)
    rho_ci_boot = (NaN, NaN)
    n_boot_kept = 0
    if B > 0
        η1f = X1 * β1; η2f = X2 * β2
        # Per-observation natural dispersion at θ̂ (constant = σ1/σ2 under the default
        # ones-column Xσ, so the sampler reduces to the old scalar bootstrap).
        sd1b = s1 ? exp.(Xsigma1 * βσ1) : fill(σ1, n)
        sd2b = s2 ? exp.(Xsigma2 * βσ2) : fill(σ2, n)
        fits = Any[]
        for _ in 1:B
            ub = randn(rng, n)
            y1b = [_mf_rand(fam1, η1f[i] + λ1 * ub[i], trials1[i], sd1b[i], rng) for i in 1:n]
            y2b = [_mf_rand(fam2, η2f[i] + λ2 * ub[i], trials2[i], sd2b[i], rng) for i in 1:n]
            fb = try
                fit_mixed_family(; y1 = y1b, X1 = X1, fam1 = fam1, y2 = y2b, X2 = X2,
                                 fam2 = fam2, trials1 = trials1, trials2 = trials2,
                                 Xsigma1 = Xsigma1, Xsigma2 = Xsigma2,
                                 K = K, g_tol = g_tol, confint = false,
                                 aghq = aghq, aghq_K = aghq_K)
            catch
                nothing
            end
            push!(fits, fb)
        end
        rho_ci_boot, n_boot_kept = _mf_boot_ci(fits, B, level)
    end

    return (; β1, β2, λ1, λ2, σ1, σ2, βσ1, βσ2, v1, v2, rho_latent = ρ,
            rho_ci_wald = rho_ci_wald, rho_ci_profile = rho_ci_profile,
            rho_ci_boot = rho_ci_boot, n_boot_kept = n_boot_kept,
            loglik = -objfn(θ̂), converged = Optim.converged(res),
            iterations = res.iterations,
            fam1 = fam1, fam2 = fam2)   # carried for post-fit accessors (mf_fitted)
end

# ---------------------------------------------------------------------------
# Formula front end for the cross-family latent-rho route.
#
# WHY THIS EXISTS. `fit_mixed_family` takes RAW DESIGN MATRICES
# (`y1, X1, fam1, y2, X2, fam2, …`) and is not exported, so cross-family is the
# only fit in DRModels.jl a user reaches by hand-building matrices — every other route
# is `drm(bf(...), Family(); data = …)`. drmTMB spells the same model with its
# ordinary formula bundle and `family = c(gaussian(), poisson())`. That gap is
# the "mixed-family API mismatch" the `cross_family_latent` capability row names
# as the blocker before any public promotion.
#
# This closes it: a `drm` method taking a TUPLE of families, mirroring drmTMB's
# vector of families, that builds the designs from the formula and delegates to
# the unchanged `fit_mixed_family`.
#
# What it deliberately REFUSES, matching the row's own claim_boundary
# ("public docs must not present rho12 formulas"):
#   * a `rho12` formula — the cross-family correlation is a LATENT SCALAR, not a
#     per-observation linear predictor. Accepting `rho12 ~ x` would imply a
#     modelled correlation this route does not fit.
#   * random effects and structured markers on either axis.

"""
    drm(f::BivariateDrmFormula, fams::Tuple; data, kwargs...)

Cross-family bivariate fit — two responses from **different** families coupled by
a latent-scale correlation `rho`. The Julia twin of drmTMB's
`family = c(gaussian(), poisson())` route.

```julia
fit = drm(bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
             sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1)),
          (Gaussian(), Poisson()); data = dat)
fit.rho_latent            # the latent-scale correlation
mf_summary(fit)
```

Each axis takes its own `mu` and (where the family has one) `sigma` formula.
`sigma` is ignored for dispersionless families (`Poisson`, `Binomial`).

!!! warning "`rho12` is not a formula here"
    In the *Gaussian* bivariate route `rho12 ~ x` models a per-observation
    residual correlation. In the cross-family route the correlation is a
    **latent scalar** — the two responses live on different scales, so there is
    no common residual to correlate per observation. Supplying `rho12` is an
    error rather than being silently ignored.

Returns the `fit_mixed_family` result; see [`mf_summary`](@ref), [`mf_coef`](@ref).
This route is **experimental** — see the `cross_family_latent` capability row.
"""
function drm(f::BivariateDrmFormula, fams::Tuple; data, K::Int = 32,
             g_tol::Real = 1e-6, confint::Bool = true, level::Real = 0.95,
             profile::Bool = false, B::Int = 0, rng = Random.default_rng())
    length(fams) == 2 || throw(ArgumentError(
        "drm: the cross-family route takes exactly two families, e.g. " *
        "`(Gaussian(), Poisson())` (got $(length(fams)))"))
    fam1, fam2 = fams
    rhs = Dict(f.forms)

    haskey(rhs, :rho12) && rhs[:rho12] !== ConstantTerm(1) && throw(ArgumentError(
        "drm: the cross-family route fits a LATENT-SCALE correlation, which is a " *
        "scalar — `rho12 ~ …` would imply a per-observation modelled correlation " *
        "this route does not fit. Drop the `rho12` formula; read the fitted value " *
        "from `fit.rho_latent`. (A per-observation `rho12` formula is the " *
        "two-Gaussian residual route: `drm(bf(...), Gaussian(); data = …)`.)"))

    for p in (:mu1, :mu2)
        _, re_p, mv_p, st_p = _split_ranef(rhs[p])
        isempty(re_p) || throw(ArgumentError(
            "drm: the cross-family route does not support random effects on `$p` yet."))
        st_p === nothing || throw(ArgumentError(
            "drm: the cross-family route does not support structured markers on `$p` yet."))
        mv_p === nothing || throw(ArgumentError(
            "drm: the cross-family route does not support `meta_V` on `$p`."))
    end

    y1, X1, _ = _design(f.response1, rhs[:mu1], data)
    y2, X2, _ = _design(f.response2, rhs[:mu2], data)
    # σ designs: the sigma formulas reuse a real LHS (only the matrix is kept),
    # exactly as the Gaussian bivariate residual route does.
    _, Xs1, _ = _design(f.response1, get(rhs, :sigma1, ConstantTerm(1)), data)
    _, Xs2, _ = _design(f.response1, get(rhs, :sigma2, ConstantTerm(1)), data)

    fit = fit_mixed_family(; y1 = Float64.(y1), X1 = X1, fam1 = fam1,
                           y2 = Float64.(y2), X2 = X2, fam2 = fam2,
                           Xsigma1 = Xs1, Xsigma2 = Xs2,
                           K = K, g_tol = Float64(g_tol), confint = confint,
                           level = Float64(level), profile = profile, B = B, rng = rng)
    # Carry the designs so the post-fit surface works WITHOUT them being handed
    # back by the user. `mf_fitted(fit, X1, X2)` demanding the matrices is the
    # same API mismatch in post-fit form: a formula user has no X1 to give it.
    return (; fit..., X1 = X1, X2 = X2)
end
