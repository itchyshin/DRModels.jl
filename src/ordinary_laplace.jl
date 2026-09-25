# ordinary_laplace.jl — opt-in `marginal = :Laplace` for an ordinary random
# intercept `(1 | g)` on the mean of Poisson, Binomial, NegBinomial2, Gamma and
# Beta (Arc 2, drmTMB same-target parity).
#
# Why a separate route. The default `marginal = :LA` integrates an ordinary
# `(1 | g)` by 32-node Gauss–Hermite quadrature per group (see the
# `_fit_*_ranef` fitters). drmTMB (TMB) integrates the same model by the
# Laplace approximation, so the two engines fit the same model but report
# different log-likelihoods and estimates. `:Laplace` fits the TMB objective:
#
#     −log L(θ) ≈ −ℓ(y | b̂) + ½ b̂′b̂/σ² + G·log σ + ½ log det H(b̂)
#
# with b̂ the joint inner mode and H the Hessian of the penalized negative
# log-likelihood in b (the `(2π)` constants cancel). The Laplace approximation is
# invariant to the affine reparameterisation b = σu, so this equals TMB's value
# on its `u`-scale parameterisation exactly.
#
# The machinery is not new: it is the verified sparse-Laplace spine of the
# structured routes (`sparse_laplace_glmm.jl`: phylo/relmat/animal), with the
# prior precision Q = I (independent groups) and the observation→latent map
# equal to the group index. Differences from those routes, all so that value
# and gradient describe TMB's function everywhere: the random-effect log-SD and
# the dispersion are unclamped (`raw_scales = true`); NB2 uses a kernel that
# stays accurate at a huge size 1/σ² (`Val(:nb2_raw)`, below); and the inner
# mode is solved to 1e-13 (`_ORDINARY_LAPLACE_NEWTON_TOL`).
#
# Scope (everything else is refused, never silently rerouted to `:LA`):
#   * exactly one ordinary `(1 | g)` on `mu`; no `(1 + x | g)`, `(0 + x | g)`,
#     crossed/multiple terms, structured markers, `meta_V`, `zi`/`hu`;
#   * NegBinomial2 / Gamma / Beta with `sigma ~ 1` (no random effect on sigma,
#     no coupled location–scale `(1 | p | g)`);
#   * maximum likelihood only (`method = :REML` is refused).
#
# The family `drm` methods delegate here with one line placed before any of
# their own routing; `:LA`, `:VA` and `:AGHQ` never reach this file.

_scalar_laplace_requested(marginal) =
    marginal isa Symbol && Symbol(uppercase(String(marginal))) === :LAPLACE

function _ordinary_laplace_reject(fam, what)
    throw(ArgumentError(
        "marginal = :Laplace is not available for $(nameof(typeof(fam)))() with $what. " *
        "This route covers exactly one ordinary random intercept `(1 | g)` on the mean for " *
        "Poisson, Binomial, NegBinomial2, Gamma and Beta (`sigma ~ 1` for the three scale " *
        "families), fitted by maximum likelihood. Omit `marginal` (the default `:LA`) for " *
        "other models."))
end

_ordinary_laplace_has_sigma(::Union{NegBinomial2,Gamma,Beta}) = true
_ordinary_laplace_has_sigma(_) = false

"""
    _drm_ordinary_laplace(f, fam; data, tree, K, A, coords, g_tol, se, method)

Front end for `drm(f, fam; marginal = :Laplace)` on Poisson, Binomial,
NegBinomial2, Gamma and Beta. Validates the admitted shape (one ordinary
`(1 | g)` on the mean; `sigma ~ 1` for scale families; ML), then fits the
TMB-convention Laplace marginal. Returns a `DrmFit` tagged `marginal = :Laplace`
with blocks `:mu`, (`:sigma`,) `:resd` (the random-effect log-SD).
"""
function _drm_ordinary_laplace(f::DrmFormula, fam; data, tree = nothing,
                               K = nothing, A = nothing, coords = nothing,
                               g_tol::Real = 1e-8, se::Bool = true,
                               method = nothing)
    meth = _reject_method_as_marginal(fam, method; allow_reml = true)
    meth === :REML && _ordinary_laplace_reject(fam, "`method = :REML`")
    missing_fit = _fit_observed_response_rows(f, data) do data_observed
        _drm_ordinary_laplace(f, fam; data = data_observed, tree = tree, K = K,
                              A = A, coords = coords, g_tol = g_tol, se = se,
                              method = method)
    end
    missing_fit !== nothing && return missing_fit

    _lss_only_gaussian_guard(f, fam)
    rhs = Dict(f.forms)
    has_sigma = _ordinary_laplace_has_sigma(fam)
    for (pname, r) in f.forms
        pname === :mu && continue
        if pname === :sigma && has_sigma
            _, re2, mv2, st2 = _split_ranef(r)
            (isempty(re2) && mv2 === nothing && st2 === nothing) ||
                _ordinary_laplace_reject(fam, "a random effect on `sigma`")
        elseif pname === :sigma && r === ConstantTerm(1)
            continue                                   # `bf` fills `sigma ~ 1` for mean-only families
        else
            _ordinary_laplace_reject(fam, "a `$pname` formula")
        end
    end
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    mv === nothing || _ordinary_laplace_reject(fam, "a `meta_V` marker")
    st === nothing || _ordinary_laplace_reject(fam, "a phylogenetic/structured random effect")
    length(re) == 1 || _ordinary_laplace_reject(fam,
        isempty(re) ? "no random effect (fixed-effects-only)" : "crossed/multiple random effects")
    # Render the term as the user wrote it: a FunctionTerm such as `1 + x`
    # prints as `:(1 + x)`, so show its original expression instead.
    lhs = re[1][1]
    lhs_text = hasproperty(lhs, :exorig) ? string(lhs.exorig) : string(lhs)
    lhs isa ConstantTerm && lhs.n == 1 ||
        _ordinary_laplace_reject(fam, "`($lhs_text | $(re[1][2]))` (only `(1 | g)` is covered)")
    grp = re[1][2]
    gidx, G = _group_index(getproperty(data, grp))

    if fam isa Binomial
        s, ntr = _binomial_response(f, data)
        _, Xμ, nmμ = _design(f.response, fixed_mu, data)
        fit = _fit_binomial_ordinary_laplace(fam, s, ntr, Xμ, gidx, G, nmμ, grp, g_tol; se = se)
    else
        y, Xμ, nmμ = _design(f.response, fixed_mu, data)
        if fam isa Poisson
            all(yi -> yi ≥ 0 && isinteger(yi), y) ||
                error("Poisson() requires non-negative integer counts as the response")
            fit = _fit_poisson_ordinary_laplace(fam, y, Xμ, gidx, G, nmμ, grp, g_tol; se = se)
        else
            _, Xσ, nmσ = _design(f.response, get(rhs, :sigma, ConstantTerm(1)), data)
            (size(Xσ, 2) == 1 && all(==(1.0), @view Xσ[:, 1])) ||
                _ordinary_laplace_reject(fam, "a non-constant `sigma` formula")
            fit = _fit_scale_ordinary_laplace(fam, y, Xμ, gidx, G, nmμ, nmσ, grp, g_tol; se = se)
        end
    end
    return _withformula(_withmarginal(fit, :Laplace), f)
end

# Shared outer loop. `fg(θ, b0, grad)` is one of the verified sparse-Laplace
# kernels closed over its data; it returns `(val, g, b, ok)` when `grad` and
# `(val, b, ok)` otherwise. The optimiser sequence is the structured routes'
# (LBFGS/backtracking, short polish, #422 boundary polish, #491 convergence
# flag, finite-difference Hessian for the Wald covariance), plus a scale-free
# Newton-decrement fallback for the convergence flag (see below).
function _ordinary_laplace_optimize(fg, θ0, n::Int, q::Int, g_tol; se::Bool,
                                    context::AbstractString,
                                    polish_iterations::Int = 15)
    last_b = zeros(q)
    function eval_laplace(θ, grad::Bool)
        out = fg(θ, last_b, grad)
        ok = out[end]
        ok || (out = fg(θ, zeros(q), grad); ok = out[end])
        if !ok
            return grad ? (1e18, zeros(length(θ))) : 1e18
        end
        last_b .= out[end-1]
        return grad ? (out[1], out[2]) : out[1]
    end
    nll(θ) = eval_laplace(θ, false)
    function grad!(Gout, θ)
        _, g = eval_laplace(θ, true)
        Gout .= g
        return Gout
    end
    od = Optim.OnceDifferentiable(nll, grad!, θ0)
    method = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking())
    res_fast = Optim.optimize(od, θ0, method, Optim.Options(g_tol = g_tol, iterations = 250))
    res = try
        θp = Optim.minimizer(res_fast)
        odp = Optim.OnceDifferentiable(nll, grad!, θp)
        Optim.optimize(odp, θp, Optim.LBFGS(),
                       Optim.Options(g_tol = g_tol, iterations = polish_iterations))
    catch
        res_fast
    end
    θ̂ = Optim.minimizer(res)
    nllhat = nll(θ̂)
    if nllhat > nll(Optim.minimizer(res_fast))          # polish may only improve
        θ̂ = Optim.minimizer(res_fast); nllhat = nll(θ̂)
    end
    θ̂, nllhat = _laplace_boundary_polish(nll, grad!, θ̂, nllhat)   # issue #422
    gfinal = zeros(length(θ̂))
    grad!(gfinal, θ̂)
    converged = _laplace_outer_converged(res, nllhat, gfinal, θ̂, n, g_tol)
    # A small family `sigma` (e.g. 0.01) makes the curvature in the mean
    # coefficients ~1e6: LBFGS then stops ~1e-6 from the optimum, where the
    # raw gradient is still ~1e-2 and the gradient rule above says "not
    # converged" although the fit matches native drmTMB. Only in that case, ask
    # the scale-free question instead. Fits that already pass are untouched.
    # LBFGS leaves θ̂ ~1e-4 SE from the optimum there, which can miss a 1e-5
    # relative parity bar on a coefficient; a few Newton steps on the outer
    # gradient close that gap before the question is asked.
    if !converged && isfinite(nllhat) && nllhat < 1e17
        θ̂, gfinal = _ordinary_laplace_newton_polish(grad!, θ̂, gfinal)
        nllhat = nll(θ̂)
        converged = _ordinary_laplace_newton_converged(grad!, θ̂, gfinal)
    end
    V = _ordinary_laplace_vcov(nll, θ̂, n; se = se, context = context)
    return θ̂, nllhat, converged, V, nll, grad!
end

# Wald covariance from the finite-difference Hessian of the outer objective, or
# a NaN matrix when `se = false`.
function _ordinary_laplace_vcov(nll, θ̂, n::Int; se::Bool, context::AbstractString)
    V = if se
        _vcov_from_hessian(_finite_hessian(nll, θ̂; h = _fd_hessian_step(n)); context = context)
    else
        fill(NaN, length(θ̂), length(θ̂))
    end
    return Matrix(V)
end

# Hessian of the outer objective by central differences of its analytic gradient.
function _ordinary_laplace_grad_hessian(grad!, θ; h::Real = 1e-5)
    p = length(θ)
    H = zeros(p, p)
    gp = zeros(p); gm = zeros(p)
    for i in 1:p
        hi = h * (1 + abs(θ[i]))
        e = zeros(p); e[i] = hi
        grad!(gp, θ .+ e); grad!(gm, θ .- e)
        H[:, i] .= (gp .- gm) ./ (2hi)
    end
    return Symmetric((H .+ H') ./ 2)
end

# Scale-free convergence: the Newton decrement λ² = g′H⁻¹g at θ̂. It does not
# change when a parameter is rescaled; λ²/2 is the objective gap to the local
# quadratic minimum, and every coordinate lies within λ standard errors of it.
# λ² ≤ 1e-8 means each estimate is within 1e-4 SE of the optimum. It uses the
# analytic gradient only: at σ ≈ 0.01 the objective VALUE is reproducible only
# to ~1e-9 (inner-mode accuracy), below which a value-based check cannot see.
# It needs a positive-definite Hessian, so a flat (collapsed-variance)
# direction never passes here; such fits keep the gradient rule's verdict.
const _ORDINARY_LAPLACE_DECREMENT_TOL = 1e-8

# Up to `maxiter` Newton steps θ ← θ − H⁻¹g with H the central-difference
# Hessian of the analytic gradient. Only from inside the quadratic basin
# (λ² ≤ 1e-2), and a step is kept only if it lowers the Newton decrement, so it
# can only move θ̂ closer to the optimum it already sits next to. It judges by
# the gradient, not the value, for the reason given below.
function _ordinary_laplace_newton_polish(grad!, θ̂, gfinal; maxiter::Int = 3)
    θ = copy(θ̂); g = copy(gfinal)
    all(isfinite, g) || return θ, g
    for _ in 1:maxiter
        C = cholesky(_ordinary_laplace_grad_hessian(grad!, θ); check = false)
        issuccess(C) || break
        step = C \ g
        λ2 = dot(g, step)
        (λ2 <= 1e-2 && λ2 > 0) || break
        θt = θ .- step
        gt = zeros(length(θ)); grad!(gt, θt)
        all(isfinite, gt) || break
        Ct = cholesky(_ordinary_laplace_grad_hessian(grad!, θt); check = false)
        issuccess(Ct) || break
        dot(gt, Ct \ gt) < λ2 || break
        θ, g = θt, gt
    end
    return θ, g
end

function _ordinary_laplace_newton_converged(grad!, θ̂, gfinal)
    all(isfinite, gfinal) || return false
    C = cholesky(_ordinary_laplace_grad_hessian(grad!, θ̂); check = false)
    issuccess(C) || return false
    return dot(gfinal, C \ gfinal) <= _ORDINARY_LAPLACE_DECREMENT_TOL
end

_ordinary_laplace_Q(G::Int) = spdiagm(0 => ones(Float64, G))

# Inner-mode tolerance (relative Newton step). The outer gradient assumes the
# inner gradient is zero at b̂; its error is ≈ H_bb·δb. A small family `sigma`
# makes H_bb large (≈ 1e6 per group at Gamma sigma = 0.003), so the structured
# routes' 1e-10 leaves a ~1e-3 error in the mean-coefficient gradient and the
# outer optimum drifts ~1e-5 from TMB's. Newton is quadratic here, so 1e-13
# costs about one extra inner step.
const _ORDINARY_LAPLACE_NEWTON_TOL = 1e-13

function _fit_poisson_ordinary_laplace(fam::Poisson, y, Xμ, gidx, G, nmμ, grp, g_tol;
                                       se::Bool = true)
    n = length(y); pμ = size(Xμ, 2)
    Q = _ordinary_laplace_Q(G)
    lf = [_logfactorial(round(Int, yi)) for yi in y]
    fg = (θ, b0, grad) -> _poisson_phylo_laplace_fg(y, Xμ, gidx, Q, 0.0, lf, θ;
                                                    grad = grad, b0 = b0, raw_scales = true,
                                                    newton_tol = _ORDINARY_LAPLACE_NEWTON_TOL)
    θ0 = vcat(_poisson_fixed_start(y, Xμ), log(0.4))
    θ̂, nllhat, conv, V, nll, grad! = _ordinary_laplace_optimize(
        fg, θ0, n, G, g_tol; se = se, context = "ordinary Laplace Poisson (1 | $grp)")
    blocks = [:mu => 1:pμ, :resd => (pμ+1):(pμ+1)]
    names = [:mu => nmμ, :resd => [String(grp)]]
    means = Dict(:mu => exp.(Xμ * θ̂[1:pμ]))
    obs = Dict(:mu => Vector{Float64}(y))
    fit = DrmFit(fam, blocks, names, θ̂, V, -nllhat, n, conv, means, obs,
                 Dict{Symbol,Vector{Float64}}())
    return _withnll(fit, nll, grad!)
end

function _fit_binomial_ordinary_laplace(fam::Binomial, s, ntr, Xμ, gidx, G, nmμ, grp,
                                        g_tol; se::Bool = true)
    n = length(s); pμ = size(Xμ, 2)
    Q = _ordinary_laplace_Q(G)
    sint = round.(Int, s); nint = round.(Int, ntr)
    logchoose = [_logfactorial(nint[i]) - _logfactorial(sint[i]) -
                 _logfactorial(nint[i] - sint[i]) for i in eachindex(sint)]
    aux = (s = sint, ntr = nint, logchoose = logchoose)
    kind = Val(:binomial)
    fg = (θ, b0, grad) -> _phylo_mean_laplace_fg(kind, aux, n, Xμ, gidx, Q, 0.0, θ;
                                                 grad = grad, b0 = b0, raw_scales = true,
                                                 newton_tol = _ORDINARY_LAPLACE_NEWTON_TOL)
    p̄ = clamp(sum(s) / max(sum(ntr), 1), 1e-4, 1 - 1e-4)
    θ0 = zeros(pμ + 1); θ0[1] = log(p̄ / (1 - p̄)); θ0[end] = log(0.4)
    θ̂, nllhat, conv, V, nll, grad! = _ordinary_laplace_optimize(
        fg, θ0, n, G, g_tol; se = se, context = "ordinary Laplace Binomial (1 | $grp)")
    blocks = [:mu => 1:pμ, :resd => (pμ+1):(pμ+1)]
    names = [:mu => nmμ, :resd => [String(grp)]]
    means = Dict(:mu => [_laplace_mean(kind, dot(@view(Xμ[i, :]), θ̂[1:pμ])) for i in 1:n])
    obs = Dict(:mu => [_laplace_obs(kind, aux, i) for i in 1:n])
    fit = DrmFit(fam, blocks, names, θ̂, V, -nllhat, n, conv, means, obs,
                 Dict(:trials => Float64.(nint)))
    return _withnll(fit, nll, grad!)
end

_ordinary_laplace_scale_setup(::NegBinomial2, y, Xμ) =
    (Val(:nb2_raw), _ordinary_laplace_nb2_setup(y, Xμ)...)
_ordinary_laplace_scale_setup(::Gamma, y, Xμ) =
    (Val(:gamma_fixed), _gamma_laplace_setup(y, Xμ; raw_scales = true)...)
_ordinary_laplace_scale_setup(::Beta, y, Xμ) =
    (Val(:beta_fixed), _beta_laplace_setup(y, Xμ; raw_scales = true)...)

# ---- NB2 on unclamped scales: a large-size-stable kernel (`Val(:nb2_raw)`) ----
# With the dispersion unclamped, near-Poisson data drive the size r = 1/σ² far
# above e^20. The structured kernel `:nb2_fixed` then subtracts numbers of size
# r·log r (`loggamma(y + r) − loggamma(r)`, `r log r − (y + r) log(r + μ)`, the
# digamma difference) and loses every digit: at r ≈ e^115 it reports a
# log-likelihood near 0 for data whose true value is −613, and the optimiser
# walks there. `:nb2_raw` is the same log-density,
#
#     log p = A(y, r) − log y! + y log μ − (y + r) log1p(μ/r),
#     A(y, r) = log Γ(y + r) − log Γ(r) − y log r,
#
# written without those cancellations (the idea behind TMB's
# `dnbinom_robust`): A and its r-derivative come from the Stirling series when
# r ≥ `_NB2_RAW_STIRLING_MIN`, and every ratio μ/r enters through log1p. It is
# used only by the ordinary `marginal = :Laplace` route; `:nb2_fixed` (the
# default and structured routes) is untouched.
const _NB2_RAW_STIRLING_MIN = 100.0

_nb2_raw_tail(z) = inv(12z) - inv(360z^3) + inv(1260z^5)          # log Γ Stirling tail
_nb2_raw_tail_d(z) = -inv(12z^2) + inv(120z^4) - inv(252z^6)      # its derivative

# A(y, r) = log Γ(y + r) − log Γ(r) − y log r.
function _nb2_raw_lgamma_ratio(y, r)
    r < _NB2_RAW_STIRLING_MIN && return loggamma(y + r) - loggamma(r) - y * log(r)
    z = r + y
    return (z - 0.5) * log1p(y / r) - y + _nb2_raw_tail(z) - _nb2_raw_tail(r)
end

# ∂A/∂r = ψ(y + r) − ψ(r) − y/r.
function _nb2_raw_digamma_ratio(y, r)
    r < _NB2_RAW_STIRLING_MIN && return digamma(y + r) - digamma(r) - y / r
    z = r + y
    return log1p(y / r) - y * (z - 0.5) / (r * z) + _nb2_raw_tail_d(z) - _nb2_raw_tail_d(r)
end

# Same starts as the structured NB2 routes; the aux carries the stable
# A(y, r) − log y! instead of `loggamma(y + r) − loggamma(r) − log y!`.
function _ordinary_laplace_nb2_setup(y, Xμ)
    _, θβ0, θσ0 = _nb2_laplace_setup(y, Xμ)
    yv = Float64.(round.(Int, y))
    function aux_from(logσ)
        r = exp(-2 * logσ)                          # size r = 1/σ² (drmTMB), unclamped
        lconst = [_nb2_raw_lgamma_ratio(yv[i], r) - _logfactorial(round(Int, yv[i]))
                  for i in eachindex(yv)]
        return (y = yv, size = r, lconst = lconst)
    end
    return aux_from, θβ0, θσ0
end

function _laplace_value(::Val{:nb2_raw}, aux, i, η)
    ηc = clamp(η, -30.0, 30.0)
    y = aux.y[i]; r = aux.size
    return -(aux.lconst[i] + y * ηc - (y + r) * log1p(exp(ηc) / r))
end

function _laplace_d12(::Val{:nb2_raw}, aux, i, η)
    μ = exp(clamp(η, -30.0, 30.0))
    y = aux.y[i]; r = aux.size
    den = r + μ
    return r * (μ - y) / den, (y + r) * (r / den) * (μ / den)
end

function _laplace_v123_nuisance(::Val{:nb2_raw}, aux, i, η)
    ηc = clamp(η, -30.0, 30.0)
    μ = exp(ηc)
    y = aux.y[i]; r = aux.size
    den = r + μ
    ρ = r / den                                      # r/(r + μ) ∈ (0, 1]
    v = -(aux.lconst[i] + y * ηc - (y + r) * log1p(μ / r))
    d1 = ρ * (μ - y)
    d2 = (y + r) * ρ * (μ / den)
    d3 = d2 * (r - μ) / den
    # ∂ log p/∂r = A′ − log1p(μ/r) + (y + r)μ/(r(r + μ)); ψ = log σ, ∂r/∂ψ = −2r.
    dlogp_dr = _nb2_raw_digamma_ratio(y, r) - log1p(μ / r) + (y / r + 1) * (μ / den)
    nv = 2 * r * dlogp_dr
    nd1 = -2 * ρ * μ * (μ - y) / den
    nd2 = -2 * ρ * μ * (μ * (y + 2r) - r * y) / den^2
    return v, d1, d2, d3, nv, nd1, nd2
end

_laplace_mean(::Val{:nb2_raw}, η) = exp(clamp(η, -30.0, 30.0))
_laplace_obs(::Val{:nb2_raw}, aux, i) = aux.y[i]

function _ordinary_laplace_check_response(::NegBinomial2, y)
    all(yi -> yi ≥ 0 && isinteger(yi), y) ||
        error("NegBinomial2() requires non-negative integer counts as the response")
end
_ordinary_laplace_check_response(::Gamma, y) =
    all(>(0), y) || error("Gamma() requires a strictly positive response")
_ordinary_laplace_check_response(::Beta, y) =
    all(yi -> 0 < yi < 1, y) || error("Beta() requires a response strictly inside (0, 1)")

# Below this fitted log σ (σ ≈ 3e-4) the scale route re-fits from log σ = −1.
const _ORDINARY_LAPLACE_PLATEAU_LOGSIGMA = -8.0

function _fit_scale_ordinary_laplace(fam, y, Xμ, gidx, G, nmμ, nmσ, grp, g_tol;
                                     se::Bool = true)
    _ordinary_laplace_check_response(fam, y)
    n = length(y); pμ = size(Xμ, 2)
    Q = _ordinary_laplace_Q(G)
    kind, aux_from, θβ0, θσ0 = _ordinary_laplace_scale_setup(fam, y, Xμ)
    fg = (θ, b0, grad) -> _phylo_mean_laplace_nuisance_fg(
        kind, aux_from, n, Xμ, gidx, Q, 0.0, θ; grad = grad, b0 = b0, raw_scales = true,
        newton_tol = _ORDINARY_LAPLACE_NEWTON_TOL)
    θ0 = vcat(θβ0, θσ0, log(0.4))
    ctx = "ordinary Laplace $(nameof(typeof(fam))) (1 | $grp)"
    # Both stages run with se = false: the covariance is computed once, below,
    # for the fit the guard keeps. A discarded plateau fit's Hessian is singular
    # in log σ and would print false warnings on a well-conditioned result.
    θ̂, nllhat, conv, _, nll, grad! = _ordinary_laplace_optimize(
        fg, θ0, n, G, g_tol; se = false, context = ctx)
    # Plateau guard. With the scale unclamped, one large LBFGS step can carry
    # log σ from the start to −20 or below, where the objective is flat (NB2:
    # the Poisson limit, ∂nll/∂log σ ~ 1e-14). The optimiser never returns and
    # the gradient rule reports convergence at a worse point (a review NB2
    # cell: logLik 0.81 below native drmTMB). Only then, re-fit from an
    # interior start and keep the lower objective; every other fit is untouched.
    if θ̂[pμ+1] < _ORDINARY_LAPLACE_PLATEAU_LOGSIGMA
        θ1 = copy(θ0); θ1[pμ+1] = -1.0
        alt = _ordinary_laplace_optimize(fg, θ1, n, G, g_tol; se = false, context = ctx)
        if alt[2] < nllhat
            θ̂, nllhat, conv, _, nll, grad! = alt
        end
    end
    V = _ordinary_laplace_vcov(nll, θ̂, n; se = se, context = ctx)
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+1), :resd => (pμ+2):(pμ+2)]
    names = [:mu => nmμ, :sigma => nmσ, :resd => [String(grp)]]
    auxhat = aux_from(θ̂[pμ+1])
    means = Dict(:mu => [_laplace_mean(kind, dot(@view(Xμ[i, :]), θ̂[1:pμ])) for i in 1:n])
    obs = Dict(:mu => [_laplace_obs(kind, auxhat, i) for i in 1:n])
    scales = Dict(:sigma => fill(exp(θ̂[pμ+1]), n))
    fit = DrmFit(fam, blocks, names, θ̂, V, -nllhat, n, conv, means, obs, scales)
    return _withnll(fit, nll, grad!)
end
