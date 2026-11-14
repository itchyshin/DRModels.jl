# indep_slopes_laplace.jl — independent random intercepts and slopes on the mean
# for Gamma() and Student(), e.g. `(1 | g) + (0 + x | g)` (#960, drmTMB parity).
#
# drmTMB fits these layouts by TMB's Laplace approximation with one independent
# N(0, σ_k²) prior per term; it refuses the correlated `(1 + x | g)` form for
# these families (drmTMB #1457), so this route deliberately covers the
# independent form only. DRModels.jl's own correlated `(1 + x | g)` GHQ routes
# are unchanged.
#
# Model. K scalar components; component k has a per-observation weight w_k
# (1 for an intercept, the covariate x for a `(0 + x | g)` slope), a grouping
# index and G_k levels. With b = [b_1; …; b_K] (q = Σ G_k effects),
#
#     η_i = X_i′β_μ + Σ_k w_k[i] · b_k[g_k(i)],      b_k ~ N(0, σ_k² I),
#
# and the TMB-convention Laplace marginal (the (2π) constants cancel)
#
#     nll(θ) = Σ_i −log f(y_i | η_i(b̂)) + ½ Σ_k ‖b̂_k‖²/σ_k² + Σ_k G_k log σ_k + ½ log det 𝐇(b̂),
#
# 𝐇 the Hessian in b of the penalised negative log-likelihood. Same machinery
# as the Student crossed-intercept route (`_fit_student_crossed_laplace`, #725):
# Newton inner mode on Float64 primals (expected-information direction where
# the observed 𝐇 is indefinite), then two Newton steps in dual arithmetic so
# ForwardDiff carries db̂/dθ (implicit-function theorem). 𝐇 is dense q×q: fine
# for hundreds of levels.
#
# Per-family observation terms return (−log f, d/dη, d²/dη², expected d²/dη²):
#   Gamma (log link, shape α = σ⁻², the `_fit_gamma` convention):
#     −log f = −α log α + log Γ(α) − (α − 1) log y + α η + α y e^{−η}
#     d/dη = α − α y e^{−η},  d²/dη² = α y e^{−η},  E = α
#   Student (identity link, scale σ, ν = 2 + e^{η_ν}, t = (y − η)/σ, r = 1/ν):
#     as `_fit_student_crossed_laplace`.

function _indep_obs_terms(::Val{:gamma}, y, η, lσ, ην)
    α = exp(-2 * lσ)
    ye = y * exp(-η)
    val = -(α * log(α) - loggamma(α) + (α - 1) * log(y) - α * η - α * ye)
    return val, α - α * ye, α * ye, α
end

function _indep_obs_terms(::Val{:student}, y, η, lσ, ην)
    σ = exp(lσ); t = (y - η) / σ
    r = ην > 0 ? exp(-ην) / (1 + 2 * exp(-ην)) : 1 / (2 + exp(ην))
    d = 1 + r * t^2
    val = lσ - _student_logpdf_std(t, ην)
    g1 = -(1 + r) * t / (σ * d)
    w = (1 + r) * (1 - r * t^2) / (σ^2 * d^2)
    wE = (1 + r) / ((1 + 3r) * σ^2)
    return val, g1, w, wE
end

# Non-Gaussian families added in wave 11 (#960 follow-up). Each returns
# (−log f, d/dη, d²/dη², expected d²/dη²) at one observation, in the same
# conventions as the crossed routes (`_laplace_value`/`_laplace_d12`):
#   Poisson (log link):       no dispersion; E = e^η.
#   NB2 (log link):           size r = exp(−2 lσ) (drmTMB sigma), E = r μ/(r + μ).
#   Binomial (logit link):    `yi = (successes, trials)`; E = n p (1 − p).
#   Beta (logit link):        precision φ = exp(−2 lσ); E drops the A·v′ term.
@inline _indep_clampη(η) = clamp(η, -30, 30)
@inline _indep_logistic(η) = η >= 0 ? 1 / (1 + exp(-η)) : exp(η) / (1 + exp(η))

function _indep_obs_terms(::Val{:poisson}, y, η, lσ, ην)
    ηc = _indep_clampη(η); μ = exp(ηc)
    return μ - y * ηc + loggamma(y + 1), μ - y, μ, μ
end

function _indep_obs_terms(::Val{:nb2}, y, η, lσ, ην)
    ηc = _indep_clampη(η); μ = exp(ηc)
    r = exp(clamp(-2 * lσ, -20, 20)); den = r + μ
    val = -(loggamma(y + r) - loggamma(r) - loggamma(y + 1) + y * ηc + r * log(r) - (y + r) * log(den))
    return val, (y + r) * μ / den - y, (y + r) * r * μ / den^2, r * μ / den
end

function _indep_obs_terms(::Val{:binomial}, ys, η, lσ, ην)
    s, n = ys
    ηc = _indep_clampη(η); p = _indep_logistic(ηc)
    l1pe = ηc > 0 ? ηc + log1p(exp(-ηc)) : log1p(exp(ηc))
    lc = loggamma(n + 1) - loggamma(s + 1) - loggamma(n - s + 1)
    return n * l1pe - s * ηc - lc, n * p - s, n * p * (1 - p), n * p * (1 - p)
end

function _indep_obs_terms(::Val{:beta}, y, η, lσ, ην)
    ηc = _indep_clampη(η); μ = _indep_logistic(ηc)
    φ = exp(clamp(-2 * lσ, -30, 30)); a = μ * φ; b = (1 - μ) * φ
    val = -(loggamma(φ) - loggamma(a) - loggamma(b) + (a - 1) * log(y) + (b - 1) * log1p(-y))
    ylogit = log(y) - log1p(-y)
    A = φ * (digamma(a) - digamma(b) - ylogit)
    B = φ^2 * (trigamma(a) + trigamma(b))
    v = μ * (1 - μ); vp = v * (1 - 2μ)
    return val, A * v, B * v^2 + A * vp, B * v^2
end

# Tweedie (log link, φ = exp(2 lσ), power p = 1 + logistic(η_ν)): only the quasi-likelihood
# part of the density depends on η, log f = (y μ^{1-p}/(1-p) - μ^{2-p}/(2-p))/φ + log a(y, φ, p), so
# d/dη and d²/dη² are closed form; the series term enters the value only. E[d²/dη²] = μ^{2-p}/φ.
function _indep_obs_terms(::Val{:tweedie}, y, η, lσ, ην)
    ηc = _indep_clampη(η); lsc = clamp(lσ, -15, 15); νc = clamp(ην, -12, 12)
    μ = exp(ηc); φ = exp(2 * lsc); p = _logit12(νc)
    a = exp((1 - p) * ηc); b = exp((2 - p) * ηc)
    return -_logpdf_tweedie(y, μ, φ, p), (b - y * a) / φ, ((2 - p) * b - (1 - p) * y * a) / φ, b / φ
end

# Response container, mean map, start values and labels per family.
_indep_response(::Val, y) = Float64.(y)
_indep_response(::Val{:binomial}, y) = y                      # vector of (successes, trials)
_indep_mean(::Val{:gamma}, η) = exp(η)
_indep_mean(::Val{:student}, η) = η
_indep_mean(::Val{:poisson}, η) = exp(η)
_indep_mean(::Val{:nb2}, η) = exp(η)
_indep_mean(::Val{:binomial}, η) = _indep_logistic(η)
_indep_mean(::Val{:beta}, η) = _indep_logistic(η)
_indep_mean(::Val{:tweedie}, η) = exp(η)
_indep_nu_scale(::Val{:student}, ην) = 2 .+ exp.(ην)
_indep_nu_scale(::Val{:tweedie}, ην) = _logit12.(ην)
_indep_obsval(::Val, yv) = yv
_indep_nu_block(::Val) = :nu
_indep_sigma_block(::Val) = :sigma
# Per-observation auxiliary linear predictors (ησ, ην) from θ. The default is the plain
# design-times-coefficient pair every kind used so far; the ordinal kind overrides it.
function _indep_aux_eta(kind::Val, yv, Xσ, Xν, θ, pμ, pσ, k0, ::Type{T}) where {T}
    n = length(yv)
    ησ = Xσ === nothing ? zeros(T, n) : Xσ * θ[pμ+1:pμ+pσ]
    ην = Xν === nothing ? zeros(T, n) : Xν * θ[pμ+pσ+1:k0]
    return ησ, ην
end
# Post-fit overrides of the stored means / scales (identity by default).
_indep_post(kind::Val, yv, means, scales, ημ̂, θ̂, pμ, pσ) = (means, scales)
# Kinds whose fit keeps the conditional modes for `fitted()`: the zero-inflated and zero-truncated
# count routes and the ordinal route (the other kinds keep their own, older behaviour).
_indep_records_modes(::Val) = false
_indep_records_modes(::Union{Val{:zip},Val{:zinb},Val{:tnb},Val{:cumlogit}}) = true
_indep_obsval(::Val{:binomial}, yv) = [s / n for (s, n) in yv]
_indep_famname(::Val{:gamma}) = "Gamma"
_indep_famname(::Val{:student}) = "Student"
_indep_famname(::Val{:poisson}) = "Poisson"
_indep_famname(::Val{:nb2}) = "NegBinomial2"
_indep_famname(::Val{:binomial}) = "Binomial"
_indep_famname(::Val{:beta}) = "Beta"
_indep_famname(::Val{:tweedie}) = "Tweedie"

# Fixed-effects start for the leading (β_μ; β_σ) block. Interior and well behaved.
_indep_fixed_start(::Val{:gamma}, fam, yv, Xμ, Xσ, Xν, nmμ, nmσ, nmν, g_tol) =
    _fit_gamma(fam, yv, Xμ, Xσ, nmμ, nmσ, g_tol).theta
_indep_fixed_start(::Val{:poisson}, fam, yv, Xμ, Xσ, Xν, nmμ, nmσ, nmν, g_tol) =
    _poisson_fixed_start(yv, Xμ)
function _indep_fixed_start(::Val{:nb2}, fam, yv, Xμ, Xσ, Xν, nmμ, nmσ, nmν, g_tol)
    θ = zeros(size(Xμ, 2) + size(Xσ, 2))
    θ[1:size(Xμ, 2)] .= _poisson_fixed_start(yv, Xμ)
    m = sum(yv) / length(yv); v = sum(abs2, yv .- m) / max(length(yv) - 1, 1)
    θ[size(Xμ, 2) + 1] = -0.5 * log(max(m^2 / max(v - m, 0.1 * m + eps()), 0.5))
    return θ
end
function _indep_fixed_start(::Val{:binomial}, fam, yv, Xμ, Xσ, Xν, nmμ, nmσ, nmν, g_tol)
    s = [t[1] for t in yv]; n = [t[2] for t in yv]
    return _fit_binomial(fam, s, n, Xμ, nmμ, g_tol).theta
end
function _indep_fixed_start(::Val{:tweedie}, fam, yv, Xμ, Xσ, Xν, nmμ, nmσ, nmν, g_tol)
    return _fit_tweedie(fam, yv, Xμ, Xσ, Xν, nmμ, nmσ, nmν, g_tol).theta
end
function _indep_fixed_start(::Val{:beta}, fam, yv, Xμ, Xσ, Xν, nmμ, nmσ, nmν, g_tol)
    return _fit_beta(fam, yv, Xμ, Xσ, nmμ, nmσ, g_tol).theta
end

_indep_primal(x::ForwardDiff.Dual) = _indep_primal(ForwardDiff.value(x))
_indep_primal(x::Real) = Float64(x)

# The objective returns its failure value when a random-effect log SD (or the log sigma) leaves
# [-30, 30]. The value is kept at 30. A fit that ends within `_INDEP_WALL_TOL` of the LOWER end has an
# SD that is zero for every practical purpose: it is reported as a boundary fit (see
# `convergence_status`), not as an interior optimum.
const _INDEP_LOGSD_GUARD = 30.0
const _INDEP_WALL_TOL = 1.0

"""
    _fit_indep_re_laplace(fam, kind, y, Xμ, Xσ, Xν, comps, nmμ, nmσ, nmν, g_tol) -> DrmFit

Laplace fit of `Gamma()` (`kind = Val(:gamma)`, `Xν = nothing`) or `Student()`
(`kind = Val(:student)`) with independent scalar random-effect components on the
mean. `comps` is a vector of `(w, gidx, G, label)` (the Gaussian multi-RE
convention; `w` is the slope covariate or ones). θ = [β_μ; β_σ; (β_ν;) log σ_k].
Blocks `:mu`, `:sigma`, (`:nu`,) `:resd`. Tagged `marginal = :Laplace`.
"""
function _fit_indep_re_laplace(fam, kind::Val, y, Xμ, Xσ, Xν, comps, nmμ, nmσ, nmν, g_tol;
                               start = nothing)
    n = length(y); yv = _indep_response(kind, y)
    has_nu = Xν !== nothing
    has_sigma = Xσ !== nothing                         # Poisson and Binomial have no dispersion
    pμ = size(Xμ, 2); pσ = has_sigma ? size(Xσ, 2) : 0; pν = has_nu ? size(Xν, 2) : 0
    K = length(comps)
    Gks = [c[3] for c in comps]
    offs = cumsum([0; Gks]); q = offs[end]
    W = [Float64.(c[1]) for c in comps]
    cols = [offs[k] .+ comps[k][2] for k in 1:K]       # column of b loaded by obs i, comp k
    k0 = pμ + pσ + pν                                  # last fixed-effect index
    # Group-level SD designs. A component `(w, gidx, G, label, Zg, nmsd)` carries an `sd(...) ~ ...`
    # regression, log σ_{kg} = Zg[g, :]·α_k; a 4-tuple component has the constant SD of the
    # no-formula route (Zg = 1, α_k = log σ_k), which this reduces to exactly.
    has_sdreg = any(c -> length(c) >= 6, comps)
    Zs = [length(c) >= 6 ? Matrix{Float64}(c[5]) : ones(c[3], 1) for c in comps]
    msd = [size(Z, 2) for Z in Zs]
    sdoffs = k0 .+ cumsum([0; msd]); nα = sdoffs[end] - k0

    function joint_terms(b, η0, ησ, ην, invs; expected::Bool = false)
        T = promote_type(eltype(b), eltype(η0), eltype(ησ), eltype(ην), eltype(invs))
        H = zeros(T, q, q); grad = zeros(T, q); data = zero(T)
        @inbounds for i in 1:n
            η = η0[i]
            for k in 1:K
                η += W[k][i] * b[cols[k][i]]
            end
            v, g1, w, wE = _indep_obs_terms(kind, yv[i], η, ησ[i], ην[i])
            expected && (w = wE)
            data += v
            for k in 1:K
                wk = W[k][i]; iszero(wk) && continue
                ck = cols[k][i]
                grad[ck] += wk * g1
                for l in 1:K
                    wl = W[l][i]; iszero(wl) && continue
                    H[ck, cols[l][i]] += wk * wl * w
                end
            end
        end
        pen = zero(T)
        @inbounds for j in 1:q
            iv = invs[j]
            grad[j] += iv * b[j]; H[j, j] += iv
            pen += iv * b[j]^2
        end
        return data + 0.5 * pen, grad, H
    end
    function inner_mode(η0, ησ, ην, invs, b0; maxiter::Int = 200, tol::Real = 1e-10)
        b = copy(b0)
        for it in 1:maxiter
            J0, grad, H = joint_terms(b, η0, ησ, ην, invs)
            isfinite(J0) || return b, false
            ch = cholesky(Symmetric(H); check = false)
            if !issuccess(ch)                          # indefinite: expected-information direction
                ch = cholesky(Symmetric(last(joint_terms(b, η0, ησ, ην, invs; expected = true)));
                              check = false)
                issuccess(ch) || return b, false
            end
            step = ch \ grad
            norm(step) <= tol * (1 + norm(b)) && return b, true
            # Rounding floor: once the predicted decrease (the Newton decrement) is below
            # what `J0` can resolve, the line search below rejects every step (`Jt <= J0`
            # is decided by rounding), halves it to nothing and loops to `maxiter`, which
            # reported a failed inner solve at a benign point (Gamma `(1 | g) + (0 + x | g)`
            # wave-10 cell: |step| stalled at 1.4e-8, |grad| at 2.6e-7). Take the step and stop.
            if 0.5 * dot(grad, step) <= 1e-13 * (1 + abs(J0))
                return b .- step, true
            end
            # Rounding floor: Newton converges in well under 25 iterations; a step that is still
            # at 1e-6 relative after that is stuck at the noise floor of the objective (NB2 with
            # heavy-tailed counts stalls at about 2e-8), not diverging.
            it > 25 && norm(step) <= 1e-6 * (1 + norm(b)) && return b, true
            α = 1.0; accepted = false
            while α >= 1e-10
                trial = b .- α .* step
                Jt = first(joint_terms(trial, η0, ησ, ην, invs))
                if isfinite(Jt) && Jt <= J0
                    b = trial; accepted = true; break
                end
                α /= 2
            end
            accepted || return b, norm(grad, Inf) <= 1e-6 * (1 + n)
        end
        return b, false
    end

    last_b = zeros(q)
    function nll(θ)
        T = eltype(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]
        lsg = vcat([Zs[k] * θ[sdoffs[k]+1:sdoffs[k+1]] for k in 1:K]...)   # log SD of every effect
        any(l -> abs(_indep_primal(l)) > _INDEP_LOGSD_GUARD, lsg) && return T(1e18)
        η0 = Xμ * βμ
        ησ, ην = _indep_aux_eta(kind, yv, has_sigma ? Xσ : nothing, has_nu ? Xν : nothing, θ, pμ, pσ, k0, T)
        any(l -> abs(_indep_primal(l)) > _INDEP_LOGSD_GUARD, ησ) && return T(1e18)
        invs = exp.(-2 .* lsg)
        f0 = (_indep_primal.(η0), _indep_primal.(ησ), _indep_primal.(ην), _indep_primal.(invs))
        b̂, ok = inner_mode(f0..., last_b)
        ok || ((b̂, ok) = inner_mode(f0..., zeros(q)))
        ok || return T(1e18)
        T === Float64 && (last_b .= b̂)
        b = Vector{T}(b̂)
        for _ in 1:2                                   # lift b̂ to b̂(θ) in dual arithmetic
            _, grad, H = joint_terms(b, η0, ησ, ην, invs)
            ch = cholesky(Symmetric(H); check = false)
            issuccess(ch) || return T(1e18)
            b = b .- (ch \ grad)
        end
        J, _, H = joint_terms(b, η0, ησ, ην, invs)
        ch = cholesky(Symmetric(H); check = false)
        issuccess(ch) || return T(1e18)
        val = J + sum(lsg) + 0.5 * logdet(ch)
        isfinite(val) || return T(1e18)
        return val
    end

    # Interior start. Gamma: the fixed-effects fit (shape and mean are well behaved).
    # Student: OLS mean, sigma at half the residual spread, nu = 2 + e^eta at 12 — NOT the
    # fixed-effects Student fit, which sits at the nu -> 2 boundary when the random
    # effects are ignored and then stalls there (ChickWeight: 18 logLik short of drmTMB).
    θ0 = zeros(k0 + nα)
    if kind isa Val{:student}
        βμ0 = Xμ \ yv
        θ0[1:pμ] .= βμ0
        θ0[pμ+1] = log(std(yv - Xμ * βμ0) + eps()) - log(2.0)
        θ0[pμ+pσ+1] = log(10.0)
    else
        θ0[1:k0] .= _indep_fixed_start(kind, fam, yv, Xμ, Xσ, Xν, nmμ, nmσ, nmν, g_tol)[1:k0]
    end
    base = kind isa Val{:student} ? θ0[pμ+1] : log(0.3)
    for k in 1:K
        sw = all(==(1.0), W[k]) ? 1.0 : max(std(W[k]), 1e-8)
        θ0[sdoffs[k]+1] = base - log(sw)
    end
    if start !== nothing                               # caller-supplied start (the zero-mass multi-start)
        length(start) == length(θ0) || throw(ArgumentError("start has length $(length(start)), expected $(length(θ0))"))
        θ0 .= Float64.(start)
    end
    # BFGS first: on the near-flat slope-variance ridge L-BFGS stalled at a 1e18 line-search
    # probe (simulated growth data) while BFGS reaches drmTMB's optimum in 5–25 iterations.
    opts = Optim.Options(g_tol = g_tol, iterations = 1000, f_reltol = NaN, x_abstol = NaN, x_reltol = NaN)
    res = Optim.optimize(nll, θ0, Optim.BFGS(), opts; autodiff = :forward)
    for k in 1:8                                       # bounded restarts (see student.jl #827)
        (Optim.converged(res) && nll(Optim.minimizer(res)) < 1e17) && break
        res2 = Optim.optimize(nll, Optim.minimizer(res), isodd(k) ? Optim.LBFGS() : Optim.BFGS(),
                              opts; autodiff = :forward)
        nll(Optim.minimizer(res2)) < nll(Optim.minimizer(res)) || break
        res = res2
    end
    θ̂ = Optim.minimizer(res)
    nllhat = nll(θ̂)
    gfinal = ForwardDiff.gradient(nll, θ̂)
    converged = nllhat < 1e17 && all(isfinite, gfinal) &&
                (Optim.converged(res) || _laplace_outer_converged(res, nllhat, gfinal, θ̂, n, g_tol))
    famname = _indep_famname(kind)
    V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂);
                           context = "Laplace $famname (independent random intercepts/slopes)")
    labels = [c[4] for c in comps]
    blocks = Pair{Symbol,UnitRange{Int}}[:mu => 1:pμ]
    names = Pair{Symbol,Vector{String}}[:mu => nmμ]
    sigblock = _indep_sigma_block(kind)                # `:sigma`, or `:cutpoints` for the ordinal kind
    has_sigma && (push!(blocks, sigblock => (pμ+1):(pμ+pσ)); push!(names, sigblock => nmσ))
    nublock = _indep_nu_block(kind)                    # `:nu`, or `:zi` / `:hu` for the zero-mass kinds
    has_nu && (push!(blocks, nublock => (pμ+pσ+1):k0); push!(names, nublock => nmν))
    if has_sdreg                                       # one `:sd` block of all α_k, as the Gaussian multi route
        sdnames = String[]
        for k in 1:K
            nmk = length(comps[k]) >= 6 ? comps[k][6] : ["(Intercept)"]
            append!(sdnames, K > 1 ? string.(labels[k], ": ", nmk) : String.(nmk))
        end
        push!(blocks, :sd => (k0+1):(k0+nα)); push!(names, :sd => sdnames)
    else
        push!(blocks, :resd => (k0+1):(k0+K)); push!(names, :resd => labels)
    end
    ημ̂ = Xμ * θ̂[1:pμ]
    means = Dict(:mu => _indep_mean.(Ref(kind), ημ̂)); obs = Dict(:mu => _indep_obsval(kind, yv))   # population mean (b = 0)
    scales = has_sigma ? Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)])) : Dict{Symbol,Vector{Float64}}()
    has_nu && (scales[nublock] = _indep_nu_scale(kind, Xν * θ̂[(pμ+pσ+1):k0]))
    means, scales = _indep_post(kind, yv, means, scales, ημ̂, θ̂, pμ, pσ)
    kind isa Val{:binomial} && (scales[:trials] = Float64[t for (_, t) in yv])   # per-row trials, as every other Binomial route
    if _indep_records_modes(kind)
        # Conditional modes at θ̂ (a fresh inner solve from zero, the same solver as the objective),
        # kept as the per-row contribution Z b̂ for `fitted(fit)` (conditional default).
        ησ̂, ην̂ = _indep_aux_eta(kind, yv, has_sigma ? Xσ : nothing, has_nu ? Xν : nothing, θ̂, pμ, pσ, k0, Float64)
        lsĝ = vcat([Zs[k] * θ̂[sdoffs[k]+1:sdoffs[k+1]] for k in 1:K]...)
        b̂m, okm = inner_mode(Xμ * θ̂[1:pμ], ησ̂, ην̂, exp.(-2 .* lsĝ), zeros(q))
        if okm && all(isfinite, b̂m)
            _record_cond_fit!(scales, [sum(W[k][i] * b̂m[cols[k][i]] for k in 1:K) for i in 1:n])
        else
            scales[:cond_fit_failed] = [1.0]
        end
    end
    lsg_hat = vcat([Zs[k] * θ̂[sdoffs[k]+1:sdoffs[k+1]] for k in 1:K]...)
    if nllhat < 1e17 && minimum(lsg_hat) <= -(_INDEP_LOGSD_GUARD - _INDEP_WALL_TOL)
        scales[:cond_logsd_wall] = [minimum(lsg_hat)]
        @warn "A random-effect log SD reached the lower search bound ($(-_INDEP_LOGSD_GUARD)): the fitted SD is " *
              "effectively zero, so this is a boundary fit (`convergence_status(fit)` is `:boundary`) and " *
              "Wald standard errors are not reliable. To test that variance against zero use " *
              "`lrt_boundary(fit, reduced)` with a reduced fit that drops the term (50:50 chi-bar " *
              "mixture). Profile intervals may not converge at the wall."
    end
    fit = DrmFit(fam, blocks, names, θ̂, V, -nllhat, n, converged, means, obs, scales)
    return _withiterations(_withmarginal(_withnll(fit, nll), :Laplace), Optim.iterations(res))
end

# Build `(w, gidx, G, label)` components for a mean made only of `(1 | g)` and
# `(0 + x | g)` terms with at least one slope; `nothing` otherwise (the caller
# keeps its existing routes). Slope labels are `"g:x"` so `vc`/`re_sd` keep the
# intercept and slope SDs of one factor apart.
function _has_indep_slopes(re)
    kinds = [_re_kind(r[1])[1] for r in re]
    return all(in((:intercept, :slope)), kinds) && any(==(:slope), kinds)
end

function _indep_slope_comps(re, data, n::Int)
    _has_indep_slopes(re) || return nothing
    return _indep_term_comps(re, data, n)
end

# The components of independent `(1 | g)` / `(0 + x | g)` terms, with or without a slope among them.
function _indep_term_comps(re, data, n::Int)
    kinds = [_re_kind(r[1]) for r in re]
    return map(zip(re, kinds)) do ((_, grp), (kind, var))
        gidx, G = _group_index(getproperty(data, grp))
        w = kind === :intercept ? ones(n) : Float64.(getproperty(data, var))
        (w, gidx, G, kind === :intercept ? String(grp) : "$(grp):$(var)")
    end
end

# ---------------------------------------------------------------------------
# `sd(g, coef(·)) ~ …` formulas on independent terms (and on crossed intercepts), non-Gaussian.
#
# Same dense TMB-convention Laplace as above with a group-level log-SD regression per component,
# log σ_{kg} = Z_k[g, ·] α_k (a component without a formula keeps the constant SD). This is the
# route drmTMB takes for these shapes; the family's own single-term SD routes keep their
# integrators and are not used here.
# ---------------------------------------------------------------------------

# The shape this route takes: two or more independent terms `(1 | g)` / `(0 + x | g)` on the
# mean, no structured or `meta_V` terms, and only ordinary mean `sd()` targets. `nothing` otherwise.
function _indep_sd_shape(f::DrmFormula; min_terms::Int = 2)
    rhs = Dict(f.forms)
    haskey(rhs, :mu) || return nothing
    # A structured term (`animal`, `relmat`, `spatial`, ...) with a slope makes `_split_ranef` throw its own
    # message; that is not this route's shape, so leave the error to the route that owns the term.
    split = try
        _split_ranef(rhs[:mu])
    catch err
        err isa ArgumentError || rethrow()
        return nothing
    end
    fixed_mu, re, mv, st = split
    (mv === nothing && st === nothing && length(re) >= min_terms) || return nothing
    all(r -> _laplace_lhs_kind(r[1]) in (:intercept, :slope), re) || return nothing
    kinds = [_re_kind(r[1]) for r in re]
    all(k -> k[1] in (:intercept, :slope), kinds) || return nothing
    isempty(_sdphylo_parts(f)) || return nothing
    targets = _sd_target_parts(f)
    isempty(targets) && return nothing
    all(item -> item.target.level === :iid && item.target.dpar === :mu &&
                item.target.block === nothing, targets) || return nothing
    return (; fixed_mu, re, kinds, targets)
end

# Build the components `(w, gidx, G, label, Zg, nmsd)` of an `sd()` model on independent terms.
function _indep_sd_comps(f::DrmFormula, shape, data, famname::AbstractString)
    re, kinds, targets = shape.re, shape.kinds, shape.targets
    n = length(getproperty(data, f.response))
    mem = Symbol[k === :intercept ? :intercept : v for (k, v) in kinds]
    ids = [(re[i][2], mem[i]) for i in eachindex(re)]
    allunique(ids) || throw(ArgumentError(
        "drm ($famname): repeated random-effect term on one grouping in the mean formula."))
    used = Set{Int}()
    comps = Any[]
    for i in eachindex(re)
        grp = re[i][2]; m = mem[i]
        hasproperty(data, grp) || throw(ArgumentError(
            "drm ($famname): grouping variable `$grp` of the random-effect term is not a column of `data`."))
        gidx, G = _group_index(getproperty(data, grp); col = grp)
        hit = findall(targets) do item
            t = item.target
            t.group === grp && (t.member === m || (t.member === nothing && m === :intercept))
        end
        length(hit) ≤ 1 || throw(ArgumentError(
            "drm ($famname): one `sd($grp, coef($m)) ~ …` formula per term."))
        if isempty(hit)
            Zg = ones(G, 1); nm = ["(Intercept)"]
        else
            item = targets[hit[1]]
            push!(used, hit[1])
            Zg, nm = _sd_group_design(f.response, item.rhs, data, gidx, G, grp;
                                      schema_cache = f.schema_cache, schema_key = item.key)
            size(Zg, 2) ≤ G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
                "drm ($famname): the SD target for `$m` has a rank-deficient group-level design."))
        end
        w = ones(n)
        if m !== :intercept
            _require_finite_data_column(data, m)
            w = Float64.(getproperty(data, m))
        end
        label = m === :intercept ? string(grp) : "$(grp):$(m)"
        push!(comps, (w, gidx, G, label, Matrix{Float64}(Zg), Vector{String}(nm)))
    end
    length(used) == length(targets) || throw(ArgumentError(
        "drm ($famname): an `sd()` target names a coefficient that is not a random-effect term in " *
        "the mean formula (terms: " * join(string.(ids), ", ") * ")."))
    return comps
end

# Shared guards: the model may carry only `mu`, the family's own dispersion formulas and SD formulas;
# no offset, no missing responses.
function _indep_sd_guard(f::DrmFormula, shape, data, famname::AbstractString; extra = (:sigma,))
    allowed = Set(vcat([:mu], collect(extra), [item.key for item in shape.targets]))
    all(pair -> first(pair) in allowed || (first(pair) === :sigma && last(pair) == ConstantTerm(1)),
        f.forms) || throw(ArgumentError(
        "drm ($famname): `sd()` formulas on independent random-effect terms are fitted with only " *
        "the `mu`" * join((", `$e`" for e in extra)) * " and `sd()` formulas."))
    for pn in extra                                    # fixed dispersion submodels only, as the single-term SD routes
        r = get(Dict(f.forms), pn, ConstantTerm(1))
        r == ConstantTerm(1) || throw(ArgumentError(
            "drm ($famname): `sd()` formulas on independent random-effect terms require a fixed " *
            "`$pn ~ 1` submodel."))
    end
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "drm ($famname): `sd()` formulas on independent random-effect terms require observed " *
        "responses; remove incomplete rows first."))
    _, off = _extract_offset(shape.fixed_mu)
    off === nothing || throw(ArgumentError(
        "drm ($famname): `offset(...)` is not available with `sd()` formulas on independent terms."))
    return nothing
end

# Poisson / NB2 / Gamma / Beta / Tweedie / Student on the log-/logit-/identity-mean (response `y`).
function _drm_indep_sd_laplace(f::DrmFormula, fam, kind::Val, shape; data, g_tol::Real = 1e-8)
    famname = _indep_famname(kind)
    has_sigma = !(kind isa Union{Val{:poisson},Val{:binomial}})
    has_nu = kind isa Union{Val{:student},Val{:tweedie}}
    extra = has_nu ? (:sigma, :nu) : has_sigma ? (:sigma,) : ()
    _indep_sd_guard(f, shape, data, famname; extra)
    rhs = Dict(f.forms)
    fixed_mu, _ = _extract_offset(shape.fixed_mu)
    y, Xμ, nmμ = _design(f.response, fixed_mu, data)
    if kind isa Val{:binomial}
        y, ntr = _binomial_response(f, data)
        y = collect(zip(y, ntr))
    elseif kind isa Union{Val{:poisson},Val{:nb2}}
        all(yi -> yi >= 0 && isinteger(yi), y) ||
            throw(ArgumentError("$famname requires non-negative integer counts as the response"))
    elseif kind isa Val{:gamma}
        all(yi -> yi > 0, y) || throw(ArgumentError("Gamma() requires strictly positive responses"))
    elseif kind isa Val{:beta}
        all(yi -> 0 < yi < 1, y) ||
            throw(ArgumentError("Beta() requires responses strictly in the open interval (0, 1)"))
    elseif kind isa Val{:tweedie}
        all(yi -> yi >= 0, y) || throw(ArgumentError("Tweedie() requires non-negative responses"))
    end
    Xσ = nmσ = Xν = nmν = nothing
    if has_sigma
        _, Xσ, nmσ = _design(f.response, get(rhs, :sigma, ConstantTerm(1)), data)
    else
        nmσ = String[]
    end
    if has_nu
        _, Xν, nmν = _design(f.response, get(rhs, :nu, ConstantTerm(1)), data)
    else
        nmν = String[]
    end
    comps = _indep_sd_comps(f, shape, data, famname)
    return _withformula(_fit_indep_re_laplace(fam, kind, y, Xμ, Xσ, Xν, comps, nmμ, nmσ, nmν, g_tol), f)
end

# Opt-in `marginal = :Laplace` on the single-term and independent-term `sd()` routes (the
# quadrature routes stay the default). Returns `nothing` unless the Laplace objective was asked for;
# then one independent term or more, each `(1 | g)` or `(0 + x | g)`, with `sd(g, coef(.)) ~ ...`
# formulas on the mean, are fitted by the dense TMB-convention Laplace of this file, which is the
# objective drmTMB optimises. Every other `sd()` shape is refused rather than rerouted.
function _laplace_sd_route(f::DrmFormula, fam, kind::Val, marginal; data, g_tol::Real = 1e-8,
                           fallback::Bool = false)
    _scalar_laplace_requested(marginal) || return nothing
    fam isa Binomial && fam.link !== :logit && throw(ArgumentError(
        "Binomial links other than :logit are not available with `marginal = :Laplace`."))
    shape = _indep_sd_shape(f; min_terms = 1)
    shape === nothing && fallback && return nothing     # the caller owns this shape
    shape === nothing && throw(ArgumentError(
        "drm ($(_indep_famname(kind))): `marginal = :Laplace` with `sd()` formulas covers independent " *
        "intercept and numeric-slope terms `(1 | g)` / `(0 + x | g)` on the mean only. Use `marginal = :LA` " *
        "(the default) for structured, correlated, or distributional-parameter SD formulas."))
    return _drm_indep_sd_laplace(f, fam, kind, shape; data, g_tol)
end
