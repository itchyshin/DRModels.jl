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
    if length(re) > 1                                     # crossed intercepts → Laplace (#725)
        (length(re) == 2 && all(_re_kind(r[1])[1] === :intercept for r in re)) ||
            error("Student() supports multiple random effects on the mean only as two crossed/nested intercepts, e.g. `(1 | g) + (1 | h)`")
        comps = map(re) do r
            gidx, G = _group_index(getproperty(data, r[2]))
            (gidx, G, String(r[2]))
        end
        return _withformula(_fit_student_crossed_laplace(fam, y, Xμ, Xσ, Xν, comps, nmμ, nmσ, nmν, g_tol), f)
    end
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
function _fit_student_ranef(fam::Student, y, Xμ, Xσ, Xν, gidx, G, nmμ, nmσ, nmν, grp, g_tol; K::Int = _RANEF1D_AGHQ_K)
    n = length(y); pμ, pσ, pν = size(Xμ, 2), size(Xσ, 2), size(Xν, 2)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    rule = _AGHQRule(1, K); Zre = ones(n, 1); bcache = zeros(1, G)   # #719: per-group AGHQ
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; βν = θ[pμ+pσ+1:pμ+pσ+pν]; σb = exp(θ[pμ+pσ+pν+1])
        η0 = Xμ * βμ; ησ = Xσ * βσ; ην = Xν * βν     # μ identity → no exp clamp on the mean
        ll = (i, η) -> (zt = (y[i] - η) * exp(-ησ[i]); _student_logpdf_std(zt, ην[i]) - ησ[i])
        L = reshape([σb], 1, 1)
        return -_aghq_marginal_loglik(ll, members, η0, Zre, L, rule, bcache)
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
# and each group is integrated by per-group ADAPTIVE Gauss–Hermite quadrature
# (`_aghq_marginal_loglik`, #834: nodes b̂_g + √2 C z at each group's mode),
# `nq` nodes per axis. Σ is the log-Cholesky parameterisation L = [exp(a) 0; cc exp(b)]
# (the `vc` convention), so vc(fit) reconstructs Σ = L Lᵀ. The scale σ and df ν stay
# fixed effects. θ = [βμ; βσ; βν; a, b, cc].
function _fit_student_corr_ranef(fam::Student, y, Xμ, Xσ, Xν, xs, gidx, G, nmμ, nmσ, nmν, grp, g_tol; nq::Int = _CORR_RANEF_AGHQ_K)
    n = length(y); pμ, pσ, pν = size(Xμ, 2), size(Xσ, 2), size(Xν, 2)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    rule = _AGHQRule(2, nq); Zre = hcat(ones(n), Float64.(xs)); bcache = zeros(2, G)   # #834: per-group AGHQ
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; βν = θ[pμ+pσ+1:pμ+pσ+pν]
        L = _corr_ranef_L(θ[pμ+pσ+pν+1], θ[pμ+pσ+pν+2], θ[pμ+pσ+pν+3])
        η0 = Xμ * βμ; ησ = Xσ * βσ; ην = Xν * βν
        ll = (i, η) -> (zt = (y[i] - η) * exp(-ησ[i]); _student_logpdf_std(zt, ην[i]) - ησ[i])
        return -_aghq_marginal_loglik(ll, members, η0, Zre, L, rule, bcache)
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

# Strip (possibly nested) ForwardDiff duals to Float64.
_student_fval(x::ForwardDiff.Dual) = _student_fval(ForwardDiff.value(x))
_student_fval(x::Real) = Float64(x)

# Student-t GLMM with two crossed (or nested) random intercepts on the mean,
# `(1 | g) + (1 | h)` (#725; drmTMB `student()` fits these by TMB's Laplace).
# b = [b_g; b_h] ~ N(0, diag(σ_g² I, σ_h² I)); the marginal likelihood is the
# Laplace approximation
#   nll(θ) = Σᵢ −log f(yᵢ | b̂) + ½ b̂ᵀΛb̂ + G log σ_g + H log σ_h + ½ logdet 𝐇(b̂),
# 𝐇 = ZᵀWZ + Λ, W = the observed (not expected) second derivative of −log f in μ.
# σ and ν may carry their own fixed-effect formulas. The Student data term is
# not log-concave, so the inner Newton falls back to the always-positive expected
# information (ν+1)/((ν+3)σ²) for its direction when 𝐇 is indefinite, with a
# line search; the Laplace term uses the observed 𝐇 and the evaluation fails
# closed (nll = 1e18) if 𝐇 is not positive definite at the mode. Derivatives are
# exact ForwardDiff: the Float64 mode b̂ is lifted to the dual numbers by two
# Newton steps at θ, which carry db̂/dθ (implicit-function theorem) accurately
# to second order, so the Hessian for vcov is exact as well.
# 𝐇 is dense (G+H)²: fine for hundreds of levels, slow for many thousands.
# θ = [βμ; βσ; βν; log σ_g; log σ_h].
function _fit_student_crossed_laplace(fam::Student, y, Xμ, Xσ, Xν, comps, nmμ, nmσ, nmν, g_tol)
    n = length(y); pμ, pσ, pν = size(Xμ, 2), size(Xσ, 2), size(Xν, 2)
    (gidx, G, lg), (hidx, Hh, lh) = comps
    q = G + Hh
    yv = Float64.(y)
    # −log f and its μ-derivatives for observation i at mean μ. With t = (y−μ)/σ
    # and r = 1/ν (overflow-free, as in `_student_logpdf_std`):
    #   d/dμ = −(1+r) t / (σ(1 + r t²)),   d²/dμ² = (1+r)(1 − r t²) / (σ²(1 + r t²)²).
    function obs_terms(i, μ, lσ, ην)
        σ = exp(lσ); t = (yv[i] - μ) / σ
        r = ην > 0 ? exp(-ην) / (1 + 2 * exp(-ην)) : 1 / (2 + exp(ην))
        d = 1 + r * t^2
        val = lσ - _student_logpdf_std(t, ην)
        g1 = -(1 + r) * t / (σ * d)
        w = (1 + r) * (1 - r * t^2) / (σ^2 * d^2)
        wE = (1 + r) / ((1 + 3r) * σ^2)
        return val, g1, w, wE
    end
    function joint_terms(b, η0, ησ, ην, invg, invh; expected::Bool = false)
        T = promote_type(eltype(b), eltype(η0), eltype(ησ), eltype(ην), typeof(invg))
        H = zeros(T, q, q); grad = zeros(T, q); data = zero(T)
        @inbounds for i in 1:n
            a = gidx[i]; c = G + hidx[i]
            v, g1, w, wE = obs_terms(i, η0[i] + b[a] + b[c], ησ[i], ην[i])
            expected && (w = wE)
            data += v; grad[a] += g1; grad[c] += g1
            H[a, a] += w; H[c, c] += w; H[a, c] += w; H[c, a] += w
        end
        @inbounds for j in 1:G
            grad[j] += invg * b[j]; H[j, j] += invg
        end
        @inbounds for j in (G+1):q
            grad[j] += invh * b[j]; H[j, j] += invh
        end
        joint = data + 0.5 * invg * sum(abs2, @view b[1:G]) + 0.5 * invh * sum(abs2, @view b[G+1:q])
        return joint, grad, H
    end
    function inner_mode(η0, ησ, ην, invg, invh, b0; maxiter::Int = 100, tol::Real = 1e-10)
        b = copy(b0)
        for _ in 1:maxiter
            J0, grad, H = joint_terms(b, η0, ησ, ην, invg, invh)
            ch = cholesky(Symmetric(H); check = false)
            if !issuccess(ch)                     # indefinite: expected-information direction
                ch = cholesky(Symmetric(last(joint_terms(b, η0, ησ, ην, invg, invh; expected = true)));
                              check = false)
                issuccess(ch) || return b, false  # fail closed: even the expected info is not PD here
            end
            step = ch \ grad
            norm(step) <= tol * (1 + norm(b)) && return b, true
            α = 1.0; accepted = false
            while α >= 1e-8
                trial = b .- α .* step
                if first(joint_terms(trial, η0, ησ, ην, invg, invh)) <= J0
                    b = trial; accepted = true; break
                end
                α /= 2
            end
            # A near-zero crossed variance makes invg/invh (and so H's diagonal) large,
            # which raises the floating-point floor the Newton gradient can reach before
            # line-search steps underflow -- confirmed on #827's sdh0 data: the gradient
            # plateaus at ~4e-6 (well converged in relative terms) but never crosses the
            # old 1e-8*(1+n) floor, so the line-search failure reads as non-convergence
            # and poisons the outer nll with a spurious 1e18. Loosened by 100x (still a
            # tight absolute tolerance relative to a Hessian diagonal of order invh).
            accepted || return b, norm(grad, Inf) <= 1e-6 * (1 + n)
        end
        return b, false
    end
    last_b = zeros(q)
    function nll(θ)
        T = eltype(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; βν = θ[pμ+pσ+1:pμ+pσ+pν]
        lsg = θ[pμ+pσ+pν+1]; lsh = θ[pμ+pσ+pν+2]
        # Wide guard, not a tight wall (#827 review): a crossed variance genuinely at
        # its zero boundary reports as log σ ≈ −12 to −13 (drmTMB's sdh0 case puts it
        # at −12.57), so a wall at 12 rejects a legitimate MLE outright. invg/invh stay
        # finite (no Float64 overflow) out to |log σ| ≈ 350, so 30 is generous headroom
        # while still catching a runaway line-search probe before invg/invh overflow.
        (abs(_student_fval(lsg)) > 30 || abs(_student_fval(lsh)) > 30) && return T(1e18)
        η0 = Xμ * βμ; ησ = Xσ * βσ; ην = Xν * βν
        invg = exp(-2lsg); invh = exp(-2lsh)
        (isfinite(_student_fval(invg)) && isfinite(_student_fval(invh))) || return T(1e18)
        f0 = (_student_fval.(η0), _student_fval.(ησ), _student_fval.(ην),
              _student_fval(invg), _student_fval(invh))
        b̂, ok = inner_mode(f0..., last_b)
        ok || ((b̂, ok) = inner_mode(f0..., zeros(q)))
        ok || return T(1e18)
        T === Float64 && (last_b .= b̂)
        b = b̂
        for _ in 1:2                              # lift b̂ to b̂(θ) in dual arithmetic
            _, grad, H = joint_terms(b, η0, ησ, ην, invg, invh)
            ch = cholesky(Symmetric(H); check = false)
            issuccess(ch) || return T(1e18)
            b = b .- (ch \ grad)
        end
        J, _, H = joint_terms(b, η0, ησ, ην, invg, invh)
        ch = cholesky(Symmetric(H); check = false)
        issuccess(ch) || return T(1e18)
        return J + G * lsg + Hh * lsh + 0.5 * logdet(ch)
    end
    βμ0 = Xμ \ yv
    k = pμ + pσ + pν
    θ0 = zeros(k + 2)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(std(yv - Xμ * βμ0) + eps()) - log(2.0)   # σ init (REs take a share)
    θ0[pμ+pσ+1] = log(10.0)                                  # ν init (mildly heavy-tailed)
    θ0[k+1] = θ0[pμ+1]; θ0[k+2] = θ0[pμ+1]                   # σ_g, σ_h init
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol, iterations = 1000);
                         autodiff = :forward)
    # Repeated restart from θ̂ on non-convergence (#827 review; mirrors #837): a non-converged
    # LBFGS run can stop early after its line search crosses the fail-closed 1e18 region even
    # on ordinary, well-conditioned data (confirmed: restarting from θ̂ walks the reviewer's
    # "gauss" data to drmTMB's optimum). Each restart's own line search can fail again a few
    # steps later (still short of the outer g_tol), so this restarts a bounded number of times
    # rather than once, keeping whichever minimizer is best each time. `Optim.minimum` is NOT
    # trustworthy for that comparison: on a "line search failed" result it reports the value of
    # the LAST (rejected) trial point, not of `Optim.minimizer` (confirmed: the restart's own
    # minimizer was a genuine improvement, nll 296.22 vs 296.37, while its reported `minimum`
    # read 1e18) -- so `nll` is re-evaluated at each minimizer directly instead.
    for _ in 1:8
        Optim.converged(res) && break
        res2 = Optim.optimize(nll, Optim.minimizer(res), Optim.LBFGS(),
                              Optim.Options(g_tol = g_tol, iterations = 1000); autodiff = :forward)
        improved = nll(Optim.minimizer(res2)) < nll(Optim.minimizer(res))
        improved || break
        res = res2
    end
    θ̂ = Optim.minimizer(res)
    nllhat = nll(θ̂)
    gfinal = ForwardDiff.gradient(nll, θ̂)
    converged = nllhat < 1e17 && all(isfinite, gfinal) &&
                (Optim.converged(res) || _laplace_outer_converged(res, nllhat, gfinal, θ̂, n, g_tol))
    V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂);
                           context = "sparse-Laplace Student (crossed intercepts)")
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :nu => (pμ+pσ+1):k, :resd => (k+1):(k+2)]
    names = [:mu => nmμ, :sigma => nmσ, :nu => nmν, :resd => [lg, lh]]
    means = Dict(:mu => Xμ * θ̂[1:pμ]); obs = Dict(:mu => yv)      # population μ (b = 0)
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]),
                  :nu => 2 .+ exp.(Xν * θ̂[(pμ+pσ+1):k]))
    return _withiterations(
        _withnll(DrmFit(fam, blocks, names, θ̂, V, -nllhat, n, converged, means, obs, scales), nll),
        Optim.iterations(res))
end
