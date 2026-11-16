# Gaussian sigma-side random effects WITH an SD formula, fitted by a dense
# Laplace approximation (twin of drmTMB `test-gaussian-mu-sigma-joint-sd.R` and
# `test-gaussian-sigma-structured-sd.R`). See `DESIGN-joint-sigma.md`.
#
# Model (standardised latent v, effects = scale * v):
#
#   y_i ~ N(mu_i + c_i va[ga_i],  exp(2 eta_i)),   eta_i = Xσ βσ + d_i vb[gb_i]
#   c_i = exp(Za αa)[ga_i]                   (mean intercept, optional)
#   d_i = loading_i * exp(Zb αb)[gb_i]       (sigma intercept/slope or structured)
#   va ~ N(0, I),   vb ~ N(0, K)   (K = I for an ordinary effect; raw animal/relmat K)
#
# No mean/sigma cross-correlation (independent latent fields, as drmTMB). The
# marginal is not Gaussian, so the objective is the Laplace value TMB reports,
#   nll(θ) = F(v̂) + ½ log det H(v̂) + ½ log det K,
# with F the joint negative log density in v (the ½ m log 2π of the prior and of
# the Laplace expansion cancel). Gradient: exact forward-mode AD of the Laplace
# value in which v̂ is replaced by ONE Newton step from the converged Float64
# mode; because the inner gradient is zero there, that step has exactly the
# implicit derivative dv̂/dθ = -H⁻¹ ∂g/∂θ, so the first derivative is exact.
# The outer loop, boundary polish, convergence rule and Wald covariance are the
# shared `_ordinary_laplace_optimize`.

const _GSL_MAX_LATENT = 600     # dense H: refuse beyond this many latent coordinates

# Per-observation negative log density f(lin, eta) and its derivatives in the
# mean linear predictor `lin` and the log-sigma predictor `eta`:
#   (f, f_lin, f_eta, f_lin_lin, f_lin_eta, f_eta_eta).
# Gaussian: identity mean, sd = exp(eta). NB2: mean exp(lin), size k = exp(-2 eta)
# (variance mu + mu^2 / k, as drmTMB). Gamma: mean exp(lin), shape k = exp(-2 eta)
# (sigma is the coefficient of variation, as drmTMB). Full densities, constants
# included; checked against ForwardDiff in test_joint_sigma_wave2.jl.
@inline function _gsl_kernel(::Gaussian, y, lin, eta)
    r = y - lin
    ie = exp(-2 * eta)
    q = r * r * ie
    return (eta + 0.5 * q + 0.5 * log(2π), -r * ie, 1 - q, ie, 2 * r * ie, 2 * q)
end
@inline function _gsl_kernel(::NegBinomial2, y, lin, eta)
    k = exp(-2 * eta)
    mu = exp(lin)
    lkm = _gsl_logaddexp(-2 * eta, lin)                 # log(k + mu)
    km = k + mu
    f = -(loggamma(y + k) - loggamma(k) - loggamma(y + 1) + k * (-2 * eta - lkm) + y * (lin - lkm))
    Dk = digamma(y + k) - digamma(k) + (-2 * eta - lkm) + 1 - (y + k) / km
    Dkk = trigamma(y + k) - trigamma(k) + 1 / k - 1 / km - (mu - y) / km^2
    return (f, -k * (y - mu) / km, 2 * k * Dk, mu * k * (y + k) / km^2,
            2 * k * mu * (y - mu) / km^2, -(4 * k * Dk + 4 * k^2 * Dkk))
end
@inline function _gsl_kernel(::Gamma, y, lin, eta)
    k = exp(-2 * eta)
    r = y * exp(-lin)
    ly = log(y)
    f = -(k * (-2 * eta) - k * lin - loggamma(k) + (k - 1) * ly - k * r)
    B = -2 * eta + 1 - lin - digamma(k) + ly - r
    return (f, -k * (r - 1), 2 * k * B, k * r, 2 * k * (r - 1),
            -(4 * k * B + 4 * k - 4 * k^2 * trigamma(k)))
end
@inline _gsl_logaddexp(a, b) = a > b ? a + log1p(exp(b - a)) : b + log1p(exp(a - b))

# Joint negative log density F and (optionally) gradient / dense Hessian in v.
# `T` is the promoted element type of θ-derived quantities and v.
# `rho` (optional): correlation between va[g] and vb[g] on ONE shared grouping
# factor (Ga == Gb, same level indices), i.e. (va[g], vb[g]) ~ N(0, [1 rho; rho 1]).
function _gsl_joint(fam, y, mu0, eta0, ga, c, gb, d, Pb, va, vb, Ga::Int, Gb::Int; deriv::Bool,
                    rho = nothing)
    T = promote_type(eltype(mu0), eltype(eta0), eltype(c), eltype(d), eltype(va), eltype(vb),
                     rho === nothing ? Float64 : typeof(rho))
    n = length(y)
    m = Ga + Gb
    F = zero(T)
    g = deriv ? zeros(T, m) : nothing
    H = deriv ? zeros(T, m, m) : nothing
    @inbounds for i in 1:n
        A = Ga > 0 ? c[i] * va[ga[i]] : zero(T)
        eta = eta0[i] + d[i] * vb[gb[i]]
        f, fa, fe, faa, fae, fee = _gsl_kernel(fam, y[i], mu0[i] + A, eta)
        F += f
        if deriv
            if Ga > 0
                ia = ga[i]
                g[ia] += c[i] * fa
                H[ia, ia] += c[i] * c[i] * faa
            end
            ib = Ga + gb[i]
            g[ib] += d[i] * fe
            H[ib, ib] += d[i] * d[i] * fee
            if Ga > 0
                x = c[i] * d[i] * fae
                H[ga[i], ib] += x
                H[ib, ga[i]] += x
            end
        end
    end
    if rho !== nothing
        sr = 1 / (1 - rho * rho)
        F += 0.5 * sr * (sum(abs2, va) - 2 * rho * dot(va, vb) + sum(abs2, vb))
        if deriv
            @inbounds for k in 1:Ga
                g[k] += sr * (va[k] - rho * vb[k])
                g[Ga + k] += sr * (vb[k] - rho * va[k])
                H[k, k] += sr
                H[Ga + k, Ga + k] += sr
                H[k, Ga + k] -= rho * sr
                H[Ga + k, k] -= rho * sr
            end
        end
        return F, g, H
    end
    F += 0.5 * sum(abs2, va)
    Pv = Pb === nothing ? vb : Pb * vb
    F += 0.5 * sum(vb .* Pv)
    if deriv
        @inbounds for k in 1:Ga
            g[k] += va[k]
            H[k, k] += 1
        end
        @inbounds for j in 1:Gb
            g[Ga + j] += Pv[j]
        end
        if Pb === nothing
            @inbounds for j in 1:Gb
                H[Ga + j, Ga + j] += 1
            end
        else
            @inbounds for j in 1:Gb, k in 1:Gb
                H[Ga + k, Ga + j] += Pb[k, j]
            end
        end
    end
    return F, g, H
end

# Damped Newton for the inner mode (Float64). The joint density is not convex in
# (A, eta), so a non-PD Hessian is regularised and every step is backtracked.
function _gsl_mode(joint, v0::Vector{Float64}; tol::Float64 = 1e-10, maxit::Int = 80)
    v = copy(v0)
    F, g, H = joint(v, true)
    for _ in 1:maxit
        maximum(abs, g) < tol && return v, true
        C = cholesky(Symmetric(H); check = false)
        step = if issuccess(C)
            C \ g
        else
            lam = 1e-3 * (sum(abs, diag(H)) / size(H, 1) + 1.0)
            local s
            while true
                C2 = cholesky(Symmetric(H + lam * I); check = false)
                if issuccess(C2)
                    s = C2 \ g
                    break
                end
                lam *= 10
                lam > 1e12 && return v, false
            end
            s
        end
        # Inside the quadratic basin (PD Hessian, small gradient) take the full
        # Newton step: the objective decrease there is below F's roundoff, so an
        # Armijo test on F would reject good steps and stall near 1e-8.
        if issuccess(C) && maximum(abs, g) < 1e-4
            v = v .- step
            F, g, H = joint(v, true)
            continue
        end
        slope = dot(g, step)
        t = 1.0
        accepted = false
        while t > 1e-12
            vn = v .- t .* step
            Fn, _, _ = joint(vn, false)
            if isfinite(Fn) && Fn <= F - 1e-4 * t * slope
                v = vn
                accepted = true
                break
            end
            t *= 0.5
        end
        accepted || return v, maximum(abs, g) < 1e-8
        F, g, H = joint(v, true)
    end
    return v, maximum(abs, g) < 1e-9
end

# `mean_re` is `nothing` or (gidx, G, Zg, nmsd, grp, block); `sigma_re` is
# (gidx, G, Zg, nmsd, grp, loading, Kmat-or-nothing, block, kind).
function _fit_gaussian_sigma_sd_laplace(fam::Union{Gaussian,NegBinomial2,Gamma}, y, Xμ, Xσ, mean_re, sigma_re,
                                        nmμ, nmσ, g_tol; se::Bool = true, corr::Bool = false, sdreg::Bool = false)
    n = length(y)
    pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    has_a = mean_re !== nothing
    ga, Ga, Za, nma, grpa, blocka = has_a ? mean_re : (Int[], 0, zeros(0, 0), String[], :none, :none)
    gb, Gb, Zb, nmb, grpb, loading, Kmat, blockb, _ = sigma_re
    pa = has_a ? size(Za, 2) : 0
    pb = size(Zb, 2)
    m = Ga + Gb
    m <= _GSL_MAX_LATENT || throw(ArgumentError(
        "drm: Gaussian sigma random effect with an SD formula uses a dense Laplace " *
        "engine limited to $(_GSL_MAX_LATENT) latent coordinates (got $m)."))
    Pb = nothing
    logdetK = 0.0
    if Kmat !== nothing
        Ck = cholesky(Symmetric(Matrix{Float64}(Kmat)); check = false)
        issuccess(Ck) || throw(ArgumentError(
            "drm: the structured sigma matrix must be symmetric positive definite."))
        Pb = Matrix(inv(Ck))
        Pb = (Pb .+ Pb') ./ 2
        logdetK = logdet(Ck)
    end
    ra = pμ + pσ + 1
    ia = ra:(ra + pa - 1)
    ib = (ra + pa):(ra + pa + pb - 1)
    ntheta = pμ + pσ + pa + pb + (corr ? 1 : 0)
    ic = ntheta                       # correlation parameter (eta; rho = tanh(eta)) when `corr`
    corr && !(has_a && Ga == Gb && ga == gb && Kmat === nothing) && throw(ArgumentError(
        "drm: a mean-sigma correlation needs both random intercepts on one grouping factor."))

    function pieces(θ)
        βμ = θ[1:pμ]; βσ = θ[(pμ+1):(pμ+pσ)]
        mu0 = Xμ * βμ
        eta0 = Xσ * βσ
        c = has_a ? exp.(Za * θ[ia])[ga] : eltype(θ)[]
        d = loading .* exp.(Zb * θ[ib])[gb]
        return mu0, eta0, c, d
    end
    joint_at(θ) = begin
        mu0, eta0, c, d = pieces(θ)
        (vv, deriv) -> begin
            vaa = @view vv[1:Ga]; vbb = @view vv[(Ga+1):m]
            _gsl_joint(fam, y, mu0, eta0, ga, c, gb, d, Pb, vaa, vbb, Ga, Gb; deriv = deriv,
                       rho = corr ? tanh(θ[ic]) : nothing)
        end
    end

    # Laplace value; for a Dual θ, v̂ is advanced one Newton step (see header).
    function laplace(θ::AbstractVector{T}, vhat::Vector{Float64}) where {T}
        joint = joint_at(θ)
        v = convert(Vector{T}, vhat)
        if T <: ForwardDiff.Dual
            _, g, H = joint(vhat, true)
            # A non-PD Hessian here used to raise PosDefException out of `drm()`; return the same
            # non-PD penalty as the line below (the fit is reported with converged = false).
            Cn = cholesky(Symmetric(H); check = false)
            issuccess(Cn) || return T(REML_NONPD_PENALTY)
            v = vhat .- (Cn \ g)
        end
        F, _, H = joint(v, true)
        C = cholesky(Symmetric(H); check = false)
        issuccess(C) || return T(REML_NONPD_PENALTY)
        ldR = corr ? 0.5 * Ga * log(1 - tanh(θ[ic])^2) : zero(T)
        return F + 0.5 * logdet(C) + 0.5 * logdetK + ldR
    end

    fg = function (θ, b0, grad)
        θf = Vector{Float64}(θ)
        joint = joint_at(θf)
        v, ok = _gsl_mode(joint, Vector{Float64}(b0))
        ok || ((v, ok) = _gsl_mode(joint, zeros(m)))
        ok || return grad ? (1e18, fill(NaN, ntheta), zeros(m), false) : (1e18, zeros(m), false)
        _, _, H = joint(v, true)
        issuccess(cholesky(Symmetric(H); check = false)) ||
            return grad ? (1e18, fill(NaN, ntheta), v, false) : (1e18, v, false)
        val = laplace(θf, v)
        grad || return (val, v, true)
        gθ = ForwardDiff.gradient(t -> laplace(t, v), θf)
        return (val, gθ, v, all(isfinite, gθ))
    end

    θ0 = zeros(ntheta)
    if fam isa Gaussian
        β0 = Xμ \ y
        sd0 = std(y - Xμ * β0) + eps()
        θ0[1:pμ] .= β0
        θ0[pμ+1] = log(0.7 * sd0)
        has_a && (θ0[ia] .= Za \ fill(log(0.5 * sd0), Ga))
    else
        # log-link families: least squares on log(y + 0.5); sigma and the mean SD
        # start at 0.5 on their log scales.
        θ0[1:pμ] .= Xμ \ log.(y .+ 0.5)
        θ0[pμ+1] = log(0.5)
        has_a && (θ0[ia] .= Za \ fill(log(0.5), Ga))
    end
    θ0[ib] .= Zb \ fill(log(0.3), Gb)

    what = "$(nameof(typeof(fam))) sigma random effect with an SD formula"
    θ̂, nllhat, conv, V, nll, grad! = _ordinary_laplace_optimize(
        fg, θ0, n, m, g_tol; se = false, context = what)
    # Multimodality check. The joint density is not log-concave in (mean effect,
    # log-sigma effect), so the inner problem can have several positive-definite
    # modes; the optimiser's warm start then decides which Laplace branch is
    # reported and the outer objective is discontinuous (diagnosed on a
    # 200 x 200 crossed `sigma ~ 1 + (1 | site)` design: two PD modes 15.7 units
    # apart at one θ; native drmTMB/TMB is exposed to the same path dependence).
    # If the prior-mode start v = 0 reaches a different Laplace value at θ̂, the
    # likelihood is not well defined there: warn and do not claim convergence.
    let joint = joint_at(θ̂)
        v0, ok0 = _gsl_mode(joint, zeros(m))
        if ok0
            _, _, H0 = joint(v0, true)
            if issuccess(cholesky(Symmetric(H0); check = false))
                nll0 = laplace(θ̂, v0)
                if isfinite(nllhat) && abs(nll0 - nllhat) > 1e-6 * max(1.0, abs(nllhat))
                    @warn "drm: $what: the random-effect mode is not unique at the estimate " *
                          "(Laplace log-likelihood $(round(-nllhat; digits = 4)) from the optimiser's " *
                          "path, $(round(-nll0; digits = 4)) from a zero start). The Laplace " *
                          "likelihood is path dependent here; the fit is marked not converged. " *
                          "Typical cause: few observations per sigma group with crossed mean effects."
                    conv = false
                end
            end
        end
    end
    # Wald covariance from central differences of the analytic gradient (the
    # value-difference Hessian of the shared helper is noisier than the inner-mode
    # tolerance allows at this size).
    if se
        Hθ = _finite_hessian_from_gradient(grad!, θ̂; context = what)
        V = all(isfinite, Hθ) ? Matrix(_vcov_from_hessian(Hθ)) : fill(NaN, ntheta, ntheta)
    end

    # Conditional modes and fitted scales at θ̂.
    vhat, _ = _gsl_mode(joint_at(θ̂), zeros(m))
    βμ = θ̂[1:pμ]; βσ = θ̂[(pμ+1):(pμ+pσ)]
    sd_a = has_a ? exp.(Za * θ̂[ia]) : Float64[]
    sd_b = exp.(Zb * θ̂[ib])
    ab = has_a ? sd_a .* vhat[1:Ga] : Float64[]                 # raw mean effects
    vb = vhat[(Ga+1):m]
    bb = sd_b .* vb                                              # raw sigma effects (log scale)
    eta_fixed = Xσ * βσ
    mean_fixed = Xμ * βμ
    lin = has_a ? mean_fixed .+ ab[ga] : mean_fixed
    means = Dict(:mu => fam isa Gaussian ? lin : exp.(lin))
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict{Symbol,Vector{Float64}}(
        :sigma => exp.(eta_fixed .+ loading .* bb[gb]),
        :sigma_fixed => exp.(eta_fixed),
        :sigma_random_effect => bb,
        :sigma_group_index => Float64.(gb),
        :sd_group => has_a ? sd_a : sd_b,
        :sd => has_a ? sd_a[ga] : sd_b[gb])
    has_a && (scales[:sd_group_index] = Float64.(ga))
    # `means[:mu]` already holds the CONDITIONAL mean on this joint route (mean mode `ab` included);
    # record the per-row mode too so `fitted(fit; re = :zero)` can take it back out.
    has_a && (_record_cond_fit!(scales, ab[ga]); scales[:cond_fit_stored] = [1.0])
    blocks = Pair{Symbol,UnitRange{Int}}[:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ)]
    names = Pair{Symbol,Vector{String}}[:mu => nmμ, :sigma => nmσ]
    if corr && sdreg
        # Correlated pair with a mean SD formula: typed SD blocks (as the independent route),
        # plus `:cor` holding eta with rho = tanh(eta).
        push!(blocks, blocka => ia); push!(names, blocka => nma)
        push!(blocks, blockb => ib); push!(names, blockb => nmb)
        push!(blocks, :cor => ic:ic); push!(names, :cor => ["cor(mu, sigma | $(grpa))"])
    elseif corr
        # Same names as the independent same-group route (`:resd`), plus `:cor`
        # holding eta with rho = tanh(eta) (drmTMB `eta_cor_mu_sigma`).
        push!(blocks, :resd => first(ia):last(ib)); push!(names, :resd => [String(grpa), "$(grpb)_logsigma"])
        push!(blocks, :cor => ic:ic); push!(names, :cor => ["cor(mu, sigma | $(grpa))"])
    else
        has_a && (push!(blocks, blocka => ia); push!(names, blocka => nma))
        push!(blocks, blockb => ib); push!(names, blockb => nmb)
    end
    ranef = Dict{Symbol,Any}()
    has_a && (ranef[Symbol(grpa)] = ab)
    ranef[Symbol(grpb, "_logsigma")] = bb
    fit = DrmFit(fam, blocks, names, θ̂, V, -nllhat, n, conv, means, obs, scales)
    return _withranef(_withmarginal(_withnll(fit, nll, grad!), :Laplace), ranef)
end

# ---------------------------------------------------------------------------------------------
# REML for the structured sigma intercept (fixed-effect mean). Native drmTMB (`REML = TRUE`)
# does not restrict the likelihood by a closed form here: it moves beta_mu AND beta_sigma into
# TMB's random block, so the reported objective is ONE Laplace approximation over
# w = (v, beta_mu, beta_sigma) with flat priors on both coefficient blocks,
#
#   nll_R(alpha) = F(w^) + 1/2 log det H(w^) + 1/2 log det K - (p/2) log 2pi,   p = p_mu + p_sigma,
#
# where F is the ML joint negative log density in v (as above) as a function of w, H = d2F/dw2
# over ALL of w, and w^ the joint inner mode. The -(p/2) log 2pi is the Laplace constant of the
# p flat coordinates (TMB's -(m_random/2) log 2pi; the v-block's cancels against the prior as in
# ML). Only the SD-formula coefficients alpha stay outer parameters. Because the sigma integral is
# a Laplace one (the density is not Gaussian in beta_sigma) this is an APPROXIMATE restricted
# likelihood, the same one native reports (`estimator_exact = FALSE`); the beta_mu integral alone
# would be exact at fixed sigma. The reported coefficients are the joint-mode beta^(alpha^); their
# covariance is TMB's sdreport form  H^-1 (beta block) + J V_alpha J' with J = d w^ / d alpha
# (implicit-function Jacobian, exact at the mode via one Newton step under ForwardDiff).
# ---------------------------------------------------------------------------------------------
function _gsl_joint_reml(fam, y, Xμ, Xσ, gb, d, Pb, w, Gb::Int; deriv::Bool)
    n = length(y); pμ = size(Xμ, 2); pσ = size(Xσ, 2)
    T = promote_type(eltype(d), eltype(w))
    v = w[1:Gb]
    lin = Xμ * w[(Gb+1):(Gb+pμ)]
    eta = Xσ * w[(Gb+pμ+1):(Gb+pμ+pσ)] .+ d .* v[gb]
    ker = [_gsl_kernel(fam, y[i], lin[i], eta[i]) for i in 1:n]
    F = sum(k -> k[1], ker)
    Pv = Pb === nothing ? v : Pb * v
    F += 0.5 * dot(v, Pv)
    deriv || return F, nothing, nothing
    fa = T[k[2] for k in ker]; fe = T[k[3] for k in ker]
    faa = T[k[4] for k in ker]; fae = T[k[5] for k in ker]; fee = T[k[6] for k in ker]
    m = Gb + pμ + pσ
    rμ = (Gb+1):(Gb+pμ); rσ = (Gb+pμ+1):m
    g = zeros(T, m); H = zeros(T, m, m)
    @inbounds for i in 1:n
        ib = gb[i]
        g[ib] += d[i] * fe[i]
        H[ib, ib] += d[i]^2 * fee[i]
    end
    g[1:Gb] .+= Pv
    if Pb === nothing
        @inbounds for j in 1:Gb; H[j, j] += 1; end
    else
        H[1:Gb, 1:Gb] .+= Pb
    end
    g[rμ] .= Xμ' * fa
    g[rσ] .= Xσ' * fe
    # beta-beta blocks
    H[rμ, rμ] .= Xμ' * (faa .* Xμ)
    H[rσ, rσ] .= Xσ' * (fee .* Xσ)
    Hμσ = Xμ' * (fae .* Xσ)
    H[rμ, rσ] .= Hμσ
    H[rσ, rμ] .= Hμσ'
    # v-beta blocks: d/dv_j of the beta gradients
    @inbounds for i in 1:n
        ib = gb[i]
        for a in 1:pμ
            x = d[i] * fae[i] * Xμ[i, a]
            H[ib, Gb + a] += x; H[Gb + a, ib] += x
        end
        for a in 1:pσ
            x = d[i] * fee[i] * Xσ[i, a]
            H[ib, Gb + pμ + a] += x; H[Gb + pμ + a, ib] += x
        end
    end
    return F, g, H
end

function _fit_gaussian_sigma_sd_laplace_reml(fam::Gaussian, y, Xμ, Xσ, sigma_re, nmμ, nmσ, g_tol;
                                             se::Bool = true)
    n = length(y)
    pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    p = pμ + pσ
    gb, Gb, Zb, nmb, grpb, loading, Kmat, blockb, _ = sigma_re
    pb = size(Zb, 2)
    m = Gb
    q = m + p
    q <= _GSL_MAX_LATENT || throw(ArgumentError(
        "drm: Gaussian sigma random effect with an SD formula uses a dense Laplace " *
        "engine limited to $(_GSL_MAX_LATENT) latent coordinates (got $q under REML)."))
    Kmat === nothing && throw(ArgumentError("drm: internal: structured sigma REML needs K."))
    Ck = cholesky(Symmetric(Matrix{Float64}(Kmat)); check = false)
    issuccess(Ck) || throw(ArgumentError("drm: the structured sigma matrix must be symmetric positive definite."))
    Pb = Matrix(inv(Ck)); Pb = (Pb .+ Pb') ./ 2
    logdetK = logdet(Ck)
    rμ = (m+1):(m+pμ); rσ = (m+pμ+1):q

    joint_at(α) = begin
        d = loading .* exp.(Zb * α)[gb]
        (w, deriv) -> _gsl_joint_reml(fam, y, Xμ, Xσ, gb, d, Pb, w, Gb; deriv = deriv)
    end
    # Warm start of the inner mode: v = 0, beta_mu = OLS, beta_sigma = (log 0.7 sd, 0, ...).
    w0 = zeros(q)
    β0 = Xμ \ y
    sd0 = std(y - Xμ * β0) + eps()
    w0[rμ] .= β0
    w0[m + pμ + 1] = log(0.7 * sd0)

    function laplace(α::AbstractVector{T}, what::Vector{Float64}) where {T}
        joint = joint_at(α)
        w = convert(Vector{T}, what)
        if T <: ForwardDiff.Dual
            _, g, H = joint(what, true)
            Cn = cholesky(Symmetric(H); check = false)
            issuccess(Cn) || return T(REML_NONPD_PENALTY)
            w = what .- (Cn \ g)
        end
        F, _, H = joint(w, true)
        C = cholesky(Symmetric(H); check = false)
        issuccess(C) || return T(REML_NONPD_PENALTY)
        return F + 0.5 * logdet(C) + 0.5 * logdetK - 0.5 * p * log(2π)
    end

    fg = function (α, b0, grad)
        αf = Vector{Float64}(α)
        joint = joint_at(αf)
        start = iszero(b0) ? w0 : Vector{Float64}(b0)
        w, ok = _gsl_mode(joint, start)
        ok || ((w, ok) = _gsl_mode(joint, w0))
        ok || return grad ? (1e18, fill(NaN, pb), zeros(q), false) : (1e18, zeros(q), false)
        _, _, H = joint(w, true)
        issuccess(cholesky(Symmetric(H); check = false)) ||
            return grad ? (1e18, fill(NaN, pb), w, false) : (1e18, w, false)
        val = laplace(αf, w)
        grad || return (val, w, true)
        gα = ForwardDiff.gradient(t -> laplace(t, w), αf)
        return (val, gα, w, all(isfinite, gα))
    end

    α0 = Zb \ fill(log(0.3), Gb)
    what = "Gaussian structured sigma random effect with an SD formula (REML)"
    α̂, nllhat, conv, _, nll, grad! = _ordinary_laplace_optimize(
        fg, α0, n, q, g_tol; se = false, context = what)
    joint̂ = joint_at(α̂)
    ŵ, okw = _gsl_mode(joint̂, w0)
    okw || (conv = false)
    _, _, Ĥ = joint̂(ŵ, true)
    Cĥ = cholesky(Symmetric(Ĥ); check = false)
    issuccess(Cĥ) || (conv = false)
    # Multimodality guard, as in ML: a second positive-definite mode from the prior start.
    let (w2, ok2) = _gsl_mode(joint̂, vcat(zeros(m), ŵ[(m+1):q]))
        if ok2 && issuccess(cholesky(Symmetric(joint̂(w2, true)[3]); check = false))
            nll2 = laplace(α̂, w2)
            if isfinite(nllhat) && abs(nll2 - nllhat) > 1e-6 * max(1.0, abs(nllhat))
                @warn "drm: $what: the joint mode is not unique at the estimate; the fit is marked not converged."
                conv = false
            end
        end
    end
    βμ = ŵ[rμ]; βσ = ŵ[rσ]
    θ̂ = vcat(βμ, βσ, α̂)
    ntheta = p + pb
    V = fill(NaN, ntheta, ntheta)
    if se && issuccess(Cĥ)
        Hα = _finite_hessian_from_gradient(grad!, α̂; context = what)
        if all(isfinite, Hα)
            Vα = Matrix(_vcov_from_hessian(Hα))
            # J = d w^ / d alpha: one Newton step from the converged mode, differentiated.
            J = ForwardDiff.jacobian(α̂) do t
                dd = loading .* exp.(Zb * t)[gb]
                _, g, H = _gsl_joint_reml(fam, y, Xμ, Xσ, gb, dd, Pb, ŵ, Gb; deriv = true)
                ŵ .- (cholesky(Symmetric(H)) \ g)
            end
            Hinv = Matrix(inv(Cĥ))
            Jβ = J[(m+1):q, :]
            Cββ = Hinv[(m+1):q, (m+1):q] + Jβ * Vα * Jβ'
            Cβα = Jβ * Vα
            V = [Cββ Cβα; Cβα' Vα]
            V = (V .+ V') ./ 2
        end
    end
    # ML Laplace value at the REML estimate (reference, comparable across models only under ML).
    ml_nll = let dd = loading .* exp.(Zb * α̂)[gb]
        vml, okm = _gsl_mode((v, deriv) -> _gsl_joint(fam, y, Xμ * βμ, Xσ * βσ, Int[], Float64[], gb, dd, Pb,
                                                       Float64[], v, 0, Gb; deriv = deriv), zeros(m))
        if okm
            Fm, _, Hm = _gsl_joint(fam, y, Xμ * βμ, Xσ * βσ, Int[], Float64[], gb, dd, Pb,
                                   Float64[], vml, 0, Gb; deriv = true)
            Cm = cholesky(Symmetric(Hm); check = false)
            issuccess(Cm) ? Fm + 0.5 * logdet(Cm) + 0.5 * logdetK : NaN
        else
            NaN
        end
    end
    # Conditional modes and fitted scales.
    sd_b = exp.(Zb * α̂)
    vb = ŵ[1:m]
    bb = sd_b .* vb
    eta_fixed = Xσ * βσ
    mean_fixed = Xμ * βμ
    means = Dict(:mu => mean_fixed)
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict{Symbol,Vector{Float64}}(
        :sigma => exp.(eta_fixed .+ loading .* bb[gb]),
        :sigma_fixed => exp.(eta_fixed),
        :sigma_random_effect => bb,
        :sigma_group_index => Float64.(gb),
        :sd_group => sd_b,
        :sd => sd_b[gb])
    blocks = Pair{Symbol,UnitRange{Int}}[:mu => 1:pμ, :sigma => (pμ+1):p, blockb => (p+1):(p+pb)]
    names = Pair{Symbol,Vector{String}}[:mu => nmμ, :sigma => nmσ, blockb => nmb]
    ranef = Dict{Symbol,Any}(Symbol(grpb, "_logsigma") => bb)
    fit = DrmFit(fam, blocks, names, θ̂, V, -nllhat, n, conv, means, obs, scales)
    fit = _withmarginal(_withranef(fit, ranef), :Laplace)
    return _withreml(fit, -nllhat, -ml_nll)
end

# Structured sigma-intercept spec shared by the Gaussian, NB2 and Gamma routers:
#   sigma ~ 1 + animal|relmat|phylo|spatial(1 | g),  sd(g, level, sigma) ~ z; mean fixed.
# K enters raw for animal/relmat; phylo is the tree's correlation matrix
# (ape::vcv(tree, corr = TRUE)); spatial is the native FIXED kernel
# exp(-d / median positive distance) + 1e-6 I (`drm_spatial_coords_precision`).
function _gsl_structured_sigma_spec(f::DrmFormula, what, sd_, sig_t, structured_sigma, data, n::Int,
                                    K, A, coords, tree, block_of)
    kind, grp = structured_sigma
    kind in (:animal, :relmat, :phylo, :spatial) || throw(ArgumentError(
        "drm: $what with `$kind(1 | $grp)` on sigma is not supported."))
    level_of = Dict(:animal => :animal, :relmat => :relmat, :phylo => :phylogenetic,
                    :spatial => :spatial)
    sd_.level === level_of[kind] || throw(ArgumentError(
        "drm: `sd($grp, $(sd_.level), dpar = sigma)` does not match the sigma-side `$kind(1 | $grp)`."))
    sd_.group === grp || throw(ArgumentError(
        "drm: `sd($(sd_.group), …, dpar = sigma)` must name the sigma-side grouping `$grp`."))
    sd_.member in (nothing, :intercept) || throw(ArgumentError(
        "drm: structured sigma SD targets the intercept; `coef($(sd_.member))` is not supported."))
    local gidx, G, Cmat
    if kind === :phylo
        (K === nothing && A === nothing && coords === nothing) || throw(ArgumentError(
            "drm: phylo sigma SD regression uses `tree = …` only."))
        tree === nothing && throw(ArgumentError("phylo(1 | $grp) on sigma needs `tree = …`"))
        phy = tree isa AbstractString ? augmented_phy(tree) : tree
        gidx = _phylo_mean_leaf_index(phy, getproperty(data, grp))   # rows -> leaves by name
        G = phy.n_leaves
        Cmat = _phylo_correlation(phy)
    elseif kind === :spatial
        (K === nothing && A === nothing && tree === nothing) || throw(ArgumentError(
            "drm: spatial sigma SD regression uses `coords = …` only."))
        coords === nothing && throw(ArgumentError("spatial(1 | $grp) on sigma needs `coords = …`"))
        gidx, G = _group_index(getproperty(data, grp); col = grp)
        Cmat = _gaussian_spatial_sd_kernel(coords, G, grp)
    else
        tree === nothing || throw(ArgumentError("drm: $kind sigma SD regression does not use `tree`."))
        gidx, G = _group_index(getproperty(data, grp); col = grp)
        Cmat = if kind === :relmat
            (A === nothing && coords === nothing) || throw(ArgumentError("drm: relmat sigma SD regression uses `K = …` only."))
            K === nothing && throw(ArgumentError("relmat(1 | $grp) on sigma needs the covariance matrix `K = …`"))
            Matrix{Float64}(K)
        else
            (K === nothing && coords === nothing) || throw(ArgumentError("drm: animal sigma SD regression uses `A = …` only."))
            A === nothing && throw(ArgumentError("animal(1 | $grp) on sigma needs the relatedness matrix `A = …`"))
            Matrix{Float64}(A)
        end
    end
    size(Cmat) == (G, G) || throw(ArgumentError(
        "structured matrix must be $(G)×$(G) (the number of `$grp` levels)"))
    all(isfinite, Cmat) || throw(ArgumentError("structured matrix must be finite"))
    Zb, nmb = _sd_group_design(f.response, sig_t.rhs, data, gidx, G, grp;
                               schema_cache = f.schema_cache, schema_key = sig_t.key)
    size(Zb, 2) <= G && rank(Zb) == size(Zb, 2) || throw(ArgumentError(
        "drm: $what group-level design is rank deficient."))
    return (gidx, G, Zb, nmb, grp, ones(n), Cmat, block_of(sig_t), :structured)
end

# Router leg, called from the Gaussian sd() dispatch when any SD target has
# `dpar = "sigma"`. Every shape outside the two supported ones is refused by name.
function _drm_gaussian_sigma_sd_laplace(f::DrmFormula, fam::Gaussian, sdp, sdpp, re, structured,
                                        structured_sigma, sigma_re, metav, has_missing_response,
                                        y, Xμ, Xσ, nmμ, nmσ, data, K, A, coords, g_tol, method,
                                        algorithm; tree = nothing)
    what = "Gaussian `sd(g, sigma)`"
    isempty(sdpp) || throw(ArgumentError("drm: $what cannot be combined with `sd(g, phylogenetic)`."))
    metav === nothing || throw(ArgumentError("drm: $what cannot be combined with `meta_V(...)`."))
    # REML is admitted only for the structured sigma intercept below (fixed-effect mean, one
    # animal/relmat/phylo/spatial sigma field); every other shape here stays ML-only.
    reml = method === :REML
    reml && structured_sigma === nothing && throw(ArgumentError(
        "drm: $what with an ordinary random effect supports maximum likelihood only " *
        "(REML is available for a structured sigma intercept: animal/relmat/phylo/spatial)."))
    algorithm === :auto || throw(ArgumentError("drm: `algorithm = :$algorithm` is not implemented for $what."))
    has_missing_response && throw(ArgumentError(
        "drm: $what requires observed responses; remove incomplete rows first (`drm_listwise`)."))
    structured === nothing || throw(ArgumentError(
        "drm: $what cannot be combined with a structured mean effect (phylo/relmat/animal/spatial)."))
    targets = _sd_target_parts(f)
    allowed = Set((:mu, :sigma))
    for item in targets
        push!(allowed, item.key)
    end
    all(p -> first(p) in allowed, f.forms) || throw(ArgumentError(
        "drm: $what supports only `mu`, `sigma` and `sd(...)` formulas."))
    sig_targets = filter(item -> item.target.dpar === :sigma, targets)
    mu_targets = filter(item -> item.target.dpar === :mu, targets)
    length(sig_targets) == 1 || throw(ArgumentError(
        "drm: $what takes exactly one sigma-side SD formula (got $(length(sig_targets)))."))
    sig_t = only(sig_targets)
    sd_ = sig_t.target
    sd_.block === nothing && sd_.level in (:iid, :animal, :relmat, :phylogenetic, :spatial) ||
        throw(ArgumentError(
        "drm: $what is implemented for an ordinary sigma random effect and for animal, relmat, " *
        "phylo and spatial structured sigma intercepts; `$(sd_.level)` is not."))
    nblocks = length(targets)
    block_of(item) = _sd_target_block(item.target; disambiguate = nblocks > 1)

    n = length(y)
    structured_route = structured_sigma !== nothing
    local sigma_spec
    if structured_route
        # 6b: sigma ~ 1 + animal|relmat|phylo|spatial(1 | g), sd(g, level, sigma) ~ z;
        # mean fixed. K enters raw for animal/relmat; phylo is the tree's correlation
        # matrix (ape::vcv(tree, corr = TRUE)); spatial is the native FIXED kernel
        # exp(-d / median positive distance) + 1e-6 I (`drm_spatial_coords_precision`).
        isempty(sigma_re) && isempty(re) && isempty(mu_targets) || throw(ArgumentError(
            "drm: structured sigma SD regression takes a fixed-effect mean and one structured " *
            "sigma intercept only (no ordinary mean or sigma random effects)."))
        sigma_spec = _gsl_structured_sigma_spec(f, what, sd_, sig_t, structured_sigma, data, n,
                                                K, A, coords, tree, block_of)
        mean_spec = nothing
    else
        # 6a: ordinary sigma random intercept/slope (+ optional mean intercept), each with SD.
        (K === nothing && A === nothing && coords === nothing) || throw(ArgumentError(
            "drm: $what with ordinary random effects does not use `K`, `A` or `coords`."))
        mean_spec, sigma_spec = _gsl_ordinary_specs(f, what, sd_, sig_t, mu_targets, sigma_re, re,
                                                    data, n, block_of)
    end
    reml && return _fit_gaussian_sigma_sd_laplace_reml(fam, y, Xμ, Xσ, sigma_spec, nmμ, nmσ, g_tol)
    return _fit_gaussian_sigma_sd_laplace(fam, y, Xμ, Xσ, mean_spec, sigma_spec, nmμ, nmσ, g_tol)
end

# Ordinary (6a) specs shared by the Gaussian, NB2 and Gamma routers: one sigma
# random intercept (or, when `allow_slope`, one independent sigma slope) with its
# SD formula, plus an optional mean random intercept with its own SD formula.
function _gsl_ordinary_specs(f::DrmFormula, what, sd_, sig_t, mu_targets, sigma_re, re,
                             data, n::Int, block_of; allow_slope::Bool = true)
    length(sigma_re) == 1 || throw(ArgumentError(
        "drm: $what needs exactly one ordinary `sigma` random-effect term (got $(length(sigma_re)))."))
    skind, sslope = _re_kind(sigma_re[1][1])
    skind in (:intercept, :slope) || throw(ArgumentError(
        "drm: $what supports one sigma intercept `(1 | g)` or one independent slope `(0 + x | g)`."))
    allow_slope || skind === :intercept || throw(ArgumentError(
        "drm: $what supports a sigma random INTERCEPT `(1 | g)` only (native drmTMB refuses a sigma slope here)."))
    sgrp = sigma_re[1][2]
    sd_.group === sgrp || throw(ArgumentError(
        "drm: `sd($(sd_.group), dpar = sigma)` must name the sigma random-effect group `$sgrp`."))
    smember = skind === :intercept ? :intercept : sslope
    (skind === :intercept ? sd_.member in (nothing, :intercept) : sd_.member === sslope) ||
        throw(ArgumentError(skind === :intercept ?
            "drm: the sigma intercept SD target must not carry a `coef(...)` selector." :
            "drm: a sigma random slope needs an explicit `coef($sslope)` SD target."))
    gb, Gb = _group_index(getproperty(data, sgrp); col = sgrp)
    loading = if skind === :intercept
        ones(n)
    else
        hasproperty(data, sslope) || throw(ArgumentError(
            "drm: sigma random-slope predictor `$sslope` is absent from data."))
        raw = getproperty(data, sslope)
        raw isa AbstractVector{<:Real} && length(raw) == n && all(isfinite, raw) || throw(ArgumentError(
            "drm: sigma random-slope predictor `$sslope` must be finite numeric values, one per response."))
        any(!iszero, raw) || throw(ArgumentError(
            "drm: sigma random-slope predictor `$sslope` must contain at least one nonzero value."))
        Float64.(raw)
    end
    Zb, nmb = _sd_group_design(f.response, sig_t.rhs, data, gb, Gb, sgrp;
                               schema_cache = f.schema_cache, schema_key = sig_t.key)
    size(Zb, 2) <= Gb && rank(Zb) == size(Zb, 2) || throw(ArgumentError(
        "drm: $what group-level design is rank deficient."))
    sigma_spec = (gb, Gb, Zb, nmb, sgrp, loading, nothing, block_of(sig_t), skind)
    mean_spec = nothing
    if !isempty(re)
        length(re) == 1 || throw(ArgumentError(
            "drm: $what allows at most one ordinary mean random-effect term (got $(length(re)))."))
        mkind, _ = _re_kind(re[1][1])
        mkind === :intercept || throw(ArgumentError(
            "drm: $what supports a mean random INTERCEPT `(1 | g)` only (no mean random slopes)."))
        length(mu_targets) == 1 || throw(ArgumentError(
            "drm: $what with a mean random intercept needs its own `sd(g) ~ …` formula; " *
            "a mean random effect without an SD formula is not supported here."))
        mu_t = only(mu_targets)
        md = mu_t.target
        mgrp = re[1][2]
        (md.level === :iid && md.group === mgrp && md.block === nothing &&
         md.member in (nothing, :intercept)) || throw(ArgumentError(
            "drm: the mean-side `sd(...)` must target the mean random intercept `(1 | $mgrp)`."))
        ga, Ga = _group_index(getproperty(data, mgrp); col = mgrp)
        Za, nma = _sd_group_design(f.response, mu_t.rhs, data, ga, Ga, mgrp;
                                   schema_cache = f.schema_cache, schema_key = mu_t.key)
        size(Za, 2) <= Ga && rank(Za) == size(Za, 2) || throw(ArgumentError(
            "drm: $what mean group-level design is rank deficient."))
        mean_spec = (ga, Ga, Za, nma, mgrp, block_of(mu_t))
    else
        isempty(mu_targets) || throw(ArgumentError(
            "drm: a mean-side `sd(g) ~ …` needs a mean random intercept `(1 | g)`."))
    end
    return mean_spec, sigma_spec
end

# NB2 / Gamma router leg (twins of drmTMB `test-nbinom2-mu-sigma-joint-sd.R`,
# `test-nbinom2-mu-sigma-same-group-sd.R`, `test-gamma-mu-sigma-joint-sd.R`):
#   bf(y ~ x + (1 | id), sigma ~ 1 + (1 | g), sd(id) ~ w, sd(g, sigma) ~ z)
# with `g` a different (crossed) or the same grouping factor as `id`. Mean and
# sigma latent fields are independent (no cross-correlation parameter), as in
# native drmTMB. Same dense Laplace engine as the Gaussian joint route; only the
# per-observation kernel differs. Shapes native drmTMB refuses (no SD formula on
# either side, mean or sigma random slopes, zi, structured terms) are refused.
function _drm_joint_sigma_sd_laplace(f::DrmFormula, fam::Union{NegBinomial2,Gamma};
                                     data, tree = nothing, K = nothing, A = nothing,
                                     coords = nothing, g_tol::Real = 1e-8, se::Bool = true)
    famname = fam isa Gamma ? "Gamma" : "NegBinomial2"
    what = "$famname joint `mu` + `sigma` SD regression"
    (tree === nothing && K === nothing && A === nothing && coords === nothing) || throw(ArgumentError(
        "drm: $what uses ordinary random intercepts only; `tree`, `K`, `A` and `coords` are not used."))
    isempty(_sdphylo_parts(f)) || throw(ArgumentError(
        "drm: $what cannot be combined with `sd(g, phylogenetic)`."))
    targets = _sd_target_parts(f)
    sig_targets = filter(item -> item.target.dpar === :sigma, targets)
    mu_targets = filter(item -> item.target.dpar === :mu, targets)
    (length(sig_targets) == 1 && length(mu_targets) == 1 && length(targets) == 2) || throw(ArgumentError(
        "drm: $what takes one `sd(id) ~ …` and one `sd(g, sigma) ~ …` formula."))
    allowed = Set{Symbol}((:mu, :sigma))
    foreach(item -> push!(allowed, item.key), targets)
    all(p -> first(p) in allowed, f.forms) || throw(ArgumentError(
        "drm: $what supports only `mu`, `sigma` and the two `sd(...)` formulas (no `zi`/`hu`)."))
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])
    (mv === nothing && st === nothing) || throw(ArgumentError(
        "drm: $what takes an ordinary mean random intercept (no `meta_V` or structured mean terms)."))
    isempty(re) && throw(ArgumentError(
        "drm: $what needs a mean random intercept `(1 | id)` for `sd(id) ~ …`."))
    sigma_rhs = get(rhs, :sigma, nothing)
    sigma_rhs === nothing && throw(ArgumentError(
        "drm: $what needs `sigma ~ … + (1 | g)`."))
    fixed_sigma, sigma_re, mvs, sts = _split_ranef(sigma_rhs)
    (mvs === nothing && sts === nothing) || throw(ArgumentError(
        "drm: $what takes an ordinary sigma random intercept (no structured sigma terms)."))
    sig_t = only(sig_targets)
    sd_ = sig_t.target
    (sd_.level === :iid && sd_.block === nothing) || throw(ArgumentError(
        "drm: $what targets an ordinary sigma random intercept; `$(sd_.level)` is not supported."))
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "drm: $what requires observed responses; remove incomplete rows first (`drm_listwise`)."))
    y, Xμ, nmμ = _design(f.response, fixed_mu, data)
    _, Xσ, nmσ = _design(f.response, fixed_sigma, data)
    if fam isa Gamma
        all(yi -> isfinite(yi) && yi > 0, y) || throw(ArgumentError(
            "Gamma() requires finite, strictly positive responses"))
    else
        all(yi -> isfinite(yi) && yi >= 0 && isinteger(yi), y) || throw(ArgumentError(
            "NegBinomial2() requires finite, non-negative integer counts as the response"))
    end
    block_of(item) = _sd_target_block(item.target; disambiguate = true)
    mean_spec, sigma_spec = _gsl_ordinary_specs(f, what, sd_, sig_t, mu_targets, sigma_re, re,
                                                data, length(y), block_of; allow_slope = false)
    return _withformula(_fit_gaussian_sigma_sd_laplace(fam, y, Xμ, Xσ, mean_spec, sigma_spec,
                                                       nmμ, nmσ, g_tol; se = se), f)
end

# `(1 | p | g)` parses as `((1 | p) | g)`: return (label, inner lhs) or nothing.
function _gsl_re_label(lhs)
    (lhs isa FunctionTerm && lhs.f === (|) && length(lhs.args) == 2 && lhs.args[2] isa Term) ||
        return nothing
    return (lhs.args[2].sym, lhs.args[1])
end

# Gaussian correlated mean + sigma random intercepts on one grouping factor
# (drmTMB `bf(y ~ x + (1 | p | g), sigma ~ 1 + (1 | p | g))`, no SD formulas):
#   (b_g, delta_g) ~ N(0, [s_mu^2, rho s_mu s_sigma; rho s_mu s_sigma, s_sigma^2]),
# fitted by the same dense Laplace engine with a correlated standardised prior.
# θ = [β_μ; β_σ; log s_mu; log s_sigma; eta], rho = tanh(eta) (drmTMB `eta_cor_mu_sigma`).
# Native drmTMB refuses this shape with SD formulas, so it is refused here too.
function _drm_gaussian_musigma_corr(fam::Gaussian, f::DrmFormula, re, sigma_re, y, Xμ, Xσ,
                                    nmμ, nmσ, data, g_tol; reml::Bool = false)
    what = "Gaussian correlated mean + sigma random intercepts `(1 | p | g)`"
    isempty(_sd_target_parts(f)) && isempty(_sdphylo_parts(f)) || throw(ArgumentError(
        "drm: $what cannot take `sd(...)` formulas (native drmTMB refuses this combination)."))
    (length(re) == 1 && length(sigma_re) == 1) || throw(ArgumentError(
        "drm: $what takes exactly one labelled term in `mu` and one in `sigma`."))
    lm = _gsl_re_label(re[1][1]); ls = _gsl_re_label(sigma_re[1][1])
    (lm !== nothing && ls !== nothing) || throw(ArgumentError(
        "drm: $what needs the same label in `mu` and `sigma`, e.g. `(1 | p | g)` in each."))
    lm[1] === ls[1] || throw(ArgumentError(
        "drm: $what: labels differ (`$(lm[1])` in mu, `$(ls[1])` in sigma)."))
    isint(t) = t isa ConstantTerm && t.n == 1
    (isint(lm[2]) && isint(ls[2])) || throw(ArgumentError(
        "drm: $what supports random intercepts `(1 | p | g)` only."))
    mgrp = re[1][2]; sgrp = sigma_re[1][2]
    mgrp === sgrp || throw(ArgumentError(
        "drm: $what needs one grouping factor (got `$mgrp` in mu and `$sgrp` in sigma)."))
    gidx, G = _group_index(getproperty(data, mgrp); col = mgrp)
    reml && return _fit_gaussian_musigma_corr_reml(fam, y, Xμ, Xσ, gidx, G, nmμ, nmσ, mgrp, g_tol)
    Z1 = ones(G, 1)
    mean_spec = (gidx, G, Z1, ["(Intercept)"], mgrp, :resd)
    sigma_spec = (gidx, G, Z1, ["(Intercept)"], mgrp, ones(length(y)), nothing, :resd, :intercept)
    return _fit_gaussian_sigma_sd_laplace(fam, y, Xμ, Xσ, mean_spec, sigma_spec, nmμ, nmσ, g_tol;
                                          corr = true)
end

# ---------------------------------------------------------------------------------------------
# REML for the correlated mean + sigma random-intercept pair `(1 | p | g)` in `mu` and `sigma`
# (fixed-effect predictors only, no SD formula). Native drmTMB (`REML = TRUE`) moves beta_mu AND
# beta_sigma into TMB's random block with flat priors, so the reported objective is ONE Laplace
# approximation over w = (va, vb, beta_mu, beta_sigma),
#
#   nll_R(alpha) = F(w^) + 1/2 log det H(w^) + (G/2) log(1 - rho^2) - (p/2) log 2pi,
#
# with F the ML joint negative log density (correlated standardised prior), H = d2F/dw2 over ALL
# of w, p = p_mu + p_sigma, and only alpha = (log s_mu, log s_sigma, atanh rho) outer. This is the
# structured-sigma REML objective above with the correlated per-group prior, so it is an
# APPROXIMATE restricted likelihood (the sigma integral is a Laplace one), as native reports.
# ---------------------------------------------------------------------------------------------
function _gmsc_joint_reml(y, Xμ, Xσ, g, G::Int, c, d, ρ, w; deriv::Bool)
    n = length(y); pμ = size(Xμ, 2); pσ = size(Xσ, 2)
    T = promote_type(typeof(c), typeof(d), typeof(ρ), eltype(w))
    va = @view w[1:G]; vb = @view w[(G+1):(2G)]
    rμ = (2G+1):(2G+pμ); rσ = (2G+pμ+1):(2G+pμ+pσ)
    lin = Xμ * w[rμ]; eta = Xσ * w[rσ]
    sr = 1 / (1 - ρ * ρ)
    F = 0.5 * sr * (sum(abs2, va) - 2 * ρ * dot(va, vb) + sum(abs2, vb))
    fa = Vector{T}(undef, n); fe = similar(fa); faa = similar(fa); fae = similar(fa); fee = similar(fa)
    @inbounds for i in 1:n
        f, a, e, aa, ae, ee = _gsl_kernel(Gaussian(), y[i], lin[i] + c * va[g[i]], eta[i] + d * vb[g[i]])
        F += f; fa[i] = a; fe[i] = e; faa[i] = aa; fae[i] = ae; fee[i] = ee
    end
    deriv || return F, nothing, nothing
    q = 2G + pμ + pσ
    gr = zeros(T, q); H = zeros(T, q, q)
    @inbounds for k in 1:G
        gr[k] += sr * (va[k] - ρ * vb[k]); gr[G+k] += sr * (vb[k] - ρ * va[k])
        H[k, k] += sr; H[G+k, G+k] += sr
        H[k, G+k] -= ρ * sr; H[G+k, k] -= ρ * sr
    end
    @inbounds for i in 1:n
        k = g[i]
        gr[k] += c * fa[i]; gr[G+k] += d * fe[i]
        H[k, k] += c * c * faa[i]; H[G+k, G+k] += d * d * fee[i]
        x = c * d * fae[i]
        H[k, G+k] += x; H[G+k, k] += x
        for a in 1:pμ
            h1 = c * faa[i] * Xμ[i, a]; h2 = d * fae[i] * Xμ[i, a]
            H[k, 2G+a] += h1; H[2G+a, k] += h1
            H[G+k, 2G+a] += h2; H[2G+a, G+k] += h2
        end
        for a in 1:pσ
            h1 = c * fae[i] * Xσ[i, a]; h2 = d * fee[i] * Xσ[i, a]
            H[k, 2G+pμ+a] += h1; H[2G+pμ+a, k] += h1
            H[G+k, 2G+pμ+a] += h2; H[2G+pμ+a, G+k] += h2
        end
    end
    gr[rμ] .= Xμ' * fa; gr[rσ] .= Xσ' * fe
    H[rμ, rμ] .= Xμ' * (faa .* Xμ)
    H[rσ, rσ] .= Xσ' * (fee .* Xσ)
    Hμσ = Xμ' * (fae .* Xσ)
    H[rμ, rσ] .= Hμσ; H[rσ, rμ] .= Hμσ'
    return F, gr, H
end

function _fit_gaussian_musigma_corr_reml(fam::Gaussian, y, Xμ, Xσ, gidx, G, nmμ, nmσ, grp, g_tol;
                                         se::Bool = true)
    n = length(y); pμ, pσ = size(Xμ, 2), size(Xσ, 2); p = pμ + pσ
    q = 2G + p
    q <= _GSL_MAX_LATENT || throw(ArgumentError(
        "drm: Gaussian correlated mean + sigma random intercepts use a dense Laplace " *
        "engine limited to $(_GSL_MAX_LATENT) latent coordinates (got $q under REML)."))
    rank(Xμ) == pμ || throw(ArgumentError("drm: method = :REML requires a full-rank `mu` fixed-effect design."))
    rμ = (2G+1):(2G+pμ); rσ = (2G+pμ+1):q
    joint_at(α) = (w, deriv) -> _gmsc_joint_reml(y, Xμ, Xσ, gidx, G, exp(α[1]), exp(α[2]), tanh(α[3]), w;
                                                 deriv = deriv)
    β0 = Xμ \ y
    sd0 = std(y - Xμ * β0) + eps()
    w0 = zeros(q); w0[rμ] .= β0; w0[2G+pμ+1] = log(0.7 * sd0)

    function laplace(α::AbstractVector{T}, what::Vector{Float64}) where {T}
        joint = joint_at(α)
        w = convert(Vector{T}, what)
        if T <: ForwardDiff.Dual
            _, g, H = joint(what, true)
            Cn = cholesky(Symmetric(H); check = false)
            issuccess(Cn) || return T(REML_NONPD_PENALTY)
            w = what .- (Cn \ g)
        end
        F, _, H = joint(w, true)
        C = cholesky(Symmetric(H); check = false)
        issuccess(C) || return T(REML_NONPD_PENALTY)
        return F + 0.5 * logdet(C) + 0.5 * G * log(1 - tanh(α[3])^2) - 0.5 * p * log(2π)
    end

    fg = function (α, b0, grad)
        αf = Vector{Float64}(α)
        joint = joint_at(αf)
        start = iszero(b0) ? w0 : Vector{Float64}(b0)
        w, ok = _gsl_mode(joint, start)
        ok || ((w, ok) = _gsl_mode(joint, w0))
        ok || return grad ? (1e18, fill(NaN, 3), zeros(q), false) : (1e18, zeros(q), false)
        _, _, H = joint(w, true)
        issuccess(cholesky(Symmetric(H); check = false)) ||
            return grad ? (1e18, fill(NaN, 3), w, false) : (1e18, w, false)
        val = laplace(αf, w)
        grad || return (val, w, true)
        gα = ForwardDiff.gradient(t -> laplace(t, w), αf)
        return (val, gα, w, all(isfinite, gα))
    end

    α0 = [log(0.5 * sd0), log(0.3), 0.0]
    what = "Gaussian correlated mean + sigma random intercepts (REML)"
    α̂, nllhat, conv, _, nll, grad! = _ordinary_laplace_optimize(
        fg, α0, n, q, g_tol; se = false, context = what)
    joint̂ = joint_at(α̂)
    ŵ, okw = _gsl_mode(joint̂, w0)
    okw || (conv = false)
    _, _, Ĥ = joint̂(ŵ, true)
    Cĥ = cholesky(Symmetric(Ĥ); check = false)
    issuccess(Cĥ) || (conv = false)
    let (w2, ok2) = _gsl_mode(joint̂, vcat(zeros(2G), ŵ[(2G+1):q]))
        if ok2 && issuccess(cholesky(Symmetric(joint̂(w2, true)[3]); check = false))
            nll2 = laplace(α̂, w2)
            if isfinite(nllhat) && abs(nll2 - nllhat) > 1e-6 * max(1.0, abs(nllhat))
                @warn "drm: $what: the joint mode is not unique at the estimate; the fit is marked not converged."
                conv = false
            end
        end
    end
    βμ = ŵ[rμ]; βσ = ŵ[rσ]
    θ̂ = vcat(βμ, βσ, α̂)
    V = fill(NaN, p + 3, p + 3)
    if se && issuccess(Cĥ)
        Hα = _finite_hessian_from_gradient(grad!, α̂; context = what)
        if all(isfinite, Hα)
            Vα = Matrix(_vcov_from_hessian(Hα))
            J = ForwardDiff.jacobian(α̂) do t
                _, g, H = _gmsc_joint_reml(y, Xμ, Xσ, gidx, G, exp(t[1]), exp(t[2]), tanh(t[3]), ŵ; deriv = true)
                ŵ .- (cholesky(Symmetric(H)) \ g)
            end
            Hinv = Matrix(inv(Cĥ))
            Jβ = J[(2G+1):q, :]
            Cββ = Hinv[(2G+1):q, (2G+1):q] + Jβ * Vα * Jβ'
            Cβα = Jβ * Vα
            V = [Cββ Cβα; Cβα' Vα]
            V = (V .+ V') ./ 2
        end
    end
    c = exp(α̂[1]); d = exp(α̂[2]); ρ = tanh(α̂[3])
    cv = fill(c, n); dv = fill(d, n)
    ml_nll = let
        jm = (v, deriv) -> _gsl_joint(fam, y, Xμ * βμ, Xσ * βσ, gidx, cv, gidx, dv, nothing,
                                      @view(v[1:G]), @view(v[(G+1):2G]), G, G; deriv = deriv, rho = ρ)
        vml, okm = _gsl_mode(jm, zeros(2G))
        if okm
            Fm, _, Hm = jm(vml, true)
            Cm = cholesky(Symmetric(Hm); check = false)
            issuccess(Cm) ? Fm + 0.5 * logdet(Cm) + 0.5 * G * log(1 - ρ^2) : NaN
        else
            NaN
        end
    end
    ab = c .* ŵ[1:G]; bb = d .* ŵ[(G+1):2G]
    eta_fixed = Xσ * βσ; mean_fixed = Xμ * βμ
    means = Dict(:mu => mean_fixed .+ ab[gidx])
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict{Symbol,Vector{Float64}}(
        :sigma => exp.(eta_fixed .+ bb[gidx]),
        :sigma_fixed => exp.(eta_fixed),
        :sigma_random_effect => bb,
        :sigma_group_index => Float64.(gidx),
        :sd_group => fill(c, G),
        :sd => fill(c, n),
        :sd_group_index => Float64.(gidx))
    blocks = Pair{Symbol,UnitRange{Int}}[:mu => 1:pμ, :sigma => (pμ+1):p, :resd => (p+1):(p+2), :cor => (p+3):(p+3)]
    names = Pair{Symbol,Vector{String}}[:mu => nmμ, :sigma => nmσ,
                                         :resd => [String(grp), "$(grp)_logsigma"],
                                         :cor => ["cor(mu, sigma | $(grp))"]]
    ranef = Dict{Symbol,Any}(Symbol(grp) => ab, Symbol(grp, "_logsigma") => bb)
    fit = DrmFit(fam, blocks, names, θ̂, V, -nllhat, n, conv, means, obs, scales)
    fit = _withmarginal(_withranef(fit, ranef), :Laplace)
    return _withreml(fit, -nllhat, -ml_nll)
end

# Gaussian mean `sd(g) ~ …` beside a random intercept on `sigma` that has NO sd() formula of
# its own (drmTMB `bf(y ~ x + (1 | g), sigma ~ 1 + (1 | h), sd(g) ~ z)`): independent mean and
# sigma intercepts (the same or different groupings), or, with the label in both,
# `(1 | p | g)` in `mu` and in `sigma`, the correlated pair on one grouping. The marginal is the
# Laplace value of the dense engine of `sd(g, sigma)`: the sigma SD takes the constant design
# `~ 1`. ML only; one mean intercept and one sigma intercept (no slopes), as the neighbouring
# routes.
function _drm_gaussian_mean_sd_sigma_re(f::DrmFormula, fam::Gaussian, sdp, sdpp, re, structured,
                                        structured_sigma, sigma_re, metav, has_missing_response,
                                        y, Xμ, Xσ, nmμ, nmσ, data, K, A, coords, g_tol, method,
                                        algorithm, penalty)
    what = "Gaussian mean `sd(g) ~ …` with a `sigma` random intercept"
    isempty(sdpp) || throw(ArgumentError("drm: $what cannot be combined with `sd(g, phylogenetic)`."))
    metav === nothing || throw(ArgumentError("drm: $what cannot be combined with `meta_V(...)`."))
    structured === nothing && structured_sigma === nothing || throw(ArgumentError(
        "drm: $what cannot be combined with a structured mean or sigma effect."))
    (K === nothing && A === nothing && coords === nothing) || throw(ArgumentError(
        "drm: $what uses ordinary random intercepts only; `K`, `A` and `coords` are not used."))
    penalty === nothing || throw(ArgumentError("drm: `penalty` is not wired for sd() submodel routes."))
    method === :REML && throw(ArgumentError(
        "drm: $what supports maximum likelihood only (`method = :REML` is not implemented for this shape)."))
    algorithm === :auto || throw(ArgumentError("drm: `algorithm = :$algorithm` is not implemented for $what."))
    has_missing_response && throw(ArgumentError(
        "drm: $what requires observed responses; remove incomplete rows first (`drm_listwise`)."))
    targets = _sd_target_parts(f)
    allowed = Set((:mu, :sigma))
    for item in targets
        push!(allowed, item.key)
    end
    all(p -> first(p) in allowed, f.forms) || throw(ArgumentError(
        "drm: $what supports only `mu`, `sigma` and the mean `sd(g) ~ …` formula."))
    (length(re) == 1 && length(sigma_re) == 1) || throw(ArgumentError(
        "drm: $what takes exactly one mean random intercept and one `sigma` random intercept " *
        "(got $(length(re)) and $(length(sigma_re)) terms)."))
    length(targets) == 1 && only(targets).target.dpar === :mu || throw(ArgumentError(
        "drm: $what takes exactly one mean-side `sd(g) ~ …` formula."))
    mu_t = only(targets)
    md = mu_t.target
    isint(t) = t isa ConstantTerm && t.n == 1
    lm = _gsl_re_label(re[1][1]); ls = _gsl_re_label(sigma_re[1][1])
    corr = lm !== nothing || ls !== nothing
    if corr
        (lm !== nothing && ls !== nothing && lm[1] === ls[1]) || throw(ArgumentError(
            "drm: $what needs the same label in `mu` and `sigma`, e.g. `(1 | p | g)` in each, " *
            "or no label for independent effects."))
        (isint(lm[2]) && isint(ls[2])) || throw(ArgumentError(
            "drm: $what supports random intercepts `(1 | p | g)` only."))
        re[1][2] === sigma_re[1][2] || throw(ArgumentError(
            "drm: a labelled mean-sigma correlation needs one grouping factor (got " *
            "`$(re[1][2])` in mu and `$(sigma_re[1][2])` in sigma)."))
    else
        isint(re[1][1]) && isint(sigma_re[1][1]) || throw(ArgumentError(
            "drm: $what supports random intercepts `(1 | g)` only (no random slopes)."))
    end
    mgrp = re[1][2]; sgrp = sigma_re[1][2]
    (md.level === :iid && md.group === mgrp && md.block === nothing &&
     md.member in (nothing, :intercept)) || throw(ArgumentError(
        "drm: the mean-side `sd(...)` must target the mean random intercept `(1 | $mgrp)`."))
    n = length(y)
    ga, Ga = _group_index(getproperty(data, mgrp); col = mgrp)
    Za, nma = _sd_group_design(f.response, mu_t.rhs, data, ga, Ga, mgrp;
                               schema_cache = f.schema_cache, schema_key = mu_t.key)
    size(Za, 2) <= Ga && rank(Za) == size(Za, 2) || throw(ArgumentError(
        "drm: $what mean group-level design is rank deficient."))
    mean_spec = (ga, Ga, Za, nma, mgrp, _sd_target_block(md; disambiguate = true))
    gb, Gb = _group_index(getproperty(data, sgrp); col = sgrp)
    sigma_spec = (gb, Gb, ones(Gb, 1), ["(Intercept)"], sgrp, ones(n), nothing,
                  _sd_target_block((group = sgrp, level = :iid, member = nothing, dpar = :sigma,
                                    block = nothing); disambiguate = true), :intercept)
    return _fit_gaussian_sigma_sd_laplace(fam, y, Xμ, Xσ, mean_spec, sigma_spec, nmμ, nmσ, g_tol;
                                          corr = corr, sdreg = true)
end
