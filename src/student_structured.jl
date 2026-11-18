# student_structured.jl: Student `spatial(1 | g, coords)` and `spatial(1 + x | g, coords)` on the mean,
# fitted as drmTMB fits them (gap-map cells `student|spatial|lab0|q{1,2}|sd-none`, R-only before this file).
#
# Model. With the fixed-range kernel R = correlation form of exp(-d / r) + 1e-6 I (r the median positive
# pairwise distance; no range parameter, as the Poisson, Gamma and NB2 routes),
#
#     eta_i = X_i' beta + b_1[g(i)] + x_i b_2[g(i)],     b_k ~ N(0, s_k^2 R) independently,
#
# the second component only for `(1 + x | g)` (drmTMB's bare form is the independent intercept and slope,
# correlation fixed at 0). Student observation model as `_fit_indep_re_laplace` (identity link, log sigma,
# nu = 2 + exp(eta_nu)). The TMB-convention Laplace marginal (the (2 pi) constants cancel) is
#
#     nll = sum_i -log f(y_i | eta_i(b)) + 1/2 sum_k b_k' R^-1 b_k / s_k^2 + sum_k [G log s_k - 1/2 logdet R^-1]
#           + 1/2 logdet H(b),
#
# H the Hessian in b of the penalised negative log-likelihood. Same Newton inner mode and dual-number
# lifting of the mode as `_fit_indep_re_laplace` (whose observation terms it reuses); H is dense,
# (K G) x (K G), which is small for site-level kernels.

function _fit_student_structured_laplace(fam::Student, y, Xμ, Xσ, Xν, C, labels, slope, slope_name,
                                         nmμ, nmσ, nmν, grp, g_tol; se::Bool = true)
    size(Xσ, 2) == 1 && size(Xν, 2) == 1 && all(==(1.0), Xσ) && all(==(1.0), Xν) || throw(ArgumentError(
        "Student `spatial(...)` on the mean requires `sigma ~ 1` and `nu ~ 1`; covariate `sigma` or `nu` " *
        "formulas are not implemented on this route."))
    Q, gidx = _general_cov_setup(C, labels; normalize = true)
    G = size(Q, 1)
    Qd = Matrix{Float64}(Q)
    logdetQ = logdet(cholesky(Symmetric(Qd)))
    n = length(y); yv = Float64.(y)
    pμ = size(Xμ, 2); k0 = pμ + 2
    w = slope === nothing ? [ones(n)] : [ones(n), Float64.(slope)]
    K = length(w); q = K * G
    cols = [(k - 1) * G .+ gidx for k in 1:K]

    function joint_terms(b, η0, ησ, ην, invs; expected::Bool = false)
        T = promote_type(eltype(b), typeof(η0[1]), typeof(ησ), typeof(ην), eltype(invs))
        H = zeros(T, q, q); grad = zeros(T, q); data = zero(T)
        @inbounds for i in 1:n
            η = η0[i]
            for k in 1:K
                η += w[k][i] * b[cols[k][i]]
            end
            v, g1, wt, wE = _indep_obs_terms(Val(:student), yv[i], η, ησ, ην)
            expected && (wt = wE)
            data += v
            for k in 1:K
                wk = w[k][i]; iszero(wk) && continue
                ck = cols[k][i]
                grad[ck] += wk * g1
                for l in 1:K
                    wl = w[l][i]; iszero(wl) && continue
                    H[ck, cols[l][i]] += wk * wl * wt
                end
            end
        end
        pen = zero(T)
        for k in 1:K
            r = (k - 1) * G .+ (1:G)
            bk = b[r]; Qb = Qd * bk
            grad[r] .+= invs[k] .* Qb
            H[r, r] .+= invs[k] .* Qd
            pen += invs[k] * dot(bk, Qb)
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
            0.5 * dot(grad, step) <= 1e-13 * (1 + abs(J0)) && return b .- step, true
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
        η0 = Xμ * θ[1:pμ]; ησ = θ[pμ+1]; ην = θ[pμ+2]; lsg = θ[k0+1:k0+K]
        any(l -> abs(_indep_primal(l)) > 30, lsg) && return T(1e18)
        abs(_indep_primal(ησ)) > 30 && return T(1e18)
        invs = exp.(-2 .* lsg)
        f0 = (_indep_primal.(η0), _indep_primal(ησ), _indep_primal(ην), _indep_primal.(invs))
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
        val = J + G * sum(lsg) - 0.5 * K * logdetQ + 0.5 * logdet(ch)
        isfinite(val) || return T(1e18)
        return val
    end

    # Start as the independent-effects route: OLS mean, sigma at half the residual spread, nu = 2 + e^eta at 12.
    θ0 = zeros(k0 + K)
    βμ0 = Xμ \ yv
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(std(yv - Xμ * βμ0) + eps()) - log(2.0)
    θ0[pμ+2] = log(10.0)
    for k in 1:K
        sw = all(==(1.0), w[k]) ? 1.0 : max(std(w[k]), 1e-8)
        θ0[k0+k] = θ0[pμ+1] - log(sw)
    end
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
    V = se ? _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂);
                                context = "Laplace Student (coordinate spatial effect)") :
             fill(NaN, length(θ̂), length(θ̂))
    sdlabels = K == 1 ? [String(grp)] : [String(grp), "$(grp):$(slope_name)"]
    blocks = Pair{Symbol,UnitRange{Int}}[:mu => 1:pμ, :sigma => (pμ+1):(pμ+1), :nu => (pμ+2):(pμ+2),
                                         :resd => (k0+1):(k0+K)]
    names = Pair{Symbol,Vector{String}}[:mu => nmμ, :sigma => nmσ, :nu => nmν, :resd => sdlabels]
    means = Dict(:mu => Xμ * θ̂[1:pμ]); obs = Dict(:mu => yv)
    scales = Dict{Symbol,Vector{Float64}}(:sigma => fill(exp(θ̂[pμ+1]), n),
                                          :nu => fill(2 + exp(θ̂[pμ+2]), n))
    fit = DrmFit(fam, blocks, names, θ̂, V, -nllhat, n, converged, means, obs, scales)
    return _withiterations(_withmarginal(_withnll(fit, nll), :Laplace), Optim.iterations(res))
end
