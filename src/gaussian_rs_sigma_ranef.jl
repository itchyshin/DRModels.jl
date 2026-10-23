# gaussian_rs_sigma_ranef.jl — Gaussian random-slope mean plus a `sigma` random
# intercept on the same grouping factor (#960, drmTMB parity):
#
#     y ~ x + … + (1 + x | g),   sigma ~ … + (1 | g)
#
# This extends the #745 cell (`_fit_musigma_ranef_gaussian`: mean `(1 | g)` +
# sigma `(1 | g)`, one 2×2 Laplace block per group) to a correlated intercept +
# slope on the mean. drmTMB fits it with its nested Laplace over the stacked
# (u_mu, u_sigma) vector, with the usual unstructured 2×2 covariance for
# `(1 + x | g)` and an independent `N(0, τ²)` prior on the sigma intercept (no
# cross-parameter correlation without a coupled `(1 | tag | g)` term). Because
# all three effects share one grouping factor, the joint Hessian is block
# diagonal with one 3×3 block per group, so a per-group 3-D Laplace
# approximation IS the whole-model Laplace approximation (same argument as the
# #745 derivation in gaussian_ranef.jl).
#
# For a group with m members, mean random effect u = (u0, u1), sigma effect δ,
# fixed-effect residual r_i = y_i − Xμ_i′β_μ, fixed-effect log σ η_i = Xσ_i′β_σ,
# z_i = (1, x_i), e_i = exp(−2(η_i + δ)), s_i = r_i − z_i′u:
#
#     h(u, δ) = Σ (η_i + δ) + ½ Σ s_i² e_i + ½ u′Σ⁻¹u + ½ δ²/τ²
#     ∇_u h  = −Σ s_i e_i z_i + Σ⁻¹u,            ∂_δ h = m − Σ s_i² e_i + δ/τ²
#     H_uu   = Σ e_i z_i z_i′ + Σ⁻¹,  H_uδ = 2 Σ s_i e_i z_i,  H_δδ = 2 Σ s_i² e_i + 1/τ²
#
# and the group's negative log marginal is
#
#     ½ m log 2π + h(û, δ̂) + ½ log det Σ + log τ + ½ log det H(û, δ̂).
#
# Σ = L L′ with L = [exp(a) 0; c exp(b)] (the `vc`/`:recov` convention of
# `_fit_correlated_ranef_gaussian`). Unlike that route this one optimises on the
# user's (uncentred) covariate with BFGS: the near-boundary ridge (ρ ≈ −1 on
# growth curves) stalled a centred L-BFGS 4.8 log-lik units short of drmTMB on
# ChickWeight, while BFGS reaches drmTMB's optimum in ~25 iterations.

# h, ∇h, H for one group at (u0, u1, δ), in the caller's number type.
function _rssig_terms(r, x, η, u0, u1, δ, Si11, Si12, Si22, invτ2)
    T = promote_type(eltype(r), eltype(η), typeof(u0), typeof(δ), typeof(Si11), typeof(invτ2))
    m = length(r)
    hval = zero(T); g0 = zero(T); g1 = zero(T); gδ = T(m)
    H00 = zero(T); H01 = zero(T); H11 = zero(T); H0δ = zero(T); H1δ = zero(T); Hδδ = zero(T)
    @inbounds for i in 1:m
        e = exp(-2 * (η[i] + δ))
        s = r[i] - u0 - u1 * x[i]
        se = s * e; s2e = s * se
        hval += η[i] + δ + 0.5 * s2e
        g0 -= se; g1 -= se * x[i]; gδ -= s2e
        H00 += e; H01 += e * x[i]; H11 += e * x[i]^2
        H0δ += 2 * se; H1δ += 2 * se * x[i]; Hδδ += 2 * s2e
    end
    # prior
    hval += 0.5 * (Si11 * u0^2 + 2 * Si12 * u0 * u1 + Si22 * u1^2) + 0.5 * invτ2 * δ^2
    g0 += Si11 * u0 + Si12 * u1; g1 += Si12 * u0 + Si22 * u1; gδ += invτ2 * δ
    H00 += Si11; H01 += Si12; H11 += Si22; Hδδ += invτ2
    H = [H00 H01 H0δ; H01 H11 H1δ; H0δ H1δ Hδδ]
    return hval, [g0, g1, gδ], H
end

_rssig_primal(x::ForwardDiff.Dual) = _rssig_primal(ForwardDiff.value(x))
_rssig_primal(x::Real) = Float64(x)

# Damped Newton on Float64 primals for the group mode; value-decrease line search,
# diagonal fallback direction when H is not PD away from the mode.
function _rssig_mode_primal(r, x, η, Si11, Si12, Si22, invτ2, start)
    v = copy(start)
    for _ in 1:200
        h0, g, H = _rssig_terms(r, x, η, v[1], v[2], v[3], Si11, Si12, Si22, invτ2)
        norm(g, Inf) < 1e-12 * (1 + abs(h0)) && break
        C = cholesky(Symmetric(H); check = false)
        Δ = issuccess(C) ? -(C \ g) : -g ./ max.(abs.(diag(H)), 1e-8)
        t = 1.0; moved = false
        while t > 1e-10
            vn = v .+ t .* Δ
            hn = first(_rssig_terms(r, x, η, vn[1], vn[2], vn[3], Si11, Si12, Si22, invτ2))
            if isfinite(hn) && hn <= h0
                v = vn; moved = true; break
            end
            t *= 0.5
        end
        moved || break
        norm(t .* Δ, Inf) < 1e-13 * (1 + norm(v, Inf)) && break
    end
    return v
end

"""
    _fit_rs_sigma_ranef_gaussian(fam, y, Xμ, Xσ, gidx, G, xs, nmμ, nmσ, grp, g_tol) -> DrmFit

Gaussian model with a correlated random intercept + slope on the mean
`(1 + x | g)` and a random intercept on `sigma` `(1 | g)`, same grouping factor
(#960; the slope extension of #745). Per-group 3-D Laplace marginal, tagged
`marginal = :Laplace` (the objective drmTMB maximises).
θ = [β_μ; β_σ; a, b, c (log-Cholesky of the 2×2 mean RE covariance); log τ].
Blocks: `:mu`, `:sigma`, `:recov` (`"g:L11"`, `"g:L22"`, `"g:L21"`, read by
`vc`), `:resd` (`"g_logsigma"`, the sigma random-intercept SD).
"""
function _fit_rs_sigma_ranef_gaussian(fam::Gaussian, y, Xμ, Xσ, gidx, G, xs, nmμ, nmσ, grp, g_tol)
    n = length(y); pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    yv = Float64.(y); xs = Float64.(xs)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    l2π = log(2π)
    ia = pμ + pσ + 1                       # first covariance parameter
    modes = [zeros(3) for _ in 1:G]        # warm starts for the per-group modes

    function nll_x(θ, xv; store::Bool = false)
        T = eltype(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]
        a, b, c, lτ = θ[ia], θ[ia+1], θ[ia+2], θ[ia+3]
        (max(abs(_rssig_primal(a)), abs(_rssig_primal(b)), abs(_rssig_primal(lτ))) > 30) && return T(1e18)
        l11 = exp(a); l22 = exp(b)
        # Σ⁻¹ = L⁻ᵀL⁻¹ with L⁻¹ = [1/l11 0; −c/(l11 l22) 1/l22]
        m21 = -c / (l11 * l22)
        Si11 = 1 / l11^2 + m21^2; Si12 = m21 / l22; Si22 = 1 / l22^2
        invτ2 = exp(-2 * lτ)
        r_all = yv .- Xμ * βμ; η_all = Xσ * βσ
        s = zero(T)
        for (k, idx) in enumerate(members)
            m = length(idx)
            m == 0 && continue
            r = r_all[idx]; η = η_all[idx]; x = xv[idx]
            v0 = _rssig_mode_primal(_rssig_primal.(r), x, _rssig_primal.(η),
                                    _rssig_primal(Si11), _rssig_primal(Si12), _rssig_primal(Si22),
                                    _rssig_primal(invτ2), modes[k])
            all(isfinite, v0) || return T(1e18)
            store && (modes[k] .= v0)
            u0 = T(v0[1]); u1 = T(v0[2]); δ = T(v0[3])
            for _ in 1:2                   # lift the mode to dual arithmetic (IFT)
                _, g, H = _rssig_terms(r, x, η, u0, u1, δ, Si11, Si12, Si22, invτ2)
                C = cholesky(Symmetric(H); check = false)
                issuccess(C) || return T(1e18)
                Δ = C \ g
                u0 -= Δ[1]; u1 -= Δ[2]; δ -= Δ[3]
            end
            hval, _, H = _rssig_terms(r, x, η, u0, u1, δ, Si11, Si12, Si22, invτ2)
            C = cholesky(Symmetric(H); check = false)
            issuccess(C) || return T(1e18)
            s += 0.5 * m * l2π + hval + (a + b) + lτ + 0.5 * logdet(C)
        end
        isfinite(s) || return T(1e18)
        return s
    end
    nll(θ) = nll_x(θ, xs)

    # Interior start (the fixed-effect OLS fit, RE SDs at half the residual spread).
    # Deliberately NOT warm-started from the no-sigma-RE correlated fit: that fit
    # often sits on the rho = ±1 boundary (L22 ~ e^-1000), a singular Sigma here.
    βμ0 = Xμ \ yv; res0 = yv - Xμ * βμ0
    φ0 = zeros(pμ + pσ + 4)
    φ0[1:pμ] .= βμ0
    φ0[pμ+1] = log(std(res0) + eps())
    sd0 = log(std(res0) / 2 + eps())
    φ0[ia] = sd0
    φ0[ia+1] = sd0 - log(max(std(xs), 1e-8))
    φ0[ia+2] = 0.0
    φ0[ia+3] = log(0.3)
    # BFGS on the user's (uncentred) covariate; NaN turns off the f-/x-stops so the
    # near-boundary ridge cannot end the run on a flat step (cf. #707).
    opts = Optim.Options(g_tol = g_tol, iterations = 2000,
                         f_reltol = NaN, x_abstol = NaN, x_reltol = NaN)
    res = Optim.optimize(nll, φ0, Optim.BFGS(), opts; autodiff = :forward)
    for k in 1:5                                   # bounded restarts, alternating BFGS / L-BFGS
        (drm_optim_converged(res) && nll(Optim.minimizer(res)) < 1e17) && break
        res2 = Optim.optimize(nll, Optim.minimizer(res), iseven(k) ? Optim.BFGS() : Optim.LBFGS(),
                              opts; autodiff = :forward)
        nll(Optim.minimizer(res2)) < nll(Optim.minimizer(res)) || break
        res = res2
    end
    θ̂ = Optim.minimizer(res)
    nllhat = nll_x(θ̂, xs; store = true)
    gfinal = ForwardDiff.gradient(nll, θ̂)
    converged = nllhat < 1e17 && all(isfinite, gfinal) &&
                (drm_optim_converged(res) || norm(gfinal, Inf) <= max(g_tol, 1e-5) * (1 + abs(nllhat)))
    V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂);
                           context = "Gaussian random-slope mean + sigma random intercept (Laplace)")

    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :recov => ia:(ia+2), :resd => (ia+3):(ia+3)]
    names = [:mu => nmμ, :sigma => nmσ, :recov => ["$(grp):L11", "$(grp):L22", "$(grp):L21"],
             :resd => ["$(grp)_logsigma"]]
    means = Dict(:mu => Xμ * θ̂[1:pμ]); obs = Dict(:mu => yv)
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]))      # population (u = δ = 0) σ
    # Conditional modes at θ̂ (uncentred x, so the slope/intercept are on the user's scale).
    blup = zeros(G, 2); blup_sigma = zeros(G)
    let
        a, b, c, lτ = θ̂[ia], θ̂[ia+1], θ̂[ia+2], θ̂[ia+3]
        l11 = exp(a); l22 = exp(b); m21 = -c / (l11 * l22)
        Si11 = 1 / l11^2 + m21^2; Si12 = m21 / l22; Si22 = 1 / l22^2
        r_all = yv .- Xμ * θ̂[1:pμ]; η_all = Xσ * θ̂[(pμ+1):(pμ+pσ)]
        for (k, idx) in enumerate(members)
            isempty(idx) && continue
            v = _rssig_mode_primal(r_all[idx], xs[idx], η_all[idx], Si11, Si12, Si22,
                                   exp(-2lτ), zeros(3))
            blup[k, 1] = v[1]; blup[k, 2] = v[2]; blup_sigma[k] = v[3]
        end
    end
    re = Dict(Symbol(grp) => blup, Symbol("$(grp)_logsigma") => blup_sigma)
    fit = _withranef(_withnll(DrmFit(fam, blocks, names, θ̂, V, -nllhat, n, converged,
                                     means, obs, scales), nll), re)
    return _withiterations(_withmarginal(fit, :Laplace), Optim.iterations(res))
end
