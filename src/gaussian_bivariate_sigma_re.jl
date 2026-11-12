# gaussian_bivariate_sigma_re.jl -- bivariate Gaussian with an ORDINARY correlated random
# intercept pair `(1 | p | g)` on sigma1 and sigma2 (drmTMB's `biv_gaussian()` with
# `sigma1 = ~ 1 + (1 | p | g)`, `sigma2 = ~ 1 + (1 | p | g)`), estimated by the Laplace
# approximation like drmTMB (TMB).
#
# Model.  Row i of group g:  (y1, y2)_i ~ N((X1 b1, X2 b2)_i, S_i),  S_i = [s1^2, rho s1 s2; rho s1 s2, s2^2],
#   log s1_i = Xs1_i c1 + u_g1,   log s2_i = Xs2_i c2 + u_g2,   u_g = (u_g1, u_g2) ~ N(0, Sigma_g),
# with atanh(rho / RHO_GUARD) linear in its own design (the residual route's link). The random
# effects enter ADDITIVELY on the log-sigma scale and are correlated across the two scales
# (drmTMB: `log_sigma1 += random effect`). The means carry no random effect.
#
# Likelihood.  The density is not Gaussian in u, so the marginal is not closed-form. With
# u = L z, Sigma_g = L L', z ~ N(0, I) (the standardised form TMB uses), group g contributes
#   f_g(z) = sum_i -log phi2(y_i | u = L z) + z'z / 2,
#   -log marginal_g  =  f_g(z_hat) + (1/2) log det grad^2 f_g(z_hat)        (d = 2: the 2 pi terms cancel).
# z_hat is found by damped Newton. The outer objective is differentiated with ForwardDiff: the
# mode enters the dual numbers through a few Newton steps taken with dual-valued parameters
# from the converged Float64 mode (each step makes one more derivative order exact), so the
# gradient and the Hessian that give the standard errors are exact derivatives of this Laplace
# objective, not finite differences. Whitening (u = L z) keeps a boundary Sigma_g (L -> 0) finite.
#
# Conventions follow the sibling mean route (`gaussian_bivariate_re.jl`): the group covariance is the
# `:groupcov` block with entries `Sigma_g:L11/L21/L22`, `fit.ranef.Sigma_g` is Sigma_g, and the modes
# are in `ranef(fit)`. `fit.scales[:sigma1]`/`[:sigma2]` hold exp(Xs c), the part without the
# group effects (as `fit.means` holds X b only); the conditional scale of row i is
# `fit.scales[:sigma1][i] * exp(U[1, group_index[i]])` with `U = fit.ranef.effects[group]`.
# ML only: REML with a random effect on the scale is not implemented.

_sre_fv(x) = x isa ForwardDiff.Dual ? _sre_fv(ForwardDiff.value(x)) : x

# f_g(z), its gradient and Hessian in z for one group, in closed form (no nested AD). With
# a = (log s1 + u1, log s2 + u2), z1 = r1 e^-a1, z2 = r2 e^-a2, a row's -log phi2 is
#   log 2pi + a1 + a2 + log(om)/2 + (z1^2 - 2 rho z1 z2 + z2^2) / (2 om),   om = 1 - rho^2,
# with d/da1 = 1 - (z1^2 - rho z1 z2)/om, d/da2 = 1 - (z2^2 - rho z1 z2)/om,
# d2/da1^2 = (2 z1^2 - rho z1 z2)/om, d2/da2^2 = (2 z2^2 - rho z1 z2)/om, d2/da1da2 = -rho z1 z2/om.
# Then u = L z gives grad_z = L' grad_a + z and Hess_z = L' Hess_a L + I. Generic in the element type,
# so ForwardDiff passes straight through it.
function _sre_fgh(z, L, rows, r1, r2, ls1, ls2, ρ, om)
    l11 = L[1, 1]; l21 = L[2, 1]; l22 = L[2, 2]
    u1 = l11 * z[1]
    u2 = l21 * z[1] + l22 * z[2]
    f = (z[1] * z[1] + z[2] * z[2]) / 2
    G1 = zero(f); G2 = zero(f); A11 = zero(f); A22 = zero(f); A12 = zero(f)
    @inbounds for i in rows
        a1 = ls1[i] + u1
        a2 = ls2[i] + u2
        z1 = r1[i] * exp(-a1)
        z2 = r2[i] * exp(-a2)
        c = ρ[i] * z1 * z2
        o = om[i]
        f += log(2π) + a1 + a2 + log(o) / 2 + (z1 * z1 - 2 * c + z2 * z2) / (2 * o)
        G1 += 1 - (z1 * z1 - c) / o
        G2 += 1 - (z2 * z2 - c) / o
        A11 += (2 * z1 * z1 - c) / o
        A22 += (2 * z2 * z2 - c) / o
        A12 -= c / o
    end
    g = [l11 * G1 + l21 * G2 + z[1], l22 * G2 + z[2]]
    # L' M L with M = [A11 A12; A12 A22], L lower triangular
    m11 = A11 * l11 + A12 * l21;  m12 = A12 * l22
    m21 = A12 * l11 + A22 * l21;  m22 = A22 * l22
    H = [l11 * m11 + l21 * m21 + 1  l11 * m12 + l21 * m22;
         l22 * m21                  l22 * m22 + 1]
    H[2, 1] = H[1, 2]
    return f, g, H
end

# Damped Newton for the group mode, Float64 only. Returns (z, converged). Stops when the Newton
# decrement g'H^-1 g / 2 (the predicted gain in f) is below 1e-13 of max(1, |f|): a gradient test
# alone cannot be met to 1e-10 when f is a few hundred (rounding in f). The derivative orders
# that the outer ForwardDiff passes need are made exact by the dual Newton steps afterwards.
function _sre_mode(fgh::F, z0::Vector{Float64}) where {F}
    z = copy(z0)
    fcur, g, H = fgh(z)
    for _ in 1:200
        λ = 0.0
        step = zeros(2)
        ok = false
        for _ in 1:60
            ch = cholesky(Symmetric(H + λ * I); check = false)
            if issuccess(ch)
                step = ch \ g
                ok = true
                break
            end
            λ = λ == 0.0 ? 1e-6 : 10λ
        end
        ok || return z, false
        scale = max(1.0, abs(fcur))
        dec = dot(g, step) / 2
        (dec <= 1e-13 * scale || norm(g, Inf) <= 1e-9 * scale) && return z, true
        t = 1.0
        accepted = false
        for _ in 1:60
            zn = z .- t .* step
            fn, gn, Hn = fgh(zn)
            if isfinite(fn) && fn <= fcur - 1e-4 * t * dot(g, step)
                z = zn; fcur = fn; g = gn; H = Hn; accepted = true
                break
            end
            t /= 2
        end
        accepted || return z, dec <= 1e-9 * scale
    end
    return z, false
end

function _fit_bivariate_sigma_re(f::BivariateDrmFormula, fam::Gaussian, data, fixed, spec;
                                 g_tol::Real, method::Symbol = :ML)
    method === :ML || throw(ArgumentError(
        "drm: `method = :$method` is not supported with a random effect on `sigma1`/`sigma2`; " *
        "use method = :ML (the default). drmTMB's REML for this model is not implemented here."))
    tag, grp = spec
    d = _biv_residual_designs(f, fixed, data)
    all(d.obs1) && all(d.obs2) || throw(ArgumentError(
        "drm: the bivariate sigma random-effect route needs both responses observed on every row; " *
        "drop incomplete rows (the default `missing` policy does) or use the residual route."))
    y1, X1, nm1, y2, X2, nm2 = Vector{Float64}(d.y1), d.X1, d.nm1, Vector{Float64}(d.y2), d.X2, d.nm2
    Xs1, nms1, Xs2, nms2, Xr, nmr = d.Xs1, d.nms1, d.Xs2, d.nms2, d.Xr, d.nmr
    n, offs = d.n, d.offs
    p1, p2 = size(X1, 2), size(X2, 2)
    pβ = p1 + p2
    nfix = offs[end]
    rng(k) = (offs[k]+1):offs[k+1]
    gidx, G = _group_index(getproperty(data, grp); col = grp)
    G >= 2 || throw(ArgumentError("drm: grouping factor `$grp` needs at least 2 levels"))
    rows_of = [findall(==(k), gidx) for k in 1:G]

    # Laplace objective at θ = (b1, b2, c1, c2, rho coefs, lc[3]). Returns (value, modes), the modes
    # being the Float64 z_hat per group (`nothing` when a group failed and the value is Inf).
    zstart = [zeros(2) for _ in 1:G]
    function nll_parts(θ)
        T = eltype(θ)
        r1 = y1 .- X1 * θ[rng(1)]
        r2 = y2 .- X2 * θ[rng(2)]
        ls1 = Xs1 * θ[rng(3)]
        ls2 = Xs2 * θ[rng(4)]
        ηr = Xr * θ[rng(5)]
        ρ = RHO_GUARD .* tanh.(ηr)
        om = 1 .- ρ .* ρ
        L = Matrix(lc_to_chol(θ[(nfix+1):(nfix+3)], 2).L)
        r1v, r2v, ls1v, ls2v, ρv, omv, Lv = _sre_fv.(r1), _sre_fv.(r2), _sre_fv.(ls1), _sre_fv.(ls2),
                                            _sre_fv.(ρ), _sre_fv.(om), _sre_fv.(L)
        tot = zero(T)
        zs = Vector{Vector{Float64}}(undef, G)
        for (k, rows) in enumerate(rows_of)
            fv = z -> _sre_fgh(z, Lv, rows, r1v, r2v, ls1v, ls2v, ρv, omv)
            zh, ok = _sre_mode(fv, zstart[k])
            if !ok
                return oftype(tot, Inf), nothing
            end
            zs[k] = zh
            fT = z -> _sre_fgh(z, L, rows, r1, r2, ls1, ls2, ρ, om)
            zT = Vector{T}(zh)
            if T !== Float64
                for _ in 1:3
                    _, gT, HT = fT(zT)
                    zT = zT .- HT \ gT
                end
            end
            fval, _, HT = fT(zT)
            ch = cholesky(Symmetric(Matrix(HT)); check = false)
            if !issuccess(ch)
                return oftype(tot, Inf), nothing
            end
            tot += fval + logdet(ch) / 2
        end
        return tot, zs
    end
    function nll(θ)
        v, zs = nll_parts(θ)
        isfinite(v) || return oftype(v, Inf)
        if zs !== nothing && eltype(θ) === Float64
            for k in 1:G; zstart[k] = zs[k]; end      # warm start the next evaluation
        end
        return v
    end

    β1 = X1 \ y1; β2 = X2 \ y2
    θ0 = zeros(nfix + 3)
    θ0[1:p1] = β1; θ0[(p1+1):pβ] = β2
    θ0[offs[3]+1] = _seed_ls(y1 - X1 * β1, y1)
    θ0[offs[4]+1] = _seed_ls(y2 - X2 * β2, y2)
    # Three starts for the group covariance: independent, and strongly correlated of either sign.
    # The likelihood in Sigma_g is flat and, on small or weakly informative data, has its best mode on
    # the boundary of a group correlation of +-1 (drmTMB's own fit does on the gap-map cell: SDs 0.048
    # and 0.030, correlation 1, log-likelihood -377.3387). A start with a modest correlation slides to
    # Sigma_g = 0 instead (-377.3505 there), a worse local optimum; the strongly correlated starts
    # reach the boundary mode. The best finished start is kept.
    best = nothing
    for (sd0, cr0) in ((0.2, 0.0), (0.1, 0.99), (0.1, -0.99))
        θs = copy(θ0)
        θs[(nfix+1):(nfix+3)] .= cov_to_lc(Matrix(Symmetric(sd0^2 .* [1.0 cr0; cr0 1.0])))
        r = Optim.optimize(nll, θs, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
        (best === nothing || Optim.minimum(r) < Optim.minimum(best)) && (best = r)
    end
    res = best
    θ̂ = Optim.minimizer(res)
    ll = -nll(θ̂)
    isfinite(ll) || throw(ArgumentError("drm: the bivariate sigma random-effect fit did not reach a finite likelihood."))
    V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂);
                           context = "bivariate Gaussian sigma random-effect (Laplace)")

    θb = θ̂[1:pβ]
    L = Matrix(lc_to_chol(θ̂[(nfix+1):(nfix+3)], 2).L)
    _, zs = nll_parts(θ̂)
    U = hcat((L * zs[k] for k in 1:G)...)
    ls1 = Xs1 * θ̂[rng(3)]; ls2 = Xs2 * θ̂[rng(4)]
    ρv = RHO_GUARD .* tanh.(Xr * θ̂[rng(5)])
    blocks = [:mu1 => rng(1), :mu2 => rng(2), :sigma1 => rng(3), :sigma2 => rng(4),
              :rho12 => rng(5), _BIV_RE_GROUP_BLOCK => (nfix+1):(nfix+3)]
    names = [:mu1 => nm1, :mu2 => nm2, :sigma1 => nms1, :sigma2 => nms2, :rho12 => nmr,
             _BIV_RE_GROUP_BLOCK => _biv_re_group_names()]
    means = Dict(:mu1 => X1 * θb[1:p1], :mu2 => X2 * θb[(p1+1):pβ])
    obs = Dict(:mu1 => y1, :mu2 => y2)
    scales = Dict(:sigma1 => exp.(ls1), :sigma2 => exp.(ls2), :rho12 => ρv)
    fit = DrmFit(fam, blocks, names, θ̂, V, ll, n, Optim.converged(res), means, obs, scales)
    re = (; effects = Dict(Symbol(grp) => U), Sigma_g = L * L', axes = (:sigma1, :sigma2),
          kind = :ordinary_sigma, tag = tag, group = grp, group_index = gidx)
    return _withiterations(_withranef(_withformula(_withnll(fit, nll), f), re), Optim.iterations(res))
end

# `simulate(fit; data)` and the parametric bootstrap: per group a fresh u_g ~ N(0, Sigma_g) shifts
# log sigma1 and log sigma2 of every row of the group, then the bivariate residual pair is drawn.
# Same refusals as the mean route: the table must be the fitted one, no row dropped.
_is_biv_sigma_re_fit(fit) =
    fit.formula isa BivariateDrmFormula && fit.ranef isa NamedTuple &&
    get(fit.ranef, :kind, nothing) === :ordinary_sigma && haskey(fit.ranef, :Sigma_g)

function _biv_sigma_re_simulator(fit::DrmFit, data)
    re = fit.ranef
    grp = Symbol(re.group)
    (data !== nothing && hasproperty(data, grp)) || throw(ArgumentError(
        "simulate/bootstrap: this bivariate sigma random-effect fit needs the fitting `data` " *
        "(with the grouping column `$grp`) to redraw the group effects."))
    col = getproperty(data, grp)
    mu1, mu2 = fit.means[:mu1], fit.means[:mu2]
    n = length(mu1)
    length(col) == n || throw(ArgumentError(
        "simulate/bootstrap: `data` has $(length(col)) rows but the fit used $n (rows with missing " *
        "values are dropped at fit time). Pass the complete table the model was fitted to, or drop " *
        "the incomplete rows first, so the group effects can be redrawn row for row."))
    gidx, G = _group_index(col; col = grp)
    gidx == re.group_index || throw(ArgumentError(
        "simulate/bootstrap: the grouping column `$grp` of `data` does not match the fitted groups."))
    σ1, σ2, ρ = fit.scales[:sigma1], fit.scales[:sigma2], fit.scales[:rho12]
    F = eigen(Symmetric(Matrix{Float64}(re.Sigma_g)))
    Lg = F.vectors * Diagonal(sqrt.(max.(F.values, 0.0)))
    return function (rng)
        u = Lg * randn(rng, 2, G)
        s1 = σ1 .* exp.(u[1, gidx]); s2 = σ2 .* exp.(u[2, gidx])
        z1 = randn(rng, n); z2 = randn(rng, n)
        return Dict(:mu1 => mu1 .+ s1 .* z1,
                    :mu2 => mu2 .+ s2 .* (ρ .* z1 .+ sqrt.(1 .- ρ .^ 2) .* z2))
    end
end
