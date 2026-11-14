# Random effects on `sigma` (log scale) with CONSTANT standard deviations, fitted by a dense
# Laplace approximation (drmTMB's `sigma ~ ... + (1 | g)` family of shapes; no `sd()` formula).
# One engine, several shapes, all on the standardised-latent scheme of
# `gaussian_sigma_sd_laplace.jl` (latent v ~ N(0, I), effects = loading * v):
#
#   * Gaussian / LogNormal / Gamma / NegBinomial2 / BetaBinomial with a fixed-effect mean and one
#     sigma random intercept `(1 | g)`;
#   * Gaussian sigma `(0 + x | g)` (independent slope) and `(1 + x | g)` (correlated pair,
#     one correlation parameter, drmTMB `eta_cor_sigma`);
#   * Gaussian sigma `(1 | g) + (1 | h)` (two independent intercepts);
#   * Gaussian mean `(1 | g)` with sigma `(1 | h)` on a DIFFERENT grouping factor.
#
# Model: lin_i = Xμ βμ + sum over mean-target fields, eta_i = Xσ βσ + sum over sigma-target fields,
# each field contributing  loading_i * v[group of i]. For a correlated pair on group g,
#   a_g = s1 z1_g,  b_g = s2 (r z1_g + sqrt(1 - r^2) z2_g),   r = tanh(eta),
# so eta_i gets z1_g (s1 + s2 r x_i) + z2_g s2 sqrt(1 - r^2) x_i with z standard normal. The
# Laplace value is invariant to this linear change of latent variable, so
#   nll(θ) = F(v̂) + 1/2 log det H(v̂)
# is the quantity TMB reports. Inner mode, outer loop, gradient (one Newton step under
# ForwardDiff, exact at the mode), Wald covariance: shared with the sd-formula engine.

# LogNormal: log y is Gaussian with mean `lin` and sd exp(eta); add the Jacobian log y.
@inline function _gsl_kernel(::LogNormal, y, lin, eta)
    f, fl, fe, fll, fle, fee = _gsl_kernel(Gaussian(), log(y), lin, eta)
    return (f + log(y), fl, fe, fll, fle, fee)
end

# BetaBinomial: y = (successes k, trials n); mean p = logistic(lin), precision phi = exp(-2 eta),
# alpha = phi p, beta = phi (1 - p) (drmTMB / DRModels `sigma` = 1 / sqrt(phi)).
@inline function _gsl_kernel(::BetaBinomial, yt::Tuple, lin, eta)
    k, n = yt
    phi = exp(-2 * eta)
    p = _logistic(lin)
    q = p * (1 - p)
    al = phi * p
    be = phi * (1 - p)
    ll = _logchoose(n, k) + loggamma(k + al) + loggamma(n - k + be) - loggamma(n + phi) -
         loggamma(al) - loggamma(be) + loggamma(phi)
    pa = digamma(k + al) - digamma(al)
    pb = digamma(n - k + be) - digamma(be)
    ta = trigamma(k + al) - trigamma(al)
    tb = trigamma(n - k + be) - trigamma(be)
    A = pa - pb
    B = ta + tb
    C = p * pa + (1 - p) * pb - digamma(n + phi) + digamma(phi)
    Cphi = p^2 * ta + (1 - p)^2 * tb - trigamma(n + phi) + trigamma(phi)
    D = p * ta - (1 - p) * tb
    ll_l = phi * q * A
    ll_ll = phi * q * (1 - 2p) * A + (phi * q)^2 * B
    ll_e = -2 * phi * C
    ll_ee = 4 * phi * C + 4 * phi^2 * Cphi
    ll_le = -2 * phi * q * A - 2 * phi^2 * q * D
    return (-ll, -ll_l, -ll_e, -ll_ll, -ll_le, -ll_ee)
end

_gsl_means(::Gaussian, lin) = lin
_gsl_means(::BetaBinomial, lin) = _logistic.(lin)
_gsl_means(::Union{NegBinomial2,Gamma,LogNormal}, lin) = exp.(lin)

# Joint negative log density in v (value, gradient, dense Hessian) over `F` fields per row:
# IDX[i, s] is the latent index of field s for row i, LD[i, s] its loading, TG[s] its target
# (1 = mean linear predictor, 2 = log sigma).
function _gsl_blocks_joint(fam, y, mu0, eta0, IDX, LD, TG, v, m::Int; deriv::Bool)
    T = promote_type(eltype(mu0), eltype(eta0), eltype(LD), eltype(v))
    n = length(y)
    nf = size(IDX, 2)
    val = zero(T)
    g = deriv ? zeros(T, m) : nothing
    H = deriv ? zeros(T, m, m) : nothing
    @inbounds for i in 1:n
        lin = zero(T) + mu0[i]
        eta = zero(T) + eta0[i]
        for s in 1:nf
            u = LD[i, s] * v[IDX[i, s]]
            TG[s] == 1 ? (lin += u) : (eta += u)
        end
        f, fl, fe, fll, fle, fee = _gsl_kernel(fam, y[i], lin, eta)
        val += f
        deriv || continue
        for s in 1:nf
            a = IDX[i, s]; ds = LD[i, s]
            g[a] += ds * (TG[s] == 1 ? fl : fe)
            for r in 1:nf
                h = TG[s] == 1 ? (TG[r] == 1 ? fll : fle) : (TG[r] == 1 ? fle : fee)
                H[a, IDX[i, r]] += ds * LD[i, r] * h
            end
        end
    end
    val += 0.5 * sum(abs2, v)
    if deriv
        g .+= v
        @inbounds for k in 1:m
            H[k, k] += 1
        end
    end
    return val, g, H
end

# `blocks`: vector of NamedTuples (kind = :ind | :corr, target = 1 | 2, gidx, G, x, grp, nm, xname)
# with `x` the slope covariate (ones for an intercept). `start` = (β_μ0, β_σ0, sdstart_mean, sdstart_sigma).
function _fit_sigma_blocks_laplace(fam, y, Xμ, Xσ, blocks, nmμ, nmσ, g_tol, start; se::Bool = true,
                                   what::String = "sigma random effect")
    n = length(y)
    pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    nb = length(blocks)
    # latent layout
    offs = Int[]; nfld = Int[]
    m = 0
    for b in blocks
        push!(offs, m); push!(nfld, b.kind === :corr ? 2 : 1)
        m += nfld[end] * b.G
    end
    m <= _GSL_MAX_LATENT || throw(ArgumentError(
        "drm: $what uses a dense Laplace engine limited to $(_GSL_MAX_LATENT) latent coordinates (got $m)."))
    nf = sum(nfld)
    IDX = zeros(Int, n, nf); TG = zeros(Int, nf)
    let s = 0
        for (k, b) in enumerate(blocks), f in 1:nfld[k]
            s += 1
            TG[s] = b.target
            IDX[:, s] .= offs[k] + (f - 1) * b.G .+ b.gidx
        end
    end
    # parameter layout: [βμ; βσ; log SDs (one per ind block, two per corr block); corr etas]
    nsd = sum(b.kind === :corr ? 2 : 1 for b in blocks)
    ncor = count(b -> b.kind === :corr, blocks)
    isd = Int[]; icor = Int[]     # first log-SD slot and corr slot of each block (0 if none)
    let ps = pμ + pσ, pc = pμ + pσ + nsd
        for b in blocks
            push!(isd, ps + 1); ps += b.kind === :corr ? 2 : 1
            if b.kind === :corr
                pc += 1; push!(icor, pc)
            else
                push!(icor, 0)
            end
        end
    end
    ntheta = pμ + pσ + nsd + ncor

    function loadings(θ::AbstractVector{T}) where {T}
        LD = zeros(T, n, nf)
        s = 0
        for (k, b) in enumerate(blocks)
            if b.kind === :ind
                s += 1
                LD[:, s] .= b.x .* exp(θ[isd[k]])
            else
                s1 = exp(θ[isd[k]]); s2 = exp(θ[isd[k] + 1]); r = tanh(θ[icor[k]])
                s += 1
                LD[:, s] .= s1 .+ s2 * r .* b.x
                s += 1
                LD[:, s] .= s2 * sqrt(1 - r^2) .* b.x
            end
        end
        return LD
    end
    joint_at(θ) = begin
        mu0 = Xμ * θ[1:pμ]
        eta0 = Xσ * θ[(pμ+1):(pμ+pσ)]
        LD = loadings(θ)
        (vv, deriv) -> _gsl_blocks_joint(fam, y, mu0, eta0, IDX, LD, TG, vv, m; deriv = deriv)
    end

    function laplace(θ::AbstractVector{T}, vhat::Vector{Float64}) where {T}
        joint = joint_at(θ)
        v = convert(Vector{T}, vhat)
        if T <: ForwardDiff.Dual
            _, g, H = joint(vhat, true)
            Cn = cholesky(Symmetric(H); check = false)
            issuccess(Cn) || return T(REML_NONPD_PENALTY)
            v = vhat .- (Cn \ g)
        end
        F, _, H = joint(v, true)
        C = cholesky(Symmetric(H); check = false)
        issuccess(C) || return T(REML_NONPD_PENALTY)
        return F + 0.5 * logdet(C)
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
    θ0[1:pμ] .= start.βμ
    θ0[pμ+1:pμ+pσ] .= start.βσ
    for (k, b) in enumerate(blocks)
        θ0[isd[k]] = b.target == 1 ? start.sd_mean : start.sd_sigma
        b.kind === :corr && (θ0[isd[k] + 1] = start.sd_sigma)
    end

    θ̂, nllhat, conv, V, nll, grad! = _ordinary_laplace_optimize(
        fg, θ0, n, m, g_tol; se = false, context = what)
    # Same multimodality check as the sd-formula engine: a zero start that reaches a different
    # Laplace value at θ̂ means the likelihood is path dependent there; do not claim convergence.
    let joint = joint_at(θ̂)
        v0, ok0 = _gsl_mode(joint, zeros(m))
        if ok0
            _, _, H0 = joint(v0, true)
            if issuccess(cholesky(Symmetric(H0); check = false))
                nll0 = laplace(θ̂, v0)
                if isfinite(nllhat) && abs(nll0 - nllhat) > 1e-6 * max(1.0, abs(nllhat))
                    @warn "drm: $what: the random-effect mode is not unique at the estimate " *
                          "(Laplace log-likelihood $(round(-nllhat; digits = 4)) from the optimiser's " *
                          "path, $(round(-nll0; digits = 4)) from a zero start); the fit is marked not converged."
                    conv = false
                end
            end
        end
    end
    if se
        Hθ = _finite_hessian_from_gradient(grad!, θ̂; context = what)
        V = all(isfinite, Hθ) ? Matrix(_vcov_from_hessian(Hθ)) : fill(NaN, ntheta, ntheta)
    end

    # Conditional modes and fitted quantities at θ̂.
    vhat, okmode = _gsl_mode(joint_at(θ̂), zeros(m))
    if !okmode
        @warn "drm: $what: the conditional modes at the estimate did not converge from a zero start; " *
              "`ranef(fit)` may be unreliable and the fit is marked not converged."
        conv = false
    end
    mu_fixed = Xμ * θ̂[1:pμ]
    eta_fixed = Xσ * θ̂[(pμ+1):(pμ+pσ)]
    # Population-level (random effects = 0) means and scales, as every other random-effect route
    # stores them (`simulate(...; re = :zero)` and `predict` read these); the conditional modes
    # are in `ranef`.
    means = Dict(:mu => _gsl_means(fam, mu_fixed))
    obs = Dict(:mu => start.obs)
    scales = Dict{Symbol,Vector{Float64}}(:sigma => exp.(eta_fixed))
    ranef = Dict{Symbol,Any}()
    resd_names = String[]; cor_names = String[]
    for (k, b) in enumerate(blocks)
        z1 = vhat[(offs[k]+1):(offs[k]+b.G)]
        axis = b.target == 1 ? "" : "_logsigma"
        xs = b.xname === nothing ? "" : "_$(b.xname)"
        if b.kind === :ind
            raw = exp(θ̂[isd[k]]) .* z1
            ranef[Symbol(b.grp, b.xname === nothing ? "" : "_$(b.xname)", axis)] = raw
            push!(resd_names, string(b.grp, xs, axis))
        else
            z2 = vhat[(offs[k]+b.G+1):(offs[k]+2b.G)]
            s1 = exp(θ̂[isd[k]]); s2 = exp(θ̂[isd[k] + 1]); r = tanh(θ̂[icor[k]])
            ranef[Symbol(b.grp, axis)] = s1 .* z1
            ranef[Symbol(b.grp, "_", b.xname, axis)] = s2 .* (r .* z1 .+ sqrt(1 - r^2) .* z2)
            push!(resd_names, string(b.grp, axis), string(b.grp, "_", b.xname, axis))
            push!(cor_names, "cor($(b.grp)$axis, $(b.grp)_$(b.xname)$axis)")
        end
    end
    first_sig = findfirst(b -> b.target == 2 && b.kind === :ind && b.xname === nothing, blocks)
    if first_sig !== nothing
        b = blocks[first_sig]
        sdg = fill(exp(θ̂[isd[first_sig]]), b.G)
        scales[:sigma_random_effect] = exp(θ̂[isd[first_sig]]) .* vhat[(offs[first_sig]+1):(offs[first_sig]+b.G)]
        scales[:sigma_group_index] = Float64.(b.gidx)
        scales[:sd_group] = sdg
        scales[:sd] = sdg[b.gidx]
    end
    start.extra_scales !== nothing && merge!(scales, start.extra_scales)
    fblocks = Pair{Symbol,UnitRange{Int}}[:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ),
                                          :resd => (pμ+pσ+1):(pμ+pσ+nsd)]
    fnames = Pair{Symbol,Vector{String}}[:mu => nmμ, :sigma => nmσ, :resd => resd_names]
    if ncor > 0
        push!(fblocks, :cor => (pμ+pσ+nsd+1):ntheta); push!(fnames, :cor => cor_names)
    end
    fit = DrmFit(fam, fblocks, fnames, θ̂, V, -nllhat, n, conv, means, obs, scales)
    return _withranef(_withnll(fit, nll, grad!), ranef)
end

# ---------------------------------------------------------------------------------------------
# Front ends.

# Block constructors. `rl` is the random-effect left-hand side (`1`, `0 + x`, `1 + x`).
function _gsl_block(rl, grp, data, n::Int, target::Int)
    kind, sym = _re_kind(rl)
    gidx, G = _group_index(getproperty(data, grp); col = grp)
    if kind === :intercept
        return (kind = :ind, target = target, gidx = gidx, G = G, x = ones(n), grp = grp, xname = nothing)
    end
    hasproperty(data, sym) || throw(ArgumentError(
        "drm: random-slope predictor `$sym` is absent from data."))
    raw = getproperty(data, sym)
    (raw isa AbstractVector{<:Real} && length(raw) == n && all(isfinite, raw)) || throw(ArgumentError(
        "drm: random-slope predictor `$sym` must be finite numeric values, one per response."))
    return (kind = kind === :corr ? :corr : :ind, target = target, gidx = gidx, G = G,
            x = Float64.(raw), grp = grp, xname = sym)
end

# Non-Gaussian families: one sigma random intercept, fixed-effect mean. Returns `nothing` when the
# formula carries no sigma random effect (the caller continues on its normal route).
function _drm_sigma_ranef_laplace_family(f::DrmFormula, fam; data, g_tol, se, fixed_mu, re, mv, st,
                                         offset_term, isva::Bool, isaghq::Bool)
    famname = string(nameof(typeof(fam)))
    rhs = Dict(f.forms)
    sig = get(rhs, :sigma, nothing)
    sig === nothing && return nothing
    fixed_sigma, sre, mvs, sts = _split_ranef(sig)
    (isempty(sre) && mvs === nothing && sts === nothing) && return nothing
    all(p -> first(p) in (:mu, :sigma), f.forms) || return nothing
    # Any other shape keeps the family's own refusal (`only the mean formula may carry a random effect`).
    (isempty(re) && mv === nothing && st === nothing && mvs === nothing && sts === nothing &&
     length(sre) == 1 && _re_kind(sre[1][1])[1] === :intercept) || return nothing
    offset_term === nothing || throw(ArgumentError("$famname(): an offset is not supported with a random effect on sigma."))
    (isva || isaghq) && throw(ArgumentError("$famname(): a random effect on sigma is fitted by the Laplace " *
                                           "approximation only (`marginal = :LA`)."))
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "$famname(): a random effect on sigma requires observed responses (rows with a missing response are dropped before this point by the generic wrapper)."))
    y, Xμ, nmμ = _design(f.response, fixed_mu, data)
    _, Xσ, nmσ = _design(f.response, fixed_sigma, data)
    n = size(Xμ, 1)
    if fam isa BetaBinomial
        f.response2 === nothing && throw(ArgumentError(
            "BetaBinomial() needs a two-column response: bf(cbind(successes, failures) ~ …)"))
        s_ = Float64.(getproperty(data, f.response)); fl_ = Float64.(getproperty(data, f.response2))
        (all(isfinite, s_) && all(isfinite, fl_) && all(v -> v >= 0 && isinteger(v), s_) &&
         all(v -> v >= 0 && isinteger(v), fl_) && all(>(0), s_ .+ fl_)) || throw(ArgumentError(
            "BetaBinomial() requires non-negative integer successes and failures with positive trials"))
        y = hcat(s_, fl_)
    elseif fam isa NegBinomial2
        all(v -> isfinite(v) && v >= 0 && isinteger(v), y) || throw(ArgumentError(
            "NegBinomial2() requires finite, non-negative integer counts as the response"))
    else
        all(v -> isfinite(v) && v > 0, y) || throw(ArgumentError(
            "$famname() requires finite, strictly positive responses"))
    end
    blk = _gsl_block(sre[1][1], sre[1][2], data, n, 2)
    βμ0 = zeros(size(Xμ, 2)); βσ0 = zeros(size(Xσ, 2))
    if fam isa BetaBinomial
        yk = [(y[i, 1], y[i, 1] + y[i, 2]) for i in 1:n]
        pbar = clamp(sum(y[:, 1]) / sum(y[:, 1] .+ y[:, 2]), 1e-3, 1 - 1e-3)
        βμ0[1] = log(pbar / (1 - pbar)); βσ0[1] = -0.5 * log(10.0)
        obs = y[:, 1] ./ (y[:, 1] .+ y[:, 2])
        extra = Dict{Symbol,Vector{Float64}}(:trials => Float64.(y[:, 1] .+ y[:, 2]))
    else
        yk = vec(y)
        if fam isa LogNormal
            βμ0 .= Xμ \ log.(yk)
            βσ0[1] = log(std(log.(yk) .- Xμ * βμ0) + eps())
        else
            βμ0 .= Xμ \ log.(yk .+ 0.5)
            βσ0[1] = log(0.5)
        end
        obs = Vector{Float64}(yk); extra = nothing
    end
    start = (βμ = βμ0, βσ = βσ0, sd_mean = log(0.5), sd_sigma = log(0.3), obs = obs, extra_scales = extra)
    return _withformula(_fit_sigma_blocks_laplace(fam, yk, Xμ, Xσ, [blk], nmμ, nmσ, g_tol, start;
                                                  se = se, what = "$famname sigma random effect"), f)
end

# Gaussian shapes beyond the single same-group intercept pair and the single sigma intercept.
# `re` / `sigma_re` are the mean / sigma random-effect term lists. Returns the fit, or throws a
# named refusal when the combination is not one of the verified shapes.
function _drm_gaussian_sigma_ranef_blocks(f::DrmFormula, fam::Gaussian, y, Xμ, Xσ, re, sigma_re,
                                          nmμ, nmσ, data, g_tol, se::Bool)
    all(p -> first(p) in (:mu, :sigma), f.forms) || throw(ArgumentError(
        "drm (Gaussian): a random effect on `sigma` beyond the single same-group intercept supports only `mu` and " *
        "`sigma` formulas (no `sd()` formula or other distributional parameter)."))
    n = length(y)
    kinds_s = [_re_kind(t[1])[1] for t in sigma_re]
    blocks = []
    if isempty(re)
        if length(sigma_re) == 1 && kinds_s[1] in (:slope, :corr)
            push!(blocks, _gsl_block(sigma_re[1][1], sigma_re[1][2], data, n, 2))
        elseif length(sigma_re) == 2 && all(==(:intercept), kinds_s) &&
               sigma_re[1][2] !== sigma_re[2][2]
            for t in sigma_re
                push!(blocks, _gsl_block(t[1], t[2], data, n, 2))
            end
        end
    elseif length(re) == 1 && length(sigma_re) == 1 && _re_kind(re[1][1])[1] === :intercept &&
           kinds_s[1] === :intercept && re[1][2] !== sigma_re[1][2]
        push!(blocks, _gsl_block(re[1][1], re[1][2], data, n, 1))
        push!(blocks, _gsl_block(sigma_re[1][1], sigma_re[1][2], data, n, 2))
    end
    isempty(blocks) && throw(ArgumentError(
        "drm (Gaussian): the supported random-effect shapes on `sigma` are one `(1 | g)`, one " *
        "`(0 + x | g)`, one correlated `(1 + x | g)`, two intercepts `(1 | g) + (1 | h)` on different " *
        "groups, and a mean `(1 | g)` with a sigma `(1 | h)` on a different group (the same group " *
        "has its own route); got $(length(re)) mean and $(length(sigma_re)) sigma term(s) of another form."))
    βμ0 = Xμ \ y
    sd0 = std(y - Xμ * βμ0) + eps()
    βσ0 = zeros(size(Xσ, 2)); βσ0[1] = log(0.7 * sd0)
    start = (βμ = βμ0, βσ = βσ0, sd_mean = log(0.5 * sd0), sd_sigma = log(0.3),
             obs = Vector{Float64}(y), extra_scales = nothing)
    return _withformula(_fit_sigma_blocks_laplace(fam, Vector{Float64}(y), Xμ, Xσ, blocks, nmμ, nmσ, g_tol,
                                                  start; se = se, what = "Gaussian sigma random effect"), f)
end

# ---------------------------------------------------------------------------------------------
# Post-fit: the fresh-random-effect simulator (`simulate` default, `bootstrap_result`). The shapes
# are exactly those fitted above; every other formula returns `nothing` and keeps its own route.
function _sigma_blocks_shape(f::DrmFormula, fam)
    try
        rhs = Dict(f.forms)
        haskey(rhs, :sigma) && isempty(_sd_target_parts(f)) || return nothing
        all(p -> first(p) in (:mu, :sigma), f.forms) || return nothing
        _, re, mv, st = _split_ranef(rhs[:mu])
        _, sre, mvs, sts = _split_ranef(rhs[:sigma])
        (mv === nothing && st === nothing && mvs === nothing && sts === nothing && !isempty(sre)) || return nothing
        ks = [_re_kind(t[1])[1] for t in sre]
        kr = [_re_kind(t[1])[1] for t in re]
        ok = if !(fam isa Gaussian)
            isempty(re) && length(sre) == 1 && ks[1] === :intercept
        elseif isempty(re)
            (length(sre) == 1 && ks[1] in (:slope, :corr)) ||
                (length(sre) == 2 && all(==(:intercept), ks) && sre[1][2] !== sre[2][2])
        else
            length(re) == 1 && length(sre) == 1 && kr[1] === :intercept && ks[1] === :intercept &&
                re[1][2] !== sre[1][2]
        end
        return ok ? (re, sre) : nothing
    catch e
        e isa InterruptException && rethrow()
        return nothing
    end
end

function _sigma_blocks_marginal_simulator(fit::DrmFit, data)
    fit.formula isa DrmFormula || return nothing
    fam = fit.family
    fam isa Union{Gaussian,LogNormal,Gamma,NegBinomial2,BetaBinomial} || return nothing
    shape = _sigma_blocks_shape(fit.formula, fam)
    shape === nothing && return nothing
    re, sre = shape
    haskey(fit.means, :mu) && haskey(fit.scales, :sigma) || return nothing
    mu0 = Vector{Float64}(fit.means[:mu]); sg0 = Vector{Float64}(fit.scales[:sigma])
    n = length(mu0)
    for t in vcat(re, sre)
        hasproperty(data, t[2]) && length(getproperty(data, t[2])) == n || return nothing
    end
    blocks = vcat([_gsl_block(t[1], t[2], data, n, 1) for t in re],
                  [_gsl_block(t[1], t[2], data, n, 2) for t in sre])
    resd = Dict(fit.blocks)[:resd]
    sds = fit.theta[resd]
    etas = haskey(Dict(fit.blocks), :cor) ? fit.theta[Dict(fit.blocks)[:cor]] : Float64[]
    lin0 = fam isa Gaussian ? mu0 : fam isa BetaBinomial ? log.(mu0 ./ (1 .- mu0)) : log.(mu0)
    eta0 = log.(sg0)
    return function (rng)
        lin = copy(lin0); eta = copy(eta0)
        j = 0; c = 0
        for b in blocks
            if b.kind === :ind
                j += 1
                add = b.x .* (exp(sds[j]) .* randn(rng, b.G))[b.gidx]
            else
                j += 2; c += 1
                s1 = exp(sds[j-1]); s2 = exp(sds[j]); r = tanh(etas[c])
                z1 = randn(rng, b.G); z2 = randn(rng, b.G)
                add = (s1 .* z1)[b.gidx] .+ b.x .* (s2 .* (r .* z1 .+ sqrt(1 - r^2) .* z2))[b.gidx]
            end
            b.target == 1 ? (lin .+= add) : (eta .+= add)
        end
        sg = exp.(eta)
        fam isa Gaussian && return lin .+ sg .* randn(rng, n)
        fam isa LogNormal && return exp.(lin .+ sg .* randn(rng, n))
        return _simulate_once(fit, rng; mu = _mean_response(fam, lin), sigma = sg)
    end
end
