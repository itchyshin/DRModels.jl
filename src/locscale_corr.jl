# locscale_corr.jl — route the per-family CORRELATED random intercept+slope
# `(1 + x | g)` and the INDEPENDENT slope `(0 + x | g)` onto the unified q=2
# location–scale Laplace core (cluster 1, #202). Promoting the bespoke dense
# K×K Gauss–Hermite `_fit_*_corr_ranef` fitters to the augmented-state Laplace
# core gives the exact O(p) gradient and (where requested) profile/Wald CIs.
#
# The mechanism is the Z_lat generalisation of `locscale_inner.jl`/`locscale_
# grad.jl`: BOTH latent axes per group load the MEAN predictor η with the
# per-obs loading row [1, xᵢ] (the intercept and slope), while the scale
# predictor ψ carries NO latent. So
#   Zη = [1 xᵢ]   (intercept + slope on η),   Zψ = [0 0]   (scale fixed-only).
# The independent slope `(0 + x | g)` is q=1: only the slope axis loads η, with
# Zη = [xᵢ 0] and the intercept axis pinned by a vanishing prior variance.
#
# The 2×2 group covariance Λ the core returns IS Σ_re of the (intercept, slope)
# axes; we map its Cholesky back to the existing `:recov` `[L11, L22, L21]`
# names so `coef`/`summary`/`vc` output is byte-identical to the GHQ fitters.
#
# Which families route here is decided by the family frontends after the
# cross-engine equivalence gate (`test/test_corr_locscale_equiv.jl`): a family
# whose unified-Laplace logLik tracks its GHQ logLik on the shared fixture is
# flipped; one that does not stays on GHQ.

# Family → engine `kind` Val for the q2 leaf kernels.
_corr_kind(::Poisson) = Val(:poisson)
_corr_kind(::Gamma) = Val(:gamma)
_corr_kind(::Beta) = Val(:beta)
_corr_kind(::NegBinomial2) = Val(:nb2)
_corr_kind(::LogNormal) = Val(:lognormal)

# Map the engine's 2×2 covariance Λ (intercept/slope axes) and the engine vcov
# (packed [βμ; βψ; λ] with λ = [logL11, L21, logL22]) into a `DrmFit` whose
# `:recov` block is the GHQ convention [log L11, log L22, L21] (so `vc(fit)`
# rebuilds Σ = L Lᵀ). `engresp` is the response the kernels consumed (raw y,
# log-y for LogNormal handled inside the kernel, or `(s,n)` tuples); `respobs`
# overrides the stored response (observed proportion for the bounded families).
function _corr_build_drmfit(kind, fam, fitres, Xμ, Xψ, nmμ, nmσ, grp::String,
                            y, link::Symbol; respobs = nothing, trials = nothing,
                            extra_blocks = nothing)
    pμ = size(Xμ, 2); pψ = size(Xψ, 2); n = size(Xμ, 1)
    βμ = fitres.beta_mu; βψ = fitres.beta_psi
    λ = fitres.θ[pμ+pψ+1:pμ+pψ+3]                         # [logL11, L21, logL22]
    theta = vcat(βμ, βψ, λ[1], λ[3], λ[2])                # → recov order
    perm = vcat(collect(1:(pμ+pψ)), [pμ+pψ+1, pμ+pψ+3, pμ+pψ+2])
    V = fitres.vcov === nothing ? fill(NaN, length(theta), length(theta)) :
        fitres.vcov[perm, perm]
    blocks = Pair{Symbol,UnitRange{Int}}[:mu => 1:pμ]
    names = Pair{Symbol,Vector{String}}[:mu => nmμ]
    if pψ > 0
        push!(blocks, :sigma => (pμ+1):(pμ+pψ)); push!(names, :sigma => nmσ)
    end
    push!(blocks, :recov => (pμ+pψ+1):(pμ+pψ+3))
    push!(names, :recov => ["$grp:L11", "$grp:L22", "$grp:L21"])
    means = Dict(:mu => link === :logit ? _logistic.(Xμ * βμ) :
                        link === :identity ? Xμ * βμ : exp.(Xμ * βμ))
    obs = Dict(:mu => respobs === nothing ? Float64.(y) : Float64.(respobs))
    scales = pψ > 0 ? Dict(:sigma => exp.(Xψ * βψ)) : Dict{Symbol,Vector{Float64}}()
    trials === nothing || (scales[:trials] = Float64.(trials))
    return DrmFit(fam, blocks, names, theta, V, -fitres.nll, n, fitres.converged,
                  means, obs, scales)
end

# Latent loadings for the correlated / independent slope on the mean axis.
#   :corr  → Zη = [1 xᵢ] (intercept + slope),  Zψ = [0 0]
#   :slope → Zη = [xᵢ 0] (slope only),          Zψ = [0 0]
function _corr_loadings(rk::Symbol, xs)
    n = length(xs)
    Zψ = zeros(n, 2)
    if rk === :corr
        Zη = hcat(ones(n), xs)
    else  # :slope (independent) — single loading column, intercept axis unused
        Zη = hcat(xs, zeros(n))
    end
    return Zη, Zψ
end

# Initial λ (log-Cholesky [logL11, L21, logL22]) for the group covariance.
# For the independent slope `(0 + x | g)` the SLOPE loads axis-1 (Zη = [xᵢ 0]),
# so axis-1 is the identified variance and axis-2 is genuinely unused (its loading
# column is zero). The start therefore seeds axis-1 (logL11) at a sensible slope
# variance and pins axis-2 (logL22) to a tiny variance so Λ stays PD while
# contributing nothing. (The `:slope` fitter also FIXES axis-2 at ε and L21 = 0
# rather than leaving it free — see `_fit_slope_axis_re`.)
_corr_λ0(rk::Symbol) = rk === :corr ? [log(0.4), 0.0, log(0.4)] :
                                      [log(0.4), 0.0, log(1e-3)]

"""
    _fit_corr_locscale(fam, kind, rk, y, Xμ, Xψ, xs, gidx, G, nmμ, nmσ, grp;
                       link, se, g_tol, respobs, trials, Q)

Fit a correlated (`rk == :corr`) or independent (`rk == :slope`) random
intercept/slope on the mean of a non-Gaussian family through the unified q2
location–scale Laplace core, and wrap it as a `DrmFit` with the GHQ `:recov`
block. `Xψ` is the scale fixed-effect design (`zeros(n,0)` for the mean-only
Poisson). `link` is the mean inverse link (`:log`/`:logit`/`:identity`).
`Q` is the G×G group-level precision; default `sparse(I,G,G)` (i.i.d. groups,
cluster 1 byte-identical behaviour). A structured Q (from `_general_cov_setup`
or `_locscale_phylo_setup`) routes the kron(Q,Λ⁻¹) prior through the same
spine (cluster 3: structured non-Gaussian random slopes).
"""
function _fit_corr_locscale(fam, kind, rk::Symbol, y, Xμ, Xψ, xs, gidx, G,
                            nmμ, nmσ, grp::String; link::Symbol,
                            se::Bool = true, g_tol::Real = 1e-6,
                            respobs = nothing, trials = nothing,
                            Q = sparse(1.0 * I, G, G))
    if rk === :slope
        # Independent slope `(0 + x | g)`: only axis-1 loads η (Zη = [xᵢ 0]); the
        # unused axis-2 variance is unidentified, so FIX it at ε and optimise only
        # the identified slope variance (mirrors `_fit_sigma_axis_re`'s discipline).
        fitres = _fit_slope_axis_re(kind, y, Xμ, Xψ, xs, gidx, G, Q; g_tol = g_tol, se = se)
    else
        Zη, Zψ = _corr_loadings(rk, xs)
        fitres = _fit_locscale(kind, y, Xμ, Xψ, gidx, G, Q;
                               Zη = Zη, Zψ = Zψ, λ0 = _corr_λ0(rk),
                               g_tol = g_tol, se = se)
    end
    return _corr_build_drmfit(kind, fam, fitres, Xμ, Xψ, nmμ, nmσ, grp, y, link;
                              respobs = respobs, trials = trials)
end

# Pinned fitter for the INDEPENDENT slope `(0 + x | g)`. The slope loads axis-1
# (Zη = [xᵢ 0], Zψ = [0 0]); axis-2 carries no loading, so its variance is not
# identified. We therefore FIX Λ = diag(L11², ε²) with L21 = 0 (reusing the
# σ-axis machinery `_sigma_re_Lambda`/`_sigma_re_grad`, which pin axis-2 at ε and
# L21 = 0) and optimise only θ = [βμ; βψ; logL11] via the exact O(p) gradient.
# The identified SLOPE variance is Λ[1,1] = L11². Returns a named tuple in the
# same 5-packing `[βμ; βψ; logL11, L21, logL22]` that `_corr_build_drmfit`
# consumes: L21 and logL22 are held at their pinned values (0 and log ε), with
# zero reported uncertainty (the vcov rows/cols for the pinned entries are 0).
function _fit_slope_axis_re(kind, y, Xμ, Xψ, xs, gidx, G, Q;
                            g_tol::Real = 1e-6, se::Bool = true)
    n = length(y); pμ = size(Xμ, 2); pψ = size(Xψ, 2)
    Zη = hcat(Float64.(xs), zeros(n))   # slope on axis-1, axis-2 unused
    Zψ = zeros(n, 2)

    warm = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    function obj(θ)
        βμ = @view θ[1:pμ]; βψ = @view θ[pμ+1:pμ+pψ]
        Λ = _sigma_re_Lambda(θ[pμ+pψ+1])       # diag(L11², ε²), L21 = 0
        P = prior_precision(Q, _ls_inv2x2(Λ))
        val, a, ok = _ls_marginal_nll(kind, y, Xμ * βμ, Xψ * βψ, gidx, G, P, Zη, Zψ; a0 = warm[])
        ok && (warm[] = copy(a))
        return ok ? val : 1e18
    end
    grad!(g, θ) = (g .= _sigma_re_grad(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ; warm = warm); g)
    # Reduced observed information over [βμ; βψ; logL11]. The full engine Hessian
    # `_ls_obs_information` is (pμ+pψ+3)² over [βμ; βψ; logL11, L21, logL22]; the
    # pinned coordinates (L21, logL22) are dropped, giving a well-conditioned
    # sub-block for the NewtonTrustRegion outer step (LBFGS alone stalls here
    # because the βψ intercept and logL11 gradients differ by ~10²× at the start).
    ired = vcat(collect(1:(pμ+pψ)), pμ+pψ+1)
    function hred(θr)
        θf = vcat(θr[1:pμ+pψ], θr[pμ+pψ+1], 0.0, log(_SIGMA_RE_EPS))
        Hf = _ls_obs_information(kind, y, Xμ, Xψ, gidx, G, Q, θf, Zη, Zψ)
        return Matrix(Hf)[ired, ired]
    end
    h!(H, θ) = (H .= hred(θ); H)

    βμ0 = _ls_default_betastart(kind, y, Xμ)
    βψ0 = zeros(pψ)
    θ0 = vcat(βμ0, βψ0, log(0.4))            # seed logL11 (identified slope axis)

    opts = Optim.Options(g_tol = g_tol, iterations = 1000)
    nm() = (warm[] = nothing; Optim.optimize(obj, θ0, Optim.NelderMead(),
                                             Optim.Options(iterations = 3000)))
    res = try
        Optim.optimize(obj, grad!, h!, θ0, Optim.NewtonTrustRegion(), opts)
    catch err
        err isa InterruptException && rethrow(err)
        warm[] = nothing
        try
            Optim.optimize(obj, grad!, θ0, Optim.LBFGS(), opts)
        catch err2
            err2 isa InterruptException && rethrow(err2)
            nm()
        end
    end
    θ̂r = Optim.minimizer(res)
    if any(!isfinite, θ̂r) || !(obj(θ̂r) < 1e17)
        res = nm()
        θ̂r = Optim.minimizer(res)
    end
    nll_val = obj(θ̂r)

    # Reduced vcov over [βμ; βψ; logL11] via FD Hessian of the exact gradient.
    npr = length(θ̂r)
    Vr = if se
        try
            h = 1e-4
            H = zeros(npr, npr)
            for j in 1:npr
                tp = copy(θ̂r); tp[j] += h
                tm = copy(θ̂r); tm[j] -= h
                gp = _sigma_re_grad(kind, y, Xμ, Xψ, gidx, G, Q, tp, Zη, Zψ)
                gm = _sigma_re_grad(kind, y, Xμ, Xψ, gidx, G, Q, tm, Zη, Zψ)
                H[:, j] .= (gp .- gm) ./ (2h)
            end
            Matrix(inv(Symmetric(H)))
        catch
            fill(NaN, npr, npr)
        end
    else
        nothing
    end

    # Embed the reduced estimate into the 5-packing [βμ; βψ; logL11, L21, logL22]
    # with the pinned entries L21 = 0 and logL22 = log ε. Pad the vcov to the full
    # 5-param layout with zero uncertainty on the two pinned coordinates.
    logL11 = θ̂r[pμ+pψ+1]
    θ_full = vcat(θ̂r[1:pμ+pψ], logL11, 0.0, log(_SIGMA_RE_EPS))
    npf = pμ + pψ + 3
    Vfull = if Vr === nothing
        nothing
    else
        Vf = zeros(npf, npf)
        idx = vcat(collect(1:(pμ+pψ)), pμ+pψ+1)          # [βμ; βψ; logL11]
        Vf[idx, idx] .= Vr
        Vf
    end
    Λ̂ = _sigma_re_Lambda(logL11)
    return (θ = θ_full,
            beta_mu = θ_full[1:pμ],
            beta_psi = θ_full[pμ+1:pμ+pψ],
            Lambda = Λ̂,
            components = _ls_components(Λ̂),
            vcov = Vfull,
            se = _ls_se(Vfull),
            nll = nll_val,
            converged = drm_optim_converged(res))
end

# Parse a formula rhs for a structured correlated slope:
#   `relmat(1 + x | g)`, `animal(1 + x | g)`, `phylo(1 + x | g)`,
#   `spatial(1 + x | g)`.
# Returns `(struct_kind::Symbol, slope_var::Symbol, grp_sym::Symbol)` or
# `nothing`. Does NOT consume ordinary `(1 + x | g)` terms (those are handled
# by the `re` path).
function _parse_structured_slope(rhs)
    terms = rhs isa Tuple ? collect(rhs) : Any[rhs]
    for t in terms
        t isa FunctionTerm || continue
        f = t.f
        (f === relmat || f === animal || f === phylo || f === spatial) || continue
        length(t.args) == 1 || continue
        inner = t.args[1]                              # should be `(1 + x | g)`
        inner isa FunctionTerm && inner.f === (|) || continue
        lhs = inner.args[1]
        grp_term = inner.args[2]
        grp_term isa Term || continue
        try
            rk, var = _re_kind(lhs)
            rk === :corr || continue                   # only correlated slope
            struct_kind = f === relmat  ? :relmat  :
                          f === animal  ? :animal  :
                          f === phylo   ? :phylo   : :spatial
            return (struct_kind, var, grp_term.sym)
        catch
            continue
        end
    end
    return nothing
end
