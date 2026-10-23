# locscale_sigma.jl — standalone σ-axis random intercept `sigma ~ 1 + (1|g)`
# for non-Gaussian families via the unified q=2 location–scale Laplace core
# (cluster 2, #XXX).
#
# DGP:  log σ_i = ψ0_i + u_{g(i)},   u_g ~ N(0, τ²),   mean fixed (no mean RE).
#
# The q=2 spine is SYMMETRIC in the mean (η) and scale (ψ) axes, so we can put
# the latent ENTIRELY on the scale axis:
#
#   Zη = zeros(n, 2)   — neither latent axis loads the mean predictor
#   Zψ = [1  0]        — latent axis-1 loads the scale predictor (σ intercept)
#                        axis-2 is unused, with variance fixed at ε ≈ 0
#
# IMPLEMENTATION NOTE: The full 2×2 Λ would have Λ[2,2] → 0 at the optimum
# (axis-2 is unused). Optimizing over logL22 is ill-conditioned (unbounded below).
# We therefore FIX Λ = diag(L11², ε²) with ε = 1e-6 and only optimize over
#   θ = [βμ; βψ; logL11],
# building P = prior_precision(Q, Λ⁻¹) with the fixed-ε Λ. The exact O(p)
# gradient wrt logL11 is recovered from the general Λ-gradient formula with
# L21=0 and dΛ/d(logL11) = diag(2L11², 0).
#
# We report τ = exp(logL11) as `:resd` so `re_sd(fit)[:g]` = τ.

using SparseArrays: sparse
using LinearAlgebra: I, Symmetric, cholesky, issuccess, logdet, norm
import Optim

const _SIGMA_RE_EPS = 1e-6   # fixed tiny variance for the unused latent axis

# Latent loadings for a STANDALONE σ-axis random intercept.
#   Zη = zeros(n, 2)           — mean axis: no latent contribution
#   Zψ = [1  0] for every row  — σ intercept on axis-1; axis-2 unused
function _sigma_re_loadings(n::Int)
    Zη = zeros(n, 2)
    Zψ = zeros(n, 2)
    @views Zψ[:, 1] .= 1.0
    return Zη, Zψ
end

# Build the 2×2 group covariance Λ = diag(L11², ε²) from logL11 (the single
# free variance parameter). L21 is always 0; axis-2 is pinned to ε.
function _sigma_re_Lambda(logL11::Real)
    L11 = exp(logL11)
    return [L11^2 0.0; 0.0 _SIGMA_RE_EPS^2]
end

# Marginal NLL at θ = [βμ; βψ; logL11] for the σ-axis RE.
# P = prior_precision(Q, Λ⁻¹) with Λ = diag(L11², ε²).
# `warm` is an optional Ref{Union{Nothing,Vector{Float64}}} for inner-mode warmstarting
# across successive evaluations (pass the same Ref for FD consistency).
function _sigma_re_nll(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ;
                       warm::Union{Nothing,Ref{Union{Nothing,Vector{Float64}}}} = nothing)
    pμ = size(Xμ, 2); pψ = size(Xψ, 2)
    βμ = @view θ[1:pμ]
    βψ = @view θ[pμ+1:pμ+pψ]
    logL11 = θ[pμ+pψ+1]
    Λ = _sigma_re_Lambda(logL11)
    Λinv = _ls_inv2x2(Λ)
    P = prior_precision(Q, Λinv)
    a0 = warm === nothing ? nothing : warm[]
    val, a, ok = _ls_marginal_nll(kind, y, Xμ * βμ, Xψ * βψ, gidx, G, P, Zη, Zψ; a0 = a0)
    warm !== nothing && ok && (warm[] = copy(a))
    return ok ? val : 1e18
end

# Exact gradient of the σ-axis marginal NLL wrt θ = [βμ; βψ; logL11].
# The βμ, βψ blocks come from the general _ls_marginal_grad logic; the logL11
# block uses dΛ/d(logL11) = diag(2L11², 0) and dΛ⁻¹/d(logL11) = diag(-2/L11², 0)
# (since Λ⁻¹ = diag(1/L11², 1/ε²) and d(1/L11²)/d(logL11) = -2/L11²).
function _sigma_re_grad(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ;
                        warm::Union{Nothing,Ref{Union{Nothing,Vector{Float64}}}} = nothing)
    pμ = size(Xμ, 2); pψ = size(Xψ, 2)
    βμ = @view θ[1:pμ]
    βψ = @view θ[pμ+1:pμ+pψ]
    logL11 = θ[pμ+pψ+1]
    # Call the general gradient with the full 5-elem packing [βμ; βψ; logL11, 0, log(ε)],
    # then extract only the [βμ; βψ; logL11] components.
    θ_full = vcat(βμ, βψ, logL11, 0.0, log(_SIGMA_RE_EPS))
    a0 = warm === nothing ? nothing : warm[]
    g_full = _ls_marginal_grad(kind, y, Xμ, Xψ, gidx, G, Q, θ_full, Zη, Zψ; a0 = a0)
    # g_full is [βμ(pμ); βψ(pψ); logL11(1); L21(1); logL22(1)].
    # Extract the components we optimize over; drop L21 and logL22 (fixed).
    grad = zeros(pμ + pψ + 1)
    grad[1:pμ+pψ] .= g_full[1:pμ+pψ]
    grad[pμ+pψ+1] = g_full[pμ+pψ+1]
    return grad
end

# Build a DrmFit from the σ-axis fit, emitting a `:resd` block with
# (log τ, i.e. logL11). The RE-SD name is tagged `<grp>_logsigma`: this random
# intercept lives on the log-σ (scale) axis, so `re_sd(fit)[Symbol(grp*"_logsigma")]`
# = exp(logL11) = τ, kept distinct from a mean-axis (response-scale) SD (#322).
function _sigma_re_build_drmfit(kind, fam, θ̂, V, nll_val, n, converged,
                                 y, Xμ, Xψ, nmμ, nmσ, grp::String;
                                 obs_prop = nothing, trials = nothing)
    pμ = size(Xμ, 2); pψ = size(Xψ, 2)
    βμ = θ̂[1:pμ]; βψ = θ̂[pμ+1:pμ+pψ]
    blocks = Pair{Symbol,UnitRange{Int}}[:mu => 1:pμ]
    names  = Pair{Symbol,Vector{String}}[:mu => nmμ]
    if pψ > 0
        push!(blocks, :sigma => (pμ+1):(pμ+pψ))
        push!(names,  :sigma => nmσ)
    end
    push!(blocks, :resd => (pμ+pψ+1):(pμ+pψ+1))
    push!(names,  :resd => ["$(grp)_logsigma"])
    logit_link = kind isa Val{:beta} || kind isa Val{:betabinomial}
    means  = Dict(:mu => logit_link ? _logistic.(Xμ * βμ) : exp.(Xμ * βμ))
    obs    = Dict(:mu => obs_prop === nothing ? Float64.(y) : Float64.(obs_prop))
    scales = pψ > 0 ? Dict(:sigma => exp.(Xψ * βψ)) : Dict{Symbol,Vector{Float64}}()
    trials === nothing || (scales[:trials] = Float64.(trials))
    return DrmFit(fam, blocks, names, θ̂, V, -nll_val, n, converged,
                  means, obs, scales)
end

"""
    _fit_sigma_axis_re(fam, kind, y, Xμ, Xψ, gidx, G, nmμ, nmσ, grp; link, se, g_tol,
                       obs_prop, trials)

Fit a standalone σ-axis random intercept `sigma ~ 1 + (1|g)` via the q=2
location–scale Laplace core. Mean is fixed-effects only (no mean RE). The latent
is placed on the scale axis: `Zη = 0`, `Zψ[:,1] = 1`. Only `logL11` (= log τ)
is optimized; the unused latent axis-2 is pinned to a fixed tiny variance ε = 1e-6.
Returns a `DrmFit` with a `:resd` block (log τ), so `re_sd(fit)[:g_logsigma]` = τ
(the `_logsigma` suffix marks this SD as living on the log-σ scale axis).
"""
function _fit_sigma_axis_re(fam, kind, y, Xμ, Xψ, gidx, G, nmμ, nmσ, grp::String;
                             link::Symbol, se::Bool = true, g_tol::Real = 1e-6,
                             obs_prop = nothing, trials = nothing)
    n = length(y)
    pμ = size(Xμ, 2); pψ = size(Xψ, 2)
    Q   = sparse(1.0 * I, G, G)
    Zη, Zψ = _sigma_re_loadings(n)

    warm = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    function obj(θ)
        pμ_ = size(Xμ, 2); pψ_ = size(Xψ, 2)
        βμ = @view θ[1:pμ_]; βψ = @view θ[pμ_+1:pμ_+pψ_]
        logL11 = θ[pμ_+pψ_+1]
        Λ = _sigma_re_Lambda(logL11)
        Λinv = _ls_inv2x2(Λ)
        P = prior_precision(Q, Λinv)
        val, a, ok = _ls_marginal_nll(kind, y, Xμ * βμ, Xψ * βψ, gidx, G, P, Zη, Zψ; a0 = warm[])
        ok && (warm[] = copy(a))
        return ok ? val : 1e18
    end
    function grad!(g, θ)
        # pass the Ref `warm` (not the dereferenced `warm[]`): _sigma_re_grad
        # takes the `warm` keyword and dereferences it internally (line ~79).
        g .= _sigma_re_grad(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ; warm = warm)
        return g
    end

    # Starting values: fixed-effect start for βμ/βψ, log(0.3) for logL11.
    βμ0 = _ls_default_betastart(kind, y, Xμ)
    βψ0 = zeros(pψ)
    logL11_0 = log(0.3)
    θ0 = vcat(βμ0, βψ0, logL11_0)

    opts = Optim.Options(g_tol = g_tol, iterations = 1000)
    res = try
        Optim.optimize(obj, grad!, θ0, Optim.LBFGS(), opts)
    catch err
        err isa InterruptException && rethrow(err)
        warm[] = nothing
        Optim.optimize(obj, θ0, Optim.NelderMead(),
                       Optim.Options(iterations = 2000))
    end
    θ̂ = Optim.minimizer(res)
    # Feasibility guard (same pattern as _fit_locscale).
    if any(!isfinite, θ̂) || !(obj(θ̂) < 1e17)
        warm[] = nothing
        res = Optim.optimize(obj, θ0, Optim.NelderMead(), Optim.Options(iterations = 3000))
        θ̂ = Optim.minimizer(res)
    end
    nll_val = obj(θ̂)

    # Wald covariance: observed information via finite-difference Hessian of obj.
    V = if se
        try
            h = 1e-4
            np = length(θ̂)
            H = zeros(np, np)
            for j in 1:np
                tp = copy(θ̂); tp[j] += h
                tm = copy(θ̂); tm[j] -= h
                gp = _sigma_re_grad(kind, y, Xμ, Xψ, gidx, G, Q, tp, Zη, Zψ)
                gm = _sigma_re_grad(kind, y, Xμ, Xψ, gidx, G, Q, tm, Zη, Zψ)
                H[:, j] .= (gp .- gm) ./ (2h)
            end
            # Symmetrize the one-sided-column FD Hessian by AVERAGING the two
            # triangles ((H+H')/2), matching `_ls_obs_information`. Near a variance
            # boundary the raw H is asymmetric/indefinite; `Symmetric(H)` alone
            # would silently discard the lower triangle. Guard PD: any non-positive
            # diagonal in the resulting covariance becomes NaN (not a fabricated SE).
            Hs = Symmetric((H .+ H') ./ 2)
            V = inv(Hs)
            @inbounds for i in 1:np
                V[i, i] > 0 || (V[i, i] = NaN)
            end
            Matrix(V)
        catch
            fill(NaN, length(θ̂), length(θ̂))
        end
    else
        fill(NaN, length(θ̂), length(θ̂))
    end

    return _sigma_re_build_drmfit(kind, fam, θ̂, V, nll_val, n, drm_optim_converged(res),
                                   y, Xμ, Xψ, nmμ, nmσ, grp;
                                   obs_prop = obs_prop, trials = trials)
end
