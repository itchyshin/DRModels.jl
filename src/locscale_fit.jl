# locscale_fit.jl — end-to-end fit for the non-Gaussian phylogenetic
# LOCATION–SCALE model (#202). Groundwork: not yet wired into `drm()`.
#
# Outer optimisation of the Laplace marginal (`_ls_marginal_nll`) over the
# fixed effects (βμ on the mean, βψ on the log-dispersion) and the 2×2
# group-level covariance Λ (log-Cholesky), driven by the exact O(p) outer
# gradient (`_ls_marginal_grad`) with LBFGS — fast and accurate enough for
# variance-component recovery. A derivative-free Nelder–Mead fallback guards the
# rare line-search stall on tiny / weakly-identified fixtures.
#
# The fitter is agnostic to where the group structure comes from: pass
#   Q = I_G                      → independent groups (crossed/i.i.d.), or
#   Q = root-conditioned tree precision (one latent per non-root node, data at
#       leaves) → the PHYLOGENETIC location–scale model.
# In the phylogenetic case `gidx`/`G` come from `_poisson_phylo_setup`, exactly
# as for the mean-only phylo routes.

using LinearAlgebra: cholesky, Symmetric

# Packed marginal NLL at θ = [βμ; βψ; λ(3)]. Each call solves the inner mode from
# a cold start, so the objective is deterministic — required by the optimiser.
# `Zη`/`Zψ` are the per-obs latent loadings (default = canonical mean/scale axes);
# the correlated-slope reroute passes Zη = [1 xᵢ], Zψ = [0 0].
function _ls_fit_nll(kind, y, Xμ, Xψ, gidx, G, Q, θ,
                     Zη = _ls_canonical_Zeta(length(y)),
                     Zψ = _ls_canonical_Zpsi(length(y)))
    pμ = size(Xμ, 2); pψ = size(Xψ, 2)
    βμ = @view θ[1:pμ]
    βψ = @view θ[pμ+1:pμ+pψ]
    λv = @view θ[pμ+pψ+1:pμ+pψ+3]
    P = prior_precision(Q, _ls_lc_inv2x2(λv))   # stable: never forms Λ
    val, _, ok = _ls_marginal_nll(kind, y, Xμ * βμ, Xψ * βψ, gidx, G, P, Zη, Zψ)
    return ok ? val : 1e18
end

# Callable objective that ALSO carries the structured design (kind + data +
# grouping precision). Stored in the `DrmFit.nll` slot (typed `Any`): it is a
# valid θ ↦ marginal-NLL objective in the engine's packing `[βμ; βψ; λ(3)]`, so
# generic code that merely evaluates the objective still works, while
# `confint(:profile)` / `profile_result` recognise the type and route to the
# robust trust-region profiler `_ls_profile_ci` (the generic LBFGS profiler
# throws on the variance boundary, which is exactly where profile CIs matter).
struct LocScaleObjective{K,Y,MX,PX,GI,QT}
    kind::K
    y::Y
    Xμ::MX
    Xψ::PX
    gidx::GI
    G::Int
    Q::QT
    whitened::Bool
end

# Preserve the original private constructor and its raw-precision semantics.
LocScaleObjective(kind, y, Xμ, Xψ, gidx, G, Q; whitened::Bool=false) =
    LocScaleObjective(kind, y, Xμ, Xψ, gidx, G, Q, whitened)

function (o::LocScaleObjective)(θ)
    o.whitened || return _ls_fit_nll(o.kind, o.y, o.Xμ, o.Xψ, o.gidx, o.G, o.Q, θ)
    result = _ls_whitened_eval(o.kind, o.y, o.Xμ, o.Xψ, o.gidx, o.G, o.Q, θ,
        _ls_canonical_Zeta(length(o.y)), _ls_canonical_Zpsi(length(o.y)); gradient=false)
    return result.status.ok ? result.value : 1e18
end

function _ls_objective_gradient(o::LocScaleObjective, θ)
    o.whitened || return _ls_marginal_grad(o.kind, o.y, o.Xμ, o.Xψ, o.gidx, o.G, o.Q, θ)
    return _ls_whitened_eval(o.kind, o.y, o.Xμ, o.Xψ, o.gidx, o.G, o.Q, θ,
        _ls_canonical_Zeta(length(o.y)), _ls_canonical_Zpsi(length(o.y))).gradient
end

# Family-appropriate mean-axis intercept start. NB2/Gamma use a log link, so the
# Poisson IRLS warm start applies directly. Beta/BetaBinomial use a logit link
# and a non-scalar response (BetaBinomial packs `(successes, trials)` per obs),
# for which `_poisson_fixed_start` is neither correct nor type-valid — start the
# intercept at the logit of the overall mean proportion instead.
function _ls_default_betastart(kind, y, Xμ)
    pμ = size(Xμ, 2)
    if kind isa Val{:beta}
        ȳ = clamp(sum(y) / length(y), 1e-3, 1 - 1e-3)
        β = zeros(pμ); β[1] = log(ȳ / (1 - ȳ)); return β
    elseif kind isa Val{:betabinomial}
        s = sum(t -> t[1], y); ntot = sum(t -> t[2], y)
        p̄ = clamp(s / max(ntot, 1), 1e-3, 1 - 1e-3)
        β = zeros(pμ); β[1] = log(p̄ / (1 - p̄)); return β
    else
        return _poisson_fixed_start(y, Xμ)
    end
end

# Stall stop for one certified-refinement run. Those runs disable Optim's
# x/f tolerances so that only the gradient criterion can mark them converged,
# which also removed every way to stop a run that has stopped moving. On a
# variance component sliding to its zero boundary the objective reaches its
# Float64 floor: measured 2026-10-03 on the Gamma phylo bootstrap fixture, one
# BFGS run sat at a bit-identical value from iteration 50 to 2000 (22k
# objective calls), with its gradient norm frozen for 150+ iterations at a time.
#
# The callback is deliberately narrow: it ends a run only after
# `_LS_REFINE_STUCK_WINDOW` consecutive iterations in which the iterate `x`, the
# objective value and the gradient norm are all bit-identical to the previous
# iteration -- the optimiser is not moving at all. Any movement, however small
# or slow (an ill-conditioned problem at a large offset whose value no longer
# changes in Float64, say), resets the count. It needs `extended_trace` for `x`;
# without `x` in the state it never stops a run. A stopped run reports
# `converged == false`, so it is rejected exactly as a run that exhausted its
# budget would be; the certificate itself is unchanged.
const _LS_REFINE_STUCK_WINDOW = 100

function _ls_refine_stall_callback(window::Int = _LS_REFINE_STUCK_WINDOW)
    last = Ref{Any}(nothing)
    stuck = Ref(0)
    return function (state)
        x = get(state.metadata, "x", nothing)
        current = x === nothing ? nothing : (state.value, state.g_norm, copy(x))
        if current !== nothing && last[] !== nothing && isequal(current, last[])
            stuck[] += 1
        else
            stuck[] = 0
        end
        last[] = current
        return stuck[] >= window
    end
end

# Options for one certified-refinement run (a fresh stall callback per run).
_ls_refine_options(g_tol, iterations) =
    Optim.Options(g_tol=g_tol, iterations=iterations,
                  x_abstol=NaN, x_reltol=NaN, f_abstol=NaN, f_reltol=NaN,
                  extended_trace=true, callback=_ls_refine_stall_callback())

"""
    _fit_locscale(kind, y, Xμ, Xψ, gidx, G, Q; ...)

Fit the q=2 non-Gaussian location–scale model. Returns a named tuple with the
packed estimate `θ`, the fixed effects `beta_mu` / `beta_psi`, the 2×2
group-level covariance `Lambda`, the marginal `nll`, and a `converged` flag.
"""
function _fit_locscale(kind, y, Xμ, Xψ, gidx, G, Q;
                       βμ0 = nothing, βψ0 = nothing,
                       λ0 = [log(0.3), 0.0, log(0.3)],
                       Zη = _ls_canonical_Zeta(length(y)),
                       Zψ = _ls_canonical_Zpsi(length(y)),
                       g_tol = 1e-6, iterations = 1000, se::Bool = false,
                       whitened::Bool = false)
    pμ = size(Xμ, 2); pψ = size(Xψ, 2)
    βμ0 === nothing && (βμ0 = _ls_default_betastart(kind, y, Xμ))
    βψ0 === nothing && (βψ0 = zeros(pψ))
    θ0 = vcat(βμ0, βψ0, λ0)

    # Gradient-based fit using the exact O(p) outer gradient (`_ls_marginal_grad`).
    # Two robustness levers, both essential on tree (phylogenetic) problems:
    #  * WARM-START the inner mode across outer iterations (`warm` Ref). The outer
    #    gradient is exact (not finite-differenced), so the converged mode — and
    #    thus the gradient — is unchanged by the start point; warm-starting only
    #    makes each inner solve cheap (1–2 Newton steps instead of a cold solve).
    #  * a TRUST-REGION NEWTON outer optimiser (exact gradient + observed
    #    information as Hessian), which converges quadratically and tolerates the
    #    ill-conditioning/indefiniteness that makes LBFGS stall on trees. LBFGS
    #    then Nelder–Mead are kept as fallbacks.
    warm = whitened ? Ref{Union{Nothing,_LSWhitenedSeed}}(nothing) :
                      Ref{Union{Nothing,Vector{Float64}}}(nothing)
    function nll(θ)
        if whitened
            result = _ls_whitened_eval(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ;
                                      seed=warm[], gradient=false)
            result.status.ok && (warm[] = result.seed)
            return result.status.ok ? result.value : 1e18
        end
        βμ = @view θ[1:pμ]; βψ = @view θ[pμ+1:pμ+pψ]
        P = prior_precision(Q, _ls_lc_inv2x2(θ[pμ+pψ+1:pμ+pψ+3]))   # stable: never forms Λ
        val, a, ok = _ls_marginal_nll(kind, y, Xμ * βμ, Xψ * βψ, gidx, G, P, Zη, Zψ; a0 = warm[])
        ok && (warm[] = copy(a))
        return ok ? val : 1e18
    end
    function g!(grad, θ)
        grad .= whitened ?
            _ls_whitened_eval(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ; seed=warm[]).gradient :
            _ls_marginal_grad(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ; a0=warm[])
        return grad
    end
    function h!(H, θ)
        H .= whitened ?
            _ls_whitened_information(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ; seed=warm[]) :
            _ls_obs_information(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ; a0=warm[])
        return H
    end
    opts = Optim.Options(g_tol = g_tol, iterations = iterations)
    nm() = (warm[] = nothing; Optim.optimize(nll, θ0, Optim.NelderMead(),
                          Optim.Options(iterations = max(iterations, 2000))))
    res = try
        Optim.optimize(nll, g!, h!, θ0, Optim.NewtonTrustRegion(), opts)
    catch err
        err isa InterruptException && rethrow(err)
        warm[] = nothing
        try
            Optim.optimize(nll, g!, θ0, Optim.LBFGS(), opts)
        catch err2
            err2 isa InterruptException && rethrow(err2)
            nm()
        end
    end
    θ̂ = Optim.minimizer(res)
    # Feasibility guard. On weakly-identified fixtures a variance MLE can sit on
    # the boundary; LBFGS may then chase λ until Λ underflows to singular, where
    # the gradient is forced to zero and LBFGS falsely "converges" to an
    # infeasible (non-finite) θ̂. A feasible point (finite mode solve) provably has
    # a PD Λ (Λ = LLᵀ, det>0), so if the result is infeasible, fall back to the
    # conservative derivative-free solve.
    if any(!isfinite, θ̂) || !(nll(θ̂) < 1e17)
        res = nm()
        θ̂ = Optim.minimizer(res)
    end
    # A trust-region solve may stop unsuccessfully near a zero-variance
    # boundary while its inner mode remains valid. Try BFGS with backtracking,
    # L-BFGS with backtracking, and default L-BFGS in that order; for each
    # method, try the failed endpoint before the canonical start. None of these
    # routes relaxes convergence: a candidate replaces the failed result only
    # after a fresh exact gradient and no-worse-objective certificate.
    # Successful fits and the legacy raw route bypass the entire ladder.
    if whitened && !Optim.converged(res)
        warm[] = nothing
        try
            baseline = _ls_whitened_eval(kind, y, Xμ, Xψ, gidx, G, Q, θ̂, Zη, Zψ;
                                         gradient=false)
            if baseline.status.ok && isfinite(baseline.value)
                function certified_refinement(start, method)
                    warm[] = nothing
                    try
                        candidate = Optim.optimize(nll, g!, copy(start), method,
                                                     _ls_refine_options(g_tol, iterations))
                        θc = Optim.minimizer(candidate)
                        if Optim.converged(candidate) && all(isfinite, θc)
                            checked = _ls_whitened_eval(kind, y, Xμ, Xψ, gidx, G, Q,
                                                        θc, Zη, Zψ)
                            # Eight ULPs of the baseline objective permit only
                            # rounding differences; this is not a relative
                            # likelihood tolerance.
                            allowance = 8 * eps(max(abs(baseline.value), 1.0))
                            if checked.status.ok && isfinite(checked.value) &&
                               all(isfinite, checked.gradient) &&
                               maximum(abs, checked.gradient) <= g_tol &&
                               checked.value <= baseline.value + allowance
                                return candidate, θc
                            end
                        end
                    catch err
                        err isa InterruptException && rethrow(err)
                    finally
                        warm[] = nothing
                    end
                    return nothing
                end
                # Full BFGS with backtracking is the robust first continuation
                # after the trust-region failure. If that is rejected, try the
                # two L-BFGS line searches. These are only
                # alternate numerical routes to the same objective; every
                # replacement passes the identical exact-gradient and
                # no-worse-objective certificate below.
                methods = (
                    Optim.BFGS(linesearch=Optim.LineSearches.BackTracking()),
                    Optim.LBFGS(linesearch=Optim.LineSearches.BackTracking()),
                    Optim.LBFGS(),
                )
                refined = nothing
                for method in methods, start in (θ̂, θ0)
                    refined = certified_refinement(start, method)
                    refined === nothing || break
                end
                if refined !== nothing
                    res, θ̂ = refined
                end
            end
        catch err
            err isa InterruptException && rethrow(err)
            # Retain the original unsuccessful result, never a partial candidate.
        finally
            warm[] = nothing
        end
    end
    Λ̂ = _ls_lc_to_Λ(θ̂[pμ+pψ+1:pμ+pψ+3])
    nll(θ̂)                       # ensure warm[] holds the mode at θ̂ for the SE solve
    # Wald inference (opt-in): observed information = Hessian of the exact gradient.
    V = !se ? nothing : whitened ?
        _ls_whitened_vcov(kind, y, Xμ, Xψ, gidx, G, Q, θ̂, Zη, Zψ; seed=warm[]) :
        _ls_vcov(kind, y, Xμ, Xψ, gidx, G, Q, θ̂, Zη, Zψ; a0=warm[])
    return (θ = θ̂,
            beta_mu = θ̂[1:pμ],
            beta_psi = θ̂[pμ+1:pμ+pψ],
            Lambda = Λ̂,
            components = _ls_components(Λ̂),
            vcov = V,
            se = _ls_se(V),
            nll = nll(θ̂),
            converged = Optim.converged(res))
end

# Convenience: derive (Q, gidx, G) for the PHYLOGENETIC case from a tree and the
# per-observation species labels (reuses the verified phylo precision assembly).
function _locscale_phylo_setup(tree, labels)
    Q, leaf_node, _ = _poisson_phylo_setup(tree, labels)
    return Q, leaf_node, size(Q, 1)
end

# Convenience: derive (Q, gidx, G) for a general user-supplied PD covariance `C`
# (relatedness / animal model / precomputed spatial), the non-tree analogue of
# `_locscale_phylo_setup`. Identical engine contract: the tree precision is
# swapped for C⁻¹ (rescaled to a unit-diagonal correlation), exactly the swap the
# mean-only `_fit_*_relmat_laplace` routes already make. `G = size(Q,1)` = the
# number of distinct group levels and `gidx` hits every level (data at every
# node, unlike the tree where G > #leaves) — both handled identically by the
# `_ls_joint*` block assembly.
function _locscale_relmat_setup(C, labels)
    Q, gidx = _general_cov_setup(C, labels)
    return Q, gidx, size(Q, 1)
end

# Cluster 3 helper: resolve (Q, gidx, G) for a structured non-Gaussian slope
# from the struct kind and the group-level covariance source. Reuses the same
# helpers used by the structured intercept (phylo/relmat) and the location–scale
# structured intercept paths. `phylo` builds Q from the tree; relmat/animal/spatial
# build Q from the user-supplied PD matrix via `_general_cov_setup`. A bare :iid
# kind (ordinary i.i.d. groups) is not supported here — use `_fit_corr_locscale`
# directly with the default Q = I.
function _locscale_structured_q(struct_kind::Symbol, grp_sym::Symbol, labels,
                                tree, K, A)
    if struct_kind === :phylo
        tree === nothing && error("phylo(1 + … | $grp_sym) needs `tree = …`")
        return _locscale_phylo_setup(tree, labels)     # (Q, gidx, G)
    else
        C = _poisson_structured_cov(struct_kind, grp_sym, K, A, nothing)
        return _locscale_relmat_setup(C, labels)       # (Q, gidx, G)
    end
end
