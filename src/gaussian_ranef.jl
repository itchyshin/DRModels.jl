# gaussian_ranef.jl — ordinary Gaussian random *intercepts* on the mean.
#
# For a mean random effect the marginal is exactly Gaussian:
#     y ~ N(Xβ, V),  V = D + σ_b² Z Zᵀ,  D = diag(σ_i²)
# where Z is the group-indicator. We never form V: the matrix-determinant lemma
# and the Woodbury identity reduce everything to O(n) accumulations plus a
# diagonal G×G capacitance (one random-intercept term), all ForwardDiff-friendly.

# Barrier used when the REML restriction matrix Xμ′V⁻¹Xμ fails its Cholesky.
# It is PSD by construction but built by Woodbury SUBTRACTION, so as σb² → ∞ it
# tends to 0 and rounding can flip its determinant negative (#499). A finite
# barrier — not Inf — because LBFGS's HagerZhang line search asserts isfinite.
const REML_NONPD_PENALTY = 1e8

# A structured marker's (`phylo`/`relmat`/`animal`/`spatial`) random-effect
# left-hand side must be the literal intercept `1` on every univariate route:
# `_split_ranef` (below) — the shared RHS splitter for every univariate family
# (Gaussian, Poisson, Gamma, Beta, BetaBinomial, Binomial, NegBinomial2, and
# more) — used to keep only the grouping symbol `g` and silently discard
# `lhs`, so e.g. `phylo(1 + x | g)` fit the exact same intercept-only model as
# `phylo(1 | g)` with no error (silent data loss, #620). drmTMB fits a genuine
# two-free-SD phylogenetic random INTERCEPT+SLOPE from `phylo(1 + x | g)` on
# Gaussian, Poisson and NegBinomial2 (all three as the SAME independent two-SD
# model -- measured on drmTMB main 2026-09-05; it refuses the formula on Gamma).
# DRModels.jl implements only the Gaussian one (#620), because that route is the exact
# closed-form marginal and does not extend to a non-Gaussian likelihood; it has no
# route at all for relmat/animal/spatial, so fail closed instead of silently
# dropping the slope.
_structured_re_lhs_text(lhs) = if lhs isa ConstantTerm
        string(lhs.n)
    elseif lhs isa Term
        string(lhs.sym)
    elseif lhs isa FunctionTerm && lhs.f === (+)
        join((a isa ConstantTerm ? string(a.n) : string(a.sym) for a in lhs.args), " + ")
    else
        string(lhs)
    end

# Returns the slope variable (a `Symbol`) for the ONE admitted slope form
# `phylo(1 + x | g)` when `allow_slope = true` (the Gaussian mean route, #620:
# two independent phylogenetic fields with separate SDs — drmTMB's model), and
# `nothing` for the intercept form `phylo(1 | g)`. Every other lhs — and the
# slope form on any route that did not opt in — throws, so no route can silently
# fit the intercept-only model in place of what the formula says.
function _check_phylo_re_lhs(lhs, grp::Symbol; allow_slope::Bool = false)
    (lhs isa ConstantTerm && lhs.n == 1) && return nothing
    lhs_text = _structured_re_lhs_text(lhs)
    if lhs isa FunctionTerm && lhs.f === (+)
        consts = filter(t -> t isa ConstantTerm, lhs.args)
        vars = filter(t -> t isa Term, lhs.args)
        is_one_slope = length(lhs.args) == 2 && length(consts) == 1 &&
            consts[1].n == 1 && length(vars) == 1
        if is_one_slope
            allow_slope && return vars[1].sym
            throw(ArgumentError("drm: `phylo($lhs_text | $grp)` is not implemented on this " *
                "route — only `phylo(1 | $grp)` (intercept) is here. The two-SD phylogenetic " *
                "random intercept + slope `phylo(1 + x | $grp)` is implemented for the " *
                "Gaussian mean only (`drm(bf(y ~ … + phylo(1 + x | $grp)), Gaussian(); tree)`). " *
                "drmTMB also fits this exact formula on Poisson and NegBinomial2, and it fits " *
                "the SAME independent two-SD model there — measured on drmTMB main 2026-09-05, " *
                "`corpars` is empty for `phylo(1 + x | g)` on Poisson; the estimated " *
                "intercept–slope correlation (`has_phylo_mu_q2_covariance`, surfaced in " *
                "`corpars`) belongs to the DIFFERENT tagged formula `phylo(1 + x | p | $grp)`. " *
                "DRModels.jl refuses the non-Gaussian families here because its route is the EXACT " *
                "closed-form Gaussian marginal, which does not extend to a non-Gaussian " *
                "likelihood — not because the target would differ. On Gamma, drmTMB refuses " *
                "this formula too (\"intercept-only in this q=1 route\"), so `engine = \"tmb\"` " *
                "is a route for Gaussian/Poisson/NegBinomial2 but not for Gamma (#620)"))
        end
    end
    throw(ArgumentError("drm: `phylo($lhs_text | $grp)` is not implemented on the " *
        "univariate routes — only `phylo(1 | $grp)` (intercept) is, plus the one-slope " *
        "`phylo(1 + x | $grp)` on the Gaussian mean (#620); slope-only `phylo(0 + x | g)` " *
        "and multi-slope forms are not implemented"))
end

# relmat/animal/spatial share the identical parser gap but, unlike phylo, have
# no verified drmTMB two-SD slope route to name as a follow-up — say only that
# the construct is unimplemented (#620).
function _check_structured_re_lhs(kind::Symbol, lhs, grp::Symbol)
    (lhs isa ConstantTerm && lhs.n == 1) && return nothing
    lhs_text = _structured_re_lhs_text(lhs)
    throw(ArgumentError("drm: `$kind($lhs_text | $grp)` is not implemented on the " *
        "univariate routes; only the intercept form is"))
end

# Split a μ right-hand side into its fixed part and any `(lhs | g)` terms.
# `structured` is the FIRST structured marker (relmat/animal/phylo/spatial) for
# backward compatibility; use `_collect_structured` to retrieve the full list
# (the Gaussian router supports two structured components in one fit).
#
# Returns a 5-tuple `(fixed_rhs, re, metav, structured, structured_slope)`. The
# fifth slot is the slope variable of a `phylo(1 + x | g)` marker (#620) and is
# `nothing` otherwise; it is only ever non-`nothing` when the caller opted in
# with `allow_phylo_slope = true` (the Gaussian mean route and the read-only
# consumers of an already-fitted Gaussian formula — predict, bridge labels).
# Every other caller keeps the default and keeps the #621 refusal, so a
# non-Gaussian family can never receive `(:phylo, g)` for a slope formula and
# silently fit the intercept-only model. Existing 4-way destructurings
# (`a, b, c, d = _split_ranef(rhs)`) are unaffected: Julia drops the extra slot.
# lme4 / glmmTMB / drmTMB semantics: `(x | g)` is `(1 + x | g)` — the intercept is
# implicit unless removed with an explicit `0 +` / `-1`. Rewrite a bar lhs made only of
# variable terms (`x`, `x + z`) to `1 + …`; anything carrying a constant, a `-`, or a
# nested bar is returned untouched, so every existing refusal keeps its message.
_implicit_re_intercept(lhs) = lhs
_implicit_re_intercept(lhs::Term) = FunctionTerm{typeof(+),Vector{StatsModels.AbstractTerm}}(
    +, StatsModels.AbstractTerm[ConstantTerm(1), lhs], :(1 + $(lhs.sym)))
function _implicit_re_intercept(lhs::FunctionTerm)
    (lhs.f === (+) && all(a -> a isa Term, lhs.args)) || return lhs
    return FunctionTerm{typeof(+),Vector{StatsModels.AbstractTerm}}(
        +, StatsModels.AbstractTerm[ConstantTerm(1), lhs.args...], :(1 + $(lhs.exorig.args[2:end]...)))
end

function _split_ranef(rhs; allow_phylo_slope::Bool = false)
    terms = rhs isa Tuple ? collect(rhs) : Any[rhs]
    fixed = Any[]
    re = Tuple{Any,Symbol}[]
    metav = nothing                                   # meta_V(v) known-variance column
    structured = nothing                              # (:relmat, grouping) — known K
    structured_slope = nothing                        # `x` of phylo(1 + x | g), Gaussian mean only
    for t in terms
        if t isa FunctionTerm && t.f === (|)
            push!(re, (_implicit_re_intercept(t.args[1]), t.args[2].sym))     # (re-lhs, grouping symbol)
        elseif t isa FunctionTerm && t.f === meta_V
            metav = t.args[1].sym
        elseif t isa FunctionTerm && t.f === relmat
            _check_structured_re_lhs(:relmat, t.args[1].args[1], t.args[1].args[2].sym)
            structured === nothing && (structured = (:relmat, t.args[1].args[2].sym))   # inner (1 | grp)
        elseif t isa FunctionTerm && t.f === animal
            _check_structured_re_lhs(:animal, t.args[1].args[1], t.args[1].args[2].sym)
            structured === nothing && (structured = (:animal, t.args[1].args[2].sym))
        elseif t isa FunctionTerm && t.f === phylo
            slope = _check_phylo_re_lhs(t.args[1].args[1], t.args[1].args[2].sym;
                                        allow_slope = allow_phylo_slope)
            if structured === nothing
                structured = (:phylo, t.args[1].args[2].sym)
                structured_slope = slope
            elseif slope !== nothing
                throw(ArgumentError("drm: `phylo(1 + $(slope) | $(t.args[1].args[2].sym))` must be " *
                    "the only structured marker on the mean; a second structured component " *
                    "alongside the phylogenetic random slope is not implemented (#620)"))
            end
        elseif t isa FunctionTerm && t.f === spatial
            _check_structured_re_lhs(:spatial, t.args[1].args[1], t.args[1].args[2].sym)
            structured === nothing && (structured = (:spatial, t.args[1].args[2].sym))
        else
            push!(fixed, t)
        end
    end
    fixed_rhs = isempty(fixed) ? ConstantTerm(1) :
                length(fixed) == 1 ? fixed[1] : Tuple(fixed)
    return fixed_rhs, re, metav, structured, structured_slope
end

# Collect EVERY structured marker on a right-hand side, in source order, as a
# Vector of (kind, grouping) tuples (kind ∈ :relmat, :animal, :phylo, :spatial).
# Empty when none are present. The Gaussian mean path can fit two such components
# (e.g. `phylo(1|species) + relmat(1|id)`); single-marker callers keep using the
# `structured` slot returned by `_split_ranef`.
function _collect_structured(rhs)
    terms = rhs isa Tuple ? collect(rhs) : Any[rhs]
    out = Tuple{Symbol,Symbol}[]
    for t in terms
        t isa FunctionTerm || continue
        if t.f === relmat
            push!(out, (:relmat, t.args[1].args[2].sym))
        elseif t.f === animal
            push!(out, (:animal, t.args[1].args[2].sym))
        elseif t.f === phylo
            push!(out, (:phylo, t.args[1].args[2].sym))
        elseif t.f === spatial
            push!(out, (:spatial, t.args[1].args[2].sym))
        end
    end
    return out
end

# Per-observation random-effect design weight from the term's lhs:
# `(1 | g)` → wᵢ = 1 (random intercept); `(0 + x | g)` → wᵢ = xᵢ (independent
# random slope). Correlated `(1 + x | g)` is planned.
function _re_kind(re_lhs)
    if re_lhs isa ConstantTerm
        re_lhs.n == 1 || error("random-effect intercept term must be `1`")
        return (:intercept, nothing)
    elseif re_lhs isa FunctionTerm && re_lhs.f === (+)
        consts = filter(t -> t isa ConstantTerm, re_lhs.args)
        vars = filter(t -> t isa Term, re_lhs.args)
        length(vars) == 1 || error("DRModels.jl supports `(1 | g)`, `(0 + x | g)`, `(1 + x | g)`")
        v = vars[1].sym
        any(c -> c.n == 0, consts) && return (:slope, v)      # 0 + x  → slope only
        any(c -> c.n == 1, consts) && return (:corr, v)       # 1 + x  → correlated intercept+slope
    end
    error("unsupported random-effect term: `($re_lhs | …)`")
end

# Map group labels to 1:G (stable first-seen order), O(n).
function _group_index(labels)
    lvl = Dict{eltype(labels),Int}()
    gidx = Vector{Int}(undef, length(labels))
    for (i, l) in enumerate(labels)
        gidx[i] = get!(lvl, l, length(lvl) + 1)
    end
    return gidx, length(lvl)
end

# Cancellation-free Woodbury quadratic form r′V⁻¹r for V = D + Z Σ_b Z′ with one
# scalar random effect per group (Z_ik = w_i·[g_i = k], Σ_b = diag(σ_b,k²)) —
# #746 / #747.
#
# The textbook Woodbury form r′V⁻¹r = r′D⁻¹r − Σ_k C_k²/M_k (C_k = Σ w_i r_i/D_i,
# M_k = 1/σ_b,k² + Σ w_i²/D_i) is a DIFFERENCE of two terms that both grow like
# 1/D_min. When the optimiser pushes one residual σ_i towards 0 (σ ~ x with a
# steep slope), both terms reach ~1e130 and their rounding error (~1e114) is far
# larger than the true O(100) value: the difference can come out hugely NEGATIVE,
# the nll → −1e133, and LBFGS "converges" into that rounding hole (logLik +1e44
# … +1e133 in the Wave8 twin cells). The same quantity is the penalised
# residual sum of squares at the conditional mode û_k = C_k/M_k:
#
#     r′V⁻¹r = Σ_i (r_i − w_i û_{g_i})²/D_i + Σ_k û_k²/σ_b,k²,
#
# a sum of NON-NEGATIVE terms, so it can never go below zero. One step of
# iterative refinement on û keeps the tiny-D terms accurate (r_i − w_i û is then
# formed from an û that is correct to rounding relative to r_i). In exact
# arithmetic the result is identical to q1 − q2 (and the refinement step is
# identically zero as a function of θ, so ForwardDiff derivatives are unchanged).
function _re_quad_stable(r::AbstractVector, invD::AbstractVector, w, gidx::AbstractVector{<:Integer},
                         invσb2::AbstractVector, S::AbstractVector, C::AbstractVector)
    T = promote_type(eltype(r), eltype(invD), eltype(invσb2), eltype(S), eltype(C))
    G = length(S)
    M = Vector{T}(undef, G); u = Vector{T}(undef, G); gk = zeros(T, G)
    @inbounds for k in 1:G
        M[k] = invσb2[k] + S[k]
        u[k] = C[k] / M[k]
    end
    @inbounds for i in eachindex(r)
        k = gidx[i]; wi = w === nothing ? one(T) : w[i]
        gk[k] += wi * (r[i] - wi * u[k]) * invD[i]
    end
    quad = zero(T)
    @inbounds for k in 1:G
        # invσb2[k] == Inf ⇒ σ_b,k → 0 exactly (an unconstrained line-search
        # probe can reach this, e.g. a `sd(g) ~ x` submodel driving log σ_b,k
        # to a large negative value). u[k] is already 0 from C[k]/M[k] with
        # M[k] = Inf, the correct conditional mode at zero group variance, and
        # the prior/shrinkage term contributes nothing to the quadratic form
        # in that limit. Skip the refinement there: `u[k] * invσb2[k]` would
        # otherwise be `0 * Inf = NaN`, which fails LineSearches' finiteness
        # assertion (regression from #835 surfaced by the location-scale-scale
        # REML docs example).
        if isfinite(invσb2[k])
            u[k] += (gk[k] - u[k] * invσb2[k]) / M[k]
            quad += u[k]^2 * invσb2[k]
        end
    end
    @inbounds for i in eachindex(r)
        k = gidx[i]; wi = w === nothing ? one(T) : w[i]
        e = r[i] - wi * u[k]
        quad += e * e * invD[i]
    end
    return quad
end

# Cancellation-free X′V⁻¹X for the REML term, same V = D + Z Σ_b Z′ as
# `_re_quad_stable` (the matrix analogue of that function).
#
# The Woodbury form X′V⁻¹X = X′D⁻¹X − Σ_k z_k z_k′/M_k (z_k = Σ w_i x_i/D_i) is a
# difference of two terms that both grow like 1/D_min. On an LBFGS line-search
# probe with log σ_i ≈ −105 it returned 8.08e110·[1 1; 1 1] (exactly singular)
# against a true ≈ 1.34e−7·[1 1; 1 1]; under ForwardDiff Duals the generic
# Cholesky accepted the zero pivot, logdet = −Inf, and HagerZhang's finiteness
# assertion threw (test/test_lss_reml_falseconv.jl). The same matrix is the
# penalised sum of squares at the conditional mode U_k = z_k/M_k:
#
#     X′V⁻¹X = Σ_i (x_i − w_i U_{g_i})(x_i − w_i U_{g_i})′/D_i + Σ_k U_k U_k′/σ_b,k²,
#
# a sum of PSD rank-one terms. As in `_re_quad_stable`, one refinement step on U
# (identically zero in exact arithmetic, so AD derivatives are unchanged), and a
# group with invσb2[k] = Inf (σ_b,k = 0) keeps U_k = 0 and adds no prior term.
function _re_xtvinvx_stable(X::AbstractMatrix, invD::AbstractVector, w, gidx::AbstractVector{<:Integer},
                            invσb2::AbstractVector, S::AbstractVector)
    T = promote_type(eltype(X), eltype(invD), eltype(invσb2), eltype(S))
    G = length(S); p = size(X, 2)
    M = Vector{T}(undef, G)
    U = zeros(T, G, p); gk = zeros(T, G, p)
    @inbounds for i in axes(X, 1)
        k = gidx[i]; wi = w === nothing ? one(T) : w[i]
        for j in 1:p
            U[k, j] += wi * X[i, j] * invD[i]
        end
    end
    @inbounds for k in 1:G
        M[k] = invσb2[k] + S[k]
        for j in 1:p
            U[k, j] /= M[k]
        end
    end
    @inbounds for i in axes(X, 1)
        k = gidx[i]; wi = w === nothing ? one(T) : w[i]
        for j in 1:p
            gk[k, j] += wi * (X[i, j] - wi * U[k, j]) * invD[i]
        end
    end
    A = zeros(T, p, p)
    @inbounds for k in 1:G
        isfinite(invσb2[k]) || continue
        for j in 1:p
            U[k, j] += (gk[k, j] - U[k, j] * invσb2[k]) / M[k]
        end
        for j in 1:p, l in 1:p
            A[j, l] += U[k, j] * invσb2[k] * U[k, l]
        end
    end
    e = Vector{T}(undef, p)
    @inbounds for i in axes(X, 1)
        k = gidx[i]; wi = w === nothing ? one(T) : w[i]
        for j in 1:p
            e[j] = X[i, j] - wi * U[k, j]
        end
        for j in 1:p, l in 1:p
            A[j, l] += e[j] * invD[i] * e[l]
        end
    end
    return A
end

# --- start values for `_fit_ranef_gaussian` / `_fit_ranef_gaussian_lss` (#747 follow-up) ---
#
# The historical start (`log(std(res0))` for the sigma intercept, all other sigma
# coefficients and the log-SD of the random intercept at fixed constants) ignores
# the mean OLS residuals' own information about the scale submodel entirely: a
# `sigma ~ x` slope always starts at 0 and the RE SD always starts at a fixed
# fraction of the *marginal* residual SD, which is inflated by the random-intercept
# variance it is trying to estimate separately. On the RUNAWAY panel
# (test/test_ranef_varying_scale_convergence.jl, sigma slope 10, n=40, G=4) that
# blind start left LBFGS on a boundary sd_g -> 0 local optimum 123 nats short of
# drmTMB's interior optimum (seed 4) and non-converged 95 nats short (seed 10) --
# in both cases DRModels.jl's own objective at drmTMB's parameters was BETTER than
# what LBFGS returned, so the fix is a better start, not the objective.
#
# `_ranef_sigma_ols_start`: OLS of log(guarded residual^2) on Xσ -- the standard
# heteroscedastic-regression start (mirrors drmTMB's own guarded log-scale
# regression start for `sigma ~ …`, #572/#570), giving a real slope instead of 0.
# Guarded at a small floor so a near-exact-zero residual cannot send log(r^2) to
# -Inf.
function _ranef_sigma_ols_start(Xσ::AbstractMatrix, res0::AbstractVector)
    floor2 = max(1e-8, 1e-6 * mean(abs2, res0))
    return Xσ \ log.(max.(abs2.(res0), floor2))  ./ 2   # log|r| ~ 0.5*log(r^2)
end

# `_ranef_sdg_mom_start`: one-way random-effects ANOVA method-of-moments
# estimator of the random-intercept variance from the OLS mean residuals,
# grouped by `gidx` (Searle, Casella & McCulloch 1992, ch. 3). Returns the
# log-SD on the same scale `_fit_ranef_gaussian` optimises. Floors at a small
# positive variance instead of the boundary itself, so the optimiser starts
# strictly interior even when the moment estimator itself is <= 0 (negative
# "between" variance relative to "within", the classic small-G/unbalanced
# symptom).
function _ranef_sdg_mom_start(res0::AbstractVector, gidx::AbstractVector{<:Integer}, G::Int)
    n = length(res0)
    nk = zeros(Int, G); sumk = zeros(G)
    @inbounds for i in 1:n
        k = gidx[i]; nk[k] += 1; sumk[k] += res0[i]
    end
    rbar = mean(res0)
    ssb = 0.0; ssw = 0.0
    @inbounds for i in 1:n
        k = gidx[i]
        ssw += (res0[i] - sumk[k] / nk[k])^2
    end
    @inbounds for k in 1:G
        ssb += nk[k] * (sumk[k] / nk[k] - rbar)^2
    end
    dfw = max(n - G, 1)
    msw = ssw / dfw
    n0 = (n - sum(abs2, nk) / n) / max(G - 1, 1)
    σb2_mom = (ssb / max(G - 1, 1) - msw) / n0
    σb2_floor = 1e-4 * (msw + eps())
    return 0.5 * log(max(σb2_mom, σb2_floor))
end

# One restart, deterministic, keep the best objective (mirrors the boundary-
# restart pattern in `_fit_correlated_ranef_gaussian`, #762/#837). `θ0` is the
# data-driven start; `θ0_restart` is always-interior (the historical default).
# Restart triggers on either symptom measured on the RUNAWAY panel
# (test/test_ranef_varying_scale_convergence.jl): the gradient criterion not
# met, or the random-intercept log-SD landing at a boundary sd_g -> 0 local
# optimum relative to `scale_ref` (the OLS mean-residual SD) -- seed 4's
# failure mode, `nll` −179.80 vs drmTMB's interior −56.97. A restart that is
# not triggered costs nothing beyond the first solve; one that is costs one
# extra LBFGS run, still O(1) versus the data.
#
# TIE-BREAK ON A GENUINE BOUNDARY (measured on seed 1 of the same panel). When
# sd_g -> 0 the objective is flat in lσb far below the boundary (the log-SD is
# only weakly identified there), so the primary and restart starts can reach
# the SAME nll at two very different lσb (seed 1: −177.4 vs −118.2, nll equal
# to 1e-10). Both are the same MLE, but the more extreme one can push
# `ForwardDiff.hessian` in the caller's `_vcov_from_hessian` step to a
# non-finite entry where the less extreme one does not (this is the vcov
# guard's job to flag as a boundary fit either way -- see its docstring -- not
# a reason to crash on one representative of the tie and not the other). On a
# near-tie, prefer whichever result sits closer to the interior.
function _re_lbfgs_with_restart(nll, θ0::AbstractVector, θ0_restart::AbstractVector,
                                g_tol::Real, lσb_idx::Int, scale_ref::Real)
    opts = Optim.Options(g_tol = g_tol)
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), opts; autodiff = :forward)
    θ̂ = Optim.minimizer(res)
    # NON-FINITE RESULT (Linux/Julia 1.10.12, seed 20 of the RUNAWAY panel, #848).
    # From the data-driven start LBFGS can report `converged` with a NaN
    # MINIMIZER (a line-search probe overflowed exp(-2ησ)); `nll(θ̂)` is NaN
    # even though `Optim.minimum(res)` is a finite stale value. Every NaN
    # comparison below is `false`, so without this check neither restart
    # trigger fired, the NaN θ̂ went on to `ForwardDiff.hessian`, and
    # `_vcov_from_hessian`'s `eigvals` threw "matrix contains Infs or NaNs".
    # A non-finite primary result is therefore itself a restart trigger.
    f_primary = _objective_at_minimizer(nll, res)
    primary_ok = isfinite(f_primary) && all(isfinite, θ̂)
    g_inf = primary_ok ? maximum(abs, ForwardDiff.gradient(nll, θ̂)) : Inf
    at_boundary = primary_ok && exp(θ̂[lσb_idx]) < 1e-3 * max(scale_ref, eps())
    if !primary_ok || !(g_inf <= g_tol) || at_boundary
        res_restart = Optim.optimize(nll, θ0_restart, Optim.LBFGS(), opts; autodiff = :forward)
        θ̂_restart = Optim.minimizer(res_restart)
        # Compare the objective AT each minimizer, not `Optim.minimum` (#849,
        # optim_minimum_guard.jl): these runs may have failed a line search.
        f_restart = _objective_at_minimizer(nll, res_restart)
        restart_ok = isfinite(f_restart) && all(isfinite, θ̂_restart)
        if !primary_ok
            return restart_ok ? res_restart : res
        elseif !restart_ok
            return res
        end
        Δ = f_restart - f_primary
        near_tie = abs(Δ) <= 1e-6 * max(1, abs(f_primary))
        if Δ < 0 && !near_tie
            return res_restart
        elseif near_tie && abs(θ̂_restart[lσb_idx]) < abs(θ̂[lσb_idx])
            return res_restart
        end
    end
    return res
end

# Gaussian location–scale with one random intercept (1 | g) on the mean.
# θ = [β_μ; β_σ (log σ); log σ_b].
# `reml=true` (#439) keeps β_μ in θ and adds the Patterson–Thompson term
# ½ logdet(Xμ′ V⁻¹ Xμ) to the Woodbury nll. ML (`reml=false`) is the default
# and uses the historical nll byte-for-byte.
"""
    _fit_ranef_gaussian(..., reml=false) -> DrmFit

Gaussian location–scale with one mean random intercept `(1 | g)` on the
Woodbury spine. `reml=false` (default) is ML, byte-for-byte with the
historical nll. `reml=true` (#439) adds `½ logdet(Xμ′ V⁻¹ Xμ) − ½ pμ log(2π)`
to that nll (Patterson–Thompson). Reached via `drm(...; method = :REML)` for
a single intercept only — σ-RE, slopes, and multi-ranef stay rejected.
Worked example: `test/test_reml_ordinary_ranef.jl` (not in the default suite yet).
"""
function _fit_ranef_gaussian(fam::Gaussian, y, Xμ, Xσ, gidx, G, w, nmμ, nmσ, grp, g_tol;
                             reml::Bool = false)
    n = length(y)
    pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    const_2pi = 0.5 * n * log(2π)
    const_pμ = 0.5 * pμ * log(2π)

    # ML Woodbury nll. The quadratic form is the cancellation-free
    # `_re_quad_stable` (#746/#747); in exact arithmetic it equals the historical
    # q1 − q2 = r′D⁻¹r − Σ C_k²/M_k.
    function nll_ml(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; lσb = θ[pμ+pσ+1]
        ημ = Xμ * βμ; ησ = Xσ * βσ                 # ησ = log σ_i
        σb² = exp(2lσb)
        T = eltype(θ)
        S = zeros(T, G); C = zeros(T, G)           # S_k = Σ 1/D_i,  C_k = Σ r_i/D_i
        rv = Vector{T}(undef, n); invDv = Vector{T}(undef, n)
        logdetD = zero(T)
        @inbounds for i in 1:n
            invD = exp(-2 * ησ[i])
            r = y[i] - ημ[i]
            rv[i] = r; invDv[i] = invD
            k = gidx[i]
            wi = w[i]
            S[k] += wi * wi * invD                 # (ZᵀD⁻¹Z)_kk = Σ w_i²/D_i
            C[k] += wi * r * invD                  # (ZᵀD⁻¹r)_k  = Σ w_i r_i/D_i
            logdetD += 2 * ησ[i]                   # log D_i
        end
        logdetCap = zero(T)
        @inbounds for k in 1:G
            logdetCap += log(1 + σb² * S[k])        # det-lemma term
        end
        quad = _re_quad_stable(rv, invDv, w, gidx, fill(1 / σb², G), S, C)
        logdetV = logdetD + logdetCap
        return 0.5 * (logdetV + quad) + const_2pi
    end

    # Restricted nll: ML Woodbury + ½ logdet(Xμ′ V⁻¹ Xμ) − ½ pμ log(2π).
    # Xμ′ V⁻¹ Xμ = Xμ′ D⁻¹ Xμ − (Z′ D⁻¹ Xμ)′ diag(1/M) (Z′ D⁻¹ Xμ) with the
    # same capacitance M_k = 1/σb² + S_k as the ML nll, evaluated in the
    # cancellation-free form `_re_xtvinvx_stable`.
    function nll_reml(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; lσb = θ[pμ+pσ+1]
        ημ = Xμ * βμ; ησ = Xσ * βσ
        σb² = exp(2lσb)
        T = eltype(θ)
        S = zeros(T, G)
        invDv = Vector{T}(undef, n)
        @inbounds for i in 1:n
            invD = exp(-2 * ησ[i])
            invDv[i] = invD
            wi = w[i]
            S[gidx[i]] += wi * wi * invD
        end
        # PSD penalised-SS form of Xμ′V⁻¹Xμ (Woodbury subtraction lost every digit
        # at σ_i → 0 line-search probes; see `_re_xtvinvx_stable`).
        XtVinvX = _re_xtvinvx_stable(Xμ, invDv, w, gidx, fill(1 / σb², G), S)
        # ML part via the cancellation-free `nll_ml` (#746/#747).
        nll_ml_θ = nll_ml(θ)
        # Xμ′V⁻¹Xμ is PSD by construction, but it is formed by Woodbury SUBTRACTION.
        # As σb² → ∞ the group means absorb the mean signal, Xμ′V⁻¹Xμ → 0, and rounding
        # noise can make its determinant NEGATIVE (measured at lσb ≈ 16, σb² ≈ 8e13).
        # `logdet` has no Symmetric method: it falls through to the LU path, where
        # logabsdet returns sign -1 and log(-1.0) throws DomainError (#499). Reject the
        # step instead. NOTE: it must be a LARGE FINITE barrier, not +Inf — LBFGS's
        # default HagerZhang line search asserts `isfinite(phi_c)` and would trade
        # the DomainError for an AssertionError.
        # A zero pivot is ACCEPTED by the generic (ForwardDiff Dual) Cholesky, so
        # also reject a non-finite logdet (-Inf there broke HagerZhang, #835 lss).
        cholXtVinvX = cholesky(Symmetric(XtVinvX); check=false)
        issuccess(cholXtVinvX) || return nll_ml_θ + T(REML_NONPD_PENALTY)
        ldX = logdet(cholXtVinvX)
        isfinite(ldX) || return nll_ml_θ + T(REML_NONPD_PENALTY)
        return nll_ml_θ + 0.5 * ldX - const_pμ
    end

    nll = reml ? nll_reml : nll_ml

    βμ0 = Xμ \ y
    res0 = y - Xμ * βμ0

    # Data-driven start (#747 follow-up: seeds 4 and 10 of the RUNAWAY panel in
    # test/test_ranef_varying_scale_convergence.jl). See the header note above
    # `_ranef_sigma_ols_start`.
    θ0 = zeros(pμ + pσ + 1)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1:pμ+pσ] .= _ranef_sigma_ols_start(Xσ, res0)
    θ0[pμ+pσ+1] = _ranef_sdg_mom_start(res0, gidx, G)

    # Restart start: the historical blind default (always interior). Used only
    # when the data-driven start above does not gradient-converge or lands at
    # a boundary sd_g -> 0 local optimum; see `_re_lbfgs_with_restart`.
    θ0_restart = zeros(pμ + pσ + 1)
    θ0_restart[1:pμ] .= βμ0
    θ0_restart[pμ+1] = log(std(res0) + eps())
    θ0_restart[pμ+pσ+1] = log(std(res0) / 2 + eps())

    # WHY THIS ROUTE NEEDS NO n-SCALED CONVERGENCE FALLBACK (measured 2026-08-26).
    #
    # `nll_ml` is a raw SUM over observations, and `g_tol` is an ABSOLUTE gradient
    # tolerance, so the #491 question applies here too: does `converged` become
    # unreliable at large n? #491 was exactly this on the sparse-Laplace route,
    # where a flat threshold graded an n-scaled gradient and good large-n fits
    # were reported as failures. INVESTIGATED AND THE ANSWER IS NO -- the flag is
    # trustworthy here, and the reason is structural rather than lucky.
    #
    # MEASURED, n = 500 .. 1,000,000 (Gaussian random intercept, StableRNG):
    #   g_converged = TRUE at every n, on the GRADIENT criterion specifically
    #   (f_converged and x_converged are false throughout, so nothing weaker is
    #   propping it up). Iteration count is 11 at every n up to 5e5, 13 at 1e6.
    #
    # THE MECHANISM. The achievable gradient floor here is MACHINE EPSILON times
    # the objective scale, not an inner-solve noise floor -- measured floor/nll is
    # constant at 2.3e-16 .. 5.8e-16 across three decades of n:
    #
    #        n        nll        achievable floor    floor/nll   headroom to 1e-8
    #      1e3        954          2.27e-13           2.4e-16      43,980x
    #      1e4        9,913        4.26e-12           4.3e-16       2,346x
    #      1e5        99,823       5.82e-11           5.8e-16         172x
    #      1e6        998,477      2.33e-10           2.3e-16          43x
    #
    # That is because the marginal here is EXACT (Woodbury + matrix-determinant
    # lemma, see the header) -- no inner Newton mode solve, so no stopping noise.
    # `sparse_laplace_glmm.jl` has five iterative-solve constructs, and its floor
    # was ~1e-4 at n = 512, some NINE ORDERS larger than the 2.3e-13 here at
    # n = 1000. That gap is the whole difference between #491 and this route.
    #
    # NOTE `autodiff = :forward` does NOT make this scale-invariant, which is the
    # tempting wrong explanation. Measured away from the optimum, the gradient
    # scales LINEARLY with n (||g|| = 249 / 2,215 / 21,610 at n = 1e3/1e4/1e5,
    # per-observation flat at ~0.22). Optim sees the true summed gradient.
    #
    # SO THE MARGIN IS FINITE AND SHRINKS AS 1/n. Since floor is proportional to n
    # and g_tol is absolute, headroom falls from ~44,000x to ~43x over 1e3 .. 1e6.
    # EXTRAPOLATED (not measured, and flagged as such): it would reach g_tol = 1e-8
    # near n ~ 4e7. Far outside any dataset this route is built for, but if that
    # ever changes, normalise the objective by n the way fit_q4_sparse_tmb.jl and
    # reml_q4.jl do -- do NOT add a #491-style fallback, which only papers over it.
    res = _re_lbfgs_with_restart(nll, θ0, θ0_restart, g_tol, pμ + pσ + 1, std(res0))
    θ̂ = Optim.minimizer(res)
    V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))

    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :resd => (pμ+pσ+1):(pμ+pσ+1)]
    names = [:mu => nmμ, :sigma => nmσ, :resd => [String(grp)]]
    means = Dict(:mu => Xμ * θ̂[1:pμ])
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]))   # residual σ (RE excluded)
    # Conditional RE estimates (BLUPs): b̂_k = C_k / M_k, the per-group posterior
    # mean of the random intercept at θ̂. M_k = 1/σ_b² + Σ w_i²/D_i, C_k = Σ w_i r_i/D_i.
    blup = let
        βμ = θ̂[1:pμ]; βσ = θ̂[pμ+1:pμ+pσ]; σb² = exp(2 * θ̂[pμ+pσ+1])
        ημ = Xμ * βμ; ησ = Xσ * βσ
        S = zeros(G); C = zeros(G)
        @inbounds for i in 1:n
            invD = exp(-2 * ησ[i]); k = gidx[i]
            S[k] += w[i]^2 * invD
            C[k] += w[i] * (y[i] - ημ[i]) * invD
        end
        [C[k] / (1 / σb² + S[k]) for k in 1:G]
    end
    re = Dict(Symbol(grp) => blup)
    # CONVERGED MEANS THE GRADIENT CRITERION HERE (#609 item 2, measured 2026-09-05).
    #
    # `Optim.converged(res)` is the OR of the x, f and g criteria. `Optim.Options(
    # g_tol = g_tol)` leaves `f_reltol`, `f_abstol`, `x_reltol` and `x_abstol` at
    # their 0.0 defaults, so `f_converged` fires on two byte-identical successive
    # objective values -- which is what "flat" means numerically, and which a
    # VARYING-scale surface (`sigma ~ x`) reaches while running away up the
    # unbounded σ_i → 0 ridge, nowhere near a stationary point. Reporting that as
    # converged is the defect #609 item 2 left open: the issue measured that this
    # route reaches its optimum (g_tol 1e-8 → 1e-16 moves the coefficients by
    # 1.3e-11) and handed the parity gap to drmTMB, but never checked the flag.
    #
    # MEASURED over a 6,400-cell varying-scale grid (n 30..400, G 3..20, σ slope
    # 0.15..8.0): 1,042 fits returned `converged = true` with the GRADIENT
    # criterion false; the true gradient ∞-norm there exceeded 1e-3 in 95 of them
    # and 1.0 in 45, worst 3.7e137 (with a POSITIVE Gaussian loglik of +980).
    #
    # `Optim.g_converged` is exactly the right test, not an approximation of it:
    # over the same grid `Optim.g_residual(res)` equalled
    # `maximum(abs, ForwardDiff.gradient(nll, θ̂))` with max absolute difference
    # 0.0 across 5,349 gradient-converged fits, and the largest such norm was
    # 9.996e-9 -- inside `g_tol`. Restarting LBFGS from the stalled point was
    # measured and REJECTED: of those 1,042 it recovered 665, left 377, made the
    # objective WORSE in 143, and produced NaN. So only the reported flag changes
    # here -- θ̂, the ML/REML objective and logLik are byte-identical.
    # Guard: test/test_ranef_varying_scale_convergence.jl.
    converged = Optim.converged(res) && Optim.g_converged(res)
    # Profile intervals reuse the ML Woodbury nll (same convention as FE REML).
    fit = _withranef(_withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, converged, means, obs, scales), nll_ml), re)
    if reml
        return _withreml(fit, -nll_reml(θ̂), -nll_ml(θ̂))
    end
    return fit
end

# Correlated random intercept+slope (1 + x | g): per group (b0,b1) ~ N(0, Σ_re),
# Σ_re a 2×2 covariance (log-Cholesky parameters a, b, c). Groups are disjoint, so
# the Woodbury capacitance is block-diagonal in 2×2 blocks → O(G), explicit 2×2
# inverse/solve/det (ForwardDiff-friendly). θ = [β_μ; β_σ; a, b, c].
#
# NUMERICALLY STABLE FORM (#762, #707). The historical objective formed
# M_k = Σ_re⁻¹ + Z_kᵀD⁻¹Z_k and took log(m11·m22 − m21²) plus the quadratic
# r′D⁻¹r − c_kᵀM_k⁻¹c_k. Both are differences of large, nearly equal numbers:
# with an uncentred covariate b22 ≈ x̄²·b11 so m11·m22 ≈ m21², and as ρ → ±1
# Σ_re⁻¹ blows up; the "determinant" then came out NEGATIVE (−3.5e41 on #707's
# cell 10) and `log` threw a DomainError (#762), or an Inf/NaN reached LBFGS's
# line search (AssertionError, #707). Both are evaluated here in the WHITENED
# coordinates v = L⁻¹b (Σ_re = L Lᵀ), with z̃_i = Lᵀ(1, x_i) and
# A_k = I + P_k, P_k = Σ_i z̃_i z̃_iᵀ/D_i:
#   G·logdetΣ_re + Σ_k logdet M_k = Σ_k logdet A_k,
#   det A_k = 1 + tr P_k + det P_k,  det P_k = (l11·l22)²·b11_k·Σ_i (x_i − x̄_k)²/D_i
# (every term ≥ 0; x̄_k is the D⁻¹-weighted group mean), and the quadratic is the
# penalised RSS Σ_i (r_i − z̃_iᵀv̂_k)²/D_i + Σ_k ‖v̂_k‖² at the conditional mode
# (non-negative; one refinement step, as in `_re_quad_stable`). Identical to the
# historical expressions in exact arithmetic.
function _corr_re_stable(r::AbstractVector, invD::AbstractVector, xs::AbstractVector,
                         gidx::AbstractVector{<:Integer}, G::Int, l11, l22, cc)
    T = promote_type(eltype(r), eltype(invD), typeof(l11), typeof(l22), typeof(cc))
    b11 = zeros(T, G); b21 = zeros(T, G)
    @inbounds for i in eachindex(r)
        k = gidx[i]; b11[k] += invD[i]; b21[k] += invD[i] * xs[i]
    end
    ssx = zeros(T, G); p11 = zeros(T, G); p21 = zeros(T, G); p22 = zeros(T, G)
    c1 = zeros(T, G); c2 = zeros(T, G)
    @inbounds for i in eachindex(r)
        k = gidx[i]; w = invD[i]; x = xs[i]
        dx = x - b21[k] / b11[k]
        ssx[k] += w * dx * dx
        z1 = l11 + cc * x; z2 = l22 * x               # z̃_i = Lᵀ(1, x_i)
        p11[k] += w * z1 * z1; p21[k] += w * z1 * z2; p22[k] += w * z2 * z2
        c1[k] += w * r[i] * z1; c2[k] += w * r[i] * z2
    end
    dL2 = (l11 * l22)^2
    detA = Vector{T}(undef, G); v1 = Vector{T}(undef, G); v2 = Vector{T}(undef, G)
    logdetA = zero(T)
    @inbounds for k in 1:G
        detA[k] = 1 + p11[k] + p22[k] + dL2 * b11[k] * ssx[k]
        logdetA += log(detA[k])
        v1[k] = ((1 + p22[k]) * c1[k] - p21[k] * c2[k]) / detA[k]
        v2[k] = (-p21[k] * c1[k] + (1 + p11[k]) * c2[k]) / detA[k]
    end
    # One refinement step on v̂ (residual of A v = c̃ re-formed from observations).
    g1 = zeros(T, G); g2 = zeros(T, G)
    @inbounds for i in eachindex(r)
        k = gidx[i]; w = invD[i]; x = xs[i]
        z1 = l11 + cc * x; z2 = l22 * x
        e = r[i] - z1 * v1[k] - z2 * v2[k]
        g1[k] += w * e * z1; g2[k] += w * e * z2
    end
    quad = zero(T)
    @inbounds for k in 1:G
        h1 = g1[k] - v1[k]; h2 = g2[k] - v2[k]
        v1[k] += ((1 + p22[k]) * h1 - p21[k] * h2) / detA[k]
        v2[k] += (-p21[k] * h1 + (1 + p11[k]) * h2) / detA[k]
        quad += v1[k]^2 + v2[k]^2
    end
    @inbounds for i in eachindex(r)
        k = gidx[i]; x = xs[i]
        e = r[i] - (l11 + cc * x) * v1[k] - l22 * x * v2[k]
        quad += invD[i] * e * e
    end
    return logdetA, quad, v1, v2
end

# Cancellation-free X′V⁻¹X for the REML term on the correlated (1 + x | g) route,
# V = D + Z Σ_re Z′. Column by column, X[:, j] is treated as a response in
# `_corr_re_stable`, whose whitened conditional mode v̂_j gives the penalised-SS form
#     (X′V⁻¹X)_jl = Σ_i (X_ij − z̃_i′v̂_{g_i,j})(X_il − z̃_i′v̂_{g_i,l})/D_i + Σ_k v̂_kj′v̂_kl,
# a sum of PSD terms (no Woodbury subtraction; same construction as `_re_xtvinvx_stable`).
function _corr_re_xtvinvx_stable(X::AbstractMatrix, invD::AbstractVector, xs::AbstractVector,
                                 gidx::AbstractVector{<:Integer}, G::Int, l11, l22, cc)
    p = size(X, 2)
    T = promote_type(eltype(X), eltype(invD), typeof(l11), typeof(l22), typeof(cc))
    V1 = Vector{Vector{T}}(undef, p); V2 = Vector{Vector{T}}(undef, p)
    for j in 1:p
        _, _, V1[j], V2[j] = _corr_re_stable(X[:, j], invD, xs, gidx, G, l11, l22, cc)
    end
    A = zeros(T, p, p)
    @inbounds for j in 1:p, l in j:p
        acc = zero(T)
        for k in 1:G
            acc += V1[j][k] * V1[l][k] + V2[j][k] * V2[l][k]
        end
        for i in axes(X, 1)
            k = gidx[i]; x = xs[i]
            z1 = l11 + cc * x; z2 = l22 * x
            ej = X[i, j] - z1 * V1[j][k] - z2 * V2[j][k]
            el = X[i, l] - z1 * V1[l][k] - z2 * V2[l][k]
            acc += invD[i] * ej * el
        end
        A[j, l] = acc; A[l, j] = acc
    end
    return A
end

function _fit_correlated_ranef_gaussian(fam::Gaussian, y, Xμ, Xσ, gidx, G, xs, nmμ, nmσ, grp, g_tol;
                                      reml::Bool = false)
    n = length(y)
    pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    # `xv` is the random-slope covariate: `xs` itself (the reported parametrisation)
    # or `xs .- x̄` (the optimisation parametrisation, below).
    function nll_x(θ, xv, Xm; reml::Bool = false, ldshift = 0.0)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]
        a = θ[pμ+pσ+1]; b = θ[pμ+pσ+2]; cc = θ[pμ+pσ+3]
        ημ = Xm * βμ; ησ = Xσ * βσ
        l11 = exp(a); l22 = exp(b)                 # L = [l11 0; cc l22], Σ_re = L Lᵀ
        r = y .- ημ
        invD = exp.(-2 .* ησ)
        logdetA, quad, _, _ = _corr_re_stable(r, invD, xv, gidx, G, l11, l22, cc)
        val = 0.5 * (sum(2 .* ησ) + logdetA + quad) + 0.5 * n * log(2π)
        if reml
            # Patterson–Thompson: + ½ logdet(Xμ′V⁻¹Xμ) − ½ pμ log(2π). `ldshift` = log|det R|
            # maps the QR-preconditioned design back to Xμ (Xμ = Q R, so
            # logdet(Xμ′V⁻¹Xμ) = logdet(Q′V⁻¹Q) + 2 log|det R|). A non-PD restriction
            # matrix gets the same large finite barrier as the intercept route.
            XtVinvX = _corr_re_xtvinvx_stable(Xm, invD, xv, gidx, G, l11, l22, cc)
            cholX = cholesky(Symmetric(XtVinvX); check = false)
            issuccess(cholX) || return oftype(val, val + REML_NONPD_PENALTY)
            ldX = logdet(cholX)
            isfinite(ldX) || return oftype(val, val + REML_NONPD_PENALTY)
            val = val + 0.5 * ldX + ldshift - 0.5 * size(Xm, 2) * log(2π)
        end
        # A line-search probe far outside the data scale can still overflow
        # (e.g. exp(a) → Inf). Return a large FINITE barrier: LBFGS's HagerZhang
        # line search asserts a finite objective (#707), and a thrown error is not
        # an answer either (#762).
        isfinite(val) || return oftype(val, 1e18)
        return val
    end
    nll_ml(θ) = nll_x(θ, xs, Xμ)
    nll(θ) = reml ? nll_x(θ, xs, Xμ; reml = true) : nll_ml(θ)

    # OPTIMISE IN CENTRED COORDINATES (#762). b0 + b1·x = (b0 + b1·x̄) + b1·(x − x̄),
    # so the model is invariant to shifting the slope covariate; only the Cholesky
    # parametrisation of Σ_re changes. With an uncentred covariate (x̄ ≈ 27 while
    # sd(x) ≈ 2) the intercept variance at x = 0 and the intercept–slope
    # correlation (→ ∓1) make the log-Cholesky surface badly scaled, and LBFGS
    # stopped up to 1.6 logLik units short of drmTMB while reporting convergence.
    # Fit Σ_c = L_c L_cᵀ for (1, x − x̄), then map back EXACTLY:
    #   Σ = S Σ_c Sᵀ, S = [1 −x̄; 0 1];  L11 = √Σ11, L21 = Σ21/L11,
    #   L22 = det(L_c)/L11 = l_c11·l_c22/L11 (no subtraction, so no cancellation
    #   when Σ is near-singular).
    #
    # The mean design is preconditioned the same way: optimise γ = Rμ·βμ against
    # the orthonormal Qμ (Xμ = Qμ·Rμ, thin QR), which makes the problem for data
    # with x and for data with x − x̄ identical and removes the intercept–slope
    # coupling of an uncentred mean covariate. Reported βμ = Rμ⁻¹γ. A rank-
    # deficient Xμ keeps the identity (the pre-existing behaviour).
    x̄ = sum(xs) / length(xs)
    xc = xs .- x̄
    Qμ, Rμ = let F = qr(Xμ)
        R = Matrix(F.R); dR = abs.(diag(R))
        if length(dR) == pμ && minimum(dR) > 1e-10 * maximum(dR)
            Matrix(F.Q)[:, 1:pμ], R
        else
            Xμ, Matrix{Float64}(I, pμ, pμ)
        end
    end
    ldR = sum(log ∘ abs, diag(Rμ))
    nllc(φ) = reml ? nll_x(φ, xc, Qμ; reml = true, ldshift = ldR) : nll_x(φ, xc, Qμ)
    βμ0 = Xμ \ y; res0 = y - Xμ * βμ0
    φ0 = zeros(pμ + pσ + 3)
    φ0[1:pμ] .= Rμ * βμ0
    φ0[pμ+1] = log(std(res0) + eps())
    sd0 = log(std(res0) / 2 + eps())
    φ0[pμ+pσ+1] = sd0; φ0[pμ+pσ+2] = sd0; φ0[pμ+pσ+3] = 0.0
    res = Optim.optimize(nllc, φ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    # BOUNDARY RESTART. Σ_re depends on l22 only through l22², so ρ = ±1 (l22 → 0,
    # log l22 → −∞) is ALWAYS a stationary limit of this parametrisation: the
    # gradient in log l22 vanishes there whether or not an interior optimum is
    # better. LBFGS can drift into it (measured: 3/60 H0 refits stopped 0.003–0.005
    # logLik short of drmTMB with ||g|| ≈ 1e-9). When the fit lands on that edge,
    # restart once from an interior point (l22 = l11/2, ρ reset to 0) and keep
    # the better objective; a genuine boundary optimum returns to the edge.
    let φ̂1 = Optim.minimizer(res), ia = pμ + pσ + 1
        if φ̂1[ia+1] - φ̂1[ia] < log(1e-3)
            φr = copy(φ̂1); φr[ia+1] = φ̂1[ia] + log(0.5); φr[ia+2] = 0.0
            res2 = Optim.optimize(nllc, φr, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
            res = _better_restart(nllc, res, res2)   # NOT Optim.minimum: see optim_minimum_guard.jl
        end
    end
    φ̂ = Optim.minimizer(res)
    θ̂ = let
        ac, bc, lcc = φ̂[pμ+pσ+1], φ̂[pμ+pσ+2], φ̂[pμ+pσ+3]
        lc11 = exp(ac); lc22 = exp(bc)
        m11 = lc11 - x̄ * lcc; m12 = -x̄ * lc22        # first row of S·L_c
        Σ11 = m11^2 + m12^2
        Σ21 = m11 * lcc + m12 * lc22
        L11 = sqrt(Σ11)
        θ = copy(φ̂)
        θ[1:pμ] .= Rμ \ φ̂[1:pμ]
        θ[pμ+pσ+1] = log(L11)
        θ[pμ+pσ+2] = ac + bc - log(L11)
        θ[pμ+pσ+3] = Σ21 / L11
        θ
    end
    V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :recov => (pμ+pσ+1):(pμ+pσ+3)]
    names = [:mu => nmμ, :sigma => nmσ, :recov => ["$(grp):L11", "$(grp):L22", "$(grp):L21"]]
    means = Dict(:mu => Xμ * θ̂[1:pμ]); obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]))
    # Conditional RE estimates (BLUPs): per group b̂_k = M_k⁻¹ c_k = L v̂_k
    # (intercept, slope), from the same stable whitened solve the nll uses.
    blup = let
        βμ = θ̂[1:pμ]; βσ = θ̂[pμ+1:pμ+pσ]
        l11 = exp(θ̂[pμ+pσ+1]); l22 = exp(θ̂[pμ+pσ+2]); cc = θ̂[pμ+pσ+3]
        r = y .- Xμ * βμ; invD = exp.(-2 .* (Xσ * βσ))
        _, _, v1, v2 = _corr_re_stable(r, invD, xs, gidx, G, l11, l22, cc)
        hcat(l11 .* v1, cc .* v1 .+ l22 .* v2)
    end
    re = Dict(Symbol(grp) => blup)
    fit = _withranef(_withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll_ml), re)
    reml && return _withreml(fit, -nll(θ̂), -nll_ml(θ̂))
    return fit
end

"""
    ranef(fit) -> Dict{Symbol,...}

Per-level conditional random-effect estimates (BLUPs), keyed by grouping factor.
These are the posterior means of the random effects at the fitted variance
components — drmTMB's `ranef()`.

- Scalar random intercept `(1 | g)`: a `Vector` of length `n_levels(g)`.
- Correlated `(1 + x | g)`: an `n_levels × 2` matrix (`[intercept slope]`).
- Multiple components `(1 | g) + (1 | h)`: one entry per factor.

Populated for the Gaussian closed-form RE paths (exact GLS conditional means)
and the structured (phylo/relmat/animal) routes. Returns an empty `Dict` for a
fit with **no** random-effect block at all. A non-Gaussian GLMM fitted by the
Gauss–Hermite/Laplace marginal routes (e.g. `Binomial`/`Poisson` with `(1|g)`)
never computes conditional modes — those routes integrate the random effect
out of the marginal likelihood without ever forming a posterior mode — so
calling `ranef` on such a fit throws an `ArgumentError` instead of silently
returning an empty `Dict`.
"""
function ranef(fit::DrmFit)
    if fit.ranef === nothing
        has_re_block = any(p -> first(p) in (:resd, :recov, :sd, :sd_phylo), fit.blocks)
        has_re_block && throw(ArgumentError("ranef: this fit has a random-effect block " *
            "but no stored conditional modes -- the non-Gaussian GLMM marginal route " *
            "(Gauss-Hermite quadrature / Laplace) integrates the random effect out of " *
            "the likelihood and never computes a posterior mode. Use `re_sd(fit)` / " *
            "`vc(fit)` for the fitted variance components."))
        return Dict{Symbol,Vector{Float64}}()
    end
    if fit.ranef isa NamedTuple && haskey(fit.ranef, :effects)
        return fit.ranef.effects
    end
    return fit.ranef
end

"""
    vc(fit) -> Dict{Symbol,Matrix{Float64}}

Random-effect covariance summary per grouping factor.

- Correlated random-effect block (`(1 + x | g)`): a 2×2 covariance matrix —
  `sqrt.(diag(vc(fit)[:g]))` are the intercept/slope SDs, the off-diagonal their
  covariance.
- Scalar / structured variance components (`(1 | g)`, `relmat`/`animal`/`phylo`):
  a 1×1 matrix holding that component's variance `σ²`. A fit with two structured
  components (e.g. `phylo(1|species) + relmat(1|id)`) reports both, keyed by
  grouping factor.
"""
function vc(fit::DrmFit)
    # Location-scale-scale fits (#544): the RE variance varies by group-level
    # covariates, so a single component matrix is ill-defined -- refuse.
    any(p -> first(p) in (:sd, :sd_phylo), fit.blocks) &&
        throw(ArgumentError("vc: this fit models the random-effect SD with covariates " *
            "(`sd(group) ~ ...`); use `coef(fit, :sd)` for the log-SD coefficients."))
    d = Dict{Symbol,Matrix{Float64}}()
    # q=4 phylogenetic coevolution: the raw 4×4 group-level Σ_a is stashed on
    # `ranef` (axes mu1,mu2,sigma1,sigma2); surface it here per #192.
    if fit.ranef isa NamedTuple && haskey(fit.ranef, :Sigma_a)
        d[Symbol(fit.ranef.group)] = Matrix{Float64}(fit.ranef.Sigma_a)
    end
    for (p, r) in fit.blocks
        if p === :recov
            a, b, cc = fit.theta[r]
            l11 = exp(a); l22 = exp(b)
            Σ = [l11^2 cc*l11; cc*l11 cc^2+l22^2]
            nm = first(cn[2] for cn in fit.coefnames if cn[1] === :recov)[1]   # "g:L11"
            d[Symbol(split(nm, ":")[1])] = Σ
        elseif p === :resd
            nms = first(cn[2] for cn in fit.coefnames if cn[1] === :resd)
            for (j, nm) in enumerate(nms)
                σ = exp(fit.theta[r[j]])
                d[Symbol(nm)] = fill(σ^2, 1, 1)            # 1×1 variance component
            end
        end
    end
    return d
end

# Multiple independent scalar random-effect components (e.g. (1|g) + (1|h)).
# comps :: Vector of (w, gidx, Gk, label). Marginal V = D + Σ_k σ_k² Z_k Z_kᵀ.
# Whitened Woodbury: fold σ into a scaled design Z̃ = Z·diag(σ_per_col), giving a
# small q×q (q = Σ G_k) capacitance M = I + Z̃ᵀD⁻¹Z̃ (the logdet(G) term is
# absorbed into logdet(M)). In exact arithmetic M is identity-plus-PSD hence PD,
# but at extreme σ the I is lost to rounding (M's entries ≫ 1) and the raw
# Z̃ᵀD⁻¹Z̃ is rank-deficient (crossed intercept columns). Forming M + I and
# Cholesky-factoring it then fails — or, worse, SUCCEEDS with pivots that are
# differences of O(1/σ_e²) numbers, so logdet(M + I) and the Woodbury quadratic
# r′D⁻¹r − c′(M + I)⁻¹c lose every digit as σ_e → 0 with one record per level
# (the #835/#837 cancellation class; measured: nll off by 0.63 at log σ_e = −16
# and −0.44 in logdet alone at −18, test_cancellation_sweep.jl). We therefore
# never form M: `_multi_re_qr` takes the QR factorisation of the stacked design
# [D^{-1/2}Z̃; I] (RᵀR = I + Z̃ᵀD⁻¹Z̃ exactly, without squaring the condition
# number), reads logdet(M + I) = 2Σ log|Rᵢᵢ|, and evaluates the quadratic as the
# penalised RSS Σ (rᵢ − z̃ᵢᵀb̂)²/Dᵢ + ‖b̂‖² at the least-squares mode b̂ (every term
# ≥ 0). Identical to the Woodbury expressions in exact arithmetic. Closed-form
# GLS; Z precomputed.
function _multi_re_qr(Z̃::AbstractMatrix, sdinv::AbstractVector, r::AbstractVector)
    T = promote_type(eltype(Z̃), eltype(sdinv), eltype(r))
    q = size(Z̃, 2)
    F = qr([sdinv .* Z̃; Matrix{T}(I, q, q)])
    bhat = F \ [sdinv .* r; zeros(T, q)]
    R = F.R
    logdetM = 2 * sum(log ∘ abs, diag(R))
    return R, bhat, logdetM
end

function _fit_multi_ranef_gaussian(fam::Gaussian, y, Xμ, Xσ, comps, nmμ, nmσ, g_tol)
    n = length(y); pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    K = length(comps)
    Gks = [c[3] for c in comps]
    q = sum(Gks)
    offs = cumsum([0; Gks])
    Z = zeros(n, q)
    colcomp = Vector{Int}(undef, q)
    for (k, (w, gidx, Gk, _)) in enumerate(comps)
        for c in 1:Gk
            colcomp[offs[k]+c] = k
        end
        @inbounds for i in 1:n
            Z[i, offs[k]+gidx[i]] = w[i]
        end
    end
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]
        ημ = Xμ * βμ
        ησ = clamp.(Xσ * βσ, -30.0, 30.0)          # bound σ predictor: keep exp finite
        invD = exp.(-2 .* ησ); r = y .- ημ
        σk = [exp(clamp(θ[pμ+pσ+k], -30.0, 30.0)) for k in 1:K]
        σcol = [σk[colcomp[c]] for c in 1:q]
        Z̃ = Z .* σcol'                             # scale each column by its σ_k
        _, bhat, logdetM = _multi_re_qr(Z̃, exp.(-ησ), r)
        e = r .- Z̃ * bhat
        quad = sum(invD .* e .^ 2) + sum(abs2, bhat)
        logdetV = sum(2 .* ησ) + logdetM
        val = 0.5 * (logdetV + quad) + 0.5 * n * log(2π)
        isfinite(val) || return oftype(val, 1e18)  # HagerZhang asserts a finite objective
        return val
    end

    function grad!(Gout, θ)
        fill!(Gout, 0.0)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]
        ημ = Xμ * βμ
        ησ = clamp.(Xσ * βσ, -30.0, 30.0)
        invD = exp.(-2 .* ησ)
        r = y .- ημ
        σk = [exp(clamp(θ[pμ+pσ+k], -30.0, 30.0)) for k in 1:K]
        σcol = [σk[colcomp[c]] for c in 1:q]
        Z̃ = Z .* σcol'
        R, bscaled, _ = _multi_re_qr(Z̃, exp.(-ησ), r)   # RᵀR = I + Z̃ᵀD⁻¹Z̃
        all(isfinite, bscaled) || return Gout
        Rinv = UpperTriangular(R) \ Matrix{Float64}(I, q, q)
        Hinv = Rinv * Rinv'
        α = invD .* (r .- Z̃ * bscaled)             # V⁻¹r

        Gout[1:pμ] .= -(Xμ' * α)

        # d nll / d ησᵢ = Dᵢ[(V⁻¹)ᵢᵢ - αᵢ²], Dᵢ = exp(2ησᵢ).
        lever = vec(sum((Z̃ * Hinv) .* Z̃, dims = 2))
        diagVinv = invD .- invD .^ 2 .* lever
        Gout[pμ+1:pμ+pσ] .= Xσ' * ((diagVinv .- α .^ 2) ./ invD)

        # With whitened Z̃, dV/dlogσₖ = 2 Z̃ₖZ̃ₖᵀ, and
        # Z̃ᵀV⁻¹Z̃ = I - (I + Z̃ᵀD⁻¹Z̃)⁻¹.
        dH = diag(Hinv)
        @inbounds for k in 1:K
            cols = (offs[k]+1):offs[k+1]
            Gout[pμ+pσ+k] = sum(1 .- dH[cols]) - sum(abs2, bscaled[cols])
        end
        return Gout
    end
    βμ0 = Xμ \ y; res0 = y - Xμ * βμ0
    s0 = std(res0) / sqrt(K + 1)                   # balanced variance split: resid + K REs
    θ0 = zeros(pμ + pσ + K)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(s0 + eps())
    for k in 1:K
        θ0[pμ+pσ+k] = log(s0 + eps())
    end
    od = Optim.OnceDifferentiable(nll, grad!, θ0)
    res = Optim.optimize(od, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol))
    θ̂ = Optim.minimizer(res)
    # vcov from the AD Hessian of `nll`. `nll` returns a flat 1e18 penalty when
    # `cholesky(M + I)` fails, so ForwardDiff's directional probes near a
    # near-singular capacitance (extreme σ ratios) can straddle that branch and
    # pollute the Hessian. Detect the leak (θ̂ on the penalty, or a non-finite /
    # non-PD Hessian) and report NaN rather than a silently corrupted vcov,
    # matching the phylo/sparse paths.
    V = let
        pen = oftype(sum(θ̂), 1e18)
        Hθ = ForwardDiff.hessian(nll, θ̂)
        if nll(θ̂) >= pen || !all(isfinite, Hθ) || !isposdef(Symmetric(Hθ))
            @warn "multi-RE Gaussian vcov: Hessian is not usable (near-singular " *
                  "capacitance or penalty leak); returning NaN standard errors."
            fill(NaN, length(θ̂), length(θ̂))
        else
            Matrix(inv(Symmetric(Hθ)))
        end
    end
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :resd => (pμ+pσ+1):(pμ+pσ+K)]
    names = [:mu => nmμ, :sigma => nmσ, :resd => [c[4] for c in comps]]
    means = Dict(:mu => Xμ * θ̂[1:pμ]); obs = Dict(:mu => Vector{Float64}(y))
    # Report σ from the SAME clamped predictor the likelihood used (the objective
    # evaluates `clamp.(Xσ*βσ, -30, 30)`), so `sigma(fit)` matches the fitted σ
    # rather than an unclamped value the model never scored. Warn when any fitted
    # ησ hits the clamp boundary: there the objective is gradient-flat and the
    # ForwardDiff Hessian collapses the σ block, so Wald SEs are not trustworthy.
    ησ̂ = Xσ * θ̂[(pμ+1):(pμ+pσ)]
    any(abs.(ησ̂) .>= 30.0) &&
        @warn "multi-RE Gaussian: a fitted σ predictor hit the ±30 log-σ clamp; the " *
              "objective is flat there, so the reported σ is the clamped value and the " *
              "scale-coefficient Wald SEs are unreliable."
    scales = Dict(:sigma => exp.(clamp.(ησ̂, -30.0, 30.0)))
    # Conditional RE estimates (BLUPs) per component. The whitened solve C\ZtDir is
    # in σ-scaled units; the BLUP is b̂ = σ_col ⊙ (C \ ZtDir), split back per factor.
    blup = let
        βμ = θ̂[1:pμ]; βσ = θ̂[pμ+1:pμ+pσ]
        ησ = clamp.(Xσ * βσ, -30.0, 30.0); invD = exp.(-2 .* ησ); r = y .- Xμ * βμ
        σk = [exp(clamp(θ̂[pμ+pσ+k], -30.0, 30.0)) for k in 1:K]
        σcol = [σk[colcomp[c]] for c in 1:q]
        Z̃ = Z .* σcol'
        ZtDir = Z̃' * (invD .* r)
        M = Z̃' * (invD .* Z̃)
        bscaled = cholesky(Symmetric(M + I)) \ ZtDir   # at θ̂ this is well-conditioned
        bfull = σcol .* bscaled                         # back to natural RE scale
        d = Dict{Symbol,Vector{Float64}}()
        for (k, c) in enumerate(comps)
            d[Symbol(c[4])] = bfull[(offs[k]+1):offs[k+1]]
        end
        d
    end
    return _withranef(_withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll, grad!), blup)
end

# Gauss–Hermite nodes/weights (Golub–Welsch) for ∫ h(x) e^{-x²} dx ≈ Σ wₖ h(xₖ).
function _gauss_hermite(K::Int)
    β = [sqrt(k / 2) for k in 1:(K-1)]
    E = eigen(SymTridiagonal(zeros(K), β))
    return E.values, sqrt(π) .* (E.vectors[1, :]) .^ 2
end

# Random intercept on the SCALE: sigma ~ <fixed> + (1 | g), with
# log σᵢ = Xσᵢᵀβσ + b_{g(i)}, b_g ~ N(0, σ_b²); the mean is fixed effects. There
# is no closed-form marginal (b enters σ nonlinearly), so each group's random
# effect is integrated out numerically. The DEFAULT (`marginal = :LA`, D-273)
# is unchanged: a fixed 32-node PRIOR-scale grid (b = √2 σ_b z), independent of
# where the group posterior actually sits — accurate for small groups, but
# tens of nats off for a large group SD (see D-273's receipt,
# docs/dev-log/evidence/arc2-sigma-re-laplace/receipt.md, and the AGHQ PR's own
# demo numbers). `marginal = :Laplace` (already implemented) swaps in the
# closed-form Laplace approximation that native drmTMB (TMB) computes for this
# model. `marginal = :AGHQ` (new) swaps in the per-group ADAPTIVE
# Gauss-Hermite helper (`_aghq_marginal_loglik`, #834/#719) at K =
# `_RANEF1D_AGHQ_K` nodes: the per-observation log-density at a shifted log σ
# is `ll(i, η) = -½log2π - η - ½ rᵢ² e^{-2η}`, η0 = Xσβσ, the random-effect
# design is a constant 1 (Zre = ones(n,1)), and the prior scale is L = [σ_b].
# `:Laplace` is exactly the K=1 case of this same adaptive grid (hand-derived
# there for speed); `:AGHQ` is the more accurate sibling for informative groups
# without committing to the Laplace/TMB-parity numbers. Everything else (start
# values, optimiser, blocks, names, reported σ) is shared across all three.
function _fit_sigma_ranef_gaussian(fam::Gaussian, y, Xμ, Xσ, gidx, G, nmμ, nmσ, grp, g_tol;
                                   laplace::Bool = false, aghq::Bool = false,
                                   K::Int = _RANEF1D_AGHQ_K)
    n = length(y); pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    z, w = _gauss_hermite(32)
    logw = log.(w); Kghq = length(z); rt2 = sqrt(2.0); lπ = log(π); l2π = log(2π)
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; σb = exp(θ[pμ+pσ+1])
        η0 = Xσ * βσ                            # fixed-effect log σ
        r = y .- Xμ * βμ
        re = r .^ 2 .* exp.(-2 .* η0)           # rᵢ² e^{-2η0ᵢ}
        T = eltype(θ)
        s = zero(T)
        for idx in members
            mg = length(idx)
            mg == 0 && continue
            Ag = sum(@view η0[idx]); Bg = sum(@view re[idx])
            terms = Vector{T}(undef, Kghq)
            for k in 1:Kghq
                δ = rt2 * σb * z[k]
                terms[k] = logw[k] - mg * δ - 0.5 * exp(-2δ) * Bg
            end
            mx = maximum(terms)
            llg = -0.5 * lπ - 0.5 * mg * l2π - Ag + mx + log(sum(exp.(terms .- mx)))
            s -= llg
        end
        return s
    end
    rule = _AGHQRule(1, K); Zre = ones(n, 1); bcache = zeros(1, G)   # #719: per-group AGHQ
    function nll_aghq(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; σb = exp(θ[pμ+pσ+1])
        η0 = Xσ * βσ                            # fixed-effect log σ
        r = y .- Xμ * βμ
        ll = (i, η) -> -0.5 * l2π - η - 0.5 * r[i]^2 * exp(-2η)
        L = reshape([σb], 1, 1)
        return -_aghq_marginal_loglik(ll, members, η0, Zre, L, rule, bcache)
    end
    obj = laplace ? _sigre_laplace_nll(y, Xμ, Xσ, members, pμ, pσ) : (aghq ? nll_aghq : nll)
    βμ0 = Xμ \ y; res0 = y - Xμ * βμ0
    θ0 = zeros(pμ + pσ + 1)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(std(res0) + eps())
    θ0[pμ+pσ+1] = log(0.5 * std(res0) + eps())   # σ_b init
    res = Optim.optimize(obj, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(obj, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :resd => (pμ+pσ+1):(pμ+pσ+1)]
    # Tag the RE-SD name with `_logsigma`: this random intercept lives on the
    # log-σ (scale) axis, so `re_sd`/`vc` surface a value that is NOT comparable to
    # a mean-axis (response-scale) random-intercept SD reported under the bare group
    # name. The suffix makes the axis explicit at the accessor level (#322).
    names = [:mu => nmμ, :sigma => nmσ, :resd => ["$(grp)_logsigma"]]
    means = Dict(:mu => Xμ * θ̂[1:pμ]); obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]))   # population (b=0) σ
    fit = _withnll(DrmFit(fam, blocks, names, θ̂, V, -obj(θ̂), n, Optim.converged(res), means, obs, scales), obj)
    laplace && return _withmarginal(fit, :Laplace)
    aghq && return _withmarginal(fit, :AGHQ)
    return fit
end

# ── marginal = :Laplace for the σ random intercept ───────────────────────────
# For group g with m members, the random effect b enters only through
#     h(b) = −m b − ½ B e^{−2b} − ½ b²/s²      (s = σ_b),
# with A = Σ η0ᵢ and B = Σ rᵢ² e^{−2η0ᵢ} as in the GHQ objective above (the
# b-free terms −½ m log 2π − A are added back per group). h is strictly concave:
# h''(b) = −2B e^{−2b} − 1/s² < 0, so the mode b̂ is unique. The Laplace
# approximation of the group's log marginal,
#     log ∫ N(y | b) N(b; 0, s²) db ≈ h_full(b̂) + ½ log 2π − ½ log(−h_full''(b̂)),
# collapses (the prior's −log s − ½ log 2π and the curvature term combine) to
#     −½ m log 2π − A + h(b̂) − ½ log1p(2 s² B e^{−2b̂}).
# This is the quantity TMB's Laplace computes for drmTMB's parameterisation
# (b = σ_b u, u ~ N(0, 1)): Laplace is invariant to that affine change of
# variable, and the groups are independent, so TMB's joint Hessian is diagonal
# and its log-determinant is this sum of per-group terms.

_sigre_primal(x::ForwardDiff.Dual) = _sigre_primal(ForwardDiff.value(x))
_sigre_primal(x::Real) = x

# Mode of h in Float64: the root of the strictly decreasing score
# f(b) = −m + B e^{−2b} − b/s². It lies between 0 and c = ½ log(B/m) (f(0) and
# f(c) have opposite signs), so Newton is safeguarded by bisection on that
# bracket and never leaves it.
function _sigre_mode(m::Int, B::Float64, s2::Float64)
    B > 0 || return -m * s2                  # every residual exactly zero
    c = 0.5 * log(B / m)
    lo, hi = min(0.0, c), max(0.0, c)
    lo == hi && return lo
    b = 0.5 * (lo + hi)
    for _ in 1:200
        e = exp(-2b)
        fb = -m + B * e - b / s2
        fb == 0 && return b
        fb > 0 ? (lo = b) : (hi = b)
        bn = b - fb / (-2B * e - 1 / s2)
        (lo < bn < hi) || (bn = 0.5 * (lo + hi))
        abs(bn - b) <= 1e-14 * (1 + abs(b)) && return bn
        b = bn
    end
    return b
end

# Laplace log marginal of one group, without the −½ m log 2π − A terms. The
# mode is found on primal values; two Newton steps are then taken in the
# caller's number type. At b̂ the score is zero, so the value is unchanged, and
# the steps carry the implicit-function derivative of b̂(θ): one step makes the
# first derivatives exact, two make the second derivatives (the Hessian used
# for the vcov) exact too.
function _sigre_laplace_group(m::Int, B, s2)
    b = _sigre_mode(m, Float64(_sigre_primal(B)), Float64(_sigre_primal(s2)))
    b = oftype(B * s2, b)
    for _ in 1:2
        e = exp(-2b)
        b -= (-m + B * e - b / s2) / (-2B * e - 1 / s2)
    end
    e = exp(-2b)
    return -m * b - 0.5 * B * e - 0.5 * b^2 / s2 - 0.5 * log1p(2 * s2 * B * e)
end

function _sigre_laplace_nll(y, Xμ, Xσ, members, pμ, pσ)
    l2π = log(2π)
    return function (θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; s2 = exp(2 * θ[pμ+pσ+1])
        η0 = Xσ * βσ
        r = y .- Xμ * βμ
        re = r .^ 2 .* exp.(-2 .* η0)
        s = zero(eltype(θ))
        for idx in members
            mg = length(idx)
            mg == 0 && continue
            Ag = sum(@view η0[idx]); Bg = sum(@view re[idx])
            s -= -0.5 * mg * l2π - Ag + _sigre_laplace_group(mg, Bg, s2)
        end
        return s
    end
end

# ── simultaneous mean + sigma random intercepts (#745, twin drmTMB #1287) ────
#
# `y ~ x + (1 | g), sigma ~ (1 | g)`, the SAME grouping factor `g` on both
# axes. Unlike `_fit_ranef_gaussian` (mean RE, `sigma` fixed effects — exact
# Woodbury marginal) this has no closed form: the sigma random effect enters
# the mean's per-observation VARIANCE, so it cannot be profiled out of the
# Gaussian integral the way `sigma ~ x` fixed effects can. drmTMB's own route
# (`src/drmTMB.cpp` model_type 1) does not exploit any closed form either — it
# hands TMB the full stacked (u_mu, u_sigma) vector and lets its black-box
# nested Laplace integrate everything at once, with independent
# `dnorm(u, 0, 1)` priors (no cross-dpar correlation for the plain `(1 | g)` +
# `(1 | g)` cell; that needs an explicit coupled `(1 | tag | group)` tag).
#
# Because both random effects here share ONE grouping factor, drmTMB's joint
# Hessian over the whole (u_mu, u_sigma) vector is exactly BLOCK-DIAGONAL, one
# 2×2 block per group: group k's likelihood contribution depends only on its
# own (b_mu,k, delta_sigma,k) pair, never on another group's. Summing an
# independent per-group 2-D Laplace approximation over those 2×2 blocks IS the
# whole-model nested Laplace approximation — nothing is lost relative to
# TMB's joint version by doing it group-by-group instead of as one big sparse
# solve; it is simply a more transparent implementation of the identical
# block-diagonal structure. (If a future formula let the two REs use
# DIFFERENT grouping factors, the blocks would no longer be 2×2 and this
# derivation would not apply — routed as a clear refusal at the call site.)
#
# For one group with m members, random intercept b (mean) and delta (sigma,
# both un-standardized — same convention as `_sigre_mode` above, b ~ N(0,σb²)
# directly rather than TMB's b = σb·u — Laplace is invariant to that affine
# reparameterisation), fixed-effect residual r_i = y_i − Xμ_i′β_μ and
# fixed-effect log σ η0_i = Xσ_i′β_σ:
#
#     h(b, δ) = Σ_i (η0_i + δ) + ½ D(b, δ) + ½ b²/σb,μ² + ½ δ²/σb,σ²
#     D(b, δ) = Σ_i (r_i − b)² exp(−2(η0_i + δ))
#
# (plus the −½m log 2π − ½log 2π − ½log 2π normalising constants for the
# response density and the two priors, added back in `nll_group` below). The
# 2×2 Hessian of h in closed form (used both for the Newton mode-finder and
# for the Laplace determinant):
#
#     H_bb = Σ_i e_i + 1/σb,μ²             (e_i = exp(−2(η0_i+δ)))
#     H_bδ = −∂D/∂b = 2 Σ_i (r_i−b) e_i
#     H_δδ = 2 D + 1/σb,σ²
#
# and Laplace's 2-D formula log∫∫e^{−h} db dδ ≈ −h(b̂,δ̂) + log(2π) −
# ½log det H(b̂,δ̂).

_musig_primal(x::ForwardDiff.Dual) = _musig_primal(ForwardDiff.value(x))
_musig_primal(x::Real) = x

# Damped-Newton 2-D mode-finder on stripped Float64 primals. Groups here are
# tiny (drmTMB twin cells run G ≈ 8..60, n_g ≈ 6..12), so this converges in a
# handful of iterations; step-halving on the gradient norm guards against the
# occasional overshoot from the δ-nonlinearity (e = exp(−2δ)) far from (0,0).
function _musig_mode_primal(r_idx::Vector{Float64}, eta0_idx::Vector{Float64},
                            sb_mu2::Float64, sb_sigma2::Float64)
    m = length(r_idx)
    b = 0.0; δ = 0.0
    for _ in 1:100
        e = exp.(-2 .* (eta0_idx .+ δ))
        resid = r_idx .- b
        D = sum(resid .^ 2 .* e)
        dDdb = -2 * sum(resid .* e)
        gb = 0.5 * dDdb + b / sb_mu2
        gδ = m - D + δ / sb_sigma2
        (gb^2 + gδ^2) < 1e-24 && break
        Hbb = sum(e) + 1 / sb_mu2
        Hbδ = -dDdb
        Hδδ = 2 * D + 1 / sb_sigma2
        detH = Hbb * Hδδ - Hbδ^2
        if !isfinite(detH) || detH <= 0
            Δb = -gb / max(Hbb, 1e-8)
            Δδ = -gδ / max(Hδδ, 1e-8)
        else
            Δb = -(Hδδ * gb - Hbδ * gδ) / detH
            Δδ = -(-Hbδ * gb + Hbb * gδ) / detH
        end
        step = 1.0
        g0 = gb^2 + gδ^2
        while step > 1e-8
            bn = b + step * Δb; δn = δ + step * Δδ
            en = exp.(-2 .* (eta0_idx .+ δn))
            residn = r_idx .- bn
            Dn = sum(residn .^ 2 .* en)
            gbn = 0.5 * (-2 * sum(residn .* en)) + bn / sb_mu2
            gδn = m - Dn + δn / sb_sigma2
            if gbn^2 + gδn^2 <= g0 || step < 1e-3
                b, δ = bn, δn
                break
            end
            step *= 0.5
        end
    end
    return b, δ
end

# Mode + final Hessian pieces in the CALLER's number type (mirrors
# `_sigre_laplace_group`'s pattern): find (b̂, δ̂) robustly on stripped Float64
# primals, then take two exact Newton steps in the full (possibly Dual) type
# so the implicit function theorem gives ForwardDiff the correct derivative of
# (b̂, δ̂) w.r.t. θ. Returns (b, δ, D, Hbb, Hbδ, Hδδ) at the refined point.
function _musig_refine(r_idx, eta0_idx, sb_mu2, sb_sigma2, m::Int)
    r0 = _musig_primal.(r_idx); eta00 = _musig_primal.(eta0_idx)
    sb_mu2_0 = _musig_primal(sb_mu2); sb_sigma2_0 = _musig_primal(sb_sigma2)
    b0, δ0 = _musig_mode_primal(r0, eta00, sb_mu2_0, sb_sigma2_0)
    T = promote_type(eltype(r_idx), eltype(eta0_idx), typeof(sb_mu2), typeof(sb_sigma2))
    b = oftype(one(T), b0); δ = oftype(one(T), δ0)
    for _ in 1:2
        e = exp.(-2 .* (eta0_idx .+ δ))
        resid = r_idx .- b
        D = sum(resid .^ 2 .* e)
        dDdb = -2 * sum(resid .* e)
        gb = 0.5 * dDdb + b / sb_mu2
        gδ = m - D + δ / sb_sigma2
        Hbb = sum(e) + 1 / sb_mu2
        Hbδ = -dDdb
        Hδδ = 2 * D + 1 / sb_sigma2
        detH = Hbb * Hδδ - Hbδ^2
        Δb = -(Hδδ * gb - Hbδ * gδ) / detH
        Δδ = -(-Hbδ * gb + Hbb * gδ) / detH
        b += Δb; δ += Δδ
    end
    e = exp.(-2 .* (eta0_idx .+ δ))
    resid = r_idx .- b
    D = sum(resid .^ 2 .* e)
    dDdb = -2 * sum(resid .* e)
    Hbb = sum(e) + 1 / sb_mu2
    Hbδ = -dDdb
    Hδδ = 2 * D + 1 / sb_sigma2
    return b, δ, D, Hbb, Hbδ, Hδδ
end

"""
    _fit_musigma_ranef_gaussian(fam, y, Xμ, Xσ, gidx, G, nmμ, nmσ, grp, g_tol) -> DrmFit

Gaussian model with SIMULTANEOUS random intercepts on the mean `(1 | g)` and
on `sigma` `(1 | g)`, sharing one grouping factor (#745, twin drmTMB #1287).
Each group's joint (b_mu, delta_sigma) pair is integrated by a 2-D Laplace
approximation (see the derivation above this function); marginal `:Laplace`
always — there is no closed-form or quadrature alternative implemented for
this cell. θ = [β_μ; β_σ (fixed-effect log σ); log σ_b,μ; log σ_b,σ].
"""
function _fit_musigma_ranef_gaussian(fam::Gaussian, y, Xμ, Xσ, gidx, G, nmμ, nmσ, grp, g_tol)
    n = length(y); pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    l2π = log(2π)

    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]
        sb_mu2 = exp(2 * θ[pμ+pσ+1]); sb_sigma2 = exp(2 * θ[pμ+pσ+2])
        r_all = y .- Xμ * βμ
        η0_all = Xσ * βσ
        T = eltype(θ)
        s = zero(T)
        for idx in members
            m = length(idx)
            m == 0 && continue
            r_idx = r_all[idx]; eta0_idx = η0_all[idx]
            b, δ, D, Hbb, Hbδ, Hδδ = _musig_refine(r_idx, eta0_idx, sb_mu2, sb_sigma2, m)
            detH = Hbb * Hδδ - Hbδ^2
            Ak = sum(eta0_idx)
            nll_group = 0.5 * m * l2π + Ak + m * δ + 0.5 * D +
                        0.5 * l2π + θ[pμ+pσ+1] + 0.5 * b^2 / sb_mu2 +
                        0.5 * l2π + θ[pμ+pσ+2] + 0.5 * δ^2 / sb_sigma2
            s += nll_group - l2π + 0.5 * log(detH)
        end
        return s
    end

    βμ0 = Xμ \ y; res0 = y - Xμ * βμ0
    θ0 = zeros(pμ + pσ + 2)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(std(res0) + eps())
    θ0[pμ+pσ+1] = log(0.5 * std(res0) + eps())
    θ0[pμ+pσ+2] = log(0.3)
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res)
    V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))

    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :resd => (pμ+pσ+1):(pμ+pσ+2)]
    # Same `_logsigma`-suffix convention as `_fit_sigma_ranef_gaussian` (#322):
    # the mean-axis RE-SD keeps the bare group name, the sigma-axis one is
    # tagged so `vc`/`re_sd` never conflate the two different scales.
    names = [:mu => nmμ, :sigma => nmσ, :resd => [String(grp), "$(grp)_logsigma"]]
    means = Dict(:mu => Xμ * θ̂[1:pμ])
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]))   # population (b=δ=0) σ

    # Conditional modes (BLUPs) at θ̂, one 2-D Laplace mode per group.
    blup_mu = zeros(G); blup_sigma = zeros(G)
    let
        βμ = θ̂[1:pμ]; βσ = θ̂[pμ+1:pμ+pσ]
        sb_mu2 = exp(2 * θ̂[pμ+pσ+1]); sb_sigma2 = exp(2 * θ̂[pμ+pσ+2])
        r_all = y .- Xμ * βμ; η0_all = Xσ * βσ
        for (k, idx) in enumerate(members)
            length(idx) == 0 && continue
            b, δ = _musig_refine(r_all[idx], η0_all[idx], sb_mu2, sb_sigma2, length(idx))
            blup_mu[k] = b; blup_sigma[k] = δ
        end
    end
    re = Dict(Symbol(grp) => blup_mu, Symbol("$(grp)_logsigma") => blup_sigma)

    fit = _withranef(_withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll), re)
    return _withmarginal(fit, :Laplace)
end

# `marginal` on the univariate Gaussian `drm`. `:LA` (the default, any case)
# keeps every route exactly as it was: each route's own integrator, which is
# exact wherever the Gaussian marginal is closed-form and GHQ-32 on `sigma ~ (1
# | g)`. On Gaussian, `:Laplace` and `:AGHQ` are implemented only for that σ
# random-intercept route (the non-Gaussian ordinary `(1 | g)` route lives in
# ordinary_laplace.jl). Returns the requested marginal as a Symbol (`:LA`,
# `:Laplace` or `:AGHQ`); the caller dispatches on it.
function _gaussian_marginal(marginal::Symbol)
    t = Symbol(uppercase(String(marginal)))
    t === :LA && return :LA
    t === :LAPLACE && return :Laplace
    t === :AGHQ && return :AGHQ
    throw(ArgumentError(
        "drm (Gaussian): `marginal = :$marginal` is not available for Gaussian(). " *
        "Use the default `marginal = :LA` (each route's default integrator; on `sigma ~ " *
        "1 + (1 | g)` that is 32-node Gauss–Hermite quadrature, not Laplace), " *
        "`marginal = :Laplace` to force the Laplace approximation drmTMB uses, or " *
        "`marginal = :AGHQ` for per-group adaptive Gauss–Hermite quadrature, for a " *
        "single random intercept `(1 | g)` on `sigma` with a fixed-effect mean."))
end

function _gaussian_laplace_reject(what; requested = "Laplace")
    throw(ArgumentError(
        "marginal = :$requested is not available for Gaussian() with $what. On the Gaussian " *
        "family this route covers exactly one random intercept `(1 | g)` on `sigma` " *
        "(fixed-effect mean, fixed-effect `sigma` predictors alongside it), fitted by " *
        "maximum likelihood. Omit `marginal` (the default `:LA`) for other models."))
end

# Admit `marginal = :Laplace`/`:AGHQ` only for the exact σ random-intercept
# shape, and refuse everything else before any route runs, so a request is
# never silently served by another integrator. Both non-default integrators
# share the same admissible shape (only the group integral differs), so one
# validator serves both; `requested` names the one in the error message.
function _gaussian_laplace_validate(f::DrmFormula, fam::Gaussian, data, algorithm, method,
                                    penalty, phylo_coupled, sparse, impute, missing;
                                    requested = "Laplace")
    rej(what) = _gaussian_laplace_reject(what; requested = requested)
    _has_joint_mi(f) && rej("an `mi()` joint missing-data formula")
    (impute === nothing && missing === nothing) ||
        rej("`impute`/`missing` controls")
    rhs = Dict(f.forms)
    extra = setdiff(keys(rhs), (:mu, :sigma))
    isempty(extra) || rej(
        "additional formula parts ($(join(sort(String.(collect(extra))), ", "))), such as `sd(g) ~ …`")
    _, re, metav, structured, _ = _split_ranef(rhs[:mu]; allow_phylo_slope = true)
    isempty(re) || rej("a random effect on the mean")
    metav === nothing || rej("`meta_V(...)`")
    (structured === nothing && isempty(_collect_structured(rhs[:mu]))) ||
        rej("a structured (phylo/relmat/animal/spatial) term on the mean")
    _, sigma_re, sigma_metav, structured_sigma = _split_ranef(rhs[:sigma])
    (structured_sigma === nothing && isempty(_collect_structured(rhs[:sigma]))) ||
        rej("a structured (phylo/relmat/animal/spatial) term on `sigma`")
    sigma_metav === nothing || rej("`meta_V(...)` on `sigma`")
    isempty(sigma_re) && rej(
        "no random effect on `sigma` (a fixed-effect Gaussian model has an exact likelihood)")
    length(sigma_re) == 1 || rej("more than one random-effect term on `sigma`")
    _re_kind(sigma_re[1][1])[1] === :intercept ||
        rej("a random slope on `sigma` (only `(1 | g)` is implemented)")
    method === :ML || rej("`method = :$method`")
    penalty === nothing || rej("`penalty`")
    algorithm === :auto || rej("`algorithm = :$algorithm`")
    # `profile_ci` is not checked: it only precomputes the sigma-phylo location-scale
    # CIs, so it is ignored on this route exactly as on the default (:LA) route, and
    # `drm_bridge_inference(method = "profile")` sets it for every univariate fit.
    phylo_coupled && rej("`phylo_coupled = true`")
    (sparse === nothing || sparse === false) || rej("`sparse = $sparse`")
    return nothing
end
