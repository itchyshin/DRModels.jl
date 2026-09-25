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
function _split_ranef(rhs; allow_phylo_slope::Bool = false)
    terms = rhs isa Tuple ? collect(rhs) : Any[rhs]
    fixed = Any[]
    re = Tuple{Any,Symbol}[]
    metav = nothing                                   # meta_V(v) known-variance column
    structured = nothing                              # (:relmat, grouping) — known K
    structured_slope = nothing                        # `x` of phylo(1 + x | g), Gaussian mean only
    for t in terms
        if t isa FunctionTerm && t.f === (|)
            push!(re, (t.args[1], t.args[2].sym))     # (re-lhs, grouping symbol)
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

    # Historical ML Woodbury nll — do not change this loop (byte-for-byte default).
    function nll_ml(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; lσb = θ[pμ+pσ+1]
        ημ = Xμ * βμ; ησ = Xσ * βσ                 # ησ = log σ_i
        σb² = exp(2lσb)
        T = eltype(θ)
        S = zeros(T, G); C = zeros(T, G)           # S_k = Σ 1/D_i,  C_k = Σ r_i/D_i
        q1 = zero(T); logdetD = zero(T)
        @inbounds for i in 1:n
            invD = exp(-2 * ησ[i])
            r = y[i] - ημ[i]
            a = r * invD
            k = gidx[i]
            wi = w[i]
            S[k] += wi * wi * invD                 # (ZᵀD⁻¹Z)_kk = Σ w_i²/D_i
            C[k] += wi * a                         # (ZᵀD⁻¹r)_k  = Σ w_i r_i/D_i
            q1 += r * a                            # rᵀD⁻¹r
            logdetD += 2 * ησ[i]                   # log D_i
        end
        q2 = zero(T); logdetCap = zero(T)
        @inbounds for k in 1:G
            Mk = 1 / σb² + S[k]                     # Woodbury capacitance (diagonal)
            q2 += C[k]^2 / Mk
            logdetCap += log(1 + σb² * S[k])        # det-lemma term
        end
        quad = q1 - q2
        logdetV = logdetD + logdetCap
        return 0.5 * (logdetV + quad) + const_2pi
    end

    # Restricted nll: ML Woodbury + ½ logdet(Xμ′ V⁻¹ Xμ) − ½ pμ log(2π).
    # Xμ′ V⁻¹ Xμ = Xμ′ D⁻¹ Xμ − (Z′ D⁻¹ Xμ)′ diag(1/M) (Z′ D⁻¹ Xμ) with the
    # same capacitance M_k = 1/σb² + S_k as the ML nll.
    function nll_reml(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; lσb = θ[pμ+pσ+1]
        ημ = Xμ * βμ; ησ = Xσ * βσ
        σb² = exp(2lσb)
        T = eltype(θ)
        S = zeros(T, G); C = zeros(T, G)
        ZtDinvX = zeros(T, G, pμ)
        XtDinvX = zeros(T, pμ, pμ)
        q1 = zero(T); logdetD = zero(T)
        @inbounds for i in 1:n
            invD = exp(-2 * ησ[i])
            r = y[i] - ημ[i]
            a = r * invD
            k = gidx[i]
            wi = w[i]
            S[k] += wi * wi * invD
            C[k] += wi * a
            q1 += r * a
            logdetD += 2 * ησ[i]
            @inbounds for j in 1:pμ
                xj = Xμ[i, j]
                ZtDinvX[k, j] += wi * invD * xj
                @inbounds for l in 1:pμ
                    XtDinvX[j, l] += invD * xj * Xμ[i, l]
                end
            end
        end
        q2 = zero(T); logdetCap = zero(T)
        XtVinvX = copy(XtDinvX)
        @inbounds for k in 1:G
            Mk = 1 / σb² + S[k]
            invMk = 1 / Mk
            q2 += C[k]^2 * invMk
            logdetCap += log(1 + σb² * S[k])
            @inbounds for j in 1:pμ
                zj = ZtDinvX[k, j]
                @inbounds for l in 1:pμ
                    XtVinvX[j, l] -= zj * invMk * ZtDinvX[k, l]
                end
            end
        end
        nll_ml_θ = 0.5 * (logdetD + logdetCap + q1 - q2) + const_2pi
        # Xμ′V⁻¹Xμ is PSD by construction, but it is formed by Woodbury SUBTRACTION.
        # As σb² → ∞ the group means absorb the mean signal, Xμ′V⁻¹Xμ → 0, and rounding
        # noise can make its determinant NEGATIVE (measured at lσb ≈ 16, σb² ≈ 8e13).
        # `logdet` has no Symmetric method: it falls through to the LU path, where
        # logabsdet returns sign -1 and log(-1.0) throws DomainError (#499). Reject the
        # step instead. NOTE: it must be a LARGE FINITE barrier, not +Inf — LBFGS's
        # default HagerZhang line search asserts `isfinite(phi_c)` and would trade
        # the DomainError for an AssertionError.
        cholXtVinvX = cholesky(Symmetric(XtVinvX); check=false)
        issuccess(cholXtVinvX) || return nll_ml_θ + T(REML_NONPD_PENALTY)
        return nll_ml_θ + 0.5 * logdet(cholXtVinvX) - const_pμ
    end

    nll = reml ? nll_reml : nll_ml

    βμ0 = Xμ \ y
    res0 = y - Xμ * βμ0
    θ0 = zeros(pμ + pσ + 1)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(std(res0) + eps())
    θ0[pμ+pσ+1] = log(std(res0) / 2 + eps())

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
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
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
function _fit_correlated_ranef_gaussian(fam::Gaussian, y, Xμ, Xσ, gidx, G, xs, nmμ, nmσ, grp, g_tol)
    n = length(y)
    pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]
        a = θ[pμ+pσ+1]; b = θ[pμ+pσ+2]; cc = θ[pμ+pσ+3]
        ημ = Xμ * βμ; ησ = Xσ * βσ
        T = eltype(θ)
        l11 = exp(a); l22 = exp(b)                 # L = [l11 0; cc l22], Σ_re = L Lᵀ
        Σ11 = l11^2; Σ21 = cc * l11; Σ22 = cc^2 + l22^2
        detΣ = Σ11 * l22^2                         # det(L Lᵀ), stable even when cc is large
        Si11 = Σ22 / detΣ; Si22 = Σ11 / detΣ; Si21 = -Σ21 / detΣ
        logdetΣre = 2a + 2b
        b11 = zeros(T, G); b21 = zeros(T, G); b22 = zeros(T, G)
        c1 = zeros(T, G); c2 = zeros(T, G)
        q1 = zero(T); logdetD = zero(T)
        @inbounds for i in 1:n
            invD = exp(-2 * ησ[i]); r = y[i] - ημ[i]; w = xs[i]; k = gidx[i]
            b11[k] += invD; b21[k] += w * invD; b22[k] += w * w * invD
            ri = r * invD; c1[k] += ri; c2[k] += w * ri
            q1 += r * ri; logdetD += 2 * ησ[i]
        end
        quad = q1; logdetM = zero(T)
        @inbounds for k in 1:G
            m11 = Si11 + b11[k]; m21 = Si21 + b21[k]; m22 = Si22 + b22[k]
            dM = m11 * m22 - m21^2
            u1 = (m22 * c1[k] - m21 * c2[k]) / dM      # M_k⁻¹ c_k
            u2 = (-m21 * c1[k] + m11 * c2[k]) / dM
            quad -= c1[k] * u1 + c2[k] * u2
            logdetM += log(dM)
        end
        logdetV = logdetD + G * logdetΣre + logdetM
        return 0.5 * (logdetV + quad) + 0.5 * n * log(2π)
    end
    βμ0 = Xμ \ y; res0 = y - Xμ * βμ0
    θ0 = zeros(pμ + pσ + 3)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(std(res0) + eps())
    sd0 = log(std(res0) / 2 + eps())
    θ0[pμ+pσ+1] = sd0; θ0[pμ+pσ+2] = sd0; θ0[pμ+pσ+3] = 0.0
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :recov => (pμ+pσ+1):(pμ+pσ+3)]
    names = [:mu => nmμ, :sigma => nmσ, :recov => ["$(grp):L11", "$(grp):L22", "$(grp):L21"]]
    means = Dict(:mu => Xμ * θ̂[1:pμ]); obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]))
    # Conditional RE estimates (BLUPs): per group b̂_k = M_k⁻¹ c_k (intercept, slope),
    # the same posterior-mean solve the marginal nll forms internally, evaluated at θ̂.
    blup = let
        βμ = θ̂[1:pμ]; βσ = θ̂[pμ+1:pμ+pσ]
        a = θ̂[pμ+pσ+1]; b = θ̂[pμ+pσ+2]; cc = θ̂[pμ+pσ+3]
        ημ = Xμ * βμ; ησ = Xσ * βσ
        l11 = exp(a); l22 = exp(b)
        Σ11 = l11^2; Σ21 = cc * l11; Σ22 = cc^2 + l22^2
        detΣ = Σ11 * l22^2
        Si11 = Σ22 / detΣ; Si22 = Σ11 / detΣ; Si21 = -Σ21 / detΣ
        b11 = zeros(G); b21 = zeros(G); b22 = zeros(G); c1 = zeros(G); c2 = zeros(G)
        @inbounds for i in 1:n
            invD = exp(-2 * ησ[i]); r = y[i] - ημ[i]; ww = xs[i]; k = gidx[i]
            b11[k] += invD; b21[k] += ww * invD; b22[k] += ww * ww * invD
            ri = r * invD; c1[k] += ri; c2[k] += ww * ri
        end
        B = Matrix{Float64}(undef, G, 2)
        @inbounds for k in 1:G
            m11 = Si11 + b11[k]; m21 = Si21 + b21[k]; m22 = Si22 + b22[k]
            dM = m11 * m22 - m21^2
            B[k, 1] = (m22 * c1[k] - m21 * c2[k]) / dM    # intercept BLUP
            B[k, 2] = (-m21 * c1[k] + m11 * c2[k]) / dM   # slope BLUP
        end
        B
    end
    re = Dict(Symbol(grp) => blup)
    return _withranef(_withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll), re)
end

"""
    ranef(fit) -> Dict{Symbol,...}

Per-level conditional random-effect estimates (BLUPs), keyed by grouping factor.
These are the posterior means of the random effects at the fitted variance
components — drmTMB's `ranef()`.

- Scalar random intercept `(1 | g)`: a `Vector` of length `n_levels(g)`.
- Correlated `(1 + x | g)`: an `n_levels × 2` matrix (`[intercept slope]`).
- Multiple components `(1 | g) + (1 | h)`: one entry per factor.

Currently populated for the Gaussian closed-form RE paths (exact GLS conditional
means). Returns an empty `Dict` for models without random effects. Non-Gaussian
GLMM posterior modes (GHQ/Laplace) are not yet available.
"""
function ranef(fit::DrmFit)
    fit.ranef === nothing && return Dict{Symbol,Vector{Float64}}()
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
# Z̃ᵀD⁻¹Z̃ is rank-deficient (crossed intercept columns), so we factor with
# check=false and return a finite penalty on failure — the optimiser's line
# search then retreats from those ill-scaled probes. Closed-form GLS; Z precomputed.
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
        ZtDir = Z̃' * (invD .* r)
        M = Z̃' * (invD .* Z̃)
        C = cholesky(Symmetric(M + I); check = false)  # check=false → never throws
        issuccess(C) || return oftype(sum(θ), 1e18)    # retreat from ill-scaled probes
        quad = sum(r .^ 2 .* invD) - dot(ZtDir, C \ ZtDir)
        logdetV = sum(2 .* ησ) + logdet(C)
        return 0.5 * (logdetV + quad) + 0.5 * n * log(2π)
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
        ZtDir = Z̃' * (invD .* r)
        M = Z̃' * (invD .* Z̃)
        C = cholesky(Symmetric(M + I); check = false)
        issuccess(C) || return Gout

        Hinv = C \ Matrix{Float64}(I, q, q)
        bscaled = Hinv * ZtDir
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
# effect is integrated out by K-node Gauss–Hermite quadrature: substituting
# b = √2 σ_b z turns the prior integral into Σₖ wₖ·(group likelihood at node k).
# Within a group every observation gets the same node shift δₖ = √2 σ_b zₖ, so a
# group reduces to Aₘ = Σ η0ᵢ and Bₘ = Σ rᵢ² e^{-2η0ᵢ}. O(n + G·K) per eval and
# fully differentiable (nodes are constants). drmTMB does this with Laplace; for
# a 1-D effect AGHQ is the standard, more accurate sibling.
#
# `laplace = true` (reached via `drm(...; marginal = :Laplace)`) swaps the GHQ-32
# objective for the Laplace approximation that native drmTMB (TMB) computes for
# this model; see `_sigre_laplace_nll` below. Everything else (start values,
# optimiser, blocks, names, reported σ) is shared, and the default `laplace =
# false` path is the GHQ-32 fit exactly as before.
function _fit_sigma_ranef_gaussian(fam::Gaussian, y, Xμ, Xσ, gidx, G, nmμ, nmσ, grp, g_tol;
                                   laplace::Bool = false)
    n = length(y); pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[gidx[i]], i)
    end
    z, w = _gauss_hermite(32)
    logw = log.(w); K = length(z); rt2 = sqrt(2.0); lπ = log(π); l2π = log(2π)
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
            terms = Vector{T}(undef, K)
            for k in 1:K
                δ = rt2 * σb * z[k]
                terms[k] = logw[k] - mg * δ - 0.5 * exp(-2δ) * Bg
            end
            mx = maximum(terms)
            llg = -0.5 * lπ - 0.5 * mg * l2π - Ag + mx + log(sum(exp.(terms .- mx)))
            s -= llg
        end
        return s
    end
    obj = laplace ? _sigre_laplace_nll(y, Xμ, Xσ, members, pμ, pσ) : nll
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
    return laplace ? _withmarginal(fit, :Laplace) : fit
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

# `marginal` on the univariate Gaussian `drm`. `:LA` (the default, any case)
# keeps every route exactly as it was: each route's own integrator, which is
# exact wherever the Gaussian marginal is closed-form and GHQ-32 on `sigma ~ (1
# | g)`. `:Laplace` is implemented only for that σ random-intercept route.
# Returns `true` when `:Laplace` was requested.
function _gaussian_marginal(marginal::Symbol)
    t = Symbol(uppercase(String(marginal)))
    t === :LA && return false
    t === :LAPLACE && return true
    throw(ArgumentError(
        "drm (Gaussian): `marginal = :$marginal` is not available for Gaussian(). " *
        "Use the default `marginal = :LA` (each route's default integrator; on `sigma ~ " *
        "1 + (1 | g)` that is 32-node Gauss–Hermite quadrature, not Laplace), or " *
        "`marginal = :Laplace` to force the Laplace approximation drmTMB uses, for a " *
        "single random intercept `(1 | g)` on `sigma` with a fixed-effect mean."))
end

function _gaussian_laplace_reject(what)
    throw(ArgumentError(
        "marginal = :Laplace is not available for Gaussian() with $what. On the Gaussian " *
        "family this route covers exactly one random intercept `(1 | g)` on `sigma` " *
        "(fixed-effect mean, fixed-effect `sigma` predictors alongside it), fitted by " *
        "maximum likelihood. Omit `marginal` (the default `:LA`) for other models."))
end

# Admit `marginal = :Laplace` only for the exact σ random-intercept shape, and
# refuse everything else before any route runs, so a request is never silently
# served by another integrator.
function _gaussian_laplace_validate(f::DrmFormula, fam::Gaussian, data, algorithm, method,
                                    penalty, profile_ci, phylo_coupled, sparse, impute, missing)
    _has_joint_mi(f) && _gaussian_laplace_reject("an `mi()` joint missing-data formula")
    (impute === nothing && missing === nothing) ||
        _gaussian_laplace_reject("`impute`/`missing` controls")
    rhs = Dict(f.forms)
    extra = setdiff(keys(rhs), (:mu, :sigma))
    isempty(extra) || _gaussian_laplace_reject(
        "additional formula parts ($(join(sort(String.(collect(extra))), ", "))), such as `sd(g) ~ …`")
    _, re, metav, structured, _ = _split_ranef(rhs[:mu]; allow_phylo_slope = true)
    isempty(re) || _gaussian_laplace_reject("a random effect on the mean")
    metav === nothing || _gaussian_laplace_reject("`meta_V(...)`")
    (structured === nothing && isempty(_collect_structured(rhs[:mu]))) ||
        _gaussian_laplace_reject("a structured (phylo/relmat/animal/spatial) term on the mean")
    _, sigma_re, sigma_metav, structured_sigma = _split_ranef(rhs[:sigma])
    (structured_sigma === nothing && isempty(_collect_structured(rhs[:sigma]))) ||
        _gaussian_laplace_reject("a structured (phylo/relmat/animal/spatial) term on `sigma`")
    sigma_metav === nothing || _gaussian_laplace_reject("`meta_V(...)` on `sigma`")
    isempty(sigma_re) && _gaussian_laplace_reject(
        "no random effect on `sigma` (a fixed-effect Gaussian model has an exact likelihood)")
    length(sigma_re) == 1 || _gaussian_laplace_reject("more than one random-effect term on `sigma`")
    _re_kind(sigma_re[1][1])[1] === :intercept ||
        _gaussian_laplace_reject("a random slope on `sigma` (only `(1 | g)` is implemented)")
    method === :ML || _gaussian_laplace_reject("`method = :$method`")
    penalty === nothing || _gaussian_laplace_reject("`penalty`")
    algorithm === :auto || _gaussian_laplace_reject("`algorithm = :$algorithm`")
    profile_ci && _gaussian_laplace_reject("`profile_ci = true`")
    phylo_coupled && _gaussian_laplace_reject("`phylo_coupled = true`")
    (sparse === nothing || sparse === false) || _gaussian_laplace_reject("`sparse = $sparse`")
    return nothing
end
