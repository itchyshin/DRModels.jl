# reml_q2.jl — REML (Patterson–Thompson) for the bivariate q=2 structured
# residual-correlation route (#470): a `phylo`/`relmat`/`animal` random effect
# on mu1 AND mu2 (2x2 among-group covariance Λ), plus an intercept-only
# residual covariance D = [[σ1², ρσ1σ2],[ρσ1σ2, σ2²]] — the model built by
# `CoevoProblem` / `fit_coevolution_q2_residual` in `coevolution_q.jl` and
# routed to from `_fit_bivariate_q2_structured` in `gaussian_bivariate.jl`.
#
# WHICH FIXED-EFFECT AXES GET MARGINALISED — the exact question `reml_q4.jl`'s
# header calls out (a fixture once asserted the wrong axes for the q=4 case;
# see `test/parity/q4-reml/biv-q4-phylo-reml/expected.toml`)
# --------------------------------------------------------------------------
# The defining rule, taken from `reml_q4.jl`'s own header: an axis is profiled
# out of the restricted likelihood IFF it has a random-effect axis to
# integrate against — i.e. its leaf linear predictor is (eta_axis + u_axis).
# `reml_q4.jl` profiles beta_mu1, beta_mu2, beta_s1, AND beta_s2 because the
# q=4 model puts a random effect on ALL FOUR axes (Sigma_a is 4x4 over
# mu1,mu2,sigma1,sigma2); beta_rho stays outer because rho12 has no u-axis.
#
# The q=2 STRUCTURED route here is a different model: Λ (the among-group
# covariance carried on the tree/relmat/animal precision) is 2x2, over
# mu1 and mu2 ONLY. sigma1, sigma2, and rho12 are ordinary fixed intercepts
# with NO random-effect axis — the caller already requires them intercept-only
# (`gaussian_bivariate.jl`, `_fit_bivariate_q2_structured`). By the same rule,
# this file profiles out ONLY beta_mu1 and beta_mu2; beta_sigma1, beta_sigma2,
# and beta_rho12 stay OUTER (part of `phi`, not marginalised) — exactly the
# role beta_rho plays in `reml_q4.jl`, generalised to "no u-axis ⇒ outer".
#
# WHY THIS IS A SINGLE EXACT NEWTON STEP, NOT reml_q4's ALTERNATING SCHEME
# --------------------------------------------------------------------------
# `reml_q4.jl` alternates an E-step (mode in u) with a conditional Newton on
# beta because its sigma axes enter the leaf through exp(eta_sigma + u_sigma)
# — nonlinear, so the joint objective jn(u, beta) is not quadratic and needs
# iterating to converge. Here mu1/mu2 enter LINEARLY (Y = X*beta + u + eps,
# no link function) and D is held fixed while profiling beta, so at fixed
# (Lambda, D) the joint negative log-likelihood jn(u, beta_mu) is an EXACT
# quadratic function of (u, beta_mu). A single Newton step from ANY starting
# point is therefore the EXACT joint minimiser — no alternation, no line
# search, no convergence tolerance to tune. This also means beta_mu's REML
# profile is (as it must be, since ML and REML share the same conditional-on-
# variance-components GLS profile of beta) identical in FORM to a classical
# GLS profile under V = Z(Q_cond⁻¹⊗Λ)Z' + Iₙ⊗D — this file just computes it
# via the same bordered (u, beta) Hessian construction `reml_q4.jl` uses, so
# there is exactly one restricted-likelihood FORMULA in this package
# (L_REML = L_ML(theta, beta_hat) − 0.5·logdet(S), S the Schur complement of
# the profiled fixed effects in the joint (u, beta) Hessian at the mode),
# applied to two different leaf models.
#
# `V` (`meta_V`, known bivariate sampling covariance) is NOT handled here: it
# is accepted only on the *residual-only* bivariate route (no structured
# marker), which has no random-effect axis at all in this package's
# architecture — see the permanent `V`+REML rejection in
# `gaussian_bivariate.jl`.

# ---------------------------------------------------------------------------
# phi layout for the q=2 REML outer optimisation: phi = (lc(Λ)[3], logσ1,
# logσ2, ηρ) — length lc_len(2) + 3 == 6. beta_mu1/beta_mu2 are ABSENT (they
# are profiled out at every phi via the exact Newton step below).
# ---------------------------------------------------------------------------

"Length of the q=2 REML outer parameter vector `phi = (lc(Λ), logσ1, logσ2, ηρ)`."
q2_reml_phi_len() = lc_len(2) + 3

function _q2_reml_unpack_phi(phi::AbstractVector{T}) where {T}
    lc    = phi[1:lc_len(2)]
    logσ1 = phi[lc_len(2) + 1]
    logσ2 = phi[lc_len(2) + 2]
    ηρ    = phi[lc_len(2) + 3]
    σ1 = exp(logσ1); σ2 = exp(logσ2)
    ρ  = RHO_GUARD * tanh(ηρ)
    D  = Matrix(Symmetric([σ1^2 ρ*σ1*σ2; ρ*σ1*σ2 σ2^2]))
    # `Λ` (formed once, in Float64) is kept only for `_q2_lambda_admissible`'s
    # scalar det/cond gate. `chΛ` is the factor built straight from `lc`, and
    # is what both `_q2_profile_and_schur` and `coevo_marginal_cov` use to
    # build H_uu -- neither ever forms or inverts `Λ` (#857 site K, #862/#865).
    return lc_to_cov(lc, 2), D, lc_to_chol(lc, 2)
end

function _q2_reml_pack_phi(Λ::AbstractMatrix, σ_res::AbstractVector, rho12::Real)
    length(σ_res) == 2 || error("q2 REML phi pack requires two residual SDs")
    ρ  = clamp(Float64(rho12), -0.95, 0.95)
    ηρ = atanh(ρ / RHO_GUARD)
    return vcat(cov_to_lc(Λ), log.(Float64.(σ_res)), ηρ)
end

# ---------------------------------------------------------------------------
# At fixed (Λ, D): build H_uu (via the existing `coevo_Huu`), the cross block
# H_u_beta and H_beta_beta over the mu1/mu2 axes only, take the ONE exact
# joint Newton step (from u0 = 0, beta0) to the joint mode, and hand back the
# Schur complement S needed for the REML correction alongside it (S is a
# byproduct of the same blocks the Newton step uses — no repeated work).
#
# Numerics (#857 site K, #862/#865 follow-up). This was the one caller left
# forming `Λ` densely and calling `inv(Λ)` to build H_uu — the exact failure
# mode #862's header measured for `coevo_marginal_cov` (silent wrong answers
# in the 833/2352 sweep, not just the throws). It is whitened the same way:
# with `L L' = Λ` (never forming `Λ⁻¹`) and `v = (I ⊗ L)⁻¹u`, the u-block
# becomes `H̃_uu = Q ⊗ I + Σₜ cₜ eₜeₜ' ⊗ L'D⁻¹L` (`coevo_Huu` with a whitened
# residual block, exactly `coevo_marginal_cov`'s construction) and the cross
# block becomes `H̃_ub = (I ⊗ L)'H_ub`. The Schur complement of the profiled
# beta block is invariant under this change of basis for the ELIMINATED u
# (S = H_bb − H_ub'H_uu⁻¹H_ub = H_bb − H̃_ub'H̃_uu⁻¹H̃_ub whenever `L` is
# invertible — the (I⊗L) factors cancel exactly), so `S` and `β̂` are
# unchanged in exact arithmetic; only `û` needs the final un-whitening
# `u = (I ⊗ L)v`, done in-place at the end exactly as
# `coevo_marginal_cov` un-whitens its own mode.
# ---------------------------------------------------------------------------
function _q2_profile_and_schur(prob::CoevoProblem, Q_cond::SparseMatrixCSC,
                               Λ::AbstractMatrix, D::AbstractMatrix,
                               β0::AbstractMatrix)
    nbeta = prob.k * prob.q
    chΛ = cholesky(Symmetric(Matrix{Float64}(Λ)); check = false)
    if !issuccess(chΛ)
        Sz = Symmetric(zeros(nbeta, nbeta))
        return (β̂ = β0, û = zeros(prob.q * prob.N), S = Sz,
                ch_S = cholesky(Sz; check = false), ok = false)
    end
    return _q2_profile_and_schur(prob, Q_cond, chΛ, D, β0)
end

function _q2_profile_and_schur(prob::CoevoProblem, Q_cond::SparseMatrixCSC,
                               chΛ::Cholesky, D::AbstractMatrix,
                               β0::AbstractMatrix)
    prob.q == 2 || error("q2 REML requires q = 2")
    q = 2; k = prob.k
    Dinv = inv(Symmetric(D))
    L = Matrix{Float64}(chΛ.L)
    size(L) == (q, q) || error("Λ factor has size $(size(L)); expected ($q, $q)")
    nu = q * prob.N; nbeta = k * q
    if !all(isfinite, L)
        Sz = Symmetric(zeros(nbeta, nbeta))
        return (β̂ = β0, û = zeros(nu), S = Sz,
                ch_S = cholesky(Sz; check = false), ok = false)
    end

    Iq = Matrix{Float64}(I, q, q)
    Pw   = prior_precision(Q_cond, Iq)             # Q ⊗ I (never Λ⁻¹)
    H_uu = coevo_Huu(prob, Pw, L' * Dinv * L)      # whitened H̃_uu, cf. `coevo_marginal_cov`
    # Defence in depth (#503), matching the `ch_S` pattern below. The real guard
    # is `_q2_lambda_admissible`; this catches anything that reaches here anyway.
    chH  = cholesky(Symmetric(H_uu); check = false)
    if !issuccess(chH)
        Sz = Symmetric(zeros(nbeta, nbeta))
        return (β̂ = β0, û = zeros(nu), S = Sz,
                ch_S = cholesky(Sz; check = false), ok = false)
    end

    LtDinv = L' * Dinv
    H_ub = zeros(nu, nbeta)     # whitened cross block H̃_ub = (I⊗L)'H_ub
    H_bb = zeros(nbeta, nbeta)  # unaffected by whitening u (no u dependence)
    @inbounds for i in eachindex(prob.leaf_node)
        t = prob.leaf_node[i]; base = q * (t - 1)
        Xi = @view prob.X[i, :]
        for a in 1:q, ap in 1:q
            dv = LtDinv[a, ap]
            d  = Dinv[a, ap]
            offap = (ap - 1) * k
            for c in 1:k
                H_ub[base + a, offap + c] += dv * Xi[c]
            end
            offa = (a - 1) * k
            for r in 1:k, c in 1:k
                H_bb[offa + r, offap + c] += d * Xi[r] * Xi[c]
            end
        end
    end

    C = Matrix{Float64}(undef, nu, nbeta)
    for j in 1:nbeta
        C[:, j] = chH \ H_ub[:, j]
    end
    S_raw = H_bb - H_ub' * C
    S = Symmetric((S_raw + S_raw') / 2)

    # Exact joint Newton step from (u0 = 0, beta0 = β0): jn(u, beta_mu) is
    # exactly quadratic at fixed (Λ, D), so this lands exactly on the joint
    # mode regardless of β0 — see the file header. Carried out in the
    # whitened v-space throughout (g_u whitened to g̃_u = (I⊗L)'g_u, exactly
    # `coevo_marginal_cov`'s rhs-whitening loop), then û is un-whitened once
    # at the end.
    rhs0   = coevo_rhs(prob, β0, Dinv)          # g_u = -rhs0, ORIGINAL u-space
    g_u    = -rhs0
    g_uw   = similar(g_u)                        # g̃_u = (I ⊗ L)' g_u  (L' upper)
    @inbounds for t in 1:prob.N, a in 1:q
        base = q * (t - 1)
        s = 0.0
        for b in a:q
            s += L[b, a] * g_u[base + b]
        end
        g_uw[base + a] = s
    end
    resid0 = prob.Y .- prob.X * β0
    Wt     = resid0 * Dinv                       # n x q, Wt[i,:] = Dinv*resid0[i,:]
    g_β    = -(prob.X' * Wt)                     # k x q

    ch_S = cholesky(S; check = false)
    issuccess(ch_S) ||
        return (β̂ = β0, û = zeros(nu), S = S, ch_S = ch_S, ok = false)

    rhs_β = vec(g_β) .- H_ub' * (chH \ g_uw)
    Δβ    = -(ch_S \ rhs_β)
    β̂     = β0 .+ reshape(Δβ, k, q)
    v̂     = -(chH \ (g_uw .+ H_ub * Δβ))          # whitened mode
    û     = similar(v̂)                            # un-whiten: u = (I ⊗ L) v
    @inbounds for t in 1:prob.N, a in 1:q
        base = q * (t - 1)
        s = 0.0
        for b in 1:a
            s += L[a, b] * v̂[base + b]
        end
        û[base + a] = s
    end

    return (β̂ = β̂, û = û, S = S, ch_S = ch_S, ok = true)
end

# REML log-likelihood at outer parameters (Λ, D): profile beta_mu (exact
# Newton step above), evaluate the ML marginal at beta_hat via the existing
# (tested) `coevo_marginal_cov`, and subtract 0.5*logdet(S).
# #503: `lc_to_cov` is PD by construction in exact arithmetic, but at extreme
# log-Cholesky values the Lambda it returns is numerically singular -- at
# lc = [-25, 1, -25] its determinant is NEGATIVE (-2.2e-38) while `isposdef`
# still answers true. `inv(Lambda)` is then meaningless, H_uu inherits it, and
# LAPACK's potrf detects the resulting indefiniteness only SOMETIMES: it throws
# PosDefException on some builds (caught upstream -> -Inf, the SAFE outcome) and
# SUCCEEDS on others, handing the optimiser a finite but meaningless REML value
# computed from a factorisation of a non-PD matrix. Measured over a 2352-point
# sweep of the REACHABLE parameter space: 20 throw, 833 return silent garbage.
# Guarding the factorisation alone would fix only the branch that already failed
# safely. This rejects the inadmissible parameter itself -- what both share.
#
# Threshold: a real fit on the shipped fixture sits at cond(Lambda) = 2.2, and a
# generous +/-3 log-Cholesky box around that optimum reaches only 3.7e8, while
# every reproducible failure needs cond >= 1.6e17. Nine orders of separation;
# 1e12 sits in the middle of the gap.
#
# This is the TREE route only: it builds N = 2p-2 augmented nodes of which only
# p carry data, so the internal-node blocks of H_uu are held up by the prior
# alone. The known-covariance/relmat route cannot reach this -- every node there
# carries data, so each block keeps a PD D^-1 term.
function _q2_lambda_admissible(Λ::AbstractMatrix; maxcond::Float64 = 1e12)
    A = Matrix{Float64}(Λ)
    all(isfinite, A) || return false
    d = det(A)
    (isfinite(d) && d > 0) || return false
    return cond(A) < maxcond
end

function _q2_reml_ll(prob::CoevoProblem, Q_cond::SparseMatrixCSC,
                     Λ::AbstractMatrix, D::AbstractMatrix, chΛ::Cholesky, β0::AbstractMatrix)
    _q2_lambda_admissible(Λ) || return -Inf, β0, zeros(prob.q * prob.N)
    local prof
    try
        prof = _q2_profile_and_schur(prob, Q_cond, chΛ, D, β0)
    catch e
        (e isa DomainError || e isa LinearAlgebra.PosDefException ||
         e isa LinearAlgebra.SingularException) || rethrow(e)
        return -Inf, β0, zeros(prob.q * prob.N)
    end
    prof.ok || return -Inf, prof.β̂, prof.û
    issuccess(prof.ch_S) || return -Inf, prof.β̂, prof.û
    # #503 (follow-up): `coevo_marginal_cov` has its OWN unguarded
    # `cholesky(Symmetric(H))` (coevolution_q.jl:233, commented "PD: P + PSD data
    # term"). It sits OUTSIDE the try above, which only wrapped
    # `_q2_profile_and_schur`, so a PosDefException from here escaped the whole
    # fit -- observed on CI's Julia 1.12.7 in test_reml_q2_structured.jl even
    # with the Λ-admissibility guard in place. A Λ can pass `cond < 1e12` and
    # still drive THIS Hessian to the definiteness boundary. Reject the step.
    #
    # `chΛ` (built straight from `lc` by `lc_to_chol`, not `Λ = L L'`) is passed
    # here in place of `Λ`: #857 site K follow-up (draft #862) -- `Λ` was
    # already formed once in Float64 by `_q2_reml_unpack_phi`, so handing it to
    # `coevo_marginal_cov` made that method re-factor an already-lossy matrix,
    # accurate only to l22 ≈ −18.
    local ml_ll, û
    try
        ml_ll, û, _, _ = coevo_marginal_cov(prob, Q_cond, prof.β̂, chΛ, D)
    catch e
        (e isa DomainError || e isa LinearAlgebra.PosDefException ||
         e isa LinearAlgebra.SingularException) || rethrow(e)
        return -Inf, prof.β̂, prof.û
    end
    isfinite(ml_ll) || return -Inf, prof.β̂, û
    return ml_ll - 0.5 * logdet(prof.ch_S), prof.β̂, û
end

"""
    fit_coevolution_q2_reml(prob, Q_cond; β0, Λ0, σ0, rho0, g_tol, iterations) -> NamedTuple

REML (Patterson–Thompson) fit for the bivariate q=2 structured residual-
correlation coevolution model — the REML counterpart of
[`fit_coevolution_q2_residual`](@ref). Marginalises `beta_mu1` and `beta_mu2`
(the two axes carrying the structured random effect Λ); `beta_sigma1`,
`beta_sigma2`, and `beta_rho12` stay outer parameters, since sigma1/sigma2/
rho12 are intercept-only fixed effects with no random-effect axis to
integrate against. See this file's header for why exactly those axes (and
not, e.g., all of them as in the q=4 engine).

Because mu1/mu2 enter the leaf residual linearly and (Λ, D) are held fixed
while profiling beta, the profile step is an EXACT single Newton step (not an
iterative/alternating scheme) — see `_q2_profile_and_schur`.

Returns `(; β, Λ, residual_cov, σ_res, rho12, reml_loglik, ml_loglik, loglik,
converged, iterations, u_hat)`, mirroring `fit_coevolution_q2_residual`'s
field names (`loglik` is set to `reml_loglik`, matching the q=4 REML fitter's
convention in `_fit_bivariate_q4_structured`).

# Normalisation convention (#477)

Like `reml_q4.jl`'s `fit_q4_reml`, `reml_loglik` here reports the
**normalised** Patterson–Thompson restricted log-likelihood: the raw objective
`ml_ll - 0.5*logdet(S)` plus the `(n_β/2)·log(2π)` that lme4/glmmTMB/TMB add
when integrating the flat prior over the marginalised fixed effects. Here
`n_β = length(β̂)` (`beta_mu1` + `beta_mu2`; `beta_sigma1`/`beta_sigma2`/
`beta_rho12` stay outer and are never marginalised, and `rho12` comes from `D̂`
rather than from `β̂`, so it is correctly not counted).

**Changed 2026-08-25 (#477)**, together with the q=4 route — see
`fit_q4_reml`'s docstring for the derivation and the evidence. Note this route
has **no parity fixture of its own**: it is verified only by sharing the q=4
route's arithmetic, which the q=4 gate confirmed by tightening from 5.5436 to
0.03. A q=2 REML parity fixture would be the way to check it directly.
"""
function fit_coevolution_q2_reml(prob::CoevoProblem, Q_cond::SparseMatrixCSC;
                                 β0 = nothing, Λ0 = nothing, σ0 = nothing,
                                 rho0::Real = 0.0, g_tol::Float64 = 1e-6,
                                 iterations::Int = 1000, fd_h::Float64 = 1e-6)
    prob.q == 2 || throw(ArgumentError("fit_coevolution_q2_reml requires q = 2"))
    β0 === nothing && (β0 = prob.X \ prob.Y)
    β0 = Matrix{Float64}(β0)
    Λ0 === nothing && (Λ0 = Matrix(0.2 * I(2)))
    if σ0 === nothing
        resid = prob.Y .- prob.X * β0
        σ0 = [std(resid[:, 1]) + eps(), std(resid[:, 2]) + eps()]
    end
    phi0 = _q2_reml_pack_phi(Matrix(Λ0), σ0, rho0)
    n = length(prob.leaf_node)

    β_cache = Ref(β0)

    function negreml(phi)
        Λ, D, chΛ = _q2_reml_unpack_phi(phi)
        rv, β̂, _ = _q2_reml_ll(prob, Q_cond, Λ, D, chΛ, β_cache[])
        isfinite(rv) || return Inf
        β_cache[] = β̂
        return -rv / n
    end

    function g!(G, phi)
        @inbounds for j in eachindex(phi)
            pp = copy(phi); pp[j] += fd_h
            pm = copy(phi); pm[j] -= fd_h
            fp = negreml(pp); fm = negreml(pm)
            G[j] = (isfinite(fp) && isfinite(fm)) ? (fp - fm) / (2fd_h) : 0.0
        end
        return G
    end

    res = Optim.optimize(negreml, g!, phi0,
                         Optim.LBFGS(linesearch = Optim.LineSearches.MoreThuente()),
                         Optim.Options(g_tol = g_tol, iterations = iterations,
                                       f_reltol = 1e-10, successive_f_tol = 2))
    phî = Optim.minimizer(res)
    Λ̂, D̂, chΛ̂ = _q2_reml_unpack_phi(phî)
    reml_ll, β̂, û = _q2_reml_ll(prob, Q_cond, Λ̂, D̂, chΛ̂, β_cache[])
    # #503 (follow-up): same unguarded factorisation, at the FINAL point. Here a
    # failure must not abort a REML fit that otherwise succeeded -- ml_loglik is
    # a secondary quantity reported for cross-structure comparison, so NaN is the
    # honest value rather than a thrown fit.
    #
    # `chΛ̂` (from `lc_to_chol`) is passed here in place of `Λ̂` for the same
    # #857 site K reason as inside `_q2_reml_ll` above.
    ml_ll = try
        first(coevo_marginal_cov(prob, Q_cond, β̂, chΛ̂, D̂))
    catch e
        (e isa DomainError || e isa LinearAlgebra.PosDefException ||
         e isa LinearAlgebra.SingularException) || rethrow(e)
        NaN
    end
    σ̂ = [sqrt(D̂[1, 1]), sqrt(D̂[2, 2])]
    ρ̂ = D̂[1, 2] / (σ̂[1] * σ̂[2])

    # #477: report the NORMALISED restricted log-likelihood, matching the q=4
    # route, DRModels.jl's univariate REML routes, and lme4/glmmTMB/TMB. `β̂` holds
    # exactly the marginalised fixed effects (`nbeta = k * q` above); `rho12`
    # comes from `D̂`, not from `β̂`, so it is correctly not counted.
    reml_ll_norm = _reml_normalise(reml_ll, length(β̂))
    return (; β = β̂, Λ = Λ̂, residual_cov = D̂, σ_res = σ̂, rho12 = ρ̂,
            reml_loglik = reml_ll_norm, ml_loglik = ml_ll, loglik = reml_ll_norm,
            converged = Optim.converged(res), iterations = Optim.iterations(res),
            u_hat = û)
end
