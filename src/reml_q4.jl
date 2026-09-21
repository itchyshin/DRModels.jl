# reml_q4.jl -- REML for the q=4 PLSM.
#
# REML = integrate out the location AND scale fixed effects -- beta_mu (mu1, mu2)
# AND beta_sigma (log-sigma1, log-sigma2) -- jointly with the latent random effects,
# via a BORDERED augmented state with ZERO / flat prior on those fixed effects.
# beta_sigma enters the leaf likelihood through (etas + u) on the log-sigma axes
# exactly as beta_mu enters through (eta + u) on the mean axes, so the SAME bordered
# correction extends to all four axes (this is what makes the SCALE among-axis SDs
# REML-corrected, not just the mean ones). Only beta_rho -- which has no u-axis --
# stays an outer parameter.
#
# STANDARD REML FORMULA (Patterson & Thompson 1971 / lme4 / ASReml):
#
#   L_REML(phi) = L_ML(phi, beta_hat)
#                 - 0.5 * logdet(Xtilde' H_uu^{-1} Xtilde)
#
# where
#   phi           = (beta_rho, Lambda) -- outer params; beta_mu AND beta_sigma profiled
#   beta_hat      = ML estimate of (beta_mu, beta_sigma) at phi (conditional Newton)
#   H_uu          = joint NLL Hessian wrt latents at mode (CHOLMOD from estep_mode)
#   Xtilde        = (n_u x n_beta) lifted design over ALL FOUR axes:
#                   row 4(t-1)+1 = X1[i,:]  (mu1),    row 4(t-1)+2 = X2[i,:]  (mu2),
#                   row 4(t-1)+3 = Xs1[i,:] (logσ1),  row 4(t-1)+4 = Xs2[i,:] (logσ2)
#
# The correction -0.5 logdet(S) is negative, so L_REML < L_ML at the same phi.
# Larger Lambda -> larger H_uu^{-1} -> larger logdet(S) -> bigger penalty:
# REML pushes Lambda LARGER than ML -- the defining less-biased REML property.
#
# KEY SIMPLIFICATION: d leaf_nll / d beta_axis[k] = d leaf_nll / d u[axis] * X_axis[i,k]
# for EVERY axis (mu and log-sigma alike, since the fixed effect and the RE enter the
# leaf as eta_axis + u_axis). So Xtilde is just each axis's design placed at that axis's
# position in u -- no new leaf derivatives, just reusing the existing leaf_etas layout.
#
# ADDITIVE BUILD: does NOT modify any of the working engine files:
#   sparse_aug_plsm.jl, fit_q4_sparse_tmb.jl, sparse_em_fit.jl, fit_ml_q4.jl
#
# GATE (task requirement):
#   (a) synthetic p=30, nrep=3: diag(Lambda_REML) >= diag(Lambda_ML) (component-wise)
#   (b) gradient ~ 0 at REML optimum (finite-diff check, max|g| < 0.5)
#   (c) runs at p=100 real q4_p100 without error, finite negative logLik
#   (baseline) fit_q4_sparse_tmb still gives logLik ~ -256.51
#
# Run:
#   ~/.juliaup/bin/julia --project="/Users/z3437171/Dropbox/Github Local/drm-julia-poc/julia" \
#       "/Users/z3437171/Dropbox/Github Local/drm-julia-poc/julia/drm_q4/reml_q4.jl"

using LinearAlgebra, SparseArrays, ForwardDiff, Statistics, Printf, Optim, Random
# Additive include into the DRModels module: the q=4 engine symbols (AugProblem,
# lc_to_Λ, Λ_to_lc, prior_precision, estep_mode, laplace_ll, leaf_etas,
# leaf_hess, leaf_nll, RHO_GUARD, fit_q4_sparse_tmb, pack_theta) are already in
# module scope via fit_q4_sparse_tmb.jl's include chain — no self-include here.

# ---------------------------------------------------------------------------
# phi layout: (beta_s1[ks1], beta_s2[ks2], beta_rho[kr], lc[10])
# total phi-length = ks1 + ks2 + kr + 10.   beta_mu is ABSENT (profiled out).
# ---------------------------------------------------------------------------

function phi_widths(prob::AugProblem)
    return size(prob.Xs1, 2), size(prob.Xs2, 2), size(prob.Xr, 2)
end

# phi now holds ONLY beta_rho (kr coefs) + the 10 log-Cholesky entries of Λ.
# beta_mu AND beta_sigma are profiled out — estimated by the conditional Newton at
# each phi — so they are ABSENT from phi.   phi layout = (beta_rho[kr], lc[10]).
function phi_len(prob::AugProblem)
    return size(prob.Xr, 2) + 10
end

function unpack_phi(prob::AugProblem, phi::AbstractVector{T}) where {T}
    kr  = size(prob.Xr, 2)
    rho = phi[1:kr]
    lc  = phi[kr+1:kr+10]
    return rho, lc
end

function pack_phi(prob::AugProblem, rho, Lam)
    vcat(rho, Λ_to_lc(Lam))
end

# ---------------------------------------------------------------------------
# Build Xtilde_mu -- the lifted (n_u x n_beta_mu) mean design matrix.
# Nonzero only at leaf mean-axis rows: axis1 gets X1 rows, axis2 gets X2 rows.
# When a leaf has multiple observations, all are accumulated (matching H_uu).
# ---------------------------------------------------------------------------
function build_Xmu_lifted(prob::AugProblem)
    k1 = size(prob.X1, 2); k2 = size(prob.X2, 2)
    nbmu = k1 + k2
    nu   = 4 * prob.n_total
    rows_I = Int[]; cols_J = Int[]; vals_V = Float64[]
    @inbounds for i in eachindex(prob.leaf_node)
        t    = prob.leaf_node[i]
        base = 4*(t - 1)
        for c in 1:k1
            push!(rows_I, base + 1); push!(cols_J, c);    push!(vals_V, prob.X1[i, c])
        end
        for c in 1:k2
            push!(rows_I, base + 2); push!(cols_J, k1+c); push!(vals_V, prob.X2[i, c])
        end
    end
    return sparse(rows_I, cols_J, vals_V, nu, nbmu)
end

# ---------------------------------------------------------------------------
# Conditional Newton on the profiled fixed effects (beta_mu1, beta_mu2, beta_s1,
# beta_s2) at fixed u_hat, over ALL data rows.  beta_rho stays fixed (from phi).
# Unlike mstep_beta (which uses 1:prob.p and misses replicate observations),
# this iterates eachindex(prob.leaf_node) so it is correct for nrep >= 1.
# ---------------------------------------------------------------------------
function cond_newton_beta(prob::AugProblem, u_hat::Vector{Float64},
                          beta_full; n_newton::Int=20, tol::Float64=1e-10)
    k1 = size(prob.X1, 2); k2 = size(prob.X2, 2)
    ks1 = size(prob.Xs1, 2); ks2 = size(prob.Xs2, 2)
    o1 = 0; o2 = k1; o3 = k1+k2; o4 = k1+k2+ks1
    etas_r = prob.Xr * beta_full.rho   # rho is NOT profiled — held fixed here

    # Objective: NLL over all data rows as a function of [bmu1; bmu2; bs1; bs2]
    function f_beta(bv)
        bm1 = bv[o1+1:o1+k1];  bm2 = bv[o2+1:o2+k2]
        bs1 = bv[o3+1:o3+ks1]; bs2 = bv[o4+1:o4+ks2]
        eta1  = prob.X1  * bm1; eta2  = prob.X2  * bm2
        etas1 = prob.Xs1 * bs1; etas2 = prob.Xs2 * bs2
        tot  = zero(eltype(bv))
        @inbounds for i in eachindex(prob.leaf_node)
            t    = prob.leaf_node[i]; base = 4*(t - 1)
            ublk = (u_hat[base+1], u_hat[base+2], u_hat[base+3], u_hat[base+4])
            tot += leaf_nll(ublk, prob.y1[i], prob.y2[i],
                            eta1[i], eta2[i], etas1[i], etas2[i], etas_r[i])
        end
        return tot
    end

    bv = vcat(beta_full.mu1, beta_full.mu2, beta_full.s1, beta_full.s2)
    for _ in 1:n_newton
        g = ForwardDiff.gradient(f_beta, bv)
        H = ForwardDiff.hessian(f_beta, bv)
        local step = g
        for lam in (0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2)
            ch = cholesky(Symmetric(H + lam*I); check=false)
            if issuccess(ch); step = ch \ g; break; end
        end
        f0 = f_beta(bv); alpha = 1.0; bvn = bv .- alpha .* step
        for _ in 1:25
            (f_beta(bvn) <= f0 || alpha < 1e-8) && break
            alpha *= 0.5; bvn = bv .- alpha .* step
        end
        bv = bvn
        norm(alpha .* step) < tol && break
    end
    return (mu1 = bv[o1+1:o1+k1],  mu2 = bv[o2+1:o2+k2],
            s1  = bv[o3+1:o3+ks1], s2  = bv[o4+1:o4+ks2])
end

# ---------------------------------------------------------------------------
# Augmented-state layout for the PROFILED fixed effects beta = (mu1, mu2, s1, s2).
# `Xax[d]` is axis d's design, `wax[d]` its width, `off[d]` its column offset in
# the stacked beta vector. Axis order 1=mu1, 2=mu2, 3=logσ1, 4=logσ2 — the same
# order the latent u block uses, which is what makes the lift R_i (docs/src/
# developer-notes/reml-q4-exact-gradient.md §1.1) a plain re-indexing.
# ---------------------------------------------------------------------------
function _reml_axis_layout(prob::AugProblem)
    Xax = (prob.X1, prob.X2, prob.Xs1, prob.Xs2)
    wax = (size(prob.X1, 2), size(prob.X2, 2),
           size(prob.Xs1, 2), size(prob.Xs2, 2))
    off = (0, wax[1], wax[1]+wax[2], wax[1]+wax[2]+wax[3])
    return Xax, wax, off, sum(wax)
end

_reml_beta_vec(prob::AugProblem, b) = vcat(b.mu1, b.mu2, b.s1, b.s2)

function _reml_beta_nt(prob::AugProblem, bv::AbstractVector, rho)
    _, wax, off, _ = _reml_axis_layout(prob)
    return (mu1 = bv[off[1]+1:off[1]+wax[1]], mu2 = bv[off[2]+1:off[2]+wax[2]],
            s1  = bv[off[3]+1:off[3]+wax[3]], s2  = bv[off[4]+1:off[4]+wax[4]],
            rho = rho)
end

# ---------------------------------------------------------------------------
# Bordered blocks of the JOINT Hessian over z = (u, beta):
#   H_u_beta[base+d, off[dp]+k] = Hb[d,dp] * X_dp[i,k]
#   H_beta_beta[off[d]+r, off[dp]+c] = Hb[d,dp] * X_d[i,r] * X_dp[i,c]
# i.e. the leaf contribution is R_i' Hb R_i for the lift R_i = [E_i | F_i].
# Extracted from reml_ll_and_mode so the exact gradient reuses the SAME build.
# ---------------------------------------------------------------------------
function _reml_border_blocks(prob::AugProblem, u_hat::Vector{Float64}, beta_full)
    nu = 4 * prob.n_total
    Xax, wax, off, nbeta = _reml_axis_layout(prob)
    e1, e2, es1, es2, er = leaf_etas(prob, beta_full)
    H_u_beta    = zeros(nu, nbeta)
    H_beta_beta = zeros(nbeta, nbeta)
    @inbounds for i in eachindex(prob.leaf_node)
        t    = prob.leaf_node[i]; base = 4*(t - 1)
        ublk = [u_hat[base+1], u_hat[base+2], u_hat[base+3], u_hat[base+4]]
        Hb   = leaf_hess(ublk, prob.y1[i], prob.y2[i],
                         e1[i], e2[i], es1[i], es2[i], er[i],
                         prob.obs1[i], prob.obs2[i])
        for d in 1:4, dp in 1:4
            Hdd = Hb[d, dp]
            Hdd == 0.0 && continue
            Xdp = Xax[dp]; odp = off[dp]
            for k in 1:wax[dp]
                H_u_beta[base+d, odp+k] += Hdd * Xdp[i, k]
            end
            Xd = Xax[d]; od = off[d]
            for r in 1:wax[d], c in 1:wax[dp]
                H_beta_beta[od+r, odp+c] += Hdd * Xd[i, r] * Xdp[i, c]
            end
        end
    end
    return H_u_beta, H_beta_beta
end

# ∇_z J at z = (u, beta): the u block is `joint_grad`; the beta block is the
# same leaf gradient chained through each axis's design (R_i' again).
function _reml_grad_z(prob::AugProblem, P::SparseMatrixCSC,
                      u::Vector{Float64}, beta_full)
    Xax, wax, off, nbeta = _reml_axis_layout(prob)
    g_u = joint_grad(prob, P, u, beta_full)
    e1, e2, es1, es2, er = leaf_etas(prob, beta_full)
    g_b = zeros(nbeta)
    @inbounds for i in eachindex(prob.leaf_node)
        t = prob.leaf_node[i]; base = 4*(t - 1)
        gb = leaf_grad([u[base+1], u[base+2], u[base+3], u[base+4]],
                       prob.y1[i], prob.y2[i], e1[i], e2[i], es1[i], es2[i], er[i],
                       prob.obs1[i], prob.obs2[i])
        for d in 1:4
            gb[d] == 0.0 && continue
            Xd = Xax[d]; od = off[d]
            for k in 1:wax[d]
                g_b[od+k] += gb[d] * Xd[i, k]
            end
        end
    end
    return g_u, g_b
end

# ---------------------------------------------------------------------------
# JOINT (u, beta) Newton mode-finder on the bordered system, never factorising
# the (n_u + n_beta) matrix: with A = H_uu (CHOLMOD), B = H_uβ, D = H_ββ,
# C = A^{-1}B and S = D - B'C,
#     𝓗^{-1} v  =  ( A^{-1}v_u + C q ,  -q ),   q = S^{-1}(C'v_u - v_β).
# Returns (u, beta_full, ch_H, C, S_chol, ‖∇_z J‖). The alternation in
# `reml_ll_and_mode` exits on a RELATIVE beta criterion (~1e-4·‖β‖), which is
# far too loose for an exact gradient certified at 1e-6: the implicit-function
# derivation is valid only at ∇_z J = 0. A few of these steps cost one extra
# sparse solve each and take ‖∇_z J‖ to ~1e-11.
# ---------------------------------------------------------------------------
function _reml_joint_newton(prob::AugProblem, P::SparseMatrixCSC,
                            u0::Vector{Float64}, beta_full;
                            max_iter::Int = 25, tol::Float64 = 1e-10)
    u  = copy(u0)
    bv = _reml_beta_vec(prob, beta_full)
    rho = beta_full.rho
    b   = _reml_beta_nt(prob, bv, rho)
    Hobs = build_Huu(prob, P, u, b); ch_H, _ = sparse_pd_chol(Hobs)
    B, D  = _reml_border_blocks(prob, u, b)
    nbeta = size(B, 2)
    C  = Matrix{Float64}(undef, length(u), nbeta)
    for j in 1:nbeta; C[:, j] = ch_H \ B[:, j]; end
    ch_S = cholesky(Symmetric((D - B'C + (D - B'C)') / 2); check = false)
    g_u, g_b = _reml_grad_z(prob, P, u, b)
    gn = sqrt(sum(abs2, g_u) + sum(abs2, g_b))
    f  = joint_nll(prob, P, u, b)

    for _ in 1:max_iter
        (gn < tol || !issuccess(ch_S)) && break
        q   = ch_S \ (C'g_u .- g_b)
        d_u = (ch_H \ g_u) .+ C * q
        d_b = -q
        (all(isfinite, d_u) && all(isfinite, d_b)) || break
        α = 1.0; accepted = false
        for _ in 1:30
            un = u .- α .* d_u
            bn = _reml_beta_nt(prob, bv .- α .* d_b, rho)
            fn = joint_nll(prob, P, un, bn)
            if isfinite(fn) && fn <= f + 1e-12 * abs(f)
                u = un; bv = bv .- α .* d_b; b = bn; f = fn; accepted = true
                break
            end
            α *= 0.5
        end
        accepted || break
        Hobs = build_Huu(prob, P, u, b); ch_H, _ = sparse_pd_chol(Hobs)
        B, D = _reml_border_blocks(prob, u, b)
        for j in 1:nbeta; C[:, j] = ch_H \ B[:, j]; end
        ch_S = cholesky(Symmetric((D - B'C + (D - B'C)') / 2); check = false)
        g_u, g_b = _reml_grad_z(prob, P, u, b)
        gn_new = sqrt(sum(abs2, g_u) + sum(abs2, g_b))
        gn_new >= gn && (gn = gn_new; break)   # no further progress
        gn = gn_new
    end
    return u, b, ch_H, C, ch_S, gn
end

# ---------------------------------------------------------------------------
# REML log-likelihood at outer parameters phi = (beta_rho, lc).
#   1. Unpack phi, build P.
#   2. Warm E-step to get u_hat.
#   3. Conditional Newton on (beta_mu1,beta_mu2,beta_s1,beta_s2) at fixed u_hat.
#   4. Re-run E-step with updated beta for consistency.
#   5. L_ML = laplace_ll(u_hat, beta_hat, P).
#   6. REML correction: S = Xtilde' H_uu^{-1} Xtilde (all 4 axes);  -0.5 logdet(S).
# Returns: (reml_ll_val, u_hat, ch_H, beta_full, P)
# ---------------------------------------------------------------------------
function reml_ll_and_mode(prob::AugProblem, Q_cond::SparseMatrixCSC,
                          phi::Vector{Float64};
                          u0=nothing, beta0=nothing, n_newton::Int=40)
    rho_coef, lc = unpack_phi(prob, phi)
    Lam  = lc_to_Λ(lc)
    P    = prior_precision(Q_cond, inv(Lam))

    # Initial (beta_mu, beta_sigma) guess: warm from beta0 (the cached ML/last state)
    # or cold (OLS mean intercepts, zero log-sigma).
    if beta0 === nothing
        bm1 = prob.X1 \ prob.y1; bm2 = prob.X2 \ prob.y2
        bs1 = zeros(size(prob.Xs1, 2)); bs2 = zeros(size(prob.Xs2, 2))
    else
        bm1 = beta0.mu1; bm2 = beta0.mu2; bs1 = beta0.s1; bs2 = beta0.s2
    end
    beta_full = (mu1 = bm1, mu2 = bm2, s1 = bs1, s2 = bs2, rho = rho_coef)

    # Alternate E-step and the conditional beta Newton until jointly converged.
    # The Schur complement S is only PD at the joint mode of jn(u, beta_profiled).
    u_hat = u0 === nothing ? zeros(4*prob.n_total) : Vector{Float64}(u0)
    ch_H  = nothing
    # #526: surface whether the alternation actually settled, rather than
    # letting a budget exhaustion pass silently into a flag that only reported
    # the outer optimiser's verdict. Two thresholds on purpose: the LOOP still
    # exits early at the strict 1e-6 (fitting behaviour untouched), but the
    # REPORTED verdict uses a relative criterion with a measured floor —
    # instrumentation on the #484 fixtures showed delta_b sitting in a flat
    # ~6.2e-6 LIMIT CYCLE across all 15 alternations while the engine-level
    # gradient met g_tol, i.e. a machine-level equilibrium around the joint
    # mode, not a stall. 1e-4·(1 + ‖β‖) passes that oscillation with ~30x
    # margin and still fails a materially moving beta (delta_b ~ 1e-2).
    last_delta = Inf
    for alt_it in 1:15    # up to 15 alternations for cold starts
        u_hat, ch_H, _ = estep_mode(prob, P, beta_full; u0=u_hat, n_newton=n_newton)
        u_hat = Vector{Float64}(u_hat)
        b_new = cond_newton_beta(prob, u_hat, beta_full; n_newton=20)
        delta_b = norm(b_new.mu1 .- beta_full.mu1) + norm(b_new.mu2 .- beta_full.mu2) +
                  norm(b_new.s1  .- beta_full.s1)  + norm(b_new.s2  .- beta_full.s2)
        beta_full = (mu1 = b_new.mu1, mu2 = b_new.mu2,
                     s1  = b_new.s1,  s2  = b_new.s2, rho = rho_coef)
        last_delta = delta_b
        delta_b < 1e-6 && break   # strict exit (kept)
        # PERF (2026-08-28 speed grid): the strict exit was measured NEVER to
        # fire at genuine optima -- delta_b sits in a flat ~6.2e-6 limit
        # cycle -- so every objective evaluation burned all 15 alternations
        # where ~2 suffice, and the q4 REML cell was the ONE route where
        # engine="julia" lost to native TMB (0.46x). Also exit at the SAME
        # calibrated relative criterion the #526 flag reports, after at least
        # two alternations so a cold start cannot exit on its first,
        # still-moving step.
        if alt_it >= 2
            bs = norm(beta_full.mu1) + norm(beta_full.mu2) +
                 norm(beta_full.s1) + norm(beta_full.s2)
            delta_b < 1e-4 * (1 + bs) && break
        end
    end
    beta_scale = norm(beta_full.mu1) + norm(beta_full.mu2) +
                 norm(beta_full.s1) + norm(beta_full.s2)
    inner_converged = last_delta < 1e-4 * (1 + beta_scale)
    # Final E-step with converged beta
    u_hat, ch_H, _ = estep_mode(prob, P, beta_full; u0=u_hat, n_newton=n_newton)
    u_hat = Vector{Float64}(u_hat)

    ml_ll = laplace_ll(prob, P, beta_full, u_hat, ch_H)

    # REML correction: -0.5 * logdet(S), S the Schur complement of the profiled
    # fixed effects beta = (beta_mu1,beta_mu2,beta_s1,beta_s2) in the joint Hessian
    # of jn(u, beta).  Because each beta_axis enters the leaf as (eta_axis + u_axis),
    #   d^2 leaf / d u[d] d beta_axis[d'][k] = Hb[d, d'] * X_axis[d'][i, k],
    # where Hb is the full 4x4 leaf Hessian and X_axis[d'] is that axis's design
    # (axis 1=X1 mu1, 2=X2 mu2, 3=Xs1 logσ1, 4=Xs2 logσ2).  This is the mean-only
    # build generalised to all 4 axes — including the mean<->scale cross blocks
    # Hb[1,3] etc. that carry the SCALE REML correction.
    nu  = 4 * prob.n_total
    _, _, _, nbeta = _reml_axis_layout(prob)
    H_u_beta, H_beta_beta = _reml_border_blocks(prob, u_hat, beta_full)

    # Schur complement: S = H_beta_beta - H_u_beta' * (H_uu^{-1} * H_u_beta)
    C   = Matrix{Float64}(undef, nu, nbeta)
    for j in 1:nbeta
        C[:, j] = ch_H \ H_u_beta[:, j]
    end
    S     = H_beta_beta - H_u_beta' * C
    S_sym = Symmetric((S + S') / 2)
    ch_S  = cholesky(S_sym; check=false)
    if !issuccess(ch_S)
        # S non-PD: mode/beta not jointly converged, or phi outside valid region.
        # Return -Inf so the optimizer avoids this point (barrier).
        return -Inf, u_hat, ch_H, beta_full, P, inner_converged
    end
    ld_S  = logdet(ch_S)

    return ml_ll - 0.5*ld_S, u_hat, ch_H, beta_full, P, inner_converged
end

# ---------------------------------------------------------------------------
# EXACT gradient of the REML objective (#575).
#
# The implemented objective collapses to a single Laplace form over the
# AUGMENTED state z = (u, beta):
#
#     L_REML(phi) = -J(z_hat; phi) - 0.5*logdet(H) + 0.5*logdet(P)
#     H = [A B; B' D],  logdet H = logdet A + logdet S
#
# — structurally identical to the ML objective in fit_q4_sparse_tmb.jl, so the
# exact O(p) implicit-function gradient transfers term-by-term with the leaf
# selected-inverse block Vblk replaced by
#
#     Omega_i = Vsel[leaf block] + G_i S^{-1} G_i',   G_i = C[leaf block, :] - F_i
#
# (F_i the leaf's rows of the lifted per-axis designs). H is NEVER factorised:
# every quantity comes from the existing CHOLMOD factor of A, the Takahashi
# selected inverse of A, and the small dense S. Full derivation with the
# term-by-term alignment table:
#   docs/src/developer-notes/reml-q4-exact-gradient.md
# ---------------------------------------------------------------------------

# The local `_reml_prior_precision` guard (#579) against `prior_precision`'s
# then-degenerate zero-dropping at an exactly diagonal Λ is gone: #577 fixed
# `prior_precision` itself (src/sparse_aug_plsm.jl) to store the full q×q
# axis block unconditionally, so the REML path now calls it directly. See
# docs/src/developer-notes/reml-q4-exact-gradient.md §3 and #563.

# Joint mode at phi, Newton-certified. Returns everything the value and the
# gradient both need (P, the certified z_hat, A's factor, C = A^{-1}B, S's
# factor, and the achieved ‖∇_z J‖).
function _reml_exact_state(prob::AugProblem, Q_cond::SparseMatrixCSC,
                           phi::AbstractVector{<:Real};
                           u0 = nothing, beta0 = nothing, n_newton::Int = 40,
                           joint_iter::Int = 25, joint_tol::Float64 = 1e-10)
    phiv = Vector{Float64}(phi)
    rho_coef, lc = unpack_phi(prob, phiv)
    Lam = lc_to_Λ(lc)
    P   = prior_precision(Q_cond, inv(Lam))
    # Reuse the existing alternation to get into the right neighbourhood, then
    # certify with joint Newton (the alternation's relative beta exit is far too
    # loose to support an exact gradient — see the derivation note §2.4).
    _, u_a, _, b_a, _, _ = reml_ll_and_mode(prob, Q_cond, phiv;
                                            u0 = u0, beta0 = beta0, n_newton = n_newton)
    u, b, ch_H, C, ch_S, gz = _reml_joint_newton(prob, P, Vector{Float64}(u_a), b_a;
                                                 max_iter = joint_iter, tol = joint_tol)
    return (rho = rho_coef, lc = lc, Lam = Lam, P = P, u = u, beta = b,
            ch_H = ch_H, C = C, ch_S = ch_S, gz = gz)
end

"""
    reml_nll_exact(prob, Q_cond, phi; u0, beta0, n_newton) -> Float64

The q=4 REML objective as a NEGATIVE, UNNORMALISED restricted log-likelihood at
`phi = (beta_rho, lc)`, evaluated at a joint-Newton-certified `(û, β̂)`. Same
objective as `reml_ll_and_mode` up to sign; the difference is that the
joint mode is driven to `‖∇_z J‖ ≈ 1e-10` instead of the alternation's relative
beta criterion, which makes the value a smooth, deterministic function of `phi`
— what a finite-difference reference (and a Newton-grade certification) needs.
Returns `Inf` on the non-PD Schur barrier.
"""
function reml_nll_exact(prob::AugProblem, Q_cond::SparseMatrixCSC,
                        phi::AbstractVector{<:Real};
                        u0 = nothing, beta0 = nothing, n_newton::Int = 40,
                        joint_iter::Int = 25, joint_tol::Float64 = 1e-10)
    st = _reml_exact_state(prob, Q_cond, phi; u0 = u0, beta0 = beta0,
                           n_newton = n_newton, joint_iter = joint_iter,
                           joint_tol = joint_tol)
    issuccess(st.ch_S) || return Inf
    ll = laplace_ll(prob, st.P, st.beta, st.u, st.ch_H) - 0.5 * logdet(st.ch_S)
    return isfinite(ll) ? -ll : Inf
end

"""
    reml_nll_and_exact_grad(prob, Q_cond, phi; u0, beta0, n_newton)
        -> (nll, grad, û, β̂, ch_H, z_residual)

Negative unnormalised REML log-likelihood and its EXACT gradient over
`phi = (beta_rho, lc)`. `z_residual` is `‖∇_z J(ẑ)‖` — the exact gradient is
valid only where this is small, so callers should check it before certifying
convergence on `grad`.
"""
function reml_nll_and_exact_grad(prob::AugProblem, Q_cond::SparseMatrixCSC,
                                 phi::AbstractVector{<:Real};
                                 u0 = nothing, beta0 = nothing, n_newton::Int = 40,
                                 joint_iter::Int = 25, joint_tol::Float64 = 1e-10)
    phiv = Vector{Float64}(phi)
    kr   = size(prob.Xr, 2)
    o_lc = kr
    nph  = length(phiv)

    st = _reml_exact_state(prob, Q_cond, phiv; u0 = u0, beta0 = beta0,
                           n_newton = n_newton, joint_iter = joint_iter,
                           joint_tol = joint_tol)
    P = st.P; u_hat = st.u; b = st.beta; ch_H = st.ch_H; C = st.C; ch_S = st.ch_S
    if !issuccess(ch_S)
        return Inf, fill(NaN, nph), u_hat, b, ch_H, st.gz
    end
    ll = laplace_ll(prob, P, b, u_hat, ch_H) - 0.5 * logdet(ch_S)
    if !isfinite(ll)
        return Inf, fill(NaN, nph), u_hat, b, ch_H, st.gz
    end

    Xax, wax, off, nbeta = _reml_axis_layout(prob)
    nu   = 4 * prob.n_total
    N    = prob.n_total
    Λi   = inv(st.Lam)
    Sinv = Matrix(inv(ch_S))          # nbeta is the marginalised width (small)
    CS   = C * Sinv                   # nu × nbeta
    Vsel = takahashi_selinv(ch_H)     # A^{-1} at the L+L' pattern, O(p)
    e1, e2, es1, es2, er = leaf_etas(prob, b)

    grad = zeros(nph)

    # --- C1: ∇_phi J(ẑ; phi) with ẑ frozen (single-level AD, no CHOLMOD). ----
    jn_of_phi = function (pv::AbstractVector)
        rho_t = pv[1:kr]; lc_t = pv[kr+1:kr+10]
        Pt = prior_precision(Q_cond, inv(lc_to_Λ(lc_t)))
        βt = (mu1 = b.mu1, mu2 = b.mu2, s1 = b.s1, s2 = b.s2, rho = rho_t)
        return joint_nll_T(prob, Pt, u_hat, βt)
    end
    grad .+= ForwardDiff.gradient(jn_of_phi, phiv)

    # --- C2: −0.5 ∇ logdet P = +0.5·N·∇_lc logdet Λ (analytic). -------------
    grad[o_lc+1:o_lc+10] .+=
        0.5 * N .* ForwardDiff.gradient(v -> logdet(Symmetric(lc_to_Λ(v))), st.lc)

    # --- C3a (beta_rho logdet-H trace) and I1 (v = 0.5 ∇_z logdet H) --------
    # One pass over data rows; both need the same per-leaf Omega_i.
    Gst = zeros(4, 4)                 # Q-pattern accumulator for the lc trace
    v_u = zeros(nu); v_b = zeros(nbeta)
    Gi  = zeros(4, nbeta)
    @inbounds for i in eachindex(prob.leaf_node)
        t = prob.leaf_node[i]; bt = 4 * (t - 1)
        ublk = [u_hat[bt+1], u_hat[bt+2], u_hat[bt+3], u_hat[bt+4]]
        # G_i = C[leaf block, :] − F_i, then Omega = Vsel_blk + G_i S^{-1} G_i'.
        for d in 1:4
            for m in 1:nbeta; Gi[d, m] = C[bt+d, m]; end
            Xd = Xax[d]; od = off[d]
            for k in 1:wax[d]; Gi[d, od+k] -= Xd[i, k]; end
        end
        Ω = Gi * Sinv * Gi'
        for a in 1:4, bb in 1:4
            Ω[a, bb] += Vsel[bt+a, bt+bb]
        end

        if kr > 0
            dHr = ForwardDiff.derivative(
                e -> vec(leaf_hess(ublk, prob.y1[i], prob.y2[i],
                                   e1[i], e2[i], es1[i], es2[i], e,
                                   prob.obs1[i], prob.obs2[i])), er[i])
            s = 0.0
            for bb in 1:4, a in 1:4; s += Ω[a, bb] * dHr[(bb-1)*4 + a]; end
            for c in 1:kr; grad[c] += 0.5 * s * prob.Xr[i, c]; end
        end

        T = leaf_hess_du(ublk, prob.y1[i], prob.y2[i],
                         e1[i], e2[i], es1[i], es2[i], er[i],
                         prob.obs1[i], prob.obs2[i])
        for c in 1:4
            acc = 0.0
            for bb in 1:4, a in 1:4; acc += Ω[a, bb] * T[a, bb, c]; end
            vt = 0.5 * acc
            v_u[bt+c] += vt
            Xc = Xax[c]; oc = off[c]
            for k in 1:wax[c]; v_b[oc+k] += vt * Xc[i, k]; end
        end
    end

    # --- C3b: lc logdet-H trace. Only the u-u block of H depends on lc, via
    # P = kron(Q, Λ^{-1}); the needed inverse block is W_uu = A^{-1} + C S^{-1} C'.
    rowsQ = rowvals(Q_cond); valsQ = nonzeros(Q_cond)
    @inbounds for tcol in 1:N
        for idx in nzrange(Q_cond, tcol)
            s = rowsQ[idx]; q = valsQ[idx]
            bs = 4 * (s - 1); btt = 4 * (tcol - 1)
            for a in 1:4, bb in 1:4
                corr = 0.0
                for m in 1:nbeta; corr += CS[btt+a, m] * C[bs+bb, m]; end
                Gst[bb, a] += q * (Vsel[btt+a, bs+bb] + corr)
            end
        end
    end
    dΛ = ForwardDiff.jacobian(lc_to_Λ, st.lc)      # 16×10
    for k in 1:10
        dΛk = reshape(@view(dΛ[:, k]), 4, 4)
        Mk  = -Λi * dΛk * Λi
        acc = 0.0
        for a in 1:4, bb in 1:4; acc += Gst[bb, a] * Mk[bb, a]; end
        grad[o_lc + k] += 0.5 * acc
    end

    # --- I2: w = H^{-1} v via the same bordered block solve. ----------------
    qv  = ch_S \ (C'v_u .- v_b)
    w_u = (ch_H \ v_u) .+ C * qv
    w_b = -qv

    # --- I3: −∇_phi[ (∇_z J)' w ] at frozen (ẑ, w). ------------------------
    scalar_of_phi = function (pv::AbstractVector)
        rho_t = pv[1:kr]; lc_t = pv[kr+1:kr+10]
        Pt = prior_precision(Q_cond, inv(lc_to_Λ(lc_t)))
        βt = (mu1 = b.mu1, mu2 = b.mu2, s1 = b.s1, s2 = b.s2, rho = rho_t)
        acc = dot(joint_grad_T(prob, Pt, u_hat, βt), w_u)
        f1, f2, fs1, fs2, fr = leaf_etas(prob, βt)
        @inbounds for i in eachindex(prob.leaf_node)
            t = prob.leaf_node[i]; bt = 4 * (t - 1)
            gb = leaf_grad([u_hat[bt+1], u_hat[bt+2], u_hat[bt+3], u_hat[bt+4]],
                           prob.y1[i], prob.y2[i], f1[i], f2[i], fs1[i], fs2[i], fr[i],
                           prob.obs1[i], prob.obs2[i])
            for d in 1:4
                Xd = Xax[d]; od = off[d]
                for k in 1:wax[d]; acc += gb[d] * Xd[i, k] * w_b[od+k]; end
            end
        end
        return acc
    end
    grad .-= ForwardDiff.gradient(scalar_of_phi, phiv)

    return -ll, grad, u_hat, b, ch_H, st.gz
end

# ---------------------------------------------------------------------------
# REML optimizer: LBFGS over phi = (beta_rho, lc).
# Gradient: central finite differences (phi is low-dimensional: kr+10).
# ---------------------------------------------------------------------------

"""
    _reml_normalise(reml_ll, n_beta)

Add the `(n_beta/2)·log(2π)` normalising constant to an unnormalised
Patterson–Thompson restricted log-likelihood, so DRModels.jl's bivariate REML routes
report on the same scale as lme4, glmmTMB, TMB — and as DRModels.jl's own univariate
REML routes.

`n_beta` counts only the **marginalised** fixed effects. Non-finite input passes
through unchanged so `-Inf` barriers and `NaN` sentinels keep their meaning.
"""
_reml_normalise(reml_ll, n_beta::Integer) =
    isfinite(reml_ll) ? reml_ll + 0.5 * n_beta * log(2π) : reml_ll

"""
    reml_objective_at(prob, Q_cond, phi; beta0=nothing, u0=nothing, n_newton=60)
        -> NamedTuple(reml_loglik, raw_reml_ll, beta, u, converged)

Evaluate the q=4 REML objective at a SUPPLIED `phi = (beta_rho, lc)` — the
variance-component / correlation half of `fit_q4_reml`'s parameter space —
reprofiling `beta_mu1/mu2/s1/s2` (the marginalised mean and scale fixed
effects) by the SAME conditional-Newton alternation `fit_q4_reml` runs
internally (`reml_ll_and_mode`), rather than re-running the outer LBFGS.

This is the diagnostic primitive behind cross-engine mode-finder-vs-
objective-translation checks: given another engine's fitted point
(mapped into DRModels.jl's `phi`/`beta` scale — see `pack_phi`, `Λ_to_lc`), call
this to ask "is DRModels.jl's OWN objective, evaluated AT that point, better or
worse than what DRModels.jl's own solver returned?" without hand-writing the
inner E-step/Newton alternation each time.

`beta0`/`u0` seed the conditional-Newton warm start (pass the other engine's
fixed effects, or `fit_q4_reml`'s returned `.beta`/`.u_hat`, as `beta0`/`u0`
respectively) so the inner profile has the best chance of reaching its true
conditional optimum at `phi` rather than stalling from a cold start.

`reml_loglik` is on the SAME normalised (Patterson–Thompson, `+ (n_β/2)·log(2π)`)
scale as `fit_q4_reml`'s `.reml_loglik` and as drmTMB/TMB/lme4/glmmTMB;
`raw_reml_ll` is the pre-normalisation value, exposed for constant-offset
sanity checks. `n_β` is `prob`'s combined `X1+X2+Xs1+Xs2` design width — the
same marginalised-fixed-effect count `fit_q4_reml` uses.

Returns `(reml_loglik, raw_reml_ll, beta, u, converged)`. `converged` is the
inner alternation's own convergence flag (`inner_converged` from
`reml_ll_and_mode`) — a barrier hit (non-PD Schur complement) surfaces as
`raw_reml_ll = reml_loglik = -Inf`, `converged = false`, rather than an error.
"""
function reml_objective_at(prob::AugProblem, Q_cond::SparseMatrixCSC, phi::AbstractVector;
                            beta0=nothing, u0=nothing, n_newton::Int=60)
    n_beta = size(prob.X1, 2) + size(prob.X2, 2) + size(prob.Xs1, 2) + size(prob.Xs2, 2)
    raw, u_hat, _, beta_hat, _, inner_converged = reml_ll_and_mode(
        prob, Q_cond, Vector{Float64}(phi); u0 = u0, beta0 = beta0, n_newton = n_newton)
    return (reml_loglik = _reml_normalise(raw, n_beta), raw_reml_ll = raw,
            beta = beta_hat, u = u_hat, converged = inner_converged)
end

"""
    fit_q4_reml(prob, Q_cond; beta0, Lambda0, [phi0], g_tol, ...) -> NamedTuple

Fit REML objective over phi = (beta_rho, lc). beta_mu AND beta_sigma (the location
and scale fixed effects) are profiled out internally; only beta_rho stays outer.

Returns NamedTuple: (phi, beta, Lambda, reml_loglik, ml_loglik, converged,
                     iterations, g_residual, f_calls, u_hat)

# Automatic warm restart

On some cells the REML LBFGS's first line-search step from the ML warm start
fails outright (zero accepted steps — a starting-value problem, not slow
convergence, so a bigger `iterations` or looser `g_tol` cannot fix it). That
exact stall (`!converged` with the minimizer still sitting at `phi0`) is
detected automatically and retried by re-deriving a coarser ML warm start and
continuing from there — judged at the SAME `g_tol` the caller passed, never a
loosened one. A REML fit whose first attempt already moves at all — converged
or not — is completely unaffected; see the block above `phi_hat` for the
mechanism. Needs `beta0`/`Lambda0` to fire (skipped if the caller supplied
`phi0` directly, since there is then nothing to re-derive a coarser start
from).

# Normalisation convention

`reml_loglik` reports the **normalised** Patterson–Thompson restricted
log-likelihood: the raw objective `ℓ_ML(θ, β̂) − ½ logdet(S)` plus
`(n_β/2)·log(2π)`, the constant from integrating the flat prior over the `n_β`
marginalised fixed effects (`n_β` = the combined width of the `beta_mu1`,
`beta_mu2`, `beta_s1`, `beta_s2` designs — exactly the Schur complement's
dimension; `beta_rho` is never marginalised, so it does not count). That matches
lme4, glmmTMB and TMB, so `reml_loglik` is directly comparable across engines.

The normalising constant affects the reported value, not the maximising
parameter estimates. In the q=4 cross-engine comparison, the constant was
**5.513631** and the remaining difference between optima was within **0.03**.
Comparisons must use the same normalisation before attributing differences to
the estimators.

The q=2 implementation uses the same derivation and normalisation arithmetic.
This q=4 comparison does not independently validate the q=2 route.
"""
function fit_q4_reml(prob::AugProblem, Q_cond::SparseMatrixCSC;
                     phi0=nothing, beta0=nothing, Lambda0=nothing,
                     g_tol::Float64=1e-3, iterations::Int=200,
                     n_newton::Int=40, h_fd::Float64=1e-5,
                     # `lc_zero`: same block-diagonal Σ_a constraint as the ML path —
                     # log-Cholesky indices (1..10) pinned to 0. Here they map into
                     # the phi vector (β_μ AND β_σ profiled out), at offset kr.
                     lc_zero::AbstractVector{<:Integer} = Int[],
                     verbose::Bool=false)
    kr   = size(prob.Xr, 2)
    o_lc = kr                            # lc block starts here in phi = (β_ρ, lc)
    lc_zero_idx = sort(unique(Int.(lc_zero)))
    all(1 .<= lc_zero_idx .<= 10) ||
        error("lc_zero indices must be in 1:10 (got $lc_zero_idx)")
    phi_zero = o_lc .+ lc_zero_idx       # absolute phi positions pinned to 0
    beta_ws = nothing                    # ML warm-start β to seed the profiled cache
    if phi0 === nothing
        beta0   === nothing && error("supply phi0 or (beta0, Lambda0)")
        Lambda0 === nothing && (Lambda0 = Matrix(0.3I(4)))
        # Warm-start from the ML optimum: at phi_ML the Schur complement S is
        # PD (the joint mode is well-defined), which stabilises the FD gradient.
        # The ML warm start honours the SAME lc_zero so the warm Λ is already
        # block-diagonal — phi0 lands on the constrained subspace.
        r_ml_ws = fit_q4_sparse_tmb(prob, Q_cond;
                                     β0=beta0, Λ0=Matrix(Lambda0),
                                     g_tol=max(g_tol*5, 1e-2),
                                     iterations=min(iterations, 100),
                                     n_newton=n_newton, lc_zero=lc_zero_idx)
        phi0 = pack_phi(prob, r_ml_ws.β.rho, r_ml_ws.Λ)
        beta_ws = (mu1=r_ml_ws.β.mu1, mu2=r_ml_ws.β.mu2,
                   s1=r_ml_ws.β.s1, s2=r_ml_ws.β.s2)
    end
    phi0 = copy(Vector{Float64}(phi0)); phi0[phi_zero] .= 0.0

    u_cache   = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    # Seed the profiled-beta cache with the ML warm fit (mu AND sigma) so the first
    # conditional Newton starts at the ML estimate, not a cold zero-log-sigma.
    beta_cache = Ref{Union{Nothing,NamedTuple}}(beta_ws)
    eval_cnt = Ref(0)
    nobs     = length(prob.leaf_node)
    nph      = length(phi0)

    # Objective AND exact gradient in one evaluation (#575).
    #
    # This replaced a central finite-difference gradient with step
    # `h_inner = max(h_fd, 5e-4)`, which re-ran the whole (u, beta) alternation
    # for each of the 2*nph perturbations. Two things were wrong with that.
    # (1) ACCURACY: each one-sided value carried the alternation's own relative
    # exit slack, so on the biv-q4-phylo-reml cell the gradient's noise was the
    # same order as `g_tol` itself -- convergence was being certified at the
    # noise floor, which is what #575 is (see docs/src/developer-notes/
    # reml-q4-exact-gradient.md). (2) COST: 2*nph + 1 mode solves per gradient
    # instead of one.
    #
    # `reml_nll_and_exact_grad` returns the value at a JOINT-Newton-certified
    # mode together with the exact gradient, so the line search and the gradient
    # see the same objective by construction. `gz_last` records the achieved
    # ‖∇_z J‖ so the convergence flag can require the mode to have been found,
    # not merely the outer optimiser to have stopped.
    gz_last = Ref(Inf)
    fg! = function (F, G, phiv)
        eval_cnt[] += 1
        pv = Vector{Float64}(phiv)
        local nllv, gv, uv, bv, gz
        try
            nllv, gv, uv, bv, _, gz = reml_nll_and_exact_grad(
                prob, Q_cond, pv; u0=u_cache[], beta0=beta_cache[],
                n_newton=n_newton)
        catch e
            (e isa DomainError || e isa LinearAlgebra.PosDefException ||
             e isa LinearAlgebra.SingularException || e isa ArgumentError) || rethrow(e)
            return Inf
        end
        (isfinite(nllv) && all(isfinite, gv)) || return Inf
        u_cache[]    = uv
        beta_cache[] = (mu1=bv.mu1, mu2=bv.mu2, s1=bv.s1, s2=bv.s2)
        gz_last[]    = gz
        if G !== nothing
            copyto!(G, gv ./ nobs)
            # Pin the constrained lc directions (block-diagonal Sigma_a): zero
            # their gradient so LBFGS never steps off the constrained subspace.
            isempty(phi_zero) || (G[phi_zero] .= 0.0)
        end
        return nllv / nobs
    end

    # REML optimization via LBFGS starting from phi0.
    # The REML landscape is well-behaved NEAR the ML optimum (S is PD there).
    # We use BackTracking with a line search that rejects Inf evaluations.
    od  = Optim.NLSolversBase.only_fg!(fg!)
    _optimize_phi(start_phi, gtol_i, iters_i) = Optim.optimize(
        od, Vector{Float64}(start_phi),
        LBFGS(m=5,
              alphaguess=Optim.LineSearches.InitialStatic(scaled=true),
              linesearch=Optim.LineSearches.BackTracking(order=3)),
        Optim.Options(g_tol=gtol_i, f_reltol=1e-5, successive_f_tol=10,
                      iterations=iters_i, show_trace=verbose, show_every=1),
    )

    res = _optimize_phi(phi0, g_tol, iterations)

    # Automatic warm restart (#484). Some phi0 land the REML LBFGS on a failed
    # line-search step right at the ML warm start -- the trace shows ONLY
    # "Iter 0" (zero ACCEPTED steps: x never moves off phi0), which is a
    # starting-value problem, not slow convergence -- more iterations or a
    # looser g_tol cannot help a run that never took a step. Detectable and
    # cheap to fix, so this is automatic rather than an opt-in kwarg --
    # silently returning a non-converged fit when a restart would succeed is
    # the worse default.
    #
    # Detection is by POSITION, not `Optim.iterations(res)`: LBFGS still
    # counts a rejected line-search pass as one iteration internally (so
    # `iterations` reads 1, not 0, on the exact failure this restart targets
    # -- confirmed by direct reproduction), but a rejected step leaves `x`
    # bit-identical to phi0, which is the true "took no step" signature.
    # Fires ONLY on that exact signature (`!converged && x == phi0`), so a
    # REML fit that already moves at all -- converged or not -- is untouched;
    # and the point reported below is judged at the CALLER's own unchanged
    # `g_tol`, never a loosened one.
    #
    # The restart re-derives the ML warm start ITSELF at a looser tolerance
    # rather than just loosening the REML g_tol check on the SAME phi0.
    # Loosening the check alone does not move x: phi0's gradient norm can
    # already sit under a 10x-looser g_tol, so the "coarse pass" would just
    # report instant convergence at the identical stalled point (confirmed by
    # direct reproduction) -- there would be nothing new to continue from. A
    # less-precise ML fit (larger `g_tol` on `fit_q4_sparse_tmb`, mirroring
    # the `phi0 === nothing` branch above but coarser) lands somewhere
    # genuinely different, off the exact point whose first line-search step
    # fails. Needs `beta0`/`Lambda0` (the caller's own starting guesses) to
    # redo that fit; if the caller bypassed them by supplying `phi0` directly,
    # there is nothing to re-derive from, so the restart is skipped.
    if !Optim.converged(res) && Optim.minimizer(res) == phi0 &&
       beta0 !== nothing && Lambda0 !== nothing
        g_tol_coarse = max(g_tol * 10, 1e-2)
        r_ml_coarse = fit_q4_sparse_tmb(prob, Q_cond;
                                         β0=beta0, Λ0=Matrix(Lambda0),
                                         g_tol=max(g_tol_coarse*5, 1e-2),
                                         iterations=min(iterations, 100),
                                         n_newton=n_newton, lc_zero=lc_zero_idx)
        phi0_coarse = pack_phi(prob, r_ml_coarse.β.rho, r_ml_coarse.Λ)
        phi0_coarse[phi_zero] .= 0.0

        res_coarse = _optimize_phi(phi0_coarse, g_tol_coarse, iterations)
        if Optim.minimizer(res_coarse) != phi0_coarse
            # Continue from the coarse optimum at the SAME g_tol the caller
            # asked for. Generous iteration budget -- convergence from here is
            # cheap (a handful of steps once started near the mode) and the
            # caller's `iterations` was sized for the cold run that just
            # failed, not this near-converged continuation.
            #
            # A single continuation can ALSO end in a line-search stall part
            # way (confirmed by direct reproduction: 5 steps of real progress,
            # then the same rejected-step signature, still short of g_tol) --
            # LBFGS carries no memory across separate `optimize` calls, so a
            # FRESH run from the last point it reached can take a step the
            # stale one's line-search state could not. Keep relaunching from
            # the current best point while it keeps moving and hasn't
            # converged; stop the moment it does either, or after a bounded
            # number of rounds so a genuinely stuck cell still terminates.
            cur = Optim.minimizer(res_coarse)
            res_try = _optimize_phi(cur, g_tol, max(iterations, 1000))
            round = 0
            while !Optim.converged(res_try) && Optim.minimizer(res_try) != cur && round < 10
                cur = Optim.minimizer(res_try)
                res_try = _optimize_phi(cur, g_tol, max(iterations, 1000))
                round += 1
            end
            res = res_try
        end
    end

    # Widened rescue (#497). The #484 block above fires only on the EXACT stall
    # signature (`x == phi0`, zero accepted steps), and needs its coarse ML
    # re-derivation because a fresh run from an identical point is deterministic
    # and would reproduce the identical rejected step. The far more common
    # failure is different: the run MOVES off phi0 (median 3 iterations), then
    # BackTracking cannot find an acceptable step and Optim stops with
    # `ls_failed`. Measured over a 300-seed ntip=64 sweep that was 51/51 of the
    # non-converged fits, and 100% of the failures at ntip=16/32 too -- one
    # mechanism at every N, not a size-dependent one.
    #
    # LBFGS carries no memory across separate `optimize` calls, so a FRESH run
    # from the point it reached can take a step the stale line-search state
    # could not. The two blocks are complementary, not alternatives.
    #
    # Also fires when Optim reports convergence via the f-criterion while the
    # gradient is still above the caller's `g_tol` (11/300 cells): those return
    # `converged = true` at a point that does not meet the tolerance asked for.
    #
    # Measured (converged AND g_residual < g_tol): 238/300 -> 297/300.
    # `reml_loglik` improves or ties in every cell, never worsens; fits that
    # already met tolerance come out bit-identical (max |Lambda change| = 0);
    # median wall time unchanged.
    _g_resid_now = try; Optim.g_residual(res); catch; NaN; end
    if !Optim.converged(res) || (isfinite(_g_resid_now) && _g_resid_now > g_tol)
        cur = Optim.minimizer(res)
        res_try = _optimize_phi(cur, g_tol, max(iterations, 1000))
        rounds = 0
        while !Optim.converged(res_try) && Optim.minimizer(res_try) != cur && rounds < 10
            cur = Optim.minimizer(res_try)
            res_try = _optimize_phi(cur, g_tol, max(iterations, 1000))
            rounds += 1
        end
        # Adopt the new point ONLY if it is genuinely better, so a rescue that
        # fails to help can never degrade what the caller receives.
        _g_resid_try = try; Optim.g_residual(res_try); catch; NaN; end
        if Optim.converged(res_try) || (isfinite(_g_resid_try) && _g_resid_try < _g_resid_now)
            res = res_try
        end
    end

    # Exact-gradient polish (#575). With a finite-difference gradient whose noise
    # sat at the same order as `g_tol`, tightening the tolerance was meaningless
    # -- there was nothing below the noise floor to find, and every polish tried
    # on the FD route either failed to move or moved to a point that no longer
    # met the gradient contract (scratchpad p12a). With an EXACT gradient at a
    # joint-Newton-certified mode, a 10x tighter run is both meaningful and
    # nearly free (one mode solve per evaluation instead of 2*nph + 1).
    #
    # Measured on biv-q4-phylo-reml: -219.614762 (g_residual 9.51e-4) ->
    # -219.613996 (g_residual 7.27e-5), i.e. the cold-start public route lands
    # within 1e-5 of drmTMB's own reported optimum.
    #
    # Adopted ONLY if the polished point is converged, no worse in objective AND
    # strictly better in gradient residual, so it can never trade the caller's
    # contract for a better number. The warm caches are snapshotted and restored
    # on rejection: `fg!` mutates them on every trial evaluation whether or not
    # the trial is adopted, and a rejected trial leaving a stale cache behind was
    # itself a confirmed regression on an earlier attempt at this route.
    if Optim.converged(res)
        _u_snap = u_cache[]; _b_snap = beta_cache[]
        _g_before = try; Optim.g_residual(res); catch; NaN; end
        res_pol = _optimize_phi(Optim.minimizer(res), max(g_tol / 10, 1e-10),
                                max(iterations, 1000))
        _g_after = try; Optim.g_residual(res_pol); catch; NaN; end
        if Optim.converged(res_pol) && isfinite(_g_after) &&
           Optim.minimum(res_pol) <= Optim.minimum(res) &&
           (!isfinite(_g_before) || _g_after < _g_before)
            res = res_pol
        else
            u_cache[] = _u_snap; beta_cache[] = _b_snap
        end
    end

    phi_hat    = Optim.minimizer(res)
    _, lc_hat  = unpack_phi(prob, phi_hat)
    Lam_hat    = lc_to_Λ(lc_hat)

    # Final evaluation through the SAME exact/joint-Newton path the optimiser
    # used, so the reported objective, the mode and the certified gradient all
    # refer to one point. `gz` = ‖∇_z J(ẑ)‖ replaces the old alternation flag as
    # the inner-convergence criterion (#526): it measures the thing that flag was
    # a proxy for, directly, and it is also exactly the condition under which the
    # exact gradient — and therefore `g_residual` — means anything.
    nll_hat, _, uhat, bhat, ch_H, gz_hat = reml_nll_and_exact_grad(
        prob, Q_cond, phi_hat; u0=u_cache[], beta0=beta_cache[], n_newton=n_newton)
    # The warm-cached u0/beta0 are the last line-search state, which need not sit at
    # the JOINT (u, β) mode for phi_hat. If that lands on the non-PD Schur barrier
    # (S not PD ⇒ nll_hat = Inf), re-evaluate COLD so the alternation and joint
    # Newton re-converge the mode at phi_hat (mirrors the Gate-B cold check).
    if !isfinite(nll_hat)
        nll_hat, _, uhat, bhat, ch_H, gz_hat = reml_nll_and_exact_grad(
            prob, Q_cond, phi_hat; n_newton=n_newton)
    end
    rhat     = isfinite(nll_hat) ? -nll_hat : -Inf
    inner_ok = isfinite(gz_hat) && gz_hat < 1e-6
    P_hat    = prior_precision(Q_cond, inv(Lam_hat))
    mlhat    = laplace_ll(prob, P_hat, bhat, uhat, ch_H)

    g_resid_val = try; Optim.g_residual(res); catch; NaN; end

    # #477: report the NORMALISED Patterson-Thompson restricted log-likelihood.
    # `rhat` is the unnormalised form; lme4, glmmTMB and TMB all add
    # `(n_beta/2)*log(2pi)`, and so do DRModels.jl's OWN univariate REML routes
    # (`gaussian_core.jl`, `gaussian_ranef.jl`, `location_only.jl`). Reporting
    # both conventions inside one package made `reml_loglik(fit)` mean different
    # things depending on which route produced the fit. `n_beta` is the combined
    # width of the four MARGINALISED axes -- exactly the Schur complement's
    # dimension (`nbeta` in `reml_ll_and_mode`); `rho` is never marginalised and
    # does not count. A constant cannot move the argmax, so the optimisation
    # above is untouched; only the reported value changes.
    return (phi        = phi_hat,
            beta       = bhat,
            Lambda     = Lam_hat,
            reml_loglik = _reml_normalise(rhat, length(bhat.mu1) + length(bhat.mu2) +
                                                length(bhat.s1) + length(bhat.s2)),
            ml_loglik   = mlhat,
            # #526: the public flag now also requires the FINAL evaluation's
            # inner (u, beta) alternation to have met its joint criterion, so
            # "converged" means the same thing users read on the ML path — at
            # the optimum, with the mode actually found — instead of only the
            # outer optimiser's verdict over phi.
            converged  = Optim.converged(res) && inner_ok,
            iterations = Optim.iterations(res),
            g_residual = g_resid_val,
            f_calls    = eval_cnt[],
            u_hat      = uhat)
end
