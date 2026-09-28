# test_reml_q4_chol.jl — src/reml_q4.jl builds the q=4 REML prior precision
# P = kron(Q_cond, inv(Λ)) (and the matching log-Cholesky-diagonal logdet) at
# SIX sites (lines 309, 437, 505-506, 516, 599, 999 on origin/main / the tip of
# claude/q4-prior-whitening before this PR): `reml_ll_and_mode`, `_reml_exact_state`,
# and the exact-gradient helper `reml_nll_and_exact_grad`. Near a singular Λ
# (correlation -> ±1, log-Cholesky diagonal around -18 to -30) naive `inv(Λ)`
# loses every digit — the same defect measured for the coevolution/q2 prior in
# test_q4_prior_whitening.jl and, for this q4 engine's ML marginal, pinned as a
# `@test_broken` in that file's own "q4 engine prior near singular Λ" testset.
#
# Two checks:
#   1. Extreme regime: the REML objective `reml_nll_exact` at log-Cholesky
#      diagonal -18/-20/-30 against a 256-bit BigFloat reference, independent
#      Newton solve over the FULL augmented (u, beta) state (nothing in this
#      reference touches sparse_pd_chol, estep_mode or the alternation code —
#      only P and the leaf likelihood). MUST currently fail: naive `inv(Λ)`
#      poisons the Newton-certified joint mode badly enough that `ch_S` comes
#      back non-PD and `reml_nll_exact` returns the `Inf` barrier outright.
#   2. Normal-regime identity: 10 random, well-conditioned phi, comparing the
#      LIVE `DRModels.reml_nll_exact` against a frozen, verbatim copy of
#      TODAY's (pre-whitening) `reml_ll_and_mode` / `_reml_exact_state` /
#      `reml_nll_exact` trio (`_old_*` below — copied, not reimplemented, so it
#      keeps exercising the SAME alternation/joint-Newton code paths and stays
#      a fair comparison regardless of how the fix reroutes the six sites).
#      Every helper the `_old_*` copy calls into (`_reml_joint_newton`,
#      `cond_newton_beta`, `build_Huu`, `_reml_border_blocks`, `estep_mode`,
#      `laplace_ll`, ...) is untouched by the whitening fix, so this isolates
#      the six naive-`inv(Λ)` sites as the only difference between the two
#      paths.
#
# Fixture: p=6 phylo leaves x 4 reps (n=24), X1/X2 = intercept+x, Xs1/Xs2 =
# intercept-only, Xr empty (kr=0, rho fixed at 0) -- nbeta=6 well below n=24,
# which matters: an EARLIER, smaller fixture (nbeta >= n) hit a genuinely
# unidentified Schur complement unrelated to Λ conditioning at all (verified
# by hand — see the after-task report). At this fixture the pre-patch code
# already matches the 256-bit reference to ~1e-14 in the well-conditioned
# regime, so the extreme-regime failure below is attributable to the six
# `inv(Λ)` sites and nothing else.
#
#   julia --project=. -e 'using DRModels, Test; include("test/test_reml_q4_chol.jl")'

module TestRemlQ4Chol

using DRModels
using Test, LinearAlgebra, Random, SparseArrays, ForwardDiff

# =============================================================================
# 1. Frozen, verbatim (renamed) copy of the PRE-WHITENING reml_ll_and_mode /
#    _reml_exact_state / reml_nll_exact trio from src/reml_q4.jl, so the
#    "normal-regime identity" testset keeps a naive-inv(Λ) oracle to compare
#    the live (possibly-patched) DRModels.reml_nll_exact against, independent
#    of whatever the fix does internally. Every other helper called here
#    (_reml_joint_newton, cond_newton_beta, build_Huu, _reml_border_blocks,
#    _reml_axis_layout, estep_mode, laplace_ll, unpack_phi) is NOT one of the
#    six sites the fix touches, so it is reused live via DRModels.<name>.
# =============================================================================

function _old_reml_ll_and_mode(prob::DRModels.AugProblem, Q_cond::SparseMatrixCSC,
                                phi::Vector{Float64}; u0 = nothing, beta0 = nothing,
                                n_newton::Int = 40)
    rho_coef, lc = DRModels.unpack_phi(prob, phi)
    Lam = DRModels.lc_to_Λ(lc)
    P   = DRModels.prior_precision(Q_cond, inv(Lam))          # site (verbatim, naive)

    if beta0 === nothing
        bm1 = prob.X1 \ prob.y1; bm2 = prob.X2 \ prob.y2
        bs1 = zeros(size(prob.Xs1, 2)); bs2 = zeros(size(prob.Xs2, 2))
    else
        bm1 = beta0.mu1; bm2 = beta0.mu2; bs1 = beta0.s1; bs2 = beta0.s2
    end
    beta_full = (mu1 = bm1, mu2 = bm2, s1 = bs1, s2 = bs2, rho = rho_coef)

    u_hat = u0 === nothing ? zeros(4 * prob.n_total) : Vector{Float64}(u0)
    ch_H  = nothing
    last_delta = Inf
    for alt_it in 1:15
        u_hat, ch_H, _ = DRModels.estep_mode(prob, P, beta_full; u0 = u_hat, n_newton = n_newton)
        u_hat = Vector{Float64}(u_hat)
        b_new = DRModels.cond_newton_beta(prob, u_hat, beta_full; n_newton = 20)
        delta_b = norm(b_new.mu1 .- beta_full.mu1) + norm(b_new.mu2 .- beta_full.mu2) +
                  norm(b_new.s1  .- beta_full.s1)  + norm(b_new.s2  .- beta_full.s2)
        beta_full = (mu1 = b_new.mu1, mu2 = b_new.mu2,
                     s1  = b_new.s1,  s2  = b_new.s2, rho = rho_coef)
        last_delta = delta_b
        delta_b < 1e-6 && break
        if alt_it >= 2
            bs = norm(beta_full.mu1) + norm(beta_full.mu2) +
                 norm(beta_full.s1) + norm(beta_full.s2)
            delta_b < 1e-4 * (1 + bs) && break
        end
    end
    beta_scale = norm(beta_full.mu1) + norm(beta_full.mu2) +
                 norm(beta_full.s1) + norm(beta_full.s2)
    inner_converged = last_delta < 1e-4 * (1 + beta_scale)
    u_hat, ch_H, _ = DRModels.estep_mode(prob, P, beta_full; u0 = u_hat, n_newton = n_newton)
    u_hat = Vector{Float64}(u_hat)

    ml_ll = DRModels.laplace_ll(prob, P, beta_full, u_hat, ch_H)

    nu = 4 * prob.n_total
    _, _, _, nbeta = DRModels._reml_axis_layout(prob)
    H_u_beta, H_beta_beta = DRModels._reml_border_blocks(prob, u_hat, beta_full)

    C = Matrix{Float64}(undef, nu, nbeta)
    for j in 1:nbeta
        C[:, j] = ch_H \ H_u_beta[:, j]
    end
    S     = H_beta_beta - H_u_beta' * C
    S_sym = Symmetric((S + S') / 2)
    ch_S  = cholesky(S_sym; check = false)
    if !issuccess(ch_S)
        return -Inf, u_hat, ch_H, beta_full, P, inner_converged
    end
    ld_S = logdet(ch_S)

    return ml_ll - 0.5 * ld_S, u_hat, ch_H, beta_full, P, inner_converged
end

function _old_reml_exact_state(prob::DRModels.AugProblem, Q_cond::SparseMatrixCSC,
                                phi::AbstractVector{<:Real}; u0 = nothing, beta0 = nothing,
                                n_newton::Int = 40, joint_iter::Int = 25, joint_tol::Float64 = 1e-10)
    phiv = Vector{Float64}(phi)
    rho_coef, lc = DRModels.unpack_phi(prob, phiv)
    Lam = DRModels.lc_to_Λ(lc)
    P   = DRModels.prior_precision(Q_cond, inv(Lam))          # site (verbatim, naive)
    _, u_a, _, b_a, _, _ = _old_reml_ll_and_mode(prob, Q_cond, phiv;
                                                  u0 = u0, beta0 = beta0, n_newton = n_newton)
    u, b, ch_H, C, ch_S, gz = DRModels._reml_joint_newton(prob, P, Vector{Float64}(u_a), b_a;
                                                          max_iter = joint_iter, tol = joint_tol)
    return (rho = rho_coef, lc = lc, Lam = Lam, P = P, u = u, beta = b,
            ch_H = ch_H, C = C, ch_S = ch_S, gz = gz)
end

function _old_reml_nll_exact(prob::DRModels.AugProblem, Q_cond::SparseMatrixCSC,
                              phi::AbstractVector{<:Real}; u0 = nothing, beta0 = nothing,
                              n_newton::Int = 40, joint_iter::Int = 25, joint_tol::Float64 = 1e-10)
    st = _old_reml_exact_state(prob, Q_cond, phi; u0 = u0, beta0 = beta0,
                                n_newton = n_newton, joint_iter = joint_iter, joint_tol = joint_tol)
    issuccess(st.ch_S) || return Inf
    ll = DRModels.laplace_ll(prob, st.P, st.beta, st.u, st.ch_H) - 0.5 * logdet(st.ch_S)
    return isfinite(ll) ? -ll : Inf
end

# =============================================================================
# 2. 256-bit BigFloat reference. Fully independent of reml_q4.jl's own
#    alternation/joint-Newton code: solves the SAME augmented-state problem
#    (0.5 u'Pu + sum_i leaf_nll, beta unconstrained/flat) with its own damped
#    Newton over z = (u, beta), then assembles the REML correction exactly as
#    the header derivation in reml_q4.jl describes it (H_uu = the (u,u) block
#    of the full z-Hessian at the mode; B, D = the cross/beta-beta blocks;
#    S = D - B'*H_uu^{-1}*B; ll = -J(z_hat) - 0.5 logdet(H_uu) + 0.5 logdet(P)
#    - 0.5 logdet(S)). Returns ll (not the negative).
# =============================================================================

function _reml_bigref_ll(prob::DRModels.AugProblem, Q_cond::SparseMatrixCSC,
                          phi::AbstractVector{<:Real}; prec::Int = 256, maxit::Int = 80)
    setprecision(BigFloat, prec) do
        T = BigFloat
        kr = size(prob.Xr, 2)
        rho = T.(phi[1:kr]); lc = T.(phi[(kr + 1):(kr + 10)])
        L = zeros(T, 4, 4); k = 0
        for j in 1:4, i in j:4
            k += 1
            L[i, j] = i == j ? exp(lc[k]) : lc[k]
        end
        Lam  = L * L'
        Qb   = T.(Matrix(Q_cond))
        P    = kron(Qb, inv(Lam))
        nu   = size(P, 1)
        Xax  = (T.(prob.X1), T.(prob.X2), T.(prob.Xs1), T.(prob.Xs2))
        wax  = size.(Xax, 2)
        off  = (0, wax[1], wax[1] + wax[2], wax[1] + wax[2] + wax[3])
        nbeta = sum(wax)

        function Jz(z)
            u = z[1:nu]; bv = z[(nu + 1):(nu + nbeta)]
            β = (mu1 = bv[(off[1] + 1):(off[1] + wax[1])], mu2 = bv[(off[2] + 1):(off[2] + wax[2])],
                 s1  = bv[(off[3] + 1):(off[3] + wax[3])], s2  = bv[(off[4] + 1):(off[4] + wax[4])], rho = rho)
            η1, η2, ηs1, ηs2, ηr = DRModels.leaf_etas(prob, β)
            val = dot(u, P * u) / 2
            for i in eachindex(prob.leaf_node)
                t = prob.leaf_node[i]; base = 4 * (t - 1)
                val += DRModels.leaf_nll((u[base + 1], u[base + 2], u[base + 3], u[base + 4]),
                                         T(prob.y1[i]), T(prob.y2[i]),
                                         η1[i], η2[i], ηs1[i], ηs2[i], ηr[i],
                                         prob.obs1[i], prob.obs2[i])
            end
            return val
        end

        nz = nu + nbeta
        z = zeros(T, nz)
        bm1 = T.(prob.X1 \ prob.y1); bm2 = T.(prob.X2 \ prob.y2)
        z[(nu + off[1] + 1):(nu + off[1] + wax[1])] .= bm1
        z[(nu + off[2] + 1):(nu + off[2] + wax[2])] .= bm2
        for _ in 1:maxit
            g  = ForwardDiff.gradient(Jz, z)
            sqrt(sum(abs2, g)) < T(10)^-50 && break
            H  = ForwardDiff.hessian(Jz, z)
            ridge = T(0); ch = nothing; ok = false
            for _ in 1:30
                ch = cholesky(Symmetric(H + ridge * I); check = false)
                issuccess(ch) && (ok = true; break)
                ridge = ridge == 0 ? T(10)^-20 : ridge * 10
            end
            ok || error("bigref: could not PD-ify the z-Hessian")
            dz = ch \ g
            s = one(T); j0 = Jz(z); znew = z - s * dz
            while (!isfinite(Jz(znew)) || Jz(znew) > j0) && s > T(10)^-40
                s /= 2; znew = z - s * dz
            end
            z = znew
        end

        Hfull = ForwardDiff.hessian(Jz, z)
        Huu = Hfull[1:nu, 1:nu]
        B   = Hfull[1:nu, (nu + 1):nz]
        D   = Hfull[(nu + 1):nz, (nu + 1):nz]
        C   = Huu \ B
        S   = D - B' * C
        jn       = Jz(z)
        logdetH  = logdet(cholesky(Symmetric(Huu)))
        logdetP  = logdet(cholesky(Symmetric(P + T(1e-10) * I)))
        ll_ml    = -jn - logdetH / 2 + logdetP / 2
        logdetS  = logdet(cholesky(Symmetric(S)))
        return Float64(ll_ml - logdetS / 2)
    end
end

# =============================================================================
# 3. Fixture: p=6 phylo leaves x 4 reps (n=24), nbeta=6 (X1/X2 intercept+x,
#    Xs1/Xs2 intercept-only, Xr empty so kr=0 and rho is fixed at 0). Confirmed
#    by hand to be well-identified (10-trial max relative error ~1e-14 between
#    the pre-patch code and the 256-bit reference in the well-conditioned
#    regime) -- an earlier, smaller draft (nbeta >= n) hit a genuinely
#    unidentified Schur complement that had nothing to do with Λ conditioning.
# =============================================================================

function _fixture()
    rng = MersenneTwister(20260927)
    p = 6; m = 4
    phy = random_balanced_tree(p; branch_length = 0.4)
    species = repeat(1:p, inner = m)
    n = length(species)
    x = randn(rng, n)
    X1 = hcat(ones(n), x); X2 = hcat(ones(n), x)
    Xs1 = ones(n, 1); Xs2 = ones(n, 1); Xr = zeros(n, 0)
    y1 = 0.5 .+ 0.3 .* x .+ 0.3 .* randn(rng, n)
    y2 = -0.2 .+ 0.2 .* x .+ 0.3 .* randn(rng, n)
    return make_problem(phy, y1, y2, X1, X2, Xs1, Xs2, Xr; species = species)
end

@testset "reml_q4 Λ-inversion: extreme regime vs 256-bit reference" begin
    prob, Q_cond = _fixture()
    lc0 = zeros(10); lc0[2] = 0.35   # L21: strong mu1-mu2 coupling
    for l22 in (-18.0, -20.0, -30.0)
        lc = copy(lc0); lc[5] = l22  # (2,2) log-Cholesky diagonal -> near-singular Λ
        ref = _reml_bigref_ll(prob, Q_cond, lc)
        val = -DRModels.reml_nll_exact(prob, Q_cond, lc)
        @test isfinite(ref)
        @test isfinite(val)
        @test abs(val - ref) / abs(ref) ≤ 1e-8
    end
end

@testset "reml_q4 Λ-inversion: normal-regime identity, old vs new" begin
    prob, Q_cond = _fixture()
    rng = MersenneTwister(2026)
    for _ in 1:10
        lc = zeros(10)
        lc[1] = 0.4 * randn(rng); lc[5] = 0.4 * randn(rng)
        lc[8] = 0.4 * randn(rng); lc[10] = 0.4 * randn(rng)
        lc[2] = 0.2 * randn(rng); lc[3] = 0.2 * randn(rng); lc[4] = 0.2 * randn(rng)
        lc[6] = 0.2 * randn(rng); lc[7] = 0.2 * randn(rng); lc[9] = 0.2 * randn(rng)
        old = _old_reml_nll_exact(prob, Q_cond, lc)
        new = DRModels.reml_nll_exact(prob, Q_cond, lc)
        @test isfinite(old) && isfinite(new)
        @test abs(old - new) ≤ 1e-10 * max(1, abs(old))
    end
end

end # module
