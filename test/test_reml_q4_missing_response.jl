# test_reml_q4_missing_response.jl — issue #578: `_reml_border_blocks`
# (src/reml_q4.jl) started passing the per-row observed-response masks
# (`prob.obs1[i]`, `prob.obs2[i]`) to `leaf_hess` as part of #575's exact-gradient
# extraction (commit f5f8a600). The inline build it replaced called `leaf_hess`
# with NO mask arguments, i.e. it silently defaulted to o1=o2=true for EVERY
# leaf regardless of the leaf's actual missingness — inconsistent with
# `build_Huu` (src/sparse_aug_plsm.jl), which has always threaded the masks.
# For complete data the two forms are identical (default trues == actual trues),
# so every no-regression suite stayed green; for missing responses they diverge,
# and nothing exercised that divergence. This file does.
#
# THE IDENTITY THIS TEST PINS (chosen over the FD/mask-triviality fallbacks in
# the task brief because it is the strongest one the code supports: an EXACT,
# non-tautological equality rather than an approximate or self-consistency
# check):
#
#   `_reml_border_blocks(prob, u, beta)`'s (H_u_beta, H_beta_beta) must equal an
#   INDEPENDENTLY reimplemented copy of the same lift, built in this test file
#   from `leaf_hess` called with the SAME per-row masks (`prob.obs1[i]`,
#   `prob.obs2[i]`).
#
# This is exact (not approximate) because both sides do the identical
# arithmetic on the identical inputs; it is non-tautological because the two
# implementations are independent code, and — the point of the "unmasked
# sanity" sub-test below — the identity FAILS if either implementation reverts
# to the pre-#575 unmasked call. Why it fails: for a leaf with exactly one
# response missing, `leaf_nll`'s masked branch (the `elseif o1` / `elseif o2`
# cases) is a function of only 2 of the 4 latent axes, so `leaf_hess` (ForwardDiff
# over the full 4-vector) is EXACTLY ZERO on the rows/columns of the missing
# response's axes (mu, log-sigma). The unmasked default instead evaluates the
# full bivariate branch on the SAME leaf (using make_problem's zero-filled
# placeholder for the missing y), which has nonzero curvature on those axes.
# The two lifts therefore disagree at every leaf with a real missing response —
# demonstrated directly below rather than merely asserted.
#
# Complementary checks (weaker but still real, mirroring test_575_exact_reml_
# gradient.jl and test_missing_response_bivariate.jl's front-end fit):
#   (2) FD-vs-exact-gradient of `reml_nll_and_exact_grad` at the FITTED REML
#       optimum on the SAME masked fixture. This does NOT by itself prove the
#       masks are threaded (both the value and the gradient route through
#       `_reml_border_blocks`, so a self-consistent mask bug would pass it too)
#       — it instead guards a DIFFERENT failure mode: that the exact-gradient
#       machinery (Xtilde dimensions, the Omega_i lift, the Schur solve) stays
#       well-posed, and the optimiser reaches a genuine joint stationary point,
#       once responses are missing.
#   (3) An end-to-end `drm(...; method = :REML)` fit on the missing-response
#       fixture: both ML and REML converge, report finite log-likelihoods that
#       genuinely differ (a mask bug that silently zeroed the REML correction
#       on the affected rows would show up as an accidental near-equality).
#
#   julia --project=. -e 'using DRModels, Test; include("test/test_reml_q4_missing_response.jl")'

module TestRemlQ4MissingResponse

using DRModels
using Test, Random, LinearAlgebra, Statistics

@testset "#578 q4 REML: _reml_border_blocks mask consistency with missing responses" begin

    # --- shared fixture: p ≈ 45 tips, m = 2 replicate rows per tip (REML on
    # the scale axes needs within-leaf replication to identify the per-leaf
    # scale random effects — a single row per leaf under-identifies sigma1/
    # sigma2's phylo variance and the REML fit does not converge). ~10% of y1
    # and ~10% of y2 rows are missing, on DIFFERENT rows, so obs1 != obs2 at
    # those rows — exactly the pattern the Schur-consistency change in
    # _reml_border_blocks has to get right (a leaf-row where only ONE of the
    # two responses is observed).
    Random.seed!(20260902)
    p = 45; m = 2
    phy = random_balanced_tree(p; branch_length = 0.4)
    Σphy = sigma_phy_dense(phy; σ²_phy = 1.0)
    Λ_among = [0.20 0.06 0.02 0.00;
               0.06 0.20 0.00 0.02;
               0.02 0.00 0.10 0.01;
               0.00 0.02 0.01 0.10]
    U = cholesky(Symmetric(Σphy)).L * randn(p, 4) * cholesky(Symmetric(Λ_among)).L'
    sp = repeat(1:p, inner = m); n = length(sp)
    x = randn(n)
    β_mu1 = [0.4, 0.3]; β_mu2 = [-0.2, 0.5]; β_s1 = -0.5; β_s2 = -0.4

    y1 = β_mu1[1] .+ β_mu1[2] .* x .+ U[sp, 1] .+ exp.(β_s1 .+ U[sp, 3]) .* randn(n)
    y2 = β_mu2[1] .+ β_mu2[2] .* x .+ U[sp, 2] .+ exp.(β_s2 .+ U[sp, 4]) .* randn(n)

    n_miss = max(1, round(Int, 0.10n))
    miss1 = randperm(MersenneTwister(20260903), n)[1:n_miss]
    pool2 = setdiff(1:n, miss1)
    miss2 = pool2[randperm(MersenneTwister(20260904), length(pool2))[1:n_miss]]
    @test isempty(intersect(miss1, miss2))    # constructed disjoint: masks differ

    y1m = copy(y1); y2m = copy(y2)
    y1m[miss1] .= NaN
    y2m[miss2] .= NaN

    X1 = hcat(ones(n), x); X2 = hcat(ones(n), x)
    Xs1 = ones(n, 1); Xs2 = ones(n, 1); Xr = ones(n, 1)

    prob, Q_cond = make_problem(phy, y1m, y2m, X1, X2, Xs1, Xs2, Xr; species = sp)
    @test count(.!prob.obs1) == n_miss
    @test count(.!prob.obs2) == n_miss
    @test prob.obs1 != prob.obs2

    β0 = (mu1 = X1[prob.obs1, :] \ y1m[prob.obs1], mu2 = X2[prob.obs2, :] \ y2m[prob.obs2],
          s1 = [-0.3], s2 = [-0.3], rho = [0.1])
    Λ0 = Matrix(Symmetric(0.25I(4) + 0.02 * (ones(4, 4) - I(4))))

    @testset "(1) independent recomputation of the masked leaf-Hessian lift — the exact identity" begin
        # A valid (u_hat, beta) point at a fixed non-diagonal Λ0 is enough — the
        # defect under test is plumbing (mask threading), not optimality, so a
        # single E-step from the observed-rows OLS start suffices.
        P0 = prior_precision(Q_cond, inv(Λ0))
        u_hat, _, _ = estep_mode(prob, P0, β0; n_newton = 60)
        u_hat = Vector{Float64}(u_hat)

        H_u_beta, H_beta_beta = DRModels._reml_border_blocks(prob, u_hat, β0)

        e1, e2, es1, es2, er = DRModels.leaf_etas(prob, β0)
        Xax = (X1, X2, Xs1, Xs2)
        wax = (2, 2, 1, 1)
        off = (0, 2, 4, 5)
        nu = 4 * prob.n_total
        nbeta = sum(wax)

        function _lift(masked::Bool)
            Hub = zeros(nu, nbeta)
            Hbb = zeros(nbeta, nbeta)
            for i in 1:n
                t = prob.leaf_node[i]; base = 4 * (t - 1)
                ublk = [u_hat[base+1], u_hat[base+2], u_hat[base+3], u_hat[base+4]]
                Hb = masked ?
                    DRModels.leaf_hess(ublk, prob.y1[i], prob.y2[i], e1[i], e2[i], es1[i], es2[i], er[i],
                                  prob.obs1[i], prob.obs2[i]) :
                    DRModels.leaf_hess(ublk, prob.y1[i], prob.y2[i], e1[i], e2[i], es1[i], es2[i], er[i])
                for d in 1:4, dp in 1:4
                    Hdd = Hb[d, dp]
                    Hdd == 0.0 && continue
                    Xdp = Xax[dp]; odp = off[dp]
                    for k in 1:wax[dp]
                        Hub[base+d, odp+k] += Hdd * Xdp[i, k]
                    end
                    Xd = Xax[d]; od = off[d]
                    for r in 1:wax[d], c in 1:wax[dp]
                        Hbb[od+r, odp+c] += Hdd * Xd[i, r] * Xdp[i, c]
                    end
                end
            end
            return Hub, Hbb
        end

        H_u_beta_ref, H_beta_beta_ref = _lift(true)
        @test H_u_beta ≈ H_u_beta_ref atol = 1e-12
        @test H_beta_beta ≈ H_beta_beta_ref atol = 1e-12

        # Non-vacuity: the pre-#575 UNMASKED call (leaf_hess with no o1/o2 args,
        # i.e. the bug #578 flags) must NOT reproduce the current src's output —
        # otherwise the equality above would hold trivially regardless of
        # whether masks are threaded at all.
        _, H_beta_beta_unmasked = _lift(false)
        @test !isapprox(H_beta_beta, H_beta_beta_unmasked; atol = 1e-8)
    end

    @testset "(2) FD-vs-exact REML gradient at the fitted optimum, with missing responses" begin
        rr = DRModels.fit_q4_reml(prob, Q_cond; beta0 = β0, Lambda0 = Λ0,
                             g_tol = 1e-3, iterations = 300, n_newton = 40)
        @test rr.converged
        phi_opt = Vector{Float64}(rr.phi)

        val, g_exact, _, _, _, zres = DRModels.reml_nll_and_exact_grad(
            prob, Q_cond, phi_opt; beta0 = β0, n_newton = 60)
        @test isfinite(val)
        @test all(isfinite, g_exact)
        # The exact gradient is only valid AT a joint stationary point: the
        # routine must certify one (mirrors test_575_exact_reml_gradient.jl).
        @test zres < 1e-4

        nph = length(phi_opt)
        g_fd = zeros(nph)
        h = 1e-4
        for k in 1:nph
            pp = copy(phi_opt); pp[k] += h
            pm = copy(phi_opt); pm[k] -= h
            fp = DRModels.reml_nll_exact(prob, Q_cond, pp; beta0 = β0)
            fm = DRModels.reml_nll_exact(prob, Q_cond, pm; beta0 = β0)
            g_fd[k] = (fp - fm) / (2h)
        end
        err = maximum(abs.(g_exact .- g_fd))
        scale = max(1.0, maximum(abs, g_fd))
        @test err / scale <= 1e-2
    end

    @testset "(3) end-to-end drm() ML and REML fits converge with genuinely missing cells" begin
        species = phy.leaf_names[sp]
        data = (; y1 = y1m, y2 = y2m, x, species)
        # phylo() on ALL FOUR sub-formulas is required to route into the q4
        # sparse engine (_fit_bivariate_q4_phylo) — the route _reml_border_blocks
        # belongs to; sigma1/sigma2 without a phylo() marker fall to the
        # q=2-structured route instead, which refuses missing responses outright
        # (a separate, unrelated gap) and would not exercise this code path.
        form = bf(mu1    = @formula(y1 ~ x + phylo(1 | species)),
                  mu2    = @formula(y2 ~ x + phylo(1 | species)),
                  sigma1 = @formula(sigma1 ~ 1 + phylo(1 | species)),
                  sigma2 = @formula(sigma2 ~ 1 + phylo(1 | species)),
                  rho12  = @formula(rho12 ~ 1))

        fit_ml   = drm(form, Gaussian(); data = data, tree = phy,
                       method = :ML, q4_vcov = false)
        fit_reml = drm(form, Gaussian(); data = data, tree = phy,
                       method = :REML, q4_vcov = false)

        @test fit_ml.converged
        @test fit_reml.converged
        @test isfinite(loglik(fit_ml))
        @test isfinite(loglik(fit_reml))
        @test nobs(fit_ml) == n              # row count, not observed-cell count
        @test nobs(fit_reml) == n
        # A mask bug that silently zeroed the REML Schur correction on the
        # affected rows would show up as an accidental near-equality here.
        @test !isapprox(loglik(fit_ml), loglik(fit_reml); atol = 1e-6)
    end
end

end # module TestRemlQ4MissingResponse
