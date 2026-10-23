# Focused unit tests for the module-wired optimizer / mode-accuracy fixes
# (#314, #317, #325.4). These exercise the DRModels-module code paths directly.
#
#   #314  locscale_grad.jl    — `_ls_marginal_grad` must return an all-NaN vector
#         (not all-zeros) when the inner Laplace mode fails, so a gradient-based
#         optimiser rejects the infeasible step instead of reading a zero gradient
#         as stationarity. Pairs with the `_ls_fit_nll` value sentinel.
#   #317  sparse_aug_plsm.jl  — `estep_mode`'s fast path must not freeze a loosely
#         converged mode into the Laplace marginal: the accepted mode's joint
#         gradient norm must be tight (≤ the tightened stall gate ~1e-6), matching
#         a robustly re-converged mode. Follow-up: the tightened gate routes more
#         warm E-steps through the robust path, so `sparse_pd_chol`/`laplace_ll`
#         must not THROW on a non-finite Hessian/prior at an extreme trial θ
#         (`cholesky(...; check=false)` still throws on Inf/NaN) — they must
#         signal failure so the optimiser rejects the step (CI regression fix).
#   #325.4 locscale_profile.jl — `_ls_profile_ci`/`_ls_profile_nll` now carry the
#         Zη/Zψ loadings (defaulting to canonical) so the profiler reconstructs the
#         EXACT model that was fit. Passing canonical loadings explicitly must give
#         bit-identical CIs to the default.

using DRModels
using Test, Random, LinearAlgebra, SparseArrays
import Distributions

@testset "optimizer robustness / mode accuracy (#314, #317, #325.4)" begin

    # ---------------------------------------------------------------------------
    # #314  NaN (not zero) outer gradient on inner-mode failure
    # ---------------------------------------------------------------------------
    @testset "#314 _ls_marginal_grad returns NaN on inner-mode failure" begin
        Random.seed!(7)
        G = 5; m = 6; n = G * m
        gidx = repeat(1:G, inner = m)
        x = randn(n); z = randn(n)
        Xμ = hcat(ones(n), x); Xψ = hcat(ones(n), z)
        Λt = DRModels._ls_lc_to_Λ([log(0.4), 0.1, log(0.5)]); Lt = cholesky(Symmetric(Λt)).L
        A = [Lt * randn(2) for _ in 1:G]
        y = [(r = exp(0.2 + A[gidx[i]][2]); μ = exp(0.3 + 0.4x[i] + A[gidx[i]][1]);
              Float64(rand(Distributions.NegativeBinomial(r, r / (r + μ))))) for i in 1:n]
        Q = sparse(1.0 * I, G, G)

        # A feasible θ: gradient is finite and populated (no NaN).
        θgood = [0.3, 0.4, 0.1, -0.05, log(0.4), 0.1, log(0.5)]
        ggood = DRModels._ls_marginal_grad(Val(:nb2), y, Xμ, Xψ, gidx, G, Q, θgood)
        @test all(isfinite, ggood)
        @test any(ggood .!= 0)

        # An infeasible θ (near-singular Λ, off-scale predictor) makes the inner
        # mode fail: `_ls_fit_nll` returns the sentinel and the gradient must be
        # all-NaN — crucially NOT all-zeros (the pre-#314 behaviour that read as
        # convergence to a gradient-based optimiser).
        θbad = [0.25, 0.35, 0.1, -0.05, log(1e-8), 5.0, log(1e-8)]
        vbad = DRModels._ls_fit_nll(Val(:nb2), y, Xμ, Xψ, gidx, G, Q, θbad)
        gbad = DRModels._ls_marginal_grad(Val(:nb2), y, Xμ, Xψ, gidx, G, Q, θbad)
        @test vbad >= 1e17                    # objective is the infeasible sentinel
        @test all(isnan, gbad)                # gradient signals infeasibility
        @test !any(iszero, gbad)              # NOT the old zero-vector stall signal
    end

    # ---------------------------------------------------------------------------
    # #325.4  profiler carries loadings; default == explicit canonical
    # ---------------------------------------------------------------------------
    @testset "#325.4 profile CI threads Zη/Zψ (canonical default unchanged)" begin
        Random.seed!(7)
        G = 5; m = 6; n = G * m
        gidx = repeat(1:G, inner = m)
        x = randn(n); z = randn(n)
        Xμ = hcat(ones(n), x); Xψ = hcat(ones(n), z)
        Λt = DRModels._ls_lc_to_Λ([log(0.4), 0.1, log(0.5)]); Lt = cholesky(Symmetric(Λt)).L
        A = [Lt * randn(2) for _ in 1:G]
        y = [(r = exp(0.2 + A[gidx[i]][2]); μ = exp(0.3 + 0.4x[i] + A[gidx[i]][1]);
              Float64(rand(Distributions.NegativeBinomial(r, r / (r + μ))))) for i in 1:n]
        Q = sparse(1.0 * I, G, G)
        θ̂ = [0.3, 0.4, 0.1, -0.05, log(0.4), 0.1, log(0.5)]

        # This hand-written θ̂ is a loading-contract reference, not a fitted
        # optimum. Strict nuisance checks may therefore report an explicit
        # failed endpoint; the contract here is that omitted and explicitly
        # canonical loadings produce the same complete result and diagnostics.
        ci_def = DRModels._ls_profile_ci_result(
            Val(:nb2), y, Xμ, Xψ, gidx, G, Q, θ̂; idx = 2, level = 0.95,
        )
        Zη = DRModels._ls_canonical_Zeta(n); Zψ = DRModels._ls_canonical_Zpsi(n)
        ci_exp = DRModels._ls_profile_ci_result(
            Val(:nb2), y, Xμ, Xψ, gidx, G, Q, θ̂;
            idx = 2, level = 0.95, Zη = Zη, Zψ = Zψ,
        )
        @test isequal(ci_def, ci_exp)          # explicit canonical == default
        @test all(s -> s.accepted || s.endpoint_failed || s.unbounded,
                  (ci_def.lower_status, ci_def.upper_status))
        @test !isnan(ci_def.lower) && !isnan(ci_def.upper)
    end

    # ---------------------------------------------------------------------------
    # #317  fast E-step freezes only a tightly-converged mode
    # ---------------------------------------------------------------------------
    @testset "#317 estep_mode returns a tight mode for the frozen Laplace" begin
        Random.seed!(11)
        p = 8
        phy = random_balanced_tree(p; branch_length = 0.2)
        Σ_phy = sigma_phy_dense(phy; σ²_phy = 1.0)
        n = p
        x1 = randn(n)
        X1 = hcat(ones(n), x1); X2 = hcat(ones(n), x1)
        Xs1 = reshape(ones(n), n, 1); Xs2 = reshape(ones(n), n, 1); Xr = reshape(ones(n), n, 1)
        β = (mu1 = [0.5, 0.3], mu2 = [-0.2, 0.4], s1 = [-0.4], s2 = [-0.5], rho = [0.3])
        Λ = Matrix(Symmetric([0.30 0.10 0.05 0.0;
                              0.10 0.30 0.0 0.04;
                              0.05 0.0 0.12 0.02;
                              0.0 0.04 0.02 0.12]))
        Lc = cholesky(Λ).L; Sc = cholesky(Symmetric(Σ_phy)).U
        U = Lc * randn(4, p) * Sc
        y1 = zeros(n); y2 = zeros(n)
        for i in 1:n
            m1 = (X1[i, :]'β.mu1) + U[1, i]; m2 = (X2[i, :]'β.mu2) + U[2, i]
            s1 = exp((Xs1[i, :]'β.s1) + U[3, i]); s2 = exp((Xs2[i, :]'β.s2) + U[4, i])
            ρ = DRModels.RHO_GUARD * tanh(Xr[i, :]'β.rho)
            e = cholesky([s1^2 ρ*s1*s2; ρ*s1*s2 s2^2]).L * randn(2)
            y1[i] = m1 + e[1]; y2[i] = m2 + e[2]
        end
        prob, Q = make_problem(phy, y1, y2, X1, X2, Xs1, Xs2, Xr)
        P = prior_precision(Q, inv(Λ))

        # Cold solve to the mode, then a WARM re-solve (exercises the fast path).
        u_cold, _, _ = estep_mode(prob, P, β)
        u_warm, ch, H = estep_mode(prob, P, β; u0 = u_cold)

        # The frozen-mode Laplace assumes ∇_u J = 0 exactly. With stall_tol tightened
        # to 1e-6 (#317), an accepted warm mode must be genuinely at the mode — its
        # joint gradient norm must be well below the old loose 1e-3 acceptance.
        ng_warm = norm(joint_grad(prob, P, u_warm, β))
        @test ng_warm < 1e-5
        # And it must agree with the cold-solved mode (no loose off-mode drift).
        @test maximum(abs, u_warm .- u_cold) < 1e-4
    end

    # ---------------------------------------------------------------------------
    # #317 follow-up: the sparse-Laplace choleskys must NOT throw on non-finite
    # input. `cholesky(...; check=false)` suppresses the not-PD exception but STILL
    # throws `ArgumentError("matrix contains Infs or NaNs")` on Inf/NaN entries.
    # Tightening the fast-path acceptance gate routed more warm E-steps through the
    # robust path, which at an extreme trial θ can hand a non-finite Hessian/prior
    # to `sparse_pd_chol`/`laplace_ll`; that ArgumentError escaped the q4 fit's
    # objective guard and crashed `test_gaussian_bivariate_phylo` on CI.
    @testset "#317 non-finite Hessian/prior does not crash the sparse-Laplace path" begin
        n = 8
        # A Hessian with NaN/Inf entries must yield a FINITE factor + a failure flag,
        # not an uncaught ArgumentError.
        Hbad = sparse(1.0I, n, n); Hbad[1, 1] = NaN; Hbad[2, 2] = Inf
        chb, flag = DRModels.sparse_pd_chol(Hbad)
        @test flag == Inf                          # failure signalled
        @test isfinite(logdet(chb))                # factor is finite/usable
        # A clean PD Hessian still factorizes normally (guard is inert in range).
        Hgood = sparse(2.0I, n, n)
        chg, fg = DRModels.sparse_pd_chol(Hgood)
        @test fg == 0.0
        @test logdet(chg) ≈ n * log(2.0)

        # laplace_ll must return -Inf (⇒ nll = +Inf, step rejected) when the prior
        # precision is non-finite, instead of throwing on its internal cholesky.
        Random.seed!(3); p = 6
        phy = random_balanced_tree(p; branch_length = 0.2)
        x1 = randn(p); X1 = hcat(ones(p), x1); X2 = hcat(ones(p), x1)
        Xs1 = reshape(ones(p), p, 1); Xs2 = reshape(ones(p), p, 1); Xr = reshape(ones(p), p, 1)
        y1 = randn(p); y2 = randn(p)
        prob, Q = make_problem(phy, y1, y2, X1, X2, Xs1, Xs2, Xr)
        Λ = Matrix(Symmetric(0.3I(4)))
        Pgood = prior_precision(Q, inv(Λ))
        β = (mu1 = [0.0, 0.0], mu2 = [0.0, 0.0], s1 = [-0.4], s2 = [-0.5], rho = [0.3])
        u, chH, _ = estep_mode(prob, Pgood, β)
        Pbad = copy(Pgood); Pbad.nzval[1] = NaN     # inject a non-finite prior entry
        @test DRModels.laplace_ll(prob, Pbad, β, u, chH) == -Inf   # signalled, no throw
        @test isfinite(DRModels.laplace_ll(prob, Pgood, β, u, chH)) # clean case unaffected
    end

    # ---------------------------------------------------------------------------
    # #317 follow-up (the ACTUAL CI crash): at an extreme trial θ the q4 driver's
    # `Λ = lc_to_Λ(lc)` overflows and `inv(Λ)` (marginal_and_exact_grad) throws
    # `ArgumentError("matrix contains Infs or NaNs")`. The #317 fast-path retuning
    # perturbed the line-search trajectory into that region; the fit objective's
    # try/catch previously omitted ArgumentError, so it escaped and crashed
    # `test_gaussian_bivariate_phylo` on CI. The objective must now REJECT such a
    # step (return Inf) rather than throw.
    @testset "#317 extreme θ (non-finite Λ) is rejected, not crashed" begin
        Random.seed!(1); p = 8
        phy = random_balanced_tree(p; branch_length = 0.2)
        n = p; x = randn(n); X1 = hcat(ones(n), x); X2 = hcat(ones(n), x)
        Xs1 = reshape(ones(n), n, 1); Xs2 = reshape(ones(n), n, 1); Xr = reshape(ones(n), n, 1)
        y1 = randn(n); y2 = randn(n)
        prob, Q = make_problem(phy, y1, y2, X1, X2, Xs1, Xs2, Xr)
        β = (mu1 = [0.0, 0.0], mu2 = [0.0, 0.0], s1 = [0.0], s2 = [0.0], rho = [0.0])
        θ = DRModels.pack_theta(β, DRModels.lc_to_Λ(fill(0.1, 10)))
        k1, k2, ks1, ks2, kr = DRModels.beta_widths(prob)
        nβ = k1 + k2 + ks1 + ks2 + kr
        θbad = copy(θ); θbad[nβ+1:nβ+10] .= 400.0        # exp(400) overflows Λ

        # The trigger: lc_to_Λ overflows and marginal_and_exact_grad throws.
        @test !all(isfinite, DRModels.lc_to_Λ(fill(400.0, 10)))
        # Pre-whitening this threw ArgumentError from `inv(Λ)`. The whitened engine
        # never inverts Λ; a non-finite Laplace value is now reported as +Inf (never
        # -Inf, which would read as the best possible value to a value-only caller)
        # with a non-finite gradient, which fit_q4_sparse_tmb's fg! rejects.
        nll_bad, g_bad, _, _ = DRModels.marginal_and_exact_grad(prob, Q, θbad; n_newton = 30)
        @test nll_bad == Inf
        @test !all(isfinite, g_bad)
        @test DRModels.marginal_nll(prob, Q, θbad; n_newton = 30)[1] == Inf
        # ...and never less than the value at a sane θ (the "gain" hazard)
        @test DRModels.marginal_nll(prob, Q, θbad; n_newton = 30)[1] >
              DRModels.marginal_nll(prob, Q, θ; n_newton = 30)[1]

        # The barrier: a normal fit from a FINITE start must COMPLETE (no uncaught
        # ArgumentError), even though the line search may probe extreme θ where the
        # marginal is non-finite — those steps are now rejected (return Inf) rather
        # than crashing the fit. (The bivariate-phylo front end exercises the exact
        # CI-failing config; here we just confirm the driver runs to completion.)
        r = DRModels.fit_q4_sparse_tmb(prob, Q;
                                  β0 = β, Λ0 = Matrix(Symmetric(0.3I(4))),
                                  iterations = 60, n_newton = 30)
        @test isfinite(r.loglik)                          # completed without throwing
    end
end
