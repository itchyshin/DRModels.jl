# test_phylo_locscale.jl — non-Gaussian phylogenetic LOCATION–SCALE (#202).
#
# This gates the q=2 (mean + log-dispersion) non-Gaussian phylogenetic
# location–scale engine that lives in `src/locscale_*.jl`: a Gamma response with
# mean μ = exp(η_μ) carrying a phylogenetic random intercept on η_μ, AND shape
# α = exp(η_σ) carrying its OWN phylogenetic random intercept on the log-σ
# (log-dispersion) axis, the two latent axes correlated through a 2×2 group-level
# covariance Λ and integrated out by a sparse augmented-state Laplace
# approximation (P = kron(Q_tree, Λ⁻¹); logdet derivatives via Takahashi selected
# inverse). This is the non-Gaussian analogue of the verified q=4 Gaussian PLSM:
# phylogenetic signal that lives in a species' *variability*, not only its mean.
#
# Two correctness anchors, the bar set by HANDOVER.md / the issue acceptance:
#
#   1. RECOVERY. From a seeded Gamma config with a KNOWN scale-axis phylogenetic
#      SD, recover the mean-axis slope β_μ, the mean-axis SD, and — the headline
#      novelty of #202 — the SCALE-axis SD, all within tolerance.
#
#   2. FD GRADIENT GATE (≤ 1e-6). The exact O(p) analytic outer gradient
#      `_ls_marginal_grad` (implicit-function + Takahashi, the same machinery as
#      the q=4 engine's `marginal_and_exact_grad`) must match a central
#      finite-difference gradient of the TRUE marginal NLL (`_ls_fit_nll`) to
#      ≤ 1e-6, evaluated at a θ OFF the optimum so the implicit (dâ/dθ)
#      corrections are nonzero — a frozen-mode gradient would fail this gate.
#      Mirrors test_poisson_phylo_grad_gate.jl / test_qgate_fd_gradient.jl.
#
# Identifiability / honesty notes (per HANDOVER.md §6 and the #202 design record):
#   * nrep ≥ 2 obs/species is REQUIRED — the scale-axis latent is unbounded below
#     at one obs/species. Both fixtures use m ≥ 2.
#   * The scale-axis SD is the near-singular (Watanabe) direction; it is only
#     well identified with enough species AND enough replicates per species, so
#     the recovery fixture is deliberately large (p=50, m=25) and the seed is
#     locked. The slope/SD tolerances are loose-but-meaningful.
#   * The log-shape intercept β_ψ carries a small-sample Laplace marginalisation
#     bias (Jensen: E[exp η_σ] ≠ exp E[η_σ]); we do NOT assert tight β_ψ recovery.
#   * The FD gate uses a SMALL, well-conditioned fixture (p=12, m=4) at a locked
#     seed: on a large/stiff fixture a far-off-optimum gradient component can be
#     O(100), where the absolute central-difference truncation/roundoff floor
#     rises above 1e-6 even though the gradient is correct. The small fixture
#     keeps |∇|≈O(1–10), where plain central differences at h=1e-4 sit at ~1e-8.

using DRModels
using Test, Random, LinearAlgebra, SparseArrays
import Distributions

# Gamma with mean μ = exp(η_μ) and shape α = exp(η_σ) (rate = α/μ), matching the
# `Val(:gamma)` kernel in src/locscale_kernels.jl: E[y]=μ, Var[y]=μ²/α.
_glsp_draw(ημ, ησ) = (α = exp(ησ); μ = exp(ημ); rand(Distributions.Gamma(α, μ / α)))

@testset "non-Gaussian phylo location–scale (#202): Gamma" begin

    # ---------------------------------------------------------------------
    # 1. Parameter recovery: mean slope, mean-axis SD, and SCALE-axis SD.
    # ---------------------------------------------------------------------
    @testset "recovery of μ- and σ-axis structure" begin
        Random.seed!(2)                          # locked: well-identified draw
        p = 50; m = 25; n = p * m                # m ≥ 2 (scale-RE identifiability)
        phy = random_balanced_tree(p; branch_length = 0.30)
        C = sigma_phy_dense(phy; σ²_phy = 1.0)
        LC = cholesky(Symmetric(C)).L

        sd_mu_true = 0.50                         # mean-axis phylo SD
        sd_psi_true = 0.50                        # SCALE-axis phylo SD (the #202 point)
        cor_true = 0.30                           # mean↔scale group-level correlation
        Λtrue = [sd_mu_true^2                    cor_true * sd_mu_true * sd_psi_true;
                 cor_true * sd_mu_true * sd_psi_true   sd_psi_true^2]
        LΛ = cholesky(Symmetric(Λtrue)).L
        A = LC * randn(p, 2) * LΛ'                # p×2 phylo-correlated (mean, scale)

        species = repeat(1:p, inner = m)
        x = randn(n)
        βμ = [0.30, 0.45]                         # mean: intercept + slope
        βψ = [0.70]                               # log-shape baseline exp(0.7)≈2.0
        Xμ = hcat(ones(n), x)
        Xψ = ones(n, 1)                           # scale axis: intercept + phylo RE
        y = [_glsp_draw(βμ[1] + βμ[2] * x[i] + A[species[i], 1],
                        βψ[1] + A[species[i], 2]) for i in 1:n]

        Q, gidx, G = DRModels._locscale_phylo_setup(phy, species)
        fit = DRModels._fit_locscale(Val(:gamma), y, Xμ, Xψ, gidx, G, Q; se = false)
        comp = fit.components                     # (sd_mu, sd_psi, cor_mu_psi)

        @test fit.converged
        @test isfinite(fit.nll)
        @test isposdef(Symmetric(fit.Lambda))     # PD group-level covariance
        # Mean-axis slope: the well-identified fixed effect.
        @test fit.beta_mu[2] ≈ βμ[2] atol = 0.10
        # Mean-axis phylogenetic SD.
        @test comp.sd_mu ≈ sd_mu_true atol = 0.18
        # SCALE-axis phylogenetic SD — the headline of #202 — recovered, and
        # genuinely away from the constant-σ boundary (sd_psi → 0).
        @test comp.sd_psi ≈ sd_psi_true atol = 0.22
        @test comp.sd_psi > 0.20
        # mean↔scale correlation is reported as a NAMED group-level summary
        # (cor_mu_psi), never as residual rho12 (per CLAUDE.md).
        @test isfinite(comp.cor_mu_psi) && -1.0 ≤ comp.cor_mu_psi ≤ 1.0
        # `_ls_components` consistency: sd_mu = sqrt(Λ[1,1]).
        @test comp.sd_mu ≈ sqrt(fit.Lambda[1, 1])
        @test comp.sd_psi ≈ sqrt(fit.Lambda[2, 2])

        # Stationarity of the EXACT outer gradient at the fitted θ (converged
        # through the tree precision + Takahashi, not merely a flat LBFGS stop).
        gmax = maximum(abs, DRModels._ls_marginal_grad(Val(:gamma), y, Xμ, Xψ, gidx, G, Q, fit.θ))
        @test gmax < 1e-3
    end

    # ---------------------------------------------------------------------
    # 2. FD gradient gate (≤ 1e-6) on the marginal logLik, OFF the optimum.
    # ---------------------------------------------------------------------
    @testset "exact outer gradient vs finite differences ≤ 1e-6" begin
        Random.seed!(165)                         # locked: well-conditioned draw
        p = 12; m = 4; n = p * m                  # small fixture, |∇|≈O(1–10); m ≥ 2
        phy = random_balanced_tree(p; branch_length = 0.25)
        C = sigma_phy_dense(phy; σ²_phy = 1.0)
        LC = cholesky(Symmetric(C)).L
        Λt = DRModels._ls_lc_to_Λ([log(0.42), 0.04, log(0.34)])
        LΛ = cholesky(Symmetric(Λt)).L
        A = LC * randn(p, 2) * LΛ'
        species = repeat(1:p, inner = m)
        x = randn(n); z = randn(n)
        Xμ = hcat(ones(n), x)
        Xψ = hcat(ones(n), z)                     # non-trivial scale design (intercept + z)
        y = [_glsp_draw(0.2 + 0.4 * x[i] + A[species[i], 1],
                        0.5 + A[species[i], 2]) for i in 1:n]

        Q, gidx, G = DRModels._locscale_phylo_setup(phy, species)

        # θ = [βμ(2); βψ(2); λ(3)], deliberately OFF the optimum so the implicit
        # dâ/dθ corrections (and every block of the gradient) are exercised.
        θ = [0.10, 0.45, 0.55, 0.05, log(0.40), 0.03, log(0.30)]

        g_an = DRModels._ls_marginal_grad(Val(:gamma), y, Xμ, Xψ, gidx, G, Q, θ)
        @test all(g_an .!= 0)                     # inner mode converged; gradient populated
        @test all(isfinite, g_an)

        # Central differences of the SAME marginal NLL `_ls_fit_nll` (each call
        # solves the inner mode cold to tol = 1e-9 → deterministic objective).
        # h = 1e-4 matches the verified q4 Q-gate: truncation O(h²) ≈ 1e-8 stays
        # under the 1e-6 bar, h large enough to avoid catastrophic cancellation.
        f = t -> DRModels._ls_fit_nll(Val(:gamma), y, Xμ, Xψ, gidx, G, Q, t)
        h = 1e-4
        g_fd = similar(g_an)
        for k in eachindex(θ)
            tp = copy(θ); tp[k] += h
            tm = copy(θ); tm[k] -= h
            g_fd[k] = (f(tp) - f(tm)) / (2h)
        end

        max_abs_diff = maximum(abs, g_an .- g_fd)
        @info "non-Gaussian phylo location–scale gradient gate (#202)" max_abs_diff
        @test max_abs_diff ≤ 1e-6
    end
end
