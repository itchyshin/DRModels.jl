# test_locscale_structured.jl — cluster ④: STRUCTURED non-Gaussian
# location–scale. Two extensions of the verified q=2 location–scale engine:
#
#   (1) STRUCTURED group-level covariance — a user-supplied PD relatedness /
#       spatial matrix `C` routed through `_locscale_relmat_setup` (= C⁻¹, the
#       non-tree analogue of `_locscale_phylo_setup`), so the (mean, log-σ) random
#       effects are correlated ACROSS group levels by `C`, not i.i.d. (Q = I).
#   (2) BETA and BETA-BINOMIAL leaves — the two-axis (logit mean, log-σ)
#       conditional kernels, with the drmTMB precision mapping φ = exp(-2ψ).
#       (A plain Binomial has no free dispersion, so the overdispersed-binomial
#       leaf IS the beta-binomial — its φ axis is the scale axis.)
#
# Engine-lane test: builds `(Q, gidx, G)` directly via `_locscale_relmat_setup`.
# The Beta recovery keeps the historical raw-engine oracle. Beta-Binomial uses
# the certified whitened route selected by the canonical coupled frontend; its
# raw mathematics remains covered by the separate off-optimum gradient gate.
# The public drm() coupled
# `(1 | tag | g)` route for these families is exercised in test_nonconst_sigma_re
# (i.i.d.) and the structured `(1 | tag | relmat(g))` route in
# test_locscale_frontend.jl.
#
# Two correctness anchors (the cluster-④ Definition of Done):
#   1. RECOVERY through the structured covariance: mean slope, mean-axis SD,
#      SCALE-axis SD, mean↔scale correlation, and the σ-axis covariate SLOPE.
#   2. FD GRADIENT GATE — the exact O(p) outer gradient `_ls_marginal_grad`
#      matches a finite-difference gradient of the marginal NLL at a θ OFF the
#      optimum (so dâ/dθ corrections are exercised), inner mode solved cold to
#      tol = 1e-9. Both leaves use the RELATIVE gate of test_locscale_grad.jl
#      (`max|g_an − g_fd| < tol·(1 + max|g_fd|)`, tol = 1e-6) — the engine-wide
#      standard, robust to the differing block magnitudes and to the inner-mode
#      noise floor (≈1e-9 in the NLL, amplified by 1/2h) that an absolute bar
#      would conflate with a true gradient error. The 5-POINT stencil (truncation
#      O(h⁴), the cluster-④ fallback) is used for both: it is REQUIRED for the
#      ForwardDiff-through-the-discrete-pmf beta-binomial leaf, and the analytic
#      Beta leaf clears the same relative bar with it. Achieved abs/rel diffs are
#      @info-logged.
#
# Identifiability / honesty notes (mirroring test_phylo_locscale §):
#   * m ≥ 2 obs/level is REQUIRED (the scale-axis latent is unbounded below at one
#     obs/level); the recovery fixtures use m = 22 (Beta) and m = 12
#     (Beta-Binomial).
#   * The σ-axis SD is the near-singular (Watanabe) direction — only well
#     identified with enough levels AND replicates. The raw Beta fixture remains
#     large (G = 70, m = 22); the certified Beta-Binomial fixture uses G = 30,
#     m = 12 at its locked seed and retains meaningful recovery bounds.
#   * Intercepts (βμ1, βψ1) carry a small-sample Laplace/Jensen bias and are NOT
#     asserted tightly. The historical raw Beta fit uses exact-gradient
#     stationarity because its optimizer flag can stop on a variance plateau;
#     the certified Beta-Binomial fit must report convergence as well.
using DRModels
using Test, Random, LinearAlgebra, SparseArrays
import Distributions

# A genuine, well-conditioned PD correlation among `G` levels from an exponential
# kernel over random latent positions — a relatedness/spatial-style matrix that
# is NOT a tree, rescaled to unit diagonal (so the recovered Λ is on the SD
# scale). Same idiom as test_relmat_counts_nb2.jl / test_gaussian_structured.jl.
function _ls_random_corr(rng, G; range = 0.8, jitter = 1e-8)
    pos = rand(rng, G, 2) .* 6.0
    D = [sqrt(sum(abs2, pos[k, :] .- pos[l, :])) for k in 1:G, l in 1:G]
    C = exp.(-D ./ range) + jitter * I
    d = sqrt.(diag(C))
    return Symmetric(C ./ (d * d'))
end

# 5-point central-difference gradient (truncation O(h⁴)); the cluster-④ fallback
# stencil. Matches test_q4_laplace.jl. h = 1e-3 keeps the inner-mode noise floor
# (≈1e-9 in the NLL, amplified by 1/h) below the relative bar.
function _ls_fd5(f, θ; h = 1e-3)
    g = similar(θ)
    for k in eachindex(θ)
        e = zeros(length(θ)); e[k] = 1.0
        g[k] = (-f(θ .+ 2h .* e) + 8f(θ .+ h .* e) - 8f(θ .- h .* e) + f(θ .- 2h .* e)) / (12h)
    end
    return g
end

# Relative agreement check, robust to differing block magnitudes (the engine-wide
# convention from test_locscale_grad.jl:24).
_ls_grad_ok(ga, gfd; tol = 1e-6) =
    maximum(abs, ga .- gfd) < tol * (1 + maximum(abs, gfd))

@testset "structured non-Gaussian location–scale (cluster ④)" begin

    # =====================================================================
    # 1a. RELMAT + BETA — recovery of μ-/σ-axis structure + σ-axis slope.
    # =====================================================================
    @testset "Beta + relmat: recovery through C⁻¹" begin
        Random.seed!(2027)                       # locked: well-identified draw
        G = 70; m = 22; n = G * m                 # m ≥ 2; large for σ-SD identifiability
        id = repeat(1:G, inner = m)
        C = _ls_random_corr(MersenneTwister(99), G)
        LC = cholesky(C).L

        sd_mu_true = 0.50; sd_psi_true = 0.50; cor_true = 0.30
        Λtrue = [sd_mu_true^2                     cor_true * sd_mu_true * sd_psi_true;
                 cor_true * sd_mu_true * sd_psi_true   sd_psi_true^2]
        LΛ = cholesky(Symmetric(Λtrue)).L
        A = LC * randn(G, 2) * LΛ'                # G×2 C-correlated (mean, scale)

        x = randn(n); z = randn(n)
        βμ = [0.0, 0.45]                          # logit mean: intercept + slope
        βψ = [-0.30, 0.40]                        # log σ: intercept + slope (γ1 = 0.4 ≠ 0)
        Xμ = hcat(ones(n), x); Xψ = hcat(ones(n), z)
        y = Float64[]
        for i in 1:n
            μ = 1 / (1 + exp(-(βμ[1] + βμ[2] * x[i] + A[id[i], 1])))
            φ = exp(-2 * (βψ[1] + βψ[2] * z[i] + A[id[i], 2]))
            push!(y, clamp(rand(Distributions.Beta(μ * φ, (1 - μ) * φ)), 1e-7, 1 - 1e-7))
        end

        Q, gidx, Gd = DRModels._locscale_relmat_setup(C, id)
        @test issparse(Q) && size(Q) == (G, G)    # structured precision wired in
        fit = DRModels._fit_locscale(Val(:beta), y, Xμ, Xψ, gidx, Gd, Q; se = false)
        comp = fit.components

        @test isfinite(fit.nll)
        @test isposdef(Symmetric(fit.Lambda))     # PD group-level covariance
        @test fit.beta_mu[2] ≈ βμ[2] atol = 0.12  # well-identified mean slope
        @test fit.beta_psi[2] ≈ βψ[2] atol = 0.15 # ← non-constant σ slope, the headline
        @test comp.sd_mu ≈ sd_mu_true atol = 0.18
        @test comp.sd_psi ≈ sd_psi_true atol = 0.22
        @test comp.sd_psi > 0.20                  # genuinely off the constant-σ boundary
        # mean↔scale correlation: a NAMED group-level summary (never residual rho12).
        @test isfinite(comp.cor_mu_psi) && -1.0 ≤ comp.cor_mu_psi ≤ 1.0
        @test comp.sd_mu ≈ sqrt(fit.Lambda[1, 1])
        @test comp.sd_psi ≈ sqrt(fit.Lambda[2, 2])
        # Stationarity of the EXACT outer gradient at the fit (converged through
        # C⁻¹ + Takahashi, not a flat LBFGS stop).
        gmax = maximum(abs, DRModels._ls_marginal_grad(Val(:beta), y, Xμ, Xψ, gidx, Gd, Q, fit.θ))
        @test gmax < 1e-3
    end

    # =====================================================================
    # 1b. RELMAT + BETA-BINOMIAL — recovery of μ-/σ-axis structure + slope.
    # =====================================================================
    @testset "BetaBinomial + relmat: recovery through C⁻¹" begin
        Random.seed!(3031)
        G = 30; m = 12; n = G * m
        id = repeat(1:G, inner = m)
        C = _ls_random_corr(MersenneTwister(131), G)
        LC = cholesky(C).L

        sd_mu_true = 0.50; sd_psi_true = 0.50; cor_true = 0.30
        Λtrue = [sd_mu_true^2                     cor_true * sd_mu_true * sd_psi_true;
                 cor_true * sd_mu_true * sd_psi_true   sd_psi_true^2]
        LΛ = cholesky(Symmetric(Λtrue)).L
        A = LC * randn(G, 2) * LΛ'

        x = randn(n); z = randn(n)
        βμ = [0.0, 0.45]; βψ = [-0.30, 0.40]
        Xμ = hcat(ones(n), x); Xψ = hcat(ones(n), z)
        ntr = 30                                   # fixed trials per obs
        y = Tuple{Float64,Float64}[]
        for i in 1:n
            μ = 1 / (1 + exp(-(βμ[1] + βμ[2] * x[i] + A[id[i], 1])))
            φ = exp(-2 * (βψ[1] + βψ[2] * z[i] + A[id[i], 2]))
            s = rand(Distributions.BetaBinomial(ntr, μ * φ, (1 - μ) * φ))
            push!(y, (Float64(s), Float64(ntr)))   # engine response = (successes, trials)
        end

        Q, gidx, Gd = DRModels._locscale_relmat_setup(C, id)
        fit = DRModels._fit_locscale(Val(:betabinomial), y, Xμ, Xψ, gidx, Gd, Q;
                                se = false, whitened = true)
        comp = fit.components

        @test fit.converged
        @test isfinite(fit.nll)
        @test isposdef(Symmetric(fit.Lambda))
        @test fit.beta_mu[2] ≈ βμ[2] atol = 0.12
        @test fit.beta_psi[2] ≈ βψ[2] atol = 0.15  # ← non-constant σ slope
        @test comp.sd_mu ≈ sd_mu_true atol = 0.18
        @test comp.sd_psi ≈ sd_psi_true atol = 0.22
        @test comp.sd_psi > 0.20
        @test isfinite(comp.cor_mu_psi) && -1.0 ≤ comp.cor_mu_psi ≤ 1.0
        certified = DRModels._ls_whitened_eval(
            Val(:betabinomial), y, Xμ, Xψ, gidx, Gd, Q, fit.θ,
            DRModels._ls_canonical_Zeta(length(y)), DRModels._ls_canonical_Zpsi(length(y)),
        )
        @test certified.status.ok
        @test maximum(abs, certified.gradient) < 1e-3
    end

    # =====================================================================
    # 2. FD GRADIENT GATE — exact O(p) outer gradient vs finite differences,
    #    on a structured (relmat) Q, OFF the optimum, inner mode cold to 1e-9.
    # =====================================================================
    @testset "exact outer gradient vs finite differences (relmat Q)" begin
        # Small, well-conditioned fixture so |∇| ≈ O(1–10) and the central-
        # difference floor stays well under the bar.
        Random.seed!(7717)
        G = 12; m = 4; n = G * m                   # small; m ≥ 2
        id = repeat(1:G, inner = m)
        C = _ls_random_corr(MersenneTwister(53), G)
        LC = cholesky(C).L
        Λt = DRModels._ls_lc_to_Λ([log(0.45), 0.04, log(0.40)]); LΛ = cholesky(Symmetric(Λt)).L
        A = LC * randn(G, 2) * LΛ'
        x = randn(n); z = randn(n)
        Xμ = hcat(ones(n), x); Xψ = hcat(ones(n), z)

        # θ deliberately OFF the optimum so dâ/dθ corrections (every gradient
        # block) are exercised — a frozen-mode gradient fails this gate.
        θ = [0.05, 0.45, 0.35, 0.05, log(0.42), 0.03, log(0.38)]

        Qb, gidxb, Gb = DRModels._locscale_relmat_setup(C, id)

        # ---- Beta (analytic kernels), 5-point stencil, relative gate. ----
        yβ = Float64[]
        for i in 1:n
            μ = 1 / (1 + exp(-(0.10 + 0.40 * x[i] + A[id[i], 1])))
            φ = exp(-2 * (0.30 + 0.40 * z[i] + A[id[i], 2]))
            push!(yβ, clamp(rand(Distributions.Beta(μ * φ, (1 - μ) * φ)), 1e-6, 1 - 1e-6))
        end
        g_an_β = DRModels._ls_marginal_grad(Val(:beta), yβ, Xμ, Xψ, gidxb, Gb, Qb, θ)
        @test all(g_an_β .!= 0)                    # inner mode converged
        @test all(isfinite, g_an_β)
        fβ = t -> DRModels._ls_fit_nll(Val(:beta), yβ, Xμ, Xψ, gidxb, Gb, Qb, t)
        g_fd_β = _ls_fd5(fβ, θ)
        diff_β = maximum(abs, g_an_β .- g_fd_β)
        @info "cluster ④ Beta relmat gradient gate" abs_diff = diff_β rel_bound = 1e-6 * (1 + maximum(abs, g_fd_β))
        @test _ls_grad_ok(g_an_β, g_fd_β)

        # ---- BetaBinomial (ForwardDiff-through-pmf), 5-point, relative gate. --
        ntr = 25; ybb = Tuple{Float64,Float64}[]
        for i in 1:n
            μ = 1 / (1 + exp(-(0.10 + 0.40 * x[i] + A[id[i], 1])))
            φ = exp(-2 * (0.30 + 0.40 * z[i] + A[id[i], 2]))
            s = rand(Distributions.BetaBinomial(ntr, μ * φ, (1 - μ) * φ))
            push!(ybb, (Float64(s), Float64(ntr)))
        end
        g_an_bb = DRModels._ls_marginal_grad(Val(:betabinomial), ybb, Xμ, Xψ, gidxb, Gb, Qb, θ)
        @test all(g_an_bb .!= 0)
        @test all(isfinite, g_an_bb)
        fbb = t -> DRModels._ls_fit_nll(Val(:betabinomial), ybb, Xμ, Xψ, gidxb, Gb, Qb, t)
        g_fd_bb = _ls_fd5(fbb, θ)
        diff_bb = maximum(abs, g_an_bb .- g_fd_bb)
        @info "cluster ④ BetaBinomial relmat gradient gate (5-point)" abs_diff = diff_bb rel_bound = 1e-6 * (1 + maximum(abs, g_fd_bb))
        @test _ls_grad_ok(g_an_bb, g_fd_bb)
    end
end
