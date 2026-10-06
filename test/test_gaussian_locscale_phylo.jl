# test_gaussian_locscale_phylo.jl — B1 of the σ-phylo plan.
#
# Univariate Gaussian location-scale with phylogenetic REs on BOTH axes via
# the q=2 augmented-state Laplace spine and the :gaussian_mean kernel.
#
# Three acceptance criteria:
#   (a) RECOVERY — separate-block DGP (G≈128, m≥2) recovers sd(μ-phylo) and
#       sd(σ-phylo) within sampling; coupled DGP recovers the correlation.
#   (b) FD GATE — analytic gradient vs 5-point central FD with h-sweep trough
#       (mirrors test_sigma_axis_re.jl's _fd_gate_min discipline), ≤ 1e-6.
#   (c) NO SILENT DROP — sigma~phylo now FITS and differs from sigma~1.
#
# Hard gate before commit: all [Pass] lines visible in the @testset output.

using DRModels
using Test, Random, LinearAlgebra, SparseArrays
import Distributions

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# 5-point stencil FD gradient (mirrors test_sigma_axis_re.jl).
function _fd5_grad(obj, θ; h = 1e-4)
    g = zeros(length(θ))
    for k in eachindex(θ)
        tp2 = copy(θ); tp2[k] += 2h
        tp1 = copy(θ); tp1[k] += h
        tm1 = copy(θ); tm1[k] -= h
        tm2 = copy(θ); tm2[k] -= 2h
        g[k] = (-obj(tp2) + 8obj(tp1) - 8obj(tm1) + obj(tm2)) / (12h)
    end
    return g
end

# h-sweep trough: minimum max-abs discrepancy across a set of stencil sizes,
# recovering the TRUE analytic accuracy instead of inner-mode-solve noise at a
# single small h.  (The #164 lesson: marginal FD at a fixed tiny h reads
# noise ≈ inner_residual / h, NOT analytic error.)
function _fd_gate_min_glsp(g_an, obj, θ; hs = (1e-2, 3e-3, 1e-3))
    minimum(maximum(abs, g_an .- _fd5_grad(obj, θ; h = h)) for h in hs)
end

# ---------------------------------------------------------------------------
# (c) NO SILENT DROP — must fit and differ from constant sigma
# ---------------------------------------------------------------------------
@testset "Gaussian σ-phylo: no silent drop (B1 route fires)" begin
    Random.seed!(202606111)
    p = 30; m = 4; n = p * m
    phy = random_balanced_tree(p; branch_length = 0.30)
    C   = sigma_phy_dense(phy; σ²_phy = 1.0)
    LC  = cholesky(Symmetric(C)).L
    u_sigma = 0.5 .* (LC * randn(p))   # phylo random effect on log σ
    u_mu    = 0.4 .* (LC * randn(p))   # phylo random effect on mean

    species = repeat(1:p, inner = m)
    x = randn(n)
    βμ = [0.5, 0.3]
    βψ = [0.2]
    Xμ = hcat(ones(n), x)
    y  = [βμ[1] + βμ[2]*x[i] + u_mu[species[i]] +
          exp(βψ[1] + u_sigma[species[i]]) * randn() for i in 1:n]
    data = (; y, x, species)

    # Fit the B1 route (both-phylo SEPARATE).
    fit_b1  = drm(bf(@formula(y ~ x + phylo(1 | species)),
                     @formula(sigma ~ phylo(1 | species))),
                  Gaussian(); data = data, tree = phy)
    # Fit the baseline (constant sigma).
    fit_base = drm(bf(@formula(y ~ x + phylo(1 | species)),
                      @formula(sigma ~ 1)),
                   Gaussian(); data = data, tree = phy)

    @test isfinite(loglik(fit_b1))
    @test isfinite(loglik(fit_base))
    # B1 must have a different (better) logLik than the constant-sigma baseline.
    @test loglik(fit_b1) > loglik(fit_base) - 1e-3
    # The sd_sigma parameter must be accessible and positive.
    sds = DRModels.gaussian_locscale_phylo_sds(fit_b1)
    @test sds.sd_sigma > 0.0
    @test sds.sd_mu    > 0.0
end

# ---------------------------------------------------------------------------
# (a) RECOVERY — separate-block (MUST-HAVE)
# ---------------------------------------------------------------------------
@testset "Gaussian σ-phylo: separate-block recovery" begin
    Random.seed!(202606112)
    p = 128; m = 4; n = p * m          # G≈128 species, m≥2 obs each
    phy = random_balanced_tree(p; branch_length = 0.30)
    C   = sigma_phy_dense(phy; σ²_phy = 1.0)
    LC  = cholesky(Symmetric(C)).L

    sd_mu_true    = 0.70
    sd_sigma_true = 0.60

    # SEPARATE DGP: u_mu and u_sigma are INDEPENDENT phylo draws.
    u_mu    = sd_mu_true    .* (LC * randn(p))
    u_sigma = sd_sigma_true .* (LC * randn(p))

    species = repeat(1:p, inner = m)
    x = randn(n)
    βμ = [0.5, 0.3]; βψ = [0.2]
    y  = [βμ[1] + βμ[2]*x[i] + u_mu[species[i]] +
          exp(βψ[1] + u_sigma[species[i]]) * randn() for i in 1:n]
    data = (; y, x, species)

    fit = drm(bf(@formula(y ~ x + phylo(1 | species)),
                 @formula(sigma ~ phylo(1 | species))),
              Gaussian(); data = data, tree = phy)

    @test is_converged(fit)
    @test isfinite(loglik(fit))
    sds = DRModels.gaussian_locscale_phylo_sds(fit)
    @info "Separate-block recovery" sd_mu_true sds.sd_mu sd_sigma_true sds.sd_sigma
    # Recovery within sampling (generous tolerances for a Laplace approximation).
    @test sds.sd_mu    ≈ sd_mu_true    atol = 0.25
    @test sds.sd_sigma ≈ sd_sigma_true atol = 0.25
    # sd_sigma must be genuinely above zero (not collapsed to the boundary).
    @test sds.sd_sigma > 0.15
end

# ---------------------------------------------------------------------------
# (a) RECOVERY — coupled block (secondary option)
# ---------------------------------------------------------------------------
@testset "Gaussian σ-phylo: coupled-block correlation recovery" begin
    Random.seed!(202606113)
    p = 80; m = 5; n = p * m
    phy = random_balanced_tree(p; branch_length = 0.25)
    C   = sigma_phy_dense(phy; σ²_phy = 1.0)
    LC  = cholesky(Symmetric(C)).L

    sd_mu_true    = 0.55
    sd_sigma_true = 0.45
    cor_true      = 0.50
    Λtrue = [sd_mu_true^2                          cor_true*sd_mu_true*sd_sigma_true;
             cor_true*sd_mu_true*sd_sigma_true      sd_sigma_true^2]
    LΛ = cholesky(Symmetric(Λtrue)).L
    # Correlated phylo draws: [u_mu, u_sigma] = C^{1/2} × Λ^{1/2} × Z.
    Z = randn(p, 2)
    A_draws = LC * Z * LΛ'   # p×2
    u_mu    = A_draws[:, 1]
    u_sigma = A_draws[:, 2]

    species = repeat(1:p, inner = m)
    x = randn(n)
    βμ = [0.4, 0.2]; βψ = [0.0]
    y  = [βμ[1] + βμ[2]*x[i] + u_mu[species[i]] +
          exp(βψ[1] + u_sigma[species[i]]) * randn() for i in 1:n]
    data = (; y, x, species)

    # The coupled route is reached via the internal function directly (the public
    # grammar always dispatches to SEPARATE when both axes carry `phylo(1|sp)`;
    # `coupled=true` is the internal kwarg for the correlated variant).
    Q, gidx, G = DRModels._locscale_phylo_setup(phy, data.species)
    Xμ = hcat(ones(n), data.x); nmμ = ["(Intercept)", "x"]   # intercept + x
    Xψ = reshape(ones(n), n, 1); nmσ = ["(Intercept)"]
    fit = DRModels._fit_gaussian_locscale_phylo(
        Gaussian(), Float64.(y), Xμ, Xψ, gidx, G, Q,
        nmμ, nmσ, "species";
        coupled = true, asymmetric = false, se = false, g_tol = 1e-6)

    @test isfinite(loglik(fit))
    sds = DRModels.gaussian_locscale_phylo_sds(fit)
    @info "Coupled-block recovery" sd_mu_true sds.sd_mu sd_sigma_true sds.sd_sigma
    cor_est = get(fit.scales, :lambda_cor, [NaN])[1]
    @info "Coupled correlation" cor_true cor_est
    # SDs within sampling (wider tolerance for the coupled block).
    @test sds.sd_mu    ≈ sd_mu_true    atol = 0.30
    @test sds.sd_sigma ≈ sd_sigma_true atol = 0.30
    # Correlation direction must be right (not necessarily tight).
    @test cor_est > 0.0
end

# ---------------------------------------------------------------------------
# (b) FD GATE — separate-block analytic gradient vs 5-point FD (≤ 1e-6)
# ---------------------------------------------------------------------------
@testset "Gaussian σ-phylo: FD gradient gate (separate, ≤1e-6)" begin
    Random.seed!(202606114)
    p = 16; m = 4; n = p * m         # small fixture: |∇| ≈ O(1–10); m ≥ 2
    phy = random_balanced_tree(p; branch_length = 0.20)
    C   = sigma_phy_dense(phy; σ²_phy = 1.0)
    LC  = cholesky(Symmetric(C)).L
    u_mu    = 0.50 .* (LC * randn(p))
    u_sigma = 0.40 .* (LC * randn(p))
    species = repeat(1:p, inner = m)
    x = randn(n)
    y = [0.3 + 0.4*x[i] + u_mu[species[i]] +
         exp(0.1 + u_sigma[species[i]]) * randn() for i in 1:n]
    Xμ = hcat(ones(n), x)
    Xψ = ones(n, 1)
    Q, gidx, G = DRModels._locscale_phylo_setup(phy, species)
    Zη = DRModels._ls_canonical_Zeta(n)
    Zψ = DRModels._ls_canonical_Zpsi(n)

    # θ = [βμ(2); βψ(1); logL11; logL22], deliberately off-optimum.
    θ = [0.20, 0.35, 0.05, log(0.38), log(0.30)]

    kind = Val(:gaussian_mean)
    g_an = DRModels._glsp_sep_grad(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ)
    @test all(isfinite, g_an)
    @test any(g_an .!= 0.0)

    # Cold-start FD objective (each call independently resolves the inner mode).
    obj_fd = θ_ -> DRModels._glsp_sep_nll(kind, y, Xμ, Xψ, gidx, G, Q, θ_, Zη, Zψ)
    trough = _fd_gate_min_glsp(g_an, obj_fd, θ)
    @info "Gaussian σ-phylo FD gate (separate, h-sweep trough)" trough
    @test trough ≤ 1e-6
end

# ---------------------------------------------------------------------------
# (b) FD GATE — asymmetric (σ-phylo only) analytic gradient vs 5-point FD
# ---------------------------------------------------------------------------
@testset "Gaussian σ-phylo: FD gradient gate (asymmetric, ≤1e-6)" begin
    Random.seed!(202606115)
    p = 16; m = 4; n = p * m
    phy = random_balanced_tree(p; branch_length = 0.20)
    C   = sigma_phy_dense(phy; σ²_phy = 1.0)
    LC  = cholesky(Symmetric(C)).L
    u_sigma = 0.40 .* (LC * randn(p))
    species = repeat(1:p, inner = m)
    x = randn(n)
    y = [0.3 + 0.4*x[i] + exp(0.1 + u_sigma[species[i]]) * randn() for i in 1:n]
    Xμ = hcat(ones(n), x)
    Xψ = ones(n, 1)
    Q, gidx, G = DRModels._locscale_phylo_setup(phy, species)
    Zη, Zψ = DRModels._glsp_asym_loadings(n)

    # θ = [βμ(2); βψ(1); logL22], off-optimum.
    θ = [0.25, 0.35, 0.05, log(0.35)]

    kind = Val(:gaussian_mean)
    g_an = DRModels._glsp_asym_grad(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ)
    @test all(isfinite, g_an)

    obj_fd = θ_ -> DRModels._glsp_asym_nll(kind, y, Xμ, Xψ, gidx, G, Q, θ_, Zη, Zψ)
    trough = _fd_gate_min_glsp(g_an, obj_fd, θ)
    @info "Gaussian σ-phylo FD gate (asymmetric, h-sweep trough)" trough
    @test trough ≤ 1e-6
end
