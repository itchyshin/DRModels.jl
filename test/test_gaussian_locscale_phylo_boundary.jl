# test_gaussian_locscale_phylo_boundary.jl — B2 of the σ-phylo plan.
#
# Boundary-aware PROFILE-LIKELIHOOD CIs for the phylo SDs (opt-in `profile_ci=true`).
# The honest-inference contract (Ayumi issue #2, bucket 3): where a σ-phylo variance
# component is identified, its CI is bounded away from 0; where the data carry NO
# scale-phylo signal, the CI is `[0, x]` — the boundary IS the result, not a crash.
#
# Two acceptance cells:
#   (A) WELL-IDENTIFIED separate block (σ=0.6) — profile CI sd_sigma bounded from 0,
#       finite upper, point inside; sd_mu likewise bounded (both axes identified).
#   (B) ABSENT σ-phylo signal (sd_σ_true = 0), asymmetric route — profile CI sd_sigma
#       lower endpoint is exactly 0 (honest [0, x]); no false-convergence / NaN crash.
#
# Hard gate before commit: both [Pass] cells visible in the @testset output.

using DRModels
using Test, Random, LinearAlgebra
import Distributions

# ---------------------------------------------------------------------------
# (A) WELL-IDENTIFIED — CI bounded away from the boundary
# ---------------------------------------------------------------------------
@testset "σ-phylo boundary: well-identified → CI excludes 0" begin
    Random.seed!(202606121)
    p = 64; m = 4; n = p * m
    phy = random_balanced_tree(p; branch_length = 0.30)
    C   = sigma_phy_dense(phy; σ²_phy = 1.0)
    LC  = cholesky(Symmetric(C)).L

    sd_mu_true    = 0.70
    sd_sigma_true = 0.60
    u_mu    = sd_mu_true    .* (LC * randn(p))
    u_sigma = sd_sigma_true .* (LC * randn(p))

    species = repeat(1:p, inner = m)
    x = randn(n); βμ = [0.5, 0.3]; βψ = [0.2]
    y  = [βμ[1] + βμ[2]*x[i] + u_mu[species[i]] +
          exp(βψ[1] + u_sigma[species[i]]) * randn() for i in 1:n]
    data = (; y, x, species)

    fit = drm(bf(@formula(y ~ x + phylo(1 | species)),
                 @formula(sigma ~ phylo(1 | species))),
              Gaussian(); data = data, tree = phy, profile_ci = true)

    @test is_converged(fit)
    ci_s = fit.scales[:profile_ci_sd_sigma]
    ci_m = fit.scales[:profile_ci_sd_mu]
    sds  = DRModels.gaussian_locscale_phylo_sds(fit)
    @info "Well-identified σ-phylo profile CIs" sds.sd_sigma ci_s sds.sd_mu ci_m
    # σ-phylo CI is bounded away from the boundary and finite above.
    @test ci_s[1] > 0.10
    @test isfinite(ci_s[2])
    @test ci_s[2] < Inf
    # Point estimate lies inside its own CI.
    @test ci_s[1] ≤ sds.sd_sigma ≤ ci_s[2]
    # μ-phylo axis is likewise identified (lower endpoint above 0).
    @test ci_m[1] > 0.0
    @test isfinite(ci_m[2])
end

# ---------------------------------------------------------------------------
# (B) ABSENT σ-phylo signal — honest [0, x] CI at the boundary
# ---------------------------------------------------------------------------
@testset "σ-phylo boundary: absent signal → CI includes 0" begin
    Random.seed!(202606122)
    p = 64; m = 4; n = p * m
    phy = random_balanced_tree(p; branch_length = 0.30)

    # NO phylogenetic signal in the variance: σ is constant across species.
    species = repeat(1:p, inner = m)
    x = randn(n); βμ = [0.4, 0.2]; βψ = [0.1]
    y  = [βμ[1] + βμ[2]*x[i] + exp(βψ[1]) * randn() for i in 1:n]
    data = (; y, x, species)

    # Asymmetric route: σ ~ phylo, mean is fixed-effects (the σ-only boundary case).
    fit = drm(bf(@formula(y ~ x),
                 @formula(sigma ~ phylo(1 | species))),
              Gaussian(); data = data, tree = phy, profile_ci = true)

    # A variance boundary is a real stop. The gradient criterion does not fire
    # there, so this check is that the fit returned a finite likelihood.
    @test isfinite(loglik(fit))
    ci_s = fit.scales[:profile_ci_sd_sigma]
    sds  = DRModels.gaussian_locscale_phylo_sds(fit)
    @info "Absent σ-phylo signal profile CI" sds.sd_sigma ci_s
    # The honest boundary interval: lower endpoint is exactly 0.
    @test ci_s[1] == 0.0
    # The point estimate is at/near the boundary too.
    @test sds.sd_sigma < 0.15
end
