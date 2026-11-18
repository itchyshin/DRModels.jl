# test_reml_q4_allaxes.jl — REML for the bivariate q=4 PLSM must correct ALL FOUR
# among-axis SDs, not just the two means.
#
# Regression for the scale-axis REML gap (issue #18). The q4 REML profiles out the
# location AND scale fixed effects (β_μ1, β_μ2, β_s1, β_s2) via a bordered augmented
# state; an earlier build profiled only the means, so the restricted correction
# reached 2 of the 4 axes and the scale among-SDs moved the WRONG way under REML
# (σ1 among-SD went DOWN vs ML). The defining REML property is that the restricted
# estimate is less downward-biased: diag(Σ_a)_REML ≥ diag(Σ_a)_ML on EVERY axis.
#
# This drives the public path (drm(method=:ML/:REML)) on synthetic data with a real
# among-scale signal, and asserts the property on all four axes.
using Random, LinearAlgebra

@testset "REML q4: restricted correction reaches all four axes (#18)" begin
    Random.seed!(20260614)
    p = 16; m = 5
    phy = random_balanced_tree(p; branch_length = 0.3)
    Σphy = sigma_phy_dense(phy; σ²_phy = 1.0)
    LC = cholesky(Symmetric(Σphy)).L
    # Among-axis covariance with genuine variance on BOTH scale axes (3,4), so the
    # scale-axis REML correction has something to correct.
    Λ_among = [0.25 0.10 0.05 0.00;
               0.10 0.25 0.00 0.04;
               0.05 0.00 0.16 0.02;
               0.00 0.04 0.02 0.16]
    U = LC * randn(p, 4) * cholesky(Symmetric(Λ_among)).L'
    sp = repeat(1:p, inner = m); n = length(sp); x = randn(n)
    y1 = 0.5 .+ 0.3 .* x .+ U[sp, 1] .+ exp.(-0.6 .+ U[sp, 3]) .* randn(n)
    y2 = -0.2 .+ 0.4 .* x .+ U[sp, 2] .+ exp.(-0.6 .+ U[sp, 4]) .* randn(n)
    dat = (; y1, y2, x, species = phy.leaf_names[sp])

    form = bf(mu1    = @formula(y1 ~ x + phylo(1 | species)),
              mu2    = @formula(y2 ~ x + phylo(1 | species)),
              sigma1 = @formula(sigma1 ~ 1 + phylo(1 | species)),
              sigma2 = @formula(sigma2 ~ 1 + phylo(1 | species)),
              rho12  = @formula(rho12 ~ 1))

    fml = drm(form, Gaussian(); data = dat, tree = phy, method = :ML,   q4_vcov = false)
    frl = drm(form, Gaussian(); data = dat, tree = phy, method = :REML, q4_vcov = false)

    sd_ml   = sqrt.(max.(diag(fml.ranef.Sigma_a), 0.0))
    sd_reml = sqrt.(max.(diag(frl.ranef.Sigma_a), 0.0))

    @test all(isfinite, sd_ml) && all(isfinite, sd_reml)
    # Defining property on ALL FOUR axes (mu1, mu2, sigma1, sigma2), not just means.
    @test all(sd_reml .>= sd_ml .- 1e-6)
    # The two SCALE axes in particular must be ≥ ML (the axes the old build got wrong).
    @test sd_reml[3] >= sd_ml[3] - 1e-6
    @test sd_reml[4] >= sd_ml[4] - 1e-6
    # REML genuinely changes the fit (not a silent no-op).
    @test maximum(abs.(sd_reml .- sd_ml)) > 1e-4

    boot = bootstrap_sigma_a(frl; data = dat, B = 2,
                             rng = Random.MersenneTwister(20260615),
                             failures = :error, check_converged = false)
    @test boot.attempted == 2
    @test boot.used == 2
    @test boot.failed == 0
    @test all(isfinite, [row.estimate for row in boot.summary])
end
