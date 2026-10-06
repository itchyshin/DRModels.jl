# Parametric bootstrap of the q=4 phylogenetic among-axis SDs (Ayumi #2).
# A deliberate σ2-axis collapse (no scale-phylo signal on trait 2) checks that the
# percentile interval is boundary-honest: the collapsed axis's whole CI sits below
# an identified axis's CI, and the SD point estimates match vc(fit).
using DRModels, Test, Random, LinearAlgebra
import Statistics

@testset "bootstrap_sigma_a — q4 among-axis SD CIs (boundary-honest)" begin
    Random.seed!(20260613)
    p, m = 24, 5
    phy = random_balanced_tree(p; branch_length = 0.3)
    C = sigma_phy_dense(phy; σ²_phy = 1.0)
    LC = cholesky(Symmetric(C)).L
    Z = randn(p, 4)
    Lmu = cholesky([0.64 0.30; 0.30 0.64]).L
    # axes: mu1, mu2, sigma1 carry phylo signal; sigma2 collapses (zeros).
    U = hcat(LC * Z[:, 1:2] * Lmu', LC * Z[:, 3] * 0.5, zeros(p))
    species = repeat(1:p, inner = m)
    n = length(species)
    x = randn(n)
    μ1 = 0.5 .+ 0.3 .* x .+ U[species, 1]
    μ2 = -0.2 .+ 0.4 .* x .+ U[species, 2]
    σ1 = exp.(-1.0 .+ U[species, 3])
    σ2 = exp.(-1.0 .+ U[species, 4])
    ρ = 0.3
    e1 = randn(n)
    e2 = ρ .* e1 .+ sqrt(1 - ρ^2) .* randn(n)
    dat = (; y1 = μ1 .+ σ1 .* e1, y2 = μ2 .+ σ2 .* e2, x, species)
    form = bf(mu1 = @formula(y1 ~ x + phylo(1 | species)),
              mu2 = @formula(y2 ~ x + phylo(1 | species)),
              sigma1 = @formula(sigma1 ~ 1 + phylo(1 | species)),
              sigma2 = @formula(sigma2 ~ 1 + phylo(1 | species)),
              rho12 = @formula(rho12 ~ 1))
    fit = drm(form, Gaussian(); data = dat, tree = phy)
    @test is_converged(fit)

    sd_fit = sqrt.(max.(diag(vc(fit)[:species]), 0.0))
    br = bootstrap_sigma_a(fit; data = dat, B = 16, rng = MersenneTwister(1))

    # shape + naming (loose convergence floor: cross-platform CI-robust; the
    # boundary-separation assertion below is the real correctness check)
    @test br.used >= 8
    @test length(br.summary) == 4
    @test [r.param for r in br.summary] == [:sd_mu1, :sd_mu2, :sd_sigma1, :sd_sigma2]

    rows = Dict(r.param => r for r in br.summary)
    # point estimates are exactly the fitted among-axis SDs
    for (k, idx) in ((:sd_mu1, 1), (:sd_mu2, 2), (:sd_sigma1, 3), (:sd_sigma2, 4))
        @test rows[k].estimate ≈ sd_fit[idx] atol = 1e-8
    end
    # well-formed intervals
    for r in br.summary
        @test r.lower <= r.upper
        @test r.lower >= 0          # SD scale — nonnegative
    end

    # boundary-honest separation: the collapsed σ2 axis CI sits entirely below the
    # identified μ1 axis CI (no detectable scale-phylo signal vs a clear signal).
    @test rows[:sd_sigma2].upper < rows[:sd_mu1].lower
    @test rows[:sd_sigma2].lower < 0.3      # collapsed axis: lower end near 0

    # among-axis correlation CIs (coevolution_cor with uncertainty)
    @test length(br.cor_summary) == 6
    @test [r.param for r in br.cor_summary] == [:cor_mu1_mu2, :cor_mu1_sigma1,
        :cor_mu1_sigma2, :cor_mu2_sigma1, :cor_mu2_sigma2, :cor_sigma1_sigma2]
    crows = Dict(r.param => r for r in br.cor_summary)
    # point estimate matches coevolution_cor
    @test crows[:cor_mu1_mu2].estimate ≈ coevolution_cor(fit).cor[1, 2] atol = 1e-8
    for r in br.cor_summary
        @test -1.0 <= r.lower <= r.upper <= 1.0
        @test -1.0 <= r.estimate <= 1.0
    end
    # identified vs unidentified: a σ2-axis (collapsed) correlation has a wider CI
    # than the identified mu1–mu2 correlation (Ayumi's "the boundary leaks into the
    # correlation" — a collapsed axis's correlations ride toward ±1, unestimable).
    w(r) = r.upper - r.lower
    @test w(crows[:cor_mu1_sigma2]) > w(crows[:cor_mu1_mu2])

    # public bootstrap_result dispatches to this route for a bivariate q4 fit
    br2 = bootstrap_result(fit; data = dat, B = 6, tree = phy, rng = MersenneTwister(2))
    @test length(br2.summary) == 4
    @test [r.param for r in br2.summary] == [:sd_mu1, :sd_mu2, :sd_sigma1, :sd_sigma2]
end
