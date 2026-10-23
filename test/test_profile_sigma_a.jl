# Profile-likelihood CIs for the q=4 phylogenetic among-axis SDs (Ayumi #2).
# The calibrated complement to bootstrap_sigma_a: needs no Hessian, respects the
# SD ≥ 0 boundary. On a deliberate σ2-axis collapse the profile lower bound is
# exactly 0 (the honest "no detectable scale-phylo signal" interval) — which a
# percentile bootstrap cannot produce — while identified axes get two-sided CIs.
using DRModels, Test, Random, LinearAlgebra

@testset "profile_sigma_a — q4 among-axis SD profile CIs (boundary lower = 0)" begin
    Random.seed!(20260613)
    p, m = 24, 5
    phy = random_balanced_tree(p; branch_length = 0.3)
    C = sigma_phy_dense(phy; σ²_phy = 1.0)
    LC = cholesky(Symmetric(C)).L
    Z = randn(p, 4)
    Lmu = cholesky([0.64 0.30; 0.30 0.64]).L
    U = hcat(LC * Z[:, 1:2] * Lmu', LC * Z[:, 3] * 0.5, zeros(p))   # σ2 collapses
    species = repeat(1:p, inner = m)
    n = length(species)
    x = randn(n)
    μ1 = 0.5 .+ 0.3 .* x .+ U[species, 1]
    μ2 = -0.2 .+ 0.4 .* x .+ U[species, 2]
    σ1 = exp.(-1.0 .+ U[species, 3])
    σ2 = exp.(-1.0 .+ U[species, 4])
    ρ = 0.3
    e1 = randn(n); e2 = ρ .* e1 .+ sqrt(1 - ρ^2) .* randn(n)
    dat = (; y1 = μ1 .+ σ1 .* e1, y2 = μ2 .+ σ2 .* e2, x, species)
    form = bf(mu1 = @formula(y1 ~ x + phylo(1 | species)),
              mu2 = @formula(y2 ~ x + phylo(1 | species)),
              sigma1 = @formula(sigma1 ~ 1 + phylo(1 | species)),
              sigma2 = @formula(sigma2 ~ 1 + phylo(1 | species)),
              rho12 = @formula(rho12 ~ 1))
    fit = drm(form, Gaussian(); data = dat, tree = phy)
    @test is_converged(fit)
    @test haskey(fit.ranef, :prob)              # the fit stashes the problem for profiling

    pr = profile_sigma_a(fit; level = 0.90)
    @test length(pr.summary) == 4
    @test [r.param for r in pr.summary] == [:sd_mu1, :sd_mu2, :sd_sigma1, :sd_sigma2]
    rows = Dict(r.param => r for r in pr.summary)

    sd_fit = sqrt.(max.(diag(vc(fit)[:species]), 0.0))
    for (k, idx) in ((:sd_mu1, 1), (:sd_mu2, 2), (:sd_sigma1, 3), (:sd_sigma2, 4))
        @test rows[k].estimate ≈ sd_fit[idx] atol = 1e-8
    end
    for r in pr.summary
        @test 0.0 <= r.lower <= r.upper          # ordered, nonnegative
        @test r.lower - 1e-6 <= r.estimate <= r.upper + 1e-6   # CI contains the point
    end

    # the headline: a COLLAPSED axis gets lower bound exactly 0 (boundary), an
    # IDENTIFIED axis gets a strictly-positive lower bound — the calibrated boundary
    # behaviour the bootstrap could not give.
    @test rows[:sd_sigma2].lower == 0.0
    @test rows[:sd_mu1].lower > 0.2
    # collapsed-axis interval sits below the identified axis interval
    @test rows[:sd_sigma2].upper < rows[:sd_mu1].lower

    # default (naive) threshold: a 0.90 CI uses χ²₁(0.90) ≈ 2.706
    @test isapprox(pr.threshold, 2.706, atol = 0.05)
end
