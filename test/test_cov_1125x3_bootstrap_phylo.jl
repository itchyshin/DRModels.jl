# Round 3 coverage — src/bootstrap_q4_phylo.jl (validation + small bootstrap).
using DRModels, Test, Random, LinearAlgebra

function _small_q4_fit()
    Random.seed!(112505)
    p, m = 8, 4
    phy = random_balanced_tree(p; branch_length = 0.25)
    species = repeat(1:p, inner = m)
    n = length(species)
    x = randn(n)
    dat = (; y1 = randn(n), y2 = randn(n), x, species)
    form = bf(mu1 = @formula(y1 ~ x + phylo(1 | species)),
              mu2 = @formula(y2 ~ x + phylo(1 | species)),
              sigma1 = @formula(sigma1 ~ 1 + phylo(1 | species)),
              sigma2 = @formula(sigma2 ~ 1 + phylo(1 | species)),
              rho12 = @formula(rho12 ~ 1))
    fit = drm(form, Gaussian(); data = dat, tree = phy, q4_iterations = 80, q4_vcov = false)
    return fit, dat
end

@testset "cov 1125x3 bootstrap_sigma_a validation" begin
    fit, dat = _small_q4_fit()
    plain = drm(bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
                   sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                   rho12 = @formula(rho12 ~ 1)),
                Gaussian(); data = dat)
    @test_throws ArgumentError bootstrap_sigma_a(plain; data = dat, B = 3)
    @test_throws ArgumentError bootstrap_sigma_a(fit; data = dat, B = 0)
    @test_throws ArgumentError bootstrap_sigma_a(fit; data = dat, B = -1)
    @test_throws ArgumentError bootstrap_sigma_a(fit; data = dat, B = 1.5)
    @test_throws ArgumentError bootstrap_sigma_a(fit; data = dat, B = 4, failures = :nope)
    @test_throws ArgumentError bootstrap_sigma_a(fit; data = dat, B = 4, level = 1.2)
    @test_throws ArgumentError bootstrap_sigma_a(fit; data = dat, B = 4, level = 0)
end

@testset "cov 1125x3 bootstrap_sigma_a small run" begin
    fit, dat = _small_q4_fit()
    br = bootstrap_sigma_a(fit; data = dat, B = 6, rng = MersenneTwister(9),
                           q4_iterations = 60, check_converged = false)
    @test br.attempted == 6
    @test br.used >= 1
    @test length(br.summary) == 4
    @test length(br.cor_summary) == 6
    @test br.axes == (:sd_mu1, :sd_mu2, :sd_sigma1, :sd_sigma2)
    @test br.prior_ridge >= 0
    for r in br.summary
        @test r.lower <= r.upper || isnan(r.lower)
        @test r.estimate >= 0
    end
    @test all(r -> isfinite(r.estimate) || isnan(r.estimate), br.cor_summary)
    br2 = bootstrap_sigma_a(fit; data = dat, B = 4, level = 0.90,
                            rng = MersenneTwister(11), q4_iterations = 50,
                            check_converged = false)
    @test br2.level == 0.90
    @test size(br2.draws, 2) == 4
    @test size(br2.cor_draws, 2) == 6
end