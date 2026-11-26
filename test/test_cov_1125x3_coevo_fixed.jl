# Round 3 coverage — src/coevo_accessors.jl and src/fixed_priors.jl branches.
using DRModels, Test, LinearAlgebra, Random

@testset "cov 1125x3 coevo accessors guards" begin
    dat = (; y = randn(12), x = randn(12), g = repeat(1:4, inner = 3))
    gfit = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = dat)
    for f in (coevolution_cor, coevolution_vc, coevolution_summary)
        err = try f(gfit); nothing catch e; e end
        @test err isa ErrorException
        @test occursin("coevolution accessors", err.msg)
    end
end

@testset "cov 1125x3 coevo accessors tiny phylo fit" begin
    Random.seed!(112506)
    p, m = 6, 3
    phy = random_balanced_tree(p; branch_length = 0.2)
    species = repeat(1:p, inner = m)
    n = length(species)
    x = randn(n)
    y1 = randn(n); y2 = randn(n)
    dat = (; y1, y2, x, species)
    form = bf(mu1 = @formula(y1 ~ x + phylo(1 | species)),
              mu2 = @formula(y2 ~ x + phylo(1 | species)),
              sigma1 = @formula(sigma1 ~ 1 + phylo(1 | species)),
              sigma2 = @formula(sigma2 ~ 1 + phylo(1 | species)),
              rho12 = @formula(rho12 ~ 1))
    fit = drm(form, Gaussian(); data = dat, tree = phy, q4_iterations = 60, q4_vcov = false)
    rc = coevolution_cor(fit)
    vc = coevolution_vc(fit)
    sm = coevolution_summary(fit)
    @test rc.axes == vc.axes == sm.axes
    @test size(rc.cor) == (4, 4)
    @test all(diag(rc.cor) .≈ 1.0)
    @test length(sm.pair) == 6
    @test sm.correlation ≈ [sm.cor[i, j] for (i, j) in ((1,2), (1,3), (1,4), (2,3), (2,4), (3,4))]
    @test vc.sd[:mu1] ≈ sqrt(vc.variance[:mu1]) atol = 1e-12
end

@testset "cov 1125x3 fixed_priors constructors" begin
    @test_throws ArgumentError drm_prior_sd_gamma(dpar = "sigma", group = "g", shape = 2, rate = 1)
    @test_throws ArgumentError drm_prior_sd_gamma(dpar = "mu", group = "", shape = 2, rate = 1)
    @test_throws ArgumentError drm_prior_sd_gamma(dpar = "mu", group = "g", coef = "x", shape = 2, rate = 1)
    @test_throws ArgumentError drm_prior_sd_gamma(dpar = "mu", group = "g", shape = 1.0, rate = 1)
    @test_throws ArgumentError drm_prior_sd_gamma(dpar = "mu", group = "g", shape = 2, rate = 0)
    @test_throws ArgumentError drm_prior_ridge(dpar = "mu", coef = "all", intercept = "maybe", lambda = 1)
    @test_throws ArgumentError drm_prior_normal(dpar = "mu", coef = "x", mean = 0, sd = 0)
    p = drm_prior_ridge(dpar = "mu", coef = "all", intercept = "exclude", lambda = 2.5)
    @test p.precision == 2.5
    @test p.intercept === :exclude
end

@testset "cov 1125x3 fixed_priors resolve_sd" begin
    priors = drm_priors(drm_prior_sd_gamma(dpar = "mu", group = "g1", shape = 2.2, rate = 1.5),
                        drm_prior_sd_gamma(dpar = "mu", group = "g2", shape = 2.5, rate = 2.0))
    groups = ["g1", "g2"]
    rows = DRModels._resolve_sd_priors(priors, groups, 5)
    @test length(rows) == 2
    @test Set(getproperty.(rows, :group)) == Set(["g1", "g2"])
    θ = zeros(7)
    θ[rows[1].index] = -0.5
    val = DRModels._sd_prior_value(rows[1], θ)
    @test isfinite(val)
    @test val == DRModels._sd_prior_value(rows[1], θ) # deterministic at fixed θ
    dup = drm_priors(drm_prior_sd_gamma(dpar = "mu", group = "g1", shape = 2, rate = 1),
                    drm_prior_sd_gamma(dpar = "mu", group = "g1", shape = 3, rate = 1))
    @test_throws ArgumentError DRModels._resolve_sd_priors(dup, groups, 1)
    badg = drm_priors(drm_prior_sd_gamma(dpar = "mu", group = "missing", shape = 2, rate = 1))
    @test_throws ArgumentError DRModels._resolve_sd_priors(badg, groups, 1)
    dat = (y = randn(20), x = randn(20), g1 = repeat(1:5, inner = 4), g2 = repeat(1:4, inner = 5))
    form = bf(@formula(y ~ x + (1 | g1) + (1 | g2)), @formula(sigma ~ 1))
    resolved = resolve_priors(form, Gaussian(); data = dat,
        priors = drm_priors(drm_prior_normal(dpar = "mu", coef = "x", mean = 0, sd = 2)))
    @test length(resolved) == 1
    @test_throws ArgumentError resolve_priors(form, Binomial(); data = (y = dat.y .> 0, x = dat.x),
        priors = drm_priors(drm_prior_sd_gamma(dpar = "mu", group = "g1", shape = 2, rate = 1)))
end