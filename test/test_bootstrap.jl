# Parametric bootstrap confidence intervals: simulate B replicates from the
# fitted model, refit each, take percentile intervals. Reuses simulate + drm.
using DRModels
using Test, Random

@testset "Parametric bootstrap CI" begin
    Random.seed!(11)
    n = 500
    x = randn(n)
    βμ = [0.4, 0.7]
    y = βμ[1] .+ βμ[2] .* x .+ exp.(-0.3 .+ 0.2 .* x) .* randn(n)
    form = bf(@formula(y ~ x), @formula(sigma ~ x))
    fit = drm(form, Gaussian(); data = (; y, x))

    bci = bootstrap_ci(form, Gaussian(); data = (; y, x), B = 300, level = 0.95, rng = MersenneTwister(1))
    @test length(bci) == 4
    @test all(r.lower < r.upper for r in bci)

    # the bootstrap intervals bracket the point estimates
    pe = coef(fit, :mu)
    mu = filter(r -> r.param === :mu, bci)
    @test mu[1].lower ≤ pe[1] ≤ mu[1].upper
    @test mu[2].lower ≤ pe[2] ≤ mu[2].upper

    # and the true slope is covered at this n
    @test mu[2].lower ≤ 0.7 ≤ mu[2].upper

    bs = bootstrap_summary(form, Gaussian(); data = (; y, x), B = 40,
        level = 0.95, rng = MersenneTwister(2))
    bci2 = bootstrap_ci(form, Gaussian(); data = (; y, x), B = 40,
        level = 0.95, rng = MersenneTwister(2))
    bs_fit = bootstrap_summary(fit; data = (; y, x), B = 40,
        level = 0.95, rng = MersenneTwister(2))
    bci_fit = bootstrap_ci(fit; data = (; y, x), B = 40,
        level = 0.95, rng = MersenneTwister(2))
    @test length(bs) == length(bci2) == 4
    @test all(r.std_error > 0 for r in bs)
    @test [(r.param, r.coef, r.estimate, r.lower, r.upper) for r in bs] ==
          [(r.param, r.coef, r.estimate, r.lower, r.upper) for r in bci2]
    @test bs_fit == bs
    @test bci_fit == bci2

    br = bootstrap_result(form, Gaussian(); data = (; y, x), B = 20,
        level = 0.95, rng = MersenneTwister(3))
    br_fit = bootstrap_result(fit; data = (; y, x), B = 20,
        level = 0.95, rng = MersenneTwister(3))
    bs2 = bootstrap_summary(form, Gaussian(); data = (; y, x), B = 20,
        level = 0.95, rng = MersenneTwister(3))
    @test br.attempted == 20
    @test br.used == 20
    @test br.failed == 0
    @test isempty(br.failures)
    @test length(br.seeds) == 20
    @test br.threaded == false
    @test br.julia_threads == Threads.nthreads()
    @test br.blas_threads >= 1
    @test br.worker_threads == 1
    @test br.blas_oversubscribed == false
    @test br.elapsed >= 0
    @test br.summary == bs2
    @test br_fit.summary == br.summary
    @test br_fit.seeds == br.seeds
    @test br_fit.attempted == br.attempted
    @test br_fit.used == br.used
    @test br_fit.failed == br.failed

    br_serial = bootstrap_result(fit; data = (; y, x), B = 12,
        level = 0.95, rng = MersenneTwister(8))
    br_threaded = bootstrap_result(fit; data = (; y, x), B = 12,
        level = 0.95, rng = MersenneTwister(8), threads = true)
    @test br_threaded.summary == br_serial.summary
    @test br_threaded.seeds == br_serial.seeds
    @test br_threaded.threaded == (Threads.nthreads() > 1)
    @test br_threaded.worker_threads == (Threads.nthreads() > 1 ? min(12, Threads.nthreads()) : 1)
    @test br_threaded.blas_oversubscribed == (br_threaded.threaded && br_threaded.blas_threads > 1)
    @test_throws ArgumentError bootstrap_result(form, Gaussian(); data = (; y, x),
        B = 2, failures = :warn)
    @test_throws ArgumentError bootstrap_result(DRModels._withformula(fit, nothing);
        data = (; y, x), B = 2)

    # Gaussian structured bootstrap refits should accept the same solver controls
    # as drm(...). This is the hook large phylogenetic bootstrap/profile workflows
    # need when :auto selects the sparse all-node route.
    Random.seed!(12)
    G = 16
    phy = random_balanced_tree(G; branch_length = 0.2)
    m = 2
    species = repeat(1:G, inner = m)
    x_phy = randn(G * m)
    y_phy = 0.2 .+ 0.5 .* x_phy .+ 0.3 .* randn(G * m)
    phy_form = bf(@formula(y_phy ~ x_phy + phylo(1 | species)),
                  @formula(sigma ~ 1))
    phy_data = (; y_phy, x_phy, species)
    phy_fit = drm(phy_form, Gaussian(); data = phy_data, tree = phy,
        algorithm = :em, g_tol = 1e-4)
    br_phy = bootstrap_result(phy_fit; data = phy_data, tree = phy, B = 2,
        rng = MersenneTwister(13), algorithm = :em, g_tol = 1e-4)
    @test br_phy.attempted == 2
    @test br_phy.used == 2
    @test br_phy.failed == 0
    @test length(br_phy.summary) == length(coef(phy_fit))
    br_phy_formula = bootstrap_result(phy_form, Gaussian(); data = phy_data,
        tree = phy, B = 1, rng = MersenneTwister(14),
        algorithm = :em, g_tol = 1e-4)
    @test br_phy_formula.used == 1

    fit0 = drm(form, Gaussian(); data = (; y, x))
    calls = Ref(0)
    function forced_refit(datab)
        calls[] += 1
        calls[] == 2 && error("forced refit failure")
        return drm(form, Gaussian(); data = datab)
    end
    forced = DRModels._bootstrap_result(fit0, form, (; y, x), 5, 0.95,
        MersenneTwister(4), false, forced_refit; failures = :skip)
    @test forced.attempted == 5
    @test forced.used == 4
    @test forced.failed == 1
    @test forced.failures[1].replicate == 2
    @test occursin("forced refit failure", forced.failures[1].message)
    @test length(forced.summary) == length(bs)

    calls[] = 0
    @test_throws ErrorException DRModels._bootstrap_result(fit0, form, (; y, x), 5,
        0.95, MersenneTwister(4), false, forced_refit; failures = :error)
end
