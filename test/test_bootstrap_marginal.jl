# Regression: the parametric bootstrap must REDRAW the random effects (#459).
#
# `simulate(fit)` is a CONDITIONAL simulator -- it returns
# `fit.means[:mu] .+ sigma .* randn(n)`, and `fit.means[:mu]` already contains the
# fitted BLUPs. The bootstrap used it directly, so every replicate re-used the same
# realised random effects, the refitted variance component barely moved, and the
# percentile CI collapsed onto the point estimate.
#
# Measured before the fix: the phylo-SD bootstrap CI was 1674x NARROWER than native
# TMB on identical data, B and seed. Nothing failed, nothing warned, and the point
# estimate was correct -- which is exactly why it survived. These tests fail if the
# conditional simulator is ever restored.

using Test, DRModels, Random, Statistics, LinearAlgebra
import Distributions   # qualified: DRModels exports its own `Poisson` FAMILY
# This file needs no direct StatsModels import: `@formula` is re-exported by DRModels.
# Tests that use StatsModels types import them explicitly from the test project.

@testset "#459 parametric bootstrap redraws random effects" begin
    Random.seed!(20260607)
    G = 32
    phy = random_balanced_tree(G; branch_length = 0.3)
    Craw = sigma_phy_dense(phy; σ²_phy = 1.0)
    d = sqrt.(diag(Craw)); Kc = Craw ./ (d * d')
    m = 4; n = G * m
    species = repeat(1:G, inner = m)
    x = randn(n)
    σ = 0.4; σs = 0.9
    u = σs .* (cholesky(Symmetric(Kc)).L * randn(G))
    y = 0.2 .+ 0.5 .* x .+ u[species] .+ σ .* randn(n)
    dat = (; y, x, species)
    fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
              Gaussian(); data = dat, tree = phy)

    sim = DRModels._marginal_simulator(fit, dat; tree = phy)
    @test sim !== nothing

    rng = MersenneTwister(11)
    sims = [sim(rng) for _ in 1:60]

    # 1. The replicates must NOT sit at residual distance from the CONDITIONAL mean.
    #    If the random effects are not redrawn, sd(y* - conditional mean) collapses
    #    to exactly the residual sigma -- that single number is the whole bug.
    resid_sigma = exp(coef(fit, :sigma)[1])
    spread = mean(std(s .- fit.means[:mu]) for s in sims)
    @test spread > 1.5 * resid_sigma

    # 2. The between-group signal must vary ACROSS replicates. With the BLUPs held
    #    fixed this equals sigma/sqrt(m); a marginal draw is far larger.
    grpmean(s, g) = mean(s[species .== g])
    across = std([grpmean(s, 1) for s in sims])
    @test across > 2 * (resid_sigma / sqrt(m))

    # 3. The variance-component CI must be non-degenerate AND contain the estimate.
    #    The pre-fix interval failed both: it was ~1e-4 wide, and because the draws
    #    were shifted it did not even bracket the point estimate.
    res = bootstrap_result(fit; data = dat, tree = phy, B = 40, level = 0.95,
                           rng = MersenneTwister(20260824),
                           failures = :skip, check_converged = false)
    sd_rows = [r for r in res.summary if occursin("sd", lowercase(String(r.param)))]
    @test !isempty(sd_rows)
    row = first(sd_rows)
    @test row.upper > row.lower
    @test row.lower <= row.estimate <= row.upper
    # A degenerate interval was ~2.5e-4 wide on the log scale; a real one is ~0.4.
    @test (row.upper - row.lower) > 0.05
end

@testset "#459 marginal simulator uses the RAW phylo covariance" begin
    # `re_sd` for a phylo term is defined against `sigma_phy_dense(phy)`, whose
    # diagonal is the TREE HEIGHT -- not against the normalised correlation that
    # `_resolve_structured_matrix` returns. Drawing with the correlation matrix
    # under-disperses by sqrt(height), which is invisible on a height-1 tree and
    # badly wrong otherwise. This pins the scale on a tall tree, where the two
    # choices differ by more than a factor of two.
    Random.seed!(4242)
    G = 24
    phy = random_balanced_tree(G; branch_length = 1.0)   # tall: height >> 1
    Craw = sigma_phy_dense(phy; σ²_phy = 1.0)
    height = mean(diag(Craw))
    @test height > 2                                      # the trap is live here
    d = sqrt.(diag(Craw)); Kc = Craw ./ (d * d')
    m = 4; n = G * m
    species = repeat(1:G, inner = m); x = randn(n)
    u = 0.8 .* (cholesky(Symmetric(Kc)).L * randn(G))
    y = 0.2 .+ 0.5 .* x .+ u[species] .+ 0.4 .* randn(n)
    dat = (; y, x, species)
    fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
              Gaussian(); data = dat, tree = phy)

    # Round-trip: simulate at the FITTED sd and refit. The correct covariance
    # returns the fitted sd back (up to ML shrinkage); the correlation matrix
    # returns it shrunk by roughly sqrt(height).
    sdhat = re_sd(fit)[:species]
    sim = DRModels._marginal_simulator(fit, dat; tree = phy)
    rng = MersenneTwister(7)
    got = Float64[]
    for _ in 1:6
        ys = sim(rng)
        f2 = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
                 Gaussian(); data = (; y = ys, x, species), tree = phy)
        push!(got, re_sd(f2)[:species])
    end
    ratio = median(got) / sdhat
    # Correlation-scale draws would land near 1/sqrt(height) < 0.6 here.
    @test 0.75 < ratio < 1.3
end

@testset "#461 a degenerate optimum is not reported as converged -- converges under the #574 guard" begin
    # With ONE row per group a structured random effect can interpolate the data:
    # sigma collapses toward 0 and, under the SPARSE Woodbury objective, the
    # Gaussian log-likelihood used to run away to +Inf (`Optim.converged` returns
    # true there). #461's `is_converged` floor caught 25% of bootstrap replicates
    # post hoc (sd_phylo = 22980, sigma = 7.5e-15, loglik = 6.8e13). #574 closes
    # the true numerical hole -- a residual variance below machine precision
    # relative to the phylo variance is now refused INSIDE the objective, before
    # the optimizer can reach that spurious region -- so refits now converge to
    # the same (bounded) optimum a dense GLS oracle finds, and no replicate
    # should be dropped as degenerate on this fixture any more.
    Random.seed!(4242)
    G = 100
    phy = random_balanced_tree(G; branch_length = 0.3)
    C = sigma_phy_dense(phy; σ²_phy = 1.0)
    dd = sqrt.(diag(C)); Kc = C ./ (dd * dd')
    species = collect(1:G)                      # exactly one row per species
    x = randn(G)
    u = 0.9 .* (cholesky(Symmetric(Kc)).L * randn(G))
    y = 0.2 .+ 0.5 .* x .+ u .+ 0.4 .* randn(G)
    dat = (; y, x, species)
    fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
              Gaussian(); data = dat, tree = phy)

    # The observed fit itself is healthy and must NOT be rejected -- the guard has
    # to be specific to degeneracy, not merely to one-row-per-group designs.
    @test is_converged(fit)

    res = bootstrap_result(fit; data = dat, tree = phy, B = 60, level = 0.95,
                           rng = MersenneTwister(20260824),
                           failures = :skip, check_converged = true)
    # #574 fixed the numerical cause of the degeneracy (catastrophic
    # cancellation in the sparse Woodbury objective near sigma -> 0), so every
    # refit now lands on the true, bounded optimum: no replicate is degenerate.
    @test res.failed == 0
    @test res.used == 60
    sd_rows = [r for r in res.summary if occursin("sd", lowercase(String(r.param)))]
    @test !isempty(sd_rows)
    row = first(sd_rows)
    # Before #461's `is_converged` floor the upper percentile reached ~179 (and
    # 22980 in the raw draws) against an estimate near 1; #574 removes the
    # runaway at its numerical source, so the bound should be comfortably
    # sane rather than merely "not astronomical."
    @test isfinite(row.upper)
    @test row.upper < 20 * max(row.estimate, 1e-3)
end

@testset "#461 guard is Gaussian-only (a small dispersion is not degeneracy)" begin
    # For NB2/Beta/Gamma the `:sigma` slot holds a dispersion or shape, where a
    # genuinely small value is legitimate. The guard must not reject those.
    Random.seed!(77)
    n = 200; x = randn(n)
    y = Float64[rand(Distributions.Poisson(exp(0.6 + 0.4 * xi))) for xi in x]
    f = drm(bf(@formula(y ~ 1 + x)), Poisson(); data = (; y, x))
    @test is_converged(f) == f.converged      # non-Gaussian: guard is a no-op
end

@testset "#459 the marginal simulator covers ordinary (1|g) and relmat, not only phylo" begin
    # The first version of the fix handled ONLY structured phylo and returned
    # `nothing` for everything else -- which meant the bootstrap silently fell back
    # to the conditional simulator, i.e. straight back to the bug, for ordinary
    # `(1 | g)` and for `relmat`/`animal`. Silent fallback is the failure mode this
    # whole slice exists to remove, so both routes are pinned here.
    #
    # These also pin the reason the fix needed `predict`. Whether `fit.means[:mu]`
    # is conditional or marginal varies BY ROUTE and does not track whether BLUPs
    # exist (measured: phylo = BLUPs + conditional; relmat = no BLUPs + marginal;
    # ordinary = BLUPs + marginal). Both plausible rules were tried and both were
    # wrong -- the ordinary route came back at ratio 1.46, the sqrt(2) signature of
    # counting the random effect twice.

    roundtrip(sim, refit, grp, sdhat, reps = 6) = begin
        rng = MersenneTwister(3)
        median([re_sd(refit(sim(rng)))[grp] for _ in 1:reps]) / sdhat
    end

    # --- ordinary (1 | g): independent intercepts, covariance = I
    Random.seed!(31)
    G = 40; m = 5; n = G * m
    g = repeat(1:G, inner = m); x = randn(n)
    y = 0.2 .+ 0.5 .* x .+ (0.9 .* randn(G))[g] .+ 0.4 .* randn(n)
    d = (; y, x, g)
    f1 = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = d)
    s1 = DRModels._marginal_simulator(f1, d)
    @test s1 !== nothing
    r1 = roundtrip(s1,
                   ys -> drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)),
                             Gaussian(); data = (; y = ys, x, g)),
                   :g, re_sd(f1)[:g])
    @test 0.75 < r1 < 1.3

    # --- relmat(1 | id): user-supplied K, used as given
    Random.seed!(909)
    G3 = 40; m3 = 4; n3 = G3 * m3
    A = randn(G3, 8); K3 = A * A'
    dd = sqrt.(diag(K3)); K3 = K3 ./ (dd * dd'); K3 += 1e-6I
    id = repeat(1:G3, inner = m3); x3 = randn(n3)
    u3 = 0.8 .* (cholesky(Symmetric(K3)).L * randn(G3))
    y3 = 0.2 .+ 0.5 .* x3 .+ u3[id] .+ 0.4 .* randn(n3)
    d3 = (; y = y3, x = x3, id)
    f3 = drm(bf(@formula(y ~ x + relmat(1 | id)), @formula(sigma ~ 1)),
             Gaussian(); data = d3, K = K3)
    s3 = DRModels._marginal_simulator(f3, d3; K = K3)
    @test s3 !== nothing
    r3 = roundtrip(s3,
                   ys -> drm(bf(@formula(y ~ x + relmat(1 | id)), @formula(sigma ~ 1)),
                             Gaussian(); data = (; y = ys, x = x3, id), K = K3),
                   :id, re_sd(f3)[:id])
    @test 0.75 < r3 < 1.3
end

@testset "#462 the marginal simulator covers non-Gaussian random-effect fits" begin
    # The Gaussian branch adds the random effect on the RESPONSE scale because
    # identity is the link. Every other family carries it on the LINK scale, so it
    # has to pass through the inverse link before the family draw. Until this was
    # added, `_marginal_simulator` returned `nothing` for non-Gaussian fits, which
    # meant the bootstrap silently fell back to the conditional simulator.
    #
    # Measured before the fix, Poisson (1|g): replicates carried a between-group SD
    # of 0.696 against 2.690 in the observed data -- about a quarter of the real
    # group structure.
    rt(mk, d, sim, reps = 5) = begin
        rng = MersenneTwister(9)
        got = Float64[]
        for _ in 1:reps
            try push!(got, re_sd(mk((; y = sim(rng), x = d.x, g = d.g)))[:g]) catch end
        end
        isempty(got) ? NaN : median(got) / re_sd(mk(d))[:g]
    end

    Random.seed!(5)
    G = 30; m = 6; n = G * m
    g = repeat(1:G, inner = m); x = randn(n)
    u = 0.8 .* randn(G)

    # --- Poisson
    yp = Float64[rand(Distributions.Poisson(exp(0.5 + 0.4 * x[i] + u[g[i]]))) for i in 1:n]
    dp = (; y = yp, x, g)
    mkp = dd -> drm(bf(@formula(y ~ 1 + x + (1 | g))), Poisson(); data = dd)
    fp = mkp(dp)
    sp = DRModels._marginal_simulator(fp, dp)
    @test sp !== nothing
    # The replicates must carry the group structure, not a quarter of it.
    grpsd(v) = std([mean(v[g .== k]) for k in 1:G])
    rng = MersenneTwister(2)
    sim_bgsd = mean(grpsd(sp(rng)) for _ in 1:20)
    @test sim_bgsd > 0.5 * grpsd(yp)
    @test 0.7 < rt(mkp, dp, DRModels._marginal_simulator(fp, dp)) < 1.4

    # --- Binomial (a different link, so the inverse-link step is genuinely exercised)
    yb = Float64[rand(Distributions.Bernoulli(1 / (1 + exp(-(0.2 + 0.6 * x[i] + u[g[i]]))))) for i in 1:n]
    db = (; y = yb, x, g)
    mkb = dd -> drm(bf(@formula(y ~ 1 + x + (1 | g))), Binomial(); data = dd)
    fb = mkb(db)
    sb = DRModels._marginal_simulator(fb, db)
    @test sb !== nothing
    @test 0.7 < rt(mkb, db, DRModels._marginal_simulator(fb, db)) < 1.4
end
