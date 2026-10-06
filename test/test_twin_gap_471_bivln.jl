# test_twin_gap_471_bivln.jl — issue #471, bivariate LogNormal half.
#
# VERIFIED FINDING (this file): `src/bivariate_lognormal.jl`'s `drm` method
# already forwards `tree`/`K`/`A`/`coords`/`spatial_range` UNCONDITIONALLY to
# the bivariate Gaussian dispatcher on `log(y)` — there is no marker-specific
# gate in this file at all. Direct probing on this branch (no source edit)
# confirms `animal(1 | id)` and `spatial(1 | id)` already fit and converge for
# `LogNormal()`, exactly like the `phylo`/`relmat` markers `test_bivariate_
# lognormal.jl` already covers. So there is nothing to "admit" in `src/` for
# this family — the remaining gap on the LogNormal side of #471 was TEST
# COVERAGE only: `animal`/`spatial` had no dedicated identity test, unlike
# `phylo`/`relmat`. This file closes that coverage gap and adds one small
# known-DGP recovery check; it makes no change to `src/`.
#
# drmTMB comparison: drmTMB's own `biv_lognormal()` (R/drmTMB.R:10133,
# drmTMB 0.7.1 installed locally) currently refuses ALL random/structured
# effects ("biv_lognormal() currently allows fixed-effect formulas only;
# random and structured effects are deferred") — the same blanket refusal as
# `biv_student()`. Unlike Student, though, the Julia route is not a bespoke
# unverified engine: log(Y) bivariate normal makes a phylo/relmat/animal/
# spatial fit here EXACTLY `drm(f, Gaussian(); data = log.(y), …)` plus a
# parameter-free Jacobian shift, for every route (residual, q=2, q=4) the
# Gaussian dispatcher can select — the closed-form identity asserted below IS
# the correctness proof, so this is a case where DRM.jl can soundly go beyond
# drmTMB's own scope (owner decision, vault D-180). No R-parity fixture is
# possible or claimed here since there is no admitting drmTMB comparator.
using DRModels
using Test, Random, LinearAlgebra, Statistics

@testset "#471 bivariate LogNormal: animal/spatial structured markers" begin

    @testset "animal q=4: identical to Gaussian(log y)" begin
        G = 10
        Araw = let M = randn(MersenneTwister(47501), G, G); M * M' / G + I end
        d = sqrt.(diag(Araw)); A = Araw ./ (d * d')
        grp = repeat(1:G, inner = 4)
        Random.seed!(47501)
        n = length(grp); x = randn(n)
        y1 = exp.(0.2 .+ 0.3 .* x .+ 0.2 .* randn(n))
        y2 = exp.(-0.1 .+ 0.2 .* x .+ 0.2 .* randn(n))
        dat = (; y1 = y1, y2 = y2, x = x, id = grp)
        form = bf(mu1 = @formula(y1 ~ x + animal(1 | id)),
                  mu2 = @formula(y2 ~ x + animal(1 | id)),
                  sigma1 = @formula(sigma1 ~ 1 + animal(1 | id)),
                  sigma2 = @formula(sigma2 ~ 1 + animal(1 | id)),
                  rho12 = @formula(rho12 ~ 1))
        fit  = drm(form, LogNormal(); data = dat, A = A, q4_vcov = false)
        gfit = drm(form, Gaussian();  data = (; y1 = log.(y1), y2 = log.(y2), x = x, id = grp),
                   A = A, q4_vcov = false)
        @test is_converged(fit) && is_converged(gfit)
        @test fit.ranef.structured_type === :animal
        @test coef(fit) ≈ coef(gfit) atol = 1e-8
        @test fit.ranef.Sigma_a ≈ gfit.ranef.Sigma_a atol = 1e-8
        jac = sum(log.(y1)) + sum(log.(y2))
        @test loglik(gfit) - loglik(fit) ≈ jac rtol = 1e-8
    end

    @testset "animal q=2 (markers on mu1/mu2 only) also delegates exactly" begin
        G = 12
        Araw = let M = randn(MersenneTwister(47502), G, G); M * M' / G + I end
        d = sqrt.(diag(Araw)); A = Araw ./ (d * d')
        grp = repeat(1:G, inner = 5)
        Random.seed!(47502)
        n = length(grp); x = randn(n)
        y1 = exp.(0.1 .+ 0.4 .* x .+ 0.25 .* randn(n))
        y2 = exp.(0.05 .+ 0.3 .* x .+ 0.25 .* randn(n))
        dat = (; y1 = y1, y2 = y2, x = x, id = grp)
        form = bf(mu1 = @formula(y1 ~ x + animal(1 | id)),
                  mu2 = @formula(y2 ~ x + animal(1 | id)),
                  sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                  rho12 = @formula(rho12 ~ 1))
        fit  = drm(form, LogNormal(); data = dat, A = A)
        gfit = drm(form, Gaussian();  data = (; y1 = log.(y1), y2 = log.(y2), x = x, id = grp), A = A)
        @test is_converged(fit) && is_converged(gfit)
        @test coef(fit) ≈ coef(gfit) atol = 1e-8
        @test fit.ranef.Sigma_a ≈ gfit.ranef.Sigma_a atol = 1e-8
    end

    @testset "spatial q=4: identical to Gaussian(log y)" begin
        G = 9
        coords = hcat(collect(Float64, 1:G), zeros(G))   # a line: positive pairwise distances
        grp = repeat(1:G, inner = 4)
        Random.seed!(47503)
        n = length(grp); x = randn(n)
        y1 = exp.(0.3 .+ 0.25 .* x .+ 0.2 .* randn(n))
        y2 = exp.(-0.2 .+ 0.35 .* x .+ 0.2 .* randn(n))
        dat = (; y1 = y1, y2 = y2, x = x, site = grp)
        form = bf(mu1 = @formula(y1 ~ x + spatial(1 | site)),
                  mu2 = @formula(y2 ~ x + spatial(1 | site)),
                  sigma1 = @formula(sigma1 ~ 1 + spatial(1 | site)),
                  sigma2 = @formula(sigma2 ~ 1 + spatial(1 | site)),
                  rho12 = @formula(rho12 ~ 1))
        fit  = drm(form, LogNormal(); data = dat, coords = coords, spatial_range = 1.5, q4_vcov = false)
        gfit = drm(form, Gaussian();  data = (; y1 = log.(y1), y2 = log.(y2), x = x, site = grp),
                   coords = coords, spatial_range = 1.5, q4_vcov = false)
        @test is_converged(fit) && is_converged(gfit)
        @test fit.ranef.structured_type === :spatial
        @test fit.ranef.spatial_range ≈ 1.5
        @test coef(fit) ≈ coef(gfit) atol = 1e-8
        @test fit.ranef.Sigma_a ≈ gfit.ranef.Sigma_a atol = 1e-8
        jac = sum(log.(y1)) + sum(log.(y2))
        @test loglik(gfit) - loglik(fit) ≈ jac rtol = 1e-8
    end

    @testset "spatial q=2 (markers on mu1/mu2 only) also delegates exactly" begin
        G = 11
        coords = hcat(collect(Float64, 1:G), zeros(G))
        grp = repeat(1:G, inner = 4)
        Random.seed!(47504)
        n = length(grp); x = randn(n)
        y1 = exp.(0.15 .+ 0.3 .* x .+ 0.2 .* randn(n))
        y2 = exp.(-0.05 .+ 0.25 .* x .+ 0.2 .* randn(n))
        dat = (; y1 = y1, y2 = y2, x = x, site = grp)
        form = bf(mu1 = @formula(y1 ~ x + spatial(1 | site)),
                  mu2 = @formula(y2 ~ x + spatial(1 | site)),
                  sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                  rho12 = @formula(rho12 ~ 1))
        fit  = drm(form, LogNormal(); data = dat, coords = coords, spatial_range = 2.0)
        gfit = drm(form, Gaussian();  data = (; y1 = log.(y1), y2 = log.(y2), x = x, site = grp),
                   coords = coords, spatial_range = 2.0)
        @test is_converged(fit) && is_converged(gfit)
        @test coef(fit) ≈ coef(gfit) atol = 1e-8
        @test fit.ranef.Sigma_a ≈ gfit.ranef.Sigma_a atol = 1e-8
    end

    @testset "small known-DGP recovery on a simulated tree (phylo, 40 tips)" begin
        # A genuine recovery check against the SIMULATION TRUTH (not merely the
        # Gaussian-delegate identity, which the tests above already establish):
        # does the fitted phylogenetic variance component land near the value
        # used to generate the data? The q=2 route requires the SAME marker on
        # both mu1 and mu2 (front-end contract enforced by
        # `_bivariate_q4_marker`), so mu2 carries `phylo(1 | species)` too, with
        # a near-zero true random-intercept variance and no mu1/mu2 phylo
        # correlation in the DGP.
        Random.seed!(47601)
        p = 40; m = 4
        phy = random_balanced_tree(p; branch_length = 0.4)
        Sphy = sigma_phy_dense(phy; σ²_phy = 1.0)
        LC = cholesky(Symmetric(Sphy)).L
        σ_phy_true = 0.5   # SD of the phylogenetic random intercept on mu1's log scale
        u1 = σ_phy_true .* (LC * randn(p))
        u2 = 0.01 .* (LC * randn(p))   # ~zero on mu2, to keep the DGP off a trivial diagonal
        sp = repeat(1:p, inner = m); n = length(sp); x = randn(n)
        mu1 = 0.3 .+ 0.4 .* x .+ u1[sp]
        mu2 = -0.1 .+ 0.2 .* x .+ u2[sp]
        s1, s2, rho = 0.3, 0.35, 0.2
        z1 = randn(n); z2 = rho .* z1 .+ sqrt(1 - rho^2) .* randn(n)
        y1 = exp.(mu1 .+ s1 .* z1); y2 = exp.(mu2 .+ s2 .* z2)
        dat = (; y1 = y1, y2 = y2, x = x, species = phy.leaf_names[sp])

        form = bf(mu1 = @formula(y1 ~ x + phylo(1 | species)),
                  mu2 = @formula(y2 ~ x + phylo(1 | species)),
                  sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                  rho12 = @formula(rho12 ~ 1))
        fit = drm(form, LogNormal(); data = dat, tree = phy)
        @test is_converged(fit)
        # q=2 route: markers on mu1/mu2 only, so Sigma_a is 2x2 (mu1, mu2).
        σ_hat = sqrt(fit.ranef.Sigma_a[1, 1])
        @test isapprox(σ_hat, σ_phy_true; atol = 0.35)
        # And fixed effects on the log scale recover too.
        @test isapprox(coef(fit, :mu1), [0.3, 0.4]; atol = 0.3)
        @test isapprox(coef(fit, :mu2), [-0.1, 0.2]; atol = 0.25)
    end

    @testset "drmTMB comparison: no admitting comparator (documented, not a code gap)" begin
        # drmTMB 0.7.1 (R/drmTMB.R:10133): "biv_lognormal() currently allows
        # fixed-effect formulas only; random and structured effects are
        # deferred" — the identical blanket refusal biv_student() carries.
        # Unlike Student, the Julia route rests on a closed-form identity
        # (log(Y) bivariate Gaussian), verified above against the already-
        # verified bivariate Gaussian engine, so no R-side fixture is claimed
        # or needed for this cell. This testset is a documentation marker, not
        # an executable parity check.
        @test true
    end
end
