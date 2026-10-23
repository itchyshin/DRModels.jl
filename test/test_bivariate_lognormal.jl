# Bivariate lognormal residual-correlation model — drmTMB's `biv_lognormal()`.
#
# The likelihood is closed form: log(Y) is bivariate normal, so
# log f_Y(y) = log phi_2(log y) - sum(log y). The Jacobian is parameter-free, so
# the MLE and its covariance must be IDENTICAL to the bivariate Gaussian fit on
# log(y) and only the likelihood value may shift. Both identities are asserted
# below — they are what makes delegating to the verified Gaussian kernel correct
# rather than merely convenient.

using DRModels
using Test
using Random
using LinearAlgebra

@testset "bivariate lognormal (drmTMB biv_lognormal)" begin

    rng = MersenneTwister(11)
    n = 500
    x = randn(rng, n)
    s1, s2, rho = 0.5, 0.8, 0.6
    z1 = randn(rng, n)
    z2 = rho .* z1 .+ sqrt(1 - rho^2) .* randn(rng, n)
    y1 = exp.(0.4 .+ 0.9 .* x .+ s1 .* z1)
    y2 = exp.(-0.2 .+ 0.5 .* x .+ s2 .* z2)
    data = (; y1 = y1, y2 = y2, x = x)
    f = bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
           sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
           rho12 = @formula(rho12 ~ 1))

    fit = drm(f, LogNormal(); data = data)

    @testset "parameter recovery on the log scale" begin
        @test is_converged(fit)
        est = coef(fit)
        truth = [0.4, 0.9, -0.2, 0.5, log(s1), log(s2), atanh(rho)]
        @test length(est) == 7
        @test isapprox(est, truth; atol = 0.12)
        # rho12 is reported on the correlation scale, guarded off ±1
        r̂ = first(values(corpairs(fit)))[1]
        @test isapprox(r̂, rho; atol = 0.08)
        @test -1 < r̂ < 1
    end

    @testset "closed-form identities vs the Gaussian kernel on log(y)" begin
        gfit = drm(f, Gaussian(); data = (; y1 = log.(y1), y2 = log.(y2), x = x))
        # The Jacobian does not depend on any parameter, so the optimum is the same.
        @test coef(fit) ≈ coef(gfit) atol = 1e-10
        @test vcov(fit) ≈ vcov(gfit) atol = 1e-10
        # ... and the likelihood differs by exactly that Jacobian.
        jac = sum(log.(y1)) + sum(log.(y2))
        @test loglik(gfit) - loglik(fit) ≈ jac rtol = 1e-10
        # aic/bic inherit the shift (same k, same n)
        # AIC = 2k - 2*loglik, and loglik(fit) = loglik(gfit) - jac, so the
        # lognormal AIC is LARGER by exactly 2*jac.
        @test aic(fit) - aic(gfit) ≈ 2 * jac rtol = 1e-10
    end

    @testset "the response must be strictly positive" begin
        bad = (; y1 = copy(y1), y2 = copy(y2), x = x)
        bad.y1[3] = -1.0
        @test_throws ArgumentError drm(f, LogNormal(); data = bad)
        zero_y = (; y1 = copy(y1), y2 = copy(y2), x = x)
        zero_y.y2[7] = 0.0
        @test_throws ArgumentError drm(f, LogNormal(); data = zero_y)
    end

    @testset "first-slice boundary is enforced, matching drmTMB" begin
        # No REML: the residual-only route has no random effects to integrate out.
        @test_throws ArgumentError drm(f, LogNormal(); data = data, method = :REML)
    end

    @testset "bridge routes the drmTMB family tag" begin
        b = drm_bridge(; formula = Dict(:mu1 => "y1 ~ x", :mu2 => "y2 ~ x",
                                        :sigma1 => "sigma1 ~ 1", :sigma2 => "sigma2 ~ 1",
                                        :rho12 => "rho12 ~ 1"),
                       family = "biv_lognormal", data = data)
        @test b["converged"] == true
        @test b["coefficients"] ≈ coef(fit)
        @test b["loglik"] ≈ loglik(fit)
        # the post-fit contract from A2a still holds for this family
        dp = b["dpars"]
        @test Set(keys(dp)) == Set(["mu1", "mu2", "sigma1", "sigma2", "rho12"])
        @test all(length(v) == n for v in values(dp))
        @test all(dp["sigma1"] .> 0) && all(dp["sigma2"] .> 0)
        @test all(-1 .< dp["rho12"] .< 1)
    end

    # --- structured markers (issue #471) --------------------------------------
    #
    # log(Y) bivariate Gaussian means a structured (phylo/relmat/animal/spatial)
    # fit here is LITERALLY `drm(f, Gaussian(); data = log.(data), …)` plus the
    # same parameter-free Jacobian shift as the residual route above — there is
    # no separate engine, so the identity check IS the correctness proof.
    #
    # THE TRAP: a phylo term's group-level covariance is defined against the RAW
    # covariance `sigma_phy_dense(phy)` (diagonal = tree height), NOT the
    # normalised correlation matrix. Using the wrong one under-disperses by
    # sqrt(height) — invisible on a height-1 tree, which is exactly what a
    # hand-written fixture usually is. Every height below is deliberately != 1.
    function _lognormal_q4_sim(height_bl::Real, seed::Int)
        Random.seed!(seed)
        p = 16; m = 5
        phy = random_balanced_tree(p; branch_length = height_bl)
        Sphy = sigma_phy_dense(phy; σ²_phy = 1.0)
        LC = cholesky(Symmetric(Sphy)).L
        Λ_among = [0.05 0.01 0.005 0.002;
                   0.01 0.05 0.002 0.005;
                   0.005 0.002 0.02 0.001;
                   0.002 0.005 0.001 0.02]
        U = LC * randn(p, 4) * cholesky(Symmetric(Λ_among)).L
        sp = repeat(1:p, inner = m); n = length(sp); x = randn(n)
        mu1 = 0.4 .+ 0.3 .* x .+ U[sp, 1]
        mu2 = -0.1 .+ 0.2 .* x .+ U[sp, 2]
        s1 = exp.(-0.8 .+ U[sp, 3]); s2 = exp.(-0.7 .+ U[sp, 4])
        rho = 0.3
        z1 = randn(n); z2 = rho .* z1 .+ sqrt(1 - rho^2) .* randn(n)
        y1 = exp.(mu1 .+ s1 .* z1); y2 = exp.(mu2 .+ s2 .* z2)
        dat = (; y1 = y1, y2 = y2, x = x, species = phy.leaf_names[sp])
        return phy, dat
    end

    form_q4 = bf(mu1    = @formula(y1 ~ x + phylo(1 | species)),
                 mu2    = @formula(y2 ~ x + phylo(1 | species)),
                 sigma1 = @formula(sigma1 ~ 1 + phylo(1 | species)),
                 sigma2 = @formula(sigma2 ~ 1 + phylo(1 | species)),
                 rho12  = @formula(rho12 ~ 1))
    form_q2 = bf(mu1 = @formula(y1 ~ x + phylo(1 | species)),
                 mu2 = @formula(y2 ~ x + phylo(1 | species)),
                 sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                 rho12 = @formula(rho12 ~ 1))

    @testset "q=4 phylo: identical to Gaussian(log y) across THREE tree heights" begin
        for (bl, seed) in ((0.125, 47101), (0.3, 47102), (0.75, 47103))
            phy, dat = _lognormal_q4_sim(bl, seed)
            h = phylo_tree_height(phy)
            @test !isapprox(h, 1.0; atol = 1e-6)   # never test only the invisible height-1 case

            fit  = drm(form_q4, LogNormal(); data = dat, tree = phy, q4_vcov = false)
            gfit = drm(form_q4, Gaussian();  data = (; y1 = log.(dat.y1), y2 = log.(dat.y2),
                                                       x = dat.x, species = dat.species),
                       tree = phy, q4_vcov = false)
            @test is_converged(fit) && is_converged(gfit)
            # Same theta, same group-level Sigma_a — the scale trap would show up
            # here as a height-dependent MISMATCH between the two, not a mismatch
            # to the simulation truth (which is a separate, noisier question).
            @test coef(fit) ≈ coef(gfit) atol = 1e-8
            @test fit.ranef.Sigma_a ≈ gfit.ranef.Sigma_a atol = 1e-8
            jac = sum(log.(dat.y1)) + sum(log.(dat.y2))
            @test loglik(gfit) - loglik(fit) ≈ jac rtol = 1e-8
        end
    end

    @testset "q=2 phylo (markers on mu1/mu2 only) also delegates exactly" begin
        for (bl, seed) in ((0.2, 47201), (1.1, 47202))
            phy, dat = _lognormal_q4_sim(bl, seed)
            fit  = drm(form_q2, LogNormal(); data = dat, tree = phy)
            gfit = drm(form_q2, Gaussian();  data = (; y1 = log.(dat.y1), y2 = log.(dat.y2),
                                                       x = dat.x, species = dat.species),
                       tree = phy)
            @test is_converged(fit) && is_converged(gfit)
            @test coef(fit) ≈ coef(gfit) atol = 1e-8
            @test fit.ranef.Sigma_a ≈ gfit.ranef.Sigma_a atol = 1e-8
        end
    end

    @testset "q=4 relmat: same delegation identity off the phylogenetic route" begin
        G = 10
        K = Matrix{Float64}(I, G, G)
        for i in 1:G, j in 1:G
            i != j && (K[i, j] = 0.15)
        end
        grp = repeat(1:G, inner = 4)
        Random.seed!(47301)
        n = length(grp); x = randn(n)
        y1 = exp.(0.2 .+ 0.3 .* x .+ 0.2 .* randn(n))
        y2 = exp.(-0.1 .+ 0.2 .* x .+ 0.2 .* randn(n))
        dat = (; y1 = y1, y2 = y2, x = x, g = grp)
        form = bf(mu1 = @formula(y1 ~ x + relmat(1 | g)),
                  mu2 = @formula(y2 ~ x + relmat(1 | g)),
                  sigma1 = @formula(sigma1 ~ 1 + relmat(1 | g)),
                  sigma2 = @formula(sigma2 ~ 1 + relmat(1 | g)),
                  rho12 = @formula(rho12 ~ 1))
        fit  = drm(form, LogNormal(); data = dat, K = K, q4_vcov = false)
        gfit = drm(form, Gaussian();  data = (; y1 = log.(y1), y2 = log.(y2), x = x, g = grp),
                   K = K, q4_vcov = false)
        @test is_converged(fit) && is_converged(gfit)
        @test coef(fit) ≈ coef(gfit) atol = 1e-8
        @test fit.ranef.Sigma_a ≈ gfit.ranef.Sigma_a atol = 1e-8
    end

    @testset "REML stays rejected — structured markers do not lift the ML-only bound" begin
        phy, dat = _lognormal_q4_sim(0.3, 47401)
        @test_throws ArgumentError drm(form_q4, LogNormal(); data = dat, tree = phy, method = :REML)
    end

    @testset "front-end errors (missing tree, bad marker combo) still surface" begin
        phy, dat = _lognormal_q4_sim(0.3, 47402)
        @test_throws ErrorException drm(form_q4, LogNormal(); data = dat)   # no tree = …
        bad = bf(mu1 = @formula(y1 ~ x + phylo(1 | species)),
                  mu2 = @formula(y2 ~ x), sigma1 = @formula(sigma1 ~ 1),
                  sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1))
        @test_throws ErrorException drm(bad, LogNormal(); data = dat, tree = phy)
    end
end
