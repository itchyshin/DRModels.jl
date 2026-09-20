# test_lognormal_structured_mean.jl — #563 S8: univariate LogNormal with a
# structured marker (`phylo`/`relmat`) on the MEAN. Mirrors the identity the
# BIVARIATE lognormal route already uses (src/bivariate_lognormal.jl): log(y) is
# exactly Gaussian, so a structured univariate lognormal fit is delegated
# WHOLESALE to `drm(f, Gaussian(); data = data-with-logged-response, tree = ...,
# K = ...)`, with the reported log-likelihood shifted by the parameter-free
# Jacobian `-sum(log y)`. theta/vcov/ranef and the fitted objective's gradient
# carry over untouched from the Gaussian-on-log(y) fit; only loglik (and
# everything derived from it: aic/bic/deviance) shifts. R evidence: drmTMB's
# `lognormal` family implements exactly `phylo`/`relmat` structured markers on
# the mean (session scratch lognormal-cells.md) — `animal`/`spatial` are not
# implemented on either side and stay refused here.
using DRModels
using Test, Random, LinearAlgebra, Statistics

@testset "#563 LogNormal structured mean (phylo/relmat)" begin

    @testset "phylo(1 | species): identity with Gaussian-on-log(y)" begin
        Random.seed!(20260901)
        G = 60
        phy = random_balanced_tree(G; branch_length = 0.3)
        C = sigma_phy_dense(phy; σ²_phy = 1.0)
        d = sqrt.(diag(C)); K = C ./ (d * d')
        m = 4; n = G * m
        species = repeat(1:G, inner = m)
        x = randn(n)
        β = [0.2, 0.4]; σ = 0.3; σphy = 0.6
        u = σphy .* (cholesky(Symmetric(K)).L * randn(G))
        logy = β[1] .+ β[2] .* x .+ u[species] .+ σ .* randn(n)
        y = exp.(logy)

        # Reference fit on the SAME logged copy the delegation builds, `log.(y)`,
        # not on the pre-`exp` truth `logy`: log∘exp is not the identity in
        # floating point (about 40% of entries come back one ulp off), so a
        # Gaussian fit on `logy` solves a 1-ulp-perturbed problem and LBFGS may
        # stop at a different iterate inside `g_tol = 1e-8`. That gap is not
        # reproducible across machines: 1.8e-15 on aarch64, 5.6e-9 on the CI x86
        # runner (PR #781, run 35526175941, Julia 1.10, ubuntu), so a 1e-10 pin
        # against `logy` flaked. Against the same `log.(y)` the delegation is a
        # plain carry-over of theta/vcov/ranef, so the identity is exact (`==`).
        fit_g = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
                    Gaussian(); data = (; y = log.(y), x, species), tree = phy)
        fit_ln = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
                     LogNormal(); data = (; y, x, species), tree = phy)

        sumlogy = sum(log, y)

        @test fit_ln.converged
        @test fit_ln.theta == fit_g.theta
        @test coef(fit_ln, :mu) == coef(fit_g, :mu)
        @test loglik(fit_ln) ≈ loglik(fit_g) - sumlogy atol = 1e-8
        @test vcov(fit_ln) ≈ vcov(fit_g) atol = 1e-6
        @test re_sd(fit_ln)[:species] == re_sd(fit_g)[:species]
        @test aic(fit_ln) ≈ aic(fit_g) + 2 * sumlogy atol = 1e-6
    end

    @testset "relmat(1 | id, K): identity with Gaussian-on-log(y)" begin
        Random.seed!(20260902)
        G = 40
        A = let M = randn(G, G); M * M' / G + I end
        d = sqrt.(diag(A)); K = A ./ (d * d')
        m = 5; n = G * m
        id = repeat(1:G, inner = m)
        x = randn(n)
        β = [0.3, 0.5]; σ = 0.4; σs = 0.5
        u = σs .* (cholesky(Symmetric(K)).L * randn(G))
        logy = β[1] .+ β[2] .* x .+ u[id] .+ σ .* randn(n)
        y = exp.(logy)

        # Same reference rule as the phylo testset above: log.(y), never logy.
        fit_g = drm(bf(@formula(y ~ x + relmat(1 | id)), @formula(sigma ~ 1)),
                    Gaussian(); data = (; y = log.(y), x, id), K = K)
        fit_ln = drm(bf(@formula(y ~ x + relmat(1 | id)), @formula(sigma ~ 1)),
                     LogNormal(); data = (; y, x, id), K = K)

        sumlogy = sum(log, y)

        @test fit_ln.converged
        @test fit_ln.theta == fit_g.theta
        @test coef(fit_ln, :mu) == coef(fit_g, :mu)
        @test loglik(fit_ln) ≈ loglik(fit_g) - sumlogy atol = 1e-8
        @test vcov(fit_ln) ≈ vcov(fit_g) atol = 1e-6
        @test re_sd(fit_ln)[:id] == re_sd(fit_g)[:id]
        @test aic(fit_ln) ≈ aic(fit_g) + 2 * sumlogy atol = 1e-6
    end

    @testset "refusal: structured marker on the SIGMA formula still errors" begin
        # Unchanged behaviour — only the mean formula may carry a random effect
        # or structured marker for LogNormal(); a marker on sigma is refused
        # exactly as before this change.
        Random.seed!(20260903)
        G = 10
        phy = random_balanced_tree(G; branch_length = 0.3)
        m = 4; n = G * m
        species = repeat(1:G, inner = m)
        x = randn(n)
        y = exp.(0.2 .+ 0.4 .* x .+ randn(n))
        err = nothing
        try
            drm(bf(@formula(y ~ x), @formula(sigma ~ phylo(1 | species))),
                LogNormal(); data = (; y, x, species), tree = phy)
        catch e
            err = e
        end
        @test err !== nothing
        @test occursin("only the mean formula may carry a random effect", sprint(showerror, err))
    end

    @testset "refusal: animal/spatial structured markers on the mean stay rejected" begin
        # drmTMB's `lognormal` family implements only phylo/relmat structured
        # markers; animal/spatial are NOT widened here without an R parity cell.
        Random.seed!(20260904)
        G = 10
        m = 4; n = G * m
        id = repeat(1:G, inner = m)
        x = randn(n)
        y = exp.(0.2 .+ 0.4 .* x .+ randn(n))
        err = nothing
        try
            drm(bf(@formula(y ~ x + animal(1 | id)), @formula(sigma ~ 1)),
                LogNormal(); data = (; y, x, id))
        catch e
            err = e
        end
        @test err !== nothing
    end
end
