# test_twin_gap_639.jl — #639.
#
# Two bugs in the model-comparison guard, both flagged by adversarial review of
# the drmTMB port (drmTMB is GPL; this file targets DRModels.jl's own
# src/comparison.jl and copies no drmTMB source):
#
#   1. `_reml_compare_guard` only refused REML-vs-REML pairs with different mean
#      structure; it silently ACCEPTED a REML fit compared against an ML fit
#      whenever `_mean_structure` happened to match — but a REML log-likelihood
#      is never comparable to an ML log-likelihood, full stop (different
#      likelihoods). Now any pair with different `estim_method` is refused
#      outright, and a REML-vs-REML pair must additionally match fixed-effect
#      (MEAN) structure — not just `:mu`, but `:mu1`/`:mu2` too on a bivariate
#      fit.
#   2. `_mean_structure` (now `_fixed_effect_structure`) matched only the
#      univariate `:mu` block, so it silently returned `String[]` for every
#      bivariate fit (`:mu1`/`:mu2`, never `:mu`) — the REML guard then passed
#      ANY bivariate REML pair, mean-structure mismatch or not. It now covers
#      every MEAN block that REML actually restricts/marginalises
#      (`:mu`/`:mu1`/`:mu2`), so bivariate mean structure is detected.
#      (An earlier version of this fix widened the fingerprint to EVERY
#      non-variance-component block, i.e. also `:sigma1`/`:sigma2`/`:rho12` —
#      that over-refused: a dispersion (`:sigma`/`:sigma1`/`:sigma2`) or
#      correlation (`:rho12`) submodel is a nuisance parameter estimated
#      INSIDE the restricted likelihood, not something REML restricts away
#      (gaussian_bivariate.jl: REML "marginalises beta_mu1/beta_mu2 only"), so
#      comparing REML fits with the SAME mean design but a DIFFERENT sigma/
#      rho12 submodel is a valid, everyday REML comparison (e.g. testing
#      heteroscedasticity) — exactly what test_reml.jl's "model-selection
#      guard" testset requires `lrtest(full_reml, var_only_reml)` (same mu,
#      differing only in sigma) to accept. Reverted the fingerprint to
#      mean-only blocks; see src/comparison.jl's `_fixed_effect_structure`.)
#   3. The variance-component boundary label (`_boundary_vc_warn`, issue #304)
#      already keys off block symbols (`:phylocov`/`:recov`/…) that are shared
#      between univariate and bivariate fits, so it was already bivariate-safe;
#      pinned here end-to-end for a bivariate q=4 phylogenetic pair.
#
# Uses only Test macros (@test_throws / @test_logs) — no extra Logging
# dependency.
using DRModels
using Test, Random, LinearAlgebra, Statistics

@testset "lrtest: REML/ML method + bivariate structure guards (#639)" begin
    Random.seed!(20260927)
    n = 500
    x = randn(n)
    y = 0.5 .- 0.8 .* x .+ exp.(-0.3 .+ 0.4 .* x) .* randn(n)
    data = (; y, x)

    @testset "1a. REML vs ML refused, even with identical mean structure" begin
        ml_fit   = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian(); data = data)
        reml_fit = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian(); data = data, method = :REML)
        @test ml_fit.estim_method === :ML
        @test reml_fit.estim_method === :REML

        @test_throws ArgumentError lrtest(ml_fit, reml_fit)
        @test_throws ArgumentError lrtest(reml_fit, ml_fit)
        @test_throws ArgumentError anova(ml_fit, reml_fit)
        try
            lrtest(ml_fit, reml_fit)
            @test false   # unreachable — the call above must throw
        catch e
            @test e isa ArgumentError
            @test occursin("estim_method", e.msg)
        end
    end

    @testset "1b. REML vs REML with different mean structure refused" begin
        full_reml    = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian(); data = data, method = :REML)
        reduced_reml = drm(bf(@formula(y ~ 1),     @formula(sigma ~ 1)), Gaussian(); data = data, method = :REML)
        @test_throws ArgumentError lrtest(reduced_reml, full_reml)
    end

    @testset "1c. REML vs REML, same fixed effects, differ only in a random intercept: accepted" begin
        G = 25; m = 10; nn = G * m
        g = repeat(1:G, inner = m); xr = randn(nn)
        b = 0.6 .* randn(G)
        yr = 0.3 .- 0.5 .* xr .+ b[g] .+ 0.6 .* randn(nn)
        d = (; y = yr, x = xr, g)

        full_reml    = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = d, method = :REML)
        reduced_reml = drm(bf(@formula(y ~ x),           @formula(sigma ~ 1)), Gaussian(); data = d, method = :REML)
        @test full_reml.estim_method === :REML
        @test reduced_reml.estim_method === :REML

        t = lrtest(reduced_reml, full_reml)
        @test t.dof == dof(full_reml) - dof(reduced_reml)
        @test 0 <= t.pvalue <= 1
    end

    @testset "2. ML nested comparison still accepted" begin
        full    = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x)), Gaussian(); data = data)
        reduced = drm(bf(@formula(y ~ 1),     @formula(sigma ~ 1)),     Gaussian(); data = data)
        t = lrtest(reduced, full)
        @test t.statistic > 0
        @test t.dof == dof(full) - dof(reduced)
        @test 0 <= t.pvalue <= 1
    end

    @testset "3. Bivariate fixed-effect structure is detected (not vacuous)" begin
        n2 = 800
        xb = randn(n2)
        y1 = 0.3 .+ 0.5 .* xb .+ 0.3 .* randn(n2)
        y2 = -0.2 .+ 0.4 .* xb .+ 0.3 .* randn(n2)
        biv_data = (; y1, y2, x = xb)

        biv_full = drm(bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
                           sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                           rho12 = @formula(rho12 ~ 1)),
                       Gaussian(); data = biv_data)

        fx = DRModels._fixed_effect_structure(biv_full)
        blocks_seen = first.(fx)
        @test :mu1 in blocks_seen
        @test :mu2 in blocks_seen
        # `:sigma1`/`:sigma2`/`:rho12` are dispersion/correlation submodels, not
        # something REML restricts away, so they are deliberately NOT part of
        # this fingerprint (see the file header note above) — a same-mean,
        # different-sigma REML pair must remain a valid comparison.
        @test :sigma1 ∉ blocks_seen
        @test :sigma2 ∉ blocks_seen
        @test :rho12 ∉ blocks_seen
        # No mean block is silently empty (the pre-fix `_mean_structure` returned
        # `String[]` for every one of these on a bivariate fit).
        @test all(!isempty(nms) for (_, nms) in fx)

        @testset "3a. bivariate REML vs REML: mu1 mismatch refused" begin
            biv_full_reml = drm(bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
                                    sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                                    rho12 = @formula(rho12 ~ 1)),
                                Gaussian(); data = biv_data, method = :REML)
            biv_reduced_reml = drm(bf(mu1 = @formula(y1 ~ 1), mu2 = @formula(y2 ~ x),
                                       sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                                       rho12 = @formula(rho12 ~ 1)),
                                   Gaussian(); data = biv_data, method = :REML)
            @test_throws ArgumentError lrtest(biv_reduced_reml, biv_full_reml)
        end

        @testset "3b. bivariate REML vs REML: identical fixed-effect structure accepted" begin
            # Two independently-fit bivariate REML models with IDENTICAL fixed-effect
            # structure in every mean/scale block: the REML guard itself must accept
            # the pair (tested directly — `lrtest` end-to-end would additionally hit
            # the unrelated "dof(full) > dof(reduced)" nesting check, which is not
            # what this case is pinning).
            biv_full_reml = drm(bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
                                    sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                                    rho12 = @formula(rho12 ~ 1)),
                                Gaussian(); data = biv_data, method = :REML)
            biv_full_reml2 = drm(bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
                                     sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                                     rho12 = @formula(rho12 ~ 1)),
                                 Gaussian(); data = biv_data, method = :REML)
            @test DRModels._reml_compare_guard(biv_full_reml, biv_full_reml2, "test") === nothing
        end
    end

    @testset "4. Bivariate boundary variance-component label (issue #304 x #639)" begin
        # A tiny q=4 phylogenetic bivariate fit (`:phylocov` block) vs the
        # matching fixed-effects-only bivariate fit (no `:phylocov`): dropping
        # the phylogenetic random effect is a boundary (variance = 0) null, and
        # the warning must fire and name the bivariate `:phylocov` block, exactly
        # as it already does for a univariate random-effect drop (#304).
        rng = MersenneTwister(639)
        p = 6; nrep = 2
        phy = random_balanced_tree(p; branch_length = 0.2)
        keep = setdiff(1:phy.n_total, [phy.root_index])
        Q_cond = phy.Q_topology[keep, keep]
        Sigma_a = Matrix(Symmetric([
            0.20 0.05 0.02 0.00
            0.05 0.20 0.00 0.02
            0.02 0.00 0.10 0.01
            0.00 0.02 0.01 0.10
        ]))
        P = prior_precision(Q_cond, inv(Sigma_a))
        F = cholesky(Symmetric(P))
        u_aug = F.UP \ randn(rng, size(P, 1))
        pos = Dict(node => i for (i, node) in enumerate(keep))
        leaf_pos = [pos[phy.leaf_indices[k]] for k in 1:p]

        species_idx = repeat(1:p, inner = nrep)
        species = [phy.leaf_names[k] for k in species_idx]
        nobs_ = length(species_idx)
        xb = randn(rng, nobs_)
        y1 = Vector{Float64}(undef, nobs_)
        y2 = Vector{Float64}(undef, nobs_)
        for i in 1:nobs_
            k = species_idx[i]
            u = @view u_aug[(4 * (leaf_pos[k] - 1) + 1):(4 * leaf_pos[k])]
            m1 = 1.0 + 0.4 * xb[i] + u[1]
            m2 = -0.3 + 0.3 * xb[i] + u[2]
            s1 = exp(-0.4 + u[3])
            s2 = exp(-0.5 + u[4])
            e = cholesky(Symmetric([s1^2 0.0; 0.0 s2^2])).L * randn(rng, 2)
            y1[i] = m1 + e[1]
            y2[i] = m2 + e[2]
        end
        d = (; y1, y2, x = xb, species)

        full = drm(bf(mu1 = @formula(y1 ~ x + phylo(1 | species)),
                       mu2 = @formula(y2 ~ x + phylo(1 | species)),
                       sigma1 = @formula(sigma1 ~ 1 + phylo(1 | species)),
                       sigma2 = @formula(sigma2 ~ 1 + phylo(1 | species)),
                       rho12 = @formula(rho12 ~ 1)),
                   Gaussian(); data = d, tree = phy,
                   q4_iterations = 100, q4_n_newton = 25, q4_vcov = false)
        reduced = drm(bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
                          sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                          rho12 = @formula(rho12 ~ 1)),
                      Gaussian(); data = d)

        @test :phylocov in first.(full.blocks)
        @test !(:phylocov in first.(reduced.blocks))
        # Fixed-effect structure matches (same mu1/mu2/sigma1/sigma2/rho12 designs),
        # so the REML guard is not the thing under test here — both are ML, and
        # dropping :phylocov is a pure variance-component (boundary) comparison.
        @test DRModels._fixed_effect_structure(full) == DRModels._fixed_effect_structure(reduced)

        @test_logs (:warn,) match_mode = :any lrtest(reduced, full)
        @test_warn "phylocov" lrtest(reduced, full)
        @test_warn "lrt_boundary" lrtest(reduced, full)
    end
end
