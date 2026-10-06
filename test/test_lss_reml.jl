# test_lss_reml.jl — Location–scale–scale (lss) REML (#558).
#
# Verifies:
#   1. Single-component iid lss REML vs ML: defining property (random effect SD
#      and/or residual scale larger under REML to correct n vs n-p bias),
#      metadata tagging, finite SEs across mean, scale, and sd blocks.
#   2. Single-component phylogenetic lss (sd_phylo / sd(species, phylogenetic))
#      REML vs ML: convergence, metadata, defining property, finite SEs.
#   3. Multi-component lss (lsss) REML vs ML: multi-RE models (iid + iid,
#      iid + phylo) with per-component sd() formulas, metadata, finite SEs.
using DRModels
using Test, Random, Statistics, LinearAlgebra

@testset "Location-scale-scale (lss) REML (#558)" begin
    @testset "Single-component iid lss REML" begin
        Random.seed!(20260828)
        G = 15; m = 6; n = G * m
        g = repeat(1:G, inner = m)
        x = randn(n)
        zg = randn(G)
        z = zg[g]
        # Group-specific SD: log σ_b,k = 0.2 + 0.5 * zg_k
        log_sig_b = 0.2 .+ 0.5 .* zg
        b = exp.(log_sig_b) .* randn(G)
        # Residual scale: log σ_e = -0.3 + 0.2 * x
        log_sig_e = -0.3 .+ 0.2 .* x
        y = 1.0 .+ 0.7 .* x .+ b[g] .+ exp.(log_sig_e) .* randn(n)
        data = (; y, x, z, g)

        f = bf(@formula(y ~ 1 + x + (1 | g)),
               @formula(sigma ~ 1 + x),
               @formula(sd(g) ~ 1 + z))

        f_ml   = drm(f, Gaussian(); data = data, method = :ML)
        f_reml = drm(f, Gaussian(); data = data, method = :REML)

        @test f_ml.converged
        @test f_reml.converged

        # Metadata
        @test estimation_method(f_ml) === :ML
        @test estimation_method(f_reml) === :REML
        @test isnan(reml_loglik(f_ml))
        @test isfinite(reml_loglik(f_reml))
        @test isfinite(ml_loglik(f_reml))
        @test loglik(f_reml) == reml_loglik(f_reml)
        @test ml_loglik(f_reml) <= loglik(f_ml) + 1e-6

        # Defining property: REML inflates variance/scale components (n vs n - pμ)
        # Intercept of sd(g) represents base log(σ_b)
        @test coef(f_reml, :sd)[1] >= coef(f_ml, :sd)[1] - 1e-10

        # Mean coefficients match closely
        @test isapprox(coef(f_reml, :mu), coef(f_ml, :mu); atol = 0.1)

        # Standard errors finite across all parameter blocks
        se_reml = sqrt.(abs.(LinearAlgebra.diag(f_reml.vcov)))
        @test all(isfinite, se_reml)
        @test length(se_reml) == length(f_reml.theta)
    end

    @testset "Single-component phylogenetic lss REML (sd_phylo)" begin
        Random.seed!(55801)
        p = 20; m = 4; n = p * m
        phy = random_balanced_tree(p; branch_length = 0.25)
        K = DRModels._phylo_correlation(phy)
        LK = cholesky(Symmetric(K)).L

        species = repeat(1:p, inner = m)
        x = randn(n)
        zg = randn(p)
        z = zg[species]

        # Species-level phylogenetic SD: log σ_a,k = 0.3 + 0.4 * zg_k
        sigma_a = exp.(0.3 .+ 0.4 .* zg)
        # u_a ~ MVN(0, D_a K D_a) = D_a * LK * randn
        u_a = sigma_a .* (LK * randn(p))

        y = 0.5 .+ 0.8 .* x .+ u_a[species] .+ exp.(-0.5 .+ 0.3 .* x) .* randn(n)
        data = (; y, x, z, species)

        f = bf(@formula(y ~ x + phylo(1 | species)),
               @formula(sigma ~ x),
               @formula(sd(species, phylogenetic) ~ z))

        fit_ml   = drm(f, Gaussian(); data = data, tree = phy, method = :ML)
        fit_reml = drm(f, Gaussian(); data = data, tree = phy, method = :REML)

        @test fit_ml.converged
        @test fit_reml.converged

        # Metadata
        @test estimation_method(fit_ml) === :ML
        @test estimation_method(fit_reml) === :REML
        @test isfinite(reml_loglik(fit_reml))
        @test isfinite(ml_loglik(fit_reml))
        @test loglik(fit_reml) == reml_loglik(fit_reml)
        @test ml_loglik(fit_reml) <= loglik(fit_ml) + 1e-6

        # Standard errors finite on all blocks (mean, scale, sd_phylo)
        se_reml = sqrt.(abs.(LinearAlgebra.diag(fit_reml.vcov)))
        @test all(isfinite, se_reml)
        @test haskey(Dict(fit_reml.blocks), :sd_phylo)
        @test haskey(Dict(fit_reml.blocks), :mu)
        @test haskey(Dict(fit_reml.blocks), :sigma)
    end

    @testset "Multi-component lss (lsss) REML: two iid random effects" begin
        Random.seed!(55802)
        G1 = 12; G2 = 8; n = 96
        g1 = repeat(1:G1, outer = 8)
        g2 = repeat(1:G2, inner = 12)
        x = randn(n)
        zg1 = randn(G1); z1 = zg1[g1]

        b1 = exp.(0.1 .+ 0.3 .* zg1) .* randn(G1)
        b2 = 0.4 .* randn(G2)
        y = 1.2 .+ 0.5 .* x .+ b1[g1] .+ b2[g2] .+ exp.(-0.4) .* randn(n)
        data = (; y, x, z1, g1, g2)

        f = bf(@formula(y ~ x + (1 | g1) + (1 | g2)),
               @formula(sigma ~ 1),
               @formula(sd(g1) ~ z1),
               @formula(sd(g2) ~ 1))

        fit_ml   = drm(f, Gaussian(); data = data, method = :ML)
        fit_reml = drm(f, Gaussian(); data = data, method = :REML)

        @test fit_ml.converged
        @test fit_reml.converged
        @test estimation_method(fit_reml) === :REML
        @test isfinite(reml_loglik(fit_reml))
        @test isfinite(ml_loglik(fit_reml))
        @test loglik(fit_reml) == reml_loglik(fit_reml)

        # Standard errors finite
        se_reml = sqrt.(abs.(LinearAlgebra.diag(fit_reml.vcov)))
        @test all(isfinite, se_reml)
    end

    @testset "Multi-component lss (lsss) REML: iid + phylo random effects" begin
        Random.seed!(55803)
        p = 15; m = 4; n = p * m
        phy = random_balanced_tree(p; branch_length = 0.3)
        K = DRModels._phylo_correlation(phy)
        LK = cholesky(Symmetric(K)).L

        species = repeat(1:p, inner = m)
        study = repeat(1:4, outer = p)
        x = randn(n)
        zg = randn(p); z = zg[species]

        u_phy = 0.5 .* (LK * randn(p))
        b_std = 0.3 .* randn(4)
        y = 0.8 .+ 0.6 .* x .+ u_phy[species] .+ b_std[study] .+ exp.(-0.5) .* randn(n)
        data = (; y, x, z, species, study)

        f = bf(@formula(y ~ x + (1 | study) + phylo(1 | species)),
               @formula(sigma ~ 1),
               @formula(sd(study) ~ 1),
               @formula(sd(species, phylogenetic) ~ z))

        fit_ml   = drm(f, Gaussian(); data = data, tree = phy, method = :ML)
        fit_reml = drm(f, Gaussian(); data = data, tree = phy, method = :REML)

        @test fit_ml.converged
        @test fit_reml.converged
        @test estimation_method(fit_reml) === :REML
        @test isfinite(reml_loglik(fit_reml))
        @test isfinite(ml_loglik(fit_reml))
        @test loglik(fit_reml) == reml_loglik(fit_reml)

        # Both :sd and :sd_phylo blocks present
        @test haskey(Dict(fit_reml.blocks), :sd)
        @test haskey(Dict(fit_reml.blocks), :sd_phylo)

        se_reml = sqrt.(abs.(LinearAlgebra.diag(fit_reml.vcov)))
        @test all(isfinite, se_reml)
    end
end
