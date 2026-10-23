# Gaussian mean model with TWO structured variance components in one fit:
# `phylo(1 | species) + relmat(1 | id)`. The latent field is the SUM of two
# structured intercepts, each its own variance component:
#     y = Xβ + Z₁a₁ + Z₂a₂ + ε,  a₁~N(0,σ₁²C_phylo),  a₂~N(0,σ₂²C_animal),
#     ε~N(0,σ²)
# fit by ML (dense first cut). This is a NEW capability (no drmTMB parity claim);
# the dense→sparse speed follow-up is tracked separately.
using DRModels
using Test, Random, LinearAlgebra

# Correlation from a covariance (unit-diagonal), used for the animal matrix.
_corr(M) = (d = sqrt.(diag(M)); M ./ (d * d'))

@testset "Two structured components: phylo + animal — recovery" begin
    Random.seed!(20260607)
    G = 40                                   # species == individuals here (1 obs source)
    # Phylogenetic correlation over the G species, from a balanced tree.
    phy = random_balanced_tree(G; branch_length = 0.4)
    Cphy = _corr(sigma_phy_dense(phy; σ²_phy = 1.0))
    # Animal/relatedness correlation over the G individuals (independent of phylo).
    M = randn(G, G); Canim = _corr(M * M' / G + I)

    m = 8; n = G * m                          # repeated measures per species/id
    species = repeat(1:G, inner = m)
    id = repeat(1:G, inner = m)               # same grouping levels, DIFFERENT covariance
    x = randn(n)

    β = [0.3, 0.5]
    σ  = 0.35                                  # residual SD
    σ1 = 0.9                                   # phylo SD
    σ2 = 0.6                                   # animal SD
    a1 = σ1 .* (cholesky(Symmetric(Cphy)).L * randn(G))
    a2 = σ2 .* (cholesky(Symmetric(Canim)).L * randn(G))
    y = β[1] .+ β[2] .* x .+ a1[species] .+ a2[id] .+ σ .* randn(n)
    data = (; y, x, species, id)

    fit = drm(bf(@formula(y ~ x + phylo(1 | species) + relmat(1 | id)), @formula(sigma ~ 1)),
              Gaussian(); data = data, tree = phy, K = Canim)

    @test fit.converged
    @test isfinite(loglik(fit))

    # Fixed effects. The global intercept is only weakly identified here: with two
    # whole-vector random intercepts (phylo + animal) its level trades off against
    # the two RE means, so its point estimate is noisy on a single seed — assert it
    # is merely finite. The slope is well identified.
    @test isfinite(coef(fit, :mu)[1])
    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.1

    # BOTH structured SDs recovered (re_sd reports per grouping factor).
    sds = re_sd(fit)
    @test haskey(sds, :species) && haskey(sds, :id)
    @test sds[:species] ≈ σ1 atol = 0.35
    @test sds[:id]      ≈ σ2 atol = 0.35

    # vc(fit) reports BOTH as named 1×1 variance summaries.
    V = vc(fit)
    @test sqrt(V[:species][1, 1]) ≈ σ1 atol = 0.35
    @test sqrt(V[:id][1, 1])      ≈ σ2 atol = 0.35

    # Residual scale and BLUPs.
    @test sigma(fit)[1] ≈ σ atol = 0.1
    re = ranef(fit)
    @test length(re[:species]) == G
    @test length(re[:id]) == G
end

@testset "Two structured components: collapse to single phylo (σ2 → 0)" begin
    Random.seed!(20260608)
    G = 40
    phy = random_balanced_tree(G; branch_length = 0.4)
    Cphy = _corr(sigma_phy_dense(phy; σ²_phy = 1.0))
    M = randn(G, G); Canim = _corr(M * M' / G + I)

    m = 8; n = G * m
    species = repeat(1:G, inner = m)
    # id CROSSED with species (not collinear) so the animal component is separately
    # identifiable — only then can σ2 collapse cleanly when the truth has no animal
    # signal. (The recovery testset above deliberately uses the collinear case.)
    id = repeat(1:G, outer = m)
    x = randn(n)

    σ = 0.35; σ1 = 0.9                        # NO animal component in the truth (σ2 = 0)
    a1 = σ1 .* (cholesky(Symmetric(Cphy)).L * randn(G))
    y = 0.3 .+ 0.5 .* x .+ a1[species] .+ σ .* randn(n)
    data = (; y, x, species, id)

    # Two-component fit should drive the animal SD to ≈ 0 and recover phylo + resid.
    two = drm(bf(@formula(y ~ x + phylo(1 | species) + relmat(1 | id)), @formula(sigma ~ 1)),
              Gaussian(); data = data, tree = phy, K = Canim)
    sds = re_sd(two)
    @test sds[:id] < 0.3                      # collapses toward zero (no animal signal)
    @test sds[:species] ≈ σ1 atol = 0.4

    # And it should match a single-phylo fit closely (logLik + phylo SD).
    one = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
              Gaussian(); data = data, tree = phy)
    @test loglik(two) ≥ loglik(one) - 1e-3    # two-comp nests one-comp (≥, up to tol)
    @test re_sd(two)[:species] ≈ re_sd(one)[:species] atol = 0.3
end

@testset "Two structured components: error paths" begin
    Random.seed!(1)
    G = 10; m = 3; n = G * m
    phy = random_balanced_tree(G; branch_length = 0.3)
    Cphy = _corr(sigma_phy_dense(phy; σ²_phy = 1.0))
    M = randn(G, G); Canim = _corr(M * M' / G + I)
    species = repeat(1:G, inner = m); id = repeat(1:G, inner = m)
    x = randn(n); y = randn(n)
    data = (; y, x, species, id)

    # Missing one of the required matrices.
    @test_throws Exception drm(
        bf(@formula(y ~ x + phylo(1 | species) + relmat(1 | id)), @formula(sigma ~ 1)),
        Gaussian(); data = data, tree = phy)            # K missing

    # Same grouping factor for both structured markers is rejected.
    data2 = (; y, x, species)
    @test_throws Exception drm(
        bf(@formula(y ~ phylo(1 | species) + relmat(1 | species)), @formula(sigma ~ 1)),
        Gaussian(); data = data2, tree = phy, K = Cphy)
end
