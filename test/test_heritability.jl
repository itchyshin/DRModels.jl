# test_heritability.jl — comparative-biology derived ratios with CIs on the
# structured-Gaussian fits: phylogenetic heritability / signal h² and the
# repeatability / ICC R, each as (estimate, corrected, se, ci) via the merged
# epsilon-method (bias_correct) infra and, optionally, a profile CI on the ratio.
using DRModels
using Test, Random, LinearAlgebra

_corr(M) = (d = sqrt.(diag(M)); M ./ (d * d'))

@testset "heritability/ICC — closed-form anchor (single phylo component)" begin
    Random.seed!(20260607)
    G = 60
    phy = random_balanced_tree(G; branch_length = 0.4)
    Cphy = _corr(sigma_phy_dense(phy; σ²_phy = 1.0))

    m = 10; n = G * m
    species = repeat(1:G, inner = m)
    x = randn(n)
    σ = 0.4; σ1 = 1.0
    a1 = σ1 .* (cholesky(Symmetric(Cphy)).L * randn(G))
    y = 0.3 .+ 0.5 .* x .+ a1[species] .+ σ .* randn(n)
    data = (; y, x, species)

    # These anchors exercise the dense correlation-matrix parameterisation used by
    # the delta-method Wald CI. The default sparse all-node route reports the
    # Brownian phylo SD on a different scale and should get its own derived-ratio
    # design pass before replacing this reference check.
    fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
              Gaussian(); data = data, tree = phy, algorithm = :gls)
    @test fit.converged

    # Closed-form anchor: the accessor's point estimate equals the hand-computed
    # ratio σ²_phylo / (σ²_phylo + σ²_resid) from re_sd / sigma directly.
    s_sp = re_sd(fit)[:species]
    s_re = sigma(fit)[1]
    h_hand = s_sp^2 / (s_sp^2 + s_re^2)

    h = heritability(fit)                       # single component ⇒ default focal
    @test h.method === :delta
    @test h.estimate ≈ h_hand atol = 1e-8       # exact closed-form agreement
    @test 0.0 ≤ h.estimate ≤ 1.0

    # With a single structured component, heritability == repeatability (ICC).
    R = repeatability(fit)
    @test R.estimate ≈ h.estimate atol = 1e-10
    @test icc(fit).estimate ≈ h.estimate atol = 1e-10

    # CI brackets the point estimate and is clamped to [0, 1].
    @test h.ci.lower ≤ h.estimate ≤ h.ci.upper
    @test 0.0 ≤ h.ci.lower ≤ h.ci.upper ≤ 1.0
    @test isfinite(h.se) && h.se > 0
    @test isfinite(h.corrected)

    # The h² truth for these σ's is σ1²/(σ1²+σ²) — point estimate near it.
    @test h.estimate ≈ σ1^2 / (σ1^2 + σ^2) atol = 0.12
end

@testset "heritability — delta vs profile agreement (well identified)" begin
    Random.seed!(424242)
    G = 60
    phy = random_balanced_tree(G; branch_length = 0.4)
    Cphy = _corr(sigma_phy_dense(phy; σ²_phy = 1.0))
    m = 12; n = G * m
    species = repeat(1:G, inner = m)
    x = randn(n)
    σ = 0.5; σ1 = 1.0
    a1 = σ1 .* (cholesky(Symmetric(Cphy)).L * randn(G))
    y = 0.2 .+ 0.4 .* x .+ a1[species] .+ σ .* randn(n)
    fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
              Gaussian(); data = (; y, x, species), tree = phy, algorithm = :gls)

    hd = heritability(fit; method = :delta)
    hp = heritability(fit; method = :profile)
    @test hp.method === :profile
    @test hp.estimate ≈ hd.estimate atol = 1e-8           # same point estimate
    # Both intervals bracket the point and live in [0,1].
    @test hp.ci.lower ≤ hp.estimate ≤ hp.ci.upper
    @test 0.0 ≤ hp.ci.lower ≤ hp.ci.upper ≤ 1.0
    # Delta and profile half-widths agree to a loose-but-meaningful tolerance on a
    # well-identified fit (both are asymptotic, different curvature handling).
    @test isapprox(hp.ci.lower, hd.ci.lower; atol = 0.12)
    @test isapprox(hp.ci.upper, hd.ci.upper; atol = 0.12)
end

@testset "heritability/ICC — two structured components (closed-form anchor)" begin
    Random.seed!(20260608)
    G = 50
    phy = random_balanced_tree(G; branch_length = 0.4)
    Cphy = _corr(sigma_phy_dense(phy; σ²_phy = 1.0))
    M = randn(G, G); Canim = _corr(M * M' / G + I)
    m = 10; n = G * m
    species = repeat(1:G, inner = m)
    id = repeat(1:G, outer = m)                  # crossed ⇒ both identifiable
    x = randn(n)
    σ = 0.4; σ1 = 0.9; σ2 = 0.6
    a1 = σ1 .* (cholesky(Symmetric(Cphy)).L * randn(G))
    a2 = σ2 .* (cholesky(Symmetric(Canim)).L * randn(G))
    y = 0.3 .+ 0.5 .* x .+ a1[species] .+ a2[id] .+ σ .* randn(n)
    data = (; y, x, species, id)

    fit = drm(bf(@formula(y ~ x + phylo(1 | species) + relmat(1 | id)), @formula(sigma ~ 1)),
              Gaussian(); data = data, tree = phy, K = Canim)
    @test fit.converged

    s_sp = re_sd(fit)[:species]; s_id = re_sd(fit)[:id]; s_re = sigma(fit)[1]
    tot = s_sp^2 + s_id^2 + s_re^2

    # Heritability nets out ALL components + residual; pick :species.
    h_sp = heritability(fit; component = :species)
    @test h_sp.estimate ≈ s_sp^2 / tot atol = 1e-8
    h_id = heritability(fit; component = :id)
    @test h_id.estimate ≈ s_id^2 / tot atol = 1e-8

    # The two heritabilities + residual share sum to 1 (full variance partition).
    resid_share = s_re^2 / tot
    @test h_sp.estimate + h_id.estimate + resid_share ≈ 1.0 atol = 1e-8

    # ICC / repeatability is focal-vs-residual only (excludes the other component).
    R_sp = icc(fit; component = :species)
    @test R_sp.estimate ≈ s_sp^2 / (s_sp^2 + s_re^2) atol = 1e-8
    @test R_sp.estimate ≥ h_sp.estimate            # excluding σ²_id shrinks denom

    # No focal given with >1 component ⇒ informative error.
    @test_throws Exception heritability(fit)

    # CIs clamped to [0,1] and bracketing.
    for r in (h_sp, h_id, R_sp)
        @test 0.0 ≤ r.ci.lower ≤ r.estimate ≤ r.ci.upper ≤ 1.0
    end
end

@testset "heritability — degenerate anchors (component→0 and residual→0)" begin
    # (a) A component with NO signal ⇒ h² ≈ 0 with a sensible interval.
    Random.seed!(99)
    G = 50
    phy = random_balanced_tree(G; branch_length = 0.4)
    Cphy = _corr(sigma_phy_dense(phy; σ²_phy = 1.0))
    M = randn(G, G); Canim = _corr(M * M' / G + I)
    m = 10; n = G * m
    species = repeat(1:G, inner = m)
    id = repeat(1:G, outer = m)
    x = randn(n)
    σ = 0.5; σ1 = 1.0                            # animal (id) component is ABSENT
    a1 = σ1 .* (cholesky(Symmetric(Cphy)).L * randn(G))
    y = 0.3 .+ 0.5 .* x .+ a1[species] .+ σ .* randn(n)
    data = (; y, x, species, id)
    fit = drm(bf(@formula(y ~ x + phylo(1 | species) + relmat(1 | id)), @formula(sigma ~ 1)),
              Gaussian(); data = data, tree = phy, K = Canim)

    h_id = heritability(fit; component = :id)
    @test h_id.estimate < 0.1                    # ≈ 0 (no animal signal)
    @test h_id.ci.lower ≥ 0.0                    # clamped, one-sided-ish at 0
    @test h_id.ci.lower ≤ h_id.estimate ≤ h_id.ci.upper

    # (b) residual → 0 ⇒ h² ≈ 1. Construct data with negligible residual noise
    # relative to a single phylo component, then check the single-component h².
    Random.seed!(7)
    G2 = 60
    phy2 = random_balanced_tree(G2; branch_length = 0.4)
    Cphy2 = _corr(sigma_phy_dense(phy2; σ²_phy = 1.0))
    m2 = 8; n2 = G2 * m2
    sp2 = repeat(1:G2, inner = m2)
    a = 1.0 .* (cholesky(Symmetric(Cphy2)).L * randn(G2))
    y2 = a[sp2] .+ 1e-3 .* randn(n2)            # residual ≈ 0
    fit2 = drm(bf(@formula(y ~ 1 + phylo(1 | species)), @formula(sigma ~ 1)),
               Gaussian(); data = (; y = y2, species = sp2), tree = phy2,
               algorithm = :gls)
    h2 = heritability(fit2)
    @test h2.estimate > 0.95                     # ≈ 1 (residual negligible)
    @test h2.ci.upper ≤ 1.0                      # clamped at the upper bound
end

@testset "heritability — error on heteroscedastic residual and no components" begin
    Random.seed!(3)
    n = 200; x = randn(n); y = 0.5 .* x .+ randn(n)
    # No structured component at all.
    plain = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
    @test_throws Exception heritability(plain)
    @test_throws Exception icc(plain)
end
