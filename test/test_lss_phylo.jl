# #545 — sd_phylo(species) ~ x: climate-dependent PHYLOGENETIC SD, the scale
# component of the Mizuno et al. QQQ model (V = D_a A D_a + D_e²).
# Plus the #548 regression: a structured mean intercept with a heteroscedastic
# residual used to return logLik +1.1e105 with converged = true.
#
# CROSS-ENGINE EVIDENCE (2026-08-28, identical CSV + Newick, 64-tip tree):
#   model                    DRModels.jl                drmTMB
#   sd_phylo ~ x   logLik    -69.13730392152723    -69.1373
#                  mu        0.9944436, 0.4478094  0.9944436, 0.4478094
#                  sigma    -1.2629298, 0.6845002 -1.26293,   0.6845002
#                  sd_phylo -0.1869284,-0.1827045 -0.1869285, -0.1827046
#   scalar (M5)    logLik    -70.38332423766208    -70.38332
using Test
using DRModels
using StableRNGs
using LinearAlgebra

function _balanced_newick(d)
    function node(prefix, depth)
        depth == 0 && return "$(prefix):$(1/d)"
        l = node(prefix * "a", depth - 1); r = node(prefix * "b", depth - 1)
        return depth == d ? "($l,$r);" : "($l,$r):$(1/d)"
    end
    node("t", d)
end

function _qqq_sim(; depth = 6)
    phy = DRModels.augmented_phy(_balanced_newick(depth))
    G = phy.n_leaves
    K0 = DRModels.sigma_phy_dense(phy; σ²_phy = 1.0)
    dK = sqrt.(diag(K0)); K = K0 ./ (dK * dK')
    rng = StableRNG(11)
    x = randn(rng, G)
    sp = String.(phy.leaf_names)          # REAL tip labels: a placeholder name
                                          # matches no leaf and the fit is refused
    sda = exp.(-0.5 .+ 0.4 .* x)
    sde = exp.(-1.0 .- 0.3 .* x)
    a = Diagonal(sda) * (cholesky(Symmetric(K)).L * randn(rng, G))
    y = 1.0 .+ 0.5 .* x .+ a .+ sde .* randn(rng, G)
    return phy, (y = y, x = x, species = sp)
end

@testset "sd_phylo: grammar and refusals" begin
    phy, dat = _qqq_sim()
    f = bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ x),
           @formula(sd(species, phylogenetic) ~ x))
    @test any(p -> first(p) === :sdphy_species, f.forms)
    # deprecated legacy spelling still routes to the same part (like the twin)
    flegacy = bf(@formula(y ~ x + phylo(1 | species)), @formula(sd_phylo(species) ~ x))
    @test any(p -> first(p) === :sdphy_species, flegacy.forms)
    # the level must be a known one, and only phylogenetic is implemented
    @test_throws ArgumentError bf(@formula(y ~ x), @formula(sd(g, bogus) ~ 1))
    @test_throws ArgumentError bf(@formula(y ~ x), @formula(sd(g, spatial) ~ 1))
    # sd_phylo without a phylo() mean marker
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | species)),
                                      @formula(sd(species, phylogenetic) ~ x)),
                                   Gaussian(); data = dat, tree = phy)
    # sd_phylo naming a different grouping than the phylo marker
    @test_throws ArgumentError drm(bf(@formula(y ~ x + phylo(1 | species)),
                                      @formula(sd(x, phylogenetic) ~ 1)),
                                   Gaussian(); data = dat, tree = phy)
    # sd() and sd_phylo() together
    @test_throws ArgumentError drm(bf(@formula(y ~ x + phylo(1 | species)),
                                      @formula(sd(species) ~ 1),
                                      @formula(sd(species, phylogenetic) ~ 1)),
                                   Gaussian(); data = dat, tree = phy)
    # REML is supported on this route (#558)
    fit_reml = drm(f, Gaussian(); data = dat, tree = phy, method = :REML)
    @test fit_reml.converged
    @test estimation_method(fit_reml) === :REML
    @test isfinite(reml_loglik(fit_reml))
    @test isfinite(ml_loglik(fit_reml))
end

@testset "sd_phylo: QQQ fit matches drmTMB and recovers the mean" begin
    phy, dat = _qqq_sim()
    fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ x),
                 @formula(sd(species, phylogenetic) ~ x)), Gaussian(); data = dat, tree = phy)
    @test fit.converged
    @test isfinite(fit.loglik)
    # pinned against drmTMB on the identical fixture (agreement to 7 s.f.)
    @test isapprox(fit.loglik, -69.1373; atol = 1e-3)
    @test isapprox(coef(fit, :mu), [0.9944436, 0.4478094]; atol = 1e-4)
    @test isapprox(coef(fit, :sigma), [-1.26293, 0.6845002]; atol = 1e-4)
    @test isapprox(coef(fit, :sd_phylo), [-0.1869285, -0.1827046]; atol = 1e-4)
    # the mean is the well-identified component and must recover truth (1.0, 0.5)
    @test isapprox(coef(fit, :mu), [1.0, 0.5]; atol = 0.15)
    # single-SD summaries refuse on a scale-scale fit
    @test_throws ArgumentError re_sd(fit)
end

@testset "sd_phylo: reduction invariant + #548 regression" begin
    phy, dat = _qqq_sim()
    f1 = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ x),
                @formula(sd(species, phylogenetic) ~ 1)), Gaussian(); data = dat, tree = phy)
    f0 = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ x)),
             Gaussian(); data = dat, tree = phy)
    @test f1.converged && f0.converged
    # #548: this scalar fit returned logLik +1.1467e105 with converged = true
    # before the stable dense assembly landed; drmTMB says -70.38332.
    @test isfinite(f0.loglik)
    @test isapprox(f0.loglik, -70.38332; atol = 1e-3)
    @test f0.loglik < 0
    # reduction: sd_phylo(s) ~ 1 IS the scalar model
    @test abs(f1.loglik - f0.loglik) ≤ 1e-8
    @test :resd in first.(f0.blocks)          # scalar route keeps its block name
    @test :sd_phylo in first.(f1.blocks)
end

@testset "sd_phylo: homoscedastic structured route is untouched (#548 guard)" begin
    phy, dat = _qqq_sim()
    # pσ == 1 takes the sparse Woodbury path.  At a residual variance below
    # machine precision relative to the phylo component it once reported a
    # spurious positive likelihood, so retain the small dense GLS oracle.
    fh = drm(bf(@formula(y ~ x + phylo(1 | species))), Gaussian(); data = dat, tree = phy)
    fh_dense = drm(bf(@formula(y ~ x + phylo(1 | species))), Gaussian();
                   data = dat, tree = phy, algorithm = :gls)
    @test fh.converged && isfinite(fh.loglik) && fh.loglik < 0
    @test :resd in first.(fh.blocks)
    @test isapprox(fh.loglik, fh_dense.loglik; atol = 1e-6)
end

@testset "#574 follow-up: every scale pair the guard ADMITS matches a dense oracle" begin
    # WHY THIS TEST EXISTS, and why the #461 bootstrap testset was not enough.
    # #574 refused the sparse Woodbury objective where the residual variance is
    # "below machine precision relative to the phylogenetic variance", coded as
    # sigma^2 / sigma_phy^2 >= eps.  That is exactly the point where NO digits
    # of the subtraction survive, so the bar had zero margin and the objective
    # was still wrong just above it: measured 2026-09-05 on the #461 fixture,
    # the sign flip that produces the runaway +Inf optimum sits at
    # log(sigma) - log(sigma_phy) = -17.086, INSIDE the old bar of -18.022.
    #
    # Whether the optimizer actually walks into that sliver is decided by
    # rounding, so the #461 bootstrap testset passed on x86_64 Linux CI and
    # failed on aarch64 macOS at the same commit with the same seeds.  A test
    # that depends on the optimizer's path cannot pin a numerical boundary.
    # This one evaluates the objective directly and is path-independent.
    phy, dat = _qqq_sim()
    G = phy.n_leaves
    C = DRModels.sigma_phy_dense(phy; σ²_phy = 1.0)
    X = hcat(ones(G), dat.x)
    prob = DRModels.make_loc_problem(phy, dat.y, X; species = 1:G)

    dense_nll(β, s2phy, s2) = begin
        V = s2 .* Matrix{Float64}(I, G, G) .+ s2phy .* C
        ch = cholesky(Symmetric(V))
        e = dat.y .- X * β
        0.5 * (G * log(2π) + logdet(ch) + dot(e, ch \ e))
    end

    lσ_phy = -0.35
    bar = 0.25 * log(eps(Float64))          # keep half the mantissa
    # Sweep PAST the old bar so the property -- admitted implies accurate --
    # is what is asserted, not the value of the constant.  Points the guard
    # refuses are skipped, so tightening the bar only shortens the sweep;
    # loosening it back to #574's value re-admits points that fail the
    # dense-oracle assertion below.
    admitted = 0
    for gap in range(0.0, 0.5 * log(eps(Float64)) - 2.0; length = 60)
        DRModels._loconly_resolvable_scales(lσ_phy + gap, lσ_phy) || continue
        admitted += 1
        β, nll, _ = DRModels._loconly_profile_beta(prob, lσ_phy + gap, lσ_phy)
        @test β !== nothing
        β === nothing && continue
        ref = dense_nll(β, exp(2 * lσ_phy), exp(2 * (lσ_phy + gap)))
        # Admitted means USABLE: the sparse objective must still be the same
        # number the dense oracle computes, not merely finite.
        @test abs(nll - ref) <= 1e-6 * max(1.0, abs(ref))
    end
    @test admitted >= 20                     # an empty sweep is not a pass

    # The measured break point must be on the refused side, with margin.
    @test !DRModels._loconly_resolvable_scales(lσ_phy - 17.086, lσ_phy)
    @test !DRModels._loconly_resolvable_scales(lσ_phy + bar - 0.01, lσ_phy)
    # ... and an ordinary fit is nowhere near it: the worst of the #461
    # fixture's 60 bootstrap refits sat at a gap of -1.033 (the 60 spanned
    # [-1.033, +1.439]), about eight nats clear of the bar.
    @test DRModels._loconly_resolvable_scales(lσ_phy - 1.5, lσ_phy)
end

@testset "bridge: sd()/sd_phylo() parts route (#546)" begin
    dat = (y = [1.0, 2.0, 3.0, 4.0], x = [0.0, 1.0, 0.0, 1.0],
           z = [1.0, 1.0, 2.0, 2.0], g = ["a", "a", "b", "b"])
    # positional spelling — what drmTMB's `drm_julia_formula_entry` emits
    b, _ = DRModels._bridge_formula(["y ~ x + (1 | g)", "sigma ~ x", "sd(g) ~ z"], "gaussian", dat)
    @test first.(b.forms) == [:mu, :sigma, :sd_g]
    # dict spelling — what JuliaCall marshals a named R list into. The key is the
    # marker call itself, so the value (already the full formula) passes through
    # rather than becoming `f(x) = body`, which Julia reads as a function def.
    dphy = (y = [1.0, 2.0], x = [0.0, 1.0], species = ["a", "b"])
    d = Dict{Symbol,Any}(:mu => "y ~ x + phylo(1 | species)", :sigma => "sigma ~ x",
                         Symbol("sd_phylo(species)") => "sd_phylo(species) ~ x")
    b2, _ = DRModels._bridge_formula(d, "gaussian", dphy)
    @test Set(first.(b2.forms)) == Set([:mu, :sigma, :sdphy_species])
    # an unknown keyed part must ERROR, never be silently dropped (issue-#2 class)
    @test_throws ArgumentError DRModels._bridge_formula(
        Dict{Symbol,Any}(:mu => "y ~ x", :bogus => "bogus ~ x"), "gaussian",
        (y = [1.0, 2.0], x = [0.0, 1.0]))
end

@testset "threaded inference on the stored nll closure (#549/#550)" begin
    # The dense-route nll closure is stored on the fit and called CONCURRENTLY by
    # threaded profile/bootstrap. #549: a closure-local named like an enclosing
    # local is ONE shared boxed variable — two threads' Dual-tagged matrices
    # crossed and the profile crashed. On a single-threaded CI runner the
    # threaded branches degrade to serial, so this is a smoke everywhere and a
    # genuine race regression test wherever JULIA_NUM_THREADS > 1.
    phy, dat = _qqq_sim()
    fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ x),
                 @formula(sd(species, phylogenetic) ~ x)),
              Gaussian(); data = dat, tree = phy)
    ci = confint(fit; parm = :sd_phylo, method = :profile, threads = true)
    @test length(ci) == 2
    @test all(r -> r.lower < r.estimate < r.upper, ci)
    bs = bootstrap_result(fit; data = dat, B = 12, tree = phy, threads = true,
                          rng = StableRNG(42), failures = :skip)
    @test bs.used ≥ 10                      # at most a couple of degenerate refits
    @test bs.attempted == 12
end
