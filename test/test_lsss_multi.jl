# #555 — location-scale-scale-scale: several random effects, each with its own
# SD submodel. Cross-engine pins against drmTMB on the committed fixtures
# (test/fixtures/lsss/): fixture A agrees on logLik to 8 s.f. and every
# coefficient to 6 decimals; fixture B (THREE random effects, five submodels)
# the same. Reference numbers in the fixtures README.
using Test
using DRModels
import LinearAlgebra

function _read_lsss_csv(path)
    raw = readlines(path)
    hdr = [String(strip(h, '"')) for h in split(raw[1], ",")]
    cols = [String[] for _ in hdr]
    for l in raw[2:end], (k, v) in enumerate(split(l, ","))
        push!(cols[k], String(strip(v, '"')))
    end
    return Dict(h => c for (h, c) in zip(hdr, cols))
end

const _LSSS_DIR = joinpath(@__DIR__, "fixtures", "lsss")

@testset "lsss multi: iid + phylo, both sd ~ x (fixture A)" begin
    A = _read_lsss_csv(joinpath(_LSSS_DIR, "lsss_A.csv"))
    dat = (y = parse.(Float64, A["y"]), x = parse.(Float64, A["x"]), species = A["species"])
    phy = DRModels.augmented_phy(read(joinpath(_LSSS_DIR, "lsss_A.nwk"), String))
    fit = drm(bf(@formula(y ~ x + (1 | species) + phylo(1 | species)),
                 @formula(sigma ~ 1),
                 @formula(sd(species) ~ x),
                 @formula(sd(species, phylogenetic) ~ x)),
              Gaussian(); data = dat, tree = phy)
    @test fit.converged
    @test isapprox(fit.loglik, -137.669160; atol = 1e-3)
    @test isapprox(coef(fit, :mu), [1.461294, 0.7002199]; atol = 1e-3)
    @test isapprox(coef(fit, :sigma), [-1.209928]; atol = 1e-3)
    @test isapprox(coef(fit, :sd), [-1.237836, -0.2841831]; atol = 1e-3)
    @test isapprox(coef(fit, :sd_phylo), [-0.8367155, 0.3104847]; atol = 1e-3)
    # single-SD summaries refuse on any scale-scale fit
    @test_throws ArgumentError re_sd(fit)
    # Wald CIs target both scale blocks
    @test length(confint(fit; parm = :sd)) == 2
    @test length(confint(fit; parm = :sd_phylo)) == 2
end

@testset "lsss multi: three REs, five submodels (fixture B)" begin
    B = _read_lsss_csv(joinpath(_LSSS_DIR, "lsss_B.csv"))
    dat = (y = parse.(Float64, B["y"]), x = parse.(Float64, B["x"]),
           species = B["species"], study = B["study"])
    phy = DRModels.augmented_phy(read(joinpath(_LSSS_DIR, "lsss_B.nwk"), String))
    fit = drm(bf(@formula(y ~ x + (1 | species) + (1 | study) + phylo(1 | species)),
                 @formula(sigma ~ 1),
                 @formula(sd(species) ~ x),
                 @formula(sd(study) ~ 1),
                 @formula(sd(species, phylogenetic) ~ x)),
              Gaussian(); data = dat, tree = phy)
    @test fit.converged
    @test isapprox(fit.loglik, -96.810070; atol = 1e-3)
    @test isapprox(coef(fit, :sd), [-0.9247607, -0.01718566, -1.222959]; atol = 2e-3)
    @test isapprox(coef(fit, :sd_phylo), [-1.012909, -0.04165254]; atol = 2e-3)
    # group-prefixed names when more than one iid component shares the :sd block
    @test Dict(fit.coefnames)[:sd] ==
          ["species: (Intercept)", "species: x", "study: (Intercept)"]
end

@testset "lsss multi: refusals" begin
    B = _read_lsss_csv(joinpath(_LSSS_DIR, "lsss_B.csv"))
    dat = (y = parse.(Float64, B["y"]), x = parse.(Float64, B["x"]),
           species = B["species"], study = B["study"])
    phy = DRModels.augmented_phy(read(joinpath(_LSSS_DIR, "lsss_B.nwk"), String))
    # sd() naming a grouping with no matching random effect
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | species)),
                                      @formula(sd(study) ~ 1)),
                                   Gaussian(); data = dat)
    # REML supported on the multi route (#558)
    fit_reml = drm(bf(@formula(y ~ x + (1 | species) + (1 | study)),
                      @formula(sd(species) ~ x), @formula(sd(study) ~ 1)),
                   Gaussian(); data = dat, method = :REML)
    @test fit_reml.converged
    @test estimation_method(fit_reml) === :REML
    @test isfinite(reml_loglik(fit_reml))
    @test isfinite(ml_loglik(fit_reml))
    # sd(g, phylogenetic) without a phylo() marker
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | species)),
                                      @formula(sd(species) ~ 1),
                                      @formula(sd(species, phylogenetic) ~ 1)),
                                   Gaussian(); data = dat)
end

@testset "sparse phylo-mean variance-block SEs (#556)" begin
    # Before #556 the sparse route's vcov was `fill(NaN, …)` outside the mean
    # block, so sigma/resd SEs were NA on every Gaussian phylo-mean fit — the
    # Mizuno M2 shape, caught by the acceptance matrix. Pinned against drmTMB
    # on the committed fixture (agreement: sigma SE 0.05907 vs 0.05906679).
    M = _read_lsss_csv(joinpath(_LSSS_DIR, "m2_sparse_se.csv"))
    tv = parse.(Float64, M["temp"]); pv = parse.(Float64, M["prec"])
    dat = (y = parse.(Float64, M["y"]), temp = tv, prec = pv,
           temp2 = tv .^ 2, prec2 = pv .^ 2, species = M["species"])
    phy = DRModels.augmented_phy(read(joinpath(_LSSS_DIR, "m2_sparse_se.nwk"), String))
    fit = drm(bf(@formula(y ~ temp + temp2 + prec + prec2 + phylo(1 | species))),
              Gaussian(); data = dat, tree = phy)
    @test fit.converged
    se = sqrt.(abs.(LinearAlgebra.diag(fit.vcov)))
    @test all(isfinite, se)
    # drmTMB reference (same data, native TMB): mean block + sigma
    @test isapprox(se[1:5], [0.31821877, 0.02659921, 0.01938464, 0.02635787, 0.01776037];
                   rtol = 5e-3)
    @test isapprox(se[6], 0.05906679; rtol = 5e-3)
    @test se[7] > 0                     # phylo-SD SE: finite and positive
end
