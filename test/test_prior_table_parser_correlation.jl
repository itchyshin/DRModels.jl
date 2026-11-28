using Test
using DRModels

const _lkj_table = (
    prior=["lkj(2)"],
    class=["correlation"],
    coef=[""],
    dpar=["mu"],
    group=["site"],
    from_coef=["x"],
    to_coef=["(Intercept)"],
    scale=["natural_correlation"],
)

@testset "prior-table LKJ schema and constructor" begin
    row = only(DRModels._parse_prior_table(_lkj_table))
    @test row.kind === :lkj
    @test row.class == "correlation"
    @test row.scale == "natural_correlation"
    @test row.dpar == "mu"
    @test row.group == "site"
    @test row.from_coef == "(Intercept)"
    @test row.to_coef == "x"
    @test row.arguments == (2.0,)

    prior = DRModels.drm_prior_lkj(
        dpar=:mu, eta=2, group="site", from_coef="x", to_coef="(Intercept)",
    )
    @test prior isa DRModels.CorrelationLKJPrior
    @test prior.scale === :natural_correlation
    @test prior.from_coef == "(Intercept)"
    @test prior.to_coef == "x"

    guard = 0.99999999
    @test exp(DRModels._lkj_log_normalizer(2.0, guard)) ≈
          2 * (guard - guard^3 / 3) atol=1e-12
    @test DRModels._lkj_log_prior(2.0, 0.5, guard) ≈
          log(0.75) - log(4 / 3) atol=1e-8
end

@testset "prior-table correlation selectors fail closed" begin
    parse_table(; prior="lkj(2)", class="correlation", coef="", dpar="mu",
                group="site", from_coef="(Intercept)", to_coef="x",
                scale="natural_correlation") =
        DRModels._parse_prior_table((;
            prior=[prior], class=[class], coef=[coef], dpar=[dpar], group=[group],
            from_coef=[from_coef], to_coef=[to_coef], scale=[scale],
        ))

    @test_throws ArgumentError parse_table(prior="lkj(1)")
    @test_throws ArgumentError parse_table(prior="normal(0,1)")
    @test_throws ArgumentError parse_table(scale="link")
    @test_throws ArgumentError parse_table(from_coef="x", to_coef="z")
    @test_throws ArgumentError parse_table(group="")
    @test_throws ArgumentError parse_table(dpar="sigma")
    @test_throws ArgumentError parse_table(coef="x")

    rho12 = (prior=["lkj(2)"], class=["correlation"], coef=["(Intercept)"],
             dpar=["rho12"], group=[""], scale=["natural_correlation"])
    rho_row = only(DRModels._parse_prior_table(rho12))
    @test rho_row.dpar == "rho12"
    @test rho_row.scale == "natural_correlation"
    @test_throws ArgumentError parse_table(dpar="rho12", group="site")
    @test_throws ArgumentError DRModels.drm_prior_lkj(dpar=:mu, eta=2,
        group="site", from_coef="x", to_coef="z")

    reversed = merge(_lkj_table, (from_coef=["x"], to_coef=["(Intercept)"]))
    row = only(DRModels._parse_prior_table(reversed))
    @test row.from_coef == "(Intercept)"
    @test row.to_coef == "x"
    @test_throws ArgumentError DRModels._prior_table_specs(
        [only(DRModels._parse_prior_table(_lkj_table)), row], ["site"])
end

@testset "natural-scale Gamma targets distinguish residual and random SDs" begin
    residual = (prior=["gamma(0.6666666666666666,2)"],
                class=["residual_sd"], coef=["(Intercept)"],
                dpar=["sigma"], scale=["natural_sd"])
    row = only(DRModels._parse_prior_table(residual))
    @test row.class == "residual_sd"
    @test row.dpar == "sigma"
    @test row.scale == "natural_sd"
    @test row.arguments == (2 / 3, 2.0)

    spec = only(DRModels._prior_table_specs([row], String[]))
    @test spec isa DRModels.ResidualSDGammaPrior
    @test spec.shape == 2
    @test spec.rate == 3

    random_sd = (prior=["gamma(0.6666666666666666,2)"],
                 class=["ranef_sd"], coef=["(Intercept)"],
                 dpar=["mu"], group=["site"], scale=["natural_sd"])
    @test only(DRModels._parse_prior_table(random_sd)).scale == "natural_sd"
    @test_throws ArgumentError DRModels._parse_prior_table(merge(residual, (scale=["link"],)))
    @test_throws ArgumentError DRModels._parse_prior_table(merge(residual, (dpar=["mu"],)))
    @test_throws ArgumentError DRModels._parse_prior_table(merge(residual, (coef=["x"],)))
end
