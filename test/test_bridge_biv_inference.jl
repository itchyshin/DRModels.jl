# Profile and bootstrap intervals for a FIXED-EFFECT target of the RESIDUAL-ONLY
# bivariate Gaussian fit (drmTMB A8b / drmTMB#544 row `biv_gaussian_residual`).
#
# Scope, stated up front: this route is `bf(mu1 = y1 ~ x, mu2 = y2 ~ x,
# sigma1 = ~1, sigma2 = ~1, rho12 = ~1)` with NO structured term — `_fit_bivariate_residual`
# builds it, so `fit.ranef === nothing` and every `bootstrap_result` method's
# `Sigma_a` guard skips it. The q=4 phylogenetic bivariate route is a DIFFERENT
# target (four among-axis SDs) and is covered by test_bridge_bivariate_inference.jl;
# nothing here touches it, and the last testset asserts it is still routed away.
#
# What is NOT claimed: interval COVERAGE. These are pipeline/shape assertions plus
# the #631 endpoint-failure contract — one fixture, one seed.
using DRModels, Test, Random, Statistics

function _biv_residual_fixture(; n = 200, seed = 20260905)
    rng = Random.MersenneTwister(seed)
    x = randn(rng, n)
    σ1, σ2, ρ = 0.7, 0.5, 0.4
    e1 = randn(rng, n)
    e2 = ρ .* e1 .+ sqrt(1 - ρ^2) .* randn(rng, n)
    y1 = 0.3 .+ 0.5 .* x .+ σ1 .* e1
    y2 = -0.2 .+ 0.8 .* x .+ σ2 .* e2
    return (; y1, y2, x)
end

const _BIV_FML = Dict(
    :mu1 => "y1 ~ x",
    :mu2 => "y2 ~ x",
    :sigma1 => "sigma1 ~ 1",
    :sigma2 => "sigma2 ~ 1",
    :rho12 => "rho12 ~ 1",
)

@testset "bivariate residual fixed-effect profile intervals" begin
    dat = _biv_residual_fixture()
    fit = drm(bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
                 sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                 rho12 = @formula(rho12 ~ 1)), Gaussian(); data = dat)
    @test fit.formula isa DRModels.BivariateDrmFormula
    @test fit.ranef === nothing           # residual-only: no Sigma_a route
    @test is_converged(fit)

    for (param, coefname) in ((:mu1, "x"), (:rho12, "(Intercept)"))
        res = profile_result(fit; level = 0.95, parm = param => coefname)
        @test res.attempted == 1
        @test res.used == 1
        @test res.failed == 0             # #631: a failed arm counts here
        row = only(res.ci)
        @test row.param === param
        @test row.coef == coefname
        @test isfinite(row.lower) && isfinite(row.upper)
        @test row.lower < row.estimate < row.upper
        s = only(res.stats)
        # #631 endpoint-failure flags: honoured (present, and false on a clean fit).
        @test s.lower_endpoint_failed === false
        @test s.upper_endpoint_failed === false
        @test s.lower_unbounded === false
        @test s.upper_unbounded === false
    end

    # The profile interval must be near, but not identical to, the Wald one:
    # identical bounds would mean the profiler never re-optimised.
    prof = only(profile_result(fit; parm = :mu1 => "x").ci)
    wald = only(filter(r -> r.param === :mu1 && r.coef == "x", confint(fit)))
    @test prof.lower != wald.lower
    @test abs(prof.lower - wald.lower) < 0.05
end

@testset "bivariate residual fixed-effect bootstrap intervals" begin
    dat = _biv_residual_fixture()
    fit = drm(bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
                 sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                 rho12 = @formula(rho12 ~ 1)), Gaussian(); data = dat)
    res = bootstrap_result(fit; data = dat, B = 40, level = 0.95,
                           rng = Random.MersenneTwister(4242),
                           failures = :skip, check_converged = true)
    @test res.attempted == 40
    @test res.failed == 0
    @test res.used == 40
    for (param, coefname) in ((:mu1, "x"), (:rho12, "(Intercept)"))
        row = only(filter(r -> r.param === param && r.coef == coefname, res.summary))
        @test isfinite(row.lower) && isfinite(row.upper)
        @test row.lower < row.upper
        @test row.std_error > 0
    end
    # Every block is represented, in θ order — the summary must not silently
    # drop the σ or ρ blocks the bivariate fit carries.
    @test Set(r.param for r in res.summary) == Set([:mu1, :mu2, :sigma1, :sigma2, :rho12])
end

@testset "drm_bridge_inference — bivariate residual fixed-effect target" begin
    dat = _biv_residual_fixture()
    for parm in ("fixef:mu1:x", "fixef:rho12:(Intercept)")
        prof = drm_bridge_inference(; formula = _BIV_FML, family = "biv_gaussian",
            data = dat, method = "profile", parm = parm)
        @test prof["status"] == "profile"
        @test isfinite(prof["lower"]) && isfinite(prof["upper"])
        @test prof["lower"] < prof["estimate"] < prof["upper"]
        @test prof["failed"] == 0

        boot = drm_bridge_inference(; formula = _BIV_FML, family = "biv_gaussian",
            data = dat, method = "bootstrap", parm = parm, B = 40, seed = 4242)
        @test boot["status"] == "bootstrap"
        @test isfinite(boot["lower"]) && isfinite(boot["upper"])
        @test boot["lower"] < boot["upper"]
        @test boot["failed"] == 0
        @test boot["used"] == 40
    end
end

@testset "bivariate bootstrap replicate data — observation pattern preserved" begin
    # A cell that was unobserved must stay unobserved in every replicate, or the
    # refits would fit a complete-data model the seed fit never fitted.
    fml = bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
             sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
             rho12 = @formula(rho12 ~ 1))
    data = (; y1 = [1.0, NaN, 3.0], y2 = [0.5, 0.6, NaN], x = [0.1, 0.2, 0.3])
    ysim = Dict(:mu1 => [9.0, 9.0, 9.0], :mu2 => [8.0, 8.0, 8.0])
    out = DRModels._bootstrap_data(fml, data, ysim)
    @test out.y1[1] == 9.0
    @test isnan(out.y1[2])
    @test out.y1[3] == 9.0
    @test out.y2[1] == 8.0
    @test out.y2[2] == 8.0
    @test isnan(out.y2[3])
    @test out.x == data.x

    # `missing`-typed columns keep their eltype and their missing cells.
    data_m = (; y1 = Union{Missing,Float64}[1.0, missing, 3.0],
                y2 = Union{Missing,Float64}[0.5, 0.6, 0.7], x = [0.1, 0.2, 0.3])
    out_m = DRModels._bootstrap_data(fml, data_m, ysim)
    @test ismissing(out_m.y1[2])
    @test out_m.y1[1] == 9.0
    @test eltype(out_m.y1) === Union{Missing,Float64}

    # A draw of the wrong shape is refused, not silently recycled.
    @test_throws ArgumentError DRModels._bootstrap_data(fml, data, [1.0, 2.0, 3.0])
    @test_throws ArgumentError DRModels._bootstrap_data(
        fml, data, Dict(:mu1 => [9.0, 9.0], :mu2 => [8.0, 8.0, 8.0]))
end

@testset "widened bootstrap gate refuses what it must" begin
    # A formula-less internal fit still has nothing to refit from.
    dat = _biv_residual_fixture(; n = 40)
    fit = drm(bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
                 sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                 rho12 = @formula(rho12 ~ 1)), Gaussian(); data = dat)
    stripped = DRModels.DrmFit(fit.family, fit.blocks, fit.coefnames, fit.theta, fit.vcov,
                          fit.loglik, fit.nobs, fit.converged, fit.means, fit.obs,
                          fit.scales)
    @test stripped.formula === nothing
    @test_throws ArgumentError DRModels._bootstrap_fit_formula(stripped)

    # The structured (q=4 phylo) bivariate fit is NOT admitted to this path: its
    # `Sigma_a` guard fires first, so bootstrap_result routes to bootstrap_sigma_a.
    Random.seed!(20260905)
    p, m = 12, 4
    phy = random_balanced_tree(p; branch_length = 0.3)
    species = repeat(1:p, inner = m)
    n = length(species)
    x = randn(n)
    dat4 = (; y1 = randn(n) .+ 0.3 .* x, y2 = randn(n) .- 0.2 .* x, x, species)
    fit4 = drm(bf(mu1 = @formula(y1 ~ x + phylo(1 | species)),
                  mu2 = @formula(y2 ~ x + phylo(1 | species)),
                  sigma1 = @formula(sigma1 ~ 1 + phylo(1 | species)),
                  sigma2 = @formula(sigma2 ~ 1 + phylo(1 | species)),
                  rho12 = @formula(rho12 ~ 1)), Gaussian(); data = dat4, tree = phy)
    @test fit4.ranef isa NamedTuple && haskey(fit4.ranef, :Sigma_a)
    res4 = bootstrap_result(fit4; data = dat4, B = 3, level = 0.95,
                            rng = Random.MersenneTwister(7), failures = :skip)
    # bootstrap_sigma_a returns the four among-axis SD rows, never the mu1/rho12
    # fixed-effect rows the residual route returns.
    @test Set(r.param for r in res4.summary) ==
          Set([:sd_mu1, :sd_mu2, :sd_sigma1, :sd_sigma2])
end
