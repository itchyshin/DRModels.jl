# R/Julia parity: Gaussian location-scale with a mean random intercept,
#     y ~ x + (1 | g),  sigma ~ x        (ML, Laplace == exact for Gaussian)
# vs drmTMB 0.7.0 (CRAN release, pinned in expected.meta.toml). Static fixtures
# only -- no R at test time. Regenerate with
# `Rscript --vanilla test/parity/gen_gaussian_locscale_re.R`.
# License: fixtures are generated numbers + simulated data, no drmTMB source.
using DRModels
using Test, LinearAlgebra, TOML
include(joinpath(@__DIR__, "parity", "compare.jl"))
include(joinpath(@__DIR__, "parity", "loadfixture.jl"))

@testset "parity: Gaussian locscale + (1|g) vs drmTMB 0.7.0" begin
    dir = joinpath(@__DIR__, "parity", "fixtures_re", "gaussian-locscale-re")
    meta = TOML.parsefile(joinpath(dir, "expected.meta.toml"))
    @test meta["drmtmb_version"] == "0.7.0"
    ref = TOML.parsefile(joinpath(dir, "expected.toml"))
    d = load_data(dir)
    fit = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ x)), Gaussian(); data = d)
    @test fit.converged

    rel(a, b) = abs(a - b) / max(abs(b), 1e-12)
    cm = coef(fit, :mu); cs = coef(fit, :sigma)
    got = Dict("mu_(Intercept)" => cm[1], "mu_x" => cm[2],
               "sigma_(Intercept)" => cs[1], "sigma_x" => cs[2])
    sd_g = re_sd(fit)[:g]
    se = stderror(fit)                       # last entry: SE of log RE-sd
    gotse = Dict("mu_(Intercept)" => se[1], "mu_x" => se[2],
                 "sigma_(Intercept)" => se[3], "sigma_x" => se[4])

    @testset "fit summary" begin
        @test nobs(fit) == ref["fit"]["n"]
        @test dof(fit) == ref["fit"]["df"]
        @test abs(loglik(fit) - ref["fit"]["loglik"]) <= 1e-3
    end
    @testset "fixed-effect coefficients" begin
        for (k, v) in got
            @test rel(v, ref["coef"][k]) <= 1e-4
        end
    end
    @testset "random-intercept SD (natural scale)" begin
        @test rel(sd_g, ref["coef"]["sd_mu_(1 | g)"]) <= 1e-4
    end
    @testset "Wald SEs" begin
        for (k, v) in gotse
            @test rel(v, ref["se"][k]) <= 1e-3
        end
        # drmTMB reports the SD-scale SE; delta method from log-sd SE
        @test rel(sd_g * se[5], ref["se"]["sd_mu_(1 | g)"]) <= 1e-3
    end
    @testset "fixed-effect vcov block" begin
        V = vcov(fit)[1:4, 1:4]
        R = reduce(hcat, ref["vcov"]["data"])'   # row-major -> matrix
        order = ref["vcov"]["order"]
        @test order == ["mu_(Intercept)", "mu_x", "sigma_(Intercept)", "sigma_x"]
        @test maximum(abs.(V .- R) ./ max.(abs.(R), 1e-6)) <= 1e-3
    end
end
