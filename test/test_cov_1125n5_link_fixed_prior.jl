# test_cov_1125n5_link_fixed_prior.jl — link_residual + fixed priors (src/link_residual.jl, fixed_priors.jl).
using DRModels
using Test
import SpecialFunctions: trigamma

@testset "cov 1125n5: link_residual families" begin
    @test DRModels.link_residual(Gaussian(), 0.0; dispersion = 2.5) == 2.5
    v = DRModels.link_residual(Poisson(), 3.0)
    @test v ≈ log1p(1 / 3.0)
    @test v > 0
    @test DRModels.link_residual(Binomial(), 0.5) ≈ (π^2) / 3
    μ = 0.4
    φ = 12.0
    vb = DRModels.link_residual(Beta(), μ; dispersion = φ)
    @test vb ≈ trigamma(μ * φ) + trigamma((1 - μ) * φ)
    @test DRModels.link_residual(Gamma(), 0.0; dispersion = 0.5) ≈ trigamma(2.0)
    @test DRModels.link_residual(NegBinomial2(), 0.0; dispersion = 1.2) ≈ trigamma(1.2)
    @test DRModels.link_residual(Poisson(), 1e-14) ≈ log1p(1 / 1e-12) rtol = 1e-10
end

@testset "cov 1125n5: drm_prior constructors" begin
    p = drm_prior_normal(; dpar = :mu, coef = "x", mean = 0.0, sd = 2.0)
    @test p.kind === :normal
    @test p.precision ≈ 0.25
    pr = drm_prior_ridge(; dpar = :sigma, coef = "(Intercept)", lambda = 3.0)
    @test pr.kind === :ridge && pr.precision == 3.0
    msg = try
        drm_prior_normal(; dpar = :mu, coef = "x", sd = -1.0)
        ""
    catch e
        e isa ArgumentError ? e.msg : rethrow()
    end
    @test occursin("sd", msg)
    priors = drm_priors(p, pr)
    @test length(priors) == 2
end

@testset "cov 1125n5: _resolve_fixed_priors" begin
    names = Dict(:mu => ["(Intercept)", "x"], :sigma => ["(Intercept)"])
    intercepts = Dict(:mu => 1, :sigma => 1)
    rows = DRModels._resolve_fixed_priors(
        drm_priors(drm_prior_ridge(; dpar = :mu, coef = "x", lambda = 1.0)), names, intercepts)
    @test length(rows) == 1
    @test rows[1].coef == "x"
    @test rows[1].index == 2
    rows_all = DRModels._resolve_fixed_priors(
        drm_priors(drm_prior_normal(; dpar = :mu, coef = "all", sd = 1.0, intercept = :exclude)),
        names, intercepts)
    @test length(rows_all) == 1
    @test rows_all[1].coef == "x"
    θ = [0.0, 0.5, -0.2]
    val = DRModels._fixed_prior_value(rows, θ)
    @test val ≈ 0.5 * 1.0 * (0.5 - 0)^2 rtol = 1e-12
    msg = try
        DRModels._resolve_fixed_priors(drm_priors(drm_prior_ridge(; dpar = :nu, coef = "x", lambda = 1.0)),
                                       names, intercepts)
        ""
    catch e
        e isa ArgumentError ? e.msg : rethrow()
    end
    @test occursin("dpar", msg) || occursin("nu", msg)
end

@testset "cov 1125n5: _fixed_prior validation" begin
    msg = try
        DRModels._fixed_prior(:normal, :mu, "", nothing, 0.0, 1.0, 0.0)
        ""
    catch e
        e isa ArgumentError ? e.msg : rethrow()
    end
    @test occursin("empty", msg)
end