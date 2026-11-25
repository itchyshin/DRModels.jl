# cov lane 1125n6 — small helpers across several uncovered src/ files (single shard file).
using DRModels
using Test, LinearAlgebra, Tables

@testset "cov 1125n6: _prior_table_specs" begin
    rows = DRModels._parse_prior_table((prior = ["normal(0,2)", "ridge(0.5)"],
                                        class = ["fixef", "fixef"],
                                        coef = ["x", "y"]))
    specs = DRModels._prior_table_specs(rows, String[])
    @test length(specs) == 2
    @test specs[1] isa DRModels.FixedPrior
    sdrow = DRModels._parse_prior_table((prior = ["gamma(1,4)"],
                                         class = ["ranef_sd"], coef = [""]))
    expanded = DRModels._expand_prior_table(sdrow, ["g1", "g2"])
    sd_specs = DRModels._prior_table_specs(expanded, ["g1", "g2"])
    @test all(s -> s isa DRModels.SDGammaPrior, sd_specs)
end

@testset "cov 1125n6: _zeroonebeta_marginal" begin
    @test !DRModels._zeroonebeta_marginal(:LA)
    @test DRModels._zeroonebeta_marginal(:Laplace)
    @test !DRModels._zeroonebeta_marginal(:la)
    @test_throws ArgumentError DRModels._zeroonebeta_marginal(:GHQ)
end

@testset "cov 1125n6: joint_missing_predictor guards" begin
    @test DRModels._joint_float_or_missing(1.5, "y") == 1.5
    @test DRModels._joint_float_or_missing(missing, "y") === missing
    @test_throws ArgumentError DRModels._joint_float_or_missing("a", "y")
    @test_throws ArgumentError DRModels._joint_float_or_missing(NaN, "y")
    M = DRModels._joint_matrix([1 2; 3 4], 2, "X")
    @test size(M) == (2, 2)
    @test DRModels._joint_names(["a", "b"], 2, "nm") == ["a", "b"]
    @test_throws ArgumentError DRModels._joint_names(["a", "a"], 2, "nm")
    @test DRModels._joint_log1pexp(0.0) ≈ log(2.0) rtol = 1e-14
    @test isfinite(DRModels._joint_log1pexp(30.0))
    @test isfinite(DRModels._joint_log1pexp(-30.0))
end

@testset "cov 1125n6: _clamp01_ci" begin
    ci = (lower = -0.2, upper = 1.2)
    out = DRModels._clamp01_ci(ci)
    @test out.lower == 0.0
    @test out.upper == 1.0
    mid = DRModels._clamp01_ci((lower = 0.1, upper = 0.8))
    @test mid.lower == 0.1 && mid.upper == 0.8
end

@testset "cov 1125n6: _ratio_closure algebra" begin
    θ = [log(2.0), log(3.0), log(1.0)]
    g = DRModels._ratio_closure(1, [1, 2, 3])
    val = g(θ)
    @test val > 0 && isfinite(val)
    g2 = DRModels._ratio_closure(2, [2, 3]; resid = 0)
    @test isfinite(g2(θ))
end

@testset "cov 1125n6: twin floor / re_sd readers" begin
    V = fill(NaN, 3, 3)
    V[1, 1] = 0.01; V[2, 2] = 0.01
    fit = DrmFit(Poisson(), [:mu => 1:2, :resd => 3:3],
                 [:mu => ["(Intercept)", "x"], :resd => ["g"]],
                 [-0.1, 0.2, -8.0 + 1e-4], V, -5.0, 3, true,
                 Dict(:mu => [1.0]), Dict(:mu => [0.0]), Dict{Symbol,Vector{Float64}}())
    coords = DRModels._twin_floor_fixed_coords(fit)
    @test coords == [3]
    fit2 = DrmFit(Gaussian(), [:mu => 1:1], [:mu => ["(Intercept)"],],
                  [0.0], [0.1;;], 10.0, 1, true,
                  Dict(:mu => [1.0]), Dict(:mu => [0.0]), Dict(:rho12 => [0.5]))
    @test DRModels._twin_rho12_values(fit2) == [0.5]
    bare = DrmFit(Poisson(), [:mu => 1:1], [:mu => ["(Intercept)"],],
                  [0.0], [0.1;;], -2.0, 5, true,
                  Dict(:mu => ones(5)), Dict(:mu => zeros(5)), Dict{Symbol,Vector{Float64}}())
    @test isempty(DRModels._twin_floor_fixed_coords(bare))
end

@testset "cov 1125n6: mixed_family postfit accessors" begin
    fit = (; converged = true, loglik = -12.5, iterations = 5,
           β1 = [0.1, -0.2], β2 = [0.3], βσ1 = Float64[], βσ2 = [log(2.0)],
           λ1 = 0.8, λ2 = 0.6, rho_latent = 0.25,
           rho_ci_wald = (0.0, 0.5), rho_ci_profile = (NaN, NaN), rho_ci_boot = (NaN, NaN),
           fam1 = Gaussian(), fam2 = Poisson())
    tbl = DRModels.mf_coef(fit)
    @test length(tbl.term) == length(tbl.estimate)
    @test count(==( :shared), tbl.axis) >= 3
    @test DRModels.mf_aic(fit) ≈ -2 * fit.loglik + 2 * DRModels._mf_nparams(fit) rtol = 1e-12
    @test DRModels.mf_bic(fit; nobs = 20) > DRModels.mf_aic(fit)
    X1 = ones(5, 2); X2 = ones(5, 1)
    mu = DRModels.mf_fitted(fit, X1, X2)
    @test length(mu.mu1) == 5 && length(mu.mu2) == 5
    @test all(mu.mu2 .> 0)
    io = IOBuffer()
    DRModels.mf_summary(fit; nobs = 5, io)
    @test occursin("Cross-family", String(take!(io)))
end

@testset "cov 1125n6: poly orthogonal basis" begin
    x = collect(range(-1, 1; length = 8))
    pt = DRModels._poly_fit(x, :x, 2)
    d = (x = x,)
    Z = DRModels.modelcols(pt, d)
    @test size(Z) == (8, 2)
    @test rank(Z) == 2
    q, _ = qr(Z)
    @test maximum(abs, q' * q - I) < 1e-10
end