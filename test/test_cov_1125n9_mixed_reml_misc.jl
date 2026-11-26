# cov lane 1125n9 — mixed-family kernels, REML φ layout, separation algebra, heritability clamps.
using DRModels
using Test, LinearAlgebra, Random, Statistics, Distributions

@testset "cov 1125n9: _mf_disp_init and _mf_obs_ll" begin
    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    @test isfinite(DRModels._mf_disp_init(DRModels.Gaussian(), y))
    @test isfinite(DRModels._mf_disp_init(DRModels.NegBinomial2(), y))
    @test isfinite(DRModels._mf_disp_init(DRModels.Beta(), [0.2, 0.3, 0.5, 0.7]))
    @test isfinite(DRModels._mf_disp_init(DRModels.Gamma(), y))
    ll_p = DRModels._mf_obs_ll(DRModels.Poisson(), 0.2, 3.0, 1.0, 1.0)
    @test ll_p < 0
    ll_b = DRModels._mf_obs_ll(DRModels.Binomial(), 0.0, 2.0, 4.0, 1.0)
    @test isfinite(ll_b)
    ll_nb = DRModels._mf_obs_ll(DRModels.NegBinomial2(), 0.0, 5.0, 1.0, 0.5)
    @test isfinite(ll_nb)
    @test DRModels._mf_mean(DRModels.Gamma(), 0.1) ≈ exp(0.1)
end

@testset "cov 1125n9: _mf_rand smoke" begin
    rng = MersenneTwister(1125)
    for fam in (DRModels.Poisson(), DRModels.Binomial(), DRModels.Beta(), DRModels.Gamma())
        y = DRModels._mf_rand(fam, 0.0, 1.0, 1.0, rng)
        @test isfinite(y) || y isa Integer
    end
    nb = DRModels._mf_rand(DRModels.NegBinomial2(), 0.0, 1.0, 0.8, rng)
    @test nb >= 0
end

@testset "cov 1125n9: reml_q4 phi pack/unpack" begin
    p = 6
    phy = random_balanced_tree(p; branch_length = 0.2)
    n = p
    y1 = randn(n); y2 = randn(n)
    X = hcat(ones(n), randn(n))
    prob, Q = make_problem(phy, y1, y2, X, X, X, X, X)
    @test DRModels.phi_len(prob) == size(prob.Xr, 2) + 10
    w = DRModels.phi_widths(prob)
    @test w == (size(prob.Xs1, 2), size(prob.Xs2, 2), size(prob.Xr, 2))
    kr = size(prob.Xr, 2)
    ρ = zeros(kr); Λ = DRModels.lc_to_Λ(randn(10))
    φ = DRModels.pack_phi(prob, ρ, Λ)
    ρ2, lc = DRModels.unpack_phi(prob, φ)
    @test ρ2 ≈ ρ
    @test DRModels.lc_to_Λ(lc) ≈ Λ rtol = 1e-10
    Xμ = DRModels.build_Xmu_lifted(prob)
    @test size(Xμ, 1) == 4 * prob.n_total
end

@testset "cov 1125n9: heritability ratio closure and clamp" begin
    g = DRModels._ratio_closure(1, [1, 2])
    θ = [log(2.0), log(3.0)]
    @test g(θ) ≈ 4 / 13 rtol = 1e-12
    ci = (lower = -0.1, upper = 1.2)
    c2 = DRModels._clamp01_ci(ci)
    @test c2.lower == 0.0 && c2.upper == 1.0
    @test DRModels._clamp01(1.5) == 1.0
end

@testset "cov 1125n9: _sep_lstsq and _sep_constraints" begin
    X = [1.0 0.0; 1.0 1.0; 0.0 1.0]
    s = [1.0, 0.0, 1.0]
    ntr = [1.0, 1.0, 1.0]
    A, keep, alias, coef = DRModels._sep_constraints(X, s, ntr)
    @test length(keep) >= 1
    z = DRModels._sep_lstsq(A, ones(size(A, 1)))
    @test length(z) == size(A, 2)
    det = DRModels._detect_separation(X, [0.0, 0.0, 0.0], ones(3))
    @test det.conclusive
end

@testset "cov 1125n9: _twin_floor_fixed_coords" begin
    V = fill(0.0, 3, 3)
    V[1, 1] = 0.01; V[2, 2] = 0.01
    V[3, :] .= NaN; V[:, 3] .= NaN
    fit = DrmFit(Gaussian(), [:mu => 1:2, :resd => 3:3],
                 [:mu => ["(Intercept)", "x"], :resd => ["g"]],
                 [0.0, 0.0, -8.0 + 5e-4], V, -1.0, 10, true,
                 Dict(:mu => ones(10)), Dict(:mu => ones(10)), Dict(:sigma => ones(10)))
    fixed = DRModels._twin_floor_fixed_coords(fit)
    @test fixed == [3]
    plain = DrmFit(fit.family, fit.blocks, fit.coefnames, [0.0, 0.0, 0.0], V,
                   fit.loglik, fit.nobs, fit.converged, fit.means, fit.obs, fit.scales)
    @test isempty(DRModels._twin_floor_fixed_coords(plain))
end