# R-bridge inference for the bivariate q=4 phylo fit (Ayumi #2): drm_bridge_inference
# with method="bootstrap" returns the four among-axis SD CIs as a multi-row payload
# (param/estimate/lower/upper as equal-length vectors → an R data.frame); profile is
# directed to bootstrap (the q4 boundary Hessian is singular).
using DRModels, Test, Random, LinearAlgebra

@testset "drm_bridge_inference — bivariate q4 among-axis SD CIs" begin
    Random.seed!(20260613)
    p, m = 20, 5
    phy = random_balanced_tree(p; branch_length = 0.3)
    C = sigma_phy_dense(phy; σ²_phy = 1.0)
    LC = cholesky(Symmetric(C)).L
    Z = randn(p, 4)
    Lmu = cholesky([0.64 0.30; 0.30 0.64]).L
    U = hcat(LC * Z[:, 1:2] * Lmu', LC * Z[:, 3] * 0.5, zeros(p))
    species = repeat(1:p, inner = m)
    n = length(species)
    x = randn(n)
    μ1 = 0.5 .+ 0.3 .* x .+ U[species, 1]
    μ2 = -0.2 .+ 0.4 .* x .+ U[species, 2]
    σ1 = exp.(-1.0 .+ U[species, 3])
    σ2 = exp.(-1.0 .+ U[species, 4])
    ρ = 0.3
    e1 = randn(n)
    e2 = ρ .* e1 .+ sqrt(1 - ρ^2) .* randn(n)
    dat = (; y1 = μ1 .+ σ1 .* e1, y2 = μ2 .+ σ2 .* e2, x, species)
    fml = Dict(
        :mu1 => "y1 ~ x + phylo(1 | species)",
        :mu2 => "y2 ~ x + phylo(1 | species)",
        :sigma1 => "sigma1 ~ 1 + phylo(1 | species)",
        :sigma2 => "sigma2 ~ 1 + phylo(1 | species)",
        :rho12 => "rho12 ~ 1",
    )

    fit_reml = DRModels.drm_bridge(; formula = fml, family = "biv_gaussian",
        data = dat, tree = phy, options = Dict("method" => "REML"))
    @test fit_reml["family"] == "biv_gaussian"
    @test isfinite(fit_reml["loglik"])
    @test all(isfinite, fit_reml["coefficients"])
    @test all(isnan, vec(fit_reml["vcov"]))

    res = DRModels.drm_bridge_inference(; formula = fml, family = "biv_gaussian",
        data = dat, tree = phy, method = "bootstrap", level = 0.90, B = 12,
        seed = 20260613)
    @test res["method"] == "bootstrap"
    @test res["multi"] === true
    @test res["param"] == ["sd_mu1", "sd_mu2", "sd_sigma1", "sd_sigma2"]
    @test length(res["estimate"]) == 4
    @test length(res["lower"]) == 4 && length(res["upper"]) == 4
    @test all(isfinite, res["estimate"])
    @test all(res["lower"] .<= res["upper"])
    @test all(res["lower"] .>= 0)          # SD scale
    @test res["used"] >= 6                 # 12 attempted; loose floor for CI robustness

    # PROFILE-likelihood CIs ARE available for the q4 among-axis SDs (hessian-free,
    # boundary-correct) — the headline strength, NOT a bootstrap fallback.
    res_p = DRModels.drm_bridge_inference(; formula = fml, family = "biv_gaussian",
        data = dat, tree = phy, method = "profile", level = 0.90)
    @test res_p["method"] == "profile"          # real profile, not redirected
    @test res_p["multi"] === true
    @test res_p["param"] == ["sd_mu1", "sd_mu2", "sd_sigma1", "sd_sigma2"]
    @test all(isfinite, res_p["estimate"])
    @test all(res_p["lower"] .>= 0)             # boundary-respecting lower bounds
    @test all(res_p["lower"] .<= res_p["estimate"] .+ 1e-8)
    @test all(res_p["estimate"] .<= res_p["upper"] .+ 1e-8)   # upper may be Inf on a flat axis
    @test all(isnan, res_p["std_error"])        # LR interval has no Wald SE

    # Wald is correctly unavailable for these boundary variance components.
    @test_throws ArgumentError DRModels.drm_bridge_inference(; formula = fml,
        family = "biv_gaussian", data = dat, tree = phy, method = "wald")
end
