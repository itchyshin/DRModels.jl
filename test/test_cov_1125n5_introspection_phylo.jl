# test_cov_1125n5_introspection_phylo.jl — bridge diagnostics + phylo penalty (src/introspection.jl, phylo_penalty.jl).
using DRModels
using Test, Random, LinearAlgebra

@testset "cov 1125n5: _bridge_optimizer_string" begin
    @test DRModels._bridge_optimizer_string(:lbfgs) == "Optim.LBFGS"
    @test DRModels._bridge_optimizer_string(:bfgs) == "Optim.BFGS"
    @test DRModels._bridge_optimizer_string(:newton) == "Optim.Newton"
    @test ismissing(DRModels._bridge_optimizer_string(:nelder_mead))
end

@testset "cov 1125n5: bridge_diagnostics on Gaussian ML fit" begin
    Random.seed!(112511)
    n = 50
    x = randn(n)
    y = 0.1 .+ x .+ randn(n) .* 0.4
    fit = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
    d = bridge_diagnostics(fit)
    @test d.converged == fit.converged
    @test d.vcov_complete == all(isfinite, fit.vcov)
    @test d.integrator == fit.marginal
    @test d.iterations == niterations(fit)
    @test d.penalized_map === (fit.estim_method === :MAP)
    @test d.boundary isa Vector{Int}
    @test :route in propertynames(d)
    @test :optimizer in propertynames(d)
end

@testset "cov 1125n5: drm_phylo_penalty construction" begin
    pen = drm_phylo_penalty(sd_u = 2.0, sd_alpha = 0.05)
    @test pen.rate ≈ -log(0.05) / 2.0
    @test pen.cor_sd === nothing
    pen2 = drm_phylo_penalty(sd_u = 1.0, sd_alpha = 0.1, cor_sd = 0.5)
    @test pen2.cor_sd == 0.5
    @test_throws ArgumentError drm_phylo_penalty(sd_u = -1.0, sd_alpha = 0.05)
    @test_throws ArgumentError drm_phylo_penalty(sd_u = 1.0, sd_alpha = 1.5)
    @test_throws ArgumentError drm_phylo_penalty(sd_u = 1.0, sd_alpha = 0.05, cor_sd = 0.0)
end

@testset "cov 1125n5: _phylo_pen_sd and _phylo_pen_cor" begin
    rate = -log(0.05) / 1.0
    val, dv = DRModels._phylo_pen_sd(rate, 0.0)
    @test isfinite(val) && isfinite(dv)
    @test dv ≈ rate - 1.0 atol = 1e-12
    vc, dz = DRModels._phylo_pen_cor(0.5, 0.2)
    @test vc > 0
    @test dz ≈ 0.2 / 0.25
    g = zeros(3)
    pen = drm_phylo_penalty(sd_u = 1.0, sd_alpha = 0.05)
    vsum = DRModels._phylo_pen_apply_single!(g, pen, [-0.5, 0.0, 0.1], 1)
    @test isfinite(vsum)
    @test g[1] != 0.0
    @test_throws PhyloCorPenaltyNeedsTwoSD DRModels._phylo_pen_apply_single!(
        nothing, drm_phylo_penalty(cor_sd = 0.5), [0.0], 1)
    g2 = zeros(4)
    v2 = DRModels._phylo_pen_apply_separate!(g2, pen, [-0.2, 0.1, 0.3, 0.0], 1, 3)
    @test isfinite(v2)
    @test g2[1] != 0.0 && g2[3] != 0.0
end

@testset "cov 1125n5: _phylo_pen_apply_coupled!" begin
    pen = drm_phylo_penalty(sd_u = 1.0, sd_alpha = 0.05, cor_sd = 0.4)
    θ = [-0.3, 0.15, -0.1]  # logL11, L21, logL22
    g = zeros(3)
    val = DRModels._phylo_pen_apply_coupled!(copy(g), pen, θ, 1)
    @test isfinite(val)
    DRModels._phylo_pen_apply_coupled!(g, pen, θ, 1)
    @test all(isfinite, g)
    @test any(!=(0.0), g)
end

@testset "cov 1125n5: _unitfree_converged and _scale_runoff" begin
    H = [4.0 0.1; 0.1 3.0]
    g = [1e-4, -2e-4]
    @test DRModels._unitfree_converged(H, g)
    @test !DRModels._unitfree_converged(H, [10.0, 10.0])
    σ = Dict(:sigma => [0.1, 0.2, 0.15, 0.18])
    @test !DRModels._scale_runoff(σ)
    σwide = Dict(:sigma => exp.([0.0, 10.0]))
    @test DRModels._scale_runoff(σwide)
    @test !DRModels._scale_runoff(Dict(:mu => [1.0]))
    @test DRModels._scale_runoff(Dict(:sigma => [NaN, 1.0]))
end