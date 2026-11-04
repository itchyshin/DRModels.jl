using DRModels
using Test, LinearAlgebra, Random

@testset "Binomial related-matrix plus IID SD regression" begin
    rng = MersenneTwister(20261029)
    G, H, m = 40, 40, 4
    rel_group = repeat(1:G, inner = H * m)
    iid_group = repeat(repeat(1:H, inner = m), outer = G)
    x = repeat(range(-1.0, 1.0; length = m), outer = G * H)
    rel_z_group = rand(rng, [-1.0, 1.0], G)
    iid_w_group = rand(rng, [-1.0, 1.0], H)
    rel_z = rel_z_group[rel_group]
    iid_w = iid_w_group[iid_group]
    K = [0.45^abs(i - j) for i in 1:G, j in 1:G]
    rel_sd = exp.(-0.65 .+ 0.35 .* rel_z_group)
    iid_sd = exp.(-0.8 .- 0.3 .* iid_w_group)
    rel_effect = cholesky(Symmetric(K)).L * randn(rng, G) .* rel_sd
    iid_effect = randn(rng, H) .* iid_sd
    eta = -0.1 .+ 0.25 .* x .+ rel_effect[rel_group] .+ iid_effect[iid_group]
    y = Float64[rand(rng, DRModels.Distributions.Bernoulli(p))
                for p in 1 ./ (1 .+ exp.(-eta))]
    data = (; y, x, rel_z, iid_w, rel_group, iid_group)
    formula = bf(
        @formula(y ~ x + (1 | iid_group) + relmat(1 | rel_group)),
        @formula(sd(iid_group) ~ iid_w),
        @formula(sd(rel_group, relmat) ~ rel_z),
    )

    fit = drm(formula, DRModels.Binomial(); data, K, se = false)
    @test fit.converged
    targets = DRModels._sd_target_parts(formula)
    rel_target = only(filter(item -> item.target.level === :relmat, targets))
    iid_target = only(filter(item -> item.target.level === :iid, targets))
    rel_block = DRModels._sd_target_block(rel_target.target; disambiguate = true)
    iid_block = DRModels._sd_target_block(iid_target.target; disambiguate = true)
    @test length(coef(fit, iid_block)) == 2
    @test length(coef(fit, rel_block)) == 2
    @test coef(fit, iid_block)[2] < 0
    @test coef(fit, rel_block)[2] > 0
    predicted = predict_parameters(fit, data; se = false)
    @test all(isfinite, predicted[iid_block])
    @test all(isfinite, predicted[rel_block])
    simulated = DRModels._marginal_simulator_build(fit, data; K)(rng)
    @test length(simulated) == length(y)
    @test all(in((0.0, 1.0)), simulated)
end
