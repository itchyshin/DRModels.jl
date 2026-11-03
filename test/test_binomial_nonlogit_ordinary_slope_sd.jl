using DRModels, Test, Random
import Distributions

@testset "Binomial probit/cloglog ordinary random-slope SD" begin
    for (link, seed_offset) in ((:probit, 0), (:cloglog, 1))
        rng = MersenneTwister(20261030 + seed_offset)
        G, m = 100, 30
        g = repeat(1:G, inner = m)
        z_group = repeat([-1.0, 1.0], inner = G ÷ 2)
        z = z_group[g]
        x = repeat(range(-1.5, 1.5; length = m), G)
        alpha = [-0.65, 0.5]
        slope_sd = exp.(alpha[1] .+ alpha[2] .* z_group)
        b = slope_sd .* randn(rng, G)
        eta = -0.1 .+ 0.3 .* x .+ x .* b[g]
        probability = if link === :probit
            Distributions.cdf.(Distributions.Normal(), eta)
        else
            -expm1.(-exp.(eta))
        end
        y = Float64.(rand.(Ref(rng), Distributions.Bernoulli.(probability)))
        data = (; y, x, z, g)
        formula = bf(@formula(y ~ x + (0 + x | g)),
                     @formula(sd(g, coef(x)) ~ z))

        fit = drm(formula, Binomial(link = link); data, nq = 9, se = false)
        sd_block = last(fit.blocks).first
        sd_coef = coef(fit, sd_block)
        predicted = predict_parameters(fit, data; se = false)[sd_block]
        simulator = DRModels._marginal_simulator_build(fit, data)
        sim1 = simulator(MersenneTwister(20261032 + seed_offset))
        sim2 = simulator(MersenneTwister(20261032 + seed_offset))

        @test fit.converged
        @test fit.family.link === link
        @test length(sd_coef) == 2
        @test sd_coef[2] > 0
        @test abs(sd_coef[2] - alpha[2]) < 0.6
        @test predicted ≈ exp.(hcat(ones(G), z_group) * sd_coef)[g]
        @test length(sim1) == length(y)
        @test sim1 == sim2
        @test all(value -> value == 0.0 || value == 1.0, sim1)

        G0, m0 = 48, 14
        g0 = repeat(1:G0, inner = m0)
        z0_group = repeat([-1.0, 1.0], inner = G0 ÷ 2)
        z0 = z0_group[g0]
        x0 = randn(rng, G0 * m0)
        b0 = exp.(-0.65 .+ 0.5 .* z0_group) .* randn(rng, G0)
        eta0 = -0.1 .+ 0.3 .* x0 .+ b0[g0]
        probability0 = if link === :probit
            Distributions.cdf.(Distributions.Normal(), eta0)
        else
            -expm1.(-exp.(eta0))
        end
        y0 = Float64.(rand.(Ref(rng), Distributions.Bernoulli.(probability0)))
        data0 = (; y = y0, x = x0, z = z0, g = g0)
        intercept_formula = bf(@formula(y ~ x + (1 | g)), @formula(sd(g) ~ z))
        fit0 = drm(intercept_formula, Binomial(link = link);
            data = data0, nq = 9, se = false)
        intercept_block = last(fit0.blocks).first
        intercept_coef = coef(fit0, intercept_block)
        predicted0 = predict_parameters(fit0, data0; se = false)[intercept_block]
        simulator0 = DRModels._marginal_simulator_build(fit0, data0)
        sim0 = simulator0(MersenneTwister(20261034 + seed_offset))

        @test fit0.converged
        @test fit0.family.link === link
        @test intercept_coef[2] > 0
        @test abs(intercept_coef[2] - 0.5) < 0.8
        @test predicted0 ≈ exp.(hcat(ones(G0), z0_group) * intercept_coef)[g0]
        @test length(sim0) == length(y0)
        @test all(value -> value == 0.0 || value == 1.0, sim0)
    end
end
