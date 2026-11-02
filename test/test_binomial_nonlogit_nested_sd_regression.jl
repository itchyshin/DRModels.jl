using DRModels, Test, Random
import Distributions

@testset "Binomial non-logit nested independent SD regressions" begin
    G, children_per_group, m = 18, 4, 12
    H = G * children_per_group
    g = repeat(1:G, inner = children_per_group * m)
    h = repeat(1:H, inner = m)
    wg = repeat([-1.0, 1.0, -1.0], inner = G ÷ 3)
    vh = repeat([-1.0, 1.0], inner = H ÷ 2)
    w, v = wg[g], vh[h]

    for (link, seed_offset) in ((:probit, 0), (:cloglog, 1))
        rng = MersenneTwister(20261029 + seed_offset)
        x = randn(rng, length(g))
        z = randn(rng, length(g))

        for (effect, alpha_g, alpha_h) in (
            (:intercept, [log(0.45), 0.30], [log(0.50), -0.25]),
            (:slope, [log(0.50), 0.25], [log(0.40), -0.20]),
        )
            tau_g = exp.(alpha_g[1] .+ alpha_g[2] .* wg)
            tau_h = exp.(alpha_h[1] .+ alpha_h[2] .* vh)
            b_g = tau_g .* randn(rng, G)
            b_h = tau_h .* randn(rng, H)
            eta = if effect === :intercept
                -0.20 .+ 0.35 .* x .+ b_g[g] .+ b_h[h]
            else
                -0.20 .+ 0.30 .* x .+ x .* b_g[g] .+ z .* b_h[h]
            end
            probability = link === :probit ?
                Distributions.cdf.(Distributions.Normal(), eta) : -expm1.(-exp.(eta))
            y = Float64.(rand.(Ref(rng), Distributions.Bernoulli.(probability)))
            data = (; y, x, z, g, h, w, v)
            formula = if effect === :intercept
                bf(@formula(y ~ x + (1 | g) + (1 | h)),
                   @formula(sd(g) ~ w), @formula(sd(h) ~ v))
            else
                bf(@formula(y ~ x + (0 + x | g) + (0 + z | h)),
                   @formula(sd(g) ~ w), @formula(sd(h) ~ v))
            end
            fit = drm(formula, Binomial(link = link); data, se = false)
            targets = DRModels._sd_target_blocks(DRModels._sd_target_parts(formula))

            @test fit.converged
            @test fit.family.link === link
            @test length(targets) == 2
            @test abs(coef(fit, targets[1])[2] - alpha_g[2]) < 0.55
            @test abs(coef(fit, targets[2])[2] - alpha_h[2]) < 0.55
            predictions = predict_parameters(fit, data)
            @test length(predictions[targets[1]]) == length(y)
            @test length(predictions[targets[2]]) == length(y)
            @test all(isfinite, predictions[targets[1]])
            @test all(isfinite, predictions[targets[2]])
            simulated = simulate(fit; rng = MersenneTwister(20261031 + seed_offset))
            @test length(simulated) == length(y)
            @test all(value -> value == 0.0 || value == 1.0, simulated)
        end
    end
end
