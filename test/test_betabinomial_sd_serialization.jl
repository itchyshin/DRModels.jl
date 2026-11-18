using DRModels, Test, Random, Serialization
import Distributions

@testset "BetaBinomial SD fit survives serialization" begin
    rng = MersenneTwister(20261022)
    G, m, trials = 32, 12, 20
    g = repeat(1:G, inner = m)
    zgroup = repeat([-1.0, 1.0], inner = G ÷ 2)
    z = zgroup[g]
    x = randn(rng, length(g))
    alpha = [-1.1, 0.45]
    random_intercept = exp.(alpha[1] .+ alpha[2] .* zgroup) .* randn(rng, G)
    eta = 0.15 .+ 0.3 .* x .+ random_intercept[g]
    mu = 1 ./ (1 .+ exp.(-eta))
    precision = 1 / 0.36^2
    successes = [rand(rng, Distributions.BetaBinomial(
        trials, mu[i] * precision, (1 - mu[i]) * precision))
        for i in eachindex(mu)]
    data = (; successes = Float64.(successes),
            failures = Float64.(trials .- successes), x, z, g)

    formula = bf(@formula(cbind(successes, failures) ~ x + (1 | g)),
                 @formula(sigma ~ 1), @formula(sd(g) ~ z))
    fit = drm(formula, BetaBinomial(); data)
    @test fit.converged

    path = tempname()
    try
        open(path, "w") do io
            serialize(io, fit)
        end
        reloaded = open(deserialize, path)

        @test coef(reloaded, :sd) ≈ coef(fit, :sd) atol = 1e-12 rtol = 1e-12
        @test reloaded.scales[:sd_group] ≈ fit.scales[:sd_group] atol = 1e-12
        @test predict_parameters(reloaded, data)[:sd] ≈
              predict_parameters(fit, data)[:sd] atol = 1e-12
        @test predict_parameters(reloaded, data; type = :link)[:sd] ≈
              predict_parameters(fit, data; type = :link)[:sd] atol = 1e-12

        rows = reverse(1:2:length(g))
        newdata = (; g = data.g[rows], x = data.x[rows], z = data.z[rows])
        full_response = predict_parameters(reloaded, data)[:sd]
        full_link = predict_parameters(reloaded, data; type = :link)[:sd]
        @test predict_parameters(reloaded, newdata)[:sd] ≈
              full_response[rows] atol = 1e-12
        @test predict_parameters(reloaded, newdata; type = :link)[:sd] ≈
              full_link[rows] atol = 1e-12
    finally
        rm(path; force = true)
    end
end
