module TestCumlogitSDRegression

using DRModels
using QuadGK
using Random
using Test

@testset "CumulativeLogit random-intercept SD regression" begin
    rng = MersenneTwister(2041)
    G, m = 48, 24
    labels = repeat(["site_$(g)" for g in 1:G], inner = m)
    z_group = repeat(vcat(zeros(G ÷ 2), ones(G ÷ 2)), inner = m)
    z = Float64.(z_group)
    x = randn(rng, G * m)
    cuts = [-1.0, 0.0, 1.0]
    b = [randn(rng) * exp(-0.45 + 0.45 * z_group[(g - 1) * m + 1]) for g in 1:G]
    y = Vector{Float64}(undef, G * m)
    for i in eachindex(y)
        η = 0.35 * x[i] + b[div(i - 1, m) + 1]
        p = (1 / (1 + exp(-(cuts[1] - η))),
             1 / (1 + exp(-(cuts[2] - η))),
             1 / (1 + exp(-(cuts[3] - η))))
        u = rand(rng)
        y[i] = u <= p[1] ? 1.0 : u <= p[2] ? 2.0 : u <= p[3] ? 3.0 : 4.0
    end
    data = (; y, x, z, id = labels)

    fit = drm(bf(@formula(y ~ x + (1 | id)), @formula(sd(id) ~ z)),
              CumulativeLogit(); data)

    @test isfinite(loglik(fit))
    α = coef(fit, :sd)
    @test length(α) == 2
    @test α[2] > 0
    @test length(fit.scales[:sd_group]) == G
    @test fit.scales[:sd] ≈ fit.scales[:sd_group][round.(Int, fit.scales[:sd_group_index])]
    @test predict_parameters(fit, data)[:sd] ≈ fit.scales[:sd]
    @test length(predict(fit, data)) == G * m
    simulator = DRModels._marginal_simulator_build(fit, data)
    @test simulator !== nothing
    ysim = simulator(MersenneTwister(2042))
    @test length(ysim) == G * m
    @test all(y -> y in 1:4, ysim)

    fit_constant = drm(bf(@formula(y ~ x + (1 | id)), @formula(sd(id) ~ 1)),
                       CumulativeLogit(); data)
    fit_base = drm(bf(@formula(y ~ x + (1 | id))), CumulativeLogit(); data)
    @test loglik(fit_constant) ≈ loglik(fit_base) atol = 1e-5
    @test coef(fit_constant, :sd)[1] ≈ coef(fit_base, :resd)[1] atol = 1e-4

    bad_data = merge(data, (; z = randn(rng, G * m)))
    @test_throws ArgumentError drm(
        bf(@formula(y ~ x + (1 | id)), @formula(sd(id) ~ z)),
        CumulativeLogit(); data = bad_data)

    incomplete_data = merge(data, (; y = [NaN; y[2:end]]))
    @test_throws ArgumentError drm(
        bf(@formula(y ~ x + (1 | id)), @formula(sd(id) ~ z)),
        CumulativeLogit(); data = incomplete_data)
end

@testset "CumulativeLogit random-slope SD regression" begin
    rng = MersenneTwister(2043)
    G, m = 64, 32
    z_group = repeat(vcat(zeros(G ÷ 2), ones(G ÷ 2)), inner = m)
    z = Float64.(z_group)
    id = repeat(["site_$(g)" for g in 1:G], inner = m)
    x = randn(rng, G * m)
    cuts = [-1.0, 0.0, 1.0]
    b = [randn(rng) * exp(-0.55 + 0.55 * z_group[(g - 1) * m + 1]) for g in 1:G]
    y = Vector{Float64}(undef, G * m)
    for i in eachindex(y)
        η = 0.35 * x[i] + x[i] * b[div(i - 1, m) + 1]
        cdf = (1 / (1 + exp(-(cuts[1] - η))),
               1 / (1 + exp(-(cuts[2] - η))),
               1 / (1 + exp(-(cuts[3] - η))))
        u = rand(rng)
        y[i] = u <= cdf[1] ? 1.0 : u <= cdf[2] ? 2.0 : u <= cdf[3] ? 3.0 : 4.0
    end
    data = (; y, x, z, id)

    fit = drm(bf(@formula(y ~ x + (0 + x | id)),
                 @formula(sd(id) ~ z)), CumulativeLogit(); data)
    @test isfinite(loglik(fit))
    α = coef(fit, :sd)
    @test length(α) == 2
    @test α[2] > 0
    @test length(fit.scales[:sd_group]) == G
    @test predict_parameters(fit, data)[:sd] ≈ fit.scales[:sd]
    fit_selected = drm(bf(@formula(y ~ x + (0 + x | id)),
                         @formula(sd(id, coef(x)) ~ z)),
                       CumulativeLogit(); data)
    @test loglik(fit_selected) ≈ loglik(fit) atol = 1e-7
    @test coef(fit_selected, :sd) ≈ coef(fit, :sd) atol = 1e-5
    sd_response = predict_parameters(fit_selected, data)[:sd]
    sd_link = predict_parameters(fit_selected, data; type = :link)[:sd]
    sd_link_se = predict_parameters(fit_selected, data; type = :link, se = true)[:sd]
    @test sd_link ≈ log.(sd_response)
    @test sd_link_se.value ≈ sd_link
    @test length(sd_link_se.se) == G * m
    @test all(isfinite, sd_link_se.se)
    conditional_draw = simulate(fit_selected; rng = MersenneTwister(20431))
    @test length(conditional_draw) == G * m
    @test all(y -> y in 1:4, conditional_draw)
    cuts_hat = fit.scales[:ordinal_cuts]
    eta_fixed = predict(fit, data; type = :link)
    logistic(v) = 1 / (1 + exp(-clamp(v, -35.0, 35.0)))
    oracle_loglik = 0.0
    group_rows = [((g - 1) * m + 1):(g * m) for g in 1:G]
    for g in 1:G
        integrand(q) = begin
            eta = eta_fixed[group_rows[g]] .+ x[group_rows[g]] .* fit.scales[:sd_group][g] .* q
            log_density = -0.5 * q^2 - 0.5 * log(2π)
            for (j, i) in enumerate(group_rows[g])
                c1, c2, c3 = logistic(cuts_hat[1] - eta[j]),
                             logistic(cuts_hat[2] - eta[j]),
                             logistic(cuts_hat[3] - eta[j])
                p = y[i] == 1 ? c1 : y[i] == 2 ? c2 - c1 :
                    y[i] == 3 ? c3 - c2 : 1 - c3
                log_density += log(max(p, floatmin(Float64)))
            end
            exp(log_density)
        end
        oracle_loglik += log(first(quadgk(integrand, -9.0, 9.0; rtol = 1e-9)))
    end
    @test loglik(fit) ≈ oracle_loglik atol = 2e-3
    @test length(predict(fit, data)) == G * m
    simulator = DRModels._marginal_simulator_build(fit, data)
    @test simulator !== nothing
    ysim = simulator(MersenneTwister(2044))
    @test length(ysim) == G * m
    @test all(y -> y in 1:4, ysim)
    expected_rng = MersenneTwister(2044)
    u = randn(expected_rng, G)
    eta = predict(fit, data; type = :link) .+
          x .* (fit.scales[:sd_group] .* u)[repeat(1:G, inner = m)]
    expected = Vector{Float64}(undef, G * m)
    for i in eachindex(expected)
        q = rand(expected_rng)
        cdf = (1 / (1 + exp(-(cuts_hat[1] - eta[i]))),
               1 / (1 + exp(-(cuts_hat[2] - eta[i]))),
               1 / (1 + exp(-(cuts_hat[3] - eta[i]))))
        expected[i] = q <= cdf[1] ? 1.0 : q <= cdf[2] ? 2.0 : q <= cdf[3] ? 3.0 : 4.0
    end
    @test ysim == expected
    boot = bootstrap_result(fit_selected; data, B = 1, rng = MersenneTwister(2045),
                            check_converged = true)
    @test boot.attempted == 1
    @test boot.used == 1
    @test boot.failed == 0

    fit_constant = drm(bf(@formula(y ~ x + (0 + x | id)),
                          @formula(sd(id, coef(x)) ~ 1)), CumulativeLogit(); data)
    fit_base = drm(bf(@formula(y ~ x + (0 + x | id))), CumulativeLogit(); data)
    @test loglik(fit_constant) ≈ loglik(fit_base) atol = 1e-5
    @test coef(fit_constant, :sd)[1] ≈ coef(fit_base, :resd)[1] atol = 1e-4
    @test_throws ArgumentError drm(
        bf(@formula(y ~ x + (0 + x | id)),
           @formula(sd(id, coef(intercept)) ~ z)), CumulativeLogit(); data)
end

end # module
