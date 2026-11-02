using DRModels, Test, Random
import Distributions
using LinearAlgebra

function _nb2_animal_sd_fixture(alpha; G = 40, m = 24, seed = 20261022)
    rng = MersenneTwister(seed)
    g = repeat(1:G, inner = m)
    zgroup = repeat([-1.0, 1.0], inner = G ÷ 2)
    z = zgroup[g]
    x = randn(rng, length(g))
    R = 0.35 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    latent = exp.(alpha[1] .+ alpha[2] .* zgroup) .* (cholesky(Symmetric(R)).L * randn(rng, G))
    mu = exp.(0.35 .+ 0.25 .* x .+ latent[g])
    sigma = 0.42
    size = 1 / sigma^2
    y = [rand(rng, Distributions.NegativeBinomial(size, size / (size + value)))
         for value in mu]

    # Scramble observation order while retaining explicit group IDs. The
    # predictor-to-provider ordering checks below must follow IDs, not rows.
    order = randperm(rng, length(g))
    fit_levels = unique(g[order])
    return (; y = Float64.(y[order]), x = x[order], z = z[order], g = g[order]),
           R[fit_levels, fit_levels], zgroup[fit_levels], fit_levels, alpha
end

@testset "NB2 animal location-intercept SD regression" begin
    data, A, zgroup, fit_levels, alpha = _nb2_animal_sd_fixture([-0.65, 0.4])
    formula = bf(@formula(y ~ x + animal(1 | g)),
                 @formula(sigma ~ 1),
                 @formula(sd(g, animal) ~ z))
    fit = drm(formula, NegBinomial2(); data, A, se = false)

    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sigma, :sd]
    @test sign(coef(fit, :sd)[2]) == sign(alpha[2])
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.55
    expected_group_sd = exp.(hcat(ones(length(zgroup)), zgroup) * coef(fit, :sd))
    @test fit.scales[:sd_group] ≈ expected_group_sd
    group_index = [findfirst(==(id), fit_levels) for id in data.g]
    @test Int.(fit.scales[:sd_group_index]) == group_index
    @test predict_parameters(fit, data)[:sd] ≈ expected_group_sd[group_index]

    # The direct precision route must reproduce the covariance route for the
    # same unit-diagonal animal kernel, including the first-seen group order
    # induced by the scrambled observation rows above.
    precision_fit = drm(formula, NegBinomial2(); data, Ainv = inv(A), se = false)
    @test precision_fit.converged
    @test coef(precision_fit, :mu) ≈ coef(fit, :mu) atol = 1e-5
    @test coef(precision_fit, :sigma) ≈ coef(fit, :sigma) atol = 1e-5
    @test coef(precision_fit, :sd) ≈ coef(fit, :sd) atol = 1e-4
    @test precision_fit.loglik ≈ fit.loglik atol = 1e-4
    @test precision_fit.scales[:sd_group] ≈ fit.scales[:sd_group] atol = 1e-4
    @test predict_parameters(precision_fit, data)[:sd] ≈
          predict_parameters(fit, data)[:sd] atol = 1e-4
    @test_throws ArgumentError drm(formula, NegBinomial2(); data, A, Ainv = inv(A), se = false)
    @test_throws ArgumentError drm(formula, NegBinomial2(); data,
                                   Ainv = -inv(A), se = false)
    @test_throws ArgumentError drm(formula, NegBinomial2(); data,
                                   Ainv = inv(A)[1:end-1, 1:end-1], se = false)
    asymmetric_Ainv = inv(A)
    asymmetric_Ainv[1, 2] += 0.1
    @test_throws ArgumentError drm(formula, NegBinomial2(); data,
                                   Ainv = asymmetric_Ainv, se = false)

    constant_formula = bf(@formula(y ~ x + animal(1 | g)),
                          @formula(sigma ~ 1),
                          @formula(sd(g, animal) ~ 1))
    constant_fit = drm(constant_formula, NegBinomial2(); data, A, se = false)
    base_fit = drm(bf(@formula(y ~ x + animal(1 | g)),
                      @formula(sigma ~ 1)), NegBinomial2(); data, A, se = false)
    @test constant_fit.loglik ≈ base_fit.loglik atol = 1e-4
end
