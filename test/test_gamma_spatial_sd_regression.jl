using DRModels, Test, Random, LinearAlgebra, Statistics
import Distributions

function _gamma_spatial_sd_fixture(; G = 48, m = 12, seed = 20261022)
    rng = MersenneTwister(seed)
    coords = hcat(mod.(0:(G - 1), 5), div.(0:(G - 1), 5))
    distances = [sqrt(sum(abs2, coords[i, :] .- coords[j, :]))
                 for i in 1:G, j in 1:G]
    range = sum(distances) / (G^2 - G)
    covariance = exp.(-distances ./ range) + 1e-8I
    group = repeat(1:G, inner = m)
    zgroup = [zeros(G ÷ 2); ones(G ÷ 2)][randperm(rng, G)]
    z = zgroup[group]
    x = randn(rng, G * m)
    alpha = [-0.7, 0.45]
    latent = cholesky(Symmetric(covariance)).L * randn(rng, G)
    eta = 0.25 .+ 0.2 .* x .+
          exp.(alpha[1] .+ alpha[2] .* zgroup)[group] .* latent[group]
    sigma = 0.32
    shape = 1 / sigma^2
    y = [rand(rng, Distributions.Gamma(shape, exp(v) / shape)) for v in eta]
    return (; data = (; y = Float64.(y), x, z, g = group), coords,
            covariance, group, zgroup, alpha, sigma)
end

@testset "Gamma spatial random-effect SD regression" begin
    fixture = _gamma_spatial_sd_fixture()
    data, coords = fixture.data, fixture.coords
    formula = bf(@formula(y ~ x + spatial(1 | g)),
                 @formula(sigma ~ 1),
                 @formula(sd(g, spatial) ~ z))
    fit = drm(formula, Gamma(); data, coords)

    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sigma, :sd]
    @test coef(fit, :sd)[2] > 0
    @test abs(coef(fit, :sd)[2] - fixture.alpha[2]) < 0.7
    @test fit.scales[:sd_group] ≈ exp.(hcat(ones(length(fixture.zgroup)),
                                             fixture.zgroup) * coef(fit, :sd))
    @test predict_parameters(fit, data)[:sd] ≈ fit.scales[:sd]
    @test all(isfinite, predict_parameters(fit, data; se = true)[:sd].se)

    constant_fit = drm(bf(@formula(y ~ x + spatial(1 | g)),
                          @formula(sigma ~ 1),
                          @formula(sd(g, spatial) ~ 1)),
                       Gamma(); data, coords, se = false)
    base_fit = drm(bf(@formula(y ~ x + spatial(1 | g)),
                      @formula(sigma ~ 1)), Gamma(); data, coords, se = false)
    @test constant_fit.loglik ≈ base_fit.loglik atol = 3e-5
    @test coef(constant_fit, :mu) ≈ coef(base_fit, :mu) atol = 1e-4
    @test coef(constant_fit, :sigma) ≈ coef(base_fit, :sigma) atol = 1e-4
end
