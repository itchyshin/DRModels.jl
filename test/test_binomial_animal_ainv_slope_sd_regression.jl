using DRModels, Test, Random, LinearAlgebra, SparseArrays
import Distributions

function _binomial_animal_ainv_slope_sd_fixture(; G = 64, m = 32,
                                                 seed = 20261024)
    rng = MersenneTwister(seed)
    group_order = randperm(rng, G)
    group_index = repeat(1:G, inner = m)
    group = repeat(group_order, inner = m)
    z_group = ifelse.(randperm(rng, G) .<= G ÷ 2, -1.0, 1.0)
    z = repeat(z_group, inner = m)
    x_within = collect(range(-1.2, 1.2; length = m))
    x = repeat(x_within, outer = G)

    correlation = [0.35^abs(i - j) for i in 1:G, j in 1:G]
    marginal_sd = collect(range(0.7, 1.5; length = G))
    A = Diagonal(marginal_sd) * correlation * Diagonal(marginal_sd)
    alpha = [-0.45, 0.65]
    latent = cholesky(Symmetric(A)).L * randn(rng, G)
    group_sd = exp.(alpha[1] .+ alpha[2] .* z_group)
    eta = -0.1 .+ 0.3 .* x .+ x .* (group_sd .* latent)[group_index]
    y = Float64[rand(rng, Distributions.Bernoulli(1 / (1 + exp(-v))))
                for v in eta]
    return (; data = (; y, x, z, g = group), A, alpha, z_group,
            group_index)
end

@testset "Binomial animal random-slope SD accepts direct precision" begin
    fixture = _binomial_animal_ainv_slope_sd_fixture()
    data, A = fixture.data, fixture.A
    Ainv = inv(Symmetric(A))
    formula = bf(@formula(y ~ x + animal(0 + x | g)),
                 @formula(sd(g, animal) ~ z))

    covariance_fit = drm(formula, Binomial(); data, A, se = false)
    precision_fit = drm(formula, Binomial(); data, Ainv, se = false)

    @test covariance_fit.converged
    @test precision_fit.converged
    @test coef(precision_fit, :sd) ≈ coef(covariance_fit, :sd) atol = 5e-3
    @test precision_fit.loglik ≈ covariance_fit.loglik atol = 2e-5
    @test predict_parameters(precision_fit, data; se = false)[:sd] ≈
          predict_parameters(covariance_fit, data; se = false)[:sd] atol = 5e-3

    # Reproduce the simulator's precision-factor draw and row-level slope
    # loading, so this checks more than the response range.
    draw_seed = 20261025
    simulator = DRModels._marginal_simulator_build(precision_fit, data; Ainv)
    actual_draw = simulator(MersenneTwister(draw_seed))
    oracle_rng = MersenneTwister(draw_seed)
    qchol = cholesky(Symmetric(sparse(Ainv)))
    latent_draw = DRModels._binomial_animal_precision_draw(oracle_rng, qchol,
                                                           length(fixture.z_group))
    group_sd = exp.(hcat(ones(length(fixture.z_group)), fixture.z_group) *
                    coef(precision_fit, :sd))
    eta_fixed = predict(precision_fit, data; type = :link)
    mu_star = DRModels._mean_response(precision_fit.family,
        eta_fixed .+ data.x .* (group_sd .* latent_draw)[fixture.group_index])
    expected_draw = DRModels._simulate_once(precision_fit, oracle_rng; mu = mu_star)
    @test actual_draw == expected_draw

    boot = bootstrap_result(precision_fit; data, Ainv, B = 1,
        rng = MersenneTwister(20261026), failures = :skip)
    @test boot.attempted == 1
    @test boot.used == 1
    @test boot.failed == 0
    @test Set(row.param for row in boot.summary) == Set([:mu, :sd])
end
