using DRModels, Test, Random, LinearAlgebra, Distributions
using SpecialFunctions: loggamma
using Serialization

function _beta_animal_sd_dense_laplace(theta, data, Xmu)
    G = length(data.z_group)
    beta = theta[1:size(Xmu, 2)]
    sigma = theta[size(Xmu, 2) + 1]
    alpha = theta[(size(Xmu, 2) + 2):end]
    # A enters RAW (as native drmTMB), so the prior precision is inv(A).
    Q = Matrix(inv(Symmetric(data.A)))
    group_sd = exp.(hcat(ones(G), data.z_group) * alpha)
    phi = exp(-2sigma)
    function conditional(u)
        eta = clamp.(Xmu * beta .+ (group_sd .* u)[data.group_index], -30.0, 30.0)
        mu = 1 ./ (1 .+ exp.(-eta))
        value = 0.5 * dot(u, Q * u)
        for i in eachindex(data.y)
            a = mu[i] * phi
            b = (1 - mu[i]) * phi
            value -= loggamma(phi) - loggamma(a) - loggamma(b) +
                     (a - 1) * log(data.y[i]) + (b - 1) * log1p(-data.y[i])
        end
        value
    end
    modefit = DRModels.Optim.optimize(conditional, zeros(G),
        DRModels.Optim.LBFGS(), DRModels.Optim.Options(
            g_tol = 1e-9, iterations = 500); autodiff = :forward)
    mode = DRModels.Optim.minimizer(modefit)
    H = DRModels.ForwardDiff.hessian(conditional, mode)
    return conditional(mode) + 0.5 * logdet(Symmetric(H)) -
           0.5 * logdet(Symmetric(Q))
end

function _beta_animal_sd_fixture(alpha; G = 24, m = 24, seed = 20261024)
    rng = MersenneTwister(seed)
    labels = ["id$(j)" for j in 1:G]
    # A is intentionally not a correlation matrix. The Beta animal route uses A
    # RAW (as native drmTMB); the DGP draws the latent field from that same A.
    R = [0.58^abs(i - j) for i in 1:G, j in 1:G]
    scale = collect(range(0.7, 1.5; length = G))
    A = Diagonal(scale) * R * Diagonal(scale)
    group_index = repeat(1:G; inner = m)
    group = labels[group_index]
    z_group = repeat([-1.0, 1.0], G ÷ 2)
    z = repeat(z_group; inner = m)
    x = randn(rng, G * m)
    u = cholesky(Symmetric(A)).L * randn(rng, G)
    animal_sd = exp.(alpha[1] .+ alpha[2] .* z_group)
    eta = 0.15 .+ 0.25 .* x .+ (animal_sd .* u)[group_index]
    mu = 1 ./ (1 .+ exp.(-eta))
    sigma = log(0.30)
    phi = exp(-2sigma)
    y = [rand(rng, Distributions.Beta(mu[i] * phi,
                                      (1 - mu[i]) * phi)) for i in eachindex(mu)]
    # Keep rare machine-rounded endpoint draws inside Beta support.
    y = clamp.(y, nextfloat(0.0), prevfloat(1.0))
    return (; y = Float64.(y), x, z, group, group_index, z_group, R, A), alpha
end

@testset "Beta animal-intercept predictor-dependent SD with supplied A" begin
    alpha = [log(0.45), 0.35]
    data, truth = _beta_animal_sd_fixture(alpha)
    formula = bf(@formula(y ~ x + animal(1 | group)),
                 @formula(sigma ~ 1),
                 @formula(sd(group, animal) ~ z))

    # Compare the fixed-parameter Laplace objective to a separately optimized
    # dense oracle, and compare the constant-SD special case to the established
    # Beta animal route before testing the public optimizer.
    beta_fixed = [0.15, 0.25]
    sigma_fixed = log(0.30)
    alpha_fixed = [log(0.45), 0.35]
    theta_fixed = vcat(beta_fixed, sigma_fixed, alpha_fixed)
    Xmu = hcat(ones(length(data.y)), data.x)
    oracle_nll = _beta_animal_sd_dense_laplace(theta_fixed, data, Xmu)
    route_nll = DRModels._beta_animal_sd_laplace_nll(
        theta_fixed, data.y, Xmu,
        hcat(ones(length(data.z_group)), data.z_group), data.group_index, data.A)
    @test route_nll ≈ oracle_nll atol = 2e-5

    # A constant SD design must reproduce the existing Beta animal likelihood.
    Wconstant = ones(length(data.z_group), 1)
    constant_theta = vcat(beta_fixed, sigma_fixed, alpha_fixed[1])
    constant_sd_nll = DRModels._beta_animal_sd_laplace_nll(
        constant_theta, data.y, Xmu, Wconstant, data.group_index, data.A)
    base = DRModels._fit_beta_relmat_laplace(
        DRModels.Beta(), data.y, Xmu, ones(length(data.y), 1), data.A,
        data.group, ["(Intercept)", "x"], ["(Intercept)"], :group, 1e-8;
        se = false)
    @test constant_sd_nll ≈ base.nll(constant_theta) atol = 2e-5
end

@testset "Beta animal SD objective varies with the SD predictor" begin
    alpha_true = [log(0.45), 0.55]
    data, _ = _beta_animal_sd_fixture(alpha_true; G = 120, m = 12, seed = 20261025)
    Xmu = hcat(ones(length(data.y)), data.x)
    Wsd = hcat(ones(length(data.z_group)), data.z_group)
    objective_at_truth = DRModels._beta_animal_sd_laplace_nll(
        vcat([0.15, 0.25], log(0.30), alpha_true), data.y, Xmu, Wsd,
        data.group_index, data.A)
    @test isfinite(objective_at_truth) && objective_at_truth < 1e18
    neutral = [log(0.5), 0.0]
    objective_at_neutral = DRModels._beta_animal_sd_laplace_nll(
        vcat([0.15, 0.25], log(0.30), neutral), data.y, Xmu, Wsd,
        data.group_index, data.A)
    @test isfinite(objective_at_neutral) && objective_at_neutral != objective_at_truth
end

@testset "Beta animal SD rejects invalid covariance and design dimensions" begin
    data, _ = _beta_animal_sd_fixture([log(0.45), 0.35]; G = 8, m = 4)
    Xmu = hcat(ones(length(data.y)), data.x)
    Wsd = hcat(ones(length(data.z_group)), data.z_group)
    theta = vcat([0.15, 0.25], log(0.30), [log(0.45), 0.35])
    @test_throws ArgumentError DRModels._beta_animal_sd_laplace_nll(
        theta, data.y, Xmu, Wsd[1:end-1, :], data.group_index, data.A)
    badA = copy(data.A); badA[1, 1] = -1.0
    @test_throws DomainError DRModels._beta_animal_sd_laplace_nll(
        theta, data.y, Xmu, Wsd, data.group_index, badA)
    outside_sigma = copy(theta)
    outside_sigma[3] = 4.5
    @test DRModels._beta_animal_sd_laplace_nll(
        outside_sigma, data.y, Xmu, Wsd, data.group_index, data.A) == 1e18
    aux_from, _, _ = DRModels._beta_laplace_setup([0.5], ones(1, 1))
    _, _, accepted = DRModels._beta_animal_sd_mode(
        [0.5], [31.0], [1.0], [1], Matrix{Float64}(I, 1, 1), aux_from(0.0))
    @test !accepted
end

@testset "Public Beta animal SD fit, predictions, and serialization" begin
    alpha_true = [log(0.45), 0.45]
    data, _ = _beta_animal_sd_fixture(alpha_true; G = 48, m = 8,
                                       seed = 20261026)
    formula = bf(@formula(y ~ x + animal(1 | group)),
                 @formula(sigma ~ 1),
                 @formula(sd(group, animal) ~ z))
    fit = drm(formula, DRModels.Beta(); data = (; y = data.y, x = data.x,
                                        z = data.z, group = data.group),
              A = data.A)
    Xmu = hcat(ones(length(data.y)), data.x)

    @test fit.converged
    @test fit.family isa DRModels.Beta
    @test :sd in first.(fit.blocks)
    @test coef(fit, :sd) ≈ alpha_true atol = 0.65
    @test sign(coef(fit, :sd)[2]) == sign(alpha_true[2])
    @test all(isfinite, coef(fit, :mu))
    @test all(isfinite, coef(fit, :sigma))
    @test all(isfinite, vcov(fit))
    # This mixed negative Hessian stencil point previously failed the inner
    # mode's gradient cutoff and returned the 1e18 sentinel, making vcov NaN.
    h = DRModels._fd_hessian_step(length(data.y))
    step_x = zeros(length(fit.theta))
    step_sigma = zeros(length(fit.theta))
    step_x[2] = h * (1 + abs(fit.theta[2]))
    step_sigma[3] = h * (1 + abs(fit.theta[3]))
    stencil_theta = fit.theta .- step_x .- step_sigma
    stencil_nll = fit.nll(stencil_theta)
    @test isfinite(stencil_nll) && stencil_nll < 1e16
    @test exp(coef(fit, :sigma)[1]) ≈ fit.scales[:sigma][1]

    constant_formula = bf(@formula(y ~ x + animal(1 | group)),
                          @formula(sigma ~ 1),
                          @formula(sd(group, animal) ~ 1))
    base_formula = bf(@formula(y ~ x + animal(1 | group)), @formula(sigma ~ 1))
    constant_fit = drm(constant_formula, DRModels.Beta();
        data = (; y = data.y, x = data.x, group = data.group), A = data.A,
        se = false)
    base_fit = drm(base_formula, DRModels.Beta();
        data = (; y = data.y, x = data.x, group = data.group), A = data.A,
        se = false)
    @test constant_fit.loglik ≈ base_fit.loglik atol = 1e-5
    @test coef(constant_fit, :sd)[1] ≈ coef(base_fit, :resd)[1] atol = 1e-3

    newdata = (; x = data.x[1:24], z = data.z[1:24],
                group = data.group[1:24])
    predicted = predict(fit, newdata)
    params = predict_parameters(fit, newdata)
    @test length(predicted) == 24
    @test all((0 .< predicted) .& (predicted .< 1))
    @test length(params[:sd]) == 24
    @test params[:sd] ≈ exp.(coef(fit, :sd)[1] .+
                              coef(fit, :sd)[2] .* newdata.z)

    no_A_formula = bf(@formula(y ~ x + animal(1 | group)),
                      @formula(sigma ~ 1),
                      @formula(sd(group, animal) ~ z))
    fit_data = (; y = data.y, x = data.x, z = data.z, group = data.group)
    @test_throws ArgumentError drm(no_A_formula, DRModels.Beta(); data = fit_data)
    multi_data = merge(fit_data, (; other_group = reverse(data.group)))
    multi_formula = bf(@formula(y ~ x + animal(1 | group) + animal(1 | other_group)),
                       @formula(sigma ~ 1),
                       @formula(sd(group, animal) ~ z))
    @test_throws ArgumentError drm(multi_formula, DRModels.Beta();
                                   data = multi_data, A = data.A)

    simulator = DRModels._marginal_simulator_build(fit, fit_data; A = data.A)
    @test simulator !== nothing
    rng_sim = MersenneTwister(20261027)
    draw = simulator(rng_sim)
    rng_oracle = MersenneTwister(20261027)
    factor = cholesky(Symmetric(data.A)).L
    Wsd = hcat(ones(length(data.z_group)), data.z_group)
    group_sd = exp.(Wsd * coef(fit, :sd))
    eta = predict(fit, fit_data; type = :link)
    u = group_sd .* (factor * randn(rng_oracle, length(group_sd)))
    mu = 1 ./ (1 .+ exp.(-clamp.(eta .+ u[data.group_index], -30.0, 30.0)))
    expected_draw = DRModels._simulate_once(fit, rng_oracle; mu)
    @test draw == expected_draw

    mktemp() do path, io
        serialize(io, fit)
        close(io)
        reloaded = open(deserialize, path)
        @test coef(reloaded, :sd) == coef(fit, :sd)
        @test predict(reloaded, newdata) ≈ predicted
        @test predict_parameters(reloaded, newdata)[:sd] ≈ params[:sd]
    end
end

@testset "Concentrated Beta data start animal SD fit within sigma bounds" begin
    rng = MersenneTwister(20261028)
    G, m = 16, 8
    group = repeat(["g$(j)" for j in 1:G], inner = m)
    y = rand(rng, Distributions.Beta(10_000.0, 10_000.0), G * m)
    _, _, sigma0 = DRModels._beta_laplace_setup(y, ones(length(y), 1))
    @test sigma0 < -4.0

    formula = bf(@formula(y ~ animal(1 | group)),
                 @formula(sigma ~ 1),
                 @formula(sd(group, animal) ~ 1))
    fit = drm(formula, DRModels.Beta(); data = (; y, group),
              A = Matrix{Float64}(I, G, G), se = false)
    @test fit.converged
    @test -4.0 <= only(coef(fit, :sigma)) <= 4.0
    @test fit.scales[:sigma] ≈ fill(exp(only(coef(fit, :sigma))), G * m)
end
