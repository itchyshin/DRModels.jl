using DRModels, Test, Random, LinearAlgebra, Serialization
import Distributions

function _poisson_animal_slope_sd_fixture(; G = 40, m = 24,
                                           seed = 20261022)
    rng = MersenneTwister(seed)
    order_group = randperm(rng, G)
    labels = ["animal_$(i)" for i in order_group]
    R = 0.4 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    kernel_scale = collect(range(0.7, 1.6; length = G))
    A = Diagonal(kernel_scale) * R * Diagonal(kernel_scale)
    zgroup = repeat([-1.0, 1.0], inner = G ÷ 2)
    xwithin = collect(range(-1.0, 1.0; length = m))
    group = repeat(labels, inner = m)
    group_index = repeat(1:G, inner = m)
    x = repeat(xwithin, outer = G)
    alpha = [-0.55, 0.35]
    latent_effect = cholesky(Symmetric(A)).L * randn(rng, G)
    slope = exp.(alpha[1] .+ alpha[2] .* zgroup) .* latent_effect
    eta = 0.15 .+ 0.25 .* x .+ x .* slope[group_index]
    y = [rand(rng, Distributions.Poisson(value)) for value in exp.(eta)]

    # Shuffle rows and align the supplied relationship matrix to first-seen IDs.
    row_order = randperm(rng, length(group))
    fitted_labels = unique(group[row_order])
    original_index = [findfirst(==(label), labels) for label in fitted_labels]
    fitted_group_index = [findfirst(==(label), fitted_labels)
                          for label in group[row_order]]
    data = (; y = y[row_order], x = x[row_order],
            z = zgroup[original_index][fitted_group_index],
            g = group[row_order])
    return (; data, A = A[original_index, original_index],
            R = R[original_index, original_index], group_index = fitted_group_index,
            zgroup = zgroup[original_index], alpha)
end

function _poisson_animal_slope_dense_laplace(theta, y, Xmu, x, Zobs,
                                              group_index, R)
    Q = inv(Symmetric(R))
    p_mu = size(Xmu, 2)
    beta = theta[1:p_mu]
    alpha = theta[p_mu+1:end]
    loading = x .* exp.(Zobs * alpha)
    eta0 = Xmu * beta
    function joint(b)
        value = 0.5 * dot(b, Q * b)
        for i in eachindex(y)
            eta = eta0[i] + loading[i] * b[group_index[i]]
            value -= Distributions.logpdf(Distributions.Poisson(exp(eta)), y[i])
        end
        value
    end
    result = DRModels.Optim.optimize(joint, zeros(size(R, 1)),
        DRModels.Optim.LBFGS(), DRModels.Optim.Options(
            g_tol = 1e-10, iterations = 400); autodiff = :forward)
    @test DRModels.Optim.converged(result)
    mode = DRModels.Optim.minimizer(result)
    hessian = DRModels.ForwardDiff.hessian(joint, mode)
    @test isposdef(Symmetric(hessian))
    return joint(mode) + 0.5 * logdet(Symmetric(hessian)) -
           0.5 * logdet(Symmetric(Q))
end

@testset "Poisson animal random-slope SD regression" begin
    fixture = _poisson_animal_slope_sd_fixture()
    data, A = fixture.data, fixture.A
    formula = bf(@formula(y ~ x + animal(0 + x | g)),
                 @formula(sd(g, animal) ~ z))
    fit = drm(formula, Poisson(); data, A)

    @test fit.converged
    @test abs(coef(fit, :mu)[1] - 0.15) < 0.4
    @test abs(coef(fit, :mu)[2] - 0.25) < 0.4
    @test abs(coef(fit, :sd)[1] - fixture.alpha[1]) < 0.5
    @test coef(fit, :sd)[2] > 0
    @test abs(coef(fit, :sd)[2] - fixture.alpha[2]) < 0.5
    @test fit.scales[:sd_group] ≈ exp.(hcat(ones(length(fixture.zgroup)),
                                             fixture.zgroup) * coef(fit, :sd))
    @test fit.scales[:sd] ≈ fit.scales[:sd_group][fixture.group_index]
    @test predict_parameters(fit, data; se = false)[:sd] ≈ fit.scales[:sd]

    # The sparse Laplace fit accepts the same animal model as a precision
    # matrix. Its marginal simulator must retain that route instead of
    # silently asking callers for covariance `A` again.
    Ainv = inv(Symmetric(A))
    precision_fit = drm(formula, Poisson(); data, Ainv, se = false)
    @test precision_fit.converged
    @test coef(precision_fit, :sd) ≈ coef(fit, :sd) atol = 2e-5
    @test precision_fit.loglik ≈ fit.loglik atol = 2e-5
    precision_simulator = DRModels._marginal_simulator_build(
        precision_fit, data; Ainv)
    @test precision_simulator(MersenneTwister(202610241)) ==
          DRModels._marginal_simulator_build(fit, data; A)(MersenneTwister(202610241))
    precision_boot = bootstrap_result(precision_fit; data, Ainv, B = 1,
        rng = MersenneTwister(202610242), failures = :skip)
    @test precision_boot.attempted == precision_boot.used == 1
    @test precision_boot.failed == 0

    Xmu = hcat(ones(length(data.y)), data.x)
    Zobs = hcat(ones(length(data.y)), data.z)
    theta = [0.15, 0.25, fixture.alpha...]
    dense_nll = _poisson_animal_slope_dense_laplace(
        theta, data.y, Xmu, data.x, Zobs, fixture.group_index, fixture.A)
    @test fit.nll(theta) ≈ dense_nll atol = 4e-3

    constant_formula = bf(@formula(y ~ x + animal(0 + x | g)),
                          @formula(sd(g, animal) ~ 1))
    constant_fit = drm(constant_formula, Poisson(); data, A, se = false)
    base_fit = drm(bf(@formula(y ~ x + animal(0 + x | g))), Poisson();
                   data, A, se = false)
    @test constant_fit.loglik ≈ base_fit.loglik atol = 3e-5

    base_simulator = DRModels._marginal_simulator_build(base_fit, data; A)
    base_seed = 202610231
    base_actual = base_simulator(MersenneTwister(base_seed))
    base_rng = MersenneTwister(base_seed)
    base_latent = exp(only(coef(base_fit, :resd))) .* (
        cholesky(Symmetric(fixture.A)).L * randn(base_rng, length(fixture.zgroup)))
    base_eta = hcat(ones(length(data.x)), data.x) * coef(base_fit, :mu)
    base_expected = [rand(base_rng, Distributions.Poisson(exp(eta + data.x[i] *
        base_latent[fixture.group_index[i]]))) for (i, eta) in enumerate(base_eta)]
    @test base_actual == base_expected

    seed = 20261023
    simulator = DRModels._marginal_simulator_build(fit, data; A)
    actual = simulator(MersenneTwister(seed))
    oracle_rng = MersenneTwister(seed)
    sd_group = exp.(hcat(ones(length(fixture.zgroup)), fixture.zgroup) *
                    coef(fit, :sd))
    latent = sd_group .* (cholesky(Symmetric(fixture.A)).L *
                          randn(oracle_rng, length(sd_group)))
    eta_fixed = predict(fit, data; type = :link)
    expected = [rand(oracle_rng, Distributions.Poisson(exp(eta + data.x[i] *
                    latent[fixture.group_index[i]]))) for (i, eta) in enumerate(eta_fixed)]
    @test actual == expected

    path = tempname()
    open(path, "w") do io
        serialize(io, fit)
    end
    reloaded = open(deserialize, path)
    rm(path; force = true)
    @test predict_parameters(reloaded, data; se = false)[:sd] ≈ fit.scales[:sd]

    bootstrap = bootstrap_result(fit; data, A, B = 1,
        rng = MersenneTwister(20261024), failures = :skip)
    @test bootstrap.attempted == bootstrap.used == 1

    z_bad = copy(data.z)
    z_bad[1] += 1.0
    data_bad = merge(data, (; z = z_bad))
    @test_throws ArgumentError drm(formula, Poisson(); data = data_bad, A)

    zero_slope = merge(data, (; x = zeros(length(data.x))))
    @test_throws ArgumentError drm(formula, Poisson(); data = zero_slope, A)
end
