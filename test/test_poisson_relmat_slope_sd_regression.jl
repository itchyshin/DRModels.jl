# Predictor-dependent random-effect SD for a Poisson relmat slope-only
# coefficient: log(lambda_i) = X_i beta + x_i b_g, b = D L z, D = exp(Z alpha).
# Candidate-level evidence only: one fixture, no calibration claim.
using DRModels, Test, Random, LinearAlgebra
import Distributions

function _poisson_relmat_slope_sd_fixture(; G = 40, m = 24, seed = 20261101)
    rng = MersenneTwister(seed)
    order_group = randperm(rng, G)
    labels = ["id_$(i)" for i in order_group]
    R = 0.4 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    # The relmat route works on the unit-diagonal correlation of the supplied
    # covariance, so K deliberately has a non-unit diagonal.
    kernel_scale = collect(range(0.7, 1.6; length = G))
    K = Diagonal(kernel_scale) * R * Diagonal(kernel_scale)
    zgroup = repeat([-1.0, 1.0], inner = G ÷ 2)
    xwithin = collect(range(-1.0, 1.0; length = m))
    group = repeat(labels, inner = m)
    group_index = repeat(1:G, inner = m)
    x = repeat(xwithin, outer = G)
    alpha = [-0.55, 0.35]
    latent_effect = cholesky(Symmetric(R)).L * randn(rng, G)
    slope = exp.(alpha[1] .+ alpha[2] .* zgroup) .* latent_effect
    eta = 0.15 .+ 0.25 .* x .+ x .* slope[group_index]
    y = Float64[rand(rng, Distributions.Poisson(value)) for value in exp.(eta)]

    # Shuffle rows and align K to the first-seen group labels.
    row_order = randperm(rng, length(group))
    fitted_labels = unique(group[row_order])
    original_index = [findfirst(==(label), labels) for label in fitted_labels]
    fitted_group_index = [findfirst(==(label), fitted_labels)
                          for label in group[row_order]]
    data = (; y = y[row_order], x = x[row_order],
            z = zgroup[original_index][fitted_group_index],
            g = group[row_order])
    return (; data, K = K[original_index, original_index],
            R = R[original_index, original_index], group_index = fitted_group_index,
            zgroup = zgroup[original_index], alpha)
end

function _poisson_relmat_slope_dense_laplace(theta, y, Xmu, x, Zobs,
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

@testset "Poisson relmat random-slope SD regression" begin
    fixture = _poisson_relmat_slope_sd_fixture()
    data, K = fixture.data, fixture.K
    formula = bf(@formula(y ~ x + relmat(0 + x | g)),
                 @formula(sd(g, relmat) ~ z))
    fit = drm(formula, Poisson(); data, K)

    @test fit.converged
    @test length(coef(fit, :sd)) == 2
    @test abs(coef(fit, :mu)[1] - 0.15) < 0.4
    @test abs(coef(fit, :mu)[2] - 0.25) < 0.4
    @test abs(coef(fit, :sd)[1] - fixture.alpha[1]) < 0.5
    # No sign assertion on the z effect: with 20 groups per level its sampling
    # error is comparable to the 0.35 truth, so the sign is not a fixed outcome.
    @test abs(coef(fit, :sd)[2] - fixture.alpha[2]) < 0.5
    @test fit.scales[:sd_group] ≈ exp.(hcat(ones(length(fixture.zgroup)),
                                             fixture.zgroup) * coef(fit, :sd))
    @test fit.scales[:sd] ≈ fit.scales[:sd_group][fixture.group_index]

    # The fitted marginal likelihood equals an independent dense Laplace
    # evaluation of the same slope model on the correlation matrix R.
    Xmu = hcat(ones(length(data.y)), data.x)
    Zobs = hcat(ones(length(data.y)), data.z)
    theta = [0.15, 0.25, fixture.alpha...]
    dense_nll = _poisson_relmat_slope_dense_laplace(
        theta, data.y, Xmu, data.x, Zobs, fixture.group_index, fixture.R)
    @test fit.nll(theta) ≈ dense_nll atol = 4e-3

    # The relmat route standardises K to its correlation, so it must coincide
    # with the animal route fed that correlation matrix as `A` (no scaling).
    animal_formula = bf(@formula(y ~ x + animal(0 + x | g)),
                        @formula(sd(g, animal) ~ z))
    animal_fit = drm(animal_formula, Poisson(); data, A = fixture.R, se = false)
    relmat_fit = drm(formula, Poisson(); data, K, se = false)
    @test animal_fit.converged && relmat_fit.converged
    @test relmat_fit.loglik ≈ animal_fit.loglik atol = 1e-6
    @test coef(relmat_fit, :sd) ≈ coef(animal_fit, :sd) atol = 1e-4
    @test coef(relmat_fit, :mu) ≈ coef(animal_fit, :mu) atol = 1e-4

    # Rescaling K by a positive diagonal must not change the fit.
    d = Diagonal(fill(2.5, size(K, 1)))
    scaled_fit = drm(formula, Poisson(); data, K = d * K * d, se = false)
    @test scaled_fit.loglik ≈ relmat_fit.loglik atol = 1e-6

    # A constant SD formula is a nested model of the predictor-dependent one.
    constant_fit = drm(bf(@formula(y ~ x + relmat(0 + x | g)),
                          @formula(sd(g, relmat) ~ 1)),
                       Poisson(); data, K, se = false)
    @test constant_fit.converged
    @test length(coef(constant_fit, :sd)) == 1
    @test relmat_fit.loglik >= constant_fit.loglik - 1e-6

    # Read side: predict_parameters, link prediction and the marginal
    # simulator must accept the fitted relmat slope. The simulator is checked
    # against an independent oracle: u = sd_group .* (chol(R).L * z) with R the
    # correlation form of K, y ~ Poisson(exp(eta + x u[g])).
    @test predict_parameters(fit, data; se = false)[:sd] ≈ fit.scales[:sd]
    eta_fixed = predict(fit, data; type = :link)
    @test length(eta_fixed) == length(data.y)

    seed = 20261101
    simulator = DRModels._marginal_simulator_build(fit, data; K)
    actual = simulator(MersenneTwister(seed))
    oracle_rng = MersenneTwister(seed)
    sd_group = exp.(hcat(ones(length(fixture.zgroup)), fixture.zgroup) *
                    coef(fit, :sd))
    latent = sd_group .* (cholesky(Symmetric(fixture.R)).L *
                          randn(oracle_rng, length(sd_group)))
    expected = [rand(oracle_rng, Distributions.Poisson(exp(eta + data.x[i] *
                    latent[fixture.group_index[i]]))) for (i, eta) in enumerate(eta_fixed)]
    @test actual == expected

    bootstrap = bootstrap_result(fit; data, K, B = 1,
        rng = MersenneTwister(20261102), failures = :skip)
    @test bootstrap.attempted == bootstrap.used == 1

    # Refusals: SD predictor varying within group, zero slope predictor,
    # missing K, animal-style input, and Ainv.
    z_bad = copy(data.z)
    z_bad[1] += 1.0
    data_bad = merge(data, (; z = z_bad))
    @test_throws ArgumentError drm(formula, Poisson(); data = data_bad, K, se = false)

    zero_slope = merge(data, (; x = zeros(length(data.x))))
    @test_throws ArgumentError drm(formula, Poisson(); data = zero_slope, K, se = false)
    @test_throws ArgumentError drm(formula, Poisson(); data, se = false)
    @test_throws ArgumentError drm(formula, Poisson(); data, A = K, se = false)
    @test_throws ArgumentError drm(formula, Poisson(); data,
                                   K, Ainv = inv(Symmetric(K)), se = false)
end
