# Predictor-dependent SD regression for a Tweedie relmat mean intercept.
using DRModels, Test, Random, LinearAlgebra, Serialization
using DRModels: Optim
import Distributions
import ForwardDiff

@testset "Tweedie relmat mode polish handles objective roundoff" begin
    objective(x) = 1e12 + 0.5 * x[1]^2
    mode = DRModels._tweedie_relmat_polish_mode(objective, [1e-6]; g_tol = 5e-7)
    @test mode !== nothing
    @test norm(ForwardDiff.gradient(objective, mode), Inf) <= 5e-7
    @test objective(mode) == objective([1e-6])
    roundoff_objective(x) = 1e12 + 0.5 * x[1]^2 + 0.00611 * x[1]
    rounded_mode = DRModels._tweedie_relmat_polish_mode(
        roundoff_objective, [-0.01]; g_tol = 5e-7)
    @test rounded_mode !== nothing
    @test roundoff_objective(rounded_mode) > roundoff_objective([-0.01])
    @test norm(ForwardDiff.gradient(roundoff_objective, rounded_mode), Inf) <= 5e-7
    nonconvex = x -> 1e12 - 0.5 * x[1]^2
    @test DRModels._tweedie_relmat_polish_mode(nonconvex, [1e-6];
                                                g_tol = 5e-7) === nothing
end

function _tweedie_relmat_sd_draw(rng, mu, sigma, power)
    phi = sigma^2
    lambda = mu^(2 - power) / (phi * (2 - power))
    count = rand(rng, Distributions.Poisson(lambda))
    count == 0 && return 0.0
    shape = count * (2 - power) / (power - 1)
    scale = phi * (power - 1) * mu^(power - 1)
    return rand(rng, Distributions.Gamma(shape, scale))
end

function _tweedie_relmat_sd_fixture(; alpha = [-0.6, 0.4], G = 32, m = 8,
                                     seed = 20261026)
    rng = MersenneTwister(seed)
    order = randperm(rng, G)
    group_labels = ["group_$(i)" for i in order]
    group_index = repeat(1:G; inner = m)
    group = group_labels[group_index]
    z_group = [zeros(G ÷ 2); ones(G ÷ 2)][randperm(rng, G)]
    base_R = 0.45 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    kernel_sd = ifelse.(z_group .== 0, 1.0, 2.0)
    K = kernel_sd .* base_R .* kernel_sd'
    z = z_group[group_index]
    x = rand(rng, G * m) .* 2 .- 1

    # The unscaled base correlation has a unit diagonal. K includes the
    # existing kernel scale, so exp(alpha0 + alpha1*z) is the SD multiplier;
    # the marginal latent-intercept SD is that multiplier times sqrt(diag(K)).
    latent = cholesky(Symmetric(K)).L * randn(rng, G)
    eta = 0.35 .+ 0.2 .* x .+
          exp.(alpha[1] .+ alpha[2] .* z_group)[group_index] .* latent[group_index]
    mu = exp.(eta)
    sigma, power = 0.65, 1.5
    y = [_tweedie_relmat_sd_draw(rng, mu[i], sigma, power) for i in eachindex(mu)]
    return (; data = (; y, x, z, group), K, group_index, z_group, kernel_sd, alpha,
            sigma, power)
end

# Independent dense Laplace calculation at fitted fixed parameters. It uses a
# fully normalized Distributions.MvNormal prior and optimizes the latent vector
# directly, rather than calling the production relmat precision/logdet helpers.
# The Tweedie observation log density is shared, so this specifically audits
# the varying-SD covariance and Laplace normalization, not the family density.
function _tweedie_relmat_dense_laplace_oracle(fit, data, K, fixture)
    G = length(fixture.z_group)
    eta0 = predict(fit, data; type = :link)
    phi = exp(2 * only(coef(fit, :sigma)))
    power = 1 + 1 / (1 + exp(-only(coef(fit, :nu))))
    alpha = coef(fit, :sd)
    sd = exp.(alpha[1] .+ alpha[2] .* fixture.z_group)
    covariance = Diagonal(sd) * K * Diagonal(sd)
    prior = Distributions.MvNormal(zeros(G), Symmetric(covariance))
    objective(b) = -sum(DRModels._logpdf_tweedie(data.y[i],
        exp(clamp(eta0[i] + b[fixture.group_index[i]], -30.0, 30.0)),
        phi, power) for i in eachindex(data.y)) - Distributions.logpdf(prior, b)
    initial = ranef(fit)[:group]
    alternating = [isodd(i) ? 1.5 : -1.5 for i in 1:G]
    starts = [initial, zeros(G), fill(1.5, G), fill(-1.5, G), alternating]
    modes = Vector{Vector{Float64}}()
    for start in starts
        result = Optim.optimize(objective, start, Optim.LBFGS(),
            Optim.Options(g_tol = 1e-9, iterations = 500); autodiff = :forward)
        Optim.converged(result) || continue
        candidate = Optim.minimizer(result)
        hessian = ForwardDiff.hessian(objective, candidate)
        issuccess(cholesky(Symmetric(hessian); check = false)) || continue
        push!(modes, candidate)
    end
    @assert !isempty(modes)
    mode_objectives = objective.(modes)
    selected_objective = objective(initial)
    best_index = argmin(mode_objectives)
    mode = modes[best_index]
    hessian = ForwardDiff.hessian(objective, mode)
    factor = cholesky(Symmetric(hessian))
    return (; loglik = -objective(mode) + G / 2 * log(2pi) - logdet(factor) / 2,
        selected_mode_gap = selected_objective - mode_objectives[best_index],
        checked_modes = length(modes))
end

@testset "Tweedie relmat predictor-dependent mean-intercept SD" begin
    fixture = _tweedie_relmat_sd_fixture()
    data, K = fixture.data, fixture.K
    formula = bf(@formula(y ~ x + relmat(1 | group)),
                 @formula(sigma ~ 1), @formula(nu ~ 1),
                 @formula(sd(group, relmat) ~ z))
    fit = drm(formula, Tweedie(); data, K, g_tol = 1e-5)

    @test fit.converged
    @test fit.nll(fit.theta) ≈ -loglik(fit) atol = 1e-6
    perturbed_theta = copy(fit.theta)
    perturbed_theta[1] += 0.01
    fit.nll(perturbed_theta)
    @test fit.nll(fit.theta) ≈ -loglik(fit) atol = 1e-6
    oracle = _tweedie_relmat_dense_laplace_oracle(fit, data, K, fixture)
    @test loglik(fit) ≈ oracle.loglik atol = 2e-4
    @test oracle.selected_mode_gap <= 1e-5
    @test oracle.checked_modes >= 3
    @test first.(fit.blocks) == [:mu, :sigma, :nu, :sd]
    @test length(coef(fit, :sd)) == 2
    @test sign(coef(fit, :sd)[2]) == sign(fixture.alpha[2])
    @test abs(coef(fit, :sd)[2] - fixture.alpha[2]) < 0.45

    group_sd = exp.(hcat(ones(length(fixture.z_group)), fixture.z_group) *
                    coef(fit, :sd))
    @test fit.scales[:sd_group] ≈ group_sd
    @test fit.scales[:sd] ≈ group_sd[fixture.group_index]
    marginal_group_sd = group_sd .* fixture.kernel_sd
    @test fit.scales[:marginal_sd_group] ≈ marginal_group_sd
    @test fit.scales[:marginal_sd] ≈ marginal_group_sd[fixture.group_index]
    @test predict_parameters(fit, data)[:sd] ≈ group_sd[fixture.group_index]

    # New-data marginal SDs use the supplied covariance in the new data's
    # first-seen group order. Reorder groups to catch positional assumptions.
    group_order = reverse(collect(eachindex(fixture.z_group)))
    row_order = vcat([collect((g - 1) * 8 + 1:g * 8) for g in group_order]...)
    newdata = (; y = data.y[row_order], x = data.x[row_order],
               z = data.z[row_order], group = data.group[row_order])
    K_new = fixture.K[group_order, group_order]
    predicted = predict_parameters(fit, newdata; K = K_new)
    new_group_index, _ = DRModels._group_index(newdata.group)
    new_group_sd = group_sd[group_order]
    new_marginal_sd = new_group_sd .* sqrt.(diag(K_new))
    @test predicted[:sd] ≈ new_group_sd[new_group_index]
    @test predicted[:marginal_sd] ≈ new_marginal_sd[new_group_index]
    predicted_link = predict_parameters(fit, newdata; K = K_new, type = :link)
    @test predicted_link[:marginal_sd] ≈ log.(new_marginal_sd[new_group_index])
    predicted_se = predict_parameters(fit, newdata; K = K_new, se = true)
    @test predicted_se[:sd].value ≈ predicted[:sd]
    @test predicted_se[:marginal_sd].value ≈ predicted[:marginal_sd]
    @test predicted_se[:marginal_sd].se ≈
          predicted_se[:sd].se .* sqrt.(diag(K_new)[new_group_index])
    K_wrong_size = K_new[1:end-1, 1:end-1]
    @test_throws ArgumentError predict_parameters(fit, newdata; K = K_wrong_size)
    K_nonfinite = Matrix(K_new)
    K_nonfinite[1, 1] = NaN
    @test_throws ArgumentError predict_parameters(fit, newdata; K = K_nonfinite)
    K_asymmetric = Matrix(K_new)
    K_asymmetric[1, 2] += 0.1
    @test_throws ArgumentError predict_parameters(fit, newdata; K = K_asymmetric)
    K_not_positive_definite = Matrix(K_new)
    K_not_positive_definite[1, :] .= 0.0
    K_not_positive_definite[:, 1] .= 0.0
    K_not_positive_definite[1, 1] = -1.0
    @test_throws ArgumentError predict_parameters(fit, newdata;
        K = K_not_positive_definite)
    @test coef(fit, :sigma)[1] ≈ log(fixture.sigma) atol = 0.4
    @test coef(fit, :nu)[1] ≈ log((fixture.power - 1) / (2 - fixture.power)) atol = 0.5

    path = tempname()
    try
        open(path, "w") do io
            serialize(io, fit)
        end
        reloaded = open(deserialize, path)
        @test coef(reloaded, :sd) ≈ coef(fit, :sd)
        @test reloaded.scales[:marginal_sd] ≈ fit.scales[:marginal_sd]
        @test predict_parameters(reloaded, data)[:sd] ≈
              predict_parameters(fit, data)[:sd]
    finally
        isfile(path) && rm(path)
    end

    # Bootstrap refits must simulate from the predictor-dependent SD model,
    # preserving the supplied kernel and the fitted target identity.
    simulator = DRModels._marginal_simulator_build(fit, data; K)
    @test simulator !== nothing
    simulation_seed = 20261028
    simulated = simulator(MersenneTwister(simulation_seed))
    @test length(simulated) == length(data.y)
    @test all(isfinite, simulated)
    oracle_rng = MersenneTwister(simulation_seed)
    latent_z = randn(oracle_rng, length(fixture.z_group))
    eta_fixed = predict(fit, data; type = :link)
    latent_effect = group_sd .* (cholesky(Symmetric(K)).L * latent_z)
    mu_star = exp.(eta_fixed .+ latent_effect[fixture.group_index])
    sigma_star = fit.scales[:sigma]
    power_star = fit.scales[:nu]
    expected = [_tweedie_relmat_sd_draw(oracle_rng, mu_star[i],
        sigma_star[i], power_star[i]) for i in eachindex(mu_star)]
    @test isapprox(simulated, expected; rtol = 1e-12, atol = 1e-12)

    # A constant-SD simulator is a deliberately wrong control. This fixture
    # has enough SD contrast for the independent oracle to detect that error.
    wrong_rng = MersenneTwister(simulation_seed)
    wrong_latent_z = randn(wrong_rng, length(fixture.z_group))
    constant_sd = fill(sum(group_sd) / length(group_sd), length(group_sd))
    wrong_effect = constant_sd .* (cholesky(Symmetric(K)).L * wrong_latent_z)
    wrong_mu = exp.(eta_fixed .+ wrong_effect[fixture.group_index])
    wrong = [_tweedie_relmat_sd_draw(wrong_rng, wrong_mu[i],
        sigma_star[i], power_star[i]) for i in eachindex(wrong_mu)]
    @test maximum(abs.(simulated .- wrong)) > 1e-6

    boot = bootstrap_result(fit; data, K, B = 1, g_tol = 1e-5,
        rng = MersenneTwister(20261029),
        check_converged = true)
    @test boot.attempted == 1
    @test boot.used == 1
    @test boot.failed == 0

    # A constant group-level predictor and the intercept-only SD formula
    # represent the same constant-SD model when a no-SD legacy relmat route is
    # unavailable for this family.
    constant_data = merge(data, (; z_constant = ones(length(data.y))))
    fit_intercept = drm(bf(@formula(y ~ x + relmat(1 | group)),
        @formula(sigma ~ 1), @formula(nu ~ 1),
        @formula(sd(group, relmat) ~ 1)), Tweedie();
        data = constant_data, K, g_tol = 1e-5)
    fit_unit_column = drm(bf(@formula(y ~ x + relmat(1 | group)),
        @formula(sigma ~ 1), @formula(nu ~ 1),
        @formula(sd(group, relmat) ~ 0 + z_constant)), Tweedie();
        data = constant_data, K, g_tol = 1e-5)
    @test loglik(fit_unit_column) ≈ loglik(fit_intercept) atol = 1e-5
    @test coef(fit_unit_column, :sd) ≈ coef(fit_intercept, :sd) atol = 1e-4
    @test coef(fit_unit_column, :mu) ≈ coef(fit_intercept, :mu) atol = 1e-4
    @test coef(fit_unit_column, :sigma) ≈ coef(fit_intercept, :sigma) atol = 1e-4
    @test coef(fit_unit_column, :nu) ≈ coef(fit_intercept, :nu) atol = 1e-4
end

@testset "Tweedie rejects unused structured covariance keywords" begin
    fixture = _tweedie_relmat_sd_fixture(G = 16, m = 8)
    formula = bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1))
    @test_throws ArgumentError drm(formula, Tweedie(); data = fixture.data,
        K = fixture.K, g_tol = 1e-5)
    @test_throws ArgumentError drm(formula, Tweedie(); data = fixture.data,
        A = fixture.K, g_tol = 1e-5)
    @test_throws ArgumentError drm(formula, Tweedie(); data = fixture.data,
        tree = fixture.K, g_tol = 1e-5)
    @test_throws ArgumentError drm(formula, Tweedie(); data = fixture.data,
        coords = zeros(16, 2), g_tol = 1e-5)
end

