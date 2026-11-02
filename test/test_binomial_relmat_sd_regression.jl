using DRModels, Test, Random, Distributions, LinearAlgebra

@testset "Binomial related-matrix SD regression" begin
    rng = MersenneTwister(20261021)
    G, n_each = 24, 24
    order = vcat(collect(5:G), 1:4) # matrix order is deliberately not sorted label order
    group_index = repeat(1:G; inner = n_each)
    group = ["g$(order[j])" for j in group_index]
    z_group = repeat([-1.0, 1.0], G ÷ 2)
    z = repeat(z_group; inner = n_each)
    x_group = collect(range(-1.0, 1.0; length = G))
    x = repeat(x_group; inner = n_each)

    # K is a covariance, not a correlation matrix: retain its non-unit diagonal.
    K = Matrix(Diagonal(collect(range(0.8, 1.8; length = G))) +
               0.10 * ones(G, G))
    u = cholesky(Symmetric(K)).L * randn(rng, G)
    alpha = [-0.65, 0.45]
    b = exp.(alpha[1] .+ alpha[2] .* z_group) .* u
    eta = -0.15 .+ 0.2 .* x .+ b[group_index]
    p = 1 ./ (1 .+ exp.(-eta))
    y = Float64.(rand.(Ref(rng), Distributions.Bernoulli.(p)))
    data = (; y, x, z, group)

    formula = bf(
        @formula(y ~ x + relmat(1 | group)),
        @formula(sd(group, relmat) ~ z),
    )
    target = only(DRModels._sd_target_parts(formula)).target
    @test target.group === :group
    @test target.level === :relmat
    @test target.dpar === :mu

    fit = drm(formula, DRModels.Binomial(); data, K, se = true, g_tol = 1e-5)
    @test fit.converged
    @test isfinite(fit.loglik)
    @test coef(fit, :sd)[2] > 0
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.45
    @test fit.scales[:sd_group] ≈ exp.(hcat(ones(G), z_group) * coef(fit, :sd))
    @test fit.scales[:sd] ≈ fit.scales[:sd_group][group_index]
    @test predict_parameters(fit, data)[:sd] ≈ fit.scales[:sd]
    pred = predict_parameters(fit, data; se = true)[:sd]
    @test pred.value ≈ fit.scales[:sd]
    @test all(isfinite, pred.se)

    # Independent dense latent-effect Laplace oracle. The supplied non-unit K
    # is the covariance itself, so the model covariance is D*K*D.
    theta = vcat(coef(fit, :mu), coef(fit, :sd))
    group_sd = exp.(hcat(ones(G), z_group) * coef(fit, :sd))
    covariance = Diagonal(group_sd) * K * Diagonal(group_sd)
    precision = inv(covariance)
    aux = (s = round.(Int, y), ntr = ones(Int, length(y)),
           logchoose = zeros(length(y)))
    eta_fixed = predict(fit, data; type = :link)
    function dense_joint(effect)
        eta = eta_fixed .+ effect[group_index]
        value = 0.5 * dot(effect, precision * effect)
        for i in eachindex(y)
            value += DRModels._laplace_value(Val(:binomial), aux, i, eta[i])
        end
        value
    end
    mode_result = DRModels.Optim.optimize(
        dense_joint, zeros(G), DRModels.Optim.LBFGS(),
        DRModels.Optim.Options(g_tol = 1e-9, iterations = 400);
        autodiff = :forward,
    )
    mode = DRModels.Optim.minimizer(mode_result)
    hessian = DRModels.ForwardDiff.hessian(dense_joint, mode)
    dense_laplace = dense_joint(mode) + 0.5 * logdet(Symmetric(hessian)) -
        0.5 * logdet(Symmetric(precision))
    @test fit.nll(theta) ≈ dense_laplace atol = 2e-5

    # A constant group SD written as an intercept or a one-column no-intercept
    # design must be the same model and likelihood.
    constant_data = merge(data, (; z_constant = ones(length(y))))
    fit_intercept = drm(bf(
        @formula(y ~ x + relmat(1 | group)),
        @formula(sd(group, relmat) ~ 1),
    ), DRModels.Binomial(); data = constant_data, K, se = false, g_tol = 1e-5)
    fit_unit_column = drm(bf(
        @formula(y ~ x + relmat(1 | group)),
        @formula(sd(group, relmat) ~ 0 + z_constant),
    ), DRModels.Binomial(); data = constant_data, K, se = false, g_tol = 1e-5)
    @test fit_unit_column.loglik ≈ fit_intercept.loglik atol = 1e-8
    @test coef(fit_unit_column, :sd) ≈ coef(fit_intercept, :sd) atol = 1e-7

    # A seeded marginal draw checks both first-seen label ordering and raw K
    # scaling independently of the fitting likelihood.
    draw_seed = 20261022
    simulator = DRModels._marginal_simulator_build(fit, data; K)
    actual_draw = simulator(MersenneTwister(draw_seed))
    oracle_rng = MersenneTwister(draw_seed)
    latent = group_sd .* (cholesky(Symmetric(K)).L * randn(oracle_rng, G))
    expected_p = 1 ./ (1 .+ exp.(-(eta_fixed .+ latent[group_index])))
    expected_draw = Float64[rand(oracle_rng, Distributions.Binomial(
        1, clamp(expected_p[i], eps(), 1 - eps()))) for i in eachindex(y)]
    @test actual_draw == expected_draw

    boot = bootstrap_result(fit; data, B = 1, K,
        rng = MersenneTwister(20261023), failures = :error)
    @test boot.attempted == 1
    @test boot.used == 1
    @test boot.failed == 0

    @test_throws ArgumentError drm(formula, DRModels.Binomial(); data,
        K = K[1:(end - 1), 1:(end - 1)], se = false)
end

@testset "Binomial probit and cloglog related-matrix SD regression" begin
    for (link, seed_offset) in ((:probit, 0), (:cloglog, 1))
        rng = MersenneTwister(20261040 + seed_offset)
        G, n_each = 64, 32
        matrix_order = vcat(collect(7:G), 1:6)
        group_index = repeat(1:G; inner = n_each)
        group = ["g$(matrix_order[j])" for j in group_index]
        z_group = vcat(fill(-1.0, G ÷ 2), fill(1.0, G ÷ 2))[randperm(rng, G)]
        z = repeat(z_group; inner = n_each)
        x = randn(rng, length(group_index))

        # Preserve a non-unit kernel diagonal and deliberately permute its labels.
        kernel_scale = collect(range(0.8, 1.8; length = G))
        correlation = [0.45^abs(i - j) for i in 1:G, j in 1:G]
        K = Diagonal(sqrt.(kernel_scale)) * correlation *
            Diagonal(sqrt.(kernel_scale))
        u = cholesky(Symmetric(K)).L * randn(rng, G)
        alpha = [-0.65, 0.45]
        latent_sd = exp.(alpha[1] .+ alpha[2] .* z_group)
        eta = -0.15 .+ 0.2 .* x .+ (latent_sd .* u)[group_index]
        probability = link === :probit ? Distributions.cdf.(Normal(), eta) :
                      -expm1.(-exp.(eta))
        y = Float64.(rand.(Ref(rng), Distributions.Bernoulli.(probability)))
        data = (; y, x, z, group)
        formula = bf(@formula(y ~ x + relmat(1 | group)),
                     @formula(sd(group, relmat) ~ z))

        fit = drm(formula, DRModels.Binomial(link = link);
                  data, K, se = false, g_tol = 1e-5)
        @test fit.converged
        @test isfinite(fit.loglik)
        @test coef(fit, :sd)[2] > 0
        @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.4
        @test fit.scales[:sd_group] ≈ exp.(hcat(ones(G), z_group) * coef(fit, :sd))
        @test predict_parameters(fit, data)[:sd] ≈ fit.scales[:sd]

        # Compare the sparse fitter against a dense latent Gaussian Laplace oracle.
        group_sd = fit.scales[:sd_group]
        covariance = Diagonal(group_sd) * K * Diagonal(group_sd)
        precision = inv(covariance)
        successes, trials = round.(Int, y), ones(Int, length(y))
        aux = (s = successes, ntr = trials, logchoose = zeros(length(y)), link)
        eta_fixed = predict(fit, data; type = :link)
        function dense_joint(effect)
            eta_dense = eta_fixed .+ effect[group_index]
            value = 0.5 * dot(effect, precision * effect)
            for i in eachindex(y)
                value += DRModels._laplace_value(Val(:binomial), aux, i, eta_dense[i])
            end
            value
        end
        mode_result = DRModels.Optim.optimize(
            dense_joint, zeros(G), DRModels.Optim.LBFGS(),
            DRModels.Optim.Options(g_tol = 1e-9, iterations = 400);
            autodiff = :forward,
        )
        mode = DRModels.Optim.minimizer(mode_result)
        hessian = DRModels.ForwardDiff.hessian(dense_joint, mode)
        dense_laplace = dense_joint(mode) + 0.5 * logdet(Symmetric(hessian)) -
                        0.5 * logdet(Symmetric(precision))
        @test fit.nll(vcat(coef(fit, :mu), coef(fit, :sd))) ≈ dense_laplace atol = 3e-5
    end
end
