using DRModels, Test, Random, LinearAlgebra, Serialization
import Distributions

function _poisson_animal_sd_fixture(; G = 36, m = 18, seed = 20261022)
    rng = MersenneTwister(seed)
    order = randperm(rng, G)
    labels = ["animal_$(i)" for i in order]
    R = 0.45 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    scale = collect(range(0.7, 1.6; length = G))
    A = Diagonal(scale) * R * Diagonal(scale)
    group = repeat(labels, inner = m)
    gidx = repeat(1:G, inner = m)
    zgroup = [zeros(G ÷ 2); ones(G - G ÷ 2)][randperm(rng, G)]
    z = zgroup[gidx]
    x = randn(rng, G * m)
    alpha = [-0.55, 0.38]
    latent = cholesky(Symmetric(A)).L * randn(rng, G)
    eta = 0.25 .+ 0.2 .* x .+ exp.(alpha[1] .+ alpha[2] .* zgroup)[gidx] .* latent[gidx]
    y = Float64[rand(rng, Distributions.Poisson(exp(v))) for v in eta]
    return (; data = (; y, x, z, g = group), A, R, gidx, zgroup, alpha)
end

function _poisson_animal_sd_dense_laplace(theta, y, Xmu, Zobs, gidx, R)
    Q = inv(Symmetric(R))
    p_mu = size(Xmu, 2)
    beta = theta[1:p_mu]
    alpha = theta[p_mu+1:end]
    eta0 = Xmu * beta
    loading = exp.(Zobs * alpha)
    function joint(b)
        value = 0.5 * dot(b, Q * b)
        for i in eachindex(y)
            eta = eta0[i] + loading[i] * b[gidx[i]]
            value += exp(eta) - y[i] * eta + DRModels._logfactorial(round(Int, y[i]))
        end
        value
    end
    result = DRModels.Optim.optimize(joint, zeros(size(R, 1)),
        DRModels.Optim.LBFGS(), DRModels.Optim.Options(
            g_tol = 1e-10, iterations = 400); autodiff = :forward)
    mode = DRModels.Optim.minimizer(result)
    hessian = DRModels.ForwardDiff.hessian(joint, mode)
    return joint(mode) + 0.5 * logdet(Symmetric(hessian)) -
           0.5 * logdet(Symmetric(Q))
end

@testset "Poisson animal random-effect SD regression" begin
    fixture = _poisson_animal_sd_fixture()
    data, A = fixture.data, fixture.A
    formula = bf(@formula(y ~ x + animal(1 | g)),
                 @formula(sd(g, animal) ~ z))
    fit = drm(formula, Poisson(); data, A)

    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sd]
    @test length(coef(fit, :sd)) == 2
    @test sign(coef(fit, :sd)[2]) == sign(fixture.alpha[2])
    @test abs(coef(fit, :sd)[2] - fixture.alpha[2]) < 0.55
    @test fit.scales[:sd_group] ≈ exp.(hcat(ones(length(fixture.zgroup)),
                                             fixture.zgroup) * coef(fit, :sd))
    predicted = predict_parameters(fit, data; se = true)[:sd]
    @test predicted.value ≈ fit.scales[:sd]
    @test all(isfinite, predicted.se)

    # `Ainv` is the precision of the same supplied covariance matrix. It should
    # enter the sparse Laplace fit directly and reproduce the `A` route.
    Ainv = inv(Symmetric(fixture.A))
    precision_fit = drm(formula, Poisson(); data, Ainv, se = false)
    @test precision_fit.converged
    @test coef(precision_fit, :sd) ≈ coef(fit, :sd) atol = 2e-5
    @test abs(coef(precision_fit, :sd)[2] - fixture.alpha[2]) < 0.55
    @test precision_fit.loglik ≈ fit.loglik atol = 2e-5
    @test predict_parameters(precision_fit, data)[:sd] ≈ fit.scales[:sd] atol = 2e-5
    precision_simulator = DRModels._marginal_simulator_build(
        precision_fit, data; Ainv)
    covariance_simulator = DRModels._marginal_simulator_build(fit, data; A)
    @test precision_simulator(MersenneTwister(202610251)) ==
          covariance_simulator(MersenneTwister(202610251))
    precision_boot = bootstrap_result(precision_fit; data, Ainv, B = 1,
        rng = MersenneTwister(202610252), failures = :skip)
    @test precision_boot.attempted == precision_boot.used == 1
    @test precision_boot.failed == 0

    row_order = randperm(MersenneTwister(20261025), length(data.y))
    reordered_data = (; y = data.y[row_order], x = data.x[row_order],
                      z = data.z[row_order], g = data.g[row_order])
    original_labels = unique(data.g)
    reordered_labels = unique(reordered_data.g)
    matrix_order = [findfirst(==(label), original_labels) for label in reordered_labels]
    reordered_fit = drm(formula, Poisson(); data = reordered_data,
                        A = A[matrix_order, matrix_order], se = false)
    original_sd_by_label = Dict(zip(original_labels, fit.scales[:sd_group]))
    reordered_sd_by_label = Dict(zip(reordered_labels, reordered_fit.scales[:sd_group]))
    @test all(isapprox(reordered_sd_by_label[label], original_sd_by_label[label];
                       atol = 2e-4) for label in original_labels)
    @test predict_parameters(reordered_fit, reordered_data)[:sd][invperm(row_order)] ≈
          predicted.value atol = 2e-4
    precision_reordered_fit = drm(formula, Poisson(); data = reordered_data,
        Ainv = Ainv[matrix_order, matrix_order], se = false)
    @test precision_reordered_fit.converged
    @test coef(precision_reordered_fit, :sd) ≈ coef(precision_fit, :sd) atol = 2e-5
    @test predict_parameters(precision_reordered_fit, reordered_data)[:sd][invperm(row_order)] ≈
          predict_parameters(precision_fit, data)[:sd] atol = 2e-4

    Xmu = hcat(ones(length(data.y)), data.x)
    Zobs = hcat(ones(length(data.y)), data.z)
    theta = [0.25, 0.2, fixture.alpha...]
    dense_nll = _poisson_animal_sd_dense_laplace(
        theta, data.y, Xmu, Zobs, fixture.gidx, fixture.A)
    @test fit.nll(theta) ≈ dense_nll atol = 3e-4

    constant_formula = bf(@formula(y ~ x + animal(1 | g)),
                          @formula(sd(g, animal) ~ 1))
    constant_fit = drm(constant_formula, Poisson(); data, A, se = false)
    constant_precision_fit = drm(constant_formula, Poisson(); data, Ainv, se = false)
    base_fit = drm(bf(@formula(y ~ x + animal(1 | g))), Poisson(); data, A, se = false)
    @test constant_fit.loglik ≈ base_fit.loglik atol = 2e-5
    @test coef(constant_fit, :mu) ≈ coef(base_fit, :mu) atol = 1e-4
    @test constant_precision_fit.loglik ≈ constant_fit.loglik atol = 2e-5
    @test coef(constant_precision_fit, :mu) ≈ coef(constant_fit, :mu) atol = 1e-4

    seed = 20261023
    simulator = DRModels._marginal_simulator_build(fit, data; A)
    actual = simulator(MersenneTwister(seed))
    oracle_rng = MersenneTwister(seed)
    sd_group = exp.(hcat(ones(length(fixture.zgroup)), fixture.zgroup) * coef(fit, :sd))
    latent_draw = sd_group .* (cholesky(Symmetric(fixture.A)).L *
                               randn(oracle_rng, length(sd_group)))
    eta_fixed = predict(fit, data; type = :link)
    expected = Float64[rand(oracle_rng, Distributions.Poisson(
        exp(eta_fixed[i] + latent_draw[fixture.gidx[i]]))) for i in eachindex(data.y)]
    @test actual == expected

    path = tempname()
    open(path, "w") do io
        serialize(io, fit)
    end
    reloaded = open(deserialize, path)
    @test predict_parameters(reloaded, data)[:sd] ≈ fit.scales[:sd]
    boot = bootstrap_result(fit; data, A, B = 1,
        rng = MersenneTwister(20261024), failures = :skip)
    @test boot.attempted == boot.used == 1
    @test boot.failed == 0
    @test Set(row.param for row in boot.summary) == Set([:mu, :sd])
    @test Set(row.coef for row in filter(row -> row.param === :sd, boot.summary)) ==
          Set(DRModels._block_coefnames(fit, :sd))

    varying = merge(data, (; z = copy(data.z)))
    varying.z[1] = 1.0 - varying.z[1]
    @test_throws ArgumentError drm(formula, Poisson(); data = varying, A, se = false)
    @test_throws ArgumentError drm(formula, Poisson(); data, A = nothing, se = false)
    @test_throws ArgumentError drm(formula, Poisson(); data, A, Ainv, se = false)
    @test_throws ArgumentError drm(formula, Poisson(); data,
        Ainv = ones(size(Ainv)), se = false)
end

@testset "Poisson animal SD regression preserves supplied covariance scale" begin
    fixture = _poisson_animal_sd_fixture(; seed = 20261026)
    rng = MersenneTwister(2026102601)
    data, A = fixture.data, fixture.A
    alpha = fixture.alpha
    latent = cholesky(Symmetric(A)).L * randn(rng, length(fixture.zgroup))
    eta = 0.25 .+ 0.2 .* data.x .+
          exp.(alpha[1] .+ alpha[2] .* fixture.zgroup)[fixture.gidx] .* latent[fixture.gidx]
    raw_data = merge(data, (; y = Float64[
        rand(rng, Distributions.Poisson(exp(value))) for value in eta
    ]))
    formula = bf(@formula(y ~ x + animal(1 | g)),
                 @formula(sd(g, animal) ~ z))
    fit = drm(formula, Poisson(); data = raw_data, A, se = false)

    Xmu = hcat(ones(length(raw_data.y)), raw_data.x)
    Zobs = hcat(ones(length(raw_data.y)), raw_data.z)
    theta = [0.25, 0.2, alpha...]
    raw_dense_nll = _poisson_animal_sd_dense_laplace(
        theta, raw_data.y, Xmu, Zobs, fixture.gidx, A,
    )
    @test fit.nll(theta) ≈ raw_dense_nll atol = 3e-4

    seed = 2026102602
    simulator = DRModels._marginal_simulator_build(fit, raw_data; A)
    actual = simulator(MersenneTwister(seed))
    oracle_rng = MersenneTwister(seed)
    sd_group = exp.(hcat(ones(length(fixture.zgroup)), fixture.zgroup) *
                     coef(fit, :sd))
    latent_draw = sd_group .* (cholesky(Symmetric(A)).L *
                               randn(oracle_rng, length(sd_group)))
    eta_fixed = predict(fit, raw_data; type = :link)
    expected = Float64[rand(oracle_rng, Distributions.Poisson(
        exp(eta_fixed[i] + latent_draw[fixture.gidx[i]]),
    )) for i in eachindex(raw_data.y)]
    @test actual == expected
end
