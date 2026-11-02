# Predictor-dependent random-effect SDs for a Poisson relmat intercept.
using DRModels, Test, Random, LinearAlgebra, Serialization
import Distributions

function _poisson_relmat_sd_fixture(; G = 40, m = 16)
    rng = MersenneTwister(20261021)
    order = randperm(rng, G)
    ids = ["id_$(i)" for i in order]
    R = 0.55 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    # The Poisson relmat route now uses the supplied K AS SUPPLIED (raw, as native
    # drmTMB does), so the latent field and the dense oracle below use K itself.
    scale = collect(range(0.7, 1.8; length = G))
    K = Diagonal(scale) * R * Diagonal(scale)
    group_id = repeat(ids, inner = m)
    gidx = repeat(1:G, inner = m)
    zgroup = [zeros(G ÷ 2); ones(G ÷ 2)][randperm(rng, G)]
    z = zgroup[gidx]
    x = randn(rng, G * m)
    alpha = [-0.5, 0.35]
    latent = cholesky(Symmetric(K)).L * randn(rng, G)
    eta = 0.35 .+ 0.2 .* x .+ exp.(alpha[1] .+ alpha[2] .* zgroup)[gidx] .* latent[gidx]
    y = Float64[rand(rng, Distributions.Poisson(exp(v))) for v in eta]
    return (; data = (; y, x, z, g = group_id), K, R = K, gidx, zgroup, alpha)  # R = raw K
end

function _poisson_relmat_sd_dense_laplace(theta, y, Xmu, Zobs, gidx, R)
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

@testset "Poisson relmat random-effect SD regression" begin
    fixture = _poisson_relmat_sd_fixture()
    data, K = fixture.data, fixture.K
    formula = bf(@formula(y ~ x + relmat(1 | g)),
                 @formula(sd(g, relmat) ~ z))
    fit = drm(formula, Poisson(); data, K)

    @test fit.converged
    @test length(coef(fit, :sd)) == 2
    @test abs(coef(fit, :sd)[2] - fixture.alpha[2]) < 0.5
    Zgroup = hcat(ones(length(fixture.zgroup)), fixture.zgroup)
    @test fit.scales[:sd_group] ≈ exp.(Zgroup * coef(fit, :sd))
    predicted = predict_parameters(fit, data; se = true)[:sd]
    @test predicted.value ≈ fit.scales[:sd]
    @test all(isfinite, predicted.se)

    Xmu = hcat(ones(length(data.y)), data.x)
    Zobs = hcat(ones(length(data.y)), data.z)
    theta = [0.35, 0.2, fixture.alpha...]
    dense_nll = _poisson_relmat_sd_dense_laplace(
        theta, data.y, Xmu, Zobs, fixture.gidx, fixture.R)
    @test fit.nll(theta) ≈ dense_nll atol = 2e-4

    constant_formula = bf(@formula(y ~ x + relmat(1 | g)),
                          @formula(sd(g, relmat) ~ 1))
    constant_fit = drm(constant_formula, Poisson(); data, K, se = false)
    base_fit = drm(bf(@formula(y ~ x + relmat(1 | g))), Poisson(); data, K, se = false)
    @test constant_fit.loglik ≈ base_fit.loglik atol = 2e-5
    @test coef(constant_fit, :mu) ≈ coef(base_fit, :mu) atol = 1e-4

    seed = 20261022
    simulator = DRModels._marginal_simulator_build(fit, data; K)
    actual = simulator(MersenneTwister(seed))
    oracle_rng = MersenneTwister(seed)
    latent = fit.scales[:sd_group] .* (cholesky(Symmetric(fixture.R)).L *
                                       randn(oracle_rng, length(fit.scales[:sd_group])))
    eta_fixed = predict(fit, data; type = :link)
    expected = Float64[rand(oracle_rng, Distributions.Poisson(
        exp(eta_fixed[i] + latent[fixture.gidx[i]]))) for i in eachindex(data.y)]
    @test actual == expected

    path = tempname()
    open(path, "w") do io
        serialize(io, fit)
    end
    reloaded = open(deserialize, path)
    @test predict_parameters(reloaded, data)[:sd] ≈ fit.scales[:sd]
    boot = bootstrap_result(fit; data, K, B = 1,
        rng = MersenneTwister(20261023), failures = :skip)
    @test boot.attempted == 1
    @test boot.used == 1
    @test boot.failed == 0
    @test boot.used + boot.failed == 1

    varying = merge(data, (; z = copy(data.z)))
    varying.z[1] = 1.0 - varying.z[1]
    @test_throws ArgumentError drm(formula, Poisson(); data = varying, K, se = false)
    @test_throws ArgumentError drm(formula, Poisson(); data, K = nothing, se = false)
end
