using DRModels, Test, Random, LinearAlgebra, Serialization
import Distributions

function _gamma_animal_sd_fixture(; G = 32, m = 16, seed = 20261022)
    rng = MersenneTwister(seed)
    order = randperm(rng, G)
    ids = ["animal_$(i)" for i in order]
    R = 0.5 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    scale = collect(range(0.75, 1.5; length = G))
    A = Diagonal(scale) * R * Diagonal(scale)
    group = repeat(ids, inner = m)
    gidx = repeat(1:G, inner = m)
    zgroup = [zeros(G ÷ 2); ones(G - G ÷ 2)][randperm(rng, G)]
    z = zgroup[gidx]
    x = randn(rng, G * m)
    alpha = [-0.65, 0.4]
    u = cholesky(Symmetric(A)).L * randn(rng, G)
    eta = 0.2 .+ 0.18 .* x .+ exp.(alpha[1] .+ alpha[2] .* zgroup)[gidx] .* u[gidx]
    sigma = 0.32
    shape = 1 / sigma^2
    y = [rand(rng, Distributions.Gamma(shape, exp(v) / shape)) for v in eta]
    return (; data = (; y = Float64.(y), x, z, g = group), A, R, gidx,
            zgroup, alpha, sigma)
end

function _gamma_animal_sd_dense_laplace(theta, y, Xmu, Zobs, gidx, A)
    Q = inv(Symmetric(A))
    p_mu = size(Xmu, 2)
    beta = theta[1:p_mu]
    sigma = exp(theta[p_mu + 1])
    alpha = theta[p_mu + 2:end]
    shape = 1 / sigma^2
    eta0 = Xmu * beta
    loading = exp.(Zobs * alpha)
    function joint(b)
        value = 0.5 * dot(b, Q * b)
        for i in eachindex(y)
            mu = exp(eta0[i] + loading[i] * b[gidx[i]])
            value -= Distributions.logpdf(
                Distributions.Gamma(shape, mu / shape), y[i])
        end
        value
    end
    result = DRModels.Optim.optimize(joint, zeros(size(A, 1)),
        DRModels.Optim.LBFGS(), DRModels.Optim.Options(
            g_tol = 1e-10, iterations = 400); autodiff = :forward)
    mode = DRModels.Optim.minimizer(result)
    hessian = DRModels.ForwardDiff.hessian(joint, mode)
    return joint(mode) + 0.5 * logdet(Symmetric(hessian)) -
           0.5 * logdet(Symmetric(Q))
end

@testset "Gamma animal random-effect SD regression" begin
    fixture = _gamma_animal_sd_fixture()
    data, A = fixture.data, fixture.A
    formula = bf(@formula(y ~ x + animal(1 | g)),
                 @formula(sigma ~ 1),
                 @formula(sd(g, animal) ~ z))
    fit = drm(formula, Gamma(); data, A)

    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sigma, :sd]
    @test coef(fit, :sd)[2] > 0
    @test abs(coef(fit, :sd)[2] - fixture.alpha[2]) < 0.65
    @test fit.scales[:sd_group] ≈ exp.(hcat(ones(length(fixture.zgroup)),
                                             fixture.zgroup) * coef(fit, :sd))
    @test predict_parameters(fit, data)[:sd] ≈ fit.scales[:sd]
    @test all(isfinite, predict_parameters(fit, data; se = true)[:sd].se)

    Xmu = hcat(ones(length(data.y)), data.x)
    Zobs = hcat(ones(length(data.y)), data.z)
    theta = [0.2, 0.18, log(fixture.sigma), fixture.alpha...]
    dense_nll = _gamma_animal_sd_dense_laplace(
        theta, data.y, Xmu, Zobs, fixture.gidx, fixture.A)
    @test fit.nll(theta) ≈ dense_nll atol = 3e-3

    constant_formula = bf(@formula(y ~ x + animal(1 | g)),
                          @formula(sigma ~ 1),
                          @formula(sd(g, animal) ~ 1))
    constant_fit = drm(constant_formula, Gamma(); data, A, se = false)
    base_fit = drm(bf(@formula(y ~ x + animal(1 | g)),
                      @formula(sigma ~ 1)), Gamma(); data, A, se = false)
    @test constant_fit.loglik ≈ base_fit.loglik atol = 3e-5
    constant_theta = theta[1:4]
    @test constant_fit.nll(constant_theta) ≈ _gamma_animal_sd_dense_laplace(
        constant_theta, data.y, Xmu, ones(length(data.y), 1), fixture.gidx,
        fixture.A) atol = 3e-3
    base_theta = [coef(base_fit, :mu); coef(base_fit, :sigma);
                  coef(base_fit, :resd)]
    @test base_fit.nll(base_theta) ≈ _gamma_animal_sd_dense_laplace(
        base_theta, data.y, Xmu, ones(length(data.y), 1), fixture.gidx,
        fixture.A) atol = 3e-3

    seed = 20261027
    simulator = DRModels._marginal_simulator_build(fit, data; A)
    actual = simulator(MersenneTwister(seed))
    oracle_rng = MersenneTwister(seed)
    sd_group = exp.(hcat(ones(length(fixture.zgroup)), fixture.zgroup) *
                    coef(fit, :sd))
    latent = sd_group .* (cholesky(Symmetric(fixture.A)).L *
                          randn(oracle_rng, length(sd_group)))
    eta_fixed = predict(fit, data; type = :link)
    shape = 1 / exp(2 * coef(fit, :sigma)[1])
    mu_star = exp.(eta_fixed .+ latent[fixture.gidx])
    expected = [rand(oracle_rng, Distributions.Gamma(shape, mu / shape))
                for mu in mu_star]
    @test actual ≈ expected atol = 1e-12 rtol = 1e-12

    path = tempname()
    open(path, "w") do io
        serialize(io, fit)
    end
    reloaded = open(deserialize, path)
    @test predict_parameters(reloaded, data)[:sd] ≈ fit.scales[:sd]
    boot = bootstrap_result(fit; data, A, B = 1,
        rng = MersenneTwister(20261028), failures = :skip)
    @test boot.attempted == boot.used == 1
    @test boot.failed == 0
end
