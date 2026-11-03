using DRModels, Test, Random, LinearAlgebra, Serialization, Statistics
import Distributions

function _gamma_relmat_sd_fixture(; G = 32, m = 16, seed = 20261022)
    rng = MersenneTwister(seed)
    order = randperm(rng, G)
    ids = ["id_$(i)" for i in order]
    R = 0.5 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    scale = collect(range(0.75, 1.5; length = G))
    K = Diagonal(scale) * R * Diagonal(scale)
    group = repeat(ids, inner = m)
    gidx = repeat(1:G, inner = m)
    zgroup = [zeros(G ÷ 2); ones(G ÷ 2)][randperm(rng, G)]
    z = zgroup[gidx]
    x = randn(rng, G * m)
    alpha = [-0.65, 0.4]
    Rlatent = cholesky(Symmetric(K)).L * randn(rng, G)   # raw K, as the fit uses
    eta = 0.25 .+ 0.2 .* x .+
          exp.(alpha[1] .+ alpha[2] .* zgroup)[gidx] .* Rlatent[gidx]
    sigma = 0.32
    shape = 1 / sigma^2
    y = [rand(rng, Distributions.Gamma(shape, exp(v) / shape)) for v in eta]
    return (; data = (; y = Float64.(y), x, z, g = group), K, R = K, gidx,  # R = raw K
            zgroup, alpha, sigma)
end

@testset "Gamma relmat SD fit reports the low-variation boundary" begin
    rng = MersenneTwister(20261025)
    G, m, sigma = 12, 5, 0.01
    group = repeat(1:G, inner = m)
    zgroup = repeat([-1.0, 1.0], inner = G ÷ 2)
    data = (; y = [rand(rng, Distributions.Gamma(1 / sigma^2,
                                                exp(0.3) * sigma^2))
                   for _ in 1:(G * m)],
            g = group, z = repeat(zgroup, inner = m))
    @test std(data.y) / mean(data.y) < exp(-4)
    fit = @test_logs (:warn, r"Wald standard errors are withheld") drm(
        bf(@formula(y ~ relmat(1 | g)),
           @formula(sigma ~ 1),
           @formula(sd(g, relmat) ~ z)), Gamma(); data,
        K = Matrix{Float64}(I, G, G))
    @test isfinite(coef(fit, :sigma)[1])
    @test coef(fit, :sigma)[1] ≈ -4 atol = 1e-5
    @test all(isnan, fit.vcov)
end

function _gamma_relmat_sd_dense_laplace(theta, y, Xmu, Zobs, gidx, R)
    Q = inv(Symmetric(R))
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
    result = DRModels.Optim.optimize(joint, zeros(size(R, 1)),
        DRModels.Optim.LBFGS(), DRModels.Optim.Options(
            g_tol = 1e-10, iterations = 400); autodiff = :forward)
    mode = DRModels.Optim.minimizer(result)
    hessian = DRModels.ForwardDiff.hessian(joint, mode)
    return joint(mode) + 0.5 * logdet(Symmetric(hessian)) -
           0.5 * logdet(Symmetric(Q))
end

@testset "Gamma relmat random-effect SD regression" begin
    fixture = _gamma_relmat_sd_fixture()
    data, K = fixture.data, fixture.K
    formula = bf(@formula(y ~ x + relmat(1 | g)),
                 @formula(sigma ~ 1),
                 @formula(sd(g, relmat) ~ z))
    fit = drm(formula, Gamma(); data, K)

    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sigma, :sd]
    @test fit.scales[:sigma][1] ≈ exp(coef(fit, :sigma)[1])
    @test coef(fit, :sd)[2] > 0
    @test abs(coef(fit, :sd)[2] - fixture.alpha[2]) < 0.55
    @test fit.scales[:sd_group] ≈ exp.(hcat(ones(length(fixture.zgroup)),
                                             fixture.zgroup) * coef(fit, :sd))
    @test predict_parameters(fit, data)[:sd] ≈ fit.scales[:sd]
    @test all(isfinite, predict_parameters(fit, data; se = true)[:sd].se)

    Xmu = hcat(ones(length(data.y)), data.x)
    Zobs = hcat(ones(length(data.y)), data.z)
    theta = [0.25, 0.2, log(fixture.sigma), fixture.alpha...]
    dense_nll = _gamma_relmat_sd_dense_laplace(
        theta, data.y, Xmu, Zobs, fixture.gidx, fixture.R)
    @test fit.nll(theta) ≈ dense_nll atol = 3e-3

    # Above log(sigma) = 3, the objective must still use the same sigma that
    # fit objects and the marginal simulator report.
    theta_high_sigma = [0.25, 0.2, 3.2, fixture.alpha...]
    dense_high_sigma = _gamma_relmat_sd_dense_laplace(
        theta_high_sigma, data.y, Xmu, Zobs, fixture.gidx, fixture.R)
    @test fit.nll(theta_high_sigma) ≈ dense_high_sigma atol = 3e-3

    constant_formula = bf(@formula(y ~ x + relmat(1 | g)),
                          @formula(sigma ~ 1),
                          @formula(sd(g, relmat) ~ 1))
    constant_fit = drm(constant_formula, Gamma(); data, K, se = false)
    base_fit = drm(bf(@formula(y ~ x + relmat(1 | g)),
                      @formula(sigma ~ 1)), Gamma(); data, K, se = false)
    @test constant_fit.loglik ≈ base_fit.loglik atol = 3e-5
    @test coef(constant_fit, :mu) ≈ coef(base_fit, :mu) atol = 1e-4
    @test coef(constant_fit, :sigma) ≈ coef(base_fit, :sigma) atol = 1e-4

    seed = 20261024
    simulator = DRModels._marginal_simulator_build(fit, data; K)
    actual = simulator(MersenneTwister(seed))
    oracle_rng = MersenneTwister(seed)
    sd_group = exp.(hcat(ones(length(fixture.zgroup)), fixture.zgroup) *
                    coef(fit, :sd))
    latent = sd_group .* (cholesky(Symmetric(fixture.R)).L *
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
    boot = bootstrap_result(fit; data, K, B = 1,
        rng = MersenneTwister(20261023), failures = :skip, check_converged = false)
    @test boot.attempted == boot.used == 1
    @test boot.failed == 0
end
