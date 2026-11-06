using DRModels, Test, Random, LinearAlgebra, Serialization
import Distributions

function _gamma_animal_slope_sd_fixture(; G = 48, m = 20,
                                        seed = 20261023)
    rng = MersenneTwister(seed)
    order = randperm(rng, G)
    ids = ["animal_$(i)" for i in order]
    R = 0.4 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    scale = collect(range(0.7, 1.4; length = G))
    A = Diagonal(scale) * R * Diagonal(scale)
    group = repeat(ids, inner = m)
    gidx = repeat(1:G, inner = m)
    zgroup = ifelse.(randperm(rng, G) .<= G ÷ 2, -1.0, 1.0)
    z = zgroup[gidx]
    xwithin = collect(range(-1.1, 1.1; length = m))
    x = repeat(xwithin, outer = G)
    alpha = [-0.65, 0.4]
    u = cholesky(Symmetric(A)).L * randn(rng, G)
    eta = 0.15 .+ 0.25 .* x .+ x .*
          (exp.(alpha[1] .+ alpha[2] .* zgroup) .* u)[gidx]
    sigma = 0.35
    shape = 1 / sigma^2
    y = [rand(rng, Distributions.Gamma(shape, exp(v) / shape)) for v in eta]
    return (; data = (; y = Float64.(y), x, z, g = group), A, R, gidx,
            zgroup, alpha, sigma)
end

function _gamma_animal_slope_dense_laplace(theta, y, Xmu, x, Zobs, gidx, A)
    Q = inv(Symmetric(A))
    p_mu = size(Xmu, 2)
    beta = theta[1:p_mu]
    sigma = exp(theta[p_mu + 1])
    alpha = theta[p_mu + 2:end]
    shape = 1 / sigma^2
    eta0 = Xmu * beta
    loading = x .* exp.(Zobs * alpha)
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

@testset "Gamma animal random-slope SD regression" begin
    fixture = _gamma_animal_slope_sd_fixture()
    data, A = fixture.data, fixture.A
    formula = bf(@formula(y ~ x + animal(0 + x | g)),
                 @formula(sigma ~ 1),
                 @formula(sd(g, animal) ~ z))
    fit = drm(formula, Gamma(); data, A)

    @test fit.converged
    @test coef(fit, :sd)[2] > 0
    @test abs(coef(fit, :sd)[2] - fixture.alpha[2]) < 0.45
    @test fit.scales[:sd_group] ≈ exp.(hcat(ones(length(fixture.zgroup)),
                                             fixture.zgroup) * coef(fit, :sd))
    @test fit.scales[:sd] ≈ fit.scales[:sd_group][fixture.gidx]
    @test predict_parameters(fit, data; se = false)[:sd] ≈ fit.scales[:sd]

    Xmu = hcat(ones(length(data.y)), data.x)
    Zobs = hcat(ones(length(data.y)), data.z)
    theta = [0.15, 0.25, log(fixture.sigma), fixture.alpha...]
    dense_nll = _gamma_animal_slope_dense_laplace(
        theta, data.y, Xmu, data.x, Zobs, fixture.gidx, fixture.A)
    @test fit.nll(theta) ≈ dense_nll atol = 4e-3

    constant_formula = bf(@formula(y ~ x + animal(0 + x | g)),
                          @formula(sigma ~ 1),
                          @formula(sd(g, animal) ~ 1))
    constant_fit = drm(constant_formula, Gamma(); data, A, se = false)
    base_fit = drm(bf(@formula(y ~ x + animal(0 + x | g)),
                      @formula(sigma ~ 1)), Gamma(); data, A, se = false)
    @test constant_fit.loglik ≈ base_fit.loglik atol = 3e-5

    seed = 20261024
    simulator = DRModels._marginal_simulator_build(fit, data; A)
    actual = simulator(MersenneTwister(seed))
    oracle_rng = MersenneTwister(seed)
    sd_group = exp.(hcat(ones(length(fixture.zgroup)), fixture.zgroup) *
                    coef(fit, :sd))
    latent = sd_group .* (cholesky(Symmetric(fixture.A)).L *
                          randn(oracle_rng, length(sd_group)))
    eta_fixed = predict(fit, data; type = :link) .+ data.x .* latent[fixture.gidx]
    shape = 1 / exp(2 * coef(fit, :sigma)[1])
    mu_star = exp.(eta_fixed)
    expected = [rand(oracle_rng, Distributions.Gamma(shape, mu / shape))
                for mu in mu_star]
    @test actual ≈ expected atol = 1e-12 rtol = 1e-12

    path = tempname()
    open(path, "w") do io
        serialize(io, fit)
    end
    reloaded = open(deserialize, path)
    @test predict_parameters(reloaded, data; se = false)[:sd] ≈ fit.scales[:sd]
    boot = bootstrap_result(fit; data, A, B = 1,
        rng = MersenneTwister(20261025), failures = :skip)
    @test boot.attempted == boot.used == 1
    @test boot.failed == 0

    data_bad_sd = merge(data, (; z = collect(1.0:length(data.z))))
    @test_throws ArgumentError drm(formula, Gamma(); data = data_bad_sd, A,
                                   se = false)

    intercept_slope_formula = bf(
        @formula(y ~ x + animal(1 + x | g)), @formula(sigma ~ 1),
        @formula(sd(g, animal) ~ z))
    animal_block_error = try
        drm(intercept_slope_formula, Gamma(); data, A, se = false)
        nothing
    catch err
        err
    end
    @test animal_block_error isa ArgumentError
    @test occursin("needs one SD formula for the intercept and one for the slope",
                   sprint(showerror, animal_block_error))
    mismatched_selector_formula = bf(
        @formula(y ~ x + animal(0 + x | g)), @formula(sigma ~ 1),
        @formula(sd(g, coef(x)) ~ z))
    @test_throws ArgumentError drm(mismatched_selector_formula, Gamma(); data, A,
                                   se = false)
end
