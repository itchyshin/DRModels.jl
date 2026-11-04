using DRModels, Test, Random, LinearAlgebra, SparseArrays
import Distributions

function _binomial_animal_ainv_sd_fixture(; G = 72, m = 24, seed = 20261023)
    rng = MersenneTwister(seed)
    order = randperm(rng, G)
    labels = ["animal_$(i)" for i in order]
    R = 0.4 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    marginal_sd = collect(range(0.7, 1.5; length = G))
    A = Diagonal(marginal_sd) * R * Diagonal(marginal_sd)
    group = repeat(labels, inner = m)
    gidx = repeat(1:G, inner = m)
    zgroup = collect(range(-1.0, 1.0; length = G))
    z = zgroup[gidx]
    x = randn(rng, G * m)
    alpha = [-0.9, 0.35]
    latent = cholesky(Symmetric(A)).L * randn(rng, G)
    eta = -0.25 .+ 0.2 .* x .+ exp.(alpha[1] .+ alpha[2] .* zgroup)[gidx] .* latent[gidx]
    y = Float64[rand(rng, Distributions.Bernoulli(1 / (1 + exp(-v)))) for v in eta]
    return (; data = (; y, x, z, g = group), A, alpha, zgroup)
end

@testset "Binomial animal SD regression accepts direct precision" begin
    fixture = _binomial_animal_ainv_sd_fixture()
    data, A = fixture.data, fixture.A
    formula = bf(@formula(y ~ x + animal(1 | g)),
                 @formula(sd(g, animal) ~ z))
    covariance_fit = drm(formula, Binomial(); data, A, se = false)
    Ainv = inv(Symmetric(A))
    precision_fit = drm(formula, Binomial(); data, Ainv, se = false)

    @test covariance_fit.converged
    @test precision_fit.converged
    # The outer finite-difference optimizer can stop at slightly different
    # points on weakly identified SD surfaces; the Laplace likelihood remains
    # the stricter equivalence check below.
    @test coef(precision_fit, :sd) ≈ coef(covariance_fit, :sd) atol = 5e-3
    @test precision_fit.loglik ≈ covariance_fit.loglik atol = 2e-5
    @test predict_parameters(precision_fit, data)[:sd] ≈
          predict_parameters(covariance_fit, data)[:sd] atol = 5e-3

    # 24000 draws put the 0.08 tolerance near four Monte Carlo standard
    # errors for the largest variance (2.25); 1500 draws left it near one.
    oracle_rng = MersenneTwister(20261024)
    qchol = cholesky(Symmetric(sparse(Ainv)))
    latent_draws = hcat((DRModels._binomial_animal_precision_draw(
        oracle_rng, qchol, size(Ainv, 1)) for _ in 1:24000)...)
    latent_draws .-= sum(latent_draws; dims = 2) ./ size(latent_draws, 2)
    sample_cov = latent_draws * latent_draws' / size(latent_draws, 2)
    diag_probe = [1, size(A, 1) ÷ 2, size(A, 1)]
    @test diag(sample_cov)[diag_probe] ≈ diag(A)[diag_probe] atol = 0.08
    @test sample_cov[1, 2] ≈ A[1, 2] atol = 0.05
    simulated = DRModels._marginal_simulator(precision_fit, data; Ainv)(
        MersenneTwister(20261026))
    @test length(simulated) == length(data.y)
    @test all(in((0.0, 1.0)), simulated)

    constant_formula = bf(@formula(y ~ x + animal(1 | g)),
                          @formula(sd(g, animal) ~ 1))
    constant_A = drm(constant_formula, Binomial(); data, A, se = false)
    constant_Ainv = drm(constant_formula, Binomial(); data, Ainv, se = false)
    @test constant_Ainv.loglik ≈ constant_A.loglik atol = 2e-5
    @test coef(constant_Ainv, :sd) ≈ coef(constant_A, :sd) atol = 2e-5

    boot = bootstrap_result(precision_fit; data, Ainv, B = 1,
        rng = MersenneTwister(20261025), failures = :skip)
    @test boot.attempted == boot.used == 1
    @test boot.failed == 0
    @test Set(row.param for row in boot.summary) == Set([:mu, :sd])

    @test_throws ArgumentError drm(formula, Binomial(); data, A, Ainv, se = false)
    @test_throws ArgumentError drm(formula, Binomial(); data,
        Ainv = ones(size(Ainv)), se = false)
    asymmetric = Matrix(Ainv)
    asymmetric[1, 2] += 0.2
    @test_throws ArgumentError drm(formula, Binomial(); data, Ainv = asymmetric, se = false)
end
