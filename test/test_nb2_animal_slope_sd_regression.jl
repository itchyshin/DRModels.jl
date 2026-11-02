# Predictor-dependent NB2 SD for one structured animal random slope.
using DRModels, Test, Random, LinearAlgebra
import Distributions
using SpecialFunctions: loggamma

function _nb2_animal_slope_sd_fixture(; alpha = [-0.65, 0.4], G = 40,
                                       repeats = 24, seed = 20261023)
    rng = MersenneTwister(seed)
    group_index = repeat(1:G; inner = repeats)
    group = group_index
    z_group = repeat([-1.0, 1.0]; inner = G ÷ 2)
    z = z_group[group_index]
    x = randn(rng, length(group_index))
    A = 0.35 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    slope = exp.(alpha[1] .+ alpha[2] .* z_group) .*
            (cholesky(Symmetric(A)).L * randn(rng, G))
    mu = exp.(0.35 .+ 0.25 .* x .+ x .* slope[group_index])
    sigma = 0.42
    size = 1 / sigma^2
    y = Float64[rand(rng, Distributions.NegativeBinomial(
        size, size / (size + value))) for value in mu]
    return (; data = (; y, x, z, g = group), A, z_group, group_index, alpha)
end

@testset "NB2 animal random-slope SD regression" begin
    fixture = _nb2_animal_slope_sd_fixture()
    data = fixture.data
    formula = bf(@formula(y ~ x + animal(0 + x | g)),
                 @formula(sigma ~ 1),
                 @formula(sd(g, animal) ~ z))
    fit = drm(formula, NegBinomial2(); data, A = fixture.A, se = false)

    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sigma, :sd]
    @test sign(coef(fit, :sd)[2]) == sign(fixture.alpha[2])
    @test abs(coef(fit, :sd)[2] - fixture.alpha[2]) < 0.55
    expected_group_sd = exp.(hcat(ones(length(fixture.z_group)),
                                  fixture.z_group) * coef(fit, :sd))
    @test fit.scales[:sd_group] ≈ expected_group_sd
    @test fit.scales[:sd] ≈ expected_group_sd[fixture.group_index]
    @test predict_parameters(fit, data)[:sd] ≈
          expected_group_sd[fixture.group_index]

    bad_data = merge(data, (; z = copy(data.z)))
    bad_data.z[1] = -bad_data.z[1]
    @test_throws ArgumentError drm(formula, NegBinomial2();
        data = bad_data, A = fixture.A, se = false)
end

# Dense Laplace oracle for the animal slope: raw A, loading = x * exp(Z alpha).
@testset "NB2 animal slope SD oracle, raw A, Ainv equivalence" begin
    fixture = _nb2_animal_slope_sd_fixture(; G = 24, repeats = 16, seed = 20261101)
    data = fixture.data
    G = length(fixture.z_group)
    # Non-unit diagonal kernel: the slope SD multiplies the raw A.
    scale = collect(range(0.7, 1.8; length = G))
    A = Diagonal(scale) * fixture.A[1:G, 1:G] * Diagonal(scale)
    formula = bf(@formula(y ~ x + animal(0 + x | g)),
                 @formula(sigma ~ 1),
                 @formula(sd(g, animal) ~ z))
    fit = drm(formula, NegBinomial2(); data, A, se = false)
    @test fit.converged

    Xmu = hcat(ones(length(data.y)), data.x)
    theta = vcat(coef(fit, :mu), coef(fit, :sigma), coef(fit, :sd))
    Q = inv(Symmetric(A))
    r = 1 / exp(theta[3])^2
    loading = data.x .* exp.(fixture.z_group[data.g] .* theta[5] .+ theta[4])
    function joint(b)
        value = 0.5 * dot(b, Q * b)
        for i in eachindex(data.y)
            eta = dot(Xmu[i, :], theta[1:2]) + loading[i] * b[data.g[i]]
            lse = max(eta, log(r)) + log1p(exp(-abs(eta - log(r))))   # log(r + mu)
            value -= loggamma(data.y[i] + r) - loggamma(r) - loggamma(data.y[i] + 1) +
                     r * (log(r) - lse) + data.y[i] * (eta - lse)
        end
        value
    end
    res = DRModels.Optim.optimize(joint, zeros(G), DRModels.Optim.LBFGS(),
        DRModels.Optim.Options(g_tol = 1e-10, iterations = 400); autodiff = :forward)
    mode = DRModels.Optim.minimizer(res)
    H = DRModels.ForwardDiff.hessian(joint, mode)
    oracle = joint(mode) + 0.5 * logdet(Symmetric(H)) - 0.5 * logdet(Symmetric(Q))
    @test fit.nll(theta) ≈ oracle atol = 2e-3

    precision_fit = drm(formula, NegBinomial2(); data, Ainv = inv(A), se = false)
    @test coef(precision_fit, :sd) ≈ coef(fit, :sd) atol = 1e-4
    @test precision_fit.loglik ≈ fit.loglik atol = 1e-4

    # Constant SD formula reduces to the plain animal-slope fit of the same likelihood.
    constant_fit = drm(bf(@formula(y ~ x + animal(0 + x | g)), @formula(sigma ~ 1),
                          @formula(sd(g, animal) ~ 1)), NegBinomial2(); data, A, se = false)
    @test constant_fit.loglik <= fit.loglik + 1e-6

    # Refusals: member mismatch, slope with relmat route, missing slope column.
    @test_throws ArgumentError drm(bf(@formula(y ~ x + animal(0 + x | g)),
        @formula(sigma ~ 1), @formula(sd(g, animal, coef(z)) ~ z)),
        NegBinomial2(); data, A, se = false)
    @test_throws ArgumentError drm(formula, NegBinomial2();
        data = (; data.y, data.x, data.z, g = data.g), K = A, se = false)
end
