# Predictor-dependent random-effect SD for a NegBinomial2 relmat intercept.
# Candidate-level test: recovery sign/size, internal consistency with the
# animal route and a dense-Laplace oracle. No calibration claim.
using DRModels, Test, Random, LinearAlgebra
import Distributions

function _nb2_relmat_sd_fixture(alpha; G = 40, m = 24, seed = 20261101)
    rng = MersenneTwister(seed)
    g = repeat(1:G, inner = m)
    zgroup = repeat([-1.0, 1.0], inner = G ÷ 2)
    z = zgroup[g]
    x = randn(rng, length(g))
    R = 0.35 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    latent = exp.(alpha[1] .+ alpha[2] .* zgroup) .* (cholesky(Symmetric(R)).L * randn(rng, G))
    mu = exp.(0.35 .+ 0.25 .* x .+ latent[g])
    sigma = 0.42
    size = 1 / sigma^2
    y = [rand(rng, Distributions.NegativeBinomial(size, size / (size + value)))
         for value in mu]
    # Scramble rows and use string IDs so the K ordering must follow first-seen IDs.
    order = randperm(rng, length(g))
    fit_levels = unique(g[order])
    ids = ["id_$(i)" for i in g[order]]
    data = (; y = Float64.(y[order]), x = x[order], z = z[order], g = ids)
    return (; data, R = R[fit_levels, fit_levels], zgroup = zgroup[fit_levels],
            fit_levels, alpha)
end

# Dense Laplace oracle with the same unit-diagonal-correlation convention.
function _nb2_relmat_sd_dense_laplace(theta, y, Xmu, gidx, zobs, R)
    Q = inv(Symmetric(R))
    pmu = size(Xmu, 2)
    beta = theta[1:pmu]
    sig = exp(theta[pmu + 1])
    alpha = theta[(pmu + 2):end]
    r = 1 / sig^2
    eta0 = Xmu * beta
    loading = exp.(hcat(ones(length(y)), zobs) * alpha)
    function joint(b)
        value = 0.5 * dot(b, Q * b)
        for i in eachindex(y)
            eta = eta0[i] + loading[i] * b[gidx[i]]
            mu = exp(eta)
            value -= Distributions.logpdf(
                Distributions.NegativeBinomial(r, r / (r + mu)), round(Int, y[i]))
        end
        value
    end
    res = DRModels.Optim.optimize(joint, zeros(size(R, 1)),
        DRModels.Optim.LBFGS(), DRModels.Optim.Options(g_tol = 1e-10, iterations = 400);
        autodiff = :forward)
    mode = DRModels.Optim.minimizer(res)
    H = DRModels.ForwardDiff.hessian(joint, mode)
    return joint(mode) + 0.5 * logdet(Symmetric(H)) - 0.5 * logdet(Symmetric(Q))
end

@testset "NB2 relmat location-intercept SD regression" begin
    fx = _nb2_relmat_sd_fixture([-0.65, 0.4])
    data, R, zgroup, fit_levels, alpha = fx.data, fx.R, fx.zgroup, fx.fit_levels, fx.alpha
    formula = bf(@formula(y ~ x + relmat(1 | g)),
                 @formula(sigma ~ 1),
                 @formula(sd(g, relmat) ~ z))
    # Non-unit-diagonal covariance: the SD is defined against the implied
    # correlation, so the scaled K must reproduce the correlation-matrix fit.
    scale = collect(range(0.7, 1.8; length = length(zgroup)))
    K = Diagonal(scale) * R * Diagonal(scale)
    fit = drm(formula, NegBinomial2(); data, K, se = false)

    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sigma, :sd]
    @test length(coef(fit, :sd)) == 2
    @test sign(coef(fit, :sd)[2]) == sign(alpha[2])
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.55
    expected_group_sd = exp.(hcat(ones(length(zgroup)), zgroup) * coef(fit, :sd))
    @test fit.scales[:sd_group] ≈ expected_group_sd
    @test length(fit.scales[:sd]) == length(data.y)
    @test predict_parameters(fit, data)[:sd] ≈ fit.scales[:sd]

    # Scaling invariance and agreement with the animal route on the same kernel.
    corr_fit = drm(formula, NegBinomial2(); data, K = R, se = false)
    @test coef(corr_fit, :sd) ≈ coef(fit, :sd) atol = 1e-4
    @test corr_fit.loglik ≈ fit.loglik atol = 1e-4
    animal_formula = bf(@formula(y ~ x + animal(1 | g)),
                        @formula(sigma ~ 1),
                        @formula(sd(g, animal) ~ z))
    animal_fit = drm(animal_formula, NegBinomial2(); data, A = R, se = false)
    @test animal_fit.converged
    @test coef(animal_fit, :mu) ≈ coef(corr_fit, :mu) atol = 1e-5
    @test coef(animal_fit, :sigma) ≈ coef(corr_fit, :sigma) atol = 1e-5
    @test coef(animal_fit, :sd) ≈ coef(corr_fit, :sd) atol = 1e-5
    @test animal_fit.loglik ≈ corr_fit.loglik atol = 1e-5

    # Independent dense-Laplace oracle at the fitted parameters.
    gidx = [findfirst(==(id), unique(data.g)) for id in data.g]
    Xmu = hcat(ones(length(data.y)), data.x)
    theta = vcat(coef(corr_fit, :mu), coef(corr_fit, :sigma), coef(corr_fit, :sd))
    dense_nll = _nb2_relmat_sd_dense_laplace(theta, data.y, Xmu, gidx, data.z, R)
    @test corr_fit.nll(theta) ≈ dense_nll atol = 2e-3

    # Constant SD formula reduces to the plain relmat NB2 fit.
    constant_formula = bf(@formula(y ~ x + relmat(1 | g)),
                          @formula(sigma ~ 1),
                          @formula(sd(g, relmat) ~ 1))
    constant_fit = drm(constant_formula, NegBinomial2(); data, K = R, se = false)
    @test constant_fit.converged
    @test length(coef(constant_fit, :sd)) == 1

    # Gate errors (refusals, not silent fallbacks).
    @test_throws Exception drm(formula, NegBinomial2(); data, se = false)            # K missing
    @test_throws ArgumentError drm(formula, NegBinomial2(); data, K = R, A = R, se = false)
    @test_throws ArgumentError drm(formula, NegBinomial2(); data, K = R, Ainv = inv(R), se = false)
    @test_throws ArgumentError drm(animal_formula, NegBinomial2(); data, K = R, A = R, se = false)
    @test_throws ArgumentError drm(formula, NegBinomial2(); data, K = R,
                                   coords = zeros(length(zgroup), 2), se = false)
    @test_throws ArgumentError drm(formula, NegBinomial2(); data, K = R, nAGQ = 5, se = false)
    ordinary = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1), @formula(sd(g) ~ z))
    @test_throws ArgumentError drm(ordinary, NegBinomial2(); data, K = R, se = false)
    # `_general_cov_setup` reports a size mismatch with `error`.
    @test_throws ErrorException drm(formula, NegBinomial2(); data,
                                     K = R[1:end-1, 1:end-1], se = false)
end
