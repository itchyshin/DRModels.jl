# NB2 relmat slope-only SD regression: relmat(0 + x | id, K) with sd(id, relmat) ~ z.
# K enters RAW (not correlation-rescaled), as in native drmTMB. Dense Laplace oracle.
using DRModels, Test, Random, LinearAlgebra
import Distributions
using SpecialFunctions: loggamma

function _nb2_relmat_slope_dense(theta, y, Xmu, loading, gidx, C)
    Q = inv(Symmetric(C))
    r = 1 / exp(theta[3])^2
    eta0 = Xmu * theta[1:2]
    function joint(b)
        value = 0.5 * dot(b, Q * b)
        for i in eachindex(y)
            eta = eta0[i] + loading[i] * b[gidx[i]]
            lse = max(eta, log(r)) + log1p(exp(-abs(eta - log(r))))
            value -= loggamma(y[i] + r) - loggamma(r) - loggamma(y[i] + 1) +
                     r * (log(r) - lse) + y[i] * (eta - lse)
        end
        value
    end
    res = DRModels.Optim.optimize(joint, zeros(size(C, 1)), DRModels.Optim.LBFGS(),
        DRModels.Optim.Options(g_tol = 1e-10, iterations = 500); autodiff = :forward)
    mode = DRModels.Optim.minimizer(res)
    H = DRModels.ForwardDiff.hessian(joint, mode)
    return joint(mode) + 0.5 * logdet(Symmetric(H)) - 0.5 * logdet(Symmetric(Q))
end

@testset "NB2 relmat slope-only SD regression (raw K, dense oracle)" begin
    rng = MersenneTwister(20261105)
    G, m = 30, 20
    R = 0.4 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    scale = collect(range(0.7, 1.7; length = G))
    K = Diagonal(scale) * R * Diagonal(scale)
    gidx = repeat(1:G, inner = m)
    zgroup = repeat([-1.0, 1.0], inner = G ÷ 2)
    alpha = [-0.6, 0.45]
    b = cholesky(Symmetric(K)).L * randn(rng, G)
    x = randn(rng, G * m)
    mu = exp.(0.3 .+ 0.25 .* x .+ x .* (exp.(alpha[1] .+ alpha[2] .* zgroup) .* b)[gidx])
    sz = 1 / 0.42^2
    y = Float64[rand(rng, Distributions.NegativeBinomial(sz, sz / (sz + v))) for v in mu]
    data = (; y, x, z = zgroup[gidx], g = gidx)
    formula = bf(@formula(y ~ x + relmat(0 + x | g)), @formula(sigma ~ 1),
                 @formula(sd(g, relmat) ~ z))
    fit = drm(formula, NegBinomial2(); data, K, se = false)

    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sigma, :sd]
    @test coef(fit, :sd)[2] > 0
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 1.0
    @test fit.scales[:sd_group] ≈ exp.(hcat(ones(G), zgroup) * coef(fit, :sd))
    @test predict_parameters(fit, data)[:sd] ≈ fit.scales[:sd_group][gidx]

    Xmu = hcat(ones(length(y)), x)
    theta = vcat(coef(fit, :mu), coef(fit, :sigma), coef(fit, :sd))
    loading = x .* exp.(theta[4] .+ theta[5] .* data.z)
    oracle = _nb2_relmat_slope_dense(theta, y, Xmu, loading, gidx, K)
    @test fit.nll(theta) ≈ oracle atol = 2e-3
    Rn = K ./ (sqrt.(diag(K)) * sqrt.(diag(K))')
    @test abs(_nb2_relmat_slope_dense(theta, y, Xmu, loading, gidx, Rn) - oracle) > 0.05

    @test_throws ArgumentError drm(formula, NegBinomial2(); data, A = K, se = false)
    @test_throws ArgumentError drm(formula, NegBinomial2();
        data = (; y, z = data.z, g = gidx), K, se = false)
end
