# NB2 coordinate spatial slope-only SD regression: spatial(0 + x | g) with
# sd(g, spatial) ~ z and `coords`. The kernel is the native fixed
# exp(-d / median positive distance) + 1e-6 I, RAW, built from `coords` only.
# The Laplace objective is checked against an independently coded dense oracle.
using DRModels, Test, Random, LinearAlgebra, Statistics
import Distributions
using SpecialFunctions: loggamma

function _nb2_spatial_slope_dense(theta, y, Xmu, loading, gidx, C)
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

function _nb2_native_spatial_kernel(coords)
    G = size(coords, 1)
    D = [sqrt(sum(abs2, coords[i, :] .- coords[j, :])) for i in 1:G, j in 1:G]
    pos = [D[i, j] for i in 1:G for j in (i + 1):G if D[i, j] > 0]
    return exp.(-D ./ median(pos)) + 1e-6 * I
end

@testset "NB2 coordinate spatial slope-only SD regression (native kernel, dense oracle)" begin
    rng = MersenneTwister(20261106)
    G, m = 30, 20
    coords = hcat(mod.(0:(G - 1), 6), div.(0:(G - 1), 6)) .* 1.3
    C = _nb2_native_spatial_kernel(coords)
    gidx = repeat(1:G, inner = m)
    zgroup = repeat([-1.0, 1.0], inner = G ÷ 2)[randperm(rng, G)]
    alpha = [-0.6, 0.45]
    b = cholesky(Symmetric(C)).L * randn(rng, G)
    x = randn(rng, G * m)
    mu = exp.(0.3 .+ 0.25 .* x .+ x .* (exp.(alpha[1] .+ alpha[2] .* zgroup) .* b)[gidx])
    sz = 1 / 0.42^2
    y = Float64[rand(rng, Distributions.NegativeBinomial(sz, sz / (sz + v))) for v in mu]
    data = (; y, x, z = zgroup[gidx], g = gidx)
    formula = bf(@formula(y ~ x + spatial(0 + x | g)), @formula(sigma ~ 1),
                 @formula(sd(g, spatial) ~ z))
    fit = drm(formula, NegBinomial2(); data, coords, se = false)

    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sigma, :sd]
    @test coef(fit, :sd)[2] > 0
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 1.0
    @test fit.scales[:sd_group] ≈ exp.(hcat(ones(G), zgroup) * coef(fit, :sd))
    @test predict_parameters(fit, data; se = false)[:sd] ≈ fit.scales[:sd_group][gidx]

    Xmu = hcat(ones(length(y)), x)
    theta = vcat(coef(fit, :mu), coef(fit, :sigma), coef(fit, :sd))
    loading = x .* exp.(theta[4] .+ theta[5] .* data.z)
    oracle = _nb2_spatial_slope_dense(theta, y, Xmu, loading, gidx, C)
    @test fit.nll(theta) ≈ oracle atol = 2e-3

    # Kernel alignment matters: coords rows in the wrong order give another kernel,
    # so the objective at the same theta changes.
    perm = [2:G; 1]
    Cmis = _nb2_native_spatial_kernel(coords[perm, :])
    @test abs(_nb2_spatial_slope_dense(theta, y, Xmu, loading, gidx, Cmis) - oracle) > 0.05
    fit_mis = drm(formula, NegBinomial2(); data, coords = coords[perm, :], se = false)
    @test abs(fit_mis.nll(theta) - fit.nll(theta)) > 0.05
    # A correlation-rescaled kernel is nearly identical here (diag = 1 + 1e-6), so
    # it is reported only as a sanity check, not as evidence of RAW scaling.
    Rn = C ./ (sqrt.(diag(C)) * sqrt.(diag(C))')
    @test abs(_nb2_spatial_slope_dense(theta, y, Xmu, loading, gidx, Rn) - oracle) < 0.05

    # Refusals: precomputed K, missing coords, missing slope column, A, intercept form.
    @test_throws ArgumentError drm(formula, NegBinomial2(); data, K = C, se = false)
    @test_throws ArgumentError drm(formula, NegBinomial2(); data, se = false)
    @test_throws ArgumentError drm(formula, NegBinomial2(); data, coords, A = C, se = false)
    @test_throws ArgumentError drm(formula, NegBinomial2();
        data = (; y, z = data.z, g = gidx), coords, se = false)
    @test_throws ArgumentError drm(
        bf(@formula(y ~ x + spatial(1 | g)), @formula(sigma ~ 1),
           @formula(sd(g, spatial) ~ z)), NegBinomial2(); data, coords, se = false)
end
