# NB2 relmat intercept SD regression: the dispersion must not stall on the
# clamped plateau. `_nb2_laplace_setup` clamps the NB2 size at exp(8), so the
# likelihood is exactly flat in log sigma below -4; before the restart in
# `_fit_nb2_animal_sd_laplace`, a first line-search step could land there and
# LBFGS reported convergence at log sigma near -10, tens of log-likelihood units
# below the interior optimum (seeds 2 and 6 of this fixture, the two tested
# below; the base source fails both).
# The oracle is an independent dense Laplace marginal built here: the NB2 pmf is
# written out by hand (no DRModels likelihood code), the inner mode is found by
# a separate Optim run and the Hessian by ForwardDiff.
# Candidate-level evidence: a few fixtures, no calibration claim.
using DRModels, Test, Random, LinearAlgebra
import Distributions, ForwardDiff, Optim
using SpecialFunctions: loggamma

function _nb2p_fixture(seed; G = 40, m = 16)
    rng = MersenneTwister(seed)
    R = 0.4 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    s = collect(range(0.7, 1.6; length = G))
    K = Matrix(Diagonal(s) * R * Diagonal(s))
    zg = repeat([-1.0, 1.0], G ÷ 2)
    u = exp.(-0.6 .+ 0.5 .* zg) .* (cholesky(Symmetric(K)).L * randn(rng, G))
    g = repeat(1:G, inner = m)
    x = randn(rng, G * m)
    sz = 1 / 0.42^2
    y = Float64[rand(rng, Distributions.NegativeBinomial(sz,
                  sz / (sz + exp(0.3 + 0.2 * x[i] + u[g[i]])))) for i in eachindex(g)]
    return (; data = (; y, x, z = zg[g], g = ["g$(i)" for i in g]), K, zg, gidx = g)
end

# Dense Laplace marginal NLL: joint(b) = b'K^{-1}b / 2 - sum log NB2(y | eta),
# eta_i = X_i beta + exp(Z alpha)_g b_g, size r = exp(-2 log sigma).
function _nb2p_oracle(theta, fx)
    y, x = fx.data.y, fx.data.x
    X = hcat(ones(length(y)), x)
    beta, ls, alpha = theta[1:2], theta[3], theta[4:5]
    r = exp(-2 * ls)
    load = exp.(hcat(ones(length(fx.zg)), fx.zg) * alpha)
    Q = inv(Symmetric(fx.K))
    G = length(load)
    function joint(b)
        v = 0.5 * dot(b, Q * b)
        for i in eachindex(y)
            eta = (X[i, :]' * beta) + load[fx.gidx[i]] * b[fx.gidx[i]]
            mu = exp(eta)
            v -= loggamma(y[i] + r) - loggamma(r) - loggamma(y[i] + 1) +
                 r * log(r / (r + mu)) + y[i] * log(mu / (r + mu))
        end
        return v
    end
    res = Optim.optimize(joint, zeros(G), Optim.LBFGS(),
                         Optim.Options(g_tol = 1e-9, iterations = 500);
                         autodiff = :forward)
    bhat = Optim.minimizer(res)
    H = ForwardDiff.hessian(joint, bhat)
    return joint(bhat) + 0.5 * logdet(Symmetric(H)) - 0.5 * logdet(Symmetric(Q))
end

@testset "NB2 relmat intercept SD: no stall on the clamped dispersion plateau" begin
    for seed in (2, 6)
        @testset "seed $seed" begin
            fx = _nb2p_fixture(seed)
            f = bf(@formula(y ~ x + relmat(1 | g)), @formula(sd(g, relmat) ~ z))
            fit = drm(f, NegBinomial2(); data = fx.data, K = fx.K, se = false)
            @test fit.converged
            theta = vcat(coef(fit, :mu), coef(fit, :sigma), coef(fit, :sd))
            # Interior dispersion (the stalled answers were -4.8 and -10.2).
            @test theta[3] > -3.0

            # The fitted objective is the independent dense Laplace NLL.
            base = _nb2p_oracle(theta, fx)
            @test -fit.loglik ≈ base rtol = 1e-5

            # The fit sits at an optimum of the oracle surface.
            for j in eachindex(theta), h in (-0.05, 0.05)
                th = copy(theta); th[j] += h
                @test _nb2p_oracle(th, fx) > base - 1e-3
            end

            # Emulated bug: the pre-fix plateau answer (log sigma near -8 with the
            # other coefficients unchanged) is far from optimal on the oracle surface.
            plateau = copy(theta); plateau[3] = -8.0
            @test _nb2p_oracle(plateau, fx) - base > 5.0
            # A constant SD formula nests inside the z model.
            cfit = drm(bf(@formula(y ~ x + relmat(1 | g)), @formula(sd(g, relmat) ~ 1)),
                       NegBinomial2(); data = fx.data, K = fx.K, se = false)
            @test cfit.converged && coef(cfit, :sigma)[1] > -3.0
            @test fit.loglik >= cfit.loglik - 1e-4
        end
    end
end
