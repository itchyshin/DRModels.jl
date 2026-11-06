# Poisson / NB2 / Gamma relmat(1 | g) intercept fits use the supplied K AS SUPPLIED
# (raw), as native drmTMB does. K has a NON-UNIT diagonal (0.5 to 2.5) and correlated
# off-diagonals; the rows are shuffled so the first-seen group order differs from the
# label order. Each fit's objective at its own optimum is compared with an independent
# dense Laplace oracle coded here from the RAW K; the oracle built from the
# correlation-rescaled K must differ, so the test would fail if K were normalised.
using DRModels, Test, Random, LinearAlgebra
import Distributions
using SpecialFunctions: loggamma

# -log marginal (Laplace) for b ~ N(0, sd^2 K), eta = Xmu*beta + b[g].
function _raw_k_dense_oracle(family::Symbol, y, Xmu, beta, aux, sd, gidx, K)
    G = size(K, 1)
    Qtot = inv(Symmetric(K)) ./ sd^2
    eta0 = Xmu * beta
    function nll_i(yi, eta)
        if family === :poisson
            return -(yi * eta - exp(eta) - loggamma(yi + 1))
        elseif family === :nb2
            r = 1 / aux^2
            lse = max(eta, log(r)) + log1p(exp(-abs(eta - log(r))))
            return -(loggamma(yi + r) - loggamma(r) - loggamma(yi + 1) +
                     r * (log(r) - lse) + yi * (eta - lse))
        else  # :gamma, shape a = 1 / aux^2, mean exp(eta)
            a = 1 / aux^2
            return -(a * (log(a) - eta) - loggamma(a) + (a - 1) * log(yi) -
                     a * yi * exp(-eta))
        end
    end
    function joint(b)
        value = 0.5 * dot(b, Qtot * b)
        for i in eachindex(y)
            value += nll_i(y[i], eta0[i] + b[gidx[i]])
        end
        value
    end
    res = DRModels.Optim.optimize(joint, zeros(G), DRModels.Optim.LBFGS(),
        DRModels.Optim.Options(g_tol = 1e-10, iterations = 1000); autodiff = :forward)
    mode = DRModels.Optim.minimizer(res)
    H = DRModels.ForwardDiff.hessian(joint, mode)
    return joint(mode) + 0.5 * logdet(Symmetric(H)) - 0.5 * logdet(Qtot)
end

function _raw_k_fixture(seed)
    rng = MersenneTwister(seed)
    G, m = 24, 6
    R = 0.5 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    scale = sqrt.(range(0.5, 2.5; length = G))          # K diagonal 0.5 ... 2.5
    Kbase = Diagonal(scale) * R * Diagonal(scale)        # in label order 1..G
    names = ["g" * lpad(string(i), 2, '0') for i in 1:G]
    rows = shuffle(rng, repeat(1:G, inner = m))          # shuffled group order
    labels = names[rows]
    order = unique(rows)                                 # first-seen order of labels
    K = Matrix(Kbase[order, order])                      # K in first-seen order
    gidx = [findfirst(==(r), order) for r in rows]
    b = 0.6 .* (cholesky(Symmetric(Kbase)).L * randn(rng, G))
    x = randn(rng, length(rows))
    return (; rng, labels, K, gidx, x, eta = 0.3 .+ 0.25 .* x .+ b[rows])
end

@testset "Poisson/NB2/Gamma relmat(1 | g) intercept use raw K (dense Laplace oracle)" begin
    for family in (:poisson, :nb2, :gamma)
        fx = _raw_k_fixture(20261102 + length(String(family)))
        mu = exp.(fx.eta)
        y, fam, aux_true = if family === :poisson
            Float64[rand(fx.rng, Distributions.Poisson(v)) for v in mu], Poisson(), nothing
        elseif family === :nb2
            sz = 1 / 0.45^2
            Float64[rand(fx.rng, Distributions.NegativeBinomial(sz, sz / (sz + v))) for v in mu],
                NegBinomial2(), 0.45
        else
            a = 1 / 0.4^2
            Float64[rand(fx.rng, Distributions.Gamma(a, v / a)) for v in mu], Gamma(), 0.4
        end
        data = (; y, x = fx.x, g = fx.labels)
        formula = family === :poisson ? bf(@formula(y ~ x + relmat(1 | g))) :
                  bf(@formula(y ~ x + relmat(1 | g)), @formula(sigma ~ 1))
        fit = drm(formula, fam; data, K = fx.K, se = false)
        @test fit.converged

        theta = coef(fit)
        p = 2
        beta = theta[1:p]
        aux = family === :poisson ? nothing : exp(theta[p + 1])
        sd = exp(theta[end])
        Xmu = hcat(ones(length(y)), fx.x)
        oracle = _raw_k_dense_oracle(family, y, Xmu, beta, aux, sd, fx.gidx, fx.K)
        @test fit.nll(theta) ≈ oracle atol = 2e-3
        d = sqrt.(diag(fx.K))
        Rn = fx.K ./ (d * d')
        wrong = _raw_k_dense_oracle(family, y, Xmu, beta, aux, sd, fx.gidx, Rn)
        @test abs(wrong - oracle) > 0.05
    end
end
