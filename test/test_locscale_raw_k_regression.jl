# Coupled location-scale `(1 | p | relmat(g))` / `(1 | p | animal(g))` fits use the
# supplied K AS SUPPLIED (raw), as native drmTMB does (chol2inv(chol(K)), no
# rescaling): the group-level prior is a ~ N(0, Lambda (x) K). K has a NON-UNIT
# diagonal (0.5 to 2.5) and correlated off-diagonals; rows are shuffled so the
# first-seen group order differs from the label order. The fitted objective is
# compared with an independent dense Laplace oracle coded here from the RAW K
# (dense kron(inv(K), inv(Lambda)) prior, own Newton-free LBFGS inner solve); the
# oracle built from the correlation-rescaled K must differ, so the test fails if K
# is normalised. Precomputed spatial K keeps its correlation form (asserted last).
using DRModels, Test, Random, LinearAlgebra, SparseArrays
import Distributions
using SpecialFunctions: loggamma

function _lsraw_nll(kind::Symbol, y, eta, psi)
    if kind === :gamma            # shape alpha = exp(psi), mean exp(eta)
        a = exp(psi)
        return -(a * log(a) - a * eta - loggamma(a) + (a - 1) * log(y) - a * y * exp(-eta))
    else                          # nb2: size r = exp(-2 psi), mean exp(eta)
        r = exp(-2 * psi); mu = exp(eta)
        return -(loggamma(y + r) - loggamma(r) - loggamma(y + 1) +
                 r * log(r) + y * eta - (y + r) * log(r + mu))
    end
end

# -log Laplace marginal for a (group-major: a[2g-1] mean axis, a[2g] scale axis)
# with prior precision kron(inv(K), inv(Lambda)).
function _lsraw_oracle(kind, y, eta0, psi0, gidx, K, Lambda)
    G = size(K, 1)
    P = kron(inv(Symmetric(K)), inv(Symmetric(Lambda)))
    function joint(a)
        v = 0.5 * dot(a, P * a)
        for i in eachindex(y)
            g = gidx[i]
            v += _lsraw_nll(kind, y[i], eta0[i] + a[2g - 1], psi0[i] + a[2g])
        end
        v
    end
    res = DRModels.Optim.optimize(joint, zeros(2G), DRModels.Optim.LBFGS(),
        DRModels.Optim.Options(g_tol = 1e-10, iterations = 2000); autodiff = :forward)
    mode = DRModels.Optim.minimizer(res)
    H = DRModels.ForwardDiff.hessian(joint, mode)
    return joint(mode) + 0.5 * logdet(Symmetric(H)) - 0.5 * logdet(Symmetric(P))
end

function _lsraw_fixture(seed)
    rng = MersenneTwister(seed)
    G, m = 20, 8
    R = 0.5 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    scale = sqrt.(range(0.5, 2.5; length = G))
    Kbase = Diagonal(scale) * R * Diagonal(scale)
    names = ["g" * lpad(string(i), 2, '0') for i in 1:G]
    rows = shuffle(rng, repeat(1:G, inner = m))
    order = unique(rows)
    K = Matrix(Kbase[order, order])                      # first-seen group order
    gidx = [findfirst(==(r), order) for r in rows]
    Lam = [0.30 0.06; 0.06 0.20]
    Z = cholesky(Symmetric(Kbase)).L * randn(rng, G, 2) * cholesky(Symmetric(Lam)).L'
    x = randn(rng, length(rows))
    return (; rng, labels = names[rows], K, gidx, x, Z, rows)
end

@testset "coupled location-scale relmat/animal K enters raw (dense Laplace oracle)" begin
    for kind in (:gamma, :nb2), provider in (:relmat, :animal)
        fx = _lsraw_fixture(20261102 + 7 * length(String(kind)) + length(String(provider)))
        eta = 0.4 .+ 0.2 .* fx.x .+ fx.Z[fx.rows, 1]
        psi = (kind === :gamma ? 1.2 : -0.5) .+ 0.1 .* fx.x .+ fx.Z[fx.rows, 2]
        mu = exp.(eta)
        y = if kind === :gamma
            Float64[rand(fx.rng, Distributions.Gamma(exp(p), m / exp(p))) for (m, p) in zip(mu, psi)]
        else
            Float64[let r = exp(-2p); rand(fx.rng, Distributions.NegativeBinomial(r, r / (r + m))) end
                    for (m, p) in zip(mu, psi)]
        end
        data = (; y, x = fx.x, g = fx.labels)
        mark = provider === :relmat ? :(relmat(g)) : :(animal(g))
        f = bf(eval(:(@formula(y ~ x + (1 | p | $mark)))), eval(:(@formula(sigma ~ x + (1 | p | $mark)))))
        fam = kind === :gamma ? Gamma() : NegBinomial2()
        fit = provider === :relmat ? drm(f, fam; data, K = fx.K, se = false) :
                                      drm(f, fam; data, A = fx.K, se = false)
        @test fit.converged
        # The engine precision is the inverse of the RAW K (not of its correlation).
        Q = Matrix(fit.nll.Q)
        @test Q ≈ inv(Symmetric(fx.K)) rtol = 1e-10
        bmu, bpsi, rc = coef(fit, :mu), coef(fit, :sigma), coef(fit, :recov)
        # public recov order is [log L11, log L22, L21]; engine packing [log L11, L21, log L22]
        theta = vcat(bmu, bpsi, rc[1], rc[3], rc[2])
        L = [exp(rc[1]) 0.0; rc[3] exp(rc[2])]
        Lambda = L * L'
        Xmu = hcat(ones(length(y)), fx.x)
        oracle = _lsraw_oracle(kind, y, Xmu * bmu, Xmu * bpsi, fx.gidx, fx.K, Lambda)
        @test fit.nll(theta) ≈ oracle atol = 1e-6
        d = sqrt.(diag(fx.K))
        wrong = _lsraw_oracle(kind, y, Xmu * bmu, Xmu * bpsi, fx.gidx, fx.K ./ (d * d'), Lambda)
        @test abs(wrong - oracle) > 0.05
        @test abs(fit.nll(theta) - wrong) > 0.05
    end
end

@testset "locscale setup: relmat raw, spatial and phylo keep the correlation form" begin
    K = [4.0 1.0 0.5; 1.0 2.0 0.3; 0.5 0.3 1.5]
    labels = ["a", "b", "c", "a", "b", "c"]
    Qraw, _, G = DRModels._locscale_relmat_setup(K, labels)
    @test G == 3 && Matrix(Qraw) ≈ inv(K)
    d = sqrt.(diag(K))
    Qsp, _, _ = DRModels._locscale_structured_q(:spatial, :g, labels, nothing, K, nothing)
    @test Matrix(Qsp) ≈ inv(K ./ (d * d'))
    Qrel, _, _ = DRModels._locscale_structured_q(:relmat, :g, labels, nothing, K, nothing)
    @test Matrix(Qrel) ≈ inv(K)
end
