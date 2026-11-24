using DRModels, Test, Random, LinearAlgebra
using StableRNGs
import Distributions

function _binomial_probit_phylo_test_newick(depth)
    function node(prefix, d)
        d == 0 && return "$(prefix):$(1 / depth)"
        left = node(prefix * "a", d - 1)
        right = node(prefix * "b", d - 1)
        return d == depth ? "($left,$right);" : "($left,$right):$(1 / depth)"
    end
    return node("tip", depth)
end

function _binomial_probit_dense_laplace(theta, successes, trials, X, ztip,
                                         tip_index, K)
    pμ = size(X, 2)
    beta, alpha = theta[1:pμ], theta[(pμ + 1):end]
    tau = exp.(hcat(ones(length(ztip)), ztip) * alpha)
    covariance = Diagonal(tau) * K * Diagonal(tau)
    precision = Matrix(cholesky(Symmetric(covariance)) \ I)
    eta0 = X * beta
    normal = Distributions.Normal()

    function joint(b)
        eta = eta0 + b[tip_index]
        value = 0.5 * dot(b, precision * b)
        @inbounds for i in eachindex(successes)
            s, n = successes[i], trials[i]
            value -= DRModels._logfactorial(n) - DRModels._logfactorial(s) -
                     DRModels._logfactorial(n - s)
            value -= s * Distributions.logcdf(normal, eta[i])
            value -= (n - s) * Distributions.logccdf(normal, eta[i])
        end
        return value
    end

    # Independent dense finite-difference mode and Hessian. The production fit
    # uses sparse tree precision and analytic observation derivatives.
    b = zeros(length(ztip))
    for iteration in 1:50
        h = 2e-4
        f0 = joint(b)
        grad = zeros(length(b))
        H = zeros(length(b), length(b))
        for j in eachindex(b)
            ej = zeros(length(b)); ej[j] = h
            grad[j] = (joint(b + ej) - joint(b - ej)) / (2h)
            H[j, j] = (joint(b + ej) - 2 * f0 + joint(b - ej)) / h^2
            for k in 1:(j - 1)
                ek = zeros(length(b)); ek[k] = h
                H[j, k] = H[k, j] =
                    (joint(b + ej + ek) - joint(b + ej - ek) -
                     joint(b - ej + ek) + joint(b - ej - ek)) / (4h^2)
            end
        end
        step = cholesky(Symmetric(H)) \ grad
        if norm(step) < 2e-7
            return joint(b) + 0.5 * logdet(Symmetric(H)) -
                0.5 * logdet(Symmetric(precision))
        end
        scale = 1.0
        while scale > 1e-5 && joint(b - scale * step) > f0
            scale /= 2
        end
        b .-= scale .* step
    end
    error("dense probit Laplace oracle did not converge")
end

@testset "Binomial probit phylogenetic SD regression" begin
    rng = StableRNG(20261025)
    tree = _binomial_probit_phylo_test_newick(4)
    phy = DRModels.augmented_phy(tree)
    G = phy.n_leaves
    n_each = 30
    tip_index = repeat(1:G; inner = n_each)
    species = repeat(phy.leaf_names; inner = n_each)
    ztip = repeat([-1.0, 1.0], G ÷ 2)
    z = ztip[tip_index]
    x = repeat(collect(range(-1.0, 1.0; length = n_each)), outer = G)
    K = DRModels.sigma_phy_dense(phy; σ²_phy = 1.0)
    Q, tip_nodes, _ = DRModels._poisson_phylo_setup(tree, phy.leaf_names)
    K_from_precision = inv(Matrix(Q))[tip_nodes, tip_nodes]
    @test K_from_precision ≈ K atol = 1e-10
    latent = cholesky(Symmetric(K)).L * randn(rng, G)
    eta = 0.15 .+ 0.30 .* x .+ exp.(-0.85 .+ 0.55 .* ztip)[tip_index] .* latent[tip_index]
    probability = Distributions.cdf.(Ref(Distributions.Normal()), eta)
    successes = rand.(Ref(rng), Distributions.Bernoulli.(probability))
    failures = 1 .- successes
    data = (; successes, failures, x, z, species)
    formula = bf(
        @formula(cbind(successes, failures) ~ x + phylo(1 | species)),
        @formula(sd(species, phylogenetic) ~ z),
    )

    fit = drm(formula, DRModels.Binomial(link = :probit);
              data, tree, se = true, g_tol = 1e-5)

    @test fit.converged
    @test isfinite(fit.loglik)
    @test all(isfinite, vcov(fit))
    @test abs(coef(fit, :sd_phylo)[2] - 0.55) < 0.35
    sd_design = hcat(ones(G), ztip)[tip_index, :]
    @test fit.scales[:sd_phylo] ≈
        DRModels._binomial_sd_scale(sd_design * fit.theta[3:4])
    sd_predictions = predict_parameters(fit, data)[:sd_phylo]
    @test sd_predictions ≈ DRModels._binomial_sd_scale(sd_design * fit.theta[3:4])
    row_order = reverse(eachindex(successes))
    permuted_data = (; successes = successes[row_order], failures = failures[row_order],
                     x = x[row_order], z = z[row_order], species = species[row_order])
    @test predict_parameters(fit, permuted_data)[:sd_phylo] ≈ sd_predictions[row_order]
    @test fitted(fit; re = :zero) ≈ Distributions.cdf.(
        Ref(Distributions.Normal()), hcat(ones(length(x)), x) * fit.theta[1:2],
    )
    @test length(predict(fit, data)) == length(successes)
    @test predict(fit, data) ≈
        Distributions.cdf.(Ref(Distributions.Normal()), predict(fit, data; type = :link))
    @test fit.nll(fit.theta) ≈ _binomial_probit_dense_laplace(
        fit.theta, successes, ones(Int, length(successes)),
        hcat(ones(length(x)), x), ztip, tip_index, K,
    ) atol = 2e-4
end

@testset "Binomial probit latent derivatives" begin
    successes = [0, 1]
    aux = (s = successes, ntr = ones(Int, 2), logchoose = zeros(2))
    h = 2e-3
    for (i, eta) in Iterators.product(eachindex(successes), (-5.0, -1.5, 0.0, 2.0, 5.0))
        value(x) = -successes[i] * Distributions.logcdf(Distributions.Normal(), x) -
                   (1 - successes[i]) * Distributions.logccdf(Distributions.Normal(), x)
        _, r, w, t = DRModels._binomial_probit_v123(aux, i, eta)
        fd1 = (value(eta + h) - value(eta - h)) / (2h)
        fd2 = (value(eta + h) - 2value(eta) + value(eta - h)) / h^2
        fd3 = (value(eta + 2h) - 2value(eta + h) + 2value(eta - h) -
               value(eta - 2h)) / (2h^3)
        @test r ≈ fd1 atol = 2e-5
        @test w ≈ fd2 atol = 2e-4
        @test t ≈ fd3 atol = 2e-3
    end
end
