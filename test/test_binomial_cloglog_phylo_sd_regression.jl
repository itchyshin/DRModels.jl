using DRModels, Test, Random, LinearAlgebra
import Distributions

function _binomial_cloglog_phylo_test_newick(depth)
    function node(prefix, d)
        d == 0 && return "$(prefix):$(1 / depth)"
        left = node(prefix * "a", d - 1)
        right = node(prefix * "b", d - 1)
        return d == depth ? "($left,$right);" : "($left,$right):$(1 / depth)"
    end
    return node("tip", depth)
end

function _binomial_cloglog_dense_laplace(theta, successes, failures, X, ztip,
                                          tip_index, K)
    pμ = size(X, 2)
    beta, alpha = theta[1:pμ], theta[(pμ + 1):end]
    tau = exp.(hcat(ones(length(ztip)), ztip) * alpha)
    covariance = Diagonal(tau) * K * Diagonal(tau)
    precision = Matrix(cholesky(Symmetric(covariance)) \ I)
    eta0 = X * beta

    function joint(b)
        eta = eta0 + b[tip_index]
        value = 0.5 * dot(b, precision * b)
        @inbounds for i in eachindex(successes)
            x = exp(eta[i])
            successes[i] == 1 && (value -= log(-expm1(-x)))
            failures[i] == 1 && (value += x)
        end
        return value
    end

    # Independent dense finite-difference mode and Hessian. The production fit
    # uses sparse tree precision and analytic cloglog observation derivatives.
    b = zeros(length(ztip))
    for _ in 1:60
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
        ch = cholesky(Symmetric(H); check = false)
        issuccess(ch) || error("dense cloglog Laplace Hessian is not positive definite")
        step = ch \ grad
        if norm(step) < 2e-7
            return joint(b) + 0.5 * logdet(ch) -
                0.5 * logdet(Symmetric(precision))
        end
        scale = 1.0
        while scale > 1e-5 && joint(b - scale * step) > f0
            scale /= 2
        end
        b .-= scale .* step
    end
    error("dense cloglog Laplace oracle did not converge")
end

function _binomial_mixed_dense_laplace(theta, y, X, ztip, tip_index, zgroup,
                                       group_index, K, link)
    Gtip, Ggroup = length(ztip), length(zgroup)
    pμ = size(X, 2)
    beta = theta[1:pμ]
    alpha_group = theta[(pμ + 1):(pμ + 2)]
    alpha_phy = theta[(pμ + 3):(pμ + 4)]
    tau_group = exp.(hcat(ones(Ggroup), zgroup) * alpha_group)
    tau_phy = exp.(hcat(ones(Gtip), ztip) * alpha_phy)
    Q = zeros(Gtip + Ggroup, Gtip + Ggroup)
    Q[1:Gtip, 1:Gtip] .= Matrix(cholesky(Symmetric(K)) \ I)
    Q[(Gtip + 1):end, (Gtip + 1):end] .= Matrix{Float64}(I, Ggroup, Ggroup)
    logdetQ = logdet(Symmetric(Q))
    eta0 = X * beta

    function joint(b)
        eta = eta0 .+ tau_phy[tip_index] .* b[tip_index] .+
              tau_group[group_index] .* b[Gtip .+ group_index]
        value = 0.5 * dot(b, Q * b)
        @inbounds for i in eachindex(y)
            if link === :probit
                normal = Distributions.Normal()
                value -= y[i] == 1 ? Distributions.logcdf(normal, eta[i]) :
                                     Distributions.logccdf(normal, eta[i])
            else
                x = exp(eta[i])
                value += y[i] == 1 ? -log(-expm1(-x)) : x
            end
        end
        return value
    end

    b = zeros(Gtip + Ggroup)
    for _ in 1:50
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
        ch = cholesky(Symmetric(H); check = false)
        issuccess(ch) || error("dense mixed Binomial Laplace Hessian is not positive definite")
        step = ch \ grad
        if norm(step) < 2e-7
            return joint(b) + 0.5 * logdet(ch) - 0.5 * logdetQ
        end
        scale = 1.0
        while scale > 1e-5 && joint(b - scale * step) > f0
            scale /= 2
        end
        b .-= scale .* step
    end
    error("dense mixed Binomial Laplace oracle did not converge")
end

@testset "Binomial cloglog phylogenetic SD regression" begin
    rng = MersenneTwister(20261026)
    tree = _binomial_cloglog_phylo_test_newick(4)
    phy = DRModels.augmented_phy(tree)
    G = phy.n_leaves
    n_each = 30
    tip_index = repeat(1:G; inner = n_each)
    species = repeat(phy.leaf_names; inner = n_each)
    ztip = repeat([-1.0, 1.0], G ÷ 2)
    z = ztip[tip_index]
    x = repeat(collect(range(-1.0, 1.0; length = n_each)), outer = G)
    K = DRModels.sigma_phy_dense(phy; σ²_phy = 1.0)
    latent = cholesky(Symmetric(K)).L * randn(rng, G)
    eta = 0.15 .+ 0.30 .* x .+
          exp.(-0.85 .+ 0.55 .* ztip)[tip_index] .* latent[tip_index]
    probability = -expm1.(-exp.(eta))
    successes = rand.(Ref(rng), Distributions.Bernoulli.(probability))
    failures = 1 .- successes
    data = (; successes, failures, x, z, species)
    formula = bf(
        @formula(cbind(successes, failures) ~ x + phylo(1 | species)),
        @formula(sd(species, phylogenetic) ~ z),
    )

    fit = drm(formula, DRModels.Binomial(link = :cloglog);
              data, tree, se = false, g_tol = 1e-5)

    @test fit.converged
    @test isfinite(fit.loglik)
    @test sign(coef(fit, :sd_phylo)[2]) == 1
    @test abs(coef(fit, :sd_phylo)[2] - 0.55) < 0.45
    sd_design = hcat(ones(G), ztip)[tip_index, :]
    expected_sd = DRModels._binomial_sd_scale(sd_design * fit.theta[3:4])
    @test fit.scales[:sd_phylo] ≈ expected_sd
    @test predict_parameters(fit, data)[:sd_phylo] ≈ expected_sd
    @test fit.nll(fit.theta) ≈ _binomial_cloglog_dense_laplace(
        fit.theta, successes, failures, hcat(ones(length(x)), x), ztip,
        tip_index, K,
    ) atol = 3e-4
end

@testset "Binomial mixed phylogenetic and IID SD Laplace oracle" begin
    tree = _binomial_cloglog_phylo_test_newick(3)
    phy = DRModels.augmented_phy(tree)
    G = phy.n_leaves
    sites_per_tip, n_each = 2, 8
    n_site = G * sites_per_tip
    site_index = repeat(1:n_site; inner = n_each)
    tip_index = repeat(repeat(1:G; inner = sites_per_tip); inner = n_each)
    species = repeat(phy.leaf_names; inner = sites_per_tip * n_each)
    site = repeat(["site$(i)" for i in 1:n_site]; inner = n_each)
    ztip = repeat([-1.0, 1.0], G ÷ 2)
    zgroup = repeat([-1.0, 1.0], n_site ÷ 2)
    K = DRModels.sigma_phy_dense(phy; σ²_phy = 1.0)
    x = repeat(collect(range(-1.0, 1.0; length = n_each)), outer = n_site)
    X = hcat(ones(length(x)), x)
    rng = MersenneTwister(20261030)
    u_phy = cholesky(Symmetric(K)).L * randn(rng, G)
    u_group = randn(rng, n_site)
    eta = -0.1 .+ 0.25 .* x .+
          exp.(-0.8 .+ 0.5 .* ztip)[tip_index] .* u_phy[tip_index] .+
          exp.(-0.65 .+ 0.4 .* zgroup)[site_index] .* u_group[site_index]
    data_template = (; x, species, site,
                     zgroup = zgroup[site_index], ztip = ztip[tip_index])
    formula = bf(
        @formula(cbind(successes, failures) ~ x + phylo(1 | species) + (1 | site)),
        @formula(sd(site) ~ zgroup),
        @formula(sd(species, phylogenetic) ~ ztip),
    )

    for (link, offset) in ((:probit, 0), (:cloglog, 1))
        link_rng = MersenneTwister(20261031 + offset)
        probability = link === :probit ? Distributions.cdf.(
            Ref(Distributions.Normal()), eta,
        ) : -expm1.(-exp.(eta))
        successes = rand.(Ref(link_rng), Distributions.Bernoulli.(probability))
        failures = 1 .- successes
        data = merge(data_template, (; successes, failures))
        fit = drm(formula, DRModels.Binomial(link = link);
                  data, tree, se = false, g_tol = 1e-5)

        @test fit.converged
        @test fit.nll(fit.theta) ≈ _binomial_mixed_dense_laplace(
            fit.theta, successes, X, ztip, tip_index, zgroup, site_index, K, link,
        ) atol = 2e-3
    end
end

@testset "Binomial probit and cloglog mixed phylogenetic plus IID SD regression" begin
    tree = _binomial_cloglog_phylo_test_newick(5)
    phy = DRModels.augmented_phy(tree)
    G = phy.n_leaves
    sites_per_tip, n_each = 4, 10
    n_site = G * sites_per_tip
    site_index = repeat(1:n_site; inner = n_each)
    tip_index = repeat(repeat(1:G; inner = sites_per_tip); inner = n_each)
    species = repeat(phy.leaf_names; inner = sites_per_tip * n_each)
    site = repeat(["site$(i)" for i in 1:n_site]; inner = n_each)
    z_phy_tip = collect(range(-1.0, 1.0; length = G))
    z_iid_site = repeat([-1.0, 1.0], G * sites_per_tip ÷ 2)
    z_phy = z_phy_tip[tip_index]
    z_iid = z_iid_site[site_index]
    x = repeat(collect(range(-1.0, 1.0; length = n_each)), outer = n_site)
    K = DRModels.sigma_phy_dense(phy; σ²_phy = 1.0)
    rng = MersenneTwister(20261027)
    u_phy = cholesky(Symmetric(K)).L * randn(rng, G)
    u_iid = randn(rng, n_site)
    eta = -0.1 .+ 0.25 .* x .+
          exp.(-0.8 .+ 0.5 .* z_phy_tip)[tip_index] .* u_phy[tip_index] .+
          exp.(-0.65 .+ 0.4 .* z_iid_site)[site_index] .* u_iid[site_index]
    data_template = (; x, z_phy, z_iid, species, site)
    formula = bf(
        @formula(cbind(successes, failures) ~ x + phylo(1 | species) + (1 | site)),
        @formula(sd(site) ~ z_iid),
        @formula(sd(species, phylogenetic) ~ z_phy),
    )

    for (link, offset) in ((:probit, 0), (:cloglog, 1))
        link_rng = MersenneTwister(20261028 + offset)
        probability = link === :probit ? Distributions.cdf.(
            Ref(Distributions.Normal()), eta,
        ) : -expm1.(-exp.(eta))
        successes = rand.(Ref(link_rng), Distributions.Bernoulli.(probability))
        failures = 1 .- successes
        data = merge(data_template, (; successes, failures))
        fam = DRModels.Binomial(link = link)
        fit = drm(formula, fam; data, tree, se = false, g_tol = 1e-5)

        @test fit.converged
        @test isfinite(fit.loglik)
        @test coef(fit, :sd_phylo)[2] > 0
        @test abs(coef(fit, :sd_phylo)[2] - 0.5) < 0.45
        @test coef(fit, :sd)[2] > 0
        @test abs(coef(fit, :sd)[2] - 0.4) < 0.4
        expected_mu = DRModels._binomial_mean.(Ref(fam),
            coef(fit, :mu)[1] .+ coef(fit, :mu)[2] .* x)
        @test fit.means[:mu] ≈ expected_mu
        parameter_predictions = predict_parameters(fit, data)
        @test parameter_predictions[:sd_phylo] ≈ fit.scales[:sd_phylo]
        @test parameter_predictions[:sd] ≈ fit.scales[:sd]
    end
end
