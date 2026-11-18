using DRModels, Test, Random, LinearAlgebra
import Distributions

function _binomial_phylo_sd_test_newick(depth; height = 1.0)
    function node(prefix, d)
        d == 0 && return "$(prefix):$(height / depth)"
        left = node(prefix * "a", d - 1)
        right = node(prefix * "b", d - 1)
        return d == depth ? "($left,$right);" : "($left,$right):$(height / depth)"
    end
    return node("tip", depth)
end

function _binomial_phylo_ordinary_sd_dense_laplace(theta, y, X, ztip,
                                                    tip_index, K, wgroup,
                                                    group_index; pord = 2,
                                                    pphy = 2)
    pμ = size(X, 2)
    G, J = length(ztip), length(wgroup)
    alpha_ord = theta[(pμ + 1):(pμ + pord)]
    alpha_phy = theta[(pμ + pord + 1):(pμ + pord + pphy)]
    Xord = pord == 1 ? ones(J, 1) : hcat(ones(J), wgroup)
    Xphy = pphy == 1 ? ones(G, 1) : hcat(ones(G), ztip)
    tau_ord = exp.(Xord * alpha_ord)
    tau_phy = exp.(Xphy * alpha_phy)
    covariance = zeros(G + J, G + J)
    covariance[1:G, 1:G] = Diagonal(tau_phy) * K * Diagonal(tau_phy)
    covariance[(G + 1):end, (G + 1):end] = Diagonal(tau_ord.^2)
    precision = Matrix(cholesky(Symmetric(covariance)) \ I)
    A = zeros(length(y), G + J)
    for i in eachindex(y)
        A[i, tip_index[i]] = 1.0
        A[i, G + group_index[i]] = 1.0
    end
    eta0 = X * theta[1:pμ]
    b = zeros(G + J)
    function joint(bb)
        eta = eta0 + A * bb
        value = 0.5 * dot(bb, precision * bb)
        for i in eachindex(y)
            value += log1p(exp(-abs(eta[i]))) + max(eta[i], 0) - y[i] * eta[i]
        end
        value
    end
    for _ in 1:40
        eta = eta0 + A * b
        prob = 1 ./ (1 .+ exp.(-eta))
        grad = precision * b + A' * (prob .- y)
        H = precision + A' * Diagonal(prob .* (1 .- prob)) * A
        ch = cholesky(Symmetric(H))
        step = ch \ grad
        norm(step) < 1e-11 && return joint(b) + 0.5 * logdet(ch) -
            0.5 * logdet(Symmetric(precision))
        b .-= step
    end
    error("dense joint Bernoulli oracle did not converge")
end

@testset "Binomial ordinary and phylogenetic SD regressions jointly" begin
    rng = MersenneTwister(20261020)
    tree = _binomial_phylo_sd_test_newick(5)
    phy = DRModels.augmented_phy(tree)
    G = phy.n_leaves
    n_group = 2
    n_each = 12
    species = repeat(phy.leaf_names; inner = n_group * n_each)
    group_index = repeat(repeat(1:n_group, inner = n_each), outer = G)
    tip_row = repeat(1:G; inner = n_group * n_each)
    crossed_group = group_index .+ 2 .* mod.(tip_row .- 1, 16)
    group = ["g$(crossed_group[i])" for i in eachindex(species)]
    ztip = repeat([-1.0, 1.0], G ÷ 2)
    z = repeat(ztip; inner = n_group * n_each)
    w = ifelse.(isodd.(crossed_group), -1.0, 1.0)
    x = repeat(collect(range(-1.0, 1.0; length = n_each)), outer = G * n_group)

    K = DRModels.sigma_phy_dense(phy; σ²_phy = 1.0)
    u = cholesky(Symmetric(K)).L * randn(rng, G)
    b_phy = exp.(-0.65 .+ 0.50 .* ztip) .* u
    w_group = repeat([-1.0, 1.0], 16)
    sd_group = exp.(-1.0 .+ 0.50 .* w_group)
    b_group = sd_group .* randn(rng, length(w_group))
    eta = -0.15 .+ 0.25 .* x .+ b_phy[repeat(1:G; inner = n_group * n_each)] .+
          b_group[crossed_group]
    p = 1 ./ (1 .+ exp.(-eta))
    successes = Float64.(rand.(Ref(rng), Distributions.Bernoulli.(p)))
    failures = 1 .- successes
    data = (; successes, failures, x, z, w, species, group)

    formula = bf(
        @formula(cbind(successes, failures) ~ x + (1 | group) + phylo(1 | species)),
        @formula(sd(group) ~ w),
        @formula(sd(species, phylogenetic) ~ z),
    )
    fit = drm(formula, DRModels.Binomial(); data, tree, se = true, g_tol = 1e-5)
    @test fit.converged
    @test isfinite(fit.loglik)
    @test all(isfinite, vcov(fit))
    # Reconstruct one marginal response draw from independent ordinary and
    # phylogenetic group effects. This catches a simulator that silently drops
    # either field while still returning a successful draw.
    draw_seed = 202610211
    simulator = DRModels._marginal_simulator_build(fit, data; tree)
    actual_draw = simulator(MersenneTwister(draw_seed))
    oracle_rng = MersenneTwister(draw_seed)
    ordinary_group_order = unique(crossed_group)
    @test ordinary_group_order == collect(1:32)
    group_id = crossed_group
    w_group = repeat([-1.0, 1.0], 16)
    group_coef = coef(fit, :sd)
    group_sd = exp.(group_coef[1] .+ group_coef[2] .* w_group)
    group_effect = group_sd .* randn(oracle_rng, length(w_group))
    tip_index = repeat(1:G; inner = n_group * n_each)
    phylo_coef = coef(fit, :sd_phylo)
    phylo_sd = exp.(phylo_coef[1] .+ phylo_coef[2] .* ztip)
    phylo_effect = phylo_sd .* (cholesky(Symmetric(K)).L * randn(oracle_rng, G))
    eta_fixed = predict(fit, data; type = :link)
    expected_mu = DRModels._mean_response(
        fit.family, eta_fixed .+ group_effect[group_id] .+ phylo_effect[tip_index],
    )
    trials = round.(Int, DRModels._scale_vector(fit, :trials))
    expected_draw = Float64[
        rand(oracle_rng, Distributions.Binomial(
            trials[i], clamp(expected_mu[i], eps(), 1 - eps()),
        )) for i in eachindex(successes)
    ]
    @test length(ordinary_group_order) == length(w_group)
    @test actual_draw == expected_draw
    boot = bootstrap_result(fit; data, B = 1, tree,
        rng = MersenneTwister(202610212), failures = :error,
        check_converged = true)
    @test boot.attempted == 1
    @test boot.used == 1
    @test boot.failed == 0
    sd_blocks = filter(p -> startswith(String(first(p)), "sd"), fit.blocks)
    @test length(sd_blocks) == 2
    @test all(isfinite, coef(fit, first(first.(sd_blocks))))
    @test coef(fit, :sd)[2] > 0
    @test coef(fit, :sd_phylo)[2] > 0
    @test length(predict(fit, data)) == length(successes)
    parameter_predictions = predict_parameters(fit, data; se = false)
    @test parameter_predictions[:sd] ≈ fit.scales[:sd]
    @test parameter_predictions[:sd_phylo] ≈ fit.scales[:sd_phylo]
    parameter_predictions_se = predict_parameters(fit, data; se = true)
    @test all(isfinite, parameter_predictions_se[:sd].value)
    @test all(isfinite, parameter_predictions_se[:sd].se)
    @test all(isfinite, parameter_predictions_se[:sd_phylo].value)
    @test all(isfinite, parameter_predictions_se[:sd_phylo].se)
    marginal = marginal_parameters(fit)
    @test marginal[:sd] ≈ parameter_predictions[:sd]
    @test marginal[:sd_phylo] ≈ parameter_predictions[:sd_phylo]
    group_levels = unique(group)
    group_map = Dict(label => i for (i, label) in enumerate(group_levels))
    group_id = [group_map[label] for label in group]
    w_by_group = [w[findfirst(==(gi), group_id)] for gi in eachindex(group_levels)]
    @test fit.nll(fit.theta) ≈ _binomial_phylo_ordinary_sd_dense_laplace(
        fit.theta, successes, hcat(ones(length(x)), x), ztip,
        repeat(1:G; inner = n_group * n_each), K, w_by_group, group_id;
    ) atol = 3e-6
    constant_formula = bf(
        @formula(cbind(successes, failures) ~ x + (1 | group) + phylo(1 | species)),
        @formula(sd(group) ~ 1),
        @formula(sd(species, phylogenetic) ~ 1),
    )
    fit_constant = drm(constant_formula, DRModels.Binomial(); data, tree,
                       se = false, g_tol = 1e-5)
    @test fit_constant.converged
    @test length(coef(fit_constant, :sd)) == 1
    @test length(coef(fit_constant, :sd_phylo)) == 1
    @test fit_constant.nll(fit_constant.theta) ≈ _binomial_phylo_ordinary_sd_dense_laplace(
        fit_constant.theta, successes, hcat(ones(length(x)), x), ztip,
        repeat(1:G; inner = n_group * n_each), K, w_by_group, group_id;
        pord = 1, pphy = 1,
    ) atol = 3e-6
end

function _binomial_phylo_sd_dense_laplace(theta, y, X, ztip, tip_index, K)
    pμ = size(X, 2)
    beta, alpha = theta[1:pμ], theta[pμ+1:end]
    tau = exp.(hcat(ones(length(ztip)), ztip) * alpha)
    covariance = Diagonal(tau) * K * Diagonal(tau)
    precision = Matrix(cholesky(Symmetric(covariance)) \ I)
    eta0 = X * beta
    b = zeros(length(tau))
    function joint(bb)
        v = 0.5 * dot(bb, precision * bb)
        for i in eachindex(y)
            eta = eta0[i] + bb[tip_index[i]]
            v += log1p(exp(-abs(eta))) + max(eta, 0) - y[i] * eta
        end
        v
    end
    for _ in 1:40
        grad = precision * b
        hdiag = zeros(length(b))
        for i in eachindex(y)
            eta = eta0[i] + b[tip_index[i]]
            prob = inv(1 + exp(-eta))
            grad[tip_index[i]] += prob - y[i]
            hdiag[tip_index[i]] += prob * (1 - prob)
        end
        H = precision + Diagonal(hdiag)
        ch = cholesky(Symmetric(H))
        step = ch \ grad
        norm(step) < 1e-11 && return joint(b) + 0.5 * logdet(ch) -
            0.5 * logdet(Symmetric(precision))
        b .-= step
    end
    error("dense oracle did not converge")
end

@testset "Binomial phylogenetic intercept SD regression" begin
    rng = MersenneTwister(20261019)
    depth = 5
    tree_height = 2.5
    tree = _binomial_phylo_sd_test_newick(depth; height = tree_height)
    phy = DRModels.augmented_phy(tree)
    G, m = phy.n_leaves, 20
    @test DRModels.phylo_tree_height(phy) ≈ tree_height
    tip_order = randperm(rng, G)
    species = repeat(phy.leaf_names[tip_order], inner = m)
    tip_index = repeat(tip_order, inner = m)
    ztip = repeat([-1.0, 1.0], G ÷ 2)
    z = ztip[tip_index]
    x = repeat(collect(range(-1.0, 1.0; length = m)), outer = G)
    alpha = [-0.6, 0.35]
    # Julia's phylogenetic SD is a multiplier of the raw branch-length K:
    # Cov(b) = D K D. Keep this tree deliberately non-unit-height so this test
    # catches accidental replacement of K by its unit-diagonal correlation.
    K = DRModels.sigma_phy_dense(phy; σ²_phy = 1.0)
    Kcor = DRModels._phylo_correlation(phy)
    @test diag(K) ≈ fill(tree_height, G) atol = 1e-12
    @test K ≈ tree_height .* Kcor atol = 1e-12
    u = cholesky(Symmetric(K)).L * randn(rng, G)
    b = exp.(alpha[1] .+ alpha[2] .* ztip) .* u
    eta = -0.2 .+ 0.3 .* x .+ b[tip_index]
    p = 1 ./ (1 .+ exp.(-eta))
    successes = Float64.(rand.(Ref(rng), Distributions.Bernoulli.(p)))
    failures = 1 .- successes
    data = (; successes, failures, x, z, species)

    formula = bf(
        @formula(cbind(successes, failures) ~ x + phylo(1 | species)),
        @formula(sd(species, phylogenetic) ~ z),
    )
    fit = drm(formula, DRModels.Binomial(); data, tree, se = true, g_tol = 1e-6)
    @test fit.converged
    @test all(isfinite, vcov(fit))
    @test :sd_phylo in first.(fit.blocks)
    @test DRModels._binomial_sd_threaded_profile_unsupported(fit)
    @test_throws ArgumentError profile_result(fit; threads = true, parm = :sd_phylo)
    alpha_hat = coef(fit, :sd_phylo)
    @test length(alpha_hat) == 2
    @test all(isfinite, alpha_hat)
    tau_hat = exp.(hcat(ones(G), ztip) * alpha_hat)
    Vraw = Diagonal(tau_hat) * K * Diagonal(tau_hat)
    Vcor = Diagonal(tau_hat) * Kcor * Diagonal(tau_hat)
    @test diag(Vraw) ≈ tree_height .* tau_hat.^2 atol = 1e-10
    @test Vraw ≈ tree_height .* Vcor atol = 1e-10
    @test !isapprox(Vraw, Vcor; atol = 1e-8)
    @test fit.nll(fit.theta) ≈ _binomial_phylo_sd_dense_laplace(
        fit.theta, successes, hcat(ones(length(x)), x), ztip, tip_index, K;
    ) atol = 2e-6
    pred = predict(fit, data)
    @test length(pred) == length(successes)
    @test all(v -> 0 < v < 1, pred)

    z_inconsistent = copy(z)
    z_inconsistent[1] += 0.1
    bad_data = merge(data, (; z = z_inconsistent))
    bad_formula = bf(
        @formula(cbind(successes, failures) ~ x + phylo(1 | species)),
        @formula(sd(species, phylogenetic) ~ z),
    )
    @test_throws ArgumentError drm(bad_formula, DRModels.Binomial();
        data = bad_data, tree, se = false)

    constant_formula = bf(
        @formula(cbind(successes, failures) ~ x + phylo(1 | species)),
        @formula(sd(species, phylogenetic) ~ 1),
    )
    fit_constant = drm(constant_formula, DRModels.Binomial(); data, tree, se = false,
                       g_tol = 1e-6)
    fit_scalar = drm(bf(@formula(cbind(successes, failures) ~ x + phylo(1 | species))),
                     DRModels.Binomial(); data, tree, se = false, g_tol = 1e-6)
    @test fit_constant.converged && fit_scalar.converged
    @test fit_constant.loglik ≈ fit_scalar.loglik atol = 2e-6
    @test coef(fit_constant, :sd_phylo)[1] ≈ coef(fit_scalar, :resd)[1] atol = 2e-5

    simulator = DRModels._marginal_simulator_build(fit, data; tree)
    @test simulator !== nothing
    ysim = simulator(MersenneTwister(20261020))
    @test length(ysim) == length(successes)
    @test all(y -> y == 0 || y == 1, ysim)
end

@testset "Binomial probit phylogenetic SD slope has finite profile endpoints" begin
    rng = MersenneTwister(20261026)
    tree_height = 2.5
    tree = _binomial_phylo_sd_test_newick(5; height = tree_height)
    phy = DRModels.augmented_phy(tree)
    G, n_each = phy.n_leaves, 30
    tip_index = repeat(1:G; inner = n_each)
    species = phy.leaf_names[tip_index]
    ztip = repeat([-1.0, 1.0], G ÷ 2)
    z = ztip[tip_index]
    x = repeat(collect(range(-1.0, 1.0; length = n_each)), outer = G)
    alpha = [log(0.4), 0.5]
    K = DRModels.sigma_phy_dense(phy; σ²_phy = 1.0)
    field = cholesky(Symmetric(K)).L * randn(rng, G)
    tau = exp.(alpha[1] .+ alpha[2] .* ztip)
    eta = 0.15 .+ 0.3 .* x .+ tau[tip_index] .* field[tip_index]
    probability = Distributions.cdf.(Ref(Distributions.Normal()), eta)
    successes = Float64.(rand.(Ref(rng), Distributions.Bernoulli.(probability)))
    failures = 1 .- successes
    data = (; successes, failures, x, z, species)
    formula = bf(
        @formula(cbind(successes, failures) ~ x + phylo(1 | species)),
        @formula(sd(species, phylogenetic) ~ z),
    )

    fit = drm(formula, Binomial(link = :probit); data, tree, se = true, g_tol = 1e-6)
    @test fit.converged
    @test coef(fit, :sd_phylo)[2] > 0

    profile = profile_result(fit; level = 0.80, parm = :sd_phylo => "z")
    if profile.failed > 0
        @info "probit phylogenetic SD profile endpoint diagnostics" stats = profile.stats
    end
    @test profile.failed == 0
    row = only(profile.ci)
    @test isfinite(row.lower)
    @test isfinite(row.upper)
end
