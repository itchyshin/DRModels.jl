using DRModels, StatsModels, Test, Random, LinearAlgebra
import Distributions

function _binomial_phylo_interaction_slope_tree(prefix; depth = 3, height = 1.0)
    function node(name, depth)
        depth == 0 && return "$(name):$(height / 3)"
        left, right = node(name * "a", depth - 1), node(name * "b", depth - 1)
        return depth == 3 ? "($left,$right)" : "($left,$right):$(height / 3)"
    end
    return node(prefix, depth) * ";"
end

for link in (:logit, :probit, :cloglog)
@testset "Binomial phylogenetic-interaction random-slope SD ($link)" begin
    rng = MersenneTwister(20261028 + findfirst(==(link), (:logit, :probit, :cloglog)))
    tree1 = _binomial_phylo_interaction_slope_tree("host")
    tree2 = _binomial_phylo_interaction_slope_tree("poll")
    phy1, phy2 = DRModels.augmented_phy(tree1), DRModels.augmented_phy(tree2)
    hosts, polls = phy1.leaf_names, phy2.leaf_names
    npair, nrep = length(hosts) * length(polls), 32
    z_by_pair = shuffle!(rng, repeat([-1.0, 1.0], inner = npair ÷ 2))
    x_grid = collect(range(-1.5, 1.5; length = nrep))
    host = String[]
    pollinator = String[]
    pair = String[]
    x = Float64[]
    z = Float64[]
    pair_index = 0
    # Q = kron(Q2, Q1), so the first tree index varies fastest.
    for p in polls, h in hosts
        pair_index += 1
        pair_x = shuffle!(rng, copy(x_grid))
        for xi in pair_x
            push!(host, h)
            push!(pollinator, p)
            push!(pair, "$(h)__$(p)")
            push!(x, xi)
            push!(z, z_by_pair[pair_index])
        end
    end

    Q1, node1, _ = DRModels._poisson_phylo_setup(tree1, host)
    Q2, node2, _ = DRModels._poisson_phylo_setup(tree2, pollinator)
    Q = kron(Q2, Q1)
    latent = cholesky(Symmetric(Q)).U \ randn(rng, size(Q, 1))
    leaf_node = (node2 .- 1) .* size(Q1, 1) .+ node1
    alpha = [log(0.55), 0.45]
    pair_sd = exp.(alpha[1] .+ alpha[2] .* z_by_pair)
    eta = 0.1 .+ pair_sd[repeat(1:npair, inner = nrep)] .* latent[leaf_node] .* x
    probability = if link === :logit
        1 ./ (1 .+ exp.(-eta))
    elseif link === :probit
        Distributions.cdf.(Distributions.Normal(), eta)
    else
        -expm1.(-exp.(eta))
    end
    successes = Float64.(rand.(Ref(rng), Distributions.Bernoulli.(probability)))
    failures = 1 .- successes
    order = randperm(rng, length(x))
    data = (; successes = successes[order], failures = failures[order],
        x = x[order], z = z[order], host = host[order],
        pollinator = pollinator[order], pair = pair[order])
    formula = bf(
        @formula(cbind(successes, failures) ~
                 1 + phylo_interaction(0 + x | host & pollinator)),
        @formula(sd(pair, phylogenetic_interaction, coef(x)) ~ z),
    )

    target = only(DRModels._sd_target_parts(formula)).target
    @test target.group === :pair
    @test target.level === :phylogenetic_interaction
    @test target.member === :x

    fit = drm(formula, Binomial(link = link); data, tree1, tree2,
        g_tol = 1e-5, se = false)
    @test fit.converged
    @test isfinite(fit.loglik)
    sd_block = last(fit.blocks).first
    sd_coef = coef(fit, sd_block)
    @test all(isfinite, sd_coef)
    @test sd_coef[2] > 0
    @test abs(sd_coef[2] - alpha[2]) < 0.4
    expected_sd = exp.(sd_coef[1] .+ sd_coef[2] .* data.z)
    predicted = DRModels.predict_parameters(fit, data)
    @test predicted[sd_block] ≈ expected_sd
    @test fit.scales[sd_block] ≈ expected_sd
    @test DRModels.predict(fit, data; type = :link) ≈ fill(coef(fit, :mu)[1], length(data.x))

    if link === :logit
        bootstrap = bootstrap_result(fit; data, tree1, tree2, B = 1,
            rng = MersenneTwister(20261032), failures = :skip)
        @test bootstrap.attempted == 1
        @test bootstrap.used == 1
        intervals = bootstrap_ci(fit; data, tree1, tree2, B = 2,
            rng = MersenneTwister(20261033), failures = :skip)
        @test any(row -> row.param === sd_block, intervals)
        summaries = bootstrap_summary(fit; data, tree1, tree2, B = 2,
            rng = MersenneTwister(20261034), failures = :skip)
        @test any(row -> row.param === sd_block, summaries)
    end

    zero_x_data = merge(data, (; x = zeros(length(data.x))))
    simulator = DRModels._marginal_simulator_build(
        fit, zero_x_data; tree1, tree2)
    zero_x_rng = MersenneTwister(20261030)
    simulated_zero_x = simulator(zero_x_rng)
    expected_rng = MersenneTwister(20261030)
    # The marginal simulator draws the latent interaction before the response.
    randn(expected_rng, size(Q, 1))
    expected_zero_x = DRModels._simulate_once(fit, expected_rng;
        mu = DRModels._mean_response(fit.family,
            DRModels.predict(fit, zero_x_data; type = :link)))
    @test simulated_zero_x == expected_zero_x

    zero_slope_data = merge(data, (; x = zeros(length(data.x))))
    err = try
        drm(formula, Binomial(link = link); data = zero_slope_data, tree1, tree2,
            g_tol = 1e-5, se = false)
        nothing
    catch caught
        caught
    end
    @test err isa ArgumentError
    @test occursin("zero for every observation", sprint(showerror, err))
end
end
