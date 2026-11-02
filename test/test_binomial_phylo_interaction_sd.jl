using DRModels, StatsModels, Test, Random, LinearAlgebra
import Distributions

function _binomial_phylo_interaction_test_tree(prefix; depth = 3, height = 1.0)
    function node(name, depth)
        depth == 0 && return "$(name):$(height / 3)"
        left, right = node(name * "a", depth - 1), node(name * "b", depth - 1)
        return depth == 3 ? "($left,$right)" : "($left,$right):$(height / 3)"
    end
    return node(prefix, depth) * ";"
end

@testset "Binomial phylogenetic-interaction SD formula target" begin
    @test isdefined(DRModels, :phylo_interaction)
    if isdefined(DRModels, :phylo_interaction)
        fmu = @formula(cbind(successes, failures) ~ x +
                       phylo_interaction(1 | host & pollinator))
        @test any(t -> t isa StatsModels.FunctionTerm &&
                       t.f === DRModels.phylo_interaction,
                  fmu.rhs isa Tuple ? fmu.rhs : (fmu.rhs,))
        parts = DRModels._binomial_phylo_interaction_parts(fmu.rhs)
        @test parts.group1 === :host
        @test parts.group2 === :pollinator
        @test parts.fixed_rhs isa StatsModels.Term
    end

    f = bf(
        @formula(cbind(successes, failures) ~ 1),
        @formula(sd(pair, phylogenetic_interaction) ~ z),
    )
    targets = DRModels._sd_target_parts(f)

    @test length(targets) == 1
    target = only(targets).target
    @test target.group === :pair
    @test target.level === :phylogenetic_interaction
    @test target.dpar === :mu
    @test target.member === nothing
end

@testset "Binomial phylogenetic-interaction SD fit and output" begin
    rng = MersenneTwister(20261027)
    tree1 = _binomial_phylo_interaction_test_tree("host")
    tree2 = _binomial_phylo_interaction_test_tree("poll")
    phy1, phy2 = DRModels.augmented_phy(tree1), DRModels.augmented_phy(tree2)
    hosts = phy1.leaf_names
    polls = phy2.leaf_names
    host = String[]; pollinator = String[]; pair = String[]
    x = Float64[]; z = Float64[]
    z_by_pair = shuffle!(rng, vcat(fill(-1.0, 32), fill(1.0, 32)))
    pair_index = 0
    for (ih, h) in enumerate(hosts), (ip, p) in enumerate(polls)
        pair_index += 1
        for rep in 1:12
        push!(host, h); push!(pollinator, p); push!(pair, "$(h)__$(p)")
        push!(x, 2rand(rng) - 1)
        push!(z, z_by_pair[pair_index])
        end
    end
    Q1, node1, _ = DRModels._poisson_phylo_setup(tree1, host)
    Q2, node2, _ = DRModels._poisson_phylo_setup(tree2, pollinator)
    Q = kron(Q2, Q1)
    latent = cholesky(Symmetric(Q)).U \ randn(rng, size(Q, 1))
    node = (node2 .- 1) .* size(Q1, 1) .+ node1
    eta = -0.2 .+ 0.3 .* x .+
          DRModels._binomial_sd_scale(-0.2 .+ 0.8 .* z) .* latent[node]
    prob = 1 ./ (1 .+ exp.(-eta))
    successes = Float64.(rand.(Ref(rng), Distributions.Bernoulli.(prob)))
    failures = 1 .- successes
    order = randperm(rng, length(x))
    data = (; successes = successes[order], failures = failures[order],
        x = x[order], z = z[order], host = host[order],
        pollinator = pollinator[order], pair = pair[order])
    formula = bf(
        @formula(cbind(successes, failures) ~ x +
                 phylo_interaction(1 | host & pollinator)),
        @formula(sd(pair, phylogenetic_interaction) ~ z),
    )
    fit = drm(formula, Binomial(); data, tree1, tree2, g_tol = 1e-5)
    @test fit.converged
    @test isfinite(fit.loglik)
    @test all(isfinite, coef(fit, :sd))
    @test coef(fit, :sd)[2] > 0
    predicted = DRModels.predict_parameters(fit, data)
    expected_sd = DRModels._binomial_sd_scale(
        coef(fit, :sd)[1] .+ coef(fit, :sd)[2] .* data.z)
    @test predicted[:sd] ≈ expected_sd
    @test fit.scales[:sd] ≈ expected_sd
    @test DRModels.predict(fit, data; type = :link) ≈
          coef(fit, :mu)[1] .+ coef(fit, :mu)[2] .* data.x
    @test all(isfinite, vcov(fit))
    marginal_simulator = DRModels._marginal_simulator_build(
        fit, data; tree1, tree2)
    simulated = marginal_simulator(MersenneTwister(20261028))
    @test length(simulated) == length(data.successes)
    @test all(y -> y == 0 || y == 1, simulated)
    bootstrap = bootstrap_result(fit; data, tree1, tree2, B = 1,
                                rng = MersenneTwister(20261029), failures = :skip)
    @test bootstrap.attempted == 1
    @test bootstrap.used == 1
    bootstrap_intervals = bootstrap_ci(
        fit; data, tree1, tree2, B = 2,
        rng = MersenneTwister(20261030), failures = :skip)
    @test !isempty(bootstrap_intervals)
    @test any(row -> row.param === :sd, bootstrap_intervals)
    bootstrap_summaries = bootstrap_summary(
        fit; data, tree1, tree2, B = 2,
        rng = MersenneTwister(20261031), failures = :skip)
    @test any(row -> row.param === :sd, bootstrap_summaries)
    formula_intervals = bootstrap_ci(
        formula, Binomial(); data, tree1, tree2, B = 2,
        rng = MersenneTwister(20261101), failures = :skip)
    @test any(row -> row.param === :sd, formula_intervals)

    for (link, seed_offset) in ((:probit, 1), (:cloglog, 2))
        linked_rng = MersenneTwister(20261110 + seed_offset)
        eta_link = eta[order]
        probability_link = link === :probit ?
            Distributions.cdf.(Distributions.Normal(), eta_link) :
            -expm1.(-exp.(eta_link))
        linked_successes = Float64.(rand.(Ref(linked_rng),
            Distributions.Bernoulli.(probability_link)))
        linked_data = merge(data, (; successes = linked_successes,
                                    failures = 1 .- linked_successes))
        linked_fit = drm(formula, Binomial(link = link);
                         data = linked_data, tree1, tree2, g_tol = 1e-5, se = false)
        @test linked_fit.converged
        @test isfinite(linked_fit.loglik)
        @test all(isfinite, coef(linked_fit, :sd))
        @test coef(linked_fit, :sd)[2] > 0
        @test abs(coef(linked_fit, :sd)[2] - 0.8) < 0.65
        linked_prediction = DRModels.predict_parameters(linked_fit, linked_data)
        expected_linked_sd = DRModels._binomial_sd_scale(
            coef(linked_fit, :sd)[1] .+ coef(linked_fit, :sd)[2] .* linked_data.z)
        @test linked_prediction[:sd] ≈ expected_linked_sd
        @test linked_fit.scales[:sd] ≈ expected_linked_sd
        linked_simulation = simulate(linked_fit; rng = MersenneTwister(20261120 + seed_offset))
        @test length(linked_simulation) == length(linked_data.successes)
        @test all(y -> y == 0.0 || y == 1.0, linked_simulation)
    end

    badpair = copy(data.pair); badpair[1] *= "_split"
    bad_data = merge(data, (; pair = badpair))
    @test_throws ArgumentError drm(formula, Binomial(); data = bad_data, tree1, tree2)

    badz = copy(data.z); badz[1] += 1
    bad_data = merge(data, (; z = badz))
    @test_throws ArgumentError drm(formula, Binomial(); data = bad_data, tree1, tree2)

    ordinary_formula = bf(@formula(successes ~ x))
    @test_throws ArgumentError drm(ordinary_formula, Binomial(); data, tree1, tree2)
end
