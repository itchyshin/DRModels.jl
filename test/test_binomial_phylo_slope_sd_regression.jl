using DRModels, Test, Random, LinearAlgebra
import Distributions
import Serialization

function _binomial_phylo_slope_test_newick(depth)
    function node(prefix, d)
        d == 0 && return "$(prefix):$(1 / depth)"
        left = node(prefix * "a", d - 1)
        right = node(prefix * "b", d - 1)
        return d == depth ? "($left,$right);" : "($left,$right):$(1 / depth)"
    end
    return node("tip", depth)
end

@testset "Binomial phylogenetic random-slope SD regression" begin
    rng = MersenneTwister(20261027)
    tree = _binomial_phylo_slope_test_newick(7)
    phy = DRModels.augmented_phy(tree)
    G = phy.n_leaves
    n_each = 64
    tip_index = repeat(1:G; inner = n_each)
    species = repeat(phy.leaf_names; inner = n_each)
    ztip = collect(range(-1.0, 1.0; length = G))
    z = ztip[tip_index]
    x = repeat(collect(range(-1.5, 1.5; length = n_each)), outer = G)
    K = DRModels.sigma_phy_dense(phy; σ²_phy = 1.0)
    latent_slope = cholesky(Symmetric(K)).L * randn(rng, G)
    alpha = [-0.85, 0.5]
    sd_slope = exp.(alpha[1] .+ alpha[2] .* ztip)
    eta = -0.1 .+ 0.45 .* x .+
          x .* sd_slope[tip_index] .* latent_slope[tip_index]
    probability = Distributions.logistic.(eta)
    successes = rand.(Ref(rng), Distributions.Bernoulli.(probability))
    failures = 1 .- successes
    data = (; successes, failures, x, z, species)
    formula = bf(
        @formula(cbind(successes, failures) ~ x + phylo(0 + x | species)),
        @formula(sd(species, phylogenetic) ~ z),
    )

    fit = drm(formula, DRModels.Binomial(link = :logit);
              data, tree, se = true, g_tol = 1e-5)

    @test fit.converged
    @test isfinite(fit.loglik)
    @test all(isfinite, vcov(fit))
    @test abs(coef(fit, :sd_phylo)[2] - alpha[2]) < 0.35
    sd_group = DRModels._binomial_sd_scale(hcat(ones(G), ztip) *
                                           coef(fit, :sd_phylo))
    @test fit.scales[:sd_phylo] ≈ sd_group[tip_index]
    @test predict_parameters(fit, data)[:sd_phylo] ≈ sd_group[tip_index]
    @test length(predict(fit, data)) == length(successes)
    serialized = IOBuffer()
    Serialization.serialize(serialized, fit)
    seekstart(serialized)
    restored = Serialization.deserialize(serialized)
    @test predict_parameters(restored, data)[:sd_phylo] ≈ sd_group[tip_index]
    simulated = simulate(fit; rng = MersenneTwister(2026102731))
    @test length(simulated) == length(successes)
    @test all(value -> value == 0.0 || value == 1.0, simulated)

    draw_seed = 2026102732
    marginal_simulator = DRModels._marginal_simulator_build(fit, data; tree)
    actual_draw = marginal_simulator(MersenneTwister(draw_seed))
    oracle_rng = MersenneTwister(draw_seed)
    tip_effect = sd_group .* (cholesky(Symmetric(K)).L * randn(oracle_rng, G))
    fixed_eta = predict(fit, data; type = :link)
    expected_probability = Distributions.logistic.(
        fixed_eta .+ x .* tip_effect[tip_index]
    )
    trials = round.(Int, DRModels._scale_vector(fit, :trials))
    expected_draw = Float64[
        rand(oracle_rng, Distributions.Binomial(
            trials[i], clamp(expected_probability[i], eps(), 1 - eps()),
        )) for i in eachindex(successes)
    ]
    @test actual_draw == expected_draw

    boot = bootstrap_result(fit; data, B = 1, tree,
        rng = MersenneTwister(2026102733), failures = :error,
        check_converged = true)
    @test boot.attempted == 1
    @test boot.used == 1
    @test boot.failed == 0

    for link in (:probit, :cloglog)
        probability = if link === :probit
            Distributions.cdf.(Ref(Distributions.Normal()), eta)
        else
            -expm1.(-exp.(eta))
        end
        link_rng = MersenneTwister(link === :probit ? 2026102734 : 2026102735)
        link_successes = rand.(Ref(link_rng), Distributions.Bernoulli.(probability))
        link_failures = 1 .- link_successes
        link_data = (; successes = link_successes, failures = link_failures,
                     x, z, species)
        link_fit = drm(formula, DRModels.Binomial(link = link);
                       data = link_data, tree, se = true, g_tol = 1e-5)
        @test link_fit.converged
        @test isfinite(link_fit.loglik)
        @test all(isfinite, vcov(link_fit))
        @test abs(coef(link_fit, :sd_phylo)[2] - alpha[2]) < 0.4
    end
end
