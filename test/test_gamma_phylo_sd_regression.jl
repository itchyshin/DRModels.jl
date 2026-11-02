using DRModels, Test, Random, LinearAlgebra, Distributions, Serialization

function _gamma_phylo_sd_fixture(; seed = 20261022, p = 16, m = 20)
    rng = MersenneTwister(seed)
    tree = random_balanced_tree(p; branch_length = 0.2)
    ztip = repeat([-1.0, 1.0], p ÷ 2)
    alpha = [-0.55, 0.50]
    tip_sd = exp.(alpha[1] .+ alpha[2] .* ztip)
    covariance = sigma_phy_dense(tree; σ²_phy = 1.0)
    latent = cholesky(Symmetric(covariance)).L * randn(rng, p)
    species_index = repeat(1:p; inner = m)
    species = tree.leaf_names[species_index]
    z = ztip[species_index]
    x = repeat([-1.0, 1.0], length(species_index) ÷ 2)
    eta = 0.25 .+ 0.25 .* x .+ (tip_sd .* latent)[species_index]
    mu = exp.(eta)
    sigma = 0.35
    shape = inv(sigma^2)
    y = Float64[rand(rng, Distributions.Gamma(shape, mui / shape)) for mui in mu]
    return (; tree, data = (; y, x, species, z), alpha, ztip, covariance,
            species_index, eta)
end

@testset "Gamma phylogenetic random-effect SD regression" begin
    fixture = _gamma_phylo_sd_fixture()
    (; tree, data, alpha, ztip, covariance, species_index) = fixture
    formula = bf(
        @formula(y ~ x + phylo(1 | species)),
        @formula(sigma ~ 1),
        @formula(sd(species, phylogenetic) ~ z),
    )

    fit = drm(formula, DRModels.Gamma(); data, tree, se = false, g_tol = 1e-5)
    @test fit.converged
    @test isfinite(fit.loglik)
    @test coef(fit, :sd_phylo)[2] > 0
    @test coef(fit, :sd_phylo)[2] ≈ alpha[2] atol = 0.45
    fitted_alpha = coef(fit, :sd_phylo)
    @test fit.scales[:sd_phylo_group] ≈
          exp.(fitted_alpha[1] .+ fitted_alpha[2] .* ztip)
    @test fit.scales[:sd_phylo] ≈ fit.scales[:sd_phylo_group][species_index]
    @test length(unique(fit.scales[:sd_phylo_group_index])) == length(ztip)
    @test predict_parameters(fit, data)[:sd_phylo] ≈ fit.scales[:sd_phylo]
    saved_fit = tempname()
    open(saved_fit, "w") do io
        serialize(io, fit)
    end
    reloaded = open(deserialize, saved_fit)
    rm(saved_fit; force = true)
    @test coef(reloaded, :sd_phylo) ≈ coef(fit, :sd_phylo)
    @test predict_parameters(reloaded, data)[:sd_phylo] ≈ fit.scales[:sd_phylo]

    constant_formula = bf(
        @formula(y ~ x + phylo(1 | species)),
        @formula(sigma ~ 1),
        @formula(sd(species, phylogenetic) ~ 1),
    )
    constant_fit = drm(constant_formula, DRModels.Gamma(); data, tree, se = false, g_tol = 1e-5)
    existing_fit = drm(
        bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
        DRModels.Gamma(); data, tree, se = false, g_tol = 1e-5,
    )
    @test constant_fit.converged
    @test existing_fit.converged
    @test constant_fit.loglik ≈ existing_fit.loglik atol = 2e-5

    # The sparse tree route must retain tree-tip identity when input rows use
    # a different first-seen order from the tree.
    permutation = reverse(collect(eachindex(data.y)))
    shuffled = (; (key => getproperty(data, key)[permutation]
                   for key in keys(data))...)
    shuffled_fit = drm(formula, DRModels.Gamma(); data = shuffled, tree, se = false, g_tol = 1e-5)
    @test shuffled_fit.converged
    @test coef(shuffled_fit, :sd_phylo) ≈ coef(fit, :sd_phylo) atol = 2e-4

    @test_throws ArgumentError drm(formula, DRModels.Gamma(); data, g_tol = 1e-5)
    missing_tip = filter(i -> data.species[i] != tree.leaf_names[end], eachindex(data.y))
    incomplete_tree_data = (; (key => getproperty(data, key)[missing_tip]
                               for key in keys(data))...)
    @test_throws ArgumentError drm(formula, DRModels.Gamma(); data = incomplete_tree_data,
                                   tree, se = false, g_tol = 1e-5)

    simulator = DRModels._marginal_simulator_build(fit, data; tree)
    @test simulator isa Function
    simulator_seed = 20261023
    draw = simulator(MersenneTwister(simulator_seed))
    @test length(draw) == length(data.y)
    @test all(isfinite, draw)
    # The raw field is indexed by original augmented-tree node IDs; rebuild
    # this simulator draw through the tree's original tip map.
    oracle_rng = MersenneTwister(simulator_seed)
    raw_field = DRModels._draw_phylo_raw_field(oracle_rng, tree)
    eta_fixed = predict(fit, data; type = :link)
    mu_oracle = exp.(eta_fixed .+
        fit.scales[:sd_phylo_group][species_index] .*
        raw_field[tree.leaf_indices[species_index]])
    expected_draw = DRModels._simulate_once(fit, oracle_rng; mu = mu_oracle)
    @test draw == expected_draw
    low_row = findfirst(i -> species_index[i] == 1 && data.x[i] == -1,
                        eachindex(species_index))
    high_row = findfirst(i -> species_index[i] == 2 && data.x[i] == -1,
                         eachindex(species_index))
    sim_rng = MersenneTwister(20261026)
    low_residuals = [log(simulator(sim_rng)[low_row]) - eta_fixed[low_row]
                     for _ in 1:600]
    high_residuals = [log(simulator(sim_rng)[high_row]) - eta_fixed[high_row]
                      for _ in 1:600]
    @test var(high_residuals) > var(low_residuals)
    field_rng = MersenneTwister(20261025)
    field_draws = hcat((DRModels._draw_phylo_raw_field(field_rng, tree)[tree.leaf_indices]
                        for _ in 1:10000)...)
    @test maximum(abs, cov(field_draws; dims = 2) .- covariance) < 0.1

    # Unequal root-to-tip lengths require tip-specific marginal variances;
    # one global tree-height multiplier would give the wrong D*K*D covariance.
    irregular_tree = augmented_phy("((a:0.2,b:0.5):0.3,(c:0.1,d:0.7):0.4);")
    irregular_covariance = sigma_phy_dense(irregular_tree; σ²_phy = 1.0)
    irregular_rng = MersenneTwister(20261027)
    irregular_draws = hcat((DRModels._draw_phylo_raw_field(irregular_rng, irregular_tree)[irregular_tree.leaf_indices]
                           for _ in 1:20000)...)
    empirical_covariance = cov(irregular_draws; dims = 2)
    @test diag(irregular_covariance) ≈ [0.5, 0.8, 0.5, 1.1]
    @test maximum(abs, empirical_covariance .- irregular_covariance) < 0.04
    tip_sd = [0.7, 1.0, 1.4, 2.0]
    scaled_target = Diagonal(tip_sd) * irregular_covariance * Diagonal(tip_sd)
    scaled_rng = MersenneTwister(20261028)
    scaled_tip_draws = hcat((DRModels._draw_scaled_phylo_tip_field(
        scaled_rng, irregular_tree, tip_sd) for _ in 1:20000)...)
    @test maximum(abs, cov(scaled_tip_draws; dims = 2) .- scaled_target) < 0.16

    bootstrap = bootstrap_result(fit; data, B = 1, tree,
        rng = MersenneTwister(20261024), failures = :error,
        check_converged = true)
    @test bootstrap.attempted == 1
    @test bootstrap.used == 1
    @test bootstrap.failed == 0
end
