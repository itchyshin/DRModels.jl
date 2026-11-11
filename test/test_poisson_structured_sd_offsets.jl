using DRModels, Test, Random, LinearAlgebra
import Distributions

function _poisson_offset_sd_laplace_oracle(theta, y, Xmu, Zobs, gidx, Q;
                                            offset = zeros(length(y)),
                                            weight = ones(length(y)))
    Qd = Matrix(Q)
    pmu = size(Xmu, 2)
    beta = theta[1:pmu]
    alpha = theta[pmu+1:end]
    eta0 = Xmu * beta .+ offset
    loading = weight .* exp.(Zobs * alpha)
    function joint(b)
        value = 0.5 * dot(b, Qd * b)
        for i in eachindex(y)
            eta = eta0[i] + loading[i] * b[gidx[i]]
            value += exp(eta) - y[i] * eta + DRModels._logfactorial(round(Int, y[i]))
        end
        value
    end
    result = DRModels.Optim.optimize(joint, zeros(size(Qd, 1)),
        DRModels.Optim.LBFGS(), DRModels.Optim.Options(
            g_tol = 1e-10, iterations = 400); autodiff = :forward)
    bmode = DRModels.Optim.minimizer(result)
    H = DRModels.ForwardDiff.hessian(joint, bmode)
    return joint(bmode) + 0.5 * logdet(Symmetric(H)) -
           0.5 * logdet(Symmetric(Qd))
end

function _poisson_offset_sd_newick(depth; height = 1.0)
    function node(prefix, d)
        d == 0 && return "$(prefix):$(height / depth)"
        left = node(prefix * "a", d - 1)
        right = node(prefix * "b", d - 1)
        return d == depth ? "($left,$right);" : "($left,$right):$(height / depth)"
    end
    return node("tip", depth)
end

@testset "Poisson animal SD regression includes row offsets" begin
    rng = MersenneTwister(20261025)
    G, m = 12, 12
    group_index = repeat(1:G, inner = m)
    group = ["animal_$g" for g in group_index]
    z_group = repeat([0.0, 1.0], inner = G ÷ 2)
    z = z_group[group_index]
    x = randn(rng, G * m)
    log_exposure = 0.25 .* sin.(range(0.0, 4.0; length = G * m))
    R = 0.4 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    scale = collect(range(0.7, 1.6; length = G))
    A = Diagonal(scale) * R * Diagonal(scale)
    alpha = [log(0.4), 0.35]
    latent = cholesky(Symmetric(R)).L * randn(rng, G)
    eta = 0.1 .+ 0.2 .* x .+ log_exposure .+
          exp.(alpha[1] .+ alpha[2] .* z_group)[group_index] .* latent[group_index]
    y = Float64[rand(rng, Distributions.Poisson(exp(v))) for v in eta]
    data = (; y, x, z, g = group, log_exposure)
    formula = bf(@formula(y ~ x + offset(log_exposure) + animal(1 | g)),
                 @formula(sd(g, animal) ~ z))
    fit = drm(formula, Poisson(); data, A, se = false)

    @test fit.converged
    @test fitted(fit; re = :zero) ≈ exp.(coef(fit, :mu)[1] .+
                              coef(fit, :mu)[2] .* x .+ log_exposure)
    @test predict_parameters(fit, data)[:mu] ≈ fitted(fit; re = :zero)
    theta = [0.1, 0.2, alpha...]
    Q, gidx = DRModels._general_cov_setup(A, group; normalize = false)
    Zg = hcat(ones(G), z_group)
    oracle = _poisson_offset_sd_laplace_oracle(
        theta, y, hcat(ones(length(y)), x), Zg[gidx, :], gidx, Q;
        offset = log_exposure)
    @test fit.nll(theta) ≈ oracle atol = 3e-4

    # The direct precision interface must include the same fixed offset.
    Ainv = inv(Symmetric(A))
    precision_fit = drm(formula, Poisson(); data, Ainv, se = false)
    @test precision_fit.converged
    @test precision_fit.loglik ≈ fit.loglik atol = 2e-5
    @test fitted(precision_fit) ≈ fitted(fit)
end

@testset "Poisson animal-slope SD regression includes row offsets" begin
    rng = MersenneTwister(20261027)
    G, m = 12, 12
    group_index = repeat(1:G, inner = m)
    group = ["animal_$g" for g in group_index]
    z_group = repeat([0.0, 1.0], inner = G ÷ 2)
    z = z_group[group_index]
    x = randn(rng, G * m)
    log_exposure = 0.2 .* cos.(range(0.0, 4.0; length = G * m))
    R = 0.4 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    scale = collect(range(0.7, 1.6; length = G))
    A = Diagonal(scale) * R * Diagonal(scale)
    alpha = [log(0.35), 0.3]
    latent = cholesky(Symmetric(R)).L * randn(rng, G)
    eta = 0.1 .+ 0.2 .* x .+ log_exposure .+
          exp.(alpha[1] .+ alpha[2] .* z_group)[group_index] .* latent[group_index] .* x
    y = Float64[rand(rng, Distributions.Poisson(exp(v))) for v in eta]
    data = (; y, x, z, g = group, log_exposure)
    formula = bf(@formula(y ~ x + offset(log_exposure) + animal(0 + x | g)),
                 @formula(sd(g, animal) ~ z))
    fit = drm(formula, Poisson(); data, A, se = false)

    @test fit.converged
    @test fitted(fit; re = :zero) ≈ exp.(coef(fit, :mu)[1] .+
                              coef(fit, :mu)[2] .* x .+ log_exposure)
    theta = [0.1, 0.2, alpha...]
    Q, gidx = DRModels._general_cov_setup(A, group; normalize = false)
    Zobs = hcat(ones(length(y)), z)
    oracle = _poisson_offset_sd_laplace_oracle(
        theta, y, hcat(ones(length(y)), x), Zobs, gidx, Q;
        offset = log_exposure, weight = x)
    @test fit.nll(theta) ≈ oracle atol = 5e-4

    # Marginal simulation must carry the fixed offset into each newly drawn
    # animal-slope predictor, just as the fitting likelihood does.
    seed = 20261029
    simulator = DRModels._marginal_simulator_build(fit, data; A)
    actual = simulator(MersenneTwister(seed))
    oracle_rng = MersenneTwister(seed)
    sd_group = exp.(hcat(ones(G), z_group) * coef(fit, :sd))
    latent_draw = sd_group .* (cholesky(Symmetric(A)).L * randn(oracle_rng, G))
    eta_fixed = coef(fit, :mu)[1] .+ coef(fit, :mu)[2] .* x .+ log_exposure
    expected = Float64[rand(oracle_rng, Distributions.Poisson(
        exp(eta_fixed[i] + x[i] * latent_draw[gidx[i]]))) for i in eachindex(y)]
    @test actual == expected
end

@testset "Poisson phylogenetic SD regression includes row offsets" begin
    rng = MersenneTwister(20261026)
    tree = _poisson_offset_sd_newick(2)
    phy = DRModels.augmented_phy(tree)
    m = 8
    labels = repeat(phy.leaf_names, inner = m)
    group_index = repeat(1:phy.n_leaves, inner = m)
    z_group = repeat([0.0, 1.0, 0.0, 1.0], inner = 1)
    z = z_group[group_index]
    x = randn(rng, length(labels))
    log_exposure = 0.2 .* cos.(range(0.0, 5.0; length = length(labels)))
    alpha = [log(0.35), 0.3]
    Q, leaf_node, _ = DRModels._poisson_phylo_setup(phy, labels)
    C = inv(Symmetric(Matrix(Q)))
    latent = cholesky(Symmetric(C)).L * randn(rng, size(C, 1))
    loading = exp.(alpha[1] .+ alpha[2] .* z)
    eta = 0.1 .+ 0.2 .* x .+ log_exposure .+ loading .* latent[leaf_node]
    y = Float64[rand(rng, Distributions.Poisson(exp(v))) for v in eta]
    data = (; y, x, z, species = labels, log_exposure)
    formula = bf(@formula(y ~ x + offset(log_exposure) + phylo(1 | species)),
                 @formula(sd(species, phylogenetic) ~ z))
    fit = drm(formula, Poisson(); data, tree, se = false, g_tol = 1e-6)

    @test fit.converged
    @test fitted(fit; re = :zero) ≈ exp.(coef(fit, :mu)[1] .+
                              coef(fit, :mu)[2] .* x .+ log_exposure)
    @test predict_parameters(fit, data)[:mu] ≈ fitted(fit; re = :zero)
    Zobs = hcat(ones(length(z)), z)
    theta = [0.1, 0.2, alpha...]
    oracle = _poisson_offset_sd_laplace_oracle(
        theta, y, hcat(ones(length(y)), x), Zobs, leaf_node, Q;
        offset = log_exposure)
    @test fit.nll(theta) ≈ oracle atol = 3e-4
end

@testset "Poisson related-matrix SD regression includes row offsets" begin
    rng = MersenneTwister(20261024)
    G, m = 12, 12
    group_index = repeat(1:G, inner = m)
    group = ["site_$g" for g in group_index]
    z_group = repeat([0.0, 1.0], inner = G ÷ 2)
    z = z_group[group_index]
    x = randn(rng, G * m)
    log_exposure = range(-0.5, 0.5; length = G * m) |> collect
    R = 0.4 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    alpha = [log(0.4), 0.35]
    latent = cholesky(Symmetric(R)).L * randn(rng, G)
    eta = 0.1 .+ 0.2 .* x .+ log_exposure .+
          exp.(alpha[1] .+ alpha[2] .* z_group)[group_index] .* latent[group_index]
    y = Float64[rand(rng, Distributions.Poisson(exp(v))) for v in eta]
    data = (; y, x, z, g = group, log_exposure)
    K = R
    formula = bf(@formula(y ~ x + offset(log_exposure) + relmat(1 | g)),
                 @formula(sd(g, relmat) ~ z))
    fit = drm(formula, Poisson(); data, K, se = false)

    @test fit.converged
    @test fitted(fit; re = :zero) ≈ exp.(coef(fit, :mu)[1] .+
                              coef(fit, :mu)[2] .* x .+ log_exposure)
    @test predict_parameters(fit, data)[:mu] ≈ fitted(fit; re = :zero)

    theta = [0.1, 0.2, alpha...]
    Q, gidx = DRModels._general_cov_setup(K, group)
    Zg = hcat(ones(G), z_group)
    Zobs = Zg[gidx, :]
    oracle = _poisson_offset_sd_laplace_oracle(
        theta, y, hcat(ones(length(y)), x), Zobs, gidx, Q;
        offset = log_exposure)
    @test fit.nll(theta) ≈ oracle atol = 3e-4
    seed = 20261028
    simulator = DRModels._marginal_simulator_build(fit, data; K)
    actual = simulator(MersenneTwister(seed))
    oracle_rng = MersenneTwister(seed)
    sd_group = exp.(Zg * coef(fit, :sd))
    latent_draw = sd_group .* (cholesky(Symmetric(R)).L * randn(oracle_rng, G))
    eta_fixed = predict(fit, data; type = :link)
    expected = Float64[rand(oracle_rng, Distributions.Poisson(
        exp(eta_fixed[i] + latent_draw[gidx[i]]))) for i in eachindex(y)]
    @test actual == expected
end
