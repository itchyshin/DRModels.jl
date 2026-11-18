using DRModels, Test, Random, LinearAlgebra, Serialization
import Distributions

function _gamma_animal_intercept_slope_fixture(; G = 48, m = 24,
                                                seed = 202610231)
    rng = MersenneTwister(seed)
    order = randperm(rng, G)
    ids = ["animal_$(i)" for i in order]
    R = 0.35 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    scale = collect(range(0.7, 1.4; length = G))
    A = Diagonal(scale) * R * Diagonal(scale)
    group = repeat(ids, inner = m)
    gidx = repeat(1:G, inner = m)
    z0group = ifelse.(randperm(rng, G) .<= G ÷ 2, -1.0, 1.0)
    z1group = collect(range(-1.0, 1.0; length = G))[randperm(rng, G)]
    z0, z1 = z0group[gidx], z1group[gidx]
    x = repeat(collect(range(-1.2, 1.2; length = m)), outer = G)
    alpha0 = [-0.65, 0.35]
    alpha1 = [-1.05, 0.45]
    L = cholesky(Symmetric(A)).L
    u0, u1 = L * randn(rng, G), L * randn(rng, G)
    eta = 0.2 .+ 0.3 .* x .+
        (exp.(alpha0[1] .+ alpha0[2] .* z0group) .* u0)[gidx] .+
        x .* (exp.(alpha1[1] .+ alpha1[2] .* z1group) .* u1)[gidx]
    sigma = 0.38
    shape = 1 / sigma^2
    y = [rand(rng, Distributions.Gamma(shape, exp(v) / shape)) for v in eta]
    return (; data = (; y = Float64.(y), x, z0, z1, g = group), A, gidx,
            z0group, z1group, alpha0, alpha1, sigma)
end

function _gamma_animal_intercept_slope_dense_laplace(theta, y, Xmu, x,
                                                       Z0, Z1, gidx, A)
    G = size(A, 1)
    Q = inv(Symmetric(A))
    p_mu = size(Xmu, 2)
    beta = theta[1:p_mu]
    sigma = exp(theta[p_mu + 1])
    p_sd = size(Z0, 2)
    alpha0 = theta[(p_mu + 2):(p_mu + 1 + p_sd)]
    alpha1 = theta[(p_mu + 2 + p_sd):end]
    shape = 1 / sigma^2
    eta0 = Xmu * beta
    load0 = exp.(Z0 * alpha0)
    load1 = x .* exp.(Z1 * alpha1)
    function joint(b)
        b0, b1 = @view(b[1:G]), @view(b[(G + 1):(2G)])
        value = 0.5 * dot(b0, Q * b0) + 0.5 * dot(b1, Q * b1)
        for i in eachindex(y)
            eta = eta0[i] + load0[i] * b0[gidx[i]] + load1[i] * b1[gidx[i]]
            value -= Distributions.logpdf(
                Distributions.Gamma(shape, exp(eta) / shape), y[i])
        end
        value
    end
    result = DRModels.Optim.optimize(joint, zeros(2G), DRModels.Optim.LBFGS(),
        DRModels.Optim.Options(g_tol = 1e-10, iterations = 500);
        autodiff = :forward)
    mode = DRModels.Optim.minimizer(result)
    hessian = DRModels.ForwardDiff.hessian(joint, mode)
    Qjoint = [Matrix(Q) zeros(G, G); zeros(G, G) Matrix(Q)]
    return joint(mode) + 0.5 * logdet(Symmetric(hessian)) -
           0.5 * logdet(Symmetric(Qjoint))
end

@testset "Gamma animal independent intercept and slope SD regression" begin
    fixture = _gamma_animal_intercept_slope_fixture()
    data = fixture.data
    formula = bf(@formula(y ~ x + animal(1 + x | g)),
        @formula(sigma ~ 1),
        @formula(sd(g, animal, coef(intercept)) ~ z0),
        @formula(sd(g, animal, coef(x)) ~ z1))

    targets = DRModels._sd_target_parts(formula)
    @test length(targets) == 2
    @test Set(item.target.member for item in targets) == Set((:intercept, :x))
    @test all(item.target.level === :animal for item in targets)

    bridge_formula = "y ~ x + animal(1 + x | g); sigma ~ 1; " *
        "sd(g, animal, coef(intercept)) ~ z0; " *
        "sd(g, animal, coef(x)) ~ z1"
    bridged, bridged_data = DRModels._bridge_formula(
        bridge_formula, "gamma", DRModels._bridge_data(data))
    @test Set(item.target.member for item in DRModels._sd_target_parts(bridged)) ==
        Set((:intercept, :x))
    @test length(bridged_data.y) == length(data.y)
    # `coef()` on a phylogenetic SD target is now a coefficient-specific target
    # (phylo(1 + x | g) q2 routes); it no longer fails at formula construction.
    phylo_coef = bf(@formula(y ~ phylo(1 | g)),
        @formula(sd(g, phylogenetic, coef(intercept)) ~ z0))
    @test only(DRModels._sd_target_parts(phylo_coef)).target.level === :phylogenetic

    fit = drm(formula, Gamma(); data, A = fixture.A)
    @test fit.converged
    bridge_fit = drm_bridge(; formula = bridge_formula, family = "gamma",
                            data, A = fixture.A)
    @test bridge_fit["converged"]
    @test bridge_fit["coefficients"] ≈ coef(fit) atol = 1e-8
    @test length(unique(bridge_fit["coef_names"])) == length(bridge_fit["coef_names"])
    ordered = [only(filter(t -> t.target.member === m, targets))
               for m in (:intercept, :x)]
    target_blocks = [DRModels._sd_target_block(item.target; disambiguate = true)
                     for item in ordered]
    @test first.(fit.blocks[3:4]) == target_blocks
    @test coef(fit, target_blocks[1])[2] > 0
    @test coef(fit, target_blocks[2])[2] > 0
    @test abs(coef(fit, target_blocks[1])[2] - fixture.alpha0[2]) < 0.6
    @test abs(coef(fit, target_blocks[2])[2] - fixture.alpha1[2]) < 0.6

    zero_loading_data = merge(data, (; x = zeros(length(data.x))))
    @test_throws ArgumentError drm(formula, Gamma(); data = zero_loading_data,
                                   A = fixture.A, se = false)
    binary_loading = repeat([0.0, 1.0], inner = 12, outer = 48)
    binary_loading_data = merge(data, (; x = binary_loading))
    # labelled A: an unlabelled one now warns on every call (robust-entry), which @test_nowarn would catch
    @test_nowarn drm(formula, Gamma(); data = binary_loading_data,
                     A = fixture.A, ids = unique(data.g), se = false)
    group_constant_x = repeat(collect(range(-1.2, 1.2; length = 48)), inner = 24)
    group_constant_x_data = merge(data, (; x = group_constant_x))
    @test_nowarn drm(formula, Gamma(); data = group_constant_x_data,
                     A = fixture.A, ids = unique(data.g), se = false)

    extra_structured_formula = bf(@formula(y ~ x + animal(1 + x | g) + animal(1 | h)),
        @formula(sigma ~ 1),
        @formula(sd(g, animal, coef(intercept)) ~ z0),
        @formula(sd(g, animal, coef(x)) ~ z1))
    extra_structured_data = merge(data, (; h = repeat(1:24, outer = 48)))
    @test_throws ArgumentError drm(extra_structured_formula, Gamma();
        data = extra_structured_data, A = fixture.A, se = false)

    Z0 = hcat(ones(length(fixture.z0group)), fixture.z0group)
    Z1 = hcat(ones(length(fixture.z1group)), fixture.z1group)
    @test fit.scales[Symbol(string(target_blocks[1]), "_group")] ≈
        exp.(Z0 * coef(fit, target_blocks[1]))
    @test fit.scales[Symbol(string(target_blocks[2]), "_group")] ≈
        exp.(Z1 * coef(fit, target_blocks[2]))
    @test predict_parameters(fit, data; se = false)[target_blocks[1]] ≈ fit.scales[target_blocks[1]]
    @test predict_parameters(fit, data; se = false)[target_blocks[2]] ≈ fit.scales[target_blocks[2]]

    one_per_group = (; g = first.(Iterators.partition(data.g, 24)),
                     z0 = fixture.z0group, z1 = fixture.z1group,
                     x = zeros(length(fixture.z0group)))
    new_parameters = predict_parameters(fit, one_per_group; se = false)
    @test new_parameters[target_blocks[1]] ≈ exp.(Z0 * coef(fit, target_blocks[1]))
    @test new_parameters[target_blocks[2]] ≈ exp.(Z1 * coef(fit, target_blocks[2]))

    simulator = DRModels._marginal_simulator_build(fit, data; A = fixture.A)
    actual = simulator(MersenneTwister(202610232))
    oracle_rng = MersenneTwister(202610232)
    sd0 = exp.(Z0 * coef(fit, target_blocks[1]))
    sd1 = exp.(Z1 * coef(fit, target_blocks[2]))
    L = cholesky(Symmetric(fixture.A)).L
    b0 = sd0 .* (L * randn(oracle_rng, length(sd0)))
    b1 = sd1 .* (L * randn(oracle_rng, length(sd1)))
    eta_fixed = predict(fit, data; type = :link)
    shape = exp(-2 * coef(fit, :sigma)[1])
    mu_star = exp.(eta_fixed .+ b0[fixture.gidx] .+
                   data.x .* b1[fixture.gidx])
    expected = [rand(oracle_rng, Distributions.Gamma(shape, mu / shape))
                for mu in mu_star]
    @test actual ≈ expected atol = 1e-12 rtol = 1e-12

    Xmu = hcat(ones(length(data.y)), data.x)
    theta = [0.2, 0.3, log(fixture.sigma), fixture.alpha0..., fixture.alpha1...]
    dense_nll = _gamma_animal_intercept_slope_dense_laplace(
        theta, data.y, Xmu, data.x, hcat(ones(length(data.y)), data.z0),
        hcat(ones(length(data.y)), data.z1), fixture.gidx, fixture.A)
    @test fit.nll(theta) ≈ dense_nll atol = 5e-3

    constant_formula = bf(@formula(y ~ x + animal(1 + x | g)),
        @formula(sigma ~ 1),
        @formula(sd(g, animal, coef(intercept)) ~ 1),
        @formula(sd(g, animal, coef(x)) ~ 1))
    unidentifiable_formula = bf(@formula(y ~ animal(1 + x | g)),
        @formula(sigma ~ 1),
        @formula(sd(g, animal, coef(intercept)) ~ 1),
        @formula(sd(g, animal, coef(x)) ~ 1))
    constant_loading_data = merge(data, (; x = ones(length(data.x))))
    @test_throws ArgumentError drm(unidentifiable_formula, Gamma();
        data = constant_loading_data, A = fixture.A, se = false)
    constant_fit = drm(constant_formula, Gamma(); data, A = fixture.A, se = false)
    @test constant_fit.converged
    constant_blocks = [DRModels._sd_target_block(item.target; disambiguate = true)
                       for item in DRModels._sd_target_parts(constant_formula)]
    @test length(coef(constant_fit, constant_blocks[1])) == 1
    @test length(coef(constant_fit, constant_blocks[2])) == 1
    constant_theta = vcat(coef(constant_fit, :mu), coef(constant_fit, :sigma),
        coef(constant_fit, constant_blocks[1]), coef(constant_fit, constant_blocks[2]))
    constant_dense_nll = _gamma_animal_intercept_slope_dense_laplace(
        constant_theta, data.y, Xmu, data.x,
        ones(length(data.y), 1), ones(length(data.y), 1), fixture.gidx, fixture.A)
    @test constant_fit.nll(constant_theta) ≈ constant_dense_nll atol = 5e-3

    boot = bootstrap_result(fit; data, A = fixture.A, B = 1,
        rng = MersenneTwister(202610233), failures = :error)
    @test boot.attempted == boot.used == 1
    @test boot.failed == 0

    path = tempname()
    open(path, "w") do io
        serialize(io, fit)
    end
    reloaded = open(deserialize, path)
    @test predict_parameters(reloaded, data; se = false)[target_blocks[2]] ≈
        fit.scales[target_blocks[2]]
end
