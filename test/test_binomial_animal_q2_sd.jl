using DRModels
using Test, LinearAlgebra, Random, SparseArrays, Serialization, StableRNGs

@testset "Bernoulli animal q2 SD regression" begin
    rng = StableRNG(20261028)
    G, m = 32, 28
    level_order = randperm(rng, G)
    group_index = repeat(1:G, inner = m)
    group = repeat(level_order, inner = m)
    z0_group = repeat([-1.0, 1.0], inner = G ÷ 2)
    z1_group = repeat([-1.0, -1.0, 1.0, 1.0], inner = G ÷ 4)
    z0, z1 = z0_group[group_index], z1_group[group_index]
    x = repeat(range(-1.2, 1.2; length = m), outer = G)
    A = [0.3^abs(i - j) for i in 1:G, j in 1:G]

    alpha0 = [-0.6, 0.5]
    alpha1 = [-0.75, 0.6]
    rho = 0.4
    tau0 = exp.(alpha0[1] .+ alpha0[2] .* z0_group)
    tau1 = exp.(alpha1[1] .+ alpha1[2] .* z1_group)
    latent0 = cholesky(Symmetric(A)).L * randn(rng, G)
    latent1_independent = cholesky(Symmetric(A)).L * randn(rng, G)
    latent1 = rho .* latent0 .+ sqrt(1 - rho^2) .* latent1_independent
    eta = -0.1 .+ 0.25 .* x .+ tau0[group_index] .* latent0[group_index] .+
          x .* tau1[group_index] .* latent1[group_index]
    y = Float64.(rand.(Ref(rng), DRModels.Distributions.Bernoulli.(
        1 ./ (1 .+ exp.(-eta)),
    )))
    data = (; y, x, z0, z1, group)
    formula = bf(
        @formula(y ~ x + animal(1 + x | p | group)),
        @formula(sd(group, animal, coef(intercept)) ~ z0),
        @formula(sd(group, animal, coef(x)) ~ z1),
    )

    fit = drm(formula, DRModels.Binomial(); data, A, se = false)
    @test fit.converged
    ordered = map((:intercept, :x)) do member
        only(filter(item -> item.target.member === member,
                    DRModels._sd_target_parts(formula)))
    end
    blocks = map(item -> DRModels._sd_target_block(item.target;
        disambiguate = true), ordered)
    alpha0_hat, alpha1_hat = coef(fit, blocks[1]), coef(fit, blocks[2])
    @test length(alpha0_hat) == 2
    @test length(alpha1_hat) == 2
    @test abs(alpha0_hat[2] - alpha0[2]) < 0.45
    @test abs(alpha1_hat[2] - alpha1[2]) < 0.45
    @test abs(tanh(only(coef(fit, :recov))) - rho) < 0.45
    @test only(fit.scales[:correlation]) ≈ tanh(only(coef(fit, :recov)))
    @test fit.scales[Symbol(string(blocks[1]), "_group")] ≈
          exp.(hcat(ones(G), z0_group) * alpha0_hat)
    @test fit.scales[Symbol(string(blocks[2]), "_group")] ≈
          exp.(hcat(ones(G), z1_group) * alpha1_hat)
    predicted_scales = predict_parameters(fit, data; se = false)
    @test predicted_scales[blocks[1]] ≈ fit.scales[blocks[1]]
    @test predicted_scales[blocks[2]] ≈ fit.scales[blocks[2]]
    serialized = IOBuffer()
    Serialization.serialize(serialized, fit)
    seekstart(serialized)
    restored = Serialization.deserialize(serialized)
    @test predict_parameters(restored, data; se = false)[blocks[1]] ≈
        predicted_scales[blocks[1]]
    @test predict_parameters(restored, data; se = false)[blocks[2]] ≈
        predicted_scales[blocks[2]]
    @test restored.scales[:correlation] ≈ fit.scales[:correlation]
    profile_target = only(filter(profile_targets(fit)) do target
        target.param === blocks[1] && target.parm == "z0"
    end)
    @test profile_target.profile_ready
    sd_profile = profile_result(fit; parm = blocks[1] => "z0")
    @test sd_profile.attempted == 1
    @test only(sd_profile.ci).param === blocks[1]
    @test only(sd_profile.ci).coef == "z0"

    # The public marginal simulator must draw the same correlated animal
    # coefficient field as the likelihood, with group-specific SDs.
    sim_rng = StableRNG(901)
    simulator = DRModels._marginal_simulator_build(fit, data; A)
    simulated = simulator(StableRNG(901))
    L = cholesky(Symmetric(A)).L
    z0, z1 = randn(sim_rng, G), randn(sim_rng, G)
    field0, field1 = DRModels._binomial_animal_q2_correlated_fields(
        cholesky(Symmetric(A)), false, z0, z1,
        tanh(only(coef(fit, :recov))))
    @test field0 ≈ L * z0
    @test field1 ≈ L * (tanh(only(coef(fit, :recov))) .* z0 .+
                        sqrt(1 - tanh(only(coef(fit, :recov)))^2) .* z1)
    eta_fixed = predict(fit, data; type = :link)
    sd0_group = fit.scales[Symbol(string(blocks[1]), "_group")]
    sd1_group = fit.scales[Symbol(string(blocks[2]), "_group")]
    eta_sim = DRModels._binomial_animal_q2_eta(
        eta_fixed, sd0_group, sd1_group, group_index, x, field0, field1)
    eta_oracle = [eta_fixed[i] + sd0_group[group_index[i]] * field0[group_index[i]] +
        x[i] * sd1_group[group_index[i]] * field1[group_index[i]]
        for i in eachindex(y)]
    @test eta_sim ≈ eta_oracle
    @test length(simulated) == length(y)
    @test all(in((0.0, 1.0)), simulated)
    Ainv = inv(A)
    precision_chol = cholesky(Symmetric(sparse(Ainv)))
    precision_transform = Matrix(precision_chol.UP \ Matrix{Float64}(I, G, G))
    @test precision_transform * precision_transform' ≈ A atol = 1e-10
    precision_field0, precision_field1 =
        DRModels._binomial_animal_q2_correlated_fields(
            precision_chol, true, z0, z1, tanh(only(coef(fit, :recov))))
    @test precision_field0 ≈ precision_transform * z0
    @test precision_field1 ≈ precision_transform * (
        tanh(only(coef(fit, :recov))) .* z0 .+
        sqrt(1 - tanh(only(coef(fit, :recov)))^2) .* z1)
    precision_simulator = DRModels._marginal_simulator_build(
        fit, data; Ainv)
    precision_simulated = precision_simulator(StableRNG(902))
    @test length(precision_simulated) == length(y)
    @test all(in((0.0, 1.0)), precision_simulated)
    boot = bootstrap_result(fit; data, A, B = 1,
        rng = StableRNG(903), failures = :skip)
    @test boot.attempted == 1
    @test boot.used == 1
    @test boot.failed == 0
    @test Set(row.param for row in boot.summary) ==
        Set((:mu, blocks..., :recov))

    # Independent dense Laplace oracle for Cov(b_g,b_h)=A[g,h]*D_g*R*D_h.
    theta = [-0.1, 0.25, alpha0..., alpha1..., atanh(rho)]
    R = [1.0 rho; rho 1.0]
    D = Diagonal(vec(permutedims(hcat(tau0, tau1))))
    Sigma = D * kron(A, R) * D
    Q = inv(Sigma)
    aux = (s = round.(Int, y), ntr = ones(Int, length(y)),
           logchoose = zeros(length(y)))
    eta_fixed = -0.1 .+ 0.25 .* x
    function dense_latent_nll(b)
        value = 0.5 * dot(b, Q * b)
        for i in eachindex(y)
            gi = group_index[i]
            value += DRModels._laplace_value(
                Val(:binomial), aux, i,
                eta_fixed[i] + b[2gi - 1] + x[i] * b[2gi],
            )
        end
        value
    end
    result = DRModels.Optim.optimize(
        dense_latent_nll, zeros(2G), DRModels.Optim.LBFGS(),
        DRModels.Optim.Options(g_tol = 1e-9, iterations = 400);
        autodiff = :forward,
    )
    b_mode = DRModels.Optim.minimizer(result)
    H = DRModels.ForwardDiff.hessian(dense_latent_nll, b_mode)
    dense_laplace = dense_latent_nll(b_mode) -
        0.5 * logdet(cholesky(Symmetric(Q))) +
        0.5 * logdet(cholesky(Symmetric(H)))
    @test isapprox(fit.nll(theta), dense_laplace; atol = 2e-5)
end

@testset "Non-logit Bernoulli animal q2 SD regression" begin
    formula = bf(
        @formula(y ~ x + animal(1 + x | p | group)),
        @formula(sd(group, animal, coef(intercept)) ~ z0),
        @formula(sd(group, animal, coef(x)) ~ z1),
    )
    for link in (:probit, :cloglog)
        @testset "$link link" begin
            rng = StableRNG(link === :probit ? 202610291 : 202610292)
            G, m = 80, 40
            group_index = repeat(1:G, inner = m)
            group = repeat(1:G, inner = m)
            z0_group = shuffle(rng, repeat([-1.0, 1.0], inner = G ÷ 2))
            z1_group = shuffle(rng, repeat([-1.0, 1.0], inner = G ÷ 2))
            z0, z1 = z0_group[group_index], z1_group[group_index]
            x = repeat(range(-0.8, 0.8; length = m), outer = G)
            A = [0.3^abs(i - j) for i in 1:G, j in 1:G]
            alpha0, alpha1, rho = [-0.8, 0.4], [-0.9, 0.4], 0.25
            sd0 = exp.(alpha0[1] .+ alpha0[2] .* z0_group)
            sd1 = exp.(alpha1[1] .+ alpha1[2] .* z1_group)
            L = cholesky(Symmetric(A)).L
            latent0 = L * randn(rng, G)
            latent1 = rho .* latent0 .+ sqrt(1 - rho^2) .* (L * randn(rng, G))
            eta = -0.1 .+ 0.15 .* x .+ sd0[group_index] .* latent0[group_index] .+
                  x .* sd1[group_index] .* latent1[group_index]
            probability = link === :probit ?
                DRModels.Distributions.cdf.(Ref(DRModels.Distributions.Normal()), eta) :
                -expm1.(-exp.(eta))
            y = Float64[rand(rng, DRModels.Distributions.Bernoulli(p))
                        for p in probability]
            data = (; y, x, z0, z1, group)
            fit = drm(formula, DRModels.Binomial(link = link);
                      data, A, se = false)

            blocks = map((:intercept, :x)) do member
                item = only(filter(item -> item.target.member === member,
                                   DRModels._sd_target_parts(formula)))
                DRModels._sd_target_block(item.target; disambiguate = true)
            end
            @test fit.converged
            @test abs(coef(fit, blocks[1])[2] - alpha0[2]) < 0.65
            @test abs(coef(fit, blocks[2])[2] - alpha1[2]) < 0.65
            @test abs(tanh(only(coef(fit, :recov))) - rho) < 0.55
            predicted = predict_parameters(fit, data; se = false)
            @test length(predicted[blocks[1]]) == length(y)
            @test length(predicted[blocks[2]]) == length(y)
            @test all(isfinite, predicted[blocks[1]])
            @test all(isfinite, predicted[blocks[2]])
            simulated = DRModels._marginal_simulator_build(fit, data; A)(
                StableRNG(202610293))
            @test length(simulated) == length(y)
            @test all(in((0.0, 1.0)), simulated)
        end
    end
end
