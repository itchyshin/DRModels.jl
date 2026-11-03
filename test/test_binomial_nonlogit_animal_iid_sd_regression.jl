using DRModels, Test, Random, LinearAlgebra
import Distributions

@testset "Non-logit Binomial animal plus IID SD regression" begin
    for link in (:probit, :cloglog)
        rng = MersenneTwister(link === :probit ? 20261027 : 20261028)
        G, H, m = 96, 48, 4
        animal_index = repeat(1:G, inner = H * m)
        site_index = repeat(repeat(1:H, inner = m), outer = G)
        animal_id = ["animal_$(i)" for i in animal_index]
        site = ["site_$(i)" for i in site_index]
        zlevels = collect(range(-1.0, 1.0; length = G))
        wlevels = collect(range(-1.0, 1.0; length = H))
        z, w = zlevels[animal_index], wlevels[site_index]
        x = randn(rng, length(animal_index))
        correlation = [0.35^abs(i - j) for i in 1:G, j in 1:G]
        A = correlation
        animal_u = cholesky(Symmetric(A)).L * randn(rng, G)
        iid_u = randn(rng, H)
        alpha_animal = [-0.65, 0.75]
        alpha_iid = [-0.35, -0.65]
        eta = -0.2 .+ 0.25 .* x .+
            exp.(alpha_animal[1] .+ alpha_animal[2] .* zlevels)[animal_index] .* animal_u[animal_index] .+
            exp.(alpha_iid[1] .+ alpha_iid[2] .* wlevels)[site_index] .* iid_u[site_index]
        probability = link === :probit ?
            Distributions.cdf.(Ref(Distributions.Normal()), eta) :
            -expm1.(-exp.(eta))
        y = Float64[rand(rng, Distributions.Bernoulli(clamp(p, eps(), 1 - eps())))
                    for p in probability]
        data = (; y, x, animal_id, site, z, w)
        formula = bf(
            @formula(y ~ x + animal(1 | animal_id) + (1 | site)),
            @formula(sd(animal_id, animal) ~ z),
            @formula(sd(site) ~ w),
        )

        fit = drm(formula, Binomial(link = link); data, A, se = false)
        precision_fit = drm(formula, Binomial(link = link);
                            data, Ainv = inv(Symmetric(A)), se = false)
        blocks = DRModels._sd_target_blocks(DRModels._sd_target_parts(formula))
        animal_block, iid_block = blocks

        @testset "$link link" begin
            @test fit.converged
            @test precision_fit.converged
            @test precision_fit.loglik ≈ fit.loglik atol = 2e-5
            @test coef(precision_fit, animal_block) ≈ coef(fit, animal_block) atol = 3e-3
            @test coef(precision_fit, iid_block) ≈ coef(fit, iid_block) atol = 3e-3
            @test sign(coef(fit, animal_block)[2]) == sign(alpha_animal[2])
            @test sign(coef(fit, iid_block)[2]) == sign(alpha_iid[2])
            @test abs(coef(fit, animal_block)[2] - alpha_animal[2]) < 0.55
            @test abs(coef(fit, iid_block)[2] - alpha_iid[2]) < 0.55
            predicted = predict_parameters(fit, data; se = false)
            @test all(isfinite, predicted[animal_block])
            @test all(isfinite, predicted[iid_block])
            @test length(predicted[animal_block]) == length(y)
            @test length(predicted[iid_block]) == length(y)
            simulated = simulate(fit; re = :zero, rng = MersenneTwister(20261029))
            @test length(simulated) == length(y)
            @test all(in((0.0, 1.0)), simulated)
        end
    end
end

function _binomial_nonlogit_animal_slope_iid_fixture(link; seed)
    rng = MersenneTwister(seed)
    G, H, m = 192, 96, 24
    animal_index = repeat(1:G, inner = m)
    sites_by_animal = shuffle!(rng, repeat(1:H, inner = G ÷ H))
    site_index = repeat(sites_by_animal, inner = m)
    animal_id = ["animal_$(i)" for i in animal_index]
    site = ["site_$(i)" for i in site_index]
    zlevels = collect(range(-1.0, 1.0; length = G))
    wlevels = collect(range(-0.8, 0.8; length = H))
    z, w = zlevels[animal_index], wlevels[site_index]
    x = randn(rng, length(animal_index))
    A = [0.2^abs(i - j) for i in 1:G, j in 1:G]
    animal_u = cholesky(Symmetric(A)).L * randn(rng, G)
    iid_u = randn(rng, H)
    alpha_animal = [-0.7, 0.75]
    alpha_iid = [-0.8, -0.65]
    eta = -0.2 .+ 0.3 .* x .+
        x .* (exp.(alpha_animal[1] .+ alpha_animal[2] .* zlevels) .* animal_u)[animal_index] .+
        exp.(alpha_iid[1] .+ alpha_iid[2] .* wlevels)[site_index] .* iid_u[site_index]
    probability = link === :probit ?
        Distributions.cdf.(Ref(Distributions.Normal()), eta) :
        -expm1.(-exp.(eta))
    y = Float64[rand(rng, Distributions.Bernoulli(clamp(p, eps(), 1 - eps())))
                for p in probability]
    return (; data = (; y, x, z, w, g = animal_id, site), A,
            alpha_animal, alpha_iid)
end

@testset "Non-logit Binomial animal-slope plus IID SD regression" begin
    for link in (:probit, :cloglog)
        fixture = _binomial_nonlogit_animal_slope_iid_fixture(link;
            seed = link === :probit ? 20261101 : 20261102)
        formula = bf(
            @formula(y ~ x + animal(0 + x | g) + (1 | site)),
            @formula(sd(g, animal) ~ z),
            @formula(sd(site) ~ w),
        )
        fit = drm(formula, Binomial(link = link);
            data = fixture.data, A = fixture.A, se = false)
        targets = DRModels._sd_target_parts(formula)
        animal_block, iid_block = DRModels._sd_target_blocks(targets)

        @testset "$link link" begin
            @test fit.converged
            @test isfinite(fit.loglik)
            @test coef(fit, animal_block)[2] > 0
            @test coef(fit, iid_block)[2] < 0
            @test abs(coef(fit, animal_block)[2] - fixture.alpha_animal[2]) < 0.55
            @test abs(coef(fit, iid_block)[2] - fixture.alpha_iid[2]) < 0.55
            predicted = predict_parameters(fit, fixture.data; se = false)
            @test length(predicted[animal_block]) == length(fixture.data.y)
            @test length(predicted[iid_block]) == length(fixture.data.y)
            @test all(isfinite, predicted[animal_block])
            @test all(isfinite, predicted[iid_block])
            @test predicted[animal_block] ≈ fit.scales[animal_block]
            @test predicted[iid_block] ≈ fit.scales[iid_block]
            simulated = simulate(fit; re = :zero, rng = MersenneTwister(20261103))
            @test length(simulated) == length(fixture.data.y)
            @test all(in((0.0, 1.0)), simulated)
        end
    end
end
