# refs #1381 — `predict(fit)` without newdata matches drmTMB `predict(newdata = NULL)`.
module TestIssue1381PredictNoData

using DRModels
using Test
using Random
using LinearAlgebra
import Distributions

function _check_predict_insample(fit, data; rtol = 1e-8)
    @test predict(fit) ≈ predict(fit, data; type = :response) rtol = rtol
    @test predict(fit; type = :response) ≈ predict(fit, data; type = :response) rtol = rtol
    @test predict(fit; type = :link) ≈ predict(fit, data; type = :link) rtol = rtol
    @test predict(fit; type = :mean) ≈ predict(fit, data; type = :mean) rtol = rtol
    @test predict(fit; type = :mean) ≈ fitted(fit) rtol = rtol
end

function _check_predict_parameters_insample(fit, data; rtol = 1e-8)
    a = predict_parameters(fit)
    b = predict_parameters(fit, data)
    @test Set(keys(a)) == Set(keys(b))
    for k in keys(a)
        @test a[k] ≈ b[k] rtol = rtol
    end
    al = predict_parameters(fit; type = :link)
    bl = predict_parameters(fit, data; type = :link)
    @test Set(keys(al)) == Set(keys(bl))
    for k in keys(al)
        @test al[k] ≈ bl[k] rtol = rtol
    end
end

@testset "#1381 predict(fit) without newdata" begin
    @testset "Gaussian location–scale" begin
        N = 40
        i = 1:N
        x = sin.(i)
        y = 1.0 .+ 0.5 .* x .+ 0.3 .* sin.(5.1 .* i)
        data = (; y, x)
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data)
        _check_predict_insample(fit, data)
        _check_predict_parameters_insample(fit, data)
    end

    @testset "Poisson" begin
        Random.seed!(1381)
        n = 120
        x = randn(n)
        y = Float64[rand(Distributions.Poisson(exp(0.2 + 0.5x[i]))) for i in 1:n]
        data = (; y, x)
        fit = drm(bf(@formula(y ~ x)), Poisson(); data)
        _check_predict_insample(fit, data)
        _check_predict_parameters_insample(fit, data)
    end

    @testset "Binomial" begin
        Random.seed!(1382)
        n = 80
        x = randn(n)
        p = 1 ./ (1 .+ exp.(-(0.1 .+ 0.7 .* x)))
        y = Float64[rand() < p[i] ? 1.0 : 0.0 for i in 1:n]
        data = (; y, x)
        fit = drm(bf(@formula(y ~ x)), Binomial(); data)
        _check_predict_insample(fit, data)
        _check_predict_parameters_insample(fit, data)
    end

    @testset "Tweedie" begin
        Random.seed!(1383)
        n = 60
        x = randn(n)
        y = exp.(0.15 .+ 0.25 .* x .+ 0.2 .* randn(n))
        y = max.(y, 1e-4)
        data = (; y, x)
        fit = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1), @formula(nu ~ 1)),
                  Tweedie(); data)
        _check_predict_insample(fit, data; rtol = 1e-6)
        _check_predict_parameters_insample(fit, data; rtol = 1e-6)
    end

    @testset "LogNormal" begin
        Random.seed!(1384)
        n = 100
        x = randn(n)
        y = exp.(0.3 .+ 0.4 .* x .+ 0.25 .* randn(n))
        data = (; y, x)
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), LogNormal(); data)
        _check_predict_insample(fit, data; rtol = 1e-7)
        _check_predict_parameters_insample(fit, data; rtol = 1e-7)
    end

    @testset "ZIP (zero-inflated Poisson)" begin
        Random.seed!(1385)
        n = 200
        x = randn(n)
        π = 0.25
        λ = exp.(0.15 .+ 0.35 .* x)
        y = Float64[
            rand() < π ? 0.0 : rand(Distributions.Poisson(λ[i])) for i in 1:n]
        data = (; y, x)
        fit = drm(bf(@formula(y ~ x), @formula(zi ~ 1)), Poisson(); data)
        _check_predict_insample(fit, data; rtol = 1e-7)
        _check_predict_parameters_insample(fit, data; rtol = 1e-7)
    end

    @testset "ZeroOneBeta" begin
        Random.seed!(1386)
        n = 300
        x = randn(n)
        μ = 1 ./ (1 .+ exp.(-(0.1 .+ 0.5 .* x)))
        y = clamp.(μ .+ 0.08 .* randn(n), 0.0, 1.0)
        data = (; y, x)
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(zoi ~ 1),
                        @formula(coi ~ 1)),
                  ZeroOneBeta(); data)
        _check_predict_insample(fit, data; rtol = 1e-6)
        _check_predict_parameters_insample(fit, data; rtol = 1e-6)
    end

    @testset "JointFiniteDrmFit" begin
        ctl = miss_control(response = "include", predictor = "model")
        data = (; y = [0.5, 1.2, missing, 0.8, 2.1],
                x = ["low", "medium", missing, "high", "low"],
                z = [0.1, -0.2, 0.0, 0.4, -0.1])
        levels = ["low", "medium", "high"]
        form = bf(@formula(y ~ z + mi(x)), @formula(sigma ~ 1))
        specification = (x = impute_model(@formula(x ~ z);
                                         family = CumulativeLogit(), levels = levels),)
        jfit = drm(form, Gaussian(); data = data, impute = specification,
                   missing = ctl, g_tol = 1e-8)
        @test jfit isa JointFiniteDrmFit
        @test length(predict(jfit)) == nobs(jfit)
        @test predict(jfit) ≈ fitted(jfit)
        @test predict(jfit; type = :link) ≈ predict(jfit; type = :response)
        obs = findall(i -> !ismissing(data.x[i]) && !ismissing(data.y[i]), eachindex(data.y))
        newdata = (; z = data.z[obs], x = String.(data.x[obs]))
        @test predict(jfit, newdata) ≈ predict(jfit) rtol = 1e-12
        @test predict(jfit, newdata; type = :link) ≈ predict(jfit; type = :link) rtol = 1e-12
    end

    @testset "one-coefficient non-intercept block is not a constant" begin
        Random.seed!(1387)
        n = 60
        x = rand(n) .+ 0.5
        y = 0.4 .+ 0.3 .* x .+ exp.(-0.6 .* x) .* randn(n)
        data = (; y, x)
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 0 + x)), Gaussian(); data)
        βσ = only(coef(fit, :sigma))
        @test predict_parameters(fit; type = :link)[:sigma] ≈ x .* βσ rtol = 1e-8
        @test predict_parameters(fit)[:sigma] ≈ exp.(x .* βσ) rtol = 1e-8
        _check_predict_parameters_insample(fit, data)
    end

    @testset "structured Tweedie SD regression: retained modes and per-row SD" begin
        rng = MersenneTwister(13881)
        G, m = 16, 8
        gi = repeat(1:G; inner = m)
        group = ["g$(k)" for k in gi]
        zg = repeat([0.0, 1.0], G ÷ 2)
        K = 0.45 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
        x = rand(rng, G * m) .* 2 .- 1
        latent = cholesky(Symmetric(K)).L * randn(rng, G)
        eta = 0.35 .+ 0.2 .* x .+ exp.(-0.3 .+ 0.4 .* zg)[gi] .* latent[gi]
        y = map(eachindex(eta)) do i
            μ, ϕ, pw = exp(eta[i]), 0.65^2, 1.5
            c = rand(rng, Distributions.Poisson(μ^(2 - pw) / (ϕ * (2 - pw))))
            c == 0 ? 0.0 : rand(rng, Distributions.Gamma(c * (2 - pw) / (pw - 1), ϕ * (pw - 1) * μ^(pw - 1)))
        end
        data = (; y, x, z = zg[gi], group)
        f = bf(@formula(y ~ x + relmat(1 | group)), @formula(sigma ~ 1), @formula(nu ~ 1),
               @formula(sd(group, relmat) ~ z))
        fit = drm(f, Tweedie(); data, K = K, g_tol = 1e-5)
        b = only(values(ranef(fit)))
        gidx = Int.(fit.scales[:sd_group_index])
        @test maximum(abs, b) > 0.05                         # the modes are not all zero
        ηpop = predict(fit, data; type = :link)
        @test predict(fit; type = :link) ≈ ηpop .+ b[gidx] rtol = 1e-8
        @test predict(fit) ≈ exp.(ηpop .+ b[gidx]) rtol = 1e-8
        pp = predict_parameters(fit)
        @test haskey(pp, :sd)
        @test pp[:sd] ≈ fit.scales[:sd] rtol = 1e-12
        @test pp[:sd] ≈ predict_parameters(fit, data)[:sd] rtol = 1e-8
        @test predict_parameters(fit; type = :link)[:sd] ≈ log.(fit.scales[:sd]) rtol = 1e-12
        # without the row map the modes cannot be placed: refuse by name, not population values
        bare = deepcopy(fit)
        delete!(bare.scales, :sd_group_index)
        @test_throws ArgumentError predict(bare)
    end

    @testset "refusals" begin
        N = 20
        x = randn(N)
        y = x .+ 0.1 .* randn(N)
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
        @test_throws ArgumentError predict(fit; se = true)
        @test_throws ArgumentError predict_parameters(fit; se = true)
    end
end

end