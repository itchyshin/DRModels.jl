# Red-proof for sweep-oct16 t6 inverse-link bugs in `_param_response` /
# `_mean_response` / `_link_deriv` (gaussian_core.jl).
#
# Student ν = 2 + exp(η)  (fit.scales[:nu]; drmTMB student)
# SkewNormal ν = identity (fit.scales[:nu]; drmTMB skew_normal slant)
# ZeroOneBeta μ = logistic(η)  (fit.scales[:beta_mu]; drmTMB zero_one_beta)
#
# LogNormal μ (Julia exp(η) median vs drmTMB identity / log-Y μ) is a methods
# choice and is NOT asserted here — see the FX2 receipt.
using DRModels
using Test, Random
import Distributions

@testset "FX2 inverse links (Student ν / SkewNormal ν / ZeroOneBeta μ)" begin

    @testset "Student ν is 2+exp(η), not exp(η)" begin
        Random.seed!(20261016)
        n = 80
        x = randn(n)
        y = 0.3 .+ 0.4 .* x .+ 0.8 .* randn(n)
        data = (; y, x)
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)),
                  Student(); data)
        ppl = predict_parameters(fit, data; type = :link)
        pp = predict_parameters(fit, data)
        @test pp[:nu] ≈ DRModels._param_response(Student(), :nu, ppl[:nu])
        @test pp[:nu] ≈ fit.scales[:nu]
        # Kernel at small η (the in-sample MLE often sits on the ν→∞ ridge,
        # where Float64 cannot tell `exp(η)` from `2+exp(η)`).
        @test DRModels._param_response(Student(), :nu, [0.0]) ≈ [3.0]
        @test DRModels._param_response(Student(), :nu, [-1.0]) ≈ [2 + exp(-1.0)]
        @test DRModels._link_deriv(Student(), :nu, [0.0]) ≈ [1.0]   # d(2+exp)/dη = exp
    end

    @testset "SkewNormal ν is identity, not exp" begin
        Random.seed!(20261017)
        n = 200
        x = randn(n)
        y = 0.5 .+ 0.4 .* x .+ randn(n)
        data = (; y, x)
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)),
                  SkewNormal(); data)
        ppl = predict_parameters(fit, data; type = :link)
        pp = predict_parameters(fit, data)
        @test pp[:nu] ≈ ppl[:nu]
        @test pp[:nu] ≈ fit.scales[:nu]
        @test DRModels._param_response(SkewNormal(), :nu, [-0.5, 1.2]) ≈ [-0.5, 1.2]
        @test DRModels._link_deriv(SkewNormal(), :nu, [-0.5, 1.2]) ≈ [1.0, 1.0]
    end

    @testset "ZeroOneBeta μ is logistic(η), not identity" begin
        Random.seed!(20261018)
        n = 400
        x = randn(n)
        μ = 1 ./ (1 .+ exp.(-(0.1 .+ 0.5 .* x)))
        y = Vector{Float64}(undef, n)
        for i in 1:n
            y[i] = rand() < 0.15 ? (rand() < 0.4 ? 1.0 : 0.0) :
                   rand(Distributions.Beta(μ[i] * 8, (1 - μ[i]) * 8))
        end
        data = (; y, x)
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1),
                     @formula(zoi ~ 1), @formula(coi ~ 1)),
                  ZeroOneBeta(); data)
        ppl = predict_parameters(fit, data; type = :link)
        pp = predict_parameters(fit, data)
        @test pp[:mu] ≈ DRModels._logistic.(ppl[:mu])
        @test pp[:mu] ≈ fit.scales[:beta_mu]          # interior β mean, not mixture
        @test all(0 .< pp[:mu] .< 1)
        @test predict(fit, data) ≈ pp[:mu]
        @test DRModels._param_response(ZeroOneBeta(), :mu, [0.0]) ≈ [0.5]
        @test DRModels._mean_response(ZeroOneBeta(), [0.0]) ≈ [0.5]
        @test DRModels._link_deriv(ZeroOneBeta(), :mu, [0.0]) ≈ [0.25]
        # fitted() stays the unconditional mixture mean (drmTMB fitted)
        @test fitted(fit) ≈ (1 .- fit.scales[:zoi]) .* fit.scales[:beta_mu] .+
                            fit.scales[:zoi] .* fit.scales[:coi]
    end

    @testset "sibling families keep their existing inverse links" begin
        @test DRModels._param_response(Tweedie(), :nu, [0.0]) ≈ [1.5]  # 1+logistic
        @test DRModels._mean_response(Poisson(), [0.0]) ≈ [1.0]
        @test DRModels._mean_response(Gamma(), [0.0]) ≈ [1.0]
        @test DRModels._mean_response(Beta(), [0.0]) ≈ [0.5]
        @test DRModels._mean_response(Binomial(), [0.0]) ≈ [0.5]
        @test DRModels._mean_response(BetaBinomial(), [0.0]) ≈ [0.5]
        @test DRModels._mean_response(Gaussian(), [0.3]) ≈ [0.3]
        @test DRModels._mean_response(Student(), [0.3]) ≈ [0.3]
        @test DRModels._mean_response(SkewNormal(), [0.3]) ≈ [0.3]
        @test DRModels._param_response(Student(), :sigma, [0.0]) ≈ [1.0]
        @test DRModels._param_response(ZeroOneBeta(), :zoi, [0.0]) ≈ [0.5]
        @test DRModels._param_response(ZeroOneBeta(), :coi, [0.0]) ≈ [0.5]
    end
end
