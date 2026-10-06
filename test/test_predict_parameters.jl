# predict_parameters / marginal_parameters — the drmTMB-style per-distributional-
# parameter prediction surface (univariate slice). The correctness anchor is
# in-sample reproduction: predicting at the training data must reproduce the
# stored fitted parameters (fit.means[:mu], fit.scales[...]), with no R needed.
using DRModels
using Test, Random
import Distributions          # qualified — DRModels exports its own `Poisson` family

@testset "predict_parameters / marginal_parameters (univariate)" begin

    @testset "Gaussian location–scale: keys + in-sample reproduction" begin
        Random.seed!(20260603)
        n = 500
        x = randn(n)
        y = 0.5 .- 0.8 .* x .+ exp.(-0.3 .+ 0.4 .* x) .* randn(n)
        data = (; y, x)

        fit = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x)), Gaussian(); data = data)

        pp = predict_parameters(fit, data)
        @test Set(keys(pp)) == Set([:mu, :sigma])

        # In-sample reproduction of the stored fitted parameters (the anchor).
        @test pp[:mu]    ≈ fit.means[:mu]      rtol = 1e-8
        @test pp[:sigma] ≈ fit.scales[:sigma]  rtol = 1e-8

        # marginal_parameters = cheap accessor straight off the fit.
        mp = marginal_parameters(fit)
        @test Set(keys(mp)) == Set([:mu, :sigma])
        @test mp[:sigma] == fit.scales[:sigma]
        @test mp[:mu]    == fit.means[:mu]
        # response-scale prediction == marginal accessor in-sample.
        @test pp[:mu]    ≈ mp[:mu]    rtol = 1e-8
        @test pp[:sigma] ≈ mp[:sigma] rtol = 1e-8
    end

    @testset "link scale == predict link scale" begin
        Random.seed!(20260604)
        n = 400
        x = randn(n)
        y = 0.5 .- 0.8 .* x .+ exp.(-0.3 .+ 0.4 .* x) .* randn(n)
        data = (; y, x)

        fit = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x)), Gaussian(); data = data)

        ppl = predict_parameters(fit, data; type = :link)
        @test ppl[:mu] ≈ predict(fit, data; type = :link)  rtol = 1e-10
        # For Gaussian σ the link is log; exponentiating recovers the response scale.
        @test exp.(ppl[:sigma]) ≈ fit.scales[:sigma]  rtol = 1e-8

        @test_throws ErrorException predict_parameters(fit, data; type = :bogus)
    end

    @testset "Poisson: params-present set matches fit.blocks (only :mu)" begin
        Random.seed!(20260615)
        n = 3000
        x = randn(n)
        β = [0.3, 0.5]
        λ = exp.(β[1] .+ β[2] .* x)
        y = Float64.([rand(Distributions.Poisson(λi)) for λi in λ])
        data = (; y, x)

        fit = drm(bf(@formula(y ~ x)), Poisson(); data = data)

        pp = predict_parameters(fit, data)
        # Poisson carries no σ block; only :mu is a predictable parameter.
        @test Set(keys(pp)) == Set([:mu])
        @test pp[:mu] ≈ fit.means[:mu] rtol = 1e-8        # response (count) scale, exp link
        @test all(pp[:mu] .> 0)
        @test predict_parameters(fit, data; type = :link)[:mu] ≈ predict(fit, data; type = :link) rtol = 1e-10

        mp = marginal_parameters(fit)
        @test Set(keys(mp)) == Set([:mu])
        @test mp[:mu] == fit.means[:mu]
    end

    @testset "bivariate Gaussian: keys + in-sample reproduction (mu1/mu2/sigma1/sigma2/rho12)" begin
        Random.seed!(20260531)
        n = 1500
        x = randn(n)
        μ1 = 0.3 .+ 0.5 .* x; μ2 = -0.2 .+ 0.4 .* x
        σ1 = exp.(-0.1 .+ 0.2 .* x); σ2 = exp.(-0.3 .* x)
        ρ = tanh.(0.4 .+ 0.3 .* x)
        z1 = randn(n); z2 = randn(n)
        y1 = μ1 .+ σ1 .* z1
        y2 = μ2 .+ σ2 .* (ρ .* z1 .+ sqrt.(1 .- ρ .^ 2) .* z2)
        data = (; y1, y2, x)

        fit = drm(bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
                     sigma1 = @formula(sigma1 ~ x), sigma2 = @formula(sigma2 ~ x),
                     rho12 = @formula(rho12 ~ x)), Gaussian(); data = data)

        pp = predict_parameters(fit, data)
        # Keys == the parameters present in the bivariate formula.
        params = Set(first.(fit.formula.forms))
        @test params == Set([:mu1, :mu2, :sigma1, :sigma2, :rho12])
        @test Set(keys(pp)) == params

        # In-sample reproduction of the stored fitted parameters (the anchor):
        # mu1/mu2 from fit.means (identity link), sigma1/sigma2 (exp link) and
        # rho12 (tanh link) from fit.scales.
        @test pp[:mu1]    ≈ fit.means[:mu1]    rtol = 1e-8
        @test pp[:mu2]    ≈ fit.means[:mu2]    rtol = 1e-8
        @test pp[:sigma1] ≈ fit.scales[:sigma1] rtol = 1e-8
        @test pp[:sigma2] ≈ fit.scales[:sigma2] rtol = 1e-8
        @test pp[:rho12]  ≈ fit.scales[:rho12]  rtol = 1e-8

        # marginal_parameters = cheap accessor straight off the fit.
        mp = marginal_parameters(fit)
        @test Set(keys(mp)) == params
        @test mp[:mu1]    == fit.means[:mu1]
        @test mp[:mu2]    == fit.means[:mu2]
        @test mp[:sigma1] == fit.scales[:sigma1]
        @test mp[:sigma2] == fit.scales[:sigma2]
        @test mp[:rho12]  == fit.scales[:rho12]

        # link scale: mu link is identity for Gaussian, so == predict link means.
        ppl = predict_parameters(fit, data; type = :link)
        @test ppl[:mu1] ≈ predict(fit, data; type = :link)[:mu1] rtol = 1e-10
        @test ppl[:mu2] ≈ predict(fit, data; type = :link)[:mu2] rtol = 1e-10
        # exp / tanh of the working scale recovers the response-scale fits.
        @test exp.(ppl[:sigma1]) ≈ fit.scales[:sigma1] rtol = 1e-8
        @test exp.(ppl[:sigma2]) ≈ fit.scales[:sigma2] rtol = 1e-8
        @test DRModels.RHO_GUARD .* tanh.(ppl[:rho12]) ≈ fit.scales[:rho12]  rtol = 1e-8
    end
end
