# Post-fit `rho12(fit)` accessor: drmTMB-parity residual-correlation getter.
# Bivariate fit ⇒ per-observation ρ12 ∈ (-1, 1); univariate fit ⇒ ArgumentError.
using DRModels
using Test, Random

@testset "rho12(fit) bivariate residual-correlation accessor" begin
    Random.seed!(20260603)
    n = 2000
    x = randn(n)
    βμ1 = [0.3, 0.5]
    βμ2 = [-0.2, 0.4]
    βσ1 = [-0.1, 0.2]      # on log σ1
    βσ2 = [0.0, -0.3]      # on log σ2
    βρ = [0.4, 0.3]        # on atanh(ρ12)  (ρ = tanh(η))

    μ1 = βμ1[1] .+ βμ1[2] .* x
    μ2 = βμ2[1] .+ βμ2[2] .* x
    σ1 = exp.(βσ1[1] .+ βσ1[2] .* x)
    σ2 = exp.(βσ2[1] .+ βσ2[2] .* x)
    ρ = tanh.(βρ[1] .+ βρ[2] .* x)

    z1 = randn(n); z2 = randn(n)
    y1 = μ1 .+ σ1 .* z1
    y2 = μ2 .+ σ2 .* (ρ .* z1 .+ sqrt.(1 .- ρ .^ 2) .* z2)
    data = (; y1, y2, x)

    fit = drm(bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
                 sigma1 = @formula(sigma1 ~ x), sigma2 = @formula(sigma2 ~ x),
                 rho12 = @formula(rho12 ~ x)), Gaussian(); data = data)

    r = rho12(fit)
    @test length(r) == n                  # one value per observation (mirrors `sigma`)
    @test all(-1 .< r .< 1)               # response-scale residual correlation

    # Univariate fit has no residual correlation ⇒ accessor errors.
    yu = 0.5 .- 0.8 .* x .+ exp.(-0.3 .+ 0.4 .* x) .* randn(n)
    datu = (; y = yu, x)
    fit_uni = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x)), Gaussian(); data = datu)
    @test_throws ArgumentError rho12(fit_uni)
end
