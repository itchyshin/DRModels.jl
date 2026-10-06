# Plain Binomial / Bernoulli family: the classic logistic regression. Logit link
# on the mean success probability μ; no dispersion parameter (mean-only, like
# Poisson). Two response forms: cbind(successes, failures) ~ x (trials =
# successes + failures, exactly as drmTMB) and a plain 0/1 Bernoulli vector.
# Fixed effects, ML. Recovery: the logit-scale coefficients β.
using DRModels
using Test, Random
import Distributions

@testset "Binomial cbind(s, f) ~ x — logit β recovery" begin
    Random.seed!(20260531)
    N = 4000
    x = randn(N)
    ntr = fill(20, N)                                   # 20 trials each
    β = [-0.4, 0.9]                                     # logit μ = -0.4 + 0.9x
    μ = 1 ./ (1 .+ exp.(-(β[1] .+ β[2] .* x)))
    s = [rand(Distributions.Binomial(ntr[i], μ[i])) for i in 1:N]
    fail = ntr .- s
    data = (; s = Float64.(s), fail = Float64.(fail), x)

    fit = drm(bf(@formula(cbind(s, fail) ~ x)), Binomial(); data = data)

    @test coef(fit, :mu)[1] ≈ β[1] atol = 0.08          # logit-mean intercept
    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.08          # logit-mean slope
    @test isfinite(loglik(fit))
    @test fit.converged
    @test all(0 .< fitted(fit) .< 1)                    # fitted mean success probabilities
end

@testset "Binomial 0/1 Bernoulli y ~ x — logit β recovery" begin
    Random.seed!(424242)
    N = 6000
    x = randn(N)
    β = [0.25, -0.7]                                    # logit μ = 0.25 - 0.7x
    μ = 1 ./ (1 .+ exp.(-(β[1] .+ β[2] .* x)))
    y = Float64.([rand() < μ[i] ? 1 : 0 for i in 1:N])
    data = (; y, x)

    fit = drm(bf(@formula(y ~ x)), Binomial(); data = data)

    @test coef(fit, :mu)[1] ≈ β[1] atol = 0.1           # logit-mean intercept
    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.1           # logit-mean slope
    @test isfinite(loglik(fit))
    @test fit.converged
    @test all(0 .< fitted(fit) .< 1)
    # mean-only: a sigma/dispersion formula must be rejected
    @test_throws ErrorException drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Binomial(); data = data)
end
