# Zero-one-inflated beta: responses on the CLOSED interval [0,1] — proportions
# that can also be exactly 0 or 1. A mixture: with prob `zoi` the value is a
# boundary (0 or 1; `coi` = P(1 | boundary)); otherwise a Beta(μ, φ) on (0,1).
# Parameters mu (logit) / sigma (log, φ=1/σ²) / zoi (logit) / coi (logit), exactly
# as drmTMB's `zero_one_beta`. Fixed effects, ML.
using DRModels
using Test, Random
import Distributions

@testset "Zero-one-inflated beta — recovery" begin
    Random.seed!(20260624)
    n = 6000; x = randn(n)
    β = [0.2, 0.6]; φ = 12.0; zoi = 0.25; coi = 0.4
    μ = 1 ./ (1 .+ exp.(-(β[1] .+ β[2] .* x)))
    y = Vector{Float64}(undef, n)
    for i in 1:n
        if rand() < zoi
            y[i] = rand() < coi ? 1.0 : 0.0
        else
            y[i] = rand(Distributions.Beta(μ[i] * φ, (1 - μ[i]) * φ))
        end
    end
    data = (; y, x)

    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(zoi ~ 1), @formula(coi ~ 1)),
        ZeroOneBeta(); data = data)

    @test coef(fit, :mu)[1] ≈ β[1] atol = 0.10                      # logit beta-mean intercept
    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.10                      # logit beta-mean slope
    @test 1 / (1 + exp(-coef(fit, :zoi)[1])) ≈ zoi atol = 0.05      # boundary probability
    @test 1 / (1 + exp(-coef(fit, :coi)[1])) ≈ coi atol = 0.08      # P(1 | boundary)
    @test exp(-2 * coef(fit, :sigma)[1]) ≈ φ atol = 5.0             # precision φ = 1/σ²
    @test isfinite(loglik(fit))
end
