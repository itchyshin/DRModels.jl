# Beta-binomial family: successes out of known trials, with extra-binomial
# overdispersion. Two-column response `cbind(successes, failures) ~ x` (trials =
# successes + failures), exactly as drmTMB. Logit link on the mean success
# probability μ; the `sigma` slot is the overdispersion σ with precision
# φ = 1/σ² (likelihood BetaBinomial(n, μφ, (1-μ)φ)). Fixed effects, ML.
using DRModels
using Test, Random
import Distributions

@testset "Beta-binomial cbind(s, f) ~ x — recovery" begin
    Random.seed!(20260623)
    N = 3000
    x = randn(N)
    ntr = fill(25, N)                                   # 25 trials each
    β = [0.2, 0.7]; φ = 12.0                             # logit μ = 0.2 + 0.7x; precision φ
    μ = 1 ./ (1 .+ exp.(-(β[1] .+ β[2] .* x)))
    s = [rand(Distributions.BetaBinomial(ntr[i], μ[i] * φ, (1 - μ[i]) * φ)) for i in 1:N]
    fail = ntr .- s
    data = (; s = Float64.(s), fail = Float64.(fail), x)

    fit = drm(bf(@formula(cbind(s, fail) ~ x), @formula(sigma ~ 1)), BetaBinomial(); data = data)

    @test coef(fit, :mu)[1] ≈ β[1] atol = 0.08          # logit-mean intercept
    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.08          # logit-mean slope
    φ̂ = exp(-2 * coef(fit, :sigma)[1])                  # φ = 1/σ²
    @test φ̂ ≈ φ atol = 5.0                               # precision — weakly identified
    @test isfinite(loglik(fit))
    @test all(0 .< fitted(fit) .< 1)                    # fitted mean success probabilities
end
