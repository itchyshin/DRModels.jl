# Cumulative-logit ordinal regression: ordered categorical response y ∈ {1,…,K}.
# Pr(y ≤ k) = logistic(θ_k − η), with ordered cutpoints θ_1 < … < θ_{K-1} and a
# single linear predictor η (the location intercept is dropped — cutpoints absorb
# it). One parameter `mu`, no `sigma`. Mirrors drmTMB's `cumulative_logit`.
using DRModels
using Test, Random

@testset "Cumulative logit (ordinal) — recovery" begin
    Random.seed!(20260626)
    n = 4000; x = randn(n)
    βslope = 0.8; θtrue = [-1.0, 0.0, 1.2]; K = 4
    η = βslope .* x
    y = Vector{Int}(undef, n)
    for i in 1:n
        u = rand(); yi = K
        for k in 1:(K-1)
            if u < 1 / (1 + exp(-(θtrue[k] - η[i])))   # u < P(y ≤ k)
                yi = k; break
            end
        end
        y[i] = yi
    end
    data = (; y = Float64.(y), x)

    fit = drm(bf(@formula(y ~ x)), CumulativeLogit(); data = data)

    @test coef(fit, :mu)[1] ≈ βslope atol = 0.10        # slope (intercept dropped)
    δ = coef(fit, :cutpoints)                           # raw (ordered-increment) params
    θ̂ = similar(δ); θ̂[1] = δ[1]
    for k in 2:length(δ); θ̂[k] = θ̂[k-1] + exp(δ[k]); end
    @test θ̂ ≈ θtrue atol = 0.2                          # reconstructed cutpoints
    @test isfinite(loglik(fit))
    @test all(1 .<= fitted(fit) .<= K)                  # expected ordered-category score
end
