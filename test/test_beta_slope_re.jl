# Beta correlated random intercept+slope (1 + x | g) on the logit mean: a
# proportion GLMM where per-group (b0,b1) ~ N(0, Σ). logit μ_i = Xμ_iᵀβ +
# b0_{g(i)} + b1_{g(i)}·x_i, precision φ = 1/σ². The 2-D group effect is
# integrated out per group by a K×K Gauss–Hermite product rule. Recovery: the
# logit-mean slope and the intercept/slope RE SDs (sqrt diag of vc(fit)[:g]).
using DRModels
using Test, Random, LinearAlgebra
import Distributions

@testset "Beta correlated random slope (1+x|g) — recovery" begin
    Random.seed!(20260703)
    G = 120; m = 25; n = G * m
    g = repeat(1:G, inner = m); x = randn(n)
    β = [0.2, 0.5]; φ = 15.0
    sd0 = 0.4; sd1 = 0.3; ρ = 0.3
    Σ = [sd0^2 ρ*sd0*sd1; ρ*sd0*sd1 sd1^2]
    B = cholesky(Symmetric(Σ)).L * randn(2, G)      # 2×G correlated (b0,b1) per group
    b0 = B[1, :]; b1 = B[2, :]
    μ = 1 ./ (1 .+ exp.(-(β[1] .+ β[2] .* x .+ b0[g] .+ b1[g] .* x)))
    y = Float64.([rand(Distributions.Beta(μ[i] * φ, (1 - μ[i]) * φ)) for i in 1:n])

    fit = drm(bf(@formula(y ~ x + (1 + x | g)), @formula(sigma ~ 1)), Beta(); data = (; y, x, g))

    @test coef(fit, :mu)[2] ≈ 0.5 atol = 0.15       # logit-mean slope
    V = vc(fit)[:g]                                  # 2×2 RE covariance
    @test sqrt(V[1, 1]) ≈ sd0 atol = 0.2             # intercept-RE SD
    @test sqrt(V[2, 2]) ≈ sd1 atol = 0.2             # slope-RE SD
    @test isfinite(loglik(fit))
end
