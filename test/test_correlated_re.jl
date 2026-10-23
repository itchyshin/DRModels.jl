# Gaussian correlated random intercept+slope (1 + x | g): per-group (b0,b1) ~
# N(0, Σ). The marginal is Gaussian; because groups are disjoint the Woodbury
# capacitance is block-diagonal in 2×2 blocks, so the fit is O(G) closed form.
# Recovery test: residual σ and the RE covariance Σ (its SDs).
using DRModels
using Test, Random, LinearAlgebra

@testset "Gaussian correlated random slope (1+x|g) — recovery" begin
    Random.seed!(20260609)
    G = 150; m = 20; n = G * m
    g = repeat(1:G, inner = m); x = randn(n)
    β = [0.4, 0.6]; σ = 0.4
    sd_int = 0.7; sd_slp = 0.5; ρre = 0.3
    Σ = [sd_int^2 ρre*sd_int*sd_slp; ρre*sd_int*sd_slp sd_slp^2]
    B = cholesky(Symmetric(Σ)).L * randn(2, G)       # 2×G correlated (b0,b1) per group
    b0 = B[1, :]; b1 = B[2, :]
    y = β[1] .+ β[2] .* x .+ b0[g] .+ b1[g] .* x .+ σ .* randn(n)
    data = (; y, x, g)

    fit = drm(bf(@formula(y ~ x + (1 + x | g)), @formula(sigma ~ 1)), Gaussian(); data = data)

    @test exp(coef(fit, :sigma)[1]) ≈ σ atol = 0.08      # residual SD
    V = vc(fit)[:g]                                       # 2×2 RE covariance
    @test sqrt(V[1, 1]) ≈ sd_int atol = 0.2               # intercept-RE SD
    @test sqrt(V[2, 2]) ≈ sd_slp atol = 0.2               # slope-RE SD
    @test isfinite(loglik(fit))
end
