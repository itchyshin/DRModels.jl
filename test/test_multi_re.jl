# Multiple (independent, scalar) random-effect terms in one formula, e.g. two
# crossed random intercepts (1|g) + (1|h). Marginal V = D + Σ_k σ_k² Z_k Z_kᵀ;
# the combined capacitance is dense (components share observations) but small, so
# a closed-form GLS via a (Σ G_k)×(Σ G_k) Cholesky. Recovery: residual σ + each
# component's SD.
using DRModels
using Test, Random

@testset "Gaussian crossed random intercepts (1|g)+(1|h) — recovery" begin
    Random.seed!(20260610)
    G = 25; H = 20; n = 800
    g = rand(1:G, n); h = rand(1:H, n); x = randn(n)
    β = [0.3, 0.5]; σ = 0.4; σg = 0.7; σh = 0.5
    bg = σg .* randn(G); bh = σh .* randn(H)
    y = β[1] .+ β[2] .* x .+ bg[g] .+ bh[h] .+ σ .* randn(n)
    data = (; y, x, g, h)

    fit = drm(bf(@formula(y ~ x + (1 | g) + (1 | h)), @formula(sigma ~ 1)), Gaussian(); data = data)

    @test exp(coef(fit, :sigma)[1]) ≈ σ atol = 0.1     # residual SD
    rs = re_sd(fit)
    @test rs[:g] ≈ σg atol = 0.25
    @test rs[:h] ≈ σh atol = 0.25
    @test isfinite(loglik(fit))
end
