# Gaussian meta-analysis with known sampling variances: y_i ~ N(x_iᵀβ, v_i + τ²),
# where v_i are supplied (known) and τ² (residual heterogeneity, the σ intercept)
# is estimated. Marker: meta_V(v). Recovery test (Fisher/Curie).
using DRModels
using Test, Random

@testset "Gaussian meta-analysis: meta_V(v) — recovery" begin
    Random.seed!(20260603)
    k = 500                                   # studies
    x = randn(k)
    β = [0.2, 0.5]
    τ = 0.4                                    # between-study SD (heterogeneity)
    v = (0.2 .+ 0.6 .* rand(k)) .^ 2          # known sampling variances
    y = β[1] .+ β[2] .* x .+ τ .* randn(k) .+ sqrt.(v) .* randn(k)
    data = (; y, x, v)

    fit = drm(bf(@formula(y ~ x + meta_V(v)), @formula(sigma ~ 1)), Gaussian(); data = data)

    @test coef(fit, :mu) ≈ β atol = 0.12
    @test exp(coef(fit, :sigma)[1]) ≈ τ atol = 0.15      # τ̂ on the σ intercept
    @test isfinite(loglik(fit))
end
