# predict() should return response-scale means for non-Gaussian families (apply
# the inverse link); type = :link gives the linear predictor. In-sample predict
# must equal fitted().
using DRModels, Test, Random
import Distributions

@testset "predict response scale (non-Gaussian inverse link)" begin
    Random.seed!(20260901)
    n = 250; x = randn(n)

    # Poisson (log link): response = exp(Xβ) = λ ; link = Xβ
    y = Float64[rand(Distributions.Poisson(exp(0.3 + 0.6x[i]))) for i in 1:n]
    fp = drm(bf(@formula(y ~ x)), Poisson(); data = (; y, x))
    @test predict(fp, (; y, x)) ≈ fitted(fp) rtol = 1e-8
    @test all(predict(fp, (; y, x)) .> 0)
    @test predict(fp, (; y, x); type = :link) ≈ log.(fitted(fp)) rtol = 1e-8

    # Beta (logit link): response in (0,1)
    Random.seed!(20260902)
    μb = 1 ./ (1 .+ exp.(-(0.2 .+ 0.5 .* x))); φ = 12.0
    yb = Float64[rand(Distributions.Beta(μb[i] * φ, (1 - μb[i]) * φ)) for i in 1:n]
    fb = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Beta(); data = (; y = yb, x))
    pr = predict(fb, (; y = yb, x))
    @test all(0 .< pr .< 1)
    @test pr ≈ fitted(fb) rtol = 1e-8

    # Gaussian unchanged (identity): response == link == fitted
    yg = 1.0 .+ 0.5 .* x .+ 0.3 .* randn(n)
    fg = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; y = yg, x))
    @test predict(fg, (; y = yg, x)) ≈ fitted(fg) rtol = 1e-8
    @test predict(fg, (; y = yg, x); type = :link) ≈ fitted(fg) rtol = 1e-8
end
