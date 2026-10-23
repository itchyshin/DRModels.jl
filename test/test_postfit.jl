# Post-fit accessors: fitted values and residuals, for univariate and bivariate
# Gaussian models. (Base-only assertions so no extra test deps are needed.)
using DRModels
using Test, Random

@testset "Post-fit: fitted + residuals" begin
    Random.seed!(7)
    n = 800
    x = randn(n)
    y = 1.0 .+ 0.8 .* x .+ 0.5 .* randn(n)

    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
    ŷ = fitted(fit)
    @test length(ŷ) == n
    @test residuals(fit) ≈ y .- ŷ
    @test abs(sum(residuals(fit))) < 1.0        # intercept ⇒ residuals sum ≈ 0
    @test maximum(ŷ) - minimum(ŷ) > 0.5         # fitted varies with x

    # bivariate: fitted is keyed per response
    y2 = -0.5 .+ 0.3 .* x .+ 0.4 .* randn(n)
    bvfit = drm(bf(mu1 = @formula(y ~ x), mu2 = @formula(y2 ~ x),
                   sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                   rho12 = @formula(rho12 ~ 1)), Gaussian(); data = (; y, y2, x))
    fm = fitted(bvfit)
    @test haskey(fm, :mu1) && haskey(fm, :mu2)
    @test length(fm[:mu1]) == n
    rm = residuals(bvfit)
    @test rm[:mu1] ≈ y .- fm[:mu1]
end
