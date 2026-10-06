# Information criteria for model selection: dof / aic / bic.
using DRModels, Test, Random
import Distributions

@testset "AIC / BIC / dof" begin
    Random.seed!(20260905)
    n = 200; x = randn(n)
    y = 1.0 .+ 0.5 .* x .+ 0.3 .* randn(n)
    f = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
    k = dof(f)
    @test k == length(coef(f))                       # β(2) + log σ(1) = 3 params
    @test aic(f) ≈ -2 * loglik(f) + 2k
    @test bic(f) ≈ -2 * loglik(f) + k * log(nobs(f))
    @test bic(f) > aic(f)                             # n > e² ⇒ BIC penalty heavier

    # a larger model (sigma ~ x) consumes one more dof
    f2 = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian(); data = (; y, x))
    @test dof(f2) == dof(f) + 1

    # works for a non-Gaussian fit too
    yp = Float64[rand(Distributions.Poisson(exp(0.3 + 0.5x[i]))) for i in 1:n]
    fp = drm(bf(@formula(y ~ x)), Poisson(); data = (; y = yp, x))
    @test dof(fp) == 2
    @test isfinite(aic(fp)) && isfinite(bic(fp))
end
