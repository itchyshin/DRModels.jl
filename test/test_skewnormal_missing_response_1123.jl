# #1123: SkewNormal() drops missing-response rows (observed-rows fit, with a
# warning) like every other univariate family, instead of throwing a bare
# LAPACK "matrix contains Infs or NaNs" from the Hessian.
using Test, DRModels, Random

@testset "SkewNormal missing response (#1123)" begin
    rng = MersenneTwister(1123)
    n = 150
    x = randn(rng, n)
    y = 1.0 .+ 0.5 .* x .+ exp(-0.2) .* abs.(randn(rng, n)) .* 1.0 .- 0.5 .+ 0.3 .* randn(rng, n)
    complete = (y = y, x = x)
    ym = Vector{Union{Missing,Float64}}(y)
    ym[1] = missing
    ym[7] = NaN
    dmiss = (y = ym, x = x)
    f = bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1))

    fit = @test_logs (:warn, r"missing/NaN response") drm(f, SkewNormal(); data = dmiss)
    @test fit.nobs == n - 2

    keep = setdiff(1:n, [1, 7])
    ref = drm(f, SkewNormal(); data = (y = y[keep], x = x[keep]))
    @test fit.loglik ≈ ref.loglik atol = 1e-6
    @test coef(fit, :mu) ≈ coef(ref, :mu) atol = 1e-5
    @test all(isfinite, coef(fit, :nu))
    # full-row rebuild stores the slant α (identity link), not exp(α) (#1120)
    @test length(fit.scales[:nu]) == n
    @test all(≈(coef(fit, :nu)[1]), fit.scales[:nu])
end
