# Issue #1533: `cbind(y, trials - y)` must not leak `FunctionTerm has no field sym`.
using DRModels
using Random
using Distributions
using Test

@testset "Binomial cbind arithmetic failures column (#1533)" begin
    Random.seed!(20)
    n = 45
    x = randn(n)
    trials = fill(10, n)
    p = (@. 1 / (1 + exp(-(0.1 + 0.5 * x))))
    y = Float64[rand(Distributions.Binomial(t, pr)) for (t, pr) in zip(trials, p)]
    fail = Float64.(trials .- y)
    d = (; y, succ = y, fail, x, trials = Float64.(trials))

  # Precomputed failures column (control).
    fit_ok = drm(bf(@formula(cbind(succ, fail) ~ x)), DRModels.Binomial(); data = d)
    fit_expr = drm(bf(@formula(cbind(y, trials - y) ~ x)), DRModels.Binomial(); data = d)

    @test coef(fit_expr) ≈ coef(fit_ok) atol = 1e-10
    @test loglik(fit_expr) ≈ loglik(fit_ok) atol = 1e-10
    @test nobs(fit_expr) == nobs(fit_ok) == n
end