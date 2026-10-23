# Post-fit drmTMB-parity accessors: fit a tiny Gaussian location–scale model and
# assert `is_converged` / `deviance` / `dof_residual` match their definitions.
using DRModels
using Test, Random

@testset "parity accessors: is_converged / deviance / dof_residual" begin
    Random.seed!(20260603)
    n = 500
    x = randn(n)
    y = 0.5 .- 0.8 .* x .+ exp.(-0.3 .+ 0.4 .* x) .* randn(n)
    data = (; y, x)

    fit = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x)), Gaussian(); data = data)

    @test is_converged(fit) isa Bool
    @test is_converged(fit) == true
    @test deviance(fit) ≈ -2 * loglik(fit)
    @test dof_residual(fit) == nobs(fit) - dof(fit)
    @test dof_residual(fit) > 0
end
