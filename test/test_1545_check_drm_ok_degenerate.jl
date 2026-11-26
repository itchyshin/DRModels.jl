# #1545: `check_drm(fit).ok` must be false when `convergence_status` is
# `:degenerate` and `is_converged` is false (Gaussian (1|g), one obs per group).
using DRModels
using Test, Random, Logging

@testset "#1545 check_drm ok false for inflated-SE :degenerate Gaussian RE" begin
    Random.seed!(20261126)
    data = (; y = randn(10), x = randn(10), g = collect(1:10))
    fit = with_logger(Logging.NullLogger()) do
        drm(bf(@formula(y ~ x + (1 | g))), Gaussian(); data)
    end
    r = with_logger(Logging.NullLogger()) do
        check_drm(fit)
    end
    @test fit.converged
    @test !is_converged(fit)
    @test r.convergence_status === :degenerate
    @test !r.converged
    @test !r.ok
    @test r.ok == r.converged
end