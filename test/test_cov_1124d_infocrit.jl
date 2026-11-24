# test_cov_1124d_infocrit.jl — lane 1124d: loglik / deviance / dof_residual accessors
# and REPL summary lines on a tiny fixed-effect Gaussian fit.
using DRModels
using Test, Random, Logging

function _sentinel_fit(ll::Real)
    Random.seed!(1124)
    n = 60
    x = randn(n)
    y = 0.2 .+ x .+ randn(n)
    base = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
    vals = Any[getfield(base, k) for k in fieldnames(typeof(base))]
    fn = fieldnames(typeof(base))
    vals[findfirst(==(:loglik), fn)] = ll
    vals[findfirst(==(:ml_loglik), fn)] = ll
    vals[findfirst(==(:converged), fn)] = true
    return typeof(base).name.wrapper(vals...)
end

@testset "cov 1124d: information-criterion accessors" begin
    Random.seed!(112406)
    n = 80
    x = randn(n)
    y = 0.5 .- 0.7 .* x .+ exp.(-0.2) .* randn(n)
    fit = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))

    @test loglik(fit) ≈ fit.loglik
    @test deviance(fit) ≈ -2 * loglik(fit)
    k = dof(fit)
    @test dof_residual(fit) == nobs(fit) - k
    @test dof_residual(fit) > 0

    @testset "deviance on sentinel loglik returns NaN" begin
        bad = _sentinel_fit(-1e18)
        @test with_logger(NullLogger()) do
            isnan(deviance(bad))
        end
        @test_logs (:warn,) deviance(bad)
    end

    @testset "MIME text/plain summary mentions family and nobs" begin
        txt = sprint(show, MIME"text/plain"(), fit)
        @test occursin("Gaussian", txt)
        @test occursin("nobs = $(nobs(fit))", txt)
        @test occursin("logLik", txt)
        @test family(fit) === fit.family
        @test converged(fit) == is_converged(fit)
    end
end