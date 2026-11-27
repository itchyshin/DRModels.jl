# #1602: compact `show(fit)` must use `is_converged` on degenerate ZeroOneBeta / Tweedie
# fits (same header mismatch as #1594 on binomial separation).
using DRModels
using Test
using Logging, Random

@testset "#1602 one-line show on degenerate ZeroOneBeta" begin
    n = 50
    x = randn(MersenneTwister(1602), n)
    y = zeros(n)
    fit = with_logger(NullLogger()) do
        drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(zoi ~ 1), @formula(coi ~ 1)),
            ZeroOneBeta(); data = (; y, x))
    end
    @test fit.converged
    @test !is_converged(fit)
    @test convergence_status(fit) === :degenerate
    txt = sprint(show, fit)
    first_line = first(split(txt, '\n'))
    @test occursin("NOT converged", first_line)
    @test !occursin("obs, converged;", first_line)
end

@testset "#1602 one-line show on mostly-zero Tweedie" begin
    n = 20
    x = randn(MersenneTwister(16021), n)
    y = zeros(n)
    y[5] = 1.2
    fit = with_logger(NullLogger()) do
        drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)),
            Tweedie(); data = (; y, x))
    end
    @test fit.converged
    @test !is_converged(fit)
    @test convergence_status(fit) === :degenerate
    txt = sprint(show, fit)
    first_line = first(split(txt, '\n'))
    @test occursin("NOT converged", first_line)
    @test !occursin("obs, converged;", first_line)
end