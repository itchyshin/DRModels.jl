# #1594: compact `show(fit)` must use `is_converged`, as MIME"text/plain" already does (#949).
using DRModels
using Test
using Logging

@testset "#1594 one-line show and binomial separation" begin
    x = collect(range(-2, 2; length = 40))
    y = Float64.(x .> 0)
    fit = with_logger(NullLogger()) do
        drm(bf(@formula(y ~ 1 + x)), Binomial(); data = (; y, x))
    end
    @test fit.converged
    @test !is_converged(fit)
    @test convergence_status(fit) === :degenerate
    txt = sprint(show, fit)
    first_line = first(split(txt, '\n'))
    @test occursin("NOT converged", first_line)
    @test !occursin("obs, converged;", first_line)
end