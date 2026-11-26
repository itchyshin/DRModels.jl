# DRModels.jl#1570 — completely separated Binomial bootstrap must not report success with
# zero-width CIs when `check_converged = true`.

using Random
using DRModels
using Test

@testset "#1570 binomial complete separation bootstrap" begin
    Random.seed!(4026)
    n = 40
    x = repeat([0.0, 1.0], outer = n ÷ 2)
    y = ifelse.(x .> 0, 1.0, 0.0)
    data = (; y, x)

    fit = drm(bf(@formula(y ~ x)), Binomial(); data = data)
    @test fit.converged
    @test !is_converged(fit)
    @test convergence_status(fit) === :degenerate

    logs, err = Test.collect_test_logs() do
        try
            bootstrap_result(
                fit; data = data, B = 20,
                rng = MersenneTwister(4026), threads = false, check_converged = true)
            return nothing
        catch err
            return err
        end
    end
    @test err isa ErrorException
    @test occursin("all 20 bootstrap replicates failed", sprint(showerror, err))
    @test occursin("degenerate optimum", sprint(showerror, err))
    @test any(l -> occursin("binomial separation", string(l.message)), logs)
    @test fit.geometry === :binomial_separation
end