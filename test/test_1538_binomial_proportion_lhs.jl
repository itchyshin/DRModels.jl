# Binomial proportion response `y / trials ~ x` used to leak FieldError on
# `FunctionTerm.sym` from `bf()` (#1538, same failure class as #1336).
using DRModels, Test

@testset "binomial proportion lhs is refused by name (#1538)" begin
    n = 10
    s = [3, 4, 2, 5, 3, 4, 2, 6, 3, 4]
    trials = fill(10, n)
    x = LinRange(-1, 1, n)
    data = (; s, trials, x, y = s ./ trials)

    err = try
        drm(bf(@formula(y / trials ~ x)), Binomial(); data = data)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    err isa ArgumentError || (println("got: ", typeof(err)); return)
    @test occursin("y / trials", err.msg)
    @test occursin("proportion", err.msg)
    @test occursin("cbind(successes, failures)", err.msg)
    @test occursin("cbind(y, trials - y)", err.msg)
    @test !occursin("FunctionTerm", err.msg)
    @test !occursin("no field `sym`", err.msg)

    # working spelling from the issue
    failures = trials .- s
    fit = drm(bf(@formula(cbind(s, failures) ~ x)), Binomial(); data = merge(data, (; failures)))
    @test isfinite(loglik(fit))
end