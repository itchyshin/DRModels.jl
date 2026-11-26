# DJ1: small selectors in src/inference.jl. Bootstrap drivers and
# `_loconly_profile_fg` are rewritten by open drafts and are not called here.
using DRModels, Test, LinearAlgebra, StatsModels

@testset "confint rejects a missing coefficient inside a block that exists" begin
    fit = DrmFit(DRModels.Gaussian(),
                 [:mu => 1:2, :sigma => 3:3],
                 [:mu => ["(Intercept)", "x"], :sigma => ["(Intercept)"]],
                 zeros(3),
                 Matrix{Float64}(I, 3, 3),
                 -1.0, 10, true,
                 Dict(:mu => ones(10)),
                 Dict(:mu => ones(10)),
                 Dict(:sigma => ones(10)))
    # A superset of block names is the documented skip, not an error.
    @test DRModels._ci_validate_parm(fit, [:mu, :sigma, :resd]) === nothing
    @test DRModels._ci_validate_parm(fit, :resd => "anything") === nothing
    @test DRModels._ci_validate_parm(fit, :sigma => "(Intercept)") === nothing
    err = try
        DRModels._ci_validate_parm(fit, :mu => "not_a_column")
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("not_a_column", err.msg)
    @test occursin("(Intercept)", err.msg)
    @test occursin("x", err.msg)
end

@testset "random-effect bar is detected inside tuples and vectors" begin
    bar = (@formula(y ~ (1 | g))).rhs
    plain = (@formula(y ~ x)).rhs
    @test bar isa FunctionTerm && bar.f === (|)
    @test DRModels._term_has_bar((bar,))
    @test DRModels._term_has_bar([bar, plain])
    @test !DRModels._term_has_bar((plain,))
    @test !DRModels._term_has_bar([plain])
    @test !DRModels._term_has_bar(1)
end
