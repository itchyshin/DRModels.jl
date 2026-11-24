using DRModels, Test

@testset "Hurdle NB2 SD design rank uses positive-count groups" begin
    # The full design has both z levels, but every positive count is in z=-1
    # groups. Groups at z=1 contribute only to hu and cannot identify the
    # count-side SD slope.
    data = (; y = [1.0, 2.0, 3.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            id = repeat(1:4, inner = 2),
            z = repeat([-1.0, -1.0, 1.0, 1.0], inner = 2))
    formula = bf(@formula(y ~ 1 + (1 | id)), @formula(sigma ~ 1),
                 @formula(hu ~ 1), @formula(sd(id) ~ z))

    err = try
        drm(formula, NegBinomial2(); data)
        nothing
    catch caught
        caught
    end
    @test err isa ArgumentError
    @test occursin("positive-count groups", sprint(showerror, err))
end
