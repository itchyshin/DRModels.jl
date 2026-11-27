using DRModels, StableRNGs, Test

@testset "#1576: fit-based bootstrap requires named `data` argument" begin
    n = 50
    y = randn(n)
    data = (; y)
    fit = drm(bf(@formula(y ~ 1), @formula(sigma ~ 1)), Gaussian(); data)
    rng = StableRNG(1)

    @testset "bootstrap_result" begin
        ex = try
            bootstrap_result(fit; B = 2, rng)
            nothing
        catch e
            e
        end
        @test ex isa ArgumentError
        @test occursin("data", sprint(showerror, ex))
        res = bootstrap_result(fit; data, B = 2, rng)
        @test haskey(res, :summary)
    end

    @testset "bootstrap_ci" begin
        ex = try
            bootstrap_ci(fit; B = 2, rng)
            nothing
        catch e
            e
        end
        @test ex isa ArgumentError
        @test occursin("data", sprint(showerror, ex))
        @test bootstrap_ci(fit; data, B = 2, rng) isa Vector
    end
end