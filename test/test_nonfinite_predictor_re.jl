using DRModels
using Test, Random
import Distributions

# #1113: a NaN / Inf in a numeric predictor on a row whose response is observed
# must raise a named ArgumentError, not return a DrmFit of starting values.
@testset "non-finite predictor with (1 | g) (#1113)" begin
    rng = MersenneTwister(1113)
    n, G = 120, 12
    g = repeat(1:G, inner = n ÷ G)
    x = randn(rng, n)
    u = 0.4 .* randn(rng, G)
    μ = exp.(0.5 .+ 0.3 .* x .+ u[g])
    ypois = Float64.(rand.(rng, Distributions.Poisson.(μ)))
    ybin = Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-(0.2 .+ 0.5 .* x .+ u[g]))))
    ynorm = 1.0 .+ 0.5 .* x .+ u[g] .+ 0.5 .* randn(rng, n)
    ygam = exp.(0.3 .+ 0.2 .* x .+ u[g]) .* (0.5 .+ rand(rng, n))

    cases = (
        ("Poisson", Poisson(), ypois),
        ("NegBinomial2", NegBinomial2(), ypois),
        ("Binomial", Binomial(), ybin),
        ("Student", Student(), ynorm),
        ("Gaussian", Gaussian(), ynorm),
        ("Gamma", Gamma(), ygam),
    )
    for (name, fam, y) in cases, bad in (NaN, Inf, -Inf)
        xb = copy(x); xb[7] = bad
        data = (; y, x = xb, g)
        @testset "$name x=$bad" begin
            err = try
                drm(bf(@formula(y ~ x + (1 | g))), fam; data = data)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            msg = err isa ArgumentError ? err.msg : ""
            @test occursin("`x`", msg)
            @test occursin("row 7", msg)
        end
    end

    @testset "Float32 predictor" begin
        xb = Float32.(x); xb[3] = Float32(NaN)
        @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | g))), Poisson();
                                       data = (; y = ypois, x = xb, g))
    end

    @testset "NaN predictor on a NaN-response row is dropped with the response" begin
        yb = copy(ypois); yb[7] = NaN
        xb = copy(x); xb[7] = NaN
        fit = drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = (; y = yb, x = xb, g))
        @test nobs(fit) == n - 1
        @test isfinite(loglik(fit))
    end
end
