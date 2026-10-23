# Post-fit `family(fit)` accessor: fit a tiny Gaussian location–scale model and
# assert the accessor returns the family object that was passed to `drm`.
using DRModels
using Test, Random

@testset "family(fit) post-fit accessor" begin
    Random.seed!(20260602)
    n = 500
    x = randn(n)
    y = 0.5 .- 0.8 .* x .+ exp.(-0.3 .+ 0.4 .* x) .* randn(n)
    data = (; y, x)

    fit = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x)), Gaussian(); data = data)

    @test family(fit) isa Gaussian
    @test family(fit) === fit.family
end
