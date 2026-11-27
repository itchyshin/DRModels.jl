using DRModels, Random, StatsAPI, Test

@testset "#1580: StatsAPI coefnames on DrmFit" begin
    Random.seed!(1001)
    data = (y = randn(40) .+ 1, x = randn(40), g = repeat(1:8, inner = 5))
    fit = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data)

    θ = coef(fit)
    nms = coefnames(fit)
    @test nms isa Vector{String}
    @test length(nms) == length(θ)
    expected = String[]
    for ((_, _), (_, nms)) in zip(fit.blocks, fit.coefnames)
        append!(expected, nms)
    end
    @test coefnames(fit) == expected
end