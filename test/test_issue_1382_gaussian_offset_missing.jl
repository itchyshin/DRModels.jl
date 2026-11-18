# refs #1382 — Gaussian `offset(...)` with missing values must listwise-drop like Poisson.
module TestIssue1382GaussianOffsetMissing

using DRModels
using Test

@testset "#1382 Gaussian offset missing values" begin
    N = 40
    i = 1:N
    x = sin.(i)
    off = 0.1 .+ mod.(0.618i, 1)
    yg = 1.0 .+ 0.5 .* x .+ off .+ 0.3 .* sin.(5.1 .* i)
    offna = Union{Missing, Float64}[o for o in off]
    offna[[3, 17]] .= missing
    d = (; yg, x, offna)
    fit = drm(bf(@formula(yg ~ x + offset(offna)), @formula(sigma ~ 1)), Gaussian(); data = d)
    @test nobs(fit) == 38
    @test length(coef(fit)) >= 2
end

end