# #1059: a fit whose scale runs off toward zero must not report converged = true.
using DRModels
using Test, Random

@testset "#1059 unbounded scale is not reported as converged" begin
    # One group has a constant response, so log sigma(x = 1) -> -Inf.
    Random.seed!(1)
    n = 60
    x = Float64.(vcat(zeros(n ÷ 2), ones(n ÷ 2)))
    y = ifelse.(x .== 1, 2.0, randn(n))
    data = (; y, x)
    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian(); data = data)
    @test fit.converged == false

    # The rule is unit-free: rescaling y does not change the verdict.
    fit_s = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian();
                data = (; y = 1e6 .* y, x))
    @test fit_s.converged == false

    # Detector contract on the fitted scales.
    @test DRModels._scale_runoff(Dict(:sigma => [1.0, 1e-6, 2.0]))
    @test DRModels._scale_runoff(Dict(:sigma => [1.0, 0.0]))
    @test !DRModels._scale_runoff(Dict(:sigma => [1.0, 5.0, 20.0]))
    @test !DRModels._scale_runoff(Dict(:sigma => [1e-9, 2e-9]))   # units: uniform tiny scale is fine
    @test !DRModels._scale_runoff(Dict{Symbol,Vector{Float64}}())

    # A well-behaved heteroscedastic fit still converges.
    Random.seed!(2)
    m = 800; z = randn(m)
    yh = 0.5 .+ 0.3 .* z .+ exp.(-0.2 .+ 0.4 .* z) .* randn(m)
    fit_ok = drm(bf(@formula(y ~ z), @formula(sigma ~ z)), Gaussian(); data = (; y = yh, z))
    @test fit_ok.converged
end
