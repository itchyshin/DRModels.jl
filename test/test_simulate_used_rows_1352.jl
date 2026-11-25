using Test, DRModels

@testset "simulate returns the rows used in the fit (#1352)" begin
    n = 60
    x = [sin(i) for i in 1:n]
    c = Union{Missing,Int}[(i % 6 == 0) ? missing : round(Int, 3 + 2x[i] + 0.5cos(7i)) for i in 1:n]
    f = drm(bf(@formula(c ~ x)), Poisson(); data = (; c, x))
    @test nobs(f) == 50
    @test length(fitted(f)) == 50
    @test length(simulate(f)) == 50
    @test size(simulate(f; nsim = 3)) == (50, 3)
    @test length(simulate(f; newdata = (x = [0.0, 1.0],))) == 2
end
