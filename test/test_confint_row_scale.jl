# A random-effect SD row of `confint(fit; method = :profile)` is on the log SD scale when the
# profile is interior and on the SD scale when the lower limit reaches the boundary. The row now
# carries a `scale` field that says which. The numbers are unchanged.
using DRModels
using Test

@testset "confint rows carry a scale field" begin
    h(i) = mod(sin(12.9898i) * 43758.5453, 1)
    G = 12; n = 6
    g = ["g" * string(1 + (i - 1) % G) for i in 1:(G * n)]
    x = [2h(i) - 1 for i in 1:(G * n)]
    noise = [2 * (h(900 + i) + h(1900 + i) - 1) for i in 1:(G * n)]
    u = [1.5 * (2h(5000 + k) - 1) for k in 1:G]
    gi = [1 + (i - 1) % G for i in 1:(G * n)]
    d_real = (; y = [0.4x[i] + u[gi[i]] + noise[i] for i in 1:(G * n)], x, g)
    d_none = (; y = [0.4x[i] + noise[i] for i in 1:(G * n)], x, g)
    f = bf(@formula(y ~ x + (1 | g)))

    @testset "interior profile row is on the log SD scale" begin
        fit = drm(f, Gaussian(); data = d_real)
        rows = confint(fit; method = :profile)
        r = only(filter(r -> r.param === :resd, rows))
        @test r.scale === :log_sd
        @test r.lower < r.estimate < r.upper && isfinite(r.lower)
        @test r.estimate ≈ only(coef(fit, :resd))
        @test all(r2.scale === :link for r2 in rows if r2.param !== :resd)
        @test length(unique(propertynames.(rows))) == 1
        w = only(filter(r -> r.param === :resd, confint(fit)))
        @test w.scale === :log_sd
    end

    @testset "boundary profile row is on the SD scale" begin
        fit = drm(f, Gaussian(); data = d_none)
        rows = confint(fit; method = :profile)
        r = only(filter(r -> r.param === :resd, rows))
        @test r.scale === :sd
        @test r.lower == 0.0
        @test r.upper > 0
        @test all(r2.scale === :link for r2 in rows if r2.param !== :resd)
        @test length(unique(propertynames.(rows))) == 1
        w = only(filter(r -> r.param === :resd, confint(fit)))
        @test w.scale === :log_sd
    end
end
