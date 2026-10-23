# #758 — the compact 2-arg `show(io, fit)` hardcoded every family as "Gaussian
# location–scale", while the `MIME"text/plain"` method correctly named the
# fitted family. `println(fit)`, string interpolation and `repr` all go
# through the 2-arg method, so a reader saw the wrong family there.
using DRModels
using Test

@testset "#758 — 2-arg show names the real family" begin
    n = 60
    x = Float64.(1:n)
    y = rand(1:5, n)  # count-like response for Poisson
    data = (; x, y)

    p = drm(bf(@formula(y ~ x)), Poisson(); data = data)

    compact = first(split(sprint(show, p), '\n'))
    verbose = first(split(sprint(show, MIME"text/plain"(), p), '\n'))

    @test occursin("Poisson", compact)
    @test !occursin("Gaussian", compact)
    @test occursin("Poisson", verbose)

    # Gaussian fits must still say Gaussian (not a regression from this fix).
    yg = 0.5 .+ 0.1 .* x .+ randn(n)
    g = drm(bf(@formula(yg ~ x)), Gaussian(); data = (; x, yg))
    @test occursin("Gaussian", first(split(sprint(show, g), '\n')))
end
