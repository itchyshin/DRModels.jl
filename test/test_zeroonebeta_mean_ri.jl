# ZeroOneBeta() mean-axis random intercept `(1 | g)` (#723; docs row pinned by #1122).
# The docstring in src/zeroonebeta.jl admits `(1 | g)` on `mu`; docs/src/capabilities.md
# must say so. `sigma`, `zoi`, `coi` stay fixed-effects-only.
using DRModels
using Test, Random
import Distributions

@testset "ZeroOneBeta mean (1 | g): fits and docs row (#1122)" begin
    rng = MersenneTwister(20261023)
    ng, m = 12, 10
    n = ng * m
    g = repeat(string.(1:ng), inner = m)
    b = 0.5 .* randn(rng, ng)
    x = randn(rng, n)
    φ = 10.0
    y = Vector{Float64}(undef, n)
    for i in 1:n
        μ = 1 / (1 + exp(-(0.2 + 0.5 * x[i] + b[parse(Int, g[i])])))
        if rand(rng) < 0.2
            y[i] = rand(rng) < 0.5 ? 1.0 : 0.0
        else
            y[i] = rand(rng, Distributions.Beta(μ * φ, (1 - μ) * φ))
        end
    end
    data = (; y, x, g)

    fit = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1), @formula(zoi ~ 1), @formula(coi ~ 1)),
        ZeroOneBeta(); data = data)
    @test isfinite(loglik(fit))

    # The capabilities table must name mean random intercepts for ZeroOneBeta.
    caps = read(joinpath(@__DIR__, "..", "docs", "src", "capabilities.md"), String)
    row = only(filter(l -> startswith(l, "| ZeroOneBeta |"), split(caps, '\n')))
    @test occursin("random intercept", row)
end
