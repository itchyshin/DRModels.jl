# Refs #1036: two `marginal = :VA` fits must be refused by lrtest / anova, because a
# VA `loglik` is an ELBO, not a log-likelihood.
using DRModels
using Test, Random
import Distributions

@testset "VA vs VA comparison refused (#1036)" begin
    rng = MersenneTwister(1036)
    G = 40; per = 10; n = G * per
    g = repeat(1:G, inner = per); x = randn(rng, n)
    λ = exp.(0.4 .- 0.3 .* x .+ 0.5 .* randn(rng, G)[g])
    y = Float64.([rand(rng, Distributions.Poisson(l)) for l in λ])
    data = (; y, x, g)
    va_full = drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = data, marginal = :VA)
    va_red  = drm(bf(@formula(y ~ 1 + (1 | g))), Poisson(); data = data, marginal = :VA)
    for f in (DRModels.lrtest, DRModels.anova)
        err = try f(va_red, va_full); nothing catch e; e end
        @test err isa ArgumentError
        @test occursin("ELBO", err.msg) && occursin("`:VA` vs `:VA`", err.msg)
    end
end
