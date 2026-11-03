# `ids = ...` must match a labelled relatedness matrix `A` by label on the
# predictor-dependent-SD (`sd(g, animal)`) routes too, not only on the plain
# routes (#957). Regression for a merge that let the SD dispatch return before
# label resolution, so `ids` was accepted and silently ignored.
using Test, DRModels, Random, LinearAlgebra
import Distributions

@testset "sd(g, animal) routes honour ids = ... (#957)" begin
    rng = MersenneTwister(3)
    G = 10; per = 12; n = G * per
    labs = ["id$(lpad(i, 2, '0'))" for i in 1:G]
    app = labs[randperm(rng, G)]                 # order of first appearance in the data
    ind = repeat(app, inner = per)
    B = randn(rng, G, G)
    Afull = Symmetric(B * B' / G + I) |> Matrix  # rows ordered as `labs` (sorted)
    ord = [findfirst(==(l), labs) for l in app]
    A_app = Afull[ord, ord]                      # the same matrix in appearance order
    x = randn(rng, n); h = repeat(randn(rng, G), inner = per)
    ys = Dict(
        "Poisson" => (Poisson(), Float64[rand(rng, Distributions.Poisson(exp(0.3 + 0.2xi))) for xi in x]),
        "NegBinomial2" => (NegBinomial2(), Float64[rand(rng, Distributions.Poisson(exp(0.5 + 0.2xi))) for xi in x]),
        "Gamma" => (Gamma(), Float64[rand(rng, Distributions.Gamma(4.0, exp(0.3 + 0.2xi) / 4)) for xi in x]),
        "Beta" => (Beta(), Float64[clamp(rand(rng, Distributions.Beta(4.0, 4.0)), 0.01, 0.99) for _ in x]),
    )
    fm = bf(@formula(y ~ x + animal(0 + x | ind)), @formula(sd(ind, animal) ~ h))
    fm_int = bf(@formula(y ~ x + animal(1 | ind)), @formula(sd(ind, animal) ~ h))   # Beta: intercept only
    for name in ("Poisson", "NegBinomial2", "Gamma", "Beta")
        fam, y = ys[name]
        dat = (; y, x, h, ind)
        form = name == "Beta" ? fm_int : fm
        ref = drm(form, fam; data = dat, A = A_app, se = false)
        lab = drm(form, fam; data = dat, A = Afull, ids = labs, se = false)
        @test lab.theta ≈ ref.theta rtol = 1e-6 atol = 1e-8
        # without `ids`, the sorted `A` is read by position and must differ
        pos = drm(form, fam; data = dat, A = Afull, se = false)
        @test !isapprox(pos.theta, ref.theta; rtol = 1e-6, atol = 1e-8)
    end
    # `ids` with nothing to label is refused on the SD route as well
    @test_throws ArgumentError drm(fm, Poisson(); data = (; y = ys["Poisson"][2], x, h, ind),
                                   ids = labs, se = false)
end
