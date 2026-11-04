# Bridge side of the Poisson relmat slope-only SD route: `drm_bridge` must
# render the typed coefficient labels for `relmat(0 + x | g)` with a Poisson
# family (previously only Binomial was admitted) and agree with a direct fit.
using DRModels, Test, Random, LinearAlgebra
import Distributions

@testset "drm_bridge Poisson relmat slope-only SD" begin
    rng = MersenneTwister(20261101)
    G, m = 20, 16
    R = 0.4 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    K = Diagonal(collect(range(0.7, 1.6; length = G))) * R *
        Diagonal(collect(range(0.7, 1.6; length = G)))
    zgroup = repeat([-1.0, 1.0], inner = G ÷ 2)
    slope = exp.(-0.55 .+ 0.35 .* zgroup) .* (cholesky(Symmetric(R)).L * randn(rng, G))
    g = repeat(1:G, inner = m)
    x = repeat(collect(range(-1.0, 1.0; length = m)), outer = G)
    y = Float64[rand(rng, Distributions.Poisson(exp(0.15 + 0.25 * x[i] + x[i] * slope[g[i]])))
                for i in eachindex(g)]
    data = (; y, x, z = zgroup[g], g)
    formula = "y ~ x + relmat(0 + x | g); sd(g, relmat) ~ z"

    out = drm_bridge(; formula, family = "poisson", data, K = K,
                     options = Dict{String,Any}("coef_labels" => Dict(
                         "mu" => ["(Intercept)", "x"],
                         "sd" => ["(Intercept)", "z"])))
    direct = drm(bf(@formula(y ~ x + relmat(0 + x | g)), @formula(sd(g, relmat) ~ z)),
                 Poisson(); data, K, se = true)
    @test out["coefficients"] ≈ coef(direct) atol = 1e-6
    @test out["loglik"] ≈ direct.loglik atol = 1e-6
    @test length(out["coef_names"]) == length(coef(direct)) == 4
    @test out["coef_names"][1:2] == ["mu_(Intercept)", "mu_x"]
end
