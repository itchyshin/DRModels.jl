# Bridge side of the Gaussian relmat and coordinate-spatial slope-only SD
# routes (added to the fitter in 40da69c72). `drm_bridge` must render the typed
# coefficient labels for `relmat(0 + x | g)` / `spatial(0 + x | g)` with a
# Gaussian family and agree with a direct fit. Spatial coordinate rows follow
# the first-seen group order, as the fitter requires.
using DRModels, Test, Random, LinearAlgebra, Statistics

@testset "drm_bridge Gaussian relmat/spatial slope-only SD" begin
    rng = MersenneTwister(20261102)
    G, m = 16, 12
    coords = randn(rng, G, 2)
    Dd = [norm(coords[k, :] - coords[l, :]) for k in 1:G, l in 1:G]
    C = exp.(-Dd ./ median([Dd[k, l] for k in 1:G for l in (k+1):G])) + 1e-6 * I
    s = collect(range(0.7, 1.6; length = G))
    R = 0.4 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    K = Matrix(Diagonal(s) * R * Diagonal(s))
    zgroup = repeat([-1.0, 1.0], inner = G ÷ 2)
    g = repeat(1:G, inner = m)
    x = randn(rng, G * m)
    for (kind, cov) in ((:relmat, K), (:spatial, C))
        u = exp.(-0.4 .+ 0.5 .* zgroup) .* (cholesky(Symmetric(cov)).L * randn(rng, G))
        y = 0.3 .+ 0.2 .* x .+ x .* u[g] .+ 0.5 .* randn(rng, G * m)
        data = (; y, x, z = zgroup[g], g)
        formula = "y ~ x + $(kind)(0 + x | g); sd(g, $(kind)) ~ z"
        labels = Dict{String,Any}("coef_labels" => Dict(
            "mu" => ["(Intercept)", "x"], "sigma" => ["(Intercept)"],
            "sd" => ["(Intercept)", "z"]))
        out = kind === :relmat ?
            drm_bridge(; formula, family = "gaussian", data, K = K, options = labels) :
            drm_bridge(; formula, family = "gaussian", data, coords = coords, options = labels)
        direct = drm(bf(kind === :relmat ? @formula(y ~ x + relmat(0 + x | g)) :
                                           @formula(y ~ x + spatial(0 + x | g)),
                        kind === :relmat ? @formula(sd(g, relmat) ~ z) :
                                           @formula(sd(g, spatial) ~ z)),
                     Gaussian(); data, (kind === :relmat ? (; K = K) : (; coords = coords))...)
        @testset "$kind" begin
            @test out["coefficients"] ≈ coef(direct) atol = 1e-6
            @test out["loglik"] ≈ direct.loglik atol = 1e-6
            @test out["coef_names"][1:2] == ["mu_(Intercept)", "mu_x"]
        end
    end
end
