# #1756: Gaussian `(1 | g) + (0 + x | g)` (same group, independent slope) used
# to label both SD components `g`, so drm_bridge aborted with "raw coefficient
# labels are not unique". The slope component is now `g:x`.
using DRModels
using Test, Random

@testset "Gaussian (1 | g) + (0 + x | g) labels and bridge (#1756)" begin
    Random.seed!(1); G = 30; m = 8; n = G * m
    g = repeat(1:G, inner = m); x = randn(n); u = randn(G); v = randn(G)
    y = 0.3 .+ 0.5 .* x .+ u[g] .+ v[g] .* x .+ randn(n)
    data = (; y, x, g)
    for method in (:ML, :REML)
        f = drm(bf(@formula(y ~ 1 + x + (1 | g) + (0 + x | g)), @formula(sigma ~ 1)),
                Gaussian(); data = data, method = method)
        @test Dict(f.coefnames)[:resd] == ["g", "g:x"]
        b = drm_bridge(; formula = "y ~ x + (1 | g) + (0 + x | g)", family = "gaussian",
                       data = data, options = Dict{String,Any}("method" => String(method),
                       "coef_labels" => Dict("mu" => ["(Intercept)", "x"], "sigma" => ["(Intercept)"],
                                             "resd" => ["g", "g:x"])))
        @test length(unique(b["raw_coef_names"])) == length(b["raw_coef_names"])
        @test any(==("resd_g:x"), b["raw_coef_names"])
        @test isapprox(b["loglik"], f.loglik; atol = 1e-8)
    end
end
