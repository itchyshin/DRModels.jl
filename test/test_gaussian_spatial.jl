# Gaussian spatial structured effect: spatial(1 | site) with coordinates. The
# spatial correlation K(ρ) = exp(-d/ρ) is built from the site coordinates and the
# range ρ is estimated jointly. Closed-form structured-GLS marginal (K depends on
# θ, so it is rebuilt each evaluation). Range is weakly identified from one
# realization, so the test asserts the robustly-recoverable parts.
using DRModels
using Test, Random, LinearAlgebra

@testset "Gaussian spatial(1|site, coords) — fit + recovery" begin
    Random.seed!(20260608)
    G = 100
    coords = rand(G, 2) .* 10.0                 # sites in [0,10]²
    Ddist = [sqrt(sum(abs2, coords[k, :] .- coords[l, :])) for k in 1:G, l in 1:G]
    ρtrue = 2.0; σs = 0.9; σ = 0.4
    Ktrue = exp.(-Ddist ./ ρtrue) + 1e-8 * I
    u = σs .* (cholesky(Symmetric(Ktrue)).L * randn(G))
    m = 4; n = G * m; site = repeat(1:G, inner = m); x = randn(n)
    β = [0.3, 0.5]
    y = β[1] .+ β[2] .* x .+ u[site] .+ σ .* randn(n)
    data = (; y, x, site)

    fit = drm(bf(@formula(y ~ x + spatial(1 | site)), @formula(sigma ~ 1)), Gaussian(); data = data, coords = coords)

    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.12        # slope (independent of the spatial RE)
    @test exp(coef(fit, :sigma)[1]) ≈ σ atol = 0.15   # residual SD
    @test re_sd(fit)[:site] > 0.3                     # spatial variance detected
    @test isfinite(loglik(fit))
end
