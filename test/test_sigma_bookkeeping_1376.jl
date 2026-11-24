# Refs #1376: `sigma(fit)` must not return internal bookkeeping from `fit.scales`.
using DRModels
using Test, LinearAlgebra
import Distributions

@testset "sigma hides fit.scales bookkeeping (#1376)" begin
    h(i) = mod(sin(12.9898 * i) * 43758.5453, 1)
    G = 24; m = 12; N = G * m
    gi = repeat(1:G, inner = m)
    g = ["animal_" * string(k) for k in gi]
    A = [0.35^abs(i - j) for i in 1:G, j in 1:G]
    L = cholesky(Symmetric(A)).L
    u0 = L * [3.4 * (h(1000 + k) - 0.5) for k in 1:G]
    u1 = L * [3.4 * (h(2000 + k) - 0.5) for k in 1:G]
    z0 = [isodd(k) ? -1.0 : 1.0 for k in gi]
    z1 = [2 * mod(7 * k, G) / (G - 1) - 1 for k in gi]
    x = [2 * mod(i - 1, m) / (m - 1) - 1 for i in 1:N]
    eta = [0.2 + 0.3 * x[i] + exp(-0.9 + 0.35 * z0[i]) * u0[gi[i]] +
           x[i] * exp(-1.0 + 0.45 * z1[i]) * u1[gi[i]] for i in 1:N]
    y = [Float64(Distributions.quantile(Distributions.Poisson(exp(eta[i])), 0.001 + 0.998 * h(3000 + i)))
         for i in 1:N]
    data = (; y, x, z0, z1, g)
    f = bf(@formula(y ~ x + animal(1 + x | g)),
           @formula(sd(g, animal, coef(intercept)) ~ z0),
           @formula(sd(g, animal, coef(x)) ~ z1))
    fit = drm(f, Poisson(); data, A, ids = unique(g))
    @test fit.converged
    @test haskey(fit.scales, :q2_restarts)
    s = sigma(fit)
    @test isempty(s)
    @test !haskey(s, :q2_restarts)
    fp = drm(bf(@formula(y ~ x)), Poisson(); data)
    @test sigma(fp) == Dict{Symbol, Vector{Float64}}()
end