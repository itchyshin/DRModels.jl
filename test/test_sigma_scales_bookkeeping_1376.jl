# sigma(fit) must not surface fit.scales bookkeeping (review of #1376).
using DRModels, LinearAlgebra, Test
import Distributions

h(i) = mod(sin(12.9898i) * 43758.5453, 1)

@testset "sigma() omits scales bookkeeping keys (#1376)" begin
    G = 24
    m = 12
    N = G * m
    gi = repeat(1:G, inner = m)
    g = ["animal_" * string(k) for k in gi]
    A = [0.35^abs(i - j) for i in 1:G, j in 1:G]
    L = cholesky(Symmetric(A)).L
    u0 = L * [3.4 * (h(1000 + k) - 0.5) for k in 1:G]
    u1 = L * [3.4 * (h(2000 + k) - 0.5) for k in 1:G]
    z0 = [isodd(k) ? -1.0 : 1.0 for k in gi]
    z1 = [2 * mod(7k, G) / (G - 1) - 1 for k in gi]
    x = [2 * mod(i - 1, m) / (m - 1) - 1 for i in 1:N]
    eta = [0.2 + 0.3x[i] + exp(-0.9 + 0.35z0[i]) * u0[gi[i]] + x[i] * exp(-1.0 + 0.45z1[i]) * u1[gi[i]] for i in 1:N]
    y = [Float64(Distributions.quantile(Distributions.Poisson(exp(eta[i])), 0.001 + 0.998h(3000 + i))) for i in 1:N]
    data = (; y, x, z0, z1, g)
    f = bf(@formula(y ~ x + animal(1 + x | g)),
        @formula(sd(g, animal, coef(intercept)) ~ z0),
        @formula(sd(g, animal, coef(x)) ~ z1))
    fit = drm(f, Poisson(); data, A, ids = unique(g))
    @test fit.converged
    @test haskey(fit.scales, :q2_restarts)
    σ = sigma(fit)
    @test σ isa Dict{Symbol, Vector{Float64}}
    @test isempty(σ)
    @test !haskey(DRModels._bridge_dpars(fit), "q2_restarts")
    for k in keys(fit.scales)
        DRModels._scales_bookkeeping_key(k) && continue
        # only non-bookkeeping keys would be public scale parameters; Poisson mean-only has none
        @test k ∉ (:zi, :hu, :nu, :phi)
    end
end