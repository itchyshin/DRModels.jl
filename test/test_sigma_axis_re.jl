# Cluster-2: standalone non-Gaussian σ-axis random intercept `sigma ~ 1 + (1|g)`
# via DRModels._fit_sigma_axis_re (the scale-axis RE on the q=2 location–scale core).
#
# Primary purpose: regression guard for the grad! keyword bug — the inner
# `grad!` called `_sigma_re_grad(...; a0 = warm[])`, but `_sigma_re_grad` only
# accepts the `warm` keyword (it dereferences the Ref internally). The buggy call
# was swallowed by the LBFGS try/catch (silent fallback to gradient-free
# NelderMead), so it never crashed loudly — it just disabled the exact O(p)
# gradient. This test exercises the gradient path directly AND end-to-end.
#
# The route is not yet wired into any drm() frontend, so it is reached as
# DRModels._fit_sigma_axis_re. Gamma is used because the σ (scale) axis is only
# meaningful for dispersion-bearing families (Poisson has no scale parameter).
using DRModels, Test, Random, LinearAlgebra, SparseArrays
import Distributions

# Gamma draw matching the locscale leaf parameterisation (cf. test_locscale_gamma_e2e):
# η = log mean, ψ = log shape; scale = μ/α.
_sigma_axis_gamma_draw(η, ψ) = (α = exp(ψ); μ = exp(η);
                                  rand(Distributions.Gamma(α, μ / α)))

@testset "σ-axis RE (cluster 2): grad! keyword contract" begin
    Random.seed!(202)
    G = 40; nper = 10; n = G * nper
    gidx = repeat(1:G, inner = nper)
    τ = 0.6
    u = τ .* randn(G)
    β0μ = log(3.0); β0ψ = log(1.5)          # mean 3, base log-shape
    y = [_sigma_axis_gamma_draw(β0μ, β0ψ + u[gidx[i]]) for i in 1:n]
    Xμ = ones(n, 1); Xψ = ones(n, 1)
    Q = sparse(1.0 * I, G, G)
    Zη, Zψ = DRModels._sigma_re_loadings(n)
    θ = vcat(β0μ, β0ψ, log(τ))               # [βμ; βψ; logL11]

    # The FIXED grad! passes the Ref `warm`; _sigma_re_grad dereferences it.
    warmref = Base.RefValue{Union{Nothing,Vector{Float64}}}(nothing)
    g = DRModels._sigma_re_grad(Val(:gamma), y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ; warm = warmref)
    @test length(g) == 3
    @test all(isfinite, g)
    # The buggy keyword (`a0`) must be rejected — this is the exact contract the
    # inner grad! relied on, and the bug was passing `a0` instead of `warm`.
    @test_throws MethodError DRModels._sigma_re_grad(Val(:gamma), y, Xμ, Xψ, gidx, G, Q,
                                                θ, Zη, Zψ; a0 = nothing)
end

@testset "σ-axis RE (cluster 2): Gamma end-to-end recovery" begin
    Random.seed!(7)
    G = 50; nper = 14; n = G * nper
    gidx = repeat(1:G, inner = nper)
    τ = 0.5
    u = τ .* randn(G)
    β0μ = log(4.0); β0ψ = log(2.0)
    y = [_sigma_axis_gamma_draw(β0μ, β0ψ + u[gidx[i]]) for i in 1:n]
    Xμ = ones(n, 1); Xψ = ones(n, 1)

    fit = DRModels._fit_sigma_axis_re(DRModels.Gamma(), Val(:gamma), y, Xμ, Xψ, gidx, G,
                                 ["(Intercept)"], ["(Intercept)"], "g";
                                 link = :log, se = false)
    @test is_converged(fit)
    @test isfinite(loglik(fit))
    τ̂ = re_sd(fit)[:g_logsigma]     # scale-axis RE SD keyed on the log-σ axis
    @test isfinite(τ̂) && τ̂ > 0
    @test 0.2 < τ̂ < 1.0                      # recovers τ = 0.5 within generous sampling bounds
end
