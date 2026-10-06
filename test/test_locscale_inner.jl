# q=2 augmented inner solver for the non-Gaussian location–scale model
# (#202 groundwork). Gates the joint gradient/Hessian against ForwardDiff and
# checks the mode-finder reaches a stationary point. Engine-lane test: calls the
# internal solver directly (not yet wired into drm()).
using DRModels
using Test, Random, LinearAlgebra, SparseArrays
import ForwardDiff
import Distributions

# Build P = kron(I_G, Λ⁻¹) for an i.i.d. grouping (reuses the shipped
# `prior_precision`, which is dimension-general).
_iid_P(G, Λ) = DRModels.prior_precision(sparse(1.0 * I, G, G), inv(Λ))

@testset "location–scale Λ log-Cholesky round-trip" begin
    for v in ([0.1, -0.3, 0.2], [-0.5, 0.4, 0.0], [0.7, 0.0, -0.2])
        Λ = DRModels._ls_lc_to_Λ(v)
        @test issymmetric(Λ)
        @test isposdef(Λ)
        @test DRModels._ls_Λ_to_lc(Λ) ≈ v rtol = 1e-10
    end
end

@testset "location–scale inner joint: grad/Hessian vs ForwardDiff" begin
    Random.seed!(20260606)
    Λ = DRModels._ls_lc_to_Λ([0.0, 0.25, -0.2])
    P = _iid_P(6, Λ)

    for (kind, gen) in [
        (Val(:nb2), (η, ψ) -> Float64(rand(Distributions.NegativeBinomial(exp(ψ), exp(ψ) / (exp(ψ) + exp(η)))))),
        (Val(:gamma), (η, ψ) -> rand(Distributions.Gamma(exp(ψ), exp(η) / exp(ψ)))),
    ]
        G = 6; m = 4; n = G * m
        gidx = repeat(1:G, inner = m)
        η0 = 0.2 .+ 0.1 .* randn(n)
        ψ0 = 0.3 .+ 0.1 .* randn(n)
        atrue = 0.3 .* randn(2G)
        y = [gen(η0[i] + atrue[2gidx[i]-1], ψ0[i] + atrue[2gidx[i]]) for i in 1:n]

        a = 0.2 .* randn(2G)                       # evaluate OFF the mode
        f = aa -> DRModels._ls_joint(kind, y, η0, ψ0, gidx, aa, P)

        g_an = DRModels._ls_joint_grad(kind, y, η0, ψ0, gidx, a, P)
        @test g_an ≈ ForwardDiff.gradient(f, a) rtol = 1e-6 atol = 1e-8

        H_an = Matrix(DRModels._ls_joint_hess(kind, y, η0, ψ0, gidx, G, a, P))
        @test H_an ≈ ForwardDiff.hessian(f, a) rtol = 1e-6 atol = 1e-8
    end
end

@testset "location–scale inner mode reaches a stationary point" begin
    Random.seed!(20260607)
    Λ = DRModels._ls_lc_to_Λ([0.1, 0.2, -0.1])
    G = 8; m = 6; n = G * m
    P = _iid_P(G, Λ)
    gidx = repeat(1:G, inner = m)
    η0 = 0.1 .+ 0.1 .* randn(n)
    ψ0 = 0.4 .+ 0.1 .* randn(n)
    atrue = 0.3 .* randn(2G)
    y = [Float64(rand(Distributions.NegativeBinomial(
             exp(ψ0[i] + atrue[2gidx[i]]),
             exp(ψ0[i] + atrue[2gidx[i]]) / (exp(ψ0[i] + atrue[2gidx[i]]) + exp(η0[i] + atrue[2gidx[i]-1]))
         ))) for i in 1:n]

    a, ch, ok = DRModels._ls_inner_mode(Val(:nb2), y, η0, ψ0, gidx, G, P)
    @test ok
    @test ch !== nothing
    g = DRModels._ls_joint_grad(Val(:nb2), y, η0, ψ0, gidx, a, P)
    @test norm(g) < 1e-6
end
