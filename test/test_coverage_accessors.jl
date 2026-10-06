# test_coverage_accessors.jl — anchor previously-UNTESTED exported symbols.
#
# Survey (names(DRModels) vs every test/*.jl reference) found five exported functions
# with no direct test anywhere in the suite. This file pins each to its documented
# contract so a signature/behaviour regression is caught:
#
#   • fixef        — user-facing accessor: fixed effects per distributional param
#   • unpack_theta — engine θ → (β NamedTuple, lc) split (inverse of pack_theta)
#   • joint_nll    — augmented-state joint negative log-likelihood at u
#   • joint_grad   — its gradient wrt u (verified against a central difference)
#   • build_Huu    — the augmented joint Hessian H_uu = P + blockdiag(leaf blocks)
#
# The engine trio is exercised through the public `DRModels.` exports on the same small
# p=8 augmented-tree problem used by test_sparse_aug.jl.

using DRModels
using Test, LinearAlgebra, SparseArrays, Random

# ---------------------------------------------------------------------------
# fixef(fit) — fixed-effect accessor on a fitted Gaussian location–scale model.
# ---------------------------------------------------------------------------
@testset "fixef(fit) — per-parameter fixed effects" begin
    Random.seed!(20260610)
    n = 400
    x = randn(n)
    y = 0.5 .- 0.8 .* x .+ exp.(-0.3 .+ 0.4 .* x) .* randn(n)
    data = (; y, x)
    fit = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x)), Gaussian(); data = data)

    fe = fixef(fit)
    @test fe isa AbstractVector
    # one entry per fitted parameter block, in block order (μ then σ here).
    @test first.(fe) == first.(fit.blocks)
    @test Set(first.(fe)) == Set([:mu, :sigma])

    for (p, nt) in fe
        # each entry pairs the coefficient names with the estimates …
        @test nt.estimate == coef(fit, p)                 # … and equals coef(fit, p)
        @test length(nt.names) == length(nt.estimate)     # names line up with estimates
        @test all(s -> s isa AbstractString, nt.names)
        @test eltype(nt.estimate) <: Real
    end

    # recovers the data-generating coefficients (loose tolerance, single fit).
    mu = Dict(fixef(fit))[:mu].estimate
    @test mu[1] ≈ 0.5 atol = 0.15      # μ intercept
    @test mu[2] ≈ -0.8 atol = 0.15     # μ slope
end

# ---------------------------------------------------------------------------
# Engine internals on a small augmented p=8 PLSM problem (built via DRModels exports,
# mirroring test_sparse_aug.jl's construction).
# ---------------------------------------------------------------------------
@testset "engine: unpack_theta / joint_nll / joint_grad / build_Huu" begin
    Random.seed!(11)
    p = 8
    phy = random_balanced_tree(p; branch_length = 0.2)
    n_total = phy.n_total
    Σ_phy = sigma_phy_dense(phy; σ²_phy = 1.0)

    n = p
    x1 = randn(n)
    X1 = hcat(ones(n), x1); X2 = hcat(ones(n), x1)
    Xs1 = reshape(ones(n), n, 1); Xs2 = reshape(ones(n), n, 1); Xr = reshape(ones(n), n, 1)

    β = (mu1 = [0.5, 0.3], mu2 = [-0.2, 0.4], s1 = [-0.4], s2 = [-0.5], rho = [0.3])
    Λ = [0.30 0.10 0.05 0.0; 0.10 0.30 0.0 0.04; 0.05 0.0 0.12 0.02; 0.0 0.04 0.02 0.12]
    Λ = (Λ + Λ') / 2
    @test isposdef(Λ)
    Λinv = inv(Λ)

    # simulate leaf data so the likelihood is non-degenerate.
    Lc = cholesky(Λ).L; Sc = cholesky(Symmetric(Σ_phy)).U
    Uleaf = Lc * randn(4, p) * Sc
    y1 = zeros(n); y2 = zeros(n)
    for i in 1:n
        m1 = (X1[i, :]' * β.mu1) + Uleaf[1, i]
        m2 = (X2[i, :]' * β.mu2) + Uleaf[2, i]
        s1 = exp((Xs1[i, :]' * β.s1) + Uleaf[3, i])
        s2 = exp((Xs2[i, :]' * β.s2) + Uleaf[4, i])
        ρ = 0.99 * tanh(Xr[i, :]' * β.rho)
        e = cholesky([s1^2 ρ*s1*s2; ρ*s1*s2 s2^2]).L * randn(2)
        y1[i] = m1 + e[1]; y2[i] = m2 + e[2]
    end

    # condition on the root (matches sigma_phy_dense) → full-rank PD prior.
    keep = setdiff(1:n_total, [phy.root_index])
    n_keep = length(keep)
    Q_cond = phy.Q_topology[keep, keep]
    pos = Dict(node => i for (i, node) in enumerate(keep))
    leaf_node = [pos[phy.leaf_indices[k]] for k in 1:p]
    prob = AugProblem(phy, n_keep, p, leaf_node, y1, y2, X1, X2, Xs1, Xs2, Xr)
    P = prior_precision(Q_cond, Λinv)

    @testset "unpack_theta — inverse of pack_theta" begin
        θ = pack_theta(β, Λ)
        @test length(θ) == 17                          # 7 β + 10 log-Cholesky
        β2, lc = unpack_theta(prob, θ)
        @test β2.mu1 == β.mu1
        @test β2.mu2 == β.mu2
        @test β2.s1 == β.s1
        @test β2.s2 == β.s2
        @test β2.rho == β.rho
        @test lc == Λ_to_lc(Λ)                         # log-Cholesky block recovered
        @test length(lc) == 10
        # eltype follows θ (the docstring contract; needed for AD over θ).
        β3, lc3 = unpack_theta(prob, Float32.(θ))
        @test eltype(β3.mu1) === Float32
        @test eltype(lc3) === Float32
    end

    @testset "joint_nll / joint_grad / build_Huu at a nonzero u" begin
        u = 0.05 .* randn(4 * n_keep)

        val = joint_nll(prob, P, u, β)
        @test val isa Float64
        @test isfinite(val)
        # at the supplied u the prior quadratic 0.5·uᵀPu is part of the value …
        @test val ≥ 0.5 * dot(u, P * u) - 1e-8         # leaf NLLs add a non-negative-ish remainder
        @test joint_nll(prob, P, zero(u), β) isa Float64

        g = joint_grad(prob, P, u, β)
        @test length(g) == 4 * n_keep
        @test all(isfinite, g)
        # gradient must match a central finite difference of joint_nll.
        h = 1e-6; gfd = similar(g)
        for k in eachindex(u)
            up = copy(u); up[k] += h; um = copy(u); um[k] -= h
            gfd[k] = (joint_nll(prob, P, up, β) - joint_nll(prob, P, um, β)) / (2h)
        end
        @test maximum(abs.(g .- gfd)) < 1e-6

        H = build_Huu(prob, P, u, β)
        @test issparse(H)
        @test size(H) == (4 * n_keep, 4 * n_keep)
        @test maximum(abs.(H .- H')) < 1e-10           # symmetric
        # H = P + (leaf data Hessian blocks), so its sparsity ⊇ that of P.
        @test nnz(H) ≥ nnz(P)
        # leaf blocks add curvature on top of the prior precision.
        @test H != P
    end
end
