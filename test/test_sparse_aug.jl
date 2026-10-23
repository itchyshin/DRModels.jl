# test_sparse_aug.jl — Checkpoint 3: the augmented-state sparse Laplace E-step
# must agree with a leaf-only DENSE Laplace at p=8 (the foundation gate).
#
# Compares, at a fixed θ on a small tree:
#   (a) augmented mode û at the leaf nodes  vs  dense leaf mode
#   (b) augmented Laplace marginal          vs  dense leaf marginal

using DRModels
using Test, LinearAlgebra, ForwardDiff, Random

const D = DRModels

@testset "augmented sparse Laplace E-step matches dense leaf-only oracle (p=8)" begin
    Random.seed!(11)
    p = 8
    phy = random_balanced_tree(p; branch_length = 0.2)
    n_total = phy.n_total
    Σ_phy = sigma_phy_dense(phy; σ²_phy = 1.0)            # dense leaf covariance (oracle)
    Σ_inv = inv(Symmetric(Σ_phy))

    # design (intercept-only mu + 1 covariate; intercept-only scale/rho)
    n = p
    x1 = randn(n)
    X1 = hcat(ones(n), x1); X2 = hcat(ones(n), x1)
    Xs1 = reshape(ones(n), n, 1); Xs2 = reshape(ones(n), n, 1); Xr = reshape(ones(n), n, 1)

    # fixed θ
    β = (mu1 = [0.5, 0.3], mu2 = [-0.2, 0.4], s1 = [-0.4], s2 = [-0.5], rho = [0.3])
    Λ = [0.30 0.10 0.05 0.0; 0.10 0.30 0.0 0.04; 0.05 0.0 0.12 0.02; 0.0 0.04 0.02 0.12]
    Λ = (Λ + Λ') / 2
    @test isposdef(Λ)
    Λinv = inv(Λ)

    # simulate leaf data from the model (so the likelihood is non-degenerate)
    Lc = cholesky(Λ).L; Sc = cholesky(Symmetric(Σ_phy)).U
    Uleaf = Lc * randn(4, p) * Sc                          # 4×p leaf random effects
    y1 = zeros(n); y2 = zeros(n)
    for i in 1:n
        m1 = (X1[i, :]'β.mu1) + Uleaf[1, i]; m2 = (X2[i, :]'β.mu2) + Uleaf[2, i]
        s1 = exp((Xs1[i, :]'β.s1) + Uleaf[3, i]); s2 = exp((Xs2[i, :]'β.s2) + Uleaf[4, i])
        ρ = D.RHO_GUARD * tanh(Xr[i, :]'β.rho)
        e = cholesky([s1^2 ρ*s1*s2; ρ*s1*s2 s2^2]).L * randn(2)
        y1[i] = m1 + e[1]; y2[i] = m2 + e[2]
    end

    # CONDITION ON THE ROOT (matches sigma_phy_dense): remove the root node so the
    # prior precision is full-rank PD and equivalent to the dense leaf covariance.
    keep = setdiff(1:n_total, [phy.root_index])
    n_keep = length(keep)                                  # 2p-2
    Q_cond = phy.Q_topology[keep, keep]
    pos_in_keep = Dict(node => i for (i, node) in enumerate(keep))
    leaf_node = [pos_in_keep[phy.leaf_indices[k]] for k in 1:p]

    # ============ (A) augmented sparse path ============
    prob = AugProblem(phy, n_keep, p, leaf_node, y1, y2, X1, X2, Xs1, Xs2, Xr)
    P = prior_precision(Q_cond, Λinv)
    u_aug, chH, H = estep_mode(prob, P, β; n_newton = 40)
    ll_aug = D.laplace_ll(prob, P, β, u_aug, chH)
    # extract augmented mode at leaf nodes (4×p)
    u_aug_leaf = zeros(4, p)
    for k in 1:p
        t = leaf_node[k]; u_aug_leaf[:, k] = u_aug[4(t - 1) + 1:4t]
    end

    # ============ (B) leaf-only DENSE Laplace (oracle) ============
    # latent v (4p), node-major over leaves: v[(i-1)*4+a]. prior precision
    # kron(Σ_inv, Λinv). data at leaf i.
    Pleaf = kron(Σ_inv, Λinv)                              # 4p×4p dense
    function leaf_joint_nll(v)
        val = 0.5 * dot(v, Pleaf * v)
        for i in 1:p
            b = 4(i - 1)
            val += D.leaf_nll((v[b + 1], v[b + 2], v[b + 3], v[b + 4]), y1[i], y2[i],
                              X1[i, :]'β.mu1, X2[i, :]'β.mu2, Xs1[i, :]'β.s1,
                              Xs2[i, :]'β.s2, Xr[i, :]'β.rho)
        end
        return val
    end
    function solve_leaf_mode()
        v = zeros(4p)
        for _ in 1:40
            g = ForwardDiff.gradient(leaf_joint_nll, v)
            Hd = ForwardDiff.hessian(leaf_joint_nll, v)
            step = Symmetric(Hd) \ g
            v -= step
            norm(step) < 1e-10 && break
        end
        return v
    end
    v = solve_leaf_mode()
    Hd = ForwardDiff.hessian(leaf_joint_nll, v)
    jn_leaf = leaf_joint_nll(v)
    ll_leaf = -jn_leaf - 0.5 * logdet(Symmetric(Hd)) + 0.5 * logdet(Symmetric(Pleaf))
    v_leaf = reshape(v, 4, p)

    # ============ compare ============
    # Checkpoint 3 (validated in the poc): mode matches to ~1e-6 and the
    # marginals agree exactly (the augmented/leaf-only parameterizations share
    # the same root-conditioned prior, so there is no residual constant here).
    @test maximum(abs.(u_aug_leaf .- v_leaf)) < 1e-8
    @test isapprox(ll_aug, ll_leaf; atol = 1e-6)
end
