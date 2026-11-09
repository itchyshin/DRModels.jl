# DJ1: uncovered q=4 helpers that open drafts are not rewriting.
# Skipped on purpose: `fit_q4_sparse_tmb` (#1270) and `fit_q4_reml` (#1285).
using DRModels, Test, Random, LinearAlgebra, SparseArrays, Statistics

function _dj1_q4_problem(; p::Int = 4, nrep::Int = 2, seed::Int = 11, missing_row::Bool = false)
    rng = MersenneTwister(seed)
    phy = random_balanced_tree(p; branch_length = 0.2)
    keep = setdiff(1:phy.n_total, [phy.root_index])
    Q_cond = phy.Q_topology[keep, keep]
    beta = (mu1 = [0.4, 0.15], mu2 = [-0.2, 0.1], s1 = [-0.4], s2 = [-0.5], rho = [0.1])
    Sigma_a = Matrix(Symmetric([
        0.20 0.03 0.01 0.00
        0.03 0.16 0.00 0.01
        0.01 0.00 0.07 0.01
        0.00 0.01 0.01 0.06
    ]))
    P = prior_precision(Q_cond, inv(Sigma_a))
    F = cholesky(Symmetric(P))
    u_aug = F.UP \ randn(rng, size(P, 1))
    pos = Dict(node => i for (i, node) in enumerate(keep))
    leaf_pos = [pos[phy.leaf_indices[k]] for k in 1:p]
    species = repeat(1:p, inner = nrep)
    n = length(species)
    x = randn(rng, n)
    X1 = hcat(ones(n), x)
    X2 = hcat(ones(n), x)
    Xs1 = ones(n, 1)
    Xs2 = ones(n, 1)
    Xr = ones(n, 1)
    y1 = Vector{Float64}(undef, n)
    y2 = similar(y1)
    @inbounds for i in 1:n
        u = @view u_aug[(4 * (leaf_pos[species[i]] - 1) + 1):(4 * leaf_pos[species[i]])]
        mu1 = dot(X1[i, :], beta.mu1) + u[1]
        mu2 = dot(X2[i, :], beta.mu2) + u[2]
        s1 = exp(dot(Xs1[i, :], beta.s1) + u[3])
        s2 = exp(dot(Xs2[i, :], beta.s2) + u[4])
        rho = DRModels.RHO_GUARD * tanh(dot(Xr[i, :], beta.rho))
        e = cholesky(Symmetric([s1^2 rho * s1 * s2; rho * s1 * s2 s2^2])).L * randn(rng, 2)
        y1[i] = mu1 + e[1]
        y2[i] = mu2 + e[2]
    end
    if missing_row
        y2[1] = NaN
    end
    prob, Q_fit = make_problem(phy, y1, y2, X1, X2, Xs1, Xs2, Xr; species = species)
    return prob, Q_fit, beta, Sigma_a
end

@testset "joint_nll_T and joint_grad_T match the Float64 joint and its u-gradient" begin
    prob, Q_cond, beta, Sigma_a = _dj1_q4_problem(; missing_row = true)
    P = sparse(prior_precision(Q_cond, inv(Sigma_a)))
    u = zeros(4 * prob.n_total)
    u[1:4] .= (0.1, -0.2, 0.05, -0.04)
    val_t = DRModels.joint_nll_T(prob, P, u, beta)
    val = joint_nll(prob, P, u, beta)
    @test val_t ≈ val atol = 1e-10 rtol = 1e-10
    @test isfinite(val_t)
    g = DRModels.joint_grad_T(prob, P, u, beta)
    @test g ≈ joint_grad(prob, P, u, beta) atol = 1e-10 rtol = 1e-10
    # `joint_nll_T` freezes `u` as Float64, so check one coordinate by a
    # central difference of the Float64 joint negative log-density.
    h = 1e-6
    up = copy(u); up[3] += h
    um = copy(u); um[3] -= h
    fd3 = (joint_nll(prob, P, up, beta) - joint_nll(prob, P, um, beta)) / (2h)
    @test g[3] ≈ fd3 atol = 1e-6 rtol = 1e-6
    @test length(g) == length(u)
end

@testset "q4 marginal diagnostic reports an unpack failure and a non-finite joint" begin
    prob, Q_cond, beta, Sigma_a = _dj1_q4_problem(; nrep = 1)
    theta = pack_theta(beta, Sigma_a)
    short = DRModels.q4_marginal_diagnostic(prob, Q_cond, theta[1:2])
    @test short.ok == false
    @test short.nll == Inf
    @test short.loglik == -Inf
    @test short.first_nonfinite.stage == :unpack_theta
    @test short.first_nonfinite.ok == false
    @test occursin("BoundsError", short.first_nonfinite.message)

    # A finite but enormous log-Cholesky entry is already covered as
    # :among_axis_covariance. A NaN placed past `unpack` is not what this
    # guards; the unpack path above is the one with no caller.
    good = DRModels.q4_marginal_diagnostic(prob, Q_cond, theta; n_newton = 15, gradient = false)
    @test good.ok
    @test good.nll ≈ marginal_nll(prob, Q_cond, theta; n_newton = 15)[1] atol = 1e-8
end

@testset "REML phi layout and lifted mean design" begin
    prob, Q_cond, beta, Sigma_a = _dj1_q4_problem()
    ks1, ks2, kr = DRModels.phi_widths(prob)
    @test (ks1, ks2, kr) == (size(prob.Xs1, 2), size(prob.Xs2, 2), size(prob.Xr, 2))
    @test DRModels.phi_len(prob) == kr + 10
    phi = DRModels.pack_phi(prob, beta.rho, Sigma_a)
    @test length(phi) == DRModels.phi_len(prob)
    rho, lc = DRModels.unpack_phi(prob, phi)
    @test rho ≈ beta.rho atol = 1e-12
    @test lc_to_Λ(lc) ≈ Sigma_a atol = 1e-10

    Xμ = DRModels.build_Xmu_lifted(prob)
    k1 = size(prob.X1, 2)
    k2 = size(prob.X2, 2)
    @test size(Xμ) == (4 * prob.n_total, k1 + k2)
    # Only the two mean axes of each leaf are loaded, and replicate rows add.
    acc = zeros(size(Xμ))
    for i in eachindex(prob.leaf_node)
        base = 4 * (prob.leaf_node[i] - 1)
        acc[base + 1, 1:k1] .+= prob.X1[i, :]
        acc[base + 2, k1 + 1:k1 + k2] .+= prob.X2[i, :]
    end
    @test Matrix(Xμ) ≈ acc atol = 1e-12
    @test nnz(Xμ) == nnz(sparse(acc))

    # The SparseMatrixCSC method is the identity on the latent coordinate.
    u = randn(MersenneTwister(3), 4 * prob.n_total)
    B1, D1 = DRModels._reml_border_blocks(prob, u, beta)
    B2, D2 = DRModels._reml_border_blocks(prob, Q_cond, u, beta)
    @test B1 ≈ B2 atol = 1e-12
    @test D1 ≈ D2 atol = 1e-12
end

@testset "REML cold start (no beta0) returns a finite restricted likelihood" begin
    # p = 4 with one replicate does not settle (the documented non-PD barrier
    # returns -Inf and inner_converged is false). Four replicates does settle.
    prob, Q_cond, beta, Sigma_a = _dj1_q4_problem(; p = 4, nrep = 4, seed = 5)
    phi = DRModels.pack_phi(prob, beta.rho, Sigma_a)
    ll, u_hat, ch_H, beta_hat, P, inner_ok = DRModels.reml_ll_and_mode(
        prob, Q_cond, phi; beta0 = nothing, n_newton = 20)
    @test isfinite(ll)
    @test inner_ok
    @test length(u_hat) == 4 * prob.n_total
    @test all(isfinite, u_hat)
    @test length(beta_hat.mu1) == size(prob.X1, 2)
    @test length(beta_hat.s1) == size(prob.Xs1, 2)
    @test ch_H !== nothing
    # Restricted ll is the Laplace marginal minus ½ logdet(S). On this seed
    # the gap was 10.36, so the correction is present and S is PD.
    ml = DRModels.laplace_ll(prob, P, beta_hat, DRModels.u_to_whitened(P, u_hat), ch_H)
    @test isfinite(ml)
    @test ml - ll > 1
end

@testset "closed-form EM on a small signalled tree ascends and reports the Laplace marginal" begin
    prob, Q_cond, beta, Sigma_a = _dj1_q4_problem(; p = 4, nrep = 2, seed = 8)
    β0 = (mu1 = prob.X1 \ prob.y1, mu2 = prob.X2 \ prob.y2,
          s1 = [log(std(prob.y1 .- prob.X1 * (prob.X1 \ prob.y1)))],
          s2 = [log(std(prob.y2 .- prob.X2 * (prob.X2 \ prob.y2)))],
          rho = [0.0])
    res = DRModels.fit_em_aug(prob, Q_cond, β0, Matrix(0.3 * I(4));
                              max_em = 8, tol = 1e-4, verbose = false)
    @test isfinite(res.loglik)
    @test res.iters >= 1
    @test length(res.ll_hist) == res.iters + 1
    @test all(diff(res.ll_hist) .>= -1e-7)
    @test res.ll_hist[end] ≈ res.loglik atol = 1e-8
    @test isposdef(Symmetric(res.Λ))
    # Same prior the EM driver uses (`prior_precision`, not the whitened form).
    P = prior_precision(Q_cond, inv(res.Λ))
    u, ch, _ = estep_mode(prob, P, res.β; n_newton = 40)
    ll = DRModels.laplace_ll(prob, P, res.β, u, ch)
    # Recomputed on the same non-whitened path. Measured gap on this seed
    # was 6e-11.
    @test res.loglik ≈ ll atol = 1e-8 rtol = 1e-8
end
