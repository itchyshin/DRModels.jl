using DRModels, Test, Random, LinearAlgebra

function _q4_diagnostic_problem(; p::Int = 8, nrep::Int = 2, seed::Int = 293)
    rng = MersenneTwister(seed)
    phy = random_balanced_tree(p; branch_length = 0.2)
    keep = setdiff(1:phy.n_total, [phy.root_index])
    Q_cond = phy.Q_topology[keep, keep]

    beta = (
        mu1 = [0.7, 0.2],
        mu2 = [-0.3, 0.1],
        s1 = [-0.5],
        s2 = [-0.6],
        rho = [0.2],
    )
    Sigma_a = Matrix(Symmetric([
        0.20 0.04 0.02 0.00
        0.04 0.18 0.00 0.02
        0.02 0.00 0.08 0.01
        0.00 0.02 0.01 0.07
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
        sp = species[i]
        u = @view u_aug[(4 * (leaf_pos[sp] - 1) + 1):(4 * leaf_pos[sp])]
        mu1 = dot(X1[i, :], beta.mu1) + u[1]
        mu2 = dot(X2[i, :], beta.mu2) + u[2]
        s1 = exp(dot(Xs1[i, :], beta.s1) + u[3])
        s2 = exp(dot(Xs2[i, :], beta.s2) + u[4])
        rho = DRModels.RHO_GUARD * tanh(dot(Xr[i, :], beta.rho))
        e = cholesky(Symmetric([s1^2 rho * s1 * s2; rho * s1 * s2 s2^2])).L *
            randn(rng, 2)
        y1[i] = mu1 + e[1]
        y2[i] = mu2 + e[2]
    end

    prob, Q_fit = make_problem(phy, y1, y2, X1, X2, Xs1, Xs2, Xr; species = species)
    return prob, Q_fit, pack_theta(beta, Sigma_a)
end

@testset "q4 ML objective diagnostic reports first non-finite component (#293)" begin
    prob, Q_cond, theta0 = _q4_diagnostic_problem()

    diag = DRModels.q4_marginal_diagnostic(prob, Q_cond, theta0;
                                      n_newton = 20, gradient = true)
    @test diag.ok
    @test diag.first_nonfinite === nothing
    @test isfinite(diag.nll)
    @test isfinite(diag.loglik)
    @test :laplace_loglik in [row.stage for row in diag.stages]
    nll, _, _, _ = marginal_nll(prob, Q_cond, theta0; n_newton = 20)
    @test diag.nll ≈ nll atol = 1e-8

    theta_nan = copy(theta0)
    theta_nan[1] = NaN
    diag_nan = DRModels.q4_marginal_diagnostic(prob, Q_cond, theta_nan)
    @test !diag_nan.ok
    @test diag_nan.first_nonfinite.stage == :theta

    theta_overflow = copy(theta0)
    theta_overflow[end] = 1000.0
    diag_overflow = DRModels.q4_marginal_diagnostic(prob, Q_cond, theta_overflow)
    @test !diag_overflow.ok
    @test diag_overflow.first_nonfinite.stage == :among_axis_covariance
end
