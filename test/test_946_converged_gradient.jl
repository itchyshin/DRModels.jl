# test_946_converged_gradient.jl — DRModels.jl#946.
#
# The q=4 ML / Fisher-z / REML fitters and the coevolution q / q=2 residual /
# q=2 REML fitters let Optim stop on a relative objective change (`f_reltol`).
# `Optim.converged` is the OR of the x, f and g criteria, so those routes reported
# `converged = true` at an f-plateau whatever the gradient. The flag must require
# the gradient: the absolute g_tol test or #983's unit-free Newton decrement, on a
# gradient re-evaluated at the minimizer, never on a budget stop. Each forcing
# case raises `f_reltol` so Optim stops early with a gradient far above g_tol;
# each control is the same fit at the route's default stop and still converges.
#
# The three finite-difference routes also zero-filled a gradient coordinate whose
# probe was not finite, which counts toward g_tol; the verdict gradient now
# propagates NaN instead (helper contracts at the end).
#
#   julia --project=. -e 'using DRModels, Test; include("test/test_946_converged_gradient.jl")'

module Test946ConvergedGradient

using DRModels
using Test, LinearAlgebra, Random, SparseArrays, Statistics
const Optim = DRModels.Optim

# --- q=2 coevolution fixture (generator of test_reml_q2_structured.jl) -------
function _q2_problem(seed; G = 14, nrep = 4)
    rng = MersenneTwister(seed)
    K = [0.55^abs(i - j) for i in 1:G, j in 1:G] + 1e-6I
    β = [0.20 -0.15; 0.25 0.10]
    Λ = Matrix(Symmetric([0.20 0.05; 0.05 0.17]))
    R = Matrix(Symmetric([0.10 0.025; 0.025 0.14]))
    Q = sparse(Matrix(inv(cholesky(Symmetric(K)))))
    P = DRModels.prior_precision(Q, inv(Λ))
    u = cholesky(Symmetric(P)).UP \ randn(rng, size(P, 1))
    group = repeat(1:G, inner = nrep)
    n = length(group)
    X = hcat(ones(n), randn(rng, n))
    L = cholesky(Symmetric(R)).L
    Y = zeros(n, 2)
    for i in 1:n
        b = 2 * (group[i] - 1)
        Y[i, 1] = X[i, :]' * β[:, 1] + u[b + 1]
        Y[i, 2] = X[i, :]' * β[:, 2] + u[b + 2]
        Y[i, :] .+= L * randn(rng, 2)
    end
    return DRModels.make_coevo_problem_from_covariance(K, Y, X; group = group)
end

# Independent check, not the code under test: central-FD gradient of the MEAN
# restricted objective at the fitted (Λ, σ, ρ), rebuilt from the public fields.
function _q2_reml_mean_grad(prob, Q, fit; h = 1e-6)
    n = length(prob.leaf_node)
    f = function (phi)
        Λ, D, chΛ = DRModels._q2_reml_unpack_phi(phi)
        rv, = DRModels._q2_reml_ll(prob, Q, Λ, D, chΛ, prob.X \ prob.Y)
        return -rv / n
    end
    phi = DRModels._q2_reml_pack_phi(fit.Λ, fit.σ_res, fit.rho12)
    return [(f(phi .+ h .* (eachindex(phi) .== k)) - f(phi .- h .* (eachindex(phi) .== k))) / 2h
            for k in eachindex(phi)]
end

# --- small well-posed q=4 phylo fixture (nrep = 4 per species) ---------------
function _q4_problem(seed; p = 16, nrep = 4)
    rng = MersenneTwister(seed)
    phy = random_balanced_tree(p; branch_length = 0.2)
    Σ_phy = sigma_phy_dense(phy; σ²_phy = 1.0)
    Λt = [0.25 0.10 0.05 0.0; 0.10 0.25 0.0 0.04; 0.05 0.0 0.09 0.02; 0.0 0.04 0.02 0.09]
    U = cholesky(Symmetric(Λt)).L * randn(rng, 4, p) * cholesky(Symmetric(Σ_phy)).U
    species = repeat(1:p, inner = nrep)
    n = length(species)
    x = randn(rng, n)
    X = hcat(ones(n), x)
    y1 = zeros(n); y2 = zeros(n)
    for i in 1:n
        s = species[i]
        s1 = exp(-0.4 + U[3, s]); s2 = exp(-0.5 + U[4, s]); ρ = tanh(0.3)
        e = cholesky([s1^2 ρ*s1*s2; ρ*s1*s2 s2^2]).L * randn(rng, 2)
        y1[i] = 1.0 + 0.5x[i] + U[1, s] + e[1]
        y2[i] = -0.3 + 0.4x[i] + U[2, s] + e[2]
    end
    prob, Q = make_problem(phy, y1, y2, X, copy(X), ones(n, 1), ones(n, 1), ones(n, 1);
                           species = species)
    β0 = (mu1 = X \ y1, mu2 = X \ y2, s1 = [log(std(y1 .- X * (X \ y1)))],
          s2 = [log(std(y2 .- X * (X \ y2)))], rho = [0.0])
    Λ0 = Matrix(0.3I(4)) + 0.05 * (ones(4, 4) - I)
    return prob, Q, β0, Λ0
end

@testset "#946 converged requires the gradient on the f_reltol routes" begin

    @testset "#946 q2 REML: an f_reltol stop with a large gradient is not converged" begin
        prob, Q = _q2_problem(20261016)
        g_tol = 1e-6
        early = DRModels.fit_coevolution_q2_reml(prob, Q; iterations = 300, g_tol = g_tol,
                                                 f_reltol = 1e-2)
        # Forcing condition, checked independently of the route: Optim stopped on the
        # f criterion with the gradient three orders of magnitude above g_tol.
        @test maximum(abs, _q2_reml_mean_grad(prob, Q, early)) > 1e3 * g_tol
        @test early.converged == false

        full = DRModels.fit_coevolution_q2_reml(prob, Q; iterations = 300, g_tol = g_tol)
        @test full.converged
        @test maximum(abs, _q2_reml_mean_grad(prob, Q, full)) < 1e1 * g_tol
        @test full.reml_loglik - early.reml_loglik > 1e-4   # the early stop was short
        @test early.g_residual > 1e3 * g_tol                # the reported residual says so
    end

    @testset "#946 coevolution q2 residual and general-q: same rule" begin
        prob, Q = _q2_problem(20261016)
        for fitfn in (DRModels.fit_coevolution_q2_residual, DRModels.fit_coevolution)
            early = fitfn(prob, Q; iterations = 300, g_tol = 1e-6, f_reltol = 1e-2)
            full = fitfn(prob, Q; iterations = 300, g_tol = 1e-6)
            @test early.converged == false
            @test early.g_residual > 1e3 * 1e-6
            @test full.converged
            @test full.loglik - early.loglik > 1e-4
        end
    end

    @testset "#946 q4 ML and Fisher-z: an f_reltol stop with a large gradient" begin
        prob, Q, β0, Λ0 = _q4_problem(20261016)
        g_tol = 1e-3
        for fitfn in (fit_q4_sparse_tmb, fit_q4_sparse_fisherz)
            early = fitfn(prob, Q; β0 = β0, Λ0 = Λ0, g_tol = g_tol, f_reltol = 0.1)
            full = fitfn(prob, Q; β0 = β0, Λ0 = Λ0, g_tol = g_tol)
            @test early.g_residual > 10 * g_tol
            @test early.converged == false
            @test full.converged
            @test full.loglik - early.loglik > 1e-2
        end
    end

    @testset "#946 q4 REML: the outer flag judges the caller's g_tol" begin
        prob, Q, β0, Λ0 = _q4_problem(20261016)
        # Every REML run stops on the f criterion (f_reltol = 0.9) short of g_tol, and
        # the #497 rescue / #575 polish adopt those f-converged runs; the flag used to
        # be `Optim.converged(res) && inner_ok`.
        early = DRModels.fit_q4_reml(prob, Q; beta0 = β0, Lambda0 = Λ0, g_tol = 1e-5,
                                     f_reltol = 0.9)
        @test early.g_residual > 10 * 1e-5
        @test early.converged == false
        full = DRModels.fit_q4_reml(prob, Q; beta0 = β0, Lambda0 = Λ0, g_tol = 1e-3)
        @test full.converged
        @test full.g_residual < 1e-3
    end

    @testset "#946 helper contracts" begin
        # _fd_gradient!: a non-finite probe is NaN for the verdict (never 0); the
        # optimiser's one-sided form keeps a genuine slope from the finite side.
        wall(x) = x[1] > 1.0 ? Inf : x[1]^2 + 3x[2]^2
        θ = [1.0 - 1e-7, 0.5]
        g = DRModels._fd_gradient!(zeros(2), wall, θ, 1e-6)
        @test isnan(g[1])
        @test g[2] ≈ 3.0 atol = 1e-6
        g1 = DRModels._fd_gradient!(zeros(2), wall, θ, 1e-6; one_sided = true)
        @test g1[1] ≈ 2.0 atol = 1e-4
        @test g1[2] == g[2]
        @test all(isnan, DRModels._fd_gradient!(zeros(2), x -> Inf, θ, 1e-6; one_sided = true))
        # all probes finite: the plain central difference, as the inline loops computed
        q(x) = (x[1] - 0.3)^2 + 2x[2]^2
        @test DRModels._fd_gradient!(zeros(2), q, [1.0, 1.0], 1e-6) ==
              [(q([1.0 + 1e-6, 1.0]) - q([1.0 - 1e-6, 1.0])) / 2e-6,
               (q([1.0, 1.0 + 1e-6]) - q([1.0, 1.0 - 1e-6])) / 2e-6]

        # _gradient_verified on real Optim results (pattern of test_sweep_blocked_944.jl).
        good = Optim.optimize(x -> (x[1] - 0.3)^2, [1.0], Optim.LBFGS(), Optim.Options(g_tol = 1e-8))
        plateau = deepcopy(good)
        plateau.stopped_by = merge(plateau.stopped_by, (f_converged = true, g_converged = false))
        capped = deepcopy(plateau)
        capped.stopped_by = merge(capped.stopped_by, (iterations = true,))
        timed = deepcopy(plateau)
        timed.stopped_by = merge(timed.stopped_by, (time_limit = true,))
        pd = () -> ([2.0 0.0; 0.0 2.0], true)
        n = 100
        calls = Ref(0)
        counted = () -> (calls[] += 1; pd())
        # absolute branch: mean-scale max |g| = |g|∞ / n against g_tol; no Hessian built
        @test DRModels._gradient_verified(good, [1e-7, 0.0], n, 1e-8, counted)
        @test calls[] == 0
        # f-plateau, large gradient: large decrement -> not converged
        @test !DRModels._gradient_verified(plateau, [1.0, 1.0], n, 1e-8, pd)
        # f-plateau, tiny decrement (0.5 * 2e-12 / 2 < n * 1e-10): stationary -> converged
        @test DRModels._gradient_verified(plateau, [1e-6, 1e-6], n, 1e-12, pd)
        # never on a budget stop, never without a Hessian, never with NaN or non-PD H
        @test !DRModels._gradient_verified(capped, [1e-6, 1e-6], n, 1e-12, pd)
        @test !DRModels._gradient_verified(timed, [1e-6, 1e-6], n, 1e-12, pd)
        @test !DRModels._gradient_verified(plateau, [1e-6, 1e-6], n, 1e-12, nothing)
        @test !DRModels._gradient_verified(good, [NaN, 0.0], n, 1e-8, pd)
        @test !DRModels._gradient_verified(plateau, [1e-6, 1e-6], n, 1e-12,
                                           () -> ([1.0 2.0; 2.0 1.0], true))
        @test !DRModels._gradient_verified(plateau, [1e-6, 1e-6], n, 1e-12,
                                           () -> (zeros(2, 2), false))
    end

end

end # module Test946ConvergedGradient
