# test_cholesky_guards.jl -- audit of bare `cholesky(...)` calls in src/.
#
# Two policies, one per call-site class:
#   * inside an optimiser objective: a near-singular probe must give a finite
#     penalty (or -Inf for a log-likelihood being maximised), never a raw
#     PosDefException that kills the optimiser;
#   * a post-fit summary / set-up step: a near-singular matrix must give a clear
#     ArgumentError naming the matrix, not "matrix is not positive definite".
# And for PD input the guarded code must return exactly what the bare call did.
# Fixtures use StableRNG or hand-written matrices (Julia-version independent).

using DRModels
using Test
using LinearAlgebra: Symmetric, cholesky, issuccess, I, PosDefException
using SparseArrays: sparse
using StableRNGs: StableRNG

const D = DRModels

_cg_msg(f) = try f(); "" catch e; sprint(showerror, e) end

@testset "_pd_chol_or_throw: PD identical, near-singular refused clearly" begin
    A = [4.0 1.0; 1.0 3.0]
    @test D._pd_chol_or_throw(A, "A").L == cholesky(Symmetric(A)).L          # bit-identical
    S = [1.0 1.0; 1.0 1.0]                                                    # singular
    @test_throws ArgumentError D._pd_chol_or_throw(S, "the test matrix")
    @test occursin("the test matrix", _cg_msg(() -> D._pd_chol_or_throw(S, "the test matrix")))
    @test occursin("positive definite", _cg_msg(() -> D._pd_chol_or_throw(S, "the test matrix")))
    @test_throws ArgumentError D._pd_chol_or_throw([1.0 NaN; NaN 1.0], "NaN matrix")
    @test_throws PosDefException cholesky(Symmetric(S))                       # what it used to do
end

@testset "location-only objective: non-PD factor is a penalty, not a crash" begin
    rng = StableRNG(20261108)
    G = 8
    phy = random_balanced_tree(G; branch_length = 0.25)
    species = repeat(1:G, inner = 2)
    n = length(species)
    X = hcat(ones(n), collect(range(-1.0, 1.0; length = n)))
    y = X * [0.25, -0.4] .+ 0.5 .* randn(rng, G)[species] .+ 0.4 .* randn(rng, n)
    prob = D.make_loc_problem(phy, y, X; species = species)
    β = [0.25, -0.4]; lσ = log(0.4); lσ_phy = log(0.5)

    # PD problem: unchanged, finite, equals -marginal_loglik.
    nll = D._loconly_marginal_nll(prob, β, lσ, lσ_phy)
    @test isfinite(nll) && nll < D._LOCONLY_PENALTY
    @test nll == -D.marginal_loglik(prob, β, exp(2lσ_phy), exp(2lσ))

    # A problem whose tip precision is negative definite: `build_M`'s last-resort
    # ridge retry cannot rescue it. (Real probes reach this only through extreme
    # scales / a broken precision; constructing it directly is the only
    # deterministic way to exercise the guard.)
    bad = D.LocOnlyProblem(prob.phy, prob.n_keep, prob.p, prob.n, -prob.Q_cond,
                           prob.leaf_pos, prob.species, prob.y, prob.X, prob.k,
                           prob.S, prob.STS_diag)
    @test_throws PosDefException D.marginal_loglik(bad, β, exp(2lσ_phy), exp(2lσ))  # public contract kept
    @test D._loconly_marginal_nll(bad, β, lσ, lσ_phy) == D._LOCONLY_PENALTY
    @test isfinite(D._loconly_marginal_nll(bad, β, lσ, lσ_phy))
end

# Small q = 4 coevolution problem (hand-rolled draws, StableRNG).
function _cg_q4_problem(rng; p = 6)
    phy = random_balanced_tree(p; branch_length = 0.2)
    n = p
    x1 = randn(rng, n)
    X1 = hcat(ones(n), x1); X2 = hcat(ones(n), x1)
    Xs1 = reshape(ones(n), n, 1); Xs2 = reshape(ones(n), n, 1); Xr = reshape(ones(n), n, 1)
    y1 = X1 * [1.0, 0.5] .+ 0.3 .* randn(rng, n)
    y2 = X2 * [-0.3, 0.4] .+ 0.3 .* randn(rng, n)
    prob, Q_cond = make_problem(phy, y1, y2, X1, X2, Xs1, Xs2, Xr)
    β = (mu1 = X1 \ y1, mu2 = X2 \ y2, s1 = [log(0.3)], s2 = [log(0.3)], rho = [0.0])
    return prob, Q_cond, β
end

@testset "fit_ml_q4 Λ objective: non-PD Λ scores -Inf; PD value unchanged" begin
    prob, Q_cond, β = _cg_q4_problem(StableRNG(20261108))
    Λ = Matrix(0.3 * I(4))
    ll = D.L_given_Λ(prob, Q_cond, β, Λ)
    # reference: the pre-guard formula, inline
    W = D.whitened_prior(Q_cond, cholesky(Symmetric(Λ)).L)
    v, ch, _ = D.estep_mode(prob, W, β; n_newton = 60)
    @test ll == D.laplace_ll(prob, W, β, v, ch)
    @test isfinite(ll)

    @test D.L_given_Λ(prob, Q_cond, β, ones(4, 4)) == -Inf                    # singular
    @test D.L_given_Λ(prob, Q_cond, β, fill(NaN, 4, 4)) == -Inf               # non-finite
    @test D.L_given_Λ(prob, Q_cond, β, Matrix(-1.0 * I(4))) == -Inf           # indefinite

    # Line search from a near-singular (PD) start must not throw.
    Λnear = D.lc_to_Λ([fill(-20.0, 4); zeros(6)])
    Λn, Ln = D.lambda_ml_step(prob, Q_cond, β, Λnear)
    @test size(Λn) == (4, 4)
    @test !(Ln > 0 && !isfinite(Ln))                                          # a number or -Inf, no exception

    @test_throws ArgumentError D.fit_ml(prob, Q_cond, β, ones(4, 4); verbose = false)
end

@testset "set-up / summary steps refuse a near-singular matrix clearly" begin
    # sparse precision square-root factor (two-structured objective set-up)
    @test_throws ArgumentError D._prec_sqrt_factor(sparse([1.0 1.0; 1.0 1.0]))
    Qpd = sparse([2.0 -0.5; -0.5 2.0])
    B = D._prec_sqrt_factor(Qpd)
    @test Matrix(B' * B) ≈ Matrix(Qpd) atol = 1e-12

    # Fisher-z q = 4 wrapper: a fully collapsed Σ_a (all axis SDs underflow to 0)
    φ_collapsed = vcat(fill(-400.0, 4), zeros(6))
    @test_throws ArgumentError D.fz_phi_to_lc(φ_collapsed)
    @test occursin("among-axis covariance", _cg_msg(() -> D.fz_phi_to_lc(φ_collapsed)))
    φ_ok = vcat(log.([0.5, 0.4, 0.3, 0.3]), 0.1 .* ones(6))
    @test all(isfinite, D.fz_phi_to_lc(φ_ok))

    # coevolution simulator: non-PD Λ
    phy = random_balanced_tree(5; branch_length = 0.2)
    β = zeros(2, 4)
    @test_throws ArgumentError simulate_coevolution(phy, β, ones(4, 4), fill(0.3, 4);
                                                   nrep = 1, rng = StableRNG(1))
end

@testset "meta_V + relmat: singular user K is refused clearly" begin
    rng = StableRNG(20261108)
    G = 6; m = 3
    id = repeat(string.("id", 1:G), inner = m)
    n = G * m
    x = randn(rng, n)
    v = 0.05 .+ 0.2 .* rand(rng, n)
    y = 0.3 .+ 0.5 .* x .+ 0.3 .* randn(rng, n)
    d = (; y, x, v, id)
    Ksing = ones(G, G)                                                        # rank 1, PSD but singular
    f = bf(@formula(y ~ x + meta_V(v) + relmat(1 | id)), @formula(sigma ~ 1))
    err = try drm(f, Gaussian(); data = d, K = Ksing); nothing catch e; e end
    @test err isa Exception
    @test !(err isa PosDefException)
    @test err isa ArgumentError || occursin("positive definite", sprint(showerror, err))
end
