# test_coevo_q6.jl — recovery gate for the GENERAL-q coevolution block (#188).
#
# Generalises the among-trait coevolution covariance Λ (= Σ_a) from the q=4
# bivariate location-scale PLSM to arbitrary q, on the SAME sparse augmented-
# state precision P = kron(Q_tree, Λ⁻¹). The model here is the canonical
# multivariate-Brownian coevolution (q Gaussian traits, shared q×q Λ on the
# tree, diagonal residual) — the part of the model that genuinely generalises;
# see report / src/coevolution_q.jl for why the q=4 LOCATION-SCALE leaf
# (2 means + 2 log-σ + one residual ρ) is intrinsically bivariate and does not.
#
# Asserts at q=6 (and a q=8 shape/identification smoke): recover the among-trait
# SDs, the strong correlations (sign + magnitude), the residual SDs, and the
# overall Λ (Frobenius backstop). Tolerances are set with margin over a 5-seed
# spread (correlations have SE ∝ 1/√p, so weak/zero entries recover only loosely
# at this p — the same honest pattern the q=4 recovery gate settled on).

using DRModels
using Test, LinearAlgebra, Random, Statistics, SparseArrays
using Distributions: MvNormal, logpdf

# True among-trait structure: real correlations (strong, weak, zero, negative)
# so recovery is a genuine test, not a diagonal special case.
const _COEVO6_SD = [0.8, 0.7, 0.9, 0.6, 0.75, 0.85]
const _COEVO6_R = [
    1.0   0.6   0.4   0.0   0.2  -0.3
    0.6   1.0   0.3  -0.2   0.0   0.1
    0.4   0.3   1.0   0.5   0.1   0.0
    0.0  -0.2   0.5   1.0   0.3   0.2
    0.2   0.0   0.1   0.3   1.0   0.4
   -0.3   0.1   0.0   0.2   0.4   1.0
]
_coevo6_Lambda() = Matrix(Symmetric(Diagonal(_COEVO6_SD) * _COEVO6_R * Diagonal(_COEVO6_SD)))

function _coevo6_truth_beta()
    q = 6
    β = zeros(2, q)
    β[1, :] = collect(range(-0.5, 0.5; length = q))   # per-trait intercepts
    β[2, :] = collect(range(0.3, -0.3; length = q))   # per-trait slope on x
    return β
end

@testset "general-q log-Cholesky (q=6) round-trips + matches q=4 engine" begin
    rng = MersenneTwister(1)
    for q in (2, 4, 6, 8)
        @test lc_len(q) == (q * (q + 1)) ÷ 2
        A = randn(rng, q, q)
        S = Matrix(Symmetric(A * A' + 0.5I))
        v = cov_to_lc(S)
        @test length(v) == lc_len(q)
        @test lc_to_cov(v, q) ≈ S
        @test isposdef(Symmetric(lc_to_cov(v, q)))
    end
    # The q=4 convention must coincide with the verified engine's lc_to_Λ/Λ_to_lc.
    B = randn(rng, 4, 4)
    S4 = Matrix(Symmetric(B * B' + 0.3I))
    v4 = cov_to_lc(S4)
    @test v4 ≈ DRModels.Λ_to_lc(S4)
    @test lc_to_cov(v4, 4) ≈ DRModels.lc_to_Λ(v4)
end

@testset "general-q coevolution marginal is exact (Gaussian)" begin
    # The conjugate-Gaussian Laplace marginal must equal the closed-form
    # multivariate-normal marginal logpdf of the stacked observations.
    rng = MersenneTwister(42)
    q = 4                                  # small p,q so the dense check is cheap
    p = 6
    phy = random_balanced_tree(p; branch_length = 0.2)
    Λ = Matrix(Symmetric([
        0.5 0.1 0.0 0.05
        0.1 0.4 0.08 0.0
        0.0 0.08 0.6 0.1
        0.05 0.0 0.1 0.45
    ]))
    σ_res = [0.3, 0.4, 0.25, 0.35]
    # k = 2 (intercept + slope), matching the simulator's X = [1  x].
    β = zeros(2, q)
    β[1, :] = range(-0.4, 0.4; length = q)
    β[2, :] = range(0.25, -0.25; length = q)
    sim = simulate_coevolution(phy, β, Λ, σ_res; nrep = 1, rng = rng)
    prob, Qc = make_coevo_problem(phy, sim.Y, sim.X; species = sim.species)

    ℓ_lap, = coevo_marginal(prob, Qc, β, Λ, σ_res)

    # Closed form: stacked y (n·q) ~ N(Xβ stacked, Z (Σ_phy ⊗ Λ) Z' + I_n ⊗ D),
    # Σ_phy = leaf covariance of the tree, Z maps tips→rows (here identity, 1/sp).
    Σ_phy = sigma_phy_dense(phy; σ²_phy = 1.0)              # p × p
    n = size(sim.Y, 1)
    # mean
    μ = similar(sim.Y)
    for a in 1:q
        μ[:, a] = sim.X * β[:, a]
    end
    # covariance over the n·q stacked vector in trait-major (row i, trait a):
    # cov((i,a),(j,b)) = Σ_phy[sp(i),sp(j)] * Λ[a,b] + (i==j) D[a,b]
    D = Diagonal(σ_res .^ 2)
    sp = sim.species
    C = zeros(n * q, n * q)
    idx(i, a) = (i - 1) * q + a
    for i in 1:n, a in 1:q, j in 1:n, b in 1:q
        v = Σ_phy[sp[i], sp[j]] * Λ[a, b]
        if i == j
            v += D[a, b]
        end
        C[idx(i, a), idx(j, b)] = v
    end
    yv = zeros(n * q); mv = zeros(n * q)
    for i in 1:n, a in 1:q
        yv[idx(i, a)] = sim.Y[i, a]; mv[idx(i, a)] = μ[i, a]
    end
    ℓ_dense = logpdf(MvNormal(mv, Matrix(Symmetric(C))), yv)
    @test ℓ_lap ≈ ℓ_dense rtol = 1e-8
end

@testset "q=6 coevolution recovers Λ (SDs, strong correlations, residuals)" begin
    q = 6
    p = 120
    nrep = 5
    seed = 2024
    Λ = _coevo6_Lambda()
    βt = _coevo6_truth_beta()
    σt = fill(0.3, q)

    phy = random_balanced_tree(p; branch_length = 0.2)
    sim = simulate_coevolution(phy, βt, Λ, σt; nrep = nrep, rng = MersenneTwister(seed))
    prob, Qc = make_coevo_problem(phy, sim.Y, sim.X; species = sim.species)
    @test prob.q == 6
    @test coevo_theta_len(prob) == 2 * q + (q * (q + 1)) ÷ 2 + q

    fit = fit_coevolution(prob, Qc)
    @test fit.converged
    @test size(fit.Λ) == (6, 6)
    @test isposdef(Symmetric(fit.Λ))

    sd̂ = sqrt.(diag(fit.Λ))
    Ĉ = fit.Λ ./ (sd̂ * sd̂')

    # (1) among-trait SDs: SE ∝ 1/√p, recovered to ~15% at p=120 (margin over the
    #     5-seed spread, worst seed 0.139).
    @test sd̂ ≈ _COEVO6_SD rtol = 0.2

    # (2) residual SDs: the diagonal residual is sharply identified (err ≤ 0.03).
    @test fit.σ_res ≈ σt atol = 0.06

    # (3) strong among-trait correlations: correct sign + magnitude. These are the
    #     coevolution signal the model exists to estimate.
    strong = ((1, 2, 0.6), (3, 4, 0.5), (5, 6, 0.4), (1, 3, 0.4), (1, 6, -0.3))
    for (i, j, ρ) in strong
        @test sign(Ĉ[i, j]) == sign(ρ)
        @test Ĉ[i, j] ≈ ρ atol = 0.3
    end

    # (4) overall Λ: Frobenius backstop (worst seed 0.235 rel; 0.45 leaves margin).
    @test norm(fit.Λ - Λ) ≤ 0.45 * norm(Λ)

    # (5) mean structure: per-trait slopes recover.
    @test fit.β[2, :] ≈ βt[2, :] atol = 0.15
end

@testset "q=8 coevolution: shape + identification smoke" begin
    # q=8 exercises the SAME code path at a larger among-trait dimension. A full,
    # converged 8×8 correlation recovery needs many tips and is expensive, so this
    # is a deliberately small, iteration-capped smoke: assert the machinery is
    # genuinely q=8 (Λ is 8×8 PD, θ has the right length), and that a short fit
    # improves the marginal and recovers the SDs + residual scale loosely. (The
    # full recovery bar is exercised at q=6 above.)
    q = 8
    p = 50
    nrep = 4
    rng = MersenneTwister(808)
    sds = collect(range(0.5, 1.2; length = q))
    A = randn(rng, q, q)
    R = let M = A * A'; d = sqrt.(diag(M)); M ./ (d * d'); end   # a valid corr matrix
    Λ = Matrix(Symmetric(Diagonal(sds) * R * Diagonal(sds)))
    @test isposdef(Symmetric(Λ))
    β = zeros(2, q); β[1, :] = range(-0.6, 0.6; length = q); β[2, :] = range(0.2, -0.2; length = q)
    σt = fill(0.3, q)

    phy = random_balanced_tree(p; branch_length = 0.2)
    sim = simulate_coevolution(phy, β, Λ, σt; nrep = nrep, rng = MersenneTwister(808))
    prob, Qc = make_coevo_problem(phy, sim.Y, sim.X; species = sim.species)
    @test prob.q == 8
    @test coevo_theta_len(prob) == 2 * q + (q * (q + 1)) ÷ 2 + q
    @test length(coevo_pack(β, Λ, σt)) == coevo_theta_len(prob)

    ℓ0, = coevo_marginal(prob, Qc, prob.X \ prob.Y, Matrix(0.2I(q)), fill(0.5, q))
    fit = fit_coevolution(prob, Qc; iterations = 120, g_tol = 1e-4)
    @test size(fit.Λ) == (8, 8)
    @test isposdef(Symmetric(fit.Λ))                      # stays a valid covariance
    @test fit.loglik > ℓ0                                  # a short fit improved the marginal
    sd̂ = sqrt.(diag(fit.Λ))
    @test sd̂ ≈ sds rtol = 0.3                             # among-trait SDs recover (loose)
    @test fit.σ_res ≈ σt atol = 0.1                       # residual scale recovers
end
