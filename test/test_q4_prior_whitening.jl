# test_q4_prior_whitening.jl — the coevolution prior ½ û'(Q ⊗ Λ⁻¹)û in whitened
# coordinates (#857 site K).
#
# `coevo_marginal_cov` used to form Λ = L L' in Float64, invert it, and build
# P = Q ⊗ Λ⁻¹. Near a singular Λ (log-Cholesky diagonal l22 → −14 … −18, i.e.
# |cor(Λ)| → 1) that lost every digit. Measured on the known-K q2 fixture below
# (origin/main 6fb5172a0, rel. error vs a 256-bit reference):
#     l22 = −12: 2.6e-8   −14: 7.6e-7   −15.5: 1.3e-4
#     l22 = −17: 2.6e-4 (a +0.03-nat fake gain)   ≤ −18: −Inf wall.
# The whitened form never builds Λ⁻¹ (v = (I ⊗ L⁻¹)u, H̃ = Q⊗I + I⊗L'D⁻¹L) and is
# identical in exact arithmetic, including the historical 1e-10 prior ridge.
#
# The q=4 PLSM engine (`marginal_nll`, sparse_aug_plsm.jl / fit_q4_sparse_tmb.jl)
# builds its prior the same way and IS affected (measured on the
# test_q4_objective_diagnostic fixture with L21 = 0.35: l22 = −17 gives +0.34
# nats, −20 gives +4226 nats; pinned by the @test_broken below). It is NOT changed here:
# the fix touches the verified engine's Newton mode-finder and exact gradient and
# is a separate, measured PR (D-298). The @test_broken flips to an error the day
# that lands, which is the signal to promote it to @test.

module TestQ4PriorWhitening

using DRModels
using Test, LinearAlgebra, Random, SparseArrays, ForwardDiff

const _EPS_RIDGE = 1e-10

function _knownK_fixture(; G = 12, seed = 7)
    rng = MersenneTwister(seed)
    A = randn(rng, G, G)
    K = Matrix(Symmetric(A * A' / G + 0.5I))
    n = 2G
    group = repeat(1:G, 2)
    X = hcat(ones(n), randn(rng, n))
    Y = randn(rng, n, 2) .+ 0.5 .* randn(rng, G, 2)[group, :]
    return make_coevo_problem_from_covariance(K, Y, X; group = group)
end

# 256-bit dense marginal, including the engine's prior ridge
# ½[logdet(P + εI) − logdet P] so the comparison isolates rounding error.
function _q2_bigref(prob, Q, β, lc, D)
    setprecision(BigFloat, 256) do
        T = BigFloat
        q = 2
        n = size(prob.Y, 1)
        L = T[exp(T(lc[1])) 0; T(lc[2]) exp(T(lc[3]))]
        Λ = L * L'
        Qb = T.(Matrix(Q))
        K = inv(Qb)
        V = zeros(T, n * q, n * q)
        for i in 1:n, j in 1:n, a in 1:q, b in 1:q
            V[(i - 1) * q + a, (j - 1) * q + b] =
                K[prob.leaf_node[i], prob.leaf_node[j]] * Λ[a, b] +
                (i == j ? T(D[a, b]) : zero(T))
        end
        r = vec((T.(prob.Y) .- T.(prob.X) * T.(β))')
        C = cholesky(Symmetric(V))
        ℓ = -(n * q * log(2 * T(pi)) + logdet(C) + dot(r, C \ r)) / 2
        P = kron(Qb, inv(Λ))
        ridge = (logdet(cholesky(Symmetric(P + T(_EPS_RIDGE) * I))) -
                 logdet(cholesky(Symmetric(P)))) / 2
        Float64(ℓ + ridge)
    end
end

# The pre-fix (origin/main 6fb5172a0) formula, kept verbatim in spirit as the
# normal-regime identity oracle: P = Q ⊗ Λ⁻¹ and logdet(P + 1e-10 I).
function _old_marginal(prob, Q, β, Λ, D)
    q = prob.q
    n = length(prob.leaf_node)
    Dinv = inv(Symmetric(D))
    P = DRModels.prior_precision(Q, inv(Λ))
    H = DRModels.coevo_Huu(prob, P, Dinv)
    chH = cholesky(Symmetric(H))
    û = chH \ DRModels.coevo_rhs(prob, β, Dinv)
    resid = prob.Y .- prob.X * β
    qd = 0.0
    for i in 1:n
        base = q * (prob.leaf_node[i] - 1)
        r = resid[i, :] .- û[(base + 1):(base + q)]
        qd += 0.5 * dot(r, Dinv * r)
    end
    jn = 0.5 * dot(û, P * û) + qd + 0.5 * n * (q * log(2π) + logdet(Symmetric(D)))
    return -jn - 0.5 * logdet(chH) + 0.5 * logdet(cholesky(Symmetric(P) + 1e-10I))
end

@testset "coevo prior whitening: q2 near-singular Λ matches 256-bit reference" begin
    prob, Q = _knownK_fixture()
    β = [0.1 -0.2; 0.3 0.05]
    D = [0.4 0.1; 0.1 0.3]
    for l22 in (-2.0, -12.0, -14.0, -15.5, -17.0, -18.0, -20.0, -30.0)
        lc = [log(0.8), 0.7, l22]
        ℓ, û, _, _ = coevo_marginal_cov(prob, Q, β, DRModels.lc_to_chol(lc, 2), D)
        @test isfinite(ℓ)
        @test abs(ℓ - _q2_bigref(prob, Q, β, lc, D)) ≤ 1e-10
        @test all(isfinite, û)
    end
end

@testset "coevo prior whitening: normal-regime identity with the pre-fix formula" begin
    rng = MersenneTwister(2026)
    prob, Q = _knownK_fixture()
    for _ in 1:10
        lc = [log(0.3 + rand(rng)), 0.5 * randn(rng), log(0.2 + rand(rng))]
        β = 0.3 .* randn(rng, 2, 2)
        s = 0.3 .+ rand(rng, 2)
        ρ = 0.9 * (2rand(rng) - 1)
        D = [s[1]^2 ρ * s[1] * s[2]; ρ * s[1] * s[2] s[2]^2]
        Λ = lc_to_cov(lc, 2)
        old = _old_marginal(prob, Q, β, Λ, D)
        @test abs(first(coevo_marginal_cov(prob, Q, β, Λ, D)) - old) ≤ 1e-10
        @test abs(first(coevo_marginal_cov(prob, Q, β, DRModels.lc_to_chol(lc, 2), D)) - old) ≤ 1e-10
    end
    for q in (2, 4, 6), p in (6, 20)
        phy = random_balanced_tree(p; branch_length = 0.2)
        A = randn(rng, q, q)
        Λ = Matrix(Symmetric(A * A' / q + 0.2I))
        β = 0.3 .* randn(rng, 2, q)
        σ = 0.3 .+ rand(rng, q)
        sim = simulate_coevolution(phy, β, Λ, σ; nrep = 2, rng = rng)
        pr, Qc = make_coevo_problem(phy, sim.Y, sim.X; species = sim.species)
        ℓ, û, _, P = coevo_marginal(pr, Qc, β, Λ, σ)
        @test abs(ℓ - _old_marginal(pr, Qc, β, Λ, Diagonal(σ .^ 2))) ≤ 1e-10
        # returned prior is still Q ⊗ Λ⁻¹ (API)
        @test Matrix(P) ≈ Matrix(DRModels.prior_precision(Qc, inv(Λ))) rtol = 1e-12
    end
end

@testset "coevo prior whitening: degenerate inputs give -Inf, not a throw" begin
    prob, Q = _knownK_fixture()
    β = zeros(2, 2)
    D = [0.4 0.0; 0.0 0.3]
    @test first(coevo_marginal_cov(prob, Q, β, [1.0 1.0; 1.0 1.0] .+ [0 0; 0 -1e-3], D)) == -Inf
    @test first(coevo_marginal_cov(prob, Q, β, DRModels.lc_to_chol([800.0, 0.0, 0.0], 2), D)) == -Inf
end

# q=4 engine: SAME construction, NOT fixed in this PR (see header). Pins the
# measured defect so a future whitened-engine PR has a ready regression.
function _q4_bigref(prob, Q_cond, θ)
    setprecision(BigFloat, 256) do
        T = BigFloat
        β, lc = DRModels.unpack_theta(prob, θ)
        L = zeros(T, 4, 4); k = 0
        for j in 1:4, i in j:4
            k += 1
            L[i, j] = i == j ? exp(T(lc[k])) : T(lc[k])
        end
        P = kron(T.(Matrix(Q_cond)), inv(L * L'))
        η1, η2, ηs1, ηs2, ηr = DRModels.leaf_etas(prob, β)
        fs = [z -> DRModels.leaf_nll(z, T(prob.y1[i]), T(prob.y2[i]), T(η1[i]), T(η2[i]),
                                     T(ηs1[i]), T(ηs2[i]), T(ηr[i]))
              for i in eachindex(prob.leaf_node)]
        jn(u) = dot(u, P * u) / 2 +
                sum(fs[i](u[(4 * (prob.leaf_node[i] - 1) + 1):(4 * prob.leaf_node[i])])
                    for i in eachindex(prob.leaf_node))
        function gH(u)
            g = P * u; H = copy(P)
            for i in eachindex(prob.leaf_node)
                blk = (4 * (prob.leaf_node[i] - 1) + 1):(4 * prob.leaf_node[i])
                g[blk] .+= ForwardDiff.gradient(fs[i], u[blk])
                H[blk, blk] .+= ForwardDiff.hessian(fs[i], u[blk])
            end
            g, H
        end
        u = zeros(T, size(P, 1))
        for _ in 1:80
            g, H = gH(u)
            norm(g) < T(10)^-50 && break
            du = H \ g; s = one(T); j0 = jn(u)
            while jn(u - s * du) > j0 && s > 1e-12
                s /= 2
            end
            u -= s * du
        end
        _, H = gH(u)
        Float64(-jn(u) - logdet(cholesky(Symmetric(H))) / 2 +
                logdet(cholesky(Symmetric(P))) / 2)
    end
end

@testset "q4 engine prior near singular Λ (known defect, not fixed here)" begin
    rng = MersenneTwister(293)
    phy = random_balanced_tree(8; branch_length = 0.2)
    keep = setdiff(1:phy.n_total, [phy.root_index])
    species = repeat(1:8, inner = 2)
    n = length(species)
    x = randn(rng, n)
    X1 = hcat(ones(n), x)
    y1 = 0.7 .+ 0.2 .* x .+ 0.5 .* randn(rng, n)
    y2 = -0.3 .+ 0.1 .* x .+ 0.5 .* randn(rng, n)
    prob, Q_cond = make_problem(phy, y1, y2, X1, copy(X1), ones(n, 1), ones(n, 1),
                                ones(n, 1); species = species)
    β = (mu1 = [0.7, 0.2], mu2 = [-0.3, 0.1], s1 = [-0.5], s2 = [-0.6], rho = [0.2])
    θ = pack_theta(β, Matrix(Diagonal([0.2, 0.18, 0.08, 0.07])))
    o = length(θ) - 10
    θ[o + 2] = 0.35                                # L21: strong μ1–μ2 dependence
    θ[o + 5] = -2.0                                # regular regime: fine
    @test abs(-first(marginal_nll(prob, Q_cond, θ)) - _q4_bigref(prob, Q_cond, θ)) ≤ 1e-8
    θ[o + 5] = -17.0                               # |cor| → 1: defect
    @test_broken abs(-first(marginal_nll(prob, Q_cond, θ)) - _q4_bigref(prob, Q_cond, θ)) ≤ 1e-8
end

# ---------------------------------------------------------------------------
# Follow-up to #862 (this PR): the bivariate Gaussian ML route
# (`gaussian_bivariate.jl:843`, inside `_fit_bivariate_q2_structured`'s `nll`
# closure) and the q=2 structured REML route (`reml_q2.jl`, `_q2_reml_ll` and
# `fit_coevolution_q2_reml`'s final `ml_ll`) both called `coevo_marginal_cov`
# with a Λ ALREADY FORMED as a matrix (`lc_to_cov`), which #862's
# `coevo_marginal_cov(::AbstractMatrix)` method then re-factors with
# `cholesky(Symmetric(Matrix(Λ)))`. That is accurate only to l22 ≈ −18 (one
# `L L'` round trip in Float64 before the whitened path ever sees it) instead
# of #862's ≤ 1e-10 to l22 = −30. Both call sites now pass
# `lc_to_chol(lc, 2)` (the factor built straight from `lc`) instead. This
# reproduces the "before" (matrix Λ) vs "after" (chΛ) behaviour of those two
# call sites directly through `coevo_marginal_cov`'s two methods -- the same
# function object each call site invokes -- without needing to reach into the
# fitting closures themselves.
# ---------------------------------------------------------------------------
@testset "lc_to_chol callers (#862 follow-up): extreme-regime accuracy" begin
    prob, Q = _knownK_fixture()
    β = [0.1 -0.2; 0.3 0.05]
    D = [0.4 0.1; 0.1 0.3]
    for l22 in (-12.0, -18.0, -20.0, -30.0)
        lc = [log(0.8), 0.7, l22]
        ref = _q2_bigref(prob, Q, β, lc, D)
        before = first(coevo_marginal_cov(prob, Q, β, lc_to_cov(lc, 2), D))  # base-branch call sites
        after  = first(coevo_marginal_cov(prob, Q, β, DRModels.lc_to_chol(lc, 2), D))  # fixed call sites
        err_before = isfinite(before) ? abs(before - ref) / abs(ref) : Inf
        err_after  = abs(after - ref) / abs(ref)
        @info "lc_to_chol callers: l22=$l22 rel. error before=$err_before after=$err_after"
        @test err_after ≤ 1e-10
    end
end

@testset "lc_to_chol callers (#862 follow-up): normal-regime identity" begin
    rng = MersenneTwister(4177)
    prob, Q = _knownK_fixture()
    for _ in 1:10
        lc = [log(0.3 + rand(rng)), 0.5 * randn(rng), log(0.2 + rand(rng))]
        β = 0.3 .* randn(rng, 2, 2)
        s = 0.3 .+ rand(rng, 2)
        ρ = 0.9 * (2rand(rng) - 1)
        D = [s[1]^2 ρ * s[1] * s[2]; ρ * s[1] * s[2] s[2]^2]
        before = first(coevo_marginal_cov(prob, Q, β, lc_to_cov(lc, 2), D))
        after  = first(coevo_marginal_cov(prob, Q, β, DRModels.lc_to_chol(lc, 2), D))
        @test abs(after - before) ≤ 1e-12
    end
end

end # module
