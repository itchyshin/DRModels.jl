# test_coevo_pack_validation.jl -- direct unit tests for the coevolution parameter
# plumbing in src/coevolution_q.jl: coevo_pack / coevo_unpack, the q=2
# residual-correlation pack / unpack, the log-Cholesky helpers, and the
# known-structure problem constructors. Test-only lane; no engine fits.

using DRModels
using Test, LinearAlgebra, Random, SparseArrays

_rand_spd(rng, q) = (A = randn(rng, q, q); Matrix(Symmetric(A * A' + 0.5I)))

@testset "coevo_pack / coevo_unpack round trip (q = 2:4)" begin
    rng = MersenneTwister(20260101)
    for q in 2:4
        k = 3
        G = 5
        Y = randn(rng, G, q)
        X = hcat(ones(G), randn(rng, G, k - 1))
        prob, _ = make_coevo_problem_from_precision(Matrix(1.0I, G, G), Y, X)
        @test prob.q == q && prob.k == k
        β = randn(rng, k, q)
        Λ = _rand_spd(rng, q)
        σ = 0.2 .+ rand(rng, q)
        θ = coevo_pack(β, Λ, σ)
        @test length(θ) == coevo_theta_len(prob)
        # layout: vec(β) | log-Cholesky of Λ | log σ_res
        @test θ[1:(k * q)] == vec(β)
        @test θ[(k * q + 1):(k * q + lc_len(q))] ≈ cov_to_lc(Λ)
        @test θ[(k * q + lc_len(q) + 1):end] ≈ log.(σ)
        β2, Λ2, σ2 = coevo_unpack(prob, θ)
        @test β2 == β
        @test isapprox(Λ2, Λ; atol = 1e-10)
        @test isapprox(σ2, σ; atol = 1e-10)
        # unpack on its own recovers a hand-built θ (σ stored on the log scale)
        v = randn(rng, lc_len(q)) .* 0.3
        θ3 = vcat(vec(β), v, log.(σ))
        _, Λ3, σ3 = coevo_unpack(prob, θ3)
        @test Λ3 ≈ lc_to_cov(v, q)
        @test σ3 ≈ σ
    end
end

@testset "coevo_q2_residual_pack / unpack round trip" begin
    rng = MersenneTwister(20260102)
    q = 2
    k = 3
    G = 5
    Y = randn(rng, G, q)
    X = hcat(ones(G), randn(rng, G, k - 1))
    prob, _ = make_coevo_problem_from_precision(Matrix(1.0I, G, G), Y, X)
    @test DRModels.coevo_q2_residual_theta_len(prob) == 2k + lc_len(2) + 3
    for ρ in (-0.6, 0.0, 0.35)
        β = randn(rng, k, q)
        Λ = _rand_spd(rng, q)
        σ = 0.2 .+ rand(rng, q)
        θ = DRModels.coevo_q2_residual_pack(β, Λ, σ, ρ)
        @test length(θ) == DRModels.coevo_q2_residual_theta_len(prob)
        β2, Λ2, D, σ2, ρ2 = DRModels.coevo_q2_residual_unpack(prob, θ)
        @test β2 == β
        @test isapprox(Λ2, Λ; atol = 1e-10)
        @test isapprox(σ2, σ; atol = 1e-10)
        @test isapprox(ρ2, ρ; atol = 1e-10)
        @test isapprox(D, [σ[1]^2 ρ*σ[1]*σ[2]; ρ*σ[1]*σ[2] σ[2]^2]; atol = 1e-10)
    end
    # ρ is clamped to ±0.95 on pack
    θc = DRModels.coevo_q2_residual_pack(zeros(k, q), Matrix(1.0I, q, q), ones(q), 0.999)
    @test DRModels.coevo_q2_residual_unpack(prob, θc)[5] ≈ 0.95 atol = 1e-8
    @test_throws ErrorException DRModels.coevo_q2_residual_pack(
        zeros(k, q), Matrix(1.0I, q, q), ones(3), 0.1)
    prob3, _ = make_coevo_problem_from_precision(Matrix(1.0I, G, G), randn(rng, G, 3), X)
    @test_throws ErrorException DRModels.coevo_q2_residual_unpack(prob3, zeros(20))
end

@testset "log-Cholesky helpers: lc_to_cov / cov_to_lc / lc_to_chol" begin
    rng = MersenneTwister(20260103)
    for q in 2:4
        v = randn(rng, lc_len(q)) .* 0.4
        @test cov_to_lc(lc_to_cov(v, q)) ≈ v atol = 1e-10
        L = _rand_spd(rng, q)
        @test lc_to_cov(cov_to_lc(L), q) ≈ L atol = 1e-10
        @test Matrix(DRModels.lc_to_chol(v, q)) ≈ lc_to_cov(v, q) atol = 1e-12
        @test DRModels.lc_to_chol(v, q).L ≈ cholesky(Symmetric(lc_to_cov(v, q))).L atol = 1e-10
        @test_throws ErrorException lc_to_cov(v[1:(end - 1)], q)
        @test_throws ErrorException DRModels.lc_to_chol(v[1:(end - 1)], q)
    end
    # substantively indefinite input is refused, not regularised
    @test_throws ArgumentError cov_to_lc([1.0 2.0; 2.0 1.0])
    @test_throws ArgumentError cov_to_lc(-Matrix(1.0I, 3, 3))
end

@testset "make_coevo_problem_from_precision: group validation" begin
    rng = MersenneTwister(20260104)
    G, n, q = 4, 8, 2
    Y = randn(rng, n, q)
    X = hcat(ones(n), randn(rng, n))
    Q = Matrix(1.0I, G, G)
    ok = [1, 2, 3, 4, 1, 2, 3, 4]
    prob, Qs = make_coevo_problem_from_precision(Q, Y, X; group = ok)
    @test prob.leaf_node == ok
    @test prob.N == G
    @test Qs isa SparseMatrixCSC
    msg = r"group indices must be 1-based integers in 1:4"
    @test_throws ErrorException make_coevo_problem_from_precision(Q, Y, X; group = fill(0, n))
    @test_throws msg make_coevo_problem_from_precision(Q, Y, X; group = fill(0, n))
    bad = copy(ok); bad[3] = G + 1
    @test_throws msg make_coevo_problem_from_precision(Q, Y, X; group = bad)
    neg = copy(ok); neg[2] = -1
    @test_throws msg make_coevo_problem_from_precision(Q, Y, X; group = neg)
    # integer-valued floats are accepted
    probf, _ = make_coevo_problem_from_precision(Q, Y, X; group = Float64.(ok))
    @test probf.leaf_node == ok
    # a non-integer vector is refused (the documented contract is "1-based
    # integers"); today it raises InexactError from Int.() before the check,
    # not the named message.
    frac = Float64.(ok); frac[3] = 2.5
    @test_throws Exception make_coevo_problem_from_precision(Q, Y, X; group = frac)
    @test_broken try
        make_coevo_problem_from_precision(Q, Y, X; group = frac)
        false
    catch e
        e isa ErrorException && occursin("group indices must be 1-based integers", e.msg)
    end
    # shape / definiteness guards
    @test_throws ErrorException make_coevo_problem_from_precision(Q, Y, X; group = ok[1:7])
    @test_throws ErrorException make_coevo_problem_from_precision(Q, Y, X[1:7, :]; group = ok)
    @test_throws ErrorException make_coevo_problem_from_precision(-Q, Y, X; group = ok)
end

@testset "make_coevo_problem_from_covariance == from_precision(inv(K))" begin
    rng = MersenneTwister(20260105)
    G, n, q = 5, 10, 3
    A = randn(rng, G, G)
    K = Matrix(Symmetric(A * A' + I))
    Y = randn(rng, n, q)
    X = hcat(ones(n), randn(rng, n))
    group = [1, 2, 3, 4, 5, 5, 4, 3, 2, 1]
    probK, QK = make_coevo_problem_from_covariance(K, Y, X; group = group)
    probQ, QQ = make_coevo_problem_from_precision(Matrix(Symmetric(inv(K))), Y, X; group = group)
    @test Matrix(QK) ≈ Matrix(QQ) atol = 1e-8
    @test probK.leaf_node == probQ.leaf_node && probK.N == probQ.N
    β = randn(rng, 2, q)
    Λ = _rand_spd(rng, q)
    σ = 0.3 .+ rand(rng, q)
    ℓK = coevo_marginal(probK, QK, β, Λ, σ)[1]
    ℓQ = coevo_marginal(probQ, QQ, β, Λ, σ)[1]
    @test isfinite(ℓK)
    @test isapprox(ℓK, ℓQ; atol = 1e-8)
    θ = coevo_pack(β, Λ, σ)
    βu, Λu, σu = coevo_unpack(probK, θ)
    @test isapprox(coevo_marginal(probK, QK, βu, Λu, σu)[1], ℓK; atol = 1e-8)
    @test_throws ErrorException make_coevo_problem_from_covariance(-K, Y, X; group = group)
end
