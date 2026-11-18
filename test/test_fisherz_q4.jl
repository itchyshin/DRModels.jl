# test_fisherz_q4.jl — coverage for the exported spherical/LKJ ("Fisher-z" by
# name only) q=4 outer parameterization in `src/fisherz_q4.jl`.
#
# The angle link is ρ = cos(π(tanh θ + 1)/2), NOT tanh θ, so a wrong inverse is
# a silent-error risk. Checks: R is a valid correlation matrix; the Σ → φ → Σ
# round trip; the φ → lc chain-rule Jacobian; the outer gradient against a
# central finite difference; and the fit driver's contract.

using DRModels
using Test, LinearAlgebra, Random, Statistics, ForwardDiff

const _FZ = DRModels   # fz_marginal_nll is internal (not exported)

@testset "fisherz_q4: spherical/LKJ D·R·D outer parameterization" begin

    @testset "fz_R / fz_correlations" begin
        Random.seed!(1)
        for _ in 1:20
            θ = 2 .* randn(6)
            R = fz_R(θ)
            @test issymmetric(R) || maximum(abs, R - R') ≤ 1e-14
            @test all(≈(1.0; atol = 1e-12), diag(R))
            @test isposdef(Symmetric(R))
            @test all(abs.(R[tril(trues(4, 4), -1)]) .< 1)

            φ = vcat(randn(4) ./ 2, θ)
            Σ = fz_DRD(φ)
            s = sqrt.(diag(Σ))
            Rimp = Σ ./ s ./ s'
            @test Rimp ≈ R atol = 1e-12                    # D cancels
            c = fz_correlations(φ)
            @test collect(c) ≈ [Rimp[2, 1], Rimp[3, 1], Rimp[4, 1],
                                Rimp[3, 2], Rimp[4, 2], Rimp[4, 3]] atol = 1e-12
        end
        # Documented link for the first angle: ρ21 = cos(π(tanh θ + 1)/2), not tanh θ.
        for θ1 in (-1.3, -0.2, 0.0, 0.7, 2.0)
            θ = [θ1, 0.1, -0.2, 0.3, 0.4, -0.5]
            @test fz_R(θ)[2, 1] ≈ cos(π * (tanh(θ1) + 1) / 2) atol = 1e-12
        end
        @test fz_R(zeros(6))[2, 1] ≈ 0 atol = 1e-12        # θ=0 ⇒ α=π/2 ⇒ ρ=0
    end

    @testset "fz_init_from_Sigma round trip" begin
        Random.seed!(2)
        for _ in 1:10
            A = randn(4, 4)
            Σ = Matrix(Symmetric(A * A' + 0.5I))
            φ = fz_init_from_Sigma(Σ)
            @test length(φ) == 10 && all(isfinite, φ)
            @test fz_DRD(φ) ≈ Σ atol = 1e-8 rtol = 1e-8
        end
        # Non-SPD input (indefinite correlation, off-diagonals beyond PD range):
        # no ridge up to 1e-6 can repair it, so it must fail loudly.
        Σbad = [1.0 0.99 -0.99 0.0; 0.99 1.0 0.99 0.0; -0.99 0.99 1.0 0.0; 0.0 0.0 0.0 1.0]
        @test !isposdef(Σbad)
        @test_throws ErrorException fz_init_from_Sigma(Σbad)
    end

    @testset "fz_phi_to_lc" begin
        Random.seed!(3)
        for _ in 1:5
            φ = vcat(randn(4) ./ 2, randn(6))
            @test fz_phi_to_lc(φ) ≈ Λ_to_lc(fz_DRD(φ)) atol = 1e-12
            J = ForwardDiff.jacobian(fz_phi_to_lc, φ)
            @test size(J) == (10, 10)
            @test all(isfinite, J)
        end
    end

    # --- small simulated tree problem (setup as in test_lc_metric.jl) -------
    Random.seed!(7)
    p = 8; n = p
    phy = random_balanced_tree(p; branch_length = 0.2)
    Σ_phy = sigma_phy_dense(phy; σ²_phy = 1.0)
    βt = (mu1 = [1.0, 0.5], mu2 = [-0.3, 0.4], s1 = [-0.4], s2 = [-0.5], rho = [0.3])
    Λt = [0.25 0.10 0.05 0.0; 0.10 0.25 0.0 0.04; 0.05 0.0 0.09 0.02; 0.0 0.04 0.02 0.09]
    Λt = (Λt + Λt') / 2
    x1 = randn(n)
    X1 = hcat(ones(n), x1); X2 = hcat(ones(n), x1)
    Xs1 = reshape(ones(n), n, 1); Xs2 = reshape(ones(n), n, 1); Xr = reshape(ones(n), n, 1)
    U = cholesky(Λt).L * randn(4, p) * cholesky(Symmetric(Σ_phy)).U
    y1 = zeros(n); y2 = zeros(n)
    for i in 1:n
        m1 = (X1[i, :]' * βt.mu1) + U[1, i]; m2 = (X2[i, :]' * βt.mu2) + U[2, i]
        s1 = exp((Xs1[i, :]' * βt.s1) + U[3, i]); s2 = exp((Xs2[i, :]' * βt.s2) + U[4, i])
        ρ = 0.99999999 * tanh(Xr[i, :]' * βt.rho)
        e = cholesky([s1^2 ρ*s1*s2; ρ*s1*s2 s2^2]).L * randn(2)
        y1[i] = m1 + e[1]; y2[i] = m2 + e[2]
    end
    prob, Q_cond = make_problem(phy, y1, y2, X1, X2, Xs1, Xs2, Xr)
    β0 = (mu1 = X1 \ y1, mu2 = X2 \ y2,
          s1 = [log(std(y1 .- X1 * (X1 \ y1)))], s2 = [log(std(y2 .- X2 * (X2 \ y2)))],
          rho = [0.0])
    Σa0 = [0.30 0.06 0.02 0.0; 0.06 0.28 0.0 0.03; 0.02 0.0 0.14 0.01; 0.0 0.03 0.01 0.16]
    Σa0 = Matrix(Symmetric((Σa0 + Σa0') / 2))
    ψ0 = vcat(β0.mu1, β0.mu2, β0.s1, β0.s2, β0.rho, fz_init_from_Sigma(Σa0))

    @testset "fz_marginal_and_grad vs central finite difference" begin
        nll, g, _, _ = fz_marginal_and_grad(prob, Q_cond, ψ0; n_newton = 80)
        @test isfinite(nll) && all(isfinite, g)
        @test nll ≈ _FZ.fz_marginal_nll(prob, Q_cond, ψ0; n_newton = 80) rtol = 1e-8
        h = 1e-5
        gfd = similar(ψ0)
        for i in eachindex(ψ0)
            ep = copy(ψ0); ep[i] += h
            em = copy(ψ0); em[i] -= h
            gfd[i] = (_FZ.fz_marginal_nll(prob, Q_cond, ep; n_newton = 80) -
                      _FZ.fz_marginal_nll(prob, Q_cond, em; n_newton = 80)) / (2h)
        end
        @test norm(g - gfd) / max(norm(gfd), 1e-8) < 1e-5
    end

    @testset "fit_q4_sparse_fisherz" begin
        fit = fit_q4_sparse_fisherz(prob, Q_cond; β0 = β0, Σa0 = Σa0, iterations = 30)
        @test isfinite(fit.nll)
        @test fit.loglik ≈ -fit.nll
        @test length(fit.correlations) == 6
        @test all(c -> -1 < c < 1, fit.correlations)
        @test fit.Σa ≈ fz_DRD(fit.φa)
        @test isposdef(Symmetric(fit.Σa))
        @test collect(fit.correlations) ≈ collect(fz_correlations(fit.φa))
        @test fit.iterations ≤ 30

        # Λ0 is accepted as an alias for the starting Σ_a.
        fitL = fit_q4_sparse_fisherz(prob, Q_cond; β0 = β0, Λ0 = Σa0, iterations = 2)
        @test isfinite(fitL.nll)

        # iterations cap respected
        fit1 = fit_q4_sparse_fisherz(prob, Q_cond; β0 = β0, Σa0 = Σa0, iterations = 1)
        @test fit1.iterations ≤ 1

        # time_limit respected: a vanishing budget stops after at most a step or two
        fitT = fit_q4_sparse_fisherz(prob, Q_cond; ψ0 = ψ0, time_limit = 1e-9,
                                     iterations = 500)
        @test fitT.iterations < 5

        @test_throws ErrorException fit_q4_sparse_fisherz(prob, Q_cond)
        err = try fit_q4_sparse_fisherz(prob, Q_cond); nothing catch e; e end
        @test err isa ErrorException
        @test occursin("supply ψ0 or (β0, Λ0/Σa0)", sprint(showerror, err))
    end
end
