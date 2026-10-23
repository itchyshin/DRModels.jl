# test_qgate_fd_gradient.jl — STANDING Q-gate for issue #14.
#
# Finite-difference-vs-analytic gradient gate (≤ 1e-6) for the verified q=4
# sparse-Laplace engine. This is the poc's grad_check_* check (validated to
# 1e-6) promoted to an always-on test.
#
# Reuses the EXACT engine API the grad_check_* scripts use (do NOT invent a new
# entry point):
#   • build a small reproducible q4 PLSM via random_balanced_tree + make_problem
#     (self-contained; no CSV/tree fixtures), exactly as test_analytic_grad.jl
#     and bench/demo_natgrad_p20.jl construct their problems;
#   • pack θ = [β(7); lc(10)] with pack_theta;
#   • analytic gradient  g_analytic = marginal_and_exact_grad(prob, Q_cond, θ)[2];
#   • central finite-difference gradient of the TRUE marginal NLL
#     marginal_nll(prob, Q_cond, θ)[1], warm-starting perturbed solves from the
#     base θ-mode so the finite-difference reference is not dominated by
#     inner-mode stopping noise.
#
# A non-diagonal Λ is used (the grad_check_diag.jl "case B / non-diagonal" setup
# that PASSED), evaluated at θ0 — away from the optimum the exact gradient must
# still match FD because it carries the implicit (dû/dθ) correction.

using DRModels
using Test, LinearAlgebra, Random, Statistics

@testset "Q-gate (#14): FD-vs-exact gradient ≤ 1e-6" begin
    Random.seed!(7)
    p = 8; n = p

    # --- reproducible q4 PLSM data (same generator family as test_analytic_grad.jl)
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

    # --- θ0 = OLS starts + a non-diagonal Λ (grad_check_diag.jl "case B") --------
    β0 = (mu1 = X1 \ y1, mu2 = X2 \ y2,
          s1 = [log(std(y1 .- X1 * (X1 \ y1)))], s2 = [log(std(y2 .- X2 * (X2 \ y2)))],
          rho = [0.0])
    Λ0 = [0.30 0.06 0.02 0.0; 0.06 0.28 0.0 0.03; 0.02 0.0 0.14 0.01; 0.0 0.03 0.01 0.16]
    Λ0 = Matrix(Symmetric((Λ0 + Λ0') / 2))
    θ = pack_theta(β0, Λ0)

    # --- analytic exact gradient (engine entry point) ---------------------------
    _, g_analytic, u_ref, _ = marginal_and_exact_grad(prob, Q_cond, θ; n_newton = 120)

    # --- central finite-difference gradient of the TRUE marginal NLL ------------
    mnll(t) = marginal_nll(prob, Q_cond, Vector{Float64}(t); u0 = u_ref, n_newton = 120)[1]
    h = 1e-4
    g_fd = similar(θ)
    for k in eachindex(θ)
        tp = copy(θ); tp[k] += h
        tm = copy(θ); tm[k] -= h
        g_fd[k] = (mnll(tp) - mnll(tm)) / (2h)
    end

    max_abs_diff = maximum(abs, g_analytic .- g_fd)
    @test max_abs_diff ≤ 1e-6
end

# ---------------------------------------------------------------------------
# #509: the SAME gate on the STRUCTURED problem constructor.
#
# The gate above covers the phylo path only — `make_problem(phy, ...)`. The
# structured q4 route (relmat / animal / spatial) reaches the identical engine
# through `make_problem_from_Q(Qdense, ...)` instead, and that path had NO FD
# gate. `test_gaussian_bivariate_q4_structured.jl` says so in its own comment:
# "Engine FD <=1e-6 is already gated on phylo Q_cond ... here we only prove the
# structured route wires the same nll / nllgrad closures."
#
# It passes (measured 4.73e-07), so this records a property that already holds
# rather than a defect. It is worth standing because the structured route's only
# other gradient test is `norm(gp) > norm(g0)`, which a wrong gradient would
# satisfy just as easily as a right one.
#
# NOTE the design constraint learned filing #509: the fixture must be
# IDENTIFIED. With q=4 latent values per group, G groups and n rows, the model
# is saturated once 4G >= 2n, Lambda collapses toward singular, and both the
# gradient and its finite-difference reference stop being meaningful. Here
# G = 8 groups x 2 replicates = 16 rows = 32 observations against 32 latent
# values would be saturated, so nrep = 4 is used deliberately.
@testset "Q-gate (#509): FD-vs-exact gradient <= 1e-6 on the STRUCTURED constructor" begin
    Random.seed!(7)
    p = 8; nrep = 4; n = p * nrep
    group = repeat(1:p, inner = nrep)

    # relmat-style known covariance on the correlation scale, as
    # `_q4_structured_precision` produces for kind = :relmat
    pos = rand(p, 2) .* 5.0
    Dist = [sqrt(sum(abs2, pos[a, :] .- pos[b, :])) for a in 1:p, b in 1:p]
    K = Symmetric(exp.(-Dist ./ 1.5) + 1e-8 * I)
    dK = sqrt.(diag(K)); K = Symmetric(Matrix(K) ./ (dK * dK'))
    Qdense = inv(Matrix(K))

    βt = (mu1 = [1.0, 0.5], mu2 = [-0.3, 0.4], s1 = [-0.4], s2 = [-0.5], rho = [0.3])
    Λt = [0.25 0.10 0.05 0.0; 0.10 0.25 0.0 0.04; 0.05 0.0 0.09 0.02; 0.0 0.04 0.02 0.09]
    Λt = (Λt + Λt') / 2

    x1 = randn(n)
    X1 = hcat(ones(n), x1); X2 = hcat(ones(n), x1)
    Xs1 = reshape(ones(n), n, 1); Xs2 = reshape(ones(n), n, 1); Xr = reshape(ones(n), n, 1)

    U = cholesky(Λt).L * randn(4, p) * cholesky(Matrix(K)).U
    y1 = zeros(n); y2 = zeros(n)
    for i in 1:n
        g = group[i]
        m1 = (X1[i, :]' * βt.mu1) + U[1, g]; m2 = (X2[i, :]' * βt.mu2) + U[2, g]
        s1 = exp((Xs1[i, :]' * βt.s1) + U[3, g]); s2 = exp((Xs2[i, :]' * βt.s2) + U[4, g])
        ρ = 0.99999999 * tanh(Xr[i, :]' * βt.rho)
        e = cholesky([s1^2 ρ*s1*s2; ρ*s1*s2 s2^2]).L * randn(2)
        y1[i] = m1 + e[1]; y2[i] = m2 + e[2]
    end

    prob, Q_cond = DRModels.make_problem_from_Q(Qdense, y1, y2, X1, X2, Xs1, Xs2, Xr;
                                           group = group)

    β0 = (mu1 = X1 \ y1, mu2 = X2 \ y2,
          s1 = [log(std(y1 .- X1 * (X1 \ y1)))], s2 = [log(std(y2 .- X2 * (X2 \ y2)))],
          rho = [0.0])
    Λ0 = [0.30 0.06 0.02 0.0; 0.06 0.28 0.0 0.03; 0.02 0.0 0.14 0.01; 0.0 0.03 0.01 0.16]
    Λ0 = Matrix(Symmetric((Λ0 + Λ0') / 2))
    θ = pack_theta(β0, Λ0)

    _, g_analytic, u_ref, _ = marginal_and_exact_grad(prob, Q_cond, θ; n_newton = 120)

    mnll(t) = marginal_nll(prob, Q_cond, Vector{Float64}(t); u0 = u_ref, n_newton = 120)[1]
    h = 1e-4
    g_fd = similar(θ)
    for k in eachindex(θ)
        tp = copy(θ); tp[k] += h
        tm = copy(θ); tm[k] -= h
        g_fd[k] = (mnll(tp) - mnll(tm)) / (2h)
    end

    @test maximum(abs, g_analytic .- g_fd) ≤ 1e-6
end
