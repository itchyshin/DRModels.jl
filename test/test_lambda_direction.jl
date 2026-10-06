# test_lambda_direction.jl — decisive diagnostic for the historical frozen-Λ
# bug: at p=8, fixed β, does the sparse-EM M-step (`mstep_Lambda`) actually
# ASCEND the true Laplace marginal in Λ, and does a plain finite-diff gradient
# ascent step on the true marginal also ascend it? `mstep_Lambda`/`fit_em_aug`
# are not on the public `drm()` path (the wired q=4 fit ascends the marginal
# directly via a line-searched gradient — `fit_ml_q4.jl`'s `lambda_ml_step`),
# but they back the `sparse_em_fit.jl` demos and have no other test coverage.

using DRModels
using Test, LinearAlgebra, Random, Statistics

const D = DRModels

@testset "sparse-EM Λ M-step ascends the true Laplace marginal (p=8)" begin
    Random.seed!(7); p = 8; n = p
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
        m1 = (X1[i, :]'βt.mu1) + U[1, i]; m2 = (X2[i, :]'βt.mu2) + U[2, i]
        s1 = exp((Xs1[i, :]'βt.s1) + U[3, i]); s2 = exp((Xs2[i, :]'βt.s2) + U[4, i])
        ρ = D.RHO_GUARD * tanh(Xr[i, :]'βt.rho)
        e = cholesky([s1^2 ρ*s1*s2; ρ*s1*s2 s2^2]).L * randn(2)
        y1[i] = m1 + e[1]; y2[i] = m2 + e[2]
    end
    prob, Q_cond = make_problem(phy, y1, y2, X1, X2, Xs1, Xs2, Xr)

    # fix β at a reasonable estimate; vary only Λ
    β = (mu1 = X1 \ y1, mu2 = X2 \ y2, s1 = [log(std(y1 .- X1 * (X1 \ y1)))],
         s2 = [log(std(y2 .- X2 * (X2 \ y2)))], rho = [0.0])

    function L_of_Λ(Λ)
        P = prior_precision(Q_cond, inv(Λ))
        u, ch, _ = estep_mode(prob, P, β)
        return D.laplace_ll(prob, P, β, u, ch)
    end
    function lc_to_Λ(v)
        C = zeros(eltype(v), 4, 4); k = 0
        for j in 1:4, i in j:4
            k += 1; C[i, j] = i == j ? exp(v[k]) : v[k]
        end
        return C * C'
    end
    function Λ_to_lc(Λ)
        C = cholesky(Symmetric(Λ)).L; v = Float64[]
        for j in 1:4, i in j:4
            push!(v, i == j ? log(C[i, j]) : C[i, j])
        end
        return v
    end

    Λ0 = Matrix(0.3 * I(4)); L0 = L_of_Λ(Λ0)

    # (1) EM closed-form M-step direction must ascend the TRUE marginal
    P0 = prior_precision(Q_cond, inv(Λ0)); u0, ch0, _ = estep_mode(prob, P0, β)
    Λ_em = D.mstep_Lambda(prob, Q_cond, u0, ch0)
    @test L_of_Λ(Λ_em) > L0
    for α in (1.0, 0.5, 0.25, 0.1, 0.01)
        Λα = Matrix(Symmetric(Λ0 .+ α .* (Λ_em .- Λ0)))
        @test L_of_Λ(Λα) > L0
    end

    # (2) finite-diff gradient of the TRUE marginal w.r.t. the 10 log-chol
    # params is a genuine ascent direction (sanity check on L_of_Λ itself).
    v0 = Λ_to_lc(Λ0); h = 1e-5
    g = similar(v0)
    for k in eachindex(v0)
        vp = copy(v0); vp[k] += h; vm = copy(v0); vm[k] -= h
        g[k] = (L_of_Λ(lc_to_Λ(vp)) - L_of_Λ(lc_to_Λ(vm))) / (2h)
    end
    @test norm(g) > 0
    # A gradient direction is only guaranteed to ascend LOCALLY; a raw
    # (unnormalized) step of s=1.0 in log-Cholesky space can overshoot, so only
    # the smaller steps are asserted here.
    for s in (0.3, 0.1, 0.03, 0.01)
        Λs = lc_to_Λ(v0 .+ s .* g)
        @test L_of_Λ(Λs) > L0
    end
end
