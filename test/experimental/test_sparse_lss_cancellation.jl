# test/experimental/test_sparse_lss_cancellation.jl
# Issue #551: Cancellation Gate Probe and Takahashi Gradient Verification for O(p) Sparse LSS Engine

using Test
using LinearAlgebra
using SparseArrays
using Random
using ForwardDiff
using Printf
using DRModels

function _make_phylo_lss_fixture(; G = 16, n_per_group = 2, seed = 42)
    rng = MersenneTwister(seed)
    phy = DRModels.random_balanced_tree(G; branch_length = 0.2)
    Q, leaf_pos, q = DRModels.augmented_tree_precision(phy)
    Qs = dropzeros!(sparse(Symmetric(Matrix(Q))))
    chQ = cholesky(Symmetric(Qs); check = false)
    Qinv = DRModels.takahashi_selinv(chQ)
    leaf_var = [Qinv[leaf_pos[t], leaf_pos[t]] for t in 1:G]
    inv_sd = [1.0 / sqrt(leaf_var[t]) for t in 1:G]
    logdetCprior = -logdet(chQ)

    K = DRModels._phylo_correlation(phy)
    n = G * n_per_group
    gidx = repeat(1:G, inner = n_per_group)

    pμ = 2
    Xμ = [ones(n) randn(rng, n)]
    βμ = [1.0, 0.5]

    pσ = 2
    Xσ = [ones(n) randn(rng, n)]
    βσ = [-0.5, 0.3]

    psd = 2
    Zg = [ones(G) randn(rng, G)]
    α = [-0.3, 0.4]

    σa_vec = exp.(Zg * α)
    Σa = (σa_vec * σa_vec') .* K
    chΣa = cholesky(Symmetric(Σa))
    u_phylo = chΣa.L * randn(rng, G)

    σε_base = exp.(Xσ * βσ)
    y = Xμ * βμ + u_phylo[gidx] + σε_base .* randn(rng, n)

    return (phy=phy, Q=Qs, leaf_pos=leaf_pos, q=q, leaf_var=leaf_var, inv_sd=inv_sd,
            logdetCprior=logdetCprior, K=K, n=n, G=G, gidx=gidx,
            Xμ=Xμ, βμ=βμ, pμ=pμ, Xσ=Xσ, βσ=βσ, pσ=pσ,
            Zg=Zg, α=α, psd=psd, y=y)
end

function _dense_bigfloat_exact(θ, fix; scale_se = 1.0)
    pμ, pσ, psd = fix.pμ, fix.pσ, fix.psd
    βμ = BigFloat.(θ[1:pμ])
    βσ = BigFloat.(θ[pμ+1:pμ+pσ])
    α = BigFloat.(θ[pμ+pσ+1:pμ+pσ+psd])

    Xμ = BigFloat.(fix.Xμ)
    Xσ = BigFloat.(fix.Xσ)
    Zg = BigFloat.(fix.Zg)
    K = BigFloat.(fix.K)
    y = BigFloat.(fix.y)

    ημ = Xμ * βμ
    ησ = Xσ * βσ
    ησa = Zg * α
    σa = exp.(ησa)
    Σa = (σa * σa') .* K

    n = fix.n
    Vm = Matrix{BigFloat}(undef, n, n)
    for j in 1:n, i in 1:n
        Vm[i, j] = Σa[fix.gidx[i], fix.gidx[j]]
    end
    for i in 1:n
        Vm[i, i] += (BigFloat(scale_se) * exp(ησ[i]))^2
    end
    Vfac = cholesky(Symmetric(Vm))
    r = y .- ημ
    q_dense = Float64(dot(r, Vfac \ r))
    logdetV = Float64(logdet(Vfac))
    nll = Float64(0.5 * (logdet(Vfac) + dot(r, Vfac \ r) + n * log(2 * BigFloat(pi))))
    return (q_dense, logdetV, nll)
end

function _eval_sparse_lss(θ, fix; scale_se = 1.0, want_grad = false)
    pμ, pσ, psd = fix.pμ, fix.pσ, fix.psd
    βμ = θ[1:pμ]
    βσ = θ[pμ+1:pμ+pσ]
    α = θ[pμ+pσ+1:pμ+pσ+psd]

    n = fix.n
    G = fix.G
    m = fix.q
    r = fix.y .- fix.Xμ * βμ

    ησ = fix.Xσ * βσ
    σε = scale_se .* exp.(ησ)
    w = 1.0 ./ (σε .^ 2)

    ησa = fix.Zg * α
    σa = exp.(ησa)
    wts_g = σa .* fix.inv_sd
    wts = [wts_g[fix.gidx[i]] for i in 1:n]

    diag_ZtWZ = zeros(m)
    for i in 1:n
        r_node = fix.leaf_pos[fix.gidx[i]]
        diag_ZtWZ[r_node] += w[i] * wts[i]^2
    end

    H = fix.Q + Diagonal(diag_ZtWZ)
    chH = cholesky(Symmetric(H); check = false)
    issuccess(chH) || return (1e18, Float64[], 0.0, 0.0, 0.0)

    b = zeros(m)
    for i in 1:n
        r_node = fix.leaf_pos[fix.gidx[i]]
        b[r_node] += wts[i] * w[i] * r[i]
    end

    â = chH \ b
    Zâ = [wts[i] * â[fix.leaf_pos[fix.gidx[i]]] for i in 1:n]
    res_lat = r .- Zâ

    logdetH = logdet(chH)
    logdetD = -sum(log, w)
    logdetV = logdetD + fix.logdetCprior + logdetH

    # Method A: Woodbury subtraction
    quad_sub = dot(r, w .* r) - dot(b, â)

    # Method B: GMRF sum of squares
    quad_sos = dot(res_lat, w .* res_lat) + dot(â, fix.Q * â)

    nll_sos = 0.5 * (logdetV + quad_sos + n * log(2π))
    nll_sub = 0.5 * (logdetV + quad_sub + n * log(2π))

    if !want_grad
        return (nll_sos, nll_sub, quad_sos, quad_sub, logdetV)
    end

    # Analytic Takahashi gradients
    Hinv = DRModels.takahashi_selinv(chH)
    u = w .* res_lat

    g_βμ = - (fix.Xμ' * u)

    g_βσ = zeros(pσ)
    for i in 1:n
        r_node = fix.leaf_pos[fix.gidx[i]]
        s_ii = Hinv[r_node, r_node]
        zHz_ii = wts[i]^2 * s_ii
        vinv_ii = w[i] - w[i]^2 * zHz_ii
        dη = vinv_ii / w[i] - (u[i]^2) / w[i]
        for k in 1:pσ
            g_βσ[k] += fix.Xσ[i, k] * dη
        end
    end

    d_g = zeros(G)
    for g in 1:G
        r_node = fix.leaf_pos[g]
        s_node = Hinv[r_node, r_node]
        ztwz_node = diag_ZtWZ[r_node]
        trace_term = ztwz_node * s_node
        quad_term = 0.0
        for i in 1:n
            if fix.gidx[i] == g
                quad_term += u[i] * Zâ[i]
            end
        end
        d_g[g] = trace_term - quad_term
    end
    g_α = fix.Zg' * d_g

    grad = vcat(g_βμ, g_βσ, g_α)
    return (nll_sos, grad, quad_sos, quad_sub, logdetV)
end

@testset "Issue #551: Gate 1 - Cancellation Probe" begin
    fix = _make_phylo_lss_fixture(G = 16, n_per_group = 1, seed = 123)
    θ0 = vcat(fix.βμ, fix.βσ, fix.α)

    # Test that GMRF SOS quadratic maintains full accuracy at small sigma_e,
    # while Woodbury subtraction catastrophically cancels as sigma_e -> 0
    for sc in [1.0, 1e-2, 1e-4, 1e-6, 1e-8]
        q_exact, _, _ = _dense_bigfloat_exact(θ0, fix; scale_se = sc)
        nll_sos, nll_sub, q_sos, q_sub, _ = _eval_sparse_lss(θ0, fix; scale_se = sc, want_grad = false)

        # GMRF SOS stays accurate to machine precision across all scales
        @test isapprox(q_sos, q_exact; rtol = 1e-10)

        if sc <= 1e-8
            # Woodbury subtraction severely breaks (relative error >= 100% or negative)
            err_sub = abs(q_sub - q_exact) / abs(q_exact)
            @test err_sub >= 0.99 || q_sub < 0
        end
    end
end

@testset "Issue #551: Gate 2 - Takahashi Gradient Verification" begin
    for (seed, G_val, npg) in [(42, 10, 2), (101, 16, 2), (2026, 8, 3)]
        fix = _make_phylo_lss_fixture(G = G_val, n_per_group = npg, seed = seed)
        θ = vcat(fix.βμ, fix.βσ, fix.α) + 0.1 * randn(MersenneTwister(seed), fix.pμ + fix.pσ + fix.psd)

        _, grad_analytic, _, _, _ = _eval_sparse_lss(θ, fix; scale_se = 1.0, want_grad = true)

        function dense_nll_fd(param)
            pμ, pσ, psd = fix.pμ, fix.pσ, fix.psd
            βμ = param[1:pμ]
            βσ = param[pμ+1:pμ+pσ]
            α = param[pμ+pσ+1:pμ+pσ+psd]
            ημ = fix.Xμ * βμ
            ησ = fix.Xσ * βσ
            ησa = fix.Zg * α
            σa = exp.(ησa)
            Σa = (σa * σa') .* fix.K
            T = eltype(param)
            Vm = Matrix{T}(undef, fix.n, fix.n)
            for j in 1:fix.n, i in 1:fix.n
                Vm[i, j] = Σa[fix.gidx[i], fix.gidx[j]]
            end
            for i in 1:fix.n
                Vm[i, i] += exp(2 * ησ[i])
            end
            Vfac = cholesky(Symmetric(Vm))
            r = fix.y .- ημ
            return 0.5 * (logdet(Vfac) + dot(r, Vfac \ r) + fix.n * log(2π))
        end

        grad_fd = ForwardDiff.gradient(dense_nll_fd, θ)

        @test isapprox(grad_analytic, grad_fd; atol = 1e-6)
    end
end
