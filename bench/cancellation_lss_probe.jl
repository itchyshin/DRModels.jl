# bench/cancellation_lss_probe.jl
# Probe for DRModels.jl issue #551:
# 1. Cancellation Gate Probe: test numerical stability of Woodbury subtraction
#    vs GMRF sum-of-squares at small σ_e (1e-1 down to 1e-10) with non-constant D_a = exp(Z_g α).
# 2. Analytic Takahashi gradient for α, β_μ, β_σ verified against ForwardDiff.

using LinearAlgebra
using SparseArrays
using Random
using ForwardDiff
using Printf
using DRModels

function build_fixture(; G = 16, n_per_group = 2, seed = 42)
    rng = MersenneTwister(seed)
    phy = DRModels.random_balanced_tree(G; branch_length = 0.2)
    Q, leaf_pos, q = DRModels.augmented_tree_precision(phy)
    Qs = dropzeros!(sparse(Symmetric(Matrix(Q))))
    chQ = cholesky(Symmetric(Qs); check = false)
    Qinv = DRModels.takahashi_selinv(chQ)
    leaf_var = [Qinv[leaf_pos[t], leaf_pos[t]] for t in 1:G]
    inv_sd = [1.0 / sqrt(leaf_var[t]) for t in 1:G]
    logdetCprior = -logdet(chQ)

    # Dense phylogenetic correlation matrix
    K = DRModels._phylo_correlation(phy)

    # Design matrices
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

    # Generate random effects and observations
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

# Dense NLL evaluator in Float64
function dense_nll(θ, fix; scale_se = 1.0)
    pμ, pσ, psd = fix.pμ, fix.pσ, fix.psd
    βμ = θ[1:pμ]
    βσ = θ[pμ+1:pμ+pσ]
    α = θ[pμ+pσ+1:pμ+pσ+psd]

    ημ = fix.Xμ * βμ
    ησ = fix.Xσ * βσ
    ησa = fix.Zg * α
    σa = exp.(ησa)
    Σa = (σa * σa') .* fix.K

    T = eltype(θ)
    n = fix.n
    Vm = Matrix{T}(undef, n, n)
    @inbounds for j in 1:n, i in 1:n
        Vm[i, j] = Σa[fix.gidx[i], fix.gidx[j]]
    end
    @inbounds for i in 1:n
        Vm[i, i] += (scale_se * exp(ησ[i]))^2
    end
    Vfac = cholesky(Symmetric(Vm); check = false)
    issuccess(Vfac) || return convert(T, 1e18)
    r = fix.y .- ημ
    return 0.5 * (logdet(Vfac) + dot(r, Vfac \ r) + n * log(2π))
end

# Dense NLL and Quad in BigFloat for ground truth when Float64 dense ill-conditions
function ground_truth_dense(θ, fix; scale_se = 1.0)
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

# Sparse NLL and component evaluator
function sparse_eval(θ, fix; scale_se = 1.0, want_grad = false)
    pμ, pσ, psd = fix.pμ, fix.pσ, fix.psd
    βμ = θ[1:pμ]
    βσ = θ[pμ+1:pμ+pσ]
    α = θ[pμ+pσ+1:pμ+pσ+psd]

    n = fix.n
    G = fix.G
    m = fix.q
    r = fix.y .- fix.Xμ * βμ

    # Residual weights w = 1/σ_e,i²
    ησ = fix.Xσ * βσ
    σε = scale_se .* exp.(ησ)
    w = 1.0 ./ (σε .^ 2)

    # Observation weights: wts[i] = exp((Zg*α)[gidx[i]]) / √v_leaf[gidx[i]]
    ησa = fix.Zg * α
    σa = exp.(ησa)
    wts_g = σa .* fix.inv_sd # length G
    wts = [wts_g[fix.gidx[i]] for i in 1:n] # length n

    # Build sparse Z (n × m)
    I_idx = 1:n
    J_idx = [fix.leaf_pos[fix.gidx[i]] for i in 1:n]
    Z = sparse(I_idx, J_idx, wts, n, m)

    # ZᵀWZ is diagonal on the augmented latent space (each obs attached to 1 leaf)
    # Diagonal elements: for leaf r = leaf_pos[g], (ZᵀWZ)_rr = wts_g[g]² * sum_{i: gidx[i]=g} w_i
    diag_ZtWZ = zeros(m)
    for i in 1:n
        r_node = fix.leaf_pos[fix.gidx[i]]
        diag_ZtWZ[r_node] += w[i] * wts[i]^2
    end

    H = fix.Q + Diagonal(diag_ZtWZ)
    chH = cholesky(Symmetric(H); check = false)
    issuccess(chH) || return (1e18, Float64[], 0.0, 0.0, 0.0)

    # b = Zᵀ W r (sparse vector with nonzeros at leaf positions)
    b = zeros(m)
    for i in 1:n
        r_node = fix.leaf_pos[fix.gidx[i]]
        b[r_node] += wts[i] * w[i] * r[i]
    end

    â = chH \ b
    Zâ = [wts[i] * â[fix.leaf_pos[fix.gidx[i]]] for i in 1:n]
    res_lat = r .- Zâ

    logdetH = logdet(chH)
    logdetD = -sum(log, w) # sum 2 log σ_e,i
    logdetV = logdetD + fix.logdetCprior + logdetH

    # Quadratic form evaluations:
    # Method A: Woodbury subtraction
    quad_sub = dot(r, w .* r) - dot(b, â)

    # Method B: GMRF sum of squares (stable, no subtraction of large numbers)
    quad_sos = dot(res_lat, w .* res_lat) + dot(â, fix.Q * â)

    nll_sub = 0.5 * (logdetV + quad_sub + n * log(2π))
    nll_sos = 0.5 * (logdetV + quad_sos + n * log(2π))

    if !want_grad
        return (nll_sos, nll_sub, quad_sos, quad_sub, logdetV)
    end

    # -----------------------------------------------------------------
    # Analytic Takahashi gradients
    # -----------------------------------------------------------------
    Hinv = DRModels.takahashi_selinv(chH)

    # u = V⁻¹ r = W (r - Z â)
    u = w .* res_lat

    # 1. Gradient w.r.t. β_μ: - Xμᵀ u
    g_βμ = - (fix.Xμ' * u)

    # 2. Gradient w.r.t. β_σ:
    # (Z H⁻¹ Zᵀ)_ii = wts[i]² * (H⁻¹)_{leaf, leaf}
    # (V⁻¹)_ii = w_i - w_i² * (Z H⁻¹ Zᵀ)_ii
    # ∂NLL / ∂ησ_i = (V⁻¹)_ii / w_i - u_i² / w_i = 1 - w_i * (Z H⁻¹ Zᵀ)_ii - σ_e,i² u_i²
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

    # 3. Gradient w.r.t. α:
    # For each group g:
    # d_g = (ZᵀWZ)_{leaf,leaf} * S_{leaf,leaf} - sum_{i: gidx[i]=g} u_i * (Z â)_i
    # g_α = Zgᵀ d
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

println("="^95)
println("GATE 1A: CANCELLATION PROBE WITH n = G (1 OBS PER SPECIES, NO RESIDUAL RESIDUE)")
println("="^95)

fix1 = build_fixture(G = 16, n_per_group = 1, seed = 123)
θ1 = vcat(fix1.βμ, fix1.βσ, fix1.α)

scales = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10]

@printf("%-10s | %-16s | %-16s | %-16s | %-12s | %-12s\n",
        "scale(σ_e)", "Exact (BigFloat)", "GMRF SOS Quad", "Woodbury Sub Quad", "RelErr(SOS)", "RelErr(Sub)")
println("-"^95)

for sc in scales
    q_exact, _, nll_exact = ground_truth_dense(θ1, fix1; scale_se = sc)
    nll_sos, nll_sub, q_sos, q_sub, _ = sparse_eval(θ1, fix1; scale_se = sc, want_grad = false)

    err_sos = abs(q_sos - q_exact) / abs(q_exact)
    err_sub = abs(q_sub - q_exact) / abs(q_exact)

    @printf("%-10.1e | %-16.8e | %-16.8e | %-16.8e | %-12.2e | %-12.2e\n",
            sc, q_exact, q_sos, q_sub, err_sos, err_sub)
end

println("\n" * "="^95)
println("GATE 1B: CANCELLATION PROBE WITH n = 2G (REPLICATES PER SPECIES)")
println("="^95)

fix2 = build_fixture(G = 16, n_per_group = 2, seed = 123)
θ2 = vcat(fix2.βμ, fix2.βσ, fix2.α)

@printf("%-10s | %-16s | %-16s | %-16s | %-12s | %-12s\n",
        "scale(σ_e)", "Exact (BigFloat)", "GMRF SOS Quad", "Woodbury Sub Quad", "RelErr(SOS)", "RelErr(Sub)")
println("-"^95)

for sc in scales
    q_exact, _, nll_exact = ground_truth_dense(θ2, fix2; scale_se = sc)
    nll_sos, nll_sub, q_sos, q_sub, _ = sparse_eval(θ2, fix2; scale_se = sc, want_grad = false)

    err_sos = abs(q_sos - q_exact) / abs(q_exact)
    err_sub = abs(q_sub - q_exact) / abs(q_exact)

    @printf("%-10.1e | %-16.8e | %-16.8e | %-16.8e | %-12.2e | %-12.2e\n",
            sc, q_exact, q_sos, q_sub, err_sos, err_sub)
end

println("\n" * "="^80)
println("GATE 2: ANALYTIC TAKAHASHI GRADIENT VERIFICATION AGAINST FORWARDDIFF")
println("="^80)

for (test_seed, test_G, test_npg) in [(42, 10, 2), (101, 16, 2), (2026, 8, 3)]
    fix_t = build_fixture(G = test_G, n_per_group = test_npg, seed = test_seed)
    θ_t = vcat(fix_t.βμ, fix_t.βσ, fix_t.α) + 0.1 * randn(MersenneTwister(test_seed), fix_t.pμ + fix_t.pσ + fix_t.psd)

    nll_val, grad_analytic, _, _, _ = sparse_eval(θ_t, fix_t; scale_se = 1.0, want_grad = true)

    # ForwardDiff on dense NLL
    f_dense(θ) = dense_nll(θ, fix_t; scale_se = 1.0)
    grad_fd_dense = ForwardDiff.gradient(f_dense, θ_t)

    # ForwardDiff on sparse NLL (using SOS form)
    # Define a pure ForwardDiff-compatible sparse nll
    function sparse_nll_autodiff(θ)
        pμ, pσ, psd = fix_t.pμ, fix_t.pσ, fix_t.psd
        βμ = θ[1:pμ]
        βσ = θ[pμ+1:pμ+pσ]
        α = θ[pμ+pσ+1:pμ+pσ+psd]
        n, G, m = fix_t.n, fix_t.G, fix_t.q
        r = fix_t.y .- fix_t.Xμ * βμ
        ησ = fix_t.Xσ * βσ
        w = exp.(-2 .* ησ)
        ησa = fix_t.Zg * α
        σa = exp.(ησa)
        wts_g = σa .* fix_t.inv_sd
        wts = [wts_g[fix_t.gidx[i]] for i in 1:n]

        T = eltype(θ)
        diag_ZtWZ = zeros(T, m)
        for i in 1:n
            r_node = fix_t.leaf_pos[fix_t.gidx[i]]
            diag_ZtWZ[r_node] += w[i] * wts[i]^2
        end

        H = Matrix(fix_t.Q) + Diagonal(diag_ZtWZ)
        chH = cholesky(Symmetric(H))
        b = zeros(T, m)
        for i in 1:n
            r_node = fix_t.leaf_pos[fix_t.gidx[i]]
            b[r_node] += wts[i] * w[i] * r[i]
        end
        â = chH \ b
        Zâ = [wts[i] * â[fix_t.leaf_pos[fix_t.gidx[i]]] for i in 1:n]
        res_lat = r .- Zâ
        logdetH = logdet(chH)
        logdetD = -sum(log, w)
        logdetV = logdetD + fix_t.logdetCprior + logdetH
        quad_sos = dot(res_lat, w .* res_lat) + dot(â, fix_t.Q * â)
        return 0.5 * (logdetV + quad_sos + n * log(2π))
    end

    grad_fd_sparse = ForwardDiff.gradient(sparse_nll_autodiff, θ_t)

    max_diff_dense = maximum(abs.(grad_analytic .- grad_fd_dense))
    max_diff_sparse = maximum(abs.(grad_analytic .- grad_fd_sparse))

    println("Seed $(test_seed) (n=$(fix_t.n), G=$(fix_t.G), params=$(length(θ_t))):")
    println("  Param names: [β_μ1, β_μ2, β_σ1, β_σ2, α_1, α_2]")
    println("  Analytic grad:     ", round.(grad_analytic, digits=8))
    println("  FD dense grad:     ", round.(grad_fd_dense, digits=8))
    println("  FD sparse grad:    ", round.(grad_fd_sparse, digits=8))
    println("  Max |Analytic - FD_dense|:  ", @sprintf("%.2e", max_diff_dense))
    println("  Max |Analytic - FD_sparse|: ", @sprintf("%.2e", max_diff_sparse))
    @assert max_diff_dense < 1e-6 "Analytic gradient vs FD dense mismatch!"
    @assert max_diff_sparse < 1e-6 "Analytic gradient vs FD sparse mismatch!"
    println("  -> PASSED (tol <= 1e-6)")
end
