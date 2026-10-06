# test/test_lss_sparse_multi.jl
# Issue #563, Phase-2 slice S7b.1: sparse MULTI-component block assembly +
# objective. Generalises test_lss_sparse.jl's (#551) single-phylo-component
# oracle to a nested two-component fixture (one phylogenetic `sd()`, one iid
# `sd()` NESTED within the phylo grouping), per
# docs/src/developer-notes/lss-sparse-multi-component.md.
#
# Oracle 1 (design note §5): DRModels._lss_sparse_multi_objective(θ, …) must equal
# an INDEPENDENT dense reconstruction of the marginal Gaussian NLL (not the
# dense route's own internal `nll_ml` closure, to avoid a shared-bug false
# positive — the same independence _s5a_dense_lss_nll gives the
# single-component route in test_sparse_precision_storage.jl) at three fixed
# θ. The assembled H's sparsity/PD structure is checked against §2.1's
# nested-fixture fill band directly.
#
# Assembly + objective only (S7b.1) — no gradients, no REML, no router
# change: this test never calls `drm(...; algorithm = :sparse)` on the
# multi-component model, only DRModels._lss_sparse_multi_objective directly.

using Test
using DRModels
using StableRNGs
using LinearAlgebra
using SparseArrays

# Same balanced-tree Newick builder as test_lss_sparse.jl (depth d -> 2^d tips).
function _make_balanced_newick(d)
    function node(prefix, depth)
        depth == 0 && return "$(prefix):$(1/d)"
        l = node(prefix * "a", depth - 1); r = node(prefix * "b", depth - 1)
        return depth == d ? "($l,$r);" : "($l,$r):$(1/d)"
    end
    node("t", d)
end

# Nested fixture: one phylogenetic component (`sd(species, phylogenetic) ~ 1
# + z`, 64 species from a balanced tree, depth = 6) and one iid component
# (`sd(site) ~ 1`, sites NESTED within species, 3 sites/species), n = 2
# rows/site (n = 384). Returns the tree, a `NamedTuple` data table for the
# dense `drm(...)` route, and every raw array the sparse-comp builders need.
function _s7b1_nested_lsss_fixture(; depth = 6, sites_per_species = 3, n_per_site = 2,
                                   seed = 20260902)
    phy = DRModels.augmented_phy(_make_balanced_newick(depth))
    Gsp = phy.n_leaves
    sp_names = String.(phy.leaf_names)
    Gsite = Gsp * sites_per_species
    rng = StableRNG(seed)

    z_sp = randn(rng, Gsp)                                # species-level covariate
    site_species = repeat(1:Gsp, inner = sites_per_species)  # site -> species (nested)
    n = Gsite * n_per_site
    site_idx = repeat(1:Gsite, inner = n_per_site)
    species_idx = site_species[site_idx]
    x = randn(rng, n)
    z = z_sp[species_idx]

    βμ = [0.8, 0.4]
    βσ = [-1.0]
    α_phy = [-0.6, 0.25]     # sd(species, phylogenetic) ~ 1 + z
    α_iid = [-0.9]           # sd(site) ~ 1

    K0 = DRModels.sigma_phy_dense(phy; σ²_phy = 1.0)
    dK = sqrt.(diag(K0)); K = K0 ./ (dK * dK')
    chK = cholesky(Symmetric(K))
    u_phy = chK.L * randn(rng, Gsp)
    σ_phy_sp = exp.(α_phy[1] .+ α_phy[2] .* z_sp)
    a_phy = σ_phy_sp .* u_phy

    σ_iid_site = fill(exp(α_iid[1]), Gsite)
    a_iid = σ_iid_site .* randn(rng, Gsite)

    σ_e = fill(exp(βσ[1]), n)
    y = βμ[1] .+ βμ[2] .* x .+ a_phy[species_idx] .+ a_iid[site_idx] .+ σ_e .* randn(rng, n)

    dat = (y = y, x = x, z = z,
           species = sp_names[species_idx],
           site = ["site$(g)" for g in site_idx])

    return (phy = phy, dat = dat, n = n, Gsp = Gsp, Gsite = Gsite,
            species_idx = species_idx, site_idx = site_idx, x = x, z_sp = z_sp)
end

# Independent dense oracle (design note §1): V = diag(σ_e,i²) +
# D_phy K_phy D_phy [gidx_phy] + D_iid² [gidx_iid] (K_iid = I). Generalises
# test_sparse_precision_storage.jl's `_s5a_dense_lss_nll` (one phylo
# component) by adding the iid component's diagonal contribution, exactly as
# gaussian_lss.jl's `Vm[i,j] += …` accumulates one term per component
# (:576, :606) — but reconstructed here from the model definition (§1), not
# by calling `_fit_gaussian_lss_multi` itself, so the two routes cannot share
# a bug.
function _s7b1_dense_multi_lss_nll(theta, phy, y, Xmu, Xsigma, Zg_phy, gidx_phy, Zg_iid, gidx_iid)
    p_mu = size(Xmu, 2); p_sigma = size(Xsigma, 2)
    p_phy = size(Zg_phy, 2); p_iid = size(Zg_iid, 2)
    beta_mu = theta[1:p_mu]
    beta_sigma = theta[(p_mu + 1):(p_mu + p_sigma)]
    alpha_phy = theta[(p_mu + p_sigma + 1):(p_mu + p_sigma + p_phy)]
    alpha_iid = theta[(p_mu + p_sigma + p_phy + 1):(p_mu + p_sigma + p_phy + p_iid)]

    Q, leaf_pos, _ = augmented_tree_precision(phy)
    leaf_covariance = inv(Matrix(Q))[leaf_pos, leaf_pos]
    leaf_sd = sqrt.(diag(leaf_covariance))
    K_phy = leaf_covariance ./ (leaf_sd * leaf_sd')

    residual_variance = exp.(2 .* (Xsigma * beta_sigma))
    phylo_sd = exp.(Zg_phy * alpha_phy)
    iid_sd = exp.(Zg_iid * alpha_iid)

    # iid contribution: K_c = I over GROUP levels, but every pair of
    # observations sharing a group level still shares that group's random
    # draw, so Cov(y_i, y_j) = σ_iid,g² for EVERY (i,j) with gidx_iid[i] ==
    # gidx_iid[j] (i≠j included) — not just the diagonal. Same accumulation
    # `_fit_gaussian_lss_multi` uses for an iid component (gaussian_lss.jl,
    # `if c.gidx[i] == c.gidx[j]; Vm[i,j] += σa[c.gidx[i]]^2`).
    n = length(y)
    Viid = zeros(n, n)
    for j in 1:n, i in 1:n
        gidx_iid[i] == gidx_iid[j] && (Viid[i, j] = iid_sd[gidx_iid[i]]^2)
    end

    V = Diagonal(residual_variance) +
        Diagonal(phylo_sd[gidx_phy]) * K_phy[gidx_phy, gidx_phy] * Diagonal(phylo_sd[gidx_phy]) +
        Viid

    factor = cholesky(Symmetric(Matrix(V)))
    residual = y - Xmu * beta_mu
    return 0.5 * (logdet(factor) + dot(residual, factor \ residual) + length(y) * log(2π))
end

@testset "Sparse multi-component LSS block assembly (#563 S7b.1)" begin
    fx = _s7b1_nested_lsss_fixture()
    phy, dat, n = fx.phy, fx.dat, fx.n

    f = bf(@formula(y ~ x + (1 | site) + phylo(1 | species)),
           @formula(sigma ~ 1),
           @formula(sd(site) ~ 1),
           @formula(sd(species, phylogenetic) ~ 1 + z))
    fit_dense = drm(f, Gaussian(); data = dat, tree = phy)
    @test fit_dense.converged

    Xmu = hcat(ones(n), fx.x)
    Xsigma = ones(n, 1)
    Zg_phy = hcat(ones(fx.Gsp), fx.z_sp)
    Zg_iid = ones(fx.Gsite, 1)

    phy_comp = DRModels._sparse_lss_phylo_comp(fx.species_idx, fx.Gsp, Zg_phy, phy)
    iid_comp = DRModels._sparse_lss_iid_comp(fx.site_idx, fx.Gsite, Zg_iid)
    comps = [phy_comp, iid_comp]

    θ_opt = vcat(coef(fit_dense, :mu), coef(fit_dense, :sigma),
                 coef(fit_dense, :sd_phylo), coef(fit_dense, :sd))

    rng2 = StableRNG(1)
    θ_perturbed = θ_opt .+ 0.15 .* randn(rng2, length(θ_opt))

    θ_boundary = copy(θ_opt)
    θ_boundary[end] = -6.0   # the iid component's (scalar) log-SD -> e^{-6}

    dense_oracle(θ) = _s7b1_dense_multi_lss_nll(θ, phy, dat.y, Xmu, Xsigma, Zg_phy,
                                                fx.species_idx, Zg_iid, fx.site_idx)
    sparse_objective(θ) = DRModels._lss_sparse_multi_objective(θ, dat.y, Xmu, Xsigma, comps)

    @testset "objective identity vs. independent dense oracle" begin
        for (label, θ) in (("dense optimum", θ_opt), ("perturbed", θ_perturbed),
                           ("boundary (iid logSD ≈ -6)", θ_boundary))
            @test isapprox(sparse_objective(θ), dense_oracle(θ); atol = 1e-8)
        end
    end

    @testset "assembled H: sparsity pattern and PD-ness (design note §2.1)" begin
        asm = DRModels._lss_sparse_multi_assemble(θ_opt, dat.y, Xmu, Xsigma, comps)
        H = asm.H
        p = size(H, 1)
        @test H ≈ H'                       # exactly symmetric by construction
        ch = cholesky(Symmetric(H); check = false)
        @test issuccess(ch)                # PD
        fillratio = nnz(sparse(ch.L)) / p
        # Measured (not assumed, per the design note's own "measure the fill"
        # discipline, §2.1): 1.99 for THIS fixture. The note's headline ≈3.0
        # was measured at G_c=50 ≪ p=1000-4000 (iid groups far fewer than
        # tips); this fixture's iid factor (192 sites) instead OUTNUMBERS its
        # 64 tips, a different corner of the same nested regime, so a
        # different — still comfortably sub-crossed (note's crossed cases
        # measure ~5-11+) — constant is expected. Band brackets the measured
        # value with margin as a regression guard, not a re-confirmation of
        # the note's own number.
        @test 1.5 <= fillratio <= 3.0

    end
end
