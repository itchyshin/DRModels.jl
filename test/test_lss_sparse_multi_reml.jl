# test/test_lss_sparse_multi_reml.jl
# Issue #563, Phase-2 sub-slice S7b.3: the REML (Patterson-Thompson) correction
# for the sparse multi-component LSS objective and its gradient, per
# docs/src/developer-notes/lss-sparse-multi-component.md §4.
#
# Self-contained: local copy of the S7b.1/S7b.2 nested fixture (distinctly
# named) so this file runs standalone.
#
# Oracle 1 (design note §5, extended to REML): DRModels._lss_sparse_multi_objective(θ,
# ...; reml = true) must equal an INDEPENDENT dense reconstruction: the S7b.1
# dense ML oracle PLUS 0.5*logdet(Xμ'V⁻¹Xμ) - 0.5*pμ*log(2π) -- the same
# normalisation convention _fit_gaussian_lss_multi's REML branch
# (gaussian_lss.jl:634-638) and #558's single-component eval_reml
# (gaussian_sparse_lss.jl:135-156) both use -- at three fixed θ, atol = 1e-8.
#
# Oracle 2: the REML gradient (DRModels._lss_sparse_multi_objective_and_grad(...;
# reml = true)) vs step-scanned central-FD on the REML objective, relative
# error <= 1e-6 on β_σ and every α coordinate at the same three θ.
#
# β_μ profiled-out contract (design note §4: XtVinvX = Xμ'V⁻¹Xμ has no β_μ
# dependence, so the REML correction term's gradient w.r.t. β_μ is exactly
# zero): the REML gradient's β_μ block must equal the ML gradient's β_μ block
# EXACTLY (not merely to FD tolerance) at every θ tested -- the same contract
# reml_q4.jl / gaussian_bivariate.jl document for their own profiled-out β's.
#
# ML bit-identity (S7b.1/S7b.2 behaviour unchanged): reml = false (the
# default) must reproduce the exact same values as calling the entries with
# no `reml` keyword at all.

using Test
using DRModels
using StableRNGs
using LinearAlgebra
using SparseArrays

# Same balanced-tree Newick builder and nested fixture as
# test_lss_sparse_multi.jl / test_lss_sparse_multi_gradient.jl, copied here
# (distinctly named) to keep this file runnable standalone.
function _s7b3_make_balanced_newick(d)
    function node(prefix, depth)
        depth == 0 && return "$(prefix):$(1/d)"
        l = node(prefix * "a", depth - 1); r = node(prefix * "b", depth - 1)
        return depth == d ? "($l,$r);" : "($l,$r):$(1/d)"
    end
    node("t", d)
end

# One phylogenetic component (`sd(species, phylogenetic) ~ 1 + z`, 64 species
# from a balanced tree, depth = 6) and one iid component (`sd(site) ~ 1`,
# sites NESTED within species, 3 sites/species), n = 2 rows/site (n = 384).
function _s7b3_nested_lsss_fixture(; depth = 6, sites_per_species = 3, n_per_site = 2,
                                   seed = 20260902)
    phy = DRModels.augmented_phy(_s7b3_make_balanced_newick(depth))
    Gsp = phy.n_leaves
    sp_names = String.(phy.leaf_names)
    Gsite = Gsp * sites_per_species
    rng = StableRNG(seed)

    z_sp = randn(rng, Gsp)
    site_species = repeat(1:Gsp, inner = sites_per_species)
    n = Gsite * n_per_site
    site_idx = repeat(1:Gsite, inner = n_per_site)
    species_idx = site_species[site_idx]
    x = randn(rng, n)
    z = z_sp[species_idx]

    βμ = [0.8, 0.4]
    βσ = [-1.0]
    α_phy = [-0.6, 0.25]
    α_iid = [-0.9]

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

# Independent dense ML oracle (copy of test_lss_sparse_multi.jl's
# _s7b1_dense_multi_lss_nll, distinctly named), extended with the REML
# correction term using the SAME normalisation convention the dense route
# reports (gaussian_lss.jl:634-638, `+ 0.5*logdet(chX) - const_pμ`) and #558's
# single-component eval_reml (gaussian_sparse_lss.jl:153-155): + 0.5*logdet(
# Xμ'V⁻¹Xμ) - 0.5*pμ*log(2π). Reconstructed from the model definition, not by
# calling any DRModels.jl fitting routine, so it cannot share a bug with the sparse
# implementation under test.
function _s7b3_dense_multi_lss_nll(theta, phy, y, Xmu, Xsigma, Zg_phy, gidx_phy, Zg_iid, gidx_iid;
                                   reml::Bool = false)
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
    nll_ml = 0.5 * (logdet(factor) + dot(residual, factor \ residual) + length(y) * log(2π))
    reml || return nll_ml

    A = factor \ Xmu
    XtVinvX = Xmu' * A
    chX = cholesky(Symmetric(XtVinvX))
    return nll_ml + 0.5 * logdet(chX) - 0.5 * p_mu * log(2π)
end

# Central-difference gradient with a step scan (as test_575_exact_reml_gradient.jl
# and test_lss_sparse_multi_gradient.jl): take the Richardson-adjacent pair
# whose difference is smallest.
function _s7b3_fd_grad(f, θ)
    np = length(θ)
    g = zeros(np)
    for k in 1:np
        best = NaN; best_gap = Inf; prev = NaN
        for h in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5)
            θp = copy(θ); θp[k] += h
            θm = copy(θ); θm[k] -= h
            d = (f(θp) - f(θm)) / (2h)
            if isfinite(prev) && abs(d - prev) < best_gap
                best_gap = abs(d - prev); best = d
            end
            prev = d
        end
        g[k] = best
    end
    return g
end

@testset "Sparse multi-component LSS REML correction (#563 S7b.3)" begin
    fx = _s7b3_nested_lsss_fixture()
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

    pμ = size(Xmu, 2); pσ = size(Xsigma, 2)
    iβμ = 1:pμ

    dense_reml_oracle(θ) = _s7b3_dense_multi_lss_nll(θ, phy, dat.y, Xmu, Xsigma, Zg_phy,
                                                      fx.species_idx, Zg_iid, fx.site_idx;
                                                      reml = true)
    ml_objective(θ) = DRModels._lss_sparse_multi_objective(θ, dat.y, Xmu, Xsigma, comps)
    reml_objective(θ) = DRModels._lss_sparse_multi_objective(θ, dat.y, Xmu, Xsigma, comps; reml = true)

    points = ("dense optimum" => θ_opt, "perturbed" => θ_perturbed,
              "boundary (iid logSD ≈ -6)" => θ_boundary)

    @testset "ML behaviour bit-identical for reml=false (S7b.1/S7b.2 unchanged)" begin
        for (label, θ) in points
            @test DRModels._lss_sparse_multi_objective(θ, dat.y, Xmu, Xsigma, comps; reml = false) ===
                  ml_objective(θ)

            nll_default, grad_default = DRModels._lss_sparse_multi_objective_and_grad(θ, dat.y, Xmu, Xsigma, comps)
            nll_explicit, grad_explicit = DRModels._lss_sparse_multi_objective_and_grad(θ, dat.y, Xmu, Xsigma, comps;
                                                                                    reml = false)
            @test nll_default === nll_explicit
            @test grad_default == grad_explicit
        end
    end

    @testset "REML objective identity vs. independent dense oracle" begin
        for (label, θ) in points
            @test isapprox(reml_objective(θ), dense_reml_oracle(θ); atol = 1e-8)
        end
    end

    @testset "REML gradient matches step-scanned central-FD; β_μ profiled out" begin
        for (label, θ) in points
            nll_ml, grad_ml = DRModels._lss_sparse_multi_objective_and_grad(θ, dat.y, Xmu, Xsigma, comps)
            nll_reml, grad_reml = DRModels._lss_sparse_multi_objective_and_grad(θ, dat.y, Xmu, Xsigma, comps;
                                                                            reml = true)
            @test isapprox(nll_reml, reml_objective(θ); atol = 1e-10)
            @test length(grad_reml) == length(θ)

            # β_μ profiled-out contract: XtVinvX = Xμ'V⁻¹Xμ has no β_μ
            # dependence, so the REML correction's gradient w.r.t. β_μ is
            # exactly zero -- the REML entry's β_μ block must equal the ML
            # entry's β_μ block EXACTLY, not merely to FD tolerance.
            @test grad_reml[iβμ] == grad_ml[iβμ]

            g_fd = _s7b3_fd_grad(reml_objective, θ)
            not_mu = (pμ+1):length(θ)
            err = maximum(abs.(grad_reml[not_mu] .- g_fd[not_mu]))
            scale = max(1.0, maximum(abs, g_fd[not_mu]))
            @info "exact-vs-FD sparse multi-component REML gradient" point = label max_abs_err = err rel = err / scale
            @test err / scale <= 1e-6
        end
    end
end
