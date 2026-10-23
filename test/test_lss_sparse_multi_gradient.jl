# test/test_lss_sparse_multi_gradient.jl
# Issue #563, Phase-2 sub-slices S7b.2/S7b.2b: exact analytic gradient of the
# sparse multi-component LSS objective (`DRModels._lss_sparse_multi_objective_and_grad`),
# generalising #551's single-component gradient to `m` components sharing the
# block augmented precision `H`, per
# docs/src/developer-notes/lss-sparse-multi-component.md §3 (alignment table)
# and §8 finding 4 (the cross-component correction to `g_α,c`).
#
# Self-contained: this file does NOT depend on test_lss_sparse_multi.jl being
# included first (its own fixture builder is a local copy, distinctly named,
# of the S7b.1 nested fixture) so it can be run standalone.
#
# Oracle 2 (design note §5): FD-vs-exact gradient at three points (dense
# optimum, a perturbed point, a boundary-ish point with one iid log-SD ≈ -6),
# central difference with a step scan (as test_575_exact_reml_gradient.jl
# does), relative error <= 1e-6 on every coordinate.
#
# Oracle 3 (design note §5, §8 finding 4): a "diagonal-only" counter-example
# — the SAME evaluator with `cross_terms = false` (own-node Hinv term only,
# no cross-component accumulation) — must DISAGREE with the FD gradient by
# more than 1e-3 relative on at least one alpha coordinate. This pins the
# review's finding so a regression to the wrong (#551-generalised) formula
# is caught, not silently reintroduced.

using Test
using DRModels
using StableRNGs
using LinearAlgebra
using SparseArrays

# Same balanced-tree Newick builder and nested fixture as test_lss_sparse_multi.jl
# (S7b.1), copied here (distinctly named) to keep this file runnable standalone.
function _s7b2_make_balanced_newick(d)
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
function _s7b2_nested_lsss_fixture(; depth = 6, sites_per_species = 3, n_per_site = 2,
                                   seed = 20260902)
    phy = DRModels.augmented_phy(_s7b2_make_balanced_newick(depth))
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

# Central-difference gradient with a step scan (as test_575_exact_reml_gradient.jl):
# take the Richardson-adjacent pair whose difference is smallest.
function _s7b2_fd_grad(f, θ)
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

@testset "Sparse multi-component LSS exact gradient (#563 S7b.2/S7b.2b)" begin
    fx = _s7b2_nested_lsss_fixture()
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

    objective(θ) = DRModels._lss_sparse_multi_objective(θ, dat.y, Xmu, Xsigma, comps)

    points = ("dense optimum" => θ_opt, "perturbed" => θ_perturbed,
              "boundary (iid logSD ≈ -6)" => θ_boundary)

    @testset "exact gradient matches step-scanned central-FD (oracle 2)" begin
        for (label, θ) in points
            nll, g_exact = DRModels._lss_sparse_multi_objective_and_grad(θ, dat.y, Xmu, Xsigma, comps)

            # Objective must be bit-identical to _lss_sparse_multi_objective (S7b.1).
            @test nll === objective(θ)

            g_fd = _s7b2_fd_grad(objective, θ)
            err = maximum(abs.(g_exact .- g_fd))
            scale = max(1.0, maximum(abs, g_fd))
            @info "exact-vs-FD sparse multi-component gradient" point = label max_abs_err = err rel = err / scale
            @test err / scale <= 1e-6
        end
    end

    @testset "diagonal-only cross-term regression guard (oracle 3, §8 finding 4)" begin
        # At every point checked, log the diagonal-only formula's disagreement
        # with FD, but assert the wide-margin failure (> 1e-3 relative) only
        # over ALL coordinates pooled across points, not point-by-point: at
        # the boundary point the iid component's `wts` -> 0, so the cross
        # term itself vanishes and the wrong formula happens to sit close to
        # FD there too (a real, expected feature of that limit, not a flaw in
        # the regression guard) — pooling keeps the guard meaningful without
        # asserting something false about that limiting case.
        pmu = size(Xmu, 2); psig = size(Xsigma, 2)
        all_rel_errs = Float64[]
        for (label, θ) in points
            _, g_diag = DRModels._lss_sparse_multi_objective_and_grad(θ, dat.y, Xmu, Xsigma, comps;
                                                                  cross_terms = false)
            g_fd = _s7b2_fd_grad(objective, θ)

            iα = (pmu + psig + 1):length(θ)
            g_diag_α = g_diag[iα]
            g_fd_α = g_fd[iα]
            scale = max(1.0, maximum(abs, g_fd_α))
            rel_errs = abs.(g_diag_α .- g_fd_α) ./ scale
            @info "diagonal-only (WRONG) alpha gradient vs FD" point = label max_rel_err = maximum(rel_errs)
            append!(all_rel_errs, rel_errs)
        end
        @test maximum(all_rel_errs) > 1e-3
    end
end
