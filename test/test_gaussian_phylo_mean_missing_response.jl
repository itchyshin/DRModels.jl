# test_gaussian_phylo_mean_missing_response.jl — issue #482.
#
# Gaussian mean-phylo (`mu ~ x + phylo(1 | species)`, `sigma ~ 1`) had NO
# working `engine = "julia"` path for a missing response:
#
#   - `response = "drop"` (the R-bridge DEFAULT) — the bridge pre-drops the
#     NA-response rows before marshalling to Julia, so a species whose every
#     row was missing vanishes from the data entirely while the phylo TREE
#     payload still carries the full leaf set. The sparse phylo-mean routes
#     (`_fit_structured_gaussian_sparse_lbfgs` / `_em`) mapped species to tree
#     leaves by first-seen POSITION (`_group_index`), which requires an exact
#     `distinct-levels == n_leaves` bijection — so a species SUBSET threw a
#     confusing dimension-mismatch error naming an unrelated algorithm.
#   - `response = "include"` — the bridge sends the NaN-response rows straight
#     through; DRModels.jl's `has_missing_response` gate refuses ANY structured
#     mean route unconditionally (native reproduction, no bridge involved).
#
# Fixed the DEFAULT (`drop`) case by matching species to tree leaves BY NAME/
# tip-index (`_phylo_mean_leaf_index`, mirroring the σ-phylo and Poisson-phylo
# routes' existing convention) instead of by position — a species subset now
# lands on the CORRECT tree leaf and simply carries no likelihood term for the
# leaves it's missing from, exactly like the σ-phylo route already documents.
# `include` remains refused (no missing-response likelihood exists yet for the
# phylo-mean route) but the message is sharpened to say so honestly.
using DRModels
using Test, Random, LinearAlgebra, Statistics

@testset "#482 Gaussian mean-phylo missing response" begin

    @testset "drop (species subset): multi-tree-height round-trip" begin
        # THE TRAP (has cost this project twice): `re_sd` for a phylo term is
        # defined against the RAW covariance `sigma_phy_dense(phy)` (diagonal =
        # tree height), NOT the normalised correlation. Invisible on a height-1
        # tree. Round-trip across several heights so a mistake could not hide.
        for (seed, branch_length) in ((20260930, 0.2), (20260931, 1.0), (20260932, 3.0))
            Random.seed!(seed)
            p = 20
            m = 4
            phy = random_balanced_tree(p; branch_length = branch_length)
            species = repeat(1:p, inner = m)
            n = length(species)
            x = randn(n)
            β = [0.2, 0.4]
            σ = 0.3
            σphy = 0.6
            C = sigma_phy_dense(phy; σ²_phy = σphy^2)
            u = cholesky(Symmetric(C)).L * randn(p)
            y = β[1] .+ β[2] .* x .+ u[species] .+ σ .* randn(n)
            dat_full = (; y, x, species)

            # Baseline: every leaf present — unaffected by the fix (regression guard).
            fit_full = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
                           Gaussian(); data = dat_full, tree = phy)
            @test fit_full.converged
            @test nobs(fit_full) == n

            # Drop every row for THREE MID-TREE species (not a tail-truncation —
            # proves the fix matches by identity, not by first-seen position).
            # This is exactly what the R bridge's default `response = "drop"`
            # produces: rows gone, the TREE still carries all `p` leaves.
            drop_species = Set([3, 7, 12])
            keep = [!(s in drop_species) for s in species]
            dat_sub = (; y = y[keep], x = x[keep], species = species[keep])
            @test length(unique(dat_sub.species)) == p - length(drop_species)  # 17 of 20
            @test phy.n_leaves == p                                            # tree untouched

            # Before the fix: this threw
            #   "algorithm = :sparse_lbfgs: the number of `species` levels (17)
            #    must equal the tree's leaf count (20)"
            fit_sub = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
                         Gaussian(); data = dat_sub, tree = phy)
            @test fit_sub.converged
            @test nobs(fit_sub) == n - m * length(drop_species)

            # `re_sd` is on phy's RAW covariance scale, so a height-swap bug that
            # substituted the correlation matrix would fail this at every height
            # except branch_length ≈ 1. The slope (independent of species) and the
            # residual SD are well-identified regardless of tree height; the
            # INTERCEPT is not (it is confounded with the mean of the phylo
            # random effect, which drifts further from 0 as the phylo variance
            # grows relative to G = 20 species) — checked instead against the
            # full-data fit on the SAME random draw, not the DGP truth.
            @test coef(fit_sub, :mu)[2] ≈ β[2] atol = 0.15
            @test exp(coef(fit_sub, :sigma)[1]) ≈ σ atol = 0.15
            @test re_sd(fit_sub)[:species] ≈ σphy atol = 0.35
            # Close to the full-data fit specifically (species dropped from the
            # LIKELIHOOD, kept in the phylo PRIOR — not a different model, so both
            # fits chase the same realised random-effect draw).
            @test coef(fit_sub, :mu) ≈ coef(fit_full, :mu) atol = 0.3
        end
    end

    @testset "drop (species subset): tree tip NAMES, not just integer indices" begin
        # The parity fixture (test/parity/phylo-mean/gaussian-phylo-mean) uses
        # string leaf names ("sp_1", …) rather than integer tip indices — confirm
        # the name-matching tier handles a subset the same way. `random_balanced_
        # tree` only names tips "L1".."Lp"; rebuild the same balanced topology
        # with `species_$i` names via the lower-level `DRModels.make_phy` edge builder
        # (same pairing algorithm `random_balanced_tree` uses internally).
        Random.seed!(20260933)
        p = 16
        m = 5
        leaf_names = ["sp_$(i)" for i in 1:p]
        edges = Tuple{Int,Int,Float64}[]
        current_level = collect(1:p)
        next_id = p + 1
        branch_length = 0.8
        while length(current_level) > 1
            new_level = Int[]
            i = 1
            while i + 1 <= length(current_level)
                parent = next_id; next_id += 1
                push!(edges, (parent, current_level[i], branch_length))
                push!(edges, (parent, current_level[i + 1], branch_length))
                push!(new_level, parent)
                i += 2
            end
            i == length(current_level) && push!(new_level, current_level[i])
            current_level = new_level
        end
        phy = DRModels.make_phy(edges, p; root_index = current_level[1], leaf_names = leaf_names)
        species = repeat(leaf_names, inner = m)
        n = length(species)
        x = randn(n)
        β = [0.1, 0.3]
        σ = 0.25
        σphy = 0.5
        C = sigma_phy_dense(phy; σ²_phy = σphy^2)
        u = cholesky(Symmetric(C)).L * randn(p)
        species_idx = repeat(1:p, inner = m)
        y = β[1] .+ β[2] .* x .+ u[species_idx] .+ σ .* randn(n)

        drop = Set(["sp_2", "sp_9"])
        keep = [!(s in drop) for s in species]
        dat_sub = (; y = y[keep], x = x[keep], species = species[keep])

        fit_sub = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
                      Gaussian(); data = dat_sub, tree = phy)
        @test fit_sub.converged
        @test nobs(fit_sub) == n - m * length(drop)
        @test coef(fit_sub, :mu) ≈ β atol = 0.2
    end

    @testset "drop: species label matching neither a leaf name nor a tip index fails informatively" begin
        Random.seed!(20260934)
        p = 10
        m = 3
        phy = random_balanced_tree(p; branch_length = 0.5)
        species = repeat(1:p, inner = m)
        species[1] = 999   # not a valid tip index (1:10) and not a leaf name
        n = length(species)
        x = randn(n)
        y = randn(n)
        dat = (; y, x, species)

        err = nothing
        try
            drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
                Gaussian(); data = dat, tree = phy)
        catch e
            err = e
        end
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin("tree tip names", msg) || occursin("integer tip indices", msg)
    end

    @testset "include (unfiltered missing response) IS the drop fit on the phylo-mean cell (D-179 #2)" begin
        # The decision experiment (2026-08-27): native drmTMB's own Gaussian
        # mean-phylo fits under `response = "include"` and `response = "drop"`
        # are BYTE-IDENTICAL — logLik and coefficients exactly equal, same rows
        # used. With rows conditionally independent given the latent field, a
        # missing Gaussian response integrates out of its own likelihood factor
        # entirely, so `include` is `drop` + prediction, not a new likelihood.
        # DRModels.jl therefore accepts masked responses on this cell by fitting the
        # observed rows against the FULL tree (the subset-tolerant #482 leaf
        # matching), instead of refusing. The two calls below must agree not
        # approximately but exactly: after the row filter they are the same
        # computation on the same bytes.
        Random.seed!(20260935)
        p = 12
        m = 4
        phy = random_balanced_tree(p; branch_length = 1.0)
        species = repeat(1:p, inner = m)
        n = length(species)
        x = randn(n)
        yv = 0.2 .+ 0.4 .* x .+ 0.3 .* randn(n)
        y = Vector{Union{Missing,Float64}}(yv)
        y[3] = missing
        y[(2 * m + 1):(3 * m)] .= missing   # species 3 entirely masked: prior-only leaf
        dat = (; y, x, species)

        fit_incl = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
                       Gaussian(); data = dat, tree = phy)

        keep = .!ismissing.(y)
        dat_dropped = (; y = Float64.(y[keep]), x = x[keep], species = species[keep])
        fit_drop = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
                       Gaussian(); data = dat_dropped, tree = phy)

        @test fit_incl.converged
        @test nobs(fit_incl) == count(keep)
        @test fit_incl.loglik == fit_drop.loglik
        @test fit_incl.theta == fit_drop.theta
    end

    @testset "the wrapper does not leak past the phylo-mean cell" begin
        # Only the exact phylo-MEAN cell is unwrapped. Routes whose row-to-level
        # matching is positional (the #482 trap) must still refuse a masked
        # response rather than silently fit against the wrong levels.
        Random.seed!(20260937)
        G = 8
        m = 5
        g = repeat(1:G, inner = m)
        n = length(g)
        x = randn(n)
        y = Vector{Union{Missing,Float64}}(randn(n))
        y[2] = missing
        K = Matrix{Float64}(I, G, G)

        # relmat mean structure: still refused.
        err = nothing
        try
            drm(bf(@formula(y ~ x + relmat(1 | g)), @formula(sigma ~ 1)),
                Gaussian(); data = (; y, x, g), K = K)
        catch e
            err = e
        end
        @test err isa ArgumentError
        @test occursin("ROUTE-level", sprint(showerror, err))

        # phylo mean but a NON-CONSTANT sigma design: falls to the dense
        # structured fitter, whose positional matching is not subset-safe —
        # still refused.
        Random.seed!(20260938)
        p = 10
        phy = random_balanced_tree(p; branch_length = 1.0)
        species = repeat(1:p, inner = 4)
        n2 = length(species)
        x2 = randn(n2)
        z2 = randn(n2)
        y2 = Vector{Union{Missing,Float64}}(randn(n2))
        y2[7] = missing
        err2 = nothing
        try
            drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ z)),
                Gaussian(); data = (; y = y2, x = x2, z = z2, species))
        catch e
            err2 = e
        end
        @test err2 isa ArgumentError
        # #527: pin WHICH gate fired — the missing-response ROUTE gate, not a
        # tree/argument error thrown before it. Without this, the assertion
        # would stay green if an unrelated earlier gate started intercepting
        # the case and the mask gate silently stopped being exercised.
        @test occursin("ROUTE-level", sprint(showerror, err2))
    end

    @testset "non-phylo Gaussian response masks are unaffected" begin
        # Regression guard: the fixed-effect location-scale missing-response path
        # (`_fit_fixed_gaussian_missing_response`, unrelated code) must still work
        # exactly as before — this PR touches only the phylo-MEAN sparse routes.
        Random.seed!(20260936)
        n = 100
        x = randn(n)
        y = Vector{Union{Missing,Float64}}(1.0 .+ 0.5 .* x .+ 0.4 .* randn(n))
        y[5] = missing; y[17] = missing
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
        @test fit.converged
        @test nobs(fit) == n - 2
    end
end
