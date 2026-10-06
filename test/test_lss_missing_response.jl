# test_lss_missing_response.jl — issue #559.
#
# Location-scale-scale (LSS) missing responses:
#   - `sd(group) ~ …` (plain iid LSS)
#   - `sd(group, phylogenetic) ~ …` (phylogenetic LSS / sd_phylo)
#   - multi-component LSS (e.g. iid + phylo, multiple iid)
#
# Observed-rows pattern (D-179 #2, issue #517 precedent):
#   - Build group index, G, and Zg from the FULL data so the scale-scale linear
#     predictor D_a is parameterised across all G levels.
#   - Subset y, Xμ, Xσ, and gidx to observed rows.
#   - A group/species whose every row is missing simply stays in the prior with
#     no likelihood term.
#   - include vs hand-dropped data are byte-identical on logLik and theta when
#     every level has at least one observation.

using DRModels
using Test, Random, LinearAlgebra, Statistics, StableRNGs

@testset "#559 Location-scale-scale missing response" begin

    @testset "plain lss sd(group): exact equality vs dropped data (ML + REML)" begin
        rng = StableRNG(20260940)
        n_id = 12
        n_each = 5
        sex = repeat([0.0, 1.0], inner = n_id ÷ 2)
        b = randn(rng, n_id) .* [0.65, 0.40][Int.(sex) .+ 1]
        id = repeat(1:n_id, inner = n_each)
        sexl = sex[id]
        n = n_id * n_each
        x = randn(rng, n)
        yv = [0.35, 0.70][Int.(sexl) .+ 1] .+ 0.2 .* x .+ b[id] .+
            randn(rng, n) .* [0.35, 0.60][Int.(sexl) .+ 1]

        y = Vector{Union{Missing,Float64}}(copy(yv))
        # Mask some rows across groups, keeping >= 1 row per group
        y[2] = missing
        y[7] = missing
        y[14] = missing
        y[28] = missing
        y[45] = missing
        dat_incl = (; y, x, sex = sexl, id)

        keep = .!ismissing.(y)
        dat_drop = (; y = Float64.(y[keep]), x = x[keep], sex = sexl[keep], id = id[keep])

        # ML fit comparison
        f = bf(@formula(y ~ x + (1 | id)), @formula(sigma ~ sex), @formula(sd(id) ~ sex))
        fit_incl = drm(f, Gaussian(); data = dat_incl, method = :ML)
        fit_drop = drm(f, Gaussian(); data = dat_drop, method = :ML)

        @test fit_incl.converged
        @test fit_drop.converged
        @test nobs(fit_incl) == count(keep)
        @test fit_incl.loglik == fit_drop.loglik
        @test fit_incl.theta == fit_drop.theta

        # REML fit comparison
        fit_incl_reml = drm(f, Gaussian(); data = dat_incl, method = :REML)
        fit_drop_reml = drm(f, Gaussian(); data = dat_drop, method = :REML)

        @test fit_incl_reml.converged
        @test fit_drop_reml.converged
        @test nobs(fit_incl_reml) == count(keep)
        @test fit_incl_reml.loglik == fit_drop_reml.loglik
        @test fit_incl_reml.theta == fit_drop_reml.theta
        @test reml_loglik(fit_incl_reml) == reml_loglik(fit_drop_reml)
    end

    @testset "sd_phylo: exact equality vs dropped data (ML + REML)" begin
        rng = StableRNG(20260941)
        p = 12
        m = 4
        phy = random_balanced_tree(p; branch_length = 0.8)
        species = repeat(1:p, inner = m)
        n = length(species)
        x = randn(rng, n)
        z = repeat(randn(rng, p), inner = m)  # species-level predictor
        Kmat = DRModels._phylo_correlation(phy)
        α_true = [0.2, 0.5]
        Zg_full = [ones(p) z[1:m:end]]
        σa = exp.(Zg_full * α_true)
        Σa = (σa * σa') .* Kmat
        u = cholesky(Symmetric(Σa)).L * randn(rng, p)
        yv = 0.5 .+ 0.3 .* x .+ u[species] .+ 0.25 .* randn(rng, n)

        y = Vector{Union{Missing,Float64}}(copy(yv))
        # Mask some rows across species (all species retain >= 1 row)
        y[1] = missing
        y[6] = missing
        y[15] = missing
        y[22] = missing
        y[35] = missing
        dat_incl = (; y, x, z, species)

        keep = .!ismissing.(y)
        dat_drop = (; y = Float64.(y[keep]), x = x[keep], z = z[keep], species = species[keep])

        f = bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ x), @formula(sd(species, phylogenetic) ~ z))

        # ML
        fit_incl = drm(f, Gaussian(); data = dat_incl, tree = phy, method = :ML)
        fit_drop = drm(f, Gaussian(); data = dat_drop, tree = phy, method = :ML)

        @test fit_incl.converged
        @test fit_drop.converged
        @test nobs(fit_incl) == count(keep)
        @test fit_incl.loglik == fit_drop.loglik
        @test fit_incl.theta == fit_drop.theta

        # REML
        fit_incl_reml = drm(f, Gaussian(); data = dat_incl, tree = phy, method = :REML)
        fit_drop_reml = drm(f, Gaussian(); data = dat_drop, tree = phy, method = :REML)

        @test fit_incl_reml.converged
        @test fit_drop_reml.converged
        @test nobs(fit_incl_reml) == count(keep)
        @test fit_incl_reml.loglik == fit_drop_reml.loglik
        @test fit_incl_reml.theta == fit_drop_reml.theta
        @test reml_loglik(fit_incl_reml) == reml_loglik(fit_drop_reml)
    end

    @testset "multi-component lss: exact equality vs dropped data (ML + REML)" begin
        rng = StableRNG(20260942)
        n_sp = 8
        n_st = 4
        m = 3
        species = repeat(1:n_sp, inner = n_st * m)
        study = repeat(repeat(1:n_st, inner = m), n_sp)
        n = length(species)
        x = randn(rng, n)
        yv = 0.4 .+ 0.2 .* x .+ 0.3 .* randn(rng, n)

        y = Vector{Union{Missing,Float64}}(copy(yv))
        y[3] = missing
        y[11] = missing
        y[25] = missing
        y[40] = missing
        dat_incl = (; y, x, species, study)

        keep = .!ismissing.(y)
        dat_drop = (; y = Float64.(y[keep]), x = x[keep], species = species[keep], study = study[keep])

        f = bf(@formula(y ~ x + (1 | species) + (1 | study)),
               @formula(sigma ~ x),
               @formula(sd(species) ~ 1),
               @formula(sd(study) ~ 1))

        # ML
        fit_incl = drm(f, Gaussian(); data = dat_incl, method = :ML)
        fit_drop = drm(f, Gaussian(); data = dat_drop, method = :ML)

        @test fit_incl.converged
        @test fit_drop.converged
        @test nobs(fit_incl) == count(keep)
        @test fit_incl.loglik == fit_drop.loglik
        @test fit_incl.theta == fit_drop.theta

        # REML
        fit_incl_reml = drm(f, Gaussian(); data = dat_incl, method = :REML)
        fit_drop_reml = drm(f, Gaussian(); data = dat_drop, method = :REML)

        @test fit_incl_reml.converged
        @test fit_drop_reml.converged
        @test nobs(fit_incl_reml) == count(keep)
        @test fit_incl_reml.loglik == fit_drop_reml.loglik
        @test fit_incl_reml.theta == fit_drop_reml.theta
        @test reml_loglik(fit_incl_reml) == reml_loglik(fit_drop_reml)
    end

    @testset "sd_phylo: entirely masked species (prior-only leaf)" begin
        rng = StableRNG(20260943)
        p = 10
        m = 4
        phy = random_balanced_tree(p; branch_length = 1.0)
        species = repeat(1:p, inner = m)
        n = length(species)
        x = randn(rng, n)
        z = repeat(randn(rng, p), inner = m)
        yv = 0.2 .+ 0.4 .* x .+ 0.3 .* randn(rng, n)

        y = Vector{Union{Missing,Float64}}(copy(yv))
        # Mask species 3 ENTIRELY
        y[(2 * m + 1):(3 * m)] .= missing
        y[1] = missing
        dat = (; y, x, z, species)

        f = bf(@formula(y ~ x + phylo(1 | species)),
               @formula(sigma ~ x),
               @formula(sd(species, phylogenetic) ~ z))

        fit = drm(f, Gaussian(); data = dat, tree = phy, method = :ML)
        @test fit.converged
        @test nobs(fit) == n - m - 1
        @test isfinite(loglik(fit))
        @test isfinite(fit.vcov[1, 1])

        fit_reml = drm(f, Gaussian(); data = dat, tree = phy, method = :REML)
        @test fit_reml.converged
        @test nobs(fit_reml) == n - m - 1
        @test isfinite(reml_loglik(fit_reml))
    end

    @testset "plain lss: entirely masked group (prior-only level)" begin
        rng = StableRNG(20260944)
        n_id = 10
        n_each = 5
        sex = repeat([0.0, 1.0], inner = n_id ÷ 2)
        id = repeat(1:n_id, inner = n_each)
        sexl = sex[id]
        n = n_id * n_each
        x = randn(rng, n)
        yv = 0.35 .+ 0.2 .* x .+ 0.35 .* randn(rng, n)

        y = Vector{Union{Missing,Float64}}(copy(yv))
        # Mask group 4 ENTIRELY
        y[(3 * n_each + 1):(4 * n_each)] .= missing
        y[2] = missing
        dat = (; y, x, sex = sexl, id)

        f = bf(@formula(y ~ x + (1 | id)), @formula(sigma ~ sex), @formula(sd(id) ~ sex))

        fit = drm(f, Gaussian(); data = dat, method = :ML)
        @test fit.converged
        @test nobs(fit) == n - n_each - 1
        @test isfinite(loglik(fit))

        fit_reml = drm(f, Gaussian(); data = dat, method = :REML)
        @test fit_reml.converged
        @test nobs(fit_reml) == n - n_each - 1
        @test isfinite(reml_loglik(fit_reml))
    end

    @testset "lss dof over-parameterisation guards" begin
        # Plain lss: pμ=2, pσ=2, psd=2 -> total_dof = 6. Supply 3 observed rows -> throws ArgumentError
        x = randn(20)
        sex = repeat([0.0, 1.0], inner = 10)
        id = repeat(1:10, inner = 2)
        y = Vector{Union{Missing,Float64}}(fill(missing, 20))
        y[1:3] .= [0.1, 0.2, 0.3]  # 3 < 6
        dat = (; y, x, sex, id)
        f_plain = bf(@formula(y ~ x + (1 | id)), @formula(sigma ~ sex), @formula(sd(id) ~ sex))
        @test_throws ArgumentError drm(f_plain, Gaussian(); data = dat)

        # sd_phylo: pμ=2, pσ=2, psd=2 -> total_dof = 6. Supply 3 observed rows -> throws ArgumentError
        phy = random_balanced_tree(10; branch_length = 0.5)
        species = id
        z = sex
        f_phylo = bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ sex), @formula(sd(species, phylogenetic) ~ z))
        @test_throws ArgumentError drm(f_phylo, Gaussian(); data = (; y, x, sex, z, species), tree = phy)

        # Multi-component: pμ=2, pσ=1, psd1=1, psd2=1 -> total_dof = 5. Supply 3 rows -> throws ArgumentError
        study = id
        f_multi = bf(@formula(y ~ x + (1 | species) + (1 | study)), @formula(sd(species) ~ 1), @formula(sd(study) ~ 1))
        @test_throws ArgumentError drm(f_multi, Gaussian(); data = (; y, x, species, study))
    end

    @testset "unsupported non-lss missing-response combinations still throw ArgumentError (leak guards)" begin
        rng = StableRNG(20260945)
        G = 8
        m = 5
        g = repeat(1:G, inner = m)
        n = length(g)
        x = randn(rng, n)
        y = Vector{Union{Missing,Float64}}(randn(rng, n))
        y[2] = missing
        K = Matrix{Float64}(I, G, G)

        # relmat mean structure: still refused
        err_relmat = nothing
        try
            drm(bf(@formula(y ~ x + relmat(1 | g)), @formula(sigma ~ 1)),
                Gaussian(); data = (; y, x, g), K = K)
        catch e
            err_relmat = e
        end
        @test err_relmat isa ArgumentError
        @test occursin("ROUTE-level", sprint(showerror, err_relmat))

        # phylo mean with non-constant sigma design: still refused
        phy = random_balanced_tree(G; branch_length = 1.0)
        z = randn(rng, n)
        err_phylo_hetero = nothing
        try
            drm(bf(@formula(y ~ x + phylo(1 | g)), @formula(sigma ~ z)),
                Gaussian(); data = (; y, x, z, g), tree = phy)
        catch e
            err_phylo_hetero = e
        end
        @test err_phylo_hetero isa ArgumentError
        @test occursin("ROUTE-level", sprint(showerror, err_phylo_hetero))

        # Plain (1 | g) WITHOUT sd(): still refused (positional matching not subset-safe)
        err_ranef = nothing
        try
            drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)),
                Gaussian(); data = (; y, x, g))
        catch e
            err_ranef = e
        end
        @test err_ranef isa ArgumentError
        @test occursin("ROUTE-level", sprint(showerror, err_ranef))

        # meta_V: still refused
        err_metav = nothing
        try
            drm(bf(@formula(y ~ x + meta_V(V)), @formula(sigma ~ 1)),
                Gaussian(); data = (; y, x, V = ones(n)))
        catch e
            err_metav = e
        end
        @test err_metav isa ArgumentError
        @test occursin("ROUTE-level", sprint(showerror, err_metav))
    end
end
