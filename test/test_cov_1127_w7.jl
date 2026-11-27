# Coverage lane w7 (27 Nov): global ranks 16–18 on main @ be38f5d — `gaussian_structured.jl`,
# `binomial_sd_regression.jl`, `gaussian_sparse_lss.jl` (rank 18 `gaussian_bivariate.jl` claimed by #1622;
# `gaussian_sparse_lss.jl` is rank 19, the next unclaimed file in the ladder).
using DRModels
using Test
using LinearAlgebra
using Random
using StableRNGs
import StatsModels

@testset "cov 1127 w7" begin

    @testset "gaussian_structured.jl — matrix helpers and structured-plus-RE refusals" begin
        DR = DRModels
        phy = random_balanced_tree(8; branch_length = 0.2)
        C = DR._phylo_correlation(phy)
        @test size(C) == (phy.n_leaves, phy.n_leaves)
        @test all(isfinite, C)
        @test maximum(abs, C .- C') < 1e-10

        gidx = repeat(1:4, inner = 2)
        Z = DR._structured_Z(gidx, 4)
        @test size(Z) == (8, 4)
        @test Z[1, 1] == 1.0 && Z[2, 1] == 1.0 && Z[3, 2] == 1.0

        kw = (; K = nothing, A = nothing, tree = nothing, coords = nothing)
        @test_throws ErrorException DR._resolve_structured_matrix(:relmat, :g, 4; kw...)
        @test_throws ErrorException DR._resolve_structured_matrix(:animal, :g, 4; kw...)
        @test_throws ErrorException DR._resolve_structured_matrix(:phylo, :g, 4; kw...)
        @test_throws ErrorException DR._resolve_structured_matrix(:spatial, :site, 4; kw...)
        Ktiny = [1.0 0.5; 0.5 1.0]
        @test_throws ErrorException DR._resolve_structured_matrix(:relmat, :g, 4; K = Ktiny, A = nothing, tree = nothing, coords = nothing)

        @test_throws ErrorException DR._structured_group_index(:phylo, :sp, ["a", "b"], nothing)
        idx, G = DR._structured_group_index(:phylo, :sp, phy.leaf_names, phy)
        @test G == phy.n_leaves
        @test length(idx) == 8

        Csym = [1.0 0.3; 0.3 1.0]
        F = DR._psd_factor(Csym)
        @test F * F' ≈ Csym atol = 1e-10

        n, pμ, pσ = 12, 2, 1
        Xμ, Xσ = ones(n, pμ), ones(n, pσ)
        y = randn(n)
        data = (; g = repeat(1:4, inner = 3))
        structured = (:phylo, :g)
        re = [(StatsModels.ConstantTerm(1), :g)]
        nmμ = ["(Intercept)", "x"]

        base_kw = (; K = nothing, A = nothing, tree = phy, algorithm = :auto, penalty = nothing, g_tol = 1e-6)
        err_meta = try
            DR._drm_gaussian_structured_plus_ranef(
                Gaussian(), structured, re, 1, y, Xμ, Xσ, nmμ, ["sigma"], data; base_kw...)
        catch e
            e
        end
        @test err_meta isa ArgumentError
        @test occursin("meta_V", sprint(showerror, err_meta))

        err_pen = try
            DR._drm_gaussian_structured_plus_ranef(
                Gaussian(), structured, re, nothing, y, Xμ, Xσ, nmμ, ["sigma"], data;
                K = nothing, A = nothing, tree = phy, algorithm = :auto, penalty = ones(4), g_tol = 1e-6)
        catch e
            e
        end
        @test err_pen isa ArgumentError
        @test occursin("penalty", sprint(showerror, err_pen))

        err_alg = try
            DR._drm_gaussian_structured_plus_ranef(
                Gaussian(), structured, re, nothing, y, Xμ, Xσ, nmμ, ["sigma"], data;
                K = nothing, A = nothing, tree = phy, algorithm = :em, penalty = nothing, g_tol = 1e-6)
        catch e
            e
        end
        @test err_alg isa ArgumentError
        @test occursin("algorithm", sprint(showerror, err_alg))

        err_sp = try
            DR._drm_gaussian_structured_plus_ranef(
                Gaussian(), (:spatial, :site), re, nothing, y, Xμ, Xσ, nmμ, ["sigma"], data;
                K = nothing, A = nothing, tree = nothing, algorithm = :auto, penalty = nothing, g_tol = 1e-6)
        catch e
            e
        end
        @test err_sp isa ArgumentError
        @test occursin("spatial", sprint(showerror, err_sp))
    end

    @testset "binomial_sd_regression.jl — interaction parsing, partial correlations, link guards" begin
        DR = DRModels
        fmu = @formula(cbind(successes, failures) ~ x + phylo_interaction(1 | host & pollinator))
        ok_rhs = fmu.rhs isa Tuple ? fmu.rhs : (fmu.rhs,)
        parts = DR._binomial_phylo_interaction_parts(ok_rhs)
        @test parts.group1 === :host && parts.group2 === :pollinator

        dup_rhs = (ok_rhs..., ok_rhs[2])
        @test_throws ArgumentError DR._binomial_phylo_interaction_parts(dup_rhs)

        fbad = @formula(cbind(successes, failures) ~ phylo_interaction(1))
        bad_pipe = fbad.rhs isa Tuple ? fbad.rhs : (fbad.rhs,)
        @test_throws ArgumentError DR._binomial_phylo_interaction_parts(bad_pipe)

        fsame = @formula(cbind(successes, failures) ~ phylo_interaction(1 | host & host))
        bad_partners = fsame.rhs isa Tuple ? fsame.rhs : (fsame.rhs,)
        @test_throws ArgumentError DR._binomial_phylo_interaction_parts(bad_partners)

        @test_throws ArgumentError DR._binomial_partial_correlation_cholesky([0.1], 1)
        @test_throws ArgumentError DR._binomial_partial_correlation_cholesky([0.1], 3)

        η = [0.2, -0.5, 0.1]
        L = DR._binomial_partial_correlation_cholesky(η, 3)
        R = L * L'
        @test R ≈ R' atol = 1e-12
        cors = DR._binomial_pairwise_correlations(η, 3)
        @test length(cors) == 3
        names = DR._binomial_partial_correlation_names(:grp, 3)
        @test length(names) == 3

        for e in (0.0, 8.0, -12.0)
            s = DR._binomial_stable_sech(e)
            ref = 2 * exp(-abs(e)) / (1 + exp(-2 * abs(e)))
            @test s ≈ ref rtol = 1e-10
        end

        aux = (; s = 10.0, ntr = 10.0, logchoose = zeros(5), link = :logit)
        @test_throws ArgumentError DR._binomial_link_v123(aux, 1, 0.0)

        deriv = DR._binomial_sd_scale_derivative([0.0, 11.0, -11.0])
        @test all(x -> x > 0, deriv)
    end

    @testset "gaussian_sparse_lss.jl — component builders and sparse multi objective" begin
        DR = DRModels
        phy = random_balanced_tree(6; branch_length = 0.15)
        gidx = repeat(1:phy.n_leaves, inner = 2)
        Zg = ones(phy.n_leaves, 1)
        comp_phy = DR._sparse_lss_phylo_comp(gidx, phy.n_leaves, Zg, phy)
        @test comp_phy.G == phy.n_leaves

        @test_throws ErrorException DR._sparse_lss_phylo_comp(gidx, phy.n_leaves - 1, Zg, phy)

        gidx_iid = repeat(1:3, inner = 4)
        comp_iid = DR._sparse_lss_iid_comp(gidx_iid, 3, ones(3, 1))
        @test comp_iid.q == 3

        rng = StableRNG(1127)
        n = length(gidx_iid)
        y = randn(rng, n)
        Xμ = hcat(ones(n), randn(rng, n))
        Xσ = ones(n, 1)
        comps = [comp_iid]
        θ = zeros(2 + 1 + 1)
        asm = DR._lss_sparse_multi_assemble(θ, y, Xμ, Xσ, comps)
        @test asm.H isa SparseMatrixCSC
        @test length(asm.w) == n

        nll = DR._lss_sparse_multi_objective(θ, y, Xμ, Xσ, comps)
        @test isfinite(nll)

        fit_stub = (theta = copy(θ),)
        @test DR._lss_multi_route(fit_stub) === :unknown
        DR._mark_lss_multi_route!(fit_stub, :sparse_multi)
        @test DR._lss_multi_route(fit_stub) === :sparse_multi
    end
end