# Coverage lane w4 (2026-11-27): global ranks 7–9 among src files not claimed by
# open 1127cov PRs — `variational.jl`, `binomial_sd_regression.jl`,
# `gaussian_structured.jl` (per merged `Pkg.test(coverage=true)` + Coverage.jl on
# `main` @ be38f5d, excluding af / pz / comparison / w2 / w3 / w5 targets).
using DRModels
using Test
using Random
using LinearAlgebra
using StatsModels
import Distributions

@testset "cursor coverage 2026-11-27 (w4 ranks 7–9)" begin

    @testset "variational.jl — marginal selector and route refusals" begin
        DR = DRModels
        @test DR._marginal_method(:AGHQ) == DR.AGHQ()
        msg_laplace = try
            DR._marginal_method(:LAPLACE)
        catch e
            sprint(showerror, e)
        end
        @test occursin("not available on this route", msg_laplace)
        @test occursin(":Laplace", msg_laplace)

        @test_throws ArgumentError DR._aghq_reject(Gaussian(), "phylo(1 | g)")
        msg_aghq = try
            DR._aghq_reject(Gaussian(), "phylo(1 | g)")
        catch e
            sprint(showerror, e)
        end
        @test occursin("marginal = :AGHQ", msg_aghq)
        @test occursin("Poisson", msg_aghq)

        @test_throws ArgumentError DR._va_reject(Beta(), "crossed (1 | g) + (1 | h)")
        msg_va = try
            DR._va_reject(Beta(), "crossed (1 | g) + (1 | h)")
        catch e
            sprint(showerror, e)
        end
        @test occursin("marginal = :VA", msg_va)

        @test DR._reject_method_as_marginal(Poisson(), nothing) === nothing
        @test DR._reject_method_as_marginal(Poisson(), :ML) === :ML
        @test_throws ArgumentError DR._reject_method_as_marginal(Poisson(), :VA)
        @test_throws ArgumentError DR._reject_method_as_marginal(Poisson(), :REML)
        @test DR._reject_method_as_marginal(Poisson(), :REML; allow_reml = true) === :REML

        @test_throws ArgumentError DR._reject_reml_route(Poisson(), "Gamma families with zi")
        msg_reml = try
            DR._reject_reml_route(Poisson(), "Gamma families with zi")
        catch e
            sprint(showerror, e)
        end
        @test occursin("method = :REML", msg_reml)
        @test occursin("Poisson", msg_reml)
    end

    @testset "gaussian_structured.jl — relmat guards and psd factor" begin
        DR = DRModels
        Cbad = [1.0 0.3; 0.31 1.0]
        @test_throws ArgumentError DR._checked_relmat_symmetric(Cbad, :id)
        msg_sym = try
            DR._checked_relmat_symmetric(Cbad, :id)
        catch e
            sprint(showerror, e)
        end
        @test occursin("not symmetric", msg_sym)
        @test occursin("id", msg_sym)

        Cind = [1.0 0.0; 0.0 -1.0]
        @test_throws ArgumentError DR._checked_relmat_chol(Cind, :grp)
        msg_pd = try
            DR._checked_relmat_chol(Cind, :grp)
        catch e
            sprint(showerror, e)
        end
        @test occursin("not positive definite", msg_pd)

        K = [1.0 0.4; 0.4 1.0]
        F = DR._psd_factor(K)
        @test size(F) == (2, 2)
        @test F * F' ≈ K atol = 1e-10

        phy = DR.random_balanced_tree(4; branch_length = 0.25)
        R = DR._phylo_correlation(phy)
        @test size(R) == (4, 4)
        @test R ≈ R' atol = 1e-12
        @test all(abs.(diag(R) .- 1.0) .< 1e-10)
    end

    @testset "binomial_sd_regression.jl — phylo interaction parsing and SD algebra" begin
        DR = DRModels
        @test DR._binomial_stable_sech(0.0) ≈ 1.0 atol = 1e-12
        @test DR._binomial_stable_sech(40.0) > 0

        eta = [0.1, -0.2, 0.35]
        L = DR._binomial_partial_correlation_cholesky(eta, 3)
        R = L * L'
        cors = DR._binomial_pairwise_correlations(eta, 3)
        @test length(cors) == 3
        @test cors[1] ≈ R[1, 2] atol = 1e-12

        @test_throws ArgumentError DR._binomial_partial_correlation_cholesky([0.1], 1)
        @test_throws ArgumentError DR._binomial_partial_correlation_cholesky([0.1, 0.2], 3)

        rhs_dup = (@formula(cbind(successes, failures) ~ phylo_interaction(1 | host & pollinator) +
            phylo_interaction(1 | host2 & pollinator2))).rhs
        @test_throws ArgumentError DR._binomial_phylo_interaction_parts(rhs_dup)

        rhs_bad = (@formula(cbind(successes, failures) ~ phylo_interaction(1 | host & host))).rhs
        @test_throws ArgumentError DR._binomial_phylo_interaction_parts(rhs_bad)

        rhs_ok = (@formula(cbind(successes, failures) ~ x + phylo_interaction(1 | host & pollinator))).rhs
        parts = DR._binomial_phylo_interaction_parts(rhs_ok)
        @test parts !== nothing
        @test parts.group1 == :host
        @test parts.group2 == :pollinator
        @test parts.slope === nothing

        z = DR._binomial_sd_scale([0.0, 2.0, -3.0])
        @test all(z .> 0)
        @test z[1] ≈ 1.0 atol = 1e-12
    end
end