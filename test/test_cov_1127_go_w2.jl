# Cursor coverage 27 Nov 2026 (w2 / g–o lane): top uncovered src files on main
# excluding the open a–f and p–z 1127cov PRs — `location_only.jl`, `inference.jl`,
# `gaussian_core.jl`. Targets user-facing contracts and cheap error branches.
using DRModels
using Test
using LinearAlgebra: BLAS
using StatsModels

@testset "cursor coverage 2026-11-27 (g–o w2)" begin
    @testset "inference.jl — threading helpers and confint refusal" begin
        DR = DRModels
        @test DR._worker_threads(false, 8) == 1
        @test DR._worker_threads(true, 0) == 1
        nthr = Threads.nthreads()
        @test DR._worker_threads(true, 4) == min(4, nthr)

        orig = BLAS.get_num_threads()
        try
            BLAS.set_num_threads(2)
            @test DR._blas_oversubscribed(false) == false
            if nthr > 1
                @test DR._blas_oversubscribed(true) == (Threads.nthreads() > 1 && BLAS.get_num_threads() > 1)
            end
        finally
            BLAS.set_num_threads(orig)
        end

        fit = drm(bf(@formula(y ~ 1)), Gaussian(); data = (y = randn(12),))
        @test_throws ArgumentError confint(fit; method = :bootstrap)
        msg = try
            confint(fit; method = :laplace)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin(":wald", msg) && occursin(":profile", msg)

        stat = (
            param = :mu,
            coef = "x",
            evaluations = 4,
            gradient_evaluations = 1,
            bracket_expansions = 0,
            root_iterations = 2,
            lower_unbounded = false,
            upper_unbounded = false,
            nonmonotone = false,
            lower_endpoint_failed = true,
            upper_endpoint_failed = false,
            lower_nuisance_method = :forward,
            upper_nuisance_method = :forward,
            lower_nuisance_fallback = false,
            upper_nuisance_fallback = false,
            lower_nuisance_reason = :bracket,
            upper_nuisance_reason = :ok,
        )
        result = (ci = Any[], stats = [stat], attempted = 1, used = 0, failed = 1)
        arms = DR._profile_failed_arm_descriptions(result)
        @test length(arms) == 1
        @test occursin("mu:x", arms[1])
        @test occursin("lower", arms[1])
        @test_throws ArgumentError DR._throw_profile_endpoint_failure(result)
    end

    @testset "gaussian_core.jl — DrmFormula constructors and dpar guards" begin
        mu = @formula(y ~ 1 + x)
        forms = Pair{Symbol,Any}[:mu => mu.rhs]
        f1 = DRModels.DrmFormula(:y, forms)
        @test f1.response == :y && f1.response2 === nothing
        @test f1.schema_cache[] == Dict{Symbol,Any}()
        @test isempty(f1.sd_targets)

        f2 = DRModels.DrmFormula(:y, forms, :fail)
        @test f2.response2 === :fail

        cache = Ref(Dict{Symbol,Any}(:mu => 1))
        f3 = DRModels.DrmFormula(:y, forms, nothing, cache)
        @test f3.schema_cache === cache

        sd = Dict{Symbol,NamedTuple}(:g => (name = :g,))
        f4 = DRModels.DrmFormula(:y, forms, nothing, sd)
        @test f4.sd_targets === sd

        seen = Set{Symbol}()
        @test DRModels._check_dpar_name!(seen, :sigma) === :sigma
        @test :sigma in seen
        err_tau = try
            DRModels._check_dpar_name!(Set{Symbol}(), :tau)
            nothing
        catch e
            e
        end
        @test err_tau isa ArgumentError && occursin("sigma", err_tau.msg) && occursin("tau", err_tau.msg)
        err_dup = try
            s = Set([:sigma])
            DRModels._check_dpar_name!(s, :sigma)
            nothing
        catch e
            e
        end
        @test err_dup isa ArgumentError && occursin("duplicate", err_dup.msg)
    end

    @testset "location_only.jl — LocOnlyRanef dictionary surface" begin
        phy = DRModels.random_caterpillar_tree(6; branch_length = 0.2)
        p = phy.n_leaves
        species = repeat(1:p, inner = 2)
        n = length(species)
        prob = DRModels.make_loc_problem(phy, zeros(n), ones(n, 1); species = species)
        blups = Dict(:species => fill(0.1, p))
        mu_fixed = 0.3 .* ones(n)
        ranef = DRModels.LocOnlyRanef(blups, mu_fixed, prob)
        @test length(ranef) == 1
        @test haskey(ranef, :species)
        @test ranef[:species] === blups[:species]
        @test get(ranef, :species, nothing) === blups[:species]
        @test get(ranef, :missing, 0.0) === 0.0
        keys = [k for (k, _) in ranef]
        @test keys == [:species]
        @test ranef.tipvar[] === nothing
        v1 = DRModels._loconly_row_tip_variance(ranef)
        @test length(v1) == n
        @test ranef.tipvar[] !== nothing
        v2 = DRModels._loconly_row_tip_variance(ranef)
        @test v1 === v2
    end
end