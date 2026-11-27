# Coverage lane w6 (27 Nov): global ranks 13–15 among eligible `src/` files on
# `main` @ be38f5d (merged sharded `Pkg.test(coverage=true)` baseline), excluding
# files already claimed by open `1127cov` PRs (#1584, #1587, #1599, #1609, #1621,
# #1622, #1623, #1628). Targets: `gaussian_sparse_lss.jl`, `sparse_aug_plsm.jl`,
# `control.jl`.
using DRModels
using Test
using Random
using LinearAlgebra
using SparseArrays
using StableRNGs

function _make_tiny_newick()
  "(t1:0.1,t2:0.1):0.1;"
end

function _argmsg(f)
    try
        f()
        return ""
    catch e
        e isa ArgumentError && return e.msg
        rethrow()
    end
end

@testset "cursor coverage 2026-11-27 (w6 ranks 13–15)" begin

    DR = DRModels

    @testset "control.jl — prior weights, keyword advice, and scope helpers" begin
        @test DR._check_prior_weights(nothing) === nothing
        @test DR._check_prior_weights([1.0, 2.0]) == [1.0, 2.0]
        @test_throws ArgumentError DR._check_prior_weights(Float64[])
        @test_throws ArgumentError DR._check_prior_weights([1.0, -0.1])
        @test_throws ArgumentError DR._check_prior_weights([0.0, 0.0])
        @test_throws ArgumentError DR._check_prior_weights([1.0, NaN])

        @test occursin("ML-only", DR._kw_advice(:method, LogNormal()))
        @test occursin("treatment contrasts", DR._kw_advice(:contrasts, Gaussian()))
        @test occursin("DRMControl(se = false)", DR._kw_advice(:se, Poisson()))
        @test DR._kw_r_note(:tree, Gaussian()) ==
            " Not verified against drmTMB for this model."

        n = 12
        g = repeat(1:4, inner = 3)
        d = (y = clamp.(rand(StableRNG(112713), n) .+ 0.2, 0.05, 0.95), x = randn(n), g = g)
        f_re = bf(@formula(y ~ 1 + x + (1 | g)), @formula(sigma ~ 1))
        msg_w = _argmsg(() -> drm(f_re, Beta(); data = d, weights = fill(1.0, n)))
        @test occursin("`weights`", msg_w)
        @test occursin("refused before it started", msg_w)

        msg_ctl = _argmsg(() -> drm(f_re, Beta(); data = d,
                                   control = DRMControl(maxit = 5)))
        @test occursin("maxit", msg_ctl) || occursin("control = DRMControl", msg_ctl)

        scope = DR._ControlScope(DRMControl(), [1.0, 1.0], f_re, false, false, false,
                                 Float64[], 0, NamedTuple[], :none)
        Base.task_local_storage(() -> begin
            @test_throws ArgumentError DR._ctl_weights(2; supported = false, route = "synthetic")
            @test DR._ctl_weights(2) == [1.0, 1.0]
            @test DR._ctl_merge_se(true) == true
        end, :DRModels_drm_control_scope, scope)
        @test DR._ctl_merge_se(true) == true

        f_fix = bf(@formula(y ~ 1 + x), @formula(sigma ~ 1))
        ypos = exp.(randn(StableRNG(112715), 20))
        msg_unk = _argmsg(() -> drm(f_fix, LogNormal(); data = (y = ypos, x = randn(20)), method = :REML))
        @test occursin("does not accept the keyword `method`", msg_unk)
        acc = last(split(msg_unk, "Keywords this route accepts: "))
        @test occursin("`control`", acc)
    end

    @testset "gaussian_sparse_lss.jl — multi-component assembly and phylo guard" begin
        phy = DR.augmented_phy(_make_tiny_newick())
        @test phy.n_leaves == 2
        gidx = [1, 1, 2, 2]
        G = 2
        Zg = ones(G, 1)
        comp = DR._sparse_lss_iid_comp(gidx, G, Zg)
        @test comp.G == G
        @test length(comp.inv_sd) == G
        @test all(comp.inv_sd .== 1)

        gidx_bad = [1, 1, 1]
        @test_throws ErrorException DR._sparse_lss_phylo_comp(gidx_bad, 3, Zg, phy)

        n = 4
        y = randn(StableRNG(112714), n)
        Xμ = ones(n, 1)
        Xσ = ones(n, 1)
        comps = [comp]
        θ = [0.0, -1.0, 0.1]
        parts = DR._lss_sparse_multi_assemble(θ, y, Xμ, Xσ, comps)
        @test parts.H isa SparseMatrixCSC
        @test length(parts.r) == n
        nll = DR._lss_sparse_multi_objective(θ, y, Xμ, Xσ, comps)
        @test isfinite(nll)
    end

    @testset "sparse_aug_plsm.jl — Newton cap, prior precision, pattern mismatch" begin
        step = [10.0, -3.0, 0.5]
        capped = DR.cap_newton_step(step, 1.0)
        @test maximum(abs, capped) <= 1.0 + 1e-15
        @test step == DR.cap_newton_step(step, 100.0)

        Q = sparse([1.0 0.0; 0.0 1.0])
        Λinv = [2.0 0.0; 0.0 3.0]
        P = DR.prior_precision(Q, Λinv)
        @test nnz(P) >= 4

        err = DR.CholPatternMismatch("unit test")
        @test sprint(showerror, err) == "CholPatternMismatch: unit test"

        H = sparse([2.0 0.0; 0.0 2.0])
        cache = DR.CholPatternCache()
        DR._chol_factorize(H, 1e-9, cache)
        @test DR._assert_chol_pattern_matches(H, cache) === nothing
        bad = SparseMatrixCSC(H.m, H.n, copy(H.colptr), copy(H.rowval), copy(H.nzval))
        push!(bad.rowval, 2)
        push!(bad.nzval, 0.01)
        bad.colptr[end] += 1
        @test_throws DR.CholPatternMismatch DR._assert_chol_pattern_matches(bad, cache)
    end
end