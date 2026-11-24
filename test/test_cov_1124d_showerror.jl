# test_cov_1124d_showerror.jl — lane 1124d: pinned showerror / validation messages
# on small helpers and accessors (no engine or parity routes).
using DRModels
using Test, Random

_msg(f) = try
    f()
    ""
catch e
    sprint(showerror, e)
end

@testset "cov 1124d: exception messages and display" begin
    @testset "CholPatternMismatch showerror" begin
        e = DRModels.CholPatternMismatch("pattern rowval mismatch at node 3")
        msg = sprint(showerror, e)
        @test occursin("CholPatternMismatch", msg)
        @test occursin("pattern rowval mismatch", msg)
    end

    @testset "PhyloCorPenaltyNeedsTwoSD showerror" begin
        e = PhyloCorPenaltyNeedsTwoSD(
            "drm_phylo_penalty(cor_sd = …) needs two phylogenetic SD parameters.")
        msg = sprint(showerror, e)
        @test occursin("cor_sd", msg)
        @test occursin("two phylogenetic SD", msg)
    end

    @testset "rho12(fit) on univariate fit" begin
        Random.seed!(1124)
        n = 40
        x = randn(n)
        y = 0.4 .+ 0.6 .* x .+ randn(n)
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
        err = _msg(() -> rho12(fit))
        @test occursin("bivariate", err)
        @test occursin("rho12", err)
    end

    @testset "DRMControl constructor validation messages" begin
        for (ctor, frag) in (
            (() -> DRMControl(multi_start = 0), "multi_start"),
            (() -> DRMControl(jitter = -1.0), "jitter"),
            (() -> DRMControl(optimizer = :nelder_mead), "optimizer"),
            (() -> DRMControl(maxit = 2.5), "maxit"),
            (() -> DRMControl(start = (mu = [1.0],)), "start"),
        )
            msg = _msg(ctor)
            @test occursin("DRMControl", msg)
            @test occursin(frag, msg)
        end
        c = DRMControl(maxit = 99, multi_start = 2, se = false, optimizer = :lbfgs)
        out = sprint(show, c)
        @test occursin("DRMControl", out)
        @test occursin("multi_start = 2", out)
        @test occursin("optimizer = :lbfgs", out)
    end

    @testset "poly() outside a formula" begin
        e = try
            poly(1.0, 2)
            nothing
        catch ex
            ex
        end
        @test e isa ArgumentError
        @test occursin("formula term", sprint(showerror, e))
    end

    @testset "PolyTerm show" begin
        t = DRModels.PolyTerm(:x1, 2, [0.0], [1.0, 40.0, 1.0])
        @test sprint(show, t) == "poly(x1, 2)"
    end
end