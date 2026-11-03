# #949: every user-facing "converged" (show, check_drm, bridge_diagnostics, and both
# bridge payload keys) must come from `is_converged`, not the raw `fit.converged`.
# Pins what src/summary.jl (show), src/inference.jl (check_drm) and
# src/introspection.jl (bridge_diagnostics) already do, on a fit where the two
# flags differ: the aliased-design fit of test_1110_aliased_design.jl, whose raw
# optimiser flag is true but whose `is_converged` is false.
# NOT covered: `check_drm`'s `ok` with `grad_source = :none` (documented in the
# check_drm docstring; issue #949 stays open for it).
using DRModels
using Test, Random, Logging

@testset "#949 converged surfaces agree with is_converged" begin
    Random.seed!(1110)
    n = 200
    x = randn(n); z = 2 .* x .+ 1
    y = 1.0 .+ 0.5 .* x .+ exp.(-0.3 .+ 0.2 .* x) .* randn(n)
    fit = with_logger(NullLogger()) do
        drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + x + z)), Gaussian();
            data = (; y, x, z))
    end
    # the case where the two flags differ
    @test fit.converged
    @test !is_converged(fit)

    @testset "show prints is_converged" begin
        txt = sprint(show, MIME"text/plain"(), fit)
        @test occursin("converged = false", txt)
        @test !occursin("converged = true", txt)
    end

    @testset "check_drm and bridge_diagnostics" begin
        @test check_drm(fit).converged == is_converged(fit)
        @test check_drm(fit).converged == false
        @test bridge_diagnostics(fit).converged == is_converged(fit)
        @test bridge_diagnostics(fit).converged == false
    end

    @testset "the two bridge payload keys agree" begin
        out = DRModels._bridge_flatten(fit; family = "gaussian")
        @test out["converged"] == is_converged(fit)
        @test out["diagnostics"]["converged"] == is_converged(fit)
        @test out["converged"] == out["diagnostics"]["converged"]
        @test out["converged"] == false
    end
end
