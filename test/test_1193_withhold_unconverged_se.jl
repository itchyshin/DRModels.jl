# #1193 (split out of #1127, Part B; Shinichi's decision of 25 Oct 2026): `show`
# and `coeftable` must not print Wald standard errors as usable when the fit is
# not converged (`is_converged` false) or its Hessian is not positive definite.
# drmTMB refuses `vcov()` there. The printed Std.Error column (and so z, p and
# the Wald interval) is withheld as NaN, and `show` says why. `stderror()` and
# `vcov()` keep the Hessian-based values: the decision is about printed SEs.
using DRModels
using Test, Random, Logging

@testset "#1193 printed Wald SEs withheld on an unconverged or non-PD fit" begin
    @testset "converged fit: SEs printed as before" begin
        Random.seed!(7)
        x = randn(100); y = 0.3 .+ 0.5 .* x .+ exp.(0.2 .* x) .* randn(100)
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + x)), Gaussian(); data = (; y, x))
        @test is_converged(fit)
        ct = coeftable(fit)
        @test all(isfinite, ct.cols[2])
        @test ct.cols[2] ≈ stderror(fit)
        txt = sprint(show, MIME"text/plain"(), fit)
        @test !occursin("Wald SEs withheld", txt)
        @test !occursin("NaN", txt)
    end

    @testset "optimiser stopped early (PD Hessian): SEs withheld" begin
        Random.seed!(7)
        x = randn(100); y = 0.3 .+ 0.5 .* x .+ exp.(0.2 .* x) .* randn(100)
        fit = with_logger(NullLogger()) do
            drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + x)), Gaussian(); data = (; y, x),
                control = DRMControl(maxit = 2))
        end
        @test !fit.converged
        @test !is_converged(fit)
        @test DRModels._twin_vcov_positive_definite(fit.vcov)   # finite SEs on main
        ct = coeftable(fit)
        @test all(isnan, ct.cols[2])                # Std.Error
        @test all(isnan, ct.cols[3])                # z
        @test all(isnan, ct.cols[4])                # p
        @test all(isnan, ct.cols[5]) && all(isnan, ct.cols[6])   # Wald interval
        @test ct.cols[1] == fit.theta               # estimates still shown
        @test all(isfinite, stderror(fit))          # accessor unchanged
        txt = sprint(show, MIME"text/plain"(), fit)
        @test occursin("Wald SEs withheld", txt)
        @test occursin("not converged", txt)
    end

    @testset "non-positive-definite Hessian: SEs withheld" begin
        # The aliased-design fit of test_1110 / test_949: the optimiser flag is
        # true, but `z = 2x + 1` aliases a sigma column, so the Hessian is not PD.
        Random.seed!(1110)
        n = 200
        x = randn(n); z = 2 .* x .+ 1
        y = 1.0 .+ 0.5 .* x .+ exp.(-0.3 .+ 0.2 .* x) .* randn(n)
        fit = with_logger(NullLogger()) do
            drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + x + z)), Gaussian(); data = (; y, x, z))
        end
        @test fit.converged
        @test !is_converged(fit)
        @test !DRModels._twin_vcov_positive_definite(fit.vcov)
        ct = coeftable(fit)
        @test all(isnan, ct.cols[2])
        @test all(isnan, ct.cols[3]) && all(isnan, ct.cols[4])
        txt = sprint(show, MIME"text/plain"(), fit)
        @test occursin("Wald SEs withheld", txt)
        @test occursin("Hessian is not positive definite", txt)
        @test !occursin("Inf", txt)
        @test all(isnan, summary(fit).cols[2])      # summary is coeftable
    end
end
