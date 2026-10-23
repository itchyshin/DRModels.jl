# In lme4, glmmTMB and drmTMB, `(x | g)` means `(1 + x | g)`: the random intercept is
# implicit unless removed with `0 +` / `- 1`. DRModels used to refuse `(x | g)`
# ("unsupported random-effect term"), found by the lme4 sleepstudy twin (#706 / #889).
# Fix: `_split_ranef` normalises an lhs with no explicit constant to `1 + lhs`.
# drmTMB (measured 2026-09-29, sleepstudy): `(Days|Subject)` and `(1+Days|Subject)`
# give the identical logLik, -875.9697.
using DRModels
using Test, Random, LinearAlgebra
import Distributions

_ric_same(a, b, key) = begin
    @test loglik(a) ≈ loglik(b) atol = 1e-10
    @test coef(a, :mu) ≈ coef(b, :mu) atol = 1e-10
    @test vc(a)[key] ≈ vc(b)[key] atol = 1e-10
end

@testset "implicit random intercept: (x | g) == (1 + x | g)" begin
    Random.seed!(20260929)
    G = 30; m = 12; n = G * m
    g = repeat(1:G, inner = m); x = randn(n); z = randn(n)
    B = cholesky(Symmetric([0.25 0.06; 0.06 0.16])).L * randn(2, G)
    η = 0.3 .+ 0.4 .* x .+ B[1, g] .+ B[2, g] .* x
    yg = η .+ 0.5 .* randn(n)
    yp = Float64.([rand(Distributions.Poisson(exp(e))) for e in η])
    dg = (; y = yg, x, z, g); dp = (; y = yp, x, z, g)

    @testset "Gaussian" begin
        a = drm(bf(@formula(y ~ x + (x | g))), Gaussian(); data = dg)
        b = drm(bf(@formula(y ~ x + (1 + x | g))), Gaussian(); data = dg)
        _ric_same(a, b, :g)
        @test size(vc(a)[:g]) == (2, 2)
    end

    @testset "Poisson" begin
        a = drm(bf(@formula(y ~ x + (x | g))), Poisson(); data = dp)
        b = drm(bf(@formula(y ~ x + (1 + x | g))), Poisson(); data = dp)
        _ric_same(a, b, :g)
        @test size(vc(a)[:g]) == (2, 2)
    end

    @testset "explicit 0 / -1 keep slope-only (or the existing refusal)" begin
        ref = drm(bf(@formula(y ~ x + (0 + x | g))), Gaussian(); data = dg)
        @test size(vc(ref)[:g]) == (1, 1)
        for f in (@formula(y ~ x + (0 + x | g)), @formula(y ~ x + (-1 + x | g)))
            r = try
                drm(bf(f), Gaussian(); data = dg)
            catch err
                err
            end
            if r isa Exception
                @test occursin("unsupported random-effect term", sprint(showerror, r))
            else
                @test loglik(r) ≈ loglik(ref) atol = 1e-10
            end
        end
    end

    @testset "(x + z | g) gets an implicit intercept (same outcome as 1 + x + z)" begin
        msg(f) = try
            drm(bf(f), Gaussian(); data = dg); ""
        catch err
            sprint(showerror, err)
        end
        m1 = msg(@formula(y ~ x + (x + z | g)))
        m2 = msg(@formula(y ~ x + (1 + x + z | g)))
        @test !isempty(m2)          # multi-slope is unsupported ...
        @test m1 == m2              # ... and reported identically once 1 is implicit
        r1 = DRModels._split_ranef(@formula(y ~ x + (x + z | g)).rhs)[2][1][1]
        r2 = DRModels._split_ranef(@formula(y ~ x + (1 + x + z | g)).rhs)[2][1][1]
        @test length(r1.args) == length(r2.args) == 3
    end
end
