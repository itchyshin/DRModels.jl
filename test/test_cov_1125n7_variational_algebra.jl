# cov lane 1125n7 — variational inner Newton algebra (src/variational.jl).
using DRModels
using SpecialFunctions: loggamma
using Test, LinearAlgebra

@testset "cov 1125n7: _va_damped_step ascent" begin
    Hmm, Hmr, Hrr = -2.0, 0.1, -3.0
    gm, gr = 0.2, -0.1
    Δm, Δr = DRModels._va_damped_step(Hmm, Hmr, Hrr, gm, gr)
    @test isfinite(Δm) && isfinite(Δr)
    # Positive-definite H needs Levenberg damping before the ascent solve
    Δm2, Δr2 = DRModels._va_damped_step(1.0, 0.0, 1.0, 0.1, 0.1)
    @test isfinite(Δm2) && isfinite(Δr2)
    @test Δm2 ≈ 0.1 && Δr2 ≈ 0.1
    Δm3, Δr3 = DRModels._va_damped_step(-1.0, 0.0, -1.0, 0.0, 0.0)
    @test Δm3 == 0.0 && Δr3 == 0.0
end

@testset "cov 1125n7: _va_cap_step preserves direction" begin
    Δm, Δr = 10.0, 5.0
    cm, cr = DRModels._va_cap_step(Δm, Δr)
    @test abs(cm) <= 4 + 1e-12
    @test abs(cr) <= 2 + 1e-12
    @test sign(cm) == sign(Δm)
    @test sign(cr) == sign(Δr)
    sm, sr = DRModels._va_cap_step(0.5, 0.2)
    @test sm ≈ 0.5 && sr ≈ 0.2
end

@testset "cov 1125n7: _va_newton_decrement2" begin
    gm, gr = 0.01, -0.02
    Hmm, Hmr, Hrr = -1.5, 0.05, -2.0
    λ2 = DRModels._va_newton_decrement2(gm, gr, Hmm, Hmr, Hrr)
    @test isfinite(λ2) && λ2 >= 0
    @test DRModels._va_newton_decrement2(gm, gr, 1.0, Hmr, Hrr) == Inf
    @test DRModels._va_newton_decrement2(NaN, gr, Hmm, Hmr, Hrr) == Inf
end

@testset "cov 1125n7: _va_negdef" begin
    @test DRModels._va_negdef(-1.0, -1.0, 0.0, 1.0)
    @test !DRModels._va_negdef(1.0, -1.0, 0.0, -1.0)
    @test !DRModels._va_negdef(-1.0, 1.0, 0.0, -1.0)
end

@testset "cov 1125n7: _lgamma_stirling_rem" begin
    xlo = 5.0
    rlo = DRModels._lgamma_stirling_rem(xlo)
    exact_lo = loggamma(xlo) - ((xlo - 0.5) * log(xlo) - xlo + 0.5 * log(2π))
    @test rlo ≈ exact_lo rtol = 1e-12
    for x in (20.0, 100.0)
        r = DRModels._lgamma_stirling_rem(x)
        @test isfinite(r) && r > 0
        @test r < 1 / x
    end
end

@testset "cov 1125n7: _binom_group_kernel shape" begin
    b = 0.2
    η0idx = [0.1, -0.3]
    sidx = [3, 7]
    nidx = [10, 12]
    val = DRModels._binom_group_kernel(b, η0idx, sidx, nidx)
    @test isfinite(val)
    val0 = DRModels._binom_group_kernel(0.0, η0idx, sidx, nidx)
    @test isfinite(val0)
    val_hi = DRModels._binom_group_kernel(5.0, η0idx, sidx, nidx)
    @test val_hi <= val0
end

@testset "cov 1125n7: _va_inner_converged" begin
    @test DRModels._va_inner_converged([0.5, 0.1], "Poisson VA")
    @test_logs (:warn,) match_mode = :any begin
        @test !DRModels._va_inner_converged([0.5, 3.0], "Poisson VA")
    end
    DRModels._without_boundary_warnings() do
        @test !DRModels._va_inner_converged([2.0], "Binomial VA")
    end
end