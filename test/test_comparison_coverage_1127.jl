# Cursor coverage 2026-11-27: `src/comparison.jl` guards and accessors not
# reached by the existing comparison / lrtest / update test cluster alone.
using DRModels
using Test, Random

function _argmsg(f)
    try
        f()
        return ""
    catch e
        e isa ArgumentError || rethrow()
        return e.msg
    end
end

@testset "comparison.jl coverage (1127)" begin
    Random.seed!(1127)

    @testset "aicc is Inf when n − k − 1 is not positive" begin
        n = 3
        x1, x2 = randn(n), randn(n)
        y = 0.2 .+ 0.3 .* x1 .+ 0.1 .* x2 .+ 0.4 .* randn(n)
        fit = drm(bf(@formula(y ~ 1 + x1 + x2), @formula(sigma ~ 1)),
                  Gaussian(); data = (; y, x1, x2))
        k = dof(fit)
        @test nobs(fit) - k - 1 <= 0
        @test aicc(fit) == Inf
        @test isfinite(aic(fit))
    end

    @testset "update and weights guards" begin
        n = 40
        x, y = randn(n), 0.4 .+ 0.6 .* randn(n) .+ 0.3 .* randn(n)
        data, form = (; y, x), bf(@formula(y ~ 1 + x), @formula(sigma ~ 1))
        full = drm(form, Gaussian(); data = data)
        reduced = bf(@formula(y ~ 1), @formula(sigma ~ 1))
        msg = _argmsg(() -> update(full, reduced, 0; data = data))
        @test occursin("named arguments", msg) && occursin("unnamed", msg)
        msg = _argmsg(() -> update(full, reduced; data = data, family = DRModels.Poisson()))
        @test occursin("family", msg) && occursin("not a refit keyword", msg)
        w = fill(1.0, n); w[1:4] .= 2.0
        fitw = drm(form, Gaussian(); data = data, weights = w)
        @test weights(fitw) == w
        msg = _argmsg(() -> update(fitw, form; data = data))
        @test occursin("weights", msg)
        unw = update(fitw, form; data = data, weights = nothing)
        @test weights(unw) === nothing
    end

    @testset "lrtest mixture refusals and marginal guard" begin
        G, m = 8, 3
        g, h = repeat(1:G, inner = m), repeat(1:G, outer = m)
        x = randn(G * m)
        y = 0.25 .+ 0.35 .* x .+ randn(G)[g] .+ 0.5 .* randn(G)[h]
        d = (; y, x, g, h)
        reduced = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)),
                      Gaussian(); data = d, control = DRMControl(se = false))
        full2 = drm(bf(@formula(y ~ x + (1 | g) + (1 | h)), @formula(sigma ~ 1)),
                    Gaussian(); data = d, control = DRMControl(se = false))
        msg = _argmsg(() -> lrtest(reduced, full2; mixture = true))
        @test occursin("mixture = true", msg) && occursin("variance", msg)
        full1 = drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + x)),
                    Gaussian(); data = d, control = DRMControl(se = false))
        msg = _argmsg(() -> lrtest(reduced, full1; mixture = true))
        @test occursin("plain", msg)
        plain = lrtest(reduced, full2)
        @test plain.dof == 2 && 0 <= plain.pvalue <= 1
    end
end