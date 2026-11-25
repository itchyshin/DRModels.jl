# cov lane 1125n7 — Tweedie density series and power-bound flags (src/tweedie.jl).
using DRModels
using Test, ForwardDiff, Random

@testset "cov 1125n7: _tweedie_window and _tweedie_logW" begin
    z, α = -2.5, -0.4
    win = DRModels._tweedie_window(z, α)
    @test win !== nothing
    jl, jh = win
    @test jl <= jh
    logW = DRModels._tweedie_logW(1.5, exp(z - α * log(1.5)), α, 1.0)
    @test isfinite(logW)
    @test DRModels._tweedie_window(50.0, α) === nothing
end

@testset "cov 1125n7: _logpdf_tweedie mass and support" begin
    μ, φ, p = 2.0, 0.5, 1.35
    l0 = DRModels._logpdf_tweedie(0.0, μ, φ, p)
    @test isfinite(l0) && l0 < 0
    y = 1.2
    ly = DRModels._logpdf_tweedie(y, μ, φ, p)
    @test isfinite(ly)
    @test ly > l0
    ys = (0.0, 0.5, 1.0, 1.5)
    dens = [DRModels._logpdf_tweedie(yv, μ, φ, p) for yv in ys]
    @test all(isfinite, dens)
    @test dens[3] > dens[2]
end

@testset "cov 1125n7: _tweedie_eta_derivs vs ForwardDiff" begin
    y, eta, phi, power = 0.8, log(2.0), 0.4, 1.4
    g1, h1 = DRModels._tweedie_eta_derivs(y, eta, phi, power)
    nll(e) = -DRModels._logpdf_tweedie(y, exp(e), phi, power)
    g_fd = ForwardDiff.derivative(nll, eta)
    h_fd = ForwardDiff.derivative(e -> ForwardDiff.derivative(nll, e), eta)
    @test g1 ≈ g_fd rtol = 1e-8
    @test h1 ≈ h_fd rtol = 1e-6
end

@testset "cov 1125n7: _tweedie_power_at_bound" begin
    lo = 1 + 1 / (1 + exp(6.0))
    hi = 1 + 1 / (1 + exp(-6.0))
    @test DRModels._tweedie_power_at_bound(lo + 1e-6)
    @test DRModels._tweedie_power_at_bound(hi - 1e-6)
    @test !DRModels._tweedie_power_at_bound((lo + hi) / 2)
    mid = 1.3
    @test !DRModels._tweedie_power_at_bound(mid)
end

@testset "cov 1125n7: _tweedie_power_at_bound geometry tag" begin
    fit = DrmFit(Tweedie(), [:mu => 1:1], [:mu => ["(Intercept)"]], [0.0], [0.1;;],
                 -5.0, 10, true, Dict(:mu => ones(10)), Dict(:mu => ones(10)),
                 Dict(:sigma => fill(0.5, 10), :nu => fill(1.35, 10)))
    flagged = DRModels._without_boundary_warnings() do
        DRModels._flag_tweedie_power_bound(fit, 1.35)
    end
    @test flagged.geometry === :power_at_bound
end