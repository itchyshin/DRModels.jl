# test_1260_optimizer_diagnostics.jl — #1260: bridge_diagnostics optimizer
# labels follow the recorded method, not "nonnegative niterations ⇒ LBFGS".
# Diagnostics/docs only: no fitted-value, likelihood, or solver default is
# changed here. Live drm() cases are small and skip R-parity / fixtures.

using DRModels
using Test, Random
import Distributions

function _fit1260(fam; iterations::Int = -1,
                  blocks = [:mu => 1:1],
                  coefnames = [:mu => ["(Intercept)"]])
    return DRModels._withiterations(
        DrmFit(fam, blocks, coefnames, [0.0], fill(1.0, 1, 1),
               0.0, 1, true, Dict{Symbol,Vector{Float64}}(),
               Dict{Symbol,Vector{Float64}}(), Dict{Symbol,Vector{Float64}}()),
        iterations)
end

function _report1260(; optimizer::Symbol = :default, actual::Symbol,
                     stage::Symbol = :primary, kept::Int = 0, runs = ())
    c = DRMControl(optimizer = optimizer)
    return DRMControlReport(c, Float64[], kept, true, :computed, nothing,
                            c.optimizer, actual, runs, stage,
                            DRModels._ctl_native_controls(c))
end

function _labeled1260(fam; iterations::Int = 4, optimizer::Symbol = :default,
                      actual::Symbol, stage::Symbol = :primary,
                      kept::Int = 0, runs = ())
    fit = _fit1260(fam; iterations = iterations)
    return DRModels._withcontrolinfo(fit, fit.vcov,
                                    _report1260(; optimizer, actual, stage, kept, runs))
end

function _indep_slopes1260(fam; iterations::Int = 8)
    return DRModels._withmarginal(
        _fit1260(fam; iterations,
                 blocks = [:mu => 1:1, :resd => 1:1],
                 coefnames = [:mu => ["(Intercept)"], :resd => ["g", "g:x"]]),
        :Laplace)
end

function _rs_sigma1260(; iterations::Int = 8)
    return DRModels._withmarginal(
        _fit1260(Gaussian(); iterations,
                 blocks = [:mu => 1:1, :recov => 1:1, :resd => 1:1],
                 coefnames = [:mu => ["(Intercept)"],
                              :recov => ["L11"],
                              :resd => ["g_logsigma"]]),
        :Laplace)
end

@testset "#1260 optimizer label vs recorded method" begin
    @testset "synthetic: default LBFGS from a known-LBFGS counted fingerprint" begin
        # Gaussian FE, no mixed-method markers — not "any nonnegative count".
        fit = _fit1260(Gaussian(); iterations = 5)
        d = bridge_diagnostics(fit)
        @test niterations(fit) == 5
        @test d.optimizer == "Optim.LBFGS"
        @test d.iterations == 5
        # Student ordinary (1 | g) Laplace: counted LBFGS, resd has no colon.
        student_ri = DRModels._withmarginal(
            _fit1260(Student(); iterations = 6,
                     blocks = [:mu => 1:1, :resd => 1:1],
                     coefnames = [:mu => ["(Intercept)"], :resd => ["g"]]),
            :Laplace)
        @test bridge_diagnostics(student_ri).optimizer == "Optim.LBFGS"
        @test !DRModels._unrecorded_mixed_method_route(student_ri)
    end

    @testset "synthetic: unsupported count stays missing" begin
        fit = _fit1260(Gaussian(); iterations = -1)
        d = bridge_diagnostics(fit)
        @test niterations(fit) == -1
        @test ismissing(d.optimizer)
        @test ismissing(d.iterations)
    end

    @testset "synthetic: qualified BFGS matches the recorded method" begin
        counted = _labeled1260(Gaussian(); iterations = 6, optimizer = :bfgs,
                               actual = :bfgs)
        d = bridge_diagnostics(counted)
        @test control_info(counted).actual_optimizer === :bfgs
        @test d.optimizer == "Optim.BFGS"
        @test d.iterations == 6
        # Gaussian mean (1 | g) records the method but does not attach a count.
        uncounted = _labeled1260(Gaussian(); iterations = -1, optimizer = :bfgs,
                                 actual = :bfgs)
        d_ri = bridge_diagnostics(uncounted)
        @test d_ri.optimizer == "Optim.BFGS"
        @test ismissing(d_ri.iterations)
    end

    @testset "synthetic: recorded LBFGS and reserved Newton" begin
        d_lbfgs = bridge_diagnostics(_labeled1260(Gaussian(); optimizer = :lbfgs,
                                                  actual = :lbfgs))
        @test d_lbfgs.optimizer == "Optim.LBFGS"
        d_newton = bridge_diagnostics(_labeled1260(Beta(); optimizer = :default,
                                                   actual = :newton))
        @test d_newton.optimizer == "Optim.Newton"
        @test d_newton.iterations == 4
    end

    @testset "synthetic: do not infer LBFGS from an unrecorded count" begin
        fit = _labeled1260(Gaussian(); iterations = 8, actual = :unrecorded)
        d = bridge_diagnostics(fit)
        @test control_info(fit).actual_optimizer === :unrecorded
        @test ismissing(d.optimizer)
        @test d.iterations == 8
    end

    @testset "synthetic: Beta Newton stage is explicitly unknown unless recorded" begin
        d_bare = bridge_diagnostics(_fit1260(Beta(); iterations = 7))
        @test ismissing(d_bare.optimizer)
        @test d_bare.iterations == 7
        # Default control still allows polish, so a stored :lbfgs may be stale.
        d_default = bridge_diagnostics(_labeled1260(Beta(); optimizer = :default,
                                                    actual = :lbfgs, iterations = 3))
        @test ismissing(d_default.optimizer)
        @test d_default.iterations == 3
        # Explicit optimizer skips polish; the recorded method is the accepted stage.
        d_bfgs = bridge_diagnostics(_labeled1260(Beta(); optimizer = :bfgs,
                                                 actual = :bfgs, iterations = 9))
        @test d_bfgs.optimizer == "Optim.BFGS"
        d_lbfgs = bridge_diagnostics(_labeled1260(Beta(); optimizer = :lbfgs,
                                                  actual = :lbfgs, iterations = 2))
        @test d_lbfgs.optimizer == "Optim.LBFGS"
    end

    @testset "synthetic: counted BFGS-first sibling routes stay unknown" begin
        for fam in (Gamma(), Student())
            fit = _indep_slopes1260(fam)
            @test DRModels._unrecorded_mixed_method_route(fit)
            @test niterations(fit) == 8
            @test isnothing(fit.control_info)
            @test ismissing(bridge_diagnostics(fit).optimizer)
            @test bridge_diagnostics(fit).iterations == 8
        end
        rs = _rs_sigma1260()
        @test DRModels._unrecorded_mixed_method_route(rs)
        @test niterations(rs) == 8
        @test ismissing(bridge_diagnostics(rs).optimizer)
        @test bridge_diagnostics(rs).iterations == 8
    end

    @testset "synthetic: multi_start kept method, not last optimizer_runs entry" begin
        runs = ((method = :lbfgs, stage = :start, start_index = 1),
                (method = :bfgs, stage = :start, start_index = 2))
        fit = _labeled1260(Gaussian(); actual = :lbfgs, kept = 1, runs = runs)
        info = control_info(fit)
        @test info.actual_optimizer === :lbfgs
        @test info.kept_start == 1
        @test last(info.optimizer_runs).method === :bfgs
        @test bridge_diagnostics(fit).optimizer == "Optim.LBFGS"
    end
end

@testset "#1260 sibling live routes (no parity fixtures)" begin
    Random.seed!(1260)
    n = 60
    x = randn(n)
    y = 0.2 .+ 0.5 .* x .+ 0.4 .* randn(n)
    data = (; y, x)
    f = bf(@formula(y ~ x), @formula(sigma ~ 1))

    default = drm(f, Gaussian(); data = data)
    d_default = bridge_diagnostics(default)
    @test niterations(default) >= 0
    @test d_default.optimizer == "Optim.LBFGS"
    @test d_default.iterations == niterations(default)
    @test isfinite(loglik(default))

    bfgs = drm(f, Gaussian(); data = data, control = DRMControl(optimizer = :bfgs))
    d_bfgs = bridge_diagnostics(bfgs)
    @test control_info(bfgs).actual_optimizer === :bfgs
    @test d_bfgs.optimizer == "Optim.BFGS"
    @test niterations(bfgs) >= 0
    @test d_bfgs.iterations == niterations(bfgs)
    @test isfinite(loglik(bfgs))

    μ = 1 ./ (1 .+ exp.(-(0.2 .+ 0.4 .* x)))
    yβ = Float64.([rand(Distributions.Beta(μi * 8, (1 - μi) * 8)) for μi in μ])
    beta = drm(f, Beta(); data = (; y = yβ, x))
    d_beta = bridge_diagnostics(beta)
    @test niterations(beta) >= 0
    @test ismissing(d_beta.optimizer)          # Newton polish unrecorded
    @test d_beta.iterations == niterations(beta)
    @test isfinite(loglik(beta))

    G, m = 12, 8
    g = repeat(1:G, inner = m)
    xg = randn(G * m)
    b = 0.35 .* randn(G)
    yg = Float64.([rand(Distributions.Poisson(exp(0.15 + 0.25 * xg[i] + b[g[i]])))
                   for i in eachindex(xg)])
    poi = drm(bf(@formula(y ~ x + (1 | g))), Poisson();
              data = (; y = yg, x = xg, g), marginal = :Laplace)
    d_poi = bridge_diagnostics(poi)
    @test niterations(poi) == -1
    @test ismissing(d_poi.iterations)
    @test ismissing(d_poi.optimizer)

    nre = 60
    Gre = 10
    gre = repeat(1:Gre, inner = 6)
    ure = 0.3 .* randn(Gre)
    xre = randn(nre)
    yre = 1 .+ 0.4 .* xre .+ ure[gre] .+ 0.5 .* randn(nre)
    dre = (; y = yre, x = xre, g = gre)
    fre = drm(bf(@formula(y ~ 1 + x + (1 | g)), @formula(sigma ~ 1)), Gaussian();
              data = dre)
    @test niterations(fre) == -1
    @test ismissing(bridge_diagnostics(fre).optimizer)
    fre_bfgs = drm(bf(@formula(y ~ 1 + x + (1 | g)), @formula(sigma ~ 1)), Gaussian();
                   data = dre, control = DRMControl(optimizer = :bfgs, se = false))
    @test control_info(fre_bfgs).actual_optimizer === :bfgs
    @test niterations(fre_bfgs) == -1
    @test bridge_diagnostics(fre_bfgs).optimizer == "Optim.BFGS"
    @test ismissing(bridge_diagnostics(fre_bfgs).iterations)
    @test isfinite(loglik(fre_bfgs))
end

# Live B1 fingerprints: unrecorded counted BFGS-first Laplace routes. No
# `control=` — a report would take the `:unrecorded` branch instead of the
# no-report mixed-method exclusion. Skip R-parity / fixtures.
@testset "#1260 B1 live mixed-method counted routes (no parity fixtures)" begin
    Random.seed!(12602)
    G, m = 8, 6
    n = G * m
    g = repeat(1:G, inner = m)
    x = randn(n)
    b0 = 0.25 .* randn(G)
    b1 = 0.15 .* randn(G)

    η = 0.35 .+ 0.2 .* x .+ b0[g] .+ b1[g] .* x
    yg = exp.(η .+ 0.12 .* randn(n))
    fg = drm(bf(@formula(y ~ 1 + x + (1 | g) + (0 + x | g)), @formula(sigma ~ 1)),
             Gamma(); data = (; y = yg, x, g))
    @test fg.marginal === :Laplace
    @test DRModels._unrecorded_mixed_method_route(fg)
    @test niterations(fg) >= 0
    @test isnothing(fg.control_info)
    @test ismissing(bridge_diagnostics(fg).optimizer)
    @test isfinite(loglik(fg))

    y = 0.3 .+ 0.4 .* x .+ b0[g] .+ b1[g] .* x .+ 0.35 .* randn(n)
    fr = drm(bf(@formula(y ~ 1 + x + (1 + x | g)), @formula(sigma ~ 1 + (1 | g))),
             Gaussian(); data = (; y, x, g))
    @test fr.marginal === :Laplace
    @test DRModels._unrecorded_mixed_method_route(fr)
    @test niterations(fr) >= 0
    @test isnothing(fr.control_info)
    @test ismissing(bridge_diagnostics(fr).optimizer)
    @test isfinite(loglik(fr))
end

println("ISSUE_1260_OPTIMIZER_DIAGNOSTICS")
