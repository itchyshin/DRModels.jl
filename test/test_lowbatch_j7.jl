# test_lowbatch_j7.jl — regression tests for the low-severity twin-review fixes
# (#322–#326). Each testset captures ONE fixed finding so a regression is caught.
using DRModels
using Test, Random
import Distributions

# --- #323.1: saturated missing-response Gaussian fit warns (residual dof 0) ---
@testset "#323.1 saturated missing-response Gaussian fit warns" begin
    # 2 mean params (intercept + x) + 1 scale intercept = 3 fixed params; give
    # exactly 3 observed responses so residual dof = 0 (saturated).
    n = 8
    x = collect(range(-1, 1; length = n))
    y = Vector{Union{Missing,Float64}}(0.2 .+ 0.5 .* x .+ 0.3 .* randn(MersenneTwister(1), n))
    y[4:end] .= missing                      # 3 observed rows, pμ+pσ = 3
    dat = (; y, x)
    f = bf(@formula(y ~ x), @formula(sigma ~ 1))
    @test_logs (:warn, r"saturated") (:warn, r"missing/NaN") match_mode = :any drm(f, Gaussian(); data = dat)
end

# --- #324.1: bridge SD-row picker errors instead of mislabelling a fixed row ---
@testset "#324.1 bridge picker errors when no SD row present" begin
    # A profile-style result with only fixed-effect rows (no :resd*) must NOT
    # silently return the first (μ-intercept) row as the SD interval.
    rows = [(param = :mu, coef = "(Intercept)", estimate = 0.3, lower = 0.1, upper = 0.5),
            (param = :mu, coef = "x",           estimate = 0.5, lower = 0.3, upper = 0.7)]
    @test_throws ArgumentError DRModels._bridge_pick_sd_row(rows)
    # And it still finds a genuine SD row.
    ok = [(param = :mu, coef = "(Intercept)", estimate = 0.3, lower = 0.1, upper = 0.5),
          (param = :resd, coef = "g", estimate = 0.4, lower = 0.2, upper = 0.6)]
    @test DRModels._bridge_pick_sd_row(ok).param === :resd
end

# --- #322.3: scale-axis RE SD is keyed `<grp>_logsigma`, distinct from mean axis ---
@testset "#322.3 scale-axis RE SD keyed on the log-sigma axis" begin
    Random.seed!(20260703)
    G = 30; m = 20; n = G * m
    g = repeat(1:G, inner = m); x = randn(n)
    β = [0.4, -0.2]; γ0 = log(0.5); σb = 0.5
    bg = σb .* randn(G)
    y = β[1] .+ β[2] .* x .+ exp.(γ0 .+ bg[g]) .* randn(n)
    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + (1 | g))), Gaussian(); data = (; y, x, g))
    @test haskey(re_sd(fit), :g_logsigma)
    @test !haskey(re_sd(fit), :g)
end

# --- #325.5: NB2/Gamma/Beta VA fitters reject sigma ~ x (non-intercept Xσ) ---
@testset "#325.5 VA fitters guard sigma ~ 1" begin
    Random.seed!(20260704)
    n = 200; ng = 20
    gidx = repeat(1:ng, inner = n ÷ ng)
    x = randn(n)
    Xμ = hcat(ones(n), x)
    Xσ_bad = hcat(ones(n), x)                 # sigma ~ x — must be rejected
    y = Float64.(rand(0:5, n))
    nmμ = ["(Intercept)", "x"]; nmσ = ["(Intercept)", "x"]
    @test_throws ErrorException DRModels._fit_nb2_ranef_va(DRModels.NegBinomial2(), y, Xμ, Xσ_bad, gidx, ng, nmμ, nmσ, :g, 1e-6)
    yg = abs.(randn(n)) .+ 0.1
    @test_throws ErrorException DRModels._fit_gamma_ranef_va(DRModels.Gamma(), yg, Xμ, Xσ_bad, gidx, ng, nmμ, nmσ, :g, 1e-6)
    yb = clamp.(rand(n), 0.01, 0.99)
    @test_throws ErrorException DRModels._fit_beta_ranef_va(DRModels.Beta(), yb, Xμ, Xσ_bad, gidx, ng, nmμ, nmσ, :g, 1e-6)
end
