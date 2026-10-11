# test_converged_scale_spread_skewnormal_nu.jl — DRModels.jl#1059 (unit-free
# runaway-scale bar in `is_converged`) and #1060 (SkewNormal nu-near-0 Wald caveat).
using DRModels, Test, Random

# Rebuild a DrmFit with a different `converged` flag / sigma scales (DrmFit is immutable).
function _with_fields(fit::DRModels.DrmFit; kw...)
    vals = map(fieldnames(DRModels.DrmFit)) do nm
        haskey(kw, nm) ? kw[nm] : getfield(fit, nm)
    end
    return DRModels.DrmFit(vals...)
end

@testset "#1059 is_converged rejects a collapsing scale in some rows" begin
    Random.seed!(1)
    n = 200
    x = randn(n)
    y = 1.0 .+ 0.5 .* x .+ exp.(0.2 .+ 0.3 .* x) .* randn(n)
    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian(); data = (; y, x))
    @test fit.converged
    @test is_converged(fit)                       # a healthy heteroscedastic fit is untouched

    # One group's sigma collapses toward 0 while the other keeps an O(1) scale:
    # the old `max(sigma)` bar passes this; the spread bar must not.
    s = copy(fit.scales[:sigma]); s[1:50] .= 1e-12
    bad = _with_fields(fit; converged = true, scales = Dict(:sigma => s))
    @test bad.converged
    @test !is_converged(bad)

    # Unit-free: scaling the whole response (and sigma) by 1e-5 must not trip the bar.
    s2 = fit.scales[:sigma] .* 1e-5
    @test is_converged(_with_fields(fit; scales = Dict(:sigma => s2),
                                    obs = Dict(:mu => fit.obs[:mu] .* 1e-5)))
end

@testset "#1060 SkewNormal warns when nu is near 0" begin
    Random.seed!(1)
    n = 300
    x = randn(n)
    y = 1.0 .+ 0.5 .* x .+ randn(n)               # symmetric errors, true nu = 0
    @test DRModels._skewnormal_nu_wald_unreliable([0.1, -0.2, 0.05])
    @test !DRModels._skewnormal_nu_wald_unreliable(Float64[])
    @test !DRModels._skewnormal_nu_wald_unreliable([0.8, 0.8], [0.8], fill(0.01, 1, 1))   # z = 8
    @test DRModels._skewnormal_nu_wald_unreliable([0.8, 0.8], [0.8], fill(0.5, 1, 1))     # z ≈ 1.1
    @test_logs (:warn, r"slant ν is compatible with 0") match_mode = :any drm(
        bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)), SkewNormal(); data = (; y, x))

    # A clearly skewed fit does not warn.
    Random.seed!(2)
    z = randn(n); w = abs.(randn(n))
    ys = 1.0 .+ 0.5 .* x .+ 2.0 .* w .+ 0.3 .* z   # strong right skew
    fit = @test_logs min_level = Base.CoreLogging.Warn drm(
        bf(@formula(ys ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)), SkewNormal(); data = (; ys, x))
    @test maximum(abs, fit.scales[:nu]) >= 0.5
end
