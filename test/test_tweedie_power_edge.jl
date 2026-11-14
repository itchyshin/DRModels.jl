# The crossed Tweedie route finds the power p by a bounded 1-D search on the
# logit-(1,2) scale, inside [-6, 6]. When the search ends at one end of that
# window the power is a bound, not an interior estimate. The fit must say so:
# `fit.geometry` is `:power_at_bound` and one warning names the power and the interval.
# (`is_converged` is already false for every crossed Tweedie fit, so it is not a test of this change.)
# The window, the search and the estimates are unchanged.
using DRModels
using Test, Logging
using StableRNGs
import Distributions

function _rtweedie_edge(rng, μ, φ, p)
    λ = μ^(2 - p) / (φ * (2 - p)); γ = φ * (p - 1) * μ^(p - 1); sh = (2 - p) / (p - 1)
    N = rand(rng, Distributions.Poisson(λ))
    N == 0 ? 0.0 : rand(rng, Distributions.Gamma(N * sh, γ))
end

# Shared with the 10-fit comparison on main: `kind = :edge` is Poisson counts
# (power runs to 1), `:interior` is a compound Poisson-Gamma with p = 1.5.
function _tweedie_edge_data(seed, kind)
    rng = StableRNG(seed); G = 10; H = 8; n = kind === :edge ? 160 : 100
    g = rand(rng, 1:G, n); h = rand(rng, 1:H, n); x = randn(rng, n)
    bg = 0.3 .* randn(rng, G); bh = 0.3 .* randn(rng, H)
    μ = exp.(0.5 .+ 0.3 .* x .+ bg[g] .+ bh[h])
    y = kind === :edge ? [Float64(rand(rng, Distributions.Poisson(m))) for m in μ] :
                         [_rtweedie_edge(rng, m, 0.6, 1.5) for m in μ]
    return (; y, x, g, h)
end

_tweedie_edge_fit(dat) = drm(bf(@formula(y ~ x + (1 | g) + (1 | h))), Tweedie(); data = dat)

@testset "crossed Tweedie: power at the end of its search interval is flagged" begin
    dat = _tweedie_edge_data(3, :edge)
    fit = nothing
    logger = Test.TestLogger()
    with_logger(logger) do
        fit = _tweedie_edge_fit(dat)
    end
    η = only(coef(fit, :nu))
    @test abs(η) > 6 - 1e-3                       # the search ended at the window end
    @test fit.geometry === :power_at_bound
    msgs = [string(r.message) for r in logger.logs if r.level == Logging.Warn]
    @test count(m -> occursin("end of its search interval", m), msgs) == 1
    @test any(m -> occursin("not an interior estimate", m), msgs)
    @test any(m -> occursin("within 1.0e-5 of an end", m), msgs)           # the rule is in p, 1e-5
    @test any(m -> occursin("no option to fix the power", m) && occursin("Poisson()", m) &&
                   occursin("Gamma()", m), msgs)                           # what to do next
    # the estimate itself is unchanged (still the window end, not moved or clamped)
    @test 1 + 1 / (1 + exp(-η)) ≈ 1.0025 atol = 1e-4
end

@testset "crossed Tweedie: an interior power is not flagged" begin
    dat = _tweedie_edge_data(11, :interior)
    fit = nothing
    logger = Test.TestLogger()
    with_logger(logger) do
        fit = _tweedie_edge_fit(dat)
    end
    @test abs(only(coef(fit, :nu))) < 5
    @test fit.geometry === :ok
    @test !any(r -> occursin("search interval", string(r.message)), logger.logs)
end

@testset "the edge rule is 1e-5 in p, not in the logit coordinate" begin
    lo = DRModels._logit12(-6.0); hi = DRModels._logit12(6.0)
    @test DRModels._tweedie_power_at_bound(lo + 5e-6)
    @test DRModels._tweedie_power_at_bound(hi - 5e-6)
    @test !DRModels._tweedie_power_at_bound(lo + 2e-5)
    @test !DRModels._tweedie_power_at_bound(hi - 2e-5)
    @test !DRModels._tweedie_power_at_bound(1.5)
end
