# Family-dispersion boundary: Poisson ⊂ NegBinomial2 tests sigma at 0
# (DRModels #1115, twin of drmTMB #1557).
#
# lrtest/anova keep the naive χ² p-value (the existing "do not silently swap"
# contract from #304) but must warn, as they already do for an added RE SD.
# This file is the t10 red-proof: it fails on f71b292 (silent χ²) and passes
# once `_boundary_vc_warn` sees the family nest.

using DRModels
using Test, Random
import Distributions

function _nb2_null_data(; n = 80)
    # Official campaign seed 20261017000 overflows UInt32; same map as #1115.
    Random.seed!(20261017000 - 19000000000)
    x = randn(n)
    y = Float64.(rand.(Distributions.Poisson.(exp.(0.4 .+ 0.3 .* x))))
    return (; y, x)
end

@testset "lrtest warns on Poisson vs NegBinomial2 (#1115)" begin
    data = _nb2_null_data()
    pois = drm(bf(@formula(y ~ 1 + x)), DRModels.Poisson(); data, se = false)
    nb = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), NegBinomial2();
             data, se = false)

    t = @test_logs (:warn, r"(?i)boundary|sigma") match_mode = :any lrtest(pois, nb)
    @test t.dof == 1
    lb = lrt_boundary(nb, pois; q = 1)
    @test t.statistic ≈ lb.statistic atol = 1e-10
    # Packaged verb stays naive χ²; lrt_boundary is the chi-bar p-value.
    @test t.pvalue ≈ lb.pvalue_naive atol = 1e-12
    @test t.pvalue ≈ 2 * lb.pvalue atol = 1e-12
    @test anova(pois, nb) == t

    # Interior Gaussian scale regression: no family nest, no warning.
    Random.seed!(1)
    n = 80
    z = randn(n)
    y = 0.3 .+ 0.5 .* z .+ randn(n)
    gdata = (; y, z)
    g0 = drm(bf(@formula(y ~ 1 + z), @formula(sigma ~ 1)), Gaussian(); data = gdata)
    g1 = drm(bf(@formula(y ~ 1 + z), @formula(sigma ~ 1 + z)), Gaussian(); data = gdata)
    @test_logs lrtest(g0, g1)
end

@testset "family-boundary detector is Poisson to NegBinomial2 only (#1115)" begin
    nest = DRModels._family_boundary_nest
    @test nest(DRModels.Poisson(), NegBinomial2()) == (added = :sigma, q = 1)
    @test nest(NegBinomial2(), DRModels.Poisson()) === nothing
    @test nest(DRModels.Poisson(), DRModels.Poisson()) === nothing
    @test nest(Gaussian(), Student()) === nothing
end
