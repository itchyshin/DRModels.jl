# residuals(fit; type = :quantile, quantile_reference = :plugin): the drmTMB-style
# plug-in quantile residual (F at the fitted-row parameters with the random-effect modes
# included). The default (:marginal, averaged) is unchanged. #1101 / SHINICHI-0728 D3.
using DRModels
using Test, Random, Statistics
import Distributions

@testset "quantile residuals: plug-in option (#1101)" begin
    y = [0.952, 0.835, 0.546, 0.057, 1.361, -0.639, 0.319, 1.008, 0.206, 0.64, 0.913, 1.14,
         -0.37, 0.057, 0.625, -0.409, 0.158, -0.445, -0.818, 0.428, -0.637, -0.122, 0.497, 0.157]
    x = [1.112, 1.037, 0.395, -1.666, 0.506, -1.691, -0.89, 0.305, 0.521, 0.309, -0.661, 0.557,
         -0.62, -0.344, 0.632, -0.447, -0.521, -0.808, 0.427, 0.048, -2.014, -0.106, 0.497, -1.465]
    g = repeat(1:6, inner = 4)
    fit = drm(bf(@formula(y ~ x + (1 | g))), Gaussian(); data = (; y, x, g))
    @test fit.converged

    qm = residuals(fit; type = :quantile)
    @test qm == residuals(fit; type = :quantile, quantile_reference = :marginal)  # default unchanged
    qp = residuals(fit; type = :quantile, quantile_reference = :plugin)

    # plug-in is the conditional z-score (y - fitted) / sigma
    @test maximum(abs.(qp .- (y .- fitted(fit)) ./ fit.scales[:sigma])) < 1e-8
    # pinned to drmTMB 0.7.1 drm_quantile_residuals() on the same data (REML-free ML fit)
    r_ref = [0.18636213, -0.01300737, -0.08087738, 0.71037531, 1.50360224, -1.05388527]
    @test maximum(abs.(qp[1:6] .- r_ref)) < 1e-6
    @test std(qp) ≈ 0.96295959 atol = 1e-6
    # and it is not the averaged residual
    @test maximum(abs.(qp .- qm)) > 0.05

    # no random effect: both references coincide
    fit0 = drm(bf(@formula(y ~ x)), Gaussian(); data = (; y, x))
    @test residuals(fit0; type = :quantile, quantile_reference = :plugin) ==
          residuals(fit0; type = :quantile)

    # Poisson (1 | g): plug-in randomises inside [F(y-1), F(y)] at the conditional mean
    yc = [0, 1, 0, 0, 3, 0, 0, 2, 1, 1, 1, 2, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]
    fp = drm(bf(@formula(yc ~ x + (1 | g))), DRModels.Poisson(); data = (; yc, x, g))
    qpp = residuals(fp; type = :quantile, quantile_reference = :plugin, rng = MersenneTwister(3))
    μc = fitted(fp)
    N01 = Distributions.Normal()
    lo = [Distributions.quantile(N01, clamp(Distributions.cdf(Distributions.Poisson(μc[i]), yc[i] - 1), eps(), 1 - eps())) for i in eachindex(yc)]
    hi = [Distributions.quantile(N01, clamp(Distributions.cdf(Distributions.Poisson(μc[i]), yc[i]), eps(), 1 - eps())) for i in eachindex(yc)]
    @test all(lo .- 1e-9 .<= qpp .<= hi .+ 1e-9)

    @test_throws ArgumentError residuals(fit; type = :quantile, quantile_reference = :bogus)
end
