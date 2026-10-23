# #1007 (contract half left open by #1054): when the delta-method SE is not finite,
# the ratio's interval must be NaN, not the clamped-to-[0, 1] "valid-looking" (0, 1).
using DRModels
using Test, Random

# Rebuild a fit with a replaced covariance (all 23 positional fields).
function _t1007_with_vcov(fit, V)
    return DRModels.DrmFit(fit.family, fit.blocks, fit.coefnames, fit.theta, V,
        fit.loglik, fit.nobs, fit.converged, fit.means, fit.obs, fit.scales, fit.formula,
        fit.nll, fit.nllgrad, fit.ranef, fit.estim_method, fit.reml_loglik, fit.ml_loglik,
        fit.marginal, fit.phylo_penalty, fit.penalty, fit.iterations, fit.phylo_scale)
end

@testset "#1007 non-finite delta SE gives a NaN interval, not (0, 1)" begin
    Random.seed!(42)
    G = 25; m = 6; nn = G * m
    g = repeat(1:G, inner = m)
    y = 1.0 .+ 0.9 .* randn(G)[g] .+ 0.5 .* randn(nn)
    fit = drm(bf(@formula(y ~ 1 + (1 | g))), Gaussian(); data = (; y, g))

    # Reference: a usable covariance gives a finite SE and a CI inside [0, 1].
    ok = heritability(fit)
    @test isfinite(ok.se)
    @test 0 <= ok.ci.lower <= ok.ci.upper <= 1

    # A finite but non-positive-definite covariance (negated) makes the delta-method
    # variance negative, so `bias_correct` returns se = Inf.
    bad = _t1007_with_vcov(fit, -fit.vcov)
    for acc in (heritability, icc, repeatability)
        r = @test_logs (:warn, r"not finite") match_mode = :any acc(bad)
        @test r.se == Inf
        @test r.status === :vcov_unavailable
        @test isnan(r.ci.lower) && isnan(r.ci.upper)     # was (0.0, 1.0)
        @test 0 <= r.estimate <= 1                        # the point estimate is unaffected
    end
end
