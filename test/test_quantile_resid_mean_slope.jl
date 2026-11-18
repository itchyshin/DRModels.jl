# Quantile residuals of a lone mean random slope `(0 + x | g)`. The fit stores one
# `:resd` block for it, the same as `(1 | g)`, and the residual code used to read it as
# a random intercept: every row was judged against N(mu, sigma^2 + sd^2) although the
# slope contributes x_i^2 sd^2. The fit does not keep x, so a slope keeps the
# population reference (random effect set to 0), as the other shapes the docstring
# lists (crossed, correlated slopes). A random intercept keeps its marginal.
using DRModels
using Test, Random
import Distributions

@testset "quantile residuals: lone mean random slope is not an intercept" begin
    rng = Random.MersenneTwister(20261113)
    G = 20; m = 15; n = G * m
    g = repeat(1:G, inner = m); x = randn(rng, n)
    a = 0.5 .* randn(rng, G); s = 0.6 .* randn(rng, G)
    η = 0.2 .+ 0.3 .* x .+ a[g] .+ s[g] .* x
    dg = (; y = η .+ 0.7 .* randn(rng, n), x, g)
    dp = (; y = Float64.([rand(rng, Distributions.Poisson(exp(0.2 + 0.3 * x[i] + s[g[i]] * x[i]))) for i in 1:n]), x, g)

    @testset "Gaussian (0 + x | g): population reference, not the intercept marginal" begin
        fit = drm(bf(@formula(y ~ x + (0 + x | g))), Gaussian(); data = dg)
        @test DRModels._ranef_marginal_mix(fit, fit.family, fit.means[:mu]) === nothing
        r = residuals(fit; type = :quantile)
        σ = fit.scales[:sigma]
        @test r ≈ (fit.obs[:mu] .- fit.means[:mu]) ./ σ atol = 1e-12
        sb = sqrt(vc(fit)[:g][1, 1])
        wrong = (fit.obs[:mu] .- fit.means[:mu]) ./ sqrt.(σ .^ 2 .+ sb^2)
        @test maximum(abs.(r .- wrong)) > 0.05
    end

    @testset "Gaussian (1 | g) keeps its closed-form marginal" begin
        fit = drm(bf(@formula(y ~ x + (1 | g))), Gaussian(); data = dg)
        r = residuals(fit; type = :quantile)
        sb = sqrt(vc(fit)[:g][1, 1])
        @test r ≈ (fit.obs[:mu] .- fit.means[:mu]) ./ sqrt.(fit.scales[:sigma] .^ 2 .+ sb^2) atol = 1e-12
    end

    @testset "Poisson (0 + x | g): population reference" begin
        fit = drm(bf(@formula(y ~ x + (0 + x | g))), Poisson(); data = dp)
        @test DRModels._ranef_marginal_mix(fit, fit.family, fit.means[:mu]) === nothing
        r = residuals(fit; type = :quantile, rng = Random.MersenneTwister(7))
        rr = Random.MersenneTwister(7)
        ref = map(1:n) do i
            d = Distributions.Poisson(fit.means[:mu][i]); y = round(Int, fit.obs[:mu][i])
            lo = Distributions.cdf(d, y - 1)
            Distributions.quantile(Distributions.Normal(),
                clamp(lo + (Distributions.cdf(d, y) - lo) * rand(rr), eps(), 1 - eps()))
        end
        @test r ≈ ref atol = 1e-10
    end

    @testset "Poisson (1 | g) still gets the marginal mixture" begin
        fit = drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = dp)
        @test DRModels._ranef_marginal_mix(fit, fit.family, fit.means[:mu]) !== nothing
    end
end
