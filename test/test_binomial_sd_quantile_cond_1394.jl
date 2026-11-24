# Refs #1394: quantile residuals on Binomial `sd(g) ~ …` use Laplace modes when stored.
using DRModels
using Test, Random
import Distributions as D

@testset "Binomial sd regression quantile uses conditional modes (#1394)" begin
    # Interior AGHQ fit: enough replication per group, moderate RE and SD effects.
    G, m, N = 40, 8, 10
    n = G * m
    rng = MersenneTwister(100)
    g = repeat(string.("g", 1:G), inner = m)
    gx_g = repeat([-0.25, 0.25], inner = cld(G, 2))[1:G]
    gx = gx_g[repeat(1:G, inner = m)]
    x1 = repeat(collect(range(-0.8, 0.8; length = m)), outer = G)
    u = 0.2 .* randn(rng, G)
    η = [0.05 + 0.25 * x1[i] + u[(i - 1) ÷ m + 1] for i in 1:n]
    p = clamp.(1 ./ (1 .+ exp.(-η)), 0.25, 0.75)
    s = [Float64(rand(rng, D.Binomial(N, p[i]))) for i in 1:n]
    fl = N .- s
    fit = drm(bf(@formula(cbind(s, fl) ~ x1 + (1 | g)), @formula(sd(g) ~ gx)),
              Binomial(); data = (; s, fl, x1, g, gx))
    @test fit.converged
    μ0 = fit.means[:mu]
    μc = DRModels._conditional_mean_at_modes(fit, Binomial(), μ0, nothing)
    @test any(!isapprox(μc[i], μ0[i]; atol = 1e-12) for i in eachindex(μ0))
    q = residuals(fit; type = :quantile, rng = MersenneTwister(1))
    lo0 = [D.quantile(D.Normal(), s[i] == 0 ? 0.0 : D.cdf(D.Binomial(N, μ0[i]), s[i] - 1))
           for i in 1:n]
    hi0 = [D.quantile(D.Normal(), D.cdf(D.Binomial(N, μ0[i]), s[i])) for i in 1:n]
    lo1 = [D.quantile(D.Normal(), s[i] == 0 ? 0.0 : D.cdf(D.Binomial(N, μc[i]), s[i] - 1))
           for i in 1:n]
    hi1 = [D.quantile(D.Normal(), D.cdf(D.Binomial(N, μc[i]), s[i])) for i in 1:n]
    inside_pop = count(i -> lo0[i] - 1e-8 <= q[i] <= hi0[i] + 1e-8, 1:n)
    inside_cond = count(i -> lo1[i] - 1e-8 <= q[i] <= hi1[i] + 1e-8, 1:n)
    @test inside_cond > inside_pop
end