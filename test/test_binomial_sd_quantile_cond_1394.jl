# Refs #1394: quantile residuals on Binomial `sd(g) ~ …` use Laplace modes when stored.
using DRModels
using Test, Random
import Distributions as D

@testset "Binomial sd regression quantile uses conditional modes (#1394)" begin
    z(i, a, b, c, d) =
        sqrt(3) * (mod(i * a, 1.0) + mod(i * b, 1.0) + mod(i * c, 1.0) + mod(i * d, 1.0) - 2)
    G = 12; m = 6; n = G * m; N = 8
    g = repeat(["g" * lpad(j, 2, '0') for j in 1:G], inner = m)
    gx = repeat([isodd(j) ? -1.0 : 1.0 for j in 1:G], inner = m)
    x1 = [z(i, sqrt(2), sqrt(3), sqrt(5), sqrt(7)) for i in 1:n]
    b = [exp(-0.3 + 0.5 * (isodd(j) ? -1 : 1)) *
         z(j, sqrt(11), sqrt(13), sqrt(17), sqrt(19)) for j in 1:G]
    p = [1 / (1 + exp(-(0.2 + 0.7 * x1[i] + b[(i - 1) ÷ m + 1]))) for i in 1:n]
    s = [Float64(D.quantile(D.Binomial(N, p[i]), mod(i * sqrt(23), 1.0))) for i in 1:n]
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
