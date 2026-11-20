# #1394: `residuals(fit; type = :quantile)` on a Binomial `(1 | g)` fit with `sd(g) ~ gx`
# must judge each row against the fitted distribution at the Laplace/AGHQ modes of the group
# effects (as drmTMB and `fitted(fit)` do), not the population distribution (b = 0).
# Fixed on main by 48bfb0924 (the route now stores `scales[:cond_eta_re]`); this file pins it.
#
# Oracle: a randomised count residual lies in [qnorm(F(y-1)), qnorm(F(y))] of the fitted
# distribution. F uses Distributions.Binomial at the CONDITIONAL mean `fitted(fit)`.
# Wrong-answer variant: the same interval at the population mean `fit.means[:mu]` (the old
# behaviour) must NOT contain every draw, so the check can tell the two apart.
# Tolerance (fixed before the first run): 1e-8 on the interval edges.
using DRModels, Test, Random
import Distributions as D

@testset "#1394 Binomial sd(g) ~ gx quantile residuals are conditional" begin
    z(i, a, b, c, d) = sqrt(3) * (mod(i * a, 1.0) + mod(i * b, 1.0) + mod(i * c, 1.0) + mod(i * d, 1.0) - 2)
    G = 24; m = 10; n = G * m; N = 8
    g = repeat(["g" * lpad(j, 2, '0') for j in 1:G], inner = m)
    gx = repeat([isodd(j) ? -1.0 : 1.0 for j in 1:G], inner = m)
    x1 = [z(i, sqrt(2), sqrt(3), sqrt(5), sqrt(7)) for i in 1:n]
    b = [exp(-0.3 + 0.5 * (isodd(j) ? -1 : 1)) * z(j, sqrt(11), sqrt(13), sqrt(17), sqrt(19)) for j in 1:G]
    p = [1 / (1 + exp(-(0.2 + 0.7 * x1[i] + b[(i - 1) ÷ m + 1]))) for i in 1:n]
    s = [Float64(D.quantile(D.Binomial(N, p[i]), mod(i * sqrt(23), 1.0))) for i in 1:n]
    fl = N .- s
    fit = drm(bf(@formula(cbind(s, fl) ~ x1 + (1 | g)), @formula(sd(g) ~ gx)), DRModels.Binomial();
              data = (; s, fl, x1, g, gx))
    q = residuals(fit; type = :quantile, rng = Random.MersenneTwister(1))

    inside(mu) = count(1:n) do i
        lo = s[i] == 0 ? -Inf : D.quantile(D.Normal(), D.cdf(D.Binomial(N, mu[i]), s[i] - 1))
        hi = D.quantile(D.Normal(), D.cdf(D.Binomial(N, mu[i]), s[i]))
        lo - 1e-8 <= q[i] <= hi + 1e-8
    end
    @test haskey(fit.scales, :cond_eta_re)
    @test inside(fitted(fit)) == n
    @test inside(fit.means[:mu]) < n
end
