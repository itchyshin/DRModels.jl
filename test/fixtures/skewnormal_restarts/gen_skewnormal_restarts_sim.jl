# gen_skewnormal_restarts_sim.jl -- regenerates sim159.csv (n = 200).
#
#   julia --project=test test/fixtures/skewnormal_restarts/gen_skewnormal_restarts_sim.jl
#
# Design of the restart study: x ~ N(0, 1); mu = 1 + 0.5 x; sigma = exp(0.2 + 0.3 x);
# slant = a + 0.7 x with a = -0.5; moment form (mu = mean, sigma = SD, slant = alpha);
# Azzalini draw y = xi + omega (delta |u0| + sqrt(1 - delta^2) u1). StableRNG seed 9000 + 159.
using StableRNGs, Printf
rng = StableRNG(9000 + 159)
n = 200; a = -0.5
x = randn(rng, n); y = zeros(n)
for i in 1:n
    mu = 1.0 + 0.5 * x[i]; sigma = exp(0.2 + 0.3 * x[i]); alpha = a + 0.7 * x[i]
    delta = alpha / sqrt(1 + alpha^2)
    omega = sigma / sqrt(1 - 2delta^2 / pi)
    xi = mu - omega * delta * sqrt(2 / pi)
    u0 = abs(randn(rng)); u1 = randn(rng)
    y[i] = xi + omega * (delta * u0 + sqrt(1 - delta^2) * u1)
end
open(joinpath(@__DIR__, "sim159.csv"), "w") do io
    println(io, "y,x")
    for i in 1:n; @printf(io, "%.17g,%.17g\n", y[i], x[i]); end
end
