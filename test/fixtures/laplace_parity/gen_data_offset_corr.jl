# Generates data_poisson_offset_corr.csv (run once; the committed CSV is the fixture). Poisson counts with a mean offset
# and a correlated random intercept and slope: log mu = 0.2 + 0.3 x + logb + u0_g + x u1_g.
using Random, DelimitedFiles
rng = MersenneTwister(20261110)
n, G = 400, 40
function rpois(rng, l); L = exp(-l); k = 0; p = 1.0; while true; p *= rand(rng); p <= L && return k; k += 1; end; end
g = repeat(1:G, inner = n ÷ G); x = randn(rng, n); logb = 0.3 .* randn(rng, n)
z1 = randn(rng, G); z2 = randn(rng, G); u0 = 0.5 .* z1; u1 = 0.3 .* (0.5 .* z1 .+ sqrt(0.75) .* z2)
c = [rpois(rng, exp(0.2 + 0.3x[i] + logb[i] + u0[g[i]] + x[i] * u1[g[i]])) for i in 1:n]
open("data_poisson_offset_corr.csv", "w") do io
    println(io, "c,x,logb,g")
    for i in 1:n; println(io, c[i], ",", x[i], ",", logb[i], ",g", lpad(g[i], 2, '0')); end
end
