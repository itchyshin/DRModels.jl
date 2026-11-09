# Self-contained Julia-version-independent generator (splitmix64 + Box-Muller): no Random stdlib.
mutable struct SM64; s::UInt64; end
function nextu(r::SM64)
    r.s += 0x9e3779b97f4a7c15
    z = r.s
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    z ⊻ (z >> 31)
end
unif(r) = (Float64(nextu(r) >> 11) + 0.5) / 9007199254740992.0
normal(r) = sqrt(-2 * log(unif(r))) * cos(2pi * unif(r))
"""G12-like Bernoulli data: 40 groups x 10 rows, x1..x3 ~ N(0,1), group covariate z equally spaced in [-1, 1],
random intercept SD exp(-0.8 + 0.3 z), random x1 slope SD exp(-0.9), fixed effects (0.5, 0.2, 0.2, -0.3)."""
function gen_binomial_indep(seed; G = 40, m = 10)
    r = SM64(UInt64(seed) * 0x2545F4914F6CDD1D + 12345)
    zlev = [-1 + 2 * (i - 1) / (G - 1) for i in 1:G]
    n = G * m
    g = String[]; x1 = Float64[]; x2 = Float64[]; x3 = Float64[]; z = Float64[]; y = Float64[]
    bi = [exp(-0.8 + 0.3 * zlev[i]) * normal(r) for i in 1:G]
    bs = [exp(-0.9) * normal(r) for i in 1:G]
    for i in 1:G, _ in 1:m
        a, b, c = normal(r), normal(r), normal(r)
        η = 0.5 + 0.2a + 0.2b - 0.3c + bi[i] + bs[i] * a
        push!(g, "g" * lpad(i, 2, '0')); push!(x1, a); push!(x2, b); push!(x3, c); push!(z, zlev[i])
        push!(y, unif(r) < 1 / (1 + exp(-η)) ? 1.0 : 0.0)
    end
    (y = y, x1 = x1, x2 = x2, x3 = x3, z = z, g = g)
end
