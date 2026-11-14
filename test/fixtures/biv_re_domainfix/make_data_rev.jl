# Writes r<seed>.csv: the three simulated data sets (seeds 7080, 7083 and 7084 of the review sweep) that guard the
# value a rejected line-search trial returns. Not run by the tests: the CSV files are the stored truth.
# The stream reproduces them on Julia 1.13.3; Julia 1.10 draws different numbers from the same seeds.
# Usage: julia make_data_rev.jl <outdir> [seed ...]
# 3 to 40 groups, balanced or not, group SD from {0.02, ..., 1.0}, group correlation from {-0.95, ..., 0.95},
# residual correlation from {-0.7, ..., 0.6}.
using Random, LinearAlgebra
function params(s)
    rng = MersenneTwister(910000 + s)
    G = rand(rng, [3, 5, 8, 12, 20, 40]); m = rand(rng, [3, 5, 8, 15]) # mean rows per group
    sdg = rand(rng, [0.02, 0.1, 0.3, 0.6, 1.0], 2)
    ru = rand(rng, [-0.95, -0.6, 0.0, 0.3, 0.7, 0.95]); rr = rand(rng, [-0.7, -0.2, 0.3, 0.6])
    s1 = rand(rng, [0.4, 0.8, 1.2]); s2 = rand(rng, [0.5, 1.0])
    bal = rand(rng) < 0.4
    (; G, m, sdg, ru, rr, s1, s2, bal)
end
function sim(s)
    p = params(s); rng = MersenneTwister(920000 + s)
    G = p.G
    n = max(G + 2, p.G * p.m)
    g = p.bal ? [1 + mod(i - 1, G) for i in 1:n] : vcat(1:G, [min(G, 1 + floor(Int, G * rand(rng)^1.7)) for _ in 1:(n - G)])
    sd1, sd2 = p.sdg
    L = cholesky([sd1^2 p.ru*sd1*sd2; p.ru*sd1*sd2 sd2^2]).L
    U = L * randn(rng, 2, G)
    x1 = randn(rng, n); x2 = randn(rng, n); e1 = randn(rng, n); e2 = randn(rng, n)
    y1 = 0.2 .+ 0.6 .* x1 .+ U[1, g] .+ p.s1 .* e1
    y2 = -0.1 .+ 0.3 .* x2 .+ U[2, g] .+ p.s2 .* (p.rr .* e1 .+ sqrt(1 - p.rr^2) .* e2)
    (; y1, y2, x1, x2, g = ["h$(lpad(k, 2, '0'))" for k in g])
end
out = ARGS[1]; mkpath(out)
for s in (length(ARGS) > 1 ? parse.(Int, ARGS[2:end]) : [7080, 7083, 7084])
    d = sim(s)
    open(joinpath(out, "r$(s).csv"), "w") do io
        println(io, "id,g,x1,x2,y1,y2")
        for i in eachindex(d.y1); println(io, i, ",", d.g[i], ",", d.x1[i], ",", d.x2[i], ",", d.y1[i], ",", d.y2[i]); end
    end
end
