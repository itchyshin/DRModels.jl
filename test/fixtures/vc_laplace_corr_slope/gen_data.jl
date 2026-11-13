# Generates the committed datasets of test_vc_laplace_corr_slope.jl (run once; the CSVs are the fixture):
#   julia --project=test gen_data.jl      (from this directory)
# Correlated random intercept and slope `(1 + x | g)`, 40 groups of 10 rows, SD 0.5 and 0.6, correlation -0.3.
#   data_binomial_s{1..4}.csv   20 trials per row:  s, fl, x, g
#   data_poisson_s{1..4}.csv    counts:              y, x, g
#   data_nb2_s{1..4}.csv        counts, size 20:      y, x, g
#   data_binomial_q3_s{1..2}.csv  intercept and two slopes (1 + x + x2 | g): s, fl, x, x2, g
using Random, StableRNGs
import Distributions as D

const G, M = 40, 10
const N = G * M
lg(e) = 1 / (1 + exp(-e))

function draw(fam::Symbol, seed::Integer; q3::Bool = false)
    rng = StableRNG(20261113 + 100 * seed + (q3 ? 7 : 0) + findfirst(==(fam), (:binomial, :poisson, :nb2)))
    gi = [(i - 1) % G + 1 for i in 1:N]
    x = round.(2 .* rand(rng, N) .- 1; digits = 6)
    x2 = q3 ? round.(randn(rng, N); digits = 6) : zeros(N)
    z1, z2, z3 = randn(rng, G), randn(rng, G), randn(rng, G)
    a = 0.5 .* z1
    b = 0.6 .* (-0.3 .* z1 .+ sqrt(1 - 0.3^2) .* z2)
    c = 0.4 .* (0.2 .* z1 .- 0.1 .* z2 .+ sqrt(1 - 0.05) .* z3)
    eta = [0.2 + 0.4x[i] + 0.2x2[i] + a[gi[i]] + b[gi[i]] * x[i] + (q3 ? c[gi[i]] * x2[i] : 0.0) for i in 1:N]
    g = ["g" * lpad(k, 2, '0') for k in gi]
    if fam === :binomial
        s = [rand(rng, D.Binomial(20, lg(e))) for e in eta]
        return q3 ? (; s, fl = 20 .- s, x, x2, g) : (; s, fl = 20 .- s, x, g)
    elseif fam === :poisson
        return (; y = [rand(rng, D.Poisson(exp(0.7 * e))) for e in eta], x, g)
    else
        return (; y = [rand(rng, D.Poisson(rand(rng, D.Gamma(20.0, exp(0.7 * e) / 20.0)))) for e in eta], x, g)
    end
end

function write_csv(name, d)
    open(name, "w") do io
        println(io, join(keys(d), ","))
        for i in 1:N
            println(io, join([string(v[i]) for v in values(d)], ","))
        end
    end
end

for fam in (:binomial, :poisson, :nb2), seed in 1:4
    write_csv("data_$(fam)_s$(seed).csv", draw(fam, seed))
end
for seed in 1:2
    write_csv("data_binomial_q3_s$(seed).csv", draw(:binomial, seed; q3 = true))
end
