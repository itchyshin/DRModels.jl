# Deterministic data for the sigma-random-effect parity cells (StableRNG: identical on every
# Julia version). One data set serves all cells; each family column is generated from the same
# design (x1, g, g2).
#   n = 320 rows, g = 40 groups of 8, g2 = 8 groups crossed with g (row i has g2 = mod(i, 8) + 1).
#   log sigma_i = -0.3 + 0.3 x1_i + a_g [+ b_g x1_i] [+ c_g2]   (SDs 0.4 / 0.3 / 0.3)
# Columns: y_g (Gaussian), y_gm (Gaussian with a mean intercept on g and sigma intercept on g2),
# y_ln, y_gam, y_nb, sb / flb (beta-binomial, 20 trials).
using StableRNGs, Distributions, Random

function sre_fixture(; seed = 2026111001)
    rng = StableRNG(seed)
    G = 40; m = 8; n = G * m
    g = repeat(1:G, inner = m)
    g2 = [mod(i, 8) + 1 for i in 1:n]
    x1 = randn(rng, n)
    a = 0.4 .* randn(rng, G)
    # correlated intercept + slope on log sigma: sd(a) = 0.4, sd(b) = 0.3, rho = 0.5
    z1 = randn(rng, G); z2 = randn(rng, G)
    a_cs = 0.4 .* z1
    b_cs = 0.3 .* (0.5 .* z1 .+ sqrt(0.75) .* z2)
    c = 0.3 .* randn(rng, 8)
    mu_g = 0.5 .* randn(rng, G)
    eta0 = -0.3 .+ 0.3 .* x1
    # Gaussian: three sigma structures share y_g only where the cell asks for them; each cell uses
    # its own response column so the generating structure matches the fitted structure.
    y_g = 0.5 .+ 0.6 .* x1 .+ exp.(eta0 .+ a_cs[g] .+ b_cs[g] .* x1 .+ c[g2]) .* randn(rng, n)
    y_gm = 0.5 .+ 0.6 .* x1 .+ mu_g[g] .+ exp.(eta0 .+ c[g2]) .* randn(rng, n)
    eta = eta0 .+ a[g]
    y_ln = exp.(0.5 .+ 0.4 .* x1 .+ exp.(eta) .* randn(rng, n))
    y_gam = [rand(rng, Distributions.Gamma(exp(-2eta[i]), exp(0.5 + 0.4 * x1[i]) * exp(2eta[i]))) for i in 1:n]
    y_nb = [begin
        k = exp(-2eta[i]); mu = exp(1.0 + 0.4 * x1[i])
        rand(rng, Distributions.Poisson(rand(rng, Distributions.Gamma(k, mu / k))))
    end for i in 1:n]
    sb = zeros(Int, n)
    for i in 1:n
        mu = 1 / (1 + exp(-(0.2 + 0.5 * x1[i]))); phi = exp(-2eta[i])
        p = rand(rng, Distributions.Beta(mu * phi, (1 - mu) * phi))
        sb[i] = rand(rng, Distributions.Binomial(20, p))
    end
    return (; y_g, y_gm, y_ln, y_gam, y_nb, sb, flb = 20 .- sb, x1, g, g2)
end

function sre_write_csv(path; kw...)
    d = sre_fixture(; kw...)
    open(path, "w") do io
        println(io, join(string.(keys(d)), ","))
        for i in 1:length(d.x1)
            println(io, join([string(v[i]) for v in values(d)], ","))
        end
    end
end
