# Shared DGM for test/test_twin_gap_747.jl (#746 / #747): Gaussian mean random
# intercept `(1 | g)` with a residual `sigma ~ x` (or `sigma ~ x + x2`) model,
# shaped like the Wave8 ADEMP cells 110 (balanced G = 12 x n_g = 8, sigma ~ x,
# truth (mu0, mu1, sd_g, s0, s1) = (0.2, 0.45, 0.5, -0.4, 0.25)) and 104
# (unbalanced G = 10, sigma ~ x + x2). StableRNGs so the DATA is identical on
# every Julia version and platform.
using StableRNGs, StatsModels

function _tg747_draw(seed::Integer, kind::Symbol)
    rng = StableRNG(seed)
    if kind === :bal
        G = 12; g = repeat(1:G, inner = 8)
    else
        G = 10; sizes = [3, 4, 5, 6, 8, 9, 10, 12, 14, 16]
        g = reduce(vcat, [fill(k, sizes[k]) for k in 1:G])
    end
    n = length(g)
    x = randn(rng, n); x2 = x .^ 2
    sdg = kind === :bal ? 0.5 : 0.4
    u = sdg .* randn(rng, G)
    ls = kind === :bal ? (-0.4 .+ 0.25 .* x) : (-0.3 .+ 0.2 .* x .+ 0.05 .* x2)
    y = 0.2 .+ 0.45 .* x .+ u[g] .+ exp.(ls) .* randn(rng, n)
    return (; y, x, x2, g)
end

_tg747_formula(kind::Symbol) = kind === :bal ?
    bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ x)) :
    bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ x + x2))
