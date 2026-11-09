# Julia-version-independent data generator (splitmix64 + Box-Muller + Fisher-Yates) for
# test_binomial_spatial_sd_regression.jl. The Random stdlib is deliberately not used: `shuffle!`,
# `randn` and `rand(Bernoulli)` draw different values from the same MersenneTwister seed on Julia
# 1.10 and 1.12, so the old fixture was a different dataset on each version.
using SpecialFunctions: erfc

mutable struct SM64Spatial; s::UInt64; end
function _nextu(r::SM64Spatial)
    r.s += 0x9e3779b97f4a7c15
    z = r.s
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    z ⊻ (z >> 31)
end
_unif(r::SM64Spatial) = (Float64(_nextu(r) >> 11) + 0.5) / 9007199254740992.0
_normal(r::SM64Spatial) = sqrt(-2 * log(_unif(r))) * cos(2pi * _unif(r))
_inv_link(link::Symbol, eta) = link === :logit ? 1 / (1 + exp(-eta)) :
    link === :probit ? 0.5 * erfc(-eta / sqrt(2)) : 1 - exp(-exp(eta))

"""
    gen_binomial_spatial_sd(link, seed; Lchol, alpha, slope, n_each = 32)

One Bernoulli draw from the spatial SD-regression design: `G = size(Lchol, 1)` sites with
half of them at `z = -1` and half at `z = +1` (positions shuffled by `seed`), random-field SD
`exp(alpha[1] + alpha[2] z)` times `Lchol * N(0, I)`, entering the linear predictor as an
intercept (`slope = false`) or multiplied by the covariate `x` (`slope = true`).
Returns `(y, x, z, z_group, group)`.
"""
function gen_binomial_spatial_sd(link::Symbol, seed::Integer; Lchol, alpha, slope::Bool,
                                 n_each::Int = 32)
    G = size(Lchol, 1)
    r = SM64Spatial(UInt64(seed) * 0x2545F4914F6CDD1D + 12345)
    z_group = vcat(fill(-1.0, G ÷ 2), fill(1.0, G ÷ 2))
    for i in G:-1:2                      # Fisher-Yates
        j = 1 + Int(_nextu(r) % UInt64(i))
        z_group[i], z_group[j] = z_group[j], z_group[i]
    end
    group_index = repeat(1:G; inner = n_each)
    n = G * n_each
    x = [_normal(r) for _ in 1:n]
    u = Lchol * [_normal(r) for _ in 1:G]
    b = exp.(alpha[1] .+ alpha[2] .* z_group) .* u
    eta = slope ? -0.2 .+ 0.25 .* x .+ x .* b[group_index] : -0.2 .+ 0.25 .* x .+ b[group_index]
    y = [_unif(r) < _inv_link(link, eta[i]) ? 1.0 : 0.0 for i in 1:n]
    (y = y, x = x, z = repeat(z_group; inner = n_each), z_group = z_group,
     group = ["s$j" for j in group_index])
end
