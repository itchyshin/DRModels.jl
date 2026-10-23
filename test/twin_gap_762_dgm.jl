# Shared DGM for test/test_twin_gap_762.jl (#762 / #707): a sparrow-like
# repeated-measures design (unbalanced, 1-5 records on each of G birds) simulated
# under H0: NO random-slope variance -- y = 75 + 0.5 x + u_g + e with
# u_g ~ N(0, 1.5^2), e ~ N(0, 1.2^2), x ~ N(27, 2^2) (an uncentred body mass, as in
# #762) -- and then fitted with the correlated `(1 + x | g)` model. StableRNGs so
# the DATA is identical on every Julia version and platform.
using StableRNGs, Statistics

function _tg762_draw(seed::Integer; centred::Bool = false, G::Int = 60)
    rng = StableRNG(seed)
    sizes = [1 + floor(Int, 5 * rand(rng)) for _ in 1:G]
    g = reduce(vcat, [fill(k, sizes[k]) for k in 1:G])
    n = length(g)
    x = 27 .+ 2 .* randn(rng, n)
    u = 1.5 .* randn(rng, G)
    y = 75 .+ 0.5 .* x .+ u[g] .+ 1.2 .* randn(rng, n)
    centred && (x = x .- mean(x))
    return (; y, x, g)
end
