# Deterministic fixtures for the Gaussian structured-sigma REML test (StableRNG: identical on
# every Julia version). Five cells: animal, relmat, phylo, spatial, and relmat with a
# covariate-carrying fixed sigma (`sigma ~ x + relmat(1 | id)`).
#   y_i = 0.2 + 0.5 x_i + exp(log 0.5 [+ 0.3 x_i] + s_g u_g) e_i,   u ~ N(0, K),
#   s_g = exp(alpha0 + alpha1 z_g),  alpha = (log 0.45, 0.5)
using StableRNGs, LinearAlgebra, Statistics

function _gsr_tree(ids; bits = 5)
    G = length(ids)
    nw(ix, root) = length(ix) == 1 ? "$(ids[ix[1]]):1.5" :
        "(" * nw(ix[1:end÷2], false) * "," * nw(ix[end÷2+1:end], false) * ")" * (root ? ";" : ":0.1")
    height = 1.5 + 0.1 * (bits - 1)
    b(i) = reverse(digits(i - 1; base = 2, pad = bits))
    C = [i == j ? 1.0 : (findfirst(b(i) .!= b(j)) - 1) * 0.1 / height for i in 1:G, j in 1:G]
    return nw(collect(1:G), true), C
end

function gsr_fixture(cell::Symbol; n_g = 32, n_each = 10)
    seed = Dict(:animal => 2026110801, :relmat => 2026110802, :phylo => 2026110803,
                :spatial => 2026110804, :relmat_sx => 2026110805)[cell]
    rng = StableRNG(seed)
    ids = [string("s", lpad(string(j), 2, '0')) for j in 1:n_g]
    tree = nothing; coords = nothing
    K = if cell === :phylo
        tree, C = _gsr_tree(ids)
        C
    elseif cell === :spatial
        coords = hcat(repeat(1:4; inner = 8) .* 1.0, repeat(1:8; outer = 4) .* 1.0)
        D = [norm(coords[i, :] .- coords[j, :]) for i in 1:n_g, j in 1:n_g]
        r = median([D[i, j] for i in 1:n_g for j in (i+1):n_g if D[i, j] > 0])
        exp.(-D ./ r) + 1e-6 * I
    else
        rho = cell === :animal ? 0.3 : 0.5
        dg = collect(range(0.7, 1.4; length = n_g))
        [rho^abs(i - j) * sqrt(dg[i] * dg[j]) for i in 1:n_g, j in 1:n_g]
    end
    z = collect(range(-1, 1; length = n_g))[sortperm(rand(rng, n_g))]
    u = cholesky(Symmetric(K)).L * randn(rng, n_g)
    alpha = [log(0.45), 0.5]
    gi = repeat(1:n_g; inner = n_each)
    x = randn(rng, length(gi))
    sc = exp.(alpha[1] .+ alpha[2] .* z)
    sx = cell === :relmat_sx ? 0.3 : 0.0
    y = 0.2 .+ 0.5 .* x .+ exp.(log(0.5) .+ sx .* x .+ sc[gi] .* u[gi]) .* randn(rng, length(gi))
    data = (; y, x, id = ids[gi], z = z[gi])
    return (; data, K, ids, tree, coords, cell, alpha)
end
