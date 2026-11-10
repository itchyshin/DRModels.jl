# Regenerates the committed fixtures in this directory (provenance only: the CSV/Newick files
# are the fixtures, so the tests do not depend on any sampler's output across Julia versions).
#   julia --project=<repo> test/fixtures/tweedie_structured_sd/generate.jl
# Five datasets, 40 groups x 8 rows. phylo_1..3: random ultrametric tree, latent effect with
# covariance sd_g^2 * C (C = tip correlation). relmat_1..2: AR(1)-type K with unequal kernel scales.
# y | b ~ Tweedie(mu = exp(b0 + b1 x + sd_g * u_g), sigma, power), z a group-level binary predictor.
using DRModels, StableRNGs, LinearAlgebra, Random
import Distributions

const OUT = @__DIR__
const G, M = 40, 8

function tweedie_draw(rng, mu, sigma, power)
    phi = sigma^2
    lambda = mu^(2 - power) / (phi * (2 - power))
    n = rand(rng, Distributions.Poisson(lambda))
    n == 0 && return 0.0
    return rand(rng, Distributions.Gamma(n * (2 - power) / (power - 1),
                                         phi * (power - 1) * mu^(power - 1)))
end

# Random ultrametric binary tree, root height 1. A node at height `h` has branch length
# `ph - h` to its parent; tips sit at height 0.
function newick(rng, tips, h, ph)
    if length(tips) == 1
        body = tips[1]
    else
        k = rand(rng, 1:(length(tips) - 1))
        left, right = tips[1:k], tips[(k + 1):end]
        child(t) = length(t) == 1 ? 0.0 : h * (0.35 + 0.6 * rand(rng))
        body = "(" * newick(rng, left, child(left), h) * "," *
               newick(rng, right, child(right), h) * ")"
    end
    return ph === nothing ? body : "$body:$(ph - h)"
end

function write_csv(path, header, rows)
    open(path, "w") do io
        println(io, join(header, ","))
        for r in rows
            println(io, join(r, ","))
        end
    end
end

function simulate(name, kind, seed; alpha, beta = [0.4, 0.3], sigma = 0.65, power = 1.5)
    rng = StableRNG(seed)
    names = ["s" * lpad(i, 2, '0') for i in 1:G]
    labels = shuffle(rng, names)                    # group label per latent position
    zg = shuffle(rng, [zeros(G ÷ 2); ones(G - G ÷ 2)])
    if kind == :phylo
        nwk = newick(rng, shuffle(rng, names), 1.0, nothing) * ";"
        write(joinpath(OUT, "$(name)_tree.nwk"), nwk * "\n")
        phy = DRModels.augmented_phy(nwk)
        C = Matrix(DRModels._phylo_correlation(phy))
        order = phy.leaf_names                       # latent position k is tip order[k]
        glabel = order
    else
        K = [0.45^abs(i - j) for i in 1:G, j in 1:G]
        ks = [isodd(i) ? 1.0 : 1.6 for i in 1:G]
        C = ks .* K .* ks'
        glabel = labels
        write_csv(joinpath(OUT, "$(name)_K.csv"), ["c$j" for j in 1:G],
                  [C[i, :] for i in 1:G])
    end
    sd_g = exp.(alpha[1] .+ alpha[2] .* zg)
    u = cholesky(Symmetric(C)).L * randn(rng, G)
    rows = Vector{Any}()
    for k in 1:G, _ in 1:M
        x = 2rand(rng) - 1
        mu = exp(beta[1] + beta[2] * x + sd_g[k] * u[k])
        push!(rows, (tweedie_draw(rng, mu, sigma, power), x, zg[k], glabel[k]))
    end
    write_csv(joinpath(OUT, "$(name).csv"), ["y", "x", "z", "g"], rows)
end

simulate("phylo_1", :phylo, 20261108; alpha = [-0.5, 0.5])
simulate("phylo_2", :phylo, 20261109; alpha = [-0.3, 0.4], beta = [0.5, -0.3], sigma = 0.8, power = 1.4)
simulate("phylo_3", :phylo, 20261110; alpha = [-0.7, 0.6], sigma = 0.55, power = 1.6)
simulate("relmat_1", :relmat, 20261111; alpha = [-0.6, 0.4])
simulate("relmat_2", :relmat, 20261112; alpha = [-0.4, 0.3], beta = [0.6, 0.25], sigma = 0.7, power = 1.45)
