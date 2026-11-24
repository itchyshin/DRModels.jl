# Generates the refusal-gap-map grid: cells.tsv, data/*.csv (StableRNG, fixed seeds) and the
# structured-covariance fixtures (tree.nwk, coords.csv, A.csv, A_ids.csv).
# Usage: julia --project=<env with StableRNGs, Distributions, DelimitedFiles> gen_cells.jl <outdir>
using Random, LinearAlgebra, StableRNGs, DelimitedFiles
import Distributions

const OUT = length(ARGS) >= 1 ? ARGS[1] : @__DIR__
const DD = joinpath(OUT, "data")
mkpath(DD)

const G = 40            # groups (g1)
const M = 10            # rows per group
const G2 = 30           # levels of the crossed second factor
const FAMILIES = ["gaussian", "poisson", "nb2", "gamma", "binomial", "beta", "lognormal", "student", "tweedie"]
const KINDS = ["ordinary", "animal", "relmat", "phylo", "spatial", "crossed"]
const HAS_SIGMA = Dict(f => !(f in ("poisson", "binomial")) for f in FAMILIES)

# ---- structured fixtures ---------------------------------------------------------------------
ids = [string("s", lpad(string(j), 2, '0')) for j in 1:G]
function build_tree(rng, labels; height = 1.0)
    function build(labs, depth)
        if length(labs) == 1
            return labs[1] * ":" * string(height - depth), Dict((labs[1], labs[1]) => height)
        end
        local k, dn, nl, Cl, nr, Cr, C
        k = rand(rng, 1:(length(labs) - 1))
        dn = depth + (height - depth) * rand(rng) * 0.6
        (nl, Cl) = build(labs[1:k], dn); (nr, Cr) = build(labs[(k+1):end], dn)
        C = merge(Cl, Cr)
        for a in labs[1:k], b in labs[(k+1):end]
            C[(a, b)] = dn; C[(b, a)] = dn
        end
        return "(" * nl * "," * nr * "):" * string(dn - depth), C
    end
    k = rand(rng, 1:(length(labels) - 1))
    (nl, Cl) = build(labels[1:k], 0.0); (nr, Cr) = build(labels[(k+1):end], 0.0)
    C = merge(Cl, Cr)
    for a in labels[1:k], b in labels[(k+1):end]
        C[(a, b)] = 0.0; C[(b, a)] = 0.0
    end
    return "(" * nl * "," * nr * ");", [C[(labels[i], labels[j])] for i in 1:length(labels), j in 1:length(labels)]
end
rngT = StableRNG(20261108)
shuf = shuffle(rngT, ids)
nwk, Cm = build_tree(rngT, shuf)           # Cm rows follow the shuffled tip order `shuf`
idx = Dict(l => i for (i, l) in enumerate(shuf))
Pm = [idx[l] for l in ids]
Cphy = Cm[Pm, Pm]
Cphy = Cphy ./ sqrt.(diag(Cphy) * diag(Cphy)')
write(joinpath(DD, "tree.nwk"), nwk * "\n")
rngC = StableRNG(20261109)
cx = repeat(1:8; inner = 8)[1:G] .* 1.0 .+ 0.3 .* randn(rngC, G)
cy = repeat(1:8; outer = 8)[1:G] .* 1.0 .+ 0.3 .* randn(rngC, G)
open(joinpath(DD, "coords.csv"), "w") do io
    println(io, "id,easting,northing")
    for j in 1:G; println(io, ids[j], ",", cx[j], ",", cy[j]); end
end
Dm = [hypot(cx[i] - cx[j], cy[i] - cy[j]) for i in 1:G, j in 1:G]
rng_range = sort([Dm[i, j] for i in 1:G for j in (i+1):G])[div(G * (G - 1) ÷ 2, 2)]
Csp = exp.(-Dm ./ rng_range) + 1e-6 * I
Cani = Matrix(0.4 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :)))
open(joinpath(DD, "A.csv"), "w") do io
    for i in 1:G; println(io, join(repr.(Cani[i, :]), ",")); end
end
open(joinpath(DD, "A_ids.csv"), "w") do io; println(io, "id"); foreach(s -> println(io, s), ids); end
Kmap = Dict("phylo" => Cphy, "spatial" => Csp, "animal" => Cani, "relmat" => Cani)

# ---- data ----------------------------------------------------------------------------------
function logistic(x) 1 / (1 + exp(-x)) end
function rtweedie(rng, mu, phi, p)
    lam = mu^(2 - p) / (phi * (2 - p)); a = (2 - p) / (p - 1); sc = phi * (p - 1) * mu^(p - 1)
    n = rand(rng, Distributions.Poisson(lam)); n == 0 && return 0.0
    return sum(rand(rng, Distributions.Gamma(n * a, sc)))
end
function simulate(fam, kind, seed)
    rng = StableRNG(seed)
    n = G * M
    gi = repeat(1:G; inner = M)
    g2i = [mod(r - 1, G2) + 1 for r in 1:n]
    X = hcat(ones(n), randn(rng, n, 3))
    z = shuffle(rng, collect(range(-1, 1; length = G)))
    z2 = shuffle(rng, collect(range(-1, 1; length = G2)))
    s = [0.5, 0.3, 0.3, 0.3]
    Rm = [1 0.3 0.3 0.3; 0.3 1 0.3 0.3; 0.3 0.3 1 0.3; 0.3 0.3 0.3 1.0]
    L = kind in ("ordinary", "crossed") ? Matrix(1.0I, G, G) : cholesky(Symmetric(Kmap[kind])).L
    E = L * randn(rng, G, 4) * cholesky(Symmetric(Rm)).U      # G x 4, correlated coefficients
    B = E .* (s' .* exp.(0.3 .* z))
    u2 = kind == "crossed" ? 0.3 .* exp.(0.3 .* z2) .* randn(rng, G2) : zeros(G2)
    beta = [0.3, 0.4, 0.3, -0.2]
    eta = [dot(X[r, :], beta) + dot(X[r, :], B[gi[r], :]) + u2[g2i[r]] for r in 1:n]
    y = zeros(n)
    for r in 1:n
        e = eta[r]
        y[r] = if fam == "gaussian"; e + 0.5 * randn(rng)
        elseif fam == "student"; e + 0.5 * rand(rng, Distributions.TDist(5))
        elseif fam == "lognormal"; exp(e + 0.4 * randn(rng))
        elseif fam == "poisson"; Float64(rand(rng, Distributions.Poisson(exp(e))))
        elseif fam == "nb2"; kk = 3.0; Float64(rand(rng, Distributions.NegativeBinomial(kk, kk / (kk + exp(e)))))
        elseif fam == "gamma"; sh = 4.0; rand(rng, Distributions.Gamma(sh, exp(e) / sh))
        elseif fam == "binomial"; Float64(rand(rng) < logistic(e))
        elseif fam == "beta"; mu = logistic(e); ph = 8.0
            clamp(rand(rng, Distributions.Beta(mu * ph, (1 - mu) * ph)), 1e-4, 1 - 1e-4)
        elseif fam == "tweedie"; rtweedie(rng, exp(e), 1.0, 1.5)
        else error("family ", fam) end
    end
    gname = kind in ("ordinary", "crossed") ? [string("g", lpad(string(j), 2, '0')) for j in gi] : ids[gi]
    return (; y, x1 = X[:, 2], x2 = X[:, 3], x3 = X[:, 4], z = z[gi], g = gname,
              g2 = [string("h", lpad(string(j), 2, '0')) for j in g2i], z2 = z2[g2i])
end

# ---- lowermode check: seeds 1..24, same seed formula as wave 9 (fi, ki are the wave-9 indices)
const FI = Dict("nb2" => 3, "gamma" => 4); const KI = Dict("spatial" => 2, "phylo" => 4)
for fam in ("nb2", "gamma"), kind in ("phylo", "spatial"), seed in 1:24
    d = simulate(fam, kind, 202611080000 + 1000seed + 100FI[fam] + KI[kind])
    open(joinpath(DD, "data_$(fam)_$(kind)_s$(seed).csv"), "w") do io
        println(io, "y,x1,x2,x3,z,g,g2,z2")
        for r in eachindex(d.y)
            println(io, join((repr(d.y[r]), repr(d.x1[r]), repr(d.x2[r]), repr(d.x3[r]), repr(d.z[r]), d.g[r], d.g2[r], repr(d.z2[r])), ","))
        end
    end
end
println("done")
