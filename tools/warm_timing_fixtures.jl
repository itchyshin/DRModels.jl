#!/usr/bin/env julia
# tools/warm_timing_fixtures.jl — S12 (#563), root gate G5.
#
# Generates every fixture in the warm-workflow registry
# (docs/dev-log/evidence/julia-r-parity/warm-workflow-registry.md) ONCE,
# deterministically, and writes it to plain CSV (+ Newick tree files for the
# phylogenetic workflows) under an output directory. tools/warm_timing.jl and
# tools/warm_timing.R both READ these same files -- neither engine generates
# its own data -- so "matched" means bit-identical inputs, not
# distributionally-similar ones drawn independently in R and Julia.
#
# Usage:
#   julia --project=. tools/warm_timing_fixtures.jl --out docs/dev-log/evidence/julia-r-parity/warm-fixtures
#
# Uses ONLY main-project deps (Random, LinearAlgebra -- both stdlib, already
# in Project.toml) plus manual text I/O -- no CSV.jl/DataFrames/StableRNGs,
# none of which are main-project dependencies (tools/*.jl runs under
# `julia --project=.`, not test/Project.toml).

using Random
using LinearAlgebra

# ---- CLI --------------------------------------------------------------------

function parse_args(argv)
    out = "docs/dev-log/evidence/julia-r-parity/warm-fixtures"
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--out"
            out = argv[i + 1]; i += 2
        else
            error("warm_timing_fixtures.jl: unknown arg $a")
        end
    end
    return out
end

# ---- balanced synthetic phylogeny -------------------------------------------
# NOT ape::rcoal (that needs R). A depth-d balanced binary tree, every branch
# length 1/d, gives root-to-tip path length 1 for every leaf (Ω unit tip
# variance, the same convention bench_fit_h2h.R uses after its own
# `edge.length / h` rescale). Covariance between two leaves = (shared root-to-
# MRCA path length)/d. The Newick text below encodes exactly this topology and
# these edge lengths, so drmTMB's ape::vcv.phylo(read.tree(...)) and DRModels.jl's
# augmented_phy(...) both recover the SAME implied covariance used to simulate
# the response here -- that is what "identical data" requires, not that the
# tree LOOK like a real phylogeny.

function balanced_tree(depth::Int; prefix::String = "sp")
    n = 2^depth
    names = ["$(prefix)$(i)" for i in 1:n]

    function node(lo::Int, hi::Int, d::Int)
        if hi == lo
            return "$(names[lo]):$(1/depth)"
        end
        mid = lo + div(hi - lo + 1, 2) - 1
        l = node(lo, mid, d - 1)
        r = node(mid + 1, hi, d - 1)
        inner = "($l,$r)"
        return d == depth ? inner * ";" : inner * ":$(1/depth)"
    end
    newick = node(1, n, depth)

    shared = zeros(Int, n, n)
    function fill_shared!(lo::Int, hi::Int, d::Int)
        hi == lo && return
        mid = lo + div(hi - lo + 1, 2) - 1
        for i in lo:mid, j in (mid + 1):hi
            shared[i, j] = depth - d
            shared[j, i] = depth - d
        end
        fill_shared!(lo, mid, d - 1)
        fill_shared!(mid + 1, hi, d - 1)
    end
    fill_shared!(1, n, depth)
    for i in 1:n
        shared[i, i] = depth
    end
    Sigma = shared ./ depth   # unit tip variance

    return newick, names, Sigma
end

# ---- I/O helpers --------------------------------------------------------------

function write_csv(path::AbstractString, header::Vector{String}, cols::Tuple)
    n = length(cols[1])
    for c in cols
        length(c) == n || error("write_csv: column length mismatch for $path")
    end
    open(path, "w") do io
        println(io, join(header, ","))
        for i in 1:n
            println(io, join((string(c[i]) for c in cols), ","))
        end
    end
end

write_newick(path::AbstractString, newick::AbstractString) = open(io -> print(io, newick), path, "w")

# ---- workflow fixtures --------------------------------------------------------
# Seeds and sizes here are the source of truth: warm-workflow-registry.md
# documents them, warm_timing.jl / warm_timing.R must NOT redefine the data
# (they read the files this script writes).

function gen_gauss_mixed_phylo_mean(dir)
    Random.seed!(20260901_01)
    newick, sp_names, Sigma = balanced_tree(4)              # 16 tips
    n_each = 6
    n = length(sp_names) * n_each
    species = repeat(sp_names, inner = n_each)
    study = repeat(1:4, outer = cld(n, 4))[1:n]
    x = randn(n)
    u = cholesky(Symmetric(Sigma)).L * randn(length(sp_names)) .* 0.5     # phylo mean RE
    b_study = randn(4) .* 0.4
    eta = 0.3 .+ 0.5 .* x .+ u[indexin(species, sp_names)] .+ b_study[study]
    y = eta .+ 0.3 .* randn(n)
    write_csv(joinpath(dir, "gauss_mixed_phylo_mean.csv"), ["y", "x", "species", "study"],
              (y, x, species, study))
    write_newick(joinpath(dir, "gauss_mixed_phylo_mean.nwk"), newick)
end

function gen_gauss_lss_sd_group(dir)
    Random.seed!(20260901_02)
    G = 30; n_each = 20; n = G * n_each
    g = repeat(1:G, inner = n_each)
    x = randn(n); z = randn(n)
    zg = randn(G)                                            # GROUP-level covariate for sd(g) --
                                                                # sd() predictors must be constant
                                                                # within each level of the group
                                                                # (drm() refuses otherwise); broadcast
                                                                # to one column per observation below.
    sd_g = exp.(-0.4 .+ 0.4 .* zg)
    b = sd_g[g] .* randn(n)
    y = 0.4 .+ 0.6 .* x .+ b .+ exp.(-0.3 .+ 0.2 .* z) .* randn(n)
    write_csv(joinpath(dir, "gauss_lss_sd_group.csv"), ["y", "x", "z", "zg", "g"], (y, x, z, zg[g], g))
end

function gen_gauss_lss_sd_phylo(dir)
    Random.seed!(20260901_03)
    newick, sp_names, Sigma = balanced_tree(6)               # 64 tips
    n_each = 3
    n = length(sp_names) * n_each
    species = repeat(sp_names, inner = n_each)
    idx = indexin(species, sp_names)
    x = randn(n)
    xs = randn(length(sp_names))                             # SPECIES-level covariate for
                                                                # sd(species, phylogenetic) -- sd()
                                                                # predictors must be constant within
                                                                # each species (drm() refuses `x`,
                                                                # which varies within species, here).
    sd_phy = exp.(-0.4 .+ 0.35 .* xs)
    u = cholesky(Symmetric(Sigma)).L * randn(length(sp_names))
    a = sd_phy .* u
    y = 1.2 .+ 0.5 .* x .+ a[idx] .+ exp.(-1.0 .- 0.2 .* x) .* randn(n)
    write_csv(joinpath(dir, "gauss_lss_sd_phylo.csv"), ["y", "x", "xs", "species"],
              (y, x, xs[idx], species))
    write_newick(joinpath(dir, "gauss_lss_sd_phylo.nwk"), newick)
end

function gen_biv_q4_phylo(dir)
    Random.seed!(20260901_04)
    newick, sp_names, Sigma = balanced_tree(4; prefix = "sp")  # 16 tips (registry documents as p~12-16)
    n_each = 3
    n = length(sp_names) * n_each
    species = repeat(sp_names, inner = n_each)
    idx = indexin(species, sp_names)
    Sigma_a = [0.22 0.07; 0.07 0.18]
    C_phy = cholesky(Symmetric(Sigma)).L
    C_axis = cholesky(Symmetric(Sigma_a)).U
    U = C_phy * randn(length(sp_names), 2) * C_axis
    x = randn(n)
    e1 = 0.32 .* randn(n)
    e2 = 0.25 .* e1 .+ sqrt(1 - 0.25^2) .* 0.36 .* randn(n)
    y1 = 0.20 .+ 0.30 .* x .+ U[idx, 1] .+ e1
    y2 = -0.15 .+ 0.15 .* x .+ U[idx, 2] .+ e2
    write_csv(joinpath(dir, "biv_q4_phylo.csv"), ["species", "x", "y1", "y2"], (species, x, y1, y2))
    write_newick(joinpath(dir, "biv_q4_phylo.nwk"), newick)
end

function gen_bernoulli_mixed(dir)
    Random.seed!(20260901_05)
    G = 25; n_each = 20; n = G * n_each
    g = repeat(1:G, inner = n_each)
    b = 0.5 .* randn(G)
    x = randn(n)
    eta = 0.2 .+ 0.6 .* x .+ b[g]
    p = 1 ./ (1 .+ exp.(-eta))
    y = Int.(rand(n) .< p)
    write_csv(joinpath(dir, "bernoulli_mixed.csv"), ["y", "x", "g"], (y, x, g))
end

function gen_poisson_mixed(dir)
    Random.seed!(20260901_06)
    G = 25; n_each = 20; n = G * n_each
    g = repeat(1:G, inner = n_each)
    b = 0.4 .* randn(G)
    x = randn(n)
    eta = 0.3 .+ 0.4 .* x .+ b[g]
    lam = exp.(eta)
    y = [rand_pois(l) for l in lam]
    write_csv(joinpath(dir, "poisson_mixed.csv"), ["y", "x", "g"], (y, x, g))
end

# Dependency-free Poisson deviate (Knuth's algorithm; lambdas here are small
# enough -- < ~10 -- that this is fine and avoids a Distributions.jl coupling
# in a tool script that otherwise needs none of it).
function rand_pois(lambda::Real)
    L = exp(-lambda); k = 0; p = 1.0
    while true
        k += 1
        p *= rand()
        p <= L && return k - 1
    end
end

function gen_lognormal_locscale(dir)
    Random.seed!(20260901_07)
    n = 400
    x = randn(n); z = randn(n)
    y = exp.(0.4 .+ 0.5 .* x .+ exp.(-0.3 .+ 0.25 .* z) .* randn(n))
    write_csv(joinpath(dir, "lognormal_locscale.csv"), ["y", "x", "z"], (y, x, z))
end

function gen_meta_analysis(dir)
    Random.seed!(20260901_08)
    k = 300
    x = randn(k)
    tau = 0.4
    v = (0.2 .+ 0.6 .* rand(k)) .^ 2
    y = 0.2 .+ 0.5 .* x .+ tau .* randn(k) .+ sqrt.(v) .* randn(k)
    write_csv(joinpath(dir, "meta_analysis.csv"), ["y", "x", "v"], (y, x, v))
end

function gen_large_sparse_lss(dir)
    Random.seed!(20260901_09)
    newick, sp_names, Sigma = balanced_tree(11)              # 2048 tips
    n_each = 2
    n = length(sp_names) * n_each
    species = repeat(sp_names, inner = n_each)
    idx = indexin(species, sp_names)
    x = randn(n)
    xs = randn(length(sp_names))                             # SPECIES-level covariate for
                                                                # sd(species, phylogenetic), same
                                                                # constant-within-group requirement
                                                                # as gauss_lss_sd_phylo above.
    sd_phy = exp.(-0.4 .+ 0.3 .* xs)
    u = cholesky(Symmetric(Sigma)).L * randn(length(sp_names))
    a = sd_phy .* u
    y = 1.0 .+ 0.4 .* x .+ a[idx] .+ exp.(-1.1 .- 0.15 .* x) .* randn(n)
    write_csv(joinpath(dir, "large_sparse_lss_p2000.csv"), ["y", "x", "xs", "species"],
              (y, x, xs[idx], species))
    write_newick(joinpath(dir, "large_sparse_lss_p2000.nwk"), newick)
end

const GENERATORS = [
    ("gauss_mixed_phylo_mean", gen_gauss_mixed_phylo_mean),
    ("gauss_lss_sd_group", gen_gauss_lss_sd_group),
    ("gauss_lss_sd_phylo", gen_gauss_lss_sd_phylo),
    ("biv_q4_phylo (shared by biv_q4_phylo_ml and biv_q4_phylo_reml)", gen_biv_q4_phylo),
    ("bernoulli_mixed", gen_bernoulli_mixed),
    ("poisson_mixed", gen_poisson_mixed),
    ("lognormal_locscale", gen_lognormal_locscale),
    ("meta_analysis", gen_meta_analysis),
    ("large_sparse_lss_p2000", gen_large_sparse_lss),
]

function main()
    out = parse_args(ARGS)
    mkpath(out)
    for (label, f) in GENERATORS
        print("generating $label ... ")
        f(out)
        println("done")
    end
    println("wrote fixtures to $out")
end

main()
