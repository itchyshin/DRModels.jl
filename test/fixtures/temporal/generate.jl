# Generator for the temporal AR1 / OU fixtures (DRModels.jl, D-310).
#
#   julia --project=test test/fixtures/temporal/generate.jl
#
# Writes `ar1_gapped.csv` and `ou_irregular.csv` next to this file. Both are
# simulated here (no drmTMB code or output is used), deterministically from
# StableRNGs, so they can be regenerated bit-for-bit. They are small enough to
# send to drmTMB for the later parity cells (drmTMB#1302):
#
#   drmTMB(bf(y ~ x + temporal(1 | id, time = occ, structure = "ar1"), sigma ~ 1),
#          data = read.csv("ar1_gapped.csv"), family = gaussian(), REML = FALSE)
#   drmTMB(bf(y ~ x + (1 | id) + temporal(1 | id, time = elapsed, structure = "ou"),
#             sigma ~ 1),
#          data = read.csv("ou_irregular.csv"), family = gaussian(), REML = FALSE)
#
# Julia spellings: `temporal(1 | id, occ, ar1)` / `temporal(1 | id, elapsed, ou)`.
using StableRNGs, LinearAlgebra, DelimitedFiles, Printf

# Dense draw of one stationary chain with correlation `corr(|t_i - t_j|)`.
function _chain(rng, t, corr, sd)
    R = [corr(abs(a - b)) for a in t, b in t]
    return sd .* (cholesky(Symmetric(R)).L * randn(rng, length(t)))
end

function write_fixture(path, header, cols)
    open(path, "w") do io
        println(io, join(header, ","))
        for i in eachindex(cols[1])
            println(io, join((c[i] isa AbstractFloat ? @sprintf("%.10g", c[i]) : string(c[i]) for c in cols), ","))
        end
    end
end

# AR1: 40 series, 5-9 integer occasions drawn from 0:14 (real gaps kept),
# rows written in shuffled order. Truth: beta = (1.0, 0.5), phi = 0.6,
# sd_temporal = 0.8, sigma = 0.5.
function make_ar1(dir)
    rng = StableRNG(20261001)
    id = String[]; occ = Int[]; x = Float64[]; y = Float64[]
    for s in 1:40
        k = rand(rng, 5:9)
        t = sort(collect(0:14)[sortperm(rand(rng, 15))[1:k]])
        xs = randn(rng, k)
        a = _chain(rng, t, d -> 0.6^d, 0.8)
        append!(id, fill(@sprintf("s%02d", s), k)); append!(occ, t); append!(x, xs)
        append!(y, 1.0 .+ 0.5 .* xs .+ a .+ 0.5 .* randn(rng, k))
    end
    p = sortperm(rand(rng, length(y)))
    write_fixture(joinpath(dir, "ar1_gapped.csv"), ["y", "x", "id", "occ"],
                  (y[p], x[p], id[p], occ[p]))
end

# OU: 24 series, 4-7 irregular elapsed times in (0, 10), a stable series
# intercept, rows shuffled. Truth: beta = (0.8, 0.35), decay = 0.45,
# sd_temporal = 0.65, sd_id = 0.45, sigma = 0.4.
function make_ou(dir)
    rng = StableRNG(20261002)
    id = String[]; el = Float64[]; x = Float64[]; y = Float64[]
    for s in 1:24
        k = rand(rng, 4:7)
        t = sort(round.(10 .* rand(rng, k); digits = 3))
        xs = randn(rng, k)
        a = _chain(rng, t, d -> exp(-0.45 * d), 0.65)
        b = 0.45 * randn(rng)
        append!(id, fill(@sprintf("site%02d", s), k)); append!(el, t); append!(x, xs)
        append!(y, 0.8 .+ 0.35 .* xs .+ b .+ a .+ 0.4 .* randn(rng, k))
    end
    p = sortperm(rand(rng, length(y)))
    write_fixture(joinpath(dir, "ou_irregular.csv"), ["y", "x", "id", "elapsed"],
                  (y[p], x[p], id[p], el[p]))
end

# Paired phylo + OU (wave 2, D-311): a random ultrametric binary tree over 24
# species, rescaled to root-to-tip height 1 (drmTMB requires an ultrametric
# tree for its correlation scale), and 3-6 irregular elapsed times in (0, 10)
# per species, rows shuffled. Truth: beta = (0.25, 0.45), sd_phylo = 0.8
# (tip correlation scale), sd_temporal = 0.75, decay = 0.45, sigma = 0.4.
# Writes `phylo_ou_species.csv` (y, x, species, elapsed) and
# `phylo_ou_species.newick`.
function make_phylo_ou(dir; seed = 20261003)
    rng = StableRNG(seed)
    m = 24
    names = [@sprintf("sp%02d", i) for i in 1:m]
    # clusters: (newick-builder, height, members); branch lengths fixed at the end
    edges = Tuple{Int,Int,Float64}[]       # (parent, child, length) on node ids
    height = zeros(m)                      # node heights, leaves 0
    cl = collect(1:m)
    next = m
    h = 0.0
    while length(cl) > 1
        i, j = sortperm(rand(rng, length(cl)))[1:2]
        a, b = cl[i], cl[j]
        h += 0.2 + rand(rng)
        next += 1
        push!(height, h)
        push!(edges, (next, a, h - height[a]), (next, b, h - height[b]))
        deleteat!(cl, sort([i, j]))
        push!(cl, next)
    end
    root = cl[1]
    H = height[root]
    kids = Dict{Int,Vector{Tuple{Int,Float64}}}()
    for (p_, c, l) in edges
        push!(get!(kids, p_, Tuple{Int,Float64}[]), (c, l / H))
    end
    nwk(v) = v <= m ? names[v] :
        "(" * join([nwk(c) * ":" * @sprintf("%.17g", l) for (c, l) in kids[v]], ",") * ")"
    open(joinpath(dir, "phylo_ou_species.newick"), "w") do io
        println(io, nwk(root), ";")
    end
    # tip correlation = shared path length (height 1 after rescaling)
    C = zeros(m, m)
    function addpath!(v, members)
        for (c, l) in get(kids, v, Tuple{Int,Float64}[])
            mem = Int[]
            collect_leaves!(c, mem)
            for p_ in mem, q in mem
                C[p_, q] += l
            end
            addpath!(c, members)
        end
    end
    collect_leaves!(v, out) = v <= m ? push!(out, v) :
        foreach(c -> collect_leaves!(c[1], out), kids[v])
    addpath!(root, nothing)
    a = 0.8 .* (cholesky(Symmetric(C)).L * randn(rng, m))
    sp = String[]; el = Float64[]; x = Float64[]; y = Float64[]
    for s in 1:m
        t = unique(sort(round.(10 .* rand(rng, rand(rng, 3:6)); digits = 3)))   # distinct times
        k = length(t)
        xs = randn(rng, k)
        b = _chain(rng, t, d -> exp(-0.45 * d), 0.75)
        append!(sp, fill(names[s], k)); append!(el, t); append!(x, xs)
        append!(y, 0.25 .+ 0.45 .* xs .+ a[s] .+ b .+ 0.4 .* randn(rng, k))
    end
    p = sortperm(rand(rng, length(y)))
    write_fixture(joinpath(dir, "phylo_ou_species.csv"), ["y", "x", "species", "elapsed"],
                  (y[p], x[p], sp[p], el[p]))
end

# Homogeneous Toeplitz (wave 2, D-311): complete equally spaced panels, the
# within-site covariance sigma^2 R with R Toeplitz, built from partial
# autocorrelations by the Durbin-Levinson recursion (written out here, not
# taken from the package). Writes
#   `homtoep_panel6.csv`: 40 sites x occasions 0:5, PACs (0.6, -0.3, 0.35, 0,
#     0.15) (non-exponential lags), sigma = 0.9, beta = (0.2, 0.4);
#   `homtoep_neg4.csv`: 30 sites x occasions 2, 4, 6, 8, PACs (-0.45, 0.2,
#     0.1) (negative first lag), sigma = 1.2, beta = (-0.3, 0.5).
# Columns `y, x, id, occ`; rows shuffled.
function _toeplitz_from_pacf(pac)
    K = length(pac) + 1
    rho = zeros(K); rho[1] = 1.0
    phi = Float64[]; v = 1.0
    for m in 1:(K-1)
        pred = m == 1 ? 0.0 : sum(phi[j] * rho[m-j+1] for j in 1:(m-1))
        rho[m+1] = pred + pac[m] * v
        phi = [[phi[j] - pac[m] * phi[m-j] for j in 1:(m-1)]; pac[m]]
        v *= 1 - pac[m]^2
    end
    return [rho[abs(i - j) + 1] for i in 1:K, j in 1:K]
end

function _homtoep_panel(rng, path, nsite, occ, pac, sigma, beta)
    L = cholesky(Symmetric(_toeplitz_from_pacf(pac))).L
    K = length(occ)
    id = String[]; t = Int[]; x = Float64[]; y = Float64[]
    for s in 1:nsite
        xs = randn(rng, K)
        append!(id, fill(@sprintf("site%02d", s), K)); append!(t, occ); append!(x, xs)
        append!(y, beta[1] .+ beta[2] .* xs .+ sigma .* (L * randn(rng, K)))
    end
    p = sortperm(rand(rng, length(y)))
    write_fixture(path, ["y", "x", "id", "occ"], (y[p], x[p], id[p], t[p]))
end

function make_homtoep(dir)
    _homtoep_panel(StableRNG(20261004), joinpath(dir, "homtoep_panel6.csv"), 40, collect(0:5),
                   [0.6, -0.3, 0.35, 0.0, 0.15], 0.9, (0.2, 0.4))
    _homtoep_panel(StableRNG(20261005), joinpath(dir, "homtoep_neg4.csv"), 30, [2, 4, 6, 8],
                   [-0.45, 0.2, 0.1], 1.2, (-0.3, 0.5))
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_ar1(@__DIR__)
    make_ou(@__DIR__)
    make_phylo_ou(@__DIR__)
    make_homtoep(@__DIR__)
end
