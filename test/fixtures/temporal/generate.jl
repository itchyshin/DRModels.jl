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

if abspath(PROGRAM_FILE) == @__FILE__
    make_ar1(@__DIR__)
    make_ou(@__DIR__)
end
