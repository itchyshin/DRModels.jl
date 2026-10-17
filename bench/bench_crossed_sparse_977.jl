# #977 — Gaussian crossed (1 | s) + (1 | d) wall-clock scaling after the sparse
# multi-RE kernel (src/gaussian_multi_re_sparse.jl).
#
# Workload: an InstEval-SHAPED synthetic design (2972 "students" × 1128
# "lecturers", a binary `service` covariate), rows drawn uniformly at random,
# y ~ 1 + service + (1 | s) + (1 | d), sigma ~ 1. The real lme4::InstEval data
# is GPL-licensed R data and is NOT used here; the shape (level counts, one
# crossed pair, one binary covariate) is what drives the factor fill.
#
# Usage:  julia --project=bench bench/bench_crossed_sparse_977.jl [n ...]
# Default n = 2000 20000 100000. Each n is fitted twice (the first n also pays
# JIT), and the second wall time is reported. Output: a table on stdout and
# bench/out/crossed_sparse_977.csv.
#
# Before #977 (dense QR of the (n + q) × q stacked design): n = 2000 took 331 s
# on the issue reporter's machine (M1 Ultra, loaded) and n = 20000 had not
# finished after 36 min. drmTMB on the same machine: 0.5 s / 4.9 s / 14.8 s
# (n = 2000 / 20000 / 73421). Those R numbers are the reporter's, not re-run
# here.
using DRModels, Random, Printf

function instEval_shaped(n; S = 2972, D = 1128, seed = 20261011)
    rng = MersenneTwister(seed)
    s = rand(rng, 1:S, n); d = rand(rng, 1:D, n)
    service = Float64.(rand(rng, n) .< 0.4)
    bs = 0.33 .* randn(rng, S); bd = 0.52 .* randn(rng, D)
    y = 3.2 .- 0.08 .* service .+ bs[s] .+ bd[d] .+ 1.18 .* randn(rng, n)
    return (y = y, s = string.(s), d = string.(d), service = service)
end

fitone(data) = drm(bf(@formula(y ~ 1 + service + (1 | s) + (1 | d)), @formula(sigma ~ 1)),
                   Gaussian(); data = data)

ns = isempty(ARGS) ? [2000, 20000, 100000] : parse.(Int, ARGS)
rows = String[]
println("n,q,seconds,loglik,converged,threads")
fitone(instEval_shaped(500))                       # JIT warm-up
for n in ns
    data = instEval_shaped(n)
    q = length(unique(data.s)) + length(unique(data.d))
    GC.gc()
    t = @elapsed (f = fitone(data))
    line = @sprintf("%d,%d,%.2f,%.6f,%s,%d", n, q, t, loglik(f), is_converged(f),
                    Threads.nthreads())
    println(line); flush(stdout)
    push!(rows, line)
end
mkpath(joinpath(@__DIR__, "out"))
open(joinpath(@__DIR__, "out", "crossed_sparse_977.csv"), "w") do io
    println(io, "n,q,seconds,loglik,converged,threads")
    foreach(r -> println(io, r), rows)
end
