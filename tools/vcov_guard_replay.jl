# Replay the covariance guard's decision on a census written by
# `tools/vcov_guard_census.jl`, under the shipped constants or candidate ones.
#
#   julia --project=. tools/vcov_guard_replay.jl census_*.tsv
#   DRM_REPLAY_RTOL=1e-8 DRM_REPLAY_FLAT=1e-5 julia --project=. tools/vcov_guard_replay.jl census_*.tsv
#
# For every recorded call it computes
#   legacy  the pre-#936 decision (eigenvalues of the UNSCALED matrix, relative
#           tolerance 3e-8)
#   new     `DRModels._vcov_guard_decision` -- the shipped decision code itself,
#           with DRM_REPLAY_RTOL / DRM_REPLAY_FLAT when set
# and prints the transition table, the two distributions the constants are
# calibrated on, and every call whose decision changes. It never runs a fit: the
# rows carry the Hessians.

using LinearAlgebra, Printf, DRModels

const RTOL = parse(Float64, get(ENV, "DRM_REPLAY_RTOL", string(DRModels._VCOV_RTOL)))
const FLAT = parse(Float64, get(ENV, "DRM_REPLAY_FLAT", string(DRModels._VCOV_FLAT_CURVATURE)))
const LEGACY_RTOL = 3e-8

struct Call
    file::String
    line::Int
    context::String
    unitfree::Vector{Int}
    H::Matrix{Float64}
end

function read_census(paths)
    calls = Call[]
    for path in paths, ln in eachline(path)
        f = split(ln, '\t')
        length(f) == 6 || continue
        p = parse(Int, f[4])
        uf = isempty(f[5]) ? Int[] : parse.(Int, split(f[5], ','))
        H = reshape(parse.(Float64, split(f[6], ',')), p, p)
        push!(calls, Call(f[1], parse(Int, f[2]), f[3], uf, H))
    end
    return calls
end

function legacy(H)
    ev = eigvals(Symmetric(H))
    sc = maximum(abs, ev)
    sc != 0 && minimum(ev) < -LEGACY_RTOL * sc && return :not_pd
    (sc == 0 || minimum(abs, ev) <= LEGACY_RTOL * sc) && return :singular
    return :inv
end

function hist(xs, label)
    println("\n", label, " (decade histogram):")
    for e in -17:4
        n = count(x -> 10.0^e <= x < 10.0^(e + 1), xs)
        n > 0 && @printf("  [1e%d, 1e%d)  %6d\n", e, e + 1, n)
    end
    @printf("  exactly 0      %6d\n", count(==(0), xs))
end

function main(paths)
    calls = read_census(paths)
    rows = map(calls) do c
        all(isfinite, c.H) || return (c = c, legacy = :nonfinite, new = :nonfinite, rcond = NaN,
                                      flat = Int[], ufd = Float64[])
        r = DRModels._vcov_guard_decision(c.H, c.unitfree; rtol = RTOL, flat_curvature = FLAT)
        (c = c, legacy = legacy(c.H), new = r.decision, rcond = r.rcond, flat = r.flat,
         ufd = [c.H[k, k] for k in c.unitfree])
    end
    @printf("%d calls from %d test files; RTOL = %g, FLAT = %g\n", length(rows),
            length(unique(r.c.file for r in rows)), RTOL, FLAT)

    println("\nTransition table (legacy -> new):")
    for a in (:inv, :singular, :not_pd, :nonfinite), b in (:inv, :singular, :not_pd, :nonfinite)
        n = count(r -> r.legacy == a && r.new == b, rows)
        n > 0 && @printf("  %-9s -> %-9s %6d\n", a, b, n)
    end

    hist(filter(isfinite, [r.rcond for r in rows]),
         "Collinearity: min|eig| / max|eig| of the equilibrated non-flat block")
    hist([abs(x) for r in rows for x in r.ufd],
         "Flatness: |curvature| of every unit-free coordinate, natural units")

    println("\nCalls whose decision changes (deduplicated by file:line and decision):")
    seen = Set{Tuple{String,Int,Symbol,Symbol}}()
    for r in rows
        r.legacy == r.new && continue
        key = (r.c.file, r.c.line, r.legacy, r.new)
        key in seen && continue
        push!(seen, key)
        mind = isempty(r.ufd) ? NaN : minimum(abs, r.ufd)
        @printf("  %-9s -> %-9s rcond=%.2e min|uf curv|=%.2e flat=%s  %s:%d  [%s]\n",
                r.legacy, r.new, r.rcond, mind, string(r.flat), r.c.file, r.c.line,
                first(r.c.context, 50))
    end
    return rows
end

isempty(ARGS) || main(ARGS)
