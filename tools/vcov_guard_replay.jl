# Replay the covariance guard's decision on a census written by
# `tools/vcov_guard_census.jl`, under the shipped constants or candidate ones.
#
#   julia --project=. tools/vcov_guard_replay.jl census_*.tsv
#   DRM_REPLAY_RTOL=1e-8 DRM_REPLAY_FLOOR=10 julia --project=. tools/vcov_guard_replay.jl census_*.tsv
#
# For every recorded call it computes
#   legacy  the decision of the pre-#936 rule (eigenvalues of the UNSCALED matrix,
#           relative tolerance 3e-8)
#   new     the decision of `_vcov_from_hessian` (eigenvalues of the matrix
#           equilibrated by `_vcov_scaling`), with DRM_REPLAY_RTOL / _FLOOR when set
# and prints the transition table, the distribution of the equilibrated ratio
# min|eig| / max|eig| in the region that matters for the threshold, and every call
# whose decision changes. It never runs a fit: the rows carry the Hessians.

using LinearAlgebra, Printf, DRModels

const RTOL = parse(Float64, get(ENV, "DRM_REPLAY_RTOL", string(DRModels._VCOV_RTOL)))
const FLOOR = parse(Float64, get(ENV, "DRM_REPLAY_FLOOR", string(DRModels._VCOV_UNITFREE_FLOOR)))
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

function scaling(H, unitfree, floor)
    s = similar(diag(H))
    for k in eachindex(s)
        c = abs(H[k, k])
        c = k in unitfree ? max(c, floor) : (c == 0 ? one(c) : c)
        s[k] = 1 / sqrt(c)
    end
    return s
end

function decide(ev, rtol)
    scale = maximum(abs, ev)
    scale != 0 && minimum(ev) < -rtol * scale && return :saddle
    (scale == 0 || minimum(abs, ev) <= rtol * scale) && return :singular
    return :inv
end

ratio(ev) = (sc = maximum(abs, ev); sc == 0 ? 0.0 : minimum(abs, ev) / sc)

function main(paths)
    calls = read_census(paths)
    rows = map(calls) do c
        fin = all(isfinite, c.H)
        if !fin
            return (c = c, legacy = :nonfinite, new = :nonfinite, rH = NaN, rM = NaN, ufmin = NaN)
        end
        evH = eigvals(Symmetric(c.H))
        s = scaling(c.H, c.unitfree, FLOOR)
        evM = eigvals(Symmetric(s .* c.H .* s'))
        ufmin = isempty(c.unitfree) ? NaN : minimum(c.H[k, k] for k in c.unitfree)
        (c = c, legacy = decide(evH, LEGACY_RTOL), new = decide(evM, RTOL),
         rH = ratio(evH), rM = ratio(evM), ufmin = ufmin)
    end
    @printf("%d calls from %d files; RTOL = %g, FLOOR = %g\n", length(rows),
            length(unique(r.c.file for r in rows)), RTOL, FLOOR)

    println("\nTransition table (legacy -> new):")
    for a in (:inv, :singular, :saddle, :nonfinite), b in (:inv, :singular, :saddle, :nonfinite)
        n = count(r -> r.legacy == a && r.new == b, rows)
        n > 0 && @printf("  %-9s -> %-9s %6d\n", a, b, n)
    end

    println("\nEquilibrated ratio min|eig|/max|eig|, sorted, in [1e-13, 1e-3] (decade histogram):")
    rM = sort(filter(isfinite, [r.rM for r in rows]))
    for e in -16:-1
        n = count(x -> 10.0^e <= x < 10.0^(e + 1), rM)
        n > 0 && @printf("  [1e%d, 1e%d)  %6d\n", e, e + 1, n)
    end
    @printf("  exactly 0      %6d\n", count(==(0), rM))
    println("\nThe 40 equilibrated ratios nearest the threshold:")
    near = sort(filter(r -> isfinite(r.rM), rows), by = r -> abs(log10(max(r.rM, 1e-300)) - log10(RTOL)))
    for r in near[1:min(40, end)]
        @printf("  %.3e  legacy=%-8s new=%-8s %s:%d  %s\n", r.rM, r.legacy, r.new,
                r.c.file, r.c.line, first(r.c.context, 60))
    end

    println("\nCalls whose decision changes:")
    for r in rows
        r.legacy == r.new && continue
        @printf("  %-8s -> %-8s rH=%.2e rM=%.2e minufdiag=%.2e  %s:%d  [%s]  uf=%s\n",
                r.legacy, r.new, r.rH, r.rM, r.ufmin, r.c.file, r.c.line,
                first(r.c.context, 60), string(r.c.unitfree))
    end
    return rows
end

isempty(ARGS) || main(ARGS)
