# Re-run the Gaussian ordinary/indep/crossed cells that wave-8 Julia refused (R fits them) on this branch,
# against R's logLik recorded in the gap map (cells.tsv, from drmTMB branch claude/gap-map-wave8-20261108,
# fuzz-gap-map-wave8/maps/w8/julia_behind_ranked.tsv). Cell definitions mirror that harness's run_julia.jl.
# Usage: julia --project=<repo> run_cells.jl <out.tsv> <cell_id> [<cell_id> ...]
using DRModels
function readcsv(path)
    ls = readlines(path); h = Symbol.(split(ls[1], ","))
    cols = [split(l, ",") for l in ls[2:end]]
    vals = map(eachindex(h)) do j
        v = [c[j] for c in cols]
        all(x -> tryparse(Float64, x) !== nothing, v) ? parse.(Float64, v) : String.(v)
    end
    NamedTuple{Tuple(h)}(Tuple(vals))
end
mkf(s) = eval(Meta.parse("@formula(" * s * ")"))
const CNAME = ["intercept", "x1", "x2", "x3"]
function build(kind, q, lab, pat)
    bar = join(["1", "x1", "x2", "x3"][1:q], " + ")
    inner = bar * (lab == 1 ? " | p | " : " | ") * "g"
    re = kind == "indep" ? "(1 | g)" * join([" + (0 + $v | g)" for v in ["x1", "x2", "x3"][1:q-1]]) :
         kind == "ordinary" ? "($inner)" : "($inner) + (1 | g2)"
    mu = mkf("y ~ x1 + x2 + x3 + $re")
    sel = pat == "none" ? Int[] : pat in ("one", "z") ? collect(1:q) : pat == "first" ? [1] : [q]
    rhs = pat == "one" ? "1" : "z"
    sds = Any[mkf("sd(g, coef($(CNAME[j]))) ~ $rhs") for j in sel]
    kind == "crossed" && !(pat in ("none", "last")) && push!(sds, mkf("sd(g2) ~ " * (pat == "one" ? "1" : "z2")))
    bf(mu, mkf("sigma ~ 1"), sds...)
end
ref = Dict(split(l, '\t')[1] => l for l in readlines(joinpath(@__DIR__, "cells.tsv"))[2:end])
open(ARGS[1], "a") do io
    for id in ARGS[2:end]
        p = split(ref[id], '\t'); kind = p[3]; q = parse(Int, p[4]); lab = parse(Int, p[5]); pat = p[6]
        rll = parse(Float64, p[2])
        d = readcsv(joinpath(@__DIR__, "data_gaussian_$(kind == "indep" ? "ordinary" : kind).csv"))
        t = @elapsed fit = try drm(build(kind, q, lab, pat), Gaussian(); data = d) catch e; e end
        if fit isa Exception
            println(io, join([id, rll, "ERROR", "", "", first(replace(sprint(showerror, fit), r"\s+" => " "), 160)], '\t'))
        else
            println(io, join([id, rll, round(loglik(fit), digits = 8), abs(loglik(fit) - rll), fit.converged, round(t, digits = 1)], '\t'))
        end
        flush(io)
    end
end
