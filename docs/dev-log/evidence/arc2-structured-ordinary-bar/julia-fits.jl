# Arc 2 `structured_with_ordinary_bar` receipt -- JULIA side.
#
# Reads the fixtures native-fits.R wrote, fits each through `drm_bridge` (the
# entry point drmTMB's R bridge calls) with the formula string and the K / A /
# tree kwargs the bridge would send, and writes
#   julia.tsv       fixture, param, value  (same param spelling as native.tsv)
#   comparison.tsv  fixture, param, native, julia, abs_diff, rel_diff
# Run from this directory (after native-fits.R):
#   JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 julia --project=<DRModels.jl> julia-fits.jl
using DRModels

function read_tsv(path)
    lines = filter(!isempty, readlines(path))
    header = split(lines[1], '\t')
    rows = [split(l, '\t') for l in lines[2:end]]
    return header, rows
end

function read_data(path)
    header, rows = read_tsv(path)
    cols = Dict{String,Any}()
    for (j, h) in enumerate(header)
        raw = [String(r[j]) for r in rows]
        parsed = tryparse.(Float64, raw)
        cols[String(h)] = any(isnothing, parsed) ? raw : Float64.(parsed)
    end
    return cols
end

function read_matrix(path)
    _, rows = read_tsv(path)
    return [parse(Float64, r[j+1]) for r in rows, j in 1:length(rows)]
end

formulas = Dict(String(split(l, '\t')[1]) => String(split(l, '\t')[2])
                for l in readlines(joinpath("fixtures", "formulas.tsv")) if !isempty(l))
_, nrows = read_tsv("native.tsv")
native = Dict((String(r[1]), String(r[2])) => parse(Float64, r[3]) for r in nrows)
fixtures = unique(String(r[1]) for r in nrows)

# Julia `:resd` label -> native `sdpars$mu` label.
function native_sd_label(fx, label, formula)
    m = match(r"(phylo|relmat|animal)\(1 \| (\w+)\)", formula)
    kind, sgrp = m.captures
    fx == "spatial_h" && (kind = "spatial")
    label == sgrp && return "$(kind)(1 | $(sgrp))"
    endswith(label, "_iid") && return "(1 | $(label[1:end-4]))"
    occursin(':', label) && return "(0 + $(split(label, ':')[2]) | $(split(label, ':')[1]))"
    return "(1 | $(label))"
end

out = Tuple{String,String,Float64}[]
for fx in fixtures
    formula = formulas[fx]
    data = read_data(joinpath("fixtures", "$fx.data.tsv"))
    kw = Dict{Symbol,Any}()
    isfile(joinpath("fixtures", "$fx.tree.nwk")) &&
        (kw[:tree] = strip(read(joinpath("fixtures", "$fx.tree.nwk"), String)))
    isfile(joinpath("fixtures", "$fx.K.tsv")) && (kw[:K] = read_matrix(joinpath("fixtures", "$fx.K.tsv")))
    isfile(joinpath("fixtures", "$fx.A.tsv")) && (kw[:A] = read_matrix(joinpath("fixtures", "$fx.A.tsv")))
    res = drm_bridge(; formula = formula, family = "gaussian", data = data, kw...)
    res["converged"] || error("$fx did not converge")
    push!(out, (fx, "df", Float64(res["df"])))
    push!(out, (fx, "logLik", res["loglik"]))
    for (nm, v) in zip(res["raw_coef_names"], res["coefficients"])
        if startswith(nm, "resd_")
            push!(out, (fx, "sd:" * native_sd_label(fx, nm[6:end], formula), exp(v)))
        else
            dp, term = split(nm, "_"; limit = 2)
            push!(out, (fx, "$(dp):$(term)", v))
        end
    end
end

open("julia.tsv", "w") do io
    println(io, "fixture\tparam\tvalue")
    for (fx, p, v) in out
        println(io, fx, '\t', p, '\t', string(round(v; digits = 10)))
    end
end
open("comparison.tsv", "w") do io
    println(io, "fixture\tparam\tnative\tjulia\tabs_diff\trel_diff")
    for (fx, p, v) in out
        nv = native[(fx, p)]
        ad = abs(v - nv)
        println(io, join((fx, p, nv, round(v; digits = 10), ad, ad / max(abs(nv), 1e-12)), '\t'))
    end
end
# every native parameter must have a Julia twin (no dropped term)
missing_twin = setdiff(Set(keys(native)), Set((fx, p) for (fx, p, _) in out))
isempty(missing_twin) || error("native parameters with no Julia twin: $(collect(missing_twin))")
println("wrote julia.tsv and comparison.tsv for $(length(fixtures)) fixtures")
