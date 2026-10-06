# Arc 2 sigma-RE Laplace receipt: DRModels side.
#
# Reads the fixtures written by native_fit.R, fits each with
# `marginal = :Laplace` (the new route) and with the default `marginal = :LA`
# (GHQ-32, for the size of the gap it closes), and writes julia.tsv plus
# comparison.tsv (per cell: df, logLik on both engines, |ΔlogLik|, the largest
# relative and absolute estimate difference, and the largest relative SE
# difference against native.tsv).
#
# Run from the DRModels.jl repo root after native_fit.R:
#   JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 julia --project=. \
#     docs/dev-log/evidence/arc2-sigma-re-laplace/julia_fit.jl
using DRModels
using Printf

const DIR = joinpath("docs", "dev-log", "evidence", "arc2-sigma-re-laplace")

function readfix(path)
    lines = readlines(path)
    hdr = replace.(split(lines[1], ","), "\"" => "")
    cols = [String[] for _ in hdr]
    for l in lines[2:end], (j, v) in enumerate(split(l, ","))
        push!(cols[j], replace(v, "\"" => ""))
    end
    col(n) = cols[findfirst(==(n), hdr)]
    return (y = parse.(Float64, col("y")), x = parse.(Float64, col("x")), g = col("g"))
end

readtsv(path) = (ls = readlines(path); h = split(ls[1], '\t');
                 [Dict(zip(h, split(l, '\t'))) for l in ls[2:end]])

native = readtsv(joinpath(DIR, "native.tsv"))
cells = unique([r["cell"] for r in native])
jrows = String["cell\tengine\tdf\tlogLik\tconverged\tparam\testimate\tse"]
crows = String["cell\tn\tG\tdf_native\tdf_julia\tlogLik_native\tlogLik_julia_Laplace\tabs_dlogLik_Laplace\t" *
               "max_rel_dest_Laplace\tmax_abs_dest_Laplace\tmax_rel_dse_Laplace\tconverged_Laplace\t" *
               "logLik_julia_LA_GHQ32\tdlogLik_LA_GHQ32\tverdict"]
for cell in cells
    nr = filter(r -> r["cell"] == cell, native)
    d = readfix(joinpath(DIR, "fixtures", cell * ".csv"))
    sig = startswith(cell, "c6") ? @formula(sigma ~ 1 + x + (1 | g)) : @formula(sigma ~ 1 + (1 | g))
    f = bf(@formula(y ~ x), sig)
    fitL = drm(f, Gaussian(); data = d, marginal = :Laplace)
    fitG = drm(f, Gaussian(); data = d)
    @assert fitL.marginal === :Laplace && fitG.marginal === :LA
    est = fitL.theta; se = stderror(fitL)
    names = [r["param"] for r in nr]
    @assert length(est) == length(names)
    for (k, nm) in enumerate(names)
        push!(jrows, @sprintf("%s\tjulia_Laplace\t%d\t%.10f\t%s\t%s\t%.10f\t%.10f",
                              cell, dof(fitL), loglik(fitL), fitL.converged, nm, est[k], se[k]))
    end
    nat = parse.(Float64, [r["estimate"] for r in nr])
    natse = parse.(Float64, [r["se"] for r in nr])
    llN = parse(Float64, nr[1]["logLik"]); dfN = parse(Int, nr[1]["df"])
    dll = abs(loglik(fitL) - llN)
    rel = maximum(abs.(est .- nat) ./ max.(abs.(nat), 1e-8))
    ab = maximum(abs.(est .- nat))
    relse = maximum(abs.(se .- natse) ./ natse)
    ok = dfN == dof(fitL) && dll <= 1e-6 && rel <= 1e-5 && fitL.converged
    push!(crows, @sprintf("%s\t%s\t%s\t%d\t%d\t%.10f\t%.10f\t%.3e\t%.3e\t%.3e\t%.3e\t%s\t%.10f\t%.4f\t%s",
                          cell, nr[1]["n"], nr[1]["G"], dfN, dof(fitL), llN, loglik(fitL), dll, rel, ab,
                          relse, fitL.converged, loglik(fitG), loglik(fitG) - llN,
                          ok ? "SAME_MODEL_MATCH" : "MISMATCH"))
    println(crows[end])
end
write(joinpath(DIR, "julia.tsv"), join(jrows, "\n") * "\n")
write(joinpath(DIR, "comparison.tsv"), join(crows, "\n") * "\n")
println("julia ", VERSION, " | DRModels ", pkgversion(DRModels))
