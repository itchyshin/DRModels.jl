# Fit every listed gap-map independent-term cell and tabulate the agreement with drmTMB 0.8.1.
# Usage: julia --project=. tools/nongauss_indep_compare.jl [cell-id-prefix ...] > table.tsv
# (no prefix: every ML `sd-none` cell of the reference; `--all` adds the cells with SD formulas). One row per cell:
# id, status, dLogLik, max fixed-effect gap, max log-SD gap, seconds, agrees.
include(joinpath(@__DIR__, "..", "test", "nongauss_indep_helpers.jl"))
ref = ngi_reference()
ids = sort([id for id in keys(ref) if occursin("|sd-none|", id)])
args = filter(a -> a != "--all", ARGS)
any(a == "--all" for a in ARGS) && (ids = sort(collect(keys(ref))))
isempty(args) || (ids = [id for id in ids if any(startswith(id, a) for a in args)])
println("cell\tstatus\tdloglik\tfixef_gap\tlogsd_gap\tseconds\tagrees\tr_clean")
for id in ids
    fam = split(id, '|')[1]
    d = ngi_data(fam; crossed = occursin("|crossed|", id))
    t = @elapsed res = try
        fit = drm(ngi_build(id), ngi_family(id); data = d)
        (fit, nothing)
    catch e
        (nothing, sprint(showerror, e))
    end
    fit, err = res
    if fit === nothing
        println(id, "\tREFUSES\t\t\t\t", round(t; digits = 1), "\tfalse\t", ref[id].clean, "\t", first(split(err, '\n')))
    else
        c = ngi_compare(id, fit, ref[id])
        println(id, "\t", fit.converged ? "FITS" : "NONCONVERGED", "\t", c.dll, "\t", c.fe, "\t", c.sdlog, "\t",
                round(t; digits = 1), "\t", ngi_agrees(c) && fit.converged, "\t", ref[id].clean)
    end
    flush(stdout)
end
