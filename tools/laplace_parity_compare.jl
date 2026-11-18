# Fit the listed gap-map cells with `marginal = :Laplace` (and, with --default, without it) and tabulate
# the agreement with drmTMB 0.8.1. Usage: julia --project=. tools/laplace_parity_compare.jl [--default] [id-prefix ...]
# (no prefix: every reference cell of the single-term, slope and independent-term shapes). One row per cell:
# id, status, dLogLik, max fixed-effect gap in R standard errors, max log-SD gap, seconds, agrees.
include(joinpath(@__DIR__, "..", "test", "laplace_parity_helpers.jl"))
ref = lp_reference()
args = filter(a -> a != "--default", ARGS)
mode = any(a == "--default" for a in ARGS) ? :LA : :Laplace
ids = sort([id for id in keys(ref) if split(id, '|')[2] in ("ordinary", "slope", "indep", "corr") && split(id, '|')[1] != "lognormal"])
isempty(args) || (ids = [id for id in ids if any(startswith(id, a) for a in args)])
println("cell\tmarginal\tstatus\tdloglik\tfixef_gap_se\tlogsd_gap\tseconds\tagrees\tr_clean")
for id in ids
    t = @elapsed res = try
        (lp_fit(id; marginal = mode), nothing)
    catch e
        (nothing, sprint(showerror, e))
    end
    fit, err = res
    if fit === nothing
        println(id, "\t", mode, "\tREFUSES\t\t\t\t", round(t; digits = 1), "\tfalse\t", ref[id].clean, "\t", first(split(err, '\n')))
    else
        c = lp_compare(id, fit, ref[id])
        println(id, "\t", mode, "\t", fit.converged ? "FITS" : "NONCONVERGED", "\t", c.dll, "\t", c.fe, "\t", c.sdlog, "\t",
                round(t; digits = 1), "\t", lp_agrees(c) && fit.converged, "\t", ref[id].clean)
    end
    flush(stdout)
end
