# Fit models of docs/dev-log/2026-11-09-gaussian-sd-groupf.md in Julia and compare with the drmTMB 0.8.1
# reference. Usage: julia --project=. tools/gaussian_sd_groupf_compare.jl <out.tsv> <dataset> <model> [<model> ...]
# (one process per dataset, so a shell pool can run them in parallel; JULIA_NUM_THREADS=1).
include(joinpath(@__DIR__, "..", "test", "gaussian_sd_groupf_helpers.jl"))
out = ARGS[1]; ds = ARGS[2]; ids = ARGS[3:end]
ref = gsf_reference(); d = gsf_read(ds)
open(out, "a") do io
    for id in ids
        t = @elapsed fit = try drm(gsf_build(id), Gaussian(); data = d) catch e; e end
        if fit isa Exception
            println(io, join([ds, id, "ERROR", first(replace(sprint(showerror, fit), r"\s+" => " "), 200)], '\t'))
            flush(io); continue
        end
        c = gsd_compare(gsf_extract(fit, id), ref[(ds, id)])
        println(io, join([ds, id, gsf_group(id), c.dll, c.fe, c.sdf, c.sdlog, c.cor, c.flagged, c.pass, round(t, digits = 1),
                          join(c.unmatched, "; ")], '\t'))
        flush(io)
    end
end
