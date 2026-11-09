include("ep_cell_j.jl")
ids_all = readlines(ARGS[1])[2:end]; outfile = ARGS[2]
io = open(outfile, "w")
emit(args...) = (println(io, join(string.(args), '\t')); flush(io))
msg(e) = replace(first(sprint(showerror, e), 250), r"[\t\n]+" => " ")
for id in ids_all
    cl = CELLS[id]; d = load_data(cl.family, cl.kind in ("indep", "slope") ? "ordinary" : cl.kind)
    fit = try
        f = build(cl); kw = kwargs_for(cl.kind)
        cl.est == "REML" ? drm(f, fam_of(cl.family); data = d, method = :REML, kw...) : drm(f, fam_of(cl.family); data = d, kw...)
    catch e; emit(id, "prof", "FIT_ERROR", msg(e)); nothing end
    fit === nothing && continue
    t0 = time()
    try
        for r in confint(fit; method = :profile); emit(id, "prof", string(r.param, ":", r.coef), @sprintf("%.10g", r.lower), @sprintf("%.10g", r.upper), "profile"); end
    catch e
        emit(id, "prof", "ERROR", msg(e))
    end
    emit(id, "prof", "secs", @sprintf("%.1f", time() - t0))
end
close(io)
