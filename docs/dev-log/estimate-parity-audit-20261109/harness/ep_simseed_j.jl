include("ep_cell_j.jl")
ids_all = readlines(ARGS[1])[2:end]; io = open(ARGS[2], "w")
for id in ids_all
    cl = CELLS[id]; d = load_data(cl.family, cl.kind in ("indep", "slope") ? "ordinary" : cl.kind)
    fit = drm(build(cl), fam_of(cl.family); data = d, kwargs_for(cl.kind)...)
    kwm = (; data = d, (k => v for (k, v) in pairs(kwargs_for(cl.kind)) if k != :ids)...)
    for s in 1:6
        sm = Matrix{Float64}(simulate(fit; nsim = 1000, rng = MersenneTwister(s), kwm...))
        println(io, join([id, "seed", s, @sprintf("%.6g", var(vec(sm))), @sprintf("%.6g", mean(sm))], '\t')); flush(io)
    end
end
close(io)
