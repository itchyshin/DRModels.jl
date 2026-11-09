# Phase 2: Julia fit of one cell, evaluate Julia objective at R's estimate, write theta_J.
using DRModels
include(joinpath(@__DIR__, "jl_cell.jl"))
id = ARGS[1]; n = ARGS[2]
cl = CELLS[id]; d = load_data(cl.family, cl.kind in ("indep","slope") ? "ordinary" : cl.kind)
f = build(cl)
fit = drm(f, fam_of(cl.family); data = d, kwargs_for(cl.kind)...)
llJ = loglikelihood(fit)
rl = split(readline(joinpath(@__DIR__, "out", "xr", n * ".tsv")), '\t')
thR = parse.(Float64, split(rl[5], ';'))
global llJ_at_R = NaN; global note = ""
if length(thR) == length(fit.theta)
    try
        fit.nll(Float64.(fit.theta))   # warm the objective's inner state before evaluating elsewhere
        global llJ_at_R = -fit.nll(thR)
    catch e
        global note = "nll error: " * first(sprint(showerror, e), 120) end
else
    global note = "npar differs: R $(length(thR)) vs Julia $(length(fit.theta))"
end
mkpath(joinpath(@__DIR__, "out", "xj"))
open(joinpath(@__DIR__, "out", "xj", n * ".tsv"), "w") do io
    println(io, join([id, string(llJ), string(llJ_at_R), string(fit.converged),
        join(string.(fit.theta), ";"), join(string.(first.(fit.coefnames)), ";"), note], '\t'))
end
