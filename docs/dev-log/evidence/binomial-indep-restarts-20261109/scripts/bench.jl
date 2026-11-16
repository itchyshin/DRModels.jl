# usage: julia --project=<DRModels> bench.jl SEED_FROM SEED_TO OUT.tsv ; one row per (seed, q, mode)
include(joinpath(@__DIR__, "cells.jl")); include(joinpath(@__DIR__, "gen.jl"))
a, b, out = parse(Int, ARGS[1]), parse(Int, ARGS[2]), ARGS[3]
# warm-up (compile) so the timed fits below are steady-state
for q in (2, 3), mode in ("off", "trigger", "force")
    ENV["DRM_BINOM_INDEP_MODE"] = mode; drm(g12_form(q), Binomial(:logit); data = gen_binomial_indep(0))
end
open(out, "a") do io
    for seed in a:b, q in (2, 3), mode in ("off", "trigger", "force")
        ENV["DRM_BINOM_INDEP_MODE"] = mode
        d = gen_binomial_indep(seed)
        try
            t = @elapsed fit = drm(g12_form(q), Binomial(:logit); data = d)
            rs = get(fit.scales, :q2_restarts, [NaN, NaN, NaN, NaN])
            println(io, join((seed, q, mode, loglikelihood(fit), fit.converged, round(t; digits = 2), rs[1], rs[2], rs[3]), '\t')); flush(io)
        catch e
            println(io, join((seed, q, mode, "ERR", first(sprint(showerror, e), 80)), '\t')); flush(io)
        end
    end
end
