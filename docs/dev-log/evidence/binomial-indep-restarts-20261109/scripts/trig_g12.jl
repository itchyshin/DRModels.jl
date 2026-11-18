include(joinpath(@__DIR__, "cells.jl"))
d = g12_data(joinpath(@__DIR__, "..", "..", "..", "..", "..", "test", "fixtures", "binomial_indep_restarts", "g12_binomial.csv"))
for mode in ("off", "trigger", "force"), q in (2, 3)
    ENV["DRM_BINOM_INDEP_MODE"] = mode
    t = @elapsed fit = drm(g12_form(q), Binomial(:logit); data = d, se = false)
    println(mode, "\tq", q, "\tll=", round(loglikelihood(fit); digits = 4), "\tconv=", fit.converged, "\trestarts=", get(fit.scales, :q2_restarts, nothing), "\tt=", round(t; digits = 2))
end
