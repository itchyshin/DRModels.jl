include(joinpath(@__DIR__, "cells.jl"))
d = g12_data(joinpath(@__DIR__, "..", "..", "..", "..", "..", "test", "fixtures", "binomial_indep_restarts", "g12_binomial.csv"))
for q in (2, 3)
    t = @elapsed fit = drm(g12_form(q), Binomial(:logit); data = d)
    println("q=$q ll=", loglikelihood(fit), " conv=", fit.converged, " restarts=", get(fit.scales, :q2_restarts, nothing), " t=", round(t; digits=1))
    println("   theta=", round.(fit.theta; digits=3))
end
