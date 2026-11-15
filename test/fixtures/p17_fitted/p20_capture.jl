# Writes the stored fit values (theta, logLik, vcov) of every phylo-group cell, exactly (`repr` round-trips a Float64).
# Run once on the pre-change commit (a6b12b0be) and once on the changed tree; the two files must be identical.
#   julia --project=<proj> p20_capture.jl out.txt
using DRModels, Logging
include(joinpath(@__DIR__, "p20_cells.jl"))
d = _pm_mr(_pm_data())
open(ARGS[1], "w") do io
    for (id, fit_it) in _PM_CELLS
        fit = with_logger(NullLogger()) do
            try fit_it(d) catch e; nothing end
        end
        fit === nothing && (println(io, id, "\tFIT FAILED"); continue)
        println(io, id, "\ttheta=", repr(fit.theta), "\tloglik=", repr(fit.loglik), "\tconv=", fit.converged,
                "\tvcov=", repr(Matrix(fit.vcov)))
    end
end
