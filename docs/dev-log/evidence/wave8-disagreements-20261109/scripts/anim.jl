using DRModels
include(joinpath(@__DIR__, "jl_cell.jl"))
for (n, id) in ((4, "binomial|animal|lab0|q2|sd-one|cm0|ML"), (7, "binomial|animal|lab0|q2|sd-z|cm0|ML"))
    cl = CELLS[id]; d = load_data(cl.family, cl.kind); fit = drm(build(cl), fam_of(cl.family); data = d, kwargs_for(cl.kind)...)
    rl = split(readline(joinpath(@__DIR__, "out", "xr", "$n.tsv")), '\t'); thR = parse.(Float64, split(rl[5], ';'))
    fit.nll(Float64.(fit.theta))
    println(id, "\n  Julia ll = ", loglikelihood(fit), "  atanh_rho_hat = ", fit.theta[end], "  (R estimate ll ", rl[2], ")")
    println("  Julia objective at R's estimate with rho = 0: ", -fit.nll([thR; 0.0]))
    println("  Julia objective at Julia's estimate with rho set to 0: ", -fit.nll([fit.theta[1:end-1]; 0.0]))
end
