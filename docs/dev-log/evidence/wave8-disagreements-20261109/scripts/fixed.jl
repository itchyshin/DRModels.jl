using DRModels
include("jl_cell.jl")
for fam in ["nb2","gamma","beta","student","tweedie"]
    d = load_data(fam, "ordinary")
    f = bf(@formula(y ~ x1 + x2 + x3), @formula(sigma ~ 1))
    fit = drm(f, fam_of(fam); data = d)
    println("JULIA ", fam, " ll=", loglikelihood(fit), " names=", fit.coefnames, " theta=", round.(fit.theta, digits=5))
end
