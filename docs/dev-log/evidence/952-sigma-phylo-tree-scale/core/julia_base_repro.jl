using DRModels, DelimitedFiles
raw = readdlm(joinpath(pwd(), "carni70.csv"), ','; header = true)[1]
dat = (; species = String.(strip.(String.(raw[:, 1]), '"')), lsize = Float64.(raw[:, 2]), lrange = Float64.(raw[:, 3]))
f = bf(@formula(lsize ~ 1 + lrange + phylo(1 | species)), @formula(sigma ~ 1 + phylo(1 | species)))
for (lab, file) in (("unit", "carni70_unit.nwk"), ("raw", "carni70.nwk")), coupled in (false, true)
    nwk = read(joinpath(pwd(), file), String)
    t = @elapsed fit = drm(f, Gaussian(); data = dat, tree = nwk, phylo_coupled = coupled)
    println("RESULT tree=$lab coupled=$coupled logLik=", round(loglik(fit), digits = 8), " converged=", is_converged(fit), " time=", round(t, digits = 1), "s theta=", fit.theta)
    flush(stdout)
end
