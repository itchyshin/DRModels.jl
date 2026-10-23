using DRModels, DelimitedFiles
raw = readdlm(joinpath(pwd(), "carni70.csv"), ','; header = true)[1]
dat = (; species = String.(strip.(String.(raw[:, 1]), '"')), lsize = Float64.(raw[:, 2]), lrange = Float64.(raw[:, 3]))
f = bf(@formula(lsize ~ 1 + lrange + phylo(1 | species)), @formula(sigma ~ 1 + phylo(1 | species)))
res = Dict()
for coupled in (false, true), (lab, file) in (("unit", "carni70_unit.nwk"), ("raw", "carni70.nwk"))
    nwk = read(joinpath(pwd(), file), String)
    t = @elapsed fit = drm(f, Gaussian(); data = dat, tree = nwk, phylo_coupled = coupled)
    res[(lab, coupled)] = fit
    println("RESULT tree=$lab coupled=$coupled logLik=", round(loglik(fit), digits = 10), " is_converged=", is_converged(fit),
            " time=", round(t, digits = 1), "s sd_mu=", fit.scales[:lambda_sd_mu], " sd_sigma=", fit.scales[:lambda_sd_sigma],
            " theta=", fit.theta)
    flush(stdout)
end
H = 53.8
for coupled in (false, true)
    u = res[("unit", coupled)]; r = res[("raw", coupled)]
    println("coupled=$coupled  |ΔlogLik| = ", abs(loglik(u) - loglik(r)),
            "  max|Δβ| = ", maximum(abs, u.theta[1:3] .- r.theta[1:3]),
            "  sd_mu ratio*sqrt(H) = ", r.scales[:lambda_sd_mu][1] * sqrt(H) / u.scales[:lambda_sd_mu][1],
            "  sd_sigma ratio*sqrt(H) = ", r.scales[:lambda_sd_sigma][1] * sqrt(H) / u.scales[:lambda_sd_sigma][1],
            "  SE(β) unit=", sqrt.(max.(0, [u.vcov[i,i] for i in 1:3])), " raw=", sqrt.(max.(0, [r.vcov[i,i] for i in 1:3])))
    coupled && println("   cor unit=", u.scales[:lambda_cor], " raw=", r.scales[:lambda_cor])
end
