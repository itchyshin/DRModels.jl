using DRModels, DelimitedFiles
include(joinpath(@__DIR__, "jl_cell.jl"))
cl = CELLS["gaussian|spatial|lab0|q1|sd-none|cm0|ML"]
d = load_data("gaussian", "spatial")
Kfix = Matrix{Float64}(readdlm(joinpath(DD, "K_fixedrange.csv"), ',', Float64))
f1 = bf(@formula(y ~ x1 + x2 + x3 + relmat(1 | g)), @formula(sigma ~ 1))
fit1 = drm(f1, Gaussian(); data = d, K = Kfix)
println("Julia relmat with R's fixed-range kernel: ll = ", loglikelihood(fit1), "   (R spatial sd-none: -393.1791601168)")
fit2 = drm(build(cl), Gaussian(); data = d, coords = COORDS)
println("Julia native spatial (estimated range): ll = ", loglikelihood(fit2), " range theta = ", fit2.theta[end], " exp = ", exp(fit2.theta[end]))
