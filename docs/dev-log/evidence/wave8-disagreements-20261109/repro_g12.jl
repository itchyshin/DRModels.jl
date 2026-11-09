# Minimal reproduction for group G12 (binomial independent intercept + slope, sd(g, ...) ~ z):
# Julia converges to a local optimum 2.0 nats below a point that its own objective scores higher.
# Run:  julia --project=<env with DRModels> repro_g12.jl    (needs only the CSV next to this file)
using DRModels, DelimitedFiles
m, h = readdlm(joinpath(@__DIR__, "data", "data_binomial_ordinary.csv"), ',', Any; header = true)
h = String.(vec(h)); col(n) = m[:, findfirst(==(n), h)]
d = (y = Float64.(col("y")), x1 = Float64.(col("x1")), x2 = Float64.(col("x2")), x3 = Float64.(col("x3")),
     z = Float64.(col("z")), g = String.(col("g")))
f = bf(@formula(y ~ x1 + x2 + x3 + (1 | g) + (0 + x1 | g)),
       @formula(sd(g, coef(intercept)) ~ z), @formula(sd(g, coef(x1)) ~ z))
fit = drm(f, Binomial(:logit); data = d)
println("Julia optimum:  ll = ", loglikelihood(fit), "   converged = ", fit.converged)
println("                sd-block theta = ", round.(fit.theta[5:end]; digits = 3))
# drmTMB 0.8.1 estimate for the same model (ll -253.5291, is_converged() = FALSE, pdHess = TRUE):
thR = [0.472920237952, 0.217143847112, 0.23791642793, -0.273548232169,
       -54.264625208, 56.949833433, -0.972683864797, 0.149484497541]
fit.nll(Float64.(fit.theta))                       # warm the objective before evaluating elsewhere
println("Julia objective at the drmTMB estimate: ll = ", -fit.nll(thR))
