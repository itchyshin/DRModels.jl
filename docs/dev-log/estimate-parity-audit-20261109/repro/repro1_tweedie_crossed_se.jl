# Defect 1: Tweedie with two crossed random intercepts converges (logLik -724.7952, same as drmTMB) but every
# standard error is Inf and is_converged is false, while drmTMB reports a positive-definite Hessian and finite SEs.
# Run: julia --project=<DRModels env at 0bfcde2bb> repro1_tweedie_crossed_se.jl   (from this directory)
using DRModels, DelimitedFiles
m, h = readdlm(joinpath(@__DIR__, "..", "data", "data_tweedie_crossed.csv"), ',', Any; header = true)
h = String.(vec(h)); col(n) = m[:, findfirst(==(n), h)]
d = (; y = Float64.(col("y")), x1 = Float64.(col("x1")), x2 = Float64.(col("x2")), x3 = Float64.(col("x3")), g = String.(col("g")), g2 = String.(col("g2")))
fit = drm(bf(@formula(y ~ x1 + x2 + x3 + (1 | g) + (1 | g2)), @formula(sigma ~ 1)), Tweedie(); data = d)
println("logLik       = ", loglikelihood(fit))
println("is_converged = ", is_converged(fit), "   (drmTMB: TRUE, pdHess TRUE)")
println("stderror     = ", stderror(fit), "   (drmTMB: 0.131 0.061 0.061 0.057 0.034 ... all finite)")
println(coeftable(fit))
# drmTMB 0.8.1 reference (R): drmTMB(bf(y ~ x1 + x2 + x3 + (1 | g) + (1 | g2), sigma = ~ 1), family = tweedie(), data = d)
#   mu:(Intercept) 0.2596 (SE 0.1314), mu:x1 0.3363 (0.0609), mu:x2 0.3102 (0.0609), mu:x3 -0.2624 (0.0574),
#   sigma:(Intercept) 0.1762 (0.0340), nu:(Intercept) 0.1877 (0.0986); logLik -724.7952
