# Defect 2: Gaussian independent intercept + slopes on one grouping factor, (1|g) + (0+x1|g) + (0+x2|g).
# The fit has three random-effect SDs, but the three :resd coefficients are all labelled "g", so re_sd(fit) and vc(fit)
# return a single entry (the last SD) instead of three. The Student and Gamma routes label them g, g:x1, g:x2.
# Run: julia --project=<DRModels env at 0bfcde2bb> repro2_re_sd_indep_gaussian.jl   (from this directory)
using DRModels, DelimitedFiles
m, h = readdlm(joinpath(@__DIR__, "..", "data", "data_gaussian_ordinary.csv"), ',', Any; header = true)
h = String.(vec(h)); col(n) = m[:, findfirst(==(n), h)]
d = (; y = Float64.(col("y")), x1 = Float64.(col("x1")), x2 = Float64.(col("x2")), x3 = Float64.(col("x3")), g = String.(col("g")))
fit = drm(bf(@formula(y ~ x1 + x2 + x3 + (1 | g) + (0 + x1 | g) + (0 + x2 | g)), @formula(sigma ~ 1)), Gaussian(); data = d)
println("resd labels : ", first(cn[2] for cn in fit.coefnames if cn[1] === :resd), "   (expected g, g:x1, g:x2)")
println("re_sd(fit)  : ", re_sd(fit), "   (expected 3 entries; drmTMB reports 3 SDs)")
println("vc(fit)     : ", vc(fit))
println("coef(fit, :resd) = ", coef(fit, :resd), "   (3 values, so the estimates exist; only the labels collide)")
