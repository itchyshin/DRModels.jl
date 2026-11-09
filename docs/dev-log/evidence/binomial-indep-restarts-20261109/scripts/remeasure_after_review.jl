using DRModels
include(joinpath(ENV["W"], "test/fixtures/binomial_indep_restarts/gen_binomial_indep.jl"))
fq(q) = q == 2 ?
    bf(@formula(y ~ x1 + x2 + x3 + (1 | g) + (0 + x1 | g)), @formula(sd(g, coef(intercept)) ~ z), @formula(sd(g, coef(x1)) ~ z)) :
    bf(@formula(y ~ x1 + x2 + x3 + (1 | g) + (0 + x1 | g) + (0 + x2 | g)), @formula(sd(g, coef(intercept)) ~ z), @formula(sd(g, coef(x1)) ~ z), @formula(sd(g, coef(x2)) ~ z))
open(ARGS[1], "w") do io
    for seed in 1:40, q in (2, 3)
        d = gen_binomial_indep(seed)
        t = @elapsed fit = drm(fq(q), Binomial(:logit); data = d, se = false)
        rs = fit.scales[:q2_restarts]
        println(io, join([seed, q, "new", loglikelihood(fit), fit.converged, round(t, digits=3), rs...], "\t")); flush(io)
    end
end
