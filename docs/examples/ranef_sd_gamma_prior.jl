# Run from the repository root: julia --project=. docs/examples/ranef_sd_gamma_prior.jl
using DRModels

data = (y = [0.6, 2.0, 0.9, 1.1, 2.4, 1.1, 1.9, 2.3, 0.6],
        x = [-1.2, 0.2, -0.8, 0.1, 1.1, -1.0, -0.3, 0.6, 1.3],
        site = [1, 1, 2, 2, 2, 3, 3, 3, 3])
formula = bf(@formula(y ~ x + (1 | site)), @formula(sigma ~ x))
priors = drm_priors(
    drm_prior_sd_gamma(dpar="mu", group="site", coef="(Intercept)", shape=2, rate=3),
    drm_prior_normal(dpar="mu", coef="x", mean=0, sd=2),
)

println("Resolved prior targets:")
foreach(println, resolve_priors(formula, Gaussian(); data=data, priors=priors))
fit = drm(formula, Gaussian(); data=data, priors=priors)
println("Estimator: ", fit.estim_method)
println("Data log likelihood: ", loglik(fit))
println("Fixed-coefficient penalty: ", fit.penalty.fixed_penalty)
println("Random-intercept SD penalty: ", fit.penalty.sd_penalty)
println("Penalized negative log objective: ", fit.penalty.objective)
println("Converged: ", fit.converged)
