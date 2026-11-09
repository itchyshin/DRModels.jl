# Coefficient and random-intercept SD priors

This interface is experimental while its supported routes are being validated.

Use coefficient priors to regularize fixed effects when estimates are
weakly determined. A prior applies to a coefficient on the linear-predictor
scale. Gaussian `mu` uses the identity link, Gaussian `sigma` uses the log link,
and binomial `mu` uses the logit link. Predictor units and factor coding change
the meaning of a coefficient prior; choose scales accordingly.

For example, an ecologist could ask whether average growth changes with age.
In the example, `y` is growth and `x` is age. The `mu` coefficient for `x`
describes the change in mean growth per age unit. The `sigma` intercept is the
log residual standard deviation, assumed constant across ages. Choose each
prior in those units; a prior does not guarantee that the data identify the
corresponding quantity.

```julia
using DRModels
dat = (y = [1.0, 2.2, 2.9, 4.1, 4.8, 6.3], x = collect(0.0:5.0))
p = drm_priors(
    drm_prior_normal(dpar = "mu", coef = "x", mean = 0, sd = 2),
    drm_prior_normal(dpar = "sigma", coef = "(Intercept)", mean = 0, sd = 3),
)
form = bf(@formula(y ~ x))
resolve_priors(form, Gaussian(); data = dat, priors = p)
fit = drm(form, Gaussian(); data = dat, priors = p)
```

Selectors are exact final design-matrix column names. To select all columns,
use `coef = "all"` and state `intercept = "include"` or `"exclude"`.
An explicit intercept uses `coef = "(Intercept)"`. Empty, unknown, overlapping,
or rank-deficient selections fail. The preview reports the distributional
parameter, coefficient, position, precision and selection reason.

A normal prior contributes its full negative log density. A ridge prior uses
`drm_prior_ridge(dpar="mu", coef="all", intercept="exclude", lambda=0.25)`
and contributes `lambda * beta^2 / 2` per selected coefficient. Lambda zero is
unpenalized. A normal prior with mean zero and standard deviation two has the
same quadratic term, but includes its density constant.

Gaussian models may also contain one independent random intercept in the mean
formula, for example `bf(@formula(y ~ x + (1 | site)))`. Fixed-effect priors
apply to coefficients of `mu` and `sigma`; an SD prior is optional, as shown
below. The default `:LA` Gaussian route integrates the
intercept exactly and adds the fixed-effect priors once to the marginal
negative log-likelihood. Explicit `marginal = :Laplace` is not supported for
a mean random intercept.

```julia
grouped = (y = [1., 1.4, 2.0, 2.7, 1.2, 1.9, 3.2, 3.4],
           x = [0., 1., 0., 1., 0., 1., 0., 1.],
           site = repeat(1:4, inner = 2))
group_form = bf(@formula(y ~ x + (1 | site)))
group_priors = drm_priors(drm_prior_normal(dpar = "mu", coef = "x", mean = 0, sd = 2))
resolve_priors(group_form, Gaussian(); data = grouped, priors = group_priors)
group_fit = drm(group_form, Gaussian(); data = grouped, priors = group_priors)
```

The supported routes are Gaussian `mu` and `sigma` (with at most that one mean
random intercept) and fixed-effect binomial-logit `mu`. Random slopes, multiple
or structured random effects, random-effect SD formulas, REML, existing
phylogenetic penalties, missing responses or predictors, non-unit weights, and
other families are refused. For separated binomial data,
a finite optimum requires all separating directions to be regularized; a prior
on just one coefficient does not guarantee this.

Active prior fits are labelled `:MAP`. `loglik(fit)` is the data log-likelihood
at the penalized estimate. The stored objective includes the prior. Its inverse
Hessian describes local curvature of that penalized objective; it is neither an
automatic frequentist covariance nor a full posterior interval. Likelihood
ratio tests, AIC/BIC, confidence intervals, likelihood profiles and bootstrap
refits are refused for these fits. Prediction standard errors and simulation are
not yet qualified.

## Gamma prior on a random-intercept SD

A Gaussian ML model with one unlabelled iid random intercept in `mu` can use a
Gamma density on that group's natural standard deviation. Supply its exact
name and inspect the resolved target before fitting:

```julia
sd_prior = drm_prior_sd_gamma(dpar = "mu", group = "site",
                              coef = "(Intercept)", shape = 2, rate = 3)
priors = drm_priors(sd_prior,
                    drm_prior_normal(dpar = "mu", coef = "x", mean = 0, sd = 2))
resolve_priors(group_form, Gaussian(); data = grouped, priors = priors)
fit = drm(group_form, Gaussian(); data = grouped, priors = priors)
```

`shape` must be finite and greater than one, and `rate` must be finite and
positive. The Gamma density is with respect to the SD itself; it does not add
a log-SD Jacobian. The normalized penalty is added once for the shared SD,
not once per group. The fit is penalized ML, or SD-scale MAP: `loglik(fit)`
remains the data log likelihood at the penalized estimate. The fixed and SD
penalties and total objective are stored separately in `fit.penalty`.
`update` repeats the prior unless explicitly replaced or dropped by changing
the estimator.

This route requires complete observations, unit weights and a unique iid
mean random intercept; random slopes, multiple or structured blocks and REML
are outside this slice. Write the random intercept only once in the formula.
Currently, Julia's `@formula` collapses repeated identical terms such as
`(1 | site) + (1 | site)` before the prior target is checked. The resolver
therefore cannot reject that spelling as a duplicate; drmTMB in R rejects it.
This difference remains unresolved.

Confidence intervals, likelihood comparisons, AIC,
BIC and uncertainty-dependent prediction or simulation are refused for active
priors. An unpenalized comparison requires unpenalized refits. Choosing the
prior depends on the response units: changing units requires changing its rate.
