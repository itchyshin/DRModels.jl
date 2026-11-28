# Coefficient, residual-SD and random-intercept SD priors

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

The supported routes are Gaussian `mu` and `sigma` (with up to two distinct iid
mean random intercepts) and fixed-effect binomial-logit `mu`. Normal priors on
Gaussian `mu` and `sigma` coefficients also support one phylogenetic mean
intercept with matching phylogenetic SD regression and fixed residual-SD
predictors. These priors do not target the phylogenetic SD-regression
coefficients. This phylogenetic route currently uses the dense solver and
refuses the automatically selected sparse route for more than 500 groups.
Omit `weights` on this route; explicit weights are refused, including unit weights.
Random slopes, other structured combinations, REML, existing phylogenetic
penalties, missing responses or predictors, non-unit weights, and other
families are refused. Supply the tree again when updating a phylogenetic fit.
The R-to-Julia bridge for this route is still being checked. For separated
binomial data,
a finite optimum requires all separating directions to be regularized; a prior
on just one coefficient does not guarantee this.

For example, the following small data set demonstrates the phylogenetic route.
Its four tips are too few for a reliable scientific analysis; use it to inspect
the API, not to assess uncertainty.

```julia
using DRModels
phy = DRModels.augmented_phy("((a:0.5,b:0.5):0.5,(c:0.5,d:0.5):0.5);")
d = (y = [.2,.7,.35,1.1,.8,1.25,1.4,1.1,1.9,1.5,2.15,.5],
     x = [-1.2,-.8,-.7,-.2,.1,.15,.4,.6,.8,1.,1.2,-.3],
     sp = ["a","a","b","b","b","c","c","d","d","d","d","a"],
     z = [-1.,-1.,-.35,-.35,-.35,.4,.4,1.,1.,1.,1.,-1.])
f = bf(@formula(y ~ x + phylo(1 | sp)), @formula(sigma ~ x),
       @formula(sd(sp, phylogenetic) ~ z))
p = drm_priors(drm_prior_normal(dpar="mu", coef="x", mean=.1, sd=.7),
               drm_prior_normal(dpar="sigma", coef="x", mean=-.2, sd=.5))
resolve_priors(f, Gaussian(); data=d, priors=p)
fit = drm(f, Gaussian(); data=d, tree=phy, priors=p)
```

Active prior fits are labelled `:MAP`. `loglik(fit)` is the data log-likelihood
at the penalized estimate. The stored objective includes the prior. Its inverse
Hessian describes local curvature of that penalized objective; it is neither an
automatic frequentist covariance nor a full posterior interval. Likelihood
ratio tests, AIC/BIC, confidence intervals, likelihood profiles and bootstrap
refits are refused for these fits. Prediction standard errors and simulation are
not yet qualified.

## Gamma prior on a random-intercept SD

A Gaussian ML model with one or two distinct unlabelled iid random intercepts
in `mu` can use Gamma densities on selected natural standard deviations. Supply each exact
name and inspect the resolved target before fitting:

```julia
sd_prior = drm_prior_sd_gamma(dpar = "mu", group = "site",
                              coef = "(Intercept)", shape = 2, rate = 3)
priors = drm_priors(sd_prior,
                    drm_prior_normal(dpar = "mu", coef = "x", mean = 0, sd = 2))
resolve_priors(group_form, Gaussian(); data = grouped, priors = priors)
fit = drm(group_form, Gaussian(); data = grouped, priors = priors)
```

For crossed intercepts, name each group to choose which SD to regularize:

```julia
crossed = merge(grouped, (observer = [1, 2, 1, 2, 2, 1, 2, 1],))
cross_form = bf(@formula(y ~ x + (1 | site) + (1 | observer)))
cross_priors = drm_priors(
    drm_prior_sd_gamma(dpar = "mu", group = "site", shape = 2, rate = 3),
    drm_prior_sd_gamma(dpar = "mu", group = "observer", shape = 2.5, rate = 2))
resolve_priors(cross_form, Gaussian(); data = crossed, priors = cross_priors)
cross_fit = drm(cross_form, Gaussian(); data = crossed, priors = cross_priors)
```

`shape` must be finite and greater than one, and `rate` must be finite and
positive. The Gamma density is with respect to the SD itself; it does not add
a log-SD Jacobian. The normalized penalty is added once for the shared SD,
not once per group. The fit is penalized ML, or SD-scale MAP: `loglik(fit)`
remains the data log likelihood at the penalized estimate. The fixed and SD
penalties and total objective are stored separately in `fit.penalty`.
`update` repeats the prior unless explicitly replaced or dropped by changing
the estimator.

This route requires complete observations, unit weights and distinct iid
mean random-intercept groups; random slopes, more than two or structured blocks and REML
are outside this slice. Write the random intercept only once in the formula.
Two grouping columns that encode the same partition are refused because their
separate SDs cannot be identified from the likelihood.
Currently, Julia's `@formula` collapses repeated identical terms such as
`(1 | site) + (1 | site)` before the prior target is checked. The resolver
therefore cannot reject that spelling as a duplicate; drmTMB in R rejects it.
This difference remains unresolved.

Confidence intervals, likelihood comparisons, AIC, BIC and
uncertainty-dependent prediction or simulation are refused for active priors.
`bootstrap_result(fit; data = ...)` warns that its intervals describe the
penalized estimator and that bootstrap coverage has not been validated. An unpenalized comparison
requires unpenalized refits. Choosing the prior depends on the response units:
changing units requires changing its rate.

## Specifying priors in a table

A Tables.jl-compatible column table uses `prior`, `class` and `coef`, with
optional `dpar`, `group`, `scale`, `from_coef` and `to_coef` columns. The `scale` column defaults to
`link` for `fixef` and `natural_sd` for `ranef_sd`; a `residual_sd` row must
state `scale = "natural_sd"` explicitly. This interface follows a documented
subset of glmmTMB's prior-table style; it does not accept every glmmTMB prior
class or distribution. For the crossed model above, the following table
regularizes the fixed slope and the `site` random-intercept SD:

```julia
prior_table = (prior = ["normal(0,2)", "gamma(0.6666666666666666,2)"],
               class = ["fixef", "ranef_sd"],
               coef = ["x", "site"])
resolve_priors(cross_form, Gaussian(); data = crossed, priors = prior_table)
table_fit = drm(cross_form, Gaussian(); data = crossed, priors = prior_table)
```

For `fixef`, `normal(mean, sd)` and `student_t(location, scale, df)` are
normalized densities; `ridge(lambda)` is an unnormalized quadratic penalty.
The Student-t degrees of freedom and scale must be finite and positive. A blank
`coef` selects all non-intercept coefficients; use `(Intercept)` to select the
intercept.
A coefficient literally named `all` cannot be selected by a prior table,
including through a blank selector, because that name is reserved for a
constructor wildcard. Optional `dpar` selects `mu` or the supported Gaussian `sigma` predictor and
defaults to `mu`.

The table form and direct constructor use the same Student-t density:

```@example table_student_t
using DRModels
student_data = (y = [.2, .8, .5, 1.4, 1.0, 1.6, 2.0, 1.3, 2.2, 1.8, 2.4, 2.1],
                x = [-1., 0., 1., -1., 0., 1., -1., 0., 1., -1., 0., 1.])
student_form = bf(@formula(y ~ x))
student_t_table = (prior = ["student_t(0,2,4)"], class = ["fixef"], coef = ["x"])
student_t_fit = drm(student_form, Gaussian(); data = student_data, priors = student_t_table)
(loglik = loglik(student_t_fit), method = student_t_fit.estim_method)
```

Both apply the normalized density to the fitted link-scale coefficient. The
penalty is a MAP objective contribution, not posterior uncertainty.

For `ranef_sd`, `gamma(mean, shape)` targets an SD on its natural scale.
This table convention uses **mean and shape**, whereas
`drm_prior_sd_gamma()` uses **shape and rate**: `rate = shape / mean`. The
shape must exceed one. With no `group` column, `coef = "site"` selects the
`site` intercept; a blank `coef` selects every supported random-intercept SD.
Alternatively, specify `group = "site"` and `coef = "(Intercept)"`.
Overlapping, unknown and unsupported selections are rejected. Random-slope SD
priors, other structured effects and other families remain outside this SD slice. The original table is stored in `table_fit.penalty.input_table`; `update()`
replays the resolved targets rather than expanding a blank selector again.

For a Gaussian model with an intercept-only `sigma` formula, `residual_sd`
targets the single natural residual SD. State the target scale explicitly:

```@example residual_sd_prior
using DRModels
residual_data = (y = [.2, .8, .5, 1.4, 1.0, 1.6, 2.0, 1.3, 2.2, 1.8, 2.4, 2.1],
                 x = [-1., 0., 1., -1., 0., 1., -1., 0., 1., -1., 0., 1.])
residual_form = bf(@formula(y ~ x), @formula(sigma ~ 1))
residual_prior = (prior = ["gamma(0.9,2.4)"], class = ["residual_sd"],
                  coef = ["(Intercept)"], dpar = ["sigma"], scale = ["natural_sd"])
residual_fit = drm(residual_form, Gaussian(); data = residual_data, priors = residual_prior)
(residual_sd = first(sigma(residual_fit)),
 residual_variance = first(sigma(residual_fit))^2,
 method = residual_fit.estim_method)
```

Here `gamma(mean, shape)` is normalized with `rate = shape / mean` and density
with respect to `d(sigma)`. The prior is evaluated at `sigma = exp(eta_sigma)`
without a log-SD Jacobian; the reported Gaussian residual variance is
`sigma^2`. A sigma predictor such as `sigma ~ x` has coefficient targets on
the link scale and cannot use this scalar residual-SD row.

## Priors for correlations

Use `lkj(eta)` with `class = "correlation"` and `scale = "natural_correlation"` to regularize a supported correlation. The density is applied to the correlation itself, rather than its transformed optimization coordinate. The current implementation requires a finite `eta` greater than 1 and normalizes the density over the guarded correlation interval.

For the constant residual correlation of a bivariate Gaussian model, select `dpar = "rho12"` and `coef = "(Intercept)"`. The `sigma1` and `sigma2` formulas must be intercept-only for this route.

```julia
rho_prior = (prior = ["lkj(2)"], class = ["correlation"],
             dpar = ["rho12"], coef = ["(Intercept)"],
             scale = ["natural_correlation"])
```

For one ordinary Gaussian random-intercept and random-slope block, select `dpar = "mu"`, identify the group, and name the intercept and slope in `from_coef` and `to_coef`. The coefficient pair must match the fitted block. A blank `coef` distinguishes this pair target from the `rho12` intercept target.

```julia
slope_prior = (prior = ["lkj(2)"], class = ["correlation"],
               dpar = ["mu"], coef = [""], group = ["site"],
               from_coef = ["(Intercept)"], to_coef = ["x"],
               scale = ["natural_correlation"])
```

These examples describe the supported constant residual-correlation and single ordinary intercept-slope targets. Other correlation structures are rejected. As with the other active priors, the resulting fit is penalized; the prior does not supply posterior samples.
