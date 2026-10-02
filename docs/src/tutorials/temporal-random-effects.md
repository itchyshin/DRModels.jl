# Temporal AR1 and OU random effects

!!! note "Status — Tested (Gaussian mean, ML)"
    Mirrors drmTMB's *Temporal AR1 and OU random effects* article
    (`vignettes/temporal-random-effects.Rmd`). **In DRModels.jl today:** one
    intercept-only `temporal(1 | id, time, ar1)` or `temporal(1 | id, time, ou)`
    term on the Gaussian **mean**, with `sigma ~ 1`, fitted by ML, optionally
    alongside an ordinary `(1 | id)` on the same `id`. Other families, temporal
    terms on `sigma`, slopes, REML and other structured terms are refused with
    an error. This page fits drmTMB's own two datasets and reproduces its
    numbers (table below).

Repeated measurements can vary for three different reasons. Individuals or
sites may have stable baseline differences, their deviations may persist
through time, and each observation has independent residual noise. Gaussian
temporal AR1 and OU models separate those components while estimating the
effects of treatments or other predictors.

For observation ``t`` in series ``i``, the model is

```math
y_{it} = x_{it}^{\top}\beta + b_i + a_{it} + \epsilon_{it},
```

where ``b_i`` is the stable series intercept, ``a_{it}`` is a stationary
temporal process with covariance
``\operatorname{Cov}(a_{it}, a_{is}) = s_a^2 \phi^{|t-s|}``, and
``\epsilon_{it}`` is independent residual noise. The process SD ``s_a``, the
ordinary-intercept SD ``s_b`` and the residual SD ``\sigma`` measure different
sources of variation. Persistence ``\phi`` is the correlation one occasion
apart and can be negative or positive.

## Writing the term in Julia

drmTMB writes the term with keyword arguments,
`temporal(1 | site, time = occasion, structure = "ar1")`. Julia's `@formula`
cannot carry keyword arguments or strings inside a formula, so DRModels.jl
uses the same three pieces **by position**, with a bare structure name:

| drmTMB (R)                                                  | DRModels.jl (Julia)                     |
|:------------------------------------------------------------|:----------------------------------------|
| `temporal(1 \| site, time = occasion, structure = "ar1")`    | `temporal(1 \| site, occasion, ar1)`     |
| `temporal(1 \| site, time = elapsed_days, structure = "ou")` | `temporal(1 \| site, elapsed_days, ou)`  |

If you call DRModels.jl from R through `drm_bridge` (see [Coming from R?](../coming-from-r.md)),
you can keep drmTMB's keyword spelling: the bridge translates it to the
positional form (shown at the end of this page).

## A repeated-site example

The data are the ones drmTMB's article simulates: 48 sites visited on the
irregular integer occasions 0, 1, 3, 4, 6 and 7, with a treatment that
alternates between visits. The gap between occasions 1 and 3 remains a
two-occasion gap when the model is fitted. The simulation is R code
(`set.seed(20260908)`), so rather than re-simulate it in Julia we read the
exact data frame it produced, which ships with the package's test fixtures.

```@example temporal
using DRModels

# A tiny CSV reader, so this page needs nothing beyond DRModels.
function read_fixture(file)
    lines = readlines(pkgdir(DRModels, "test", "fixtures", "temporal", file))
    header = split(lines[1], ',')
    rows = split.(lines[2:end], ',')
    return name -> [r[findfirst(==(name), header)] for r in rows]
end

col = read_fixture("vignette_ar1_sites.csv")
dat = (y         = parse.(Float64, col("y")),
       treatment = parse.(Float64, col("treatment")),
       occasion  = parse.(Int, col("occasion")),      # AR1 needs integer occasions
       site      = String.(col("site")))
(n = length(dat.y), sites = length(unique(dat.site)), occasions = sort(unique(dat.occasion)))
```

Fit an ordinary random intercept and the temporal process together. Both
terms use `site`; the first captures stable site-to-site differences and the
second captures deviations through time within each site. Here `treatment`
compares treated and untreated visits within sites after accounting for a
linear occasion trend.

```@example temporal
fit = drm(bf(@formula(y ~ treatment + occasion + (1 | site) +
                          temporal(1 | site, occasion, ar1)),
             @formula(sigma ~ 1)),
          Gaussian(); data = dat)
coeftable(fit)
```

The temporal process and the variance components are collected by
[`temporal_parameters`](@ref): the process SD (`sd`), the persistence (`phi`,
or `decay` for OU), the ordinary-intercept SD (`sd_iid`) and the residual SD
(`sigma`), all on their natural scales.

```@example temporal
tp = temporal_parameters(fit)
```

The mean-model rows report the intercept, treatment and occasion effects.
Read a positive persistence estimate as deviations tending to continue in the
same direction over adjacent occasions; read a negative value as alternating
deviations.

### Same numbers as drmTMB

The drmTMB article prints its key estimates rounded to four decimals. The
reference values below are drmTMB's full-precision output on the same data
(drmTMB 0.7.1, R 4.6.1; recorded in `test/parity/temporal/vignette-ar1-ri/`).

```@example temporal
drmtmb_ar1 = [
    "logLik"         => -345.34482899469231,
    "(Intercept)"    => 1.0837697434675801,
    "treatment"      => 0.49554761766962618,
    "occasion"       => 0.063853165014370952,
    "sd_site"        => 0.56241063630758481,
    "sd_temporal"    => 0.58154558894996577,
    "phi"            => 0.67287364953874829,
    "sigma"          => 0.51841259520806404,
]
β = coef(fit, :mu)
julia_ar1 = [loglik(fit), β..., tp.sd_iid, tp.sd, tp.phi, tp.sigma]

compare(ref, est) = [(quantity = k, drmTMB = round(r; digits = 4),
                      DRModels = round(e; digits = 4), abs_diff = abs(e - r))
                     for ((k, r), e) in zip(ref, est)]
compare(drmtmb_ar1, julia_ar1)
```

Every difference is far below the fourth decimal: the two packages reach the
same maximum-likelihood optimum.

### Fitted values and conditional effects

One convention differs from drmTMB. In DRModels.jl, `fitted(fit)` (and
`predict`) is the **population-level** mean ``X\hat\beta``, the convention for
every structured term in this package; drmTMB's `fitted()` adds the estimated
site and temporal deviations. Those conditional deviations are available from
`ranef`: `ranef(fit)[:site]` holds the temporal deviation for each row (in data
order) and `ranef(fit)[:site_iid]` the stable intercept for each site.

```@example temporal
conditional = fitted(fit) .+ ranef(fit)[:site] .+
              ranef(fit)[:site_iid][indexin(dat.site, unique(dat.site))]
drmtmb_head_fitted = [0.342813400738, 1.208231134835, 1.092193017083,
                      1.972908249725, 1.079472727340, 1.639981963517]
(DRModels = first(conditional, 6), drmTMB = drmtmb_head_fitted,
 max_abs_diff = maximum(abs.(first(conditional, 6) .- drmtmb_head_fitted)))
```

These conditional values agree with drmTMB's `head(fitted(fit))` to about
``10^{-7}``, looser than the ``10^{-11}`` of the estimates because the two
packages compute the conditional modes by different numerical routes.

`simulate(fit)` draws a new realization from the fitted model: fresh stable
and temporal effects plus residual noise, as drmTMB's default `simulate()`.

```@example temporal
using Random
first(simulate(fit; rng = Xoshiro(1)), 6)
```

## What the intervals cover

drmTMB deliberately exposes Wald intervals only for the **mean coefficients of
an AR1 fit**, using the full observed marginal-likelihood Hessian, and treats
the process SD, ordinary-intercept SD, persistence or decay and residual SD as
point estimates. Read DRModels.jl the same way:

```@example temporal
confint(fit; parm = :mu)
```

DRModels.jl's `vcov`, `stderror` and `coeftable` also print Wald standard
errors for the variance and persistence coordinates (on their working scales),
because every DRModels.jl route does. drmTMB does not report them, and their
calibration has not been established in either package; do not use them as
intervals for the temporal parameters.

Likelihood profiles are available for the mean coefficients of both AR1 and OU
fits. Use them when reporting a treatment or other regression effect:

```@example temporal
confint(fit; method = :profile, parm = :mu => "treatment")
```

On these data the Wald limits match drmTMB's `confint(fit, method = "wald")`
(treatment: 0.3530935, 0.6380017) to better than ``10^{-9}``, and the profile
limits match drmTMB's `method = "profile"` (0.3521367, 0.6391744) to within the
profile root-search tolerance, about ``10^{-5}``.

No coverage is claimed for AR1 intervals of either kind. drmTMB's first AR1
calibration pilot found a residual-SD boundary in a primary short-series case,
so its AR1 profile and Wald intervals carry no coverage claim, and DRModels.jl
makes none either. Variance-component, persistence or decay intervals,
forecasting and `newdata` prediction of the temporal process are not
available.

Supply a finite integer time column for AR1, keep its real gaps, and make sure
every site–occasion pair is unique; if a site was measured more than once at
an occasion, aggregate those records before fitting. Rows may be in any order.
An AR1-only fit omits `(1 | site)` but keeps the same `temporal()` term.

## Irregular elapsed time with OU

Use an OU process when elapsed intervals carry meaning, such as visits at 0,
0.5, 2.5 and 5 days. It keeps the same three-way variance separation but
estimates a positive decay rate, so the correlation over a gap ``d`` is
``\exp(-\lambda d)``. It cannot represent alternating, negative temporal
correlation. Again we load drmTMB's simulated data (`set.seed(20260909)`, 24
sites).

```@example temporal
col = read_fixture("vignette_ou_sites.csv")
dat_irregular = (y            = parse.(Float64, col("y")),
                 treatment    = parse.(Float64, col("treatment")),
                 site         = String.(col("site")),
                 elapsed_days = parse.(Float64, col("elapsed_days")))

ou_fit = drm(bf(@formula(y ~ treatment + (1 | site) +
                             temporal(1 | site, elapsed_days, ou)),
                @formula(sigma ~ 1)),
             Gaussian(); data = dat_irregular)
ou_tp = temporal_parameters(ou_fit)
```

```@example temporal
drmtmb_ou = [
    "logLik"      => -102.58057815043882,
    "(Intercept)" => 0.84940947165378300,
    "treatment"   => 0.37134308754180773,
    "sd_site"     => 0.59768102547432966,
    "sd_temporal" => 0.67845554172768419,
    "decay"       => 0.49279374267021420,
    "sigma"       => 0.19011181501037316,
]
julia_ou = [loglik(ou_fit), coef(ou_fit, :mu)..., ou_tp.sd_iid, ou_tp.sd, ou_tp.decay, ou_tp.sigma]
compare(drmtmb_ou, julia_ou)
```

For OU, `elapsed_days` must be finite and numeric, genuine gaps are retained,
and duplicate site–time records must be aggregated before fitting. `decay` is
the positive rate in the units of `elapsed_days`: if you change days to hours,
the numerical rate changes by the inverse factor while the fitted correlation
over a physical gap does not.

drmTMB withholds OU Wald inference altogether and recommends a profile
interval for a mean effect:

```@example temporal
confint(ou_fit; method = :profile, parm = :mu => "treatment")
```

drmTMB reports (0.1835334, 0.5452362) for the same profile interval.

drmTMB qualified fixed-mean profile intervals for OU in a retained 3,000-fit
campaign across three predeclared scenarios (80 sites measured at 6 or 12
irregular occasions, with and without stable site intercepts). That result
does not establish general temporal coverage, it was obtained with drmTMB,
and this page does not extend it: DRModels.jl reproduces the same likelihood,
but no separate calibration study has been run on its intervals. OU decay and
variance-component intervals are not available.

## The drmTMB spelling through the bridge

An R user calling DRModels.jl through `drm_bridge` (see [Coming from R?](../coming-from-r.md)) can
send drmTMB's own keyword spelling as a formula string. The bridge translates
it and reaches the same optimum:

```@example temporal
out = drm_bridge(; formula = "y ~ treatment + (1 | site) + " *
                     "temporal(1 | site, time = elapsed_days, structure = \"ou\"); sigma ~ 1",
                 family = "gaussian", data = dat_irregular)
(bridge_loglik = out["loglik"], native_loglik = loglik(ou_fit))
```

## Evidence

Eight drmTMB parity cells — AR1 and OU, each with and without `(1 | id)`, on
these two datasets and on two further simulated fixtures — are checked on
every test run (`test/test_parity_temporal.jl`, cells in
`test/parity/temporal/`). Across them the log-likelihood agrees with drmTMB to
better than ``10^{-10}`` and every coefficient, SD, persistence and decay to
better than ``10^{-8}`` relative. The likelihood itself is also checked against
a dense multivariate-normal oracle (`test/test_temporal_ar1.jl`,
`test/test_temporal_ou.jl`).

## See also

- [What is tested](../capabilities.md) ·
  [What can I fit today?](../model-guides/model-map.md)
- [Profile likelihood intervals](../diagnostics-and-validation/profile-likelihood.md)
- [Structured dependence](structural-dependence.md)
