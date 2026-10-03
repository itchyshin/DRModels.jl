# Temporal AR1, OU and Toeplitz effects

!!! note "Status — Tested (Gaussian mean, ML)"
    Mirrors drmTMB's *Temporal AR1, OU, and Toeplitz effects* article
    (`vignettes/temporal-random-effects.Rmd`). **In DRModels.jl today:** one
    intercept-only `temporal(1 | id, time, ar1)` or `temporal(1 | id, time, ou)`
    term on the Gaussian **mean**, with `sigma ~ 1`, fitted by ML, optionally
    alongside an ordinary `(1 | id)` on the same `id`, or (OU only) alongside a
    phylogenetic stable intercept `phylo(1 | species)` on the same grouping;
    or a homogeneous Toeplitz `temporal(1 | id, time, homtoep)` covariance on a
    complete, equally spaced panel.
    Other families, temporal terms on `sigma`, slopes, REML and other
    structured terms are refused with an error. This page fits drmTMB's own two datasets and reproduces its
    numbers (table below).

!!! info "Attribution"
    The explanatory prose on this page is adapted from the drmTMB article
    *Temporal AR1 and OU random effects* (`vignettes/temporal-random-effects.Rmd`;
    its Toeplitz section from the development version of that article,
    *Temporal AR1, OU, and Toeplitz effects*) by its copyright holder, Shinichi
    Nakagawa, and is reused here under the MIT licence. The code, the
    Julia-specific text and the comparisons are new.

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
reference values below are drmTMB's full-precision output on the same data,
read from the parity cell `test/parity/temporal/vignette-ar1-ri/` (generated by
drmTMB 0.7.1 under R 4.6.1; the drmTMB commit is recorded in that cell's
`expected.meta.toml`).

```@example temporal
using TOML
parity_cell(name) = TOML.parsefile(pkgdir(DRModels, "test", "parity", "temporal",
                                          name, "expected.toml"))

# drmTMB's numbers for one cell, as (quantity => value) pairs.
function drmtmb_estimates(ref)
    t = ref["temporal"]
    persistence = haskey(t, "phi") ? "phi" => t["phi"] : "decay" => t["decay"]
    return ["logLik" => ref["fit"]["loglik"],
            [replace(k, "mu_" => "") => v for (k, v) in ref["coef"]]...,
            "sd_site" => t["sd_iid"], "sd_temporal" => t["sd"],
            persistence, "sigma" => t["sigma"]]
end

# The same quantities from a DRModels.jl fit, matched by name.
function julia_estimates(fit)
    tp = temporal_parameters(fit)
    names = Dict(fit.coefnames)[:mu]
    β = coef(fit, :mu)
    return Dict("logLik" => loglik(fit), (names .=> β)...,
                "sd_site" => tp.sd_iid, "sd_temporal" => tp.sd,
                "phi" => tp.phi, "decay" => tp.decay, "sigma" => tp.sigma)
end

compare(ref, fit) = [(quantity = k, drmTMB = round(r; digits = 4),
                      DRModels = round(julia_estimates(fit)[k]; digits = 4),
                      abs_diff = abs(julia_estimates(fit)[k] - r))
                     for (k, r) in drmtmb_estimates(ref)]

ar1_ref = parity_cell("vignette-ar1-ri")
compare(ar1_ref, fit)
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
drmtmb_fitted = Float64.(ar1_ref["conditional"]["fitted"])   # drmTMB's fitted(fit)
(rows = length(conditional), first_rows = first(conditional, 3),
 drmTMB_first_rows = first(drmtmb_fitted, 3),
 max_abs_diff = maximum(abs.(conditional .- drmtmb_fitted)))
```

All 288 conditional values agree with drmTMB's `fitted(fit)` to the
`max_abs_diff` shown. That agreement is looser than the agreement of the
estimates because of drmTMB's side: for this fit, the modes drmTMB stores are
not exactly the modes at its own reported optimum, while DRModels.jl's modes
are exact. The residuals below inherit the same small difference.

Standardised residuals use the same conditional convention as drmTMB's
Pearson and quantile residuals. `residuals(fit; type = :quantile)` is
``(y - \hat\mu^{\mathrm{cond}})/\hat\sigma``, where
``\hat\mu^{\mathrm{cond}}`` is the conditional mean above. This is the
estimated residual noise left after the fitted site and temporal deviations.
It is not whitened against the correlation within a site, and the deviations
are fitted to the same data, so its spread is below 1 even when the model is
correct. Use it to look for outliers and for patterns against covariates or
time; do not read its spread as a calibration check. Check
`check_drm(fit).temporal_boundary` first: at the residual-SD boundary
(`sigma_ratio`, described below) the temporal path absorbs the data, so as
``\hat\sigma`` goes to 0 these residuals collapse toward 0 and cannot reveal
outliers. (`residuals(fit)` stays ``y - X\hat\beta``, matching `fitted`.)

```@example temporal
z = residuals(fit; type = :quantile)
drmtmb_pearson = Float64.(ar1_ref["residuals"]["pearson"])   # drmTMB's residuals(fit, type = "pearson")
(rms = sqrt(sum(abs2, z) / length(z)), first_rows = first(z, 3),
 drmTMB_first_rows = first(drmtmb_pearson, 3),
 max_abs_diff = maximum(abs.(z .- drmtmb_pearson)))
```

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
wald = confint(fit; parm = :mu)
```

The same Wald limits from drmTMB (`confint(fit, method = "wald")`, stored in
the parity cell), side by side:

```@example temporal
# Pair DRModels.jl interval rows with drmTMB rows ("fixef:mu:<coef>").
function compare_ci(julia_rows, drmtmb_rows)
    map(drmtmb_rows) do r
        coefname = replace(r["parm"], "fixef:mu:" => "")
        j = only(filter(x -> x.coef == coefname, julia_rows))
        (coef = coefname, lower = j.lower, upper = j.upper,
         drmTMB_lower = r["lower"], drmTMB_upper = r["upper"],
         max_abs_diff = max(abs(j.lower - r["lower"]), abs(j.upper - r["upper"])))
    end
end
compare_ci(wald, ar1_ref["wald"])
```

DRModels.jl's `vcov`, `stderror` and `coeftable` also print Wald standard
errors for the variance and persistence coordinates (on their working scales),
because every DRModels.jl route does. drmTMB does not report them, and their
calibration has not been established in either package; do not use them as
intervals for the temporal parameters.

Likelihood profiles are available for the mean coefficients of both AR1 and OU
fits. Use them when reporting a treatment or other regression effect:

```@example temporal
compare_ci(confint(fit; method = :profile, parm = :mu => "treatment"), ar1_ref["profile"])
```

Each package locates profile endpoints with its own numerical root search, so
expect profile limits to agree less tightly than the Wald limits; the
differences on this page are measured, not asserted.

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

## Free correlation by discrete lag with homogeneous Toeplitz

Use homogeneous Toeplitz covariance when every site is measured at the same
complete, equally spaced set of discrete occasions and the scientific question
is whether correlation departs from AR1's exponential pattern. It estimates a
separate correlation for each lag: visits one occasion apart share
`cor_lag1`, visits two occasions apart share `cor_lag2`, and so on. This is
more flexible than AR1, so it needs a common panel rather than the irregular
or incomplete schedules accepted by AR1 and OU. In Julia the structure name is
`homtoep`:

| drmTMB (R)                                                     | DRModels.jl (Julia)                        |
|:---------------------------------------------------------------|:-------------------------------------------|
| `temporal(1 \| site, time = occasion, structure = "homtoep")`  | `temporal(1 \| site, occasion, homtoep)`    |

drmTMB's article simulates 80 sites, each visited at the same six equally
spaced occasions (`set.seed(20261002)`), with a treatment that alternates
between visits. Its lag correlations (0.60, 0.45, 0.40, 0.30, 0.20) decay
more slowly than an AR1 pattern with the same first-lag value would (0.60,
0.36, 0.22, 0.13, 0.08), and the total SD is 0.80. We read the exact data it
produced.

```@example temporal
col = read_fixture("vignette_homtoep_sites.csv")
panel = (y         = parse.(Float64, col("y")),
         treatment = parse.(Float64, col("treatment")),
         site      = String.(col("site")),
         occasion  = parse.(Int, col("occasion")))   # integer occasions

toeplitz_fit = drm(bf(@formula(y ~ treatment + temporal(1 | site, occasion, homtoep)),
                      @formula(sigma ~ 1)),
                   Gaussian(); data = panel)
toeplitz_tp = temporal_parameters(toeplitz_fit)
(sigma = toeplitz_tp.sigma, cor_lag = round.(toeplitz_tp.cor; digits = 3))
```

Here `sigma` is the total within-site SD of the Toeplitz covariance. The model
does not separately estimate a temporal-process SD, an independent residual SD
or an ordinary `(1 | site)` intercept, because those components are not
separately identifiable when the correlation at every lag is free; both
packages refuse `(1 | site)` beside `homtoep`. The lag correlations are
estimated through their partial autocorrelations (`coef(toeplitz_fit,
:temporal_pac)`, on the atanh scale), which keeps every fitted correlation
matrix positive definite. There are no latent temporal states: `fitted` is
``X\hat\beta`` in both packages, and `simulate` draws each site's six values
jointly from the fitted ``\sigma^2 R``.

The same model fitted by drmTMB to the same data (parity cell
`test/parity/temporal/vignette-homtoep/`; drmTMB's article prints these
estimates rounded to four decimals):

```@example temporal
toep_ref = parity_cell("vignette-homtoep")
toep_names = Dict(toeplitz_fit.coefnames)[:mu]
[(quantity = q, drmTMB = round(r; digits = 4), DRModels = round(j; digits = 4),
  abs_diff = abs(j - r))
 for (q, r, j) in [("logLik", toep_ref["fit"]["loglik"], loglik(toeplitz_fit)),
                   [(k, v, coef(toeplitz_fit, :mu)[findfirst(==(replace(k, "mu_" => "")), toep_names)])
                    for (k, v) in toep_ref["coef"]]...,
                   ("sigma", toep_ref["temporal"]["sigma"], toeplitz_tp.sigma),
                   [("cor_lag$m", r, toeplitz_tp.cor[m])
                    for (m, r) in enumerate(toep_ref["temporal"]["cor"])]...]]
```

drmTMB qualifies only likelihood-profile intervals for mean regression effects
of this model (its article computes this one with `profile_engine =
"tmbprofile"`):

```@example temporal
compare_ci(confint(toeplitz_fit; method = :profile, parm = :mu => "treatment"), toep_ref["profile"])
```

A retained 4,000-fit drmTMB campaign qualified these mean-effect profiles in
three predeclared 80-site, six-occasion panels: AR1-shaped, non-exponential,
and negative first-lag correlation patterns. The 20-site stress panel had
lower intercept coverage, so this is evidence for those primary panel
designs, not a coverage claim for every Toeplitz analysis, and it was obtained
with drmTMB: DRModels.jl reproduces the same likelihood, but no separate
calibration study has been run on its intervals. Both packages refuse
everything else: Wald covariance and intervals (`vcov`, `stderror`,
`confint(fit)`, `predict(...; se = true)`), and profile intervals for `sigma`
or the lag correlations. `confint(fit; method = :profile)` without `parm`
profiles the mean coefficients only, and `coeftable` prints the standard-error
columns as `NaN`. If elapsed gaps are genuinely irregular, use OU instead.

Standardised residuals account for the correlation within a site:
`residuals(toeplitz_fit; type = :quantile)` returns the whitened
``L^{-1}(y - X\hat\beta)``, with ``L`` the Cholesky factor of each site's
``\hat\sigma^2 R``. These are drmTMB's Pearson residuals for this model.

The panel rules are drmTMB's: integer occasions, at least 3 and at most 12
common occasions, equally spaced, every site observing all of them, and at
least as many sites as occasions (the K − 1 free lag correlations need that
many independent series to be estimable at all; this is a floor, not a design
recommendation). Missing responses are handled as drmTMB handles them. The
rows with a missing response are dropped first, and then the panel rules
apply to the rows that remain. A site that loses one occasion is therefore
refused by name as incomplete, and is never silently dropped. An occasion
that is missing for every site leaves a smaller common panel, which still
fits if it stays equally spaced.

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
ou_ref = parity_cell("vignette-ou-ri")
compare(ou_ref, ou_fit)
```

For OU, `elapsed_days` must be finite and numeric, genuine gaps are retained,
and duplicate site–time records must be aggregated before fitting. `decay` is
the positive rate in the units of `elapsed_days`: if you change days to hours,
the numerical rate changes by the inverse factor while the fitted correlation
over a physical gap does not.

drmTMB withholds OU Wald inference altogether and recommends a profile
interval for a mean effect:

```@example temporal
compare_ci(confint(ou_fit; method = :profile, parm = :mu => "treatment"), ou_ref["profile"])
```

drmTMB attaches a condition to reporting this interval: report it only when
`check_drm(ou_fit)` raises no `temporal_mean_profile` warning and the
interval's `conf.status` shows no problem; finite endpoints alone are not
calibration evidence. On these data drmTMB's check gives:

```@example temporal
ou_ref["check_drm"]
```

The status is a `note` (profile intervals are available for this fit, with
calibration unqualified), not a warning, so by drmTMB's rule the interval may
be reported, without a coverage claim.

DRModels.jl has **no equivalent check yet**: its `check_drm` does not assess
temporal profile intervals. Before reporting a DRModels.jl OU profile
interval, run drmTMB's check on the same model, or treat the interval as
uncalibrated.

drmTMB qualified fixed-mean profile intervals for OU in a retained 3,000-fit
campaign across three predeclared scenarios (80 sites measured at 6 or 12
irregular occasions, with and without stable site intercepts). That result
does not establish general temporal coverage, it was obtained with drmTMB,
and this page does not extend it: DRModels.jl reproduces the same likelihood,
but no separate calibration study has been run on its intervals. OU decay and
variance-component intervals are not available.

## Phylogenetic stable effects with OU deviations

!!! info "Attribution"
    This section adapts the drmTMB article *Phylogenetic stable effects and
    temporal OU deviations* (`vignettes/phylogenetic-temporal-effects.Rmd`,
    drmTMB development version) by its copyright holder, Shinichi Nakagawa, under
    the MIT licence.

A comparative time-series data set can contain three different kinds of
variation. Closely related species can have similar *stable* baselines because
of shared evolutionary history. Each species can also make a short-lived
departure from its baseline that persists across nearby observation times.
Finally, observations have independent residual noise. The paired
phylogenetic–temporal model keeps these quantities separate. For observation
time ``t`` in species ``i``,

```math
y_{it} = x_{it}^{\top}\beta + b_i + a_{it} + \epsilon_{it},\qquad
\mathbf b \sim \mathcal N(\mathbf 0, s_b^2 A),\qquad
\operatorname{Cov}(a_{it}, a_{js}) = \mathbb 1_{i=j}\, s_a^2 e^{-\lambda |t-s|},
```

with ``\epsilon_{it} \sim \mathcal N(0, \sigma^2)``. Here ``A`` is the tip
correlation matrix of the tree. The stable SD ``s_b`` describes
between-species variation that follows the tree, ``s_a`` the stationary SD of
an OU departure *within the same species*, ``\lambda`` its decay rate in the
units of the elapsed-time column, and ``\sigma`` the observation noise.

The model is additive. It does **not** say that a temporal departure is shared
between related species: for two different species the covariance is
``s_b^2 A_{ij}``, even when their observations are close together in time. A
separable phylogeny-by-time field is a different model with a different
scientific question, and neither package fits it yet.

The data are those simulated by drmTMB's article (`set.seed(20260909)`): 60
species on a random coalescent tree, each observed at the genuinely irregular
elapsed times 0, 0.5, 2.5 and 5 with an alternating treatment, simulated with
`sd_phylo_stable = 0.45`, `sd_temporal = 0.55`, `decay_temporal = 0.45`,
`sigma = 0.35`, intercept 1 and treatment effect 0.4. As above, we read the
exact data and tree it produced.

```@example temporal
col = read_fixture("vignette_phylo_ou_species.csv")
dat_phylo = (y            = parse.(Float64, col("y")),
             treatment    = parse.(Float64, col("treatment")),
             species      = String.(col("species")),
             elapsed_days = parse.(Float64, col("elapsed_days")))
tree = read(pkgdir(DRModels, "test", "fixtures", "temporal",
                   "vignette_phylo_ou_species.newick"), String)

phylo_fit = drm(bf(@formula(y ~ treatment + phylo(1 | species) +
                                temporal(1 | species, elapsed_days, ou)),
                   @formula(sigma ~ 1)),
                Gaussian(); data = dat_phylo, tree = tree)
phylo_tp = temporal_parameters(phylo_fit)
```

The two random terms must use the same `species` column, and the tree is
passed to `drm` (drmTMB writes it inside the term, `phylo(1 | species, tree =
tree)`). The three variance components are reported separately:
`sd_phylo` is drmTMB's `sd_phylo_stable`, `sd` its `sd_temporal`, and `decay`
its `decay_temporal`. Like drmTMB, DRModels.jl uses the tree's tip
**correlation** matrix, so rescaling every branch length leaves the fit
unchanged.

```@example temporal
phylo_ref = parity_cell("vignette-phylo-ou")
pt = phylo_ref["temporal"]
phylo_names = Dict(phylo_fit.coefnames)[:mu]
[(quantity = q, drmTMB = round(r; digits = 4), DRModels = round(j; digits = 4),
  abs_diff = abs(j - r))
 for (q, r, j) in [("logLik", phylo_ref["fit"]["loglik"], loglik(phylo_fit)),
                   [(k, v, coef(phylo_fit, :mu)[findfirst(==(replace(k, "mu_" => "")), phylo_names)])
                    for (k, v) in phylo_ref["coef"]]...,
                   ("sd_phylo_stable", pt["sd_phylo"], phylo_tp.sd_phylo),
                   ("sd_temporal", pt["sd"], phylo_tp.sd),
                   ("decay_temporal", pt["decay"], phylo_tp.decay),
                   ("sigma", pt["sigma"], phylo_tp.sigma)]]
```

The two packages reach the same maximum-likelihood optimum. With a smaller
panel (drmTMB's article mentions 8 or 16 species at these four times), the
residual SD can run to zero while the OU term absorbs it; a fit at that
boundary should not be read component by component.

drmTMB's `fitted()` adds both conditional components. In DRModels.jl they are
in `ranef`: `ranef(phylo_fit)[:species_phylo]` holds the stable effect of each
species (first-seen order) and `ranef(phylo_fit)[:species]` the OU departure
of each row.

```@example temporal
re = ranef(phylo_fit)
phylo_conditional = fitted(phylo_fit) .+ re[:species] .+
    re[:species_phylo][indexin(dat_phylo.species, unique(dat_phylo.species))]
(max_abs_diff = maximum(abs.(phylo_conditional .- Float64.(phylo_ref["conditional"]["fitted"]))),)
```

`residuals(phylo_fit; type = :quantile)` subtracts both conditional components
and divides by ``\hat\sigma``, as drmTMB's Pearson residuals do, so the
caveats for the AR1 residuals above apply here too.

`simulate(phylo_fit)` draws a new phylogenetic stable vector, a new
independent OU path for each species and new noise.

The pairing is intentionally narrow, and both packages refuse the same
departures from it: it needs `ou` (not `ar1`), an unlabelled
`phylo(1 | species)` intercept on the same grouping as `temporal()`, no
ordinary `(1 | species)` (the stable between-species component is already the
phylogenetic term), at least three species with at least two distinct times
each, at least three distinct positive lags across the data, tree tips that
are exactly the observed species, and an **ultrametric** tree (all tips at the
same depth, to drmTMB's relative `sqrt(eps)` tolerance). Zero-length branches
are fine.

Every temporal fit is also checked against drmTMB's boundary rules: a
random-effect SD (here `sd_phylo` or `sd`) below 1e-4, a residual SD below
1e-3 of the response SD, an OU decay that leaves correlation at about 1 or 0
at every observed lag, or AR1 persistence beyond ±0.999. Such a fit warns when
it is fitted and reports `check_drm(fit).temporal_boundary.at_boundary = true`;
do not read its variance components separately. As in drmTMB, the residual-SD
rule compares with the marginal SD of the response, so it can also fire when a
covariate explains most of that SD.

**What can be reported now.** drmTMB's article marks this model as a
development workflow: its parser, dense-likelihood, method and fixed-mean
profile checks pass, and a replacement point-recovery study met its
predeclared point criteria, but no interval-calibration study has been run,
so it asks readers not to report confidence intervals from the paired model
yet. DRModels.jl reproduces the same likelihood and makes no stronger claim:
use the fit to inspect the variance components, not for interval inference.
For the record only, the fixed-mean profile for `treatment` agrees between the
two packages. Profile `confint` on the paired fit warns that the interval is
not calibrated, as drmTMB's does; Wald (`confint(fit)`) and bootstrap
intervals do not warn, also as in drmTMB, so the same caution applies to them
without a reminder:

```@example temporal
compare_ci(confint(phylo_fit; method = :profile, parm = :mu => "treatment"), phylo_ref["profile"])
```

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
`test/parity/temporal/`). In the measured run (Julia 1.12.6 on Linux, Totoro),
the log-likelihood agreed with drmTMB to better than ``10^{-10}`` and every
coefficient, SD, persistence and decay to better than ``10^{-8}`` relative.
The tests enforce looser tolerances, ``10^{-8}`` absolute for the
log-likelihood and ``10^{-6}`` relative for the parameters, to leave room for
other platforms. The likelihood itself is also checked against
a dense multivariate-normal oracle (`test/test_temporal_ar1.jl`,
`test/test_temporal_ou.jl`).

Three cells cover homogeneous Toeplitz (drmTMB's article data above, a
40-site six-occasion panel with a non-exponential lag pattern, and a 30-site
four-occasion panel with a negative first-lag correlation), generated from
the drmTMB development version: logLik within ``10^{-11}``, β and σ within
``2 \times 10^{-11}`` relative and every lag correlation within
``5 \times 10^{-10}`` (Julia 1.10.12 and 1.13, Linux, Totoro). The Toeplitz likelihood is checked against a BigFloat dense oracle
to ``2 \times 10^{-15}`` relative, including partial autocorrelations near
±1, and its partial-autocorrelation parameterisation against dense Schur
complements (`test/test_temporal_homtoep.jl`).

Two more cells cover the paired phylogenetic + OU model: drmTMB's article data
above and a 24-species simulated fixture. On Julia 1.10.12 and 1.13 (Linux, Totoro)
they agree with drmTMB to about ``10^{-11}`` in logLik and to
``2 \times 10^{-9}`` (relative) or better in every estimate; the conditional
fitted values and the conditional residuals (`residuals(fit; type = :quantile)`
against drmTMB's Pearson residuals) agree to ``10^{-10}`` or better. The exact
figures vary a little with the Julia version and with bounds checking. Their numbers come from drmTMB
development commit `012258e9f` (recorded in each cell's
`expected.meta.toml`). The paired likelihood is checked against a dense oracle
(``9.2 \times 10^{-13}``, including the decay and stable-SD extremes and a
tree with zero-length branches), and the pruning pass alone on
non-ultrametric trees, in `test/test_temporal_phylo_ou.jl`.

## See also

- [What is tested](../capabilities.md) ·
  [What can I fit today?](../model-guides/model-map.md)
- [Profile likelihood intervals](../diagnostics-and-validation/profile-likelihood.md)
- [Structured dependence](structural-dependence.md)
