# Estimate-parity audit beyond logLik, 2026-11-09

Reader: whoever owns the R-Julia parity lane. Purpose: the wave-10 gap map says Julia and R agree on log-likelihood in 260 of 3008 cells. A user also reads coefficients, standard errors, intervals, predictions, simulations and residuals. This note checks those on a stratified sample of the 260 cells and says which differences matter. Verification only: no Julia source or test was edited. Tip measured: `0bfcde2bb0ec534bd81d3a759a509722bb463efd` (PR #1272), Julia 1.10.12, against drmTMB 0.8.1 (`engine = "tmb"`).

## Bottom line

- **90 cells, all fit on both sides** (every family, every route, 76 ML and 14 REML). In 81 of them every compared quantity agrees: coefficients to 1.5e-6 at worst (median 2.5e-9), Hessian standard errors to 0.3% at worst (all but two cells within 0.02%), response and link predictions to 3e-6, Wald intervals to 4e-3 (median 7.5e-8), random-effect SDs to 2e-7. The parameter count is identical to R in all 90 and AIC and BIC agree to 5e-6, REML included.
- **8 cells sit on a boundary** (a random-effect SD near zero, all binomial or NB2 structured routes). Both sides report the same log-likelihood to 3e-6 but the SD coefficient differs by up to 6.6 on the log scale and the standard errors differ by factors up to 5e13. R flags 8 of the 8 as not converged; Julia's `is_converged` is false in 7 of them. Nothing to fix, but nothing to compare either.
- **Two real defects**, both with reproducers in `estimate-parity-audit-20261109/repro/`:
  1. **Tweedie with crossed random intercepts: every Julia standard error is `Inf`** and `is_converged` is false, while R has a positive-definite Hessian and finite SEs. Same log-likelihood and estimates (3.6e-6). A user gets no SEs and no intervals.
  2. **Gaussian independent terms `(1 | g) + (0 + x1 | g) + (0 + x2 | g)`: `re_sd()` and `vc()` return one SD instead of three.** The three `:resd` coefficients all carry the label `g`, so the dictionary keeps the last. A proposed one-line patch is in the same folder, not applied.
- **Everything else that differs is a convention**, most of it already in the Julia docstrings: `fitted()` and response residuals, quantile residuals, LogNormal `predict(type = "response")`, which simulation modes exist, and which routes refuse `re_sd()`. The user-facing ones are ranked below.
- Not done: the bridge comparison, the simulation noise check and the patch check, all blocked by Totoro load (see "Not verified").

## What was compared

Sample: `harness/sample_cells.py` (seed 20261109) takes one cell from each of the 55 family, route and estimator strata of the 260 agreeing cells, then 35 more that add variety in SD pattern, number of slopes and labelling. Result: 90 cells (Gaussian 34, Poisson 10, Binomial 9, Gamma 9, Beta 8, NB2 8, LogNormal 5, Tweedie 4, Student 3; ordinary 10, slope 3, indep 8, crossed 11, animal 10, relmat 16, phylo 22, spatial 10).

The harness reads the wave-8 `data/` folder (drmTMB branch `claude/gap-map-wave8-20261108`, `docs/dev-log/parity-crosswalk-evidence-20261107/fuzz-gap-map-wave8/data`); only the six files the reproducers need are copied into this folder. Cells, data, builders and thresholds are the wave-8 harness's, unchanged (`harness/ep_cell_r.R`, `ep_cell_j.jl` are cut from `run_r.R` and `run_julia.jl`). Each side fits the cell once and dumps everything to one long TSV (`harness/ep_r.R`, `ep_j.jl`); `compare_ep.py` matches and differences them.

| Quantity | R call | Julia call | How matched |
| --- | --- | --- | --- |
| Coefficients and Hessian SEs | `summary(fit)$coefficients` | `coef`, `stderror` | by distributional parameter (`mu`, `sigma`, `nu`, any SD-formula block), in order inside each |
| SDs, correlations, SEs | `summary(fit)$parameters` | `vc(fit)`, `re_sd(fit)`; SEs by numerical delta method on `vcov` | by order within sd and cor |
| Information criteria | `logLik` df, `AIC`, `BIC` | `dof`, `aic`, `bic` | direct |
| Wald intervals | `confint(fit)` | `confint(fit)` | like coefficients |
| Profile intervals (10 cells) | `confint(fit, parm, method = "profile")` | `confint(fit; method = :profile)` | like coefficients |
| Predictions | `predict(fit, newdata)`, 8 in-sample rows | `predict(fit, newdata)` | direct, both scales |
| Fitted, residuals | `fitted`, `residuals(type = "response" / "quantile")` | `fitted`, `residuals` | direct, all rows |
| Simulation | `simulate(fit, nsim = 1000)` | `simulate(fit; nsim = 1000)` | moments: grand mean, grand variance, mean per-observation variance, share of zeros |

Julia's random-effect parameters are Cholesky or log-SD entries in its coefficient table; R's are in a separate table. They are compared on the SD scale, not as coefficients. A Julia coefficient table can hold extra rows for SD coefficients that have no `sd()` formula (3 cells); those extra rows are ignored.

Compute: Totoro only, 3 R and 3 Julia processes at one thread each (6 cores of 150), `nice -n 10`. Estimate beforehand 30 to 60 minutes; actual 10 minutes for the main run. A second Julia pass (warnings, convergence flags, `re_sd`, simulation without a harness-only keyword) took a few minutes. I launched only when the one-minute load was 100 or below, with one exception: a 2-process seed check was started at load 396 because my launch line printed the load without testing it, and I killed it within a minute (no output).

## Agreement by family, route and estimator

Clean cells only: the 81 of 90 that are neither boundary cells nor the Tweedie case below. Each cell contributes its worst value. The SE column is the largest `|SE ratio − 1|` over the cell's coefficients (the larger SE over the smaller).

### By family

| Family | Cells | max abs logLik diff | max abs coef diff | max abs (SE ratio − 1) | max abs SD diff | max abs pred link diff | max abs pred response diff | max abs Wald bound diff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Beta | 8 | 6.7e-09 | 2.0e-07 | 1.5e-04 | 5.2e-09 | 3.7e-08 | 7.0e-09 | 8.4e-05 |
| Binomial | 3 | 1.0e-09 | 1.3e-07 | 0.00316 | n/a | 7.8e-08 | 1.7e-08 | 0.00354 |
| Gamma | 9 | 7.0e-09 | 9.4e-08 | 0 | 1.5e-10 | 1.7e-07 | 4.2e-07 | 2.2e-06 |
| Gaussian | 34 | 1.0e-09 | 1.3e-08 | 0.00123 | 9.9e-10 | 2.2e-08 | 2.2e-08 | 1.6e-06 |
| LogNormal | 5 | 1.0e-09 | 6.8e-10 | 0 | 3.2e-10 | 7.0e-11 | 2.72 (convention, below) | 1.5e-07 |
| NB2 | 6 | 8.0e-09 | 1.1e-07 | 2.0e-05 | 2.9e-08 | 2.0e-07 | 2.2e-07 | 2.2e-05 |
| Poisson | 10 | 1.0e-09 | 1.5e-06 | 0 | 1.8e-07 | 1.7e-06 | 3.0e-06 | 2.1e-06 |
| Student | 3 | 0 | 6.0e-11 | 0 | 2.9e-11 | 3.9e-11 | 3.9e-11 | 1.2e-06 |
| Tweedie | 3 | 0 | 6.0e-08 | 0 | n/a | 1.6e-08 | 3.0e-08 | 2.1e-06 |

### By route

| Route | Cells | max abs logLik diff | max abs coef diff | max abs (SE ratio − 1) | max abs SD diff | max abs pred link diff | max abs pred response diff | max abs Wald bound diff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| animal | 9 | 2.0e-09 | 1.3e-07 | 3.0e-05 | 1.0e-12 | 8.5e-09 | 1.5e-08 | 4.0e-05 |
| crossed | 10 | 8.0e-09 | 1.5e-06 | 0.00316 | 1.8e-07 | 1.7e-06 | 2.72 (LogNormal) | 0.00354 |
| indep | 8 | 3.0e-09 | 1.1e-07 | 1.0e-05 | 2.9e-11 | 3.0e-08 | 1.6e-07 | 1.5e-05 |
| ordinary | 10 | 0 | 1.2e-07 | 2.0e-05 | 0 | 6.9e-08 | 1.84 (LogNormal) | 2.2e-05 |
| phylo | 17 | 1.0e-09 | 1.4e-07 | 0.00123 | 5.0e-09 | 1.3e-07 | 1.63 (LogNormal) | 5.2e-05 |
| relmat | 16 | 3.0e-09 | 9.4e-08 | 1.5e-04 | 7.0e-12 | 3.3e-08 | 1.31 (LogNormal) | 8.4e-05 |
| slope | 3 | 0 | 1.0e-11 | 0 | 2.0e-12 | 2.0e-12 | 1.99 (LogNormal) | 1.5e-07 |
| spatial | 8 | 0 | 2.0e-07 | 0 | n/a | 2.0e-07 | 4.2e-07 | 2.2e-06 |

### By estimator

| Estimator | Cells | max abs logLik diff | max abs coef diff | max abs (SE ratio − 1) | max abs SD diff | max abs pred link diff | max abs pred response diff | max abs Wald bound diff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ML | 67 | 8.0e-09 | 1.5e-06 | 0.00316 | 1.8e-07 | 1.7e-06 | 2.72 (LogNormal) | 0.00354 |
| REML | 14 | 1.0e-09 | 1.3e-08 | 0.00123 | 9.9e-10 | 2.2e-08 | 2.2e-08 | 2.3e-07 |

"n/a" in the SD column means there was no SD to compare. `re_sd()` and `vc()` refuse when the model has an SD formula (65 of the 90 cells), so SD-formula blocks are compared through their coefficients, which are in the coefficient column. The other 25 cells have plain SDs: `vc()` gives them for 23 (21 clean), and `re_sd()` in the second pass agrees with R's SDs and SEs to 5 digits on the same cells. The exceptions are the two Gaussian independent-slope cells (defect 2) and the binomial boundary cell. **Correlations were not compared on the sample.** The sampled correlated blocks are either q of 3 or more or carry an SD formula, and there Julia's `vc()` refuses and the correlation sits in Cholesky entries. In an exploratory Gaussian ordinary q = 2 cell outside the sample the correlation and its SE agree with R to 1.7e-11 and a ratio of 1.

Reading of the 81: the differences are numerical noise from two different optimisers and Hessians, not from different models. Over 566 compared coefficient rows only 10 differ by more than 1e-6, the largest 1.5e-6 (Poisson crossed). The largest SE gaps (0.12% Gaussian phylo REML, 0.3% Binomial crossed with SD formulas) are well below anything a user would act on.

### Excluded cells

| Group | Cells | max abs logLik diff | max abs coef diff | max SE ratio | max abs pred link diff |
| --- | --- | --- | --- | --- | --- |
| Boundary or flat (R not converged, or an SD coefficient below −8) | 8 | 2.6e-06 | 6.62 | 5.0e+13 | 1.8e-07 |
| Julia unconverged while R converged (Tweedie crossed) | 1 | 1.0e-09 | 3.6e-06 | Inf | 2.7e-07 |

Even on the boundary cells the predictions agree (1.8e-7), since the fixed effects are well determined. The boundary cells: Binomial phylo (4 cells; the log-SD coefficient is −11.6 to −12.0 in R and −13.9 to −18.3 in Julia, i.e. an SD of about 1e-5 to 1e-8), Binomial spatial (2), NB2 animal q = 2 (1), NB2 phylo q = 2 SD formula (1).

## Classification of differences, ranked by user impact

| Rank | Class | What differs | Cells | Size | Documented? |
| --- | --- | --- | --- | --- | --- |
| 1 | (iii) defect | Tweedie crossed: all Julia SEs are `Inf`, `is_converged` false, `coeftable` SEs NaN; R finite | 1 of 1 sampled | all SEs | no |
| 2 | (iii) defect | Gaussian `(1 \| g) + (0 + x \| g) + ...`: `re_sd` and `vc` return 1 of q SDs | 2 of 2 sampled plain cells (q = 3 ML, REML) | q − 1 SDs lost | no |
| 3 | (i) convention | Quantile residuals: R conditions on fitted random-effect modes on every route; Julia does so only on some routes and elsewhere sets the random effects to 0 | 35 of 59 continuous-family cells differ | residual SD 1.23 median (up to 1.89) in Julia against 0.95 in R | yes, per route, in the `residuals` docstring |
| 4 | (i) convention | `fitted()` and response residuals: R adds the fitted random-effect modes, Julia is population level | 79 of 90 differ (agree in 11) | up to 82 on the response scale | yes (`fitted`, `predict`, `residuals` docstrings) |
| 5 | (i) convention | LogNormal `predict(type = "response")`: R returns the log-scale mean, Julia `exp(mu)` | 5 of 5 | exp of the link value | yes (`predict` docstring: compare with `type = :link`) |
| 6 | (i) coverage, (?) moments | `simulate()`: marginal draws exist in both for 41 cells; Julia only in 19 where R refuses, R only in 22 where Julia refuses (Julia offers `re = :zero` there). On the 41, 8 count, Gamma and LogNormal cells differ in grand variance by 9% to 105% in both directions, not shown to be noise | 90 | see Simulation | refusals are explicit |
| 7 | (i) convention | `re_sd` and `vc` refuse when an SD formula is present; R reports per-level SDs | 65 of 90 | by design | yes |
| 8 | (i) coverage | Profile intervals: Julia refuses for 5 of 7 non-Gaussian cells where R works; R refuses Gaussian REML profiles where Julia works | 10 profiled | see below | refusal text names the reason |
| 9 | (ii) noise | Hessian SE and Wald bounds at 1e-4 to 3e-3 on cells with large SE; coefficients to 1.5e-6 | 81 | stated above | n/a |
| 10 | (ii) boundary | SD near zero: estimates, SEs and `is_converged` disagree in the flat direction; the likelihood and predictions agree | 8 | up to 6.6 log units | n/a |
| 11 | (iii) minor | Two `simulate()` refusals leak internal names: `no parameter sd_phylo in fit (have [...])` (Beta, Tweedie phylo with an SD formula, 3 cells) and a raw `MethodError` for `coords` passed as a `Dict` (Beta spatial, 1 cell; `drm()` accepts the Dict) | 4 | n/a | no |

### Notes on the conventions (all numbers measured here)

- **Conventions that cost nothing:** AIC, BIC, df and nobs match R in all 90 cells. REML: log-likelihood, AIC, BIC, coefficients and SEs match R in all 14 REML cells (coefficients to 1.3e-8, SE ratio at most 1.0012); there is no ML/REML convention gap in this sample.
- **Quantile residuals (rank 3).** For continuous families (59 cells) the two agree to 1e-6 on 24 cells and on the other 35 correlate at 0.83 (median, range 0.65 to 0.97). The cells that agree are mostly SD-formula routes, which the docstring says condition on the modes. Cells that differ include all 7 crossed, all 7 independent-slope and all 3 slope cells and the plain structured intercept cells. A user fitting the same model in both packages and drawing a QQ plot sees Julia's residuals more spread out: the SD of the 35 differing cells is 1.23 at the median (range 0.87 to 1.89) against 0.95 in R. The docstring warns of this; the practical advice is to compare Julia's plot against R's only on the routes it names. For Tweedie R has quantile residuals and Julia refuses (4 cells; no closed-form CDF).
- **Predictions.** Both packages agree to 3e-6 on both scales for 84 of 90 cells. The other six are 5 LogNormal cells (rank 5) and one Poisson cell at 3e-6 (noise).
- **Wald intervals** are the same construction in both (estimate ± z·SE on the working scale). They agree to the SE differences above.

## Simulation moments

Both sides draw 1000 data sets from the fitted model with fresh random effects ("marginal"), seeds differ because the generators differ, so only moments can be compared. Which cells have a marginal draw:

| | Julia draws | Julia refuses |
| --- | --- | --- |
| **R draws** | 41 | 22 (Julia has `re = :zero` only) |
| **R refuses** | 19 | 8 |

Julia's refusals are mostly the crossed shapes `(1 | g) + (1 | g2)` (15 cells), where R draws. R's refusals are SD formulas (12) and structured terms (13) plus 2 correlated q = 3 cells, where Julia draws in 19 of the 27.

On the 41 cells where both draw (distance as a fraction of R's simulated total SD for the mean, relative for the variances):

| Moment | median | 90th percentile | max |
| --- | --- | --- | --- |
| grand mean (in SDs) | 0.0056 | 0.029 | 0.034 |
| grand variance (relative) | 0.6% | 25% | 105% |
| mean per-observation variance (relative) | 0.9% | 31% | 115% |
| share of zeros (absolute) | 0 | 0.001 | 0.002 |

The mean and the zero share agree everywhere. Gaussian cells agree on variance to 1% or better. Eight count, Gamma and LogNormal cells with random slopes or SD formulas differ in variance by 9% to 105%, with no consistent sign: R is higher in 4 (NB2 ordinary SD formula 61.3 against 34.8, Poisson relmat SD formula 96.0 against 64.8, Poisson ordinary SD formula 21.9 against 18.8, Gamma relmat labelled 10.2 against 9.0) and Julia is higher in 4 (NB2 indep SD formula 37.4 against 76.6, NB2 spatial 37.9 against 47.5, LogNormal slope 16.8 against 21.6, Poisson ordinary 17.4 against 19.7). Those models give heavy-tailed draws (40 groups, log-linked mean, random SD), so a grand variance from 1000 draws is noisy. The mixed sign fits noise. I planned to measure the seed-to-seed spread of the grand variance with six seeds on each side for these eight cells (`harness/ep_simseed_*`), but Totoro's one-minute load stayed above 360 for the last 80 minutes, so I did not run it. **Whether the eight differences are noise is unresolved.**

## Intervals

Wald: see the table above. Profile, 10 non-boundary cells (`results/profile_r.tsv`, `profile_julia.tsv`), `mu` and `sigma` coefficients:

| Cell family and route | R | Julia | Bound difference |
| --- | --- | --- | --- |
| LogNormal crossed ML | 5 intervals, 6 s | 5 intervals, 0.6 s | 7.2e-6 |
| Poisson crossed ML | 4 intervals, 4 s | 4 intervals, 0.6 s | 1.3e-5 |
| Gaussian crossed, ordinary, slope REML (3 cells) | refuses: "not ready for direct profiling" | 5 to 6 intervals each | not comparable |
| Beta crossed ML | 5 intervals, 268 s | refuses (`sigma` lower endpoint did not converge) | none |
| Gamma crossed ML | 5 intervals, 52 s | refuses (3 of 7 endpoints) | none |
| NB2 crossed ML | 5 intervals, 30 s | refuses (2 of 7 endpoints) | none |
| Student crossed, indep q = 2 ML | 6 intervals each, 115 s, 123 s | refuses (3 and 4 endpoints; 57 s and 243 s) | none |

Where both produce intervals they agree to 1.3e-5. Julia's refusals are explicit (they name the coefficient and the arm), so there is no wrong answer here, only a gap: five non-Gaussian cells where a user can profile in R and not in Julia.

## The `engine = "julia"` bridge

Not compared. `harness/ep_r.R` has an `EP_ENGINE=julia` switch that runs the same dump through `drmTMB(..., engine = "julia")`, so the bridge output could be differenced against native R and native Julia with the existing comparison script, but I did not run it (see "Not verified").

## Defects with reproducers

Both are minimal scripts in `estimate-parity-audit-20261109/repro/`, using committed data in `data/`, and they run from that folder with the DRModels environment at `0bfcde2bb`.

1. `repro1_tweedie_crossed_se.jl`. Tweedie, `y ~ x1 + x2 + x3 + (1 | g) + (1 | g2)`, `sigma ~ 1`. Julia: log-likelihood −724.7952 (same as R), eight coefficients, `stderror` all `Inf`, `is_converged` false (status "degenerate"), warning "a scale parameter ended at a clamp inside the likelihood". R: pdHess TRUE, SEs 0.131, 0.061, 0.061, 0.057, 0.034, 0.099. Estimates agree to 3.6e-6, so the optimum is the same; the Hessian is the problem. Cause is not diagnosed: I did not find which coordinate Julia thinks is at the clamp (the SDs are −0.49 and −1.06 on the log scale, the dispersion 0.18, the power 0.19, none near a bound).
2. `repro2_re_sd_indep_gaussian.jl`. Gaussian `(1 | g) + (0 + x1 | g) + (0 + x2 | g)`: `re_sd(fit)` returns one entry, `vc(fit)` one matrix, while `coef(fit, :resd)` has three values. The cause is `src/gaussian_core.jl:1904`, where the label of every scalar component is `String(grp)`. The Student and Gamma independent-slope routes label the same terms `g`, `g:x1`, `g:x2`. `proposed-gaussian-indep-resd-labels.patch` gives slope terms the label `g:x`. It is **not applied**. The patch changes the coefficient label for slope terms on this route (`resd:g` to `resd:g:x1`) and I have not run the test suite or this reproducer against it.

## Not verified

- **Sample, not census.** 90 of 260 cells; no claim about the other 170. The two defects were each found in one route family (Tweedie crossed, Gaussian indep), so related cells (Tweedie in other crossed shapes, Gaussian indep q = 2 and 4 with no SD formula) probably share them but were not fitted here.
- **Matching rule.** Coefficients are matched by position inside each distributional parameter, SDs by order. Names were spot-checked, not machine-matched. The positional rule gave zero count mismatches after ignoring Julia's extra coefficient rows (3 cells).
- **SD standard errors on the Julia side** are numerical delta-method values computed in the harness (central differences on `vcov`), not a package output.
- **Cause of defect 1** is not found. **Impact of the patch** for defect 2 on tests and on any code that keys on the label `g` is unchecked.
- **Residuals for discrete families** (Poisson, NB2, Binomial) are randomised with different generators, so they were not compared pointwise; only the two conventions above were read from the continuous families.
- **Only the logLik-agreeing cells.** Cells where the logLik disagrees (54) or Julia refuses (113) were not looked at.
- **The `engine = "julia"` bridge was not compared.** The task asked for bridge output against native R for cells the bridge reaches; the switch is in the harness but I did not run it. Totoro's one-minute load stayed between 360 and 440 from 06:39 to 07:32 and I do not run on it above 100. I also did not run the seed check for the eight simulation cells or the patch check. Everything above is native R against native Julia.
- Evidence is committed under `docs/dev-log/estimate-parity-audit-20261109/`: `harness/`, `data/` (only the files the reproducers need), `repro/`, `results/` (raw dumps gzipped, per-cell table, detail tables).
