# Bivariate missing-response defaults and remaining fitted()/predict() gaps, 2026-11-09

Reader: whoever owns the R-Julia parity lane. Purpose: re-measure the gap-map v2 `fitted()` and `logLik` mismatches on the stack of PRs #1277 and #1282, fix what Julia can fix, and make Julia's bivariate missing-response default match drmTMB 0.8.1.

## Pre-registered agreement rule (written before any Julia number of the new fit was read)

The R side of the bivariate missing-response runs (`biv/r_out.txt`, three seeds) was produced first and read for its behaviour (default, row counts, messages). No Julia number from the new code had been seen when this rule was written.

A cell agrees with R drmTMB 0.8.1 at the optimum when all of these hold:

| Quantity | Rule |
| --- | --- |
| log-likelihood | `abs(dLL) <= 1e-4` |
| mean-model and `rho12` coefficients | `abs(diff) <= 1e-3 * SE_R` |
| log standard deviation coefficients (`sigma1`, `sigma2` on the log scale) | `abs(diff) <= 5e-3` |
| fitted values on the response scale | `abs(diff) <= 1e-6` |
| rows used (`nobs`) | equal |

Three seeds (1, 2, 3), three missingness patterns (both responses partly missing, only `y1`, only `y2`), two modes (default and `miss_control(response = "include")`). A failure on any seed is reported as a failure; thresholds are not changed after reading.

For the gap-map re-measurement the rule of `classify.py` stands unchanged (logLik within 1e-4, fitted mean and sd within 1e-4 relative).

## What drmTMB 0.8.1 does (measured, `biv/r_out.txt`)

- `miss_control()` has `response = c("drop", "include")`, default `"drop"`. Leaving `missing =` out, passing `miss_control()` and passing `miss_control(response = "drop")` give identical fits (same logLik to 8 digits).
- Default: a row with either response missing is removed before fitting (complete cases). No warning and no message. `nobs()` counts the kept rows (117 of 160 for seed 1, both responses partly missing); `fitted()` has one row per kept row.
- `"include"`: rows with at least one observed response stay; a row with both missing is removed from the likelihood. A partly observed row contributes the univariate marginal of its observed response. `nobs()` counts rows with at least one observed response (158 for the same data); `fitted()` has one row per input row (160).

## Step 1: re-measurement on the stacked tip (gap map v2, 492 cells)

Tip measured: PR #1272 (`0bfcde2bb`) with #1277 and #1282 merged (`e27db3897`). Julia 1.10.12 on Totoro, the same harness and the stored R side of `claude/gap-map-v2-20261109`, one Julia pass of 8 workers (5 minutes, as estimated). The bivariate cells (areas G and M, 92 cells) were then rerun on a Mac with Julia 1.10.0 after the code changes of this PR; the other 400 cells run no code this PR touches and keep the Totoro result. Raw output and the classified maps are in `remeasure/`.

| Class | #1272 tip (gap map v2) | + #1277 + #1282 | + this PR |
| --- | --- | --- | --- |
| both fit, agree on logLik and fitted (harness rule) | 216 | 270 | 270 |
| both fit, logLik agrees, harness flags `fitted()` | 104 | 50 | 56 |
| both fit, logLik differs | 27 | 27 | 21 |

The harness rule compares mean and sd of one fitted vector. For a bivariate fit it reads the first column of Julia's `Dict(:mu1, :mu2)` against R's stacked matrix, so 32 bivariate Gaussian and Student cells are flagged although both engines agree. The 56 flagged cells after this PR were therefore compared row by row (`remeasure/fitted_full_vector_comparison.txt`, scripts `dump_r.R`, `dump_jl.jl`, `cmp.py`):

| Signature of the 56 flagged cells | Cells | Row-by-row result |
| --- | --- | --- |
| Bivariate Gaussian and Student (36 and 2; includes the six `missing` default cells and the six `include` cells) | 38 | agree, max difference 1.2e-7 (optimiser tolerance on `rho12 ~ x`; 2e-11 or smaller on `rho12 ~ 1`) |
| Bivariate LogNormal | 3 | agree after this PR (max 1e-12). At the merged tip they differed: `fitted()` was the log-scale location, R gives `exp(mu + sigma^2/2)` |
| Random-effect fits, R adds the fitted modes to `fitted()` | 12 | differ by 0.016 to 1.5 (Julia is population level) |
| Univariate response with NA rows | 3 | agree on the 144 rows R keeps (max 3e-11); Julia returns all 160 rows |

So the number of real `fitted()` mismatches is 15 (12 random-effect, 3 row-count), not 56. Of the 50 flagged at the merged tip before this PR, 32 were harness artefacts, 3 were genuine (bivariate LogNormal, fixed here), 12 random-effect and 3 row-count. The six bivariate default cells entered the flagged class only because their logLik now agrees. The six bivariate `R-default(drop)` cells in the logLik-differs class now agree; the six `R-include` cells need `missing = miss_control(response = "include")` in the Julia string (the grid's Julia string has no option, so after the default change they differ by construction; `cells_biv_patched.tsv` adds the keyword, and they agree to 3e-6).

`predict()` was not part of the harness comparison beyond internal consistency. drmTMB's `predict(fit)` for a bivariate fit returns the first location only (a plain vector); Julia returns both responses as a `Dict`. For a bivariate LogNormal, drmTMB's `predict()` is the log-scale location; Julia returned `exp(mu)`. Fixed here for LogNormal.

### Remaining logLik differences (21)

- 19 random-effect cells (Poisson, NB2, Binomial probit and cloglog, cumulative logit, Gamma, Beta, zero-one Beta; |diff| 0.005 to 0.19). Julia's default for these routes is adaptive Gauss-Hermite, drmTMB's is Laplace. Refitting with `marginal = :Laplace` (the documented switch) gives agreement to 6e-10 or better on all 9 cells whose route accepts it (`remeasure/laplace_switch_check.tsv`). The other 10 routes refuse `:Laplace` (Binomial probit and cloglog with a random effect, cumulative logit, and the `sigma ~ x` plus `(1 | g)` routes for Gamma, Beta, NB2 and zero-one Beta), so Julia cannot reproduce R's number there. See decision 2.
- 2 SkewNormal fixed-effect cells where Julia stops at a lower local optimum (8.7e-2 and 3e-3). Not touched. See decision 3.

## Step 2: what was fixed

1. Bivariate `LogNormal` `fitted()`: now `exp(mu_k + sigma_k^2 / 2)` per response, as drmTMB. `residuals()` is now `y - fitted(fit)` on the response scale; before it subtracted the log-scale location from the response. `predict(type = :response)` is the log-scale location and `type = :median` its exponential, as for the univariate LogNormal in #1282. Test against an R matrix (`fitted_lognormal_seed1.csv`), 1e-6.
2. A bivariate `LogNormal` fit with `Union{Missing,Float64}` response columns failed in a `Float64` conversion under `include`; it now fits (it only handled `NaN`).

## Step 3: bivariate missing-response default

Julia change, all three bivariate routes (Gaussian including REML and the structured q=2 and q=4 routes, `LogNormal`, `Student`):

- New keyword `missing = nothing`. `nothing` means drmTMB's default: rows with either response missing are dropped before fitting, with a counted warning. drmTMB drops silently; the warning changes no number and matches Julia's other drops.
- `missing = miss_control(response = "include")` is the old behaviour: partly observed rows stay and contribute the marginal likelihood of their observed response. `miss_control(response = "drop")` is the default spelled out. `miss_control(response = "fail")` refuses data with a missing response. `miss_control(predictor = "model")` is refused for a bivariate response. `miss_control` now accepts `response = "drop"`.
- The R bridge sends `missing = miss_control(response = "include")` for these routes: the R wrapper already applied its own `drop`, so any row that reaches Julia with a missing response was kept on purpose.
- Not changed: the cross-family (mixed family) route, which already refuses missing responses.

Agreement against drmTMB 0.8.1 under the rule above (`check_agreement.jl`; the same checks run in `test/test_bivariate_missing_default.jl`): 27 of 27 cells agree (3 seeds, 3 missingness patterns, default, `include` and `drop`). Largest |dLL| 5e-11, largest coefficient difference 8.5e-6 of one R standard error, largest log SD difference 7e-11, largest `fitted()` difference 8e-11, rows used equal in every cell. The fixtures are in `test/fixtures/bivariate_missing/`.

Test status: `test/test_bivariate_missing_default.jl` is new (94 assertions); on the base `e27db3897` the default cells fail (Julia kept the partly observed rows: logLik -345.4 against R's -292.9 for seed 1) and on this branch they pass. `test/test_reml_q4_missing_response.jl` asserted the old default (`nobs == n`) and now asks for `include` explicitly.

## Step 4: decisions needed (not guessed)

1. **`fitted()` on random-effect fits: population level or conditional?** (12 cells.) drmTMB adds the fitted random-effect modes to `fitted()` and `residuals()`; Julia returns the population value with random effects at zero on most routes, a choice pinned by #1092 and by many tests. Recommendation: keep the default, add a keyword `fitted(fit; conditional = true)` (and the same for `residuals`) that adds the stored modes on the routes that keep them (ordinary, crossed, slope, structured), document it in the rosetta page next to drmTMB's default, and flip the default only in a breaking release if you want R-identical output. Cost: one PR, about a day with tests on every route; it changes no current number.
2. **Laplace against adaptive Gauss-Hermite as the default for random-effect GLMMs.** (19 cells.) Julia's AGHQ is the more accurate integral, and `marginal = :Laplace` reproduces R exactly where it exists (9 of 19 cells, 6e-10). Recommendation: keep AGHQ as the default, state the difference in the parity page, make the parity harness use `marginal = :Laplace`, and add Laplace to the 10 routes that refuse it only if you want a number-for-number match there.
3. **SkewNormal `nu ~ x` local optimum.** (2 cells.) Julia reports converged at a lower optimum than R. Recommendation: add a multi-start for the skew slope; effort about half a day. Not started here.
4. **Row count of `fitted()` when responses are missing.** (3 cells.) Julia returns a value for every input row, R only for the rows it fitted (144 of 160 by default). The values agree on the common rows. Recommendation: keep Julia's longer vector, document it, and let the bridge subset; the alternative (drop to the fitted rows) loses information a user may want for the rows with no response.
5. **Bivariate `predict()` shape.** drmTMB returns the first location only; Julia returns both responses as a `Dict`. Recommendation: keep Julia's, which is a superset.
6. **Warning on the default drop.** drmTMB is silent, Julia now warns with the count (house style for every other Julia drop). Recommendation: keep the warning.
7. **`miss_control()` defaults.** Julia's `miss_control()` has `response = "fail"` (kept for the joint missing-predictor frontend); drmTMB's is `"drop"`. A bivariate call with no `missing` follows drmTMB, but an explicit `miss_control()` still means "fail" here. Recommendation: leave it and say so, since changing it would silently change the joint frontend.

## Environment notes and unverified items

- Totoro's ControlMaster socket disappeared during the session and sshd refused connections for a while (no fresh login was attempted); it came back later. The bivariate R reference was therefore produced with drmTMB 0.8.1 built from this repository's source into a private library, and reproduces the Totoro run (`~/drm_campaign_20261108/Rlib`, 0.8.1) to 1e-8 in logLik on all 27 cells. The stored R side of the gap map is unchanged.
- Bivariate gap-map cells were rerun on a Mac (Julia 1.10.0, 2 workers, load above 60 from other users); the 400 other cells were not rerun after the code change because no touched code runs in them. The full test suite is the check on that claim.
- Julia 1.12 gap-map runs were not repeated; the unit tests of this PR are run on 1.10 and 1.12 on Totoro (see the PR).
