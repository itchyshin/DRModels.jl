# Structured and positive-family SD cells from the wave-10 gap map, 2026-11-09

Reader: whoever owns the R-Julia parity lane. Branch `claude/struct-sd-counts-20261109`, from `0bfcde2bb` (tip of PR #1272). Scope: the structured-covariance and positive-family SD cells of the wave-10 gap map (ranks 21 to 31 and the small signatures) that are not independent-term or correlated ordinary blocks. Draft only; nothing here is merged by us.

## Step 1: where each refusal lives, and the plan

| Cell group (R fits them) | Refusal text | Source | Plan | Effort |
| --- | --- | --- | --- | --- |
| Poisson `spatial(1 \| g)` + `sd(g, spatial) ~ 1 / z` (2) | "targets one ordinary mean random-effect coefficient" | `src/poisson.jl`, `_drm_poisson_sd_regression` | New leg beside the relmat leg: kernel `exp(-d / median positive d) + 1e-6 I` from `_gaussian_spatial_sd_kernel`, then the existing relmat SD Laplace objective | S |
| NB2 `spatial(1 \| g)` + SD (2) | "supports only the slope-only `spatial(0 + x \| g)`" | `src/negbinomial.jl`, one guard | Drop the guard: the kernel and the animal-SD Laplace objective were already wired for the spatial slope | S |
| NB2 `phylo(1 \| g)` + `sd(g, phylogenetic)` (2) | "supports ordinary effects, an animal intercept ... other structured routes are not implemented" | `src/negbinomial.jl` | New leg: tip-correlation kernel (`_structured_q2_kernel`, as the Beta structured route), rows reordered to first-seen groups, animal-SD Laplace objective | S |
| Gamma `(1 \| g) + (0 + x \| g)` with ONE SD formula (`sd-first`, `sd-last`) (2) | "requires exactly one ordinary independent random effect" | `src/twofield_sd_laplace.jl` (`_twofield_build`) | Complete the single target with the default intercept-only SD of the other coefficient, then the existing two-field engine. Gamma only; NB2 and Poisson share the shape but belong to the independent-terms lane | S |
| LogNormal `(1 \| g) + (1 \| h)` with `sd(g)`, `sd(h)` (2) | "requires one ordinary mean-effect `sd(group)` formula" | `src/lognormal.jl` | log y is exactly Gaussian: delegate to the Gaussian dispatcher on log y plus the Jacobian shift (`_lognormal_jacobian_shift`), as the other LogNormal routes do | S |
| Student `(1 \| g) + (1 \| h)` with `sd(g)`, `sd(h)` (2) | "requires exactly one `sd(group)` formula" | `src/student.jl` | Per-level SD designs in the crossed Laplace fitter | M |
| Binomial `relmat(1 + x \| g)` + SD (2) | "requires exactly one `sd(group, relmat)` formula" | `src/binomial_sd_regression.jl` | **Record only.** R's `is_converged()` is FALSE for both cells and the x1 SD sits at -10.7 and -5.7 with SE above 1e4 | none |
| NB2/Poisson/LogNormal `ordinary` q2 `sd-first/last`, NB2 `indep` q3/q4 | correlated-block and independent-term refusals | various | Out of scope (correlated and independent lanes) | n/a |
| Poisson spatial plain (no SD) NaN | start values | `_fit_poisson_spatial_coord` | Not touched (triage lane owns it) | n/a |

LogNormal `ordinary` q2 `sd-one` and `sd-z` (correlated `(1 + x1 | g)` with two coefficient SDs) are unlocked by the same delegation as a side effect. They belong to the correlated-blocks lane; they are recorded in the verdict table but not tested here beyond the logLik.

## Pre-registered thresholds

Same as the Beta structured route (2026-11-08), fixed before the new fits were compared coefficient by coefficient. The stored wave-10 R logLik had been looked at while choosing cells, so T1 is not blind for the first fits; T2 and T3 are.

- T1: `|loglik_julia - loglik_R| <= 1e-4`.
- T2: every mu, sigma, nu and SD coefficient `|diff| <= 1e-3`; and `|diff| / R-SE <= 0.02`.
- T3: `|se_julia / se_R - 1| <= 1e-2` for every finite R SE.
- Verdict classes per cell: (a) fits and agrees, (b) R does not converge: record and leave refused, (c) Julia stops at a worse optimum than R, (d) too large: record.

R references: drmTMB 0.8.1, library `~/drm_campaign_20261108/Rlib` on Totoro, one core, `nice -n 10`, about 45 s for 14 cells. Script and output: `test/fixtures/struct_sd_counts/ref_struct_sd_counts.R`, `reference_drmtmb.csv`. Datasets are the wave-8 gap-map data (400 rows).

## Step 2: results

Filled in by the agreement test `test/test_struct_sd_counts.jl` (see the PR body for the table).
