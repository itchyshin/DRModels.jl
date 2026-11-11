# Julia crashes from gap map v2: root causes and dispositions (2026-11-09)

Reader: whoever owns the R-Julia parity lane. Purpose: say what each of the 22
uncaught Julia exceptions in `docs/dev-log/2026-11-09-gap-map-v2.md` (that file lives on the branch `claude/gap-map-v2-20261109`, not on this branch or its base) was, what
changed, and what was left alone and why. Base: `0bfcde2bb` (tip of PR #1272),
Julia 1.10.12 and 1.12. R reference: drmTMB 0.8.1 from `~/drm_campaign_20261108/Rlib`.

## Pre-registered thresholds (written before the fixes were run)

A cell that Julia now fits counts as agreeing with R when all of these hold:

- `loglik`: `abs(julia - r) <= 1e-4`. This is the gap-map v2 rule for "agree".
- `nobs`: equal.
- Every coefficient: `abs(julia - r) <= 1e-3`. The cells here are closed-form or
  well-conditioned fixed-effect fits, so 1e-3 is generous; it is not widened
  after the fact.

A cell that is turned into a refusal counts as done when `drm()` throws an
`ArgumentError` whose message names the cause and what to do, and a test pins
the exact message. A refusal is not a claim that R refuses too unless the table
says so.

R references: `test/fixtures/gap_v2_crashes/make_references.R` (one single-process
call on Totoro, `nice -n 10`, one thread) writes `r_reference.csv` next to it.
Data: `test/fixtures/gap_v2_crashes/data_v2.csv`, the gap-map v2 data
(md5 `ae47edf77c31bcd8c3b4c933e9201239`), so Julia and R read the same bytes.

## The 22 crash cells, by root cause

Reproducer for every row: `test/test_gap_v2_crashes.jl` (data in `test/fixtures/gap_v2_crashes/`).
All 22 reproduced on the base tip before any change.

| Cells | Exception on base | Root cause | Disposition | Where |
| --- | --- | --- | --- | --- |
| 7 | `MethodError: Float64(String)` | A factor used as a random slope reached `Float64.(column)` in each family route. R refuses ("Random-slope variable f must be numeric"). | **(b) refusal.** One check in `_drm_entry_guard`, so every univariate family gets it. | `src/input_validation.jl` |
| 1 | `MethodError: \|(Int64, String)` | `zi ~ 1 + (1 \| g)`: the `\|` term reached `modelcols`. R refuses ("Zero-inflation random effects are not implemented."). The gap-map note lists this cell as "R fits; Julia fails"; R was re-run for this PR and refuses, so this PR corrects the gap map there. | **(b) refusal**, same guard, for `zi` and `hu`. | `src/input_validation.jl` |
| 1 | `MethodError: _mf_init(LogNormal, …)` | The cross-family route has six families; `LogNormal` fell through to a missing method. R refuses the pair. | **(b) refusal** that lists the supported families. | `src/mixed_family.jl` |
| 1 | `DomainError: log(-1.0)` (`dual.jl:254`) | Gaussian REML. A line-search trial with `sigma` coefficients near 40 gave weights from 1e-108 to 6e79, so `Xμ' W Xμ` was numerically indefinite and `logdet` threw. R fits it. | **(a) fits.** The restricted objective returns `Inf` when the determinant sign is not positive; the line search backs off. | `src/gaussian_core.jl` |
| 1 | `MethodError: Float64(String)` | Ordinal response given as string labels. drmTMB refuses a character response and an unordered factor (it asks for an ordered factor or integer scores), so this is a Julia-side convenience; the R comparison uses an ordered factor whose levels are in label order. | **(a) fits.** String or symbol labels are coded 1..K in sorted order, with an `@info`; another element type is refused. | `src/cumulative.jl` |
| 1 | `DimensionMismatch` | `weights` of length n with 16 missing responses. The observed-rows wrapper dropped rows but not their weights; helper `_drop_scope_weight_rows!` already existed for the other drop paths. R fits. | **(a) fits.** The two observed-rows paths now drop the weights with their rows. A vector whose length is neither n nor the observed count is still a `DimensionMismatch`. | `src/gaussian_core.jl` |
| 2 | `LoadError: non-call expression … \|\|` | `@formula` cannot parse `\|\|`. The error is raised when the formula is written, before `drm` runs. R fits it as independent intercept and slope. | **(c) not fixable inside `drm`.** Spelling documented and tested: `(1 \| g) + (0 + x1 \| g)` gives R's log-likelihood. Own macro or string front-end for `\|\|`: M. | capabilities, news |
| 8 | `UndefVarError: poly` | `@formula` has no `poly`; the symbol is looked up in the caller's scope. R fits it. | **(c) not fixable inside `drm`.** A raw quadratic function term spans the same space and gives R's ML log-likelihood (tested). A native orthogonal `poly` needs a stateful StatsModels term that stores the basis for `predict`: M. The string route already reproduces R's `poly` (#492). | capabilities, news |

Result: 12 cells change from crash to a fit (3 cells) or a clean refusal (9 cells); 10 cells are
documented as not fixable inside `drm`, each with a tested spelling.

### Agreement with drmTMB 0.8.1 (thresholds above)

| Fit | logLik Julia | logLik R | Max coefficient difference |
| --- | --- | --- | --- |
| REML `y ~ f * x1`, `sigma ~ x1 + f` | -238.89994 | -238.89994 | within 1e-3 on the coefficients the test compares; the sigma intercept (R 0.0981624) is not asserted, and the logLik tolerance of 1e-4 bounds the rest |
| `CumulativeLogit`, labels | -209.65178 | -209.65178 | slope within 1e-3 (tested); the cutpoints are not compared (the R reference has no cutpoint rows) |
| `weights = w`, 16 missing responses | -277.20962 | -277.20962 | the first two coefficients within 1e-3 (tested); the rest are bounded only by the 1e-4 logLik tolerance |
| `(1 \| g) + (0 + x1 \| g)` against R `(1 + x1 \|\| g)` | -242.40879 | -242.40879 | not compared |
| `y ~ x1 + sq(x1)` against R `poly(x1, 2)` | -243.58556 | -243.58556 | not compared (different basis) |

The test asserts `abs(diff) <= 1e-4`; the table shows the values rounded to five places.

## Behind cells (37) sharing a root cause with the crashes

- **Factor slopes: no behind cell.** R refuses them, so the seven crash cells were not parity gaps.
- **Group with NA labels (2 cells): left as a refusal.** #1112 chose "refuse, point to `drm_listwise`"
  and `test_1112_missing_group_label.jl` pins it. drmTMB drops the 10 rows (`nobs` 150). On the
  `drm_listwise` data Julia agrees with R (logLik -223.59546, `nobs` 150, tested). Making `drm`
  drop silently with a warning, as it already does for a missing predictor, is S: add the grouping
  columns of ordinary `|` terms to `_drop_missing_predictor_rows`. It reverses a tested decision, so
  it needs the owner's call.
- **`||` and `poly` as syntax** (8 + 2 cells above): M each, see the table.

## Remaining Julia-behind cells, ranked by effort

Cell counts are from the gap-map v2 note. Efforts are my estimate from the refusal text, not from a
read of each route; none was started. Row 1's remark rests on one call site only: `_extract_offset`
returns a single `offset_term` (`gaussian_core.jl`, the Gaussian entry).

| Rank | Cells | What Julia needs | Effort |
| --- | --- | --- | --- |
| 1 | 3 | Offsets: two `offset()` terms, `offset` with `zi`, `offset` with `meta_V` (sum the terms; lift one limit per route) | S to M. `_extract_offset` returns one term that every route consumes; summing needs that contract widened. |
| 2 | 2 | `method = :REML` with `meta_V` (fixed and `(1 \| g)`) | M |
| 3 | 2 | Cumulative logit with `(1 \| g) + (0 + x1 \| g)`; Binomial with three random intercepts | M each |
| 4 | 6 | `weights` with a random effect, `meta_V` or the bivariate route | M: row weights through the grouped likelihoods |
| 5 | 8 | `sigma` random effects on lognormal, Gamma, NB2, beta-binomial; Gaussian `sigma` with slopes, two groups, or a group different from `mu` | M each |
| 6 | 2 | SkewNormal with a random intercept on `mu` | M |
| 7 | 4 | `zi`/`hu` with a random effect on `mu` (Poisson, NB2, hurdle NB2, truncated NB2) | L |
| 8 | 6 | Bivariate Gaussian with ordinary random effects `(1 \| p \| g)` (also `sigma1`, `sigma2`, REML, missing, weights) | L |
| 9 | 2 | Gaussian `link = "log"` / `"inverse"` | Decision first: R accepts the argument and ignores it (gap-map v2, section 3); matching that would copy a bug. |

## Not verified

- Julia 1.12 and the full suite: see the Verification section of the PR.
- Mixed-family refusal and the `zi` refusal are pinned by message only; no R run was needed (both are R refusals in the gap-map v2 data).
- The REML fix is a guard on the objective, not a rewrite of the line search. A REML fit whose optimum lies where `Xμ' W Xμ` is genuinely singular would now report a non-converged or failed fit instead of a `DomainError`; none appeared in the grid.
- `poly` and `||` were checked through their equivalent spellings, not through new Julia syntax.
