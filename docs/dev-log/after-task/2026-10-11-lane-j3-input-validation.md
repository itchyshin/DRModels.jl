# After-task: lane J3 — fit-entry input validation

## 1. Goal

Refuse, with an `ArgumentError` that names the argument, inputs the `drm`
entry points used to accept and then drop or misread silently. Code-review
issues #1027, #1026, #1022, #1020, #1014, #1013, #1010, #966 and #1017, plus
#939, #938, #926 and the Gaussian half of #951 added by the coordinator, and the
approved decisions 15 (#974, #963) and 16 (#957) (campaign of 2026-10-11, lane J3).

## 2. Implemented

New `src/input_validation.jl`, included after the family files:

- `_drm_entry_guard(f, fam, data; tree, K, A, coords)` is the first line of
  every univariate `drm(::DrmFormula, family)` method. It runs:
  - `_require_family_dpars` (#1027): a formula for a parameter the family
    does not model is refused. A per-family table (`_family_dpars`) lists the
    secondary parameters each method consumes. A constant `sigma` (which `bf`
    always adds) is accepted for families without sigma. Binomial and the
    truncated families keep their own existing checks. This also fixes #939
    (CumulativeLogit `sigma ~ x`).
  - `_require_offset_supported` (#938): `offset(...)` is refused anywhere
    except the Poisson mean, where Poisson applies it with coefficient 1 and
    keeps its own limits. Elsewhere it used to get a free coefficient.
  - `_require_formula_column_lengths` (#1022): every column the formula reads
    (responses, predictors, grouping, slope, `meta_V`, temporal time) must have
    one value per response row. The bivariate Gaussian, Student, LogNormal and
    cross-family entries run the same length check.
  - `_require_structured_keywords_used` (#1010): `tree` needs `phylo()` (or
    `sd(g, phylogenetic) ~`), `K` needs `relmat()` (or, off the Gaussian
    family, `spatial()`), `A` needs `animal()`, `coords` needs `spatial()`.
- `_refuse_unmatched_tip_labels` (#1026) is called by `_poisson_phylo_setup`
  and `_phylo_mean_leaf_index` before the positional fallback. Name-bearing
  labels that are not all tip names are refused. Purely numeric codes keep
  the old fallback.
- `_require_symmetric_finite_covariance` (#1020) in
  `make_coevo_problem_from_covariance` (q=2) and `_q4_structured_precision`.
- `_require_meta_V_variances` (#1014) in both meta_V fitters.
- `drm_listwise` (#1013) uses the same column walker. That walker skips the
  `temporal()` structure keyword and length-checks before its scan.
- Cross-family `drm(bf, (fam1, fam2))` (#966) refuses a missing response,
  naming the column. `fit_mixed_family` clears `converged` when the optimum is
  the 1e10 sentinel.
- `algorithm = :sparse` / `sparse = true` on an unstructured Gaussian cell is
  refused (#951); `:gls` / `:lbfgs` stay documented aliases there.
- `_resolve_relatedness_labels` (#957, decision 16): `A` is matched to the data
  by label when it is a table (column names are the labels) or comes with the
  new `ids = …` keyword (Gaussian, Poisson, NB2, Gamma, Beta, ZeroOneBeta).
  Labels absent from the data are dropped; a data level without a row in `A`
  is refused. An unlabelled matrix keeps the first-appearance convention and
  warns once per session.
- Readable refusals (decision 15): univariate Gaussian `V = …` (#974, used to
  be a bare MethodError) and `animal`/`relmat`/`spatial` on `sigma` (#963).
- `r2_constant_sigma` (#1017) refuses any fit whose formula has a random,
  structured, `meta_V` or temporal part, or which has blocks other than
  mu/sigma. It drops non-finite rows from both sums. Because the walk covers
  the `sigma` formula, `sigma ~ 1 + (1 | g)` is refused too (#926, r2 part).

## 3a. Decisions and Rejected Alternatives

- #1009, #1012, #1019 and #1021 (non-finite inputs and sentinel convergence)
  are already fixed by the open PR #1045, so they were not duplicated here.
  That PR also covers non-finite `K`/`A` and the raw-array half of #966.
- `profile_ci` (#1010): left as accept-and-ignore, because
  `test_sigma_re_laplace.jl` asserts that the bridge sets it on every profile
  call. Changing that is an owner decision.
- #926 bootstrap part (the marginal simulator ignores a scale random
  effect) is in the bootstrap code, lane J1's area, and was not changed.
- #938: refusing was chosen over implementing a coefficient-1 offset for
  every family; drmTMB applies offsets for all families, so implementing it
  is a feature for the owner to schedule.
- #957: done per decision 16 for univariate `A`. Not changed: `K` (relmat)
  and `coords` (spatial) keep the first-appearance convention; the bivariate
  animal route is unchanged; bootstrap / LRT refit helpers do not forward
  `ids`, so pass `A` as a labelled table when refitting.

## 4. Verification

Julia could not be installed in the agent container (network policy), so no
local tests were run. PR CI is the test runner. See the PR comment for the run.
One existing fixture changed: `test_bridge_formula_translation.jl` used
`"sp1".."sp16"` labels against tips `"L1".."L16"`, relying on the positional
fallback (#1026). It also passed `tree` to formulas with no phylo term
(#1010). Both were updated; the fitted models are identical.

## 5. Follow-ups

See "Needs Shinichi's decision" in the PR body.
