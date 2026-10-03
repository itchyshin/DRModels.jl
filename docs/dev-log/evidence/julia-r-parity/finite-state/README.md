# Shared finite-state prepared likelihood — development checkpoint

Issue DRM.jl#563 remains open. One ordinal or categorical predictor with a Gaussian
response is implemented through a common state-expanded prepared likelihood.
Direct Julia formula admission and R bridge transport for these routes remain
required next. All24 native missing-predictor obligations remain open; these two
prepared cases do not close entire capability axes.

## Current evidence

- `finite-native-003.json`: unchanged native defaults;180rows each,3states,
  all4response/predictor masks,3parameter points. Complete source/DLL/control hashes.
  Raw native ordinal order is beta,delta,cuts,alpha; transport explicitly permutes
  to beta,delta,alpha,cuts. Categorical alpha is level-major, then designterm.
- `finite-reference-003.toml`: validated numerical transport plus independent
  finite-sum likelihood/posterior/prediction values; no GPL implementation source.
- `finite-julia-002.toml`:48fixed-point checks against current source. Likelihood,
  gradient, allrow probabilities, ordinal score SD, categorical mode and full
  state-weighted predictions pass. This is not default fitting parity.
- `finite-fit-002.toml`: actual default prepared fits, raw covariance, actual
  imputed SDs and availability/status masks, predictions, native errors and losses.
  Both fits converged with observed-information covariance; numerical independent
  Hessian inversion checks pass (absolute max|HV-I|<=1e-4, predeclared).
- Native oracle18damage controls and fit oracle17damage controls pass normally and
  with Python assertions disabled. The fit oracle rejects arbitrary1000I covariance,
  changed SDs/masks and dishonest success flags. All raw failures are retained.
- The updated developer page executes its3examples (`finite-kernel-002` build).
  This is source-build evidence only, not rendered/full-site/deployment evidence.

## Strict native-default parity (refreshed 2026-10-03, #606)

`finite-native-003.json`, `finite-reference-003.toml` and `finite-fit-002.toml` were
regenerated against drmTMB `main` 0eb0467851, which Newton-polishes fits by
default. The comparator, the 4e-6 threshold and the estimators are unchanged.
Both default fits now pass:

- ordinal: theta 2.7e-11, prediction 1.7e-11, imputation 2.8e-11, conditional SD
  2.2e-11;
- categorical: theta 1.0e-10, prediction 7.7e-11.

`check_finite_fit_receipt.py ... --require-parity` now exits 0. The earlier
losses were real disagreements with where the old build's `nlminb` stopped (ordinal
theta 2.163e-6, prediction 7.561e-6, imputation 5.124e-6; categorical theta
1.741e-5, prediction 9.576e-6). They are kept in git history and explained by
`../finite-stopping/`, which now reads the retained pre-polish fixture
`../finite-stopping/finite-native-003-prepolish.json`. Provenance, data
reproduction (≤1.3e-15 from the frozen data), the validator anchor edits and the
before/after checks are in `../nongaussian-refresh-20261003/`.
`finite-julia-003.toml` is the fixed-point receipt against the refreshed reference.

## History and interpretation

Native001 incorrectly recorded a nonexistent control field;002 corrected controls
but stored a one-row gradient matrix.003 corrects both. Earlier JSONs are retained,
not current anchors. Julia001 fixed-point/fit receipts predate the final tiny-spacing
arithmetic repair or actual-SD retention;002 is the current source receipt.
Several stress-test expectations were corrected from independent state calculations;
logs preserve each failure. An initial sandbox attempt could not write Julia's
cache. An initial gate-wrapper attempt used the ledger directory instead of the
checkout; explicit --cwd corrected it. No tolerance or estimator was changed.

Rose approved the bounded prepared kernel and independently checked covariance/SD
receipts. The representable1e308 arithmetic repair was inspected; there is no
specific1e308 test assertion, so no such test coverage is claimed. Actual cutraw
-1000, logits±1000 and large spacing+10 are exercised.

No frontend/bridge parity, recovery, coverage, profile/bootstrap, warm-performance,
release or registration claim. The original full programme and its gates remain open.
