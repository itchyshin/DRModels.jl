`finite-public-007.json` is the current source-stamped public R-to-Julia receipt.
It was generated on 2026-10-03 (#606) against the refreshed native anchor
`../finite-state/finite-native-003.json` (drmTMB `main` 0eb0467851, Newton polish
on by default); provenance is in `../nongaussian-refresh-20261003/`. It retains
raw/public covariance, separate coefficient blocks and names, actual
imputation/SD/status fields, posterior probabilities, cutpoints, and actual
new-data predictions. Both adapter verdicts pass, and both native default-fit
verdicts now pass the unchanged 4e-6 tolerance. Runtime includes startup and is
not warm performance.

001–006 are historical receipts, kept byte-for-byte. 003 introduced independent
replay fields and remains the immutable stopping-diagnostic input. 005 refreshes
final factor source and the native R regression covariance repair. 006 refreshes
final prediction source and transformed uncertainty code; elapsed 21.053 seconds
includes startup. 005 and 006 record native FAIL verdicts against the pre-polish
anchor (now `../finite-stopping/finite-native-003-prepolish.json`), so they no
longer pass the checker, which requires the current anchor hash. No receipt is a
replacement for the native comparator `../finite-state/finite-native-003.json`.
The checker replays the finite sums independently and checks inverse curvature,
public covariance axes and every retained conditional output. Its 20 corruption
controls must reject normally and with Python `-O`. Three of them guard the
verdict direction: a forged PASS against an anchor moved by 1e-5, a forged PASS
at a theta error of 4.004e-6 (just above the bar), and a positive control at
3.996e-6 that must be accepted. The checker compares source manifests by
repository-relative path and sha256, so it runs from any checkout.

R fit objects remain in the R repository only. No R implementation source is
included in the MIT Julia repository. Source and test failures are retained in
the accompanying logs. Documentation checks execute two source pages; they do
not establish visual or deployed-site completeness.

Direct Julia still uses raw coefficient/covariance coordinates including ordinal
cuts; R public accessors omit predictor cuts. Full accessor parity and typed ordered factors remain required. Direct known-state
newdata prediction now has bounded evidence in `../finite-prediction/`; broader
missing-state operations remain required. Plain-string/Boolean additional mean factors now have generated
native design evidence; see `../finite-factor-coding/`. All programme gates remain open.

Provenance describes the tested working trees, including the preserved foreign R bridge edits. It is not a clean committed-head full-suite qualification; integration must refresh that evidence after all owned changes are reconciled.
