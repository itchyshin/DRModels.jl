# Finite-state stopping diagnostic

This receipt records a diagnostic explanation of the retained finite-state native/Julia stopping differences. It is not a new fit, optimizer change, global optimum certificate, or parity waiver.

The checker independently evaluates the frozen prepared-kernel likelihood, finite-difference gradient, and local Hessian after applying the explicit native parameter permutation. The resulting Newton displacement reproduces the Julia/native displacement to a maximum residual of `6.356001848094714e-11` for the ordinal case and `1.9967956024348845e-10` for the categorical case. The independently evaluated Julia gradients are `1.1368683772161601e-08` and `8.526512829121202e-09`, respectively. These explain the small local stopping displacement; they do not replace the native default fit.

The retained native default parity verdict remains **FAIL** at the strict `4e-6` tolerance for both finite-state cases. The diagnostic reports native maximum gradients `0.001573169705340634` (ordinal) and `0.0009399121840727998` (categorical). It also records the original finite-state errors without suppressing them. The companion damage check rejected all six intentionally damaged receipts (`FINITE_STOPPING_DAMAGES_REJECTED 6`); that attribution is from the retained checker output, not a new campaign.

The frozen inputs are:

- native receipt: `finite-stopping/finite-native-003-prepolish.json`, SHA-256 `d8f75d1d4652d5580cee935b3eeb22d003b7ccb33d7b58f149cef190a712e7cb`. It is byte-identical to `finite-state/finite-native-003.json` as it stood before the 2026-10-03 refresh (#606), which replaced that file with the Newton-polished anchor;
- Julia/public receipt: `finite-frontends/finite-public-003.json`, SHA-256 `0147b2657c81b223e5c4e5742e0d66b90ca570b576da5ce04887e7dcf3ef2ee2`;
- diagnostic: `finite-stopping/diagnostic-001.json`;
- checker: `tools/check_finite_stopping_diagnostic.py`.

**Cross-check after the 2026-10-03 refresh (#606).** The diagnostic's own Newton
step, taken from the pre-polish anchor with the independently evaluated gradient
and Hessian, lands on the new polished anchor (`finite-state/finite-native-003.json`,
SHA-256 `734560e4…`): max |old anchor + Newton step − new anchor| is `4.5e-11`
(ordinal) and `9.9e-11` (categorical), against a raw displacement of `2.16e-6` and
`1.74e-5`. So drmTMB's Newton polish moves the native fit to the point this
diagnostic predicted. Script and output:
`../nongaussian-refresh-20261003/newton_xcheck.py` and `newton-xcheck-out.txt`.

Reproducible checks (read-only):

```sh
python3 tools/check_finite_stopping_diagnostic.py --check \
  docs/dev-log/evidence/julia-r-parity/finite-stopping/diagnostic-001.json
python3 tools/check_finite_stopping_diagnostic.py --check \
  docs/dev-log/evidence/julia-r-parity/finite-stopping/diagnostic-001.json --damage
```

The original native reference also exposed a separate public covariance mapping bug: `mi_<variable>` was resolved as `beta_mi_<variable>`, instead of `beta_mi`/`beta_mi2`. The bounded R accessor repair and its independent review are retained in the drmTMB checkout under `native-mi-covariance/`. It changes no fitted objective or frozen reference. Positive-scale and mixture-probability summaries still require transformation-aware covariance work.
