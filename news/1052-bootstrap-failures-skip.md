- **Bootstrap records a thrown refit by default (#1052).** `bootstrap_ci`,
  `bootstrap_summary`, `bootstrap_result` and `_bootstrap_result` now default
  to `failures = :skip`, matching drmTMB: a refit that throws is stored in
  `failures`, successful draws are kept, and the result is
  `bootstrap_incomplete` with a warning. `failures = :error` still aborts.
  When every replicate fails the error names the first failure's cause.
  `used == 0` still errors.
