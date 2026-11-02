- **Bootstrap records a thrown refit and names an all-fail cause (#1052).**
  A thrown refit is stored in `failures`. The default remains
  `failures = :error` (abort after the thrown refits are recorded). Pass
  `failures = :skip` to keep successful draws and return
  `bootstrap_incomplete`. When every replicate fails the error names the
  first failure's cause. `used == 0` still errors.
