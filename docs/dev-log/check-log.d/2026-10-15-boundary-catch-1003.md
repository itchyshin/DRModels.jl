# #1003 narrow the boundary-diagnostic catch

- Change: `_boundary_diagnostic_failed` (src/boundary_diagnostics.jl) now downgrades only
  `KeyError`/`BoundsError`/`DimensionMismatch` to a warning; every other exception is rethrown.
  (Lane J2 had already replaced the silent `catch -> nothing` with a warning for all but `InterruptException`.)
- Tests: test/test_j2_convergence_vcov.jl, "#1003 only shape errors are caught ..." (red on main f71b292).
- Not run locally: Julia is unavailable in this container; verified by CI (see PR body).
