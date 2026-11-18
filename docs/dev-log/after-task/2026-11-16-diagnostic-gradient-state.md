# Diagnostic gradient state

A Gamma correlated-slope fit returned different gradient diagnostics on repeated calls at unchanged parameters. Its AGHQ objective updates a retained warm-start cache; a near-boundary fit showed measurable sensitivity to that state. The same diagnostic is used by `check_drm`, the fit report and bridge diagnostics.

The Gamma and Student correlated-slope routes now retain a callable wrapper that leaves ordinary objective evaluation on its existing path. Reporting gradients use a private snapshot and do not update the live cache or count diagnostic-only mode resets. Finite-difference evaluations start from the same snapshot. The wrapper's type appears in the bridge's diagnostic route label.

The bounded checks and their limits are in `docs/dev-log/check-log.d/2026-11-16-diagnostic-gradient-state.md`. Repeated diagnostic values do not prove gradient accuracy near a boundary. The full package suite, hosted checks and independent post-integration review remain required.
