# After-task: #1003 boundary-diagnostic catch

Shannon; no subagents were running. Finding: main already warned on a failed diagnostic (J2), but caught
every non-interrupt exception, so a genuine bug inside `_variance_boundary`/`_temporal_boundary` was
hidden behind a warning. Owner direction: catch the expected (fit-shape) errors only, rethrow the rest.
Behaviour change: an unexpected exception in a boundary diagnostic now propagates out of `drm()`.
Out of scope (left on #1003): a tagged `:not_covered` return so `check_drm` can distinguish "not covered" from "clean".
Rose audit: pending.
