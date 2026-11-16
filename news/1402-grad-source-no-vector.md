- **A bridge fit that sends no gradient vector no longer claims `grad_source = "forward"` (#1402).**
  The top-level `grad_source` is the provenance of the `"gradient"` vector drmTMB reads.
  Routes with no stored gradient callback still omit that vector. They now omit the
  top-level source as well, unless it is already `"none"` or `"unavailable"`.
  The nested diagnostics keep the scalar `max_abs_grad` and its real producer
  (often `"forward"`). `check_drm` in Julia is unchanged.
