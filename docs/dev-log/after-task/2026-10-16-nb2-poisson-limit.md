# After-task: NB2 at the Poisson limit

Shannon; no subagents were running. Three reports, one root: with no overdispersion the NB2 likelihood
has no interior maximum in sigma, and the fitter treated the resulting plateau as an ordinary optimum.
Fixes: say so (warning naming the Poisson limit; `converged` untouched), survive the line-search
assertion (#1039), and look once for a higher interior optimum before accepting the plateau (#1069).
Decision recorded: the interior refit is a pragmatic multi-start, not a root-cause fix of why the
search left the interior; the cause on the plateau is untraced. The `-8` threshold is a judgement
(NB2 and Poisson logLik differ by far less than any tolerance beyond it), not a calibrated cut.
Out of scope: location SEs at the limit (SE path, #1073 stack).
Rose audit: pending.
