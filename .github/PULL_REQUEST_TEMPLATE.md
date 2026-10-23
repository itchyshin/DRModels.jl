<!-- DRModels.jl pull request. Keep slices narrow: one issue → one branch → one PR. -->

Closes #

## What changed

<!-- One paragraph. Which persona's lane? -->

## Definition of Done

- [ ] Implementation wired into the module
- [ ] Tests (failing-first where applicable) as `test/test_*.jl` — auto-discovered by
      `test/runtests.jl`, no edit to that file needed (see its CONFLICT-FREE
      REGISTRATION note)
- [ ] NEWS entry as a fragment in `news/<slug>.md`, not a `NEWS.md` edit (see `news/README.md`)
- [ ] Docstrings + a worked example
- [ ] Per-slice entry added to `docs/dev-log/check-log.d/` (not the frozen `check-log.md` table)
- [ ] After-task report in `docs/dev-log/after-task/`
- [ ] Rose audit — claim-vs-evidence, status tag honest, no doc drift

## Verification

<!-- Paste the commands you ran and their output. Verify before claiming. -->

- [ ] Engine not regressed (`bench/run_sparse_tmb_nd.jl` → logLik −256.51) *(if `src/` touched)*
- [ ] License boundary intact (no drmTMB GPL source vendored)
