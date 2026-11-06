# After-task: bootstrap thrown-refit recording (#1052)

Shannon speaking, with Noether (bootstrap status contract) and Rose
(claim-vs-evidence). No subagents ran.

## 1. Goal

Close #1052 from current `main` (`64ccb2d8d`) without flipping the public
default. A thrown bootstrap refit is recorded. `failures = :skip` keeps
the successful draws. The default stays `failures = :error` (abort).
`used == 0` still errors and names the first failure.

#1048 and #1070 were also named for this pass. Both are already on `main`
(#1048 closed by #1202; #1070 closed after #1100). They are not restaged.
The #1048 phylo fixture still dropped one EM refit on Julia 1.10.10
(`m = 2`); this slice widens it to four observations per tip so the default
`check_converged = true` keeps both replicates. `check_converged = false`
is not restored.

## 2. Implemented

- Thrown refits are recorded on every `bootstrap_*` path (already the
  `_bootstrap_result` contract). `failures = :skip` keeps the run;
  `failures = :error` (default) aborts after the thrown refits are recorded.
- All-fail error names the first failure's replicate, seed and message.
- Docstrings state the default is still `:error` and that `:skip` is opt-in.
- Failing-first test: `test/test_1052_bootstrap_failures_skip.jl`.
- `test/test_bootstrap.jl` phylo EM block: `m = 2` → `m = 4`.

## 3a. Decisions and rejected alternatives

- The caller asked this PR to keep the public default as `:error`. Decision
  18's default flip is not applied here.
- Do not restore `check_converged = false` for the phylo sibling (#1048).

## 3b. Evidence

- Explicit `failures = :skip` records a thrown refit and returns
  `bootstrap_incomplete`.
- No-keyword and `failures = :error` abort with
  `bootstrap failed in 1 of 4 replicates`.
- All-fail under `:skip` names `always fails here`.
- Phylo sibling on the #1202 fixture (`m = 2`, seed 12 / boot 13): one EM
  refit reported `converged = false` (log sd_phy ≈ −3.18) and was dropped.
  With `m = 4`, `used == 2` and `failed == 0`.

## 4. Files touched

- `src/inference.jl` (all-fail message and docstring wording; defaults stay
  `:error`)
- `test/test_1052_bootstrap_failures_skip.jl`
- `test/test_bootstrap.jl` (phylo `m`)
- `test/test_triage_g_twin_inputs.jl` (comment only)
- `news/1052-bootstrap-failures-skip.md`
- `docs/dev-log/check-log.d/2026-10-30-1052-bootstrap-failures-skip.md`
- this after-task

## 5. Rose audit

- Claim: default remains `:error`. Evidence: the new test throws without a
  `failures` keyword.
- Claim: `failures = :skip` records a thrown refit. Evidence: the same file.
- Claim: #1048 phylo block keeps both replicates without
  `check_converged = false`. Evidence: `used == 2` after `m = 4`.
- Scope: recording, explicit `:skip`, all-fail naming, and a test-data width.
  No engine or formula change. No default flip. No drmTMB source vendored.
- Drift: #1048 and #1070 stay closed; this PR does not claim them.

## 6. License

MIT. No drmTMB GPL source.
