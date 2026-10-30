# After-task: bootstrap default `failures = :skip` (#1052)

Shannon speaking, with Noether (bootstrap status contract) and Rose
(claim-vs-evidence). No subagents ran.

## 1. Goal

Close #1052 from current `main` (`64ccb2d8d`). A thrown bootstrap refit must
be recorded and the successful draws kept, matching drmTMB. The abort stays
available as `failures = :error`. `used == 0` still errors.

#1048 and #1070 were also named for this pass. Both are already on `main`
(#1048 closed by #1202; #1070 closed after #1100). They are not restaged.
The #1048 phylo fixture still dropped one EM refit on Julia 1.10.10
(`m = 2`); this slice widens it to four observations per tip so the default
`check_converged = true` keeps both replicates. `check_converged = false`
is not restored.

## 2. Implemented

- Default `failures` on `bootstrap_ci`, `bootstrap_summary`,
  `bootstrap_result` and `_bootstrap_result` is `:skip`.
- All-fail error names the first failure's replicate, seed and message.
- `failures = :error` still aborts after the thrown refits are recorded.
- Docstrings updated. `bootstrap_sigma_a` still maps a non-`:error` mode to
  its existing `:warn`.
- Failing-first test: `test/test_1052_bootstrap_failures_skip.jl`.
- `test/test_bootstrap.jl` phylo EM block: `m = 2` → `m = 4`.

## 3a. Decisions and rejected alternatives

- Owner decision 18 (11 Oct 2026) is the source: change the default, do not
  invent a success floor. Draft #1055 already implements this on a dirty
  stacked branch. This PR restages only #1052 from current `main` so the
  default change is reviewable without the rest of that draft.
- Level-1 "do not change package defaults" is followed except for this
  already-settled owner decision.
- Do not restore `check_converged = false` for the phylo sibling (#1048).

## 3b. Evidence

- Red on unmodified `main`: the new test threw
  `bootstrap failed in 1 of 4 replicates` under the default, and the all-fail
  path used the `:error` message rather than `all 2 bootstrap replicates failed`.
- Green after the default change: `test/test_1052_bootstrap_failures_skip.jl`
  10/10.
- Phylo sibling on the #1202 fixture (`m = 2`, seed 12 / boot 13): one EM
  refit reported `converged = false` (log sd_phy ≈ −3.18) and was dropped.
  With `m = 4`, `used == 2` and `failed == 0`.
- Focused siblings, Julia 1.10.10, `--check-bounds=yes`, BLAS=1:
  `test/test_bootstrap.jl` 46/46; `test/test_triage_g_twin_inputs.jl` 135/135.

## 4. Files touched

- `src/inference.jl`
- `test/test_1052_bootstrap_failures_skip.jl`
- `test/test_bootstrap.jl` (phylo `m`)
- `test/test_triage_g_twin_inputs.jl` (comment only)
- `news/1052-bootstrap-failures-skip.md`
- `docs/dev-log/check-log.d/2026-10-30-1052-bootstrap-failures-skip.md`
- this after-task

## 5. Rose audit

- Claim: default records a thrown refit and returns `bootstrap_incomplete`.
  Evidence: the new test, red-then-green on Julia 1.10.10.
- Claim: #1048 phylo block keeps both replicates without
  `check_converged = false`. Evidence: `used == 2` after `m = 4`.
- Scope: one decided default, the all-fail message, and a test-data width.
  No engine or formula change. No drmTMB source vendored.
- Drift: #1048 and #1070 stay closed; this PR does not claim them.
  #1070 docs (adaptive GHQ, 5 nodes for ordinary `(1|g)`) are already on
  `main` via #1100.

## 6. License

MIT. No drmTMB GPL source.
