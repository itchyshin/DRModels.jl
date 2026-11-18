# Active validation checkpoint

Goal: diagnose the prior full-suite timeout and finish default Julia validation.
The user approved this continuation on 5 November 2026. No merge or release.

Local root: `/Users/z3437171/.codex/worktrees/drmodels-fixed-priors-20261105/DRM.jl`.
Branch: `codex/fixed-effect-priors-20261105`; starting HEAD: `60175d6e2`.
Product source aggregate remains `a30e347285f925af2e128ed6350fafe4d6b9035668444606bfa2fcea13e26653`.
The two existing `.claude/settings.json` and `.cursor/hooks.json` modifications
are outside the task and must remain unstaged. No drmTMB edits are planned.

Remote root on Totoro:
`/home/snakagaw/hsq_work/drm-priors-20261105-01a1ad0b`.
Candidate copy: `julia-validation8`; evidence: `evidence/validation8`.
Eight native Pkg.test shards are running, each under a 170-minute timeout.
Master PID: 1612427. Exact process-group IDs are in process-groups.tsv.
Meter: `drmodels-prior-validation8-20261105`, estimate 150 minutes, total 8 exits.
Each exit receipt records completion only; the verifier determines pass/fail.

A 949-file code/test/project manifest and 202 auxiliary-file manifest match the
remote copy before tests. The Julia executable, root dependency manifest and
all eight Pkg.test sandbox environments have separate hash receipts. Only the
disposable runner is instrumented. Native discovery selects 554 files, partitioned
70/70/69/69/69/69/69/69; assertion files and thresholds are unchanged.

Next: query remote status-prior-validation.py; inspect any nonzero shard exit
immediately; retain raw logs. At completion copy receipts, verify every start
and completion against expected-partition.json, recheck source hashes, and
classify any failure against exact baseline 24ac6ca on the same host. Do not
restart the whole run or change tolerances without a measured reason.

The existing Julia worker independently identified the interrupted Tweedie
stack; the integration reviewer independently checked the partition and timing
wrapper. Reuse that reviewer for final evidence. Both are currently idle.
The exact baseline CI measured the crossed Tweedie test at 122m31.1s; this explains
why a 90-minute total budget was insufficient on that measured workload, without
establishing candidate speed or attributing the entire prior remainder to it.
