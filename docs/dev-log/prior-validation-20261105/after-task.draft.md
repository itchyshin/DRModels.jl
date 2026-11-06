# Julia full-suite validation after the prior slice

Status: PASS for the pinned Julia 1.10.10 default suite. All eight shards exited 0; all 554 selected files completed. No merge or release.

## 1. Goal

Finish Julia validation for the fixed-effect prior slice after the earlier
90-minute full-suite timeout. Shinichi approved diagnosis and continued checks
in the isolated prior lane, with cleanup and PR #1651 protected and no merge or
release.

## 2. Implemented

Added task-specific timing, collection and verification tools outside package
source. A disposable runner prints and flushes each selected file's start and
completion around the existing include call. Eight native positional shards
cover the same default suite without changing assertions, tolerances or test
order within a shard. No numerical kernel, public API or CI workflow changed.

## 3. Mathematical Contract

The prior objective, coefficient selectors and inference restrictions remain
unchanged on source aggregate `a30e347285f925af2e128ed6350fafe4d6b9035668444606bfa2fcea13e26653`.
This task measures test execution and verifies completion. It does not change
an estimator or establish interval calibration.

## 3a. Decisions and Rejected Alternatives

Use the existing shard mechanism and preserve every selected assertion. Keep
optional parity and existing deferred tests under their original guards. Put
timing only in the disposable runner, so package source and the repository's
normal test runner remain unchanged. Retain raw logs separately from compact
completion transcripts and bind them with hashes.

## 4. Files Touched

New files live in `tools/prior-validation-20261105/` and
`docs/dev-log/prior-validation-20261105/`, with this report and a new per-slice
check-log entry. The original prior implementation is `07b25dc4`; its earlier
evidence closeout is `60175d6e`. The diagnostic plan checkpoint is `9a352386e`.
Existing `.claude/settings.json` and `.cursor/hooks.json` modifications remain
outside this task. No drmTMB source or protected cleanup work was edited. The R report has
a documentation-only follow-up note so its earlier timeout is not mistaken
for the latest validation result.

## 5. Checks Run

The retained timeout stack identifies `test/test_crossed_tweedie.jl:28`, whose
fit is at line 46. Its 352 completed summary rows account for 2,028.7 of the
5,400 seconds. The other 3,371.3 seconds include setup, compilation, untimed
work and the interrupted test; the old log cannot divide that remainder.

The exact baseline `24ac6ca` completed CI on Julia 1.10.13. Its crossed-Tweedie
testset took 7,351.1 seconds inside a 10,872-second test step. Thus the old
90-minute total budget was shorter than one baseline testset on that measured
host. These are not matched-host speed comparisons with the candidate.
[Baseline job](https://github.com/itchyshin/DRModels.jl/actions/runs/42056992643/job/127349180304).

The new Totoro run uses Julia 1.10.10, four Julia threads per shard and one
BLAS/OMP thread. The estimate is 150 minutes wall time, with a 170-minute cap
per shard. Native discovery selects 554 files, partitioned
70/70/69/69/69/69/69/69. A reviewer independently reconstructed this partition.
The remote copy matches 949 code/test/project files and 202 auxiliary inputs.
All eight sandbox Project and Manifest files have identical respective hashes;
the runtime snapshots and executable hash are recorded separately.

Final result: all eight shards exited 0 and all 554 selected files completed.
The full run took 7,891 seconds (2h11m31s), within the 150-minute estimate.
The crossed-Tweedie file completed in 6,820.025 seconds (113m40s), longer than
the entire earlier 90-minute budget. No test assertion or tolerance changed.
The verifier checks ordered markers, package verdicts, JET and parity status,
transcript/exit bindings, raw-log checksums, source and runtime provenance.
All checks pass; complete raw logs remain on Totoro and compact transcripts
and timing receipts are retained in the validation directory. Compact transcripts
trim trailing horizontal whitespace; the separately retained raw logs are
identified by their original hashes.

## 6. Tests of the Tests

The verifier accepts the expected completion format, including Julia's
scientific-notation timestamps. Six negative controls reject a timeout,
missing completion, duplicate file, missing package success, altered transcript
hash and mismatched exit receipt. Full verification also binds source,
auxiliary inputs, the timed runner, runtime snapshots and independent raw-log
checksums. A missing receipt or selected completion fails verification.

## 7a. Issue Ledger

No issue, PR, comment or release was created. Existing issue #737 identifies the
crossed-Tweedie test's original feature; this task does not close it or change
that feature. The validation plan and receipts remain on the isolated branch.

## 8. Consistency Audit

The Tweedie test and its numerical kernel files are byte-identical to baseline.
The new runner preserves the include order and BLAS restoration check. JET is
available in the test environment; opt-in parity remains off. No family or
capability status, signed tolerance, likelihood target or license boundary was
changed. Final independent claim review is recorded in the validation receipts.

## 9. What Did Not Go Smoothly

The earlier budget did not account for the completed baseline's long Tweedie
test. Buffered stdout then obscured which file was active. The retained signal
stack located it; the new stderr markers provide direct per-file timing.

The first status parser did not accept scientific-notation timestamps and was
corrected before interpreting progress. Independent review found that the first
verifier checked markers without binding hashes. It now checks receipts,
source and runtime provenance, including eight distinct sandbox paths. The
launcher recipe also refuses future reuse of an existing evidence directory;
the active run used the recipe checkpointed at `9a352386e`.

The R closeout checker initially refused to re-run its oracles because R
startup reordered PATH. Rebinding the already approved read-only checks
resolved that harness failure; all three original receipt checks still pass.

## 10. Known Residuals

The default suite completed on the pinned Julia 1.10.10 environment.
The separately gated parity suites and three pre-existing deferred test files
are not admitted by a default-suite pass. Other Julia versions, operating
systems, coverage calibration and the full documentation site are outside
this continuation. The earlier matched-baseline documentation failure remains
separate; no documentation-audit PR was altered.

## 11. Team Learning

Read completed baseline phase timings before setting a full-suite budget.
Record starts and completions at the file boundary, and flush them separately
from test summaries. Count a shard as passed only after its package verdict,
exit code and complete selection agree. Pin runtime dependencies alongside
source so later work can reproduce the tested environment.

## 12. Cross-Product Coverage

This continuation covers the default Julia package suite on the pinned
Julia 1.10.10 environment, including the first prior slice and existing
univariate/bivariate regression tests selected by the native runner. It adds
no scientific capability.

This does NOT cover opt-in R parity, new inference calibration, SD priors,
posterior sampling, a wider recovery campaign, release or deployment.
