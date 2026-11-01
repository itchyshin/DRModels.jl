## 1. Goal

Reconcile current DRM parity PR and CI states with lane owners, update the Julia parity crosswalk from exact reviewed evidence, and align a stale Wald-test label with current behavior.

## 2. Implemented

Updated the internal Julia parity crosswalk with the exact R `fixef()`
dispatch limitation for Julia bridge classes, a date-stamped export-name
comparison for exact R and Julia main pins, and the current Julia-main CI
failure with its unmerged fixture-only PR. Recorded the maintainer-selected
`meta_V` contract (`sigma(fit) = τ` in both packages), the exact-main accessor
mismatch, and the retained source-matched candidate evidence, plus the
still-unimplemented dense univariate covariance-mode parity decision. Corrected
the stale claim that the public capabilities page omits SkewNormal, q4 REML,
and chi-bar-square; documented the current hurdle-NB2 `fitted()`/residual gap
and its old-pin numeric receipt boundary; and summarized the unresolved
current-source validation holds without promoting historical calibration
results. Corrected three stale source-line citations. Existing route-status
counts and unrelated dirty files remain intact. After reviewing the current source and test, corrected the sigma and random-effect Wald test labels that still said z/p were blanked although the assertions require finite values. Added the live Julia `meta_V` guide mismatch, stale Julia README version, and drmTMB GitHub release-description conflict with the live CRAN record to the crosswalk. These notes separate published wording, exact-main behavior, the unmerged candidate, and tag/release/registry records; no external text or page was changed. A Totoro exact-main paired hurdle-NB2 diagnostic measured a 1.2416 maximum fitted/residual gap while dpars and log likelihood agreed closely. The ordinary R bridge test file still errors before its output assertions because `fixef()` dispatch fails; the diagnostic avoided that extractor without changing package source. Curie's read-only debt review identified a paired q4 phylogenetic REML admission and current-source replay as a separate next validation proposal; no such run was launched or approved here.

## 3a. Decisions and Rejected Alternatives

Assumptions without asking: current `main` is the baseline for crosswalk
claims; if an owner admits a different source pair, rerun the ledger against
that pair. I did not promote a candidate change based on its PR status or
green checks. I left the broader SD work out because it remains in an
unmerged lane. I recorded the PR checks for #1435 and #1653 as owner-reported
because the workflow API did not return their jobs.

## 4. Files Touched

- `docs/design/capability-status.md`
- `docs/dev-log/after-task/2026-11-01-parity-crosswalk-reconciliation.md`
- `docs/dev-log/check-log.d/2026-11-01-parity-crosswalk-reconciliation.md`
- `docs/dev-log/evidence/cross-repo-parity/hurdle-nb2-20261101/` (paired diagnostic, native controls, scripts, logs, and receipt)
- `test/test_summary_zp_suppress.jl`

## 5. Checks Run

The test-only edit changes labels and comments, not assertions or runtime code. No test suite was run for that wording-only change. The targeted file was clean before editing; its preflight-reported rename-branch blob was byte-identical to the current file.

The raw `test.log` contains trailing whitespace emitted by testthat. It is kept
byte-for-byte and passes its SHA-256 receipt; `git diff --check` applies to the
report, check-log, and label-only test edit, excluding raw `.log` artifacts.


`python3 tools/parity_ledger.py --drmtmb '/Users/z3437171/Dropbox/Github Local/drmTMB' --ref origin/main --julia-ref origin/main` exited 0. It read R 0.7.1 at `b535bfbcce2a9039d869c2e34fe9da476394bf62` and Julia 0.7.2 at `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716`; found 73 versus 197 exports, zero unaccounted R export gaps, 15 documented raw name mismatches, 40 Julia-only exports, and closure PASS for 37 route rows plus 18 refusal gates. This is name and ledger accounting, not behavioral parity.

The retained `meta_V` candidate receipts were rechecked read-only. All seven
listed local source/test hashes match the Oct. 28 accessor receipt, and the
focused-test and same-fixture manifests pass `sha256sum -c`. The receipts
support candidate-only accessor and simulation behavior. They do not exercise
the separately approved dense univariate covariance mode. Exact-main R and
Julia sources and the dirty candidate were compared; no engine source was
changed in this slice.

Live PR API checks on 2026-11-01:

- DRModels.jl #1210 is open, non-draft, mergeable. Exact-head CI run `40355893345` and Documenter run `40355893421` passed; preview status is successful. No reviews or inline threads. The PR body still says final render reconciliation and `/dev/` publication are pending.
- DRModels.jl #1214 is open, draft, mergeable. Exact-head CI `40466175353`, Documenter `40466175340`, and preview passed; no reviews or threads.
- DRModels.jl #1215 is open, draft, mergeable. Exact-head CI `40500799901`, Documenter `40500799934`, and preview passed; no reviews or threads.
- DRModels.jl #1216 is open, non-draft, and mergeable at exact head `2d4dc87294d5c6e67b1f834c767c24639ddd0987` over current main. Exact-head CI `41152894330` passes all four Julia shards and `ci-ok`; Documenter `41152894311` passes docs and preview, with deploy skipped. The PR body records a seed-13 phylogenetic EM refit reaching `log(sd_phy) ≈ -3.18` and being dropped as nonconverged. Changing the fixture from two to four observations per tip stabilizes that test and does not repair the EM fit.
- DRModels.jl #1226 is open, draft, and mergeable at exact head `26087694a72ea645aa3191752dbd32f0fc0ae758` over the same main. Exact-head CI run `41244566371` passed all four Julia shards and Documenter run `41244566368` passed docs and preview, with deploy skipped. It makes the same `m = 2` to `m = 4` edit to `test/test_bootstrap.jl` as #1216. The PR body reports 14/1,000 boundary refits at `m = 2` and 0/1,000 at `m = 4` on Julia 1.10; the change is test-only and does not alter the EM method. It has no assignee, PR comments, reviews, or inline threads. The #1224 issue discussion says to keep the underlying EM-boundary question open and distinguish a fixture change from resolving that question. The current #1216 body still has an older sentence saying no open issue tracks the boundary, but appended text now references #1224; the issue link is present while that sentence remains internally stale. No owner has recorded a resolution of the duplicate fixture edit; keep the fixture-only change distinct from #1216's bootstrap-status implementation until they coordinate.
- drmTMB #1435 is open, non-draft, mergeable at `d012cc28593691168167e07c3f8e0ff9122ccc61` over base `b535bfbcce2a9039d869c2e34fe9da476394bf62`. No reviews or threads. Its latest PR comment from the lane says run `41133539626` passed all six checks; that run ID could not be fetched and current `gh pr checks` returned no checks, so the result remains owner-reported. Current R main `R/zzz.R` confirms the bridge-class registrations are absent.
- drmTMB #1653 is open, draft, mergeable at `181ae21ed00f5a85682509b6c7397eda15f3339a` over `b535bfbcce2a9039d869c2e34fe9da476394bf62`; no reviews or threads. Its lane comment reports six green checks on run `41133418965`, but the run ID could not be fetched and current `gh pr checks` returned no checks; treat the result as owner-reported.
- drmTMB #1627 is open and draft at `36dc7a70cabcf0dec8428af4bd14fa2e0fc0625b`. Its exact-head run `41272767129` failed all four Ubuntu release shards at generated-ledger validation; `os-matrix` and source-tree tests passed. Each shard reports the same `mc-0568` mismatch: receipt blob `8b715877f3be24c9fa4aaffb53f4b1c431780f70` versus candidate `R/drmTMB.R` blob `b887d0d00573544f12607671288aa2dbe8423a93`, while the authenticated model-15 source fingerprint matches. Independent Git object lookup confirms `8b715...` is the file blob at both current main `b535bfbcce2a9039d869c2e34fe9da476394bf62` and the PR base; only the candidate changes it. The release jobs stop before `R CMD check`. This identifies candidate-specific file-provenance drift; package tests did not run. The PR metadata has no assignee or owner comment. Its lane must regenerate and review the receipt before the full checks can run.
- DRModels.jl main CI run `41315894403` is on exact main SHA `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716`. Its Julia 1.10 shard 2 failed in `test/test_bootstrap.jl:105-106`: `br_phy.used == 2` evaluated as 1 and `br_phy.failed == 0` evaluated as 1 (44 pass, 2 fail in that file). At 15:54 UTC, Julia 1.10 shard 1 and Julia 1 shard 1 were still in progress; the other six completed Julia shards passed. A 16:01 UTC read found the same two shards active. At 16:16 UTC, the workflow still had no terminal conclusion, with the same two shards active and the shard-2 failure unchanged. At 16:31 UTC, the same run remained in progress: six of eight Julia shards had completed (five successful, one failed), and Julia 1.10 shard 1 plus Julia 1 shard 1 were still active. No rerun was requested. Both #1216 and #1226 have independently verified green exact-head CI, and both edit the same fixture file. Their statuses do not resolve their duplicate ownership or the separate EM-boundary question, and neither candidate establishes interval calibration.
- The GitHub Pages API reports the Julia site build as successful, but its latest published source commit is `645b435`; comparing that source with current main `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` shows main is two commits ahead. This confirms that a documentation build exists, not that the current-main docs changes are published.
- drmTMB main is `b535bfbcce2a9039d869c2e34fe9da476394bf62`. Scheduled pkgdown run `41217879763` passed on that SHA. A fresh scheduled run `41329710630` on the same exact main SHA also passed, including both `pkgdown` and `deploy` jobs. This is current-main deployment workflow evidence; no separate live-page retrieval was performed in this continuation. Exact-main receipt-staleness run `40681027801` failed at the receipt detector. The newest full `R CMD check` found is run `40375071745`, failed on older SHA `02095c67d7e70bbb648f45d2bbdd5aa0b1ea7be5`; no full `R CMD check` result for current main was found.

Lane reconciliation: the documentation-audit owner confirmed the exact main pins, the bridge dispatch defect, and that their lane did not establish an edit to the protected capability ledger. The SD lane remained separate and unmerged; no SD route was added to the current-main crosswalk. The bootstrap PR authors' recorded evidence agrees that the fixture change is test-only and that the EM refit remains correctly excluded when unconverged. The current #1216 body now references #1224 but still contains an older sentence saying no issue tracks the boundary; no direct handoff resolving the duplicate fixture edit was found. A fresh file preflight emitted 31 live lanes, identified three refs with different `capability-status.md` content, and listed open PR lanes, then exceeded its 8-second timeout before its full scan completed. I reviewed the three reported diffs; the differences are retained, and the user-authorized scope and existing lease restricted this update to the capability page and its two slice records.

`git diff --check` passed. The crosswalk status table still has 51 rows.
`python3 tools/check_capability_citations.py` passed with 2 citations verified
and 17 not precisely resolvable. After the live CI refresh, the after-task
structure check passed; the absolute-path `slop_check.py` run on this report
reported 0 findings. The report
passed the brain evidence
promotion gate with 8 registered claims. The after-task structure validator
passed from the package worktree. The first unguarded `closeout.py check`
failed because its R validator was launched from the brain repository root;
rerunning the compiler with its recursion guard passed after the full R
validator had already passed from this package root. The exact-main parity
ledger command above exited 0. No full `Pkg.test()` or package-wide check was
run. The focused exact-main bootstrap fixture reproduction below is the only
local fit/test executed in this continuation.
The refreshed four #1627 release-shard logs were read. Their identical
`mc-0568` failure was compared with `R/drmTMB.R` Git blobs at the exact main,
PR base, and candidate commits. It is a provenance guard failure before
`R CMD check`, not evidence of a package test failure.

Focused exact-main Julia reproduction: archived source SHA
`64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716`; archived
`test/test_bootstrap.jl` SHA-256
`ee4c5a8e0aa5752e92d48b69c3e8ca24eec19c4b1e6ae04c4576d47e727ca57c`;
Julia 1.10.12, one Julia thread, one BLAS thread, offline-resolved test
environment in an owned temporary depot overlay. A scratch runner copied the
fixture block from lines 84-106 without changing the source tree. With `m = 2`,
the local run reproduced the two failing assertions: `used == 1` and
`failed == 1` for two attempts; the stored record says the refit did not
converge or landed on a degenerate optimum. Runner SHA-256
`455923451e70b08a82ebda3b4950645dae4cd323ba6a022bc7c6344521a7e9e6`, log
SHA-256 `837886cb7647c32f7c356e8849b3a24035daab449329a293816ff747d740ba84`.
Changing only `m` to 4 in the scratch runner passed all 3 assertions, with no
failed replicate; runner SHA-256
`cddb279042a230daafce2e5ea7566a779ab361a55a0dc6c769f57a34f328932f`, log
SHA-256 `41d7974b7d3099a8afe04472d16fe651f4aa7fbbe38a57a2c2e854374d1a66df`.
The temporary logs are under `/tmp/drmodels-main-bootstrap.1ykDCp/` and may be
purged. This is one fixture-specific reproduction, not evidence about the
algorithm's broader reliability; the generic failure message does not resolve
the #1224 question. No repository test or engine file was modified.

Additional exact-main hurdle checks (2026-11-01, Totoro): R native
`test-hurdle-nbinom2.R` passed 60 assertions in 4.83 seconds. Julia
`test_bridge_zi_marginal_mean.jl` passed 24 assertions in 39.13 seconds and
printed three boundary-Hessian warnings. R bridge
`test-julia-family-hurdle_nbinom2.R` ran three live cases but each errored at
`fixef()` with no method for `drmTMB_julia`; its output assertions did not run.
A one-pair diagnostic then used the same seed-20260623, n=1,800 fixture and
the exact main source copies. Both fits converged. Maximum dpar difference was
`9.7686e-13`, log-likelihood gap was about `2.6e-11`, and maximum fitted-value
and response-residual gaps were both `1.241592632021761`. Native R fitted
values agreed with the R hurdle formula reconstructed from Julia dpars to
`7.6672e-13`; Julia fitted values equalled its `mu` dpar to `4.4409e-16`.
The diagnostic bypassed the failing extractor but did not alter source, test
assertions, formula, or estimator. The receipt and logs are under
`docs/dev-log/evidence/cross-repo-parity/hurdle-nb2-20261101/`.

## 6. Tests of the Tests

No tests were added or altered in this lane. The parity ledger printed its
accounted/unaccounted categories and closure result. The scratch exact-main
runner fails on the `m = 2` fixture and passes after changing only `m` to 4;
the assertions detect a non-usable refit, but this does not show whether the
optimizer's boundary handling is correct. The R native hurdle test and Julia
bridge unit test passed at their recorded scopes. The normal exact-main R
bridge file failed at the `fixef()` dispatcher before its output checks. The
separate diagnostic measured a large fitted/residual mismatch despite close
dpars and likelihood; see the receipt for exact pins, run commands, hashes,
warnings, and limits. The main-suite CI failure remains recorded; no
repository fixture or engine code was changed.

## 7a. Issue Ledger

Fixed in the crosswalk: the exact-source bridge dispatch caveat and export
name-accounting snapshot, and the exact-main `meta_V` scale discrepancy.
Deferred: #1435 remains an open candidate; #1216 and #1226 remain unmerged,
both have verified green exact-head CI, and overlap on
`test/test_bootstrap.jl`; the #1224 follow-up leaves the underlying EM boundary
question open, and #1216's body now links the issue while retaining a stale
sentence saying no issue tracks it; #1210,
#1214, and #1215 remain open documentation PRs; #1653 remains draft. Julia
main CI has two failed shards at the terminal refresh. drmTMB main has an
exact-main receipt-staleness failure; the local exact-main R CMD check
completed at 16:55 UTC with one `testthat` failure. The separate R source-tree
suite completed from an exact-main Git clone because its harness requires Git
metadata and a package tarball does not include `.git`.
drmTMB #1627 has four
release shards stopped before package checks by candidate-specific receipt
provenance drift; its lane has not documented a receipt refresh in the reviewed
PR metadata. The approved dense univariate covariance mode remains unimplemented
and file ownership is unresolved.

Live PR metadata and owner comments were rechecked after the CI refresh. #1216
remains open, non-draft, clean, and at the reviewed head; the maintainer's
comment says to ship after its exact-head checks pass and asks that #1052 be
referenced rather than closed. Its green suite run is verified. #1226 remains
open and draft at its reviewed head, with no owner comment; its exact-head
suite and Documenter runs are verified green. The two PRs still edit the same
fixture. R #1435 remains open and non-draft, while #1653 remains open and
draft; their comments report six passing checks each, but direct run lookup
returns HTTP 404, so those checks remain owner-reported. R #1627 remains open,
draft, and unstable at its reviewed head; its exact-head release validation
stops at candidate receipt provenance. No owner message was sent in this
refresh.

### Parity PR refresh at 17:35 UTC

Live branch queries reconfirmed Julia main `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716`
and drmTMB main `b535bfbcce2a9039d869c2e34fe9da476394bf62`. Julia #1242
(`c56ecfb3f528287bde3e179846456ba9a8fff7ca`), #1239
(`e8a6c20a8c19da2aa1616b4f33dc8df7b73ea34e`), and #1234
(`0b85b7cf2fb47b2a12d8edb555494ff2705614ee`) are open, non-draft, and their
exact-head Julia shards and aggregate checks pass. The PR bodies retain
integrator-specific broken comparisons; #1234's Gamma cell remains skipped
without a generated reference. #1240 remains a draft at
`03b93e680266369dff10e63eb4c846826a102ec4`; its exact-head run
`41248908491` fails on Julia shard 2 because the new fixture metadata lacks
`drmtmb_code_hash`. This fails the fixture-provenance test before its model
parity assertions. R #1659 is open and non-draft at
`1539b2b52df017aec9159e4d6ce288626e8f0b8b`; all six checks in run
`41248855941` pass. Its fixtures target the exact Julia main pin and include
14 fixed-effect/random-intercept cells, but six default-integrator cells are
skipped and the PR body reports only a one-dataset-per-cell comparison. The
five selected PRs have no GitHub assignees, comments, or review decisions;
their detailed scope therefore comes from their maintainer-authored PR bodies.
The crosswalk records them as candidate evidence only. No merge, PR edit, or
owner message was made.

### `meta_V` candidate repair reconciliation

The exact-main `sigma(fit) = tau` discrepancy remains open on Julia main, while
the dirty candidate implements the maintainer-selected accessor contract. A
separate stored candidate fix repairs the reproduced
`simulate(fit; newdata)` omission of known sampling variance. Its manifest
matches the present source: `src/gaussian_core.jl` SHA-256
`7f49c383a0f3ce646c0a9a1121e99d03a679c5a9117c7de417d4cadc6b94fc41` and
`test/test_meta.jl` SHA-256
`75938ec3eb2ad0024624471c6ade21e28f8fdfeca63ce6d15a818b64d96f69f4`. The
retained Totoro/Julia 1.10.12 run passed 13/13 assertions, including 12,000
new-data draws checked against `sqrt(v_new + tau_new^2)` and explicit
missing/invalid variance refusals. This closes that native fixed-effect
simulation defect for the source-pinned fixture only. It does not establish
bridge simulation, random-effect simulation, or exact-main behavior. PR #1210
is open with green exact-head CI, but its head does not contain this repair;
its PR body still says required checks are pending, so the live checks
supersede that sentence. No new implementation ownership or merge decision is
inferred; the implementation remains dirty and unmerged.

At 18:08 UTC I rechecked the seven Julia source/test hashes in the October 28
`meta_V` receipt against this dirty worktree: 7/7 match. All 14 retained logs
and runners listed in that receipt's `SHA256SUMS` also verify. This confirms
the identity of the stored focused tests and paired-fit evidence; no test was
rerun. The fresh file preflight still finds four refs carrying work on
`src/gaussian_meta.jl` and 31 live lanes, so I made no source changes.

### PR and CI owner refresh at 18:03 UTC

Re-read live metadata and exact-head workflow records for the PRs cited by the
crosswalk. Julia #1210, #1216, #1226, #1234, #1239, and #1242 each have
retrievable successful CI and Documenter runs on their recorded heads. #1240
has a failed CI run and a successful Documenter run; the known fixture-hash
failure is unchanged. drmTMB #1659 has a retrievable successful exact-head
`R-CMD-check`. These remain PR-candidate evidence, not merged-main results.

The latest #1216 lane note from `cursor[bot]` says it is ready for maintainer
review and repeats the successful exact-head checks. #1226 remains draft with
no conversation comment; it still edits the same bootstrap fixture as #1216,
and no lane-owner resolution is recorded. drmTMB #1435, #1652, and #1653 have
latest `cursor[bot]` comments reporting six green checks each, but the current
commit-run queries returned no records, commit status was empty, and direct
Actions URLs returned HTTP 404. Their CI remains lane-reported. #1435 is
non-draft in GitHub metadata although its body retains “Draft only”; the lane
comment says “Do not merge from this agent.” #1652 and #1653 remain draft.
drmTMB #1627 remains draft with an exact-head run failing at the generated
receipt-provenance gate before `R CMD check`; its PR conversation has no owner
status note. No owner message, PR update, merge, or source change was made.

### Julia main failure-owner refresh at 18:13 UTC

I re-read the exact job logs for main run `41315894403`: Julia 1.10 shard 2
fails `test_bootstrap.jl:105` because `br_phy.used` is 1 rather than 2; Julia
1.10 shard 1 fails `test_j3a_gradient_hessian_vcov.jl:158`, with 27/28 tests in
that J3a block passing. Current issue metadata shows #1224 open with no
assignee. #1216 is non-draft with green exact-head CI and a lane note that its
`m = 2` to `m = 4` change is a fixture change, not an EM fix; #1226 remains
draft, duplicates the fixture edit, and has no PR comment. No resolution of
the overlap is recorded.

J3a maps to #1190, open with no assignee; its R twin #1594 is closed. The
current failing fit is nonconverged and the retained source-pinned diagnostic
finds both the gradient and value Hessians non-positive. That does not prove
the #1190 acceptance failure, which concerns a fit reported converged with
unusable SEs. The current Ubuntu failure also contradicts the older issue
comment that Linux CI produced a positive-definite Hessian for this assertion.
Preflight finds two refs touching the test file, neither a current fix: the
J3d ref addresses a separate #1073 Hessian path, and the J3a red-test ref is
explicitly throwaway. No source owner or corrective PR was found; no assertion
or engine change was made.

## 8. Consistency Audit

Checked the Julia status table and terminal shard logs, both R route-status axes, exact R `meta_V`
documentation/tests, Julia main and dirty-candidate `meta_V` source and hashes,
the new-data simulation repair manifest against current source/test hashes,
the R bridge registration, the exact R route ledger, the two sets of export
names, candidate PR patches/comments, reviews and threads, exact-head checks on
both bootstrap PRs, the #1224 follow-up, and the current Julia-main CI job log.
Kept the 51-row capability count and the R bridge and claim
status counts unchanged. Checked the neighboring documentation-lane boundary
and left its public pages untouched.

## 9. What Did Not Go Smoothly

The lane preflight named many live lanes and historical refs, then timed out
before completing its scan. Its reported page diffs were reviewed against the
current file before continuing under the existing user authorization and narrow
lease. The first generated closeout
template resolved a relative path inside the brain repository; that accidental
template was removed, and the report was regenerated at its absolute path in
this checkout. The closeout wrapper's initial R invocation used the brain
repository as its working root, whereas the same validator passes when run
from this package root. GitHub PR-comment claims of green R checks for #1435
and #1653 were not independently confirmable through the workflow API; direct
run lookup returned HTTP 404, so they remain owner-reported. Julia bootstrap
PR #1216's exact-head CI and #1226's exact-head CI and Documenter run were
verified green. The maintainer comment on #1216 says to ship after CI passes
and requests referencing #1052 rather than closing it. The PRs still overlap
on the fixture file, so no owner resolution is inferred and no message was
sent in this refresh.

## 10. Known Residuals

Julia main CI run `41315894403` completed on SHA
`64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716`; six of eight Julia shards passed
and two failed. Julia 1.10 shard 2 fails the phylogenetic bootstrap fixture at
`test/test_bootstrap.jl:105-106` (one usable refit and one failed refit). Julia
1.10 shard 1 fails `test/test_j3a_gradient_hessian_vcov.jl:158`: the non-clamped
diagonal entries of `vcov(fit)` are not all finite in the seeded NB2 crossed
fit at a random-effect SD clamp. The exact-main test file matches the
worktree's `origin/main` blob. The full log shows this as the sole assertion
failure in that testset (27 passed, one failed, no errors). No source change
was made. The two open bootstrap fixture PRs remain overlapping and have no
recorded owner resolution; issue #1224 keeps the underlying EM boundary
question open.

A read-only reproduction in a pristine scratch clone of exact main narrows
the J3a case. Julia 1.10.12 marks the fit nonconverged; the maximum absolute
gradient over its free coordinates is `5.15e-3`. Its random-effect log-SDs
are `-6.389` and `-8.059`, with the latter at the lower clamp. The ordinary
stored-gradient Hessian has positive eigenvalues, but the same cold-started
gradient Hessian used by the SE path has minimum eigenvalue `-1.36e-5` on the
four coordinates left after fixing the clamped SD. The value-based fallback
also reports a non-positive-definite Hessian. This is evidence that the
existing assertion asks for finite Wald SEs at a nonconverged, near-boundary
fit where the SE guard withholds them; it does not prove a general Hessian
implementation defect. No test or engine change was made. The exact-main
reproduction and cold-stencil diagnostics are retained at
`/Users/z3437171/Documents/Codex/2026-11-01/drmodels-main-j3a-repro/`.

drmTMB main's receipt-staleness workflow failed; its latest full CI `R CMD
check` still refers to an older source SHA. The local exact-main check on R
4.6.0/macOS arm64 completed at 16:55 UTC with `Status: 1 ERROR`. Its one failed
assertion is `test-associate-pairs-bernoulli-bernoulli.R:40`: forward and
reverse Bernoulli association `eta` values were `0.551583976` and
`0.551583956`, against tolerance `1e-8` (difference about `2e-8`). The test
file SHA-256 `f487548071de2808d3f546bbeb90a83419beb7e66d6ff1b4b86846c484ec5be4`
exactly matches the file on `origin/main`; this is not a PR-only failure. The
suite reports 29,547 passes, one failure, 102 warnings, and 473 skips; 72 live
Julia bridge tests were skipped. Vignette rebuilding passed. The `student_shape_grid`
message was from a test that explicitly expects strict rendering to fail when
artifacts are missing, and was not counted as a failing assertion. No R source
or test was changed.

The built R tarball and check logs are retained at
`/Users/z3437171/Documents/Codex/2026-11-01/drmtmb-main-check-b535bfbcce2a/`.
The tarball SHA-256 is
`2b1ef044cb452e967fad5faca202f736058a1415de4d02189fc09841b4a9c50a`; log
hashes are in `SHA256SUMS` in that directory. The separate R source-tree suite
passed 26,519 assertions with zero failures, 14 warnings, and five skips in
577.5 seconds. `071-four-fixture-summary` passed 182 assertions in the Git
clone after an initial tarball run stopped because `.git` metadata was absent.
The raw log SHA-256 is
`876ea6d2f63ccf939d45a606f75bb31c727b7444e69bdc1376fa864713803791`; run
details and its checksum are in the retained evidence folder. Five skips were
one installed-baked-file path, three parity cases requiring `DRM_JL_PATH` at
the pinned Julia checkout, and the disabled Phase 18 stub. No live Julia
bridge tests ran in this source-tree configuration. No
current-main CI `R CMD check` result was verified. #1435 and #1653 remain
unmerged and owner-reported green; #1627 remains blocked before package tests
by candidate receipt provenance. R #1652 is a new open draft that changes
only the failing association-symmetry tolerance from `1e-8` to `1e-7`; its
lane reports six passing checks, but direct workflow lookup returns no
records, so it remains owner-reported and awaits maintainer review. On Julia,
#1216 is open, non-draft, and mergeable with independently verified green
exact-head CI and Documenter runs; #1226 remains an open draft with the same
fixture edit and green exact-head checks. Their overlap has no recorded owner
resolution. No interval-coverage claim is made.

The authorized checkout is
`/Users/z3437171/.codex/worktrees/parity-ledger-aliases/DRM.jl`, HEAD
`64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716`; this slice owns
`docs/design/capability-status.md`, the two dated parity-crosswalk records,
their hurdle-NB2 receipt directory, and the label-only change in
`test/test_summary_zp_suppress.jl`. Preserve every other dirty file. Julia CI,
the R package check, and the separate R source-tree harness are terminal. The
R `R CMD check` meter entry has been retired.

### Live-state follow-up (2026-11-01 19:08 UTC)

- drmTMB #1600 was already merged at `88e4c7c1b8812e1bec10e191b1b81b89d9bd6bfa`.
  All six checks listed on its exact PR head succeeded. No merge was performed
  in this follow-up. [PR #1600](https://github.com/itchyshin/drmTMB/pull/1600).
- DRModels.jl #1167 was already merged at
  `6b39b8cf20efba09987a6fa68667160a04ba8f00`. Its Julia 1 and 1.10 shards,
  aggregate check, and docs build succeeded; the preview check was cancelled
  and deploy was skipped. No merge was performed in this follow-up.
  [PR #1167](https://github.com/itchyshin/DRModels.jl/pull/1167).
- The R temporal-boundary test-only correction is pushed at
  `9630944f4c3b9e63d05ad0ad8945b0db11bcf3ad` on
  `codex/temporal-ou-cmdcheck-fix-20261101`. The focused test passed locally
  earlier on R 4.6.0/macOS; the branch has no PR or CI run. It does not establish
  a passing full `R CMD check`.
- Julia main run `41315894403` is terminal at
  `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716`: six of eight test shards passed.
  The bootstrap fixture and J3a covariance assertions failed. PR #1216 and
  draft #1226 still overlap on the fixture change, with no recorded owner
  resolution. Issue #1190 remains open; its current main failure is a
  nonconverged fit and is not the issue's converged-fit acceptance case.

No PR state was changed and no external message was sent in this follow-up.

### Live-state follow-up (2026-11-01 21:15 UTC)

- Julia #1245 remains open at exact head
  `0a338a0bc6f1532e7b48c029c82cfcd4b463b30c`. Run `41343324208` has the
  change-classification check and Julia shards 1/4, 3/4, and 4/4 passing;
  shard 2/4 is still running, and the scaling sweep was skipped. Documenter
  run `41343324175` passed the docs build and preview; stable deployment was
  skipped. Its PR description was refreshed at 21:08 UTC while preserving the
  19:08 historical snapshot.
- drmTMB #1662 remains open at exact head
  `01a2c2a49a1450cc79e9d35a138028ec1a27c720`. After review identified that the
  expected-status calculation repeated production predicates, the test gained
  a direct assertion that optimizer code 1 yields `degenerate`. The focused
  test passed locally in about 7 seconds. Run `41350326046` has a passing
  OS-matrix and source-tree test; all four Ubuntu release shards were still
  running at this refresh. Its PR description now distinguishes the previous
  green head from this pending exact-head run.
- Both PR descriptions were re-read after update and match their live heads.
  No PR was merged, and no stable documentation deployment was run. These are
  test and point-fit evidence only; they do not validate interval coverage or
  establish broad R–Julia parity.

### meta_V guide and overlapping PR ownership refresh (2026-11-01 21:30 UTC)

The current cached Julia main is `1d6ef3fe53a64ca680c034f3d297709b36cf1729`,
after PR #1216 merged. Its public meta-analysis guide still says
`sigma(fit)` is the heterogeneity SD, while `src/gaussian_meta.jl` stores
`sqrt(v + tau^2)` in `fit.scales[:sigma]`. The guide claim therefore remains
stale on current main. PR #1245 records the mismatch in its capability ledger,
but does not change the public guide.

Path preflight for `docs/src/model-guides/meta-analysis.md` found four refs
with edits absent from this checkout, including `rename/drmodels-20260917`,
`claude/docs-backlog-20260905`, and `cursor/pr423-ci-tol`. The working tree
still reports 31 live lanes and the 2026-10-04 handover to Cursor. I did not
edit the guide or infer an owner from branch or PR authorship.

The latest PR metadata shows #1106 open, draft, conflicting, with a failed
Documenter preview and no assignee or review; #1055 open, draft, conflicting,
with passing checks and no assignee or review; #1236 open, non-draft and
mergeable, with passing checks and no assignee or review; and #1232 open,
draft and mergeable, with Julia shards passing, its Documenter preview
cancelled, and no assignee or review. The latest #1055 comment says the #1052
change was restaged in #1216. These PR states do not establish ownership of
the dirty `meta_V` source. The earlier hunk audit found the `src/bridge.jl`
and `src/summary.jl` changes disjoint from those PRs, but the ownership
question remains open.

Independent reviews found no blocking issue in the scoped #1662 test change
and no false promotion in the reviewed #1245 ledger claims. At this refresh,
Julia run `41343324208` still has shard 2/4 active and drmTMB run
`41350326046` still has all four release shards active, on the same exact
heads recorded above. No PR, public guide, or source file was changed.

### Bootstrap fixture PR disposition refresh (2026-11-01 21:34 UTC)

PR #1216 merged at 2026-11-01 19:17:25 UTC as
`1d6ef3fe53a64ca680c034f3d297709b36cf1729`. Current `origin/main` contains
the four-observation phylogenetic bootstrap fixture (`m = 4`) in
`test/test_bootstrap.jl`. Diff review confirms #1226 changes the same fixture
from `m = 2` to `m = 4`; it remains open and draft at
`26087694a72ea645aa3191752dbd32f0fc0ae758`, with merge state `CONFLICTING`.
Its listed Julia checks and Documenter preview pass on that head, but no
assignee or submitted review is recorded. #1226 also adds its own explanatory
news note; that remaining documentation is not enough to infer a disposition
or owner. No PR or source was changed.

### Exact-head CI completion refresh (2026-11-01 21:40 UTC)

The exact-head drmTMB #1662 run `41350326046` completed successfully. Its OS
matrix, source-tree tests, and all four Ubuntu release shards passed. This is
current-head package CI evidence for the test-only change; the focused Mac test
remains narrower than a full macOS `R CMD check`. No merge was performed.

Julia #1245 remains open at exact head
`0a338a0bc6f1532e7b48c029c82cfcd4b463b30c`. In run `41343324208`, shards 1/4,
3/4, and 4/4 and the change-classification check passed, while shard 2/4 was
still running at the 21:40 UTC poll; the scaling sweep was skipped. Documenter
run `41343324175` passed its preview, with stable deployment skipped. No PR
metadata or source was changed. The Julia run still needs a final status check
before the local crosswalk commit can be pushed.

### Long-running Julia shard checkpoint (2026-11-01 22:04 UTC)

Julia shard 2/4 remains active in run `41343324208`, on exact PR head
`0a338a0bc6f1532e7b48c029c82cfcd4b463b30c`. GitHub reports its `Pkg.test`
step active since 19:46:17 UTC and does not make logs available while the job
is running. No failure or terminal result is reported. The local crosswalk
commit is `fd005581c679de5e8c4da013c77c38099048dbf9`, one commit ahead of the
remote PR branch; pushing now would replace the tested head and restart CI, so
the push remains pending. The scoped lease is released. All other dirty files
in this worktree belong outside this commit and remain untouched.

### Owner and PR-state refresh (2026-11-01 22:07 UTC)

Read-only GitHub API checks and independent PR-status reviews found no
assignees, submitted reviews, or open review threads for drmTMB #1659/#1662 or
DRModels.jl #1106/#1055/#1236/#1232. PR authorship and branch names are
provenance, not proof of current lane ownership.

- drmTMB #1659 is open, non-draft, mergeable at
  `1539b2b52df017aec9159e4d6ce288626e8f0b8b`; its exact-head R-CMD-check run
  `41248855941` passed. Its author reports a historical full check with one
  error and one warning, with the error reproduced on main; that attribution
  was not independently rerun in this refresh. The parity fixture checks point
  estimates and Wald SEs, not interval coverage; six default-integrator
  comparisons are skipped as expected approximation differences.
- drmTMB #1662 is open, non-draft, mergeable at
  `01a2c2a49a1450cc79e9d35a138028ec1a27c720`; its OS matrix, source-tree tests,
  and four Ubuntu release shards passed in `41350326046`. The local Mac result
  is the focused test, not a full Mac `R CMD check`.
- DRModels.jl #1106 is draft and marked “do not merge”; CI and the docs build
  passed, but its Documenter preview failed. #1055 is draft and marked “do not
  merge before #1044”; its CI and docs passed, but the stack dependency and
  owner decisions remain unresolved. Neither has a submitted review or owner
  assignment.
- DRModels.jl #1236 is open, ready for review, mergeable, and green on CI and
  Documenter. Its PR body records an unresolved choice about recommending
  Wald intervals in bridge errors. #1232 is open draft and green on Julia CI;
  its docs preview was cancelled and its body leaves behavior decisions open.

No merge, PR update, or owner assignment was made. Green CI alone does not
clear these PRs for merge.

Independent exact-head diff review adds two readiness distinctions:

- #1659 is source-attributed to Shinichi and its seven-family test design is
  internally consistent. It covers fixed effects and `mu` random intercepts
  for univariate log likelihood, working-scale coefficients, and Wald SEs.
  Six default-integrator comparisons are retained as explicit expected
  mismatch skips; they are not counted as parity passes. The live Julia layer
  covers eight bridge-admitted cases and can skip when its optional checkout
  or dependencies are unavailable. No bivariate, wider-structure, or interval
  coverage claim follows.
- #1236 changes `_bridge_fit` to convert recognized unsupported-keyword
  `MethodError`s into named `ArgumentError`s; it leaves fit defaults and
  estimates unchanged. Its new error test does not assert the conditional
  “wald is not served” clause. It leaves the #1011 profile-failure message
  invariant untouched. The broader choice about recommending R-side Wald
  intervals remains unowned, but need not block this narrow implementation if
  the current wording is retained. No tests were rerun in this review.

## 11. Team Learning

Memory receipt: `route.py` returned no LOAD-FIRST manifest for this worktree.
The brain search returned no directly applicable new decision. I applied the
repository rules to preserve all unrelated dirty files, read the exact-source
PR and CI receipts, and distinguish merged main evidence from candidates.

### Naturalness assessment

Style: 4/10, medium confidence. Genre and coverage: technical status report;
reviewed §§7a and 9–10, the J3a diagnostic paragraph in the capability ledger,
the updated testset labels, and the new check-log row. The run IDs, source hashes, and measured Hessian
eigenvalue keep the update specific. The sentence “This is evidence that the
existing assertion asks for finite Wald SEs” in §10 could sound more final
than the evidence warrants; it is followed by the boundary and says this does
not prove a general engine defect. The live-state follow-up is also narrow: it
reports the PRs as already merged, calls out the cancelled preview, and keeps
the pushed R test branch separate from full-check evidence. No change needed.

Gates: science = pass for the limited numerical interpretation; the
nonconverged fit and cold-Hessian eigenvalue support withholding this fit's
Wald SEs but make no coverage claim. Facts = pass for the refreshed PR states,
terminal workflow records, exact-main test result, and scratch diagnostic
logs/checksums. References = pass for the directly checked PR/run records and
local file receipts. Inherited historical claims outside the reviewed sections
were not re-audited. Provenance: self-review of saved report v5 on 2026-11-01;
the follow-up PR states and branch identity were checked against live GitHub
and the remote ref. No held-out examples or controls were used.

Golden Set: not run. The Julia testset-name edit changes labels only; no
assertions or runtime code changed. The focused R test result is historical
and is recorded separately above.

### Naturalness assessment: 21:15 UTC follow-up

Style: 3/10, medium confidence. Genre and coverage: short technical status
addendum and its check-log row. The two bullets tie each PR to an exact head,
workflow, and pending state, so readers can distinguish passed checks from
active ones. No change needed. Gates: science = not applicable because this
update makes no model-validity claim; facts = pass against live PR metadata,
workflow jobs, and the local focused-test result; references = pass for the
linked PR/run records. Provenance: self-review of saved report v6 on
2026-11-01; the same status facts were checked against live GitHub. No held-out
examples or controls were used.

### Naturalness assessment: 21:30 UTC ownership follow-up

Style: 3/10, medium confidence. Genre and coverage: short technical status
addendum. It names the exact current-main commit and stored scale, identifies
the four unresolved guide refs, and records PR states without assigning lane
ownership. The current source, GitHub metadata, and file-level preflight were
checked directly. It makes no model-validation or public-readiness claim. No
held-out examples or controls were used.

### Naturalness assessment: 21:34 UTC bootstrap disposition follow-up

Style: 3/10, medium confidence. Genre and coverage: short PR-state addendum.
It names the merge commit, the exact remaining PR head, the shared fixture
change, and the conflict state. It separates passing checks from mergeability
and does not infer an owner or recommend closing the PR. Facts were checked
against GitHub metadata and current cached main. No held-out examples or
controls were used.

### Naturalness assessment: 21:40 UTC exact-head CI follow-up

Style: 3/10, medium confidence. Genre and coverage: technical status addendum.
It separates the completed drmTMB run from Julia's still-active shard, names
exact workflow and source identities, and retains the no-merge and no-deploy
limits. Facts were checked against the exact-head job records; no model
validity or interval-coverage claim is made. No held-out examples or controls
were used.

### Naturalness assessment: 22:04 UTC CI checkpoint

Style: 3/10, medium confidence. Genre and coverage: handoff status addendum.
It gives the exact run, job, head, and local commit, explains why the push is
pending, and says unrelated changes remain untouched. GitHub confirms the job
is active and that logs are unavailable before completion. No test or
calibration claim is inferred from the active run. No held-out examples or
controls were used.

### Naturalness assessment: 22:07 UTC PR ownership refresh

Style: 3/10, medium confidence. Genre and coverage: short technical PR-state
addendum. It ties each candidate to its exact source and checks, separates
author-reported full-check evidence from live Actions results, and records
missing owner/review signals without assigning ownership from authorship. No
merge or scientific validation claim is made. Facts were checked against
GitHub PR, review, comment, and Actions APIs. No held-out examples or controls
were used.

### Naturalness assessment: 22:11 UTC diff-review follow-up

Style: 3/10, medium confidence. Genre and coverage: technical PR-readiness
addendum. It narrows #1659 to the seven-family point-estimate/SE matrix and
states its explicit skipped comparisons; it describes #1236's behavior change
and leaves the wider Wald choice with the owner. No tests or merges are
claimed. The two PR diffs, tests, exact-head checks, and owner metadata were
reviewed directly. No held-out examples or controls were used.

## 12. Cross-Product Coverage

Covers: the exact-main Julia capability snapshot, the R-to-Julia `fixef`
dispatch caveat, export-name accounting, the observed current-main Julia
bootstrap fixture failure and its disposition (#1216 merged; #1226 remains a
conflicting draft), and the public Julia `meta_V`, README, and drmTMB
release-description drift, plus
the current-main hurdle-NB2 bridge test boundary and measured fixed-effects
output mismatch observed during this review. It also records the already-merged
state of #1600/#1167 and the pushed R test-only branch, with their evidence
limits.
It also records the live CI refresh on Julia #1245 and drmTMB #1662, including
the completed exact-head R run and Julia's remaining active shard at the 21:40
UTC poll, plus the 22:04 active-shard checkpoint. It records
the current-main `meta_V` guide mismatch and the unresolved ownership evidence
for the four refs that change that guide.

Does NOT cover: implementation or merge of any PR, package-wide R or Julia
acceptance, a current-main CI `R CMD check` result, deployed documentation,
runtime R-to-Julia behavior after a future merge, parity beyond the one
fixed-effects hurdle-NB2 cell, random-effect routes, or interval calibration
and coverage. The four refs carrying changes to the public `meta_V` guide were
not reconciled by owner. PR author/branch metadata for #1659/#1662 and
#1106/#1055/#1236/#1232 did not establish current lane ownership. No
public-page or release-metadata edit was made. Independent diff review found
#1659 source-attributed with narrow point-estimate/SE parity scope; #1236 has
one missing assertion for its conditional Wald refusal text and an unresolved
broader wording decision.
