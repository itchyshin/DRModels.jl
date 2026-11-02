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
unmerged lane. I recorded the PR-comment-reported checks for #1435 and #1653
as unverified because direct workflow lookup did not return their jobs.

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
It also records terminal exact-head CI for Julia #1245 and drmTMB #1662 as of
22:23 UTC, and the current failed checks for drmTMB #1663. The `meta_V` guide
mismatch and unresolved ownership of its four editing refs remain open.

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

### Final PR and CI status (2026-11-01 22:23 UTC)

Julia #1245 remains open, non-draft, and mergeable at exact head
`0a338a0bc6f1532e7b48c029c82cfcd4b463b30c`. Run `41343324208` completed with
all four Julia test shards and the `ci-ok` gate successful. Documenter preview
`41343324175` passed; stable deployment was skipped. The remote branch still
points at the tested head. Local commit `3c56b6dc32fccee253c8ff53aec780a5e049bdf3`
contains only the two crosswalk files and is one commit ahead; unrelated dirty
worktree changes remain unstaged.

drmTMB #1662 remains open at `01a2c2a49a1450cc79e9d35a138028ec1a27c720`.
Exact-head run `41350326046` completed successfully, including the source-tree
tests and all four Ubuntu release shards. Its Mac evidence remains the focused
affected test, not a full Mac `R CMD check`.

drmTMB #1663 remains open at `7309d3cd813f9c0f42da9974470c23d87da5bcf4`.
Exact-head run `41354826786` failed in the source-tree test and at generated
capability-ledger validation in all four Ubuntu shards, before package checks;
the OS matrix passed. The PR has no requested reviewers, submitted reviews,
review threads, or comments. Its broad source changes remain with its owner.
No PR was merged, no source from #1663 was edited, and no model, parity, or
interval-calibration claim follows from these check results.

**Style assessment:** 3/10, medium confidence; short technical status addendum.
“All four Julia test shards” gives a concrete scope; no wording change is
needed. **Gates:** science = pass for the stated CI-only scope; facts = pass
against current GitHub PR/run records; references = not applicable, no
literature claims. **Provenance:** self-review of this 22:23 UTC draft update;
no held-out examples or controls.

### Exact-head refresh (2026-11-01 23:01 UTC)

Read-only audit confirmed that PR #1245 is still open, non-draft, and
mergeable at exact head `10e28c250c12931d92efc9e07dbfdeceafa05e60`, based on
Julia main `1d6ef3fe53a64ca680c034f3d297709b36cf1729`. Test run
[`41357071138`](https://github.com/itchyshin/DRModels.jl/actions/runs/41357071138)
was still running with all four Julia shards active. Documenter run
[`41357071224`](https://github.com/itchyshin/DRModels.jl/actions/runs/41357071224)
passed the docs build and preview; deployment was skipped. The earlier 22:23
status remains a historical snapshot for head `0a338a0…` and run `41343324208`.

The current PR contains five commits across 30 paths, including the capability
ledger, parity-ledger source pinning and tests, the evidence crosswalk, the
checksum-bound Hurdle NB2 receipt, and a test-label change. No production
source files are in the PR diff. The latest pushed commit is only the CI-status
documentation refresh. Its local worktree remains dirty in files outside this
crosswalk. Do not stage those files.

The #1216 merge at `1d6ef3f…` includes the `m = 4` bootstrap fixture. PR #1226
remains an open draft and is now non-mergeable; its duplicate fixture edit is
already present on main through #1216, though no direct owner handoff on #1226
was verified. Checks cited in PR comments for drmTMB #1435 and #1653 remain
unverified because direct Actions lookup returned 404; label them
PR-comment-reported, not CI-verified. PR #1627 and #1663 exact-head checks
remain failed. PR #1662 exact-head checks remain green, with its OS matrix
limited to `FULL=false`.

**Next check:** refresh #1245 no earlier than 23:12 UTC, retaining the exact
head and run IDs. If the checks are terminal, append that result. If the head
has moved, stop and reconcile with the owner before editing further. No PR
metadata, source code, merge, or deployment action was performed here.

### Scope count correction (2026-11-01 23:08 UTC)

GitHub's compare API for base `1d6ef3fe53a64ca680c034f3d297709b36cf1729` and
head `10e28c250c12931d92efc9e07dbfdeceafa05e60` confirms five commits ahead,
three commits behind, and 30 changed paths. This replaces the earlier
33-path count. The PR remains based on the recorded main SHA; its branch is
not yet up to date with that base.

### CI poll (2026-11-01 23:12 UTC)

PR #1245 still points to head `10e28c250c12931d92efc9e07dbfdeceafa05e60`
over base `1d6ef3fe53a64ca680c034f3d297709b36cf1729`. Run `41357071138`
remains in progress: the change-classification job passed, scaling-sweep was
skipped, and all four Julia test shards were still running. Documenter
`41357071224` remains green from the earlier exact-head read. No new head has
been pushed.

The drmTMB #1662 run `41350326046` is now terminal and successful at exact head
`01a2c2a49a1450cc79e9d35a138028ec1a27c720`: source-tree tests, the OS-matrix
job, and all four Ubuntu release shards passed. The OS-matrix selector used
`FULL=false`; this does not constitute a full macOS R CMD check. The latest
main-branch R CMD-check run returned by GitHub is older run `40375071745` at
`02095c67d7e70bbb648f45d2bbdd5aa0b1ea7be5`; its macOS release job failed while
source-tree, Ubuntu, Windows, and OS-matrix jobs passed. Thus the red badge
does not report a check of current main `b535bfbcce2a9039d869c2e34fe9da476394bf62`
or of #1662's current head. GitHub returned HTTP 410 for the old macOS job
logs, so this refresh does not independently attribute that failure. No PR
was merged or otherwise changed.

### CI progress update (2026-11-01 23:16 UTC)

The PR head and base are unchanged. Run `41357071138` remains active; the
change-classification job passed, scaling-sweep was skipped, and Julia shard
4/4 has now passed. Shards 1/4, 2/4, and 3/4 remain in progress. This is a
material change from the 23:12 poll, not a terminal result. The local
documentation commit remains unpushed so it does not cancel the running
workflow. No other PR, source, or remote state changed.

### Reviewed PR and CI refresh (2026-11-02 01:23 UTC)

The live paired `main` pins are drmTMB `b535bfbcce2a9039d869c2e34fe9da476394bf62` and DRModels.jl `1d6ef3fe53a64ca680c034f3d297709b36cf1729`. These are the comparison baseline for this refresh. Exact-head candidate checks below establish PR-scoped evidence only; they do not establish merged-main behavior, interval calibration, or broad parity.

| PR | Exact head and current checks | Reviewed scope and boundary |
|---|---|---|
| DRModels.jl #1233 | `d80e5543cf13a99ec50934c2c95cc033740e5e14`; CI `41247477178` and Documenter `41247477288` passed; no submitted review | Test-only animal-A label guards and VA comparison behavior. The same-tag VA refusal remains `@test_broken`; a VA ELBO is not a Laplace log-likelihood. Test-count wording in the PR description is ambiguous. No blocker to human review for this bounded test-only scope. |
| DRModels.jl #1234 | `0b85b7cf2fb47b2a12d8edb555494ff2705614ee`; CI `41247508362` and Documenter `41247508281` passed; no submitted review | Student parity disagreements remain `@test_broken`; the Gamma cell is scaffolding and skipped because no reference fixture has been generated. #969 is separate. A comment says “same parameterisation” where it checks parameter count. No blocker to human review as evidence-only work. |
| DRModels.jl #1235 | `3421a1642e84e88a3ec6a5793862396b9c914a19`; CI `41247707278` and Documenter `41247707274` passed; no submitted review | Tests/docstrings assert current marginal-simulator fallback behavior and preserve the unresolved throw-versus-warning choice. Review found the sentence “Only a `meta_V(...)` plus random-effect fit throws” too broad: other structured/random-field and phylogenetic-slope paths also throw and are not covered. The helper tests show `nothing` fallback cases, not a complete `bootstrap_result` interval validation. The author should narrow or complete the diagnostic sentence before treating the docs claim as reviewed; `@test_broken` currently means a throw expectation, not a warning expectation. |
| DRModels.jl #1236 | `c1f95b5debc5cfc3142428ba966a91611a4ad90d`; CI `41247941735` and Documenter `41247941696` passed; no submitted review | Diagnostic-text change only. Recognized unsupported keyword calls become `ArgumentError`; unrelated `MethodError`s are rethrown. Wald wording avoids recommending Wald after profile failures. The #1011 wording decision is explicitly left open. No blocker to human review. |
| DRModels.jl #1237 | `503b3797acdf44dba38af1096ebce7fc338f6e9b`; CI `41249874224` and Documenter `41249874289` passed; no submitted review | Fixes RNG stability only for the target Poisson phylogenetic bootstrap fixture. Neighboring test blocks still use the global RNG. The separate boundary-interval finding remains open. No blocker to human review. |
| DRModels.jl #1238 | `0fee29a1f8c44a5d3775b7f60f258181ee2580ad`; CI `41248270821` and Documenter `41248270842` passed; no submitted review | Fixed-seed q4 bootstrap test checks both failed and usable refits, full failure accounting, incomplete status, and a dropped-refit warning. It avoids a brittle exact cross-platform failure count. Minor gap: warning count is not asserted equal to `brw.failed`. No blocker to human review. |
| DRModels.jl #1239 | `e8a6c20a8c19da2aa1616b4f33dc8df7b73ea34e`; CI `41248334383` and Documenter `41248334332` passed; no submitted review | Six fixture cells cover fixed effects and one random intercept across Beta, Binomial, and BetaBinomial. Three likelihood checks remain `@test_broken` for the Laplace versus adaptive Gauss-Hermite gap. Generator does not enforce the stated drmTMB version or convergence, a reproducibility risk but not a scope blocker. |
| DRModels.jl #1242 | `c56ecfb3f528287bde3e179846456ba9a8fff7ca`; CI `41249837752` and Documenter `41249837766` passed; no submitted review | Poisson/NB2 parity fixtures distinguish `:Laplace` from default adaptive quadrature. Default-route `@test_broken` assertions only test that a tolerance is exceeded; they do not bound or sign-check the discrepancy. Linked boundary-SE and crossed-effect issues are not tested by these single-intercept examples. No blocker to human review with that limitation visible. |
| DRModels.jl #1210 | `88b0a3d1d930528bffee292cf1c1d14d2f34c913`; CI `40355893345` and Documenter `40355893421` passed; no submitted review | Broad documentation/API-contract reconciliation and diagnostic text. Human review may start, but the final rendered-site audit must be completed against this candidate before merge/closeout; `/dev/` publication is a later gate. |
| DRModels.jl #1215 | Draft `abda662cce362dc6108349dda96ba40ed2844142`; CI `40500799901` and Documenter `40500799934` passed; no submitted review | One-file wording change labels ROADMAP.md as a historical phase snapshot and points to current capabilities/API guides; both links resolve. One minor ambiguity remains: a later `Target` subsection mentions an old drmTMB version and could look current unless explicitly covered by the historical label. No blocker to human review; PR remains draft and was not changed here. |
| DRModels.jl #1214 | Draft `b657264262757505c866bca8ea50e678335142c3`; CI `40466175353` passed all four Julia shards; Documenter `40466175340` passed docs and preview, deployment skipped; no submitted review | Large docs-audit candidate touching manual pages, tests, news, and source docstrings. Independent review found no major formula/example inconsistency in checked ordinal, family, and marginal-approximation examples; it distinguishes computational availability from interval coverage and VA ELBO from a likelihood. `which-scale.md` still calls bootstrap recommended at a true boundary despite no calibration evidence; review suggests describing it as an Hessian-independent computational option. This PR overlaps substantially with #1210, which should be reconciled to one canonical patch before merge. Human review may start while draft; merge needs overlap disposition and the boundary-bootstrap wording decision. |
| DRModels.jl #1222 | Draft `d2a750e1dd97123b68b4055edf8e1f5d62de4e30`; CI `41241169945` and Documenter `41241170042` passed; no submitted review | Adds docstrings and `@docs` entries for 16 internal exports, with coverage and news. Issue #985 comment `6400708085` records the owner decision: document remaining exports, add StatsModels for `@formula`, set `checkdocs=:exports`, and do not un-export. This PR still has `checkdocs=:none`; a scratch `:exports` run found six more symbols and an unresolved `lc_to_chol` reference, while the curated `pagesonly` setup/navigation test conflicts with surfacing the internal page. The decision is known; implementation/configuration remains unreconciled. Keep draft and issue open until the selected policy is completed. |
| DRModels.jl #1223 | Draft `64940f742395b19e22e2936697b6f5ccb23c2b4c`; CI `41240893500` passed all four Julia shards; Documenter `41240893497` passed docs and preview; no submitted review | The non-finite Hessian guard checks entries before `eigvals`, preserves `ArgumentError`, and tests the actual separated BetaBinomial case with `se=true` and `se=false` plus a multi-index NaN guard. Independent review found no correctness blocker and confirms the unresolved `se=false`/false-convergence issue is kept out of scope. User guidance should say rescaling helps conditioning but cannot fix a nonexistent finite MLE from complete separation; a different/regularized model is needed. A direct Inf/single-index test would strengthen the message's broader claim. |
| DRModels.jl #1228 | Draft `fe064dce02087cc69c9500f678967f6d258f848a`; CI `41245284485` passed; Documenter `41245284449` built docs but preview was cancelled; no submitted review | Notes two deliberate R/Julia route differences: hurdle Poisson support and structured BetaBinomial support. Both linked issues (#726, #740) ask for alignment decisions that this docs-only PR leaves open. No model change or issue closure is claimed; docs preview is still missing. |
| DRModels.jl #1229 | Draft `10c5c2b6633d8cdb0de84291288d7e8d59b16a15`; CI `41245468054` passed; Documenter `41245467942` built docs but preview was cancelled; no submitted review | Documents a claimed large-`sigma_b` quantile-residual accuracy limit and a separate unreproduced #930 data case. The PR explicitly says the tail limitation is not fixed and requests the missing tree/data or script for #930. These are reported limitations, not a verified fix or a reproducible regression test; preview remains incomplete. |
| DRModels.jl #1227 | Draft `dfcb5dcbe5e393ed44e9225d327afaf46f3fd560`; CI `41245221802` passed; Documenter `41245221782` built docs successfully but preview was cancelled; no submitted review | Moves contributor process notes from README and documents the three-way quantile-residual convention difference (Julia marginal, R code conditional, R help fixed-effect-only). Review found the opening cross-package residual claim generalizes one measured Gaussian phylo case to `phylo()`, `relmat()`, and `animal()` and needs narrowing or explicit unverified wording. The standard-normal caveat should say residuals are marginally standardized but dependent, so their set need not resemble an iid normal sample. The drmTMB convention decision is explicitly deferred to #1101. Keep draft until wording is corrected and Documenter preview passes. |
| DRModels.jl #1245 | `b1abfab8b9f2d6e483051edf7ac772090c067be7`; Documenter `41370506455` passed docs and preview, deployment skipped. CI `41370506532`: classification passed, sweep skipped, four Julia shards still in progress at 01:23 UTC. | Crosswalk/capability-ledger evidence PR. Do not push while those shards run; local unrelated dirty files remain outside the edited crosswalk scope. Reconcile the next crosswalk update with the terminal exact-head result. |
| drmTMB #1662 | `01a2c2a49a1450cc79e9d35a138028ec1a27c720`; exact-head R-CMD-check `41350326046` passed; no submitted review | Test-only temporal OU boundary classification. Focused Mac test passed as reported, but the separate local full Mac check found an unrelated Bernoulli-pair symmetry failure. GitHub OS-matrix run selected `FULL=false`; it is not a full Mac check. No merge was made. |
| drmTMB #1663 | `53504dd397ca7678b86d85b384b69d8c4ba81446`; run `41370624656` terminal failure | All four Ubuntu shards stopped pre-test at the model-15 `mc-0568` receipt fingerprint mismatch. Source-tree job `125040115444` had two failures in `test-reader-vignette-contracts.R`: a manifest/source-vignette mismatch naming absent `predictor-dependent-random-sd.Rmd`, and two reader-contract lint findings not detailed in the log. The live Julia bridge did not run. Keep the SD receipt runner and reader-contract correction with their existing owners; no code or receipt is changed here. |

This refresh did not submit GitHub reviews or messages and did not merge, deploy, or change source code. The parity claims remain scoped to the exact tests and approximations above. Merging or publishing still requires the applicable owner and release gates.

At 01:32 UTC, run `41370506532` for #1245 still has all four Julia shards active; classification passed and the scaling sweep was skipped. Its remote head remains `b1abfab8b9f2d6e483051edf7ac772090c067be7`. The local branch is five commits ahead at `3e52d81c8622241ab0b1368d26fc9aa1d5a38917`; cumulative diff against its upstream contains only this after-task report and matching check-log entry, and `git diff --check` passes. These docs updates stay local until the active CI reaches a terminal result; after pushing, checks must be verified again on the new head.

Independent follow-up: issue #985 comment `6400708085` confirms that the `checkdocs=:exports`/StatsModels/no-unexport choice for #1222 was already decided by the owner. The draft's missing symbols, `lc_to_chol` link, and `pagesonly` conflict are unfinished implementation, not an unresolved owner choice. For #1227, review found the residual-convention summary overgeneralizes one Gaussian phylo measurement to `relmat()` and `animal()` and needs narrower language; the marginal standardization caveat should also retain dependence. For #1214, review supports starting human review, but calls out its substantial overlap with #1210 and a bootstrap recommendation at true boundaries that should not imply calibrated validity. At 01:29 UTC, exact-head #1245 run `41370506532` still has the same four Julia shards active; remote head/base are unchanged. No code, PR metadata, or external comments changed.

The independent #1215 review found the banner and current-guide links sound. One small ambiguity remains in a later `Target` subsection that names an old drmTMB version; the blanket historical label likely covers it, but a local label there would make the scope clearer. At 01:30 UTC, exact-head #1245 run `41370506532` still has all four Julia shards active at the unchanged remote head. No branch update was pushed.

The read-only #1223 review found no correctness blocker in the finite-Hessian guard or its separated-BetaBinomial test. Its error advice should distinguish numerical conditioning from complete separation, where rescaling cannot create a finite MLE. A direct `Inf`/single-index check is a minor test improvement; the draft already preserves the unresolved `se=false` convergence issue. At 01:32 UTC, #1245 run `41370506532` remains active with all four Julia shards running; no new head or terminal result is present.

### Status refresh (2026-11-02 01:35 UTC)

PR #1245 remains open at exact head `b1abfab8b9f2d6e483051edf7ac772090c067be7`, based on `1d6ef3fe53a64ca680c034f3d297709b36cf1729`. Documenter run `41370506455` passed its docs build and preview; stable deployment was skipped. CI run `41370506532` has classification success, scaling sweep skipped, and all four Julia shards still in progress. The six local documentation commits remain unpushed to avoid replacing the tested head. Once the run is terminal, recheck the remote PR head before pushing and then verify checks on the pushed head.

For drmTMB #1659, the live PR still points to `1539b2b52df017aec9159e4d6ce288626e8f0b8b` over current main `b535bfbcce2a9039d869c2e34fe9da476394bf62`. The earlier crosswalk records six successful checks in run `41248855941`, but the current Actions lookup for that run returned 404, and the current commit workflow/status queries returned no associated runs or statuses. Keep the earlier result as stored historical evidence, not a newly retrievable check. Its fixture provenance still targets DRModels.jl `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716`, behind current main `1d6ef3fe53a64ca680c034f3d297709b36cf1729`; current-pair parity is unverified. No test, PR, source, merge, or external state changed.

### Safe stopping point (2026-11-02 01:38 UTC)

The latest check of #1245 at 01:35 UTC still showed all four Julia shards running at unchanged head `b1abfab8b9f2d6e483051edf7ac772090c067be7`. The local branch now has seven committed crosswalk updates ahead of that remote head; `git diff --name-only b1abfab8b9f2d6e483051edf7ac772090c067be7..HEAD` contains only this report and its paired check-log file, and the diff passes `git diff --check`.

The handoff gate at 01:37 UTC did not pass: this Julia worktree has 70 uncommitted paths, seven local commits ahead on the current branch, and 197 unpushed commits on other branches. The remaining dirty paths and branches belong to concurrent or unverified lanes and are protected. No unrelated files were staged or changed. Carry this branch forward; after run `41370506532` reaches a terminal result, recheck PR #1245's head, push only the seven crosswalk commits if the head is unchanged, and verify checks on the resulting head. Do not merge or publish.

### Previously authorized PR merge verification (2026-11-02 01:39 UTC)

The live GitHub API confirms drmTMB #1600 is closed and merged at
`88e4c7c1b8812e1bec10e191b1b81b89d9bd6bfa` from exact head
`80c8cad31ca178ac2b9103b830d3265f7d976f64`. DRModels.jl #1167 is closed and
merged at `6b39b8cf20efba09987a6fa68667160a04ba8f00` from exact head
`ee16f8f4ad1b26edfb46ce9b486cc57112aec7ba`. Both merge commits are ancestors
of the current main refs, drmTMB `b535bfbcce2a9039d869c2e34fe9da476394bf62`
and DRModels.jl `1d6ef3fe53a64ca680c034f3d297709b36cf1729`. Their historical
run records show successful required package test shards; #1167's docs preview
was cancelled and deployment skipped. This confirms the prior merges and main
lineage, not live documentation deployment or broader calibration. No PR was
changed in this verification.

### Why the #1245 push remains gated (2026-11-02 01:40 UTC)

At 01:39 UTC, PR #1245 remained at `b1abfab8b9f2d6e483051edf7ac772090c067be7`
and its four Julia shards were active. The exact-base workflow
`.github/workflows/CI.yml` at `1d6ef3fe53a64ca680c034f3d297709b36cf1729`
uses a concurrency group based on workflow and branch, with
`cancel-in-progress: true`. Pushing to this PR branch now would cancel its
active run and start checks on the new head. Keep the local crosswalk commits
unpushed until this run is terminal; then verify the PR head, push the scoped
branch, and wait for checks on that new head.

### drmTMB #1656 process-log PR review (2026-11-02 01:42 UTC)

Exact head `2e9556282c914d37fce3196cce7da62b07e2dabe` is open, non-draft, and
mergeable over current main `b535bfbcce2a9039d869c2e34fe9da476394bf62`. The
diff freezes historical `check-log.md`, directs new entries to
`check-log.d/`, and adds a five-assertion process-contract test with census
updates. It does not change model likelihoods or defaults. A Cursor bot comment
reports six successful checks in run `41144231647`, but current run lookup
returns 404 and the commit has no retrievable check runs or statuses. Treat
that pass as historical reported evidence. No submitted review, open review
thread, or assignee is present; the bot's request for merge approval is not
human review. This bounded PR is not yet merge-ready: it needs retrievable
exact-head checks and owner or human review. No PR state was changed.

### drmTMB #1657 bridge-order PR review (2026-11-02 01:42 UTC)

Exact head `6d21c804afa392d0b77c370631daecdc220b5f98` is open, non-draft, and
mergeable over current main `b535bfbcce2a9039d869c2e34fe9da476394bf62`. Its
changes preserve row names for labelled Julia spatial coordinates and add an
offline permuted-payload test plus a live TMB-versus-Julia relmat K-order
check. They do not change the likelihood or model defaults. The final commit
adds the row-order assertion requested in a maintainer issue comment. No P0/P1
correctness issue was found; the offline fixture's unseeded `rnorm(4)` is a
minor clarity issue because the generated values do not affect the payload
oracle.

The PR body cites CI run `41235399713`, but the run lookup returns 404; current
commit workflow and status queries return no checks for the final head. The
issue reproduction cites DRModels.jl `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716`,
older than current main, so that reproduction is historical. There is no
assignee, formal review, or inline review thread. The branch is a good human
review candidate, but exact-head checks need to be rerun or made retrievable
before it is treated as merge-ready. No source or PR state changed.

### drmTMB #1651 source-pin documentation PR review (2026-11-02 01:44 UTC)

Exact head `d228429a28633b0657e569630128c66b578f255c` is an open draft over
current main `b535bfbcce2a9039d869c2e34fe9da476394bf62`. This docs/source-pin
change updates R documentation, NEWS, generated Rd, and a semantics test, with
a C17 receipt/check-log recertification; it does not change likelihoods,
defaults, or the API. The body reports four live family round-trips using
Julia `19e5e2fe` and drmTMB `02095c67`; those are not current-main pins.

A bot comment reports six passing checks in run `41135048434`, but current
workflow lookup returns no run and the exact head has no combined status.
There is no formal review or assignee. The prior owner handoff says to keep the
PR draft unless Shinichi asks to mark it ready. No correctness blocker was
found; retain the draft and require retrievable exact-head checks before
calling it merge-ready. No PR state changed.

### drmTMB #1655 and #1653 narrow test/docs PR review (2026-11-02 01:46 UTC)

PR #1655 is an open draft at exact head
`063279f44f3f343239e90d9ad4c070779fe2ac47`, based on current main
`b535bfbcce2a9039d869c2e34fe9da476394bf62`. It changes only
`tests/testthat/test-cran-lane-filter.R`. The test mocks the CRAN lane
predicate only in interactive sessions, where production intentionally
permits use; non-interactive sessions still exercise the setup guard. No
correctness or scope issue was found. The bot reports six passing checks, but
the run is no longer retrievable and current commit status is empty. Keep the
draft until exact-head checks and the owner's readiness decision are clear.

PR #1653 is an open draft at exact head
`181ae21ed00f5a85682509b6c7397eda15f3339a`, also based on current main. It adds
explicit roxygen usage for `update.drm_pair_association()` and a regression
check that regenerated help is stable with Matrix attached. The patch leaves
generated Rd and NAMESPACE unchanged and adds skip-census entries; no model or
public behavior changes. No correctness issue was found. Its six passing
checks are bot-reported in run `41133418965`, but current lookup returns 404
and the head has no statuses. Neither draft has a formal review or assignee.
Both are suitable for human review after exact-head checks can be verified;
neither is merge-ready now. No PR metadata changed.

### CMR package-separation boundary (2026-11-02 01:47 UTC)

drmTMB PR #1612 remains open, non-draft, and mergeable at head
`0b0f4aa7d73835cc4acfd0ca48234f71c2453f72` over current drmTMB main
`b535bfbcce2a9039d869c2e34fe9da476394bf62`. Its scope adds first-capture CJS
models and the article directly to the drmTMB package. The PR body reports a
main-based Totoro source-package check and says repeated-sample interval
availability and coverage are not established. Current commit workflow lookup
returns no runs; there are no submitted reviews or comments in the current PR
timeline. This proposal is outside the DRM parity cleanup and must remain
separate from core DRM readiness. Per the active goal, do not merge or describe
the interval methods as statistically calibrated. Resolve the package
destination independently before any integration or publication action. No PR
state changed.

### drmTMB #1658 stable prior-kernel PR review (2026-11-02)

PR #1658 is open, non-draft, and mergeable at exact head
`2db6b25ccc909748b56695ccf8375c838c723b59`, based on current main
`b535bfbcce2a9039d869c2e34fe9da476394bf62`. The reviewed Beta/Gamma `mi()`
substitutions reuse the existing stable response-density kernels with matching
parameters. The beta-binomial rewrite uses an equivalent gamma-ratio identity.
Tests cover high-shape references, ordinary-shape agreement, and a gradient
check. No likelihood defect was found.

Two gates remain. First, #1658 requires #1529 to merge, but #1529 is
non-mergeable and based on older main; its Beta/Gamma prior changes overlap the
stable-density replacements and C17 receipt entries in #1658. The owners must
resolve the overlap on a rebased combined tree and recertify C17. Second, the
body reports four passing Linux shards, but current workflow and commit-status
queries return no records. The C17 receipt pins `a6131fc`, three commits before
this PR head. Although the intervening commits do not change the receipt's
listed source files, the combined candidate still needs fresh exact-tree CI and
C17 verification. Do not treat #1658 as merge-ready until those gates close.
No source or PR state changed.
