# Current parity PR review notes

Reviewed 2026-11-02 against DRModels.jl main `973bbb675dc6a10b436acf86175e089e38ec467e` and drmTMB main `129c09c8c8d56fa91bb09146f02dff24e65e6e7b`.

## DRModels.jl #1228

- Head `fe064dce02087cc69c9500f678967f6d258f848a`; base `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716`. The base is behind current main.
- CI run [41245284485](https://github.com/itchyshin/DRModels.jl/actions/runs/41245284485): four Julia shards passed.
- Documenter and rendered page audit [41245284449](https://github.com/itchyshin/DRModels.jl/actions/runs/41245284449): passed; preview was cancelled.
- The NB2 hurdle numerical comparison is pinned to the older Julia source. Its receipt reports agreement in component predictions and log likelihood, but a maximum fitted mean/residual difference of 1.24. Avoid saying fitted values agree without qualifying the target.
- BetaBinomial structured support in current Julia source is phylogenetic `phylo(1 | group)` with constant `sigma`; source refuses other structured markers. Do not imply all structured covariance providers.
- Claims about drmTMB refusals need current R source pins and a cited route-specific test or receipt.
- The PR remains open/draft. This review did not contact its owner or change the PR.

## DRModels.jl #1247

- Head `822f9b04c2d25862007141fb12d8ebde0c3bc084`; base `87a8ed4e57a9f5a4b61ca66673ca02d48515b8ee`, behind current main.
- CI run [41493253608](https://github.com/itchyshin/DRModels.jl/actions/runs/41493253608): all four Julia 1.13.3 shards failed; Documenter run [41493253627](https://github.com/itchyshin/DRModels.jl/actions/runs/41493253627) passed.
- Shard 1, job [125447816343](https://github.com/itchyshin/DRModels.jl/actions/runs/41493253608/job/125447816343): `test_tnb2_nan.jl:95`, timing comparison observed `0.116476224 < 0.10539398`, false.
- Shard 2, job [125447816133](https://github.com/itchyshin/DRModels.jl/actions/runs/41493253608/job/125447816133): `test_temporal_ou.jl:59`, dense-oracle discrepancy `1.709850039e-10` exceeded `1e-10`.
- Shard 3, job [125447816262](https://github.com/itchyshin/DRModels.jl/actions/runs/41493253608/job/125447816262): `test_gaussian_ranef.jl:59-62`, all-singleton Hessian warning, convergence, and covariance expectations failed.
- Shard 4, job [125447816180](https://github.com/itchyshin/DRModels.jl/actions/runs/41493253608/job/125447816180): `test_ls_inv2x2_chol.jl:123`, theta tolerance `1e-8` failed; maximum observed difference was about `1.82e-8`.
- These are exact-head CI failures on the PR merge ref, not evidence against the StableRNG test-data change itself. The J3a assertion was not separately identified in this log review. The owner should rebase onto current main and rerun exact-head CI after the unrelated test fixes land.

## `meta_V` accessor parity

- Current R main's `sigma.drmTMB` documentation says meta-analytic sigma is residual heterogeneity, and the method routes to the `sigma` distributional predictor (`R/methods.R:4317-4350`).
- Current Julia main calculates `sqrt(v + tau^2)` into `fit.scales[:sigma]` (`src/gaussian_meta.jl`); `sigma(fit)` returns that field (`src/gaussian_core.jl:2143-2159`). The Julia guide says sigma is tau (`docs/src/model-guides/meta-analysis.md:36`), which conflicts with the accessor implementation. `test/test_meta.jl:17-21` checks the sigma coefficient, not the accessor.
- This confirms a current-main accessor and documentation mismatch. A dirty candidate reportedly stores tau separately, but overlapping source ownership remains unresolved; no source changes were made.

## Limits

This is a read-only PR and source review. No package tests were run here, no raw workflow log was copied into this folder, and no owner was contacted. The workflow links point to the authoritative GitHub logs. No parity, interval-calibration, release, or merge claim follows from this review.

## Live PR and CI refresh, 2026-11-02 21:45 UTC

Read from GitHub REST API at the stated time. Current default-branch pins were DRModels.jl `973bbb675dc6a10b436acf86175e089e38ec467e` and drmTMB `129c09c8c8d56fa91bb09146f02dff24e65e6e7b`. PR base/head pins below are the API values, not inferred from a local checkout. Branch names identify the submitted lane, but do not confirm who currently owns the live process. No owner was contacted.

| PR | Head branch and SHA | Base SHA | Current checks / state | Next eligible action |
|---|---|---|---|---|
| DRModels.jl #1249 | `claude/julia1-tolerance-fixes` · `e01e0fadb1ba7773a41f08d8a0802a2c0b018970` | `973bbb675dc6a10b436acf86175e089e38ec467e` (current main) | Julia 1 run `41519305510`: four shards still in progress. Docs-preview and docs succeeded in `41519305443`; API merge state `blocked`. | Wait for the four shards; then reconcile the separate Julia 1.10 #1114 failure before treating this as a full matrix pass. |
| DRModels.jl #1228 | `docs/726-740-twin-notes` · `fe064dce02087cc69c9500f678967f6d258f848a` | `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` (stale) | Julia 1 run `41245284485` and docs run `41245284449` passed; preview was cancelled; API merge state `unstable`. | Owner must refresh the stale base and address the documented claim/pin findings; rerun exact-head CI and inspect the rendered preview. |
| DRModels.jl #1237 | `fix/bootstrap-formula-structured-stablerng` · `503b3797acdf44dba38af1096ebce7fc338f6e9b` | `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` (stale) | Run `41401878711`: Julia 1.10 shards 1 and 2 failed; other listed shards and scaling sweep passed; aggregate failed. A prior run `41249874224` passed. | Keep with its lane owner; diagnose the exact Julia 1.10 failures and update the stale base before another exact-head run. |
| DRModels.jl #1247 | `claude/julia110-j3a-fix` · `822f9b04c2d25862007141fb12d8ebde0c3bc084` | `87a8ed4e57a9f5a4b61ca66673ca02d48515b8ee` (stale) | Run `41493253608`: all four Julia 1 shards failed; docs run `41493253627` passed; API merge state `blocked`. | Keep with its lane owner. Rebase and separate the J3a case from the four unrelated failures before re-running. |
| DRModels.jl #1240 | `claude/intelligent-lovelace-s0fgio` · `03b93e680266369dff10e63eb4c846826a102ec4` | `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` (stale) | Run `41248908491`: Julia 1 shard 2 failed and the other three passed; docs and preview passed; aggregate failed. | Keep with its lane owner; attribute shard 2 and refresh the base before claiming parity. |
| drmTMB #1657 | `fix/drm1512-snak` · `21f769d0f80f4651592dac9d80d7a7ee3dc8a049` | `129c09c8c8d56fa91bb09146f02dff24e65e6e7b` (current main) | Run `41518765724`: source-tree and OS-matrix checks passed; four Ubuntu release shards still in progress; API merge state `unstable`. | Wait for the four shards; then review exact-head results and the bridge matrix-order behavior. |
| drmTMB #1659 | `test/drmjl-parity-fixtures-snak` · `e2d617b54b9a94b8b8e2758db9a6c80fa4bcab87` | `71ceee671aac0e4c8bedf265b3b735b1ad633390` (stale) | Current run `41512891602`: source-tree, OS matrix and macOS passed; Ubuntu and Windows release checks still in progress. Earlier run `41491401329` passed Linux/macOS but Windows was cancelled. API merge state `clean`. | Wait for current Windows/Ubuntu jobs; then refresh the stale base and rerun exact-head checks. Its evidence remains point-estimate/SE parity, not interval calibration. |
| drmTMB #1663 | `codex/general-sd-regression` · `83bfcf3c369eace9cf8d1363440f9e11953c758f` | `b535bfbcce2a9039d869c2e34fe9da476394bf62` (stale; current main is `129c09c8c8d56fa91bb09146f02dff24e65e6e7b`) | No check runs were returned for this head; API merge state `dirty`. The shared worktree and candidate remain owner-held. | Do not edit or retarget. The owning lane must reconcile its source/base and publish fresh checks. |

Queued jobs were intentionally not polled repeatedly in this snapshot. Next refresh should follow the status-receipt backoff rule, or happen sooner only after a job completion, head change, relevant notification, or a decision that needs fresh evidence. Totoro and DRAC availability was confirmed by Shinichi, but this PR-status review required no compute; neither host was used. No files in PR branches were changed, and no PR metadata was changed.

## Targeted recheck, 2026-11-02 21:50 UTC

After the five-minute backoff, the same exact heads were refreshed. DRModels.jl #1249 (`e01e0fadb1ba7773a41f08d8a0802a2c0b018970`, run `41519305510`) still had all four Julia 1 shards in progress. drmTMB #1659 (`e2d617b54b9a94b8b8e2758db9a6c80fa4bcab87`, run `41512891602`) still had Ubuntu and Windows in progress; macOS, OS matrix, and source-tree checks remained successful. drmTMB #1657 (`21f769d0f80f4651592dac9d80d7a7ee3dc8a049`, run `41518765724`) had advanced: Ubuntu shard 4 completed successfully while shards 1–3 remained in progress; source-tree and OS-matrix checks remained successful. Heads and bases were unchanged, so the previous base/ownership assessment still applies. No new detailed-log fetch was needed because no failure appeared in these jobs.

## Targeted recheck, 2026-11-02 21:55 UTC

The exact heads and bases remained unchanged. DRModels.jl #1249, run `41519305510`, still had all four Julia 1 shards in progress. drmTMB #1657, run `41518765724`, now had Ubuntu shards 3 and 4 successful and shards 1 and 2 in progress; source-tree and OS-matrix checks were successful. drmTMB #1659, run `41512891602`, now had Ubuntu and macOS successful, with Windows still in progress; source-tree and OS-matrix checks were successful. At 21:53 UTC the review API showed all three PRs authored by `itchyshin`, with no assignees, requested reviewers, or submitted reviews. This records formal review status, not live lane ownership.

## Targeted recheck, 2026-11-02 22:01 UTC

Fetched diagnostics pinned to the same exact PR heads. DRModels.jl #1249 (`e01e0fadb1ba7773a41f08d8a0802a2c0b018970`, base current main `973bbb675dc6a10b436acf86175e089e38ec467e`) still has all four Julia 1 shards running in run `41519305510`; docs and preview remain successful. drmTMB #1657 (`21f769d0f80f4651592dac9d80d7a7ee3dc8a049`, base current main `129c09c8c8d56fa91bb09146f02dff24e65e6e7b`) has shards 1, 3, and 4 successful and shard 2 running in run `41518765724`. A newer run `41522911628` has begun on the same head: source-tree checks are running and the OS matrix is queued. drmTMB #1659 (`e2d617b54b9a94b8b8e2758db9a6c80fa4bcab87`) remains based on stale main `71ceee671aac0e4c8bedf265b3b735b1ad633390`; Ubuntu and macOS checks passed, Windows was cancelled in run `41512891602`, and all Windows jobs in earlier attempts were also cancelled. It needs a successful Windows result after refreshing the base. No submitted reviews were present in the preceding 21:53 metadata read; no lane owner was contacted. Totoro and DRAC availability was confirmed by Shinichi, but these metadata checks required no compute. No PR is ready for merge based on this refresh.

## Targeted recheck, 2026-11-02 22:06 UTC

Diagnostics were pinned to the same exact heads. #1249 remains unchanged with all four Julia shards running in `41519305510`. For #1657, run `41522911628` advanced: the OS matrix passed, and its Windows, macOS, Ubuntu, and source-tree jobs are now running. The earlier run `41518765724` still reports Ubuntu shard 2 in progress, with the other three shards and source-tree/OS checks successful. #1659 remains unchanged: Windows was cancelled and its base is stale. No merge action is supported by this status; no lane owner was contacted and no compute was started.

## #1659 Windows job detail, 2026-11-02 22:08 UTC

Fetched steps for job `125531174791` in run `41512891602`, pinned to head `e2d617b54b9a94b8b8e2758db9a6c80fa4bcab87`. The `Run r-lib/actions/check-r-package@v2` step finished successfully, and the returned step list shows setup, cleanup, and job completion steps as successful. The parent Windows job still has conclusion `cancelled`. Therefore the log supports a successful package-check step on this Windows runner, but the overall required job is not green. Keep #1659's Windows status as cancelled until a terminal successful job-level check appears.

## #1662 merge closeout, 2026-11-02 22:10 UTC

GitHub reports drmTMB #1662 merged at `2026-11-02T11:30:13Z` by merge commit `71ceee671aac0e4c8bedf265b3b735b1ad633390`. The live `main` ref is `129c09c8c8d56fa91bb09146f02dff24e65e6e7b`; GitHub's compare API reports that ref is four commits ahead of the merge commit and zero behind, confirming the merge commit is in current main. The PR body reports exact-head CI run `41350326046` passed its listed checks. Current check diagnostics return no records for that PR head, and a direct job-list lookup for the cited run returns 404, so the CI result remains owner-reported/historical rather than freshly inspectable. Post-merge main check status was not established in this lookup.

## R lane ownership check, 2026-11-02 22:12 UTC

R-repository lane preflight on the primary checkout listed #1659's branch `test/drmjl-parity-fixtures-snak` among 31 live lanes. It reported no live lease; a read-only `git worktree list` filter found no local checkout for that branch. This identifies the branch as active in the lane census but does not identify a human task owner. No source files or PR metadata were changed. Preserve the branch and wait for a clear handoff before taking code action.

## Additional DRModels.jl PRs, 2026-11-02 22:14 UTC

Live main is `973bbb675dc6a10b436acf86175e089e38ec467e`. Exact-head PR metadata and diagnostics show:

| PR | Head / base | Current exact-head evidence | Owner-safe next step |
|---|---|---|---|
| #1228 | `fe064dce02087cc69c9500f678967f6d258f848a` / `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` | Four Julia shards and `ci-ok` passed; docs build passed; docs preview was cancelled. The base is stale. The reviewed BetaBinomial wording still needs to say phylogenetic intercept only with constant `sigma`; the hurdle-Poisson distinction is supported. | Keep draft. Owner should narrow the BetaBinomial claim, rebase, and obtain a fresh preview. |
| #1237 | `503b3797acdf44dba38af1096ebce7fc338f6e9b` / `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` | Latest run `41401878711` failed on Julia 1.10 shards 1 and 2; its four Julia 1 shards passed. A previous run `41249874224` passed. The base is stale. | Owner should rebase and diagnose the Julia 1.10 failures before another full claim. Stable test data alone did not make the latest workflow green. |
| #1240 | `03b93e680266369dff10e63eb4c846826a102ec4` / `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` | Julia 1 shard 2 and aggregate `ci-ok` fail in run `41248908491`; the other three shards, docs, and preview pass. The base is stale. The PR remains a draft. The failure is the missing `drmtmb_code_hash` fixture field; see the exact-head diagnosis below. | Keep draft until its owner adds the required provenance value (or the allowed unknown marker), rebases, and reruns. |
| #1247 | `822f9b04c2d25862007141fb12d8ebde0c3bc084` / `87a8ed4e57a9f5a4b61ca66673ca02d48515b8ee` | All four Julia 1 shards failed in run `41493253608`; docs and preview passed. The base is stale. | Wait for current-main #1249, which targets three unrelated Julia 1 failures reported in this run. Then the owner can rebase and rerun #1247; the separate all-singleton Hessian failure still needs attribution. |

For context, #1249 is based on current main `973bbb675dc6a10b436acf86175e089e38ec467e`. At the last exact-head check at 22:11:46 UTC, all four Julia shards remained active and docs/preview had passed. These findings are from GitHub metadata and diagnostics, not new local test runs. Branch authorship does not establish active task ownership; no PR branches or metadata were changed.

## #1240 exact-head CI diagnosis, 2026-11-02 22:18 UTC

Refreshed PR metadata and diagnostics from GitHub. PR #1240 remains open and draft at head `03b93e680266369dff10e63eb4c846826a102ec4`, based on stale main `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716`; current Julia main is `973bbb675dc6a10b436acf86175e089e38ec467e`. The exact-head Julia 1 shard 2 job [124711274391](https://github.com/itchyshin/DRModels.jl/actions/runs/41248908491/job/124711274391) fails in `test/test_fixture_provenance.jl:34`: the new `fixtures_re/gaussian-locscale-re/expected.meta.toml` lacks `drmtmb_code_hash`. The fixture records drmTMB 0.7.0 from CRAN but says the generating sandbox could not obtain a Git commit hash. Current main's #473 provenance test requires this field for every drmTMB comparison fixture and permits the explicit value `unknown (pre-#473)` when the generating build cannot be recovered ([test contract at current main](https://github.com/itchyshin/DRModels.jl/blob/973bbb675dc6a10b436acf86175e089e38ec467e/test/test_fixture_provenance.jl)). The failed provenance test occurs before the new `test_parity_gaussian_locscale_re.jl` in the shard's deterministic test order, so this CI run provides no numerical-parity result for the new fixture. `ci-ok` failed; the other three Julia 1 shards, docs, and preview passed; docs-deploy was skipped.

This records a fixture-provenance contract failure. CI provides no numerical-parity result for the new fixture because the failing provenance test aborts the shard before that test runs. The PR remains based on an older main, and its local 17/17 result does not cover the full suite. The PR owner can add a permitted provenance value, rebase, and rerun exact-head CI. No PR branch, source file, or PR metadata was changed. Totoro/DRAC compute was unnecessary for this deterministic metadata check.

## Active PR check refresh, 2026-11-02 22:20 UTC

Refreshed GitHub metadata and check diagnostics after the prior five-minute backoff. All results below match their PR head SHAs.

- DRModels.jl #1249 remains at head `e01e0fadb1ba7773a41f08d8a0802a2c0b018970`, based on current main `973bbb675dc6a10b436acf86175e089e38ec467e`. Julia 1 shard 1 passed; shards 2–4 remain in progress in run `41519305510`. Docs and preview passed. Keep it open until all shards finish and any failures are classified.
- drmTMB #1657 remains at head `21f769d0f80f4651592dac9d80d7a7ee3dc8a049`, based on current main `129c09c8c8d56fa91bb09146f02dff24e65e6e7b`. Source-tree and OS-matrix checks passed in run `41522911628`; Windows, macOS, and Ubuntu jobs remain in progress. All four Ubuntu shards passed in earlier run `41518765724` on the same head. Keep open until the current OS matrix completes.
- drmTMB #1659 remains at head `e2d617b54b9a94b8b8e2758db9a6c80fa4bcab87`, based on stale main `71ceee671aac0e4c8bedf265b3b735b1ad633390`. Ubuntu and macOS passed; Windows is cancelled in run `41512891602`. A successful package-check step inside that job does not change the cancelled job conclusion. It needs a current-main base refresh and successful Windows job-level check.

No runner result or base changed enough to support a merge recommendation. The earlier owner caveat remains: PR author and branch name do not establish who currently holds the live lane. No PR branches, metadata, or source files were changed. This status refresh required no compute.

## Active PR check refresh, 2026-11-02 22:24 UTC

Refreshed PR metadata and exact-head diagnostics after the prior five-minute backoff.

- DRModels.jl #1249 remains at head `e01e0fadb1ba7773a41f08d8a0802a2c0b018970`, based on current main `973bbb675dc6a10b436acf86175e089e38ec467e`. Julia 1 shards 1 and 2 passed; shards 3 and 4 remain in progress in run `41519305510`. Docs and preview passed.
- drmTMB #1657 remains at head `21f769d0f80f4651592dac9d80d7a7ee3dc8a049`, based on current main `129c09c8c8d56fa91bb09146f02dff24e65e6e7b`. Source-tree and OS-matrix checks passed in run `41522911628`; Windows, macOS, and Ubuntu remain in progress. All four Ubuntu shards passed in earlier run `41518765724` on the same head.
- drmTMB #1659 remains at head `e2d617b54b9a94b8b8e2758db9a6c80fa4bcab87`, based on stale main `71ceee671aac0e4c8bedf265b3b735b1ad633390`. Ubuntu and macOS passed; Windows remains cancelled in run `41512891602`.

The refresh changes no owner or source attribution. #1249 and #1657 still have live runners, while #1659 needs an owner-led base refresh and successful Windows job. The current evidence does not support merging any of these PRs.

## Active PR check refresh, 2026-11-02 22:40 UTC

Refreshed PR metadata and exact-head check runs from GitHub. The head SHAs and
bases are unchanged from the preceding snapshots.

- DRModels.jl [#1249](https://github.com/itchyshin/DRModels.jl/pull/1249)
  remains at head `e01e0fadb1ba7773a41f08d8a0802a2c0b018970` on current main
  `973bbb675dc6a10b436acf86175e089e38ec467e`. In run
  [41519305510](https://github.com/itchyshin/DRModels.jl/actions/runs/41519305510),
  Julia 1 shards 1, 2, and 4 passed; shard 3 is still running. Docs and preview
  passed in run 41519305443; scaling and stable docs deployment were skipped.
  No formal review or review thread is recorded. Wait for shard 3, then
  classify the full exact-head result.
- drmTMB [#1657](https://github.com/itchyshin/drmTMB/pull/1657) remains at
  head `21f769d0f80f4651592dac9d80d7a7ee3dc8a049` on current main
  `129c09c8c8d56fa91bb09146f02dff24e65e6e7b`. In current run
  [41522911628](https://github.com/itchyshin/drmTMB/actions/runs/41522911628),
  source-tree and OS-matrix checks passed; Windows, macOS, and Ubuntu release
  jobs remain in progress. All four Ubuntu shards passed on the same head in
  run 41518765724. No formal review or review thread is recorded. Wait for the
  current platform jobs before reviewing the exact-head check set.
- drmTMB [#1659](https://github.com/itchyshin/drmTMB/pull/1659) remains at
  head `e2d617b54b9a94b8b8e2758db9a6c80fa4bcab87` on stale base
  `71ceee671aac0e4c8bedf265b3b735b1ad633390`. Its latest run
  [41512891602](https://github.com/itchyshin/drmTMB/actions/runs/41512891602)
  has successful Ubuntu, macOS, OS-matrix, and source-tree checks, but the
  Windows release job is cancelled. The successful inner package-check step
  documented above does not change the parent job's cancelled result. No
  formal review or review thread is recorded. A base refresh and successful
  Windows job remain owner actions.
- drmTMB [#1664](https://github.com/itchyshin/drmTMB/pull/1664) is at head
  `015c0e1fac9cad68e82f2711eeebad4b1b6fcd69` on current main
  `129c09c8c8d56fa91bb09146f02dff24e65e6e7b`. Its four Ubuntu shards,
  OS-matrix, and source-tree checks passed in run
  [41513019278](https://github.com/itchyshin/drmTMB/actions/runs/41513019278).
  GitHub reports it open and mergeable; no formal review or review thread is
  recorded. The read-only code review found no blocker, so it is ready for
  maintainer review, not merged.

DRModels.jl #1240 remains open and draft at head
`03b93e680266369dff10e63eb4c846826a102ec4` on stale base
`64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716`. The provenance failure still
precedes the new parity test, so that exact-head CI supplies no numerical
parity result. No PR was changed, reviewed on GitHub, or merged during this
refresh. No compute was needed for this metadata work.

## Additional parity-related PR census, 2026-11-02 22:45 UTC

The following exact-head states were refreshed from GitHub after the previous
snapshot. Worktree observations came from the local lane census; they identify
checkouts, not human ownership.

- drmTMB [#1663](https://github.com/itchyshin/drmTMB/pull/1663) advanced to
  head `993905ff6aa53d70304b988f73be397422dcf274` on current main
  `129c09c8c8d56fa91bb09146f02dff24e65e6e7b`. In run
  [41532211007](https://github.com/itchyshin/drmTMB/actions/runs/41532211007),
  source-tree and OS-matrix checks passed, Ubuntu shards 1-3 passed, and shard
  4 is still running. GitHub reports the open PR mergeable, with no formal
  review or requested reviewer. Its exact-head local worktree is heavily
  modified and contains substantial untracked evidence. Preserve it. The PR
  claims native TMB support only; it does not claim Julia parity or calibrated
  intervals.
- drmTMB [#1647](https://github.com/itchyshin/drmTMB/pull/1647) remains at
  head `b55083409ffd87ea0750d2ab7731f4b967b83cf0` on current main
  `129c09c8c8d56fa91bb09146f02dff24e65e6e7b`. Current run
  [41522917395](https://github.com/itchyshin/drmTMB/actions/runs/41522917395)
  has Ubuntu, macOS, OS-matrix, and source-tree checks passed; Windows remains
  active. All four Ubuntu shards passed on the same head in run
  [41522127422](https://github.com/itchyshin/drmTMB/actions/runs/41522127422).
  No formal review is recorded.
- drmTMB [#1435](https://github.com/itchyshin/drmTMB/pull/1435) is a small
  `fixef()` dispatch change at head
  `3f4a82f361d777f79a6bf1121af748791017d224`. Its Ubuntu, macOS, Windows,
  OS-matrix, and source-tree checks passed in run
  [41497247921](https://github.com/itchyshin/drmTMB/actions/runs/41497247921),
  but its base `71ceee671aac0e4c8bedf265b3b735b1ad633390` is stale relative to
  current main `129c09c8c8d56fa91bb09146f02dff24e65e6e7b`. A prior PR comment
  says it was ready for a maintainer merge decision on the then-current base;
  that is not fresh-base validation or proof of present ownership. No matching
  current-head worktree, formal review, or review thread was found. Do not
  rebase or merge it without resolving ownership and refreshing checks.
- DRModels.jl [#1215](https://github.com/itchyshin/DRModels.jl/pull/1215) is a
  one-file roadmap wording change at head
  `1ee71530ebcd31bb62cce2fba6601ffced4f65ea` on current main
  `973bbb675dc6a10b436acf86175e089e38ec467e`. It is a draft; the changes job
  and docs job are queued in runs
  [41539026157](https://github.com/itchyshin/DRModels.jl/actions/runs/41539026157)
  and
  [41539026225](https://github.com/itchyshin/DRModels.jl/actions/runs/41539026225).
  The local
  checkout is at an older commit, so wait for checks and reconcile its owner
  before advancing it.
- DRModels.jl [#1214](https://github.com/itchyshin/DRModels.jl/pull/1214) and
  [#1210](https://github.com/itchyshin/DRModels.jl/pull/1210) overlap across
  55 files. #1214 is a 55-file draft at current-main head
  `5ba76b1cc0fdda189732011fecc4b9dc220c52b1`; its changes and docs jobs are
  queued in runs
  [41539113997](https://github.com/itchyshin/DRModels.jl/actions/runs/41539113997)
  and
  [41539114018](https://github.com/itchyshin/DRModels.jl/actions/runs/41539114018).
  #1210 is a 57-file draft at stale-base head
  `88b0a3d1d930528bffee292cf1c1d14d2f34c913`, with older passing CI in runs
  [40355893345](https://github.com/itchyshin/DRModels.jl/actions/runs/40355893345)
  and
  [40355893421](https://github.com/itchyshin/DRModels.jl/actions/runs/40355893421).
  Neither has formal review, and neither current head has a matching local
  checkout. A single owner decision is needed to choose the canonical
  overlapping patch before either proceeds.

These PR checks are metadata evidence, not source review or permission to
change an owner-held branch. No review, message, PR change, merge, or compute
run occurred in this census.

## Full open-PR inventory, 2026-11-02 22:50 UTC

The complete read-only census contains 35 open drmTMB PRs and 24 open
DRModels.jl PRs. None had a formal review or review thread. Exact head/base
SHAs, current check counts, registered worktree evidence, and file-overlap
groups are recorded in the [full open-PR census](../2026-11-02-open-pr-census/README.md).

Since the 22:45 snapshot, drmTMB #1663 completed all six exact-head checks
successfully at head `993905ff6aa53d70304b988f73be397422dcf274`; its exact-head
worktree remains heavily dirty, so it stays owner-held. DRModels.jl #1249
still has Julia 1 shard 3 running its test step. drmTMB #1657 still has its
three platform checks running; #1647 has only Windows active. DRModels.jl
#1215 remains queued, while #1214 has its docs build active and changes check
queued. The full census identifies #1664 as the clearest low-risk review
candidate, not an automatic merge. CMR #1612 remains separate from core DRM;
its deferred interval-calibration work is not promoted by this PR inventory.

## Live PR and owner refresh, 2026-11-02 23:00 UTC

GitHub metadata, exact-head workflow jobs, and local worktree/coordination
records were rechecked. The default-branch pins remain drmTMB
`129c09c8c8d56fa91bb09146f02dff24e65e6e7b` and DRModels.jl
`973bbb675dc6a10b436acf86175e089e38ec467e`. The latest open-PR search
returned 34 drmTMB PRs and 24 DRModels.jl PRs. The earlier full inventory is
a 22:50 snapshot; this section supersedes only the rows below. No PR was
changed, merged, or reviewed on GitHub by this refresh.

### drmTMB

- **#1663 is merged**, not open. GitHub records merge time
  `2026-11-02T22:50:57Z`, exact head
  `993905ff6aa53d70304b988f73be397422dcf274`, merge commit
  `e5cd86667c97d1b34f0ead22017a675377f46930`, and successful exact-head run
  [41532211007](https://github.com/itchyshin/drmTMB/actions/runs/41532211007).
  Its registered worktree
  `/Users/z3437171/Dropbox/Github Local/drmTMB/.worktrees/general-sd-regression`
  remains at that head with modified lane/vignette files and extensive
  untracked evidence. The active-lane ledger assigns cross-engine SD work to
  Shinichi in that lane. The merged PR is not evidence that the additional
  dirty work has landed; preserve that checkout and its files.
- **#1659 remains open** at the same head as the 22:50 census,
  `e2d617b54b9a94b8b8e2758db9a6c80fa4bcab87`, on stale base
  `71ceee671aac0e4c8bedf265b3b735b1ad633390`. Exact-head run
  [41491340175](https://github.com/itchyshin/drmTMB/actions/runs/41491340175)
  completed successfully for source-tree tests, OS matrix, and all four
  Ubuntu release shards. GitHub reports it mergeable. Its registered audit
  worktree is prunable and absent, so no live owner checkout was verified.
  The PR's 14 stored-reference cells are one-dataset point-estimate and SE
  comparisons at the older Julia pin `64ccb2d8`; the six default
  GHQ-versus-Laplace comparisons are skipped. This does not establish
  recovery, interval calibration, or current-pair parity.
- **#1664** remains open at exact head
  `015c0e1fac9cad68e82f2711eeebad4b1b6fcd69` on current main. All six checks
  in run [41513019278](https://github.com/itchyshin/drmTMB/actions/runs/41513019278)
  passed, but GitHub currently reports the PR non-mergeable. It has no
  submitted formal review in the refreshed R candidate set. Keep it as a
  small review candidate, not a merge-ready change, until mergeability and
  review are resolved.
- **#1657, #1647, #1646, and #1623** have successful exact-head checks in
  runs `41518765724`, `41522127422`, `41522870646`, and `41521376664`,
  respectively. GitHub reports each non-mergeable. **#1435** has successful
  run `41497210984`, but remains on stale base
  `71ceee671aac0e4c8bedf265b3b735b1ad633390` and is also reported
  non-mergeable. These checks do not resolve source conflicts or identify a
  current lane owner. The eight R candidates inspected in this refresh had
  no formal review submissions.

### DRModels.jl

- **#1242, #1239, and #1234 are merged** at exact heads `c56ecfb3`,
  `e8a6c20a`, and `0b85b7cf`; all recorded CI and Documenter runs succeeded.
  Their merge commits are ancestors of the locally cached `origin/main`
  `973bbb675dc6a10b436acf86175e089e38ec467e`. The tests retain important
  limits: #1242 has nine default-integrator mismatches marked broken while
  fixed effects and explicit `:Laplace` agree; #1239 has three broken
  marginal log-likelihood comparisons with gaps 0.12, 0.06, and 0.19; #1234
  retains four Student Gaussian-limit mismatches and skips Gamma because its
  reference fixture has not been generated. Merging these tests does not
  make the broken or skipped cells pass.
- **#1240 remains an open draft** at head
  `03b93e680266369dff10e63eb4c846826a102ec4` on stale base
  `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716`. CI run
  [41248908491](https://github.com/itchyshin/DRModels.jl/actions/runs/41248908491)
  fails fixture-provenance validation because `expected.meta.toml` lacks
  `drmtmb_code_hash`, before the new numerical fixture test runs. Documenter
  run `41248908497` passed. This is a provenance failure, not a measured
  model mismatch or parity result. Current review lookup found no formal
  review submission.
- **#1249** remains active at current-main head
  `e01e0fadb1ba7773a41f08d8a0802a2c0b018970`: docs passed, while Julia 1
  shard 3 was still running in
  [41519305510](https://github.com/itchyshin/DRModels.jl/actions/runs/41519305510)
  at 23:00 UTC. #1248 had its test run queued and docs passed; #1247's four
  Julia 1 shards failed while docs passed; #1237's recorded CI and docs
  passed. #1215's jobs were queued, #1214's changes job was queued and docs
  job active, and #1210's older-base CI/docs passed. #1214 and #1210 still
  overlap across 55 files, so keep the canonical patch decision with their
  owner.
- A separate detached worktree at
  `/Users/z3437171/.codex/worktrees/capability-status-reconcile/DRM.jl`
  contains uncommitted changes to `docs/design/capability-status.md` and
  untracked after-task/check-log records. Its ownership was not identified
  in the coordination board or live lease list. This refresh reviewed the
  diff and preserves it; it does not overwrite or consolidate that page.

The read-only R owner refresh and Julia owner refresh were performed in
parallel. The R refresh found 33 live lanes in its repo census and verified
the assigned owner record for #1663's dirty SD lane. The Julia refresh found
no authoritative owner for the detached capability-page edit. PR authorship,
branch names, green CI, and a merged PR were not treated as proof that adjacent
uncommitted work had been completed or released. Totoro and DRAC were not used:
this was metadata, source, and worktree reconciliation, not a compute task.

## Owner-safe follow-up, 2026-11-02 23:10 UTC

The current exact-head checks and registered worktrees were rechecked after the
23:00 snapshot. No candidate met both conditions for safe advancement: a
current exact-head checkout with a demonstrably live owner, and no overlapping
lane. PR authorship and an earlier “ready” comment did not satisfy that owner
check. No source branch or PR was changed.

- **drmTMB #1657** remains at head
  `21f769d0f80f4651592dac9d80d7a7ee3dc8a049`, based on current main. Run
  [41522911628](https://github.com/itchyshin/drmTMB/actions/runs/41522911628)
  has source-tree, OS-matrix, macOS, and Windows jobs successful; its Ubuntu
  release check is still running at 23:09 UTC. The PR is reported
  non-mergeable, has no formal review, and no registered exact-head worktree
  was found. Keep it with its current lane pending a terminal check and owner
  handoff.
- **DRModels.jl #1249** remains at head
  `e01e0fadb1ba7773a41f08d8a0802a2c0b018970` on current main. Run
  [41519305510](https://github.com/itchyshin/DRModels.jl/actions/runs/41519305510)
  has Julia 1 shards 1, 2, and 4 successful; shard 3 is still running at
  23:09 UTC. Documenter passed. No formal review or exact-head checkout was
  found, so keep it waiting for its current lane and final shard.
- **DRModels.jl #1248** remains open at head
  `1d7eeff21145b1ecdf598a97257db5cbb14702f9`, based on stale base
  `87a8ed4e57a9f5a4b61ca66673ca02d48515b8ee`. CI run
  [41511040751](https://github.com/itchyshin/DRModels.jl/actions/runs/41511040751)
  fails on three Julia 1 shards. The inspected logs show distinct failures:
  the temporal OU dense-oracle gap is `1.7098500393e-10` against a `1e-10`
  bound; the coupled NB2 location-scale fixture misses its `1e-8` parameter
  tolerance; and the all-singleton Gaussian test expects a warning,
  non-convergence, and NaN covariance but receives a converged fit with finite
  covariance. These are exact-head failures; the cause and relation to #1248
  remain unattributed. Documenter passed. Do not call this a #1248 regression
  without a current-base or owner-led reproduction.
- Julia #1214 and #1215 remain draft documentation PRs. Their latest recorded
  checks were queued, and neither has an exact-head checkout. #1214 overlaps
  #1210 across 55 files. The smallest file count does not make either branch
  safe to take.
- In R, #1664 has six green exact-head jobs but is non-mergeable, has no formal
  review, and lacks an exact-head checkout; #1647 overlaps its methods file.
  #1646 overlaps the active #1529 Gamma-mixture lane. #1659 is green and
  reported mergeable but has a stale base and no registered owner checkout.
  The other inspected green R candidates also lack a collision-free
  owner-and-checkout path. Leave them untouched pending lane-owner disposition.

No Totoro/DRAC compute was appropriate for this API, CI-log, and worktree
review. The next useful refresh event is completion of #1657 run 41522911628
or #1249 run 41519305510; otherwise wait for an owner handoff rather than
rechecking unchanged queued or running work.

## #1657 terminal CI refresh, 2026-11-02 23:12 UTC

Run [41522911628](https://github.com/itchyshin/drmTMB/actions/runs/41522911628)
is now terminal and all five listed jobs passed at exact head
`21f769d0f80f4651592dac9d80d7a7ee3dc8a049`: source-tree tests, OS matrix, and
Ubuntu, macOS, and Windows release checks. PR #1657 remains open and
non-draft, with no formal reviews, no registered exact-head worktree, and
GitHub still reports `mergeable = false`. The comparison from current main
`129c09c8c8d56fa91bb09146f02dff24e65e6e7b` to the PR head reports six commits
ahead, zero behind, and current main as the merge base. The reason for the
PR API's non-mergeable status is not established; do not infer a text conflict
or merge readiness from these fields alone.

DRModels.jl #1249 remains open at head
`e01e0fadb1ba7773a41f08d8a0802a2c0b018970`; its last observed state at 23:09
UTC was three Julia 1 shards successful and shard 3 still running, with
Documenter passed. No new poll was needed for this update. No branch or PR
metadata was changed, and no Totoro/DRAC compute was used.

## Current-main and R-CMD-check correction, 2026-11-02 23:18 UTC

Live GitHub state supersedes the R main pin in the earlier 23:12 note. drmTMB
main advanced to `e5cd86667c97d1b34f0ead22017a675377f46930` when #1663 merged
at 22:50:56 UTC. The latest R-CMD-check run returned by the main-branch
workflow query is
[40375071745](https://github.com/itchyshin/drmTMB/actions/runs/40375071745),
created October 26 at commit
`02095c67d7e70bbb648f45d2bbdd5aa0b1ea7be5`; its macOS release job failed,
while the source-tree, Ubuntu, Windows, and OS-matrix jobs passed. This is not
a check of current main. Its macOS job log now returns HTTP 410, so this
refresh cannot attribute the failure to a particular test. No later
R-CMD-check run appears in the 50 most recent main workflow runs. The latest
main run is receipt-staleness run
[41540682271](https://github.com/itchyshin/drmTMB/actions/runs/41540682271),
queued on `e5cd866`; it is not an R-CMD-check.

drmTMB #1657 remains open at head
`21f769d0f80f4651592dac9d80d7a7ee3dc8a049`, with its five-job exact-head run
`41522911628` green. However, its API base remains the old main
`129c09c8c8d56fa91bb09146f02dff24e65e6e7b`. The live compare from new main
`e5cd866` to the PR head is `diverged`, six commits ahead and 47 behind, with
`129c09c` as merge base. GitHub reports `mergeable = false` and
`mergeable_state = dirty`; no formal review or requested reviewer is recorded.
Treat the green run as exact-head evidence only, not current-main validation
or merge readiness. Leave rebasing and new CI to the owning lane.

DRModels.jl #1249 remains open at head
`e01e0fadb1ba7773a41f08d8a0802a2c0b018970` on current main
`973bbb675dc6a10b436acf86175e089e38ec467e`. In run
[41519305510](https://github.com/itchyshin/DRModels.jl/actions/runs/41519305510),
shards 1, 2, and 4 passed and shard 3 remained in progress at 23:18 UTC;
GitHub reports the PR mergeable but blocked. No formal review was recorded.

Shinichi confirmed Totoro and DRAC are available again. Neither was used for
this metadata-only refresh. Route future CPU validation to Totoro within the
150-core cap; reserve DRAC for queued, multi-node, or larger-than-150-core
work, with an estimate and required pre-run approval for campaigns over three
hours.
