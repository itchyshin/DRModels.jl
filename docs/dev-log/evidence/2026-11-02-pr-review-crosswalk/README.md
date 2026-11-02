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

Refreshed PR metadata and diagnostics from GitHub. PR #1240 remains open and draft at head `03b93e680266369dff10e63eb4c846826a102ec4`, based on stale main `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716`; current Julia main is `973bbb675dc6a10b436acf86175e089e38ec467e`. The exact-head Julia 1 shard 2 job [124711274391](https://github.com/itchyshin/DRModels.jl/actions/runs/41248908491/job/124711274391) fails in `test/test_fixture_provenance.jl:34`: the new `fixtures_re/gaussian-locscale-re/expected.meta.toml` lacks `drmtmb_code_hash`. The fixture records drmTMB 0.7.0 from CRAN but says the generating sandbox could not obtain a Git commit hash. Current main's #473 provenance test requires this field for every drmTMB comparison fixture and permits the explicit value `unknown (pre-#473)` when the generating build cannot be recovered ([test contract at current main](https://github.com/itchyshin/DRModels.jl/blob/973bbb675dc6a10b436acf86175e089e38ec467e/test/test_fixture_provenance.jl)). The shard's other testsets passed before this single provenance assertion failed. `ci-ok` failed; the other three Julia 1 shards, docs, and preview passed; docs-deploy was skipped.

This records a fixture-provenance contract failure. The job log reports no numerical R/Julia mismatch. The PR remains based on an older main, and its local 17/17 result does not cover the full suite. The PR owner can add a permitted provenance value, rebase, and rerun exact-head CI. No PR branch, source file, or PR metadata was changed. Totoro/DRAC compute was unnecessary for this deterministic metadata check.
