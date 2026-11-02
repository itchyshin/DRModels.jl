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
