# Open PR census for drmTMB and DRModels.jl

Refreshed 2026-11-02 at 22:50 UTC from GitHub REST metadata, exact-head check
runs, formal reviews, review threads, and registered local worktrees. Default
branches at the refresh were drmTMB `129c09c8c8d56fa91bb09146f02dff24e65e6e7b`
and DRModels.jl `973bbb675dc6a10b436acf86175e089e38ec467e`.

This is a status inventory, not a source review. Every PR in both repositories
had zero formal review submissions, zero review threads, and no requested
reviewer or assignee at the refresh. PR authorship and branch names do not
establish who currently owns a lane. A green check, mergeable flag, or local
checkout does not authorize a merge or branch edit. `D` means draft. Check
counts below are exact-head records, with active and cancelled jobs kept
separate from successful checks. Recheck volatile states before taking action.

## drmTMB

Thirty-five PRs were open.

| PR | Head SHA | Base SHA | Current check state | Owner and next-action evidence |
|---|---|---|---|---|
| [#1664](https://github.com/itchyshin/drmTMB/pull/1664) | `015c0e1fac9cad68e82f2711eeebad4b1b6fcd69` | `129c09c8c8d56fa91bb09146f02dff24e65e6e7b` | Open, mergeable, 6 passed. Run [41513019278](https://github.com/itchyshin/drmTMB/actions/runs/41513019278). | Small `vcov()` message correction. Read-only code review found no blocker. Ready for maintainer review, not merged. |
| [#1663](https://github.com/itchyshin/drmTMB/pull/1663) | `993905ff6aa53d70304b988f73be397422dcf274` | `129c09c8c8d56fa91bb09146f02dff24e65e6e7b` | Open, mergeable, 6 passed. Run [41532211007](https://github.com/itchyshin/drmTMB/actions/runs/41532211007). | Exact-head worktree exists and is heavily dirty with calibration/evidence files. Preserve it. Native TMB only; no Julia parity or calibrated-interval claim. |
| [#1659](https://github.com/itchyshin/drmTMB/pull/1659) | `e2d617b54b9a94b8b8e2758db9a6c80fa4bcab87` | `71ceee671aac0e4c8bedf265b3b735b1ad633390` | Open, mergeable, 14 passed / 2 cancelled. Both Windows jobs cancelled. | Stale base; owner-led refresh and a successful Windows job are needed. Stored estimates/SEs are not coverage evidence. |
| [#1658](https://github.com/itchyshin/drmTMB/pull/1658) | `15a66e3b21d282c14e9c240838b676ad9a046a25` | `71ceee671aac0e4c8bedf265b3b735b1ad633390` | Open, non-mergeable, 11 passed. | Stale base. Its shared source file with #1438 needs scope coordination. |
| [#1657](https://github.com/itchyshin/drmTMB/pull/1657) | `21f769d0f80f4651592dac9d80d7a7ee3dc8a049` | `129c09c8c8d56fa91bb09146f02dff24e65e6e7b` | Open, mergeable, 8 passed / 3 active in run [41522911628](https://github.com/itchyshin/drmTMB/actions/runs/41522911628). Earlier same-head Ubuntu shards all passed. | Current OS checks remain active; no matching exact-head worktree was found. Do not infer an owner from the branch name. |
| [#1656](https://github.com/itchyshin/drmTMB/pull/1656) | `2e9556282c914d37fce3196cce7da62b07e2dabe` | `b535bfbcce2a9039d869c2e34fe9da476394bf62` | Open, non-mergeable, 4 passed / 1 failed. macOS failure in run [41475140388](https://github.com/itchyshin/drmTMB/actions/runs/41475140388). | Stale base; failure and ownership need owner review. |
| #1655 D | `063279f44f3f343239e90d9ad4c070779fe2ac47` | `b535bfbcce2a9039d869c2e34fe9da476394bf62` | Open draft, no checks. | Registered checkout is two commits behind the PR head. Preserve lane work. |
| #1653 D | `181ae21ed00f5a85682509b6c7397eda15f3339a` | `b535bfbcce2a9039d869c2e34fe9da476394bf62` | Open draft, no checks. | No matching checkout found. |
| #1652 D | `deb5246002d2f3c6a97a1c5a111e83f378b5c232` | `b535bfbcce2a9039d869c2e34fe9da476394bf62` | Open draft, no checks. | Registered checkout is two commits behind the PR head. Preserve lane work. |
| #1651 D | `d228429a28633b0657e569630128c66b578f255c` | `b535bfbcce2a9039d869c2e34fe9da476394bf62` | Open draft, no checks. | Registered docs-audit checkout is three commits behind and has modified workflow article content. Preserve it. |
| #1648 D | `1d9b40467c8e442febfe02fbc89d672541f234dc` | `b535bfbcce2a9039d869c2e34fe9da476394bf62` | Open draft, no checks. | No matching checkout found. |
| [#1647](https://github.com/itchyshin/drmTMB/pull/1647) | `b55083409ffd87ea0750d2ab7731f4b967b83cf0` | `129c09c8c8d56fa91bb09146f02dff24e65e6e7b` | Open, mergeable, 10 passed / 1 active. Windows active in run [41522917395](https://github.com/itchyshin/drmTMB/actions/runs/41522917395); same-head Ubuntu shards passed in [41522127422](https://github.com/itchyshin/drmTMB/actions/runs/41522127422). | Await Windows and a source review. |
| [#1646](https://github.com/itchyshin/drmTMB/pull/1646) | `5a37f316bdf99f3fd6cd839fc8117bdcb44e2d9b` | `129c09c8c8d56fa91bb09146f02dff24e65e6e7b` | Open, mergeable, 8 passed / 3 active in run [41522923449](https://github.com/itchyshin/drmTMB/actions/runs/41522923449). | Await current OS jobs and a source review. |
| #1642 D | `0bb97689ed130c85b2dfdb7d586f7119370aa411` | `b535bfbcce2a9039d869c2e34fe9da476394bf62` | Open draft, no checks. | No matching checkout found. |
| #1629 D | `6e4745ef676bb3c81fe57aa54283a18530718f50` | `b535bfbcce2a9039d869c2e34fe9da476394bf62` | Open draft, no checks. | No matching checkout found. |
| #1627 D | `36dc7a70cabcf0dec8428af4bd14fa2e0fc0625b` | `b535bfbcce2a9039d869c2e34fe9da476394bf62` | Open draft, no checks. | No matching checkout found. |
| [#1623](https://github.com/itchyshin/drmTMB/pull/1623) | `270a55069d8d3ae054dd42658b2d07abf132c3be` | `129c09c8c8d56fa91bb09146f02dff24e65e6e7b` | Open, mergeable, 10 passed / 1 active in run [41522905962](https://github.com/itchyshin/drmTMB/actions/runs/41522905962). | Await active Windows job and a source review. |
| [#1612](https://github.com/itchyshin/drmTMB/pull/1612) D | `0b0f4aa7d73835cc4acfd0ca48234f71c2453f72` | `b535bfbcce2a9039d869c2e34fe9da476394bf62` | Open draft, no checks. | CMR work stays separate from the DRM-core parity finish. Its checkout is 1 ahead / 499 behind; preserve it. |
| #1575 D | `df71b12297058e2bfee996f41618094bacbc511d` | `f0771de99d19324746a8cbe262405ce7c032dc3f` | Open draft, no checks. | Stacked on a Cursor boundary-LRT branch. |
| #1565 D | `970da71600ab3042d69bcb608bf16bd72032f74d` | `3493b187c91750a6cda810e536ac873e2ef1eaa9` | Open draft, no checks. | Older JuliaCall setup lane. |
| #1563 D | `7881fabcccf64607b63873820a5287f8f74bb148` | `129c09c8c8d56fa91bb09146f02dff24e65e6e7b` | Open draft, two checks active in run [41540482150](https://github.com/itchyshin/drmTMB/actions/runs/41540482150). | Head changed during census; re-read before any action. Local checkout is at a different SHA, 1 ahead / 310 behind. |
| #1546 D | `f505b62d2cb258288f2a69615c4b4180a95c289a` | `1b52d4a7b3597d4155e681a3788cdbef79d09009` | Open draft, no checks. | Stacked on Lane C. |
| [#1529](https://github.com/itchyshin/drmTMB/pull/1529) | `bde443a14652323f269c0f6e75ee62f879e2b7db` | `e17921e9b65bd418405107ed3b7954dedb0d88ac` | Open, non-mergeable, no checks. | PR body explicitly says merge after #1513. Do not advance it independently. |
| #1524 D | `5b236c2cb18bc6fda461122b06406e6d6fc39db0` | `ee23dd6cd46400debecfd1c24bf5e95034acb404` | Open draft, 5 passed / 1 cancelled. Ubuntu shard 2 cancelled in [40281179286](https://github.com/itchyshin/drmTMB/actions/runs/40281179286). | Overlaps #1443 on core zero-inflated/hurdle files. |
| #1514 D | `11cca91bac1a1165a7f9ea3f1cb3d34fc7e2991e` | `16647cdc2a426e669faa4c750abaddd5a0935d89` | Open draft, non-mergeable, 6 passed. | Base is behind current main; no matching checkout found. |
| #1509 D | `4d8a039c681137b023561e6bfda46dc9a68c4bf9` | `16647cdc2a426e669faa4c750abaddd5a0935d89` | Open draft, non-mergeable, 6 passed. | Overlaps triage changes in `R/check.R`, `R/methods.R`, and `R/drmTMB.R`. |
| #1505 D | `beb34168a147521d2624bfdbb42f77dbe8298083` | `16647cdc2a426e669faa4c750abaddd5a0935d89` | Open draft, non-mergeable, 6 passed. | Overlaps triage changes in `R/check.R`, `R/methods.R`, and `R/drmTMB.R`. |
| #1503 D | `16e10b0866a80ccf2ffc72f7464c473a0e814ac5` | `16647cdc2a426e669faa4c750abaddd5a0935d89` | Open draft, non-mergeable, 6 passed. | Overlaps triage changes in `R/check.R`, `R/methods.R`, and `R/drmTMB.R`. |
| #1502 D | `e870c45f2bee88f00c47aa4a0798283dfdb528d9` | `bdd0122c7f91a2289fad51ecc35947ea238c63e9` | Open draft, non-mergeable, 6 passed. | Stacked base; do not treat green checks as independent integration. |
| [#1501](https://github.com/itchyshin/drmTMB/pull/1501) | `0a71c9cbc3d8aa5abb2041e6d4c08e54b6adc1d4` | `2f2033a7c0f76d08eb80794493739cda3ae66dab` | Open, non-mergeable, 5 passed / 1 cancelled. Ubuntu shard 1 cancelled in [40281819295](https://github.com/itchyshin/drmTMB/actions/runs/40281819295). | Stacked base and overlaps bridge/profile work. |
| #1500 D | `d0f272d9ecc47214c9a9d904feabe1ec526e78fe` | `2f2033a7c0f76d08eb80794493739cda3ae66dab` | Open draft, non-mergeable, 6 passed. | Stacked base and overlaps bridge/profile work. |
| #1443 D | `a690e9571f7bd6adec5c1415fde7261bc0205dc3` | `b535bfbcce2a9039d869c2e34fe9da476394bf62` | Open draft, no checks. | Overlaps #1524 on `R/drmTMB.R`, `src/drmTMB.cpp`, `NAMESPACE`, and `_pkgdown.yml`. |
| #1438 D | `36ebce399d7e38bfb4f523a3fe297d865359c2a7` | `b535bfbcce2a9039d869c2e34fe9da476394bf62` | Open draft, no checks. | Shares `src/drmTMB.cpp` with #1658. Coordinate numeric-kernel and structured-route changes. |
| [#1435](https://github.com/itchyshin/drmTMB/pull/1435) | `3f4a82f361d777f79a6bf1121af748791017d224` | `71ceee671aac0e4c8bedf265b3b735b1ad633390` | Open, mergeable, 11 passed in run [41497247921](https://github.com/itchyshin/drmTMB/actions/runs/41497247921). | Small `fixef()` dispatch change with a prior maintainer handoff comment, but stale base and no current matching checkout. Rebase and revalidate only after ownership is resolved. |
| #1033 | `b638fc9686cfed782390084beb4338f15d1d49e5` | `16647cdc2a426e669faa4c750abaddd5a0935d89` | Open, non-mergeable, 2 passed / 4 failed in run [39978273242](https://github.com/itchyshin/drmTMB/actions/runs/39978273242). | Old stale-base PR; owner status unverified. |

### R-repository interpretation

- All 35 open PRs lacked a formal review and review thread. This includes
  #1664: it is the smallest fully green current-main candidate, but it still
  needs maintainer review. No merge was authorized by this census.
- The clean low-risk next action is to review #1664, not to merge it. #1435
  has a historical maintainer handoff but stale base and no current checkout.
- #1663 has current-main CI green but its exact-head worktree is dirty. Do not
  clean or rebase it. #1657, #1647, #1646, and #1623 had checks active at the
  census.
- R conflicts include #1658/#1438 sharing `src/drmTMB.cpp`; #1524/#1443
  overlapping on four core files; and triage PRs #1503/#1505/#1509 overlapping
  in check and method files. Treat these groups as owner coordination work.
- #1612 is the separated CMR lane. Its full-population scope stays deferred,
  and interval calibration is not claimed as complete.

## DRModels.jl

Twenty-four PRs were open. All had zero formal reviews and review threads.
The PR author was the same maintainer throughout, which is not proof of active
task ownership.

| PR | Head SHA | Base SHA | Current check state | Owner and next-action evidence |
|---|---|---|---|---|
| [#1249](https://github.com/itchyshin/DRModels.jl/pull/1249) | `e01e0fadb1ba7773a41f08d8a0802a2c0b018970` | `973bbb675dc6a10b436acf86175e089e38ec467e` | Open, non-draft. Julia 1 shards 1, 2, 4 passed; shard 3 active in [41519305510](https://github.com/itchyshin/DRModels.jl/actions/runs/41519305510). Docs and preview passed in [41519305443](https://github.com/itchyshin/DRModels.jl/actions/runs/41519305443); scale/deploy skipped. | Wait for shard 3 and classify the full result. |
| [#1215](https://github.com/itchyshin/DRModels.jl/pull/1215) D | `1ee71530ebcd31bb62cce2fba6601ffced4f65ea` | `973bbb675dc6a10b436acf86175e089e38ec467e` | Draft. Changes [41539026157](https://github.com/itchyshin/DRModels.jl/actions/runs/41539026157) and docs [41539026225](https://github.com/itchyshin/DRModels.jl/actions/runs/41539026225) queued; scale skipped. | One-file roadmap wording change, but checkout is stale and no current owner is verified. Wait for checks and a handoff. |
| [#1214](https://github.com/itchyshin/DRModels.jl/pull/1214) D | `5ba76b1cc0fdda189732011fecc4b9dc220c52b1` | `973bbb675dc6a10b436acf86175e089e38ec467e` | Draft. Changes queued in [41539113997](https://github.com/itchyshin/DRModels.jl/actions/runs/41539113997); docs active in [41539114018](https://github.com/itchyshin/DRModels.jl/actions/runs/41539114018); scale skipped. | Overlaps #1210 in 55 files; both lack current-head worktrees. Owner must select a canonical change. |
| [#1074](https://github.com/itchyshin/DRModels.jl/pull/1074) | `8d676c009f6d6021c70f66a441b1f1b050a9f951` | `973bbb675dc6a10b436acf86175e089e38ec467e` | Open. Changes [41539155647](https://github.com/itchyshin/DRModels.jl/actions/runs/41539155647) and docs [41539155583](https://github.com/itchyshin/DRModels.jl/actions/runs/41539155583) queued; scale skipped. | A possibly lane-held local checkout is 134 commits behind the PR. Do not modify it. |
| [#1248](https://github.com/itchyshin/DRModels.jl/pull/1248) | `1d7eeff21145b1ecdf598a97257db5cbb14702f9` | `87a8ed4e57a9f5a4b61ca66673ca02d48515b8ee` | Stale base. Run [41511040751](https://github.com/itchyshin/DRModels.jl/actions/runs/41511040751): shard 1 passed, shards 2 and 4 failed, shard 3 active; docs [41511040762](https://github.com/itchyshin/DRModels.jl/actions/runs/41511040762) passed. | Not ready for cleanup; failure diagnosis and owner action required. |
| [#1247](https://github.com/itchyshin/DRModels.jl/pull/1247) | `822f9b04c2d25862007141fb12d8ebde0c3bc084` | `87a8ed4e57a9f5a4b61ca66673ca02d48515b8ee` | Stale base. Four Julia 1 shards and aggregate failed in [41493253608](https://github.com/itchyshin/DRModels.jl/actions/runs/41493253608); docs [41493253627](https://github.com/itchyshin/DRModels.jl/actions/runs/41493253627) passed. | Keep the failure attribution distinct from the PR's StableRNG source change. |
| [#1243](https://github.com/itchyshin/DRModels.jl/pull/1243) | `0769611adb503bb54a940190f65a39de47dea109` | `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` | Stale base. All Julia shards and aggregate passed in [41250039255](https://github.com/itchyshin/DRModels.jl/actions/runs/41250039255); docs [41250039254](https://github.com/itchyshin/DRModels.jl/actions/runs/41250039254) passed. | Current-main compatibility and ownership remain unverified. |
| [#1240](https://github.com/itchyshin/DRModels.jl/pull/1240) D | `03b93e680266369dff10e63eb4c846826a102ec4` | `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` | Stale base. Shard 2 and aggregate fail in [41248908491](https://github.com/itchyshin/DRModels.jl/actions/runs/41248908491) because `drmtmb_code_hash` is missing before the numerical fixture runs; other shards/docs passed. | No CI numerical result for the new parity fixture. Add allowed provenance and rebase under its owner. |
| [#1237](https://github.com/itchyshin/DRModels.jl/pull/1237) | `503b3797acdf44dba38af1096ebce7fc338f6e9b` | `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` | Stale base. Julia 1.10 shards 1/2 and aggregate failed in [41401878711](https://github.com/itchyshin/DRModels.jl/actions/runs/41401878711); 1.x shards/scaling passed. Earlier run [41249874224](https://github.com/itchyshin/DRModels.jl/actions/runs/41249874224) had a pass. | Reproduce and classify current failures before treating as ready. |
| [#1232](https://github.com/itchyshin/DRModels.jl/pull/1232) | `a592efcc354ec82073b0cdc37bf689d9e2fb579e` | `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` | Stale base. Julia shards/aggregate and docs passed in [41246031667](https://github.com/itchyshin/DRModels.jl/actions/runs/41246031667); preview cancelled. | Stale-base compatibility and owner status unverified. |
| [#1231](https://github.com/itchyshin/DRModels.jl/pull/1231) | `77dd48d465b77b0795414ca447bdf3344e7fdc77` | `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` | Stale base. Julia shards/aggregate and docs passed in [41245901898](https://github.com/itchyshin/DRModels.jl/actions/runs/41245901898); preview cancelled. | Stale-base compatibility and owner status unverified. |
| [#1230](https://github.com/itchyshin/DRModels.jl/pull/1230) | `af1729902ab4beffcd083773ad5b09fb4f384ee3` | `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` | Stale base. Julia shards/aggregate and docs passed in [41245718439](https://github.com/itchyshin/DRModels.jl/actions/runs/41245718439); preview cancelled. | Overlaps #1214/#1210 on control and documentation files. |
| [#1229](https://github.com/itchyshin/DRModels.jl/pull/1229) | `10c5c2b6633d8cdb0de84291288d7e8d59b16a15` | `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` | Stale base. Julia shards/aggregate and docs passed in [41245468054](https://github.com/itchyshin/DRModels.jl/actions/runs/41245468054); preview cancelled. | Stale-base compatibility and owner status unverified. |
| [#1228](https://github.com/itchyshin/DRModels.jl/pull/1228) | `fe064dce02087cc69c9500f678967f6d258f848a` | `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` | Stale base. Julia shards/aggregate and docs passed in [41245284485](https://github.com/itchyshin/DRModels.jl/actions/runs/41245284485); preview cancelled. | Prior review identified claims to narrow: hurdle fitted mean vs component predictions, structured BetaBinomial restricted to `phylo(1|group)` with constant sigma, and stale R refusal pins. |
| [#1227](https://github.com/itchyshin/DRModels.jl/pull/1227) | `dfcb5dcbe5e393ed44e9225d327afaf46f3fd560` | `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` | Stale base. Julia shards/aggregate and docs passed in [41245221802](https://github.com/itchyshin/DRModels.jl/actions/runs/41245221802); preview cancelled. | Prior review flagged overgeneralized residual-convention wording. |
| [#1223](https://github.com/itchyshin/DRModels.jl/pull/1223) | `64940f742395b19e22e2936697b6f5ccb23c2b4c` | `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` | Stale base. Julia shards/aggregate and docs/preview passed in [41240893500](https://github.com/itchyshin/DRModels.jl/actions/runs/41240893500) and [41240893497](https://github.com/itchyshin/DRModels.jl/actions/runs/41240893497). | Current-main compatibility and owner status unverified. |
| [#1222](https://github.com/itchyshin/DRModels.jl/pull/1222) | `d2a750e1dd97123b68b4055edf8e1f5d62de4e30` | `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` | Stale base. Julia shards/aggregate and docs/preview passed in [41241169945](https://github.com/itchyshin/DRModels.jl/actions/runs/41241169945) and [41241170042](https://github.com/itchyshin/DRModels.jl/actions/runs/41241170042). | Prior record notes internal symbols/doc gaps; source review remains needed. |
| [#1218](https://github.com/itchyshin/DRModels.jl/pull/1218) | `2fc48d03972f4b7ca0677c354250916f7bbb947d` | `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` | Stale base. Julia shards/aggregate and docs passed in [41245197960](https://github.com/itchyshin/DRModels.jl/actions/runs/41245197960); preview cancelled. | Overlaps #1106 on four files; coordinate before rebasing. |
| [#1210](https://github.com/itchyshin/DRModels.jl/pull/1210) D | `88b0a3d1d930528bffee292cf1c1d14d2f34c913` | `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` | Stale base. Julia shards/aggregate and docs/preview passed in [40355893345](https://github.com/itchyshin/DRModels.jl/actions/runs/40355893345) and [40355893421](https://github.com/itchyshin/DRModels.jl/actions/runs/40355893421). | Overlaps #1214 in 55 files; no current-head checkout. |
| [#1208](https://github.com/itchyshin/DRModels.jl/pull/1208) | `7606420b84a7e2a6b0da005050b39f7a68124263` | `19f7edb34380e09c6a5602387803bc7838e27f01` | Stacked/stale base. Julia shards/aggregate and docs/preview passed in [40249455856](https://github.com/itchyshin/DRModels.jl/actions/runs/40249455856) and [40249455870](https://github.com/itchyshin/DRModels.jl/actions/runs/40249455870). | Current-main compatibility and owner status unverified. |
| [#1106](https://github.com/itchyshin/DRModels.jl/pull/1106) D | `830a106f744f9e4493d7a94c22022842cda3ed0d` | `e4333387050549d49bf95738543e1e11bc81fa8c` | Stacked/stale. Julia 1/1.10 shards and aggregate passed in [39888530858](https://github.com/itchyshin/DRModels.jl/actions/runs/39888530858); docs passed, preview failed. | Overlaps #1214/#1210 on 18 files and #1218 on four. Coordinate before action. |
| [#1091](https://github.com/itchyshin/DRModels.jl/pull/1091) | `5eeac9f20db32da1991daf00c0ee881ae9e6e311` | `4bb45e4ebb44a7597b86f8505b759904e6574ba5` | Stacked base. Julia 1/1.10 shards, aggregate, and scaling passed in [38962690102](https://github.com/itchyshin/DRModels.jl/actions/runs/38962690102). | Shares three source files with #1106. |
| [#1084](https://github.com/itchyshin/DRModels.jl/pull/1084) | `fc25177eed057f487e29c15b0570a4e1afe17513` | `99000bae8c953715137a83517027f22556b533e6` | Stacked base. Julia 1/1.10 shards, aggregate, and scaling passed in [38971048355](https://github.com/itchyshin/DRModels.jl/actions/runs/38971048355). | Base is another branch; not an independent current-main pass. |
| [#1055](https://github.com/itchyshin/DRModels.jl/pull/1055) | `9c4d925a0967ba1248a4b8a8f9243c496dd9896a` | `2fb75c425252c6a9ecacd5d0f8c58a2af0dc8309` | Stacked/stale. Julia 1/1.10 shards and aggregate passed in [39968275179](https://github.com/itchyshin/DRModels.jl/actions/runs/39968275179); docs/preview passed in [39968275002](https://github.com/itchyshin/DRModels.jl/actions/runs/39968275002). | Current-main compatibility and owner status unverified. |

### DRModels.jl interpretation

- Four PRs target current main: #1249, #1215, #1214, and #1074. The first
  three have active or queued checks. No PR is current-main, fully green,
  conflict-free, formally reviewed, and ready for owner-safe merge.
- The smallest candidate is #1215, a one-file roadmap wording change, but its
  checks were queued and its local checkout was stale. Wait for exact-head
  checks and establish ownership before action.
- The #1074 worktree may be lane-held and is 134 commits behind its PR. Do not
  modify it. Twenty-one PR heads had no matching branch checkout in this
  clone; absence does not prove abandonment.
- #1214 and #1210 overlap across 55 files; #1214/#1106 and #1210/#1106 overlap
  across 18 files; #1218/#1106 share four files; #1230 overlaps #1214/#1210
  on three files; #1106/#1091 share three files. Reconcile these owner groups
  before considering a branch available.

## Current action boundary

At this snapshot, #1664 is the clearest low-risk maintainer review candidate.
That is not a merge instruction. #1435 is small but stale-base and not
owner-confirmed; #1663 is green but has a dirty exact-head checkout; #1215 is
small but still has queued checks and stale local state. All other green checks
on stale or stacked bases require current-base and scope review before they
become actionable.

No PR, branch, review, comment, or checkout was modified for this census. No
package tests, simulation, benchmark, or compute-server job was run. Raw
workflow evidence remains on GitHub at the linked run URLs. CMR PR #1612 stays
separate from the DRM core, with interval calibration still deferred.

## Post-census state, 2026-11-02 23:00 UTC

The open-PR search now returns 34 drmTMB PRs and 24 DRModels.jl PRs. drmTMB
#1663 merged at 22:50:57 UTC at the exact head recorded above. Its dirty
`general-sd-regression` worktree remains active and protected. drmTMB #1659
has successful exact-head source-tree, OS-matrix, and four Ubuntu checks in
run `41491340175`; its base remains stale and its registered audit worktree
is prunable/missing. DRModels.jl #1242, #1239, and #1234 merged with their
recorded CI and Documenter checks successful; their known broken/skipped
parity cells remain documented in the rolling crosswalk. Selected R and
Julia check updates and owner evidence are recorded in the
[rolling parity PR crosswalk](../2026-11-02-pr-review-crosswalk/README.md).
All other table rows remain the 22:50 snapshot and must be refreshed before
action.
