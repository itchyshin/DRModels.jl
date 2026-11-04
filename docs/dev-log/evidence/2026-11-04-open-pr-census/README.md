# Current open PR census: drmTMB and DRModels.jl

Snapshot refreshed on 2026-11-04. PR metadata and exact-head checks were read from GitHub at 15:34 UTC for DRModels.jl and 15:36 UTC for drmTMB. The current default branch SHAs were rechecked at 15:43 UTC: drmTMB `9e2c03f6fdcb9091418cb629f2ca7f30da695e3d`; DRModels.jl `f4fa3aed451636b6ffa442ce2632515513d23b1f`. DRModels.jl #1252 was rechecked at 15:43 UTC. This refresh supersedes the counts in the 2026-11-02 census, while the earlier file remains the historical snapshot.

This snapshot records GitHub PR metadata and local worktree status. PR titles link to the live PR. Full head/base SHAs identify the inspected revisions. GitHub's `mergeable` flag reports its mergeability estimate; current-base compatibility, owner release, formal review, and required checks need separate evidence. Branch names and author identity do not establish lane ownership.

## Current summary

| Repository | Current main | Open PRs | Exact-head and ownership summary |
|---|---|---:|---|
| drmTMB | `9e2c03f6fdcb9091418cb629f2ca7f30da695e3d` | 33 | 28 PRs had no exact-head check runs or statuses. Five had six successful statuses each (`#1514`, `#1509`, `#1505`, `#1503`, `#1502`), but all five are draft, stale-based, and conflict-marked. The inspected cleanup/parity candidates had no exact-head checks. |
| DRModels.jl | `f4fa3aed451636b6ffa442ce2632515513d23b1f` | 22 | #1252 is based on current main; the other 21 use older or stacked bases. None had a formal review submission. #1252's docs passed, Julia shards 1 and 2 passed, shard 3 failed, and shard 4 remained in progress at 15:43 UTC. |

The Nov 2 census listed 35 R and 24 Julia PRs. Since then drmTMB #1670 and #1671 merged; DRModels.jl #1247, #1249, and #1250 merged, and #1252 opened. The resulting live totals are 33 and 22. The 2026-11-02 census remains useful for prior decisions, but its CI and ownership states are historical.

### Merged SD-regression closeout

- DRModels.jl #1250 merged at `f4fa3aed451636b6ffa442ce2632515513d23b1f`.
- drmTMB #1670 merged at `8d30ed99a1d6c621dc23ce7fbb067ce7dc4cdcee`; #1671 then merged at `9e2c03f6fdcb9091418cb629f2ca7f30da695e3d`.
- The #1671 receipt refresh reports 8/8 receipt checks passed, with all 64 R and 114 Julia source hashes matching. This supports receipt consistency. Full package checks, R–Julia parity, and calibration require separate evidence.
- Calibration G5 remains deferred. D-342, two disclosed tolerance adjustments, and known numerical/export limits remain unresolved. These are not closed by the SD merges.

### Current checks on main

- drmTMB main's receipt-staleness run [41834464919](https://github.com/itchyshin/drmTMB/actions/runs/41834464919) passed on `9e2c03f6fdcb9091418cb629f2ca7f30da695e3d`.
- No R-CMD-check run was listed for current drmTMB main. The latest identified R-CMD-check was failed run [40375071745](https://github.com/itchyshin/drmTMB/actions/runs/40375071745), on the older SHA `02095c67d7e70bbb648f45d2bbdd5aa0b1ea7be5`. This does not establish failure of current main; it means current main lacks that check's evidence.
- DRModels.jl #1252's Documenter build and preview passed. The preview is [available here](https://itchyshin.github.io/DRModels.jl/previews/PR1252/). The same run's shard 3 failure is at `test/test_binomial_sd_public.jl:92`; the cached log reports a singular fitted Hessian immediately before the convergence assertion. A previous run on the byte-identical tree passed this testset, so the cause remains unproven. Shard 4 was still running at the last read. Run [41842212229](https://github.com/itchyshin/DRModels.jl/actions/runs/41842212229); Documenter run [41842211469](https://github.com/itchyshin/DRModels.jl/actions/runs/41842211469).

## drmTMB open PRs

At 15:36 UTC, `gh pr list --state open --limit 100` returned 33. All bases differ from current main. The exact-head check summary below is from the connected GitHub checks/statuses; absent means zero checks and zero statuses were returned for that SHA.

| PR | Exact head / base | State, checks, and ownership evidence |
|---|---|---|
| [#1033 missing-response evidence ledger](https://github.com/itchyshin/drmTMB/pull/1033) | `69476355ff0cdf4c81d088ca766ecfbcca31bece` / `129c09c8c8d56fa91bb09146f02dff24e65e6e7b` | Open, non-draft, conflict-marked; no exact-head checks. Broad 69-file change. |
| [#1501 contrasts, Gaussian links, bootstrap status](https://github.com/itchyshin/drmTMB/pull/1501) | `645c649fc89df05ae586e237b0c5e8f0c3012a91` / `129c09c8c8d56fa91bb09146f02dff24e65e6e7b` | Open, non-draft, conflict-marked; no exact-head checks. |
| [#1500 triage fixes](https://github.com/itchyshin/drmTMB/pull/1500) | `7630b6a4653d4ee7ffbd69ee5e6be87d330d8890` / `e5cd86667c97d1b34f0ead22017a675377f46930` | Open draft, conflict-marked; no exact-head checks. |
| [#1524 zero-inflated and hurdle random intercept](https://github.com/itchyshin/drmTMB/pull/1524) | `886ee6b8e77f925163858871e4b242680ba9a33d` / `129c09c8c8d56fa91bb09146f02dff24e65e6e7b` | Open draft, conflict-marked; no exact-head checks; source, tests, and docs. |
| [#1563 Poisson versus NB2 boundary test](https://github.com/itchyshin/drmTMB/pull/1563) | `7881fabcccf64607b63873820a5287f8f74bb148` / `129c09c8c8d56fa91bb09146f02dff24e65e6e7b` | Open draft, conflict-marked; no exact-head checks. A local branch-only checkout is at a different HEAD. |
| [#1435 fixef dispatch](https://github.com/itchyshin/drmTMB/pull/1435) | `3f4a82f361d777f79a6bf1121af748791017d224` / `71ceee671aac0e4c8bedf265b3b735b1ad633390` | Open, non-draft, conflict-marked; no exact-head checks. |
| [#1646 Gamma mi quadrature](https://github.com/itchyshin/drmTMB/pull/1646) | `5a37f316bdf99f3fd6cd839fc8117bdcb44e2d9b` / `129c09c8c8d56fa91bb09146f02dff24e65e6e7b` | Open, non-draft, conflict-marked; no exact-head checks. |
| [#1647 predictive mean](https://github.com/itchyshin/drmTMB/pull/1647) | `b55083409ffd87ea0750d2ab7731f4b967b83cf0` / `129c09c8c8d56fa91bb09146f02dff24e65e6e7b` | Open, non-draft, conflict-marked; no exact-head checks. |
| [#1623 rank-deficient design refusal](https://github.com/itchyshin/drmTMB/pull/1623) | `270a55069d8d3ae054dd42658b2d07abf132c3be` / `129c09c8c8d56fa91bb09146f02dff24e65e6e7b` | Open, non-draft, conflict-marked; no exact-head checks. |
| [#1657 Julia bridge spatial-coordinate parity](https://github.com/itchyshin/drmTMB/pull/1657) | `21f769d0f80f4651592dac9d80d7a7ee3dc8a049` / `129c09c8c8d56fa91bb09146f02dff24e65e6e7b` | Open, non-draft, conflict-marked; no exact-head checks; no matching local worktree. |
| [#1664 vcov guidance](https://github.com/itchyshin/drmTMB/pull/1664) | `015c0e1fac9cad68e82f2711eeebad4b1b6fcd69` / `129c09c8c8d56fa91bb09146f02dff24e65e6e7b` | Open, non-draft, conflict-marked; no exact-head checks. |
| [#1658 stable mi priors](https://github.com/itchyshin/drmTMB/pull/1658) | `15a66e3b21d282c14e9c240838b676ad9a046a25` / `71ceee671aac0e4c8bedf265b3b735b1ad633390` | Open, non-draft, conflict-marked; no exact-head checks. |
| [#1612 first-capture CJS models and article](https://github.com/itchyshin/drmTMB/pull/1612) | `0b0f4aa7d73835cc4acfd0ca48234f71c2453f72` / `b535bfbcce2a9039d869c2e34fe9da476394bf62` | Open draft, conflict-marked; no exact-head checks. This is CMR work, kept separate from the DRM core. Its registered checkout is branch-only and at a different HEAD. |
| [#1627 shared TMB data defaults refactor](https://github.com/itchyshin/drmTMB/pull/1627) | `36dc7a70cabcf0dec8428af4bd14fa2e0fc0625b` / `b535bfbcce2a9039d869c2e34fe9da476394bf62` | Open draft, conflict-marked; no exact-head checks; large source refactor. |
| [#1443 hurdle Poisson](https://github.com/itchyshin/drmTMB/pull/1443) | `a690e9571f7bd6adec5c1415fde7261bc0205dc3` / `b535bfbcce2a9039d869c2e34fe9da476394bf62` | Open draft, conflict-marked; no exact-head checks. |
| [#1438 structured beta-binomial](https://github.com/itchyshin/drmTMB/pull/1438) | `36ebce399d7e38bfb4f523a3fe297d865359c2a7` / `b535bfbcce2a9039d869c2e34fe9da476394bf62` | Open draft, conflict-marked; no exact-head checks; overlaps other beta-binomial work. |
| [#1656 check-log.d adoption](https://github.com/itchyshin/drmTMB/pull/1656) | `2e9556282c914d37fce3196cce7da62b07e2dabe` / `b535bfbcce2a9039d869c2e34fe9da476394bf62` | Open, non-draft, conflict-marked; no exact-head checks. Earlier green comment is historical. |
| [#1629 refuse censored responses](https://github.com/itchyshin/drmTMB/pull/1629) | `6e4745ef676bb3c81fe57aa54283a18530718f50` / `b535bfbcce2a9039d869c2e34fe9da476394bf62` | Open draft, conflict-marked; no exact-head checks. |
| [#1648 fitted-row quantile outputs](https://github.com/itchyshin/drmTMB/pull/1648) | `1d9b40467c8e442febfe02fbc89d672541f234dc` / `b535bfbcce2a9039d869c2e34fe9da476394bf62` | Open draft, conflict-marked; no exact-head checks. |
| [#1651 bind Julia tolerance claim](https://github.com/itchyshin/drmTMB/pull/1651) | `d228429a28633b0657e569630128c66b578f255c` / `b535bfbcce2a9039d869c2e34fe9da476394bf62` | Open draft, conflict-marked; no exact-head checks. A branch-only docs-audit checkout is dirty and at a different HEAD. |
| [#1642 label REML mean-block SE](https://github.com/itchyshin/drmTMB/pull/1642) | `0bb97689ed130c85b2dfdb7d586f7119370aa411` / `b535bfbcce2a9039d869c2e34fe9da476394bf62` | Open draft, conflict-marked; no exact-head checks. |
| [#1653 association help with Matrix](https://github.com/itchyshin/drmTMB/pull/1653) | `181ae21ed00f5a85682509b6c7397eda15f3339a` / `b535bfbcce2a9039d869c2e34fe9da476394bf62` | Open draft, clean/mergeable, but no exact-head checks; lane census still includes it and ownership is unconfirmed. |
| [#1652 optimizer-path test tolerance](https://github.com/itchyshin/drmTMB/pull/1652) | `deb5246002d2f3c6a97a1c5a111e83f378b5c232` / `b535bfbcce2a9039d869c2e34fe9da476394bf62` | Open draft, clean/mergeable, but no exact-head checks. Its registered docs-audit checkout is at a different HEAD. |
| [#1655 CRAN lane guard test](https://github.com/itchyshin/drmTMB/pull/1655) | `063279f44f3f343239e90d9ad4c070779fe2ac47` / `b535bfbcce2a9039d869c2e34fe9da476394bf62` | Open draft, clean/mergeable, but no exact-head checks. Its registered docs-audit checkout is at a different HEAD. |
| [#1502 Julia weights and convergence](https://github.com/itchyshin/drmTMB/pull/1502) | `e870c45f2bee88f00c47aa4a0798283dfdb528d9` / `bdd0122c7f91a2289fad51ecc35947ea238c63e9` | Open draft, conflict-marked; six successful statuses, but stale-base and no current-base checks. |
| [#1575 refresh boundary-LRT skip census](https://github.com/itchyshin/drmTMB/pull/1575) | `df71b12297058e2bfee996f41618094bacbc511d` / `f0771de99d19324746a8cbe262405ce7c032dc3f` | Open draft, conflict-marked; no exact-head checks; stacked on a non-main branch. |
| [#1565 JuliaCall setup SIGSEGV](https://github.com/itchyshin/drmTMB/pull/1565) | `970da71600ab3042d69bcb608bf16bd72032f74d` / `3493b187c91750a6cda810e536ac873e2ef1eaa9` | Open draft, conflict-marked; no exact-head checks. |
| [#1503 factor responses and gradients](https://github.com/itchyshin/drmTMB/pull/1503) | `16e10b0866a80ccf2ffc72f7464c473a0e814ac5` / `16647cdc2a426e669faa4c750abaddd5a0935d89` | Open draft, conflict-marked; six successful statuses, but stale-base and no current-base checks. |
| [#1505 unusable convergence flags](https://github.com/itchyshin/drmTMB/pull/1505) | `beb34168a147521d2624bfdbb42f77dbe8298083` / `16647cdc2a426e669faa4c750abaddd5a0935d89` | Open draft, conflict-marked; six successful statuses, but stale-base and no current-base checks. |
| [#1509 Lane C inference and bootstrap](https://github.com/itchyshin/drmTMB/pull/1509) | `4d8a039c681137b023561e6bfda46dc9a68c4bf9` / `16647cdc2a426e669faa4c750abaddd5a0935d89` | Open draft, conflict-marked; six successful statuses, but stale-base and no current-base checks. |
| [#1514 stable beta-binomial and Tweedie kernels](https://github.com/itchyshin/drmTMB/pull/1514) | `11cca91bac1a1165a7f9ea3f1cb3d34fc7e2991e` / `16647cdc2a426e669faa4c750abaddd5a0935d89` | Open draft, conflict-marked; six successful statuses, but stale-base and no current-base checks. |
| [#1546 check_drm print formatting](https://github.com/itchyshin/drmTMB/pull/1546) | `f505b62d2cb258288f2a69615c4b4180a95c289a` / `1b52d4a7b3597d4155e681a3788cdbef79d09009` | Open draft, clean/mergeable, but no exact-head checks; stacked base. |
| [#1529 weighted mi priors and quadrature](https://github.com/itchyshin/drmTMB/pull/1529) | `bde443a14652323f269c0f6e75ee62f879e2b7db` / `e17921e9b65bd418405107ed3b7954dedb0d88ac` | Open, non-draft, conflict-marked; no exact-head checks; stacked base. |

R lane context: the read-only census found 83 registered worktrees and 32 live lanes. The preflight showed overlapping work and no lease before this census-refresh lease. Matching R worktrees for PR candidates are owned or occupied by current lanes, or are at a different head. A clean or absent worktree is not proof of owner release. No R candidate in this inventory is both exact-head checked, current-base, and owner-free.

## DRModels.jl open PRs

At 15:34 UTC, GitHub search returned 22 open PRs. The PR metadata was re-read at 15:43 UTC for exact full head/base SHAs. Only #1252 uses current main as its base. None had a formal review submission at the 15:34 UTC review read.

| PR | Exact head / base | State and latest check evidence |
|---|---|---|
| [#1055 LRT/bootstrap boundary status](https://github.com/itchyshin/DRModels.jl/pull/1055) | `9c4d925a0967ba1248a4b8a8f9243c496dd9896a` / `2fb75c425252c6a9ecacd5d0f8c58a2af0dc8309` | Draft, stacked and conflicting; eight Julia shards, ci-ok, docs and preview passed on the recorded head. |
| [#1074 boundary diagnostic catch](https://github.com/itchyshin/DRModels.jl/pull/1074) | `8d676c009f6d6021c70f66a441b1f1b050a9f951` / `973bbb675dc6a10b436acf86175e089e38ec467e` | Draft; Julia shard 1 passed, shards 2–4 and ci-ok failed; docs and preview passed. |
| [#1084 NB2 Poisson limit](https://github.com/itchyshin/DRModels.jl/pull/1084) | `fc25177eed057f487e29c15b0570a4e1afe17513` / `99000bae8c953715137a83517027f22556b533e6` | Draft and stacked; Julia 1 and 1.10 shards, ci-ok, scaling and changes passed on the recorded head. Base is not current main. |
| [#1091 REML score and mean SE](https://github.com/itchyshin/DRModels.jl/pull/1091) | `5eeac9f20db32da1991daf00c0ee881ae9e6e311` / `4bb45e4ebb44a7597b86f8505b759904e6574ba5` | Draft, 64 files, base not an ancestor of current main. Eight Julia shards and other recorded checks passed; D-342 remains unresolved, so the PR cannot settle that contract by itself. |
| [#1106 singular/boundary vcov guard](https://github.com/itchyshin/DRModels.jl/pull/1106) | `830a106f744f9e4493d7a94c22022842cda3ed0d` / `e4333387050549d49bf95738543e1e11bc81fa8c` | Draft, 41 files; Julia, 1.10, ci-ok and docs passed, preview failed. Broad change and stale base. |
| [#1208 Student Gaussian-limit start](https://github.com/itchyshin/DRModels.jl/pull/1208) | `7606420b84a7e2a6b0da005050b39f7a68124263` / `19f7edb34380e09c6a5602387803bc7838e27f01` | Draft; four Julia shards, ci-ok, docs and preview passed on the recorded head. |
| [#1214 current docs audit repair](https://github.com/itchyshin/DRModels.jl/pull/1214) | `5ba76b1cc0fdda189732011fecc4b9dc220c52b1` / `973bbb675dc6a10b436acf86175e089e38ec467e` | Draft, 55 files; Julia shards 1, 3 and 4 and ci-ok failed; shard 2, docs and preview passed. |
| [#1215 roadmap snapshot](https://github.com/itchyshin/DRModels.jl/pull/1215) | `1ee71530ebcd31bb62cce2fba6601ffced4f65ea` / `973bbb675dc6a10b436acf86175e089e38ec467e` | Draft; Julia shards, ci-ok and docs passed; preview cancelled. |
| [#1218 unit-free convergence flag](https://github.com/itchyshin/DRModels.jl/pull/1218) | `2fc48d03972f4b7ca0677c354250916f7bbb947d` / `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` | Draft, stale base; four Julia shards, ci-ok and docs passed; preview cancelled. It overlaps #1106. |
| [#1222 engine-internal docs](https://github.com/itchyshin/DRModels.jl/pull/1222) | `d2a750e1dd97123b68b4055edf8e1f5d62de4e30` / `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` | Draft, stale base; checks passed. Its body treats the docs policy as unresolved, but live #985 records the owner decision: document remaining exports, add StatsModels to docs environment, and set `checkdocs = :exports`. The diff does not implement the docs-environment or `make.jl` changes. |
| [#1223 non-finite Hessian error](https://github.com/itchyshin/DRModels.jl/pull/1223) | `64940f742395b19e22e2936697b6f5ccb23c2b4c` / `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` | Draft, stale base; Julia shards and docs passed, preview cancelled. PR body requires an edited/squashed commit message to avoid closing #1220; a separate #480 bootstrap failure is reported. |
| [#1227 contributor notes and residual convention](https://github.com/itchyshin/DRModels.jl/pull/1227) | `dfcb5dcbe5e393ed44e9225d327afaf46f3fd560` / `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` | Draft, stale base; Julia and docs passed; preview cancelled. |
| [#1228 hurdle and structured BetaBinomial docs](https://github.com/itchyshin/DRModels.jl/pull/1228) | `fe064dce02087cc69c9500f678967f6d258f848a` / `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` | Draft, stale base, mergeable=false; Julia and docs passed; preview cancelled. |
| [#1229 quantile-residual limit](https://github.com/itchyshin/DRModels.jl/pull/1229) | `10c5c2b6633d8cdb0de84291288d7e8d59b16a15` / `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` | Draft, stale base; Julia and docs passed; preview cancelled. |
| [#1230 multi-start refusal docs](https://github.com/itchyshin/DRModels.jl/pull/1230) | `af1729902ab4beffcd083773ad5b09fb4f384ee3` / `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` | Draft, stale base; Julia and docs passed; preview cancelled. |
| [#1231 Beta location-scale test](https://github.com/itchyshin/DRModels.jl/pull/1231) | `77dd48d465b77b0795414ca447bdf3344e7fdc77` / `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` | Draft, stale base; Julia and docs passed; preview cancelled. Test-only scope. |
| [#1232 separation and Student limit](https://github.com/itchyshin/DRModels.jl/pull/1232) | `a592efcc354ec82073b0cdc37bf689d9e2fb579e` / `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` | Draft, stale base, mergeable=false; Julia and docs passed; preview cancelled. |
| [#1237 StableRNG bootstrap test](https://github.com/itchyshin/DRModels.jl/pull/1237) | `503b3797acdf44dba38af1096ebce7fc338f6e9b` / `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` | Non-draft, stale base; Julia 1.10 shards 1 and 2 and ci-ok failed; Julia 1.13 shards and docs/preview passed. Run [41401878711](https://github.com/itchyshin/DRModels.jl/actions/runs/41401878711). |
| [#1240 Gaussian location-scale parity test](https://github.com/itchyshin/DRModels.jl/pull/1240) | `03b93e680266369dff10e63eb4c846826a102ec4` / `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` | Draft, stale base; shard 2 and ci-ok failed; other shards and docs passed. The current receipt says the failed shard reached a missing provenance field before the fixture's numerical comparison. |
| [#1243 mixed-family bootstrap status](https://github.com/itchyshin/DRModels.jl/pull/1243) | `0769611adb503bb54a940190f65a39de47dea109` / `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` | Draft, stale base; recorded Julia and docs checks passed. Its own full-suite ledger still lists seven items not passed, including one unexplained load-sensitive failure. |
| [#1248 VA follow-ups](https://github.com/itchyshin/DRModels.jl/pull/1248) | `1d7eeff21145b1ecdf598a97257db5cbb14702f9` / `87a8ed4e57a9f5a4b61ca66673ca02d48515b8ee` | Non-draft, stale base; Julia shards 2–4 and ci-ok failed; shard 1 and docs/preview passed. |
| [#1252 parity crosswalk](https://github.com/itchyshin/DRModels.jl/pull/1252) | `466a801484d6a3b5ba49402b1b401d430aa91387` / `f4fa3aed451636b6ffa442ce2632515513d23b1f` | Non-draft, current base; docs and preview passed; Julia shard 3 failed on `test_binomial_sd_public.jl:92`, shard 4 in progress at 15:43 UTC. No formal review submission was listed. |

Julia lane context: the 15:34 preflight showed 23 live lanes and no pre-existing lease. The exact #1252 branch has a clean registered worktree. Among the 22 PRs, no candidate was both owner-free and ready for an owner-safe merge. The absence of a matching worktree does not establish that its owner has released the lane.

## Worktree and owner boundaries

- The current checkout used for this report is an isolated worktree at exact Julia main `f4fa3aed451636b6ffa442ce2632515513d23b1f`. The original crosswalk worktree remains untouched: branch `codex/current-main-parity-crosswalk-20261102`, HEAD `3a348a4c5321a6cdb92c075a0af17bfe10c68918`, 43 commits ahead and 55 behind current main, with a modified crosswalk README, a modified pycache file, and an untracked after-task report.
- The PR #1252 worktree remains separately registered, clean, at `466a801484d6a3b5ba49402b1b401d430aa91387`.
- This report claims only its unique dated evidence directory and check-log/after-task files. It does not edit the existing crosswalk README, any source, any PR branch, or another lane's worktree.
- A scoped lane lease for those report paths was granted under `codex:open-pr-census-refresh-20261104`; release it when the report slice is complete.

### Registered worktree inventory, 2026-11-04

Separate read-only scans covered every registered worktree in both repositories. drmTMB had 83 accessible registrations: 48 dirty, 35 clean, one with an unresolved merge conflict, 44 behind-only, 38 diverged from live main, and one at main. No worktree matched an open PR's exact head. Five worktrees matched PR branch names while their local SHAs differed from the PR heads. The primary drmTMB checkout itself was 1 commit ahead and 698 behind live main, with 77 tracked and 101 untracked status entries.

DRModels.jl had 90 registered worktrees: 78 accessible and scanned, and 12 whose gitdir paths were missing. Their contents and dirty state remain unknown, so they were left untouched. Across local branch refs, 729 distinct commits were not reachable from live main; this includes historical and owner-held branches and is not a count of uncommitted tasks. The primary Julia checkout was 569 commits behind live main and had tracked and untracked edits.

The scans found no explicit current owner release for dirty or diverged lanes. Six Julia lease records and the drmTMB lease records were expired; preflight showed 23 live Julia lanes and 32 live R lanes. The top-level coordination records were stale relative to the current PR set. Branch names and absent leases do not establish permission to adopt or archive work.

Several high-risk groups remain separate: parallel detached drmTMB SD edits overlap in `R/drmTMB.R`; the CMR worktrees retain large uncommitted source and recovery evidence; Julia covariance and parity worktrees contain hundreds of untracked evidence files; one historical drmTMB verification worktree has 7,571 tracked deletions; and Julia has 12 missing registrations with unknown content. No cleanup or consolidation was attempted. The CMR inventory stays separate from the DRM-core parity work.

The original dirty crosswalk README and after-task report were compared with #1252's exact head. Every local line is present in the PR version; the PR version has 11 additional README lines and 14 additional after-task lines. The old worktree, including its unrelated pycache edit, remains intact.

Exact checks during this follow-up: #1252 remains at head `466a801484d6a3b5ba49402b1b401d430aa91387`; shard 3 failed and shard 4 was still running at 16:11 UTC. Log download for shard 4 reported that logs will be available after completion. #1253 was created after the original 22-PR Julia snapshot, so the live Julia count rose to 23; its checks were queued at 16:11 UTC on head `5169619d66ea736b880a8b3804198e5f2d0ffce1`. These are dated observations and must be refreshed before any PR action.

## What this snapshot does not establish

This is not a source-level review of all 55 PRs, a full package test, a fresh R-CMD-check on current main, R–Julia numerical parity, parameter recovery, interval calibration, approval to edit another lane, or approval to merge, release, publish, or deploy. The CMR work stays separate; its interval-calibration G5 remains deferred. The broader R–Julia parity goal remains open. Preserve the known D-342 decision, tolerance adjustments, and numerical/export limitations in their existing records.

## Sources

- GitHub open-PR search and per-PR metadata/check records, captured at the timestamps above: [drmTMB](https://github.com/itchyshin/drmTMB/pulls) and [DRModels.jl](https://github.com/itchyshin/DRModels.jl/pulls).
- Main-branch pointers: [drmTMB main](https://github.com/itchyshin/drmTMB/commits/main/) and [DRModels.jl main](https://github.com/itchyshin/DRModels.jl/commits/main/).
- Prior complete census: `docs/dev-log/evidence/2026-11-02-open-pr-census/README.md` on the previous crosswalk work branch; it is retained as the historical baseline.
- CMR/source boundaries: R PR #1612 and the separately maintained CMR contract and programme ledger.
- Main receipt check: [drmTMB run 41834464919](https://github.com/itchyshin/drmTMB/actions/runs/41834464919). Current-main R-CMD-check absence was verified from the workflow-run list; the older failed run is [40375071745](https://github.com/itchyshin/drmTMB/actions/runs/40375071745).
