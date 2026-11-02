## 1. Goal

Refresh the exact-current-main DRM parity crosswalk with live PR and CI evidence. Run one bounded R-to-Julia bridge smoke on the matched cached-main source pair and record its narrow evidence boundary.

## 2. Implemented

- Added a dated current-main refresh to `docs/design/capability-status.md` for drmTMB `129c09c8c8d56fa91bb09146f02dff24e65e6e7b` and DRModels.jl `973bbb675dc6a10b436acf86175e089e38ec467e`.
- Recorded current metadata and exact-head CI for PRs #1663, #1657, #1659, #1237, and #1240, plus merged PR context for #1666 and #1245. Distinguishes missing checks, queued jobs, active jobs, and failures.
- Ran the filtered `julia-structured` test set from a curated R archive and Julia source archive on Totoro. The R tests completed without failures; one live Poisson `relmat()` Julia bridge test ran. The log records seven matrix-order warnings for unlabelled provider matrices.
- Bound the run to the exact source pins, archived source hashes, runtime versions, one thread, environment workaround, and log SHA-256. The evidence says startup and one route smoke; it does not claim numerical parity or interval calibration.
- Ran `test/test_gaussian_ranef.jl` and `test/test_tnb2_nan.jl` against an isolated exact-main archive on Totoro with Julia 1.10.10 and one Julia/BLAS thread; both passed 12/12. Then ran `test/test_temporal_ou.jl`, `test/test_ls_inv2x2_chol.jl`, and `test/test_tnb2_nan.jl` with Julia 1.13.1; all 185 assertions passed. These test separate #1114 and #1249 cases, not the J3a failure reported by #1247. The logs, source/test hashes, and both input and resolved test manifests are preserved under `docs/dev-log/evidence/2026-11-02-totoro-main-ci-attribution/`.
- On exact drmTMB main `129c09c8c8d56fa91bb09146f02dff24e65e6e7b`, the focused Bernoulli association test passed all 36 expectations on Totoro, R 4.6.1 Linux x86_64. Its test file is byte-identical to the older baseline in PR #1659. This does not reproduce the historical Mac arm64 failure or identify its cause; it does not establish a full current-main `R CMD check`. Raw output and source/test archive hashes are in `docs/dev-log/evidence/2026-11-02-totoro-drmTMB-main-associate-pairs/`.
- Reviewed DRModels.jl #1228's docs claims and exact-head CI. CI run 41245284485 and Documenter run 41245284449 passed, but its head is based on stale main `64ccb2d`. The NB2 hurdle claim needs to distinguish matching component predictions/log likelihood from a reported fitted-mean/residual gap; BetaBinomial structured wording must specify phylogenetic intercept support and constant `sigma`; drmTMB refusal claims need current R source pins. Findings and links are retained in `docs/dev-log/evidence/2026-11-02-pr-review-crosswalk/README.md`.
- Inspected #1247's four failed 1.13.3 shard logs in run 41493253608. Failures were the TNB2 timing bound, temporal dense-oracle tolerance, Gaussian all-singleton Hessian expectations, and location-scale theta tolerance. Its Documenter run passed. J3a was not isolated in this log review. The PR remains based behind current main; no owner or PR was changed.
- Rechecked `meta_V` accessor behavior from both exact current-main source pins. R documents `sigma()` as residual heterogeneity. Julia stores and returns `sqrt(v + tau^2)` although its guide says `tau`; the registered test checks coefficient recovery rather than `sigma(fit)`. This remains an open code/documentation parity gap, and source ownership is unresolved.

## 3a. Decisions and Rejected Alternatives

- Used Totoro for the single filtered bridge smoke. DRAC was available but unnecessary for this short CPU check; no campaign or large simulation was launched.
- Kept the current-main page draft separate from the older `codex/drm-parity-crosswalk-20261102` page. Their histories and contents differ materially; copying one over the other would discard claims or evidence.
- Did not edit the owner-held #1663 worktree, the report/evidence files held by the status-refresh lane, or any PR metadata.
- Assumption: a one-test bridge smoke is useful as route evidence while explicitly withholding parity and calibration claims. If this is too narrow for the page, remove the smoke paragraph; no model claim depends on it.

## 4. Files Touched

- `docs/design/capability-status.md`
- `docs/dev-log/after-task/2026-11-02-current-main-crosswalk-smoke.md`
- `docs/dev-log/check-log.d/2026-11-02-current-main-crosswalk-smoke.md`
- `docs/dev-log/evidence/2026-11-02-totoro-main-ci-attribution/gaussian-ranef.log.gz`
- `docs/dev-log/evidence/2026-11-02-totoro-main-ci-attribution/tnb2-nan.log.gz`
- `docs/dev-log/evidence/2026-11-02-totoro-main-ci-attribution/pre-resolution-manifest.sha256`
- `docs/dev-log/evidence/2026-11-02-totoro-main-ci-attribution/README.md`
- `docs/dev-log/evidence/2026-11-02-totoro-main-ci-attribution/julia1131-focused-run.log.gz`
- `docs/dev-log/evidence/2026-11-02-totoro-main-ci-attribution/julia1131-alias-not-installed.log.gz`
- `docs/dev-log/evidence/2026-11-02-totoro-main-ci-attribution/julia1131-resolved-Manifest.toml`
- `docs/dev-log/evidence/2026-11-02-totoro-main-ci-attribution/pre-resolution-TestManifest-julia110.toml`
- `docs/dev-log/evidence/2026-11-02-totoro-main-ci-attribution/resolved-TestProject.toml`
- `docs/dev-log/evidence/2026-11-02-totoro-drmTMB-main-associate-pairs/README.md`
- `docs/dev-log/evidence/2026-11-02-totoro-drmTMB-main-associate-pairs/focused-test.log.gz`
- `docs/dev-log/evidence/2026-11-02-pr-review-crosswalk/README.md`
- `tools/__pycache__/runtests_files.cpython-313.pyc`: incidental modification from the first citation-check invocation; not staged or restored.

The current-main worktree is on local branch `codex/current-main-parity-crosswalk-20261102`, created from exact cached main `973bbb675dc6a10b436acf86175e089e38ec467e`.

## 5. Checks Run

- `PYTHONDONTWRITEBYTECODE=1 python3 tools/check_capability_citations.py`: passed; 2 citations verified, 22 skipped as not precisely resolvable.
- `git diff --check`: passed after storing raw logs as gzip files. The evidence READMEs record hashes of the uncompressed bytes.
- `python3 ~/shinichi-brain/tools/slop_check.py "$PWD/docs/design/capability-status.md"`: `FINDINGS: 0`.
- `Rscript tools/check-after-task.R <report>` from the DRM.jl repository: structure passed. `python3 ~/shinichi-brain/tools/evidence_gate.py`: passed, 8 registered claims.
- `python3 ~/shinichi-brain/tools/closeout.py check <report>`: failed at the brain-wide acceptance-ledger check because 9 unrelated `.unlazy` gates are unmet in book-format, brain-campaign, and brain-visuals. Their files are outside this task and were not changed.
- On Totoro, `NOT_CRAN=true DRMTMB_JULIA_TESTS=true DRMODELS_JL_PATH="$JROOT" DRM_JL_RELMAT_PATH="$JROOT" Rscript -e '...; devtools::test(".", filter = "julia-structured", reporter = "summary")'` completed successfully with no failures and one live bridge test. Log: `~/sim-archive/drm-parity/crosswalk-smoke-2026-11-02-2036/logs/run-v4.log`, SHA-256 `c663d0169d581c2aa761dadfe93704a5c6e52ab5fd1f8dbe33cb04ae79af4f98`.
- Source archive SHA-256 values: R `49ff87ce28da6cc96b4f302b4df19f952aa9cde2daf234fb570258f33cc26f00`; Julia `48565e446632a6086182f08770827ea5e980839d42ccdb92f64385b16d9aa86d`.
- Runtime: Totoro, R 4.6.1, Julia 1.12.6, JuliaCall 0.17.6, one thread. Julia's bundled `libunwind` was preloaded; `JULIA_PROJECT` was unset for the R process.
- Additional current-main tests: Totoro, Julia 1.10.10 and 1.13.1, one Julia and BLAS thread. The 1.10.10 run passed 12/12 in each of `test/test_gaussian_ranef.jl` and `test/test_tnb2_nan.jl`. The 1.13.1 run passed all 185 assertions in `test/test_temporal_ou.jl`, `test/test_ls_inv2x2_chol.jl`, and `test/test_tnb2_nan.jl`; it did not rerun Gaussian random effects. It warned that the manifest was resolved with Julia 1.10.10, and JET precompilation emitted an error; the focused test sets completed and the Julia process exited zero. The exact-main archive SHA-256 is `4b4808c6acef38087e06c8779a563755bc5d2ae8855e3c401165d39bafd1ffb4`. Manifest and log hashes are in the crosswalk and evidence directory.
- R association test: exact drmTMB main archive, R 4.6.1, Totoro Linux x86_64, one BLAS/OpenMP thread. All 36 expectations passed. Archive SHA-256 `df361671a9b1dbbbe4cd60e142bd099900693990eab460b0ceb78c35e567441b`; test blob SHA-256 `f487548071de2808d3f546bbeb90a83419beb7e66d6ff1b4b86846c484ec5be4`; raw log SHA-256 `1798fbdfbc97dcdd5e55c6d4a09049dead66ca11ebfe2297853f0eb8637a6856`.
- Read-only live PR refresh and diagnostics: R #1659 run 41491340175 passed; R #1657 run 41518765724 queued; Julia #1247 run 41493253608 failed in four 1.13.3 shards while Documenter run 41493253627 passed; Julia #1249 run 41519305510 queued while Documenter run 41519305443 passed. DRModels.jl #1228 runs 41245284485 and 41245284449 passed, with claim limitations recorded above. Totoro was reachable but heavily loaded (196/257/280); DRAC's control socket was present and the queue query returned no jobs. No additional compute was dispatched.
- No full R `CMD check`, full Julia suite, numerical comparison, or calibration study was run.

## 6. Tests of the Tests

The existing live test fits a Poisson `relmat()` model through `engine = "julia"` in a clean subprocess and checks finite, sensible fit output. The filtered test run reached that guarded live path. No deliberate fault injection or negative-control rerun was performed in this task.

## 7a. Issue Ledger

- drmTMB #1663 remains open, unmergeable, with no checks on exact head `83bfcf3c369eace9cf8d1363440f9e11953c758f`; its base is stale and the diff spans 3,825 files. Its owner worktree is extremely dirty and preserved.
- drmTMB #1657 remains open. Exact-head run 41518765724 has source-tree and OS-matrix jobs queued. No formal review is recorded; its Julia comparison pin is older than current main.
- drmTMB #1659 remains open. At 21:27 UTC, source-tree and OS-matrix checks passed, macOS passed, and current Ubuntu/Windows jobs were running; previous Ubuntu shards passed. No formal review is recorded; its base and Julia fixture pin are stale, and its stored parity evidence covers point estimates and Wald SEs rather than recovery or coverage. The separate Bernoulli tolerance test passed on exact current R main on Totoro/Linux; full current-main `R CMD check` remains unrun.
- DRModels.jl #1237 remains open. Exact-head run 41401878711 has two failing Julia 1.10 shards, including the J3a Hessian assertion and separate bootstrap assertions; attribution to current main remains unverified.
- DRModels.jl #1247 remains open and its exact-head Julia 1 shards still fail. Its reported J3a fixture at `test/test_j3a_gradient_hessian_vcov.jl:158` was not rerun. The separate all-singleton Hessian (#1114) and platform-sensitive TNB2 timing files pass on the focused exact-main Totoro runs; that does not clear #1247.
- Detailed logs for #1247 run 41493253608 attribute the four Julia 1.13.3 shard failures to the TNB2 timing bound, temporal dense-oracle tolerance, Gaussian all-singleton Hessian expectations, and location-scale theta tolerance. The Julia 1.13.3 J3a test was not isolated in this review. This clarifies the earlier aggregate failure note; it does not clear #1247.
- DRModels.jl #1228 is docs-only, based on stale `64ccb2d`; its four Julia shards and Documenter build passed, but the reviewed NB2-hurdle agreement claim, BetaBinomial structured-scope wording, and drmTMB refusal claims need tighter, current-source evidence before treating it as current documentation.
- A current-main source recheck confirms the `meta_V` accessor mismatch remains between drmTMB `129c09c8` and DRModels.jl `973bbb675`. The Julia guide says `sigma(fit) = tau`, while current Julia main returns `sqrt(v + tau^2)` from `sigma(fit)`. The current R method describes and returns residual heterogeneity. No overlapping source edit was made.
- DRModels.jl #1249 is open on current main `973bbb675`. At the 21:15 UTC refresh its four Julia shards were queued and its docs build and preview had passed. This is not a completed exact-head check.
- DRModels.jl #1240 remains a draft. Run 41248908491 fails shard 2 and aggregate `ci-ok` because fixture provenance is missing before parity assertions; other Julia shards and docs checks passed.
- #1666 and #1245 are merged at the current cached-main pair; their current exact-source checks are described in the crosswalk.

## 8. Consistency Audit

Compared the current-main page with the separate crosswalk branch and preserved both. Confirmed the latter has committed capability-page work based on an earlier main pin and dirty report/evidence files owned by another lane. Confirmed the current-main page lease is scoped to `docs/design/capability-status.md`; the status-refresh lease covers separate report, assessment, check-log, and evidence paths. Current PR heads and check states were reread from GitHub after the smoke. The page labels the Totoro run as a smoke and separates it from numerical parity and calibration.
The metadata-only census found 183 registered worktree/stash entries across both repositories: 161 usable worktrees, 22 missing paths, 101 dirty entries, and 21 stashes. These are preserved; the count is not a list of active owners. The separate exact-main Julia archive and focused tests were run on Totoro, not in any registered checkout.
The 21:27 UTC check refresh found R #1657 source-tree and OS-matrix success, with shard 4/4 running and shards 1/4 through 3/4 queued; R #1659 source-tree and OS-matrix success, macOS success, and current Ubuntu/Windows jobs running; and R #1663 without exact-head checks, still unmergeable. Julia #1247 still failed all four Julia 1 shards and aggregate `ci-ok`, while docs passed. Julia #1249 had shards 1/4 and 3/4 running, shards 2/4 and 4/4 queued, and docs/preview passed.

At 21:29 UTC, the connected GitHub refresh found R #1659 exact-head workflow 41491340175 successful, R #1657 workflow 41518765724 queued, Julia #1247 CI run 41493253608 failed with Documenter run 41493253627 successful, and Julia #1249 CI run 41519305510 queued with Documenter run 41519305443 successful. The PR metadata marks #1247 and #1249 mergeable. Workflow lookup returned no run for R #1663. These records update the earlier snapshot; no PR was changed or merged. Totoro was reachable, but its load averages were 196, 257, and 280. The existing DRAC control socket was present and the account queue query returned no jobs; no compute was dispatched.

## 9. What Did Not Go Smoothly

The initial local JuliaCall attempts failed during startup because the host needed Julia's bundled `libunwind` preloaded; a model-free setup probe then succeeded, and the final bounded bridge test completed. For the additional Totoro run, the first Julia 1.13 version alias was wrong and produced no test result; the failed launch is retained. The successful launch used Juliaup's installed `+1.13` alias. Its manifest was inherited from a Julia 1.10.10 environment, which Julia 1.13.1 warned about; JET precompilation emitted an error even though the requested test sets completed and the command exited zero. I kept that limitation visible. The `closeout.py new` command resolved a relative report path under the Shinichi brain instead of this repository; the generated template was moved intact into this worktree and the brain checkout was checked clean for that path. The repo-local after-task structure check and evidence gate passed, but the full closeout wrapper fails on 9 unmet gates in unrelated brain projects. I left those gates and files untouched. `route.py` returned that no repository LOAD-FIRST manifest exists for this worktree. The first remote source grep used `rg`, which is not installed on Totoro; the needed test lines were checked with `grep` instead.

## 10. Known Residuals

- This smoke does not establish numerical parity, parameter recovery, interval availability, or interval calibration.
- The full R and Julia package suites have not been qualified at this source pair. The focused exact-main R association test passed on Totoro/Linux but does not establish a full `R CMD check`.
- At the latest refresh, #1659's exact-head workflow passed and #1657's was queued. #1663 remains owner-held and unmergeable; #1237 and #1240 remain unresolved as described above. #1247's CI failed while docs passed, for the four shard failures listed above; its named J3a case was not isolated in this review. #1249's CI was queued while docs passed. #1228's checks passed, but its doc claims need revision before they are current-source evidence.
- The current-main page change is local and uncommitted. It has not been pushed or opened as a PR. Selective reconciliation with the older crosswalk branch remains for its page owner after the current-main refresh is reviewed.
- The full closeout compiler remains red because of the unrelated brain-project gates above; the repo-local after-task structure check and evidence gate passed.
- The incidental Python cache modification remains in the working tree and is excluded from staging.

## 11. Team Learning

Memory receipt: read `00-INDEX`, `WORKING-STYLE`, `WHAT-WORKS`, `DECISIONS`, and `PROJECTS` through the Shinichi brain connector; the repo `route.py` call found no LOAD-FIRST manifest. The Totoro-vs-DRAC choice followed the compute guidance: route a bounded single fit to Totoro and reserve DRAC for work that benefits from its queue or scale.

Golden Set: not run; this documentation and smoke task did not change model code or exercise a known-mistake regression class.

Reusable lesson: keep host startup workarounds in the run receipt, separate from model results. A successful bridge smoke is route evidence, not a parity comparison.

## 12. Cross-Product Coverage

Covers: one live drmTMB-to-DRModels.jl Poisson `relmat()` bridge route on the exact cached-main pair; four focused exact-main Julia test files across Julia 1.10.10 and 1.13.1; one focused R association test on exact current R main; and a source-pinned current PR/CI snapshot.

Does NOT cover: direct Julia parity for the bridge route, other response families or providers, numerical comparison, broad cross-language parity, recovery, inference, interval availability or calibration, full package checks, CMR, release, or deployment. The focused Julia tests do not establish full-suite or cross-version CI status and do not reproduce #1247's J3a failure. The focused R test does not establish full current-main `R CMD check`.
