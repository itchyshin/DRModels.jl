# Handover to Cursor: DRModels.jl twin-gap, temporal waves, issue sweep, v0.7.2

**You are Cursor**, taking over the DRModels.jl lane from Claude Code. This file is the state; you inherit no chat history. Written 2026-10-04 by Claude (Opus 5.5).

## Critical context

- DRModels.jl is the MIT Julia twin of the GPL R package drmTMB. **Never copy drmTMB source.** Parity uses generated outputs only.
- **Owner rules that bind you:**
  - **drmTMB merges:** never merge anything in drmTMB (D-164). Its PRs are owner-merged.
  - **Force-push and admin:** never force-push or rebase a pushed branch, and never `--admin`.
  - **Mentions:** never @-mention anyone. Run `python3 ~/shinichi-brain/tools/agent_mention_check.py --text <file>` on every PR body.
  - **Merges:** merge DRModels.jl PRs only when the owner says "merge N", and then only through `~/shinichi-brain/tools/pr_merge_when_green.sh` (the queue wrapper is `~/local-scratch/tools/land_queue.sh N`).
  - **Compute:** no Julia on the Mac. Use Totoro through the existing socket: `ssh -S ~/.ssh/cm-snakagaw@totoro.biology.ualberta.ca:22 snakagaw@totoro.biology.ualberta.ca`. Always prefix `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4`, and stay at 150 cores or fewer.
  - **Run length:** a run estimated over 3 h needs owner approval first (D-287).
  - **R libraries:** install drmTMB into PRIVATE R libraries on Totoro, never `~/R/lib`.
- **Foreign lanes, do not touch:**
  - Codex docs PRs #770, #777, #779, #789, #793, #795–#800, plus #576.
  - Cursor PR #809.
  - Claude reader PRs #801 and #802 (from another session).
  - Several issues are blocked on these lanes (see Next Steps).

## Goals

Close the R ↔ Julia twin gap. This session shipped temporal random effects (AR1, OU, phylo + OU, homogeneous Toeplitz) in both packages, residual fixes, an issue triage with owner decisions (D-316), and the v0.7.2 release prep that rebuilds the broken `/stable` docs.

## What was accomplished (merged on DRModels.jl main)

- **Temporal random effects:**
  - #917: phylo + OU and the temporal boundary diagnostic.
  - #918: homtoep.
  - #921: homtoep bootstrap and profile mean-only scope, the dropped-row bootstrap bug, honest Wald/bootstrap wording.
  - #924: AR1/OU conditional residuals, as in drmTMB; wave-1 parity regenerated at drmTMB 07d1612ea.
- **Residual fixes:**
  - #929: zero-inflated and hurdle quantile residuals use the mixture CDF (#922).
  - #954: a sigma-only random effect is no longer treated as a mean random effect (#923).
  - #925: `repeatability` / `icc` / `heritability` no longer count a log-σ random effect as a variance component. They use the marginal E[σ²] = exp(2b₀ + 2Σω²), as drmTMB does; the same fix applies to `_variance_boundary`.
- **Docs:**
  - #928: 12 small docs fixes.
  - #932: D-316 docs (#685, #700, #702, #711, #776).
  - #980: README first example uses drmTMB's growth story (Refs #674).
- **CI and evidence:**
  - #933: the Documenter build no longer deploys; a separate `docs-deploy` job publishes (closes #757).
  - #934: S9 non-Gaussian native fixtures refreshed against drmTMB 0eb0467851, with strengthened negative controls (Refs #606).
- **Issue closures:**
  - Closed with evidence: #49, #280, #467, #616, #627, #649, #671, #672, #673, #695.
  - New issues filed: #926, #927, #930, #931.

## Current working state

| Item | State | Branch / head | Next |
|---|---|---|---|
| **#967** location-scale stall stops (slow bootstrap test) | OPEN, auto-merge ARMED (owner said merge) | `claude/bootstrap-bridge-slow` | Lands when CI is green. Its review blocker (lost SEs) is fixed and verified: SEs are bit-identical to main. |
| **#1037** release v0.7.2 (NEWS assembled, version 0.7.2) | OPEN, auto-merge ARMED (owner said merge) | `claude/version-0.7.2` | Lands when CI is green. |
| **Tag `v0.7.2`** | **WRONG: on GitHub it points at `b28bee51c`** (pre-release, Project.toml still 0.7.1) | - | See Next Steps 1. |
| drmTMB drafts #1442, #1443, #1445, #1446→#1449 | All CI-green, base main, reviewed, owner-merge only | various `claude/*` | See Next Steps 3. |

## Next immediate steps (classify each OWED / DONE / RETRACTED / PROTECTED first)

1. **Fix the v0.7.2 tag (owner action).**
   - The owner pushed `v0.7.2` before #1037 merged. The owner then ran `git push --delete origin v0.7.2`, but at 2026-10-04 GitHub still showed the tag at `b28bee51c` (`gh api repos/itchyshin/DRModels.jl/git/refs/tags/v0.7.2`).
   - Once #1037 AND #967 have merged, give the owner these commands to run in order:
     - `git push --delete origin v0.7.2`
     - `git tag -d v0.7.2`
     - `git fetch origin && git tag v0.7.2 origin/main && git push origin v0.7.2`
   - The owner tags by hand: the repo has no `DOCUMENTER_KEY`, so TagBot tags would not trigger docs.
   - Then verify that the tag run's `docs-deploy` job succeeded, and that https://itchyshin.github.io/DRModels.jl/stable/ loads with CSS and working nav. DocumenterVitepress keeps `stable/` and `v0.7/` only; no `v0.7.2/` folder is expected.
   - Then close #670, #681 and #688, and check #679 and #699 (they stay open until `/stable` is rebuilt).
2. **#606 ledger (owner or ledger holder).**
   - #934 strengthened the batteries. The programme ledger `.unlazy/julia-r-parity` (not on this Mac) needs its EXPECT strings moved to `FINITE_FIT_NEGATIVE_CONTROLS_PASS 20`, `FINITE_PUBLIC_DAMAGES_REJECTED 20` and `JOINT_PUBLIC_NEGATIVE_CONTROLS_PASS mutations=24`, and the finite-public CHECK from `finite-public-005.json` to `-007.json`.
   - Then close #606.
3. **drmTMB drafts (owner merges, one at a time, in this order):** #1445 → #1446 → #1447 → #1448 → #1449 → #1442 → #1443.
   - After each merge, refresh the next draft: merge main in (never rebase), re-run the C17 runner and refresh TSV rows mc-0568, mc-0569 and mc-0576, resolve NEWS/check-log/census conflicts, and repin the lss-tip receipt.
   - #1442 and the temporal stack both edit `vcov.drmTMB()`; keep both blocks.
   - After the temporal stack merges, regenerate the DRModels.jl temporal parity cells at the merged drmTMB SHA (generators: `test/parity/gen_temporal_parity.R`, `gen_temporal_wave2_parity.R`).
4. **Blocked on foreign lanes (do not edit their files):**
   - #691 (`weights()` honesty): `src/comparison.jl` is in Codex #789, and `docs/src/rosetta.md` is in #802/#770/#779.
   - #674 follow-up: move `docs/src/index.md` (#798) and `docs/src/getting-started.md` (#800) to the README's growth example.
   - `docs/make.jl:96-97` has a stale `DRM_DOCS_DEPLOY` comment (make.jl is in #779/#576).
5. **Fixable, unowned:**
   - Triage verdicts are in the vault: `~/shinichi-brain/docs/dev-log/handover/2026-10-03-drmodels-issue-triage.json`.
   - Still fixable: #609 (R oracle reruns, about 1.5 h), #686 (a text correction, about 60 min) and #687 (the CONTRACT benchmark-labelling protocol).
   - New issues: #926 (`r2_constant_sigma` and the bootstrap simulator ignore a σ random effect), #927 (crash: mean + σ random intercepts with a σ covariate, `log(detH)` DomainError), #930 (whole-tree sparse-LSS profile `fallback_not_converged`, 10,970 tips), #931 (prior weights).
   - Small follow-ups:
     - `src/variational.jl` comments (lines 464, 479, 512, 554, 579) still call the NB2 slot log θ; it is log σ.
     - In the `docs` build job, permissions could drop to read (#933 review).
     - The ZI NB2 count CDF saturates at log σ < about −19; this is documented as inherited (#929).
6. **Owner decisions outstanding (drmTMB side):**
   - Temporal `sigma_ratio` uses marginal sd(y), so it can fire falsely when covariates dominate.
   - drmTMB's quantile-residual docs say "population level", but its code conditions on the modes.
   - drmTMB's Julia runners still call `pathof(DRM)`, the pre-rename name.

## Key decisions and rationale

- D-310 / D-311: temporal waves 1 and 2, twinned. D-316: issue-triage answers ("all recommended").
- Twin fidelity: where drmTMB refuses intervals (temporal bootstrap; Wald for OU and paired fits), DRModels.jl documents its extra intervals as an uncalibrated extension rather than claiming "as in drmTMB".
- Residual conventions: temporal AR1/OU/paired residuals are conditional on the modes; homtoep residuals are whitened; ordinary `(1|g)` residuals are σ_b-marginal; ZI/hurdle residuals use the mixture CDF.
- Correctness beats speed: #967 kept main's SEs bit-identical and accepted a smaller speed-up on the ZEN-kernel runners (22 min there, not 1.5 min).

## Gotchas and failed approaches

- **Do not call a CI job "hung" without reading its log.** Claude cancelled two progressing runs; the slow `Julia 1 - shard 2/4` was `test_bridge_bootstrap_tree.jl` taking 12–74 min, not a hang (fixed by #967).
- Agents twice ran Julia on the Mac or force-pushed with lease. Restate the rules in every brief.
- Fixing a flipped negative control can silently weaken a guard. Mutation-test the validator (the #934 lesson).
- Name-suffix routing (`_logsigma`) misfires on user columns named `*_logsigma`. Use `_is_logsigma_re(fit, nm)` (src/heritability.jl), which checks the sigma formula.
- GitHub runner queues are slow, because GLLVModels.jl uses most of the account's concurrent jobs.
- `tools/handoff_gate.sh` reports 14 `.unlazy/twin-gap` ledgers as INCONCLUSIVE because node/gate-check is unavailable. That is not a pass; run them with `node ~/.claude/skills/unlazy/scripts/gate-check.mjs` if needed.

## Files created / modified this session (all merged, except the open PRs)

- **src:** `temporal.jl`, `inference.jl`, `introspection.jl`, `gaussian_core.jl`, `quantile_residuals.jl`, `heritability.jl`, `boundary_diagnostics.jl`, `visualization.jl`, `bridge.jl` (comments). Open PR #967: `locscale_inner.jl`, `locscale_fit.jl`.
- **tests:**
  - `test_temporal_{ar1,ou,phylo_ou,homtoep,boundary,residuals}.jl`
  - `test_parity_temporal.jl`
  - `test/parity/temporal/*`
  - `test_quantile_residuals_{zi_hurdle,re_axis}.jl`
  - `test_repeatability_logsigma_re.jl`
  - `test/fixtures/musigma_ranef_745/native_repeatability.tsv`
- **docs:**
  - `docs/src/tutorials/temporal-random-effects.md`
  - `capabilities.md`, `README.md`, `large-data.md`, `coming-from-r.md`, `api-stability.md`, `location-scale.md`, `location-scale-scale.md`, `profile-likelihood.md`
  - deleted: `changelog.md`, `get-started.md`
- **CI:** `.github/workflows/Documenter.yml`, `docs/deploy_preview.jl`.
- **Evidence:** `docs/dev-log/evidence/julia-r-parity/nongaussian-refresh-20261003/`, plus `tools/receipt_paths.py` and the validator batteries in `tools/`.
- **Release PR #1037:** `NEWS.md`, `Project.toml`, and the removal of `news/*.md`.
- **This handover:** `docs/dev-log/handover/2026-10-04-cursor-handover.md` and `docs/dev-log/coordination-board.md`.

## Environment for Cursor

- **Repo:** `/Users/z3437171/Dropbox/Github Local/DRM.jl` (GitHub `itchyshin/DRModels.jl`, default branch `main`, protected; the required checks are `docs` and `ci-ok`).
- **Do not stage** the untracked `INBOX.md` in the main checkout.
- **Worktrees** under `/Users/z3437171/local-scratch/lanes/DRM.jl-*` are Claude's; leave them unless you own the branch.
- **Verify (on Totoro, not the Mac):** clone your branch there, then run `julia +1.13 --project=<env> -e 'include("test/<file>.jl")'` with the env developed from the clone. Run the full suite with `DRM_TEST_SHARD=k/4`.
- **Docs audit:** `python3 tools/reader_surface_audit.py --public-only` (Python runs fine on the Mac).
- **Totoro scratch from this session (safe to ignore):** `~/claude-*` dirs, e.g. `claude-rls3`, `claude-606b`, `claude-606c`, `claude-bootslow`, `claude-rv*`, `claude-q922`, `claude-q923*`, `claude-rel072`, `claude-tfu`. Another lane's long `Pkg.test` in `~/lanes/DRModels.jl-class3-j110` is not ours.
- **Running-log context:** `~/shinichi-brain/docs/dev-log/handover/2026-09-28-twin-gap-morning-report.md`.

## How to resume

Start a fresh Cursor agent in the repository and paste:

```text
Read AGENTS.md and docs/dev-log/handover/2026-10-04-cursor-handover.md. Run the handover rehydration steps, reconcile them with the current git state, then continue only the OWED Next Immediate Steps.
```
