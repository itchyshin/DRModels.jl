## 1. Goal

Correct one false R-to-Julia export gap in the DRModels.jl parity ledger and prevent it from returning.

## 2. Implemented

- Added the explicit mapping `beta_family` to Julia's `Beta` constructor in `tools/parity_ledger.py`.
- Extracted same-name and alias matching into `has_julia_twin()` so the rule can be tested directly.
- Added `tools/tests/test_parity_ledger.py` with a regression test for the Beta alias.
- Added an explicit Julia `--julia-ref` (default `HEAD`) and read exports from that committed ref rather than the dirty working tree. The ledger prints both source refs and SHAs.
- Exact-main parity ledger now reports zero unmatched drmTMB exports. Capability rows remain separately classified.

## 3a. Decisions and Rejected Alternatives

- Kept the public Julia API unchanged. `Beta()` is already the equivalent family constructor; adding a duplicate API would be misleading.
- Rechecked same-name matches for `converged` and `convergence_status`; they already match on exact Julia main.
- The Julia ref defaults to `HEAD`; callers comparing main branches can select `--julia-ref origin/main`. Both refs are resolved to commit SHAs and displayed so the source pair is explicit.

## 4. Files Touched

- `tools/parity_ledger.py`
- `tools/tests/test_parity_ledger.py`
- `docs/dev-log/check-log.d/2026-10-27-parity-ledger-beta-alias.md`
- `docs/dev-log/after-task/2026-10-27-parity-ledger-beta-alias.md`
- Brain receipt: `Shinichi/docs/dev-log/after-task/2026-10-27-parity-ledger-beta-alias.md`

## 5. Checks Run

### 2026-11-01 source-pinned recheck

- Revalidated live GitHub `main` against the local refs: drmTMB `b535bfbcce2a9039d869c2e34fe9da476394bf62`; DRModels.jl `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716`.
- `python3 tools/tests/test_parity_ledger.py`: 2 tests passed in 5.459 seconds.
- `python3 tools/parity_ledger.py --drmtmb '/Users/z3437171/Dropbox/Github Local/drmTMB' --ref origin/main --julia-ref origin/main`: exit 0; zero unmatched R exports; 37 capability rows classified; closure passed. Current counts: 15 raw unmatched names, all 15 accounted for; 40 Julia-only exports, 98 accounted for; 18 intentional-error gates.
- `git diff --check -- tools/parity_ledger.py`: passed.
- This recheck read Git source refs and ran the focused Python tests. It did not run R or Julia model fits, package-wide tests, CI, or interval validation.

### Original 2026-10-27 checks (historical)

- Current DRModels.jl main SHA: `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716`; drmTMB main SHA: `02095c67d7e70bbb648f45d2bbdd5aa0b1ea7be5`.
- `python3 tools/tests/test_parity_ledger.py`: 2 tests passed, including a dirty-worktree fixture proving Julia exports come from the requested commit.
- `python3 tools/parity_ledger.py --drmtmb '/Users/z3437171/Dropbox/Github Local/drmTMB' --ref origin/main --julia-ref origin/main`: source SHAs printed as R `02095c67d7e70bbb648f45d2bbdd5aa0b1ea7be5` and Julia `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716`; zero unmatched exports; 37 capability rows classified as 10 covered, 22 partial, 4 experimental, and 1 unsupported; closure passed under the ledger's documented invariant.
- Repeated both commands on 2026-10-28 after resuming the lane: 2/2 tests passed, and the exact-main ledger again reported zero unmatched exports and 37 classified capability rows.
- `git diff --check`: passed.
- `julia tools/build_check_log.jl`: not run; the `julia` executable is unavailable in this shell.
- Rose read-only review: no blocking findings.

## 6. Tests of the Tests

Before implementation, the alias regression failed because `has_julia_twin()` did not exist; after extraction and the alias mapping it passed. The source-pinning test first failed because `drmjl_exports()` had no ref argument; after the fix it verified that a dirty file does not affect exports read from `HEAD`.

## 7a. Issue Ledger

- Fixed: one false export-gap count caused by the R and Julia names differing.
- No issue or PR is attached to this checker-only slice.
- Still open: broad route parity, two Julia exact-main CI failures, and the R macOS/R 4.6.1 CMD-check failure. No merge or CI dispatch was performed.

## 8. Consistency Audit

Compared the current R and Julia main refs, reran the parity ledger from an exact-main Julia worktree, checked the matching Julia exports and family docs, and had an independent reviewer inspect this patch. The correction changes ledger accounting only; it does not claim equivalent behavior beyond the established family and status APIs.

## 9. What Did Not Go Smoothly

The first ledger run used a stale Julia checkout and counted names absent from that checkout. The exact-main rerun showed only the Beta alias needed correction. The report generator initially listed unrelated files from the shared Brain checkout; the list was replaced with the actual files for this task. The final attempt to claim `DECISIONS.md` and `AGENT_LOG.md` remained pending in the shared lease scan, so no entries were made there.

## 10. Known Residuals

This closes one export-name accounting error and makes the Julia side of each ledger run source-pinned. Broader parity remains open. Full package tests are not green. CMR Phase B interval availability and coverage/calibration remain deferred. At the original 2026-10-27 checkpoint, the patch was uncommitted and unpublished; see the dated source-pinned recheck above and Git for its current landing state.

## 11. Team Learning

Memory receipt: `route.py` found no LOAD-FIRST manifest for this worktree. Read the supplied AGENTS.md, repository handover and roadmap, and ran path-specific lane preflight. Exact main refs were used to distinguish current capabilities from stale checkout state.

Golden Set: not in scope; this was a Python export-name matching correction. No statistical or model-kernel failure class was investigated.

## 12. Cross-Product Coverage

Covers: one drmTMB export name mapped to its existing DRModels.jl equivalent, and reproducible export comparison against an explicit Julia Git ref.

Does NOT cover: runtime fit parity, the R-to-Julia bridge, package-wide tests, statistical recovery, interval calibration, deployment, or release readiness.
