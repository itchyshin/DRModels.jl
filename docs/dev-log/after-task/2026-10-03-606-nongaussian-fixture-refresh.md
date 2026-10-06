# After-Task Report: S9 non-Gaussian native fixture refresh (#606)

- **Date:** 2026-10-03
- **Issue:** #606 (part of #563, S9). Branch `claude/606-nongaussian-bar`, draft PR.
- **Perspectives:** Shannon (Rose after-task pass). No subagents.

## 1. Goal

Regenerate the frozen native fixtures that the S9 non-Gaussian ledger leaves
compare against. The source is drmTMB `main` 0eb0467851, which Newton-polishes
fits by default. The change is evidence-only: no `src/` change and no tolerance
change.

## 2. Implemented

- Fixtures were regenerated with the repository's own generators, on Totoro,
  against the private drmTMB library:
  - joint: `native-mi-oracle-003.json`, then `joint-reference-002.json`, then
    `native_reference.toml`. The `native-uncertainty` probe produced
    `joint-native-uncertainty-current-002.json`, from which
    `native_uncertainty.toml` was exported.
  - finite: `finite-native-003.json`, then `finite-reference-003.toml`.
- Receipts were regenerated against the new fixtures:
  - in place: `finite-fit-002.toml`, `joint-public-003.json`;
  - the public receipt as the new `finite-public-007.json` (after review; 005
    restored from `main`);
  - new files: `finite-julia-003.toml`, `joint-fit-003.toml`,
    `joint-native-003.toml`, and `joint-frontend-fit-002.toml` (the same file
    as `joint-direct-bridge-002.toml`).
- Validator edits:
  - the finite anchor hash moves to the new fixture, whose `fit_control` now
    includes `newton_polish` and `start`;
  - the finite-fit runtime pin changes from `1.10.0` to `1.10.12`;
  - the stopping diagnostic is repinned to a retained pre-polish copy;
  - three negative controls that assumed a failing verdict now invert the
    honest verdict instead.
- `nongaussian-refresh-20261003/` holds the provenance, before/after table,
  harness scripts and every run summary.

## 3a. Decisions and Rejected Alternatives

- **Regenerate in place, not as 004.** The ledger CHECK lines pin the 003 and
  002 paths, and the task named those files. Git history keeps the old bytes,
  and the stopping diagnostic keeps its own copy.
- **Repository generators, unmodified.** The drmTMB runners use `load_all()`.
  A shim attached the private build and `source()`d each runner in place, so no
  drmTMB file was copied or edited. A refit-on-frozen-data generator was
  rejected: it would be a new tool, not the repository's own.
- **The 1.10.0 runtime pin moves to 1.10.12.** Installing Julia 1.10.0 to
  satisfy an old pin was rejected, because the brief fixed the runtimes at
  1.10.12 and 1.13.
- **Manifests not rewritten.** The per-directory `manifest.json` files are
  dated snapshots. `sha256.txt` lists the new hashes instead.

## 4. Files Touched

The PR changes only evidence files, test fixtures and five `tools/*.py`
validators. `src/` is untouched.

## 5. Checks Run

All runs were on Totoro with one OMP/BLAS thread; scripts that require one
Julia thread got one.

- `run_checks.sh`: every affected CHECK, including generators, oracles and
  negative controls. Run before and after the refresh on 1.10.12, then after
  on 1.13.1.
- `final_checks.sh`: the CHECKs on the committed paths, plus all 12
  `test_joint_missing_*` files, on both runtimes.
- `docs_gates.sh`: the two docs-subset gates, on 1.10.12.
- The results and the before/after table are in the evidence README.

## 6. Tests of the Tests

- The "before" run reproduced the known failures on frozen fixtures:
  - finite ordinal theta 2.16e-6 and categorical theta 1.74e-5;
  - joint Bernoulli 1.0015e-5;
  - an md5 preflight stop for `native-uncertainty`.
  The checks therefore fail for the right reason before the change and pass
  after it.
- Negative-control batteries run normally and under `python3 -O`.
- Three controls stopped rejecting their damage once parity became true. They
  were fixed so that they invert the honest verdict, and still reject in both
  directions.

## 7a. Issue Ledger

Every S9 leaf named in #606 passes its own CHECK:

- `native-uncertainty:G2`;
- `finite-public:G3` (and the collateral `G2`, `G4`);
- `finite-state-evidence:G4` (and the collateral `G2`, `G3`);
- `joint-fit-parity:G1`, `joint-public-fit:G5`, `r-joint-native:G1`.

The ledger file itself (`.unlazy/julia-r-parity`) was not available, so no box
was ticked here.

## 8. Consistency Audit

- READMEs updated: `finite-state/` and `joint-bridge/`.
- `missing-predictor-progress.json`: the `default_fit_parity` field updated.
- `required_gaps` / `next` in the progress file still list strict native
  parity. Left for the ledger owner.

## 9. What Did Not Go Smoothly

- JuliaCall segfaulted in the system `libunwind`. Preloading Julia's bundled
  copy fixed it.
- The drmTMB public runners still call `pathof(DRM)`, the pre-rename module
  name. A shim alias bridged it; the runners should be fixed on the drmTMB side.
- JuliaCall's dependency installer for 1.13 failed once and succeeded on rerun.

## 10. Known Residuals

- `S10 matched-native:G1` was not run.
- The full `Pkg.test()` suite was not run.
- On 1.13, the finite-fit receipt fails only on its runtime pin, by design.
- The public receipts pin absolute paths under `/home/snakagaw/claude-606b/after`.
- Regenerated data differ from the frozen Mac data by ≤1.3e-15, from
  platform-level `rnorm` differences.

## 11. Team Learning

A comparator that changes its optimizer default turns every frozen native
fixture into a stale stopping point. Re-measure with the new default switched
off before blaming the engine.

## 12. Cross-Product Coverage

- **Covers:** the 2 joint and 2 finite S9 cases at the 4e-6 bar, on Julia
  1.10.12 and 1.13.1.
- **Does NOT cover:**
  - other missing-predictor cases;
  - recovery or coverage;
  - warm performance;
  - S10 leaves;
  - any programme gate G0–G8.

## Addendum: PR #934 review fixes (2026-10-03)

- **B1.** Both finite batteries, and the joint public bridge battery (same gap),
  gain a forged-PASS control (native theta +1e-5), a threshold control (theta
  error 4.004e-6) and a just-below positive control (3.996e-6). Counts: 17 → 20,
  17 → 20, 21 → 24. Mutation test: mutants A, D, F, G survived before and are
  killed after; B, C, E stay killed; the joint verdict mutants H, I survived the
  joint battery before this fix and are killed after. Evidence:
  `nongaussian-refresh-20261003/mutation-{before,after}.txt`.
- **M1.** The refreshed public receipt is `finite-public-007.json`; 005 is restored
  byte-for-byte from `main`.
- **M2.** `tools/receipt_paths.py`: the finite-fit, finite-public and joint-bridge
  validators compare source by repository-relative path and sha256. Receipt bytes
  are unchanged.
- **m1–m3.** Stopping README points at the pre-polish copy and records the Newton
  cross-check (4.5e-11, 9.9e-11); drmTMB git SHA added to the provenance file;
  host-library packages named.
- Checks rerun from `~/claude-606c/tree` on Julia 1.10.12 and 1.13.1
  (`summary-review-{110,113}.txt`).
- Harness artefact: one 1.10.12 gate run failed on a stale `.pyc` left by a
  same-size mutant; after invalidating the cache it passes. The mutation tables
  were rerun with bytecode caching off. Recorded in the evidence README.
