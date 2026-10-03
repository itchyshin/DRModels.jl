# S9 non-Gaussian native fixture refresh (issue #606), 2026-10-03

Evidence-only refresh. No `src/` change and no tolerance change. The 4e-6 native
bar, every comparator and every estimator are unchanged.

## Why

drmTMB `main` 0eb0467851 (0.7.1) Newton-polishes fits by default
(`drm_control(newton_polish = TRUE)`). The frozen native fixtures were written by
the 2026-08-30 build, whose `nlminb` stopped with a maximum gradient of 2.9e-4
to 1.6e-3. The S9 ledger checks compared current Julia fits against those
stopping points, so they stayed red even though Julia was already at the
optimum. The re-measure on #606 (2026-10-03) showed this: with polish off,
current drmTMB returns the old frozen estimates; with polish on, every gap falls
below 1.1e-10.

## Comparator provenance

| Item | Value |
|---|---|
| drmTMB checkout | `main` 0eb0467851cc2902556764bdca5441b7bc053be6 (Totoro `~/claude-mr/1442/main`) |
| drmTMB build | private library `~/claude-mr/1442/Rlib-main`; see `drmtmb-provenance.toml` |
| drmTMB build git SHA | `0eb0467851cc2902556764bdca5441b7bc053be6`, `GitDirty: FALSE` (the build's own `build-provenance.dcf`) |
| drmTMB `code_hash` | `4c37c67c63ecba08ea16459250f80915ee0ee972c1030ac4df8855bbf95277b3` (`tools/drmtmb_provenance.R`) |
| Native DLL sha256 (finite export) | `2028c2cf8afdc88e038707972755eda7ebb7739e9cf381cf5973c0b8faef7f2f` |
| R / TMB | R 4.6.1 / TMB 1.9.25, Totoro, one BLAS/OMP thread |
| Julia | 1.10.12 (fixtures and committed receipts); 1.13.1 rerun of every check |
| DRModels.jl source | `origin/main` 959d71fb7; `src/` byte-identical in every run |

Only drmTMB came from the private library `~/claude-mr/1442/Rlib-main`; it was
never loaded from `~/R/lib`. Every other R package loaded from Totoro's `~/R/lib`:
TMB 1.9.25, JuliaCall 0.17.6, jsonlite 2.0.0, digest 0.6.39 and pkgload 1.5.1. The
harness exports `R_LIBS_USER=/nonexistent`, but Totoro's `~/.Renviron` sets
`R_LIBS_USER=~/R/lib`, which wins, so `.libPaths()` is `~/R/lib`,
`/usr/local/lib/R/site-library`, `/usr/lib/R/site-library`, `/usr/lib/R/library`.
Nothing was installed into `~/R/lib`.

## Generators (the repository's own, unmodified)

| Fixture | Generator |
|---|---|
| `missing-predictor-oracle/native-mi-oracle-003.json` | `tools/check_missing_predictor_oracle.R` (DRModels.jl) |
| `joint-prototype/joint-reference-002.json` | `tools/export_joint_predictor_reference.R` |
| `test/fixtures/joint_missing_predictor/native_reference.toml` | `tools/joint_reference_to_toml.py` |
| `joint-frontend/joint-native-uncertainty-current-002.json` | `tools/joint_native_uncertainty_probe.R` (the `native-uncertainty:G2` CHECK) |
| `test/fixtures/joint_missing_predictor/native_uncertainty.toml` | `tools/export_joint_uncertainty_reference.py` |
| `finite-state/finite-native-003.json` | drmTMB `tools/export-finite-joint-reference.R` (runner sha256 `50976bdd…`, unchanged) |
| `finite-state/finite-reference-003.toml` | `tools/finite_reference_to_toml.py` |
| `finite-state/finite-fit-002.toml`, `finite-julia-003.toml` | `tools/check_finite_joint_fit.jl`, `tools/check_finite_joint_reference.jl` |
| `finite-frontends/finite-public-007.json` (new receipt; 005 is historical) | drmTMB `tools/run-julia-joint-finite-public.R` |
| `joint-bridge/joint-public-003.json` | drmTMB `tools/run-julia-joint-public.R` |
| `joint-prototype/joint-{fit,native}-003.toml`, `joint-frontend/joint-frontend-fit-002.toml` (= `joint-bridge/joint-direct-bridge-002.toml`) | `tools/check_joint_predictor_fit.jl`, `check_joint_predictor_reference.jl`, `check_joint_frontend_fit.jl` |

Three things about the run environment, all recorded here and not hidden:

1. **drmTMB runners and the private library.** The drmTMB runners call
   `pkgload::load_all()` on the checkout. `privlib_shim.R` attaches the private
   build with `library(drmTMB)`, turns `load_all` into a no-op, and then
   `source()`s the unmodified runner in place. No drmTMB file is copied or
   edited, so each receipt's `runner_sha256` is the real runner's hash.
2. **The `DRM` module alias.** The two public runners record the loaded Julia
   source as `pathof(DRM)`, the module's name before the DRModels rename. drmTMB
   binds the loaded backend as `Main.drmTMB_backend`, so the shim defines
   `const DRM = drmTMB_backend` after `drm_julia_setup()`. This only records an
   identity and changes no numbers. Follow-up on the drmTMB side: the runners
   should use `pathof(drmTMB_backend)`.
3. **JuliaCall on Totoro.** Embedding Julia in R segfaulted inside the system
   `libunwind` while Julia was unwinding the stack, both in `jp_run` and in a
   minimal `julia_setup()`. Preloading Julia's own
   `lib/julia/libunwind.so.8` (`LD_PRELOAD`) fixes it for both runtimes.

## Data reproduction

Both generators simulate their data from fixed seeds (20260830; 96403/96404).
On Totoro (Linux, R 4.6.1) the regenerated data agree with the frozen Mac data
(R 4.6.0) to within 1.3e-15. They are not bit-identical: the last-ulp
differences come from the platform's `rnorm`. Masks, categorical codes and row
denominators are identical. Every receipt is internally consistent with the
regenerated data.

## Before / after (same Julia source, Julia 1.10.12)

"Before" means the frozen fixtures and current drmTMB (`summary-before.txt`);
"after" means the refreshed fixtures (`summary-after-out.txt`,
`summary-final-110.txt`).

| Leaf / quantity | Before | After | Bar |
|---|---|---|---|
| `finite-state-evidence:G4` ordinal theta / prediction / imputation / cond. SD | 2.16e-6 / 7.56e-6 / 5.12e-6 / 2.81e-6 FAIL | 2.7e-11 / 1.7e-11 / 2.8e-11 / 2.2e-11 PASS | 4e-6 |
| `finite-state-evidence:G4` categorical theta / prediction | 1.74e-5 / 9.58e-6 FAIL | 1.0e-10 / 7.7e-11 PASS | 4e-6 |
| `finite-public:G3` `native_status` (ordinal, categorical) | FAIL, FAIL | PASS, PASS | 4e-6 |
| `joint-fit-parity:G1` max \|native − Julia theta\| (Gaussian, Bernoulli) | 2.75e-6, 1.0015e-5 FAIL | 3.8e-12, 1.09e-11 PASS | 4e-6 |
| `joint-public-fit:G5` native theta (Gaussian, Bernoulli) | 2.75e-6, 1.0015e-5 FAIL | 3.8e-12, 1.09e-11 PASS | 4e-6 |
| `r-joint-native:G1`, committed receipt (worst native error) | Gaussian training mean 0.933; Bernoulli `newdata` ERROR — FAIL | ≤3.5e-9 (Gaussian imputed SE); all others ≤3.1e-11 — PASS | 4e-6 |
| `native-uncertainty:G1` preflight | FAIL (frozen JSON pins old DLL md5) | PASS | — |
| `native-uncertainty:G2` mean / SE error (Gaussian) | not reached (8.6e-4 on 2026-09-02) | 4.4e-16 / 2.8e-13 PASS | 1e-6 |

Every other check passes, including all negative-control batteries (normal and
`python3 -O`) and the historical finite stopping diagnostic. Two failures in the
logs belong to the negative controls described under "Validator edits" below,
not to parity. `fp_G2` in `summary-after-out.txt` and the first two
`finite-state-evidence_G2` lines in `summary-final-110.txt` failed because those
controls assumed a failing verdict. Both were fixed and rerun; the reruns are
the later lines.

On Julia 1.13.1 (`summary-after13-out.txt`, `summary-final-113.txt`) every
generator, oracle and negative-control check passes, and all 12
`test/test_joint_missing_*.jl` files pass. There is one exception: the
finite-fit receipt validator pins `julia_version == "1.10.12"`, so a 1.13
receipt fails its `runtime` field by design. Its parity numbers match 1.10.12
(categorical theta 1.0078e-10). The docs-subset gates (`finite-state-evidence:G3`,
`finite-public:G4`) pass on 1.10.12 (`summary-docs-110.txt`).

## Validator edits (anchors and negative controls only)

- `check_finite_native_reference.py`: `REFERENCE_SHA256` moves to the
  regenerated anchor. The anchor's `fit_control` now carries `newton_polish`
  and `start`, and the validator compares against the anchor's own complete
  list, so a missing or changed control is still rejected.
- `check_finite_fit_receipt.py`: the runtime pin changes from `1.10.0` (the
  2026-08-30 Mac runtime) to `1.10.12`, the runtime that produced the refreshed
  receipt.
- `check_finite_stopping_diagnostic.py`: still explains the pre-polish stopping
  point. It now reads the retained historical fixture
  `finite-stopping/finite-native-003-prepolish.json`, byte-identical to the old
  anchor (sha256 `d8f75d1d…`).
- Three negative controls had assumed a failing native verdict. They now
  invert the honest verdict or offset the reported error:
  `check_finite_public_receipt.py` (`native_status`) and
  `test_finite_fit_receipt.py` (`false_pass`, `reported_error`). On their own
  these only test the harmless direction (a false FAIL); see the review fixes
  below, which add the forged-PASS direction.

## Review fixes (PR #934 review, 2026-10-03)

- **Forged-PASS and threshold controls (B1).** With the honest verdict now PASS,
  a validator that always says PASS, or that loosens 4e-6 to 1e-3, passed both
  batteries. Each battery now builds a temporary copy of the native anchor whose
  theta is moved, updates the receipt's reported theta error to the honest value
  against that copy, and keeps the verdict at PASS. Two such receipts must be
  rejected on the verdict check itself: anchor theta + 1e-5, and a theta error of
  4.004e-6 (just above the bar). A third at 3.996e-6 must be accepted, which shows
  the other two fail only on the verdict. Battery sizes: `test_finite_fit_receipt.py`
  17 → 20 (`FINITE_FIT_NEGATIVE_CONTROLS_PASS 20`), `check_finite_public_receipt.py
  --damage` 17 → 20 (`FINITE_PUBLIC_DAMAGES_REJECTED 20`). The same gap existed in
  the joint public bridge battery, so it gets the same three controls (native
  theta moved in the receipt, native log-likelihood recomputed):
  `test_joint_bridge_public_receipt.py` 21 → 24
  (`JOINT_PUBLIC_NEGATIVE_CONTROLS_PASS mutations=24`). Ledger EXPECT strings that
  pin 17, 17 or 21 must move to 20, 20 and 24. Mutation results are in
  `mutation-before.txt` and `mutation-after.txt`.
- **Receipt numbering (M1).** The refreshed public receipt is the new file
  `finite-frontends/finite-public-007.json`. `finite-public-005.json` is restored
  byte-for-byte from `main`; it and 006 are historical FAIL receipts against the
  pre-polish anchor. The progress ledger, `final_checks.sh` and the
  `finite-frontends/` README point at 007.
- **Path portability (M2).** See "Path portability" below.
- **Newton cross-check (m1).** Pre-polish anchor + one independent Newton step
  equals the polished anchor to 4.5e-11 (ordinal) and 9.9e-11 (categorical)
  (`newton_xcheck.py`, `newton-xcheck-out.txt`; also in `../finite-stopping/`).
- **Provenance (m2, m3).** The drmTMB build's git SHA and dirty flag are in
  `drmtmb-provenance.toml`; the host-library packages are named above.

The validators and batteries were rerun from a different checkout path
(`~/claude-606c/tree`) on Julia 1.10.12 and 1.13.1: `summary-review-110.txt`,
`summary-review-113.txt`.

## Path portability

The runner receipts record absolute paths from the machine that wrote them
(`/home/snakagaw/claude-606b/after/...` for Julia, `/home/snakagaw/claude-mr/1442/main/...`
for drmTMB). The receipt bytes are evidence and were not rewritten. Instead, the
three validators that compared those paths now compare source manifests keyed by
root and repository-relative path, with the same per-file sha256
(`tools/receipt_paths.py`). The recorded Julia root is the directory of the loaded
`src/DRModels.jl`, and the recorded drmTMB root is the directory of the recorded
`NAMESPACE`; every recorded path must sit under one of them.

| Receipt | Absolute paths recorded | Validator | Portable now |
|---|---|---|---|
| `finite-state/finite-fit-002.toml` | `runtime.loaded_source` | `check_finite_fit_receipt.py` | yes (was: exact path) |
| `finite-frontends/finite-public-007.json` | `runtime.source`, `source_before/after` keys | `check_finite_public_receipt.py` | yes (was: exact paths) |
| `joint-bridge/joint-public-003.json` | `runtime.source`, `source_before/after` keys | `check_joint_bridge_public_receipt.py` | yes (was: exact paths) |
| `joint-prototype/joint-{fit,native}-003.toml`, `joint-frontend/joint-frontend-fit-002.toml` (= `joint-bridge/joint-direct-bridge-002.toml`) | `loaded_source` only | `check_joint_predictor_{fit_,}receipt.py`, `check_joint_frontend_fit_receipt.py` (relative-path manifests) | yes (path never checked) |
| `finite-state/finite-julia-003.toml` | `loaded_source` only | none in Python; `tools/check_finite_joint_reference.jl` asserts while writing it | not applicable |
| `missing-predictor-oracle/native-mi-oracle-003.json`, `joint-frontend/joint-native-uncertainty-current-002.json` | private drmTMB library paths | the R probes compare md5 of the loaded files | yes on any host with that build; the library is an argument |

What this does not cover: the drmTMB checkout is still a validator argument, and
its `R/`, `NAMESPACE`, `src/` and runner bytes must match the recorded hashes, so
the public checks need a drmTMB checkout at 0eb0467851. The finite-fit validator
still pins `julia_version == "1.10.12"`.

## Not covered

- `S10 matched-native:G1` was not run.
- The programme ledger itself (`.unlazy/julia-r-parity`) is not on this
  machine. The CHECK commands were run with their environment-specific paths
  substituted: `/private/tmp/drm-parity-20260830/{drmTMB,R-lib,DRM.jl}` became
  the drmTMB checkout, the private library and the tree above. Ticking the
  ledger is left to whoever holds it.
- The evidence-set `manifest.json` files in `joint-bridge/`, `joint-prototype/`,
  `joint-frontend/` and `missing-predictor-oracle/` are dated snapshots and were
  not rewritten. `sha256.txt` here lists the current hashes.
- Receipt bytes still record absolute Totoro paths; the validators no longer
  depend on them (see "Path portability").
- The full `Pkg.test()` suite was not run. Only the 12 affected
  `test_joint_missing_*` files were run.
