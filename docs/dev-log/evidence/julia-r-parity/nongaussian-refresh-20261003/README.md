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
| drmTMB `code_hash` | `4c37c67c63ecba08ea16459250f80915ee0ee972c1030ac4df8855bbf95277b3` (`tools/drmtmb_provenance.R`) |
| Native DLL sha256 (finite export) | `2028c2cf8afdc88e038707972755eda7ebb7739e9cf381cf5973c0b8faef7f2f` |
| R / TMB | R 4.6.1 / TMB 1.9.25, Totoro, one BLAS/OMP thread |
| Julia | 1.10.12 (fixtures and committed receipts); 1.13.1 rerun of every check |
| DRModels.jl source | `origin/main` 959d71fb7; `src/` byte-identical in every run |

drmTMB was always loaded from the private library, never from `~/R/lib`. Its
dependencies (TMB, jsonlite, digest, JuliaCall, pkgload) come from the host
library, as before.

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
| `finite-frontends/finite-public-005.json` | drmTMB `tools/run-julia-joint-finite-public.R` |
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
  `test_finite_fit_receipt.py` (`false_pass`, `reported_error`). The battery
  counts (17 and 17) are unchanged.

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
- `joint-public-003.json` and `finite-public-005.json` pin absolute source paths
  under `/home/snakagaw/claude-606b/after`, as earlier receipts pinned
  `/private/tmp/...`. They validate only against that tree.
- The full `Pkg.test()` suite was not run. Only the 12 affected
  `test_joint_missing_*` files were run.
