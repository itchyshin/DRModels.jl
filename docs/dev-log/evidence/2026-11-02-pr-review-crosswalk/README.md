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
