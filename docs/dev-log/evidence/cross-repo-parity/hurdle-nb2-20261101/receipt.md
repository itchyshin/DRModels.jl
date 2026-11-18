# Exact-main hurdle NB2 bridge output comparison

## Claim and scope

This receipt measures one fixed-effects hurdle-NB2 output cell on the current
drmTMB and DRModels.jl main sources. It compares distributional predictions,
response-scale fitted values, response residuals, and log likelihood for the
same simulated 1,800-observation data set. Both fits reported convergence.
This is a point-fit parity check; it says nothing about intervals, interval
availability, coverage, recovery, random effects, or package-wide validity.

## Source and environment

| Item | Value |
|---|---|
| drmTMB source | `b535bfbcce2a9039d869c2e34fe9da476394bf62` |
| DRModels.jl source | `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` |
| Host | Totoro, Linux x86_64 |
| R | 4.6.1 |
| Julia | 1.10.12 |
| Julia / BLAS threads | 1 / 1 |
| Fixture | seed 20260623; n = 1,800; 492 zeros; 1,308 positive counts |
| Command runtime | 25.92 seconds elapsed, 25.15 seconds user, 0.84 seconds system |

The R and Julia source snapshots were already present on Totoro. Their
`R/methods.R`, `src/bridge.jl`, and corresponding test files were SHA-256
checked against the exact local `origin/main` blobs before execution:

| Source file | SHA-256 |
|---|---|
| R `R/methods.R` | `ffce4b76731dc1673b731a1b9896302350326f83831c01ba071e9908c2cd1c31` |
| R hurdle bridge test | `c78ea415299ef41471e1e4f1fe5e3f5fc53694bf95a7b2407e6276f401259ce4` |
| R native hurdle test | `4bff1dc640a03f797de19a6429fa6a2712807175ae7a00d4edb4224cefb5bd1c` |
| Julia `src/bridge.jl` | `0b4ff2b509bf63a98a222181684c6356238e93f19d157548b41bacee8713b065` |
| Julia bridge test | `e5c37e84be3ae44458c700393fc3f5897bf97c9cd6c3807b3d467eeffd0ae113` |

## Results

The R simulation uses the committed hurdle-NB2 fixture definition in
`tests/testthat/test-julia-family-hurdle_nbinom2.R`, seed 20260623. The same
generated data were passed to native TMB and the live Julia bridge. The
diagnostic harness did not call `fixef()`, because the exact-main bridge test
file currently errors at that extractor before it returns its comparison
object. It reports the predicted distributions and output vectors directly.

| Quantity | Result |
|---|---:|
| R and Julia fits converged | yes / yes |
| Maximum absolute difference across `mu`, `sigma`, and `hu` predictions | `9.768574837920596e-13` |
| Native R fitted mean vs hurdle mean reconstructed from Julia dpars | `7.667200208061331e-13` |
| Julia bridge fitted mean vs its `mu` dpar | `4.440892098500626e-16` |
| Maximum absolute R vs Julia fitted-value difference | `1.241592632021761` |
| Maximum absolute response-residual difference | `1.241592632021761` |
| Maximum absolute R response-residual identity error, `resid - (y - fitted)` | `0` |
| Maximum absolute Julia response-residual identity error, `resid - (y - fitted)` | `0` |
| R log likelihood | `-2996.364475110835` |
| Julia log likelihood | `-2996.364475110861` |
| Absolute log-likelihood difference | about `2.6e-11` |

The measured result confirms that parameter and likelihood parity do not imply
response-output parity. R uses the hurdle-adjusted mean
`(1 - hu) * E[Y | Y > 0]`; the current Julia bridge returns the untruncated
NB2 component mean `mu`. Response residuals inherit the same discrepancy.
This is a verified exact-main fixed-effects output gap, and the route must stay
partial on the bridge axis.

## Test outcomes and a separate dispatcher error

- The exact-main R native file `test-hurdle-nbinom2.R` passed all 60
  assertions in 4.83 seconds. It does not exercise the Julia bridge.
- The exact-main Julia file `test_bridge_zi_marginal_mean.jl` passed 24/24
  assertions in 39.13 seconds. Its hurdle test explicitly preserves the
  component-mean behavior. The log contains three numerical-singularity
  warnings at boundary fits; the warning states that the flagged Wald SEs are
  not trustworthy.
- The exact-main R bridge file `test-julia-family-hurdle_nbinom2.R` ran three
  live bridge cases, but all three errored at `fixef()` with `no applicable
  method for 'fixef' applied to an object of class "drmTMB_julia"`. The model
  fits returned far enough to reach coefficient extraction, but the test
  assertions after extraction did not run. This is the separate open #1435
  load-order registration defect, not a hurdle likelihood discrepancy. Direct
  workflow lookup for #1435's head returned no run records; its PR-body check
  claims are not independently verified here.

The successful paired diagnostic avoids `fixef()` and does not alter package
source, tests, formula, or estimator. It therefore measures model output
behavior while retaining the public extractor failure as a separate main
regression. It is not a workaround for promoting the ordinary R bridge test
to green.

## Reproduction files and checksums

| File | SHA-256 |
|---|---|
| `paired-diagnostic.R` | `e6bf4227daa692ce02c0d8f4f2fcd86eff6eb9713cd6faa2461684f470ee7e62` |
| `run-pair.sh` | `59b12ced1ab59133f8e518ccd338039e7748a40354bedf8261e3626f00c13951` |
| `paired-diagnostic.log` | `604eed537e236e94e916d938c73ecab9e176408f7719c4d2747c0b04dae532cd` |
| `run-full-bridge-tests.R` | `f59a91aea27b274925f198223b95a5fa5fdbf827289573f77149a1104ed055e7` |
| `run-full-bridge-tests.sh` | `b2e120fd05c500e07ac9d2ec3adf703c7f89a1696bf189eba12df0db028b83ec` |
| `native-controls/run-r.sh` | `8c15888e517c2edd9d7629628972465b1bddcb652ea659e5dd9864f02ddfd6bf` |
| `native-controls/run-julia.sh` | `d880d98bb285a3eda440a8425fb9630d5d5aa732ae087813b88296e31dd20ab8` |
| original exact-main bridge test log | `3fc624d3489e172149d922f761e20f5420637006d71cca03782c28a5a5f1de05` |
| native R test log | `3ff11da22691930a1d15cf1340a3e4fc162073766441493ea110008bf9354c84` |
| Julia bridge test log | `d56bb4786f5873f17ef4704380e4003c50e1fb919fd459bfe646cac68425814d` |

The native-control outputs and their runners are under `native-controls/`.
All retained outputs were copied from Totoro after exit and checksum-verified.
No package source or test file was edited, and no CI or external service was
used for computation.
