# After-task: #936 — unit-free singular / boundary decision in the vcov guard

## 1. Goal

`_vcov_from_hessian` decided "numerically singular / variance boundary" from
`min|eig(H)| / max|eig(H)|` of the unscaled Hessian against 3e-8. That ratio
moves with the units of `y` and of the predictors, so an uncentred `year` or
`y * 1e4` produced a false boundary warning, and at `y * 1e8` the pseudo-inverse
truncated real directions. Make the decision independent of units, keep
true-boundary warnings, and verify the SEs against an exact reference and
drmTMB. Shannon, with Noether (guard numerics) and Rose (claims versus
evidence). Subagents mapped and annotated the 92 guard calls and reviewed the
first design; none are running.

## 2. Implemented

- **`_vcov_guard_decision`** (`src/vcov_guard.jl`), a pure function that the
  census replay also calls. It asks two separate questions:
  - *Flat (boundary).* A coordinate the caller declares unit-free is flat when
    `|H_kk| <= _VCOV_FLAT_CURVATURE = 1e-4` in natural units, i.e. a Wald SE of
    100 or more on the log, atanh or logit scale.
  - *Non-identified.* The other coordinates, each equilibrated by its own
    curvature with no floor, are singular when `min eig / max eig <= _VCOV_RTOL
    = 3e-8`.
  - *Not positive definite.* This is judged on the floored matrix `M`
    (`_vcov_scaling`; Sylvester: same signs as `H`), anchored at 1. Noise of
    either sign on a flat coordinate is therefore not read as a saddle. An
    ordinary coordinate with non-positive curvature keeps scale 1, so a
    numerically zero diagonal stays zero. A non-finite `M` (an indefinite `H`
    with a subnormal diagonal) is not positive definite instead of throwing. A
    NaN or Inf `H` still throws (owner decision 14).
- A healthy Hessian is inverted as `inv(H)`, bit for bit as before. The fallback
  is `S pinv(M) S`, so it is equivariant under a change of units, and a
  numerically flat coordinate still gets zero variance (#969 semantics
  unchanged).
- The singular warning names its cause: a variance/dispersion boundary
  (`flat_coordinates`) or a non-identified combination
  (`null_direction_coordinates`, from the basis-invariant projector weight of
  the near-null space).
- **Call sites.** Every guard call declares `unitfree`. Two helpers support
  this. `_unitfree_columns(X, offset)` returns the 0/1 columns, which are
  intercepts and indicators. `_location_unitfree_columns(fam, Xμ)` returns those
  columns for log/logit-link location formulas only; it returns none for
  Gaussian, Student or skew-normal, where even the intercept carries `y` units.
  The declared coordinates are log-SDs, log-Cholesky diagonals, atanh
  correlations, cut-points, scalar log/logit shape parameters, and those 0/1
  columns. Never declared: identity-link location coefficients, coefficients on
  continuous predictors, raw Cholesky off-diagonals, and signed loadings.
  In total, 95 call sites in 29 files set `unitfree`. The wrappers
  `_vcov_or_nan`, `_q4_fd_vcov`, `_q4_fd_vcov_from_grad`,
  `_ordinary_laplace_optimize` and `_ordinary_laplace_vcov` forward the keyword.
  The PR body lists the files.
- **Crossed-mean nuisance route** (`src/sparse_laplace_glmm.jl`). A coordinate
  past its likelihood clamp is now zeroed before the guard, as the crossed-mean
  route already did. Its finite-difference "curvature" is inner-solve noise:
  4.7e-5 locally, above 1e-4 on CI Julia 1.10, and -1.1e-3 on another fixture.
  That noise would otherwise decide between "boundary" and "not a minimum".
- **Tools.** `tools/vcov_guard_census.jl` records every Hessian the suite sends to
  the guard, through `_VCOV_CENSUS_HOOK`, which is `nothing` in normal use.
  `tools/vcov_guard_replay.jl` replays the legacy and new decisions on those
  Hessians under any candidate constants.

## 3. Calibration

The census ran on 39e623e with 2,796 guard calls from 149 test files (Linux,
Julia 1.11.7).

- **Collinearity.** The equilibrated ratio is bimodal. The non-identified
  cluster has 11 values of exactly 0 and 7 in 4.0e-17 .. 1.5e-15, with a top of
  1.5e-10. The next point is 6.2e-6. 3e-8 sits 2.3 orders from each edge of this
  gap.
- **Flatness.** Above 3.6e-7 the natural curvatures are a continuum, so 1e-4 is
  anchored on the fixtures. The boundary fixtures reach 7.2e-5: NB2 r → ∞ at
  8.2e-6, q=2 structured log L22 at 9.2e-6 .. 1.2e-5, the binomial clamp at
  4.0e-5, and the crossed-mean degenerate grouping at 7.2e-5. The next recorded
  point is an interior q=2 REML log L22 at 1.32e-4. The margin is thin (0.14
  and 0.12 orders), and the code comment records it together with its
  falsification criterion.
- **Transitions against the legacy rule** on the same Hessians. Four calls newly
  warn, all genuine boundaries: NB2 r → ∞ (twice), a q=2 structured log L22, and
  a binomial clamp. Three calls stop warning: two separation fixtures, which
  `_separation_guard` still flags, and one uncentred-x `(1 + x | g)` AGHQ fit,
  which was the #936 artefact itself. One call moves from "singular" to "not
  positive definite": a saturated 3-observation bivariate fit whose Hessian has
  eigenvalues down to -3.4e13. The legacy rule called it singular only because
  of the 1e30 scale (#972 principle).

## 3a. Decisions and rejected alternatives

- **The issue's proposal (equilibrate every coordinate) cannot keep boundary
  warnings.** A test invariant to every diagonal rescaling cannot see a
  decoupled flat coordinate, and at a variance boundary a log-SD's whole row
  decays together. In the census, every call the legacy rule flagged was
  exactly that: eigenvector alignment 1.000, with a fully equilibrated ratio of
  0.2 .. 1. Boundary detection therefore needs the caller to say which
  coordinates have a natural scale.
- **The first design (531dc25: one eigenvalue test on a floored matrix) was
  rejected after an adversarial review.** The floor leaked into the collinearity
  test, so an uncentred covariate next to a declared intercept (`sigma ~ year`,
  `zi ~ year`, an ordinal cut-point) warned again. Leaving location columns
  undeclared also silenced separated and all-zero levels on log/logit-link
  random-effect routes. Both are now covered by red-proof tests: 6 fail and 5
  error on 120746b.
- No absolute flatness test is possible for coordinates measured in units,
  such as identity-link location coefficients or slopes on continuous
  predictors. They are judged only through collinearity.
- `mixed_family`'s cross-family ρ Wald CI declares log λ1 and the dispersion
  columns, but not the location columns. It is a CI for ρ only.
- Out of scope, noted on the PR:
  - `bivariate_student.jl:217-219` writes the start values into a copy.
  - The Poisson spatial log-range clamp is in absolute coordinate units.
  - `_fit_multi_ranef_gaussian` uses `isposdef` + `inv`, not the guard.
- #1049 (PR #1044's `_unitfree_converged`) can reuse `_vcov_scaling` or
  `_vcov_guard_decision` for its singular fallback.

## 4. Verification

- **Red-proof.**
  - `test/test_vcov_unitfree_936.jl` fails on base f71b292 at 13 of 30
    fit-level assertions: false warnings for uncentred year and `y * 1e4`, and
    SEs of 13,088 vs 9.6e6 and 2.39 vs 2.28e9 at `y * 1e8`. The guard-level
    testsets need the new API.
  - The review-driven testsets fail on 120746b (6 fail, 5 error).
  - The suite passes on the branch: 119 assertions together with
    `test/test_vcov_guard.jl`. `test/test_crossed_se_forwarding.jl` gives 21/21
    after the nuisance-route zeroing.
- **Exactness.** The SEs equal the centred or rescaled fit after the exact
  transformation, and the closed-form Gaussian ML SEs in BigFloat, to 1e-8
  relative.
- **drmTMB parity.** drmTMB 0.7.1 at ca3ddefa; only numbers are committed, in
  `test/fixtures/vcov_unitfree_936/` with `gen_drmtmb.R`. SEs agree to
  ≤ 3.3e-7 relative where drmTMB converges. On both `y * 1e4` fits drmTMB
  reports `converged = false`. On the uncentred one its SEs are 5.3e-6 off the
  closed form, while DRModels matches it to 1e-8.
- **Full suite.** The local census shards 1–3 ran on 39e623e. Shard 4 is the
  slow shard (3–4 h on CI) and was still running when this was written. PR CI
  is the full record.
- **Local-only failures, not this PR's.**
  - `test_reml_q4_missing_response.jl:210` (`fit_reml.converged`) also fails on
    base f71b292 locally with Julia 1.11.7. That route never calls the guard
    (`q4_vcov = false`).
  - The wall-time assertions in `test_lss_sparse_multi_public.jl` failed under
    a 4-shard load; that shard is green on CI Julia 1.10.

## 5. Rose

The claims are limited to what the tests, the census replay and CI show. The
flatness margin is stated as thin, not as a gap. No drmTMB source was vendored:
the fixture holds generated numbers plus our own MIT generator script. No
bridge payload shape changed. Two things are new: the singular warning carries
a `null_direction_coordinates` keyword, and its cause line differs between
boundary and non-identified.
