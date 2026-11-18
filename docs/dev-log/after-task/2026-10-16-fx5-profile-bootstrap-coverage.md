# After-task: FX5 t9 profile / bootstrap coverage (#1117, #1119, drmTMB #1562)

Shannon, with Noether (profile contract) and Rose (claim vs evidence). No
subagents.

## 1. Goal

Lane t9 (receipt `sweep-oct16/t9` on drmTMB, pins drmTMB `ca3ddefa` /
DRModels.jl `f71b292`) found three coverage defects. This slice implements the
one prescribed code fix and reports the two methods questions.

## 2. Implemented

**#1117 (code).** Ordinary Gaussian location-scale `(1 | g)` profile CIs for
`sd_g` at the numerical SD floor must never throw and must never return a
successful `(-Inf, Inf)`.

Two mechanisms, both on the generic profiler (`src/inference.jl`):

1. **Lower-arm boundary** (drmTMB `allow_lower_boundary` for `exp`
   targets). A `:resd` / `:resd_mu` / `:resd_sigma` lower arm that reaches
   `log(sqrt(eps))` ≈ −18.4 without an LR crossing, or whose nuisance solve
   fails at that floor, is a **boundary**, not a failed endpoint. The
   user-facing row is rewritten to the SD scale: `[0, exp(u)]`. `confint`
   therefore does not throw (`failed == 0`) and does not print `(-Inf, Inf)`.
2. **Usable profile step.** A collapsed fit's pseudo-inverse SE can be
   ~1e-19. The old `se > 0` test accepted that, so 40 expansions of ×1.6
   never left the floor (seed `20261016902`). Steps below `1e-6` now fall
   back to `1.0` on the working scale.

Location-only `:resd` rows use the same `[0, u]` rewrite (sibling path).
Interior `:resd` rows stay on `log σ_b` (existing `test_profile_ci.jl`
contract).

**#1119 (options only; not implemented).** Percentile bootstrap under-covers
the sigma intercept 83–90% in all four t9 cells, one-sided from above. This
is the Dinnage S6 / drmTMB #1315 mechanism (percentile interval built from
an already-shrunk ML estimate). Changing the default interval type is a
methods choice. See §3a.

**drmTMB #1562 (not a construction bug).** See the twin investigation on the
drmTMB branch. Sigma-slope profile coverage 90.3% at G=30, `sd_g=0.5` equals
Wald on the same 300 fits (271/300). Julia on the same DGP covers 93.3%
(pass bar). `sd_g` overcover at true 0.1 is the expected `[0, u]` Self–Liang
boundary (287/300 and 282/300 `profile.boundary`).

## 3a. Decisions and rejected alternatives

- **Do not change the working-scale contract for interior SDs.** Only a
  boundary lower arm is rewritten to `[0, u]`. Converting every `:resd`
  profile to the SD scale would break existing tests and Wald/profile
  comparisons on healthy fits.
- **Do not return `[0, Inf]` as the happy path.** After the step-size fix
  both t9 seeds have a finite upper. `[0, Inf]` remains only if the upper
  arm is genuinely unbounded; that is not `(-Inf, Inf)`.
- **#1119 options (STOP; do not pick a default):**
  1. Keep the percentile default; document the bias-doubling; recommend
     Wald/profile for scale intercepts at small n. This is the owner's
     11 Oct 2026 decision on drmTMB #1315 (option a).
  2. Opt-in **basic / reflected** interval `2θ̂ − q*` (centres near the
     truth when the ML is biased).
  3. Opt-in **BCa**.
  4. Change the **default** to basic or BCa — a methods choice, not taken.
  5. Studentized bootstrap — more refits, not proposed here.

## 3b. Evidence (red-proof)

Before the fix, `test/test_resd_profile_boundary.jl` on pin `f71b292`:

| seed | `profile_result` | `confint` |
| --- | --- | --- |
| 20261016900 | `(-Inf, −0.630]`, `failed=1`, `not_converged` | throws `ArgumentError` |
| 20261016902 | `(-Inf, Inf)`, `failed=0` | returns `(-Inf, Inf)` |

After:

| seed | interval (SD scale) | covers 0.1 |
| --- | --- | --- |
| 20261016900 | `[0, 0.533]`, estimate 0.200 | yes |
| 20261016902 | `[0, 0.229]`, estimate 0 | yes |

Sibling tests (generic profile, locscale status, loconly #1023, bridge
status, crossed Gaussian/Poisson `:resd`) all passed after the rewrite.

## 4. Files touched

`src/inference.jl`, `test/test_resd_profile_boundary.jl`,
`docs/src/diagnostics-and-validation/profile-likelihood.md`, this receipt,
`docs/dev-log/check-log.d/2026-10-16-1117-resd-profile-boundary.md`,
`news/1117-resd-profile-boundary.md`.

## 5. Checks run

- Red-proof: FAIL on the unfixed tree, PASS (25/25) after; re-confirmed
  25/25 in 30.7 s on head `0bd19b879`.
- Sibling profile battery: `test_profile_ci.jl`,
  `test_profile_nuisance_status.jl`, `test_profile_acceptance_oracles.jl`,
  `test_sentinel_generic_profile.jl`, `test_j2_convergence_vcov.jl`,
  `test_bridge_profile_status.jl`, `test_locscale_profile_status.jl`.
- Full unsharded `Pkg.test()`: 179 `@testset`s passed, then stalled on
  pre-existing `test_crossed_tweedie.jl` (nested ForwardDiff + Brent
  power-p profile; not on the generic `confint` path this slice edits)
  for >50 min at 99.9% CPU with no further log. Not treated as a
  regression of #1117. CI run ids: `38983946104` (test shards /
  `changes`), `38983946098` (docs).

## 6. License / Rose

No drmTMB GPL source was vendored. The `[0, u]` rule is the published
drmTMB behaviour (generated-output parity), implemented in fresh Julia.
No speed or coverage *rate* is claimed beyond the two reproduced seeds and
the t9 receipt numbers cited as the finding, not as a new measurement.

## 7. Issue ledger

- Closes #1117 — implemented and red-proofed.
- Refs #1119 — options only; twin of itchyshin/drmTMB#1315.
- Refs itchyshin/drmTMB#1562 — investigation, not a profile-construction
  bug. drmTMB push was 403 (`Permission to itchyshin/drmTMB.git denied to
  cursor[bot]`). Branch `cursor/fx5-t9-slope-coverage-e28d` @
  `5fea1f634073ae162bbbe78ae5e1a053fb77b1ac` is local only. Apply
  `docs/dev-log/evidence/2026-10-16-drmtmb-1562.patch` on that pin.
