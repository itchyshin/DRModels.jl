# After-task — parity vcov/SE default bars (#1503)

## Slice

Always-on Workflow G guard: `test/test_parity_vcov_se_defaults.jl` refits every
in-cohort `test/parity/fixtures/*` cell that stores `[vcov]`, strips fixture
`[tol]` overrides for `rtol_vcov` / `atol_vcov` / `rtol_se` / `atol_se` only, and
asserts vcov / SE failures from `compare_fit` / `compare_bridge` are empty at
`compare.jl` keyword defaults.

## Evidence

Julia 1.13.4, `julia --project -e 'using Test; include("test/test_parity_vcov_se_defaults.jl")'`:
29 pass, 2 broken (`meta-analysis-V`, `robust-student` vcov only); SE on those
cells passes at defaults.

## Mismatches (documented, not widened)

| Cell | Quantity | |Δ| (default bar) |
|---|---|---|
| meta-analysis-V | `vcov[mu_(Intercept),sigma_(Intercept)]` (+ symmetric) | ≈1.46e-8 |
| robust-student | `vcov[mu_x,nu_(Intercept)]` (+ symmetric) | ≈2.04e-8 |

Fixture `[tol] atol_vcov = 3e-8` lines unchanged for gated `DRM_PARITY_TESTS=1`.

## Rose

- No src/ engine edits; no tolerance loosening; no GPL boundary touch.
- Claim scope: static-fixture vcov/SE agreement at documented defaults only.

refs #1503