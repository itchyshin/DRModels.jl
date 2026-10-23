# DRModels.jl tests

## CI RNG policy (#388)

Julia’s default seeded RNG (`Random.seed!` / `MersenneTwister`) is **not stable
across minor versions**. A test that (1) draws its dataset from that stream and
(2) asserts recovery against known truth with `atol`/`rtol` can pass on CI’s
`1.10` leg and fail on `1` (≈1.12), or the reverse — the estimator is fine; the
two legs never saw the same data. Measured on this repo (2026-08-07): with
`Random.seed!(7)`, `randn(50_000)` streams agree for 8188 draws then diverge
(`docs/dev-log/evidence/2026-08-07-388-rng-probe.md`). Twin: HSquared.jl
2026-08-04 CI RNG fix; decision “CI stays RNG-free” for recovery claims.

**Preference order for new or repaired recovery tests** (do **not** add
`StableRNGs` to the main `Project.toml`):

1. **Keep stochastic recovery out of default CI** — env-gate
   (`DRM_SLOW_TESTS=1`, a dedicated `sim/` driver, or similar); CI keeps
   shapes, identities, guards, and cross-estimator same-draw checks.
2. **Literal deterministic fixtures** — explicit arrays / committed CSV
   (parity fixtures under `test/parity/` are the model).
3. **`StableRNGs` in the test env only** — already listed in
   `test/Project.toml`; use for a residual case that must stay in CI and
   cannot be fixtured. Example: `test_mixed_family.jl`.

Safer patterns already in the suite: native ≈ bridge on one draw
(`test_bridge.jl`); sparse ≈ dense; public ≈ direct locscale; Workflow G
parity fixtures behind `DRM_PARITY_TESTS=1`.

Standalone `julia --project=. test/….jl` does **not** load `test/Project.toml`.
For tests that `using StableRNGs`, run under `Pkg.test()` or
`julia --project=test` after `Pkg.develop(path=".")`.
