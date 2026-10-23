# After-Task Report: `sigma(fit)` now reports the scale the likelihood scored, on every NB2 / Gamma / Beta route

- **Date:** 2026-09-15
- **Issue:** Dinnage wave-3 audit Item 2 (M2); vault decision **D-268**; #324
  follow-up. Branch `fix/report-clamped-sigma-20260915` off DRM.jl `origin/main`
  `27fc92020`
- **Perspectives:** Hopper (R↔Julia twin), Rose (after-task)

## 1. Goal

The audit found the same defect in kind on three fixed-effects routes: the
likelihood soft-clamps the log-scale linear predictor, and then the fit reports
the **raw** `exp(Xσ θ̂)` — a scale the model never evaluated. D-268 records the
band difference from drmTMB as deliberate and asks for the reporting fix.
The R twin landed it as `drm_clamped_sigma_eta()` (drmTMB `0526baa2f`).

## 2. Implemented

Two one-line helpers in `src/negbinomial.jl` next to `_softclamp`, plus their
derivation in a comment:

```julia
_reported_sigma(ησ, lo, hi, margin) = exp.(_softclamp.(ησ, lo, hi, margin))   # soft-guarded routes
_reported_sigma(ησ, lo, hi) = exp.(clamp.(ησ, lo, hi))                        # hard-clamped routes
```

applied at **twelve** `scales = Dict(:sigma => …)` sites — not the three the audit
named. The audit named the three routes that soft-clamp; grepping each file for
`Xσ` after the fit turned up nine more of the same shape, where the likelihood
hard-`clamp`s `ησ` and the fit still reported the raw value. Bands are unchanged
(D-268): NB2 ±20, Gamma ±15, Beta ±15, margin 3 on the soft ones.

Everything downstream reads `fit.scales[:sigma]` — `sigma(fit)`
(`src/gaussian_core.jl:1356`), `simulate` (`:1762`), the quantile residuals
(`src/quantile_residuals.jl`) — so the twelve fit sites really are the choke
point on this side, and all four accessors move together.

One paragraph in `docs/src/rosetta.md` states the deliberate difference: both
twins now report the guarded scale, but drmTMB clamps every scale family at
±12 while DRM.jl keeps NB2 ±20 / Gamma ±15 / Beta ±15 and does **not** clamp the
Gaussian fixed-effects core at all.

## 3a. Decisions and rejected alternatives

- **Rejected: narrow the bands to drmTMB's ±12.** D-268 records the wider bands
  as deliberate — `size`/`shape`/`precision` `= exp(-2·eta_sigma)` means a
  near-Poisson NB2 legitimately wants `eta_sigma ≈ -15`. Documented instead.
- **Rejected: add a clamp to the Gaussian core.** D-268 again: that route is
  deliberately unclamped and self-consistent. Untouched.
- **Rejected: fix only the three soft-clamped routes.** They are three of twelve
  instances of one mistake in the three files this lane holds; the Rose principle
  says fix them all. The hard-clamped ones bite only past ±20/±15, so in practice
  the change there is an identity — which is exactly why it is cheap to make right.
- **Rejected: warn when the guard bites** (the multi-RE Gaussian route does).
  It is the right follow-up but it changes the console output of every affected
  fit, which is a separate, user-visible decision. Named in §10.
- **Kept: the raw reported MEAN.** `means[:mu]` is still `exp.(Xμ θ̂)` while the
  likelihood soft-clamps `ημ` at ±20/±30. That is the same shape of gap, but
  drmTMB does not clamp the mean at all, so the raw value is the one closer to
  the twin; and `|η_μ| > 20` is not reachable by a real count mean. Named in §10.

## 4. Files touched

`src/negbinomial.jl`, `src/gamma.jl`, `src/beta.jl`, `test/runtests.jl`,
`test/test_report_clamped_sigma.jl` (new), `docs/src/rosetta.md`, this receipt,
and its `check-log.d` row.

## 5. Checks run

aarch64 macOS, Julia 1.10.0, `JULIA_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=1`.

**Red first.** The new file on the pre-fix sources (restored from `HEAD`, then
restored forward from a byte copy — no `git checkout` in a live worktree):

```
NB2 fixed effects — sigma predictor outside the ±20 band: Test Failed at test/test_report_clamped_sigma.jl:55
  Expression: ≈(σ̂, _scored_sigma_soft(Xσ, coef(fit, :sigma), -20.0, 20.0, 3.0), rtol = 1.0e-12)
   Evaluated: [… 6.994436087074515e11 ×6] ≈ [… 9.300647771199884e9 ×6]
NB2 …:57   Expression: exp(ηraw[end]) / σ̂[end] > 10.0          Evaluated: 1.0 > 10.0
Gamma …:79 Expression: ≈(σ̂, _scored_sigma_soft(…, -15.0, 15.0, 3.0), rtol = 1.0e-12)
   Evaluated: [7.450580597421455e-9 …] ≈ [2.4248010114790695e-8 …]
Gamma …:80 Expression: σ̂[1] / exp(ηraw[1]) > 2.0               Evaluated: 1.0 > 2.0
Beta  …:101 Expression: ≈(σ̂, _scored_sigma_soft(…, -15.0, 15.0, 3.0), rtol = 1.0e-12)
   Evaluated: [2.9952460982601884e-8 …] ≈ [4.357837710963805e-8 …]
Beta  …:102 Expression: σ̂[1] / exp(ηraw[1]) > 1.2              Evaluated: 1.0 > 1.2

Test Summary:                                                          | Pass  Fail  Total   Time
reported sigma is the scale the likelihood scored (D-268, audit M2)    |   15     6     21  20.3s
  NB2 fixed effects — sigma predictor outside the ±20 band             |    2     2      4  11.4s
  Gamma fixed effects — sigma predictor outside the ±15 band           |    1     2      3   2.4s
  Beta fixed effects — sigma predictor outside the ±15 band            |    1     2      3   1.1s
  in-band fits are bit-for-bit unchanged (the guard is identity there) |    9            9   3.1s
  hard-guarded NB2 routes report the guarded scale too                 |    2            2   2.3s
```

The reported NB2 scale was **75×** the scale the likelihood scored
(`6.99e11` vs `9.30e9`), Gamma **3.3×** low, Beta **1.45×** low.

**Green after:**

```
Test Summary:                                                       | Pass  Total   Time
reported sigma is the scale the likelihood scored (D-268, audit M2) |   21     21  17.7s
```

**Every test file that fits NB2 / Gamma / Beta / TruncatedNB2** — 53 files,
found by `grep -rln "NegBinomial2()\|Gamma()\|Beta()\|TruncatedNegBinomial2()" test/*.jl`,
run in one session, ~10 min: **51 green**, plus `test_mixed_family.jl` green when
run under `--project=test` (it needs `StableRNGs`, which the package env lacks),
i.e. **52 of 53**. The one red is `test_corr_locscale_equiv.jl`, two assertions in
its **Poisson** `(1+x|g)` case — a family this change does not touch. Verified
pre-existing: the same two assertions fail identically on the unmodified
`origin/main` sources in this worktree.

The full 271-file suite was **not** run here (it is the >30-minute campaign class,
D-139); the family subset above is the measured evidence.

## 6. Tests of the tests

Making the guard bind through the public API is the whole difficulty, because a
guard only binds where the likelihood is already flat. Each fixture is built so
that the predictor leaves the band for a reason that is stable, not because an
optimiser happened to wander:

- **NB2** — six rows carry `sigma` covariate `z = 30` and `y = 0`. Zeros are flat
  in `η_σ` for NB2 (their gradient decays like `exp(-2 η_σ)`), so they do not pull
  the slope; the `z ∈ {0,1,2}` rows pin it, and `η_σ ≈ 27` follows by arithmetic.
- **Gamma / Beta** — data with a coefficient of variation of `2e-9`, i.e. whose
  own MLE `log σ ≈ -20` is outside the band by construction.
- Each testset asserts the fixture actually left the band (`maximum(ηraw) > 20`,
  `minimum(ηraw) < -15`) **before** asserting the identity, so a future
  environment where the fit stops short fails loudly instead of passing vacuously.
- The in-band control asserts `sigma(fit) == exp.(Xσ θ̂)` with `==`, not `≈`:
  the guard is exactly the identity inside the band, so ordinary fits must be
  bit-for-bit what they were.
- The NB2 loglik round-trip **passes on the old code too** and is labelled as
  such in the file. At a size of `~1e-20` the raw and the guarded scale give the
  same near-degenerate NB2 mass for `y = 0`; the row-by-row identity is what pins
  that route. Claiming it as part of the red would have been false.

## 7a. Issue ledger

Closes audit Item 2 (M2) for the NB2/Gamma/Beta routes and records the band
difference per D-268. It does **not** close M2 for the Gaussian core (deliberate,
per D-268), nor for the kernel-based Laplace routes (audit Item 2c). No release,
tag, collaborator message or remote compute. **Merge is Shinichi's.**

## 8. Consistency audit

Inside the band every touched expression is the identity, so every fit that never
approached the guard reports exactly what it reported before — asserted, with
`==`, for all three families. Outside the band the reported scale, `simulate`,
the quantile residuals and `residuals` all move together, because all four read
the one `fit.scales[:sigma]` this change writes. `src/DRM.jl` exports, the public
API, the estimands and the optimiser are untouched; no likelihood changed, so no
banked parity number moves.

## 9. What did not go smoothly

The first Gamma and Beta fixtures drove the predictor out of band by a genuine
runaway (a saturated group with identical `y`). That works, but the overshoot is
whatever the optimiser's last step happened to be — 5.5 units past the bound on
one seed, 0.0 on another — so the test would have been a coin flip. Replaced with
fixtures whose data have an out-of-band dispersion by construction.

The Gamma/Beta loglik round-trip had to be dropped **in the out-of-band testsets**
and moved to the in-band ones. Where the guard binds, the precision is `σ⁻² ≈ 1e15`,
and recovering it by squaring a reported `σ` loses the last bits of an exponent the
density is extremely sensitive to: one ulp in `φ` moved the Beta log-likelihood by
**317 nats** at `n = 150` (`5.265721604115561e14` vs `5.26572160411556e14`). That
is a real property of the regime, not a defect introduced here, and the test file
says so at the point where the assertion is missing.

## 10. Known residuals

- **No warning at the boundary.** `src/gaussian_ranef.jl` warns when a fitted
  `η_σ` hits its clamp; these twelve routes do not. A user whose fit lands there
  gets an honest number with no flag that the Wald SEs beside it are meaningless.
- **`predict` / `predict_parameters` on `newdata` is NOT covered.**
  `_param_response` / `_link_deriv` (`src/gaussian_core.jl:1470-1520`) map
  `η → σ` with a bare `exp.(η)` for every family, so a `sigma` prediction at a new
  extreme covariate value is still the unguarded value. drmTMB's M2 fix *did*
  cover that path. Fixing it properly needs a per-family band table (the analogue
  of `drm_clamped_scale_families()`), and `gaussian_core.jl` is outside this
  lane's lease with four lanes live.
- **The kernel-based Laplace routes** (`src/locscale_kernels.jl`,
  `src/mixed_family.jl`, `src/locscale_sigma.jl:109`) have the same gap past
  their hard `clamp(ψ, ±15/±30)`. Audit Item 2c. Not touched.
- **The reported MEAN** is still raw (see §3a).
- **Stale comment, not fixed:** `src/quantile_residuals.jl:35-36` says NB2 stores
  the size in the sigma slot "directly, NOT σ⁻²", while the code three lines
  below (`:69`) correctly computes `1 / σ²`. The comment is wrong; left alone
  because it is outside this fix.
- **Stale reference, not fixed:** `_softclamp`'s comment cites
  `docs/design/03-likelihoods.md`, which does not exist in the repo. The new
  comment points at `docs/src/rosetta.md` instead.
- Measured on one platform (aarch64 macOS, Julia 1.10.0) and on the 53-file
  family subset, not the full suite.

## 11. Team learning

"One choke point" is a hypothesis, not a finding. The audit named three sites; the
same file held six, and the three files held twelve. Grep the *shape* of the
mistake (`exp.(Xσ * θ̂…)`), not the line numbers you were handed.

And a guard only binds where the likelihood is flat — which is why a
clamp-reporting bug is invisible to ordinary tests, and why a test for one has to
build the out-of-band state by construction rather than wait for an optimiser to
wander there.

## 12. Cross-product coverage

Covers the post-fit reported scale for NB2 (fixed / ranef / correlated-ranef /
zi / hurdle / truncated), Gamma (fixed / ranef / correlated-ranef) and Beta
(fixed / ranef / correlated-ranef), on one platform, at the fit-time reporting
site. Does not cover `predict`-on-`newdata`, the Laplace kernel routes, the
Gaussian core (deliberately), the R bridge, the mean predictor, boundary
warnings, performance, R-side parity fixtures, DRAC/Totoro, or release gates.
