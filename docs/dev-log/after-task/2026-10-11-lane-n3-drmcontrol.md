# After-task: `DRMControl`, `gradient` / `hessian`, prior weights (lane N3)

## 1. Goal

Implement owner decisions 8 (option a) and 9 (option b), recorded 11 Oct 2026 on
#942, #948, #994, #971 and #931: one `control = DRMControl(...)` keyword with
drmTMB's `drm_control()` names, documented `gradient(fit)` / `hessian(fit)`,
an honest `weights(fit)`, and prior weights on the fixed-effect families.

Shannon led. No subagents ran. Perspectives: Boole (front end and API), Noether
(optimiser and gradient scale), Rose (claims versus evidence), Pat (reader docs).

## 2. Implemented

- `src/control.jl`: `DRMControl`, `DRMControlReport`, the `@drm_entry` macro.
  The macro adds `control` and `weights` keywords to each public `drm` method
  and runs its body in a task-local scope. Fixed-effect fitters read the scope
  through three helpers (`_ctl_optimize`, `_ctl_se`, `_ctl_weights`). Each marks
  what it used. When the route returns, `_ctl_finish` throws an `ArgumentError`
  for any requested field that nothing consumed. This is what makes a refusal
  named rather than a silent no-op.
- 19 `drm` methods are wrapped with one prefix, `@drm_entry`; no signature was
  rewritten by hand, so the J1-J5 lanes' edits to those lines merge as one-line
  conflicts at most.
- 18 fixed-effect fitters (Gaussian, LogNormal, Gamma, Student, SkewNormal,
  Poisson plus zero-inflated and hurdle, NegBinomial2 plus zero-inflated,
  hurdle and truncated, TruncatedPoisson, Beta, BetaBinomial, Binomial,
  ZeroOneBeta, Tweedie, CumulativeLogit) call the helpers.
- `src/fit_derivatives.jl`: `gradient`, `hessian`, `control_info`.
- `src/comparison.jl`: `weights(fit)` returns the weights or `nothing`.
- `DrmFit` gains a trailing `control_info` field. A 23-argument compatibility
  constructor keeps all 131 existing `DrmFit(...)` call sites unchanged.
- `stderror` returns NaN, not Inf, for an all-NaN vcov.
- Bootstrap refits forward the seed fit's weights.
- Docs: `model-guides/convergence.md` worked example, Rosetta and capabilities rows.

## 3a. Decisions and rejected alternatives

- **A scope instead of a keyword on every fitter.** The alternative was to add
  `control` to ~130 fitter signatures across 40 files, which would collide with
  all five open J lanes. The scope costs one hidden channel, and the
  consumed-field check removes its main risk (silent ignoring).
- **Refuse rather than approximate.** Random-effect, structured, bivariate and
  REML routes refuse `maxit` / `start` / `multi_start` / `weights`. The owner
  scoped weights to the fixed-effect families; optimiser fields follow the same
  boundary so no route half-honours a control.
- **`se = false` works everywhere.** Routes with no skip path compute the
  Hessian and drop it (`control_info(fit).se === :discarded`). This saves no
  time there, but the contract is uniform and honest about the cost.
- **Weighted likelihood.** Row `i`'s log-likelihood is multiplied by `w[i]`.
  Normalising constants that count rows (`n log 2π` in Gaussian, the Jacobian
  `Σ log y` in LogNormal) are weighted too, so integer weights equal row
  duplication exactly. `nobs` still counts rows.
- **REML refuses weights.** Weights would change the restricted-likelihood
  correction; deferred to the rollout in #931.
- **`start` is on the internal scale in `coef(fit)` order.** Documented, and a
  wrong length errors by name.
- `seed` and `jitter` are DRModels.jl extensions to drmTMB's names, documented as such.
- `control_info` / `DRMControlReport` are classed experimental in the API
  stability test, because the owner's decisions named only the struct and the
  two accessors.

## 3b. Not done (tracked in #931 / follow-ups)

- Weights on random-effect, structured, bivariate, meta and REML routes.
- Weights in `predict` / residuals / `simulate` (nothing reads them there yet).
- Bridge still refuses `weights =` from drmTMB's `engine = "julia"` route.
- `maxit` / `start` / `multi_start` on random-effect routes.
- No engine benchmark: q4 engine files untouched.

## 4. Verification

See the check-log fragment for exact counts. Julia 1.10.4 (conda-forge); the
CHOLMOD 5.3 / 4.0 mismatch warning from that build was present and sparse
Cholesky was spot-checked.
