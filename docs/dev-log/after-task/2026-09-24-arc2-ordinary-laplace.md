# After-task: Arc 2 — `marginal = :Laplace` on an ordinary `(1 | g)` (2026-09-24)

## Scope

drmTMB's `engine = "julia"` refuses non-Gaussian ordinary random intercepts
(gate `nongaussian_mu_ordinary_random_effect`) because DRModels' default `:LA`
route integrates `(1 | g)` by 32-node Gauss–Hermite quadrature while native
drmTMB (TMB) uses the Laplace approximation: same model, different integrator,
different log-likelihood. This slice adds an opt-in `marginal = :Laplace` that
fits the TMB objective for one ordinary `(1 | g)` on the mean of Poisson,
Binomial, NegBinomial2, Gamma and Beta (`sigma ~ 1` for the scale families),
leaving `:LA` untouched.

## Outcome

- `src/ordinary_laplace.jl` (new): front end, shape validation, and the three
  fitters (Poisson; Binomial; NB2/Gamma/Beta through one scale-family fitter).
  It calls the existing sparse-Laplace kernels of the structured routes with
  Q = I and the group index as the latent map, and adds one likelihood kernel
  of its own: `Val(:nb2_raw)`, an NB2 log-density that stays accurate at a
  huge size 1/σ² (Stirling tail `_nb2_raw_tail`, ratios through `log1p`).
  A scale-family fit that ends with log σ < −8 (the flat Poisson-limit
  plateau for NB2) is re-fitted from log σ = −1, keeping the lower objective.
- `src/sparse_laplace_glmm.jl`: a `raw_scales = false` keyword on three kernels
  plus the Gamma and Beta setups (`_nb2_laplace_setup` is unchanged; the route
  builds its own NB2 setup). With `true` the RE log-SD and the dispersion are not
  clamped, so value and analytic gradient describe TMB's function everywhere
  (the clamp-with-unclamped-gradient defect found on the earlier #071 branch).
  Default `false`: structured routes unchanged.
- Family `drm` methods (`poisson.jl`, `binomial.jl`, `negbinomial.jl`,
  `gamma.jl`, `beta.jl`): one three-line delegation, placed before any routing.
  `negbinomial.jl`, `gamma.jl`, `beta.jl` are also touched by open PR #770; the
  insertion is in the `drm` front end, away from #770's hunks (fitter bodies).
  `binomial.jl` also moves its response parse into `_binomial_response` so both
  routes share it (behaviour identical).
- `src/variational.jl`: `:Laplace` named in the selector messages;
  `method = :Laplace` points at `marginal`.
- `src/bridge.jl`: `marginal` bridge option forwarded to `drm`; every bridge
  result now carries `"marginal"` (the integrator the fit used).
- `src/comparison.jl`: `lrtest`/`anova` accept a random-effect-free fit against
  any non-VA fit (its log-likelihood is exact). This also admits fixed vs
  `marginal = :AGHQ`, which `origin/main` refused; `:LA` vs `:Laplace`
  random-effect pairs and any VA fit are still refused.
- Covered: Poisson, NB2, Binomial (Bernoulli and `cbind` trials), Gamma, Beta.
  Not covered (refused, never rerouted): `(1 + x | g)` — the kernels map each
  observation to one scalar latent with one σ, so a correlated 2-D per-group
  effect needs a new kernel; `(0 + x | g)`; crossed/multiple terms;
  `sigma ~ covariates`; RE on `sigma`; coupled location–scale; `zi`/`hu`;
  structured markers; REML.

## Evidence

`docs/dev-log/evidence/arc2-ordinary-laplace/`: ten fixtures (five families ×
two seeds), native drmTMB vs `:Laplace`: df equal, |ΔlogLik| ≤ 3.8e-10, max
relative estimate difference ≤ 6.1e-8. The `:LA` gap on the same data is 0.057
to 1.34 log-likelihood units.

## Rose

Claim is point estimates and log-likelihood on the ten fixtures plus five
small-sigma fixtures (receipt, "Small family σ"); SEs, intervals, boundary fits
and NB2 data whose optimum is exactly Poisson (σ → 0) are not claimed. Review
round 3 fixed two route defects found at small sigma: the NB2 kernel lost all
precision at size r > e^20 (now a separate large-size-stable kernel,
`Val(:nb2_raw)`), and Gamma σ = 0.003 reported `converged = true` 7.2e-5 from
native (now inner tolerance 1e-13 plus a Newton polish). Review round 4 found
an NB2 cell (σ = 0.05, x and z, 28 unbalanced groups) whose first fit stopped
on the Poisson-limit plateau (log σ −19.03, `converged = true`, logLik 0.81
below native); the plateau guard now reaches native's optimum (|ΔlogLik|
1.1e-11), and the cell is a test fixture. The R
bridge refusal stays until the conductor lifts it. `:LA` answers are guarded by
tests, not merely asserted.

## Checks run

- `test/test_ordinary_laplace.jl` (new): relationship tests against an
  independent per-group Laplace, `relmat(1 | g)` with K = I, one-point AGHQ,
  `:LA` guards, kernel `raw_scales` guard, refusals, bridge passthrough.
- Neighbouring files (Poisson/NB2/Binomial/Gamma/Beta default, RE, phylo and
  relmat Laplace, VA, AGHQ, Cox–Reid, missing response, bridge): see the PR body.

## Follow-ups

- drmTMB bridge: admit the covered shape, send `marginal = "Laplace"`, supply
  `coef_labels` for dpar `resd`, check the returned `marginal`.
- `(1 + x | g)` by Laplace needs a 2-D per-group kernel (not in this slice).
- `sigma ~ covariates` could reuse the heteroscedastic kernel with a Q-generic
  front end and unclamped per-observation scales.
