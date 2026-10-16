# After-task: lane n3 — `predict` sets random effects to zero; name the default marginal (#1092, #1070)

## 1. Goal

Docs-only fix lane at base `f71b292` (drmTMB pin `ca3ddef`), PR #1100.

- **#1092.** The `predict` docstring said random effects were "integrated out", but
  `type = :response` returns `g⁻¹(Xβ̂)` with every random effect at zero. Reword
  it as "set to zero (prediction at the average group)", say how to get the
  marginal mean, and add one line on what drmTMB's `predict()` returns. Acceptance:
  a test asserting `predict(:response) == invlink(Xβ̂)` at RE = 0.
- **#1070.** On the drmTMB-parity and Getting-started pages, name the default
  marginal approximation, say that `marginal = :Laplace` matches drmTMB, that
  LRT/AIC are valid only within one approximation, and that `:Laplace` is for
  parity rather than a better default (plus, from the issue thread, which models
  accept `:Laplace`). Grep the other pages for "integrated out" and "matches
  drmTMB" and fix each wrong hit.

## 2. Implemented

- `src/gaussian_core.jl` docstrings: `predict` (RE = 0, the average group; grouping
  columns ignored; scoped in-sample identity; log-link marginal-mean recipe
  `predict .* exp(re_sd(fit)[:g]^2 / 2)`; logit attenuation; drmTMB's `predict`
  with and without `newdata`; the `se.fit` parity claim corrected, since drmTMB's
  `predict()` has no `se.fit`; SEs carry `β̂` uncertainty only). `fitted`
  (population-level, b = 0, with the exceptions: the Gaussian `phylo` mean routes
  store `Xβ̂ + Sû`, plus ZeroOneBeta, CumulativeLogit and LogNormal). `predict_parameters` and
  `marginal_parameters` ("integrated out" removed; in-sample identity scoped). `re_sd`
  (the mean-axis SD is on the link scale).
- `src/poisson.jl`, `src/beta.jl`, `src/gamma.jl`, `src/negbinomial.jl`: rendered
  docstrings no longer call the ordinary `(1 | g)` default "GHQ-32". `src/temporal.jl`:
  "every structured term" scoped.
- `docs/src/drmtmb-parity.md`: new section "Default marginal approximation" (per-route
  defaults; the gap to drmTMB with committed evidence; `:Laplace` parity, the accepted
  models and how others refuse; parity-not-better, scoped to the adaptive cells with
  the Gaussian `sigma` exception; the like-with-like rule and what `lrtest`/`aic` do
  and do not check). Footer records provenance.
- `docs/src/getting-started.md`: new section "Random effects and the default
  approximation"; `marginal_parameters` no longer called "population-averaged";
  predictions described as RE = 0.
- Grep-hit fixes and consistency: `model-guides/model-workflow.md`, `families.md`
  (Tweedie, CumulativeLogit), `model-guides/marginal-la-vs-va.md` (stale GHQ-32 default,
  historical measurements labelled as pre-0.7.2), `capabilities.md` (default-integrator
  row; small-sigma advice; new Student `:Laplace` row), `rosetta.md`,
  `diagnostics-and-validation/prediction-and-postfit.md`,
  `tutorials/temporal-random-effects.md`.
- Test: `test/test_predict_random_effects.jl` (auto-discovered). Poisson and
  Bernoulli `(1 | g)`: `predict(:link) == Xβ̂`, `predict(:response) == g⁻¹(Xβ̂)`,
  fitted / unseen / absent group levels give identical predictions, in-sample
  `predict ≈ fitted`, the log-link recipe equals a QuadGK marginal mean, and the logit
  marginal mean is strictly closer to 1/2. `test/test_predict.jl` header comment fixed.
- `news/1092-1070-predict-marginal-docs.md`.

## 3a. Decisions and rejected alternatives

- **"Adaptive GHQ (32 points)" was not written.** The brief and #1070 say the default
  is adaptive GHQ with 32 points; at `f71b292` no route does that. The ordinary
  non-Gaussian `(1 | g)` default is per-group adaptive GHQ with 5 nodes
  (`src/adaptive_ghq.jl:84`, #719, v0.7.2; Tweedie 21, CumulativeLogit 41, LogNormal 1
  and exact), and the fixed 32-node grid survives only on Gaussian `sigma ~ (1 | g)`,
  ZeroOneBeta `(1 | g)` and the cross-family latent. The docs state the code.
- **The gap figure.** #1070's Poisson comparison (0.0014 → 5.6e-9) has no committed
  evidence and is the low end of what the repo has measured: the class-1 Laplace parity
  receipt (2026-09-30, after #719) shows 0.01–0.24 log-likelihood units on seven
  Poisson/Beta/NB2/Gamma/Student cells, with the default within about 5e-4 of a
  61-node reference and drmTMB the further. The pages lead with the receipt and give
  0.0014 as one example, attributed by date and commit (published pages may not cite
  `dev-log/` paths or "issue #N"). The first push led with 0.0014; the audit caught it.
- **`:Laplace` "for parity, not a better default" is scoped** to the adaptive cells:
  on Gaussian `sigma ~ (1 | g)` the default is the fixed grid and `capabilities.md`
  already (correctly) recommends `:Laplace` for large groups.
- **Code-side defects found but not fixed (docs-only lane)**, listed in the PR as
  follow-ups: `offset()` + `(1 | g)` + `marginal = :Laplace` estimates the offset
  coefficient instead of refusing or fixing it at 1 (`src/ordinary_laplace.jl`);
  `predict(:response)` returns η for ZeroOneBeta and CumulativeLogit
  (`_mean_response`); `predict_parameters` gives Student `nu` as `exp(η)` (stored
  `2 + exp(η)`) and SkewNormal `nu` as `exp(slant)`; BetaBinomial silently ignores
  `marginal = :VA`; `ArgumentError` texts (`src/variational.jl`, `src/comparison.jl`,
  `src/poisson.jl`) still call the default "GHQ-32"; `fit.marginal` tags the option,
  not the integrator, so `lrtest` cannot catch two `:LA` fits on different routes.
- A doctest was rejected in favour of a test file: no local Julia, and a doctest
  must match printed output exactly.

## 4. Verification

- Julia cannot be installed in the authoring container (the proxy blocks the Julia
  download and package servers), so `Pkg.test()` and the Documenter build ran only in
  the PR's CI. The test was written against existing idioms and traced line by line
  against `src/` by an independent reviewer.
- Local: `python3 tools/tests/test_reader_surface_audit.py` (17 OK);
  `python3 tools/reader_surface_audit.py --public-only` (PASSED, 46 files);
  `git diff --check`; the rendered-audit regexes applied to all added text, including
  docstrings (0 hits); no `$` or `\` in added docstring lines.

## 5. Rose audit (claim vs evidence, scope, drift, license)

Two adversarial passes; see the PR for the full list.

- Round 1: five reviewers and one skeptic per finding (44 agents). 38 of 39 findings
  survived. Fixed in round 2: the gap figure (above); the unscoped parity-not-better
  bullet; the in-sample `predict ≈ fitted` and `predict_parameters ≈ marginal_parameters`
  identities and the drmTMB like-with-like line, now scoped to the GLM-type families with the
  exceptions named; `aic`/`bic`/`aicc` do refuse `:VA`; `fit.marginal` exception
  (Gaussian mean + `sigma` `(1 | g)` default is tagged `:Laplace`); Gaussian `sigma`
  covariates allowed under `:Laplace`; LogNormal node count; missing Student `:Laplace`
  row; the prediction page's `se.fit` claim; temporal "every structured term"; `re_sd`
  scale wording; logit "closer to 1/2" (now "strictly, whenever Xβ̂ ≠ 0"); test
  comment; NEWS fragment overstatements.
- Round 2 (targeted re-check of the fixes): pending at this commit; recorded in the next one.
- License: drmTMB behaviour is described from `R/methods.R` / `man/*.Rd` at `ca3ddef`
  in our own words; no GPL text copied.
