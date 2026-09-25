# DRModels.jl — NEWS

All notable changes are recorded here. The live work ledger is
[GitHub Issues](https://github.com/itchyshin/DRModels.jl/issues); this file is the
human-readable changelog and mirrors `docs/src/changelog.md`.

## Development

- **A structured marker plus an ordinary random effect no longer drops the ordinary
  term.** `drm(bf(y ~ x + phylo(1 | sp) + (1 | h), sigma ~ 1), Gaussian(); …)` (and
  the same shape with `relmat`/`animal`, or with `(0 + x | h)` or several bars) used
  to reach the single-structured fitter, which fitted the marker alone and silently
  discarded every `(1 | h)` term. It now fits drmTMB's model: independent blocks,
  `V = D + Σ_k σ_k² Z_k K_k Z_kᵀ`, ML, reported in one `:resd` block (ordinary bars
  first, the marker last; an ordinary intercept that shares the marker's grouping is
  keyed `<g>_iid`). Nine fixtures match drmTMB `engine = "tmb"` to |ΔlogLik| ≤ 1.4e-10
  (`docs/dev-log/evidence/arc2-structured-ordinary-bar/`). REML, `(1 + x | h)`,
  range-estimated `spatial()`, `meta_V()`, `penalty` and sparse algorithms with this
  shape now raise an `ArgumentError` instead of dropping a term.
- **Package renamed to DRModels.jl.** The Julia package and module are now
  `DRModels`, while the modelling API remains `drm()`, `bf()`, and the existing
  fit/post-fit surface. `DRModels.DRM` is a soft-deprecated qualified alias for
  migration only; `using DRM` cannot survive a Julia package rename. The
  canonical `using DRModels` import is intentionally quiet; this changelog and
  the migration documentation carry the deprecation notice instead of a
  package-startup warning. The UUID and package version remain unchanged.
  The GitHub repository rename is complete; the stable Pages deployment awaits
  this unmerged PR landing. Historical
  `docs/dev-log/` records intentionally retain their original spelling.

## v0.7.1 — 2026-09-05

- **Bootstrap replicates keep a masked fit's response mask (drmTMB #1188).** `_bootstrap_data`
  merged each simulated draw into the original table wholesale, so on the missing-response routes
  every replicate refitted on the FULL design regardless of how many rows the seed fit observed.
  (The draw itself must span the full design -- `means`/`scales` are rebuilt over all rows by #646 --
  so the mask has to be re-imposed one step later, when the replicate table is built.) A bootstrap
  whose replicates are richer than the original understates uncertainty, and the shortfall grows with
  the missing fraction. `_bootstrap_data` now restores the seed fit's response mask, reading it with
  `_is_response_missing` so `missing` and `NaN` are treated exactly as the fit itself treated them; a
  `cbind(successes, failures)` response is masked where EITHER cell is absent, and the failure column
  inherits the mask through NaN arithmetic. MEASURED end to end through the drmTMB bridge
  (`engine = "julia"`, `y = 0.3 + 0.5x + N(0,1) exp(0.1x)`, `n = 60`, `bf(y ~ x, sigma ~ x)`,
  Gaussian, 30 of 60 responses masked, `B = 99`, 99/99 replicates converged in both arms): the
  bootstrap interval on `fixef:mu:x` was 0.4748 wide against a Wald width of 0.8153 before the fix
  (ratio 0.58) and 0.8754 after (ratio 1.07). MEASURED on the R side of the same defect, which is
  where coverage is affordable (`S = 200` datasets, `B = 99` replicates each, nominal 95% percentile
  interval, truth 0.5, same datasets and seeds in both arms): coverage 0.895 / 0.820 / 0.720 before
  at 10% / 30% / 50% masked and 0.910 / 0.895 / 0.910 after, against a Wald reference of
  0.920 / 0.925 / 0.915; Monte Carlo standard errors 0.020-0.032, and the paired improvement is
  significant at 30% (p = 2.75e-04) and 50% (p = 7.28e-12) but NOT at 10% (p = 0.375). Receipt:
  drmTMB `docs/dev-log/evidence/julia-r-parity/p2-g3/1188-bootstrap-mask-receipt.md`.
  KNOWN CONSEQUENCE: replicates now legitimately drop rows, so `drm`'s #258 "never silent" warning
  fires once per replicate. It was left in place deliberately -- the same channel carries the
  saturated-fit (`residual dof 0`) warning, which is exactly what a user needs when replicates on a
  heavily masked fit degenerate. `test/test_bridge_response_mask_inference.jl`.
- **The `zi`/`hu` count mixtures now have a test at the `drm_bridge` MARSHALLING
  boundary, and a family spelled `zi_poisson` gets an actionable refusal.**
  `test_zi.jl` and `test_hurdle.jl` covered these mixtures through native `drm`
  only; nothing covered the `drm_bridge` route, which is the boundary drmTMB's
  `engine = "julia"` actually crosses and the one its banked `zi_poisson` parity
  receipt rests on. New `test/test_bridge_zi.jl` (45 assertions) pins
  bridge-vs-native agreement for ZIP (`family = "poisson"` + a keyed `zi` entry)
  and ZINB (`family = "nbinom2"` + `sigma` + `zi`): coefficients agree to
  `< 1e-8` and log-likelihoods to `1e-8` on both, with a plain-Poisson contrast
  proving the `zi` entry is not silently dropped in marshalling. Separately,
  `_bridge_family("zi_poisson")` (or `zi_nbinom2`, `zero_inflated_poisson`,
  `hurdle_nbinom2`) previously threw a bare ``unsupported family `zi_poisson` ``;
  it now names the spelling that works -- `family = "poisson"` plus a `zi` entry
  in `formula`. It is deliberately NOT an alias: mapping `zi_poisson` to
  `Poisson()` would fit a PLAIN Poisson, silently and without error, whenever
  the caller omitted the `zi ~` part. A mixture prefix on a non-count family
  (`zi_gaussian`, `hurdle_beta`) and any other unknown tag still get the plain
  message, with no suggestion.
- **The #574 resolvable-scale guard had zero margin, so the #461 runaway degenerate optimum was
  still reachable (`src/location_only.jl`).** #574 refused the sparse phylogenetic location-only
  Woodbury objective where the residual variance is "below machine precision relative to the
  phylogenetic variance", coded as `sigma^2 / sigma_phy^2 >= eps`. The relative error of that
  Woodbury subtraction grows as `eps * (sigma_phy^2 / sigma^2)`, so `eps` is exactly the point at
  which NO digits survive: the bar sat *on* the cliff rather than back from it, and the objective
  was still wrong just above it. Measured 2026-09-05 against a dense `V = sigma^2 I + sigma_phy^2 C`
  oracle on the #461 fixture (G = 100, one row per species) at `log(sigma_phy) = -26.9176`: the
  Woodbury log-likelihood tracks the oracle to 1.37e-05 relative at `log(sigma) = -40.0`, then
  FLIPS SIGN at `log(sigma) = -44.0` (`+4.37745e+25` against the oracle's `-3.15149e+25`). That
  break is at `log(sigma) - log(sigma_phy) = -17.086`, INSIDE #574's bar of `-18.022`, and it is
  the whole #461 runaway: `Optim` reports convergence there, `sigma = 7.75e-20`,
  `sd_phylo = 2.04e-12`, `loglik = +4.38e+25`. Whether the optimiser walks into that sliver is
  decided by rounding, so `test/test_bootstrap_marginal.jl`'s `#461` testset passed on x86_64
  Linux CI and failed on aarch64 macOS at the same commit with the same seeds and B = 60
  (`res.failed == 0` evaluated `1 == 0`; `res.used == 60` evaluated `59 == 60` -- replicate 38).
  The bar is now the standard rule for a cancellation-limited difference: keep at least HALF the
  mantissa, `sigma^2 / sigma_phy^2 >= sqrt(eps)`, i.e. `log(sigma) - log(sigma_phy) >=
  0.25 * log(eps)` = `-9.011`. It refuses `sigma / sigma_phy < 1.2e-04`; on the #461 fixture the
  60 bootstrap refits span `[-1.033, +1.439]`, about eight nats clear of it. This is a
  floating-point representation limit, not a statistical lower bound; `algorithm = :gls` remains
  available for a genuine boundary estimate. New path-independent pin in
  `test/test_lss_phylo.jl`: every `(log sigma, log sigma_phy)` pair the guard ADMITS must agree
  with the dense oracle to 1e-06 relative -- the property, not the constant, so re-loosening the
  bar fails it (17 assertions and 1 error under #574's value).

- **REML for the Gaussian mean-only phylogenetic cell (#624 item (c)).** `drm(bf(y ~ x +
  phylo(1 | species), sigma ~ 1), Gaussian(); tree, method = :REML)` now FITS. It used to reach
  the generic univariate REML gate and throw, even though the sparse location-only spine already
  carried a validated restricted objective. The objective is the same Patterson–Thompson
  restriction the `sd()` / Woodbury routes use, `nll_REML = nll_ML + 0.5·logdet(Xμ′V⁻¹Xμ) −
  0.5·pμ·log(2π)`, evaluated by `_loconly_reml_components`; because β_μ is profiled out exactly
  by GLS at every variance point, the restriction is exact rather than approximate. The
  integrated-out set is {u_phylo, β_μ} with the same additive `+0.5·pμ·log(2π)` that drmTMB gets
  from TMB's Laplace fold of `beta_mu` into `random=`, so the two engines' REML log-likelihoods
  are on ONE convention with no offset to remove. `fit.estim_method` is `:REML`, `reml_loglik` /
  `ml_loglik` are both populated, and the variance-component SEs use the restricted curvature.
  Measured against native drmTMB REML on the drmTMB fixture (n = 90, 30 tips): logLik
  −76.000977125105 vs −76.000977125761 (6.56e-10), coefficients to 3.71e-08, phylogenetic SD
  0.480176374690 vs 0.480176234530. ML is untouched (same fixture logLik before and after:
  −61.46784165162242). The widening is scoped to exactly the shape the sparse route serves:
  `sigma ~ x`, `relmat` / `animal` / `spatial`, `algorithm = :em` and a phylo random slope all
  still refuse (`test/test_reml_reml_phylo_mean.jl`). A REML fit also carries the
  RESTRICTED objective in `fit.nll` and no `fit.nllgrad`: the route's analytic
  score belongs to the ML marginal, and at the REML optimum it is (1.01, 0.99) on
  the two variance parameters — reporting it would have read as "not converged"
  for a converged fit, through drmTMB's `fit$bridge$gradient`. The ML path keeps
  its analytic score unchanged.
- **Profile and bootstrap intervals on a fixed-effect target of the RESIDUAL-ONLY bivariate
  Gaussian fit (drmTMB A8b).** `bf(mu1 = y1 ~ x, mu2 = y2 ~ x, sigma1 = ~ 1, sigma2 = ~ 1,
  rho12 = ~ 1)` with no structured term had no bootstrap route at all: `_bootstrap_fit_formula`
  refused every `BivariateDrmFormula`. That gate now admits the residual-only case, and the
  three blockers behind it are fixed — the `bootstrap_result(::DrmFit{<:Gaussian})` refit
  closure no longer forwards `algorithm` (which `drm(::BivariateDrmFormula, ::Gaussian; ...)`
  does not declare), `_bootstrap_result` accepts a `BivariateDrmFormula`, and `_bootstrap_data`
  gained a bivariate method for the `Dict(:mu1, :mu2)` draw `_simulate_once` returns. That
  method PRESERVES THE OBSERVATION PATTERN: a cell unobserved in the data stays unobserved in
  every replicate, so the refits fit the model the seed fit fitted. `profile_result` needed no
  change (it already reached the generic ForwardDiff path) and is now covered by assertions on
  the #631 endpoint-failure flags. Measured on an n = 200 fixture: 40/40 replicates, 0 failed,
  finite bounds on all five blocks; `drm_bridge_inference(parm = "fixef:mu1:x" |
  "fixef:rho12:(Intercept)")` returns finite rows for both methods
  (`test/test_bridge_biv_inference.jl`, 71 assertions). BOUNDARY: the STRUCTURED (q = 4
  phylogenetic) bivariate route is untouched — its `Sigma_a` guard fires first and it still
  routes to `bootstrap_sigma_a`; a `DrmFit` with no `.formula` is still refused; and this is a
  capability claim, NOT an interval-coverage claim (one fixture, one seed).
- **`converged` on the varying-scale Gaussian random-intercept route is now the GRADIENT
  criterion (#609 item 2).** `bf(y ~ x + (1 | g), sigma ~ x)` fits through
  `_fit_ranef_gaussian`, which reported `Optim.converged(res)` -- the OR of Optim's x, f and g
  criteria. Because `Optim.Options(g_tol = ...)` leaves `f_reltol`/`f_abstol`/`x_reltol`/
  `x_abstol` at their `0.0` defaults, `f_converged` fires on two byte-identical successive
  objective values, which a VARYING-scale surface reaches while running away up the unbounded
  sigma_i -> 0 ridge, nowhere near a stationary point. Measured over a 6,400-cell grid (n
  30..400, G 3..20, sigma slope 0.15..8.0): 1,042 fits returned `converged = true` with the
  gradient criterion false; the true gradient inf-norm there exceeded 1e-3 in 95 of them and 1.0
  in 45, worst 3.7e137 alongside a POSITIVE Gaussian logLik of +980. The reported flag now also
  requires `Optim.g_converged(res)`, so `fit.converged` implies the gradient met `g_tol`.
  NOTHING ELSE MOVES: theta-hat, the ML and REML objectives, logLik, vcov and the BLUPs are
  byte-identical (five fixtures across four routes, logLik equal to 17 significant figures
  before and after). This is the DRModels.jl half of #609 item 2; the ~1e-5 conditional-prediction
  parity gap the issue diagnosed is drmTMB's `nlminb` stopping rule and is tracked as
  drmTMB #1130. Guard: `test/test_ranef_varying_scale_convergence.jl`.
- **`coef_labels` count mismatch now names the construct behind it (#467, #609).** When the R side
  supplies more names for a dpar than the fit has columns, the commonest cause is a factor level
  with no rows in the data: base-R `model.matrix()` gives every DECLARED level a column (an
  all-zero one for a level nothing uses), while DRModels.jl codes only the levels it OBSERVES. The
  message previously said only "the R side must send exactly one name per column", naming neither
  the column nor the fix. It now names the coded columns DRModels.jl actually built and points at
  `droplevels()`. Measured through drmTMB on 2026-09-05 against this repository: `y ~ gempty` with
  `levels = c("a", "b", "c", "zz")` produced exactly this mismatch (R supplied 4 names, DRModels.jl built
  `["(Intercept)", "gempty: b", "gempty: c"]`). The hint is emitted only when the block has coded
  (`"<column>: <level>"`) columns and the supply is too LONG; a short supply, and a block of purely
  continuous columns, keep the bare count message unchanged. Behaviour is otherwise identical: this
  is refusal WORDING, not a new refusal, and every existing count/echo/fidelity test is unchanged
  (`test_bridge_formula_constructs.jl` 188 assertions, `test_bridge_coef_labels_echo.jl` 42,
  `test_bridge_formula_labels.jl` 819, `test_bridge_base_r_names.jl` 20, all green).
- **`check_drm()` CRASHED on four shipping routes instead of reporting on them, and a NaN gradient
  could not be told apart from a verified one.** `_check_max_abs_grad` (`src/inference.jl`) ended in
  an unguarded `maximum(abs, ForwardDiff.gradient(fit.nll, fit.theta))`. Four routes store a bare
  objective (no gradient callback) that is exact on `Float64` but NOT dual-number safe, so that line
  threw and the health check died on the very fits it exists to report on -- against the comment
  written two functions below it, "A diagnostic must REPORT trouble, not crash on it". Measured
  2026-09-05 at origin/main 109b6421c, in two distinct exception types: the bivariate q=2 structured
  route (`gaussian_bivariate.jl:626`, whose objective factorises a sparse `H_uu` through CHOLMOD
  inside `coevo_marginal_cov`) and the sparse two-structured Gaussian mean route
  (`gaussian_structured.jl:652`, same factorisation) both raised `TypeError: in Sparse, in Tv,
  expected Tv<:Union{Float64, ComplexF64}, got Type{ForwardDiff.Dual{...}}`; sparse LSS under REML,
  single-component (`gaussian_sparse_lss.jl:285`) and multi-component (`:1018`), raised the
  different `MethodError: no method matching Float64(::ForwardDiff.Dual{...})` from a Float64 work
  array. Only the REML arm of those last two ever reaches the line: both store
  `reml ? nothing : nllgrad!`, and the ML arm's stored callback is consulted first (its
  `max_abs_grad` is unchanged). The probe now falls back to a CENTRAL FINITE DIFFERENCE of the same
  objective -- the pattern `_profile_autodiff_mode` (`:805-821`) already uses for exactly this
  situation -- rather than returning `NaN`. Returning `NaN` would have been the quiet version of the
  same defect: `ok = converged && (penalized || isnan(mag) || mag <= grad_tol) && pd` treats a NaN
  magnitude as a PASS, a disjunct written for a fit that stores no objective at all, so an untested
  route would have reported `ok = true` and looked identical to a verified one. `check_drm` gains a
  `grad_source` field naming the producer -- `:locscale`, `:stored`, `:forward`, `:finite`, `:none`,
  `:unavailable`, reusing `profile_result`'s `autodiff` vocabulary -- warns when the magnitude is a
  finite difference (good to about 1e-6 relative, not to machine precision), and warns again on
  `:unavailable`, the NaN that is NOT `:none` and whose `ok` was never scored against a gradient.
  Measured after the fix, all four now RETURN with `grad_source = :finite`: q=2 structured
  2.7504093225161337e-3, two-structured sparse 2.7248279090903387e-8, sparse LSS REML
  1.7936919476824185, multi-component sparse LSS REML 1.0092392038529852. The last two are an
  accuracy check on the fallback rather than just a smoke test: the dense engine fits the same two
  models with a dual-safe objective, and its exact ForwardDiff magnitudes on the identical fixtures
  are 1.7936919467546497 and 1.0092392105794588 -- the finite difference agrees to 5.2e-10 and
  6.7e-9 relative. Eleven dual-safe fixtures spanning `:forward`, `:stored` and `:locscale` are
  unchanged to the last digit (e.g. plain Gaussian 2.580158309228864e-13, canonical location-scale
  1.651913078548617e-9), with every `check_drm` verdict field identical. Guarded by
  `test/test_check_drm.jl` (77 assertions, one testset per AD-hostile route plus `:none` vs
  `:unavailable` and a non-finite stored callback); removing the guard errors 5 of its testsets with
  the exceptions quoted above.

- **Bivariate q=2 `spatial(1 | group)` is admitted at the formula front end.** The q=2 marker
  allow-list took only `phylo`, `relmat` and `animal`, so a `spatial(...)` marker on `mu1`/`mu2`
  was refused outright -- a cell drmTMB fits natively AND through the bridge, which reaches it
  only by rewriting `spatial(...)` into `relmat(...) + K` on the R side (`R/julia-bridge.R`).
  `spatial` now supplies a covariance exactly as `relmat(K = ...)` and `animal(A = ...)` do:
  the exponential kernel `exp(-d / rho)` over the group levels at a FIXED range `rho` (keyword
  `spatial_range`, else the mean off-diagonal pairwise distance), which is the identical rule
  the q=4 structured route already used -- that construction is now one shared helper, so the
  two cells cannot drift apart. Same solver call as the other providers, so `method = :REML`
  comes on the same path, and `fit.ranef.structured_type` carries `:spatial` through to the
  bridge export target `gaussian_q2_mu1_mu2_spatial_residual_correlation` with no change to
  `src/bridge.jl`. Measured: `spatial(coords)` and `relmat(K = the same matrix)` on one fixture
  agree BIT-FOR-BIT -- max abs difference 0.0 on logLik and on all ten coefficients, under both
  ML and REML (`test/test_reml_q2_structured.jl`). The range in force is readable off the fit as
  `fit.ranef.spatial_range`, the same field the q=4 route already records, because the default is
  data-dependent and would otherwise be invisible; `relmat`/`animal` fits report `nothing`.
  BOUNDARY: `rho` is FIXED, not estimated. This is not the univariate `spatial` route, which
  carries `log rho` as a free parameter -- the two are different models and must not be compared
  as one. Note also that DRModels.jl's default range is the MEAN off-diagonal distance with a 1e-8
  ridge, while drmTMB's native R-side rule is the MEDIAN positive distance with a 1e-6 jitter,
  so the two engines build different matrices from the same coordinates unless `spatial_range`
  is pinned; a coords-to-coords cross-engine comparison is therefore not a parity check.
  REPAIR shipped with it: the q=2 route had a second, broader guard that refused ANY q=2
  structured fit merely for carrying a `coords =` keyword, so an ordinary `relmat`/`animal` q=2
  fit with a stray `coords =` errored. That guard is gone; each provider branch refuses when its
  own input is missing.
- **`drm_bridge` admits `skew_normal`** — `_bridge_family` now maps drmTMB's
  `skew_normal()` (tags `skew_normal` / `skewnormal`) to the native `SkewNormal()`
  family. The family was already implemented (`src/skewnormal.jl`) but the R
  bridge had no case for it, so drmTMB's `engine = "julia"` could not admit it.
  Both packages use the same public moment parameterisation (`mu` = mean,
  `sigma` = SD, `nu` = Azzalini slant), so the bridged coefficients are the
  native ones. Fixed effects only, ML only — exactly what `SkewNormal()` fits.
- **REML on the residual-only bivariate Gaussian route (#624; drmTMB #1142).** `drm(bf(mu1 = y1 ~ x, mu2 = y2 ~ x, sigma1 = ~1, sigma2 = ~1, rho12 = ~1), Gaussian(); method = :REML)` fits instead of refusing. The old message ("`method = :REML` needs random effects to restrict") mistook "no random effects" for "nothing to restrict": REML here integrates the MEAN fixed effects `beta_mu1`/`beta_mu2` out of the Gaussian likelihood under the row-wise 2x2 residual covariance — the classical Patterson–Thompson/SUR case, and the same set native drmTMB hands to TMB's Laplace approximation for this cell (whose Laplace step is EXACT, the integrand being quadratic). Closed form: profile `beta` by GLS, add `-0.5*logdet(sum_i Z_i' S_i^-1 Z_i)` and the `+(p_beta/2)*log(2*pi)` normalising constant, and optimise over `(beta_sigma1, beta_sigma2, beta_rho)` alone. Same-target against drmTMB `engine = "tmb"` with `REML = TRUE` on the committed fixture: logLik `-97.021205818372` on both engines (difference 0.0 at 12 decimal places), coefficients to 4.33e-7, standard errors to 6.54e-7 relative. `vcov` reports the joint mean block `H^-1 + G Var(phi) G'` that TMB's `sdreport()` reports. The ML route is unchanged (same fixture, `-90.202703298791` on both engines, difference 4.8e-13). `meta_V` plus `:REML` keeps its permanent refusal — that route marginalises nothing. `test/test_reml_reml_biv_residual.jl`.
- **Profile CIs on the canonical (non-sparse) location-scale route returned a signed infinite
  bound because ONE unevaluable trial abandoned the whole endpoint arm (#651).** #631/#633 closed
  this class for the SPARSE phylogenetic LSS engine and made `confint(...; method = :profile)` fail
  closed; the canonical whitened route kept the defect, and it has been failing unrelated pull
  requests at random on `Julia 1.10 - shard 1/4` (`test/test_locscale_profile_threads.jl:55` and
  `:91`, the two `isfinite` assertions). The endpoint search
  (`_ls_profile_root_result`, `src/locscale_profile.jl`) treated a single trial whose nuisance
  solve did not converge as terminal, in both the bracket-expansion and guarded-Newton phases, and
  reported that premature give-up as `Inf`/`-Inf`. It is not a data race and not run-to-run
  non-determinism: 600 repeats of the fixture at 1, 2 and 4 Julia threads produced zero failures
  and exactly four distinct `gradient_maxabs` values, bit-identical across thread counts. The
  fixture's accepted arms simply sit against the acceptance bar -- worst measured 7.96e-08 against
  a hard 1e-07 -- so a last-bit difference between macOS aarch64 and Linux x86-64 decides whether
  an arm is accepted, which is why it reproduced in CI and not locally. Both phases now CONTRACT
  toward a point already known to be feasible, on the bounded budget `maxcontract` (default 8):
  expansion halves `thi` toward the feasible `tlo`, and refinement halves the TRIAL toward `tlo`,
  leaving the maintained bracket -- and its proof that a crossing exists -- untouched. Measured with
  a +/-2-ULP perturbation of the response, the honest local surrogate for that platform difference:
  17 of 200 draws returned a non-finite bound before, 0 of 200 after, with all 800 endpoint arms
  accepted and 37 contractions across exactly the 17 draws that used to fail. The search is still
  bounded -- at most `(maxexpand + maxnewton + 1) * (1 + maxcontract)` evaluations -- so it cannot
  hang, and it is inert where nothing fails: the unperturbed fixture's intervals and residuals are
  bit-identical before and after. Contraction only POSTPONES failure; an arm that cannot be
  evaluated anywhere down to `xtol` is still a failed, signed-infinity endpoint, so
  `profile_result` remains the auditable surface #631 made it and `confint` still refuses it. One
  behaviour change beyond that: an expansion that had to contract can no longer certify
  UNBOUNDEDNESS, because part of the range was never evaluated -- it reports the new
  `:infeasible_region` failed endpoint instead of `:no_crossing`. Guarded by
  `test/test_locscale_profile_threads.jl` (29 assertions, 0.3 s, deterministic and
  platform-independent: it pins the search logic, which is what changed).

- **Inference on a missing-response ("response mask") Gaussian fit: `is_converged` read false on a
  converged fit, and every bootstrap replicate failed (#646).** Both defects sat downstream of the
  fit itself -- the point estimates were already right, matching drmTMB's `engine = "tmb"` to ~7e-06
  on the qualification fixture -- which is why nothing caught them until bridge-side inference was
  measured. (1) `_nondegenerate_fit` (`src/summary.jl`) took its scale-free degeneracy bar as
  `std(fit.obs[:mu])`, but the missing-response routes keep the full-design response there with NaN
  in the masked positions, so `yscale` was NaN and `smax > 1e-6 * max(NaN, eps)` was false for every
  `smax` under IEEE-754. `is_converged` therefore returned false on EVERY masked Gaussian fit,
  independent of the optimiser: measured on the fixture, `Optim.converged` true, `|grad|inf`
  6.41e-12 against `g_tol` 1e-8, and `theta` bit-identical to the complete-case fit (max abs diff
  0.0), yet `is_converged` false. The bar is now taken over the observed (finite) responses only;
  a genuinely degenerate sigma is still rejected. (2) `_simulate_once` (`src/gaussian_core.jl`)
  drew `randn(rng, fit.nobs)` -- the count that entered the likelihood, 54 on the fixture -- and
  broadcast it against the length-60 `means`/`scales` that `_with_full_fixed_gaussian_rows` rebuilds
  over the full design, throwing `DimensionMismatch` deterministically on every replicate, so
  `bootstrap_result` raised `"all B bootstrap replicates failed"`. Draws are now taken per row of
  the design, which is also the length `_bootstrap_data` needs to merge back into the caller's
  table. (3) Found while fixing those: `_with_full_fixed_gaussian_rows` and `_with_full_response_rows`
  used the 19-argument compatibility constructor, silently resetting `iterations` to -1
  ("not recorded") -- the identical complete-case fit reported 7 -- and dropping the MAP penalty
  slots; both now pass all 22 fields. Guarded by `test/test_bridge_response_mask_inference.jl`
  (21 assertions: converged status, a 5-replicate bootstrap on a masked response, the iteration
  count, and a degenerate-sigma control proving the widened bar still rejects).
- **Standard errors on the bivariate q=2 structured route (relmat / animal / phylo, ML and
  REML).** `vcov` on this route was an all-NaN placeholder for every provider and both
  estimators, which `stderror` then mapped to `Inf` and `confint` to `(-Inf, Inf)` -- the
  package's signal for "at a variance boundary", so a converged fit was indistinguishable from
  an unidentified one. The route now second-differences the marginal ML negative log-likelihood
  it already builds, through the shared `_finite_hessian` helper, and passes the result to the
  same `_vcov_from_hessian` guard the dense ML bivariate route uses; that guard eigen-tests the
  matrix, pseudo-inverts a singular one and warns, naming this route. ForwardDiff is not an
  option here because `coevo_marginal_cov` factorises a sparse H_uu through CHOLMOD, which
  rejects dual numbers -- the same reason the sparse-LSS routes finite-difference. RECEIPT: the
  beta block of the finite-difference Hessian is compared against the EXACT Schur complement
  `S = X' V^-1 X` that `_q2_profile_and_schur` assembles by an entirely different route; max
  relative difference 3.96e-9 (ML) and 9.75e-9 (REML) on the committed fixture, asserted at
  1e-6 in `test/test_q2_structured_vcov.jl`. That oracle is exact but covers only the four
  beta coordinates, so the variance/correlation axes are corroborated separately by profile
  likelihood, which never touches the Hessian: profile/Wald half-width ratio 1.0082 for
  `rho12` and 1.0050 for `sigma1`. `confint(fit, method = :profile)` works on this route
  through the existing `:finite` autodiff fallback -- and note that it did NOT before: the
  profile endpoint search seeds its bracket from the Wald standard error, so an infinite SE
  made it refuse with `not_converged`. The all-NaN covariance had therefore also disabled the
  documented fallback that the boundary warning tells users to reach for. Cost is 0.012 s at the usual ten parameters and
  scales as O(p^2): 2.6 s at 46 parameters (a 20-covariate mean design), against a 6.1 s fit,
  so there is no opt-out keyword -- unlike the q=4 route's `q4_vcov`.
  ESTIMATOR NOTE: under `method = :REML` the reported covariance is the ML curvature evaluated
  at the REML point, with the restricted-penalty curvature omitted -- the policy already
  documented for the q=4 sibling. REML Wald intervals on this route are therefore mildly
  anti-conservative; prefer `confint(fit, method = :profile)` when that matters.
  KNOWN WART: when the finite-difference Hessian is poor, the warning that says so is emitted
  under a hardcoded "sparse-Laplace vcov:" label, because the shared helper carries it. The
  route-naming text reaches the user through the guard's own singularity warning.

- **Gaussian two-SD phylogenetic random slope `phylo(1 + x | species)` on the mean (#620).** The
  #621 refusal is replaced by the fit on the Gaussian mean route: two INDEPENDENT phylogenetic
  fields, `a ~ N(0, σₐ² C)` for the intercept and `b ~ N(0, σ_b² C)` for the slope on the same
  tree correlation `C`, no intercept–slope correlation. That is drmTMB's Gaussian model
  term for term (`src/drmTMB.cpp`, `model_type == 1`, `has_phylo_mu` with `q_phylo == 2`: both
  fields sit on `mu`, so `has_cross_dpar_phylo` is false and the else-branch adds one
  `exp(-2·log_sd_phylo(k))·uₖᵀQuₖ` term per field with no cross term). Fitted here as the
  closed-form dense marginal `V = D + σₐ² Z C Zᵀ + σ_b² Zₓ C Zₓᵀ`, which is that Laplace
  objective evaluated exactly. Reported as a two-row `:resd` block (`species`, `species:x`;
  `re_sd`/`vc`/`ranef` all carry both fields), five free parameters — the same five drmTMB
  optimizes. Measured same-target against drmTMB 67703f541 on the committed fixture: logLik,
  both SDs, β and log σ all agree to ≤ 6.7e-13 (`test/test_phylo_slope_two_sd.jl`).
  BOUNDARY (corrected 2026-09-05 after measurement): drmTMB fits `phylo(1 + x | g)` on Poisson
  and NegBinomial2 as well, and it fits the SAME independent two-SD model there — measured on
  drmTMB main, `corpars` is empty for the untagged `phylo(1 + x | g)` on Poisson (two SD rows,
  `phylo(1 | site)` and `phylo(0 + x | site)`), while the estimated intercept–slope correlation
  (`has_phylo_mu_q2_covariance`, reported in `corpars` as `cor(mu:(Intercept),mu:x | p | site)`)
  appears only for the DIFFERENT tagged formula `phylo(1 + x | p | g)`. drmTMB refuses the
  formula on Gamma ("intercept-only in this q=1 route"). DRModels.jl still refuses every non-Gaussian
  family here, for the accurate reason: this route is the EXACT closed-form Gaussian marginal
  and does not extend to a non-Gaussian likelihood, so the count families need a two-field
  Laplace route that does not exist yet. The earlier claim that drmTMB fits "a DIFFERENT model"
  for Poisson/NegBinomial2 on this formula was wrong; the refusal it justified is unchanged. `phylo(0 + x | g)`, multi-slope
  forms, a second structured component, ordinary and `sigma`-side random effects, a structured
  (`phylo`) `sigma`, the `sd(...)`/`sd_phylo(...)` location-scale-scale submodels, REML, missing
  responses, the sparse algorithms and the parametric-bootstrap simulator all stay refused with
  a named error. Those refusals are raised next to `_split_ranef`, ABOVE every route that can
  return, so no engine written for the intercept-only cell can receive this formula and quietly
  fit `phylo(1 | g)` instead.

## v0.7.0 — 2026-08-28

**The twin-parity release.** Versioning now twin-tracks drmTMB (D-181, like GLLVM.jl↔gllvmTMB):
v0.7.0 means "at parity with drmTMB 0.7.0", pinned at drmTMB commit `a6de6bb71`; drmTMB's
fast-moving mi() missing-data axis is a named, fenced delta (D-180 #1) and the designated
post-0.7 headline. Registration in Julia General is planned at v0.7.1, mirroring the twin.

- **The API freeze** — every one of the 159 exports classified into Stable / Experimental /
  Engine tiers, machine-checked by `test/test_api_stability.jl` (total classification: a new
  export cannot appear untiered, a stable name cannot vanish silently). The promise and its
  honest Julia-0.x framing: `docs/src/api-stability.md`.
- **The comprehensive engine speed grid** (`report/engine-speed-grid.md`) — Julia wins 14 of 15
  comparable warm-fit cells (2.3×–42×); the one loss (q4 phylo REML, 0.46×) was diagnosed (the
  inner alternation burned 15 iterations per objective evaluation against a never-firing exit)
  and fixed to ~parity (0.94×, 2.1× faster).
- **The user-journey sweep** (`tools/user_journey_sweep.R`) — 29 models typed the way users type
  them, both engines: 25 matches at 1e-6..1e-11, refusal quality graded, and one real bug found
  and fixed (drmTMB#1099: `predict(newdata)` on factors/interactions failed on
  Julia-vs-R coefficient-name conventions).
- **The threaded bootstrap, demonstrated** — `confint(fit, parm, method = "bootstrap",
  threads = TRUE)` from R: 10× native TMB at R=199 on the same model and data
  (`docs/dev-log/evidence/2026-08-28-engine-julia-usability-demo.md`).
- Wave Q quality debt to zero: drmTMB#1090 (univariate `nu` tripping the bivariate bridge
  branch) fixed; #526 (q4 REML flag folds in the inner alternation, measured criterion);
  #527 hardening both repos; #482 and #8 closed.

## v0.2.0 — 2026-08-28

### The completion arc (2026-08-24 → 2026-08-27, D-179)

- **The R↔Julia capability ledger is COMPLETE** (drmTMB PRs #1085, #1087, and
  the Phase 2 promotions): every one of the 11 rows in drmTMB's generated
  `julia-capabilities.tsv` is now `covered` (9), an owner-signed permanent
  boundary (`cross_family_latent`, drmTMB #1089 — with a retraction: the route
  IS reachable from R), or unsupported by design (`engine_control_surface`).
  Every promotion is backed by SE-grade parity on a stamped comparator build.
  `covered` remains a capability claim, never interval coverage — the
  `coverage_claimed` fences are permanent documented boundaries by owner
  decision (D-179 #4).
- **`converged` answers a fixed question (#491, #517)** — the sparse-Laplace
  flag now tests the mean per-observation gradient against 1e-6 and no longer
  short-circuits on `Optim.converged`; a deliberately sloppy `g_tol` can no
  longer buy a green flag ("converged is not for sale" gate).
- **The large-p SE gap is fixed at source (#517)** — the outer vcov
  finite-difference step now grows with n (`clamp(2.5e-7·n, 1e-4, 1e-2)`);
  measured SE parity vs native TMB at p=1000 went 1.2e-03 → 4.0e-06 and at
  p=3000 9.0e-04 → 4.5e-06. Below n=400 nothing changes.
- **`response = "include"` fits the Gaussian phylo-mean cell (#517)** —
  measured byte-identical to `drop` on drmTMB's own route, so it is
  implemented as observed-rows + full tree (the σ-phylo convention); the
  wrapper deliberately does not extend to positional-matching routes.
- **q4 `converged` is gated on Λ admissibility (#509, #518)** — success is no
  longer claimed at a numerically singular Λ (the saturated-fixture regime);
  the smoke fixture is de-saturated and the saturated one kept as the
  negative control.
- **`mstep_Lambda`'s descent is a wired characterisation (#472, #519)** — the
  measured negative result now runs in the default suite as a tripwire, and
  `src/experimental/` carries per-file verdicts (#520).
- **Per-family speed table (#9, #521)** — `report/speed-per-family.md`
  consolidates every engine-vs-engine timing with its caveats; README swept to
  match the promoted ledger (its REML sentence was three routes stale).
- Earlier in the same arc: bivariate q=2 structured REML (#470,
  `reml_q2.jl`), per-coefficient `parm` for `confint` (#495, #514), comparator
  provenance stamping in all four parity harnesses (#473, #512), and the inner
  `newton_tol` tightening whose SE effect was measured, not assumed (#513).


- **VA Rung 2+3 (#136)** — scaffold anchors a/b/c in `test/test_variational.jl`
  are live (RE→0 ELBO = GLM loglik; ELBO ≤ adaptive GHQ; NB2 `r→∞` ≈ Poisson-VA).
  `aicc` on a VA fit errors even when `n − k − 1 ≤ 0` (no `Inf` short-circuit).
  Docs status is **Experimental** for Poisson + Binomial + NB2 + Gamma + Beta
  `(1 | g)`, not Planned-only. **Does not close #136.**

- **Public variational marginal Rung 1 (#136)** — Experimental
  `drm(...; marginal = :VA)` now covers Poisson **and** Binomial / NegBinomial2 /
  Gamma / Beta random intercept `(1 | g)` (scale families need `sigma ~ 1`),
  routing to the existing ELBO kernels. Default remains Laplace. `loglik` on a
  VA fit is an ELBO; mixed LA/VA AIC/LRT errors. Unsupported VA
  (phylo/crossed/corr/zi/hu/FE/`sigma ~ x`) rejects. `method = :VA` points at
  `marginal`. **Does not close #136.**

- **Non-Gaussian phylogenetic location–scale (#202)** — public `drm()` path for
  `NegBinomial2()` / `Gamma()` with a shared structured RE on **μ and log σ**
  via grammar B `(1 | p | phylo(species))` on both axes (`tree=` forwarded into
  the locscale frontend). `vc(fit)[:species]` reports the 2×2 group-level
  covariance (never residual `rho12`). Dual issue-text `phylo(1|sp)` on both
  axes stays rejected. Evidence: `test/test_public_phylo_locscale.jl` (NB2
  recovery + Gamma smoke). R `nbinom2-locscale` fixture deferred until drmTMB
  supports coupled `(1|p|species)` (D-94; R q=1 structured-σ covers scale-axis
  existence).

## v0.1.2 (2026-08-01)

**Version catch-up on the post–v0.1.1 `main` tip (registry candidate — not yet
in Julia General).** `Project.toml` / `CITATION.cff` advance `0.1.0` → `0.1.2`
so a first General registration does not collide with the existing git tags
`v0.1.0` / `v0.1.1` (May 2026; tip is hundreds of commits past those tags).
Prep + gates: [`docs/dev-log/plans/registrator-prep-2026-08-01.md`](docs/dev-log/plans/registrator-prep-2026-08-01.md).
**Registrator submit and tagging wait explicit Shinichi OK** (ultra-plan S4).
This note does **not** claim General membership, Phase 1.5 / #5 bridge closeout,
or a full public wire of remaining `src/experimental/`.

Post–v0.1.1 tip highlights already on `main` (Rose-honest; measured claims stay
in `HANDOVER.md` / `report/`):

- **Registry hygiene (S2/S3)** — ayumi↔main integrate (#340), load-banner silence
  (#341), docs honesty on Next / bridge / version drift (#342), post-merge
  checkpoint (#343).
- **σ-phylo REML correctness** — REML restricts both β_μ and β_ψ (#337).
- **Opt-in Gaussian REML**, conjugate-EM phylo-mean solver, heritability /
  repeatability / ICC accessors, DHARMa-style randomised quantile residuals, and
  the coevolution q=4 phylo front end (see HANDOVER TL;DR).
- **Julia-side bridge surface** (`drm_bridge` / `drm_bridge_inference`) remains
  in-tree; R-side `engine = "julia"` finish is still open (#5).
- **Bivariate q=4 coevolution bootstrap uncertainty** — `bootstrap_sigma_a` gives
  parametric-bootstrap intervals for the q=4 phylogenetic among-axis SDs
  (`sqrt(diag(Σ_a))`) and the six among-axis coevolution correlations, reachable
  from R via JuliaCall; `bootstrap_result(fit)` and the R bridge route a bivariate
  q4 fit here. Boundary-honest: a collapsing scale axis returns a small interval
  near the floor and its correlations come back unidentified (`~[−1, 1]`). A
  coverage study (`report/finish-audit/bootstrap-coverage-findings.md`) confirms
  the mean-axis SD CIs are well-calibrated and the detection/correlation reads are
  robust; the scale-axis *precise* CIs are anti-conservative (a known
  boundary-bootstrap effect) — read them as detection/indication, with a calibrated
  interval (BCa / bias-correction) tracked as a follow-up.
- **Per-parameter prediction** — `predict_parameters` (fitted distributional
  parameters on new data), `marginal_parameters` (population-averaged), and
  `prediction_grid` (build a swept `newdata` grid from a reference table).
- **Auditable profile-likelihood CIs** — `profile_result` exposes the full
  profile object behind `confint(fit; method = :profile)`.
- **Post-fit accessors** — `summary`, `family`, `is_converged`, `deviance`,
  `dof_residual`, and `rho12` (bivariate residual correlation).
- **Non-Gaussian phylogenetic random effects** — `phylo(1 | species, tree)` on
  the mean for Poisson, NegBinomial2, Gamma, Beta, and Binomial families
  (constant `σ`), via a sparse Laplace approximation, plus crossed intercepts
  `(1 | g) + (1 | h)` for those families.

## v0.1.1 (2026-05-31)

**drmTMB family parity complete** — the four remaining families, each
recovery-tested and shipped one-PR-per-family with green CI. DRModels.jl now fits
every distribution family drmTMB offers.

- **Beta-binomial** `BetaBinomial()` — successes out of known trials with
  extra-binomial overdispersion. Two-column `cbind(successes, failures) ~ …`
  response (drmTMB-exact, via a `cbind` formula marker + a second-response field
  on `DrmFormula`), logit mean + `φ = 1/σ²`. (drmTMB has no standalone
  `binomial`; ordinary binomial is the `φ → ∞` limit of beta-binomial.)
- **Zero-one-inflated beta** `ZeroOneBeta()` — proportions on the closed `[0,1]`;
  mixture `P(0)=zoi(1-coi)`, `P(1)=zoi·coi`, `(1-zoi)·Beta(μ,φ)`. Params
  `mu`/`sigma`/`zoi`/`coi`.
- **Tweedie** `Tweedie()` — semicontinuous (compound Poisson–Gamma, `1<p<2`):
  exact-zero mass + positive continuous part. Mean `μ` (log), `sigma` =
  √dispersion, `nu` = the estimated power `p` (logit-`(1,2)`). Density via the
  Dunn–Smyth series (adds the `SpecialFunctions` dependency).
- **Cumulative-logit** `CumulativeLogit()` — ordinal: `Pr(y≤k)=logistic(θ_k−η)`
  with `K-1` ordered cutpoints; intercept dropped.

Full set (12 univariate + bivariate Gaussian): Gaussian, Student-t, LogNormal,
Gamma, Tweedie, Beta, zero-one-inflated beta, beta-binomial, Poisson,
NegBinomial2, truncated-NB2, cumulative-logit. Families are validated by
simulation parameter recovery; the numerical drmTMB-parity gate is #17.

## v0.1.0 (2026-05-31)

First tagged release: the `drm()` / `bf()` distributional-regression front end
with **8 response families**, the count `zi` / `hu` modifiers, the complete
Gaussian random-effect / structured / inference surface, and a published
DocumenterVitepress site. Formula syntax mirrors drmTMB exactly; families are
validated by simulation parameter recovery (numerical drmTMB-parity gate: #17).

### Phase 0 — Team & workflows (2026-05-30)

- Stood up the 12-persona team (`AGENTS.md`), extended the project `CLAUDE.md`,
  and added `ROADMAP.md` (phases → v1.0).
- Added 10 scripted workflows in `.claude/workflows/` (W0/Q/A/B/D/F/G/H/S/R)
  and 12 Codex agent configs in `.codex/agents/`.
- Established the **GitHub work ledger**: labels, milestones (Phase 0 → v1.0),
  and the initial near-term issues; issue + PR templates.
- Added the `docs/dev-log/` discipline (check-log, coordination-board,
  after-task, decisions, recovery-checkpoints, scout) and
  `tools/drm-checkpoint.jl`.
- Scaffolded the **Documenter** site mirroring drmTMB's pkgdown navbar — 36
  status-tagged stub pages, reference index in 6 workflow-ordered categories.
- Project meta: `bench/Project.toml`, `test/Project.toml`, `CITATION.cff`,
  `.JuliaFormatter.toml`, and `Documenter.yml` / `TagBot.yml` CI.
- **Engine unchanged.** The verified q=4 PLSM engine (2.18× over drmTMB on the
  single fit, O(p) to p=10,000, valid CIs where drmTMB's Hessian is singular)
  is exactly as handed over. See `HANDOVER.md`.

### Gaussian surface — first tranche (2026-05-30)

The public `drm()` / `bf()` front end (StatsModels `@formula`, mirroring drmTMB)
and the Gaussian family, built test-first with recovery tests and merged via
PRs #21–#27 (green CI each):

- **Univariate location–scale** — `drm(bf(y ~ x, sigma ~ x), Gaussian())`, ML.
- **Bivariate location–scale + residual correlation** —
  `bf(mu1=…, mu2=…, sigma1=…, sigma2=…, rho12=…)` (tanh link on ρ12).
- **Ordinary random intercept** `(1 | g)` on the mean — closed-form Gaussian
  marginal (matrix-determinant lemma + Woodbury); `re_sd`.
- **Meta-analysis** — `meta_V(v)` known sampling variances + estimated
  heterogeneity τ.
- **Inference & post-fit** — `coef` / `vcov` / `stderror` / `confint` (Wald) /
  `fitted` / `residuals` / `loglik` / `nobs` / `fixef`.
- **Docs** — landing page rewritten as a real stats-package page; Get started,
  location-scale, bivariate-coscale, which-scale, meta-analysis, model-workflow,
  and the "What can I fit today?" capability map filled with **executed**
  examples.
- Fixed R's implicit intercept (`y ~ x` ⇒ `y ~ 1 + x`). Verified `src/` engine
  unchanged.

### Gaussian surface — completed (2026-05-31)

Completing the Gaussian distributional-regression surface, all recovery-tested,
each shipped as one PR with green CI:

- **Random effects on the mean** — independent slopes `(0 + x | g)`, correlated
  intercept+slope `(1 + x | g)` (`vc`), and multiple crossed / nested terms
  `(1 | g) + (1 | h)` (whitened-Woodbury capacitance).
- **Random effects on the scale** — `sigma ~ … + (1 | g)`, integrated out by
  per-group Gauss–Hermite quadrature (#40).
- **Structured effects on the mean** — `relmat(1|id, K)`, `animal(1|id, A)`,
  `phylo(1|species, tree)`, `spatial(1|site, coords)` — closed-form GLS.
- **Parametric bootstrap** (`bootstrap_ci`) and **profile-likelihood** intervals
  (`confint(fit; method = :profile)`, #38), alongside Wald.
- **Post-fit** — `predict` (new data) and `simulate`.

### Phase 2 — response families (2026-05-31)

Eight families behind the same `bf()` grammar, each with its own per-parameter
formulas and a simulation recovery test:

- **Student-t** `Student()` — robust location–scale–shape (μ, σ, ν).
- **Poisson** `Poisson()` — counts (log-link mean).
- **NegBinomial2** `NegBinomial2()` — overdispersed counts (dispersion θ).
- **TruncatedNegBinomial2** `TruncatedNegBinomial2()` — positive counts (≥ 1).
- **Beta** `Beta()` — proportions in (0,1) (logit mean; precision φ = 1/σ²).
- **Gamma** `Gamma()` — positive continuous (shape α = 1/σ²).
- **LogNormal** `LogNormal()` — positive, multiplicative.
- **Count modifiers** — `zi` (zero-inflation: ZIP / ZINB) and `hu` (hurdle).

### Documentation — the makie-style site (2026-05-31)

- Adopted the **DocumenterVitepress** backend (the docs.makie.org look); Node is
  supplied by `NodeJS_jll`, so there is no system install and no extra CI step.
- **CairoMakie** figure gallery rendered from live fits, including the
  **Confidence Eye** (pale compatibility lens + outline + hollow estimate).
- Landing page, capability matrix, family guide, and tutorials filled with
  executed examples and honest status tags.

Planned next: the R↔Julia bridge (`engine = "julia"`), beta-binomial (needs a
trials column — drmTMB has no standalone binomial), the bespoke families
(Tweedie / cumulative_logit / zero-one-inflated beta), wiring `src/experimental/`,
and the RCall numerical drmTMB-parity gate (#17).

[parity anchor: drmTMB v0.1.3 (2026-05-20)]
