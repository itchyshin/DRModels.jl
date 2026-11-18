# Integration decisions, 2026-11-09

This note records how the lane pull requests were combined on
`claude/integration-final-20261109`, which starts from the tip of the wave-10 base
(`claude/wave10-binomial-restarts-20261109`, commit 0bfcde2bb). The pull requests were merged
one at a time, in the order below, and the test files each one touched were run on Julia 1.10
after its merge, so that a semantic clash is attributed to the right pull request.

Merge order: #1273, #1280, #1275, #1274, #1276, #1278, #1283, #1286, #1277, #1279, #1282, then
the three fixes that are based on main: #1284, #1285 and #1287.

Rule used throughout: where two pull requests changed the same behaviour, the version with the
stricter test or the wider checked coverage was kept, and the weaker test was dropped or folded
into it.

## Clean merges

#1273, #1280, #1275, #1274, #1284 and #1287 merged without a source or test conflict. The
shard assignment file conflicted in almost every merge and is handled in the last section.

## Conflicts and how they were settled

### #1276 (Gaussian REML routes, second campaign) against #1273 (Gaussian `sd()` group F)

`src/gaussian_corr_crossed_sd.jl` was rewritten by #1273 to take several correlated blocks and
slope terms, with a new fit function. #1276 had added REML to the old single-block version.
The #1273 structure was kept and the #1276 REML objective was ported onto it: one dense
objective with a `restricted` switch, the same Patterson-Thompson term, and `_withreml` on the
result. REML is admitted only where #1276 compared it with drmTMB: exactly one correlated
block beside random intercepts on other groupings. Two blocks, or a block beside a slope term,
keep a maximum-likelihood-only refusal, because no REML comparison exists for them. The refusal
text keeps the phrase `method = :REML is not implemented` and the word `INTERCEPTS`, which the
older and newer tests both look for.

`src/gaussian_ordinary_q34_sd.jl` carried an unconditional REML refusal from the wave-9 base.
#1276 removed it and added a gate that refuses REML when `sd()` formulas are present. The
gate was kept and the unconditional refusal dropped.

`src/gaussian_core.jl` conflicted only in the text of one error message. The #1276 text was
kept, because it lists the REML routes that now exist.

`docs/src/capabilities.md` was combined by hand: the wave-9 and wave-10 refusal list from the
base, the #1276 wording about the bare `spatial(1 | g)` (it estimates the range where drmTMB
fixes it), and the new REML routes.

`test/test_reml.jl` and `test/test_reml_ordinary_ranef.jl`: the base asserted that a block
beside `(1 | h)` refuses REML. That shape now fits. The #1276 tests were kept, which assert
refusal for a block beside a slope term `(0 + x | h)`, the case that is still refused.

`test/test_gaussian_ordinary_q3_q4_sd.jl` (from #1280) asserted that a bare q = 3 block refuses
REML. #1276 makes it fit. The assertion was changed to require a converged REML fit, and the
refusal with `sd()` formulas stays.

### #1278 (non-Gaussian independent terms) against #1274 (structured SD counts)

Both added a route for several `sd()` formulas.

- Student: #1274 routes two targets to a crossed-intercept function that was checked against
  drmTMB. #1278 routes every independent-term shape to a shared Laplace route. The merged
  dispatcher sends the crossed pair of two random intercepts to the #1274 function and every
  other independent shape to the #1278 route.
- LogNormal: #1274 delegates all multi-`sd()` fits to the public Gaussian dispatcher on
  `log(y)`. #1278 added a narrower crossed function. The #1274 delegation was kept, since it
  covers the same shapes and inherits any Gaussian refusal.
- `src/indep_slopes_laplace.jl`: both pull requests added a noise-floor exit to the inner
  Newton loop, with different tests. Both exits were kept; each returns a converged inner
  solve only when its own condition holds.
- `test/test_gamma_nb2_twofield_sd.jl`: the #1274 test (two SD blocks, the unselected one
  constant, and a refusal for a coefficient outside the pair) is stricter than the #1278
  test, which only checked a finite log-likelihood. The #1274 test was kept.

### #1283 (non-Gaussian correlated blocks) against #1278 and #1274

`src/lognormal.jl`: the correlated-block route of #1283 is tried first, then the #1274
multi-`sd()` delegation. Otherwise a two-target correlated block would be sent to the
delegation.

`docs/src/capabilities.md`: the Poisson, NegBinomial2, Binomial, Gamma and LogNormal rows were
combined so each row states both the independent-term coverage and the correlated-block
coverage. #1283 said that independent `(0 + x | g)` terms are refused for Binomial; the
merged rows say they are admitted, as #1278 implements and tests.

### #1286 (animal slope) against #1276

Both complete the bare `animal(1 + x | g)` term under REML. #1276 also covers `spatial` and
`relmat`; #1286 covers `animal` and `relmat`, and adds Beta to the ML routes. The merged rule
completes REML for Gaussian `animal`, `relmat` and `spatial`, and for no other family. The
test that checks the REML estimation method of the bare form was kept.

### #1285 against #1284

`test/test_binomial_spatial_sd_regression.jl`: #1284 builds the legacy mean-range kernel
explicitly, and #1285 needs the Cholesky factor of the same matrix. The merged test builds the
legacy kernel and then the factor.

### #1282 (fitted semantics) against #1277 (predict on new data)

There was no textual conflict, but a semantic one. #1282 makes `predict(fit, newdata)` on a
LogNormal fit return the log-scale mean by default, as drmTMB does (`type = :median` gives
`exp(mu)`, `type = :mean` gives `E[Y]`), and checks that against drmTMB 0.8.1 reference values.
The #1277 factor-level test assumed the older `exp` inverse link for LogNormal. The #1282
behaviour was kept, because it is the one with an R reference, and the #1277 test now uses the
identity link for the LogNormal case. All the new-data checks of #1277 (single row, subset of
levels, reordered levels) still apply to it.

### Later folds: #1289, #1291, #1292

#1289 (stable NB2 density at tiny sigma) merged cleanly. #1291 (residual refusals) conflicted in the
capabilities table and in `test_gap_close_q2.jl`: the Gaussian spatial REML assertion from #1276 was
kept, together with the new NegBinomial2 labelled phylo fit from #1291, and the Beta and Binomial spatial
rows take the #1291 text. #1292 (bivariate predict) conflicted only in the shard file. Left out of this
tip: the spatial-parity and Laplace-parity lanes and #1290 (its failures did not reproduce), and the
Codex PR #1281.

### Late folds: #1298 and #1299

#1298 (review fixes, documentation and test assertions only) merged cleanly. #1299 (spatial parity)
conflicted in `_q2_boundary_restarts`, which now takes both the `big_gain` argument from the wave-10
base and the `starts` argument from #1299, and in the capabilities table (the Poisson spatial row and
the NegBinomial2 labelled note were combined). The Poisson `coords` route now uses the fixed native
range, which is a user-visible default change.

## User-visible default changes on this tip

`fitted()` returns the response mean `E[Y]` for LogNormal, zero-inflated, hurdle and truncated fits,
and `predict()` for LogNormal returns the log-scale mean by default (#1282). REML now fits for
correlated blocks of 3 or 4 coefficients, a block beside random intercepts, structured intercepts with
an `sd()` formula, and the bare structured slope term (#1276, #1286). The new independent-term fits
for the non-Gaussian families (#1278) use the Laplace marginal and are tagged `marginal = :Laplace`,
while the older single-term routes stay on quadrature. An unlabelled Binomial `animal(1 + x | g)` now
means independent fields and a labelled one means correlated, consistent with the other families; this
change in #1286 was approved by Shinichi.

## Shard assignment

`test/shard_assign_N16.txt` conflicted in every merge, because each pull request appended its
own file to the same lines. It was regenerated once at the end, by a small script: the assignment lines were merged shard by shard, files that no
longer exist were dropped, duplicates were removed, and files that no pull request assigned
were added to the shard with the fewest files. The result lists every discoverable test file
exactly once. Discoverable files are the `test_*.jl` files in `test/` minus the four in
`_TEST_EXCLUDE`.
