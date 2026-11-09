# Animal / relmat intercept-plus-slope routes: scope, thresholds, verdicts (2026-11-09)

Reader: the owner of the R-Julia parity lane. Purpose: decide which of the wave-10 gap-map ranks 5 and 9 cells can be closed cheaply, say what the rest costs, and fix the agreement thresholds before the agreement run.

## Scope finding

Ranks 5 and 9 read as "L: new C++-equivalent code". Measured on the wave-10 tip, most of them are not. The animal and relmat intercept-plus-slope likelihoods already exist for every family involved; what is missing is route admission.

1. **Bare `animal|relmat(1 + x | g)` with no `sd()` formula.** In drmTMB 0.8.1 this is the two-SD model with constant SDs, the same model as `sd(g, level, coef(intercept)) ~ 1` plus `sd(g, level, coef(x)) ~ 1`. `_default_structured_q2_sd` (structured_q2_phylo_spatial.jl) already completes the bare form for Poisson, NB2, Gamma and Gaussian ML. Beta and Gaussian REML were not in its list. Measured before any change: Julia's explicit `~ 1` form already equals R's bare form (Beta animal 143.0657524500, Beta relmat 120.8833800786, Gaussian REML animal -464.6771053456, relmat -419.4355765760, all to 1e-10). Cost: S.
2. **Binomial `animal(1 + x | p | g)` with SD formulas.** The Binomial animal q2 Laplace fitter already has a `correlated` switch (used by the phylo route). The animal entry point never set it: it always fitted the correlated model, so the *unlabelled* form fitted a different model from R's independent-field model. The two Julia cells that "fit and disagree" (`binomial|animal|lab0|q2|sd-one` 1.44, `sd-z` 0.90 logLik units) are this bug, and R's labelled value equals what Julia's unlabelled route returned (-254.4081). Cost: S, and it fixes 2 disagreements as well as 2 refusals.

## Verdicts per cell

| Cell | Verdict |
| --- | --- |
| beta animal q2 sd-none (ML) | (a) fits and agrees |
| beta relmat q2 sd-none (ML) | (a) |
| gaussian animal q2 sd-none REML | (a) |
| gaussian relmat q2 sd-none REML | (a) |
| binomial animal lab1 q2 sd-one, sd-z (ML) | (a) |
| binomial animal lab0 q2 sd-one, sd-z (ML), previously disagreeing | (a) now agree |
| binomial relmat q2 sd-none, sd-one, sd-z | (c) no Binomial relmat q2 route exists; R has `is_converged() = FALSE` on all three. Would reuse the animal Laplace fitter with a raw-K kernel (S to M), but there is no converged reference to hold it to. Left refused. |
| beta spatial q2 sd-none | (b) R `is_converged() = FALSE`; needs a Beta spatial q2 route. Left refused. |
| gaussian spatial q2 REML | (c) Julia's spatial ML is 0.9 logLik from R (rank 39); fix ML first. Left refused. |

Left alone on purpose: Binomial animal bare (no SD formula) form, which R refuses; labelled bare forms except Poisson (R refuses); Binomial and Beta REML (ML-only by design).

## Pre-registered agreement thresholds

Written before any coefficient or SE comparison; the logLik agreement had already been seen in the probe run (it is why the cells were chosen). Same as the Beta structured SD study:

- T1, logLik: |julia - R| <= 1e-3.
- T2, coefficients (mu, sigma, SD blocks, atanh correlation): |diff| <= 5e-3.
- T3, standard errors: Julia SE / R SE in [0.95, 1.05], zero non-finite Julia SEs.
- Reference: `test/fixtures/animal_slope/reference.tsv`, made by `reference.R` with drmTMB 0.8.1 on Totoro (`reference.R` says how to rerun). All eight R fits report `is_converged() = TRUE`.

## Label semantics (drmTMB 0.8.1 against Julia)

drmTMB 0.8.1: `kind(1 + x | g)` fits two independent fields (correlation pinned at 0); `kind(1 + x | p | g)` estimates the intercept-slope correlation. Measured on the wave-8 data, Binomial animal with `sd ~ 1`: R unlabelled -255.8525, labelled -254.4081.

Julia, from the 200-cell probe on the wave-10 tip: Gaussian, Poisson and the phylo/spatial Binomial route already honour the label (unlabelled and labelled logLik differ, e.g. Poisson animal -662.249 against -660.279). The one route that ignored it was the Binomial animal q2 entry point, which always fitted the correlated model. This is the cause of the two Binomial animal disagreements the disagreement report lists (cells 4 and 7). Fixed here: the label now selects `correlated` on that route; unlabelled gives R's independent-field logLik to 2e-9 and labelled to 3e-9. Existing Binomial tests that generated correlated data and fitted the unlabelled form now use the labelled form. Beta accepts only the unlabelled form (R refuses the labelled Beta form too). Relmat and spatial Binomial q2 are not routed (see verdicts). Not checked beyond the animal routes.
