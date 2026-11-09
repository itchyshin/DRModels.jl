# Residual refusals after the open lane PRs, 2026-11-09

Reader: whoever owns the R-Julia parity lane. Purpose: say which cells drmTMB 0.8.1 fits and DRModels.jl still refuses once the open lane PRs land, what is done about each signature, and what stays open and why.

## How the list was built

Base: the wave-10 gap map (tip `0bfcde2bb`, 3008 cells), 107 R-only cells (R fits, Julia refuses). Then, per PR, the cells it reports as flipped (PR bodies and `docs/dev-log/` of each branch; only #1278 ships a cell-level rerun table, the others were mapped by shape from their notes):

| Step | Cells |
|---|---|
| base R-only | 107 |
| #1278 non-Gaussian independent terms: flips to agreement | 48 |
| #1278: Binomial `sd-first` q2, q4 become honest non-converged fits | 2 |
| #1276 Gaussian REML (correlated q3-q4, crossed q2-q4, structured SD and bare q2 REML) | 19 |
| #1283 non-Gaussian correlated blocks (Poisson/NB2 one-target 4, LogNormal 4, Binomial q3-q4 2, fit but disagree with R) | 10 |
| #1274 structured SD counts (Poisson spatial 2, NB2 spatial 2, NB2 phylo 2) | 6 |
| #1286 animal slope (Beta animal/relmat bare 2, Binomial animal labelled 2) | 4 |
| #1273 Gaussian SD group F (mean+sigma correlated block with `sd(g) ~ z`) | 1 |
| **residual R-only** | **17** |

Caveat: the stacked branches were not merged and rerun together, so this is a subtraction by shape, not a measurement. The integration lane's rerun is the check. #1286 was merged into the work branch and its two Binomial animal cells re-ran here and match R (-254.40813, -253.98079).

Not in scope here: the 54 both-fit-disagree cells (Julia fits, log-likelihood differs from R by 0.03 to 1.5; older quadrature routes against R's Laplace). They are a bigger parity gap than the 17 residual refusals and are not R-only cells, so they stay with the owner of the SD-regression routes (see `2026-11-09-nongauss-indep-design.md`, verdict (e)).

## Residual 17, by signature, ranked by cells closed per hour

Effort is the estimate made before reading code in detail; "hours" is of focused work.

| Rank | Signature | Cells | Verdict | Effort | Cells/hour |
|---|---|---|---|---|---|
| 1 | Bare structured `animal`/`relmat`/`spatial(1 \| g)`, Binomial; bare `spatial(1 \| g)`, Beta (R fits the one-SD model; explicit `sd(g, kind) ~ 1` already agrees) | 4 | close | 0.5 | 8 |
| 2 | Bare Beta `phylo(1 + x1 \| g)` and `spatial(1 + x1 \| g)` (explicit `~ 1` forms already agree) | 2 | close | 0.2 | 10 |
| 3 | NB2 `phylo(1 + x1 \| p \| g)` labelled, no SD formula (explicit `~ 1` already agrees) | 1 | close | 0.2 | 5 |
| 4 | NB2 bare `spatial(1 \| g)` with `coords` | 1 | defer: the NB2 spatial route (src in #1274) is owned by the spatial-parity lane | 0.3 | n/a |
| 5 | Student `spatial(1 \| g)`, `spatial(1 + x1 \| g)` with `coords` (R fits 2 of 50 Student spatial cells) | 2 | defer: no Student spatial route; needs a kernel Laplace fit, also spatial-parity territory | 5 | 0.4 |
| 6 | Binomial REML, `(1 \| g)` and `(0 + x1 \| g)` (drmTMB integrates the fixed effects out by Laplace; Julia Binomial is ML-only) | 2 | defer: new objective (fixed effects as flat-prior latent variables), needs its own design | 4 | 0.5 |
| 7 | Gaussian `(1 \| p \| g)` on mean and sigma, REML | 1 | defer: restriction term for the location-scale correlated route | 3 | 0.3 |
| 8 | Binomial `relmat(1 + x1 \| g)` bare, `sd ~ 1`, `sd ~ z` | 3 | record: drmTMB `is_converged()` is FALSE on all three (x1 SD at -10.7) | 0 | n/a |
| 9 | Gaussian bare `spatial(1 \| g)` REML | 1 | record: Julia estimates the exponential range, drmTMB fixes it at the median positive distance; ML gap 0.94 is a kernel difference. The explicit `sd(g, spatial) ~ 1` REML form fits and agrees (#1276) | 0 | n/a |

Cells closed here: 7 (ranks 1 to 3). Recorded: 4. Deferred: 6 (ranks 4 to 7). Total 17. R is not converged on Binomial spatial q1 (rank 1) and Beta spatial q2 (rank 2) either; they are closed on log-likelihood and flagged, see the results.

## Pre-registered agreement rule (written before the reference run of this lane was read)

Honest disclosure: the R log-likelihoods of these cells were already visible in the gap map when the cells were chosen, and Julia's explicit `sd ~ 1` forms agreeing with them is what made them candidates. The coefficient and SE comparison below was not read before this section was written.

Reference: drmTMB 0.8.1, `engine = "tmb"`, Totoro library `~/drm_campaign_20261108/Rlib`, the gap-map datasets, script `test/fixtures/residual_refusals/reference.R`, output `reference.tsv`. A cell **agrees** when all hold:

- `|logLik_Julia - logLik_R| <= 1e-4`;
- every fixed-effect coefficient (mean and log sigma) within `1e-3`;
- every log SD (and correlation, where estimated) within `5e-3`;
- Julia reports `converged = true`, unless R reports `is_converged() = FALSE` for the same cell, in which case the verdict is "agrees at R's own non-converged point, flagged".

The limits are not to be widened after the results are read. A cell that misses one is not closed.

## Change

`src/structured_q2_phylo_spatial.jl`:

1. `_default_structured_q1_sd`: a formula with one bare structured intercept `animal|relmat|spatial(1 | g)` and no SD formula is completed with `sd(g, kind) ~ 1` for Binomial (animal, relmat, spatial) and Beta (spatial). Same conditions as the q2 completion: nothing else random on the mean, `sigma ~ 1`, ML.
2. `_default_q2_sd_families`: Beta added for `:phylo` and `:spatial`; a label on the bare phylo block is accepted for NegBinomial2 as it already is for Poisson.

`src/binomial.jl` and `src/beta.jl` call the new helper next to the q2 one. No likelihood code changes.

## Results

Reference: drmTMB 0.8.1 on Totoro, `reference.R`, committed with `reference.tsv`. Julia 1.10.12 on a Mac; the test passes on 1.10.12 and 1.12.6 (39 checks).

| Cell | dLL | max fixed-effect gap | max log-SD gap | Julia converged / R converged | Verdict |
|---|---|---|---|---|---|
| Binomial animal `(1 \| g)` | 4.5e-13 | 5.0e-9 | 1.4e-7 | yes / yes | (a) agrees |
| Binomial relmat `(1 \| g)` | 1.2e-11 | 8.3e-9 | 2.9e-8 | yes / yes | (a) agrees |
| Binomial spatial `(1 \| g)` | 2.9e-9 | 7.8e-9 | 8.6 | yes / no | (b) agrees on log-likelihood and fixed effects; both fits put the SD on the zero boundary (R log SD -11.4, SE 9331), so the log-SD limit is not met and is not applicable. Flagged |
| Beta spatial `(1 \| g)` | 7.7e-13 | 2.9e-8 | 1.4e-8 | yes / yes | (a) agrees |
| Beta spatial `(1 + x1 \| g)` | 4.8e-10 | 3.0e-8 | 1.4 | yes / no | (b) agrees on log-likelihood and fixed effects; R is not converged and the slope SD is at its boundary. Flagged |
| Beta phylo `(1 + x1 \| g)` | 5.6e-12 | 1.1e-8 | 8.9e-8 | yes / yes | (a) agrees |
| NB2 phylo `(1 + x1 \| p \| g)` | 3.7e-11 | 5.0e-8 | 3.5e-7 | yes / yes | (a) agrees |

Strict reading of the rule: 5 cells meet every criterion; 2 meet the log-likelihood and fixed-effect criteria and miss the log-SD one because the SD is on a boundary in R as well. They are counted as closed with a flag, not as clean agreements. Standard errors were not compared with R's (R's are unusable at the two boundary cells).

The new test `test/test_residual_refusals.jl` fails on the base (6 errors, 2 passes: the two guard checks) and passes now. Items checked by it besides the seven cells: completing the Binomial animal term gives the same fit as the explicit `sd(g, animal) ~ 1` form to 1e-8, a second random term keeps its refusal, and `method = :REML` on Binomial stays refused.

## Not verified here

- Totoro was unreachable after the reference run (ssh refused, control socket gone), so the full suite on Totoro (16 shards per version) is not in this note; see the PR for what was run on the Mac.
- The residual list is a subtraction by shape across unmerged PRs, not a combined rerun.
- The 6 deferred cells are untouched; the 4 recorded cells have no drmTMB number to match.
