# Class-1 twin parity: ordinary `(1 | g)`, DRModels.jl vs drmTMB

**Question.** The Class-1 twin issues (#709 Poisson, #710 Beta, #713 binomial, #714 Student,
#715 NegBinomial2, #716 Gamma; analyses #717/#718) report random-intercept fits where the two
engines disagree beyond 1e-5. Is that the integrator (DRModels default vs drmTMB's Laplace), or a bug?

**Answer.** The integrator, in all six families. On the SAME integrator (drmTMB's TMB Laplace vs DRModels
`marginal = :Laplace`) the twins agree to at most 5.4e-8 relative on every working-scale estimate and
at most 5e-11 absolute on logLik, on all ten regenerated cells. Student (#714), the one family
with no `:Laplace` option, got one in this PR; its sigma / nu / sd gap (2.8e-2 default vs drmTMB) is the
integrator too (Laplace vs Laplace: 1.2e-10). No bug in either twin was found.

## Inventory: integrators for an ordinary `(1 | g)` on the mean (main at d31b9f4cd)

| family | default (`:LA`) | `marginal = :Laplace` | `:AGHQ` | `:VA` |
|---|---|---|---|---|
| Poisson | per-group adaptive GHQ, 5 nodes (#719) | yes (ordinary_laplace.jl) | yes (`nAGQ`; 1 node = Laplace) | yes |
| Binomial | per-group adaptive GHQ, **3** nodes | yes | crossed only | yes |
| Beta | adaptive GHQ, 5 | yes (`sigma ~ 1`) | crossed only | yes |
| NegBinomial2 | adaptive GHQ, 5 | yes (`sigma ~ 1`) | crossed only | yes |
| Gamma | adaptive GHQ, 5 | yes (`sigma ~ 1`) | crossed only | yes |
| Student | adaptive GHQ, 5 | **no; added here** (`sigma`, `nu` may be formulas) | no | no |

The issues describe the default as "GHQ-32". That is stale: since #719 the 1-D default is per-group
**adaptive** GHQ with 5 nodes (3 for Binomial); `docs/src/model-guides/marginal-la-vs-va.md` still says
32-node non-adaptive. Defaults are untouched (D-273): default `theta` and `logLik` on all ten
cells are byte-identical to main (`repr` of the Float64 vectors compared with `cmp`).

## Parity table (`comparison.tsv`)

"ref" is the default integrator at 61 nodes per group (internal fitters), the best available proxy
for the true marginal likelihood. Estimates compared are every working-scale outer parameter
(mu coefficients, log sigma, nu coefficient, log sd_mu).

| cell | logLik drmTMB | logLik DRM `:Laplace` | max rel est (`:Laplace` vs drmTMB) | logLik DRM default | max rel est (default vs drmTMB) | logLik ref (61 nodes) | sd_mu drmTMB / default / ref |
|---|---|---|---|---|---|---|---|
| #709 Poisson | -105.096931 | -105.096931 | 8.0e-10 | -105.079772 | 9.2e-03 | -105.079266 | 0.5852 / 0.5881 / 0.5882 |
| #710 Beta | 47.882759 | 47.882759 | 2.1e-08 | 47.917726 | 2.6e-03 | 47.917750 | 0.3113 / 0.3123 / 0.3123 |
| #713 binomial (binary) | -63.427036 | -63.427036 | 9.6e-11 | -63.397075 | 6.0e-02 | -63.396756 | 0.4750 / 0.4965 / 0.4972 |
| #713 binomial (bacteria) | -101.102236 | -101.102236 | 1.9e-08 | -101.037224 | 8.4e-03 | -100.820515 | 1.2864 / 1.2866 / 1.3562 |
| #713 binomial (cbind) | -164.469716 | -164.469716 | 5.4e-08 | -164.458540 | 2.7e-03 | -164.458445 | 0.2616 / 0.2625 / 0.2625 |
| #714 Student | -105.952014 | -105.952014 | 9.3e-11 | -105.897512 | 2.8e-02 | -105.897478 | 0.3480 / 0.3502 / 0.3502 |
| #714 Student (30x10, sd .5) | -303.273557 | -303.273557 | 7.3e-11 | -303.054437 | 3.1e-02 | -303.054053 | 0.3432 / 0.3449 / 0.3449 |
| #714 Student (30x10, sd .8) | -271.203258 | -271.203258 | 1.2e-10 | -270.968429 | 1.4e-02 | -270.968001 | 0.8671 / 0.8675 / 0.8675 |
| #715 NB2 | -198.509004 | -198.509004 | 2.2e-08 | -198.497350 | 2.9e-03 | -198.497346 | 0.2646 / 0.2657 / 0.2657 |
| #716 Gamma | -119.476824 | -119.476824 | 5.5e-11 | -119.466070 | 8.9e-04 | -119.465975 | 0.3744 / 0.3748 / 0.3748 |

Reading it:

* **Same integrator.** `:Laplace` vs drmTMB: max relative estimate gap 5.4e-8 (cbind), most cells
  at or below 1e-10; |ΔlogLik| below 5e-11. Both ≪ the 1e-5 twin rule. This demonstrates parity for
  every family.
* **Which engine is closer to the truth.** The default (adaptive GHQ) is within 5e-4 nat of the
  61-node reference on every cell except bacteria; drmTMB's Laplace is 0.01 to 0.28 nat off it, and
  its sd_mu is 0.1% to 5.2% low. So the twin gap is the Laplace error in drmTMB, not a defect in the DRModels default.
* **Exception: bacteria (`MASS::bacteria`, sparse binary).** The default Binomial route (K = 3)
  is 0.22 nat and 5% (sd_mu 1.2866 vs 1.3562) off the reference, no closer than Laplace. This is a
  separate accuracy limit of the 3-node default on sparse binary data, not part of the twin gap.
  Not changed here (D-273); it deserves its own issue.

## Student (#714)

drmTMB `nu` is `2 + exp(beta_nu)`, as in DRModels, so the working-scale parameters line up. Laplace vs
Laplace agrees to 9.3e-11 / 7.3e-11 / 1.2e-10 (issue-shaped 12x8 cell and two 30x10 cells). The 4.4e-2 `sigma`
gap in the issue is large because Student sigma and nu are strongly coupled to the group variance
and the Laplace approximation is poorest for the heavy-tailed, non-log-concave per-group integrand; the
default and the 61-node reference agree with each other to 6e-5 relative. Not a bug.

The new route (`_fit_student_ordinary_laplace`) is per-group scalar Newton with an expected-information
fallback (the Student data term is not log-concave), observed curvature in the Laplace term, ForwardDiff
outer gradient. A first version stopped the inner Newton on a roundoff-limited line search
(`1e18` failure); fixed by accepting decreases below the roundoff of J and a gradient floor.
Tests: `test/test_student_ordinary_laplace.jl` (independent Laplace reference at θ̂ and a perturbed θ to 1e-7;
default-route objective at one adaptive node equals the Laplace objective to 1e-8; default route and
`:LA` identical; refusals).

## Data provenance (read this before citing the table)

The issues' burn-cell CSVs (`/workspace/drm-twin-grid-burn`) were not available. #713 binary uses the
issue's exact R simulator (drmTMB logLik -63.42703552 reproduced), and `MASS::bacteria` is the issue's
dataset (-101.10 reproduced). The other cells keep the issue's design (n, groups, formula, seed) but the DGP constants are
ours (see `native_fit.R`), so they reproduce the *class* of gap, not the issues' exact numbers.

## Reproduce

```
Rscript docs/dev-log/evidence/class1-laplace-parity/native_fit.R          # drmTMB 0.7.1, writes fixtures + native.tsv
julia --project=. docs/dev-log/evidence/class1-laplace-parity/julia_fit.jl # writes comparison.tsv (about 1 min)
```

Runs: Julia 1.10.10 on Totoro (`OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1`), R 4.6.0 / drmTMB 0.7.1 / TMB 1.9.21 on the Mac.
Full suite `DRM_TEST_SHARD=1/4..4/4` passed on Julia 1.10.10 and 1.13.1 (shard 2 re-run after the
bridge-test fix below).
