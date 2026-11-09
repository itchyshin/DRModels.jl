# Wave-8 log-likelihood disagreements: classification of the 54 cells, 2026-11-09

Reader: whoever owns the R-Julia parity lane. Purpose: say, for each of the 54 both-fit cells where Julia (DRM.jl wave 8, `479bbf963`, PR #1270) and R (drmTMB 0.8.1) log-likelihoods differ by more than 1e-4, whether the gap is a convention, a deliberate model difference, one engine sitting at a lower optimum, or a Julia defect. The 54 cells come from the drmTMB branch `claude/gap-map-wave8-20261108`, folder `docs/dev-log/parity-crosswalk-evidence-20261107/fuzz-gap-map-wave8/`. No code in `src/` or `test/` was changed.

## Bottom line

- **No cell is a Julia log-likelihood defect (class d: 0 of 54).** In every cell where I could compute the exact marginal likelihood, Julia's reported value equals it at Julia's own estimate (36 one-dimensional cells, at most 2e-3 off) or matches Julia's own objective at R's estimate (7 correlated cells, at most 2.3e-3 off). For the 3 correlated `sd-none` cells (11, 19, 53) the parameter vectors do not line up, so only Julia's own optimum could be compared with the exact value at R's estimate (+0.035, +0.006, -0.001).
- **Class b, 47 cells.** 46 are an integrator difference: R's Laplace approximation against Julia's adaptive Gauss-Hermite quadrature (AGHQ). Exact quadrature shows Julia is the accurate one: R's Laplace value is below the exact value in 36 cells and above it in 10 (the 4 Poisson and Gamma slope-only cells and the 6 Tweedie cells). 1 cell fits a different model: Julia's native `spatial()` estimates a range that R fixes.
- **Class c, 4 cells.** 2 binomial spatial cells where R stops 1.5 to 1.8 nats below a point its own objective scores higher; 2 binomial independent-slope cells where Julia stops 1.3 to 2.0 nats below a point its own objective scores higher (probably a single-start optimiser on a ridge; I did not test restarts; minimal reproduction below). On the Julia side this looks like an optimiser limitation; I did not verify that Julia's value is right at the extreme point (see Unverified).
- **Class a (convention), 2 cells.** Binomial `animal(1 + x | g)` q=2 (cells 4, 7): R reads the unlabelled form as independent intercept and slope and pins the correlation at 0; Julia always estimates it. With the label `(1 + x | p | g)` R fits it too and both engines agree to 1e-8. This is a parity gap in Julia (it ignores the label for `animal` q=2), not a likelihood defect. No constant offset was found anywhere. The earlier README's "unexplained" Tweedie ordinary and slope groups (6 cells) are class b: same density in both engines (identical fixed-effect fits, logLik -790.0291), Julia's value is the exact marginal and R's Laplace value is 0.03 to 0.08 higher.
- **Class e, 1 cell (cell 9, binomial independent q=4 `sd-z`).** Exact quadrature puts the likelihood 3.6 to 4.5 nats below both engines' values; neither integrator is accurate in this near-singular region (details in the cell 9 entry under Unverified).
- In the integrator cells the two engines also reach essentially the same parameter estimates: R's own objective at Julia's estimate is within 0.015 of R's optimum in all 36 one-dimensional cells, and within 0.11 in the correlated cells whose parameter vectors line up. Only the reported value differs.

| Class | Cells |
| --- | --- |
| a, convention (animal label handling; parity gap in Julia) | 2 |
| b, integrator (Laplace vs AGHQ) | 46 |
| b, different model (Julia estimates the spatial range) | 1 |
| c, one engine at a lower optimum | 4 (2 R lower, 2 Julia lower) |
| d, Julia defect | 0 |
| e, neither engine accurate (cell 9) | 1 |

## How each cell was tested

All runs on Totoro through the existing ControlMaster socket, Julia 1.10.12, R drmTMB 0.8.1 from `~/drm_campaign_20261108/Rlib` (the library the original map used), one thread per process, `nice -n 10`, at most 22 processes at once. Estimated up front: under 10 minutes per pass. Actual: Laplace pass about 5 minutes, cross-evaluation 2 minutes, exact quadrature under 1 minute.

1. **Laplace re-run (T1).** The 54 cells were rerun in Julia with `marginal = :Laplace`. Only the 6 plain `(1 | g)` cells accept it (Beta, Binomial, Gamma, NB2, Poisson, Student, `sd-none`, ordinary, q=1): all 6 reproduce R to at most 3e-9 (for example Beta 124.9158693215 on both sides). The other 48 refuse with "supports `marginal = :LA` only" or "not available for `(1 + x1 | g)`" (Tweedie does not accept the keyword), so they need T2 and T3.
2. **Cross-evaluation (T2).** Each cell was fitted in both engines. R's objective was evaluated at Julia's estimate (`obj$fn` on the retained TMB object after `drm_pin_tmb_object_to_optimum`, the same step `objective_at()` uses), and Julia's objective at R's estimate (`fit.nll`). The parameter vectors line up position by position for the mean, sigma, nu and SD blocks: fixed-effect fits give identical vectors and logLik in both engines for NB2, Gamma, Beta, Student and Tweedie. For correlated `sd-none` blocks the vectors do not line up (Poisson q=2: R's objective at Julia's vector is 2.4 below R's optimum; Julia's at R's vector is 11.5 below), so T2 is not used there.
3. **Exact quadrature (T3).** For the 36 one-dimensional cells, an adaptive Gauss-Hermite integral with 100 nodes (60 for Tweedie) per group, centred at the conditional mode, written independently in R from the family densities (calibrated to the fixed-effect fits). Halving the node count changes the result by under 1e-8. For correlated ordinary blocks (q=2 to 4) and independent blocks, a tensor version (25, 13, 9 nodes per dimension for q=2, 3, 4) built from R's fitted group covariances. Evaluated at the engine estimates listed in the table.
4. **Specific probes (T4).** Julia objective with the extra animal correlation fixed at zero; Julia with R's fixed-range spatial kernel passed as `relmat`; one repro for class c.

## Groups

- G1: Beta, ordinary and slope-only, q=1
- G2: Student, ordinary and slope-only, q=1
- G4: Binomial, ordinary and slope-only, q=1
- G5: NB2, ordinary and slope-only, q=1
- G6: Poisson and Gamma, ordinary, q=1
- G7: Poisson and Gamma, slope-only, q=1
- G8: Tweedie, ordinary and slope-only, q=1
- G9: Poisson and NB2, plain `(1 + x1 | g)`, q=2
- G10: Binomial, correlated ordinary block, q=2 to 4
- G11: Binomial, independent intercept and slopes, sd-one q=2 and sd-z q=4
- G12: Binomial, independent intercept and slope, sd-z, q=2 and 3
- G13: Binomial, spatial, sd-z, q=1 and 2
- G14: Binomial, animal, q=2
- G15: Gaussian, spatial, sd-none

Per-group reading:

- **G1 to G8 (36 cells).** One random effect, exact value computable. At Julia's estimate Julia's ll equals exact quadrature (at most 2e-3 off; within 2e-4 on the SD-formula routes; on the plain `sd-none` route 3 of 8 cells, Beta, Poisson and Gamma, are 0.8 to 1.8e-3 off and the rest about 0; I did not establish why). R's Laplace value is lower than exact by 0.06 to 0.25 for Beta, Student, Binomial, NB2, Poisson (ordinary) and Gamma (ordinary), and higher than exact by 0.03 to 0.17 for Poisson and Gamma slope-only and for Tweedie. The sign depends on the family and the term. Julia's `sd-one` and `sd-none` cells are the same model with the same value in each engine (R -744.6958 for both, for example), so the plain-cell Laplace re-run (T1) also covers the `sd-one` cells.
- **G9 (2 cells).** Julia's plain correlated block uses AGHQ; the same model written with SD formulas (`sd-one`, `sd-z`) uses Laplace in Julia and agrees with R to 1e-10 (Poisson q=2 `sd-one` -712.4434 on both sides, NB2 `-699.8291`). Exact quadrature at R's estimate: Poisson -712.4096 (Julia -712.4106, R Laplace -712.4434), NB2 -699.5928 (Julia -699.5866, R -699.8291). Julia uses two integrators for one model (0.03 apart for Poisson, 0.24 for NB2); the AGHQ value is the accurate one.
- **G10, G11 (9 cells).** Binomial q=2 to 4 blocks (cell 11, `sd-none`, is covered through its twin cell 12: same R and Julia values). Exact quadrature at R's estimate against Julia's objective at the same point (the exact values are converged: 1.5x and 2x the node counts move the q=2 values by under 1e-8 and the q=3 and q=4 values by under 2e-5): q=2 `sd-one` -254.9217 against -254.9217, `sd-first` -254.7336 against -254.7336, `sd-z` -254.7349 against -254.7350, `sd-last` -254.9299 against -254.9302, independent `sd-one` -255.5294 against -255.5294; q=3 `sd-first` -253.0528 against -253.0540; q=4 `sd-first` -252.7467 against -252.7490 (AGHQ node limit at four dimensions). R's Laplace value is 0.25 to 0.54 below the exact value. Cell 9 has no exact value; see its row.
- **G12 (class c, Julia lower).** See below.
- **G13 (class c, R lower).** See below.
- **G14 (convention gap).** R documents that `animal`, `phylo`, `spatial` and `relmat` `(1 + x | g)` terms fit independent intercept and slope fields with no correlation, and fits the correlation only for the labelled `(1 + x | p | g)`. Julia's binomial `animal` q=2 route always estimates it (7 parameters against R's 6; estimate 1.83, standard error 3.8). Julia's objective at R's estimate with the correlation set to zero equals R's ll to 1e-9 (-255.85246113882 against -255.8524611388; -254.88524414 against -254.8852441418). With the label, R fits 7 and 9 parameters and gives -254.408130114 and -253.98079180889; Julia gives -254.40813011126 and -253.98079181798, so the engines agree to 1e-8 on the same model. The binomial phylo and spatial q=2 routes in Julia honour the label, so the animal route is the odd one out (read from source by a probe; Julia's intent is unconfirmed). The q=1 animal cells agree with R to 1e-10.
- **G15 (different model).** Julia's native `spatial(1 | g)` estimates the range jointly (documented in its docstring); the drmTMB bridge instead sends the fixed-range kernel `K = exp(-d / median(d)) + 1e-6 I`. Julia's `relmat` with that kernel gives -393.17916011678, R's value to 1e-10. The `sd-one` and `sd-z` Gaussian spatial cells already agree to 0.0 for the same reason.

## Class c details

**G13, binomial spatial `sd-z`, q=1 and q=2 (cells 1, 3): R at a lower optimum.** R reports `pdHess = FALSE` and `is_converged() = FALSE` and ends at -263.3873 in both cells, the same number, with SD coefficients near -12 (spatial variance switched off). R's own objective at Julia's estimate is -261.5872 and -261.8340, matching Julia's ll to 0.0000 and 0.0144, and Julia's objective at R's estimate is -263.3873, equal to R's ll. So the two objectives agree at both points; R's optimiser stopped early. Not a Julia defect.

**G12, binomial independent `(1 | g) + (0 + x1 | g)` with `sd(...) ~ z` on both terms, q=2 and q=3 (cells 2, 5): Julia at a lower local optimum, on an exact-likelihood plateau.** R's estimate has intercept log-SD -54.26 + 56.95 z: the intercept variance is zero except in the two groups with z of 0.95 and 1 (SD 0.9 and 14.7). Group g12 (z = 1) has y = 1 in all 10 rows, so its marginal tends to 0.5 as its intercept SD grows: the exact likelihood rises monotonically along the ridge (exact -253.20809 at SD 14.7, -253.07936 at SD 1.04e4, limit -253.07919) and has no finite maximum, which fits R's `is_converged() = FALSE`. Julia's objective at R's point is -253.2382 (exact there -253.20809, 0.030 low; R's Laplace -253.5291, 0.321 low), 1.99 above Julia's own optimum of -255.2241 (exact there -255.22394, 0.0002 off). Julia has two local optima in cell 2: the interior one, with intercept log-SD -0.93 + 0.76 z, which `drm()` returns, and a ridge one at -253.14505 (SD 52.9 at z = 1, gradient 1.4e-7), reached by a finite-difference BFGS and Nelder-Mead on `fit.nll` from R's estimate and from 14 perturbed R starts. All 8 starts near Julia's optimum return to -255.22414. Julia's AGHQ loses accuracy as the SD grows (low by 0.03 at SD 14.7, 0.049 at 52.9, 0.141 at 390, 0.220 at 2015), so its finite ridge maximum comes from the quadrature, not from the likelihood. Julia refuses `DRMControl(start = ...)` on mixed models (its own error cites issue #1219), so there is no supported way to restart `drm()` from the ridge. Cell 5 (q=3): exact at R's estimate -253.20809 (Julia 0.030 low); Julia has at least three further local optima, all above its reported -254.5596, the best found being -252.75123 (exact there -252.70625), 1.81 above it; the global maximum for q=3 was not established. I did not change code: a fix is a multi-start feature (and a better large-SD integrator), not a small change. Minimal reproduction: `docs/dev-log/evidence/wave8-disagreements-20261109/repro_g12.jl` (self-contained, committed data), output:

```
Julia optimum:  ll = -255.2241433762659   converged = true
                sd-block theta = [-0.926, 0.758, -0.932, 0.068]
Julia objective at the drmTMB estimate: ll = -253.2381590624687
```

## Unverified

- **Cells 2 and 5:** the global maximum is not established (cell 2 has an unbounded exact plateau; cell 5 had 20 restarts and only 2 reached the best local optimum, -252.75123). The restart and exact-quadrature runs are on Totoro in `~/disagree_w8_20261109/ridge/` and are not in this repository.
- **Cell 9** (binomial independent q=4 `sd-z`): exact marginal computed by quadrature on the independent effects (terms with SD below 1e-6 dropped). At R's estimate it is -253.742 (R Laplace -250.138, Julia's objective there -249.587); at Julia's estimate -254.029 (Julia AGHQ -249.558, R's objective there -250.177). Both engines overstate the likelihood by 3.6 to 4.5 nats, far more than the 0.58 gap between them. The error is concentrated in group g12, whose intercept SD is about 9000 and x3 slope SD about 740 (Laplace -1.51 against exact -5.63 at Julia's estimate). Neither integrator is accurate in this regime and both optimisers run coefficients past 50 in size, so cell 9 is not an integrator difference with Julia accurate; I class it e. The exact value for R's g12 is good to about 0.005 nats (grid and hcubature disagree by that much; a nested-integral run was killed at 14.5 minutes). The cross-evaluation still shows no likelihood defect.
- **Cells 11, 19, 53** (correlated `sd-none`): parameter vectors do not line up, so T2 is not usable; classed from the exact value at R's estimate and from the equal-model `sd-one` twins.
- **For the 10 correlated cells with an exact value it is computed at R's estimate, not Julia's** (Julia's covariance parameterisation was not decoded), and compared with Julia's objective at that same point (cells 12 to 16, 8, 10) or, where the vectors do not line up (cells 11, 19, 53), with Julia's value at its own optimum. Julia's own optimum is above the exact value at R's estimate by 0.03 to 0.08 in the binomial cells and by 0.006 in the NB2 cell; in the Poisson cell Julia is 0.001 below it. These are different points, so no sign is expected.
- **Julia 1.12.7**: cells 12, 21, 41, 18, 2, 4 refitted with dependency versions re-resolved (not pinned to the 1.10 manifest): ll equal to 1.7e-10 or better, theta to 1e-8 (cells 12, 21, 41, 18, 2: identical or 3e-14), so the gaps are not a Julia-version effect for those six. The other 48 cells were not rerun. One data set and seed per cell, as in the original map.

## All 54 cells

Columns: R and Julia log-likelihoods from the wave-8 fuzz; gap is Julia minus R; evidence is from the tests above.

| # | Cell | Group | Class | R ll | Julia ll | Gap | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `binomial\|spatial\|lab0\|q2\|sd-z\|cm0\|ML` | G13 | c | -263.3873 | -261.5872 | +1.8001 | R objective at Julia estimate -261.5872 vs Julia ll -261.5872 (diff 0.0000); Julia objective at R estimate -263.3873 vs R ll -263.3873 (diff 0.0000); R stops 1.80 nats below a point its own objective scores higher |
| 2 | `binomial\|indep\|lab0\|q2\|sd-z\|cm0\|ML` | G12 | c | -253.5291 | -255.2241 | -1.6950 | Julia objective at R estimate -253.2382, above Julia's own optimum -255.2241 by 1.99; R objective there -253.5291 |
| 3 | `binomial\|spatial\|lab0\|q1\|sd-z\|cm0\|ML` | G13 | c | -263.3873 | -261.8484 | +1.5389 | R objective at Julia estimate -261.8340 vs Julia ll -261.8484 (diff 0.0144); Julia objective at R estimate -263.3873 vs R ll -263.3873 (diff 0.0000); R stops 1.55 nats below a point its own objective scores higher |
| 4 | `binomial\|animal\|lab0\|q2\|sd-one\|cm0\|ML` | G14 | a | -255.8525 | -254.4081 | +1.4443 | Julia has 1 extra parameter (atanh rho); Julia objective at R estimate with rho = 0 equals R ll -255.8525 to 1e-9 |
| 5 | `binomial\|indep\|lab0\|q3\|sd-z\|cm0\|ML` | G12 | c | -253.5291 | -254.5596 | -1.0305 | Julia objective at R estimate -253.2382, above Julia's own optimum -254.5596 by 1.32; R objective there -253.5291 |
| 6 | `gaussian\|spatial\|lab0\|q1\|sd-none\|cm0\|ML` | G15 | b | -393.1792 | -392.2383 | +0.9409 | Julia relmat with R's fixed-range kernel gives -393.17916011678 = R ll (1e-10); native Julia spatial estimates range 1.105 (R fixes 3.256) |
| 7 | `binomial\|animal\|lab0\|q2\|sd-z\|cm0\|ML` | G14 | a | -254.8852 | -253.9808 | +0.9045 | Julia has 1 extra parameter (atanh rho); Julia objective at R estimate with rho = 0 equals R ll -254.8852 to 1e-9 |
| 8 | `binomial\|ordinary\|lab0\|q4\|sd-first\|cm0\|ML` | G10 | b | -253.2909 | -252.6683 | +0.6226 | exact at R estimate -252.7467; Julia objective there -252.7490; R ll -253.2909 |
| 9 | `binomial\|indep\|lab0\|q4\|sd-z\|cm0\|ML` | G11 | e | -250.1383 | -249.5580 | +0.5804 | R ll -250.1383, Julia objective at R estimate -249.5866, R objective at Julia estimate -250.1775; both sd-z blocks have coefficients above 50 in size (R is_converged FALSE); no exact value (singular covariance at R estimate) |
| 10 | `binomial\|ordinary\|lab0\|q3\|sd-first\|cm0\|ML` | G10 | b | -253.5277 | -252.9782 | +0.5495 | exact at R estimate -253.0528; Julia objective there -253.0540; R ll -253.5277 |
| 11 | `binomial\|ordinary\|lab0\|q2\|sd-none\|cm0\|ML` | G10 | b | -255.2132 | -254.8870 | +0.3262 | exact at R estimate -254.9217; Julia ll -254.8870; R ll -255.2132 (parameter vectors not interchangeable on this route) |
| 12 | `binomial\|ordinary\|lab0\|q2\|sd-one\|cm0\|ML` | G10 | b | -255.2132 | -254.8870 | +0.3262 | exact at R estimate -254.9217; Julia objective there -254.9217; R ll -255.2132 |
| 13 | `binomial\|ordinary\|lab0\|q2\|sd-first\|cm0\|ML` | G10 | b | -255.0176 | -254.6983 | +0.3193 | exact at R estimate -254.7336; Julia objective there -254.7336; R ll -255.0176 |
| 14 | `binomial\|ordinary\|lab0\|q2\|sd-z\|cm0\|ML` | G10 | b | -255.0168 | -254.6982 | +0.3186 | exact at R estimate -254.7349; Julia objective there -254.7350; R ll -255.0168 |
| 15 | `binomial\|ordinary\|lab0\|q2\|sd-last\|cm0\|ML` | G10 | b | -255.1800 | -254.8751 | +0.3048 | exact at R estimate -254.9299; Julia objective there -254.9302; R ll -255.1800 |
| 16 | `binomial\|indep\|lab0\|q2\|sd-one\|cm0\|ML` | G11 | b | -255.7796 | -255.4850 | +0.2946 | exact at R estimate -255.5294; Julia objective there -255.5294; R ll -255.7796 |
| 17 | `beta\|ordinary\|lab0\|q1\|sd-one\|cm0\|ML` | G1 | b | 124.9159 | 125.1621 | +0.2462 | exact at Julia estimate 125.1621; Julia minus exact +0.0000; R minus exact -0.2462 |
| 18 | `beta\|ordinary\|lab0\|q1\|sd-none\|cm0\|ML` | G1 | b | 124.9159 | 125.1613 | +0.2454 | exact at Julia estimate 125.1621; Julia minus exact -0.0008; R minus exact -0.2462 |
| 19 | `nb2\|ordinary\|lab0\|q2\|sd-none\|cm0\|ML` | G9 | b | -699.8291 | -699.5866 | +0.2424 | exact at R estimate -699.5928; Julia ll -699.5866; R ll -699.8291 (parameter vectors not interchangeable on this route) |
| 20 | `beta\|ordinary\|lab0\|q1\|sd-z\|cm0\|ML` | G1 | b | 126.1364 | 126.3718 | +0.2354 | exact at Julia estimate 126.3718; Julia minus exact +0.0000; R minus exact -0.2354 |
| 21 | `poisson\|slope\|lab0\|q1\|sd-one\|cm0\|ML` | G7 | b | -757.0856 | -757.2575 | -0.1719 | exact at Julia estimate -757.2575; Julia minus exact +0.0000; R minus exact +0.1719 |
| 22 | `poisson\|slope\|lab0\|q1\|sd-z\|cm0\|ML` | G7 | b | -755.9330 | -756.0997 | -0.1667 | exact at Julia estimate -756.0997; Julia minus exact +0.0000; R minus exact +0.1667 |
| 23 | `gamma\|slope\|lab0\|q1\|sd-one\|cm0\|ML` | G7 | b | -617.6482 | -617.8088 | -0.1607 | exact at Julia estimate -617.8088; Julia minus exact +0.0000; R minus exact +0.1607 |
| 24 | `gamma\|slope\|lab0\|q1\|sd-z\|cm0\|ML` | G7 | b | -616.8826 | -617.0391 | -0.1564 | exact at Julia estimate -617.0391; Julia minus exact +0.0000; R minus exact +0.1564 |
| 25 | `student\|ordinary\|lab0\|q1\|sd-z\|cm0\|ML` | G2 | b | -520.3764 | -520.2462 | +0.1302 | exact at Julia estimate -520.2462; Julia minus exact -0.0000; R minus exact -0.1302 |
| 26 | `student\|ordinary\|lab0\|q1\|sd-none\|cm0\|ML` | G2 | b | -520.4095 | -520.2795 | +0.1300 | exact at Julia estimate -520.2794; Julia minus exact -0.0000; R minus exact -0.1301 |
| 27 | `student\|ordinary\|lab0\|q1\|sd-one\|cm0\|ML` | G2 | b | -520.4095 | -520.2795 | +0.1300 | exact at Julia estimate -520.2794; Julia minus exact -0.0000; R minus exact -0.1301 |
| 28 | `beta\|slope\|lab0\|q1\|sd-one\|cm0\|ML` | G1 | b | 79.9159 | 80.0341 | +0.1182 | exact at Julia estimate 80.0341; Julia minus exact +0.0000; R minus exact -0.1182 |
| 29 | `student\|slope\|lab0\|q1\|sd-one\|cm0\|ML` | G2 | b | -526.9645 | -526.8474 | +0.1172 | exact at Julia estimate -526.8473; Julia minus exact -0.0000; R minus exact -0.1172 |
| 30 | `student\|slope\|lab0\|q1\|sd-z\|cm0\|ML` | G2 | b | -526.7746 | -526.6575 | +0.1171 | exact at Julia estimate -526.6575; Julia minus exact -0.0000; R minus exact -0.1171 |
| 31 | `beta\|slope\|lab0\|q1\|sd-z\|cm0\|ML` | G1 | b | 80.8031 | 80.9157 | +0.1126 | exact at Julia estimate 80.9157; Julia minus exact +0.0000; R minus exact -0.1126 |
| 32 | `binomial\|slope\|lab0\|q1\|sd-z\|cm0\|ML` | G4 | b | -257.0377 | -256.9337 | +0.1040 | exact at Julia estimate -256.9337; Julia minus exact -0.0000; R minus exact -0.1040 |
| 33 | `binomial\|slope\|lab0\|q1\|sd-one\|cm0\|ML` | G4 | b | -257.0384 | -256.9348 | +0.1036 | exact at Julia estimate -256.9348; Julia minus exact -0.0000; R minus exact -0.1036 |
| 34 | `nb2\|ordinary\|lab0\|q1\|sd-one\|cm0\|ML` | G5 | b | -702.1252 | -702.0341 | +0.0910 | exact at Julia estimate -702.0341; Julia minus exact +0.0000; R minus exact -0.0910 |
| 35 | `nb2\|ordinary\|lab0\|q1\|sd-none\|cm0\|ML` | G5 | b | -702.1252 | -702.0342 | +0.0910 | exact at Julia estimate -702.0341; Julia minus exact -0.0000; R minus exact -0.0910 |
| 36 | `nb2\|slope\|lab0\|q1\|sd-one\|cm0\|ML` | G5 | b | -706.8046 | -706.7157 | +0.0888 | exact at Julia estimate -706.7157; Julia minus exact +0.0000; R minus exact -0.0888 |
| 37 | `binomial\|ordinary\|lab0\|q1\|sd-none\|cm0\|ML` | G4 | b | -256.1458 | -256.0589 | +0.0869 | exact at Julia estimate -256.0589; Julia minus exact -0.0000; R minus exact -0.0869 |
| 38 | `binomial\|ordinary\|lab0\|q1\|sd-one\|cm0\|ML` | G4 | b | -256.1458 | -256.0589 | +0.0869 | exact at Julia estimate -256.0589; Julia minus exact -0.0000; R minus exact -0.0869 |
| 39 | `binomial\|ordinary\|lab0\|q1\|sd-z\|cm0\|ML` | G4 | b | -255.8732 | -255.7899 | +0.0833 | exact at Julia estimate -255.7897; Julia minus exact -0.0002; R minus exact -0.0835 |
| 40 | `tweedie\|ordinary\|lab0\|q1\|sd-z\|cm0\|ML` | G8 | b | -766.5357 | -766.6167 | -0.0810 | exact at Julia estimate -766.6167; Julia minus exact +0.0000; R minus exact +0.0810 |
| 41 | `tweedie\|ordinary\|lab0\|q1\|sd-none\|cm0\|ML` | G8 | b | -766.5668 | -766.6477 | -0.0809 | exact at Julia estimate -766.6477; Julia minus exact +0.0000; R minus exact +0.0809 |
| 42 | `tweedie\|ordinary\|lab0\|q1\|sd-one\|cm0\|ML` | G8 | b | -766.5668 | -766.6477 | -0.0809 | exact at Julia estimate -766.6477; Julia minus exact +0.0000; R minus exact +0.0809 |
| 43 | `poisson\|ordinary\|lab0\|q1\|sd-one\|cm0\|ML` | G6 | b | -744.6958 | -744.6181 | +0.0777 | exact at Julia estimate -744.6181; Julia minus exact +0.0000; R minus exact -0.0777 |
| 44 | `poisson\|ordinary\|lab0\|q1\|sd-z\|cm0\|ML` | G6 | b | -744.6532 | -744.5756 | +0.0776 | exact at Julia estimate -744.5756; Julia minus exact +0.0000; R minus exact -0.0776 |
| 45 | `gamma\|ordinary\|lab0\|q1\|sd-one\|cm0\|ML` | G6 | b | -574.2828 | -574.2062 | +0.0766 | exact at Julia estimate -574.2062; Julia minus exact +0.0000; R minus exact -0.0766 |
| 46 | `gamma\|ordinary\|lab0\|q1\|sd-z\|cm0\|ML` | G6 | b | -574.2203 | -574.1440 | +0.0763 | exact at Julia estimate -574.1440; Julia minus exact +0.0000; R minus exact -0.0763 |
| 47 | `poisson\|ordinary\|lab0\|q1\|sd-none\|cm0\|ML` | G6 | b | -744.6958 | -744.6198 | +0.0760 | exact at Julia estimate -744.6181; Julia minus exact -0.0018; R minus exact -0.0777 |
| 48 | `gamma\|ordinary\|lab0\|q1\|sd-none\|cm0\|ML` | G6 | b | -574.2828 | -574.2076 | +0.0752 | exact at Julia estimate -574.2062; Julia minus exact -0.0014; R minus exact -0.0766 |
| 49 | `nb2\|ordinary\|lab0\|q1\|sd-z\|cm0\|ML` | G5 | b | -698.2373 | -698.1651 | +0.0722 | exact at Julia estimate -698.1651; Julia minus exact +0.0000; R minus exact -0.0722 |
| 50 | `tweedie\|slope\|lab0\|q1\|sd-none\|cm0\|ML` | G8 | b | -780.8964 | -780.9643 | -0.0678 | exact at Julia estimate -780.9643; Julia minus exact +0.0000; R minus exact +0.0678 |
| 51 | `tweedie\|slope\|lab0\|q1\|sd-one\|cm0\|ML` | G8 | b | -780.8964 | -780.9643 | -0.0678 | exact at Julia estimate -780.9643; Julia minus exact +0.0000; R minus exact +0.0678 |
| 52 | `nb2\|slope\|lab0\|q1\|sd-z\|cm0\|ML` | G5 | b | -703.9092 | -703.8490 | +0.0601 | exact at Julia estimate -703.8490; Julia minus exact +0.0000; R minus exact -0.0601 |
| 53 | `poisson\|ordinary\|lab0\|q2\|sd-none\|cm0\|ML` | G9 | b | -712.4434 | -712.4106 | +0.0328 | exact at R estimate -712.4096; Julia ll -712.4106; R ll -712.4434 (parameter vectors not interchangeable on this route) |
| 54 | `tweedie\|slope\|lab0\|q1\|sd-z\|cm0\|ML` | G8 | b | -777.0721 | -777.1023 | -0.0303 | exact at Julia estimate -777.1023; Julia minus exact +0.0000; R minus exact +0.0303 |

## Files

`docs/dev-log/evidence/wave8-disagreements-20261109/`: `cross_eval.tsv` (per cell: both ll, both cross-evaluations, exact values, Laplace re-run ll), `exact_md.txt` (correlated-block exact values), `repro_g12.jl` and `data/`, `scripts/` (R and Julia probes and the shell drivers; they expect a copy of the drmTMB `fuzz-gap-map-wave8` folder named `harness/`).

## Reproduce

The scripts were run from a Totoro directory `~/disagree_w8_20261109/` that held the drmTMB branch folder as `harness/`, `env.sh`, `jl_cell.jl`, `xj.jl`, `anim.jl`, `gsp.jl`, `repro_g12.jl` at its top level, and `r_cell.R`, `xr1.R`, `xr2.R`, `exact.R`, `exact_md.R`, `gsp.R` inside `harness/`. The shell drivers hard-code that layout; they are a record of what ran, not a portable package. Order: `run_x.sh` (fits and cross-evaluation), `run_x4.sh` (exact values; `run_x3.sh` is the earlier pass with a broken Julia-at-R step and a wrong first exact version, kept for the record), `run_md.sh` (correlated exact values), `gsp.R` then `gsp.jl` (writes and reads `K_fixedrange.csv`), `anim.jl`. Probe outputs not in the TSV are in `probes.txt`.
