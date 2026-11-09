# Spatial parity: Poisson fixed-range kernel and NB2 spatial sd-z restarts (2026-11-09)

Branch `claude/spatial-parity-20261109`, cut from `claude/wave10-binomial-restarts-20261109`
with `claude/triage-nonconv-20261109` (PR #1275) merged in. Draft only.

## Pre-registered rule (written before the new R run was read)

One caveat on "before reading numbers": the committed test of PR #1275 already quotes drmTMB
0.8.1's log-likelihood for the Poisson cell (-675.6946875) and Julia's free-range value (-674.46).
Those two numbers were visible when this note was written. The rule below is therefore fixed
before the fresh R run and before any run of the new Julia route, and the fresh R run is the
reference, not the quoted number.

### Part B: Poisson `spatial(1 | g, coords)` uses drmTMB's fixed-range kernel

Reference: drmTMB 0.8.1 (library `~/drm_campaign_20261108/Rlib` on Totoro), ML, on the committed
cell data `test/fixtures/nonconv_triage/data_poisson_spatial.csv` with the shared coordinates
`test/fixtures/gamma_sdz/coords.csv`, formula `y ~ x1 + x2 + x3 + spatial(1 | g, coords = coords)`.
Julia: `drm(...; coords)` on the same data with the same site order.

Agreement is required on all of:

1. `|logLik_Julia - logLik_R| <= 1e-4`.
2. Each fixed effect: `|beta_Julia - beta_R| <= 1e-3 * SE_R` (R's standard error is the scale).
3. Log spatial SD: `|log sd_Julia - log sd_R| <= 5e-3`.

Both fits must report convergence. If any item fails, the route is not declared aligned, and the
note says which item and by how much. The thresholds are not relaxed after the numbers are read.

The kernel is `exp(-d / r) + 1e-6 I` with `r` the median positive pairwise distance, built by the
same helper as the Gamma route (`_spatial_covariance_from_coords`). The old estimated-range
behaviour (a sixth parameter, `:range` block) is removed rather than kept behind an option: no
existing test needs it except those that assert a `:range` block, and those are updated.

NB2 intercept `spatial(1 | g)` has no coordinate route in Julia (it takes a precomputed `K` only),
so there is no estimated-range difference to remove there. The NB2 slope and q = 2 routes already
use the native kernel (spatial-kernel audit of 2026-11-08, rows 4 and 5).

### Part A: NB2 spatial sd-z optimiser restarts

Cell: `nb2|spatial|lab1|q2|sd-z|cm0|ML`. Pattern: `_q2_boundary_restarts`
(`src/sparse_laplace_glmm.jl`), including its acceptance rule: a restart is accepted only if it
improves the objective by more than 1e-5; a converged fit is replaced by an unconverged one only
for a gain above 1e-3; the result is never worse than the original.

Acceptance:

1. The reproduction cell reaches logLik -554.0242 within 1e-4.
2. On 20 fresh seeds (same data-generating design, new seeds), report logLik, converged flag and
   run-time ratio before and after. Any seed with a lower logLik after than before is reported and
   blocks the change until explained.
3. The other NB2 and Poisson structured sd-z routes are checked on the same seeds; a route is
   changed only if the evidence shows the same stuck-lower-mode pattern (a restart gains more
   than 1e-3 logLik on at least one seed).

Part A, as pre-registered, was written for the NB2 route only. During the work the same-seed
check (item 3) turned up the same pattern on the Poisson q = 2 routes, so the rule for them is
the one in item 3: change a route only if a restart gains more than 1e-3 on at least one seed.

## Results (appended after the run; the rule above was not edited)

### Part B: Poisson fixed-range kernel

drmTMB 0.8.1 reference (Totoro, `2026-11-09-spatial-parity-evidence/ref_pois.R`): logLik -675.6946875075, converged
(code 0), pdHess TRUE. Julia, same data and site order (Mac, Julia 1.10.0; the committed test asserts the same rule on 1.10.12 and 1.12):

| Item | Rule | Measured | Verdict |
|---|---|---|---|
| logLik | \|dLL\| <= 1e-4 | 1.8e-9 | pass |
| Fixed effects (4) | <= 1e-3 R SE | 5.7e-6, 5.6e-7, 1.4e-7, 6.6e-7 R SE | pass |
| Log spatial SD | <= 5e-3 | 1.8e-7 | pass |
| Convergence | both converged | Julia converged, 5 parameters | pass |

Before the change Julia reported -674.46 with a sixth parameter (the range).

### Part A: NB2 spatial q = 2 sd-z

Reproduction cell `nb2|spatial|lab1|q2|sd-z|cm0|ML`: before -554.212716, after -554.024198
(target -554.0242, difference 2e-6, converged). The lower mode differs from the better one
mainly in the sign of the z coefficient of the intercept SD (-0.329 against +0.42). The old
alternative starts only tilted the slope-SD block, which is why they never reached it.

Twenty fresh seeds (91001 to 91020; G = 40 sites, 400 rows, NB2 size 2, correlated intercept and
slope fields, rho 0.3, SDs depend on a group covariate z; generator and raw tables in
`2026-11-09-spatial-parity-evidence/`, files `nb2_seeds.jl` and `nb2_spatial_q2_sdz_*.tsv`). Julia 1.10.0 (the Mac default), one
thread, both arms ran at the same time on the same machine.

| seed | logLik before | logLik after | change | converged b/a | secs b/a |
|---|---|---|---|---|---|
| 91001 | -588.6237 | -587.9568 | +0.6670 | true / true | 13.0 / 16.1 |
| 91002 | -618.8329 | -618.8329 | +0.0000 | true / true | 0.7 / 4.3 |
| 91003 | -783.9249 | -783.9249 | +0.0000 | true / true | 9.7 / 13.1 |
| 91004 | -595.3970 | -595.3970 | +0.0000 | true / true | 0.8 / 4.5 |
| 91005 | -640.4308 | -640.4308 | +0.0000 | true / true | 0.9 / 5.4 |
| 91006 | -735.9402 | -735.9402 | +0.0000 | true / true | 0.8 / 4.4 |
| 91007 | -665.6162 | -665.6162 | +0.0000 | true / true | 1.9 / 10.5 |
| 91008 | -768.8899 | -768.8899 | +0.0000 | true / true | 0.8 / 4.2 |
| 91009 | -651.6455 | -651.6455 | +0.0000 | true / true | 0.8 / 6.5 |
| 91010 | -559.7149 | -559.7149 | +0.0000 | true / true | 9.5 / 13.1 |
| 91011 | -620.4112 | -620.4112 | +0.0000 | true / true | 2.7 / 12.5 |
| 91012 | -694.3451 | -694.3451 | +0.0000 | true / true | 0.7 / 3.8 |
| 91013 | -627.6255 | -627.6255 | +0.0000 | true / true | 12.2 / 18.8 |
| 91014 | -757.8015 | -757.8015 | +0.0000 | true / true | 0.8 / 4.1 |
| 91015 | -710.9990 | -710.9990 | +0.0000 | true / true | 8.3 / 14.1 |
| 91016 | -554.2440 | -554.2440 | +0.0000 | true / true | 1.1 / 5.1 |
| 91017 | -727.2805 | -727.2805 | +0.0000 | true / true | 1.6 / 7.5 |
| 91018 | -660.1169 | -660.1169 | +0.0000 | true / true | 1.2 / 6.5 |
| 91019 | -751.0343 | -750.8027 | +0.2316 | true / true | 0.8 / 5.8 |
| 91020 | -687.0056 | -687.0056 | +0.0000 | true / true | 1.6 / 8.5 |

Two seeds improve (91001 by 0.667, 91019 by 0.232); eighteen are unchanged to the printed
digits; none is worse; every fit converged before and after. Total time 69.9 s before, 168.8 s
after (ratio 2.41). The ratio per fast seed is about 5 because four extra optimisations run
whatever the first fit did. A first version with ten extra starts (the four tilts plus the six
existing alternatives) gave the same logLik on every seed at a total ratio of 4.1, so the six
were dropped from the forced pass. The boundary rescue (existing alternatives, only when the
first fit ends at a boundary or unconverged) runs as before.

### Same-seed check of the other structured sd-z q = 2 routes

Probe: refit from four tilted copies of each fit's own SD coefficients (z coefficient of each
block set to plus or minus 2 / max |z|) and report the best logLik gain. Same seeds and design,
Poisson draws for the Poisson routes; relmat uses the same kernel matrix as `K`. Julia 1.12.6.

| Route | Seeds with gain > 1e-3 before | Gains | After the change |
|---|---|---|---|
| NB2 spatial lab1 (this PR) | 2 of 20 | 0.667, 0.232 | 0 |
| NB2 relmat lab1 | 0 of 20 | none | not changed |
| Poisson spatial lab1 | 2 of 20 | 0.081 (91007), 0.143 (91015) | 0; both seeds reach the tilted mode, none worse |
| Poisson relmat lab0 | 1 of 20 | 0.098 (91016) | 0; none worse |

So the Poisson q = 2 fitter (spatial, relmat and, through the same code, animal) now also tries
the four tilts for `sd-z` designs, kept only if the objective improves by more than 1e-6. No
seed got worse and no converged flag changed (the timing in the probe files includes the probe's
own local optimisations, so it is not a fit-time ratio). NB2 phylo and animal, Poisson phylo,
and the correlated variants not listed were not probed. Poisson animal was not probed
separately; it shares the relmat code path.
