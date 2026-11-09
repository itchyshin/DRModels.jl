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
