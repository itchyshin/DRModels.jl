# After-task: DRModels speed kinds toward 20 (2026-09-23)

## Scope

Fill the three-package speed board DRM column from diversity plan §4.2 first
wave, plus an earlier Mac Studio bridge/fixture bank. Lease
`cursor:DRM.jl-speed-kinds-toward20`. Soft A3 skipped. Did not touch `src/`.

## Outcome

### Wave A (Mac Studio; prior commit on this branch)

1. Promoted **4** banked bridge kinds from #803 (student, biv-rho12, gamma, beta-fenced).
2. Measured **3** shared-CSV H2H cells vs drmTMB 0.7.1 on tip `e9d50a110`.
3. Measured **1** tip-abs phylo Beta p=128.
4. Net then: **10 → 18**.

### Wave B (Totoro; this follow-up)

5. Ran diversity §4.2 first-wave driver on Totoro tip `e9d50a110`, threads=1.
6. Banked **10** new `has_receipt` cells (phylo Poisson/NB2/binomial/gamma,
   H2H q4 p1000 Julia-abs, crossed binomial, biv rho12, profile CI, animal, LSS).
7. Soft A3 measure-first beta-block **skipped** (`sections_fine` not restorable
   on tip in <30 min).
8. drmTMB pairs **not** run on the Totoro wave (load / H2H harness note).
9. Net now: floor 10 + Wave A 8 + Wave B 10 = **28** DRM `has_receipt` cells.

Evidence: `docs/dev-log/evidence/2026-09-23-speed-kinds-toward20/`
(`board_drm_first_wave_20260923_e9d50a110.csv`, drivers, logs).

## Rose

- Numbers from Totoro / Mac script stdout only.
- Absolute Totoro walls; no cross-host ×; H2H cell Julia-only this receipt.
- No public README/NEWS speed claim.

## Checks

- Totoro first-wave + phylo-repair logs green (10/10 cells)
- `slop_check.py` on shipped prose
- No heavy local Julia

## Coordination

Wave B is the Totoro diversity first-wave previously noted as foreign/unstaged
relative to Wave A; now banked on the same lease/PR.
