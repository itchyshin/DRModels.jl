# After-task: DRModels speed kinds toward 20 (2026-09-23)

## Scope

Fill the three-package speed board DRM column toward 20 diverse kinds.
Lease `cursor:DRM.jl-speed-kinds-toward20`. Did not touch reader-doc PRs or `src/`.

## Outcome

1. Promoted **4** banked bridge kinds onto the board from #803 evidence
   (student, biv-rho12, gamma, beta-with-R-fence).
2. Measured **3** new shared-CSV H2H cells vs drmTMB 0.7.1 on tip `e9d50a110`
   (missing-response Gaussian, sigma RE, Tweedie FE).
3. Measured **1** tip-abs phylo Beta p=128 cell.
4. Net: DRM board kinds **10 → 18**. Evidence under
   `docs/dev-log/evidence/2026-09-23-speed-kinds-toward20/`.

## Rose

- Numbers from script stdout / banked TOML/JSON only.
- Sigma RE Δll disclosed; beta R arm fenced; no public speed claim.

## Checks

- Local Julia tip timings (J=1, OB=1)
- R drmTMB 0.7.1 fixture H2H
- `slop_check.py` on shipped prose

## Coordination note

While this lease held `docs/dev-log/evidence/`, a foreign Totoro first-wave script
(`bench/speed_board_firstwave_drm.jl`) and a `CLAIMED_BY_TOTORO.md` appeared in the
worktree. Left unstaged. Distinct cell_ids from that wave; no merge of their files
into this PR.
