# Claude Code instructions for DRModels.jl

`DRModels.jl` is the Julia twin of the R package **drmTMB** (univariate &
bivariate distributional regression). Sister to GLLVModels.jl. **Read `HANDOVER.md`
first** — it is the source of truth for scope, the verified state, and the
engine. Then [`AGENTS.md`](AGENTS.md) (the team, lanes, Definition of Done, and
the parity / bridge / license contracts) and [`ROADMAP.md`](ROADMAP.md)
(phases + gates). The live work ledger is **GitHub Issues**, not a private file.

## Identity & syntax (keep stable)
- Distributional regression: a formula per parameter — mean μ, scale **σ** (use
  `sigma`, never `tau`), correlation **ρ12** (`rho12`) for bivariate residual
  correlation. Group-level (phylo/spatial/study) correlations are named
  covariance summaries, not residual `rho12`.
- Front end (planned): `bf(mu1=…, mu2=…, sigma1=…, sigma2=…, rho12=…)` (mirror
  drmTMB). Meta-analysis = `gaussian()` + `meta_V()` (the older `meta_known_V`
  is deprecated; keep it only as a parity stub).
- **ML is the default** (REML likelihoods aren't comparable across fixed-effect
  structures — needed for model selection); REML is an option.

## Current state
- `src/` = verified q=4 PLSM engine (sparse augmented-state Laplace, exact O(p)
  gradient, robust mode-finder). `src/experimental/` = REML / inference /
  location-only / EM variants, migrated but NOT yet wired into the module.
- Verified: 2.18× over drmTMB single fit; O(p) to p=10,000; valid CIs where
  drmTMB's Hessian fails. See `report/comparison-grid.md`.

## Discipline
- **Local checks over CI.** Run `Pkg.test()` / benchmarks locally first; CI is
  Linux-only, PR + `workflow_dispatch` (cost-disciplined). Add macOS/Windows only
  before a release.
- **Verify before claiming.** Every speed/accuracy claim in this repo was
  reproduced by an independent run; keep that bar. Don't promote extrapolated
  numbers (e.g. drmTMB at p=10,000) to measured results.
- Don't revert human/Codex changes unless asked. Update `report/` when the
  grammar, likelihood, RE, or phylogenetic behaviour changes.

## Team, ledger & recovery
- **Speak as Shannon**; name the active perspectives and say explicitly when no
  subagents are running. The 12 personas + their lanes are in `AGENTS.md`.
- **Work ledger = GitHub Issues.** Milestones = phases; one issue → one branch →
  one PR → merge; PRs `closes #NN`. Don't pile up local commits.
- **Evidence-first rehydration.** On resume, reconstruct state from `git status`
  / recent commits / `docs/dev-log/` (check-log, after-task, recovery-
  checkpoints) — not chat memory. Helper: `tools/drm-checkpoint.jl`.
- **License boundary.** drmTMB is GPL(≥3); DRModels.jl is MIT. **Never vendor drmTMB
  GPL source.** R-parity uses generated outputs only. Rose audits this per tag.
- **Definition of Done** (per `AGENTS.md`): impl + tests + docstrings + worked
  example + check-log + after-task + Rose audit.

<!-- shinichi-hub -->
@/Users/z3437171/Dropbox/Github Local/Shinichi/AGENTS.md
