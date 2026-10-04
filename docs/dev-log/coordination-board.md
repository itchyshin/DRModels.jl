# coordination-board.md — branch / PR overlap

Check this before editing shared files (`src/DRM.jl`, `AGENTS.md`, `CLAUDE.md`,
`ROADMAP.md`, `test/runtests.jl`, `docs/`). Record active branches + which files
they touch so two agents don't collide.

## Active-Lane-Split (2026-10-04 — Claude lane HANDED to Cursor; v0.7.2 release in flight)

_**START HERE (Cursor):** [`docs/dev-log/handover/2026-10-04-cursor-handover.md`](handover/2026-10-04-cursor-handover.md)._

_| Open Claude PRs being landed by the merge gate: #967 (location-scale stall stops) and #1037 (release v0.7.2). The `v0.7.2` tag currently points at the wrong commit and must be re-tagged by the owner after both merge. **Other live lanes, unchanged and not owned by this handover:** Codex docs PRs #770, #777, #779, #789, #793, #795–#800, #576; Cursor #809; Claude reader PRs #801/#802. drmTMB drafts #1442/#1443/#1445/#1446–#1449 are owner-merge only._

## Historical — Active-Lane-Split (2026-08-28 — v0.7.0 TAGGED; the lss arc is MERGED; lane handed to Cursor)

_**START HERE:** [`docs/dev-log/handover/2026-08-28-cursor-handover.md`](handover/2026-08-28-cursor-handover.md)._

_| **Everything below the 2026-08-25 block is HISTORICAL.** Since then: `v0.7.0` tagged (merge commit
`051bdc11`, twin-tracked to drmTMB 0.7.0 at pin `a6de6bb71`), and the **location–scale–scale arc**
merged on both mains — DRM.jl #547 (carrying #554) + #553, drmTMB #1100. Capability ledger is now
**12 rows** (`location_scale_scale` added as `partial`), CLOSURE: PASS._

_| **ACTIVE LANE: Cursor**, on the "true parity + better" arc: **#558** lss REML → **#559** lss
missing responses → **#551** sparse O(p) lss → ledger row 12 → `covered`. No other lane is live;
DRM.jl `main` and drmTMB `main` are both clean. The scientific driver is
`Ayumi-495/LS_ecogeographical-rules` (her issue **#28** is the user-facing contract)._

_| **Owner gates in force:** D-164 (no CRAN) · D-111/D-181 (registration at v0.7.1, not before) ·
interval-COVERAGE fences permanent (D-179 #4 / D-180 #2) · D-139 (the p=10,000 scaling run in #551
needs a time estimate + owner approval before it runs)._

## Historical — Active-Lane-Split (2026-08-25 — Waves 1+2 landed on a branch; THREE PRs await the owner)

_**START HERE:** [`docs/dev-log/handover/2026-08-25-claude-handover-drmtmb-catchup.md`](handover/2026-08-25-claude-handover-drmtmb-catchup.md)._

_| **NOTHING IS MERGED. Both repos' `main` are untouched** — DRM.jl `8d45b651`, drmTMB `fb8e6c1a5`.
Three PRs are open and are the OWNER's call, not a lane's: **DRM.jl#485** (draft, 44 commits),
**drmTMB#1080** (#460 bridge CI routing, repaired after adversarial review), **drmTMB#1082** (four
capability rows moved). `AGENTS.md` requires maintainer approval for `src/`, the formula grammar, and
`AGENTS.md` — all three are touched._

_| **EIGHT CAPABILITY ROWS MOVED** (drmTMB#1082): `0 covered · 6 partial · 4 experimental · 1 unsupported`
→ **`5 covered · 4 partial · 1 experimental · 1 unsupported`**, CLOSURE: PASS. To covered:
`biv_gaussian_residual`, `plain_binomial_nonphylo`, `base_gaussian_location_scale`,
`gaussian_phylo_mean`, `biv_q4_phylo_reml`. To partial: `general_covariance_structured`,
`phylo_count_large_p`, `cross_family_latent`. Each checked against design/168's four limbs, not against
an impression. **Every row that can move without the owner has now moved** — the one remaining
`experimental` row (`phylo_gamma_beta_binomial`) is gated on the #473 reinstall decision, and
`engine_control_surface` is `unsupported` by design and must never move._

_| **OUR OWN CAPABILITY PAGE WAS EVIDENCE AGAINST US** (DRM.jl#490): `docs/src/capabilities.md` listed the
cross-family bivariate model as "Absent — no cross-family bivariate model is implemented" while
`src/mixed_family.jl`, three wired test files and a ~450-line methods guide sat on `main`. Public
documentation is one of design/168's four limbs, so the page was holding a row down by asserting the
capability did not exist. Swept the page's other two negative claims — both hold._

_| **ONE ROW HELD BACK ON PURPOSE:** `gaussian_response_mask` stays `partial` though its four limbs
are arguably met. `response = "include"` is still refused on the Gaussian mean-phylo route, and that is
a hole in the **named** capability, not an excluded neighbour. Its old boundary — which said the default
`drop` path also fails — was corrected: #482 fixed it (the cause was positional species-to-leaf mapping,
not a missing tree re-prune)._

_| **`supported` IS NOT A STATUS** — the vocabulary is `covered > partial > experimental > planned >
unsupported`. The countdown used to count `claim_status != 'supported'` and print "N unsupported rows":
`len(caps)` by construction, unable to ever register progress. Fixed in `d265d876`._

_| **STILL TRUE:** both `interval_status != "coverage_claimed"` fences intact — **no row claims interval
coverage**; the #468 campaign is UNRUN and awaits an owner go/no-go (D-139). #420/#406 OPEN and DIRTY
(do not rebase). #49, #136 PARKED. D-111 OFF. `.codex/agents/shannon-coordinator.toml` PROTECTED._

_| **DO NOT reinstall drmTMB casually (#473):** "0.7.0" identifies ≥16 builds; the installed one is 16
shipped commits behind `origin/main`. Reinstalling moves the comparator under every banked number at
once — record `Rscript tools/drmtmb_provenance.R --toml` first, then re-run the harnesses._

## Active-Lane-Split (2026-08-24 — G0 NAMED) — SUPERSEDED by the entry above

_**START HERE:** this entry. **Lane `feat/drmtmb-catchup`** (Claude Code), branched off `origin/main`
@ **`5de340fe`**. The previous entry's "lane is IDLE pending an owner-named G0" is **no longer true** —
Shinichi named the G0 on 2026-08-24: **catch up with drmTMB, then complete the package.**_

_| **THE drmTMB FENCE IS SETTLED, AND RECORDED.** The previous entry says *"Do not start #460 — blocked
by CRAN quiesce"*, and that #460 line is **SUPERSEDED**. Shinichi was asked directly, with the stricter
"DRM.jl only" reading offered as a real alternative, and chose the D-164 reading: **both repos are in
scope.** Recorded — not merely asserted — as a clarification block inside **D-164** in
`~/shinichi-brain/memory/DECISIONS.md` @ **`ed5132b`** (the vault is local-only per D-37, so a local
commit IS landed state). It licenses reversible drmTMB work: branches, PRs, tests, and edits to
`inst/extdata/julia-capabilities.tsv`._

_| **WHAT THE FENCE STILL FORBIDS, undisturbed:** no `submit_cran()`, no upload, **no release tag, no
announcement**. D-164's hold on **RELEASE** stands. drmTMB work lands as **branch + PR and is NOT merged
to drmTMB `main`** without Shinichi's say-so._

_| **CLAIMED BY THIS LANE — do not start these:** [#460](https://github.com/itchyshin/DRM.jl/issues/460)
(bridge CI routing; its stated prerequisite #459 is already closed, so it is unblocked) ·
[#465](https://github.com/itchyshin/DRM.jl/issues/465) 9 orphan test files ·
[#466](https://github.com/itchyshin/DRM.jl/issues/466) `niterations` ·
[#467](https://github.com/itchyshin/DRM.jl/issues/467) bridge formula constructs ·
[#468](https://github.com/itchyshin/DRM.jl/issues/468) interval-coverage **pre-run** (D-139 gate).
Also in scope: bivariate `:REML`, coupled phylo REML, structured markers for bivariate Student/LogNormal,
exact REML gradient (#11). The three "candidates" the previous handover offered a future session are
**now claimed here** — anyone reading cold should come to this lane, not start them._

_| **COMPUTE (owner-directed): Totoro AND DRAC.** Totoro for pilot cells (≤150 cores, D-143 binding;
`OPENBLAS_NUM_THREADS=1`); DRAC for the wide multi-seed grid via SLURM **job arrays**, one seed per
`$SLURM_ARRAY_TASK_ID` — never on a login node, `sbatch`/`salloc` with `--time`/`--account`, depot and
R library on `/project` (never `/scratch`, purged ~60 d). Attach via the existing `~/.ssh/cm-*` sockets;
never trigger Duo (D-64). **D-139 binds: #468 stops for a go/no-go before any grid runs.**_

_| **VOCABULARY CORRECTION (2026-08-24) — `supported` IS NOT A STATUS.** Earlier entries below, this
lane's own first entry, and Mission Control all said "no capability row is `supported`". The governing
registry (drmTMB `docs/design/168`) defines exactly five: **`covered > partial > experimental > planned
> unsupported`**. Nothing can ever become `supported`. Worse, `tools/parity_ledger.py` counted
`claim_status != 'supported'` and printed the result as "N unsupported capability rows" — a number equal
to `len(caps)` by construction, which could never move however much evidence landed, and which labelled
6 `partial` + 4 `experimental` rows with a word the registry reserves for *deliberately rejected*. Fixed
in `d265d876`; the countdown now reads **`11 capability rows [6 partial · 4 experimental · 1
unsupported]`**. **Promotion here means `experimental → partial` or `partial → covered`.**_

_| **SUPERSEDED 2026-08-25** — the "no row is covered" line below was true when written and is not
now: five rows are `covered` on drmTMB#1082 (unmerged). **Still true and unchanged:**
**both `interval_status != "coverage_claimed"` fences intact** — agreement
between two engines is not calibration against truth, and only the coverage campaign can earn their
removal. #420/#406 remain OPEN (DIRTY) — **do not rebase**. #49, #136 PARKED. D-111 OFF.
`.codex/agents/shannon-coordinator.toml` PROTECTED — never stage._

_| **CORRECTION to the entry below:** it says *"drmTMB was NOT edited (dirty-set hash verified
unchanged)"* — the **hash pin `0c7636897a…` is now stale**. drmTMB's tree is **CLEAN (0 dirty files)**,
not 102; a Codex lane committed that work to `codex/rescue-claude-handover-freshness-0718-20260824`.
Independently verified by both lanes._

## Active-Lane-Split (2026-08-24 — parity catch-up MERGED) — SUPERSEDED by the entry above

_**START HERE:** [`docs/dev-log/handover/2026-08-24-claude-handover-parity-merged.md`](handover/2026-08-24-claude-handover-parity-merged.md).
`origin/main` @ **`acf3d4fb3`** — merge of [#458](https://github.com/itchyshin/DRM.jl/pull/458)
(20 commits, 65 files). Issues #457/#459/#461/#462 **closed**. Full suite 304 testsets, zero
failures; CI green both Julia versions._

_| **Lane is IDLE pending an owner-named G0.** The 2026-08-24 vacation STOP was lifted, the
campaign ran, and it is now finished. Do not start [#460](https://github.com/itchyshin/DRM.jl/issues/460) —
it is drmTMB-side bridge routing, blocked by CRAN quiesce._

_| **TWO LANES LIVE from ~21:30 (D-88).** A second DRM.jl session (`drm-jl-2b`) opened while this
entry was being written. Claim a lease before writing shared files:
`tools/lane_lease.sh --claim DRM.jl --paths <...>`; a REFUSED lease means go sequential.
Separately: `lane_preflight.sh` also reports the #458 **merge itself** as a direct-to-main lane,
because all commits here carry Shinichi's identity — distinguish by timestamp against
`acf3d4fb3`._

_| **STILL TRUE:** no capability row is `supported`; drmTMB was NOT edited (dirty-set hash
verified unchanged); no coverage claim — both `interval_status != "coverage_claimed"` fences
intact. #420/#406 remain OPEN (DIRTY). #49, #136 PARKED. D-111 OFF._

_| **Behaviour change to know about:** `is_converged` is now stricter than `fit.converged` — it
rejects degenerate optima (#461). `fit.converged` still gives the raw optimiser flag._

## Active-Lane-Split (2026-08-24 — parity catch-up ACTIVE) — SUPERSEDED by the entry above

_**START HERE:** PR [#458](https://github.com/itchyshin/DRM.jl/pull/458), issue
[#457](https://github.com/itchyshin/DRM.jl/issues/457), after-task
`docs/dev-log/after-task/2026-08-24-parity-catchup.md`. Branch `parity/se-axis`.
The vacation STOP below was **lifted by the owner on 2026-08-24**, who named
R↔Julia parity as the next G0 and authorised DRM.jl chip flips._

_| **The harness gained an SE / interval axis it never had** — the axis drmTMB's own q4
`claim_boundary` names as blocking ("interval reliability"). Four capability rows now carry
live same-target evidence: `plain_binomial_nonphylo`, `phylo_count_large_p` (p=300),
`general_covariance_structured`, `biv_gaussian_residual`._

_| All 11 canonical fixtures re-measured at **drmTMB 0.7.0** (were frozen at 0.6.0):
**zero drift**, seeds preserved. AGHQ chip `missing`→`implemented` (Poisson `(1|g)` only —
scope narrowed, not oversold). Tally 40/1/1/4 → **41/1/1/3**._

_| **STILL TRUE, do not misread:** no capability row is `supported` anywhere; **no drmTMB
edit** (CRAN prepare-only quiesce); **no coverage claim** — both
`interval_status != "coverage_claimed"` fences stay. #420/#406 remain OPEN (DIRTY). #49,
#136 PARKED. D-111 OFF._

_| **CARRIED-OVER:** benchmark timings are **not measured**. The machine never went quiet
(a foreign lane's `devtools::test()` plus DRM.jl's `Pkg.test()` held load ~18/20 cores), and
the provisional data showed the direction **flip** between bootstrap and single fits, so a
ratio under contention could be sign-wrong. Harnesses exist (`tools/bench_fit_h2h.R`,
`tools/bench_bootstrap.R`); resume command in `.unlazy/parity-catchup/GATES.md` `ABANDON: G4A`.
Run them **sequentially on an idle machine**. Also: `fit$bridge$iterations` is `NA`, so the
optimizer-mechanism question cannot be answered until iteration counts cross the bridge._

_| **New engine finding:** DRM.jl clamps at `_LAPLACE_LOG_SD_FLOOR = log(1e-6)`
(`sparse_laplace_glmm.jl:149`); drmTMB has **no equivalent bound**. On near-degenerate draws
the engines land on different optima — a feasible-set difference, not a likelihood bug. See
`docs/dev-log/evidence/2026-08-24-sd-floor-asymmetry.md`._

## Active-Lane-Split (2026-08-24 — vacation STOP; Claude pickup) — SUPERSEDED, kept for history

_**START HERE:** [`docs/dev-log/handover/2026-08-24-claude-handover.md`](handover/2026-08-24-claude-handover.md).
Canonical tree: `/Users/z3437171/Dropbox/Github Local/DRM.jl` on `main` @
`6ee03fd` (GitHub API 2026-08-24). **Engine DONE:** #449 AGHQ + #451 Cox–Reid
phylo. **Scout sequence 1–5 STOP** (#452–#455 merged). Lane **IDLE** — Shinichi
away; next G0 is owner-named only._

_| Open PRs: [#420](https://github.com/itchyshin/DRM.jl/pull/420) + [#406](https://github.com/itchyshin/DRM.jl/pull/406) — DIRTY; leave OPEN per scout #455. Do not rebase while away._

_| AGHQ capability chip still **missing**. #49 **PARKED**. D-111 Julia General **OFF**._

| Lane | Repo | State | Pointer |
|---|---|---|---|
| DRM.jl Julia | this repo | **IDLE** / vacation STOP | `docs/dev-log/handover/2026-08-24-claude-handover.md` |
| drmTMB (R) | sibling | **unknown** | that repo's handover — do not start from DRM.jl |

_Light tip refresh 2026-08-14 (`docs/claude-handover-idle`) — **historical**:
`origin/main` @ `d0fac9d7` (#405 MERGED). DRM.jl Julia = **IDLE** pending
owner G0. Next pickup = **Claude**. Cursor lane idle/handing off. Issue 136
stays OPEN (136e public Gamma RI report landed; VA stays Experimental).
#49 PARKED. drmTMB = **sibling** (status unknown here; do not claim finished;
do not start from this tree). 2026-08-13 Cursor idle handover is historical.
Live START HERE:
`docs/dev-log/handover/2026-08-14-claude-handover-drm-idle.md`.
Prior 2026-08-07 Phase 3 / #7 complete-with-carveouts still holds._

_GitHub auto-merge (2026-08-13, option A): `allow_auto_merge` is ON.
Agents may `gh pr merge N --auto --merge` after opening a PR. Pause
(no `--auto`) on `src/` engine / formula grammar / version bump /
`AGENTS.md` / `CLAUDE.md` / unfinished epic / foreign lane. Never put
close/fix/resolve next to `#NN` unless the slice is meant to finish
that issue. D-111 Julia General still OFF. Durable decision file is on
OPEN [#406](https://github.com/itchyshin/DRM.jl/pull/406) (BLOCKED on
CI `test (1.10)` at last check) until that PR lands._

## Active-Lane-Split (2026-08-16 — Claude lane HANDED OVER to Cursor)

_**START HERE:** `docs/dev-log/handover/2026-08-16-cursor-handover.md`, then
`LOOP/GOAL.md`. Working tree stays `~/local-scratch/lanes/DRM.jl-catchup`._

_**The ledger countdown reached 0 export gaps** (17 raw, 17 accounted for) —
`CLOSURE: PASS`. Read that as "no drmTMB export lacks a DRM.jl twin", **not** as
parity complete: 11 unsupported capability rows remain, each with a written
`claim_boundary`. Re-measure before trusting it._

_Landed 2026-08-15/16: #414 (A4c penalized-MAP phylo), #415 (A4d), #416 (A4e
ledger honesty), #417 (`check_drm` NaN vcov), #418 (A5), #419 (A6 tree scale).
`origin/main` = `0a4c2dc9`._

_**CARRIED-OVER — nine PRs open, #420–#429.** An earlier note here called the
`BLOCKED` PRs pure queue saturation "with nothing wrong". That was wrong. **#423
had a real defect**: `meta-analysis.md` linked to `` [`meta_vcov_bivariate`](@ref) ``
while that function sat in no `@docs` block, so under `warnonly = true`
Documenter warned instead of failing, wrote a literal `./@ref` into the built
page, and **VitePress** died on the dead link — an npm-shaped error two stages
from its cause. Fixed in `docs/src/reference/structured-effect-markers.md`.
**#420 and #425** failed separately on `git push upstream HEAD:gh-pages` —
concurrent preview deploys racing on that branch; both re-run._

_**#428 (A11) is deliberately unarmed — it touches `src/`; owner call.** #429 is
**stacked on #423** and retargets to `main` when that merges; do not rebase it by
hand. #406 is a **foreign** PR — untouched._

_**Lesson for every lane here:** a red Documenter job is not self-evidently a
prose problem, and "the queue is busy" is not a diagnosis. `warnonly = true`
means a broken `@ref` never fails Documenter — it fails VitePress later, naming
npm. Read `gh run view <id> --log-failed` down to the failing process._

_**Tree-scale trap, worth knowing across lanes:** `ape::vcv(corr=TRUE)` gives
unit tip variance; raw Newick branch lengths give tip variance = tree height `h`.
A DGP that ignores this manufactures a convincing ~30% phylo variance-component
"bias" that is entirely the simulator's error (predicted −29.3%, observed
−29.4%). Suspect the DGP before the engine._

| Lane | Repo | State | Pointer |
|---|---|---|---|
| DRM.jl catch-up | this repo | **HANDED OVER** to Cursor; ledger at 0; 5 PRs blocked on CI, 3 branches unPR'd | `docs/dev-log/handover/2026-08-16-cursor-handover.md` · `LOOP/GOAL.md` |
| drmTMB bridge (narrow) | `/Users/z3437171/Dropbox/Github Local/drmTMB` | **#1049 + #1050 OPEN — NEVER MERGE UNATTENDED** (9 live lanes + release slice #959). #1032 and #1038 landed | drmTMB PRs #1049, #1050 |
| drmTMB `engine="julia"` Workflow G | `/Users/z3437171/Dropbox/Github Local/drmTMB` | **sibling — status UNKNOWN from here**; do not claim finished, do not start from DRM.jl | that repo's own handover |

## Active-Lane-Split (2026-08-15 — arc-loop lane ACTIVE)

_The `engine = "julia"` catch-up campaign runs as an **/arc-loop** in a bounded
worktree at `~/local-scratch/lanes/DRM.jl-catchup` on `claude/lane-catchup` (off
the Dropbox path). **START HERE:**
`docs/dev-log/handover/2026-08-15-claude-handover.md`, then `LOOP/GOAL.md`._

_Anchor: drmTMB **0.7.0 INSTALLED**. The twin moves fast — re-run
`python3 tools/parity_ledger.py --drmtmb ../drmTMB --ref origin/main` before
trusting any count. Landed via #408/#409/#410: A0, A1, A2a, A3a, A3b, A-fix.
On the lane, unmerged: A3c-1/2/3, **A-nb2**, A-sigma, A4-design. Next: **A4c**._

_**A-nb2 is worth knowing about across lanes:** DRM.jl's NB2 dispersion seed was
on the size scale rather than log-σ (the `-0.5` was missing at 6 sites), so NB2
fits could silently converge to the Poisson boundary. Fixed; it changes NB2
starting values everywhere. Found only by a cross-implementation parity fixture._

| Lane | Repo | State | Pointer |
|---|---|---|---|
| DRM.jl catch-up (arc-loop) | this repo | **ACTIVE** — `claude/lane-catchup`; A4c next; two owner gates open (A4a→#49, A4b→not-ported) | `docs/dev-log/handover/2026-08-15-claude-handover.md` · `LOOP/GOAL.md` |
| DRM.jl #412 | this repo | `docs/a3c-design` — auto-merge ARMED, CI running | PR #412 |
| drmTMB bridge (narrow) | `/Users/z3437171/Dropbox/Github Local/drmTMB` | **PR #1032 OPEN — NEVER MERGE** (9 live lanes + release slice #959). Evidence citations only, no status changed | drmTMB PR #1032 |
| drmTMB `engine="julia"` Workflow G | `/Users/z3437171/Dropbox/Github Local/drmTMB` | **sibling — status UNKNOWN from here**; do not claim finished, do not start from DRM.jl | that repo's own handover |
| DRM.jl (prior, historical) | this repo | superseded — tip-idle handover of 2026-08-14 | `docs/dev-log/handover/2026-08-14-claude-handover-drm-idle.md` |

Rehydrate must read **every** row. A single pointer must not orphan the drmTMB
sibling lane or the parked epics (#136 OPEN, #49 PARKED — which now also holds
`categorical`, an imputation family, not a response family).

## Lane split — Claude ↔ Codex

Full Codex brief: **#76** (pinned).

- **Claude (Shannon)** — family front-ends, post-fit, docs.
  Files: family `src/*.jl`
  (`poisson` / `negbinomial` / `beta` / `gamma` / `student` / `lognormal` /
  `betabinomial` / `binomial` / `zeroonebeta` / `tweedie` / `cumulative`),
  `summary.jl`, `inference.jl`, `variational.jl`, `docs/`, `test/runtests.jl`,
  `model-map.md`, `check-log.d/`.
- **Codex** — engine core + estimators.
  Files: `src/sparse_aug_plsm.jl`, `src/fit_q4_sparse_tmb.jl`,
  `src/takahashi_selinv.jl`, `src/experimental/*`, `bench/*`, `report/*`.
- **Shared — coordinate on the PR:** `src/DRM.jl` (include/export list).
  Append engine/experimental symbols in their own spot; flag on the PR.

## Verified state (as of HEAD)

- **All 13 families done / exported / tested.** Each has a `struct … end`
  marker in `src/` (`gaussian_core.jl` `Gaussian`, `student.jl` `Student`,
  `poisson.jl` `Poisson`, `negbinomial.jl` `NegBinomial2` + `TruncatedNegBinomial2`,
  `beta.jl` `Beta`, `betabinomial.jl` `BetaBinomial`, `binomial.jl` `Binomial`,
  `gamma.jl` `Gamma`, `lognormal.jl` `LogNormal`, `zeroonebeta.jl` `ZeroOneBeta`,
  `tweedie.jl` `Tweedie`, `cumulative.jl` `CumulativeLogit`) and all 13 appear in
  the `src/DRM.jl` export list. Family parity completed at **v0.1.1**
  (see `NEWS.md`).
- **Inference is wired** in `src/inference.jl` — Wald (`confint(…, method=:wald)`,
  boundary-aware SEs, #106), profile-likelihood (`method=:profile`, #103), and
  bootstrap (coefficient summaries #101 / auditable results #105 / fit-based
  entry points #132).
- **Tagged releases:** `v0.1.0` and `v0.1.1` (`git tag`).
- **Engine — crossed/structured Laplace merged.** Codex's lane landed crossed
  random effects (`closes #70`) plus a series of crossed/structured Laplace
  speed/correctness merges: #89, #97, #108, #111, #114, #119, #123, #126, #128.
- **Phase 3 / #7 complete-with-carveouts (2026-08-07).** All 26 drmTMB-target
  Documenter slugs exist; carve-outs named (phylo×spatial Theory+roadmap;
  VA Experimental #136). Public VA is Experimental `(1 | g)` on Poisson /
  Binomial / NB2 / Gamma / Beta (`marginal = :VA`); Laplace remains default;
  `_fit_va` still errors for unwired families; #136 stays OPEN. Inventory:
  `docs/dev-log/evidence/2026-08-07-7-phase3-inventory.md`.
- **#136e public-path report (2026-08-13 on main via #404).** Gamma `(1 | g)`
  smoke: LA ≈ VA on α; LA faster warm. VA stays Experimental. Not ZINB / not
  two-part. Report: `report/va-vs-laplace-bias.md`.

## Active branches

| Branch | Owner | Touching | Status |
|---|---|---|---|
| `main` | — | tip @ `d0fac9d7` (Merge #405) | landed idle START HERE; ship = idle pending G0 |
| `docs/claude-handover-idle` | Shannon (Cursor → Claude) | Claude handover + LOOP + coord + DoD docs | **this docs lane**; no closer for issue 136 |
| `docs/github-auto-merge` | Shannon (Cursor) | auto-merge policy note | **OPEN #406** / BLOCKED on CI `test (1.10)`; not a G0 |
| `docs/tip-idle-after-404` | — | handover + LOOP + coord + DoD docs | **MERGED** via #405; now historical |
| `feat/136e-va-bias-report` | — | report + bench + docs honesty | **MERGED** via #404; do not resume |
| `docs/handover-2026-08-09-drm-julia-lane` | — | Julia-lane START HERE | **MERGED** via #403; now historical |
| `feat/136-va-rung2-3` | — | — | **MERGED** via #401; do not resume |
| #49 | — | fenced | **PARKED** — owner-named only |
| drmTMB `engine="julia"` | — | **other repo** | **sibling** — possibly in progress; unknown here |

> Note: Phase 3 article-fill work is exhausted (26/26). Do not invent tip-idle
> SHA padding after this PR. Never stage `.worktrees/`. Coordinate on
> `src/DRM.jl` if an engine lane reopens.
