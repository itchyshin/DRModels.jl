# GOAL: DRM.jl lane speed6-20260919 (Fable speed plan step 6; GitHub name DRModels.jl)

Immutable for the run. Re-read at the top of every arc.

## Mission
Profile the q=4 bivariate phylogenetic route at p = 100/1000/5000 so a section table says what owns the
time, then land the three hygiene fixes the Fable panel named (closed-form logdet P, cholesky! symbolic
reuse in the inner Newton and build_M, warm u0 into _q4_fd_vcov), each an identity gated against
origin/main, tests first, without moving the repo's guarded baseline (2.18x vs drmTMB, logLik -256.51).

## Invariants (never violated)
- Profile first: no src/ edit until leaf-S3 has passed and its table is in the checkpoint (gate G3.5).
- Every src/ change is an identity: marginal NLL at fixed theta equals origin/main (90fbb0e28) at
  rtol 1e-12; Newton iteration counts and the ridge lambda sequence identical; cholesky! fallback count
  zero; _q4_fd_vcov Wald vcov rtol 1e-8 (the one stated non-bitwise gate); parity fixtures unchanged.
- The repo's own AGENTS.md rules bind: TDD (failing test first), docstrings, check-log entry, Noether
  audit before the draft PR; Shinichi's sign-off is the landing gate. Never widen a tolerance.
- Compute: Mac Studio, JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1; p <= 5000; each run under 30 min
  (D-139). The p=10,000 arm is not in this lane.
- Never push, never merge, never open a PR from inside this lane. Never stage files you did not create.
  Do not touch src/DRModels.jl or NEWS.md (another lane's D-269 paths).
- Local folder and module: the checkout says DRModels after the rename; bench/Project.toml on origin/main
  may still name `DRM`; if the bench env fails to load, run bench scripts with `--project=.` and say so.

## Definition of done
- leaf-S3 gates PASS under
  `node ~/shinichi-brain/skills/unlazy/scripts/gate-check.mjs --reverify --timeout 1800 .unlazy/julia-speed-20260919/gates/leaf-S3.md`;
  then leaf-S5 (written before S5 starts) PASS the same way; full `Pkg.test()` green.
- bench/results/ holds the section TSV with git SHA, Julia version, BLAS config, threads in the header.
- LOOP/lanes/speed6-20260919/checkpoint.md states TRUTH LIVES IN with paths and the branch SHA.

## Arcs
See arcs.md. Order: S3 (profile) -> checkpoint -> S5a logdet P -> S5b cholesky! reuse -> S5c warm u0,
each re-profiled after landing -> verify.

## Gates (STOP and surface)
Any change that moves the p=100 logLik -256.51 baseline or the 2.18x number; a gate failing twice on the
same cause; a proposed edit outside the OWNS list; any compute above 30 min.

## Pre-authorised
Scoped edits under bench/, the OWNS src files, new test files and the one runtests.jl include; local
Julia runs under 30 min; `Pkg.test()`; local commits on claude/lane-speed6-20260919; .unlazy ledgers.

## Resume order
LOOP/lanes/speed6-20260919/GOAL.md -> checkpoint.md -> ultra-plan.md -> AGENTS.md (repo) ->
docs/dev-log/coordination-board.md -> .unlazy/julia-speed-20260919/GATES.md and gates/.
