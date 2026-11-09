# Integration verification, 2026-11-09

Branch `claude/integration-final-20261109`, built from the wave-10 base (0bfcde2bb) with the
eleven stacked lane pull requests and the three main-based fixes merged in. How the conflicts
were settled is in `2026-11-09-integration-decisions.md`. This note records what was run and what
it showed. Source files were identical for all runs below (tip 76763f66f, which differs from the
final commit only by documentation).

## Full test suite

Julia 1.10.12 and Julia 1.12.7 on Totoro, 16 time-balanced shards per version
(`test/shard_by_time.jl` with `test/shard_assign_N16.txt`), one thread per process, 32 processes in
all. The assignment lists 616 discoverable test files (every `test_*.jl` in `test/` except the four
in `_TEST_EXCLUDE`), each exactly once; the runs executed 616 distinct files per version.

| Julia | Files run | Failing files | Shard exit codes | Slowest shard |
| --- | --- | --- | --- | --- |
| 1.10.12 | 616 | 0 | 16 of 16 zero | 1047 s |
| 1.12.7 | 616 | 0 | 16 of 16 zero | 1287 s |

Failing files on the integrated tip: none, on either version. Because there are none, there is no
list to compare against the wave-10 base. To check the five failures named at the start of the task,
the same files were run on the unmodified wave-10 base with Julia 1.12.7:
`test_stored_loglik_at_optimum`, `test_binomial_spatial_sd_regression`,
`test_locscale_raw_k_regression`, `test_relmat_raw_k_regression` and `test_reml_q4_missing_response`
all fail there and all pass on the integrated tip (the fixes of #1284 and #1285). The two further
files reported as failing on the base tips, `test_poisson_structured_sd_offsets` and
`test_binomial_animal_q2_sd`, passed on the base in this run, so that failure was not reproduced.
One test on the Mac with Julia 1.10.12, `test_lane_j4_predict_numerics` (the cumulative logit
crossed-effect check), failed on the Mac on both the base and the integrated tip with the same
value, and passes on Totoro on both versions, so it depends on the machine.

## Gap map, 3008 cells

The wave-8 fuzz harness was rerun on the integrated tip (Julia 1.10.12, 24 workers, 300 s per cell),
and classified with the same classifier. The R side is the stored wave-8 result, as in the wave-10
rerun. "Before" is the wave-10 base map; "after" is the integrated tip. Files are in
`gap-map-integration-final-20261109/` (`maps/`, `compare/`, `raw-julia/`).

| Class | Before | After |
| --- | --- | --- |
| Both fit, logLik agrees (1e-4) | 260 | 350 |
| Both fit, logLik disagrees | 54 | 55 |
| R fits, Julia refuses (R-only) | 107 | 17 |
| Julia fits, R refuses (Julia-only) | 43 | 83 |
| Not converged | 5 | 4 |
| Timeout | 1 | 1 |
| Crashes (either side) | 0 | 0 |
| Both refuse | 2538 | 2498 |
| Julia fits (any) | 357 | 488 |

Transitions: 86 R-only cells now agree with R, 2 R-only cells now fit but disagree, 2 R-only cells
now fit without converging; 2 non-converged cells now agree and 1 now fits but disagrees; 2
disagreeing cells now agree; 40 cells that both sides refused are now fitted by Julia only; 2873
others keep their class.

Regressions under the pre-registered rules (a fitted cell that stops fitting, an agreeing cell that
starts to disagree, a refusal that becomes a crash, a new crash type): none. The harness also flags
any fitted cell whose logLik moved by more than 1e-6. Two cells are flagged, both
`binomial|animal|lab0|q2|ML` (`sd-one` and `sd-z`): the logLik fell by 1.44 and 0.90, and it now
equals R's value to 1e-10. They had been fitting above R's maximum before. So they moved into
agreement, not away from it.

The net rise in disagreements (54 to 55) is three cells that now fit where they did not before:
Binomial ordinary q = 3 and q = 4 without SD formulas (Julia is 0.57 and 0.65 above R's Laplace
value, the adaptive Gauss-Hermite difference documented for the Binomial blocks), and one Poisson
spatial cell that R did not converge on. Two disagreeing cells moved to agreement, so the net is one.

Still open: 17 R-only cells (Binomial 8, Beta 3, Gaussian 2, NB2 2, Student 2), four non-converged
cells (NB2 animal q = 2, Binomial ordinary q = 3, Binomial indep q = 2 and q = 4), and one timeout
(Binomial ordinary q = 4 with `sd-z`, which R also fails to converge on).

## Not verified

- R was not rerun; the stored wave-8 R results were reused.
- The new fits were compared through logLik only, not through estimates or standard errors.
- The 83 Julia-only cells have no R reference.
- Merge-by-merge test runs on Julia 1.10 were complete for the first four merges. For the later
  merges the Totoro connection dropped part way, and their touched test files were run on the Mac
  (Julia 1.10.12) on the final tree, then everything again in the full suites above.
