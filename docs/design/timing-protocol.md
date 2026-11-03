# Cross-engine timing protocol (DRModels.jl ↔ drmTMB)

Any table that compares DRModels.jl with drmTMB (or any other engine) on time
must label how each side was run. Hidden parallelism makes timings
incomparable: R forked workers multiplied by hidden BLAS or OpenMP threads are
not the same thing as Julia threads multiplied by hidden BLAS threads. This
page is the shared checklist; drmTMB's 2026-06-08 threading comparison uses the
same backend names.

## Required columns

Every row of a cross-engine timing table records, for **each** engine:

| Column | What to write | Examples |
|---|---|---|
| backend | how work is parallelised | `none`, `multicore` (R forked workers), `OpenMP` (TMB), `JuliaThreads` |
| process model | one process or several | `single process`, `forked workers` |
| workers / threads | count used by the backend | `1`, `8` |
| BLAS threads | `BLAS.get_num_threads()` in Julia; the R BLAS setting (for example `RhpcBLASctl::blas_get_num_procs()`) | `1` |
| OpenMP threads | `OMP_NUM_THREADS` / TMB `openmp()` setting | `1` |
| platform | OS, CPU, core count, Julia / R and package versions | `Linux, 16 cores, Julia 1.12.6, R 4.6.1` |
| timing | warm or cold, median of how many runs, wall clock | `warm median of 5` |

Also state the model, the data, and both log-likelihoods, so the reader can
see that the two engines fitted the same model.

## Pairing rule

Compare like with like:

- drmTMB `parallel = "none"` with DRModels.jl serial (`threads = false`);
- drmTMB `parallel = "multicore"` with N workers with DRModels.jl
  `threads = true` on N Julia threads, BLAS pinned to one thread on both
  sides.

A row that compares a parallel run on one side with a serial run on the other
must say so in the row, not only in a footnote.

## Where this applies

- `report/speed-per-family.md` and `report/comparison-grid.md` rows added from
  now on;
- bootstrap and profile timings (for example `bench/threaded_bootstrap_demo.jl`,
  whose drmTMB line is `parallel = "none"` only);
- future OpenMP work on the drmTMB side and any bridge timing.

Earlier rows that predate this protocol stay as recorded; where they compare a
threaded DRModels.jl run with serial drmTMB, the text next to them says so.
