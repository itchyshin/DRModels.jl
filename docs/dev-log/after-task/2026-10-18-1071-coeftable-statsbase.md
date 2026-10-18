# After Task: keep `coeftable` a real StatsBase CoefTable (#1071 / #1087)

## 1. Goal

PR #1087 must keep returning a real `StatsBase.CoefTable` from `coeftable` /
`summary`, and print the #1071 integrator note in `show(fit)` only. The thin
`DrmCoefTable` print wrapper broke `ct isa CoefTable` and `DataFrame(ct)`.
Shannon, with Pat (UX / summary surface) and Rose (claim vs evidence). No
subagents.

## 2. Implemented

- Removed `DrmCoefTable`. `coeftable(fit)` again returns `CoefTable(...)`.
- `show(fit)` still prints `_marginal_note` (RE block or non-default integrator;
  Poisson `(1|g)` still keys off the variance-component block, not `fit.ranef`).
- File header at `src/summary.jl:8-10` no longer claims `coeftable` and
  `CoefTable` both come from StatsModels, and states that the note lives on
  `show(fit)`.
- Tests: `ct isa CoefTable` and `DataFrame(ct)` on default `:LA`, `:Laplace`,
  and a fixed-effect fit. `DataFrames` is a test-only dep (`test/Project.toml`).

Not touched: engine files, `check_drm`, `bridge.jl`, `_finite_hessian`.

## 3a. Decisions and Rejected Alternatives

- Rejected keeping the wrapper and forwarding `isa` via a subtype: `CoefTable`
  is a concrete StatsBase struct; a wrapper is not one, so `DataFrame(ct)`
  would still fail.
- Rejected stuffing the note into a `CoefTable` row or column name: that would
  change `.rownms` / `.colnms`.

## 4. Verification

Host: this container, Julia 1.10.10, single BLAS thread via `runtests.jl`.

```
julia --project=test -e 'using DRModels; include("test/test_summary_marginal_label.jl"); include("test/test_summary.jl"); include("test/test_summary_method.jl"); include("test/test_summary_zp_suppress.jl")'
# show names the marginal approximation (#1071) | 16 16  22.3s
# DrmFit summary (show + coeftable)              18/18
# summary(fit) method + coef/vcov/nobs shapes      5/5
# coeftable/show z-p suppression                  34/34

julia --project=test test/runtests.jl
# DRModels tests: all 372 files (unsharded); EXIT 0; 16,610 pass, 0 fail, 0 error; 8342 s
```

Red-proof (worktree at e0816e5, tests only, no `_marginal_note`):
`show names the marginal approximation (#1071) | 6 pass, 2 fail` — the two
`occursin("marginal :…", shown(fit))` assertions, as intended. GitHub
`gh run rerun 38846684642` returned 403 (integration token has no
Actions write); local red-proof is the completed check.

No engine bench: `src/` change is print-path only in `summary.jl`.

## 5. Rose audit

Scope is print-path only. No logLik values change. No drmTMB source vendored.
`coeftable` is again a real `CoefTable`; the #1071 claim is `show(fit)`, which
is what the issue asked to name.

## 6. Follow-ups

`closes #1071` stays on PR #1087.
