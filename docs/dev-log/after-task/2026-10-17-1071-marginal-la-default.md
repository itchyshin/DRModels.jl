# After Task: print `marginal :LA` on the default RE path (#1071 / #1087)

## 1. Goal

CI run 38846719656 failed on shard 2/4 at `test/test_summary_marginal_label.jl:24`
because a printed Poisson random-effects fit showed no `marginal :LA` on the
default path. Reviewers asked for that label on both `summary` and `coeftable`,
with the existing wording kept for `:Laplace` / `:VA` / `:AGHQ`.

## 2. Implemented

- `_marginal_note` now treats a fit as having random effects when `fit.ranef`
  is present **or** a variance-component block (`:resd`, `:recov`, `:sd`,
  `:sd_phylo`, `:phylocov`, `:resd_mu`, `:resd_sigma`) is present. Poisson
  `(1|g)` never stores BLUPs (`ranef()` already documents this), so the old
  `fit.ranef === nothing && fit.marginal === :LA` test hid the default label.
- `show(fit)`, `summary(fit)` and `coeftable(fit)` all print the same note.
  `summary` remains an alias of `coeftable`; the table wraps `CoefTable` so
  `.cols` / `.rownms` stay intact.
- Other-method labels are unchanged. Fixed-effect `:LA` output is still silent.

Not touched: `src/` files outside `summary.jl` (including anything in the #1513
lane).

## 3a. Decisions and Rejected Alternatives

- Rejected storing BLUPs on the Poisson GHQ route just to make the label appear:
  that would edit the engine and is out of scope.
- Rejected stuffing the note into a `CoefTable` row or column name: that would
  change `.rownms` / `.colnms` that existing tests read.

## 4. Verification

Host: this container, Julia 1.10.12 (juliaup), single BLAS thread via `runtests.jl`.

```
julia --project=. -e 'using DRModels; include("test/test_summary_marginal_label.jl")'
# Test Summary: show names the marginal approximation (#1071) | 18 18  23.2s

julia --project=. -e 'using DRModels; include("test/test_summary.jl"); include("test/test_summary_method.jl"); include("test/test_summary_zp_suppress.jl")'
# DrmFit summary (show + coeftable)              18/18
# summary(fit) method + coef/vcov/nobs shapes      5/5
# coeftable/show z-p suppression                  34/34

DRM_TEST_SHARD=2/4 julia --project=test test/runtests.jl
# 93 of 372 files; EXIT 0; no Fail / Error During Test
# includes: show names the marginal approximation (#1071) | 18 18  2.5s
```

`Pkg.test()` was not used: Julia 1.10 reported `can not merge projects` after a local `test/` instantiate (no Manifest is committed). The shard command above is the same include list CI uses.

No engine bench: `src/` change is print-path only in `summary.jl`.

## 5. Rose audit

Scope is print-path only. No logLik values change. No drmTMB source vendored.

## 6. Follow-ups

None from this slice. `closes #1071` stays on PR #1087.
