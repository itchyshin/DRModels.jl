# After Task: Poisson vs NegBinomial2 boundary warning (#1115)

## Goal

A user who calls `lrtest(pois, nb)` or `anova(pois, nb)` on a Poisson fit
versus a nested `NegBinomial2()` fit should be told that the extra `sigma`
is tested at 0. That is the same boundary warning `lrtest` already emits
for an added `(1 | g)` variance component. Lane t10 filed the silent naive
χ² as #1115 (twin of drmTMB #1557).

This change does not swap the packaged p-value. Julia issue #304 says
`lrtest` warns and leaves the χ² reference in place. drmTMB `anova()`
applies the 50:50 chi-bar mixture for the same pair. Call
`lrt_boundary(nb, pois; q = 1)` to match that p-value.

#1116 and #1118 (type-I 3.0% and 3.4% under chi-bar on the shared 500-rep
grids) are not closed. Hitting the t10 bar `[4%, 6%]` needs a methods
choice. Options live in the drmTMB twin report
`docs/dev-log/after-task/2026-10-16-fx6-poisson-nb2-boundary-lrt.md`.

## Implemented

`_family_boundary_nest` recognises `Poisson()` ⊂ `NegBinomial2()`.
`_boundary_vc_warn` emits the existing boundary warning text for that
nest. `lrtest` and `anova` stay aliases of each other. The reported
p-value stays `ccdf(Chisq(Δdof), max(stat, 0))`.

Sibling family nests are not wired. An interior Gaussian `sigma ~ 1` vs
`sigma ~ x` comparison stays silent.

## Files Changed

`src/comparison.jl`, `test/test_family_boundary_lrt.jl`, `NEWS.md`, this
report.

## Checks Run

`julia --project=.` on pin `f71b292` plus this branch:
`test/test_family_boundary_lrt.jl` 11 pass; `test/test_comparison.jl` 12
pass. The new file is the red-proof: before the warning existed,
`@test_logs (:warn, r"(?i)boundary|sigma") lrtest(pois, nb)` failed.

## Next Actions

Keep `lrtest` on naive χ² unless the owner reverses #304. Use
`lrt_boundary` for the chi-bar p-value. Do not pick mixture weights or a
parametric bootstrap without the methods decision recorded on the drmTMB
side.
