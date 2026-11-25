# After-task: Binomial animal q2 pinned recov label (#1401)

Personas: **Hopper** (R bridge coef labels), **Noether** (the unlabelled animal
likelihood stays the independent-fields model). No spawned subagents.

## What changed

`_bridge_accept_pinned_binomial_animal_recov_label` drops
`coef_labels["recov"]` only when the fit is an unlabelled Binomial
`animal(1 + x | group)` with two animal SD targets and the supplied name is
exactly `<group>:atanh_rho` (a length-1 vector or a bare string). The
coefficient vector is unchanged: there is still no `:recov` parameter.
`_bridge_echo_coef_labels` still rejects every other unknown dpar.

Mapping that label onto the `sdtarget_*` blocks was the wrong contract. Those
blocks are the SD-regression coefficients (two columns each for `~ z0` / `~ z1`),
and drmTMB already sends those under the hex `sdtarget_*` keys. The `recov`
entry is the correlation name, one string.

Estimating a free correlation for the unlabelled form would change the
public model. That default is the independent-fields fit
(`docs/dev-log/2026-11-09-integration-decisions.md`,
`test/test_animal_slope_routes.jl`). This slice does not touch it.

## Rose

Claim: the bridge call that drmTMB makes for an unlabelled Binomial animal
intercept-plus-slope no longer dies on unknown dpar `"recov"`, and a
genuinely unknown dpar still does. Evidence: the new regression test, 26/26,
including a `drm_bridge` call whose coefficients match `drm` of the same
formula (`atol = 1e-8`), plus the existing echo suite 42/42. No drmTMB source
is vendored. The q=4 logLik baseline was not re-run; the Laplace objective
is untouched.

## Verification

Linux, Julia 1.10.13, `julia --project=.`:

- `include("test/test_bridge_binomial_animal_q2_recov_labels.jl")` — 26/26
- `include("test/test_bridge_coef_labels_echo.jl")` — 42/42

## Open

drmTMB's oracle in `test-julia-binomial-animal-q2-oracle-b31ja.R` also
expects `corpars$mu` for the unlabelled formula. This fit still pins that
correlation at 0 and does not add a `recov_` coefficient, so that expectation
is unchanged.
