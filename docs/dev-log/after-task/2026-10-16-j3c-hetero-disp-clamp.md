# After-task: J3c hetero per-observation log-sigma clamp (PR #1080)

## 1. Goal

Task spec `fix1080.md` on PR #1080 head `92cefb58`: `_fit_phylo_mean_laplace_hetero`
picked its Wald Hessian path from the phylogenetic log-SD coordinate alone.
Check whether `_phylo_mean_laplace_hetero_fg` clamps log sigma per observation;
if yes, route those cases to the value Hessian and red-proof a test that hits
the clamp. Shannon, with Noether (engine contract). No subagents.

## 2. Finding (yes)

`_phylo_mean_laplace_hetero_fg` **does** clamp log sigma per observation:

```
ησ = clamp.(Xσ * βσ, -8.0, 8.0)     # src/sparse_laplace_glmm.jl:1047
aux = aux_from(ησ)                  # NB2/Gamma/Beta: clamp(-2 ησ, -8, 8) → ησ ∈ [-4, 4]
```

The SE block at `:1198` only asked `_se_hessian_path` about the trailing log-SD.
A fit can sit on the family's ±4 dispersion clamp with an interior phylo SD;
the analytic gradient is then non-zero where the objective is flat, so the
gradient-difference Hessian is undefined.

## 3. Implemented

- `_se_hessian_path` accepts optional `eta` (per-observation `Xσ·βσ`).
- `_fit_phylo_mean_laplace_hetero` passes `eta = Xσ * βσ` and
  `disp_bound = _dispersion_clamp(kind)`. Checking the βσ coordinates themselves
  would miss an interior slope whose linear predictor is already on the bound.
- Red-proof test `test/test_j3c_hetero_clamp_hessian.jl`: on `92cefb58` the
  `eta=` keyword was a `MethodError` and an e2e NB2 phylo `sigma ~ x` fit
  warned for #959 clamp (fitted ησ range −4.967 to 5.612) **without** the
  value-Hessian fallback. After the route: 12/12.

## 3a. Decisions and rejected alternatives

- Do not treat βσ coefficients as `disp` coordinates. The clamp is on ησ.
- Family bound is `_dispersion_clamp` (±4 for NB2/Gamma/Beta hetero), not the
  fg's wider ±8 box, matching the J3a review of the scalar nuisance route.
- No change to the verified q=4 engine. No drmTMB source vendored.

## 4. Verification

- Red-proof on `92cefb58` (test added, src unchanged): 8 pass / 2 fail / 2 error.
- After the route: `test/test_j3c_hetero_clamp_hessian.jl` 12/12 in 24.7 s
  (Julia 1.10.10).
- Full `Pkg.test()` recorded in the check-log row for this slice.

## 5. Rose

Claim is the routing decision, not a new SE-accuracy number. The e2e cell
asserts the warning and that some fitted ησ is on the family's bound with
an interior phylo log-SD; it does not claim drmTMB `sdreport` agreement on
that clamped design (value-path SEs stay known-noisy). License boundary
intact. Status of PR #1080 stays draft; this does not close #1073.
