- **`confint(fit; method = :profile)` returns `[0, u]` for a random-effect SD that is
  estimated at zero, instead of refusing.** When an SD estimate was at or near zero,
  the upper-limit search started from a Wald step that could be 1e7 on the log scale
  and never bracketed the crossing, so `confint` threw (for example "upper (nuisance
  solve: nonfinite_objective)") on a Gaussian `(1 | g)` fit and on both SDs of
  `(1 | g) + (0 + x | g)`. The search now starts one log unit above the estimate in
  that case. The lower limit is exactly 0 and the upper limit is where the profile
  deviance reaches the chi-square(1) quantile; on a Gaussian `(1 | g)` fit it agrees
  with a hand-written profile to 1e-4. Interior SDs are unchanged bit for bit.
