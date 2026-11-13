- **`predict_parameters()` returns every estimated parameter.** A Student,
  SkewNormal or Tweedie fit without a `nu` formula left `:nu` out of the result,
  and a ZeroOneBeta fit without `zoi` or `coi` formulas left those out, although
  they are estimated and `fixef()` reports them. They are now returned (link and
  response scale, with `se = true` too), with the same values as the model
  written out with `nu ~ 1`, `zoi ~ 1` and `coi ~ 1`.
  The same fix changes four other outputs:
  the bridge `newdata` payload (`dpars_newdata`) for Student and SkewNormal fits
  gains a `"nu"` entry;
  `residuals(type = :quantile)` and `simulate` on a Student fit with missing
  responses now work (they threw a `BoundsError`);
  `predict(type = :mean)` on a ZeroOneBeta fit with missing responses now works
  (it threw a `KeyError` for `zoi`);
  and `simulate` on a SkewNormal fit with missing responses now returns
  different numbers, because the old draws used a `nu` vector that was too short
  and were wrong.
