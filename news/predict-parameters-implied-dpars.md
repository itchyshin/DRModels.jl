- **`predict_parameters()` returns every estimated parameter.** A Student,
  SkewNormal or Tweedie fit without a `nu` formula left `:nu` out of the result,
  and a ZeroOneBeta fit without `zoi` or `coi` formulas left those out, although
  they are estimated and `fixef()` reports them. They are now returned (link and
  response scale, with `se = true` too), with the same values as the model
  written out with `nu ~ 1`, `zoi ~ 1` and `coi ~ 1`.
