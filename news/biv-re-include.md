- **The bivariate `(1 | p | g)` route accepts rows with one response missing.** With
  `missing = miss_control(response = "include")`, a row with one of the two responses
  missing now stays in the fit and contributes the marginal likelihood of the response that
  was observed, as in drmTMB 0.8.1 and as on the fixed-effect bivariate route. Before, the
  fit refused with "needs both responses observed on every row". The default (`"drop"`)
  is unchanged, and complete data give the same fit to the last bit. Checked against a
  drmTMB fit (`test/fixtures/biv_re_include/`): log-likelihood to 1e-4, fitted values to
  1e-6, and against a dense Gaussian likelihood of the observed cells written separately in
  the test, with a finite-difference gradient check.
