- **`simulate(fit; data)` for several random intercepts.** A fit with
  `(1 | g) + (1 | g2) [+ ...]` on the mean (distinct groups, any family except
  ZeroOneBeta) now draws a fresh independent effect per term, as drmTMB 0.8.1
  does, instead of refusing. Slope terms, repeated groups, correlated blocks
  and `meta_V` plus two fields stay refusals. `simulate(fit)` without `data=`
  is unchanged: it refuses, because the fit does not keep the grouping columns.
- **Behaviour change in `bootstrap`.** For these crossed-intercept fits the
  parametric bootstrap previously fell back, without a warning, to the
  conditional draw (random effects held at zero; #997). It now redraws every
  field. Measured on simulated Gaussian and Poisson data with two crossed
  intercepts (12 datasets, B = 60): the 95% interval for the log random-effect
  SDs was about 14 to 27 log units wide and covered the truth 0 of 12 times;
  it is now 0.7 to 2.0 wide and covers 9 to 12 of 12. Fixed-effect intervals
  changed little (slope width 0.27 to 0.28 Gaussian, 0.26 to 0.24 Poisson).
