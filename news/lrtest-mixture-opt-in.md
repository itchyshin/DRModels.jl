- **`lrtest(reduced, full; mixture = true)` gives the boundary-correct p-value for
  one variance component.** When `full` adds exactly one variance (SD) parameter,
  no correlation parameter, and the fixed effects are identical (for example
  `(1 | g)` left out of `reduced`), the reference is the equal mixture of
  chi-square(0) and chi-square(1), so `pvalue = 0.5 * P(chi-square(1) >
  statistic)` (Self and Liang 1987; Stram and Lee 1994). The result also carries
  `pvalue_naive` and `reference`. The default is unchanged: plain chi-square with
  the existing boundary warning. The option is refused by name for any other pair:
  a difference in fixed effects, more than one added variance, or an added
  correlation (a random slope with its correlation, where a different mixture
  applies, and `(1 | g) + (0 + x | g)` against `(1 + x | g)`, where the plain
  chi-square(1) is the right test). It is also refused when the statistic is
  clearly negative (the pair is not nested or a fit did not converge). With few
  groups the mixture is conservative: in null data sets with 10 rows per group it
  rejected at the 0.05 level in 2.9% of 3000 data sets with 20 groups and 4.1% of
  5000 with 100 groups; lme4 shows the same.
