- **`lrtest(reduced, full; mixture = true)` gives the boundary-correct p-value for
  one variance component.** When `full` adds exactly one variance parameter
  (for example `(1 | g)` left out of `reduced`), the reference is the equal
  mixture of chi-square(0) and chi-square(1), so `pvalue = 0.5 * P(chi-square(1) >
  statistic)` (Self and Liang 1987; Stram and Lee 1994). The result also carries
  `pvalue_naive` and `reference`. The default is unchanged: plain chi-square with
  the existing boundary warning. The option is refused by name for pairs that are
  not a single-variance test, including a random slope with its correlation, where
  a different mixture applies. In 2000 null data sets (20 groups of 10, no random-effect variance) at the 0.05
  level, the plain test rejected in 1.1% and the mixture in 2.95%; the mixture is
  closer to the nominal level but still conservative with this few groups.
