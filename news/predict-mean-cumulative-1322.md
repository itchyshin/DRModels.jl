# `predict(type = :mean)` now works for CumulativeLogit (#1322)

`predict(fit, newdata; type = :mean)` on a `CumulativeLogit` fit used to throw. It now returns the
expected category score at `newdata` (`sum_k k * P(Y = k)`), matching drmTMB's
`predict(type = "mean")` (drmTMB #1320, PR #1647). In-sample it equals `fitted(fit)`.
Thanks to Russell Dinnage for the original report (drmTMB #1320).
