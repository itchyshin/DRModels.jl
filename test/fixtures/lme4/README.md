# lme4 teaching-dataset fixtures (issue #706)

- `cbpp.csv` - `lme4::cbpp` (contagious bovine pleuropneumonia; 56 rows: herd, incidence, size, period).
- `sleepstudy.csv` - `lme4::sleepstudy` (180 rows: Reaction, Days, Subject).

Written with `write.csv(lme4::<name>, row.names = FALSE)` from lme4 2.0.1 (R).
Only the datasets are vendored; no lme4 code is copied. lme4's DESCRIPTION
license is `GPL (>= 2)`, and these datasets ship inside the package under it
(both are also published in Bates et al. 2015, J. Stat. Softw. 67(1), and
Belenky et al. 2003 for sleepstudy). They are used by `test/test_lme4_twins.jl`.
