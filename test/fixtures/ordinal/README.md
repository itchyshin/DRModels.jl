# ordinal teaching-dataset fixture (issue #953)

- `wine.csv` - `ordinal::wine` (Randall 1989, bitterness of wine; 72 rows: 9 judges x
  8 bottles, temp x contact). Columns: `rating` (1-5), `temp`, `contact`,
  `bottle` (`b1`..`b8`), `judge` (`j1`..`j9`); the continuous `response` column
  of the original is not used.

Transcribed from the data block in DRModels.jl issue #953 (written there from
`ordinal::wine`, ordinal 2023.12-4.1). Only the dataset is vendored; no ordinal
code is copied. ordinal's DESCRIPTION license is `GPL (>= 2)`, and the dataset
ships inside the package under it; it is also published in Randall (1989),
"The analysis of sensory data by generalised linear model", Biometrical Journal
31, 781-793. Same arrangement as `test/fixtures/lme4/README.md`. Used by
`test/test_lane_j4_predict_numerics.jl`.
