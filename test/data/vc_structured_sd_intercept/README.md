# vc / re_sd on an intercept-only structured SD

`vcA.csv` is the data (20 groups of 8 rows, 8 trials per row, an exponential
correlation matrix over the groups). `vcA_ref.R` fits it in drmTMB 0.8.1 and
`vcA_ref.out` is its recorded output (R 4.6.1): log-likelihood -281.1827097 and
an animal SD of 0.1649602. `test_vc_structured_sd_intercept.jl` pins DRModels.jl
against these numbers.
