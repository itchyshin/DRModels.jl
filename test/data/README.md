# test/data

`habitat_growth.csv` — the first-model example data of the drmTMB package
(120 individuals: `habitat`, `temperature`, `growth`), used by the
Getting started page.

Provenance: generated from drmTMB's README code (drmTMB PR 1506,
`set.seed(1)`, `n = 120`, `mu = 1 + 0.6*grassland + 0.4*temperature`,
`sigma = exp(-0.5 + 0.45*grassland)`). Julia's random numbers cannot
reproduce R's `set.seed(1)`, so the table is shipped instead. R was not
available when it was made: the numbers were produced by a pure-Python
re-implementation of R's default generator (Mersenne-Twister, inversion
normals), which reproduces R's `set.seed(1); rnorm(5)` and `runif(3)` values,
and an ML refit of the table reproduces the estimates printed in drmTMB's
README to all printed digits (mu: 0.9703459, 0.5467444, 0.4134070; log sigma:
-0.4851669, 0.4776546). The data are simulated; they carry no licence
restriction beyond this repository's MIT licence.
