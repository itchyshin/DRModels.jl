- **Binomial phylo / spatial `(1 + x | g)` q2 SD regression (logit, 0/1).**
  `sigma ~ 1 + phylo(1 + x | g)` and the coordinate-spatial form fit with a
  Laplace likelihood, with `predict`, bootstrap simulation and the R bridge.
  The labelled form `phylo(1 + x | p | g)` is accepted with a free correlation
  `rho` (as R 0.8.1 does) and is fitted by the Binomial animal q2 Laplace
  route. The labelled spatial form stays refused, with a message that matches
  R. `marginal` other than `:LA` is refused on this route, and
  `simulate(data = more rows than the fit)` or `tree1`/`tree2`/`K`/`A`/`Ainv`
  on it raise a clear error instead of a `BoundsError` or a silent ignore.
  Standard errors use a finite-difference step of 3e-3, which agrees with
  drmTMB to about 0.4% on weakly identified slope SDs.
- **Boundary restarts on the structured q2 Laplace fitters.** When the first
  L-BFGS fit ends with a fitted group SD below 0.05 or fails the outer
  gradient test, the NB2 and Binomial (animal, phylo, spatial) q2 fitters
  re-run from alternative starts (constant and heterogeneous slope SD) and
  keep one only on a clear objective gain; a kept first fit is returned
  unchanged. The Binomial animal q2 inner solve treats a failed line search at
  a negligible Newton decrement as convergence at the noise floor.
- **Test-suite speed.** A capped first L-BFGS step makes the crossed Tweedie fit
  about 200 times faster, and `test/shard_by_time.jl` runs a time-balanced
  shard assignment (`test/shard_assign_N8.txt`, `test/shard_assign_N16.txt`).
