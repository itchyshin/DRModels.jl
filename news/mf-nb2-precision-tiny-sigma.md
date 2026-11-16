- **The cross-family fitter scores the NB2 count density correctly at tiny sigma.** The
  mixed-family NB2 log-density (`fit_mixed_family`) formed `loggamma(y + θ) - loggamma(θ)`
  directly, with size θ = 1/σ². Once σ fell below about 0.03 the two terms (each of size
  θ·log θ, up to 1e13 at the σ clamp) cancelled, and the density was off by up to 0.046 nats
  per observation at the clamp, so the Poisson limit was scored higher than the truth (a
  spurious gain of about 3.5 nats on a 150-row fixture). For θ at or above 1e3 the density
  now uses the Stirling-stable form already used by the ordinary Laplace route, written with
  `log1p(μ/θ)`; below that the old expression is kept as it was, so fits with
  log σ ≥ -3 are unchanged. Fits that sit on the near-Poisson plateau keep the same
  log-likelihood but can report a different σ̂ along the flat direction.
