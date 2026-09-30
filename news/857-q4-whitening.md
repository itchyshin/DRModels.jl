- **q=4 REML engine: whitened prior, no Λ inverse (#857 site q4-REML).**
  `reml_ll_and_mode`, `_reml_exact_state` and the exact-gradient helper
  `reml_nll_and_exact_grad` built the REML prior precision
  `P = kron(Q_cond, inv(Λ))` from a formed `Λ` at six sites, plus a bare
  `logdet(Symmetric(Λ))` for its gradient term. Near a singular Λ (correlation
  → ±1, log-Cholesky diagonal around −18 to −30) this poisoned the
  Newton-certified joint mode badly enough that the Schur complement came back
  non-PD and `reml_nll_exact` returned the `Inf` barrier outright — on a
  well-identified p=6-leaf/n=24 fixture, l22 = −18/−20/−30 all failed this way
  while a 256-bit reference gives a finite value (≤ 1e-8 relative once fixed).
  Now built in whitened coordinates (`v = (I ⊗ L⁻¹)u`), matching the sister
  fix to the ML engine (#857 site q4) and the coevolution/q2 prior (#857 site
  K). Normal-regime values are unchanged: a frozen copy of the pre-fix
  algorithm agrees with the live one to ≤ 1e-10 over 10 random well-conditioned
  points.

- **Whitened q=4 engine: acceptance bar and test changes.** Owner-approved bar
  (2026-09-30): (i) identity with the unwhitened engine at a *polished* inner mode
  (`test_q4_prior_whitening.jl` compares `marginal_nll` with main's `P = Q ⊗ Λ⁻¹`
  construction solved to ‖∇J‖ < 1e-11; `test_reml_q4_chol.jl` compares the REML
  objective with a frozen pre-whitening copy), plus (ii) accuracy against a 256-bit
  reference (ML: l22 from −2 to −30; REML: −2, −12, −18, −20, −30). The inner mode
  gets a Newton polish after the LM λ-break so finite-difference vcov checks are not
  noise-dominated. `marginal_nll`'s 4th return is a `WhitenedPrior` (see below);
  `test_coverage_engine.jl` and `test_optimizer_robustness.jl` follow that contract
  (an extreme θ now gives a non-finite objective/gradient, which the fit rejects,
  instead of an `ArgumentError`). The factorisation floor in
  `test_q4_perf_identities.jl` drops from ≥ 100 to ≥ 90 because the fit converges in
  fewer iterations, and the `test_gaussian_bivariate_q4_structured.jl` positive
  control uses `q4_iterations = 300` (the whitened objective needs > 120 there).

- **q=4 ML PLSM engine: whitened prior, no Λ inverse (#857 site q4).** The
  Newton mode-finder, `laplace_ll`, `marginal_nll`, the exact O(p) gradient and
  `q4_marginal_diagnostic` now work in `v = (I ⊗ L⁻¹)u` (`Λ = LLᵀ` built from θ
  by `lc_to_chol`), so `H̃ = Q⊗I + blockdiag(LᵀD L)` and the `Σ log Lᵢᵢ` terms
  cancel analytically; the 1e-10 prior ridge is kept exactly. On the
  `test_q4_objective_diagnostic` fixture the marginal was 6.5e-7 relatively wrong
  at l22 = −12 and 2.4 nats off at l22 ≤ −20; it is now ≤ 4e-11 of a 256-bit
  reference from −2 to −30. `marginal_nll`'s 4th return value is now the
  `WhitenedPrior` and its `ch_H` the factor of `H̃`. The q=4 REML is separate.
