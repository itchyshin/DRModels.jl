# #1567: integer weights vs duplicated rows at a grouped-variance boundary share
# logLik and fixed effects, but log-SD θ can diverge. Under the FX3b twin contract,
# `convergence_status` must be `:boundary` when random-effect SD < 1e-4, including
# `control = DRMControl(se = false)` fits where the Hessian was computed then discarded.
using DRModels, Random, StatsAPI, Test, Logging

@testset "#1567 weights vs replication at RE variance boundary" begin
  rng = MersenneTwister(5)
  n = 30
  g = repeat(["a", "b", "c"], inner = 10)
  x = randn(rng, n)
  y = 0.5 .+ 0.35 .* x .+ randn(rng, n)
  d = (; y, x, g)
  w = fill(2.0, n)
  NOSE = DRMControl(se = false)
  f = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1))

  m_w, m_rep = with_logger(NullLogger()) do
    m_w = drm(f, Gaussian(); data = d, weights = w, control = NOSE)
    d2 = (; y = repeat(d.y, inner = 2), x = repeat(d.x, inner = 2), g = repeat(d.g, inner = 2))
    m_rep = drm(f, Gaussian(); data = d2, control = NOSE)
    m_w, m_rep
  end

  @test m_w.uncertainty === :skipped
  @test m_rep.uncertainty === :skipped
  @test control_info(m_w).se === :discarded

  @test loglikelihood(m_w) ≈ loglikelihood(m_rep) atol = 1e-9
  @test m_w.theta[1:3] ≈ m_rep.theta[1:3] atol = 1e-6
  @test maximum(values(re_sd(m_w))) < 1e-4
  @test maximum(values(re_sd(m_rep))) < 1e-4

  @test convergence_status(m_w) === :boundary
  @test convergence_status(m_rep) === :boundary
  @test !is_converged(m_w)
  @test !converged(m_w)
  @test m_w.converged  # optimiser flag unchanged
end