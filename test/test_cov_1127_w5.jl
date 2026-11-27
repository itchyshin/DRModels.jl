# Coverage lane w5 (27 Nov): ranks 10–12 global uncovered lines on `main` @ be38f5d —
# `gaussian_bivariate.jl`, `gaussian_lss.jl`, `temporal.jl` (not claimed by open 1127cov PRs).
using DRModels
using Test
using LinearAlgebra
using Random
using StableRNGs

@testset "cov 1127 w5: gaussian_bivariate.jl" begin
  _rhs_dict(f) = Dict(f.forms)

  @testset "bf: placeholder LHS and response-column guards" begin
      @test_throws ArgumentError bf(
          mu1 = @formula(y1 ~ 1), mu2 = @formula(y2 ~ 1),
          sigma1 = @formula(tau ~ 1), sigma2 = @formula(sigma2 ~ 1),
          rho12 = @formula(rho12 ~ 1))
      @test_throws ArgumentError bf(
          mu1 = @formula(y1 ~ 1), mu2 = @formula(y2 ~ 1),
          sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma1 ~ 1),
          rho12 = @formula(rho12 ~ 1))
      @test_throws ArgumentError bf(
          mu1 = @formula(cbind(y1, y2) ~ 1), mu2 = @formula(y2 ~ 1),
          sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
          rho12 = @formula(rho12 ~ 1))
  end

  @testset "_bivariate_q4_marker: rho12 and structured marker contracts" begin
      rho_re = bf(
          mu1 = @formula(y1 ~ 1 + phylo(1 | species)),
          mu2 = @formula(y2 ~ 1 + phylo(1 | species)),
          sigma1 = @formula(sigma1 ~ 1 + phylo(1 | species)),
          sigma2 = @formula(sigma2 ~ 1 + phylo(1 | species)),
          rho12 = @formula(rho12 ~ 1 + (1 | species)))
      @test_throws ErrorException DRModels._bivariate_q4_marker(_rhs_dict(rho_re))

      rho_meta = bf(
          mu1 = @formula(y1 ~ 1), mu2 = @formula(y2 ~ 1),
          sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
          rho12 = @formula(rho12 ~ 1 + meta_V(1)))
      @test_throws ErrorException DRModels._bivariate_q4_marker(_rhs_dict(rho_meta))

      q2_kind = bf(
          mu1 = @formula(y1 ~ 1 + phylo(1 | species)),
          mu2 = @formula(y2 ~ 1 + relmat(1 | species)),
          sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
          rho12 = @formula(rho12 ~ 1))
      @test_throws ErrorException DRModels._bivariate_q4_marker(_rhs_dict(q2_kind))

      q2_group = bf(
          mu1 = @formula(y1 ~ 1 + phylo(1 | species)),
          mu2 = @formula(y2 ~ 1 + phylo(1 | g)),
          sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
          rho12 = @formula(rho12 ~ 1))
      @test_throws ErrorException DRModels._bivariate_q4_marker(_rhs_dict(q2_group))

      q2_tag = bf(
          mu1 = @formula(y1 ~ 1 + phylo(1 | tagA | species)),
          mu2 = @formula(y2 ~ 1 + phylo(1 | tagB | species)),
          sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
          rho12 = @formula(rho12 ~ 1))
      @test_throws ErrorException DRModels._bivariate_q4_marker(_rhs_dict(q2_tag))

      mixed_tag = bf(
          mu1 = @formula(y1 ~ 1 + phylo(1 | tagA | species)),
          mu2 = @formula(y2 ~ 1 + phylo(1 | species)),
          sigma1 = @formula(sigma1 ~ 1 + phylo(1 | species)),
          sigma2 = @formula(sigma2 ~ 1 + phylo(1 | species)),
          rho12 = @formula(rho12 ~ 1))
      @test_throws ErrorException DRModels._bivariate_q4_marker(_rhs_dict(mixed_tag))
  end
end

@testset "cov 1127 w5: gaussian_lss.jl" begin
  @testset "_lss_only_gaussian_guard refuses non-Gaussian LSS sd()" begin
      f = bf(@formula(y ~ 1 + (1 | g)), @formula(sigma ~ 1), @formula(sd(g) ~ 1))
      @test_throws ArgumentError DRModels._lss_only_gaussian_guard(f, Gamma())
      msg = ""
      try
          DRModels._lss_only_gaussian_guard(f, Gamma())
      catch e
          msg = sprint(showerror, e)
      end
      @test occursin("Gaussian LSS", msg)
      @test occursin("Gamma", msg)
  end

  @testset "_sd_target_block disambiguates multiple sd() targets" begin
      f = bf(@formula(y ~ 1 + (1 | g) + (1 | h)), @formula(sigma ~ 1),
             @formula(sd(g) ~ 1), @formula(sd(h) ~ 1))
      parts = DRModels._sd_target_parts(f)
      @test length(parts) == 2
      b1 = DRModels._sd_target_block(parts[1].target; disambiguate = true)
      b2 = DRModels._sd_target_block(parts[2].target; disambiguate = true)
      @test b1 !== :sd && b2 !== :sd && b1 !== b2
      @test DRModels._sd_target_block(parts[1].target) === :sd
  end
end

@testset "cov 1127 w5: temporal.jl" begin
  function _homtoep_panel()
      rng = StableRNG(1127)
      S, K = 8, 4
      id = repeat(string.(1:S), inner = K)
      occ = repeat(0:(K - 1), outer = S)
      x = randn(rng, S * K)
      y = randn(rng, S * K)
      return (; y, x, id, occ)
  end

  f = bf(@formula(y ~ x + temporal(1 | id, occ, homtoep)), @formula(sigma ~ 1))

  @testset "temporal_parameters and profile refusals" begin
      dat = _homtoep_panel()
      fit = drm(f, Gaussian(); data = dat, control = DRMControl(se = false))
      @test fit.converged
      tp = temporal_parameters(fit)
      @test tp.structure === :homtoep
      @test tp.cor !== nothing && length(tp.cor) == 3

      plain = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = dat)
      @test_throws ArgumentError temporal_parameters(plain)

      σ_idx = findfirst(k -> begin
          p, _ = DRModels._coef_metadata(fit, k)
          p === :sigma
      end, 1:length(fit.theta))
      @test σ_idx !== nothing
      @test_throws ArgumentError DRModels._homtoep_refuse_nonmean_index(fit, "profile_curve", σ_idx)
  end

  @testset "bf refuses temporal() on bivariate non-mean slots" begin
      @test_throws ArgumentError bf(
          mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
          sigma1 = @formula(sigma1 ~ 1 + temporal(1 | id, occ, ar1)),
          sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1))
  end
end