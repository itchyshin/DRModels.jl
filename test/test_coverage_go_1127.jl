# test_coverage_go_1127.jl — raise line coverage on src/[g-o]*.jl (27 Nov 2026).
# Focus: routes excluded from the default suite or thin error/edge branches with
# documented contracts. Small DGPs only (runtime budget ~60 s on CI hardware).

using DRModels
using Test, Random, LinearAlgebra, SparseArrays
import Distributions

const _fcloc = DRModels._fit_corr_locscale
const _corrkind = DRModels._corr_kind

@testset "coverage go: locscale_corr.jl (unified q2 correlated RE path)" begin
  Random.seed!(20261127_01)
  G = 30; m = 8; n = G * m
  g = repeat(1:G, inner = m)
  x = randn(n)
  Σ = [0.4^2 0.3 * 0.4 * 0.3; 0.3 * 0.4 * 0.3 0.3^2]
  B = cholesky(Symmetric(Σ)).L * randn(2, G)
  b0 = B[1, :]; b1 = B[2, :]
  β = [0.2, 0.35]
  λ = exp.(β[1] .+ β[2] .* x .+ b0[g] .+ b1[g] .* x)
  y = Float64.([rand(Distributions.Poisson(λi)) for λi in λ])
  Xμ = hcat(ones(n), x); nmμ = ["(Intercept)", "x"]
  Xψ = zeros(n, 0); nmσ = String[]
  xs = Float64.(x)

  @test _corrkind(Poisson()) === Val(:poisson)
  fit_p = _fcloc(Poisson(), Val(:poisson), :corr, y, Xμ, Xψ, xs, g, G,
                 nmμ, nmσ, "g"; link = :log, se = true)
  @test fit_p isa DrmFit && isfinite(loglik(fit_p)) && haskey(vc(fit_p), :g)

  μg = exp.(0.15 .+ 0.25 .* x .+ b0[g] .+ b1[g] .* x)
  yγ = Float64.([rand(Distributions.Gamma(6.0, μg[i] / 6.0)) for i in 1:n])
  fit_g = _fcloc(Gamma(), Val(:gamma), :corr, yγ, Xμ, Xψ, xs, g, G,
                 nmμ, nmσ, "g"; link = :log, se = true)
  @test isfinite(loglik(fit_g))

  ηb = 0.1 .+ 0.3 .* x .+ b0[g] .+ b1[g] .* x
  μb = 1 ./ (1 .+ exp.(-ηb))
  yb = Float64.([rand(Distributions.Beta(μb[i] * 12, (1 - μb[i]) * 12)) for i in 1:n])
  fit_b = _fcloc(Beta(), Val(:beta), :corr, yb, Xμ, Xψ, xs, g, G,
                 nmμ, nmσ, "g"; link = :logit, se = true, respobs = yb)
  @test isfinite(loglik(fit_b))

  # Independent slope `(0 + x | g)` — axis-1 loading, pinned axis-2 (rk = :slope).
  γ = repeat(1:G, inner = m)
  x₂ = randn(n)
  b₁ = 0.3 .* randn(G)
  μγ = exp.(0.1 .+ 0.2 .* x₂ .+ b₁[γ] .* x₂)
  yγ = Float64.([rand(Distributions.Gamma(5.0, μγ[i] / 5.0)) for i in 1:n])
  fit_slope = _fcloc(Gamma(), Val(:gamma), :slope, yγ, hcat(ones(n), x₂), ones(n, 1),
                     x₂, γ, G, ["(Intercept)", "x"], ["(Intercept)"], "g";
                     link = :log, se = true)
  @test fit_slope isa DrmFit
  @test vc(fit_slope)[:g][1, 1] > 0

  Zη, Zψ = DRModels._corr_loadings(:corr, x)
  @test size(Zη) == (n, 2) && size(Zψ) == (n, 2)
  @test all(Zη[:, 1] .== 1)
  Zηs, _ = DRModels._corr_loadings(:slope, x)
  @test Zηs[:, 2] == zeros(n)

  parsed = DRModels._parse_structured_slope((@formula(y ~ x + relmat(1 + x | g))).rhs)
  @test parsed == (:relmat, :x, :g)
  @test DRModels._parse_structured_slope((@formula(y ~ x + (1 + x | g))).rhs) === nothing
end

@testset "coverage go: locscale_sigma.jl (σ-axis RE nll + se=false)" begin
  Random.seed!(20261127_02)
  G = 20; n = G * 6
  g = repeat(1:G, inner = 6)
  x = randn(n)
  y = exp.(0.2 .+ 0.15 .* x .+ 0.25 .* randn(G)[g] .+ randn(n) .* 0.3)
  Xμ = hcat(ones(n), x); Xψ = ones(n, 1)
  Q = sparse(1.0I, G, G)
  Zη, Zψ = DRModels._sigma_re_loadings(n)
  θ = [0.1, 0.05, -0.2, log(0.35)]
  warm = Ref{Union{Nothing, Vector{Float64}}}(nothing)
  nll, a, ok = DRModels._ls_marginal_nll(Val(:gamma), y, Xμ * θ[1:2], Xψ * θ[3], g, G,
      DRModels.prior_precision(Q, DRModels._ls_inv2x2(DRModels._sigma_re_Lambda(θ[4]))),
      Zη, Zψ; a0 = warm[])
  @test ok && isfinite(nll)
  gvec = DRModels._sigma_re_grad(Val(:gamma), y, Xμ, Xψ, g, G, Q, θ, Zη, Zψ; warm = warm)
  @test length(gvec) == 4 && all(isfinite, gvec)

  fit = DRModels._fit_sigma_axis_re(Poisson(), Val(:poisson), y, Xμ, Xψ, g, G,
                                    ["(Intercept)", "x"], ["(Intercept)"], "g";
                                    link = :log, se = false, g_tol = 1e-5)
  @test fit isa DrmFit
  @test any(isnan, diag(fit.vcov))
  @test re_sd(fit)[:g_logsigma] > 0
end

@testset "coverage go: locscale_infer.jl (degraded Wald vcov)" begin
  # Near-singular mean–scale Λ (review #870); `_ls_vcov` must warn, not throw.
  Random.seed!(20261127_03)
  G = 15; n = G * 6
  gidx = repeat(1:G, inner = 6)
  x = randn(n)
  Xμ = hcat(ones(n), x)
  Λt = [0.25 0.1998; 0.1998 0.16]
  LΛ = cholesky(Symmetric(Λt + 1e-12I)).L
  A = [LΛ * randn(2) for _ in 1:G]
  y = Float64.([rand(Distributions.Poisson(exp(0.2 + 0.3x[i] + A[gidx[i]][1]))) for i in 1:n])
  Xψ = ones(n, 1)
  Q = sparse(1.0I, G, G)
  fit = DRModels._fit_locscale(Val(:poisson), y, Xμ, Xψ, gidx, G, Q; g_tol = 1e-8, se = true)
  @test isfinite(fit.nll)
  if fit.vcov === nothing
    @test fit.se === nothing
  else
  @test all(isfinite, fit.vcov) && all(>(0), fit.se)
  end
end

@testset "coverage go: joint_missing_uncertainty.jl" begin
  z = [-0.5, 0.2, 0.4, -0.2]
  x = Union{Missing, Float64}[0.5, missing, 0.9, missing]
  y = Union{Missing, Float64}[1.1, -0.1, missing, missing]
  X = hcat(ones(4), z)
  model = prepared_joint_model(y, x, X, ones(4, 1), X; predictor = :gaussian)
  θ = [0.1, -0.2, 0.5, 0.0, 0.1, 0.0, log(0.8)]
  V = Matrix{Float64}(I, 7, 7) * 0.02

  fail = DRModels._joint_imputation_uncertainty(model, θ, V;
        covariance_status = :hessian_unavailable)
  @test all(==("sdreport_failed"), fail.uncertainty_status)

  wrong_size = DRModels._joint_imputation_uncertainty(model, θ, V[1:3, 1:3])
  @test all(==("sdreport_unavailable"), wrong_size.uncertainty_status)

  nonfinite = DRModels._joint_imputation_uncertainty(model, θ, fill(NaN, 7, 7))
  @test all(==("sdreport_unavailable"), nonfinite.uncertainty_status)
end

@testset "coverage go: input_validation.jl (_maybe_table_column)" begin
  @test DRModels._maybe_table_column((a = 1,), :a) == 1
  @test DRModels._maybe_table_column(Dict("b" => 2), :b) == 2
  @test DRModels._maybe_table_column(Dict(:c => 3), :c) == 3
  @test DRModels._maybe_table_column((a = 1,), :missing_col) === nothing
end

@testset "coverage go: introspection.jl" begin
  @test DRModels._bridge_optimizer_string(:lbfgs) == "Optim.LBFGS"
  @test DRModels._bridge_optimizer_string(:bfgs) == "Optim.BFGS"
  @test DRModels._bridge_optimizer_string(:newton) == "Optim.Newton"
  @test ismissing(DRModels._bridge_optimizer_string(:unknown))

  phy = DRModels.random_balanced_tree(6; branch_length = 0.4)
  C = DRModels.sigma_phy_dense(phy; σ²_phy = 1.0)
  d = sqrt.(diag(C)); K = C ./ (d * d'); L = cholesky(Symmetric(K)).L
  G = 6; m = 3; species = repeat(1:G, inner = m); n = G * m
  rng = MersenneTwister(20261127_04)
  u1 = 0.5 .* (L * randn(rng, G))
  y = [1.0 + u1[species[i]] + 0.3 * randn(rng) for i in 1:n]
  fit = drm(bf(@formula(y ~ phylo(1 | species)), @formula(sigma ~ 1)), Gaussian();
            data = (y = y, species = species), tree = phy, control = DRMControl(se = false))
  @test fit.nll isa DRModels.LocOnlyObjective
  tg = profile_targets(fit)
  @test all(r -> r.profile_ready, tg)
  bd = bridge_diagnostics(fit)
  @test bd.route == "LocOnlyObjective"
  @test bd.converged
end