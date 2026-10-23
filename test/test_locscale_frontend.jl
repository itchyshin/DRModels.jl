# Public `drm()` routing for the non-Gaussian location–scale model (#202, 3b).
# The drmTMB/brms-style coupled tag `(1 | p | species)` shared by the mean and
# sigma formulas must route to the augmented-state engine and produce a DrmFit
# whose coefficients/Λ match a direct `_fit_locscale` call (parity), with a
# working summary and `vc` accessor.
using DRModels
using Test, Random, LinearAlgebra, SparseArrays
import Distributions

_nb2_draw_f(η, ψ) = (r = exp(ψ); μ = exp(η);
                     Float64(rand(Distributions.NegativeBinomial(r, r / (r + μ)))))

@testset "drm() location–scale routing via (1|p|group)" begin
    Random.seed!(2026)
    G = 25; m = 25; n = G * m
    species = repeat(1:G, inner = m)
    x = randn(n)
    Λt = [0.25 0.05; 0.05 0.16]
    LΛ = cholesky(Symmetric(Λt)).L
    A = [LΛ * randn(2) for _ in 1:G]
    y = [_nb2_draw_f(0.5 + 0.4x[i] + A[species[i]][1], 0.3 + 0.2x[i] + A[species[i]][2])
         for i in 1:n]
    data = (; y, x, species)

    fit = drm(bf(@formula(y ~ x + (1 | p | species)),
                 @formula(sigma ~ x + (1 | p | species))), NegBinomial2(); data = data)

    # Parity with a direct engine call on the same design.
    Xμ = hcat(ones(n), x); Xψ = hcat(ones(n), x)
    gidx, Gd = DRModels._group_index(species)
    Q = sparse(1.0 * I, Gd, Gd)
    # match the drm() default g_tol (NegBinomial2 uses 1e-8) so the fits coincide
    direct = DRModels._fit_locscale(Val(:nb2), Float64.(y), Xμ, Xψ, gidx, Gd, Q; g_tol = 1e-8, se = true)

    @test coef(fit, :mu) ≈ direct.beta_mu rtol = 1e-6
    @test coef(fit, :sigma) ≈ direct.beta_psi rtol = 1e-6
    @test DRModels.vc(fit)[:species] ≈ direct.Lambda rtol = 1e-6
    @test nobs(fit) == n
    @test isfinite(loglik(fit))
    @test all(isfinite, stderror(fit))

    # Summary renders, including the group-level covariance block.
    s = sprint(show, MIME("text/plain"), fit)
    @test occursin("Random-effect covariance", s)

    # Public Wald inference surface works end-to-end (mirrors drmTMB's confint).
    ci = confint(fit)                                  # method = :wald
    @test !isempty(ci)
    slope = first(r for r in ci if r.param === :mu && r.coef == "x")
    @test slope.lower < slope.estimate < slope.upper
    @test summary(fit) isa typeof(coeftable(fit))      # coeftable/summary build cleanly
end

@testset "drm() location–scale: a lone coupled tag is an error" begin
    Random.seed!(11)
    n = 60; x = randn(n); species = repeat(1:6, inner = 10)
    y = Float64.(rand(0:5, n))
    data = (; y, x, species)
    # Coupled tag on the mean only → must error (no partner on sigma).
    @test_throws Exception drm(bf(@formula(y ~ x + (1 | p | species)),
                                  @formula(sigma ~ x)), NegBinomial2(); data = data)
end
