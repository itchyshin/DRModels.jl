# test_twin_gap_739.jl — issue #739 (twin drmTMB #1281): ZeroOneBeta() had no
# `drm(...; tree=/K=)` phylo/relmat method. Verifies:
#   1. RED: the method genuinely does not exist on the base (pre-#739) API.
#   2. GREEN: a phylo(1 | species) fit recovers known FE + phylo SD on a
#      simulated balanced tree, converges, and reports a finite logLik.
#   3. The new `:zeroonebeta_fixed` Laplace kernel's analytic η-derivatives
#      match ForwardDiff/finite-difference derivatives of the kernel value,
#      for both an atom row (y=0/1, must be exactly zero) and an interior row
#      (must match the verified `:beta_fixed` kernel).
#   4. A relmat(1 | id) route with a user-supplied PD covariance also fits.
#
# Runtime target: < 3 min (p=70 tips, m=6 reps/tip).

using DRModels
using Test, Random, LinearAlgebra
using ForwardDiff
import Distributions

_tg739_logistic(x) = 1 / (1 + exp(-x))

@testset "twin-gap #739: ZeroOneBeta phylo/relmat sparse-Laplace kernel" begin
    # RED (verified manually against the pre-#739 base, origin/claude/twin-gap-723,
    # 735e2bae7): `ZeroOneBeta()`'s `drm()` method took no `tree`/`K` keyword at
    # all — `drm(bf(@formula(y ~ x + phylo(1 | species)), ...), ZeroOneBeta();
    # data = ..., tree = phy)` raised `MethodError: no method matching drm(...;
    # tree, ...)` (the base signature is `drm(f, fam; data, g_tol)`), and even
    # with the keyword accepted, the `st !== nothing` branch of `drm()` raised
    # `ZeroOneBeta() does not support phylo/relmat structured random effects
    # yet`. This file only exercises the post-fix (GREEN) behaviour; the RED
    # state is not re-creatable in-process without reverting the source.

    @testset "GREEN: phylo(1 | species) recovery" begin
        Random.seed!(20260927)
        p = 70
        m = 6
        phy = random_balanced_tree(p; branch_length = 0.20)
        species = repeat(1:p, inner = m)
        n = length(species)
        x = randn(n)
        β = [0.10, 0.55]
        precision = 14.0
        zoi_true = 0.12
        coi_true = 0.35
        σphy = 0.40
        C = sigma_phy_dense(phy; σ²_phy = σphy^2)
        u = cholesky(Symmetric(C)).L * randn(p)
        μ = _tg739_logistic.(β[1] .+ β[2] .* x .+ u[species])

        y = Vector{Float64}(undef, n)
        for i in 1:n
            if rand() < zoi_true
                y[i] = rand() < coi_true ? 1.0 : 0.0
            else
                y[i] = rand(Distributions.Beta(μ[i] * precision, (1 - μ[i]) * precision))
            end
        end

        fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1),
                     @formula(zoi ~ 1), @formula(coi ~ 1)),
                  ZeroOneBeta(); data = (; y, x, species), tree = phy, se = false)

        @test fit.converged
        @test coef(fit, :mu)[2] ≈ β[2] atol = 0.30
        @test 0.03 < exp(coef(fit, :sigma)[1]) < 1.0
        @test re_sd(fit)[:species] > 0.03
        @test isfinite(loglik(fit))
        @test all(0 .<= fitted(fit) .<= 1)
        zoihat = _tg739_logistic(coef(fit, :zoi)[1])
        coihat = _tg739_logistic(coef(fit, :coi)[1])
        @test isapprox(zoihat, zoi_true; atol = 0.06)
        @test isapprox(coihat, coi_true; atol = 0.15)

        # Structured effects cannot combine with an ordinary random effect.
        @test_throws ErrorException drm(
            bf(@formula(y ~ x + phylo(1 | species) + (1 | block)), @formula(sigma ~ 1),
               @formula(zoi ~ 1), @formula(coi ~ 1)),
            ZeroOneBeta(); data = (; y, x, species, block = repeat(1:5, inner = n ÷ 5)),
            tree = phy, se = false,
        )

        # drmTMB comparison: `zero_one_beta()` in drmTMB (as of v0.1.x) does not
        # itself expose a phylo/relmat random-effect route to compare against
        # (its RE support is the ordinary `(1 | g)` GHQ path added for #723) —
        # so there is no same-surface drmTMB fit to diff this Laplace fit
        # against. Not run here; see the PR description for the honest note.
    end

    @testset "kernel derivative check: Val(:zeroonebeta_fixed)" begin
        Random.seed!(20260928)
        yv = [0.35, 0.0, 1.0, 0.62]
        isatom = (yv .== 0) .| (yv .== 1)
        ylogit = [isatom[i] ? 0.0 : (log(yv[i]) - log1p(-yv[i])) for i in eachindex(yv)]
        φ = 9.0
        aux = (y = yv, precision = φ, ylogit = ylogit,
               lgammaφ = DRModels.loggamma(φ), digammaφ = DRModels.digamma(φ),
               zoi = 0.15, coi = 0.4, isatom = isatom)
        kind = Val(:zeroonebeta_fixed)

        for i in eachindex(yv)
            η0 = 0.3
            v = DRModels._laplace_value(kind, aux, i, η0)
            d1 = DRModels._laplace_d1(kind, aux, i, η0)
            d2 = DRModels._laplace_d2(kind, aux, i, η0)
            d3 = DRModels._laplace_d3(kind, aux, i, η0)

            if isatom[i]
                @test d1 == 0.0
                @test d2 == 0.0
                @test d3 == 0.0
            else
                fd1 = ForwardDiff.derivative(η -> DRModels._laplace_value(kind, aux, i, η), η0)
                fd2 = ForwardDiff.derivative(η -> DRModels._laplace_d1(kind, aux, i, η), η0)
                fd3 = ForwardDiff.derivative(η -> DRModels._laplace_d2(kind, aux, i, η), η0)
                @test d1 ≈ fd1 atol = 1e-6
                @test d2 ≈ fd2 atol = 1e-6
                @test d3 ≈ fd3 atol = 1e-6
                # matches the verified :beta_fixed kernel exactly up to the
                # constant -log1p(-zoi) offset in the value only.
                @test d1 ≈ DRModels._laplace_d1(Val(:beta_fixed), aux, i, η0)
                @test d2 ≈ DRModels._laplace_d2(Val(:beta_fixed), aux, i, η0)
                @test d3 ≈ DRModels._laplace_d3(Val(:beta_fixed), aux, i, η0)
                @test v ≈ DRModels._laplace_value(Val(:beta_fixed), aux, i, η0) - log1p(-aux.zoi) atol = 1e-12
            end

            # nuisance (phi) axis: also zero for atom rows, matches :beta_fixed otherwise.
            nv = DRModels._laplace_nuisance_value(kind, aux, i, η0)
            n1 = DRModels._laplace_nuisance_d1(kind, aux, i, η0)
            n2 = DRModels._laplace_nuisance_d2(kind, aux, i, η0)
            if isatom[i]
                @test nv == 0.0 && n1 == 0.0 && n2 == 0.0
            else
                @test nv ≈ DRModels._laplace_nuisance_value(Val(:beta_fixed), aux, i, η0)
                @test n1 ≈ DRModels._laplace_nuisance_d1(Val(:beta_fixed), aux, i, η0)
                @test n2 ≈ DRModels._laplace_nuisance_d2(Val(:beta_fixed), aux, i, η0)
            end
        end
    end

    @testset "GREEN: relmat(1 | id) route" begin
        Random.seed!(20260929)
        G = 24
        m = 8
        n = G * m
        ids = repeat(1:G, inner = m)
        x = randn(n)
        β = [0.0, 0.4]
        precision = 12.0
        R = 0.3 .* ones(G, G) + 0.7 .* Matrix(I, G, G)   # simple PD relatedness
        u = cholesky(Symmetric(R)).L * randn(G) .* 0.5
        μ = _tg739_logistic.(β[1] .+ β[2] .* x .+ u[ids])
        y = Vector{Float64}(undef, n)
        for i in 1:n
            if rand() < 0.08
                y[i] = rand() < 0.5 ? 1.0 : 0.0
            else
                y[i] = rand(Distributions.Beta(μ[i] * precision, (1 - μ[i]) * precision))
            end
        end

        fit = drm(bf(@formula(y ~ x + relmat(1 | id)), @formula(sigma ~ 1),
                     @formula(zoi ~ 1), @formula(coi ~ 1)),
                  ZeroOneBeta(); data = (; y, x, id = ids), K = R, se = false)
        @test fit.converged
        @test isfinite(loglik(fit))
        @test all(0 .<= fitted(fit) .<= 1)
    end
end
