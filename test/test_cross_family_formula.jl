# test_cross_family_formula.jl — A11: the formula front end for the cross-family
# latent-rho route.
#
# WHY. `fit_mixed_family` takes RAW DESIGN MATRICES and is not exported, so
# cross-family was the only fit in DRModels.jl a user reached by hand-building
# matrices; everything else is `drm(bf(...), Family(); data = …)`, and drmTMB
# spells the same model with its ordinary bundle plus
# `family = c(gaussian(), poisson())`. The `cross_family_latent` capability row
# names that gap — "resolve the mixed-family API mismatch" — as the blocker
# before any public promotion.
#
# THE LOAD-BEARING TEST is equivalence: the formula route must produce a fit
# IDENTICAL to the matrix call it replaces. A front end that quietly builds a
# different design would be worse than no front end.

using DRModels
using Test
using Random
using LinearAlgebra
using Statistics

function _xfam_fixture(seed::Int; n = 220, rho = 0.5)
    rng = MersenneTwister(seed)
    x = randn(rng, n)
    z1 = randn(rng, n)
    z2 = rho .* z1 .+ sqrt(1 - rho^2) .* randn(rng, n)
    y1 = 0.3 .+ 0.7 .* x .+ z1                      # gaussian axis
    y2 = Float64.([rand(rng) < 1 / (1 + exp(-(0.2 + 0.5 * x[i] + z2[i]))) for i in 1:n])
    (; x, y1, y2)
end

const _BF_XFAM = bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
                    sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1))

@testset "cross-family formula front end (A11)" begin

    @testset "formula route == the hand-built matrix call it replaces" begin
        d = _xfam_fixture(31)
        X = hcat(ones(length(d.x)), d.x)

        viaformula = drm(_BF_XFAM, (Gaussian(), DRModels.Binomial()); data = d, confint = false)
        viamatrix = DRModels.fit_mixed_family(
            y1 = d.y1, X1 = X, fam1 = Gaussian(),
            y2 = d.y2, X2 = X, fam2 = DRModels.Binomial(),
            confint = false)

        # identical, not merely close: the front end must build the SAME design
        @test viaformula.loglik ≈ viamatrix.loglik atol = 1e-10
        @test viaformula.rho_latent ≈ viamatrix.rho_latent atol = 1e-10
        @test viaformula.β1 ≈ viamatrix.β1 atol = 1e-10
        @test viaformula.β2 ≈ viamatrix.β2 atol = 1e-10
        @test viaformula.converged == viamatrix.converged
    end

    @testset "it recovers a latent correlation" begin
        d = _xfam_fixture(32; rho = 0.6)
        fit = drm(_BF_XFAM, (Gaussian(), DRModels.Binomial()); data = d, confint = false)
        @test fit.converged
        @test fit.rho_latent > 0.2          # sign and rough magnitude, not a point claim
        @test fit.rho_latent < 0.95
    end

    @testset "the post-fit accessors work off the formula fit" begin
        d = _xfam_fixture(33)
        fit = drm(_BF_XFAM, (Gaussian(), DRModels.Binomial()); data = d, confint = false)
        @test mf_coef(fit) !== nothing
        @test isfinite(mf_aic(fit))
        @test length(mf_fitted(fit)) > 0
    end

    @testset "a mu-only bundle works (dispersionless second axis)" begin
        d = _xfam_fixture(34)
        fit = drm(bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x)),
                  (Gaussian(), DRModels.Binomial()); data = d, confint = false)
        @test fit.converged
        @test isfinite(fit.loglik)
    end

    @testset "refusals" begin
        d = _xfam_fixture(35)

        # rho12 as a FORMULA: the cross-family correlation is a latent SCALAR.
        # Accepting `rho12 ~ x` would imply a modelled per-observation
        # correlation this route does not fit -- the capability row's own
        # claim_boundary says public docs must not present rho12 formulas here.
        bad = bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
                 sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                 rho12 = @formula(rho12 ~ x))
        err = try
            drm(bad, (Gaussian(), DRModels.Binomial()); data = d, confint = false); nothing
        catch e; e end
        @test err isa ArgumentError
        @test occursin("latent", sprint(showerror, err))
        @test occursin("rho_latent", sprint(showerror, err))   # says where to read it

        # wrong number of families
        @test_throws ArgumentError drm(_BF_XFAM, (Gaussian(),); data = d)

        # random effects / structured markers are not in this route
        withre = bf(mu1 = @formula(y1 ~ x + (1 | g)), mu2 = @formula(y2 ~ x),
                    sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1))
        dre = (; d..., g = repeat(1:11, inner = 20))
        @test_throws ArgumentError drm(withre, (Gaussian(), DRModels.Binomial()); data = dre)
    end
end
