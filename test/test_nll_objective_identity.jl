# test_nll_objective_identity.jl -- `fit.nll` must be the objective the fit reports.
#
# Found by hooking `_withnll` and comparing `fit.loglik` with `-fit.nll(fit.theta)`
# on every fit in the suite. Two defect classes:
#   1. A REML fit stored the plain ML objective in `fit.nll`
#      (Gaussian lss iid / structured / multi-component, dense bivariate
#      residual). `gradient(fit)` is documented as the gradient of the REML
#      criterion on a REML fit, and `check_drm` reads the same objective, so a
#      converged REML fit looked non-stationary (the ML score at a REML optimum
#      is O(1)).
#   2. A coupled location-scale fit stores a `LocScaleObjective` in ENGINE packing
#      while `fit.theta` is in `:recov` order, so `gradient(fit)`, `hessian(fit)` and
#      `profile_curve(fit, k)` evaluated the objective at the wrong point.
using DRModels
using Test, Random, LinearAlgebra, SparseArrays, Statistics
import Distributions

_nll_at_theta(fit) = DRModels._fit_objective(fit, "test")(Vector{Float64}(fit.theta))

@testset "fit.nll is the reported objective (loglik == -nll(theta))" begin
    @testset "Gaussian lss iid, REML" begin
        Random.seed!(20260828)
        G = 15; m = 6; n = G * m
        g = repeat(1:G, inner = m); x = randn(n); zg = randn(G); z = zg[g]
        b = exp.(0.2 .+ 0.5 .* zg) .* randn(G)
        y = 1.0 .+ 0.7 .* x .+ b[g] .+ exp.(-0.3 .+ 0.2 .* x) .* randn(n)
        data = (; y, x, z, g)
        f = bf(@formula(y ~ 1 + x + (1 | g)), @formula(sigma ~ 1 + x), @formula(sd(g) ~ 1 + z))
        for method in (:ML, :REML)
            fit = drm(f, Gaussian(); data = data, method = method)
            @test fit.converged
            @test loglik(fit) ≈ -_nll_at_theta(fit) atol = 1e-8
            @test maximum(abs, gradient(fit)) < 1e-4
            @test check_drm(fit).ok
            # vcov is the inverse Hessian of this same objective on this route
            @test inv(hessian(fit)) ≈ vcov(fit) rtol = 1e-5
        end
    end

    @testset "bivariate residual, REML" begin
        Random.seed!(20260905)
        n = 120; x = randn(n)
        y1 = 0.4 .+ 0.6 .* x .+ 0.5 .* randn(n)
        y2 = -0.2 .+ 0.3 .* x .+ 0.4 .* y1 .+ 0.5 .* randn(n)
        f = bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x), sigma1 = @formula(sigma1 ~ 1),
               sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1))
        for method in (:ML, :REML)
            fit = drm(f, Gaussian(); data = (; y1, y2, x), method = method)
            @test loglik(fit) ≈ -_nll_at_theta(fit) atol = 1e-8
            @test maximum(abs, gradient(fit)) < 1e-4
            @test check_drm(fit).ok
        end
    end

    @testset "sparse phylo lss, REML (algorithm = :sparse)" begin
        # depth-3 balanced tree, 8 tips, 3 observations per tip
        function node(prefix, depth, d)
            depth == 0 && return "$(prefix):$(1/d)"
            l = node(prefix * "a", depth - 1, d); r = node(prefix * "b", depth - 1, d)
            return depth == d ? "($l,$r);" : "($l,$r):$(1/d)"
        end
        phy = DRModels.augmented_phy(node("t", 3, 3))
        G = phy.n_leaves
        K0 = DRModels.sigma_phy_dense(phy; σ²_phy = 1.0)
        dK = sqrt.(diag(K0)); K = K0 ./ (dK * dK')
        rng = MersenneTwister(123)
        gidx = repeat(1:G, inner = 3); n = length(gidx)
        x1 = randn(rng, n); z1_g = randn(rng, G)
        a = exp.(-0.4 .+ 0.35 .* z1_g) .* (cholesky(Symmetric(K)).L * randn(rng, G))
        y = 1.2 .+ 0.5 .* x1 .+ a[gidx] .+ exp.(-1.1 .- 0.3 .* x1) .* randn(rng, n)
        data = (; y, x1, z1 = z1_g[gidx], species = String.(phy.leaf_names)[gidx])
        f = bf(@formula(y ~ x1 + phylo(1 | species)), @formula(sigma ~ x1),
               @formula(sd(species, phylogenetic) ~ z1))
        fit = drm(f, Gaussian(); data = data, tree = phy, algorithm = :sparse, method = :REML, g_tol = 1e-8)
        @test estimation_method(fit) === :REML
        @test loglik(fit) ≈ -_nll_at_theta(fit) atol = 1e-8
        @test maximum(abs, gradient(fit)) < 1e-3
    end

    @testset "coupled location-scale: objective is in coef(fit) order" begin
        Random.seed!(7)
        G = 40; m = 20; n = G * m
        sp = repeat(1:G, inner = m); x = randn(n)
        A = randn(G, 2) .* 0.5
        y = [(r = exp(-2 * (-0.35 + A[sp[i], 2])); μ = exp(0.3 + 0.45 * x[i] + A[sp[i], 1]);
              Float64(rand(Distributions.NegativeBinomial(r, r / (r + μ))))) for i in 1:n]
        data = (; y, x, species = sp)
        f = bf(@formula(y ~ 1 + x + (1 | p | species)), @formula(sigma ~ 1 + (1 | p | species)))
        fit = drm(f, NegBinomial2(); data = data)
        @test fit.nll isa DRModels.LocScaleObjective
        @test loglik(fit) ≈ -_nll_at_theta(fit) atol = 1e-6
        g = gradient(fit)
        @test length(g) == length(coef(fit))
        @test all(isfinite, g)
        H = hessian(fit)
        @test size(H) == (length(g), length(g)) && all(isfinite, H)
        # profile_curve is centred at the fit: deviance 0 at the estimate
        pc = profile_curve(fit, 1; npoints = 5, span = 1.0)
        @test minimum(pc.deviance) ≥ -1e-6
    end
end
