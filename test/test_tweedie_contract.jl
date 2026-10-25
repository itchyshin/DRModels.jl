# Tweedie() contract suite: the density's exact-zero mass and normalisation, every
# input refusal in `drm(..., Tweedie())` with its message, the accessor contract on
# a small fixed-effect fit, and the degenerate-response behaviour as it stands today.
# Complements test_tweedie.jl (recovery only) and the ranef / aghq / crossed files
# (random effects only). Test-only: no src/ change.
using DRModels
using Test, Random, Statistics, LinearAlgebra
import Distributions
using QuadGK: quadgk

function _rtweedie_contract(μ, φ, p)
    λ = μ^(2 - p) / (φ * (2 - p)); γ = φ * (p - 1) * μ^(p - 1); sh = (2 - p) / (p - 1)
    N = rand(Distributions.Poisson(λ))
    N == 0 ? 0.0 : rand(Distributions.Gamma(N * sh, γ))
end

@testset "Tweedie contract" begin

    @testset "density: point mass at 0 and normalisation" begin
        μ, φ = 2.0, 1.5
        for p in (1.2, 1.5, 1.9)
            mass0 = -μ^(2 - p) / (φ * (2 - p))
            @test DRModels._logpdf_tweedie(0.0, μ, φ, p) ≈ mass0 atol = 1e-12
            # P(y = 0) + ∫_0^∞ f(y) dy = 1; split the range so the quadrature
            # resolves the peak (mean μ) and the tail.
            cont = sum(first(quadgk(y -> exp(DRModels._logpdf_tweedie(y, μ, φ, p)), a, b;
                                    rtol = 1e-8)) for (a, b) in
                       ((0.0, 0.5), (0.5, 2.0), (2.0, 8.0), (8.0, 40.0), (40.0, Inf)))
            @test exp(mass0) + cont ≈ 1 atol = 1e-4
        end
    end

    @testset "refusals" begin
        Random.seed!(11)
        n = 60
        x = randn(n); g = repeat(1:6, inner = 10); h = repeat(1:5, 12)
        v = fill(0.1, n)
        y = abs.(randn(n)) .+ 0.1
        data = (; y, x, g, h, v, species = g, id = g, site = g)

        @test_throws ErrorException(
            "Tweedie() requires non-negative responses (zeros allowed)") drm(
            bf(@formula(y ~ x)), Tweedie(); data = merge(data, (; y = vcat(-0.5, y[2:end]))))

        @test_throws r"does not support meta_V" drm(
            bf(@formula(y ~ x + meta_V(v))), Tweedie(); data = data)

        for pname in (:sigma, :nu)
            f = pname === :sigma ?
                bf(@formula(y ~ x), @formula(sigma ~ 1 + (1 | g))) :
                bf(@formula(y ~ x), @formula(nu ~ 1 + (1 | g)))
            @test_throws r"not implemented for `sigma`/`nu`" drm(f, Tweedie(); data = data)
        end

        for (lab, f) in (
                "phylo" => bf(@formula(y ~ x + phylo(1 | species))),
                "relmat" => bf(@formula(y ~ x + relmat(1 | id))),
                "animal" => bf(@formula(y ~ x + animal(1 | id))),
                "spatial" => bf(@formula(y ~ x + spatial(1 | site))))
            @test_throws r"structured \(phylo/relmat/animal/spatial\) random effects on `mu` are not implemented" drm(
                f, Tweedie(); data = data)
        end

        # unsupported multiple / non-intercept random-effect shapes
        @test_throws r"multiple random effects only as crossed/nested intercepts" drm(
            bf(@formula(y ~ x + (1 | g) + (0 + x | h))), Tweedie(); data = data)
        @test_throws r"require a constant `sigma` formula" drm(
            bf(@formula(y ~ x + (1 | g) + (1 | h)), @formula(sigma ~ x)), Tweedie(); data = data)
        @test_throws r"require a constant `nu` formula" drm(
            bf(@formula(y ~ x + (1 | g) + (1 | h)), @formula(nu ~ x)), Tweedie(); data = data)
        @test_throws r"correlated random slopes `\(1 \+ x \| g\)` are not implemented" drm(
            bf(@formula(y ~ x + (1 + x | g))), Tweedie(); data = data)
    end

    @testset "accessor contract (fixed-effect fit)" begin
        Random.seed!(20261019)
        n = 800; x = randn(n)
        μ = exp.(0.5 .+ 0.3 .* x)
        y = [_rtweedie_contract(μ[i], 2.0, 1.5) for i in 1:n]
        data = (; y, x)
        obs_zero = count(==(0.0), y) / n
        @test 0.1 < obs_zero < 0.6

        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)), Tweedie(); data = data)
        k = length(coef(fit))
        @test k == 4
        @test is_converged(fit)

        V = vcov(fit)
        @test size(V) == (k, k)
        @test V ≈ V' rtol = 1e-10
        @test all(diag(V) .> 0)
        @test stderror(fit) ≈ sqrt.(diag(V))

        ci = confint(fit)
        @test length(ci) == k
        @test all(c.lower < c.estimate < c.upper for c in ci)
        @test [c.estimate for c in ci] ≈ coef(fit)

        ct = coeftable(fit)
        @test size(ct.cols[1], 1) == k
        @test length(ct.rownms) == k

        ŷ = fitted(fit)
        @test length(ŷ) == n
        @test all(ŷ .> 0)
        @test mean(ŷ) ≈ mean(y) rtol = 0.05

        @test predict(fit, data) ≈ ŷ

        pp = predict_parameters(fit, data)
        @test pp[:mu] ≈ ŷ                      # response scale, not log scale
        @test all(1 .< pp[:nu] .< 2)
        @test all(pp[:sigma] .> 0)

        s = simulate(fit; rng = MersenneTwister(5))
        @test length(s) == n
        @test all(s .>= 0)
        @test abs(count(==(0.0), s) / n - obs_zero) < 0.05

        r = residuals(fit)
        @test length(r) == n && all(isfinite, r)
        # no closed-form Tweedie CDF: quantile residuals are refused (see
        # test_quantile_residuals.jl)
        @test_throws ArgumentError residuals(fit; type = :quantile)

        @test aic(fit) ≈ -2 * loglik(fit) + 2 * dof(fit)
        @test dof(fit) == k
        @test isfinite(deviance(fit))
        @test isfinite(dof_residual(fit)) && dof_residual(fit) == n - k
    end

    # is_converged is true only when convergence_status is :converged (FX3b).
    # Both fits warn that the Hessian is numerically singular. The covariance
    # on the fit comes from that guard. convergence_status returns :degenerate
    # from its covariance block (rules 3 to 5). The all-zero log-likelihood
    # is about 0 and the one-spike log-likelihood is finite, so both stay
    # above the -1e15 sentinel.
    @testset "degenerate responses (current behaviour)" begin
        n = 200; x = randn(MersenneTwister(3), n)
        f = bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1))

        fit0 = @test_logs (:warn, r"numerically singular") (:warn, r"ended at a clamp") match_mode = :any drm(
            f, Tweedie(); data = (; y = zeros(n), x))
        @test !is_converged(fit0)
        @test coef(fit0, :mu)[1] < -20          # log mean driven to -inf
        @test loglik(fit0) ≈ 0 atol = 1e-6

        y1 = vcat(1.3, zeros(n - 1))
        fit1 = @test_logs (:warn, r"numerically singular") (:warn, r"ended at a clamp") match_mode = :any drm(
            f, Tweedie(); data = (; y = y1, x))
        @test !is_converged(fit1)
        @test coef(fit1, :nu)[1] < -11.9  # power pinned at its lower clamp
        @test isfinite(loglik(fit1))
    end
end
