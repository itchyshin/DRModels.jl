# test_bias_correct.jl — epsilon-method / generalized-delta bias correction for
# nonlinear derived quantities (TMB sdreport bias.correct analogue, #227 B11).
#
# Anchors that make this CI-verifiable:
#   1. IDENTITY: for a linear g the correction is exactly zero and the
#      estimate/SE/CI equal the plug-in Wald values.
#   2. CURVATURE: for g = exp on a coordinate with Wald mean m and variance v,
#      the corrected value matches the second-order expansion of the analytic
#      E[exp] = exp(m + v/2), i.e. exp(m)(1 + v/2), to tolerance — a closed-form
#      check proving the correction direction AND magnitude.
using DRModels
using Test, Random, LinearAlgebra
using Distributions: Normal, quantile

z95 = quantile(Normal(), 0.975)

@testset "bias_correct — identity anchor (linear g ⇒ zero correction)" begin
    # A hand-built (θ̂, V) so the Wald values are known exactly.
    θ̂ = [1.5, -0.7, 0.3]
    V = [0.04 0.01 0.0;
         0.01 0.09 0.0;
         0.0  0.0  0.25]

    # g(θ) = θ_2 — a coordinate projection (linear ⇒ Hessian 0).
    bc = bias_correct(θ̂, V, θ -> θ[2])
    @test bc.estimate ≈ θ̂[2]
    @test bc.bias ≈ 0.0 atol = 1e-12          # exact: H_g = 0
    @test bc.corrected ≈ bc.estimate          # no correction
    @test bc.se ≈ sqrt(V[2, 2])               # plug-in Wald SE
    @test bc.ci.lower ≈ bc.estimate - z95 * bc.se
    @test bc.ci.upper ≈ bc.estimate + z95 * bc.se

    # A general AFFINE g = a·θ + b is still exactly linear ⇒ zero correction,
    # and the delta SE equals √(aᵀ V a).
    a = [2.0, -1.0, 0.5]; b = 3.0
    bc2 = bias_correct(θ̂, V, θ -> dot(a, θ) + b)
    @test bc2.bias ≈ 0.0 atol = 1e-12
    @test bc2.corrected ≈ bc2.estimate
    @test bc2.se ≈ sqrt(dot(a, V * a))
end

@testset "bias_correct — curvature anchor (g = exp, closed form)" begin
    # Single curved coordinate: θ̂ = m, V = [v]. The epsilon-method correction is
    #   ½·tr(H_g·V) = ½·exp(m)·v,
    # so corrected = exp(m)(1 + v/2), which is the 2nd-order Taylor of the EXACT
    # E[exp(θ)] = exp(m + v/2). They agree to O(v²); check at small v.
    m = 0.4
    for v in (0.01, 0.02, 0.05)
        bc = bias_correct([m], reshape([v], 1, 1), θ -> exp(θ[1]))
        @test bc.estimate ≈ exp(m)                       # plug-in
        @test bc.bias ≈ 0.5 * exp(m) * v                 # epsilon correction (exact form)
        @test bc.corrected ≈ exp(m) * (1 + v / 2)        # closed form
        # Matches the analytic second-order expectation to O(v²): the gap is the
        # remainder exp(m)(exp(v/2) − 1 − v/2) ≈ exp(m)·v²/8.
        analytic = exp(m + v / 2)
        @test bc.corrected ≈ analytic atol = exp(m) * v^2   # within the O(v²) remainder
        @test bc.corrected < analytic                    # 2nd-order undershoots exp(m+v/2)
        # Correction has the right SIGN: exp is convex ⇒ positive bias.
        @test bc.bias > 0
        # Delta SE: ∇g = exp(m), so se = exp(m)·√v.
        @test bc.se ≈ exp(m) * sqrt(v)
    end
end

@testset "bias_correct — quadratic g is exact (g = θ², closed form)" begin
    # For g(θ) = θ² with θ̂ = N(m, v): E[θ²] = m² + v EXACTLY, and the
    # epsilon-method correction ½·H·v = ½·2·v = v is exact (g is quadratic ⇒ the
    # 2nd-order Taylor is exact, no remainder).
    m, v = 1.3, 0.2
    bc = bias_correct([m], reshape([v], 1, 1), θ -> θ[1]^2)
    @test bc.estimate ≈ m^2
    @test bc.bias ≈ v
    @test bc.corrected ≈ m^2 + v          # EXACT E[θ²]
    @test bc.se ≈ abs(2m) * sqrt(v)       # ∇g = 2m
end

@testset "bias_correct — cross terms via off-diagonal covariance (g = θ₁θ₂)" begin
    # g = θ₁θ₂, H_g = [[0,1],[1,0]] ⇒ ½·tr(H·V) = V₁₂. And E[θ₁θ₂] = m₁m₂ + Cov,
    # so corrected = m₁m₂ + V₁₂ is EXACT (bilinear ⇒ 2nd-order exact).
    m = [0.8, -0.5]
    V = [0.10 0.03; 0.03 0.07]
    bc = bias_correct(m, V, θ -> θ[1] * θ[2])
    @test bc.estimate ≈ m[1] * m[2]
    @test bc.bias ≈ V[1, 2]
    @test bc.corrected ≈ m[1] * m[2] + V[1, 2]
end

@testset "bias_correct — level controls CI width; se=0 ⇒ point interval" begin
    bc = bias_correct([0.4], reshape([0.02], 1, 1), θ -> exp(θ[1]))
    bc90 = bias_correct([0.4], reshape([0.02], 1, 1), θ -> exp(θ[1]); level = 0.90)
    @test (bc90.ci.upper - bc90.ci.lower) < (bc.ci.upper - bc.ci.lower)
    @test_throws ArgumentError bias_correct([0.0], reshape([1.0], 1, 1), θ -> θ[1]; level = 1.5)

    # Degenerate covariance ⇒ zero SE ⇒ CI collapses to the corrected point.
    bc0 = bias_correct([0.4], reshape([0.0], 1, 1), θ -> exp(θ[1]))
    @test bc0.se == 0.0
    @test bc0.ci.lower == bc0.ci.upper == bc0.corrected
end

@testset "bias_correct — on a real DrmFit (DrmFit method == manual θ̂,V form)" begin
    Random.seed!(20260607)
    n = 4000
    x = randn(n)
    βμ = [0.5, -0.8]
    βσ = [-0.3, 0.4]
    y = βμ[1] .+ βμ[2] .* x .+ exp.(βσ[1] .+ βσ[2] .* x) .* randn(n)
    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian(); data = (; y, x))

    θ̂ = coef(fit); V = vcov(fit)

    # Linear g on a real fit: pick the μ-slope index. Correction ~ 0, Wald exact.
    kmu = fit.blocks[findfirst(p -> p.first === :mu, fit.blocks)].second
    islope = kmu[2]
    bclin = bias_correct(fit, θ -> θ[islope])
    @test bclin.bias ≈ 0.0 atol = 1e-10
    @test bclin.corrected ≈ bclin.estimate
    @test bclin.se ≈ stderror(fit)[islope]
    @test bclin.ci.lower ≈ bclin.estimate - z95 * bclin.se
    @test bclin.ci.upper ≈ bclin.estimate + z95 * bclin.se

    # Curved DERIVED quantity: back-transformed σ at the σ intercept (exp(log σ)).
    ksig = fit.blocks[findfirst(p -> p.first === :sigma, fit.blocks)].second
    isig = ksig[1]
    gσ = θ -> exp(θ[isig])
    bc = bias_correct(fit, gσ)
    # DrmFit method must agree with the explicit (θ̂, V) form.
    bc_manual = bias_correct(θ̂, V, gσ)
    @test bc.estimate ≈ bc_manual.estimate
    @test bc.corrected ≈ bc_manual.corrected
    @test bc.se ≈ bc_manual.se
    @test bc.ci.lower ≈ bc_manual.ci.lower
    @test bc.ci.upper ≈ bc_manual.ci.upper
    # Convex transform ⇒ positive bias; correction matches ½·exp(m)·v with the
    # actual fit's working-scale mean/variance for this single coordinate.
    m = θ̂[isig]; v = V[isig, isig]
    @test bc.bias ≈ 0.5 * exp(m) * v rtol = 1e-8
    @test bc.bias > 0
    @test bc.corrected > bc.estimate
end
