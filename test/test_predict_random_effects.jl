# #1092: `predict` on a random-effect fit is the prediction for an average group,
# with every random effect set to zero, `g⁻¹(Xβ̂)`. It is NOT the marginal mean
# averaged over the random-effect distribution, which differs under a nonlinear
# link. These checks pin the behaviour the `predict` docstring documents: the
# b = 0 identity, that grouping columns in `newdata` are ignored, the in-sample
# agreement with `fitted` on the ordinary `(1 | g)` route, and the log-link
# marginal-mean recipe `predict .* exp(σ_b²/2)` against an independent quadrature.
using DRModels
using Test, StableRNGs, QuadGK
import Distributions

@testset "#1092 predict sets random effects to zero (b = 0), not the marginal mean" begin
    # E[h(Z)] for Z ~ N(0, 1), by Gauss–Kronrod quadrature on [-12, 12]; the
    # truncated tails cost under 1e-25 of each integral here (σ_b ≲ 1.5), far
    # below the test tolerances.
    enorm(h) = quadgk(z -> h(z) * exp(-z^2 / 2) / sqrt(2π), -12.0, 12.0; rtol = 1e-12)[1]

    @testset "Poisson (1 | g): log link" begin
        rng = StableRNG(1092)
        G = 30; m = 15; n = G * m
        g = repeat(1:G, inner = m); x = randn(rng, n); b = 0.8 .* randn(rng, G)
        y = Float64[rand(rng, Distributions.Poisson(exp(0.3 + 0.5 * x[i] + b[g[i]]))) for i in 1:n]
        fit = drm(bf(@formula(y ~ x + (1 | g))), DRModels.Poisson(); data = (; y, x, g))
        β = coef(fit, :mu); σb = re_sd(fit)[:g]
        xn = [-1.0, 0.0, 1.0]; η = β[1] .+ β[2] .* xn
        p = predict(fit, (; x = xn))

        @test predict(fit, (; x = xn); type = :link) ≈ η rtol = 1e-12
        @test p ≈ exp.(η) rtol = 1e-12                           # g⁻¹(Xβ̂) at b = 0
        # The grouping column is not used: fitted levels, unseen levels and no
        # column at all give the same prediction.
        @test predict(fit, (; x = xn, g = [1, 2, 3])) == p
        @test predict(fit, (; x = xn, g = [997, 998, 999])) == p
        # This route stores the b = 0 value as `fitted`, so in-sample they agree.
        @test predict(fit, (; y, x, g)) ≈ fitted(fit) rtol = 1e-10

        # The marginal mean E_b[exp(η + b)], b ~ N(0, σ_b²), is exp(η + σ_b²/2):
        # the docstring's recipe, checked against quadrature, and strictly larger.
        @test σb > 0.1
        marg = [enorm(z -> exp(ηi + σb * z)) for ηi in η]
        @test p .* exp(σb^2 / 2) ≈ marg rtol = 1e-8
        @test all(marg .> p)
    end

    @testset "Binomial (1 | g), 0/1 response: logit link" begin
        rng = StableRNG(10920)
        G = 40; m = 15; n = G * m
        g = repeat(1:G, inner = m); x = randn(rng, n); b = randn(rng, G)
        μ = [1 / (1 + exp(-(-0.3 + 0.8 * x[i] + b[g[i]]))) for i in 1:n]
        y = Float64[rand(rng) < μ[i] ? 1.0 : 0.0 for i in 1:n]
        fit = drm(bf(@formula(y ~ x + (1 | g))), DRModels.Binomial(); data = (; y, x, g))
        β = coef(fit, :mu); σb = re_sd(fit)[:g]
        xn = [-1.5, 1.5]; η = β[1] .+ β[2] .* xn
        p = predict(fit, (; x = xn))

        @test p ≈ 1 ./ (1 .+ exp.(-η)) rtol = 1e-12               # logistic(Xβ̂) at b = 0
        @test predict(fit, (; x = xn, g = [998, 999])) == p
        @test predict(fit, (; y, x, g)) ≈ fitted(fit) rtol = 1e-10

        # The logit marginal mean has no closed form; for η ≠ 0 and σ_b > 0 it
        # lies strictly closer to 1/2 than the b = 0 prediction.
        @test σb > 0.1
        @test all(!iszero, η)
        marg = [enorm(z -> 1 / (1 + exp(-(ηi + σb * z)))) for ηi in η]
        @test all(abs.(marg .- 0.5) .< abs.(p .- 0.5))
    end
end
