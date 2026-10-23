# Wald inference for fitted DRModels models: standard errors from the observed
# information (vcov) and Wald confidence intervals. Mirrors drmTMB's default
# `confint(..., method = "wald")`.
using DRModels
using Test, Random, LinearAlgebra

@testset "Wald inference: stderror + confint" begin
    Random.seed!(20260602)
    n = 3000
    x = randn(n)
    βμ = [0.5, -0.8]
    βσ = [-0.3, 0.4]
    y = βμ[1] .+ βμ[2] .* x .+ exp.(βσ[1] .+ βσ[2] .* x) .* randn(n)
    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian(); data = (; y, x))

    se = stderror(fit)
    @test se ≈ sqrt.(diag(vcov(fit)))
    @test all(se .> 0)

    ci = confint(fit; level = 0.95)
    @test length(ci) == 4
    @test all(c.lower < c.estimate < c.upper for c in ci)

    # confint is exactly estimate ± z·se (deterministic Wald property)
    z = 1.959963984540054      # qnorm(0.975)
    for (k, c) in enumerate(ci)
        @test c.lower ≈ c.estimate - z * se[k]
        @test c.upper ≈ c.estimate + z * se[k]
    end

    # μ estimates recover the truth (loose), in [intercept, slope] order
    mu_ci = filter(c -> c.param === :mu, ci)
    @test [c.estimate for c in mu_ci] ≈ βμ atol = 0.1

    # narrower confidence level ⇒ narrower interval
    ci90 = confint(fit; level = 0.90)
    @test (ci90[1].upper - ci90[1].lower) < (ci[1].upper - ci[1].lower)
end

@testset "Wald SE at a singular boundary (non-PD vcov ⇒ Inf, not NaN)" begin
    # At the Watanabe-singular boundary the observed information is not
    # positive-definite, so inv(I_obs) carries a non-positive variance for the
    # unidentified direction. stderror must report Inf there (not NaN), and the
    # Wald interval must be unbounded — matching the experimental infer_q4.jl
    # boundary handling that this wires into the public API (#10).
    blocks = [:mu => 1:2]
    coefnames = [:mu => ["(Intercept)", "x"]]
    theta = [0.5, -0.8]
    # slot 1 identified (var 0.04); slot 2 on the boundary (negative variance)
    V = [0.04 0.0; 0.0 -1.0e-6]
    empty = Dict{Symbol,Vector{Float64}}()
    fit = DRModels.DrmFit(Gaussian(), blocks, coefnames, theta, V, -10.0, 100, false,
                     empty, empty, empty)

    se = stderror(fit)
    @test se[1] ≈ 0.2
    @test isinf(se[2]) && se[2] > 0          # undefined SE, not NaN
    @test !any(isnan, se)

    ci = confint(fit; method = :wald)
    @test ci[1].lower < ci[1].estimate < ci[1].upper      # identified: finite CI
    @test ci[2].lower == -Inf && ci[2].upper == Inf       # unidentified: unbounded
    @test !any(isnan(c.lower) || isnan(c.upper) for c in ci)
end
