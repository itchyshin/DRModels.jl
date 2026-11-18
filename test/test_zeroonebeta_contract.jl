# ZeroOneBeta() contract suite (test-only): refusal messages, the
# parameter / fitted / simulate / post-fit contract on a fixed-effect fit,
# the no-exact-1s sample (#986) pinned to its CURRENT behaviour, and row-order
# invariance. Recovery lives in test_zeroonebeta.jl. Refs #975, #986.
using DRModels
using Test, Random
import Distributions

_zob_logistic(x) = 1 / (1 + exp(-x))

# n rows; boundary with prob `zoi` (P(1 | boundary) = `coi`), else Beta(μ, φ).
function _zob_data(n; zoi = 0.2, coi = 0.5, φ = 10.0, no_ones = false, seed = 20261019)
    rng = MersenneTwister(seed)
    x = randn(rng, n)
    y = Vector{Float64}(undef, n)
    for i in 1:n
        μ = _zob_logistic(log(0.4 / 0.6) + 0.5 * x[i])
        if rand(rng) < zoi
            y[i] = (!no_ones && rand(rng) < coi) ? 1.0 : 0.0
        else
            y[i] = rand(rng, Distributions.Beta(μ * φ, (1 - μ) * φ))
        end
    end
    return (; y, x, g = string.(rand(rng, 1:20, n)), v = fill(0.1, n))
end

# true iff `f()` throws an ErrorException whose message matches `rx`
function _zob_thrown(rx, f)
    try
        f()
    catch e
        return e isa ErrorException && occursin(rx, sprint(showerror, e))
    end
    return false
end

const _ZOB_F = bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(zoi ~ 1), @formula(coi ~ 1))

@testset "ZeroOneBeta contract" begin
    d = _zob_data(1500)

    @testset "refusals" begin
        bad = (; d.x, y = [1.2; d.y[2:end]])
        @test_throws ErrorException drm(_ZOB_F, ZeroOneBeta(); data = bad)
        @test _zob_thrown(r"requires responses in the closed interval", () -> drm(_ZOB_F, ZeroOneBeta(); data = bad))
        neg = (; d.x, y = [-0.1; d.y[2:end]])
        @test _zob_thrown(r"requires responses in the closed interval", () -> drm(_ZOB_F, ZeroOneBeta(); data = neg))

        @test _zob_thrown(r"does not support meta_V", () -> drm(
            bf(@formula(y ~ x + meta_V(v)), @formula(sigma ~ 1), @formula(zoi ~ 1), @formula(coi ~ 1)),
            ZeroOneBeta(); data = d))

        for p in (:sigma, :zoi, :coi)
            fs = Dict{Symbol,Any}(:sigma => @formula(sigma ~ 1), :zoi => @formula(zoi ~ 1), :coi => @formula(coi ~ 1))
            fs[p] = p === :sigma ? @formula(sigma ~ 1 + (1 | g)) :
                    p === :zoi ? @formula(zoi ~ 1 + (1 | g)) : @formula(coi ~ 1 + (1 | g))
            @test _zob_thrown(r"only the mean formula may carry a random effect", () -> drm(
                bf(@formula(y ~ x), fs[:sigma], fs[:zoi], fs[:coi]), ZeroOneBeta(); data = d))
        end

        @test _zob_thrown(r"cannot be combined with ordinary random effects", () -> drm(
            bf(@formula(y ~ x + phylo(1 | g) + (1 | g)), @formula(sigma ~ 1), @formula(zoi ~ 1), @formula(coi ~ 1)),
            ZeroOneBeta(); data = d))

        for (p, rx) in ((:sigma, r"constant `sigma ~ 1`"), (:zoi, r"constant `zoi ~ 1`"), (:coi, r"constant `coi ~ 1`"))
            fs = Dict{Symbol,Any}(:sigma => @formula(sigma ~ 1), :zoi => @formula(zoi ~ 1), :coi => @formula(coi ~ 1))
            fs[p] = p === :sigma ? @formula(sigma ~ x) : p === :zoi ? @formula(zoi ~ x) : @formula(coi ~ x)
            @test _zob_thrown(rx, () -> drm(bf(@formula(y ~ x + phylo(1 | g)), fs[:sigma], fs[:zoi], fs[:coi]),
                ZeroOneBeta(); data = d))
            @test _zob_thrown(rx, () -> drm(bf(@formula(y ~ x + relmat(1 | g)), fs[:sigma], fs[:zoi], fs[:coi]),
                ZeroOneBeta(); data = d))
        end

        @test _zob_thrown(r"needs `tree = \.\.\.`", () -> drm(
            bf(@formula(y ~ x + phylo(1 | g)), @formula(sigma ~ 1), @formula(zoi ~ 1), @formula(coi ~ 1)),
            ZeroOneBeta(); data = d))

        @test _zob_thrown(r"not `\(0 \+ x \| g\)`", () -> drm(
            bf(@formula(y ~ x + (0 + x | g)), @formula(sigma ~ 1), @formula(zoi ~ 1), @formula(coi ~ 1)),
            ZeroOneBeta(); data = d))
    end

    fit = drm(_ZOB_F, ZeroOneBeta(); data = d)

    @testset "parameter contract" begin
        @test any(==(0.0), d.y) && any(==(1.0), d.y)
        @test fit.converged
        pp = predict_parameters(fit, d)
        for k in (:mu, :zoi, :coi)
            @test all(v -> 0 < v < 1, pp[k])
        end
        @test pp[:zoi][1] ≈ 0.2 atol = 0.05
        @test pp[:coi][1] ≈ 0.5 atol = 0.10
        # fitted = unconditional mean: zoi*coi + (1 - zoi)*mu (docstring)
        @test fitted(fit) ≈ pp[:zoi] .* pp[:coi] .+ (1 .- pp[:zoi]) .* pp[:mu] atol = 1e-10

        s = simulate(fit; rng = MersenneTwister(7))
        @test length(s) == length(d.y)
        @test all(v -> 0 <= v <= 1, s)
        p0(v) = count(==(0.0), v) / length(v)
        p1(v) = count(==(1.0), v) / length(v)
        @test p0(s) > 0 && p1(s) > 0
        @test abs(p0(s) - p0(d.y)) < 0.05
        @test abs(p1(s) - p1(d.y)) < 0.05

        @test all(isfinite, residuals(fit))
        V = vcov(fit)
        @test V == V'
        @test size(V) == (length(coef(fit)), length(coef(fit)))
        ci = confint(fit)
        @test length(ci) == length(coef(fit))
        @test all(c -> c.lower < c.estimate < c.upper, ci)
        @test [c.estimate for c in ci] ≈ coef(fit)
        @test aic(fit) ≈ -2 * loglik(fit) + 2 * dof(fit)
        @test dof(fit) == 5
    end

    # #986 situation: no exact 1s. Records CURRENT behaviour only; if any of
    # these fail the behaviour changed (intended or not) and #986 should be consulted.
    @testset "no exact 1s: current behaviour (#986)" begin
        d2 = _zob_data(1500; no_ones = true)
        @test count(==(0.0), d2.y) > 0 && count(==(1.0), d2.y) == 0
        fit2 = @test_logs (:warn, r"Hessian is numerically singular") match_mode = :any drm(
            _ZOB_F, ZeroOneBeta(); data = d2)
        @test fit2.converged                              # reported converged despite the degenerate coi
        @test coef(fit2, :coi)[1] < -29                   # coi runs off to the logit clamp (-30)
        @test vcov(fit2)[end, end] == 0.0                 # SE of coi is exactly zero, not NaN or large
        @test all(isfinite, vcov(fit2))
        @test predict_parameters(fit2, d2)[:coi][1] < 1e-10
    end

    @testset "row-order invariance" begin
        p = randperm(MersenneTwister(3), length(d.y))
        fitp = drm(_ZOB_F, ZeroOneBeta(); data = (; y = d.y[p], x = d.x[p]))
        @test loglik(fitp) ≈ loglik(fit) atol = 1e-8
    end
end
