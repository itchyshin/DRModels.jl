# CumulativeLogit() contract suite for the fixed-effect fit with K = 3..5 ordered
# categories: response refusals, threshold naming and ordering, the probabilities
# implied by the thresholds and linear predictor, the shape of `fitted` / `predict`,
# the covariance / interval / information-criterion accessors, the log-likelihood
# against an independent sum, and `simulate`. Complements test_cumulative.jl
# (recovery only). Test-only: no src/ change.
#
# Shape facts the source fixes today (src/cumulative.jl): there is no per-category
# probability accessor. `fitted(fit)` is the expected category score
# `Σ_k k·P(y = k)` (length n, in (1, K)); `predict(fit, data)` is the linear
# predictor η = X·β on both scales; `predict_parameters` carries only `:mu` (also η). The
# category probabilities are therefore rebuilt here from `coef(fit, :cutpoints)`
# and η, independently of the fitter's helpers.
#
# Not asserted on purpose: predict(newdata) behaviour (#1141, src/cumulative_predict.jl)
# beyond the training data, and the `sigma` formula (#939, open design question).
using DRModels
using Test, Random, LinearAlgebra
import Distributions

_cl_logistic(z) = 1 / (1 + exp(-z))

# Thresholds from the raw increments: θ_1 = δ_1, θ_k = θ_{k-1} + exp(δ_k).
function _cl_thresholds(δ)
    θ = similar(δ); θ[1] = δ[1]
    for k in 2:length(δ); θ[k] = θ[k-1] + exp(δ[k]); end
    θ
end

# n × K matrix of P(y = k | η), built from the cumulative probabilities.
function _cl_probs(η, θ)
    K = length(θ) + 1
    P = Matrix{Float64}(undef, length(η), K)
    for i in eachindex(η)
        cum = [_cl_logistic(θ[k] - η[i]) for k in 1:K-1]
        for k in 1:K
            P[i, k] = k == 1 ? cum[1] : k == K ? 1 - cum[K-1] : cum[k] - cum[k-1]
        end
    end
    P
end

function _cl_draw(rng, η, θ, K)
    y = Vector{Float64}(undef, length(η))
    P = _cl_probs(η, θ)
    for i in eachindex(η)
        u = rand(rng); acc = 0.0; yi = K
        for k in 1:K
            acc += P[i, k]
            if u <= acc; yi = k; break; end
        end
        y[i] = yi
    end
    y
end

# Does `f()` throw? (the refusals below only need to know whether the call is refused)
function _cl_throws(f)
    try; f(); return false; catch; return true; end
end

function _cl_fit(K; n = 600, seed = 20261020, β = 0.7)
    rng = MersenneTwister(seed)
    θ = collect(range(-1.2, 1.4; length = K - 1))
    x = randn(rng, n)
    y = _cl_draw(rng, β .* x, θ, K)
    data = (; y, x)
    drm(bf(@formula(y ~ x)), CumulativeLogit(); data = data), data, θ
end

@testset "CumulativeLogit contract" begin

    @testset "refusals" begin
        x5 = [0.3, -0.2, 1.1, 0.5, -0.7]
        run(y; x = x5) = drm(bf(@formula(y ~ x)), CumulativeLogit(); data = (; y, x))

        # non-integer response values
        @test_throws r"ordered integer categories coded 1, 2" run([1.5, 2.0, 3.0, 1.0, 2.0])
        # a code below 1
        @test_throws r"ordered integer categories coded 1, 2" run([0.0, 1.0, 2.0, 1.0, 2.0])
        # a single level (all 1s: K would be 1)
        @test_throws r"K ≥ 2" run(fill(1.0, 5))

        # A response whose levels skip a category (here 3 of 1..4 is absent) is
        # NOT refused today: the fit runs, with a threshold gap running off to
        # -Inf. Expected behaviour is a refusal naming the missing level.
        ymiss = [1.0, 2.0, 4.0, 4.0, 2.0, 1.0, 4.0, 2.0]
        @test_broken _cl_throws(() -> run(ymiss; x = randn(MersenneTwister(1), 8)))
        # Only one observed level above 1 (all 2s, so level 1 is missing) is also
        # accepted today and reports is_converged == true.
        @test_broken _cl_throws(() -> run(fill(2.0, 5)))
    end

    for K in (3, 4, 5)
        @testset "K = $K fixed-effect contract" begin
            fit, data, _ = _cl_fit(K)
            n = length(data.y)
            @test is_converged(fit)

            # coefficient layout: the slope, then K-1 thresholds named theta1.. in order
            @test length(coef(fit)) == 1 + (K - 1)
            @test length(coef(fit, :mu)) == 1
            δ = coef(fit, :cutpoints)
            @test length(δ) == K - 1
            θ̂ = _cl_thresholds(δ)
            @test issorted(θ̂; lt = <=)                       # strictly increasing
            @test θ̂ ≈ fit.scales[:ordinal_cuts]
            ct = coeftable(fit)
            @test length(ct.rownms) == K
            @test ct.rownms[end-K+2:end] == ["cutpoints: theta$k" for k in 1:K-1]

            # probabilities from thresholds and η: rows sum to 1, all in (0, 1)
            η = predict(fit, data)
            @test η ≈ coef(fit, :mu)[1] .* data.x
            @test predict(fit, data; type = :link) ≈ η
            P = _cl_probs(η, θ̂)
            @test size(P) == (n, K)
            @test all(sum(P; dims = 2) .≈ 1)
            @test all(0 .< P .< 1)

            # fitted is the expected category score implied by those probabilities
            ŷ = fitted(fit)
            @test length(ŷ) == n
            @test all(1 .< ŷ .< K)
            @test ŷ ≈ P * collect(1.0:K)
            @test collect(keys(predict_parameters(fit, data))) == [:mu]
            @test predict_parameters(fit, data)[:mu] ≈ η          # the linear predictor, not the score

            # covariance, standard errors, intervals
            V = vcov(fit)
            k = length(coef(fit))
            @test size(V) == (k, k)
            @test V ≈ V' rtol = 1e-10
            @test all(diag(V) .> 0)
            @test all(eigvals(Symmetric(V)) .> 0)
            @test stderror(fit) ≈ sqrt.(diag(V))

            ci95 = confint(fit; level = 0.95)
            ci80 = confint(fit; level = 0.80)
            @test length(ci95) == k && length(ci80) == k
            @test [c.estimate for c in ci95] ≈ coef(fit)
            for (a, b) in zip(ci95, ci80)
                @test a.lower < b.lower < a.estimate < b.upper < a.upper
            end
            @test [(c.upper - c.lower) / 2 for c in ci95] ≈
                  Distributions.quantile(Distributions.Normal(), 0.975) .* stderror(fit) rtol = 1e-6

            # information criteria
            @test dof(fit) == k
            @test aic(fit) ≈ -2 * loglik(fit) + 2 * k
            @test bic(fit) ≈ -2 * loglik(fit) + k * log(n)
            @test nobs(fit) == n

            # loglik equals an independent sum of log category probabilities
            yi = round.(Int, data.y)
            @test sum(log(P[i, yi[i]]) for i in 1:n) ≈ loglik(fit) rtol = 1e-9
            @test loglik(fit) < 0

            # simulate: levels in 1:K, seed reproducible, shape sensible
            s1 = simulate(fit; rng = MersenneTwister(5))
            s2 = simulate(fit; rng = MersenneTwister(5))
            s3 = simulate(fit; rng = MersenneTwister(6))
            @test length(s1) == n
            @test s1 == s2
            @test s1 != s3
            @test all(v -> v == round(v) && 1 <= v <= K, s1)
            sim_freq = [count(==(Float64(c)), s1) / n for c in 1:K]
            obs_freq = [count(==(c), yi) / n for c in 1:K]
            @test maximum(abs.(sim_freq .- obs_freq)) < 0.08
        end
    end
end
