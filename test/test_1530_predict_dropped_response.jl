# #1530 / #1540: after an observed-rows fit, `predict` on training `newdata` that
# still carries the response must mark dropped rows as `missing`, not finite values.

using DRModels
using Test
using StatsModels: @formula

@testset "#1530 predict masks rows dropped for a missing/NaN response" begin
    @testset "Poisson counts" begin
        n = 60
        x = [sin(i) for i in 1:n]
        c = Union{Missing, Int}[(i % 6 == 0) ? missing : round(Int, 3 + 2 * x[i]) for i in 1:n]
        data = (; c, x)
        fit = @test_logs (:warn, r"missing/NaN response") drm(
            bf(@formula(c ~ x), @formula(sigma ~ 1)), Poisson(); data)
        @test nobs(fit) == 50
        @test length(fitted(fit)) == 50
        pred = predict(fit, data)
        @test length(pred) == n
        @test count(ismissing, pred) == 10
        @test all(isfinite, filter(!ismissing, pred))
        keep = findall(!ismissing, data.c)
        @test pred[keep] ≈ predict(fit, (; c = c[keep], x = x[keep])) atol = 1e-10
        pr = predict(fit, data; se = true)
        @test length(pr.prediction) == n
        @test count(ismissing, pr.prediction) == 10
        @test count(ismissing, pr.se) == 10
    end

    @testset "Gaussian location–scale" begin
        n = 60
        x = [sin(i) for i in 1:n]
        y = Union{Missing, Float64}[(i % 6 == 0) ? missing : 0.5 + x[i] + 0.1 * sin(2i) for i in 1:n]
        data = (; y, x)
        fit = @test_logs (:warn, r"missing/NaN response") drm(
            bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data)
        @test nobs(fit) == 50
        pred = predict(fit, data)
        @test length(pred) == n
        @test count(ismissing, pred) == 10
    end
end

@testset "#1540 predict(type=:mean) marks dropped response rows missing" begin
    n = 20
    x = collect(range(-1, 1; length = n))
    y = Union{Missing, Float64}[clamp(0.3 + 0.4 * x[i], 0.05, 0.95) for i in 1:n]
    y[3] = missing
    data = (; y, x)
    fit = @test_logs match_mode = :any (:warn, r"missing/NaN response") drm(
        bf(@formula(y ~ x), @formula(sigma ~ 1),
           @formula(zoi ~ 1), @formula(coi ~ 1)),
        ZeroOneBeta(); data)
    @test nobs(fit) == n - 1
    meanp = predict(fit, data; type = :mean)
    @test length(meanp) == n
    @test ismissing(meanp[3])
    @test all(isfinite, filter(!ismissing, meanp))
    diff = data.y .- meanp
    @test ismissing(diff[3])
    @test all(ismissing.(diff[findall(ismissing, data.y)]))
end