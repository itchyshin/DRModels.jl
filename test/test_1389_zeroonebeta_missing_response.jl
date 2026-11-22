using DRModels
using Test
using StatsModels: @formula

# #1389: a ZeroOneBeta fit whose response has missing values threw
# `KeyError: key :zoi not found` after the optimiser finished. `bf(y ~ x)` stores only the
# formulas that were written, and `predict_parameters` returned no `:zoi` / `:coi` for a fit
# that did not write `zoi ~ ...` / `coi ~ ...`, although both are estimated (intercept only).
# The reference numbers are drmTMB 0.8.1 on the same deterministic data
# (`zero_one_beta()`, 10 of 60 responses missing; recorded in the issue).
@testset "#1389 ZeroOneBeta with a missing response, no zoi/coi formula" begin
    n = 60; i = 1:n
    x = [sin(k) for k in i]
    g = [string("g", 1 + (k - 1) % 6) for k in i]
    y = Union{Missing,Float64}[
        (k % 6 == 0) ? missing :
        (k % 10 == 3) ? 0.0 :
        (k % 13 == 5) ? 1.0 :
        clamp(0.5 + 0.25 * sin(k) + 0.12 * cos(5k) + 0.1 * (((k - 1) % 6) - 2.5) / 2.5, 0.05, 0.95)
        for k in i]
    d = (; y, x, g)
    keep = findall(!ismissing, y)
    d_obs = (; y = Float64.(y[keep]), x = x[keep], g = g[keep])

    @testset "fixed effects: nobs and logLik equal drmTMB" begin
        fit = drm(bf(@formula(y ~ x)), ZeroOneBeta(); data = d)
        @test nobs(fit) == 50
        @test isapprox(loglikelihood(fit), 2.533589; atol = 1e-5)

        fit_s = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), ZeroOneBeta(); data = d)
        @test nobs(fit_s) == 50
        @test isapprox(loglikelihood(fit_s), 2.585723; atol = 1e-5)
    end

    @testset "random intercept: the fit no longer throws" begin
        fit = drm(bf(@formula(y ~ x + (1 | g))), ZeroOneBeta(); data = d)
        @test nobs(fit) == 50
        @test isfinite(loglikelihood(fit))
    end

    @testset "an explicit zoi ~ 1, coi ~ 1 gives the same fit" begin
        fit0 = drm(bf(@formula(y ~ x)), ZeroOneBeta(); data = d)
        fit1 = drm(bf(@formula(y ~ x), @formula(zoi ~ 1), @formula(coi ~ 1)),
                   ZeroOneBeta(); data = d)
        @test loglikelihood(fit0) == loglikelihood(fit1)
        @test fitted(fit0) == fitted(fit1)
    end

    @testset "predict_parameters and predict(type = :mean) on complete data" begin
        fit = drm(bf(@formula(y ~ x)), ZeroOneBeta(); data = d_obs)
        nd = (; x = [-0.5, 0.0, 0.7])
        explicit = drm(bf(@formula(y ~ x), @formula(zoi ~ 1), @formula(coi ~ 1)),
                       ZeroOneBeta(); data = d_obs)
        for type in (:response, :link)
            pp = predict_parameters(fit, nd; type)
            pe = predict_parameters(explicit, nd; type)
            @test Set(keys(pp)) == Set(keys(pe))
            @test haskey(pp, :zoi) && haskey(pp, :coi)
            for k in keys(pe)
                @test pp[k] == pe[k]
            end
        end
        @test length(predict(fit, nd; type = :mean)) == 3
        @test predict(fit, nd; type = :mean) == predict(explicit, nd; type = :mean)
        zoi = predict_parameters(fit, nd)[:zoi]
        @test all(z -> 0 < z < 1, zoi)
        @test allequal(zoi)
    end

    @testset "se = true returns zoi and coi" begin
        fit = drm(bf(@formula(y ~ x)), ZeroOneBeta(); data = d_obs)
        ps = predict_parameters(fit, (; x = [0.0, 0.5]); se = true)
        @test haskey(ps, :zoi) && haskey(ps, :coi)
        @test all(isfinite, ps[:zoi].se)
    end
end
