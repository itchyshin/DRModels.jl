# test_predict_se.jl — delta-method standard errors on the prediction surface
# (glmmTMB/drmTMB `se.fit` parity). The correctness anchor is an independent
# recomputation of the link-scale SE from vcov(fit) + the rebuilt μ design,
# plus back-compatibility of the default (se = false) path. No R needed.
using DRModels
using Test, Random, LinearAlgebra

@testset "predict / predict_parameters delta-method SE (Gaussian loc-scale)" begin
    Random.seed!(20260604)
    n = 500
    x = randn(n)
    y = 0.5 .- 0.8 .* x .+ exp.(-0.3 .+ 0.4 .* x) .* randn(n)
    data = (; y, x)

    fit = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x)), Gaussian(); data = data)

    @testset "back-compat: se = false is unchanged" begin
        @test predict(fit, data) == predict(fit, data; se = false)
        @test predict(fit, data; type = :link) == predict(fit, data; type = :link, se = false)
        @test predict_parameters(fit, data) == predict_parameters(fit, data; se = false)
        @test predict_parameters(fit, data; type = :link) ==
              predict_parameters(fit, data; type = :link, se = false)
    end

    # Rebuild the μ design exactly as `predict` does, and the μ-block vcov.
    f = fit.formula
    fixed_mu, _, _, _ = DRModels._split_ranef(Dict(f.forms)[:mu])
    ndr = merge(NamedTuple(pairs(data)), NamedTuple{(f.response,)}((zeros(n),)))
    _, Xμ, _ = DRModels._design(f.response, fixed_mu, ndr)
    rμ = DRModels._block_range(fit, :mu)
    Vμ = vcov(fit)[rμ, rμ]

    @testset "link-scale SE matches independent recomputation" begin
        r = predict(fit, data; type = :link, se = true)
        @test r isa NamedTuple
        @test Set(keys(r)) == Set([:prediction, :se])
        @test r.prediction == predict(fit, data; type = :link)   # point unchanged

        se_ref = [sqrt(dot(view(Xμ, i, :), Vμ, view(Xμ, i, :))) for i in 1:n]
        @test r.se ≈ se_ref
        @test all(r.se .> 0)
    end

    @testset "Gaussian identity link: response SE == link SE" begin
        r_link = predict(fit, data; type = :link, se = true)
        r_resp = predict(fit, data; type = :response, se = true)
        @test r_resp.se ≈ r_link.se               # dμ/dη = 1 for identity link
        @test r_resp.prediction == predict(fit, data; type = :response)
        @test all(r_resp.se .> 0)
    end

    @testset "predict_parameters: per-parameter (; value, se)" begin
        pp = predict_parameters(fit, data; se = true)
        @test Set(keys(pp)) == Set([:mu, :sigma])
        for p in (:mu, :sigma)
            @test pp[p] isa NamedTuple
            @test Set(keys(pp[p])) == Set([:value, :se])
            @test all(pp[p].se .> 0)
        end
        # value matches the se = false point prediction.
        ref = predict_parameters(fit, data)
        @test pp[:mu].value    ≈ ref[:mu]
        @test pp[:sigma].value ≈ ref[:sigma]

        # mu SE on the response scale equals the independently recomputed link SE
        # (identity link → derivative 1).
        se_ref = [sqrt(dot(view(Xμ, i, :), Vμ, view(Xμ, i, :))) for i in 1:n]
        @test pp[:mu].se ≈ se_ref
    end
end
