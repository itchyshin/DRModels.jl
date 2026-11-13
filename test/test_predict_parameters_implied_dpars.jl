# predict_parameters() returns every estimated distributional parameter, also
# the ones with an implied intercept-only predictor (`nu` for Student,
# SkewNormal and Tweedie; `zoi` and `coi` for ZeroOneBeta). The values must equal
# those of the same model with the formula written out (`nu ~ 1`).
using DRModels
using Test, Random

@testset "predict_parameters includes implied dpars (nu, zoi, coi)" begin
    rng = Random.Xoshiro(20261112)
    n = 120
    x = [sin(0.37i) + 0.3cos(1.7i) for i in 1:n]
    nd = (; x = [-1.0, 0.0, 1.0])
    ystud = [0.5 + 0.6x[i] + 1.2tan(0.45sin(13i)) + 0.3cos(5i) for i in 1:n]
    yskew = 0.5 .+ 0.6 .* x .+ 0.7 .* abs.(randn(rng, n)) .+ 0.3 .* randn(rng, n)
    ytw = [rand(rng) < 0.25 ? 0.0 : exp(0.3 + 0.3xi) * (0.3 + rand(rng)) for xi in x]
    ybeta = clamp.(1 ./ (1 .+ exp.(-(0.2 .+ 0.5 .* x .+ 0.4 .* randn(rng, n)))), 0.02, 0.98)
    yzob = [r < 0.1 ? 0.0 : r > 0.9 ? 1.0 : ybeta[i] for (i, r) in enumerate(rand(rng, n))]
    d = (; x, ystud, yskew, ytw, yzob)

    function same_as_explicit(fit_implied, fit_explicit, dpar)
        for type in (:link, :response)
            pi_ = predict_parameters(fit_implied, nd; type)
            pe = predict_parameters(fit_explicit, nd; type)
            @test dpar in keys(pi_)
            @test Set(keys(pi_)) == Set(keys(pe))
            @test pi_[dpar] == pe[dpar]
            @test length(pi_[dpar]) == length(nd.x)
            @test all(==(first(pi_[dpar])), pi_[dpar])      # constant across rows
        end
        link = predict_parameters(fit_implied, nd; type = :link)[dpar]
        @test link[1] == only(coef(fit_implied, dpar))
    end

    @testset "Student nu" begin
        f1 = drm(bf(@formula(ystud ~ x), @formula(sigma ~ 1)), Student(); data = d)
        f2 = drm(bf(@formula(ystud ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)), Student(); data = d)
        same_as_explicit(f1, f2, :nu)
        @test predict_parameters(f1, nd)[:nu][1] ≈ 2 + exp(only(coef(f1, :nu)))
    end

    @testset "SkewNormal nu" begin
        f1 = drm(bf(@formula(yskew ~ x), @formula(sigma ~ 1)), SkewNormal(); data = d)
        f2 = drm(bf(@formula(yskew ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)), SkewNormal(); data = d)
        same_as_explicit(f1, f2, :nu)
    end

    @testset "Tweedie nu" begin
        f1 = drm(bf(@formula(ytw ~ x), @formula(sigma ~ 1)), Tweedie(); data = d)
        f2 = drm(bf(@formula(ytw ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)), Tweedie(); data = d)
        same_as_explicit(f1, f2, :nu)
    end

    @testset "ZeroOneBeta zoi and coi" begin
        f1 = drm(bf(@formula(yzob ~ x)), ZeroOneBeta(); data = d)
        f2 = drm(bf(@formula(yzob ~ x), @formula(zoi ~ 1), @formula(coi ~ 1)), ZeroOneBeta(); data = d)
        same_as_explicit(f1, f2, :zoi)
        same_as_explicit(f1, f2, :coi)
    end

    @testset "se = true carries the implied dpar too" begin
        f1 = drm(bf(@formula(ystud ~ x), @formula(sigma ~ 1)), Student(); data = d)
        p = predict_parameters(f1, nd; se = true)
        @test haskey(p, :nu)
        @test all(isfinite, p[:nu].se)
    end

    @testset "families without extra parameters are unchanged" begin
        yg = 0.5 .+ 0.6 .* x .+ 0.7 .* randn(rng, n)
        f = drm(bf(@formula(yg ~ x)), Gaussian(); data = (; yg, x))
        @test Set(keys(predict_parameters(f, nd))) == Set([:mu, :sigma])
    end
end
