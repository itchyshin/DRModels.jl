# P53c: univariate Gaussian `missing = miss_control(response = "include")`.
# Design shared with the bivariate include (train 4, #1409): rows with a missing response keep a
# NaN in `fit.obs[:mu]` (read by `_used_row_mask`), contribute no likelihood term, and every
# per-row output returns the rows used (`nobs(fit)`). The likelihood of the kept rows is the
# likelihood of the observed rows, so each fit must equal the fit on the pre-dropped data exactly.
using DRModels
using Test, Random, Logging

function _uni_data(seed; n = 80, miss = (3, 11, 12, 40, 77))
    rng = MersenneTwister(seed)
    x = randn(rng, n); g = repeat(1:8, inner = n ÷ 8)
    u = randn(rng, 8)
    y = Vector{Union{Missing,Float64}}(1 .+ 0.5 .* x .+ 0.7 .* u[g] .+ exp.(0.1 .+ 0.2 .* x) .* randn(rng, n))
    y[collect(miss)] .= missing
    return (y = y, x = x, g = g)
end
_complete(d) = (k = findall(!ismissing, d.y); (y = Float64.(d.y[k]), x = d.x[k], g = d.g[k]))
const _INC = miss_control(response = "include")
const _quiet = Logging.NullLogger()

@testset "univariate Gaussian response include (P53c)" begin
    d = _uni_data(1); dc = _complete(d); nmiss = count(ismissing, d.y)
    f = bf(@formula(y ~ x), @formula(sigma ~ x))

    @testset "fixed-effect location-scale: same fit as the complete cases, silent, used rows" begin
        fi = nothing; @test_logs min_level = Logging.Warn (fi = drm(f, Gaussian(); data = d, missing = _INC))
        fc = drm(f, Gaussian(); data = dc)
        @test nobs(fi) == length(dc.y) == nobs(fc)
        @test loglik(fi) ≈ loglik(fc) atol = 1e-9
        @test coef(fi, :mu) ≈ coef(fc, :mu) atol = 1e-7
        @test coef(fi, :sigma) ≈ coef(fc, :sigma) atol = 1e-7
        @test length(fitted(fi)) == length(residuals(fi)) == length(sigma(fi)) == nobs(fi)
        @test fitted(fi) ≈ fitted(fc) atol = 1e-7
        @test length(fi.obs[:mu]) == length(d.y) && count(isnan, fi.obs[:mu]) == nmiss   # the marker
        # the default policy gives the same fit but says so
        fdef = nothing; @test_logs (:warn, r"dropped") (fdef = drm(f, Gaussian(); data = d))
        @test loglik(fdef) ≈ loglik(fi) atol = 1e-12
        # `"drop"` is the default
        fdrop = nothing; @test_logs (:warn, r"dropped") (fdrop = drm(f, Gaussian(); data = d, missing = miss_control(response = "drop")))
        @test loglik(fdrop) ≈ loglik(fi) atol = 1e-12
    end

    @testset "no missing response: include changes nothing" begin
        a = drm(f, Gaussian(); data = dc, missing = _INC); b = drm(f, Gaussian(); data = dc)
        @test a.theta == b.theta && loglik(a) == loglik(b) && nobs(a) == nobs(b)
    end

    @testset "weights follow their rows" begin
        w = 0.5 .+ rand(MersenneTwister(5), length(d.y)); k = findall(!ismissing, d.y)
        fi = drm(f, Gaussian(); data = d, weights = w, missing = _INC)
        fc = drm(f, Gaussian(); data = dc, weights = w[k])
        @test loglik(fi) ≈ loglik(fc) atol = 1e-9
        @test coef(fi, :mu) ≈ coef(fc, :mu) atol = 1e-7
    end

    @testset "REML on the observed rows" begin
        fi = drm(f, Gaussian(); data = d, method = :ML, missing = _INC)   # ML reference
        r1 = drm(bf(@formula(y ~ x)), Gaussian(); data = d, method = :REML, missing = _INC)
        r2 = drm(bf(@formula(y ~ x)), Gaussian(); data = dc, method = :REML)
        @test loglik(r1) ≈ loglik(r2) atol = 1e-9
        @test coef(r1, :mu) ≈ coef(r2, :mu) atol = 1e-7
    end

    @testset "mean random intercept (1 | g)" begin
        fr = bf(@formula(y ~ x + (1 | g)))
        fi = drm(fr, Gaussian(); data = d, missing = _INC)
        fc = drm(fr, Gaussian(); data = dc)
        @test loglik(fi) ≈ loglik(fc) atol = 1e-8
        @test coef(fi, :mu) ≈ coef(fc, :mu) atol = 1e-6
        @test length(fitted(fi)) == nobs(fi)
        fq = nothing; @test_logs min_level = Logging.Warn (fq = drm(fr, Gaussian(); data = d, missing = _INC))
        @test length(residuals(fq; type = :quantile)) == nobs(fq)
    end

    @testset "fail refuses; predictor = \"model\" needs mi(x)" begin
        err = try drm(f, Gaussian(); data = d, missing = miss_control(response = "fail")); nothing catch e; e end
        @test err isa ArgumentError && occursin("miss_control(response = \"fail\")", err.msg) &&
              occursin("$nmiss of $(length(d.y))", err.msg)
        ok = drm(f, Gaussian(); data = dc, missing = miss_control(response = "fail"))   # complete data passes
        @test nobs(ok) == length(dc.y)
        err2 = try drm(f, Gaussian(); data = d, missing = miss_control(response = "include", predictor = "model")); nothing catch e; e end
        @test err2 isa ArgumentError && occursin("mi(x)", err2.msg)
    end

    @testset "a predictor missing on a row is still dropped, with its warning" begin
        x2 = Vector{Union{Missing,Float64}}(d.x); x2[5] = missing
        d2 = (y = d.y, x = x2, g = d.g)
        fi = nothing; @test_logs (:warn, r"predictor|missing") match_mode = :any (fi = drm(f, Gaussian(); data = d2, missing = _INC))
        @test nobs(fi) == length(dc.y) - 1
    end
end
