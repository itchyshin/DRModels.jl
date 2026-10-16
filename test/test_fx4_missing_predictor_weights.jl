# FX4 (a)(b): missing predictor KeyError (#1089 / #1111) and weights MethodError (#931).
#
# Red-proof: on the pin these throw KeyError / MethodError. After the fix they
# error clearly (predictor `missing`; RE `weights`) or fit (complete Union{Missing}
# predictor; FE integer weights ≡ row duplication).
using DRModels
using Test
using Random
using LinearAlgebra

@testset "FX4 missing predictor (#1089, #1111) and weights (#931)" begin

    @testset "#1089 missing in a predictor is a named error, not KeyError" begin
        Random.seed!(20261016)
        n = 40
        x = Vector{Union{Missing,Float64}}(randn(n))
        x[3] = missing
        z = randn(n)
        y = 1.0 .+ 0.5 .* replace(x, missing => 0.0) .+ 0.3 .* randn(n)
        g = repeat(1:8, inner = 5)
        f = bf(@formula(y ~ 1 + x + z + (1 | g)), @formula(sigma ~ 1))
        err = try
            drm(f, Gaussian(); data = (; y, x, z, g))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin("missing", msg)
        @test occursin("drm_listwise", msg)
        @test !isa(err, KeyError)
        # The documented complete-case path still matches drmTMB's nobs drop.
        clean = drm_listwise(f, (; y, x, z, g); verbose = false)
        fit = drm(f, Gaussian(); data = clean)
        @test nobs(fit) == n - 1
        @test isfinite(loglik(fit))
    end

    @testset "#1111 Union{Missing,Float64} predictor with no missings stays numeric" begin
        Random.seed!(20261017)
        n = 60
        x = Vector{Union{Missing,Float64}}(randn(n))
        y = 0.4 .+ 0.7 .* x .+ 0.25 .* randn(n)
        fit = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian();
                  data = (; y, x))
        @test length(coef(fit, :mu)) == 2
        @test isfinite(loglik(fit))
        @test is_converged(fit)
        # drm_listwise no-drop must also narrow (#1111); it used to return the
        # Union column unchanged.
        f = bf(@formula(y ~ 1 + x), @formula(sigma ~ 1))
        clean = drm_listwise(f, (; y, x); verbose = false)
        @test eltype(clean.x) === Float64
        @test length(coef(drm(f, Gaussian(); data = clean), :mu)) == 2
        # Vector{Any} is the bridge's R `factor()` encoding — must stay
        # categorical, not be identity.'d into a slope. Four numeric levels
        # dummy-code as intercept + 3 dummies; a continuous column is 2.
        h = Any[Float64(mod1(i, 4)) for i in 1:n]
        fit_fac = drm(bf(@formula(y ~ 1 + h), @formula(sigma ~ 1)), Gaussian();
                      data = (; y, h))
        @test length(coef(fit_fac, :mu)) == 4
    end

    @testset "#931 weights= is a named keyword, not a MethodError" begin
        Random.seed!(20261018)
        n = 80
        g = repeat(1:10, inner = 8)
        x = randn(n)
        y = 0.2 .+ 0.4 .* x .+ 0.4 .* randn(n)
        w = fill(1.0, n)
        f_re = bf(@formula(y ~ 1 + x + (1 | g)), @formula(sigma ~ 1))
        err = try
            drm(f_re, Gaussian(); data = (; y, x, g), weights = w)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("931", sprint(showerror, err))
        @test !isa(err, MethodError)

        # Poisson / Gamma (1 | g) — the sweep t5 MethodError cells.
        yp = rand(0:4, n)
        yg = abs.(y) .+ 0.2
        @test_throws ArgumentError drm(bf(@formula(y ~ 1 + x + (1 | g))), Poisson();
                                       data = (; y = yp, x, g), weights = w)
        @test_throws ArgumentError drm(bf(@formula(y ~ 1 + x + (1 | g)), @formula(sigma ~ 1)),
                                       Gamma(); data = (; y = yg, x, g), weights = w)
        @test_throws ArgumentError drm(bf(@formula(y ~ 1 + x)), Binomial();
                                       data = (; y = rand(0:1, n), x), weights = w)
        err_len = try
            drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian();
                data = (; y, x), weights = w[1:end-1])
            nothing
        catch e
            e
        end
        @test err_len isa ArgumentError
        @test occursin("length", sprint(showerror, err_len))
    end

    @testset "#931 FE integer weights match row duplication" begin
        Random.seed!(20261019)
        n = 30
        x = randn(n)
        y = 0.3 .+ 0.5 .* x .+ 0.35 .* randn(n)
        w = [1, 2, 3][mod1.(1:n, 3)]
        dat = (; y, x)
        f = bf(@formula(y ~ 1 + x), @formula(sigma ~ 1))
        fit_w = drm(f, Gaussian(); data = dat, weights = w)
        ydup = vcat((fill(y[i], w[i]) for i in 1:n)...)
        xdup = vcat((fill(x[i], w[i]) for i in 1:n)...)
        fit_d = drm(f, Gaussian(); data = (; y = ydup, x = xdup))
        @test loglik(fit_w) ≈ loglik(fit_d) atol = 1e-8
        @test coef(fit_w) ≈ coef(fit_d) atol = 1e-8
        @test weights(fit_w) == Float64.(w)
        @test weights(fit_d) === nothing

        # Poisson FE: same identity on the log-likelihood.
        yp = rand(0:5, n)
        fp = bf(@formula(y ~ 1 + x))
        fitp_w = drm(fp, Poisson(); data = (; y = yp, x), weights = w)
        ypdup = vcat((fill(yp[i], w[i]) for i in 1:n)...)
        xpdup = vcat((fill(x[i], w[i]) for i in 1:n)...)
        fitp_d = drm(fp, Poisson(); data = (; y = ypdup, x = xpdup))
        @test loglik(fitp_w) ≈ loglik(fitp_d) atol = 1e-8
        @test coef(fitp_w) ≈ coef(fitp_d) atol = 1e-6

        yg = abs.(y) .+ 0.2
        fg = bf(@formula(y ~ 1 + x), @formula(sigma ~ 1))
        fitg_w = drm(fg, Gamma(); data = (; y = yg, x), weights = w)
        ygdup = vcat((fill(yg[i], w[i]) for i in 1:n)...)
        xgdup = vcat((fill(x[i], w[i]) for i in 1:n)...)
        fitg_d = drm(fg, Gamma(); data = (; y = ygdup, x = xgdup))
        @test loglik(fitg_w) ≈ loglik(fitg_d) atol = 1e-6
        @test coef(fitg_w) ≈ coef(fitg_d) atol = 1e-5

        yb = 1 ./ (1 .+ exp.(-y))
        clamp!(yb, 1e-4, 1 - 1e-4)
        fb = bf(@formula(y ~ 1 + x), @formula(sigma ~ 1))
        fitb_w = drm(fb, Beta(); data = (; y = yb, x), weights = w)
        ybdup = vcat((fill(yb[i], w[i]) for i in 1:n)...)
        xbdup = vcat((fill(x[i], w[i]) for i in 1:n)...)
        fitb_d = drm(fb, Beta(); data = (; y = ybdup, x = xbdup))
        @test loglik(fitb_w) ≈ loglik(fitb_d) atol = 1e-6
        @test coef(fitb_w) ≈ coef(fitb_d) atol = 1e-5
    end

    @testset "#931 unweighted fits report nothing, not a fake ones() stub" begin
        Random.seed!(20261020)
        n = 40
        x = randn(n)
        y = randn(n)
        fit = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
        @test weights(fit) === nothing
    end
end
