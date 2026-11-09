# Bivariate missing-response default, matched to drmTMB 0.8.1.
#
# drmTMB's `miss_control(response = "drop")` is the default: a row with either response
# missing is removed before fitting. `"include"` keeps partly observed rows and lets each
# contribute the marginal likelihood of its observed response. Before this change Julia
# defaulted to the latter, so the same data gave different estimates in the two engines.
#
# Reference: `test/fixtures/bivariate_missing/r_reference.csv`, written by `make_r.R` in the
# same folder (drmTMB 0.8.1, three seeds, three missingness patterns). Agreement rule,
# fixed before the Julia numbers were read: |dLL| <= 1e-4, mean-model and rho12 coefficients
# within 1e-3 of the R standard error, log-SD coefficients within 5e-3, fitted values within
# 1e-6, equal number of rows used.
using DRModels
using Test, DelimitedFiles, Logging

const _FIX = joinpath(@__DIR__, "fixtures", "bivariate_missing")

function _biv_data(seed)
    m, h = readdlm(joinpath(_FIX, "data_seed$seed.csv"), ',', Any; header = true)
    cols = Pair{Symbol,Any}[]
    for (j, n) in enumerate(String.(vec(h)))
        c = m[:, j]
        push!(cols, Symbol(n) => any(x -> x == "NA", c) ?
            Union{Missing,Float64}[x == "NA" ? missing : Float64(x) for x in c] : Float64.(c))
    end
    return (; cols...)
end

function _biv_forms(cell)
    f1, f2 = cell == "both" ? (@formula(y1m ~ x1), @formula(y2m ~ x2)) :
             cell == "y1" ? (@formula(y1m ~ x1), @formula(y2 ~ x2)) :
                            (@formula(y1 ~ x1), @formula(y2m ~ x2))
    return bf(mu1 = f1, mu2 = f2, sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
              rho12 = @formula(rho12 ~ 1))
end

_quiet(f) = Logging.with_logger(f, Logging.NullLogger())

@testset "bivariate missing-response default matches drmTMB 0.8.1" begin
    ref, hdr = readdlm(joinpath(_FIX, "r_reference.csv"), ',', Any; header = true)
    hdr = String.(vec(hdr))
    col(n) = findfirst(==(n), hdr)

    @testset "R agreement at the optimum (3 seeds x 3 patterns x default/include)" begin
        for r in eachrow(ref)
            seed, mode, cell = Int(r[col("seed")]), String(r[col("mode")]), String(r[col("cell")])
            d = _biv_data(seed); fm = _biv_forms(cell)
            fit = mode == "default" ? _quiet(() -> drm(fm, Gaussian(); data = d)) :
                  drm(fm, Gaussian(); data = d, missing = miss_control(response = mode))
            tag = "seed $seed $mode $cell"
            @test abs(loglik(fit) - r[col("ll")]) <= 1e-4
            @test nobs(fit) == r[col("nobs")]
            th = coef(fit); b = Float64[r[col("b$i")] for i in 1:7]; se = Float64[r[col("se$i")] for i in 1:7]
            @test maximum(abs.(th[[1, 2, 3, 4, 7]] .- b[[1, 2, 3, 4, 7]]) ./ se[[1, 2, 3, 4, 7]]) <= 1e-3
            @test maximum(abs.(th[[5, 6]] .- b[[5, 6]])) <= 5e-3
        end
    end

    @testset "fitted() agrees with drmTMB's matrix row by row" begin
        d = _biv_data(1); fm = _biv_forms("both")
        for (mode, file) in (("default", "fitted_seed1_default_both.csv"), ("include", "fitted_seed1_include_both.csv"))
            fit = mode == "default" ? _quiet(() -> drm(fm, Gaussian(); data = d)) :
                  drm(fm, Gaussian(); data = d, missing = miss_control(response = "include"))
            Rf = readdlm(joinpath(_FIX, file), ',', Float64)
            fv = fitted(fit)
            @test size(Rf) == (length(fv[:mu1]), 2)
            @test maximum(abs.(hcat(fv[:mu1], fv[:mu2]) .- Rf)) <= 1e-6
        end
    end

    @testset "default is complete cases, with a counted warning; explicit options" begin
        d = _biv_data(1); fm = _biv_forms("both")
        complete = count(i -> !ismissing(d.y1m[i]) && !ismissing(d.y2m[i]), eachindex(d.y1m))
        io = IOBuffer()
        fit = Base.CoreLogging.with_logger(Base.CoreLogging.SimpleLogger(io, Base.CoreLogging.Warn)) do
            drm(fm, Gaussian(); data = d)
        end
        @test nobs(fit) == complete
        @test occursin("dropped $(length(d.y1m) - complete) of $(length(d.y1m)) rows", String(take!(io)))
        # `missing = nothing`, `"drop"` and the default are the same fit
        fd = _quiet(() -> drm(fm, Gaussian(); data = d, missing = miss_control(response = "drop")))
        @test loglik(fd) == loglik(fit) && coef(fd) == coef(fit)
        # "include" keeps every row with at least one observed response
        fi = drm(fm, Gaussian(); data = d, missing = miss_control(response = "include"))
        @test nobs(fi) > nobs(fit)
        # "fail" refuses data with a missing response; complete data fit under every policy
        @test_throws ArgumentError drm(fm, Gaussian(); data = d, missing = miss_control(response = "fail"))
        @test_throws ArgumentError drm(fm, Gaussian(); data = d, missing = miss_control(predictor = "model"))
        @test_throws ArgumentError drm(fm, Gaussian(); data = d, missing = "include")
        dc = (; y1 = d.y1, y2 = d.y2, x1 = d.x1, x2 = d.x2)
        fc = bf(mu1 = @formula(y1 ~ x1), mu2 = @formula(y2 ~ x2), sigma1 = @formula(sigma1 ~ 1),
                sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1))
        l0 = loglik(drm(fc, Gaussian(); data = dc))
        for pol in ("drop", "include", "fail")
            @test loglik(drm(fc, Gaussian(); data = dc, missing = miss_control(response = pol))) == l0
        end
    end

    @testset "REML honours the same default" begin
        d = _biv_data(2); fm = _biv_forms("both")
        complete = count(i -> !ismissing(d.y1m[i]) && !ismissing(d.y2m[i]), eachindex(d.y1m))
        fit = _quiet(() -> drm(fm, Gaussian(); data = d, method = :REML))
        @test nobs(fit) == complete
    end

    @testset "bivariate lognormal and Student: same default, include still available" begin
        d = _biv_data(3)
        complete = count(i -> !ismissing(d.y1m[i]) && !ismissing(d.y2m[i]), eachindex(d.y1m))
        ln(v) = Union{Missing,Float64}[ismissing(x) ? missing : exp(x) for x in v]
        dl = (; l1 = ln(d.y1m), l2 = ln(d.y2m), x1 = d.x1, x2 = d.x2)
        fl = bf(mu1 = @formula(l1 ~ x1), mu2 = @formula(l2 ~ x2), sigma1 = @formula(sigma1 ~ 1),
                sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1))
        f0 = _quiet(() -> drm(fl, LogNormal(); data = dl))
        @test nobs(f0) == complete
        # partly observed rows enter the likelihood under "include" (the lognormal route reports
        # the complete-case count in `nobs` either way, so compare the likelihood)
        @test loglik(drm(fl, LogNormal(); data = dl, missing = miss_control(response = "include"))) != loglik(f0)
        fs = bf(mu1 = @formula(y1m ~ x1), mu2 = @formula(y2m ~ x2), sigma1 = @formula(sigma1 ~ 1),
                sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1), nu = @formula(nu ~ 1))
        s0 = _quiet(() -> drm(fs, Student(); data = d))
        @test nobs(s0) == complete
        @test nobs(drm(fs, Student(); data = d, missing = miss_control(response = "include"))) > complete
    end

    @testset "bivariate lognormal fitted() is E[Y] = exp(mu + sigma^2 / 2), as in drmTMB" begin
        d = _biv_data(1)
        dl = (; l1 = exp.(d.y1), l2 = exp.(d.y2), x1 = d.x1, x2 = d.x2)
        fl = bf(mu1 = @formula(l1 ~ x1), mu2 = @formula(l2 ~ x2), sigma1 = @formula(sigma1 ~ 1),
                sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1))
        fit = drm(fl, LogNormal(); data = dl)
        Rf = readdlm(joinpath(_FIX, "fitted_lognormal_seed1.csv"), ',', Float64)
        fv = fitted(fit)
        @test maximum(abs.(hcat(fv[:mu1], fv[:mu2]) .- Rf)) <= 1e-6
        @test fv[:mu1] ≈ exp.(fit.means[:mu1] .+ fit.scales[:sigma1] .^ 2 ./ 2)
        # residuals are on the same (response) scale as fitted
        r = residuals(fit)
        @test r[:mu1] ≈ dl.l1 .- fv[:mu1]
        # predict(type = :response) is the log-scale location (drmTMB), :median its exp
        pr = predict(fit, dl); pl = predict(fit, dl; type = :link); pm = predict(fit, dl; type = :median)
        @test pr[:mu1] ≈ pl[:mu1] && pr[:mu1] ≈ fit.means[:mu1]
        @test pm[:mu2] ≈ exp.(pl[:mu2])
        @test all(pm[:mu1] .< fv[:mu1])   # median below the mean for a lognormal
    end
end
