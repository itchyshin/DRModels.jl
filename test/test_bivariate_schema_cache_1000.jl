# #1000: bivariate predict used to rebuild factor contrasts from newdata.
# A level subset then either threw a raw StatsModels error or (same column
# count, different reference) predicted the wrong value. The training schema
# is now cached on BivariateDrmFormula, so a subset matches the full-data
# prediction on the shared rows.
using DRModels
using Test, Random

@testset "bivariate predict reuses the training factor schema (#1000)" begin
    Random.seed!(1000)
    n = 120
    f = repeat(["a", "b", "c"], inner = n ÷ 3)
    x = randn(n)
    y1 = 1.0 .+ 0.4 .* x .+ 0.8 .* (f .== "b") .- 0.5 .* (f .== "c") .+ 0.35 .* randn(n)
    y2 = -0.4 .+ 0.3 .* x .+ 0.6 .* (f .== "c") .+ 0.35 .* randn(n)
    dat = (; y1, y2, x, f)
    fit = drm(bf(mu1 = @formula(y1 ~ x + f), mu2 = @formula(y2 ~ x + f),
                 sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                 rho12 = @formula(rho12 ~ 1)), Gaussian(); data = dat)
    @test fit.formula isa DRModels.BivariateDrmFormula
    @test haskey(fit.formula.schema_cache[], :mu1)
    @test haskey(fit.formula.schema_cache[], :mu2)

    pall = predict(fit, dat)
    keep = f .== "b"
    nd = (; x = x[keep], f = f[keep])
    psub = predict(fit, nd)
    @test psub[:mu1] ≈ pall[:mu1][keep]
    @test psub[:mu2] ≈ pall[:mu2][keep]

    pp_all = predict_parameters(fit, dat)
    pp_sub = predict_parameters(fit, nd)
    @test pp_sub[:mu1] ≈ pp_all[:mu1][keep]
    @test pp_sub[:sigma1] ≈ pp_all[:sigma1][keep]

    # An unseen level is still refused by name, not silently recoded.
    err = try
        predict(fit, (; x = [0.0], f = ["d"]))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("mu1", err.msg) || occursin("level", lowercase(err.msg))
end

@testset "bivariate Student-t predict reuses the training schema (#1000)" begin
    Random.seed!(1001)
    n = 90
    f = repeat(["a", "b", "c"], inner = n ÷ 3)
    x = randn(n)
    y1 = 0.4 .* x .+ 0.6 .* (f .== "b") .+ 0.4 .* randn(n)
    y2 = 0.3 .* x .+ 0.5 .* (f .== "c") .+ 0.4 .* randn(n)
    dat = (; y1, y2, x, f)
    fit = drm(bf(mu1 = @formula(y1 ~ x + f), mu2 = @formula(y2 ~ x + f),
                 sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                 nu = @formula(nu ~ 1), rho12 = @formula(rho12 ~ 1)),
              Student(); data = dat)
    @test haskey(fit.formula.schema_cache[], :mu1)
    @test haskey(fit.formula.schema_cache[], :nu)
    keep = f .== "b"
    nd = (; x = x[keep], f = f[keep])
    psub = predict(fit, nd)
    pall = predict(fit, dat)
    @test psub[:mu1] ≈ pall[:mu1][keep]
    @test psub[:mu2] ≈ pall[:mu2][keep]
end
