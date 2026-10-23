using DRModels
using Test, Random

@testset "Gaussian response-missing rows" begin
    Random.seed!(20260610)
    n = 120
    x = collect(range(-1, 1; length = n))
    y = 0.2 .+ 0.7 .* x .+ exp.(-0.3 .+ 0.2 .* x) .* randn(n)
    miss = [4, 17, 41, 88]
    keep = trues(n)
    keep[miss] .= false

    y_missing = Vector{Union{Missing,Float64}}(y)
    y_missing[miss] .= missing
    dat_missing = (; y = y_missing, x)
    dat_observed = (; y = y[keep], x = x[keep])

    f = bf(@formula(y ~ x), @formula(sigma ~ x))
    fit_missing = drm(f, Gaussian(); data = dat_missing)
    fit_observed = drm(f, Gaussian(); data = dat_observed)

    @test nobs(fit_missing) == count(keep)
    @test coef(fit_missing) ≈ coef(fit_observed)
    @test loglik(fit_missing) ≈ loglik(fit_observed)
    @test length(fitted(fit_missing)) == n
    @test fitted(fit_missing)[keep] ≈ fitted(fit_observed)
    @test sigma(fit_missing)[keep] ≈ sigma(fit_observed)
    @test all(isnan, residuals(fit_missing)[miss])

    bridged_missing = drm_bridge(;
        formula = Dict(:mu => "y ~ x", :sigma => "sigma ~ x"),
        family = "gaussian",
        data = dat_missing,
    )
    @test bridged_missing["coefficients"] ≈ coef(fit_observed)
    @test bridged_missing["loglik"] ≈ loglik(fit_observed)
    @test bridged_missing["nobs"] == count(keep)
    @test length(bridged_missing["fitted"]) == n
    @test all(isnan, bridged_missing["residuals"][miss])

    y_nan = copy(y)
    y_nan[miss] .= NaN
    bridged_nan = drm_bridge(;
        formula = "y ~ x; sigma ~ x",
        family = "gaussian",
        data = (; y = y_nan, x),
    )
    @test bridged_nan["coefficients"] ≈ coef(fit_observed)
    @test bridged_nan["loglik"] ≈ loglik(fit_observed)
    @test all(isnan, bridged_nan["residuals"][miss])
end

@testset "Bivariate Gaussian response-missing rows" begin
    Random.seed!(20260610)
    n = 360
    x = randn(n)
    μ1 = 0.4 .+ 0.5 .* x
    μ2 = -0.1 .+ 0.3 .* x
    σ1 = exp.(-0.2 .+ 0.1 .* x)
    σ2 = exp.(0.1 .- 0.15 .* x)
    ρ = tanh.(0.25 .+ 0.1 .* x)
    z1 = randn(n)
    z2 = randn(n)
    y1 = μ1 .+ σ1 .* z1
    y2 = μ2 .+ σ2 .* (ρ .* z1 .+ sqrt.(1 .- ρ .^ 2) .* z2)

    miss1 = [5, 23, 79, 200]
    miss2 = [12, 79, 104, 255]
    y1_missing = Vector{Union{Missing,Float64}}(y1)
    y2_missing = Vector{Union{Missing,Float64}}(y2)
    y1_missing[miss1] .= missing
    y2_missing[miss2] .= missing
    dat = (; y1 = y1_missing, y2 = y2_missing, x)

    f = bf(
        mu1 = @formula(y1 ~ x),
        mu2 = @formula(y2 ~ x),
        sigma1 = @formula(sigma1 ~ x),
        sigma2 = @formula(sigma2 ~ x),
        rho12 = @formula(rho12 ~ x),
    )
    fit = drm(f, Gaussian(); data = dat)

    observed1 = .!ismissing.(y1_missing)
    observed2 = .!ismissing.(y2_missing)
    @test nobs(fit) == count(observed1 .| observed2)
    @test isfinite(loglik(fit))
    @test length(fitted(fit)[:mu1]) == n
    @test length(fitted(fit)[:mu2]) == n
    @test all(isnan, residuals(fit)[:mu1][miss1])
    @test all(isnan, residuals(fit)[:mu2][miss2])
    @test all(isfinite, coef(fit))

    bridged = drm_bridge(;
        formula = Dict(
            :mu1 => "y1 ~ x",
            :mu2 => "y2 ~ x",
            :sigma1 => "sigma1 ~ x",
            :sigma2 => "sigma2 ~ x",
            :rho12 => "rho12 ~ x",
        ),
        family = "biv_gaussian",
        data = dat,
    )
    @test bridged["coefficients"] ≈ coef(fit)
    @test bridged["loglik"] ≈ loglik(fit)
    @test bridged["nobs"] == nobs(fit)
    @test all(isnan, bridged["residuals"]["mu1"][miss1])
    @test all(isnan, bridged["residuals"]["mu2"][miss2])
end
