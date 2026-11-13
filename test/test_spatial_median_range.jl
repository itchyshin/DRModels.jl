# spatial(1 | g) with coordinates fixes the range at the median positive pairwise distance
# (kernel exp(-d / median) + 1e-6 I), as drmTMB does, on the Gaussian and Binomial routes.
# Reference values: drmTMB 0.8.1, fixtures/spatial_median_range/make_reference.R (data, coordinates
# and the reference table are committed next to it). Before this change the Gaussian route
# estimated the range and the Binomial route used the mean pairwise distance.
using DRModels
using Test, DelimitedFiles, LinearAlgebra, Statistics

const _SMR_DIR = joinpath(@__DIR__, "fixtures", "spatial_median_range")

function _smr_load()
    m, h = readdlm(joinpath(_SMR_DIR, "data.csv"), ','; header = true)
    col(name) = m[:, findfirst(==(name), vec(h))]
    data = (; y = Float64.(col("y")), s = Int.(col("s")), fl = Int.(col("fl")),
            x = Float64.(col("x")), g = String.(col("g")))
    cm = readdlm(joinpath(_SMR_DIR, "coords.csv"), ','; skipstart = 1)
    coords = Dict(String(cm[i, 1]) => [Float64(cm[i, 2]), Float64(cm[i, 3])] for i in 1:size(cm, 1))
    rm = readdlm(joinpath(_SMR_DIR, "reference.csv"), ','; skipstart = 1)
    ref = Dict{String,Dict{String,Float64}}()
    for i in 1:size(rm, 1)
        d = get!(ref, String(rm[i, 1]), Dict{String,Float64}())
        d[String(rm[i, 2])] = Float64(rm[i, 3]); d["loglik"] = Float64(rm[i, 4])
    end
    return data, coords, ref
end

@testset "spatial(1 | g): range fixed at the median distance, pinned to drmTMB 0.8.1" begin
    data, coords, ref = _smr_load()
    @testset "Gaussian ML" begin
        r = ref["gaussian_ML"]
        fit = drm(bf(@formula(y ~ x + spatial(1 | g)), @formula(sigma ~ 1)), Gaussian(); data, coords)
        @test fit.converged
        @test !any(b -> first(b) === :range, fit.blocks)       # no estimated range
        @test loglikelihood(fit) ≈ r["loglik"] atol = 1e-4
        @test coef(fit, :mu) ≈ [r["mu:(Intercept)"], r["mu:x"]] atol = 1e-4
        @test coef(fit, :sigma)[1] ≈ r["sigma:(Intercept)"] atol = 1e-4
        @test re_sd(fit)[:g] ≈ r["sd:mu:spatial(1 | g)"] atol = 1e-4
    end
    @testset "Gaussian REML" begin
        r = ref["gaussian_REML"]
        fit = drm(bf(@formula(y ~ x + spatial(1 | g)), @formula(sigma ~ 1)), Gaussian(); data, coords,
                  method = :REML)
        @test fit.converged
        @test loglikelihood(fit) ≈ r["loglik"] atol = 1e-4
        @test coef(fit, :mu) ≈ [r["mu:(Intercept)"], r["mu:x"]] atol = 1e-4
        @test re_sd(fit)[:g] ≈ r["sd:mu:spatial(1 | g)"] atol = 1e-4
    end
    @testset "Binomial ML" begin
        r = ref["binomial_ML"]
        fit = drm(bf(@formula(cbind(s, fl) ~ x + spatial(1 | g))), Binomial(); data, coords)
        @test fit.converged
        @test loglikelihood(fit) ≈ r["loglik"] atol = 1e-4
        @test coef(fit, :mu) ≈ [r["mu:(Intercept)"], r["mu:x"]] atol = 1e-4
        @test exp(only(coef(fit, :sd))) ≈ r["sd:mu:spatial(1 | g)"] atol = 1e-4
    end
end

@testset "coef(fit, :range) says the range is fixed" begin
    data, coords, _ = _smr_load()
    fit = drm(bf(@formula(y ~ x + spatial(1 | g)), @formula(sigma ~ 1)), Gaussian(); data, coords)
    err = try coef(fit, :range); nothing catch e e end
    @test err isa ArgumentError
    @test occursin("fixed at the median positive pairwise distance", sprint(showerror, err))
    # A fit with no spatial term keeps the generic message.
    plain = drm(bf(@formula(y ~ x)), Gaussian(); data)
    err2 = try coef(plain, :range); nothing catch e e end
    @test err2 isa ArgumentError
    @test !occursin("fixed at the median", sprint(showerror, err2))
end
