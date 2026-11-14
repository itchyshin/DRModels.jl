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

@testset "the fixed range is stored, printed, and is not a parameter" begin
    data, coords, _ = _smr_load()
    # hand computation: median of the positive pairwise distances between the sites
    sites = sort(collect(keys(coords)))
    xy = reduce(vcat, [permutedims(coords[k]) for k in sites])
    pos = [norm(xy[a, :] - xy[b, :]) for a in 1:20 for b in (a + 1):20
           if norm(xy[a, :] - xy[b, :]) > 0]
    m = median(pos)
    fits = Dict(
        "Gaussian ML" => drm(bf(@formula(y ~ x + spatial(1 | g)), @formula(sigma ~ 1)), Gaussian();
                             data, coords),
        "Gaussian REML" => drm(bf(@formula(y ~ x + spatial(1 | g)), @formula(sigma ~ 1)), Gaussian();
                               data, coords, method = :REML),
        "Binomial" => drm(bf(@formula(cbind(s, fl) ~ x + spatial(1 | g))), Binomial(); data, coords),
        "Poisson" => drm(bf(@formula(s ~ x + spatial(1 | g))), Poisson(); data, coords))
    @testset "$lab" for (lab, fit) in fits
        @test coef(fit, :range) == [m]                      # stored, distance scale
        txt = sprint(show, MIME("text/plain"), fit)
        @test occursin("Spatial range = $(round(m; sigdigits = 4))", txt)
        @test occursin("fixed at the median distance between sites", txt)
        @test occursin("not estimated", txt)
        # a fixed quantity is not a parameter
        @test !any(b -> first(b) === :range, fit.blocks)
        @test size(vcov(fit)) == (length(fit.theta), length(fit.theta))
        @test length(stderror(fit)) == length(fit.theta)
        @test length(coef(fit)) == length(fit.theta)
        @test aic(fit) ≈ -2 * loglikelihood(fit) + 2 * length(fit.theta)
        @test dof(fit) == length(fit.theta)
    end
    # The Gaussian ML fit had five parameters when the range was estimated: now four.
    @test length(fits["Gaussian ML"].theta) == 4
    # No spatial term: no stored range, and the generic message.
    plain = drm(bf(@formula(y ~ x)), Gaussian(); data)
    @test !haskey(plain.scales, :spatial_range)
    @test !occursin("Spatial range", sprint(show, MIME("text/plain"), plain))
    err = try coef(plain, :range); nothing catch e e end
    @test err isa ArgumentError
    @test !occursin("fixed at the median", sprint(showerror, err))
end
