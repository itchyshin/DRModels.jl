# Refs #1297: generic `gaussian_core.jl` `_simulate_once(fit::DrmFit, …)` must not
# carry shadowed branches for families with typed methods.
using DRModels
using Test, Random

function _method_file(f)
    m = Base.which(DRModels._simulate_once, Tuple{typeof(f), Random.AbstractRNG})
    m === nothing && return ""
    return String(m.file)
end

@testset "typed _simulate_once shadow generic branches (#1297)" begin
    rng = MersenneTwister(1297)
    n = 12; x = randn(rng, n); g = repeat(1:4, inner = 3)
    y_ln = exp.(0.2 .+ 0.1 .* x .+ 0.3 .* randn(rng, n))
    y_st = 0.1 .+ 0.4 .* x .+ 0.5 .* randn(rng, n)
    ord = repeat(1:4, inner = 3)

    fit_ln = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), LogNormal(); data = (y = y_ln, x = x))
    fit_st = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Student(); data = (y = y_st, x = x))
    fit_cl = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), CumulativeLogit();
        data = (y = ord, x = x))

    @test occursin("bivariate_lognormal", _method_file(fit_ln))
    @test occursin("bivariate_student", _method_file(fit_st))
    @test occursin("cumulative", _method_file(fit_cl))

    @test DRModels._simulate_once(fit_ln, MersenneTwister(41)) isa AbstractVector
    @test DRModels._simulate_once(fit_st, MersenneTwister(42)) isa AbstractVector
    @test DRModels._simulate_once(fit_cl, MersenneTwister(43)) isa AbstractVector

    src = read(joinpath(@__DIR__, "..", "src", "gaussian_core.jl"), String)
    generic = match(r"function _simulate_once\(fit::DrmFit", src)
    @test generic !== nothing
    body = generic === nothing ? "" : src[generic.offset:end]
    @test !occursin("elseif fam isa LogNormal", body)
    @test !occursin("elseif fam isa Student", body)
    @test !occursin("elseif fam isa CumulativeLogit", body)
end