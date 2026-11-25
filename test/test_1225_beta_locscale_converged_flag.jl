# #1225 / #1050: the fixed-effect Beta location-scale route must not report
# `converged = true` when the fit stopped far from stationary. Data: agridat
# wedderburn.barley, frozen in test/fixtures/beta_locscale_1225. The optimum
# quality itself is pinned separately in test_1225_beta_locscale_barley_broken.jl.
using DRModels
using Test
using DelimitedFiles
using Logging

function _barley_1225_flag()
    m, h = readdlm(joinpath(@__DIR__, "fixtures", "beta_locscale_1225", "wedderburn_barley.csv"),
                   ',', Any; header = true)
    nm = String.(vec(h))
    col(n) = m[:, findfirst(==(n), nm)]
    y = Float64.(col("y")); n = length(y)
    return (; y = (y ./ 100 .* (n - 1) .+ 0.5) ./ n,
            site = String.(col("site")), gen = String.(col("gen")))
end

@testset "#1225 Beta location-scale converged flag (unit-free)" begin
    d = _barley_1225_flag()
    quiet(f) = with_logger(f, NullLogger())

    # sigma ~ site: stalls with max|grad| ~ 5e4, so it must not be called converged.
    fm = bf(@formula(y ~ site + gen), @formula(sigma ~ site))
    fit = quiet(() -> drm(fm, Beta(); data = d))
    @test maximum(abs, gradient(fit)) > 1.0
    @test !fit.converged

    fit5 = quiet(() -> drm(fm, Beta(); data = d, control = DRMControl(multi_start = 5)))
    @test maximum(abs, gradient(fit5)) > 1.0
    @test !fit5.converged

    # Control: sigma ~ 1 is a clean stationary fit and stays converged.
    fm0 = bf(@formula(y ~ site + gen), @formula(sigma ~ 1))
    fit0 = quiet(() -> drm(fm0, Beta(); data = d))
    @test maximum(abs, gradient(fit0)) < 1e-4
    @test fit0.converged
end
