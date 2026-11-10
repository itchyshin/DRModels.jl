# NB2 `spatial(1 | g)` with site `coords` (no SD formula): the coordinate route uses drmTMB's
# fixed-range kernel, exp(-d / r) + 1e-6 I with r the median positive pairwise distance, like the
# Poisson and Gamma routes. Before this slice Julia refused the cell (gap map cell
# `nb2|spatial|lab0|q1|sd-none|cm0|ML`, R-only).
# Reference: drmTMB 0.8.1, ML (test/fixtures/nb2_spatial_intercept/reference_drmtmb.csv; script
# ref_nb2_spatial_intercept.R, same folder). Rule fixed before the Julia run:
#   |logLik diff| <= 1e-4; each fixed effect within 1e-3 R standard errors; log SD within 5e-3;
#   R reports is_converged() = TRUE (otherwise the cell is flagged, not counted).
using DRModels, Test, DelimitedFiles

const _N2S_DIR = joinpath(@__DIR__, "fixtures", "struct_sd_counts")
const _N2S_REF = joinpath(@__DIR__, "fixtures", "nb2_spatial_intercept", "reference_drmtmb.csv")

@testset "NB2 spatial(1 | g, coords) reproduces drmTMB's fixed-range fit" begin
    m, h = readdlm(joinpath(_N2S_DIR, "data_nb2_spatial.csv"), ',', Any; header = true)
    h = String.(vec(h)); col(n) = m[:, findfirst(==(n), h)]
    d = (; y = Float64.(col("y")), x1 = Float64.(col("x1")), x2 = Float64.(col("x2")),
         x3 = Float64.(col("x3")), g = String.(col("g")))
    cm = readdlm(joinpath(_N2S_DIR, "coords.csv"), ',', Any; skipstart = 1)
    coords = Dict(String(cm[i, 1]) => [Float64(cm[i, 2]), Float64(cm[i, 3])] for i in 1:size(cm, 1))
    r, rh = readdlm(_N2S_REF, ',', Any; header = true)
    est = Float64.(r[:, 2]); se = Float64.(r[1:5, 3]); llR = Float64(r[1, 4])
    @test all(lowercase.(string.(r[:, 5])) .== "true")                     # R: is_converged() TRUE

    fit = drm(bf(@formula(y ~ x1 + x2 + x3 + spatial(1 | g)), @formula(sigma ~ 1)), NegBinomial2();
              data = d, coords = coords)
    @test fit.converged
    @test length(fit.theta) == 6                     # 4 betas, log sigma, log SD; no range parameter
    @test loglikelihood(fit) ≈ llR atol = 1e-4
    @test all(abs.(fit.theta[1:5] .- est[1:5]) .<= 1e-3 .* se)
    @test abs(fit.theta[6] - est[6]) <= 5e-3
end
