# Student `spatial(1 | g, coords)` and `spatial(1 + x1 | g, coords)` (no SD formula): the coordinate route uses
# drmTMB's fixed-range kernel, exp(-d / r) + 1e-6 I with r the median positive pairwise distance, and
# TMB-convention Laplace, like the Poisson, Gamma and NB2 routes. The `+ x1` form is drmTMB's independent
# intercept and slope (two SDs, correlation fixed at 0). Before this slice Julia refused both cells (gap map
# cells `student|spatial|lab0|q1|sd-none|cm0|ML` and `...|q2|...`, R-only).
# Reference: drmTMB 0.8.1, ML (test/fixtures/student_spatial/reference_drmtmb.csv; script
# ref_student_spatial.R, same folder). Rule fixed before the Julia run:
#   |logLik diff| <= 1e-4; each fixed effect within 1e-3 R standard errors; each log SD within 5e-3;
#   R reports is_converged() = TRUE (otherwise the cell is flagged, not counted).
using DRModels, Test, DelimitedFiles

const _STS_DIR = joinpath(@__DIR__, "fixtures", "student_spatial")
const _STS_COORDS = joinpath(@__DIR__, "fixtures", "struct_sd_counts", "coords.csv")

@testset "Student spatial(coords) reproduces drmTMB's fixed-range fits" begin
    m, h = readdlm(joinpath(_STS_DIR, "data_student_spatial.csv"), ',', Any; header = true)
    h = String.(vec(h)); col(n) = m[:, findfirst(==(n), h)]
    d = (; y = Float64.(col("y")), x1 = Float64.(col("x1")), x2 = Float64.(col("x2")),
         x3 = Float64.(col("x3")), g = String.(col("g")))
    cm = readdlm(_STS_COORDS, ',', Any; skipstart = 1)
    coords = Dict(String(cm[i, 1]) => [Float64(cm[i, 2]), Float64(cm[i, 3])] for i in 1:size(cm, 1))
    r, rh = readdlm(joinpath(_STS_DIR, "reference_drmtmb.csv"), ',', Any; header = true)
    for (cell, term, nsd) in (("q1", "spatial(1 | g)", 1), ("q2", "spatial(1 + x1 | g)", 2))
        rows = findall(==(cell), String.(r[:, 1]))
        est = Float64.(r[rows, 4]); llR = Float64(r[rows[1], 6])
        nfix = length(rows) - nsd
        se = Float64.(r[rows[1:nfix], 5])
        @test all(lowercase.(string.(r[rows, 7])) .== "true")              # R: is_converged() TRUE
        fml = cell == "q1" ? @formula(y ~ x1 + x2 + x3 + spatial(1 | g)) :
                             @formula(y ~ x1 + x2 + x3 + spatial(1 + x1 | g))
        fit = drm(bf(fml, @formula(sigma ~ 1)), Student(); data = d, coords = coords)
        @testset "$term" begin
            @test fit.converged
            @test length(fit.theta) == length(rows)       # betas, log sigma, nu predictor, log SDs; no range parameter
            @test loglikelihood(fit) ≈ llR atol = 1e-4
            @test all(abs.(fit.theta[1:nfix] .- est[1:nfix]) .<= 1e-3 .* se)
            @test all(abs.(fit.theta[nfix+1:end] .- est[nfix+1:end]) .<= 5e-3)
        end
    end
end
