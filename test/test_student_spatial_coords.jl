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
    fits = Dict{String,Any}()
    for (cell, term, nsd) in (("q1", "spatial(1 | g)", 1), ("q2", "spatial(1 + x1 | g)", 2))
        rows = findall(==(cell), String.(r[:, 1]))
        est = Float64.(r[rows, 4]); llR = Float64(r[rows[1], 6])
        nfix = length(rows) - nsd
        se = Float64.(r[rows[1:nfix], 5])
        @test all(lowercase.(string.(r[rows, 7])) .== "true")              # R: is_converged() TRUE
        fml = cell == "q1" ? @formula(y ~ x1 + x2 + x3 + spatial(1 | g)) :
                             @formula(y ~ x1 + x2 + x3 + spatial(1 + x1 | g))
        fit = drm(bf(fml, @formula(sigma ~ 1)), Student(); data = d, coords = coords)
        fits[cell] = fit
        @testset "$term" begin
            @test fit.converged
            @test length(fit.theta) == length(rows)       # betas, log sigma, nu predictor, log SDs; no range parameter
            @test loglikelihood(fit) ≈ llR atol = 1e-4
            @test all(abs.(fit.theta[1:nfix] .- est[1:nfix]) .<= 1e-3 .* se)
            @test all(abs.(fit.theta[nfix+1:end] .- est[nfix+1:end]) .<= 5e-3)
        end
    end

    # Structure of the fit: names, nobs, finite uncertainty, SD labels use the slope variable name.
    q1, q2 = fits["q1"], fits["q2"]
    @test nobs(q2) == 400 && nobs(q1) == 400
    for f in (q1, q2)
        @test all(isfinite, stderror(f)) && all(isfinite, vcov(f))
        @test size(vcov(f)) == (length(f.theta), length(f.theta))
    end
    @test Dict(q2.blocks)[:resd] == 7:8 && Dict(q1.blocks)[:resd] == 7:7
    @test Dict(q2.coefnames)[:resd] == ["g", "g:x1"] && Dict(q1.coefnames)[:resd] == ["g"]
    @test sort(collect(keys(re_sd(q2))); by = string) == [:g, Symbol("g:x1")]
    @test collect(keys(re_sd(q1))) == [:g]
    @test re_sd(q2)[:g] ≈ exp(q2.theta[7]) && re_sd(q2)[Symbol("g:x1")] ≈ exp(q2.theta[8])

    # se = false: same point estimate, uncertainty skipped (NaN), still converged.
    q2n = drm(bf(@formula(y ~ x1 + x2 + x3 + spatial(1 + x1 | g)), @formula(sigma ~ 1)), Student();
              data = d, coords = coords, se = false)
    @test q2n.converged && all(isnan, vcov(q2n))
    @test maximum(abs.(q2n.theta .- q2.theta)) < 1e-6

    # A second structured marker is refused, not silently dropped (it used to fit spatial(1 | g) alone).
    h2 = String.(col("g2")); dh = merge(d, (; h = h2))
    for second in (:(relmat(1 | h)), :(animal(1 | h)), :(phylo(1 | h)), :(spatial(1 | h)))
        for lhs in (:(spatial(1 | g)), :(spatial(1 + x1 | g)))
            f = bf(@eval(@formula(y ~ x1 + $lhs + $second)), @formula(sigma ~ 1))
            err = try drm(f, Student(); data = dh, coords = coords); nothing catch e; e end
            @test err !== nothing
            @test occursin("one structured term per formula", sprint(showerror, err))
        end
    end

    # `sigma` / `nu` must be the constant intercept: a single non-intercept column is refused too.
    dz = merge(d, (; z = Float64.(col("z"))))
    for (sg, nu) in ((@formula(sigma ~ 0 + z), @formula(nu ~ 1)), (@formula(sigma ~ 1), @formula(nu ~ 0 + z)),
                     (@formula(sigma ~ z), @formula(nu ~ 1)))
        err = try drm(bf(@formula(y ~ x1 + spatial(1 | g)), sg, nu), Student(); data = dz, coords = coords); nothing catch e; e end
        @test err isa ArgumentError && occursin("requires `sigma ~ 1` and `nu ~ 1`", err.msg)
    end

    # Forgotten `coords`: a refusal that names the term (slope form) or the structured marker (intercept form).
    err1 = try drm(bf(@formula(y ~ x1 + spatial(1 | g)), @formula(sigma ~ 1)), Student(); data = d); nothing catch e; e end
    err2 = try drm(bf(@formula(y ~ x1 + spatial(1 + x1 | g)), @formula(sigma ~ 1)), Student(); data = d); nothing catch e; e end
    @test err1 !== nothing && occursin("structured", sprint(showerror, err1))
    @test err2 !== nothing && occursin("spatial(1 + x1 | g)", sprint(showerror, err2))

    # A missing value in the slope column drops that row (the shared complete-case rule), nobs 399.
    x1m = Union{Missing,Float64}[i == 5 ? missing : v for (i, v) in enumerate(d.x1)]
    dm = merge(d, (; x1 = x1m))
    fm = @test_logs (:warn, r"dropped 1 of 400") match_mode = :any drm(
        bf(@formula(y ~ x1 + x2 + x3 + spatial(1 + x1 | g)), @formula(sigma ~ 1)), Student(); data = dm, coords = coords)
    @test nobs(fm) == 399
end
