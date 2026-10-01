# Separation screen for Binomial / Bernoulli fixed effects (#731, #728; drmTMB
# twin #1268). Policy: detect and warn -- the fit is returned, a warning names
# the affected coefficients, their SEs are Inf. The same data and the same
# expected flagged coefficients are checked in the drmTMB twin
# (tests/testthat/test-binomial-separation.R); keep the constants identical.
using DRModels
using Test, Logging, LinearAlgebra

# #728 cell-47 recipe: n = 60, x clustered near -2 / +2 (x = ±2 + N(0, 0.25),
# rounded to 3 dp; NumPy default_rng seed 20307905), true slope 3.5, y the
# Bernoulli draw -- it came out completely separated (y = 1 iff x > 0).
const _SEP_X47 = [
    -2.059, 2.3, -2.066, 1.799, -1.967, 1.757, -1.857, 1.903, -2.071, 2.125,
    -2.237, 2.02, -2.267, 1.801, -2.096, 1.58, -2.342, 1.706, -1.99, 1.843,
    -1.9, 2.578, -1.68, 1.907, -1.887, 1.945, -1.937, 1.715, -1.654, 2.21,
    -1.599, 1.656, -2.49, 2.34, -1.856, 1.961, -2.144, 2.277, -2.012, 1.795,
    -2.026, 2.361, -1.96, 1.932, -1.989, 2.349, -1.871, 2.043, -2.222, 2.174,
    -1.983, 1.885, -2.175, 2.686, -2.048, 1.724, -2.19, 1.964, -1.983, 2.275,
]
const _SEP_Y47 = Int.(_SEP_X47 .> 0)

function _sep_fit(y, x)
    logger = Test.TestLogger(min_level = Logging.Warn)
    fit = with_logger(logger) do
        drm(bf(@formula(y ~ 1 + x)), Binomial(); data = (; y = y, x = x))
    end
    recs = filter(r -> haskey(r.kwargs, :separation), logger.logs)
    return fit, recs, logger.logs
end

@testset "binomial separation: detect and warn (#731/#728)" begin
    @testset "_detect_separation: LP screen" begin
        X = [ones(6) [-3.0, -2, -1, 1, 2, 3]]
        d = DRModels._detect_separation(X, [0.0, 0, 0, 1, 1, 1], ones(6))
        @test d.separated && d.flagged == [1, 2] && d.conclusive          # complete

        xq = [-2.0, -1, 0, 0, 1, 2]; yq = [0.0, 0, 0, 1, 1, 1]            # tie at x = 0
        dq = DRModels._detect_separation([ones(6) xq], yq, ones(6))
        @test dq.separated && dq.flagged == [2]                           # quasi: slope only

        xo = [-3.0, -2, -1, 1, 2, 3]; yo = [0.0, 1, 0, 1, 0, 1]           # overlap
        do_ = DRModels._detect_separation([ones(6) xo], yo, ones(6))
        @test !do_.separated && isempty(do_.flagged)

        # binomial counts expand to rows: same verdict as the Bernoulli expansion
        dc = DRModels._detect_separation([ones(3) [-1.0, 0, 1]], [0.0, 2, 3], [3.0, 3, 3])
        xb = [-1.0, -1, -1, 0, 0, 0, 1, 1, 1]; yb = [0.0, 0, 0, 1, 1, 0, 1, 1, 1]
        db = DRModels._detect_separation([ones(9) xb], yb, ones(9))
        @test dc.separated && dc.flagged == db.flagged == [2]
    end

    @testset "complete separation toy" begin
        x = [-3.0, -2, -1, 1, 2, 3]; y = [0, 0, 0, 1, 1, 1]
        fit, recs, _ = _sep_fit(y, x)
        @test length(recs) == 1
        @test recs[1].kwargs[:flagged_coefficients] == ["(Intercept)", "x"]
        @test all(isinf, stderror(fit))
    end

    @testset "quasi-complete separation toy (flags the slope only)" begin
        x = [-2.0, -1, 0, 0, 1, 2]; y = [0, 0, 0, 1, 1, 1]
        fit, recs, _ = _sep_fit(y, x)
        @test length(recs) == 1
        @test recs[1].kwargs[:flagged_coefficients] == ["x"]
        se = stderror(fit)
        @test isinf(se[2])
    end

    @testset "#728 cell-47 near-separated dataset" begin
        fit, recs, _ = _sep_fit(_SEP_Y47, _SEP_X47)
        @test length(recs) == 1
        @test recs[1].kwargs[:flagged_coefficients] == ["(Intercept)", "x"]
        @test recs[1].kwargs[:separation] === :complete_or_quasi
        @test all(isinf, stderror(fit))
        @test fit.converged                                               # the fit is still returned
    end

    @testset "non-separated controls: no warning, vcov untouched" begin
        # strong-ish slope but overlap (eight flipped observations): finite MLE
        y = copy(_SEP_Y47); y[[1, 3, 5, 7]] .= 1; y[[2, 4, 6, 8]] .= 0
        fit, recs, logs = _sep_fit(y, _SEP_X47)
        @test isempty(recs)
        @test all(isfinite, stderror(fit))
        @test isempty(filter(r -> occursin("separation", string(r.message)), logs))
        # moderate-slope control
        xc = collect(range(-2, 2; length = 40))
        yc = [0,0,1,0,0,0,1,0,1,0,0,1,0,1,0,1,1,0,1,0, 0,1,1,0,1,1,0,1,1,1,0,1,1,1,1,0,1,1,1,1]
        fitc, recsc, _ = _sep_fit(yc, xc)
        @test isempty(recsc) && all(isfinite, stderror(fitc))
    end

    @testset "near-separation rule (numeric degeneracy)" begin
        μ = [0.5, 1e-9, 0.9]; 
        @test DRModels._near_separation(μ, [0.3, 5e4]) == [2]
        @test DRModels._near_separation(μ, [0.3, 5.0]) == Int[]           # SE not huge
        @test DRModels._near_separation([0.5, 0.2], [Inf, 1.0]) == Int[]  # no probability near 0/1
        @test DRModels._near_separation(μ, [NaN, 1.0]) == [1]
    end
end
