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

# #910-review fixture (drmTMB twin: the same constants in
# tests/testthat/test-binomial-separation.R): R `set.seed(3); x <- round(rnorm(200),
# 3); y <- rbinom(200, 1, plogis(-1 + 9 * x))`. A steep but well-identified slope
# (glm: beta = 9.308, SE = 2.014, z = 4.62; min fitted probability 4.9e-11). The
# pre-fix near rule (a fitted probability within 1e-8 of 0/1 AND an SE > 1e4)
# flagged it as soon as x was divided by 1e5 (SE 2.0e5). glm reference values:
const _SEP_XH = [
    -0.962, -0.293, 0.259, -1.152, 0.196, 0.030, 0.085, 1.117, -1.219, 1.267, -0.745,
    -1.131, -0.716, 0.253, 0.152, -0.308, -0.953, -0.648, 1.224, 0.200, -0.578, -0.942,
    -0.204, -1.666, -0.484, -0.741, 1.161, 1.012, -0.072, -1.137, 0.901, 0.852, 0.728,
    0.737, -0.352, 0.706, 1.300, 0.038, -0.979, 0.794, 0.787, -0.310, 1.699, -0.795,
    0.348, -2.265, -0.162, 1.131, -0.456, -0.899, 0.727, -0.809, 0.267, -1.737, -1.411,
    -0.454, -1.035, 1.362, 0.917, -0.785, 0.574, 0.918, 0.256, 0.352, 1.174, -0.481,
    -0.419, 0.955, -1.289, 0.186, -0.031, 0.467, 1.024, 0.267, 0.232, 0.748, 1.217,
    0.383, -0.988, -0.157, 1.736, -0.352, 0.689, 1.224, 0.794, -0.006, 0.219, -0.886,
    0.440, -0.886, -0.854, -0.990, -0.651, 1.054, -0.391, -0.071, -0.462, 0.541, 0.932,
    -0.209, 0.617, -0.405, 1.053, 0.602, 1.017, 0.608, 0.207, -1.898, -0.683, 0.481,
    -0.463, -0.280, -0.414, 1.619, -0.721, -0.453, 0.014, 0.216, 0.189, -0.050, -1.495,
    0.368, 0.517, -0.484, 0.675, -0.762, 0.386, -0.664, -1.724, 1.156, 0.694, 0.143,
    1.493, -1.632, 0.128, -2.404, 1.444, -0.879, -1.306, -0.877, -1.164, -1.982, -0.990,
    -0.152, 0.913, 0.408, -1.242, -0.643, 1.930, 0.410, -1.291, 2.635, 0.487, 0.854,
    1.088, 0.226, 0.068, -0.985, -1.311, 2.464, -0.665, 0.913, 0.965, 1.608, 1.835,
    0.702, 1.218, -1.124, 0.668, 1.216, 0.235, -0.419, 0.238, -0.551, -0.501, 1.164,
    2.156, -1.709, -1.601, -1.039, 0.323, -0.889, 0.394, 0.237, -0.430, -0.548, -1.322,
    0.682, 2.163, -0.417, -1.357, -0.671, 0.650, 0.771, 2.677, -1.371, 0.058, -0.197,
    -1.262, -0.662
]
const _SEP_YH = [
    0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1,
    0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0,
    0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1,
    1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0,
    0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0,
    0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0,
    1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0,
    0, 0, 0, 0
]
const _SEP_GLM_H = (b0 = -1.1765550222, b1 = 9.3079484214, loglik = -24.9124059683)

function _sep_capture(f)
    logger = Test.TestLogger(min_level = Logging.Warn)
    fit = with_logger(f, logger)
    recs = filter(r -> haskey(r.kwargs, :separation), logger.logs)
    return fit, recs, logger.logs
end
_sep_fit(y, x) = _sep_capture(() -> drm(bf(@formula(y ~ 1 + x)), Binomial(); data = (; y = y, x = x)))

# A flagged coefficient: Inf variance, NaN covariances, SE Inf, and the Wald
# interval and coeftable row agree on (-Inf, Inf).
function _sep_check_flagged(fit, j)
    V = vcov(fit)
    @test V[j, j] == Inf
    @test all(isnan, V[j, setdiff(axes(V, 2), [j])])
    @test isinf(stderror(fit)[j])
    ci = confint(fit)[j]
    @test ci.lower == -Inf && ci.upper == Inf
    ct = coeftable(fit)
    @test ct.cols[5][j] == -Inf && ct.cols[6][j] == Inf
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
        _sep_check_flagged(fit, 1); _sep_check_flagged(fit, 2)
    end

    @testset "quasi-complete separation toy (flags the slope only)" begin
        x = [-2.0, -1, 0, 0, 1, 2]; y = [0, 0, 0, 1, 1, 1]
        fit, recs, _ = _sep_fit(y, x)
        @test length(recs) == 1
        @test recs[1].kwargs[:flagged_coefficients] == ["x"]
        se = stderror(fit)
        @test isinf(se[2]) && isfinite(se[1])
        _sep_check_flagged(fit, 2)
        @test isfinite(vcov(fit)[1, 1])
        ci = confint(fit)[1]
        @test isfinite(ci.lower) && isfinite(ci.upper)
    end

    @testset "cbind(successes, failures) through drm (quasi: slope only)" begin
        # grouped counts; same verdict as the Bernoulli expansion in the LP testset
        d = (; s = [0, 2, 3], fl = [3, 1, 0], x = [-1.0, 0, 1])
        fit, recs, _ = _sep_capture(() -> drm(bf(@formula(cbind(s, fl) ~ 1 + x)), Binomial(); data = d))
        @test length(recs) == 1
        @test recs[1].kwargs[:flagged_coefficients] == ["x"]
        _sep_check_flagged(fit, 2)
        # Julia's Binomial() has no `weights` argument (drmTMB tests that path).
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

    @testset "near-separation rule: scale-free, never from a missing SE" begin
        μ = [0.5, 1e-9, 0.9]
        span = DRModels._SEP_NEAR_SPAN
        @test span ≈ 2 * log((1 - 1e-8) / 1e-8)                                # 36.84
        # coefficient 2 spans the whole logit range with |z| = 0.01: flagged
        @test DRModels._near_separation(μ, [0.1, 40.0], [1.0, 4000.0], [0.0, 1.0]) == [2]
        # same coefficient on a covariate divided by 1e5: β, SE and the span rescale,
        # the verdict does not
        @test DRModels._near_separation(μ, [0.1, 4e6], [1.0, 4e8], [0.0, 1e-5]) == [2]
        @test DRModels._near_separation(μ, [0.1, 40.0], [1.0, 8.0], [0.0, 1.0]) == Int[]  # z = 5
        @test DRModels._near_separation(μ, [0.1, 20.0], [1.0, 4000.0], [0.0, 1.0]) == Int[]  # span 20
        @test DRModels._near_separation([0.5, 0.2], [0.1, 40.0], [1.0, 4000.0], [0.0, 1.0]) == Int[]
        # an unavailable SE (failed Hessian -> NaN / Inf) is never near separation
        @test DRModels._near_separation(μ, [0.1, 40.0], [NaN, NaN], [0.0, 1.0]) == Int[]
        @test DRModels._near_separation(μ, [0.1, 40.0], [Inf, Inf], [0.0, 1.0]) == Int[]
    end

    @testset "healthy steep slope, raw and rescaled covariate: not flagged (#910 review)" begin
        fit, recs, logs = _sep_fit(_SEP_YH, _SEP_XH)
        @test isempty(recs)
        @test isempty(filter(r -> occursin("separation", string(r.message)), logs))
        @test coef(fit)[1] ≈ _SEP_GLM_H.b0 atol = 1e-5
        @test coef(fit)[2] ≈ _SEP_GLM_H.b1 atol = 1e-5
        @test loglikelihood(fit) ≈ _SEP_GLM_H.loglik atol = 1e-8
        @test all(isfinite, stderror(fit))
        fits, recss, _ = _sep_fit(_SEP_YH, _SEP_XH ./ 1e5)
        @test isempty(recss)
        @test stderror(fits)[2] > 1e4                     # what the old rule tripped on
        @test minimum(m -> min(m, 1 - m), fitted(fits)) < 1e-8
        @test all(isfinite, stderror(fits))
        @test coef(fits)[2] / stderror(fits)[2] ≈ coef(fit)[2] / stderror(fit)[2] rtol = 1e-3
    end

    @testset "healthy fit: the guard returns the vcov untouched (byte-identical)" begin
        X = [ones(length(_SEP_XH)) _SEP_XH]
        fit, _, _ = _sep_fit(_SEP_YH, _SEP_XH)
        V = copy(vcov(fit))
        V2 = DRModels._separation_guard(X, Float64.(_SEP_YH), ones(length(_SEP_YH)),
                                        coef(fit), V, fitted(fit), ["(Intercept)", "x"])
        @test V2 === V
    end

    @testset "inconclusive check warns, never reads as no separation" begin
        # overlapping toy: proving "no separation" needs NNLS steps, so a zero
        # budget leaves the check inconclusive (a complete-separation design is
        # decided at y = 0 without any step)
        X = [ones(6) [-3.0, -2, -1, 1, 2, 3]]; y = [0.0, 1, 0, 1, 0, 1]
        d = DRModels._detect_separation(X, y, ones(6); maxiter = 0)
        @test !d.conclusive && !d.separated
        @test DRModels._detect_separation(X, y, ones(6)).conclusive
        θ = [0.0, 0.3]; V = [1.0 0.0; 0.0 1.0]
        logger = Test.TestLogger(min_level = Logging.Warn)
        V2 = with_logger(logger) do
            DRModels._separation_guard(X, y, ones(6), θ, V, fill(0.5, 6), ["(Intercept)", "x"];
                                       maxiter = 0)
        end
        recs = filter(r -> haskey(r.kwargs, :separation), logger.logs)
        @test length(recs) == 1 && recs[1].kwargs[:separation] === :inconclusive
        @test occursin("inconclusive", string(recs[1].message))
        @test V2 == V
    end

    @testset "timing guard: separated designs with many columns stay fast" begin
        # 40-level factor, n = 1000: every third level all-0, every third all-1
        n = 1000; lv = [mod1(7i, 40) for i in 1:n]
        X = zeros(n, 40); X[:, 1] .= 1
        for i in 1:n
            lv[i] > 1 && (X[i, lv[i]] = 1)
        end
        y = [lv[i] % 3 == 0 ? 0.0 : lv[i] % 3 == 1 ? 1.0 : Float64(isodd(i)) for i in 1:n]
        DRModels._detect_separation(X, y, ones(n))               # compile
        t = @elapsed d = DRModels._detect_separation(X, y, ones(n))
        @test (d.separated && d.conclusive && t < 5) || !d.conclusive
        # n = 10,000, p = 10, complete separation on one column
        Xb = hcat(ones(10_000), [sin(0.37i * j) for i in 1:10_000, j in 1:9])
        yb = Float64.(Xb[:, 2] .> 0)
        t2 = @elapsed db = DRModels._detect_separation(Xb, yb, ones(10_000))
        @test db.separated && db.flagged == collect(1:10) && t2 < 5
    end

    @testset "bootstrap refits do not repeat the separation warning" begin
        x = [-3.0, -2, -1, 1, 2, 3]; y = [0, 0, 0, 1, 1, 1]
        fit, recs, _ = _sep_fit(y, x)
        @test length(recs) == 1
        _, brecs, _ = _sep_capture(() -> bootstrap_result(fit; data = (; y = y, x = x), B = 3,
                                                          failures = :skip))
        @test isempty(brecs)
    end
end
