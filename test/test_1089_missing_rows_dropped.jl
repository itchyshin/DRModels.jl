# test_1089_missing_rows_dropped.jl — issue #1089.
#
# A `missing` in a predictor made `drm` throw `KeyError: key missing not found`
# from the StatsModels schema, on every family. A missing response on the
# Gaussian `(1 | g)` and `meta_V` routes was refused with a route-level
# `ArgumentError`. drmTMB drops those rows (R's `na.omit`), and so does `drm`
# now, with a warning that gives the count. Each fit below must equal the fit
# on the complete-case data that `drm_listwise` builds.

using DRModels
using Test, Random, LinearAlgebra, StableRNGs
using StatsModels: @formula

# The issue's DGP: y = 1 + 0.5x - 0.3z + u_g + e, G = 20, n = 400, with 30
# missing values in y and 20 in x (independent masks, so some rows overlap).
function _data_1089(; seed = 20281017801, n = 400, G = 20, nmiss_y = 30, nmiss_x = 20)
    rng = StableRNG(seed)
    g = repeat(1:G, inner = n ÷ G)
    x = randn(rng, n)
    z = randn(rng, n)
    u = 0.5 .* randn(rng, G)
    yv = 1 .+ 0.5 .* x .- 0.3 .* z .+ u[g] .+ randn(rng, n)
    y = Vector{Union{Missing,Float64}}(yv)
    xm = Vector{Union{Missing,Float64}}(x)
    y[randperm(rng, n)[1:nmiss_y]] .= missing
    xm[randperm(rng, n)[1:nmiss_x]] .= missing
    return (; y, x = xm, z, g)
end

@testset "#1089: rows with a missing predictor or response are dropped, with a count" begin
    d = _data_1089()
    n = length(d.y)
    nx = count(ismissing, d.x)
    ncc = count(.!ismissing.(d.y) .& .!ismissing.(d.x))
    ny_left = (n - nx) - ncc                 # missing responses among rows with x present
    @test 0 < ncc < n - nx                   # the masks overlap only partly

    @testset "Gaussian (1 | g): the issue's call fits the complete cases" begin
        mk() = bf(@formula(y ~ 1 + x + z + (1 | g)), @formula(sigma ~ 1))
        fit = @test_logs (:warn, Regex("dropped $nx of $n rows with a missing predictor value \\(x\\)")) (:warn, Regex("$ny_left of $(n - nx) rows have a missing/NaN response")) match_mode = :any drm(mk(), Gaussian(); data = d)
        ref = drm(mk(), Gaussian(); data = drm_listwise(mk(), d; verbose = false))
        @test nobs(fit) == ncc
        @test fit.coefnames == ref.coefnames  # x stays continuous after the drop
        @test fit.theta == ref.theta
        @test fit.loglik == ref.loglik
    end

    @testset "Gaussian fixed effects: missing x only" begin
        dx = (; y = Float64.(coalesce.(d.y, 0.0)), x = d.x, z = d.z)
        mk() = bf(@formula(y ~ x + z), @formula(sigma ~ z))
        fit = @test_logs (:warn, Regex("dropped $nx of $n rows")) match_mode = :any drm(mk(), Gaussian(); data = dx)
        ref = drm(mk(), Gaussian(); data = drm_listwise(mk(), dx; verbose = false))
        @test nobs(fit) == n - nx
        @test fit.theta == ref.theta
        @test fit.loglik == ref.loglik
    end

    @testset "prior weights follow the dropped rows ($(nameof(typeof(fam))))" for fam in (Gaussian(), Poisson())
        rng = StableRNG(20281017805)
        n3 = 120
        x = randn(rng, n3)
        yv = fam isa Poisson ? Float64.(rand(rng, 0:4, n3)) : 0.5 .+ x .+ randn(rng, n3)
        xm = Vector{Union{Missing,Float64}}(x)
        xm[[4, 50, 99]] .= missing
        w = 0.5 .+ rand(rng, n3)
        keep = .!ismissing.(xm)
        mk() = bf(@formula(y ~ x))
        fit = @test_logs (:warn, r"dropped 3 of 120 rows") match_mode = :any drm(mk(), fam; data = (; y = yv, x = xm), weights = w)
        ref = drm(mk(), fam; data = (; y = yv[keep], x = x[keep]), weights = w[keep])
        @test nobs(fit) == n3 - 3
        @test fit.theta == ref.theta
        @test control_info(fit).weights == w[keep]
    end

    @testset "$(nameof(typeof(fam))) (1 | g): missing x and y" for fam in (Poisson(), Gamma())
        rng = StableRNG(20281017802)
        G = 20; n2 = 400
        g = repeat(1:G, inner = n2 ÷ G)
        x = randn(rng, n2)
        η = 0.3 .+ 0.4 .* x .+ 0.3 .* randn(rng, G)[g]
        yv = fam isa Poisson ? Float64.([rand(rng, 0:max(0, round(Int, 2exp(e))))  for e in η]) :
                               exp.(η) .* (0.5 .+ rand(rng, n2))
        y = Vector{Union{Missing,Float64}}(yv)
        xm = Vector{Union{Missing,Float64}}(x)
        y[randperm(rng, n2)[1:15]] .= missing
        xm[randperm(rng, n2)[1:25]] .= missing
        dd = (; y, x = xm, g)
        mk() = bf(@formula(y ~ x + (1 | g)))
        fit = @test_logs (:warn, Regex("dropped 25 of $n2 rows with a missing predictor value \\(x\\)")) match_mode = :any drm(mk(), fam; data = dd)
        ref = drm(mk(), fam; data = drm_listwise(mk(), dd; verbose = false))
        @test nobs(fit) == count(.!ismissing.(y) .& .!ismissing.(xm))
        @test fit.theta == ref.theta
    end

    @testset "bivariate $(nameof(typeof(fam))): missing x" for fam in (Gaussian(), Student())
        rng = StableRNG(20281017806)
        n4 = 150
        x = randn(rng, n4)
        y1 = 0.3 .+ 0.5 .* x .+ randn(rng, n4)
        y2 = -0.2 .+ 0.4 .* x .+ 0.5 .* randn(rng, n4)
        xm = Vector{Union{Missing,Float64}}(x)
        xm[[2, 77, 140, 141]] .= missing
        db = (; y1, y2, x = xm)
        mk() = bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x))
        fit = @test_logs (:warn, r"dropped 4 of 150 rows with a missing predictor value \(x\)") match_mode = :any drm(mk(), fam; data = db)
        ref = drm(mk(), fam; data = drm_listwise(mk(), db; verbose = false))
        @test nobs(fit) == nobs(ref) == n4 - 4
        @test coef(fit) == coef(ref)
    end

    @testset "bivariate LogNormal and (Gaussian, Poisson): missing x" begin
        rng = StableRNG(20281017808)
        n6 = 120
        x = randn(rng, n6)
        xm = Vector{Union{Missing,Float64}}(x)
        xm[[3, 60]] .= missing
        y1 = exp.(0.2 .+ 0.3 .* x .+ 0.4 .* randn(rng, n6))
        y2 = exp.(-0.1 .+ 0.2 .* x .+ 0.3 .* randn(rng, n6))
        dl = (; y1, y2, x = xm)
        mkl() = bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x))
        fit = @test_logs (:warn, r"dropped 2 of 120 rows") match_mode = :any drm(mkl(), LogNormal(); data = dl)
        ref = drm(mkl(), LogNormal(); data = drm_listwise(mkl(), dl; verbose = false))
        @test nobs(fit) == n6 - 2
        @test coef(fit) == coef(ref)

        c2 = Float64.(rand(rng, 0:5, n6))
        dm = (; y1 = log.(y1), y2 = c2, x = xm)
        mkm() = bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x))
        fitm = @test_logs (:warn, r"dropped 2 of 120 rows") match_mode = :any drm(mkm(), (Gaussian(), Poisson()); data = dm, confint = false)
        refm = drm(mkm(), (Gaussian(), Poisson()); data = drm_listwise(mkm(), dm; verbose = false), confint = false)
        @test size(fitm.X1, 1) == n6 - 2         # the route returns a NamedTuple, not a DrmFit
        @test fitm.β1 == refm.β1 && fitm.β2 == refm.β2
        @test fitm.loglik == refm.loglik
    end

    @testset "SkewNormal: missing x and y" begin
        rng = StableRNG(20281017809)
        n7 = 150
        x = randn(rng, n7)
        y = Vector{Union{Missing,Float64}}(0.3 .+ 0.6 .* x .+ abs.(randn(rng, n7)) .- 0.5 .* randn(rng, n7))
        xm = Vector{Union{Missing,Float64}}(x)
        xm[[7, 8]] .= missing
        y[[20, 90, 91]] .= missing
        ds = (; y, x = xm)
        mks() = bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1))
        fit = @test_logs (:warn, r"dropped 2 of 150 rows") (:warn, r"3 of 148 rows have a missing/NaN response") match_mode = :any drm(mks(), SkewNormal(); data = ds)
        ref = drm(mks(), SkewNormal(); data = drm_listwise(mks(), ds; verbose = false))
        @test nobs(fit) == n7 - 5
        @test fit.theta == ref.theta
    end

    @testset "Gaussian meta_V: a missing response is dropped, not refused" begin
        rng = StableRNG(20281017803)
        k = 40
        v = 0.05 .+ 0.2 .* rand(rng, k)
        x = randn(rng, k)
        y = Vector{Union{Missing,Float64}}(0.2 .+ 0.3 .* x .+ sqrt.(v) .* randn(rng, k) .+ 0.2 .* randn(rng, k))
        y[[3, 17, 29]] .= missing
        dm = (; y, x, v)
        mk() = bf(@formula(y ~ x + meta_V(v)), @formula(sigma ~ 1))
        fit = @test_logs (:warn, r"3 of 40 rows have a missing/NaN response") match_mode = :any drm(mk(), Gaussian(); data = dm)
        ref = drm(mk(), Gaussian(); data = drm_listwise(mk(), dm; verbose = false))
        @test nobs(fit) == k - 3
        @test fit.theta == ref.theta
    end

    @testset "relmat: drop only when the level order of the matrix survives" begin
        rng = StableRNG(20281017804)
        G = 10; m = 6
        K = [0.6^abs(i - j) for i in 1:G, j in 1:G]       # not permutation-invariant
        # Grouped rows: dropping a row never changes the first-appearance order.
        g = repeat(1:G, inner = m)
        x = Vector{Union{Missing,Float64}}(randn(rng, G * m))
        y = 0.4 .* coalesce.(x, 0.0) .+ randn(rng, G)[g] .+ randn(rng, G * m)
        x[[1, 14, 33]] .= missing
        dr = (; y, x, g)
        mk() = bf(@formula(y ~ x + relmat(1 | g)), @formula(sigma ~ 1))
        fit = @test_logs (:warn, r"dropped 3 of 60 rows") match_mode = :any drm(mk(), Gaussian(); data = dr, K = K)
        ref = drm(mk(), Gaussian(); data = drm_listwise(mk(), dr; verbose = false), K = K)
        @test fit.theta == ref.theta

        # Interleaved rows: dropping row 1 moves level 1 behind levels 2..G, so
        # `K` would silently line up with the wrong levels. Refuse, by name.
        g2 = repeat(1:G, outer = m)
        x2 = Vector{Union{Missing,Float64}}(randn(rng, G * m))
        x2[1] = missing
        dr2 = (; y, x = x2, g = g2)
        err = try
            drm(mk(), Gaussian(); data = dr2, K = K); nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("relmat(1 | g)", sprint(showerror, err))
    end

    @testset "bootstrap after a drop: the length error names the fix" begin
        rng = StableRNG(20281017807)
        n5 = 60
        x = randn(rng, n5)
        y = 0.5 .+ x .+ randn(rng, n5)
        xm = Vector{Union{Missing,Float64}}(x)
        xm[[5, 9]] .= missing
        d5 = (; y, x = xm)
        mk() = bf(@formula(y ~ x))
        fit = drm(mk(), Gaussian(); data = d5)
        err = try
            bootstrap_ci(fit; data = d5, B = 3, rng = MersenneTwister(1)); nothing
        catch e
            e
        end
        @test occursin("drm_listwise", sprint(showerror, err))
        ci = bootstrap_ci(fit; data = drm_listwise(mk(), d5; verbose = false), B = 3,
                          rng = MersenneTwister(1))
        @test length(ci) == length(coef(fit))
    end

    @testset "every row has a missing predictor: refused by name" begin
        da = (; y = randn(StableRNG(1), 10), x = fill(missing, 10))
        err = try
            drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = da); nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("missing", sprint(showerror, err))
    end
end

# #1123: SkewNormal() was the one univariate fitter not wrapped in the observed-rows
# helper. It dropped the rows in its body (#1089) but, unlike the other families,
# did not rebuild the full rows (nobs, fitted parameters and scales per input row).
@testset "SkewNormal missing response (#1123)" begin
    rng = MersenneTwister(1123)
    n = 150
    x = randn(rng, n)
    y = 1.0 .+ 0.5 .* x .+ exp(-0.2) .* abs.(randn(rng, n)) .* 1.0 .- 0.5 .+ 0.3 .* randn(rng, n)
    complete = (y = y, x = x)
    ym = Vector{Union{Missing,Float64}}(y)
    ym[1] = missing
    ym[7] = NaN
    dmiss = (y = ym, x = x)
    f = bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1))

    fit = @test_logs (:warn, r"missing/NaN response") drm(f, SkewNormal(); data = dmiss)
    @test fit.nobs == n - 2

    keep = setdiff(1:n, [1, 7])
    ref = drm(f, SkewNormal(); data = (y = y[keep], x = x[keep]))
    @test fit.loglik ≈ ref.loglik atol = 1e-6
    @test coef(fit, :mu) ≈ coef(ref, :mu) atol = 1e-5
    @test all(isfinite, coef(fit, :nu))
    # full-row rebuild stores the slant α (identity link), not exp(α) (#1120)
    @test length(fit.scales[:nu]) == n
    @test all(≈(coef(fit, :nu)[1]), fit.scales[:nu])
end
