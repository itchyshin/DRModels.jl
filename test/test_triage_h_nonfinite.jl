# Non-finite inputs must be rejected at the front door, and a non-finite or
# sentinel optimum must not be stored as converged = true.
# Issues #1019 (Student random intercept), #1012 (spatial coords), #1009
# (temporal), #1021 (associate_pairs dropped rows). The same helper covers
# the sibling routes that build a design or a response.

using DRModels
using Test, Random, Logging

function _err(thunk)
    try
        thunk()
        return nothing
    catch e
        return e
    end
end

_names_arg(e, needle) = e isa ArgumentError && occursin(needle, e.msg)

@testset "triage H: non-finite inputs and converged backstop" begin
    Random.seed!(1019)

    @testset "shared helper names the argument" begin
        e = _err(() -> DRModels._require_finite_inputs(; weights=[1.0, NaN]))
        @test _names_arg(e, "weights")
        @test occursin("row 2", e.msg)
        e = _err(() -> DRModels._require_finite_inputs(; offset=[0.0, Inf]))
        @test _names_arg(e, "offset")
        e = _err(() -> DRModels._require_finite_inputs(; coords=[0.0 1.0; NaN 2.0]))
        @test _names_arg(e, "coords")
        @test occursin("row 2, column 1", e.msg)
        @test DRModels._report_converged(true, -12.0, [0.1, -0.2])
        @test !DRModels._report_converged(true, -Inf, [0.1])
        @test !DRModels._report_converged(true, -1e18, [0.1])
        @test !DRModels._report_converged(true, -floatmax(Float64), [0.1])
        @test !DRModels._report_converged(true, -4.0, [0.1, NaN])
        @test !DRModels._report_converged(false, -4.0, [0.1])
    end

    @testset "constructor backstop agrees with is_converged" begin
        y = [0.2, -0.1, 0.4, 0.0, -0.3]
        blocks = [:mu => 1:1]
        names = [:mu => ["(Intercept)"]]
        V = reshape([1.0], 1, 1)
        means = Dict(:mu => y)
        obs = Dict(:mu => y)
        scales = Dict(:sigma => ones(length(y)))
        bad_coef = DrmFit(Gaussian(), blocks, names, [NaN], V, -4.0, length(y), true,
                          means, obs, scales)
        @test !bad_coef.converged
        @test !is_converged(bad_coef)
        for ll in (-Inf, -1e18, -floatmax(Float64))
            bad_ll = DrmFit(Gaussian(), blocks, names, [0.2], V, ll, length(y), true,
                            means, obs, scales)
            @test !bad_ll.converged
            @test !is_converged(bad_ll)
        end
        ok = DrmFit(Gaussian(), blocks, names, [0.2], V, -4.0, length(y), true,
                    means, obs, scales)
        @test ok.converged
        @test is_converged(ok)
    end

    @testset "#1019 Student (1 | g) rejects Inf in the response" begin
        n = 40
        x = randn(n)
        g = repeat(1:4, inner = 10)
        y = 0.3 .+ 0.3 .* x .+ 0.2 .* randn(n)
        y[3] = Inf
        data = (; y, x, g)
        f = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1))
        e = _err(() -> drm(f, Student(); data = data))
        @test _names_arg(e, "response `y`")
        @test occursin("Inf", e.msg)
        @test occursin("row 3", e.msg)
        eL = _err(() -> drm(f, Student(); data = data, marginal = :Laplace))
        @test _names_arg(eL, "response `y`")
    end

    @testset "#1012 spatial coords reject NaN and Inf" begin
        n = 12
        site = repeat(1:3, inner = 4)
        x = randn(n)
        y = 1 .+ 0.2 .* x .+ 0.1 .* randn(n)
        f = bf(@formula(y ~ x + spatial(1 | site)), @formula(sigma ~ 1))
        data = (; y, x, site)
        for bad in (NaN, Inf)
            c = [0.0 0.0; 1.0 0.0; 3.0 1.0]
            c[2, 1] = bad
            e = _err(() -> drm(f, Gaussian(); data = data, coords = c))
            @test _names_arg(e, "coords")
            @test occursin("row 2, column 1", e.msg)
        end
    end

    @testset "#1009 temporal rejects Inf in the response" begin
        ids = repeat(1:8, inner = 4)
        occ = repeat(1:4, 8)
        x = randn(length(ids))
        y = 1 .+ 0.4 .* x .+ 0.2 .* randn(length(ids))
        y[5] = Inf
        data = (; y, x, id = ids, occ)
        f = bf(@formula(y ~ x + temporal(1 | id, occ, ar1)), @formula(sigma ~ 1))
        e = _err(() -> drm(f, Gaussian(); data = data))
        @test _names_arg(e, "response `y`")
        @test occursin("row 5", e.msg)
    end

    @testset "#1021 associate_pairs refuses a margin with dropped NaN rows" begin
        n = 40
        x = randn(n)
        b = Float64.(rand(n) .< 0.5)
        y = 1 .+ 0.4 .* x .+ 0.2 .* randn(n)
        yn = copy(y)
        yn[[3, 7, 9]] .= NaN
        fb = drm(bf(@formula(b ~ x)), Binomial(); data = (; b, x))
        fgn = with_logger(NullLogger()) do
            drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; y = yn, x))
        end
        @test nobs(fgn) == n - 3
        @test is_converged(fgn)
        e = _err(() -> associate_pairs(fgn, fb; kernel = latent_normal()))
        @test _names_arg(e, "dropped")
        @test occursin("fit_1", e.msg)

        fg = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
        a = associate_pairs(fg, fb; kernel = latent_normal())
        @test isfinite(a.loglik)
        @test a.loglik > -1e15
        @test isfinite(a.eta)
    end

    @testset "sibling routes share the front door" begin
        n = 24
        x = randn(n)
        y = 0.2 .+ 0.3 .* x .+ 0.2 .* randn(n)
        g = repeat(1:4, inner = 6)

        yinf = copy(y); yinf[2] = -Inf
        e = _err(() -> drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian();
                           data = (; y = yinf, x)))
        @test _names_arg(e, "response `y`")
        @test occursin("-Inf", e.msg)

        e = _err(() -> drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian();
                           data = (; y = yinf, x, g)))
        @test _names_arg(e, "response `y`")

        tree = "((t1:0.5,t2:0.5):0.5,(t3:0.5,t4:0.5):0.5);"
        sp = repeat(["t1", "t2", "t3", "t4"], inner = 6)
        e = _err(() -> drm(bf(@formula(y ~ x + phylo(1 | sp)), @formula(sigma ~ 1)),
                           Gaussian(); data = (; y = yinf, x, sp), tree = tree))
        @test _names_arg(e, "response `y`")

        y1 = copy(y); y2 = copy(y); y2[4] = Inf
        e = _err(() -> drm(bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
                              sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                              rho12 = @formula(rho12 ~ 1)),
                           Gaussian(); data = (; y1, y2, x)))
        @test _names_arg(e, "response `y2`")

        y2c = Float64.(rand(n) .< 0.5)
        e = _err(() -> drm(bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
                              sigma1 = @formula(sigma1 ~ 1)),
                           (Gaussian(), Poisson());
                           data = (; y1 = yinf, y2 = y2c, x)))
        @test _names_arg(e, "response `y1`")

        X = hcat(ones(n), x)
        e = _err(() -> DRModels.fit_mixed_family(; y1 = yinf, X1 = X, fam1 = Gaussian(),
                                                 y2 = y, X2 = X, fam2 = Gaussian()))
        @test _names_arg(e, "response `y1`")

        xp = copy(x); xp[6] = NaN
        e = _err(() -> drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian();
                           data = (; y, x = xp)))
        @test _names_arg(e, "predictor `x`")
        @test occursin("row 6", e.msg)

        counts = [1, 0, 2, 1, 3, 0]
        xx = randn(6)
        exposure = [1.0, 1.2, Inf, 0.8, 1.0, 0.9]
        e = _err(() -> drm(bf(@formula(y ~ x + offset(exposure))), Poisson();
                           data = (; y = counts, x = xx, exposure)))
        @test _names_arg(e, "offset")

        e = _err(() -> drm_bridge(; formula = "y ~ x; sigma ~ 1", family = "gaussian",
                                  data = (; y = yinf, x)))
        @test _names_arg(e, "response `y`")

        Y = randn(4, 2); Y[1, 2] = Inf
        e = _err(() -> DRModels.drm_bridge_q2_phylo(; Y = Y, X = randn(4, 1),
                                                   species = 1:4, tree = "unused"))
        @test _names_arg(e, "`Y`")
        Q = [1.0 NaN; NaN 1.0]
        e = _err(() -> DRModels.drm_bridge_q2_known_precision(; Y = randn(3, 2), X = randn(3, 1),
                                                            group = 1:3, Q = Q))
        @test _names_arg(e, "`Q`")

        c = [0.0 0.0; 1.0 NaN; 2.0 0.5]
        id = repeat(1:3, inner = 4)
        e = _err(() -> drm(bf(@formula(y ~ x + spatial(1 | id))), Poisson();
                           data = (; y = fill(1, length(id)), x = randn(length(id)), id),
                           coords = c, se = false))
        @test _names_arg(e, "coords")
    end

    @testset "finite data still fits, and NaN responses are still omitted" begin
        n = 40
        x = randn(n)
        y = 0.2 .+ 0.5 .* x .+ 0.3 .* randn(n)
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
        @test fit.converged
        @test is_converged(fit)
        @test isfinite(loglik(fit))
        @test all(isfinite, coef(fit))
        yn = copy(y); yn[3] = NaN
        fitn = with_logger(NullLogger()) do
            drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; y = yn, x))
        end
        @test nobs(fitn) == n - 1
        @test fitn.converged
        @test isfinite(loglik(fitn))
        @test all(isfinite, coef(fitn))
    end

    @testset "missing response drops a NaN predictor on that row" begin
        # Review case F. Base fitted this (nobs = n - 1). The predictor check
        # used to run before the row was dropped.
        rng = MersenneTwister(7)
        n = 120
        x = randn(rng, n)
        y = 1 .+ 0.5 .* x .+ randn(rng, n)
        yb = Vector{Union{Missing,Float64}}(y)
        yb[7] = missing
        xb = copy(x)
        xb[7] = NaN
        fit = with_logger(NullLogger()) do
            drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian();
                data = (; y = yb, x = xb))
        end
        @test nobs(fit) == n - 1
        @test fit.converged
        @test is_converged(fit)
        @test isfinite(loglik(fit))
        @test all(isfinite, coef(fit))

        # The same row pattern on a family that subsets first, then rebuilds
        # the full-length prediction from the original table.
        yp = Float64.(rand(rng, 0:3, n))
        ypm = Vector{Union{Missing,Float64}}(yp)
        ypm[7] = missing
        fitp = with_logger(NullLogger()) do
            drm(bf(@formula(y ~ x)), Poisson(); data = (; y = ypm, x = xb))
        end
        @test nobs(fitp) == n - 1
        @test isfinite(loglik(fitp))

        # A NaN predictor on an observed row is still an error.
        xbad = copy(x)
        xbad[9] = NaN
        e = _err(() -> drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian();
                           data = (; y = yb, x = xbad)))
        @test _names_arg(e, "predictor `x`")
        @test occursin("row 9", e.msg)
    end

    @testset "named checks outside the fixed design" begin
        n = 24
        x = randn(n)
        y = 0.2 .+ 0.3 .* x .+ 0.2 .* randn(n)
        g = repeat(1:4, inner = 6)
        z = randn(n)
        z[3] = Inf
        e = _err(() -> drm(bf(@formula(y ~ x + (1 + z | g)), @formula(sigma ~ 1)),
                           Gaussian(); data = (; y, x, z, g)))
        @test _names_arg(e, "predictor `z`")
        @test occursin("row 3", e.msg)

        v = fill(0.25, n)
        v[2] = Inf
        e = _err(() -> drm(bf(@formula(y ~ x + meta_V(v)), @formula(sigma ~ 1)),
                           Gaussian(); data = (; y, x, v)))
        @test _names_arg(e, "meta_V(v)")
        @test occursin("row 2", e.msg)

        tree = "((t1:0.5,t2:0.5):0.5,(t3:0.5,t4:0.5):0.5);"
        sp = repeat(["t1", "t2", "t3", "t4"], inner = 6)
        e = _err(() -> drm(bf(@formula(y ~ x + phylo(1 + z | sp)), @formula(sigma ~ 1)),
                           Gaussian(); data = (; y, x, z, sp), tree = tree))
        @test _names_arg(e, "predictor `z`")
        @test occursin("row 3", e.msg)

        coords = (easting = [0.0, NaN, 1.0], northing = [0.0, 1.0, 0.0])
        id = repeat(1:3, inner = 4)
        e = _err(() -> drm(bf(@formula(y ~ 1 + spatial(1 | id)), @formula(sigma ~ 1)),
                           Gaussian();
                           data = (; y = randn(length(id)), id), coords = coords))
        @test _names_arg(e, "coords")
        @test occursin("row 2", e.msg)

        K = [1.0 0.2; 0.2 NaN]
        id2 = repeat(1:2, inner = 4)
        e = _err(() -> drm(bf(@formula(y ~ 1 + relmat(1 | id)), @formula(sigma ~ 1)),
                           Gaussian();
                           data = (; y = randn(length(id2)), id = id2), K = K))
        @test _names_arg(e, "`K`")

        e = _err(() -> DRModels.make_phy([(3, 1, 0.5), (3, 2, Inf)], 2))
        @test _names_arg(e, "branch length")
        @test occursin("3 -> 2", e.msg)

        e = _err(() -> DRModels.fit_phylo_interaction(
            [1.0, Inf, 0.0, 0.0], ones(4, 1),
            [1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0]))
        @test _names_arg(e, "response `y`")
        @test occursin("row 2", e.msg)

        s = [1.0, 0.0, 2.0, 1.0]
        f = [1.0, Inf, 0.0, 1.0]
        e = _err(() -> drm(bf(@formula(cbind(s, f) ~ x)), Binomial();
                           data = (; s, f, x = randn(4))))
        @test _names_arg(e, "`f`")
        @test occursin("row 2", e.msg)
    end

    @testset "predict(newdata) rejects a NaN predictor" begin
        n = 30
        x = randn(n)
        y = 0.2 .+ 0.4 .* x .+ 0.2 .* randn(n)
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
        newx = copy(x)
        newx[4] = NaN
        e = _err(() -> predict(fit, (; x = newx)))
        @test _names_arg(e, "predictor `x`")
        @test occursin("row 4", e.msg)
        @test all(isfinite, predict(fit, (; x = x)))
    end
end
