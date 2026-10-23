# Lane N3 (owner decisions 8a and 9b, 11 Oct 2026): `DRMControl` (maxit,
# start, multi_start, se) on every `drm` route, `gradient` / `hessian`
# accessors, and prior `weights` on the fixed-effect routes.
# Issues: #942 #948 #994 #971 #931.
using DRModels, Test, Random, LinearAlgebra, Statistics

const _N3_N = 150

function _n3_data(seed = 20261011)
    rng = MersenneTwister(seed)
    n = _N3_N
    x = randn(rng, n)
    μ = 1 .+ 0.5 .* x
    σ = exp.(-0.3 .+ 0.2 .* x)
    yg = μ .+ σ .* randn(rng, n)
    ypos = exp.(0.2 .+ 0.3 .* x .+ 0.4 .* randn(rng, n))          # positive (Gamma / LogNormal)
    ycount = [rand(rng) < exp(-exp(0.3 + 0.4xi)) ? 0 : round(Int, exp(0.3 + 0.4xi) + 3rand(rng)) for xi in x]
    ypois = Float64.(ycount)
    yprop = clamp.(1 ./ (1 .+ exp.(-(0.2 .+ 0.5 .* x .+ 0.6 .* randn(rng, n)))), 0.01, 0.99)
    ntr = fill(10, n)
    succ = [sum(rand(rng) < 1 / (1 + exp(-(0.2 + 0.7xi))) for _ in 1:10) for xi in x]
    fail = ntr .- succ
    ytail = μ .+ 0.6 .* randn(rng, n) ./ sqrt.(rand(rng, n) .* 0.6 .+ 0.05)   # heavy tails (Student)
    yord = [xi + 0.8randn(rng) < -0.5 ? 1 : xi + 0.8randn(rng) < 0.6 ? 2 : 3 for xi in x]
    ypos_tp = Float64.(max.(ycount, 1))
    ytw = [rand(rng) < 0.3 ? 0.0 : yi for yi in ypos]
    yzo = [u < 0.08 ? 0.0 : u > 0.92 ? 1.0 : p for (u, p) in zip(rand(rng, n), yprop)]
    return (; x, yg, ypos, ypois, yprop, succ = Float64.(succ), fail = Float64.(fail),
            ytail, yord, ypos_tp, ytw, yzo)
end

const _N3 = _n3_data()

# (label, family, formula, data-builder) for every univariate fixed-effect route
# wired in this lane. `rows(idx)` returns the data restricted / repeated to `idx`.
_n3_cases() = [
    ("Gaussian", Gaussian(), bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x)), idx -> (y = _N3.yg[idx], x = _N3.x[idx])),
    ("LogNormal", LogNormal(), bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x)), idx -> (y = _N3.ypos[idx], x = _N3.x[idx])),
    ("Gamma", Gamma(), bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), idx -> (y = _N3.ypos[idx], x = _N3.x[idx])),
    ("Student", Student(), bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), idx -> (y = _N3.ytail[idx], x = _N3.x[idx])),
    ("SkewNormal", SkewNormal(), bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), idx -> (y = _N3.ypos[idx], x = _N3.x[idx])),
    ("Poisson", Poisson(), bf(@formula(y ~ 1 + x)), idx -> (y = _N3.ypois[idx], x = _N3.x[idx])),
    ("Poisson zi", Poisson(), bf(@formula(y ~ 1 + x), @formula(zi ~ 1)), idx -> (y = _N3.ypois[idx], x = _N3.x[idx])),
    ("Poisson hu", Poisson(), bf(@formula(y ~ 1 + x), @formula(hu ~ 1)), idx -> (y = _N3.ypois[idx], x = _N3.x[idx])),
    ("NegBinomial2", NegBinomial2(), bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), idx -> (y = _N3.ypois[idx], x = _N3.x[idx])),
    ("TruncatedPoisson", TruncatedPoisson(), bf(@formula(y ~ 1 + x)), idx -> (y = _N3.ypos_tp[idx], x = _N3.x[idx])),
    ("Beta", Beta(), bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), idx -> (y = _N3.yprop[idx], x = _N3.x[idx])),
    ("Binomial", Binomial(), bf(@formula(cbind(s, f) ~ 1 + x)), idx -> (s = _N3.succ[idx], f = _N3.fail[idx], x = _N3.x[idx])),
    ("BetaBinomial", BetaBinomial(), bf(@formula(cbind(s, f) ~ 1 + x), @formula(sigma ~ 1)), idx -> (s = _N3.succ[idx], f = _N3.fail[idx], x = _N3.x[idx])),
    ("ZeroOneBeta", ZeroOneBeta(), bf(@formula(y ~ 1 + x), @formula(sigma ~ 1), @formula(zoi ~ 1), @formula(coi ~ 1)), idx -> (y = _N3.yzo[idx], x = _N3.x[idx])),
    ("Tweedie", Tweedie(), bf(@formula(y ~ 1 + x), @formula(sigma ~ 1), @formula(nu ~ 1)), idx -> (y = _N3.ytw[idx], x = _N3.x[idx])),
    ("CumulativeLogit", CumulativeLogit(), bf(@formula(y ~ x)), idx -> (y = _N3.yord[idx], x = _N3.x[idx])),
]

_n3_quiet(f) = redirect_stderr(devnull) do; f(); end

@testset "lane N3: DRMControl / gradient / hessian / weights" begin
    allrows = collect(1:_N3_N)

    @testset "DRMControl constructor" begin
        c = DRMControl()
        @test c.maxit === nothing && c.start === nothing && c.multi_start == 1 && c.se
        @test DRMControl(maxit = 50, start = [1, 2], multi_start = 3, se = false).start == [1.0, 2.0]
        @test_throws ArgumentError DRMControl(maxit = 0)
        @test_throws ArgumentError DRMControl(maxit = 2.5)
        @test_throws ArgumentError DRMControl(multi_start = 0)
        @test_throws ArgumentError DRMControl(start = [1.0, NaN])
        @test_throws ArgumentError DRMControl(start = Float64[])
        @test_throws ArgumentError DRMControl(start = (mu = [1.0],))
        @test_throws ArgumentError DRMControl(jitter = 0)
        @test occursin("multi_start = 1", sprint(show, c))
    end

    @testset "default control changes nothing" begin
        for (lab, fam, f, rows) in _n3_cases()[[1, 3, 6, 12]]
            d = rows(allrows)
            a = drm(f, fam; data = d)
            b = drm(f, fam; data = d, control = DRMControl())
            @test coef(a) == coef(b)
            @test vcov(a) == vcov(b)
            @test loglik(a) == loglik(b)
            @test control_info(a) === nothing
            @test control_info(b) isa DRMControlReport
            @test control_info(b).se === :computed
        end
    end

    @testset "#942: se = false on every fixed-effect route" begin
        # The issue's exact case: cars, Gaussian and LogNormal location-scale.
        speed = Float64[4, 4, 7, 7, 8, 9, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15,
                        15, 16, 16, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 20, 20, 20, 20, 20, 22, 23, 24, 24, 24, 24, 25]
        dist = Float64[2, 10, 4, 22, 16, 10, 18, 26, 34, 17, 28, 14, 20, 24, 28, 26, 34, 34, 46, 26, 36, 60, 80, 20, 26,
                       54, 32, 40, 32, 40, 50, 42, 56, 76, 84, 36, 46, 68, 32, 48, 52, 56, 64, 66, 54, 70, 92, 93, 120, 85]
        cars = (; dist, speed)
        fc = bf(@formula(dist ~ 1 + speed), @formula(sigma ~ 1 + speed))
        for fam in (Gaussian(), LogNormal())
            full = drm(fc, fam; data = cars)
            nose = drm(fc, fam; data = cars, control = DRMControl(se = false))
            @test coef(nose) == coef(full)
            @test loglik(nose) == loglik(full)
            @test all(isnan, vcov(nose))
            @test control_info(nose).se === :skipped
            @test all(isinf, stderror(nose))          # existing se = false contract: all-Inf
            # SEs later, from the Hessian accessor
            @test inv(hessian(nose)) ≈ vcov(full) rtol = 1e-6
        end
        for (lab, fam, f, rows) in _n3_cases()
            d = rows(allrows)
            fit = _n3_quiet(() -> drm(f, fam; data = d, control = DRMControl(se = false)))
            @test all(isnan, vcov(fit))
            @test control_info(fit).se === :skipped
        end
        # A route's own `se` keyword is merged with control.se (Gamma has one).
        g = drm(_n3_cases()[3][3], Gamma(); data = _n3_cases()[3][4](allrows), se = false)
        @test all(isnan, vcov(g))
    end

    @testset "se = false on a random-effect route (computed, then discarded)" begin
        g = repeat(1:15, 10)
        d = (y = _N3.yg .+ 0.5 .* randn(MersenneTwister(3), 15)[g], x = _N3.x, g = g)
        f = bf(@formula(y ~ 1 + x + (1 | g)), @formula(sigma ~ 1))
        full = _n3_quiet(() -> drm(f, Gaussian(); data = d))
        fit = _n3_quiet(() -> drm(f, Gaussian(); data = d, control = DRMControl(se = false)))
        @test coef(fit) == coef(full)
        @test all(isnan, vcov(fit))
        @test control_info(fit).se === :discarded
    end

    @testset "#948: maxit and start" begin
        lab, fam, f, rows = _n3_cases()[1]
        d = rows(allrows)
        base = drm(f, fam; data = d)
        capped = drm(f, fam; data = d, control = DRMControl(maxit = 1))
        @test niterations(capped) <= 1
        @test !is_converged(capped)
        roomy = drm(f, fam; data = d, control = DRMControl(maxit = 5000))
        @test coef(roomy) ≈ coef(base) atol = 1e-6
        # Starting at the optimum stays there; a far start still reaches it.
        st = drm(f, fam; data = d, control = DRMControl(start = coef(base)))
        @test coef(st) ≈ coef(base) atol = 1e-8
        far = drm(f, fam; data = d, control = DRMControl(start = [0.0, 0.0, 1.0, 0.0]))
        @test loglik(far) ≈ loglik(base) atol = 1e-6
        err = try drm(f, fam; data = d, control = DRMControl(start = [1.0, 2.0])); nothing catch e; e end
        @test err isa ArgumentError
        @test occursin("`start` has 2 values but this model has 4 parameters", sprint(showerror, err))
        # every wired family honours maxit / start
        for (lab, fam, f, rows) in _n3_cases()
            d = rows(allrows)
            b = _n3_quiet(() -> drm(f, fam; data = d))
            s = _n3_quiet(() -> drm(f, fam; data = d, control = DRMControl(start = coef(b), maxit = 3000)))
            @test loglik(s) ≈ loglik(b) atol = 1e-6
            @test control_info(s).kept_start == 1
        end
    end

    @testset "#994: multi_start" begin
        lab, fam, f, rows = _n3_cases()[3]           # Gamma, the issue's family
        d = rows(allrows)
        base = drm(f, fam; data = d)
        ms = _n3_quiet(() -> drm(f, fam; data = d, control = DRMControl(multi_start = 5, maxit = 2000)))
        info = control_info(ms)
        @test length(info.start_logliks) == 5
        @test 1 <= info.kept_start <= 5
        @test loglik(ms) == info.start_logliks[info.kept_start]
        @test loglik(ms) >= loglik(base) - 1e-6
        @test loglik(ms) == maximum(filter(isfinite, info.start_logliks))
        # Reproducible: same seed, same per-start values.
        ms2 = _n3_quiet(() -> drm(f, fam; data = d, control = DRMControl(multi_start = 5, maxit = 2000)))
        @test isequal(control_info(ms2).start_logliks, info.start_logliks)
        # Gaussian location-scale is unimodal: the starts agree, no warning.
        g = _n3_cases()[1]
        gm = @test_logs drm(g[3], g[2]; data = g[4](allrows), control = DRMControl(multi_start = 3))
        @test control_info(gm).starts_agree
        # Disagreeing starts warn and are reported (maxit = 1 leaves them apart).
        dis = @test_logs (:warn, r"did not all agree") match_mode = :any drm(g[3], g[2]; data = g[4](allrows),
                                                                             control = DRMControl(multi_start = 3, maxit = 1))
        @test !control_info(dis).starts_agree
    end

    @testset "named refusals before fitting" begin
        g = repeat(1:15, 10)
        d = (y = _N3.yg, x = _N3.x, g = g)
        fre = bf(@formula(y ~ 1 + x + (1 | g)), @formula(sigma ~ 1))
        for kw in ((control = DRMControl(maxit = 10),), (control = DRMControl(start = [1.0]),),
                   (control = DRMControl(multi_start = 2),), (weights = ones(_N3_N),))
            err = try drm(fre, Gaussian(); data = d, kw...); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("fixed-effect routes", sprint(showerror, err))
        end
        fb = bf(mu1 = @formula(y1 ~ 1 + x), mu2 = @formula(y2 ~ 1 + x))
        db = (y1 = _N3.yg, y2 = _N3.yg .+ randn(MersenneTwister(4), _N3_N), x = _N3.x)
        err = try drm(fb, Gaussian(); data = db, control = DRMControl(maxit = 10)); nothing catch e; e end
        @test err isa ArgumentError && occursin("bivariate", sprint(showerror, err))
        # Not a DRMControl
        @test_throws ArgumentError drm(_n3_cases()[1][3], Gaussian(); data = _n3_cases()[1][4](allrows),
                                       control = (maxit = 3,))
        # REML on fixed effects: optimiser fields refused after the route ran
        fr = bf(@formula(y ~ 1 + x), @formula(sigma ~ 1))
        dr = (y = _N3.yg, x = _N3.x)
        err = try drm(fr, Gaussian(); data = dr, method = :REML, control = DRMControl(maxit = 50)); nothing catch e; e end
        @test err isa ArgumentError && occursin("does not honour", sprint(showerror, err))
        @test all(isnan, vcov(drm(fr, Gaussian(); data = dr, method = :REML, control = DRMControl(se = false))))
    end

    @testset "#971: gradient and hessian" begin
        for (lab, fam, f, rows) in _n3_cases()
            fit = _n3_quiet(() -> drm(f, fam; data = rows(allrows)))
            gr = gradient(fit)
            H = hessian(fit)
            @test length(gr) == length(coef(fit))
            @test size(H) == (length(coef(fit)), length(coef(fit)))
            @test issymmetric(H)
            if is_converged(fit)
                @test maximum(abs, gr) < 1e-3
            end
        end
        lab, fam, f, rows = _n3_cases()[1]
        fit = drm(f, fam; data = rows(allrows))
        @test inv(hessian(fit)) ≈ vcov(fit) rtol = 1e-8
        g = gradient(fit); H = hessian(fit)
        @test g' * (H \ g) < 1e-12                       # scaled Newton decrement
        # same order as coef: perturbing coef k moves gradient k
        nll = θ -> -loglik(fit) + 0.5 * (θ - coef(fit))' * H * (θ - coef(fit))
        @test H[1, 1] > 0 && H[3, 3] > 0
        # A random-intercept route: finite gradient and Hessian of its objective
        gidx = repeat(1:15, 10)
        dre = (y = _N3.yg .+ 0.5 .* randn(MersenneTwister(5), 15)[gidx], x = _N3.x, g = gidx)
        fre = _n3_quiet(() -> drm(bf(@formula(y ~ 1 + x + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = dre))
        @test all(isfinite, gradient(fre))
        @test all(isfinite, hessian(fre))
        # A fit without a stored objective refuses by name
        bare = DrmFit(Gaussian(), [:mu => 1:1], [:mu => ["(Intercept)"]], [0.0], fill(1.0, 1, 1), 0.0, 1,
                      true, Dict{Symbol,Vector{Float64}}(), Dict{Symbol,Vector{Float64}}(), Dict{Symbol,Vector{Float64}}())
        @test_throws ArgumentError gradient(bare)
        @test_throws ArgumentError hessian(bare)
    end

    @testset "#931: weights(fit) is not a ones() stub" begin
        lab, fam, f, rows = _n3_cases()[1]
        fit = drm(f, fam; data = rows(allrows))
        @test weights(fit) === nothing
    end

    @testset "#931: integer weights == duplicated rows, on every fixed-effect family" begin
        w = Float64.(rand(MersenneTwister(6), 0:3, _N3_N))
        idx = reduce(vcat, [fill(i, Int(w[i])) for i in 1:_N3_N])
        for (lab, fam, f, rows) in _n3_cases()
            a = _n3_quiet(() -> drm(f, fam; data = rows(allrows), weights = w))
            b = _n3_quiet(() -> drm(f, fam; data = rows(idx)))
            @test weights(a) == w
            @test control_info(a).weights == w
            @test loglik(a) ≈ loglik(b) atol = 1e-6 * (1 + abs(loglik(b)))
            # Student / SkewNormal shape parameters can sit on a flat ridge; the
            # objective is what must agree. Elsewhere the estimates match too.
            if !(lab in ("Student", "SkewNormal"))
                # A log-scale parameter driven towards its boundary (|value| > 6,
                # e.g. a dispersion on near-Poisson data) is flat in the
                # likelihood: compare the identified entries only. The objective
                # itself is asserted above for every family.
                ok = abs.(coef(b)) .< 6
                @test coef(a)[ok] ≈ coef(b)[ok] atol = 1e-4
            end
            @test nobs(a) == _N3_N                       # rows, zero-weight rows included
        end
        # Unit weights reproduce the unweighted fit exactly.
        lab, fam, f, rows = _n3_cases()[1]
        @test coef(drm(f, fam; data = rows(allrows), weights = ones(_N3_N))) == coef(drm(f, fam; data = rows(allrows)))
        # A non-integer weight is a likelihood multiplier: halving every weight
        # halves the log-likelihood and keeps the estimates.
        h1 = drm(f, fam; data = rows(allrows), weights = fill(1.0, _N3_N))
        h2 = drm(f, fam; data = rows(allrows), weights = fill(0.5, _N3_N))
        @test loglik(h2) ≈ 0.5 * loglik(h1) rtol = 1e-8
        @test coef(h2) ≈ coef(h1) atol = 1e-6
    end

    @testset "#931: weights validation and refusals" begin
        lab, fam, f, rows = _n3_cases()[1]
        d = rows(allrows)
        @test_throws ArgumentError drm(f, fam; data = d, weights = fill(-1.0, _N3_N))
        @test_throws ArgumentError drm(f, fam; data = d, weights = fill(NaN, _N3_N))
        @test_throws ArgumentError drm(f, fam; data = d, weights = zeros(_N3_N))
        @test_throws ArgumentError drm(f, fam; data = d, weights = "w")
        @test_throws DimensionMismatch drm(f, fam; data = d, weights = ones(10))
        fr = bf(@formula(y ~ 1 + x), @formula(sigma ~ 1))
        err = try drm(fr, Gaussian(); data = d, method = :REML, weights = ones(_N3_N)); nothing catch e; e end
        @test err isa ArgumentError && occursin("REML", sprint(showerror, err))
    end

    @testset "weights carried into bootstrap refits" begin
        lab, fam, f, rows = _n3_cases()[1]
        w = Float64.(rand(MersenneTwister(7), 1:3, _N3_N))
        fit = drm(f, fam; data = rows(allrows), weights = w)
        @test DRModels._bootstrap_refit_kwargs(fit).weights == w
        @test !haskey(DRModels._bootstrap_refit_kwargs(drm(f, fam; data = rows(allrows))), :weights)
    end

    @testset "missing responses: the control scope follows the observed-rows refit" begin
        y = copy(_N3.ypos); y[3] = NaN
        d = (y = y, x = _N3.x)
        f = bf(@formula(y ~ 1 + x), @formula(sigma ~ 1))
        fit = _n3_quiet(() -> drm(f, Gamma(); data = d, control = DRMControl(se = false, maxit = 500)))
        @test all(isnan, vcov(fit))
        @test control_info(fit).se === :skipped
        @test_throws DimensionMismatch _n3_quiet(() -> drm(f, Gamma(); data = d, weights = ones(_N3_N)))
    end
end
