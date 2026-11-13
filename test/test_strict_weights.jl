# Prior `weights` give one value per row of `data` as passed to `drm`, on every route (as in drmTMB:
# "`weights` must have one value per row of `data`"). Rows dropped for a missing response or predictor take
# their weights with them. A vector already cut to the kept rows is refused: on main the fixed-effect
# univariate routes accepted it silently, which gives a wrong likelihood when it is misaligned.
#
# One case per route group. For each, with missing responses, missing predictors and both:
#   * full-length weights fit and equal the fit on the hand-cut data and hand-cut weights;
#   * a pre-cut vector and a vector of any other length are refused with the one-weight-per-row message.
# Routes that do not take weights at all (random-effect routes, REML, bivariate) refuse a wrong length
# with the same message, and keep their own refusal for a full-length vector.
using DRModels, Test, Random

const _SW_N = 90
const _SW_MSG = r"`weights` must have one value per row of `data`"

function _sw_data()
    rng = MersenneTwister(20261112)
    n = _SW_N
    x = randn(rng, n)
    yg = 1 .+ 0.5 .* x .+ exp.(-0.3 .+ 0.2 .* x) .* randn(rng, n)
    ypos = exp.(0.2 .+ 0.3 .* x .+ 0.4 .* randn(rng, n))
    ypois = Float64.([rand(rng) < 0.3 ? 0 : 1 + floor(Int, 3rand(rng) + abs(xi)) for xi in x])
    yprop = clamp.(1 ./ (1 .+ exp.(-(0.2 .+ 0.5 .* x .+ 0.6 .* randn(rng, n)))), 0.01, 0.99)
    succ = Float64.([sum(rand(rng) < 1 / (1 + exp(-(0.2 + 0.7xi))) for _ in 1:10) for xi in x])
    yord = [xi + 0.8randn(rng) < -0.5 ? 1.0 : xi + 0.8randn(rng) < 0.6 ? 2.0 : 3.0 for xi in x]
    ytw = [rand(rng) < 0.3 ? 0.0 : yi for yi in ypos]
    yzo = [u < 0.08 ? 0.0 : u > 0.92 ? 1.0 : p for (u, p) in zip(rand(rng, n), yprop)]
    y2 = 0.4 .* x .+ 0.8 .* randn(rng, n)
    g = repeat(string.('a':'i'), inner = 10)
    (; x, yg, ypos, ypois, yprop, succ, fail = 10 .- succ, yord, ytw, yzo, y2, g)
end
const _SW = _sw_data()
const _SW_W = 0.5 .+ rand(MersenneTwister(5), _SW_N)

# (route group, label, family, formula, columns, response columns that get a missing value)
const _SW_CASES = [
    ("Gaussian location-scale", "Gaussian", Gaussian(), bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x)), (y = _SW.yg,), [:y]),
    ("continuous positive", "LogNormal", LogNormal(), bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x)), (y = _SW.ypos,), [:y]),
    ("continuous positive", "Gamma", Gamma(), bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), (y = _SW.ypos,), [:y]),
    ("continuous positive", "SkewNormal", SkewNormal(), bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), (y = _SW.ypos,), [:y]),
    ("Student", "Student", Student(), bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), (y = _SW.yg,), [:y]),
    ("count", "Poisson", Poisson(), bf(@formula(y ~ 1 + x)), (y = _SW.ypois,), [:y]),
    ("count", "Poisson zi", Poisson(), bf(@formula(y ~ 1 + x), @formula(zi ~ 1)), (y = _SW.ypois,), [:y]),
    ("count", "NegBinomial2", NegBinomial2(), bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), (y = _SW.ypois,), [:y]),
    ("count", "TruncatedPoisson", TruncatedPoisson(), bf(@formula(y ~ 1 + x)), (y = max.(_SW.ypois, 1.0),), [:y]),
    ("bounded 0 to 1", "Beta", Beta(), bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), (y = _SW.yprop,), [:y]),
    ("bounded 0 to 1", "ZeroOneBeta", ZeroOneBeta(), bf(@formula(y ~ 1 + x), @formula(sigma ~ 1), @formula(zoi ~ 1), @formula(coi ~ 1)), (y = _SW.yzo,), [:y]),
    ("binomial", "Binomial", Binomial(), bf(@formula(cbind(s, f) ~ 1 + x)), (s = _SW.succ, f = _SW.fail), [:s, :f]),
    ("binomial", "BetaBinomial", BetaBinomial(), bf(@formula(cbind(s, f) ~ 1 + x), @formula(sigma ~ 1)), (s = _SW.succ, f = _SW.fail), [:s, :f]),
    ("other", "Tweedie", Tweedie(), bf(@formula(y ~ 1 + x), @formula(sigma ~ 1), @formula(nu ~ 1)), (y = _SW.ytw,), [:y]),
    ("other", "CumulativeLogit", CumulativeLogit(), bf(@formula(y ~ x)), (y = _SW.yord,), [:y]),
]

function _sw_build(cols, blank, scenario)
    d = Dict{Symbol,Any}(pairs(cols)); d[:x] = copy(_SW.x)
    keep = trues(_SW_N)
    rrows = scenario === :resp ? [3, 40, 77] : scenario === :both ? [3, 40] : Int[]
    prows = scenario === :pred ? [7, 50, 80] : scenario === :both ? [7, 50] : Int[]
    for c in blank, i in rrows
        d[c] = copy(d[c]); d[c][i] = NaN
    end
    keep[rrows] .= false
    if !isempty(prows)
        xm = Vector{Union{Missing,Float64}}(_SW.x); xm[prows] .= missing
        d[:x] = xm; keep[prows] .= false
    end
    return (; d...), keep
end

_sw_quiet(f) = redirect_stderr(devnull) do; f(); end
_sw_cut(d, keep) = NamedTuple{keys(d)}(map(c -> c[keep], values(d)))
function _sw_refused(thunk)
    err = try; _sw_quiet(thunk); nothing; catch e; e; end
    return err isa DimensionMismatch && occursin(_SW_MSG, err.msg)
end

@testset "strict weights: one weight per data row, every route group" begin
    for (group, lab, fam, f, cols, blank) in _SW_CASES
        @testset "$group / $lab" begin
            for scenario in (:resp, :pred, :both)
                d, keep = _sw_build(cols, blank, scenario)
                nk = count(keep)
                # full-length weights: the dropped rows take their weights with them
                fit = _sw_quiet(() -> drm(f, fam; data = d, weights = _SW_W))
                ref = _sw_quiet(() -> drm(f, fam; data = _sw_cut(d, keep), weights = _SW_W[keep]))
                @test nobs(fit) == nk
                @test isapprox(loglik(fit), loglik(ref); rtol = 1e-9)
                @test weights(fit) == _SW_W[keep]
                # a vector already cut to the kept rows, and vectors of other lengths, are refused
                @test _sw_refused(() -> drm(f, fam; data = d, weights = _SW_W[keep]))
                @test _sw_refused(() -> drm(f, fam; data = d, weights = _SW_W[1:(nk - 1)]))
                @test _sw_refused(() -> drm(f, fam; data = d, weights = vcat(_SW_W, 1.0)))
            end
            # no missing value: a vector of the wrong length is refused with the same message
            d, _ = _sw_build(cols, blank, :none)
            @test _sw_refused(() -> drm(f, fam; data = d, weights = _SW_W[1:(_SW_N - 3)]))
            @test loglik(_sw_quiet(() -> drm(f, fam; data = d, weights = _SW_W))) isa Real
        end
    end

    @testset "routes that take no weights refuse a wrong length with the same message" begin
        d, keep = _sw_build((y = _SW.yg, g = _SW.g), [:y], :resp)
        fre = bf(@formula(y ~ 1 + x + (1 | g)), @formula(sigma ~ 1))
        @test _sw_refused(() -> drm(fre, Gaussian(); data = d, weights = _SW_W[keep]))
        @test_throws ArgumentError _sw_quiet(() -> drm(fre, Gaussian(); data = d, weights = _SW_W))   # its own refusal
        fr = bf(@formula(y ~ 1 + x), @formula(sigma ~ 1))
        @test _sw_refused(() -> drm(fr, Gaussian(); data = d, method = :REML, weights = _SW_W[keep]))
        fb = bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x), sigma1 = @formula(sigma1 ~ 1),
                sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1))
        db, keepb = _sw_build((y1 = _SW.yg, y2 = _SW.y2), [:y1], :resp)
        @test _sw_refused(() -> drm(fb, Gaussian(); data = db, weights = _SW_W[keepb]))
    end

    @testset "the refusal comes before any fitting and names the counts" begin
        d, keep = _sw_build((y = _SW.yg,), [:y], :resp)
        f = bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x))
        err = try; drm(f, Gaussian(); data = d, weights = _SW_W[keep]); nothing; catch e; e; end
        @test err isa DimensionMismatch
        @test occursin("87 values", err.msg) && occursin("90 rows", err.msg)
        # `weights` of a type that is not a vector keeps its own refusal
        @test_throws ArgumentError drm(f, Gaussian(); data = d, weights = "w")
    end
end

# Post-fit routes that refit or re-evaluate with the stored weights. `weights(fit)` is the vector cut to the
# kept rows. The bootstrap refit gets the user's data with every row, so it must forward the weights as the
# user passed them (`control_info(fit).input_weights`). Before this was fixed, every replicate of
# `bootstrap_summary` and `bootstrap_ci` on a weighted fit with a missing response failed with the
# one-weight-per-row refusal.
# Bootstrap of a fit with a missing response works on main for these routes (Student and Binomial fail
# on main too, with an error that has nothing to do with weights, so they are not tested here).
const _SW_BOOT_LABELS = ("Gaussian", "LogNormal", "Gamma", "Poisson", "NegBinomial2", "Beta")
const _SW_LISTWISE_LABELS = ("Gaussian", "LogNormal", "Gamma", "Student", "Poisson", "NegBinomial2", "Beta", "Binomial")

@testset "strict weights: post-fit refits on a weighted fit with missing values" begin
    @testset "$lab: missing response, bootstrap_summary and bootstrap_ci" for lab in _SW_BOOT_LABELS
        (group, _, fam, f, cols, blank) = only(c for c in _SW_CASES if c[2] == lab)
        d, keep = _sw_build(cols, blank, :resp)
        fit = _sw_quiet(() -> drm(f, fam; data = d, weights = _SW_W))
        @test weights(fit) == _SW_W[keep]                       # the kept rows
        @test control_info(fit).input_weights == _SW_W          # as passed
        s = _sw_quiet(() -> bootstrap_summary(fit; data = d, B = 4, rng = MersenneTwister(11)))
        @test !isempty(s)
        ci = _sw_quiet(() -> bootstrap_ci(fit; data = d, B = 4, rng = MersenneTwister(11)))
        @test !isempty(ci)
    end

    @testset "$lab: missing predictor, bootstrap on data cut by drm_listwise" for lab in _SW_LISTWISE_LABELS
        # The data are already cut to the kept rows, so the refit gets the cut weights (`weights(fit)`).
        (group, _, fam, f, cols, blank) = only(c for c in _SW_CASES if c[2] == lab)
        d, keep = _sw_build(cols, blank, :pred)
        fit = _sw_quiet(() -> drm(f, fam; data = d, weights = _SW_W))
        @test weights(fit) == _SW_W[keep]
        dl = _sw_cut(d, keep)
        dl = merge(dl, (x = Float64.(dl.x),))
        s = _sw_quiet(() -> bootstrap_summary(fit; data = dl, B = 4, rng = MersenneTwister(11)))
        @test !isempty(s)
    end

    # The two testsets above only check that a bootstrap returns something. They would still pass if the
    # refits silently dropped the weights (the replicates would then be unweighted fits). This pins what the
    # bootstrap passes to every refit: the weights, and the right vector for the data the refit gets.
    @testset "$lab: bootstrap refits are weighted, with the vector that fits the refit data" for lab in _SW_BOOT_LABELS
        (group, _, fam, f, cols, blank) = only(c for c in _SW_CASES if c[2] == lab)
        d, keep = _sw_build(cols, blank, :resp)
        fit = _sw_quiet(() -> drm(f, fam; data = d, weights = _SW_W))
        kw = DRModels._bootstrap_refit_kwargs(fit; data = d)        # data with every row: the input weights
        @test haskey(kw, :weights)
        @test kw.weights == _SW_W
        unweighted = _sw_quiet(() -> drm(f, fam; data = d))
        @test !haskey(DRModels._bootstrap_refit_kwargs(unweighted; data = d), :weights)
        dp, keepp = _sw_build(cols, blank, :pred)
        fitp = _sw_quiet(() -> drm(f, fam; data = dp, weights = _SW_W))
        dl = _sw_cut(dp, keepp); dl = merge(dl, (x = Float64.(dl.x),))
        kwl = DRModels._bootstrap_refit_kwargs(fitp; data = dl)     # data already cut to the kept rows: the cut weights
        @test haskey(kwl, :weights)
        @test kwl.weights == _SW_W[keepp]
    end

    @testset "no missing value: the two weight vectors are the same" begin
        d, _ = _sw_build((y = _SW.yg,), [:y], :none)
        fit = _sw_quiet(() -> drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x)), Gaussian(); data = d, weights = _SW_W))
        @test weights(fit) == control_info(fit).input_weights == _SW_W
    end

    @testset "update and profile" begin
        d, keep = _sw_build((y = _SW.yg,), [:y], :resp)
        f = bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x))
        fit = _sw_quiet(() -> drm(f, Gaussian(); data = d, weights = _SW_W))
        # update with the weights as passed works; with `weights(fit)` (cut) it is refused like `drm`
        up = _sw_quiet(() -> update(fit, f; data = d, weights = _SW_W))
        @test loglik(up) == loglik(fit)
        @test _sw_refused(() -> update(fit, f; data = d, weights = weights(fit)))
        # the profile uses the stored objective, which holds the kept-row weights
        pr = _sw_quiet(() -> profile_result(fit))
        @test !isempty(pr.ci)
    end
end
