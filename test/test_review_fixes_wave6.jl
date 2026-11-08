# Wave-6 adversarial-review fixes (2026-11-08). One testset per fix.
using DRModels, Test, LinearAlgebra, DelimitedFiles, Logging, StableRNGs
import Distributions

const _FX6 = joinpath(@__DIR__, "fixtures", "entry_robust")
_quiet6(f) = Logging.with_logger(f, Logging.NullLogger())

@testset "(a) information criteria and LR guards share loglik()'s failed-fit rule" begin
    g, _ = readdlm(joinpath(_FX6, "gaussian_q2_x1e6_data.csv"), ','; header = true)
    hdr = vec(readdlm(joinpath(_FX6, "gaussian_q2_x1e6_data.csv"), ','; header = true)[2])
    col(n) = g[:, findfirst(==(n), String.(hdr))]
    dq = (; y = Float64.(col("y")), x = Float64.(col("x")), z0 = Float64.(col("z0")),
          z1 = Float64.(col("z1")), g = string.(col("g")))
    fq = _quiet6(() -> drm(bf(@formula(y ~ x + (1 + x | g)), @formula(sigma ~ 1),
                             @formula(sd(g, coef(intercept)) ~ z0), @formula(sd(g, coef(x)) ~ z1)),
                          Gaussian(); data = dq))
    @test fq.converged == false
    @test abs(fq.loglik) >= 1e15                       # the runaway +3.6e76 stored value
    for f in (aic, bic, aicc, deviance)
        @test isnan(_quiet6(() -> f(fq)))              # never a finite, best-looking value
    end
    ok = _quiet6(() -> drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = dq))
    ok2 = _quiet6(() -> drm(bf(@formula(y ~ 1 + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = dq))
    @test_throws ArgumentError lrtest(ok2, fq)
    @test_throws ArgumentError lrt_boundary(fq, ok2)
    @test_throws ArgumentError lrtest(fq, ok2)
end

@testset "(b) unlabelled A / coords warning: once per bootstrap_ci call, every ordinary drm() call" begin
    G, m = 6, 5
    rng = StableRNG(11)
    g = repeat(["a$i" for i in 1:G], inner = m)
    x = randn(rng, G * m)
    y = 1 .+ 0.4 .* x .+ repeat(randn(rng, G), inner = m) .+ 0.5 .* randn(rng, G * m)
    d = (; y, x, g)
    R0 = [0.4^abs(i - j) for i in 1:G, j in 1:G]
    f = bf(@formula(y ~ x + animal(1 | g)), @formula(sigma ~ 1))
    nlab(logs) = count(l -> occursin("has no labels", l.message), logs)
    # ordinary calls: one warning each
    for _ in 1:2
        logs, _ = Test.collect_test_logs(() -> drm(f, Gaussian(); data = d, A = R0))
        @test nlab(logs) == 1
    end
    # bootstrap_ci: the base fit warns once, the B refits stay quiet
    B = 4
    logs, _ = Test.collect_test_logs(() -> bootstrap_ci(f, Gaussian(); data = d, A = R0, B,
                                                      rng = StableRNG(3), failures = :skip, check_converged = false))
    @test nlab(logs) == 1
end

function _refusal6(f, fam, d; kw...)
    try
        drm(f, fam; data = d, kw...)
    catch e
        e isa ArgumentError && return e.msg
        rethrow()
    end
    return ""
end

@testset "(c) the drmTMB note only claims what the refusal-gap map shows" begin
    rng = StableRNG(21)
    n = 60
    g = repeat(["a$i" for i in 1:6], inner = 10)
    x = randn(rng, n)
    y = randn(rng, n)
    d = (; y, x, g)
    f = bf(@formula(y ~ x + (1 | g)))
    msg = _refusal6(f, Student(), d; tree = 1)
    @test occursin("does not accept the keyword `tree`", msg)
    @test !occursin("refuses this model too", msg)              # the old, over-broad claim
    @test occursin("refused all 50", msg)                       # what the grid shows
    @test occursin("nu ~ phylo(1 | id, tree = tree)", msg)      # drmTMB does fit this Student model
    @test occursin("not verified against drmTMB", msg)
    # a family with R fits in the grid says so, with the count
    mlog = _refusal6(bf(@formula(y ~ x + (1 | g))), DRModels.LogNormal(), d; coords = 1)
    @test occursin("refused all 50", mlog)
    @test occursin("fitted 3 of the 50", DRModels._kw_r_note(:tree, DRModels.Tweedie()))
    @test occursin("fitted 2 of the 50", DRModels._kw_r_note(:coords, Student()))
    # a family/keyword outside the grid makes no claim about drmTMB
    @test DRModels._kw_r_note(:tree, Gaussian()) == " Not verified against drmTMB for this model."
end

@testset "(d) se= refusal names control = DRMControl(se = false); accepted lists carry control and weights" begin
    rng = StableRNG(22)
    n = 60
    g = repeat(["a$i" for i in 1:6], inner = 10)
    x = randn(rng, n)
    y = randn(rng, n)
    d = (; y, x, g)
    msg = _refusal6(bf(@formula(y ~ x + (1 | g))), Gaussian(), d; se = false)
    @test occursin("does not accept the keyword `se`", msg)
    @test occursin("control = DRMControl(se = false)", msg)
    @test occursin("works on every route", msg)
    acc = last(split(msg, "Keywords this route accepts: "))
    @test occursin("`control`", acc) && occursin("`weights`", acc)
    # derived from the signatures: a splat route and a fixed-effect route agree
    msg2 = _refusal6(bf(@formula(y ~ x + (1 | g))), Student(), d; tree = 1)
    acc2 = last(split(msg2, "Keywords this route accepts: "))
    @test occursin("`control`", acc2) && occursin("`weights`", acc2)
    # the advice is true: control = DRMControl(se = false) fits where `se = false` was refused
    fit = _quiet6(() -> drm(bf(@formula(y ~ x + (1 | g))), Gaussian(); data = d, control = DRMControl(se = false)))
    @test fit isa DRModels.DrmFit
end
