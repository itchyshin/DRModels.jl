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
