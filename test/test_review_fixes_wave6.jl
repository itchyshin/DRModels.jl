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
