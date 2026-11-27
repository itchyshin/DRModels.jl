# #1562: nested lrtest on the same response but perturbed predictors must be refused.
using DRModels, StatsModels, Random, Test

function _argmsg(f)
    try
        f()
        error("expected ArgumentError")
    catch err
        err isa ArgumentError || rethrow()
        return err.msg
    end
end

@testset "#1562 lrtest predictor fingerprint guard" begin
    Random.seed!(2020)
    n = 40
    x = randn(n)
    y = 1.0 .+ 0.5 .* x .+ 0.3 .* randn(n)

    reduced = drm(bf(@formula(y ~ x)), Gaussian(); data = (; y, x))
    x_bad = copy(x)
    x_bad[10:15] .= 1_000.0
    z = randn(n)
    full = drm(bf(@formula(y ~ x + z)), Gaussian(); data = (; y, x = x_bad, z))

    msg = _argmsg(() -> lrtest(reduced, full))
    @test occursin("lrtest", msg)
    @test occursin("different predictor", msg) || occursin("design matrix", msg)

    # identical data still compares
    full_ok = drm(bf(@formula(y ~ x + z)), Gaussian(); data = (; y, x, z))
    t = lrtest(reduced, full_ok)
    @test t.dof == 1
    @test t.statistic > 0
end