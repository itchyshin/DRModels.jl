# NB2 sigma random-intercept SD regression: the default HagerZhang line search used to throw
# `AssertionError: B > A` out of `drm()` on a few percent of fuzz draws (about 6 of 360 in the
# refusal-gap fuzz; 3 of 160 in the re-run on Julia 1.12). The fitter now retries with
# BackTracking (`_optimize_with_fallback`). The three datasets are committed as CSV (seeds 9, 94,
# 95 of a StableRNG fuzz generator: G = 8-30 groups of 4-12 rows, sigma random intercept with
# `sd(group, sigma) ~ w`, extreme dispersion / mean offsets), so the test does not depend on any
# random-number stream. Contract tested: never an AssertionError; a DrmFit with an honest
# `converged` flag, or a clear ArgumentError refusal.
using DRModels, Test, DelimitedFiles, Logging

const _NB2LS = joinpath(@__DIR__, "fixtures", "nb2_sigma_sd_line_search", "seeds.csv")

function _nb2ls_data(seed)
    raw, _ = readdlm(_NB2LS, ','; header = true)
    rows = raw[:, 1] .== seed
    return (; y = Float64.(raw[rows, 2]), x = Float64.(raw[rows, 3]),
            w = Float64.(raw[rows, 4]), group = Int.(raw[rows, 5]))
end

@testset "NB2 sigma-SD: line-search assertion never escapes" begin
    f = bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + (1 | group)),
           @formula(sd(group, sigma) ~ w))
    for seed in (9, 94, 95)
        data = _nb2ls_data(seed)
        @test length(data.y) > 30
        res = Logging.with_logger(Logging.NullLogger()) do
            try
                drm(f, NegBinomial2(); data)
            catch e
                e
            end
        end
        @test !(res isa AssertionError)
        @test res isa DRModels.DrmFit || res isa ArgumentError
        if res isa DRModels.DrmFit
            @test res.converged isa Bool
            ll = Logging.with_logger(Logging.NullLogger()) do
                loglik(res)
            end
            @test isnan(ll) || abs(ll) < 1e15
            res.converged && @test isfinite(ll)
        end
    end
end
