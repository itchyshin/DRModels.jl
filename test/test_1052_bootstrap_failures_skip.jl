# #1052: a thrown bootstrap refit is recorded. The public default is
# record-and-continue (`failures = :skip`): successful draws are kept, the result
# carries `bootstrap_incomplete = true`, and a warning names how many draws
# failed. `failures = :error` stays available to abort. `used == 0` still errors
# and names the first failure.
module Test1052BootstrapFailuresSkip

using DRModels
using Test
using Random

@testset "#1052 thrown refit recorded; default is record-and-continue" begin
    rng = MersenneTwister(20261030)
    n = 40
    x = randn(rng, n)
    y = 0.3 .+ 0.5 .* x .+ randn(rng, n)
    form = bf(@formula(y ~ x), @formula(sigma ~ 1))
    data = (; y, x)
    fit0 = drm(form, Gaussian(); data)

    calls = Ref(0)
    function flaky(datab)
        calls[] += 1
        calls[] == 2 && error("forced refit failure")
        return drm(form, Gaussian(); data = datab)
    end

    # Default (no `failures` argument): record, warn with the denominator, continue.
    res = @test_logs (:warn, r"dropping 1 of 4 refits \(3 retained\)") match_mode = :any DRModels._bootstrap_result(
        fit0, form, data, 4, 0.95, MersenneTwister(3), false, flaky)
    @test res.attempted == 4
    @test res.used == 3
    @test res.failed == 1
    @test res.bootstrap_incomplete === true
    @test res.status == "bootstrap_incomplete"
    @test occursin("forced refit failure", res.failures[1].message)
    # Summaries use the completed draws only and stay finite.
    @test size(res.summary, 1) == length(coef(fit0)) || length(res.summary) == length(coef(fit0))
    @test all(r -> isfinite(r.lower) && isfinite(r.upper), res.summary)

    # A complete run is not flagged.
    clean = DRModels._bootstrap_result(
        fit0, form, data, 3, 0.95, MersenneTwister(3), false,
        d -> drm(form, Gaussian(); data = d))
    @test clean.bootstrap_incomplete === false
    @test clean.status == "bootstrap"

    # Public wrapper: failures = :skip is still accepted explicitly.
    calls[] = 0
    res_skip = @test_logs (:warn, r"bootstrap_incomplete") match_mode = :any DRModels._bootstrap_result(
        fit0, form, data, 4, 0.95, MersenneTwister(3), false, flaky; failures = :skip)
    @test res_skip.used == 3

    # Explicit failures = :error aborts and names the first failure.
    calls[] = 0
    err = try
        DRModels._bootstrap_result(
            fit0, form, data, 4, 0.95, MersenneTwister(3), false, flaky; failures = :error)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("bootstrap failed in 1 of 4 replicates", err.msg)
    @test occursin("forced refit failure", err.msg)

    allfail = try
        DRModels._bootstrap_result(fit0, form, data, 2, 0.95, MersenneTwister(3), false,
                                   _ -> error("always fails here"))
        nothing
    catch err
        err
    end
    @test allfail isa ErrorException
    @test occursin("all 2 bootstrap replicates failed", allfail.msg)
    @test occursin("always fails here", allfail.msg)
end

end
