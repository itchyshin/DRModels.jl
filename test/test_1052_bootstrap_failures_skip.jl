# #1052: a thrown bootstrap refit is recorded under the default, not aborted.
# Decision 18 (11 Oct 2026): default `failures = :skip`, matching drmTMB.
# `failures = :error` still aborts. `used == 0` still errors and names the cause.
module Test1052BootstrapFailuresSkip

using DRModels
using Test
using Random

@testset "#1052 default failures=:skip records a thrown refit" begin
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

    # No `failures` keyword: the public default must record the thrown
    # replicate and return bootstrap_incomplete. On main this still throws.
    res = @test_logs (:warn, r"bootstrap_incomplete") match_mode = :any DRModels._bootstrap_result(
        fit0, form, data, 4, 0.95, MersenneTwister(3), false, flaky)
    @test res.attempted == 4
    @test res.used == 3
    @test res.failed == 1
    @test res.status == "bootstrap_incomplete"
    @test occursin("forced refit failure", res.failures[1].message)

    calls[] = 0
    @test_throws ErrorException DRModels._bootstrap_result(
        fit0, form, data, 4, 0.95, MersenneTwister(3), false, flaky; failures = :error)

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
