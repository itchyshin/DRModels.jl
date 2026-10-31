# #1052: a thrown bootstrap refit is recorded. The public default remains
# `failures = :error` (abort). Pass `failures = :skip` to keep successful
# draws. `used == 0` still errors and names the first failure.
module Test1052BootstrapFailuresSkip

using DRModels
using Test
using Random

@testset "#1052 thrown refit recorded; default stays :error" begin
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

    res = @test_logs (:warn, r"bootstrap_incomplete") match_mode = :any DRModels._bootstrap_result(
        fit0, form, data, 4, 0.95, MersenneTwister(3), false, flaky; failures = :skip)
    @test res.attempted == 4
    @test res.used == 3
    @test res.failed == 1
    @test res.status == "bootstrap_incomplete"
    @test occursin("forced refit failure", res.failures[1].message)

    calls[] = 0
    default_err = try
        DRModels._bootstrap_result(
            fit0, form, data, 4, 0.95, MersenneTwister(3), false, flaky)
        nothing
    catch err
        err
    end
    @test default_err isa ErrorException
    @test occursin("bootstrap failed in 1 of 4 replicates", default_err.msg)
    @test occursin("forced refit failure", default_err.msg)

    calls[] = 0
    @test_throws ErrorException DRModels._bootstrap_result(
        fit0, form, data, 4, 0.95, MersenneTwister(3), false, flaky; failures = :error)

    allfail = try
        DRModels._bootstrap_result(fit0, form, data, 2, 0.95, MersenneTwister(3), false,
                                   _ -> error("always fails here"); failures = :skip)
        nothing
    catch err
        err
    end
    @test allfail isa ErrorException
    @test occursin("all 2 bootstrap replicates failed", allfail.msg)
    @test occursin("always fails here", allfail.msg)
end

end
