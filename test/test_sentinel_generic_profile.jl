# Sentinel (1e18 wall) objectives must never be read as a profile crossing in the
# generic profile-CI machinery, nor leak into plotted deviances.
using DRModels
using Test, Random

@testset "generic profile: sentinel objective is not a crossing" begin
    cliff(θ) = abs(θ[1]) > 0.3 ? 1e18 : 0.5 * (θ[1] / 0.2)^2 + 0.5 * (θ[2] - 1)^2
    nllhat = cliff([0.0, 1.0])
    half = 1.92
    # true endpoint (0.392) lies beyond the cliff at 0.3: unreachable.
    val, st = DRModels._profile_endpoint_result(
        cliff, nothing, [0.0, 1.0], 1, nllhat, half, 0.2, +1, [1.0], :finite,
    )
    @test !isfinite(val)
    @test st.endpoint_failed
    @test st.nuisance_reason === :sentinel_objective

    @test DRModels._profile_reference_difference(1e18, 0.0).status === :sentinel_objective
    @test isnan(DRModels._profile_plot_deviance(1e18, 0.0, "test", "x"))
    @test DRModels._profile_plot_deviance(3.0, 1.0, "test", "x") == 4.0

    # healthy problem (no cliff) is unchanged: exact endpoint sqrt(2*half)*0.2
    ok(θ) = 0.5 * (θ[1] / 0.2)^2 + 0.5 * (θ[2] - 1)^2
    v, s2 = DRModels._profile_endpoint_result(
        ok, nothing, [0.0, 1.0], 1, 0.0, half, 0.2, +1, [1.0], :finite,
    )
    @test !s2.endpoint_failed
    @test isapprox(v, 0.2 * sqrt(2 * half); atol=1e-6)
end
