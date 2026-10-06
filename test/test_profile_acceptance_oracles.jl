using DRModels, Test

# Independent acceptance checks owned by the coordinator, not the implementation
# lane. These retain the original normal-exit precision and exercise the actual
# bracket-collapse path without fitting a statistical model.
@testset "independent profile acceptance oracles" begin
    f(θ) = θ[1]^2/2 + (θ[2]-2θ[1])^2/2
    function grad!(g, θ)
        g[1] = θ[1]-2*(θ[2]-2θ[1])
        g[2] = θ[2]-2θ[1]
    end
    endpoint, stats = DRModels._profile_endpoint_result(
        f, grad!, [0.0,0.0], 1, 0.0, 2.0, 1.0, 1, [0.0], :stored)
    @test !stats.endpoint_failed
    @test abs(endpoint^2/2 - 2.0) < 1e-9
    @test abs(endpoint - 2.0) < 1e-6

    visited = Float64[]
    function steep(θ)
        push!(visited, θ[1])
        return 5e4*θ[1]^2
    end
    endpoint, stats = DRModels._profile_endpoint_result(
        steep, nothing, [0.0], 1, 0.0, 1.92, 1.0, 1, Float64[], :finite)
    @test !stats.endpoint_failed
    # Check membership before any evaluation at the returned coordinate.
    @test endpoint in visited
    @test abs(5e4*endpoint^2 - 1.92) <= 1e-4*1.92
    @test stats.root_iterations >= 20 # actually exercise fine bisection

    _, flat = DRModels._profile_endpoint_result(
        θ -> 0.0, nothing, [0.0], 1, 0.0, 2.0, 1.0, 1, Float64[], :finite)
    @test flat.unbounded
    @test !flat.endpoint_failed
    _, unresolved = DRModels._profile_endpoint_result(
        θ -> 1e16, nothing, [0.0], 1, 1e16, 2.0, 1.0, 1, Float64[], :finite)
    @test unresolved.endpoint_failed
    @test !unresolved.unbounded
    @test unresolved.nuisance_reason == :insufficient_precision

    probe(θ) = eltype(θ) <: DRModels.ForwardDiff.Dual ? throw(InterruptException()) : sum(abs2, θ)
    @test_throws InterruptException DRModels._profile_autodiff_mode(probe, nothing, [0.0,0.0])
end
