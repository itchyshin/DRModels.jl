# FD-Hessian fallback: a probe whose gradient evaluation fails (the sparse LSS
# engines return an EMPTY gradient on a non-finite nll/gradient) must give
# ok = false -> NaN vcov, never a DimensionMismatch. Tests the shared helper
# directly: the double failure (probe AND retry) is not reachable from public
# inputs on a small fixture.
using Test
using DRModels

@testset "_fd_hessian_from_grad fallback" begin
    fd = DRModels._fd_hessian_from_grad
    A = [2.0 0.3; 0.3 1.0]
    θ̂ = [0.5, -0.2]
    quad(θ) = A * θ

    # healthy gradient: recovers A
    H, ok = fd(quad, θ̂)
    @test ok
    @test H ≈ A atol = 1e-5

    # first probe fails, retry succeeds -> still ok
    calls = Ref(0)
    flaky(θ) = (calls[] += 1; calls[] <= 2 ? Float64[] : quad(θ))
    H, ok = fd(flaky, θ̂)
    @test ok
    @test H ≈ A atol = 1e-5

    # probe and retry both return an empty gradient -> flagged, no throw
    H, ok = fd(θ -> Float64[], θ̂)
    @test !ok

    # non-finite or wrong-length gradient is also flagged
    @test !fd(θ -> fill(NaN, 2), θ̂)[2]
    @test !fd(θ -> zeros(3), θ̂)[2]

    # only the second parameter's probes fail
    bad_second(θ) = abs(θ[2] - θ̂[2]) > 0 ? Float64[] : quad(θ)
    @test !fd(bad_second, θ̂)[2]
end
