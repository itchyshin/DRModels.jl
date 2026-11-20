# refs #1381 — predict(fit) without data must be a named ArgumentError, not a MethodError.
module TestIssue1381PredictNoData

using DRModels
using Test

@testset "#1381 predict without newdata gives a named refusal" begin
    N = 40
    i = 1:N
    x = sin.(i)
    y = 1.0 .+ 0.5 .* x .+ 0.3 .* sin.(5.1 .* i)
    d = (; y, x)
    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = d)

    for thunk in (() -> predict(fit), () -> predict(fit; type = :response),
                  () -> predict(fit; type = :link))
        err = try thunk(); nothing catch e; e end
        @test err isa ArgumentError
        msg = err isa ArgumentError ? err.msg : ""
        @test occursin("predict(fit, data)", msg)
        @test occursin("fitted(fit)", msg)
    end

    # existing routes are unchanged
    @test predict(fit, d) ≈ fitted(fit)
end

end
