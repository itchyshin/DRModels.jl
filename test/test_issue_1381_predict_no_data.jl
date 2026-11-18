# refs #1381 — `predict(fit)` without newdata must not surface a raw MethodError.
module TestIssue1381PredictNoData

using DRModels
using Test

@testset "#1381 predict(fit) without newdata" begin
    N = 40
    i = 1:N
    x = sin.(i)
    y = 1.0 .+ 0.5 .* x .+ 0.3 .* sin.(5.1 .* i)
    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
    @test predict(fit, (; y, x)) ≈ fitted(fit)
    err = try
        predict(fit)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("predict(fit, data", sprint(showerror, err))
    err2 = try
        predict(fit; type = :response)
        nothing
    catch e
        e
    end
    @test err2 isa ArgumentError
end

end