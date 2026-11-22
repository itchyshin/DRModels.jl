# refs #1351 — named formula failures instead of raw Julia errors.
module TestIssue1351FormulaHints

using DRModels
using Test

@testset "#1351 R colon interaction spelling" begin
    x1 = [2sin(i) for i in 1:60]
    x2 = [cos(3i) for i in 1:60]
    y = [1 + x1[i] - x2[i] + 0.3sin(5i) for i in 1:60]
    d = (; y, x1, x2)
    err = try
        drm(bf(@formula(y ~ x1 + x2 + x1:x2), @formula(sigma ~ 1)), Gaussian(); data = d)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    msg = sprint(showerror, err)
    @test occursin("x1:x2", msg)
    @test occursin("x1 & x2", msg) || occursin("x1 * x2", msg)
    fit = drm(bf(@formula(y ~ x1 + x2 + x1 & x2), @formula(sigma ~ 1)), Gaussian(); data = d)
    @test loglikelihood(fit) ≈ loglikelihood(
        drm(bf(@formula(y ~ x1 * x2), @formula(sigma ~ 1)), Gaussian(); data = d)) rtol = 1e-10
end

end