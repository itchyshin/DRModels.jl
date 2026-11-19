# refs #1382 — Gaussian `offset(...)` with missing values must listwise-drop like Poisson.
module TestIssue1382GaussianOffsetMissing

using DRModels
using Test

@testset "#1382 Gaussian offset missing values" begin
    N = 40
    i = 1:N
    x = sin.(i)
    off = 0.1 .+ mod.(0.618i, 1)
    yg = 1.0 .+ 0.5 .* x .+ off .+ 0.3 .* sin.(5.1 .* i)
    drop_ix = (3, 17)
    keep = [j ∉ drop_ix for j in i]
    d_cc = (; yg = yg[keep], x = x[keep], off = off[keep])
    fit_cc = drm(bf(@formula(yg ~ x + offset(off)), @formula(sigma ~ 1)), Gaussian(); data = d_cc)

    offna = Union{Missing, Float64}[o for o in off]
    for j in drop_ix
        offna[j] = missing
    end
    d = (; yg, x, offna)
    fit = @test_logs (:warn, r"dropped 2 of 40") match_mode = :any drm(
        bf(@formula(yg ~ x + offset(offna)), @formula(sigma ~ 1)), Gaussian(); data = d)
    @test nobs(fit) == 38
    @test length(coef(fit)) >= 2
    @test coef(fit) ≈ coef(fit_cc)
end

@testset "#1382 Gaussian offset missing response" begin
    N = 40
    i = 1:N
    x = sin.(i)
    off = 0.1 .+ mod.(0.618i, 1)
    yg = 1.0 .+ 0.5 .* x .+ off .+ 0.3 .* sin.(5.1 .* i)
    drop_ix = (5, 22)
    keep = [j ∉ drop_ix for j in i]
    d_cc = (; yg = yg[keep], x = x[keep], off = off[keep])
    fit_cc = drm(bf(@formula(yg ~ x + offset(off)), @formula(sigma ~ 1)), Gaussian(); data = d_cc)

    ygm = Union{Missing, Float64}[v for v in yg]
    for j in drop_ix
        ygm[j] = missing
    end
    d = (; yg = ygm, x, off)
    fit = @test_logs (:warn, r"2 of 40 rows have a missing/NaN response") match_mode = :any drm(
        bf(@formula(yg ~ x + offset(off)), @formula(sigma ~ 1)), Gaussian(); data = d)
    @test nobs(fit) == 38
    @test coef(fit) ≈ coef(fit_cc)
end

end