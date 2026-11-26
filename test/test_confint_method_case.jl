# Issue #1537 — confint(method) accepts any Symbol casing (drmTMB spells "wald").
using Test
using DRModels, Random

@testset "confint method is case-insensitive (#1537)" begin
    Random.seed!(11261126)
    data = (; y = randn(30), x = randn(30))
    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = data,
              control = DRMControl(se = true))
    lo = confint(fit; method = :wald)
    @test lo == confint(fit; method = :Wald)
    @test confint(fit; method = :WALD) == lo
    hi = confint(fit; method = :profile, parm = :mu => "x")
    @test hi == confint(fit; method = :Profile, parm = :mu => "x")
    err = try confint(fit; method = :bootstrap); nothing catch e; e end
    @test err isa ArgumentError && occursin(":wald or :profile", sprint(showerror, err))
end