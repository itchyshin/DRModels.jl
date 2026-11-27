# #1568: confint(method = :Profile) matches :profile (case-insensitive drmTMB spelling).
using DRModels, Random, Test

@testset "#1568 confint method case insensitivity" begin
    Random.seed!(701)
    y = randn(60)
    fit = drm(bf(@formula(y ~ 1), @formula(sigma ~ 1)), Gaussian(); data = (; y))
    ci_lo = confint(fit; method = :profile)
    ci_hi = confint(fit; method = :Profile)
    @test ci_hi == ci_lo
    ci_w_lo = confint(fit; method = :wald)
    ci_w_hi = confint(fit; method = :Wald)
    @test ci_w_hi == ci_w_lo
    @test_throws ArgumentError confint(fit; method = :Bootstrap)
end
