# Sentinel guard for the heritability :profile CI (`_profile_side`, `nll_at_ratio`).
# A failed profile evaluation (1e18 sentinel / NaN / non-converged) must never be read
# as "above the LRT target": the arm is flagged unresolved (NaN), not a silent cliff.
using DRModels, Test, Random

@testset "heritability profile: sentinel is not a CI bound" begin
    nll(θ) = abs(θ[1]) > 0.3 ? 1e18 : 0.5 * (θ[1] / 0.2)^2 + 0.5 * (θ[2] - 1)^2
    θ̂ = [0.0, 1.0]
    target = nll(θ̂) + 1.92
    b = @test_logs (:warn, r"unresolved") match_mode = :any DRModels._profile_side(
        v -> nll([3v, 1.0]), 0.0, target, +1)
    @test isnan(b)                       # cliff at 0.1 is NOT returned as endpoint
    # a reachable true crossing is still found
    nll2(θ) = 0.5 * (θ[1] / 0.2)^2
    b2 = DRModels._profile_side(v -> nll2([v]), 0.0, 1.92, +1)
    @test b2 ≈ 0.2 * sqrt(2 * 1.92) atol = 1e-5
    # NaN evaluation is also unresolved
    @test isnan(DRModels._profile_side(v -> v > 0.2 ? NaN : 0.0, 0.0, 1.0, +1))
end

@testset "heritability profile: ordinary CI unchanged" begin
    Random.seed!(42)
    G = 25; m = 6; nn = G * m
    g = repeat(1:G, inner = m)
    b1 = 0.9 .* randn(G)
    y = 1.0 .+ b1[g] .+ 0.5 .* randn(nn)
    fit = drm(bf(@formula(y ~ 1 + (1 | g))), Gaussian(); data = (; y, g))
    hp = heritability(fit; method = :profile)
    # Reference values captured from the pre-fix code.
    @test hp.ci.lower ≈ 0.6622079035790497 atol = 1e-8
    @test hp.ci.upper ≈ 0.8761805598290496 atol = 1e-8
end
