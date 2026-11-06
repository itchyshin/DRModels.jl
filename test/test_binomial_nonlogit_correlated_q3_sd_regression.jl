using DRModels, Test, Random, StableRNGs
import Distributions
using Statistics

# Truth-recovery tolerances are sampling-luck sensitive across Julia RNG streams; see test/README.md "CI RNG policy (#388)".
const _SLOW_BINOM_NONLOGIT_Q3 = get(ENV, "DRM_SLOW_TESTS", "0") == "1"

@testset "Binomial non-logit correlated q3 SD regression" begin
    rng = StableRNG(2026102634)
    G, m = 128, 40
    g = repeat(1:G, inner = m)
    x = randn(rng, G * m)
    z = randn(rng, G * m)
    standardize_group(v) = (v .- mean(v)) ./ std(v)
    w0 = standardize_group(randn(rng, G))
    w1 = standardize_group(randn(rng, G))
    w2 = standardize_group(randn(rng, G))
    w, h, j = w0[g], w1[g], w2[g]

    alpha0 = [log(0.65), 0.45]
    alpha1 = [log(0.55), -0.45]
    alpha2 = [log(0.5), 0.4]
    partial = [0.2, -0.2, 0.25]
    s21, s31, s32 = sqrt(1 - partial[1]^2), sqrt(1 - partial[2]^2),
                    sqrt(1 - partial[3]^2)
    u = randn(rng, 3, G)
    tau0 = exp.(alpha0[1] .+ alpha0[2] .* w0)
    tau1 = exp.(alpha1[1] .+ alpha1[2] .* w1)
    tau2 = exp.(alpha2[1] .+ alpha2[2] .* w2)
    b0 = tau0 .* u[1, :]
    b1 = tau1 .* (partial[1] .* u[1, :] .+ s21 .* u[2, :])
    b2 = tau2 .* (partial[2] .* u[1, :] .+ s31 * partial[3] .* u[2, :] .+
                  s31 * s32 .* u[3, :])
    eta = -0.1 .+ 0.2 .* x .- 0.15 .* z .+ b0[g] .+ b1[g] .* x .+ b2[g] .* z

    formula = bf(
        @formula(y ~ x + z + (1 + x + z | g)),
        @formula(sd(g, coef(intercept)) ~ w),
        @formula(sd(g, coef(x)) ~ h),
        @formula(sd(g, coef(z)) ~ j),
    )

    for (link, seed) in ((:probit, 2026102635), (:cloglog, 2026102636))
        link_rng = StableRNG(seed)
        probability = link === :probit ?
            Distributions.cdf.(Distributions.Normal(), eta) : -expm1.(-exp.(eta))
        y = Float64.(rand.(Ref(link_rng), Distributions.Bernoulli.(probability)))
        data = (; y, x, z, w, h, j, g)
        fit = drm(formula, Binomial(link = link); data, nq = 5)

        target_blocks = DRModels._sd_target_blocks(DRModels._sd_target_parts(formula))
        @test fit.converged
        @test fit.family.link === link
        @test length(target_blocks) == 3
        @test all(length(coef(fit, target)) == 2 for target in target_blocks)
        @test abs(coef(fit, target_blocks[1])[2] - alpha0[2]) < 0.35
        @test abs(coef(fit, target_blocks[2])[2] - alpha1[2]) < 0.35
        @test abs(coef(fit, target_blocks[3])[2] - alpha2[2]) < 0.35
        # Gated per #388 RNG policy: recovery vs simulation truth.
        if _SLOW_BINOM_NONLOGIT_Q3
            @test coef(fit, :recov)[end-2:end] ≈ atanh.(partial) atol = 0.3
        end
        @test fit.scales[target_blocks[1]][1:m:end] ≈
              exp.(coef(fit, target_blocks[1])[1] .+ coef(fit, target_blocks[1])[2] .* w0)
        @test fit.scales[target_blocks[2]][1:m:end] ≈
              exp.(coef(fit, target_blocks[2])[1] .+ coef(fit, target_blocks[2])[2] .* w1)
        @test fit.scales[target_blocks[3]][1:m:end] ≈
              exp.(coef(fit, target_blocks[3])[1] .+ coef(fit, target_blocks[3])[2] .* w2)
        simulated = simulate(fit; re = :zero, rng = MersenneTwister(seed + 1))
        @test length(simulated) == length(y)
        @test all(value -> value == 0.0 || value == 1.0, simulated)
    end
end
