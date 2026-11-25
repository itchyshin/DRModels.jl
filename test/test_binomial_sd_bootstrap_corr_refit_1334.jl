# Julia 1.10: correlated Binomial SD bootstrap refits must converge with the
# fixture seed draw used in test_binomial_sd_bootstrap_nq_1334.jl (#1334).
using DRModels, Test, Random, StableRNGs, Distributions

@testset "Binomial SD correlated bootstrap refit stability (#1334)" begin
    rng = StableRNG(20261020)
    G, m = 64, 18
    g = repeat(1:G, inner = m)
    z0_group = repeat([-1.0, 1.0], inner = G ÷ 2)
    z1_group = repeat([0.0, 1.0, 1.0, 0.0], inner = G ÷ 4)
    z0, z1 = z0_group[g], z1_group[g]
    x = repeat(range(-1.2, 1.2; length = m), outer = G)
    alpha0 = [-0.65, 0.55]
    alpha1 = [-0.75, 0.6]
    rho = 0.4
    tau0 = exp.(alpha0[1] .+ alpha0[2] .* z0_group)
    tau1 = exp.(alpha1[1] .+ alpha1[2] .* z1_group)
    u = randn(rng, 2, G)
    b0 = tau0 .* u[1, :]
    b1 = tau1 .* (rho .* u[1, :] .+ sqrt(1 - rho^2) .* u[2, :])
    eta = -0.1 .+ 0.25 .* x .+ b0[g] .+ x .* b1[g]
    p = 1 ./ (1 .+ exp.(-eta))
    y = Float64.(rand.(Ref(rng), Distributions.Bernoulli.(p)))
    data = (; y, x, z0, z1, g)
    slope_only = bf(
        @formula(y ~ x + (1 + x | g)),
        @formula(sd(g, coef(x)) ~ z1),
    )
    fit = DRModels._drm_binomial_sd_regression(slope_only, DRModels.Binomial(); data, nq = 9)
    boot = bootstrap_result(
        fit; data, B = 1, rng = MersenneTwister(30), failures = :error,
    )
    @test boot.used == 1 && boot.failed == 0
end