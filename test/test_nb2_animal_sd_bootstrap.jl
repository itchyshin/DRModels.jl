using DRModels, Test, Random, LinearAlgebra
import Distributions

@testset "NB2 animal SD bootstrap accepts sparse precision input" begin
    rng = MersenneTwister(2026102612)
    G, m = 12, 12
    g = repeat(1:G, inner = m)
    z_group = repeat([-1.0, 1.0], inner = G ÷ 2)
    z = z_group[g]
    x = randn(rng, length(g))
    rho = 0.35
    A = [rho^abs(i - j) for i in 1:G, j in 1:G]
    A_factor = cholesky(Symmetric(A))
    b = exp.(-0.8 .+ 0.3 .* z_group) .* (A_factor.L * randn(rng, G))
    mu = exp.(0.25 .+ 0.2 .* x .+ b[g])
    sigma = 0.42
    size = inv(sigma^2)
    y = Float64[rand(rng, Distributions.NegativeBinomial(size, size / (size + v))) for v in mu]
    data = (; y, x, z, g)
    Q = inv(A)
    formula = bf(@formula(y ~ x + animal(1 | g)), @formula(sigma ~ 1),
                 @formula(sd(g, animal) ~ z))

    fit = drm(formula, NegBinomial2(); data, Ainv = Q, se = false)
    @test fit.converged

    sim = DRModels._marginal_simulator_build(fit, data; Ainv = Q)
    @test sim isa Function
    simulated = sim(MersenneTwister(2026102613))
    @test length(simulated) == length(y)
    @test all(value -> isfinite(value) && value >= 0 && isinteger(value), simulated)

    boot = bootstrap_result(fit; data, Ainv = Q, B = 1,
                            rng = MersenneTwister(2026102614), failures = :skip)
    @test boot.attempted == 1
    @test boot.used == 1
    @test boot.failed == 0

    # Profiling a selected SD regression coefficient should reuse the fitted
    # precision objective and retain the exact named target.
    profile = profile_result(fit; level = 0.80, parm = :sd => "z")
    @test profile.failed == 0
    row = only(profile.ci)
    @test row.param == :sd
    @test row.coef == "z"
    @test isfinite(row.lower)
    @test isfinite(row.upper)
end
