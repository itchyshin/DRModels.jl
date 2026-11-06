# Marginal/bootstrap simulator for the NB2 animal slope-only SD route:
#   u = sd_group .* (chol(A).L * z),  eta = eta_fixed + x * u[g],  y ~ NB2(exp(eta), sigma)
# on the RAW A (the NB2 fit uses raw A). Candidate-level evidence only.
#
# Latent check. The simulator returns y, not u. Because it draws u first from the
# supplied rng, a copy of the rng replays the same standard normals, so the
# oracle below rebuilds u = D chol(A).L z independently and (i) must reproduce the
# simulator's y exactly when pushed through the same observation draw, and
# (ii) over many fixed-seed replays must have covariance D A D within Monte Carlo
# error. Step (ii) tests the latent formula and the raw-A choice; the link from
# the closure to that formula is step (i).
using DRModels, Test, Random, LinearAlgebra, Statistics
import Distributions

@testset "NB2 animal slope-only SD marginal simulator" begin
    rng0 = MersenneTwister(20261101)
    G, m = 24, 14
    group_index = repeat(1:G, inner = m)
    z_group = repeat([-1.0, 1.0], inner = G ÷ 2)
    R = 0.35 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    s = collect(range(0.7, 1.8; length = G))
    A = Matrix(Diagonal(s) * R * Diagonal(s))          # non-unit diagonal
    x = randn(rng0, G * m)
    alpha = [-0.6, 0.4]
    u = exp.(alpha[1] .+ alpha[2] .* z_group) .* (cholesky(Symmetric(A)).L * randn(rng0, G))
    mu = exp.(0.3 .+ 0.25 .* x .+ x .* u[group_index])
    r = 1 / 0.42^2
    y = Float64[rand(rng0, Distributions.NegativeBinomial(r, r / (r + v))) for v in mu]
    data = (; y, x, z = z_group[group_index], g = group_index)
    formula = bf(@formula(y ~ x + animal(0 + x | g)), @formula(sigma ~ 1),
                 @formula(sd(g, animal) ~ z))
    fit = drm(formula, NegBinomial2(); data, A, se = false)
    @test fit.converged

    sim = DRModels._marginal_simulator_build(fit, data; A)
    @test sim isa Function
    sd_group = exp.(hcat(ones(G), z_group) * coef(fit, :sd))
    @test sd_group ≈ fit.scales[:sd_group]
    L = cholesky(Symmetric(A)).L
    eta_fixed = Vector{Float64}(predict(fit, data; type = :link))

    # (i) exact replay of the simulator's draw.
    for seed in (11, 12, 13)
        rng_a = MersenneTwister(seed)
        rng_b = copy(rng_a)
        actual = sim(rng_a)
        latent = sd_group .* (L * randn(rng_b, G))
        mu_star = exp.(eta_fixed .+ data.x .* latent[group_index])
        @test actual == DRModels._simulate_once(fit, rng_b; mu = mu_star)
    end
    # The raw-A choice is visible: a correlation-rescaled A gives different draws.
    dA = sqrt.(diag(A))
    rng_a = MersenneTwister(21); rng_b = copy(rng_a)
    latent_norm = sd_group .* (cholesky(Symmetric(A ./ (dA * dA'))).L * randn(rng_b, G))
    wrong = DRModels._simulate_once(fit, rng_b;
        mu = exp.(eta_fixed .+ data.x .* latent_norm[group_index]))
    @test sim(rng_a) != wrong

    # (ii) latent covariance D A D within Monte Carlo error (fixed seed).
    Rdraw = 6000
    rng = MersenneTwister(20261102)
    U = zeros(Rdraw, G)
    for k in 1:Rdraw
        rng_b = copy(rng)
        sim(rng)                                   # advance exactly as the simulator does
        U[k, :] = sd_group .* (L * randn(rng_b, G))
    end
    target = Diagonal(sd_group) * A * Diagonal(sd_group)
    emp = (U' * U) / Rdraw                         # known zero mean
    se = sqrt.((target .^ 2 .+ diag(target) * diag(target)') ./ Rdraw)
    @test maximum(abs.(emp .- target) ./ se) < 5.0
    @test norm(emp - target) / norm(target) < 0.08

    # Ainv input reproduces the A-based simulator.
    sim_inv = DRModels._marginal_simulator_build(fit, data; Ainv = inv(Symmetric(A)))
    @test sim_inv(MersenneTwister(5)) == sim(MersenneTwister(5))
    @test_throws ArgumentError DRModels._marginal_simulator_build(fit, data;
        A, Ainv = inv(Symmetric(A)))
    @test_throws ArgumentError DRModels._marginal_simulator_build(fit, data)   # no matrix

    boot = bootstrap_result(fit; data, A, B = 1, rng = MersenneTwister(20261103),
                            failures = :skip)
    @test boot.attempted == boot.used == 1
end
