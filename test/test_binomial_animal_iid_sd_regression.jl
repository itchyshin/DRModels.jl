using DRModels, Test, Random, LinearAlgebra
import Distributions

function dense_binomial_animal_iid_laplace(y, X, animal_index, iid_index,
                                           A, animal_sd, iid_sd, fit)
    n, G, H = length(y), length(animal_sd), length(iid_sd)
    M = zeros(n, G + H)
    for i in 1:n
        M[i, animal_index[i]] = animal_sd[animal_index[i]]
        M[i, G + iid_index[i]] = iid_sd[iid_index[i]]
    end
    Q = zeros(G + H, G + H)
    Q[1:G, 1:G] .= inv(A)
    Q[(G + 1):end, (G + 1):end] .= Matrix{Float64}(I, H, H)
    eta_fixed = X * coef(fit, :mu)
    u = zeros(G + H)
    function joint(uval)
        eta = eta_fixed + M * uval
        return 0.5 * dot(uval, Q * uval) +
            sum(max(eta[i], 0.0) + log1p(exp(-abs(eta[i]))) - y[i] * eta[i]
                for i in eachindex(y))
    end
    for _ in 1:100
        eta = eta_fixed + M * u
        p = 1.0 ./ (1.0 .+ exp.(-eta))
        grad = Q * u + M' * (p - y)
        Hjoint = Q + M' * Diagonal(p .* (1 .- p)) * M
        ch = cholesky(Symmetric(Hjoint))
        step = ch \ grad
        norm(step, Inf) < 1e-9 && break
        old = joint(u)
        rate = 1.0
        while rate > 1e-8
            trial = u .- rate .* step
            if joint(trial) <= old
                u = trial
                break
            end
            rate *= 0.5
        end
        rate > 1e-8 || error("independent animal-plus-IID Laplace mode did not converge")
    end
    eta = eta_fixed + M * u
    p = 1.0 ./ (1.0 .+ exp.(-eta))
    Hjoint = Q + M' * Diagonal(p .* (1 .- p)) * M
    return -(joint(u) - 0.5 * logdet(Symmetric(Q)) +
             0.5 * logdet(cholesky(Symmetric(Hjoint))))
end

@testset "Binomial animal and IID effects have separate SD regressions" begin
    rng = MersenneTwister(20261024)
    G, H, m = 36, 30, 4
    animal_index = repeat(1:G, inner = H * m)
    site_index = repeat(repeat(1:H, inner = m), outer = G)
    animal_id = ["animal_$(i)" for i in animal_index]
    site = ["site_$(i)" for i in site_index]
    zlevels = collect(range(-1.0, 1.0; length = G))
    wlevels = collect(range(-1.0, 1.0; length = H))
    z, w = zlevels[animal_index], wlevels[site_index]
    x = randn(rng, length(animal_index))
    correlation = [0.45^abs(i - j) for i in 1:G, j in 1:G]
    animal_scale = randperm(rng, G) ./ G .+ 0.5
    A = Diagonal(animal_scale) * correlation * Diagonal(animal_scale)
    animal_u = cholesky(Symmetric(A)).L * randn(rng, G)
    site_u = randn(rng, H)
    alpha_animal = [-0.75, 0.50]
    alpha_site = [-0.35, -0.60]
    eta = -0.2 .+ 0.3 .* x .+
        exp.(alpha_animal[1] .+ alpha_animal[2] .* zlevels)[animal_index] .* animal_u[animal_index] .+
        exp.(alpha_site[1] .+ alpha_site[2] .* wlevels)[site_index] .* site_u[site_index]
    y = Float64[rand(rng, Distributions.Bernoulli(1 / (1 + exp(-v)))) for v in eta]
    data = (; y, x, animal_id, site, z, w)
    formula = bf(
        @formula(y ~ x + animal(1 | animal_id) + (1 | site)),
        @formula(sd(animal_id, animal) ~ z),
        @formula(sd(site) ~ w),
    )

    fit = drm(formula, Binomial(); data, A, se = false)
    precision_fit = drm(formula, Binomial(); data, Ainv = inv(A), se = false)
    targets = DRModels._sd_target_parts(formula)
    blocks = DRModels._sd_target_blocks(targets)

    @test fit.converged
    @test precision_fit.converged
    @test Set(first.(fit.blocks)) == Set([:mu; blocks])
    @test precision_fit.loglik ≈ fit.loglik atol = 2e-5
    @test coef(precision_fit, blocks[1]) ≈ coef(fit, blocks[1]) atol = 2e-3
    @test coef(precision_fit, blocks[2]) ≈ coef(fit, blocks[2]) atol = 2e-3
    @test coef(fit, blocks[1])[2] > 0
    @test coef(fit, blocks[2])[2] < 0
    pred = predict_parameters(fit, data; se = false)
    @test all(isfinite, pred[blocks[1]])
    @test all(isfinite, pred[blocks[2]])
    @test length(pred[blocks[1]]) == length(y)
    @test length(pred[blocks[2]]) == length(y)
    beta_animal = coef(fit, blocks[1])
    beta_site = coef(fit, blocks[2])
    animal_sd = exp.(beta_animal[1] .+ beta_animal[2] .* zlevels)
    site_sd = exp.(beta_site[1] .+ beta_site[2] .* wlevels)
    dense_loglik = dense_binomial_animal_iid_laplace(y,
        hcat(ones(length(y)), x), animal_index, site_index, A, animal_sd, site_sd, fit)
    @test fit.loglik ≈ dense_loglik atol = 2e-5
    marginal_simulator = DRModels._marginal_simulator(fit, data; A)
    simulated = marginal_simulator(MersenneTwister(20261025))
    @test length(simulated) == length(y)
    @test all(in((0.0, 1.0)), simulated)
    precision_simulator = DRModels._marginal_simulator(precision_fit, data; Ainv = inv(A))
    precision_simulated = precision_simulator(MersenneTwister(20261026))
    @test length(precision_simulated) == length(y)
    @test all(in((0.0, 1.0)), precision_simulated)
    boot = bootstrap_result(fit; data, A, B = 1,
        rng = MersenneTwister(20261027), failures = :error)
    @test boot.attempted == boot.used == 1
    @test boot.failed == 0
    @test Set(row.param for row in boot.summary) == Set([:mu; blocks])
    profile_target = only(filter(profile_targets(fit)) do target
        target.param === blocks[1] && target.parm == "z"
    end)
    @test profile_target.profile_ready
end
