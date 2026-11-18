# Four-member correlated Binomial SD regression. This is a focused synthetic
# parameter-recovery check for the generic partial-correlation parameterization;
# it is not interval-coverage validation.
using DRModels, Test, Random, StableRNGs
import Distributions
using LinearAlgebra
using Statistics

# Truth-recovery tolerances are sampling-luck sensitive across Julia RNG streams; see test/README.md "CI RNG policy (#388)".
const _SLOW_BINOM_Q4 = get(ENV, "DRM_SLOW_TESTS", "0") == "1"

@testset "Binomial correlated q4 SD regression" begin
    rng = StableRNG(2026102741)
    G, m = 256, 32
    g = repeat(1:G, inner = m)
    x = randn(rng, G * m)
    z = randn(rng, G * m)
    v = randn(rng, G * m)
    W = randn(rng, G, 4)
    W .-= mean(W; dims = 1)
    W ./= std(W; dims = 1)
    w1, w2, w3, w4 = (W[:, j][g] for j in 1:4)

    beta = [-0.2, 0.35, -0.25, 0.15]
    alpha = ([log(0.55), 0.6], [log(0.5), -0.55],
             [log(0.45), 0.5], [log(0.5), -0.5])
    tau = hcat((exp.(a[1] .+ a[2] .* W[:, j]) for (j, a) in enumerate(alpha))...)
    partial = [0.45, -0.42, 0.46, 0.4, -0.44, 0.41]
    eta_partial = atanh.(partial)

    # Independent construction of the unit-lower Cholesky factor in row-major
    # partial-correlation order: (2,1), (3,1), (3,2|1), (4,1), (4,2|1),
    # (4,3|1,2).
    L = zeros(4, 4)
    L[1, 1] = 1
    position = 1
    for i in 2:4
        product = 1.0
        for j in 1:(i - 1)
            p = tanh(eta_partial[position])
            L[i, j] = p * product
            product *= sqrt(1 - p^2)
            position += 1
        end
        L[i, i] = product
    end
    @test position == 7
    @test diag(L * L') ≈ ones(4)
    @test DRModels._binomial_partial_correlation_cholesky(eta_partial, 4) ≈ L
    expected_correlations = begin
        R = L * L'
        [R[i, j] for i in 1:3 for j in (i + 1):4]
    end
    @test DRModels._binomial_pairwise_correlations(eta_partial, 4) ≈ expected_correlations
    eta_q2 = [atanh(0.3)]
    eta_q3 = atanh.([0.3, -0.2, 0.4])
    @test DRModels._binomial_partial_correlation_cholesky(eta_q2, 2) ≈
          [1.0 0.0; 0.3 sqrt(1 - 0.3^2)]
    @test DRModels._binomial_partial_correlation_cholesky(eta_q3, 3) ≈
          [1.0 0.0 0.0;
           0.3 sqrt(1 - 0.3^2) 0.0;
           -0.2 sqrt(1 - 0.2^2) * 0.4 sqrt(1 - 0.2^2) * sqrt(1 - 0.4^2)]
    @test DRModels._binomial_partial_correlation_names(:g, 2) == ["g:atanh_rho"]
    @test DRModels._binomial_partial_correlation_names(:g, 3) == [
        "g:atanh_partial_rho_21",
        "g:atanh_partial_rho_31",
        "g:atanh_partial_rho_32_given_1",
    ]
    @test DRModels._binomial_pairwise_correlations(eta_q2, 2) ≈ [0.3]
    @test DRModels._binomial_pairwise_correlations(eta_q3, 3) ≈ [
        0.3, -0.2, 0.3 * -0.2 + sqrt(1 - 0.3^2) * sqrt(1 - 0.2^2) * 0.4,
    ]

    innovations = randn(rng, 4, G)
    b = (tau .* (L * innovations)')'
    eta = beta[1] .+ beta[2] .* x .+ beta[3] .* z .+ beta[4] .* v .+
          b[1, g] .+ b[2, g] .* x .+ b[3, g] .* z .+ b[4, g] .* v
    probability = 1 ./ (1 .+ exp.(-eta))
    y = Float64.(rand.(Ref(rng), Distributions.Bernoulli.(probability)))
    data = (; y, x, z, v, w1, w2, w3, w4, g)
    f = bf(
        @formula(y ~ x + z + v + (1 + x + z + v | g)),
        @formula(sd(g, coef(intercept)) ~ w1),
        @formula(sd(g, coef(x)) ~ w2),
        @formula(sd(g, coef(z)) ~ w3),
        @formula(sd(g, coef(v)) ~ w4),
    )

    # Low-node fit checks q=4 parameter recovery and public outputs. It is not
    # an interval-coverage calibration.
    fit = drm(f, Binomial(); data, nq = 3, g_tol = 1e-5)
    targets = DRModels._sd_target_blocks(DRModels._sd_target_parts(f))
    @test fit.converged
    @test -1.0e6 < fit.loglik < 0.0
    @test fit.scales[:aghq_nodes] == [3.0]
    @test length(targets) == 4
    @test all(length(coef(fit, target)) == 2 for target in targets)
    @test length(coef(fit, :recov)) == 6
    @test length(fit.scales[:corpars]) == 6
    @test all(isfinite, fit.scales[:corpars])
    @test all(length(fit.scales[target]) == length(y) for target in targets)
    for (j, target) in enumerate(targets)
        estimated = coef(fit, target)
        @test all(isfinite, estimated)
        @test sign(estimated[2]) == sign(alpha[j][2])
        @test abs(estimated[2] - alpha[j][2]) < 0.35
        @test fit.scales[target][1:m:end] ≈ exp.(estimated[1] .+ estimated[2] .* W[:, j])
    end
    estimated_partial = coef(fit, :recov)
    @test all(sign.(estimated_partial) .== sign.(eta_partial))
    # Gated per #388 RNG policy: recovery vs simulation truth.
    if _SLOW_BINOM_Q4
        @test maximum(abs.(estimated_partial .- eta_partial)) < 0.35
    end
    @test fit.scales[:corpars] ≈
          DRModels._binomial_pairwise_correlations(estimated_partial, 4)
    recov_names = DRModels._block_coefnames(fit, :recov)
    @test recov_names == [
        "g:atanh_partial_rho_21",
        "g:atanh_partial_rho_31",
        "g:atanh_partial_rho_32_given_1",
        "g:atanh_partial_rho_41",
        "g:atanh_partial_rho_42_given_1",
        "g:atanh_partial_rho_43_given_1_2",
    ]
    simulated = simulate(fit; re = :zero, rng = MersenneTwister(2026102742))
    @test length(simulated) == length(y)
    @test all(value -> value == 0.0 || value == 1.0, simulated)
end

@testset "correlation parameter clamp (_binomial_bounded_partial)" begin
    # Identity, bit for bit, inside |eta| <= atanh(0.999999); finite and monotone beyond it, never past tanh(10.254).
    edge = atanh(DRModels._Q2_RHO_BOUND)
    for eta in (-edge, -3.3, -0.2, 0.0, 0.7, 5.0, edge)
        @test DRModels._binomial_bounded_partial(eta) == (tanh(eta), DRModels._binomial_stable_sech(eta))
    end
    etas = collect(range(edge, 40.0; length = 200))
    rho = first.(DRModels._binomial_bounded_partial.(etas))
    sech = last.(DRModels._binomial_bounded_partial.(etas))
    @test all(isfinite, rho) && all(isfinite, sech) && all(>(0), sech)
    @test issorted(rho) && issorted(sech; rev = true)
    @test maximum(rho) < 1 - 2e-9
    @test DRModels._binomial_bounded_partial(-13.0) == (-DRModels._binomial_bounded_partial(13.0)[1], DRModels._binomial_bounded_partial(13.0)[2])
end
