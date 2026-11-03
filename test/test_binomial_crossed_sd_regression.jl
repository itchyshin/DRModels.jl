# Crossed Binomial intercepts with separate group-level SD regressions.
using DRModels, Test, Random, LinearAlgebra
import Distributions

function crossed_binomial_sd_laplace_oracle(y, X, gidx, hidx, tau_g, tau_h)
    G, H = length(tau_g), length(tau_h)
    A = zeros(length(y), G + H)
    for i in eachindex(y)
        A[i, gidx[i]] = tau_g[gidx[i]]
        A[i, G + hidx[i]] = tau_h[hidx[i]]
    end
    # The oracle accepts fixed coefficients through a closure so its numerical
    # mode solve and dense Hessian stay independent of the package Laplace code.
    function evaluate(beta)
        eta_fixed = X * beta
        u = zeros(G + H)
        function joint(uvec)
            eta = eta_fixed + A * uvec
            sum(log1p(exp(-abs(eta[i]))) + max(eta[i], 0.0) - y[i] * eta[i]
                for i in eachindex(y)) + 0.5 * dot(uvec, uvec)
        end
        for _ in 1:100
            eta = eta_fixed + A * u
            p = 1.0 ./ (1.0 .+ exp.(-eta))
            grad = u + A' * (p - y)
            weights = p .* (1.0 .- p)
            Hjoint = Matrix{Float64}(I, G + H, G + H) + A' * Diagonal(weights) * A
            ch = cholesky(Symmetric(Hjoint))
            step = ch \ grad
            norm(step, Inf) < 1e-10 && break
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
            # No descent step at machine precision: accept the mode only if the
            # gradient is already negligible (rounding stall near the optimum,
            # seen on x86-64 Julia 1.13.3); otherwise the oracle has failed.
            if rate <= 1e-8
                norm(grad, Inf) < 1e-8 && break
                error("independent crossed Binomial oracle mode did not converge")
            end
        end
        eta = eta_fixed + A * u
        p = 1.0 ./ (1.0 .+ exp.(-eta))
        weights = p .* (1.0 .- p)
        Hjoint = Matrix{Float64}(I, G + H, G + H) + A' * Diagonal(weights) * A
        return -(joint(u) + 0.5 * logdet(Symmetric(Hjoint)))
    end
    return evaluate
end

@testset "Crossed Bernoulli intercepts have separate predictor-dependent SDs" begin
    rng = MersenneTwister(20261023)
    G, H, m = 40, 40, 4
    g = repeat(1:G, inner = H * m)
    h = repeat(repeat(1:H, inner = m), outer = G)
    wg = repeat([-1.0, 1.0], inner = G ÷ 2)
    vh = repeat([-1.0, 1.0], inner = H ÷ 2)
    w, v = wg[g], vh[h]
    x = randn(rng, length(g))
    alpha_g = [log(0.45), 0.30]
    alpha_h = [log(0.50), -0.25]
    tau_g = exp.(alpha_g[1] .+ alpha_g[2] .* wg)
    tau_h = exp.(alpha_h[1] .+ alpha_h[2] .* vh)
    b_g = tau_g .* randn(rng, G)
    b_h = tau_h .* randn(rng, H)
    eta = -0.20 .+ 0.35 .* x .+ b_g[g] .+ b_h[h]
    y = Float64.(rand.(Ref(rng), Distributions.Bernoulli.(1.0 ./ (1.0 .+ exp.(-eta)))))
    data = (; y, x, g, h, w, v)

    formula = bf(
        @formula(y ~ x + (1 | g) + (1 | h)),
        @formula(sd(g) ~ w),
        @formula(sd(h) ~ v),
    )
    fit = drm(formula, Binomial(); data)

    @test fit.converged
    targets = DRModels._sd_target_parts(formula)
    blocks = DRModels._sd_target_blocks(targets)
    @test first.(fit.blocks[2:3]) == blocks
    @test coef(fit, blocks[1]) ≈ alpha_g atol = 0.45
    @test coef(fit, blocks[2]) ≈ alpha_h atol = 0.45

    predictions = predict_parameters(fit, data; se = false)
    @test all(isfinite, predictions[blocks[1]])
    @test all(isfinite, predictions[blocks[2]])
    @test length(predictions[blocks[1]]) == length(y)
    @test length(predictions[blocks[2]]) == length(y)

    slope_signs = map(blocks) do block
        slope = coef(fit, block)[2]
        iszero(slope) ? 1.0 : sign(slope)
    end
    clamp_data = (; y, x, g, h,
        w = fill(1.0e6 * slope_signs[1], length(y)),
        v = fill(1.0e6 * slope_signs[2], length(y)))
    clamped_predictions = predict_parameters(fit, clamp_data; se = true)
    @test all(isfinite, clamped_predictions[blocks[1]].value)
    @test all(isfinite, clamped_predictions[blocks[2]].value)
    @test all(clamped_predictions[blocks[1]].value .≈ exp(15.0))
    @test all(clamped_predictions[blocks[2]].value .≈ exp(15.0))
    @test all(isfinite, clamped_predictions[blocks[1]].se)
    @test all(isfinite, clamped_predictions[blocks[2]].se)
    clamp_low_data = (; y, x, g, h,
        w = fill(-1.0e6 * slope_signs[1], length(y)),
        v = fill(-1.0e6 * slope_signs[2], length(y)))
    lower_clamped = predict_parameters(fit, clamp_low_data; se = false)
    @test all(lower_clamped[blocks[1]] .≈ exp(-15.0))
    @test all(lower_clamped[blocks[2]] .≈ exp(-15.0))

    bootstrap = bootstrap_result(fit; data, B = 1,
        rng = MersenneTwister(202610231), failures = :error)
    @test bootstrap.attempted == 1
    @test bootstrap.used == 1
    @test bootstrap.failed == 0

    beta = coef(fit, :mu)
    oracle = crossed_binomial_sd_laplace_oracle(
        y, hcat(ones(length(y)), x), g, h,
        exp.(coef(fit, blocks[1])[1] .+ coef(fit, blocks[1])[2] .* wg),
        exp.(coef(fit, blocks[2])[1] .+ coef(fit, blocks[2])[2] .* vh),
    )
    @test abs(fit.loglik - oracle(beta)) / fit.nobs < 1e-7
end
