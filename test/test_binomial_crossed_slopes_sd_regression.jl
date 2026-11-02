# Independent synthetic oracle for two crossed ordinary Binomial effects,
# with one intercept and one numeric slope.
using DRModels, Test, Random, LinearAlgebra
import Distributions

function crossed_binomial_slope_laplace_oracle(y, X, gidx, hidx, loading_g, loading_h,
                                                tau_g, tau_h)
    G, H = length(tau_g), length(tau_h)
    A = zeros(length(y), G + H)
    for i in eachindex(y)
        A[i, gidx[i]] = tau_g[gidx[i]] * loading_g[i]
        A[i, G + hidx[i]] = tau_h[hidx[i]] * loading_h[i]
    end
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
            rate > 1e-8 || error("independent crossed Binomial oracle mode did not converge")
        end
        return -(joint(u) + 0.5 * logdet(Symmetric(
            Matrix{Float64}(I, G + H, G + H) +
                A' * Diagonal((1.0 ./ (1.0 .+ exp.(-(eta_fixed + A * u)))) .*
                    (1.0 .- 1.0 ./ (1.0 .+ exp.(-(eta_fixed + A * u))))) * A)))
    end
    return evaluate
end

@testset "Crossed Bernoulli intercept and slope have separate SD regressions" begin
    rng = MersenneTwister(20261024)
    G, H, m = 30, 30, 8
    g = repeat(1:G, inner = H * m)
    h = repeat(repeat(1:H, inner = m), outer = G)
    wg = [isodd(j) ? -1.0 : 1.0 for j in 1:G]
    vh = [isodd(j) ? -1.0 : 1.0 for j in 1:H]
    w, v = wg[g], vh[h]
    x = randn(rng, length(g))
    z = randn(rng, length(g))
    alpha_g = [log(0.50), 0.25]
    alpha_h = [log(0.40), -0.20]
    tau_g = exp.(alpha_g[1] .+ alpha_g[2] .* wg)
    tau_h = exp.(alpha_h[1] .+ alpha_h[2] .* vh)
    b_g = tau_g .* randn(rng, G)
    b_h = tau_h .* randn(rng, H)
    eta = -0.20 .+ 0.30 .* x .+ x .* b_g[g] .+ z .* b_h[h]
    y = Float64.(rand.(Ref(rng), Distributions.Bernoulli.(
        1.0 ./ (1.0 .+ exp.(-eta)))))
    data = (; y, x, z, g, h, w, v)

    formula = bf(
        @formula(y ~ x + (0 + x | g) + (0 + z | h)),
        @formula(sd(g) ~ w),
        @formula(sd(h) ~ v),
    )
    fit_or_error = try
        drm(formula, Binomial(); data)
    catch err
        err
    end
    @test !(fit_or_error isa Exception)
    if !(fit_or_error isa Exception)
        fit = fit_or_error
        targets = DRModels._sd_target_parts(formula)
        blocks = DRModels._sd_target_blocks(targets)
        @test fit.converged
        @test first.(fit.blocks[2:3]) == blocks
        @test coef(fit, blocks[1]) ≈ alpha_g atol = 0.45
        @test coef(fit, blocks[2]) ≈ alpha_h atol = 0.45

        predictions = predict_parameters(fit, data; se = false)
        @test length(predictions[blocks[1]]) == length(y)
        @test length(predictions[blocks[2]]) == length(y)
        @test all(isfinite, predictions[blocks[1]])
        @test all(isfinite, predictions[blocks[2]])

        bootstrap = bootstrap_result(fit; data, B = 1,
            rng = MersenneTwister(202610241), failures = :error)
        @test bootstrap.attempted == 1
        @test bootstrap.used == 1
        @test bootstrap.failed == 0

        oracle = crossed_binomial_slope_laplace_oracle(
            y, hcat(ones(length(y)), x), g, h, x, z,
            exp.(coef(fit, blocks[1])[1] .+ coef(fit, blocks[1])[2] .* wg),
            exp.(coef(fit, blocks[2])[1] .+ coef(fit, blocks[2])[2] .* vh),
        )
        @test abs(fit.loglik - oracle(coef(fit, :mu))) / fit.nobs < 1e-7
    end

    zero_slope_data = merge(data, (; z = zeros(length(z))))
    zero_slope_error = try
        drm(formula, Binomial(); data = zero_slope_data)
        nothing
    catch err
        err
    end
    @test zero_slope_error isa ArgumentError
    @test occursin("nonzero", sprint(showerror, zero_slope_error))

    one_informative_level = zeros(length(z))
    one_informative_level[h .== 1] .= z[h .== 1]
    rank_deficient_data = merge(data, (; z = one_informative_level))
    rank_error = try
        drm(formula, Binomial(); data = rank_deficient_data)
        nothing
    catch err
        err
    end
    @test rank_error isa ArgumentError
    @test occursin("contributing", sprint(showerror, rank_error))

    # Also exercise the mixed intercept/slope combination after the common
    # loading and informative-level guards were added.
    mixed_b_g = tau_g .* randn(rng, G)
    mixed_b_h = tau_h .* randn(rng, H)
    mixed_eta = -0.20 .+ 0.30 .* x .+ mixed_b_g[g] .+ z .* mixed_b_h[h]
    mixed_prob = 1.0 ./ (1.0 .+ exp.(-mixed_eta))
    mixed_y = Float64.(rand.(Ref(rng), Distributions.Bernoulli.(mixed_prob)))
    mixed_data = merge(data, (; y = mixed_y))
    mixed_formula = bf(
        @formula(y ~ x + (1 | g) + (0 + z | h)),
        @formula(sd(g) ~ w),
        @formula(sd(h) ~ v),
    )
    mixed_fit_or_error = try
        drm(mixed_formula, Binomial(); data = mixed_data)
    catch err
        err
    end
    @test !(mixed_fit_or_error isa Exception)
    if !(mixed_fit_or_error isa Exception)
        mixed_fit = mixed_fit_or_error
        mixed_targets = DRModels._sd_target_parts(mixed_formula)
        mixed_blocks = DRModels._sd_target_blocks(mixed_targets)
        @test mixed_fit.converged
        @test coef(mixed_fit, mixed_blocks[1]) ≈ alpha_g atol = 0.45
        @test coef(mixed_fit, mixed_blocks[2]) ≈ alpha_h atol = 0.45
    end
end
