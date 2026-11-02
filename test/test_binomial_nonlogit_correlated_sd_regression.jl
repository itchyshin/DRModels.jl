# Characterize the ordinary correlated q=2 Binomial SD route for probit and
# cloglog links. This is recovery evidence for this exact family/block slice,
# not interval-coverage validation.
using DRModels, Test, Random
import Distributions
using LinearAlgebra, Statistics

function _normal_gh_rule(order)
    jacobi = SymTridiagonal(zeros(order), sqrt.(collect(1:(order - 1)) ./ 2))
    decomposition = eigen(jacobi)
    nodes = sqrt(2) .* decomposition.values
    weights = abs2.(decomposition.vectors[1, :])
    return nodes, weights ./ sum(weights)
end

_oracle_log1pexp(x) = x > 0 ? x + log1p(exp(-x)) : log1p(exp(x))

function _binomial_correlated_q2_prior_gh_loglik(
    y, x, group, beta, sd0, sd1, rho, link; order = 28
)
    nodes, weights = _normal_gh_rule(order)
    logweights = log.(weights)
    normal = Distributions.Normal()
    members = [findall(==(g), group) for g in 1:maximum(group)]
    total = 0.0
    for g in eachindex(members)
        rows = members[g]
        node_loglik = Vector{Float64}(undef, order^2)
        position = 0
        for i in eachindex(nodes), j in eachindex(nodes)
            position += 1
            b0 = sd0[g] * nodes[i]
            b1 = sd1[g] * (rho * nodes[i] + sqrt(1 - rho^2) * nodes[j])
            value = logweights[i] + logweights[j]
            for row in rows
                eta = beta[1] + beta[2] * x[row] + b0 + b1 * x[row]
                logp, log1mp = if link === :probit
                    (Distributions.logcdf(normal, eta),
                     Distributions.logcdf(normal, -eta))
                elseif link === :cloglog
                    (log(-expm1(-exp(eta))), -exp(eta))
                else
                    (-_oracle_log1pexp(-eta), -_oracle_log1pexp(eta))
                end
                value += y[row] * logp + (1 - y[row]) * log1mp
            end
            node_loglik[position] = value
        end
        peak = maximum(node_loglik)
        total += peak + log(sum(exp.(node_loglik .- peak)))
    end
    return total
end

@testset "Binomial correlated q3 SD regression" begin
    rng = MersenneTwister(2026102632)
    G, m = 128, 40
    g = repeat(1:G, inner = m)
    x = randn(rng, G * m)
    z = randn(rng, G * m)
    w0 = randn(rng, G)
    w1 = randn(rng, G)
    w2 = randn(rng, G)
    w0 = (w0 .- mean(w0)) ./ std(w0)
    w1 = (w1 .- mean(w1)) ./ std(w1)
    w2 = (w2 .- mean(w2)) ./ std(w2)
    w, h, j = w0[g], w1[g], w2[g]
    alpha0, alpha1, alpha2 = [log(0.65), 0.45], [log(0.55), -0.45], [log(0.5), 0.4]
    tau0 = exp.(alpha0[1] .+ alpha0[2] .* w0)
    tau1 = exp.(alpha1[1] .+ alpha1[2] .* w1)
    tau2 = exp.(alpha2[1] .+ alpha2[2] .* w2)
    partial = [0.2, -0.2, 0.25]
    s21, s31, s32 = sqrt(1 - partial[1]^2), sqrt(1 - partial[2]^2),
                    sqrt(1 - partial[3]^2)
    u = randn(rng, 3, G)
    b0 = tau0 .* u[1, :]
    b1 = tau1 .* (partial[1] .* u[1, :] .+ s21 .* u[2, :])
    b2 = tau2 .* (partial[2] .* u[1, :] .+ s31 * partial[3] .* u[2, :] .+
                  s31 * s32 .* u[3, :])
    eta = -0.1 .+ 0.2 .* x .- 0.15 .* z .+ b0[g] .+ b1[g] .* x .+ b2[g] .* z
    probability = 1 ./ (1 .+ exp.(-eta))
    y = Float64.(rand.(Ref(rng), Distributions.Bernoulli.(probability)))
    data = (; y, x, z, w, h, j, g)
    f = bf(
        @formula(y ~ x + z + (1 + x + z | g)),
        @formula(sd(g, coef(intercept)) ~ w),
        @formula(sd(g, coef(x)) ~ h),
        @formula(sd(g, coef(z)) ~ j),
    )

    fit = drm(f, Binomial(); data, nq = 5)
    targets = DRModels._sd_target_blocks(DRModels._sd_target_parts(f))
    @test fit.converged
    @test length(targets) == 3
    @test all(length(coef(fit, target)) == 2 for target in targets)
    @test abs(coef(fit, targets[1])[2] - alpha0[2]) < 0.3
    @test abs(coef(fit, targets[2])[2] - alpha1[2]) < 0.3
    @test abs(coef(fit, targets[3])[2] - alpha2[2]) < 0.3
    @test coef(fit, :recov)[end-2:end] ≈ atanh.(partial) atol = 0.3
    expected_pairwise = [partial[1], partial[2],
        partial[1] * partial[2] + sqrt(1 - partial[1]^2) *
            sqrt(1 - partial[2]^2) * partial[3]]
    @test fit.scales[:corpars] ≈ expected_pairwise atol = 0.3
    recov_names = DRModels._block_coefnames(fit, :recov)
    @test occursin("partial_rho_21", recov_names[end-2])
    @test occursin("partial_rho_31", recov_names[end-1])
    @test occursin("partial_rho_32_given_1", recov_names[end])
    @test fit.scales[targets[1]][1:m:end] ≈ exp.(coef(fit, targets[1])[1] .+
                                                     coef(fit, targets[1])[2] .* w0)
    @test fit.scales[targets[2]][1:m:end] ≈ exp.(coef(fit, targets[2])[1] .+
                                                     coef(fit, targets[2])[2] .* w1)
    @test fit.scales[targets[3]][1:m:end] ≈ exp.(coef(fit, targets[3])[1] .+
                                                     coef(fit, targets[3])[2] .* w2)
    simulated = simulate(fit; rng = MersenneTwister(2026102633))
    @test length(simulated) == length(y)
    @test all(value -> value == 0.0 || value == 1.0, simulated)
end

@testset "Binomial non-logit correlated-block SD regression" begin
    for (link, seed) in ((:probit, 202610251), (:cloglog, 202610252))
        rng = MersenneTwister(seed)
        G, m = 160, 32
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
        probability = link === :probit ?
            Distributions.cdf.(Distributions.Normal(), eta) : -expm1.(-exp.(eta))
        y = Float64.(rand.(Ref(rng), Distributions.Bernoulli.(probability)))
        data = (; y, x, z0, z1, g)
        f = bf(
            @formula(y ~ x + (1 + x | g)),
            @formula(sd(g, coef(intercept)) ~ z0),
            @formula(sd(g, coef(x)) ~ z1),
        )

        fit = drm(f, Binomial(link = link); data, nq = 9)
        @test fit.converged
        @test fit.family.link === link
        target_blocks = DRModels._sd_target_blocks(DRModels._sd_target_parts(f))
        sd0, sd1 = coef(fit, target_blocks[1]), coef(fit, target_blocks[2])
        @test length(sd0) == 2
        @test length(sd1) == 2
        @test sd0[2] > 0
        @test sd1[2] > 0
        @test abs(sd0[2] - alpha0[2]) < 0.35
        @test abs(sd1[2] - alpha1[2]) < 0.35
        @test abs(tanh(fit.theta[end]) - rho) < 0.25
        @test fit.scales[target_blocks[1]][1:m:end] ≈ exp.(sd0[1] .+ sd0[2] .* z0_group)
        @test fit.scales[target_blocks[2]][1:m:end] ≈ exp.(sd1[1] .+ sd1[2] .* z1_group)
        eta_hat = predict(fit, data; type = :link)
        @test predict(fit, data) ≈ DRModels._binomial_mean.(Ref(fit.family), eta_hat)
        simulated = simulate(fit; rng = MersenneTwister(seed + 1))
        @test length(simulated) == length(y)
        @test all(value -> value == 0.0 || value == 1.0, simulated)

        beta = coef(fit, :mu)
        sd0_group = fit.scales[target_blocks[1]][1:m:end]
        sd1_group = fit.scales[target_blocks[2]][1:m:end]
        rho_hat = fit.scales[:corpars][1]
        oracle96 = _binomial_correlated_q2_prior_gh_loglik(
            y, x, g, beta, sd0_group, sd1_group, rho_hat, link; order = 96
        )
        oracle128 = _binomial_correlated_q2_prior_gh_loglik(
            y, x, g, beta, sd0_group, sd1_group, rho_hat, link; order = 128
        )
        @test oracle96 ≈ oracle128 atol = 0.01
        @test fit.loglik ≈ oracle128 atol = 0.5
    end
end
