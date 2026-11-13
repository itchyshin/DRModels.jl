# Public Bernoulli/binomial random-intercept SD-regression route.
using DRModels, Test, Random, QuadGK, StableRNGs
using LinearAlgebra
import Distributions
import ForwardDiff

# Truth-recovery assertions on stochastic fixtures are sampling-luck sensitive;
# gated per test/README.md "CI RNG policy (#388)".
const _SLOW_BINOMIAL_SD = get(ENV, "DRM_SLOW_TESTS", "0") == "1"

@testset "Binomial non-logit likelihood skips zero-count tail terms" begin
    cases = (
        (DRModels.Binomial(link = :probit), 0.0, -1000.0),
        (DRModels.Binomial(link = :probit), 1.0, 1000.0),
        (DRModels.Binomial(link = :cloglog), 0.0, -1000.0),
        (DRModels.Binomial(link = :cloglog), 1.0, 1000.0),
    )
    for (family, successes, eta) in cases
        ll(x) = DRModels._binomial_link_ll(family, 1, successes, x, 0.0)
        @test isfinite(ll(eta))
        @test isfinite(ForwardDiff.derivative(ll, eta))
    end
end

@testset "Binomial SD scale uses the native default smooth log clamp" begin
    eta = [-20.0, -12.0, -1.0, 0.0, 1.0, 12.0, 20.0]
    expected_eta = map(eta) do x
        x > 12.0 ? 12.0 + 3.0 * tanh((x - 12.0) / 3.0) :
        x < -12.0 ? -12.0 - 3.0 * tanh((-12.0 - x) / 3.0) : x
    end
    @test DRModels._binomial_sd_scale(eta) ≈ exp.(expected_eta)
    @test DRModels._binomial_sd_scale(eta[3:5]) ≈ exp.(eta[3:5])
end

@testset "Bernoulli correlated-block SD regression" begin
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
    f = bf(
        @formula(y ~ x + (1 + x | g)),
        @formula(sd(g, coef(intercept)) ~ z0),
        @formula(sd(g, coef(x)) ~ z1),
    )
    fit = DRModels._drm_binomial_sd_regression(f, Binomial(); data, nq = 9)
    @test fit.converged

    # Independent 2-D integration over standard-normal innovations. This
    # checks the marginal-SD/correlation parameterisation and the correlated
    # intercept/slope loading independently of the package AGHQ objective.
    theta = [-0.1, 0.25, alpha0..., alpha1..., atanh(rho)]
    normal = Distributions.Normal()
    reference = 0.0
    for gi in 1:G
        rows = findall(==(gi), g)
        s0 = exp(alpha0[1] + alpha0[2] * z0_group[gi])
        s1 = exp(alpha1[1] + alpha1[2] * z1_group[gi])
        loglik(u0, u1) = sum(rows) do i
            b0i = s0 * u0
            b1i = s1 * (rho * u0 + sqrt(1 - rho^2) * u1)
            e = -0.1 + 0.25 * x[i] + b0i + x[i] * b1i
            y[i] == 1 ? -log1p(exp(-e)) : -log1p(exp(e))
        end
        integrate_u0(u0) = begin
            value, _ = QuadGK.quadgk(
                u1 -> exp(loglik(u0, u1)) * Distributions.pdf(normal, u1),
                -9.0, 9.0; rtol = 1e-7,
            )
            value * Distributions.pdf(normal, u0)
        end
        value, _ = QuadGK.quadgk(integrate_u0, -9.0, 9.0; rtol = 1e-7)
        reference += log(value)
    end
    L_contract = [tau0[1] 0.0;
                  rho * tau1[1] tau1[1] * sqrt(1 - rho^2)]
    Sigma_contract = L_contract * transpose(L_contract)
    @test Sigma_contract[1, 1] ≈ tau0[1]^2
    @test Sigma_contract[1, 2] ≈ rho * tau0[1] * tau1[1]
    @test Sigma_contract[2, 2] ≈ tau1[1]^2
    L_wrong_innovation = [tau0[1] 0.0; rho * tau1[1] tau1[1]]
    @test !isapprox(L_wrong_innovation * transpose(L_wrong_innovation),
                    Sigma_contract; atol = 1e-10)
    Sigma_independent = [tau0[1]^2 0.0; 0.0 tau1[1]^2]
    @test !isapprox(Sigma_independent, Sigma_contract; atol = 1e-10)
    @test isapprox(-fit.nll(theta), reference; atol = 2e-3)
    theta_independent = copy(theta)
    theta_independent[end] = 0.0
    theta_swapped_sd = [theta[1:2]; alpha1; alpha0; theta[end]]
    @test abs(fit.nll(theta) - fit.nll(theta_independent)) > 0.01
    @test abs(fit.nll(theta) - fit.nll(theta_swapped_sd)) > 0.01
    theta_near_boundary = copy(theta)
    # eta=10 gives rho > 0.99999999 while keeping the represented covariance
    # numerically resolvable by the quadrature routine.
    theta_near_boundary[end] = 10.0
    @test isfinite(fit.nll(theta_near_boundary))
    gradient_near_boundary = DRModels.ForwardDiff.gradient(
        fit.nll, theta_near_boundary,
    )
    @test all(isfinite, gradient_near_boundary)
    @test DRModels._binomial_stable_sech(20.0) > 0
    fit_near_boundary = deepcopy(fit)
    fit_near_boundary.theta[end] = 20.0
    simulator_near_boundary = DRModels._marginal_simulator_build(
        fit_near_boundary, data,
    )
    simulated_near_boundary = simulator_near_boundary(MersenneTwister(2026102026))
    @test length(simulated_near_boundary) == length(y)
    @test all(isfinite, simulated_near_boundary)

    # A constant-SD target pair must reduce to the existing correlated
    # random-intercept/slope likelihood after mapping marginal SDs to its
    # log-Cholesky innovation parameterisation.
    constant = bf(
        @formula(y ~ x + (1 + x | g)),
        @formula(sd(g, coef(intercept)) ~ 1),
        @formula(sd(g, coef(x)) ~ 1),
    )
    fit_constant = DRModels._drm_binomial_sd_regression(
        constant, Binomial(); data, nq = 9,
    )
    corr = DRModels._fit_binomial_corr_ranef(
        Binomial(), y, ones(length(y)), hcat(ones(length(y)), x), x,
        DRModels._group_index(g)..., ["(Intercept)", "x"], :g, 1e-8; nq = 9,
    )
    t = [-0.1, 0.25, log(0.7), log(0.85), atanh(rho)]
    cc = rho * 0.85
    log_innov = log(0.85 * sqrt(1 - rho^2))
    old_t = [-0.1, 0.25, log(0.7), log_innov, cc]
    @test isapprox(fit_constant.nll(t), corr.nll(old_t); atol = 2e-8)

    intercept_only = bf(
        @formula(y ~ x + (1 + x | g)),
        @formula(sd(g, coef(intercept)) ~ z0),
    )
    fit_intercept_only = DRModels._drm_binomial_sd_regression(
        intercept_only, Binomial(); data, nq = 9,
    )
    @test fit_intercept_only.converged
    @test length(coef(fit_intercept_only, :recov)) == 2
    @test last(only(filter(p -> first(p) === :recov, fit_intercept_only.coefnames))) ==
        ["g:log_sd_x", "g:atanh_rho"]
    data_missing_x = (; y, z0, z1, g)
    @test_throws ArgumentError DRModels._marginal_simulator_build(
        fit_intercept_only, data_missing_x,
    )
    sim_i = DRModels._marginal_simulator_build(fit_intercept_only, data)
    @test length(sim_i(MersenneTwister(2026102023))) == length(y)
    boot_i = bootstrap_result(fit_intercept_only; data, B = 1,
        rng = MersenneTwister(2026102025), failures = :error)
    @test boot_i.attempted == 1
    @test boot_i.used == 1
    @test boot_i.failed == 0

    slope_only = bf(
        @formula(y ~ x + (1 + x | g)),
        @formula(sd(g, coef(x)) ~ z1),
    )
    fit_slope_only = DRModels._drm_binomial_sd_regression(
        slope_only, Binomial(); data, nq = 9,
    )
    @test fit_slope_only.converged
    @test length(coef(fit_slope_only, :recov)) == 2
    @test last(only(filter(p -> first(p) === :recov, fit_slope_only.coefnames))) ==
        ["g:log_sd_intercept", "g:atanh_rho"]
    sim_s = DRModels._marginal_simulator_build(fit_slope_only, data)
    @test length(sim_s(MersenneTwister(2026102024))) == length(y)
    boot_s = bootstrap_result(fit_slope_only; data, B = 1,
        rng = MersenneTwister(2026102027), failures = :error)
    @test boot_s.attempted == 1
    @test boot_s.used == 1
    @test boot_s.failed == 0

    sd_blocks = [DRModels._sd_target_block(item.target)
                 for item in DRModels._sd_target_parts(f)]
    @test all(haskey(fit.scales, block) for block in sd_blocks)
    @test !DRModels._binomial_sd_threaded_profile_unsupported(fit)
    predicted = predict_parameters(fit, data; se = false)
    @test all(isfinite, predicted[first(sd_blocks)])
    @test all(isfinite, predicted[last(sd_blocks)])
    simulator = DRModels._marginal_simulator_build(fit, data)
    @test simulator !== nothing
    simulated = simulator(MersenneTwister(2026102021))
    @test length(simulated) == length(y)
    boot = bootstrap_result(fit; data, B = 1,
        rng = MersenneTwister(2026102022), failures = :error)
    @test boot.attempted == 1
    @test boot.used == 1
    @test boot.failed == 0
end

@testset "Bernoulli SD formula retains coefficient-specific targets" begin
    f = bf(
        @formula(cbind(s, fail) ~ x + (1 | g) + (0 + x | g)),
        @formula(sd(g, coef(intercept)) ~ z0),
        @formula(sd(g, coef(x)) ~ z1),
    )
    targets = DRModels._sd_target_parts(f)
    @test length(targets) == 2
    @test [x.target.group for x in targets] == [:g, :g]
    @test [x.target.member for x in targets] == [:intercept, :x]
    @test [string(x.rhs) for x in targets] == ["z0", "z1"]
    @test length(unique(x.key for x in targets)) == 2
end

@testset "Bernoulli independent intercept and slope have separate SD regressions" begin
    rng = StableRNG(20261019)
    G, m = 60, 40
    g = repeat(1:G, inner = m)
    z0group = repeat([0.0, 1.0], inner = G ÷ 2)
    z1group = repeat([0.0, 1.0, 1.0, 0.0], inner = G ÷ 4)
    z0, z1 = z0group[g], z1group[g]
    x = randn(rng, G * m)
    alpha0, alpha1 = [-0.7, 0.35], [-0.9, 0.45]
    b0 = exp.(alpha0[1] .+ alpha0[2] .* z0group) .* randn(rng, G)
    b1 = exp.(alpha1[1] .+ alpha1[2] .* z1group) .* randn(rng, G)
    eta = 0.2 .+ 0.3 .* x .+ b0[g] .+ x .* b1[g]
    p = 1 ./ (1 .+ exp.(-eta))
    y = rand(rng, length(p)) .< p
    data = (; y = Float64.(y), x, z0, z1, g)
    f = bf(
        @formula(y ~ x + (1 | g) + (0 + x | g)),
        @formula(sd(g, coef(intercept)) ~ z0),
        @formula(sd(g, coef(x)) ~ z1),
    )

    fit = drm(f, Binomial(); data = data)

    @test fit.converged
    target_blocks = [DRModels._sd_target_block(item.target)
                     for item in DRModels._sd_target_parts(f)]
    @test target_blocks == first.(fit.blocks[2:3])
    @test !haskey(fit.scales, :sd)
    @test !haskey(fit.scales, :sd_group)
    @test coef(fit, target_blocks[1]) ≈ alpha0 atol = 0.45
    @test coef(fit, target_blocks[2]) ≈ alpha1 atol = 0.45
    for block in target_blocks
        sd_names = only(last.(filter(p -> first(p) === block, fit.coefnames)))
        @test allunique(sd_names)
    end
    predictions = predict_parameters(fit, data; se = false)
    @test all(isfinite, predictions[target_blocks[1]])
    @test all(isfinite, predictions[target_blocks[2]])
    @test length(predictions[target_blocks[1]]) == length(y)
    @test length(predictions[target_blocks[2]]) == length(y)
    marginal = marginal_parameters(fit)
    for block in target_blocks
        @test haskey(marginal, block)
        @test marginal[block] ≈ predictions[block]
    end
    simulator = DRModels._marginal_simulator_build(fit, data)
    @test simulator !== nothing
    simulated = simulator(MersenneTwister(2026101921))
    @test length(simulated) == length(y)
    bootstrap = bootstrap_result(fit; data, B = 1,
        rng = MersenneTwister(2026101922), failures = :error)
    @test bootstrap.attempted == 1
    @test bootstrap.used == 1
    @test bootstrap.failed == 0

    # Independent two-dimensional normal integration checks both coefficient
    # loadings and their separate group-level SD designs at fixed parameters.
    theta = [0.2, 0.3, alpha0..., alpha1...]
    normal = Distributions.Normal()
    reference = 0.0
    for gi in 1:G
        rows = findall(==(gi), g)
        sd0 = exp(alpha0[1] + alpha0[2] * z0group[gi])
        sd1 = exp(alpha1[1] + alpha1[2] * z1group[gi])
        loglik(u0, u1) = sum(rows) do i
            eta_i = 0.2 + 0.3 * x[i] + sd0 * u0 + x[i] * sd1 * u1
            y[i] == 1 ? -log1p(exp(-eta_i)) : -log1p(exp(eta_i))
        end
        integrate_u0(u0) = begin
            value, _ = QuadGK.quadgk(
                u1 -> exp(loglik(u0, u1)) * Distributions.pdf(normal, u1),
                -9.0, 9.0; rtol = 1e-8,
            )
            value * Distributions.pdf(normal, u0)
        end
        value, _ = QuadGK.quadgk(integrate_u0, -9.0, 9.0; rtol = 1e-8)
        reference += log(value)
    end
    # The public optimizer uses its production AGHQ order, so compare it with
    # the high-order oracle within the measured default-order integration
    # error. A second, high-order objective must match the direct integral.
    @test isapprox(-fit.nll(theta), reference; atol = 3e-3)
    theta_swapped_sd_designs = [0.2, 0.3, alpha1..., alpha0...]
    @test abs(fit.nll(theta) - fit.nll(theta_swapped_sd_designs)) > 0.01
    fit_high_order = DRModels._drm_binomial_sd_regression(
        f, Binomial(); data = data, nq = 15,
    )
    @test fit_high_order.converged
    @test isapprox(-fit_high_order.nll(theta), reference; atol = 2e-4)
end

function _binomial_sd_fixture(; G = 28, m = 8, trials = 12)
    rng = StableRNG(20261017)
    g = repeat(1:G, inner = m)
    zgroup = repeat([0.0, 1.0], inner = G ÷ 2)
    z = zgroup[g]
    x = randn(rng, length(g))
    alpha = [-0.55, 0.55]
    b = exp.(alpha[1] .+ alpha[2] .* zgroup) .* randn(rng, G)
    eta = 0.15 .+ 0.45 .* x .+ b[g]
    prob = 1.0 ./ (1.0 .+ exp.(-eta))
    s = [rand(rng, Distributions.Binomial(trials, p)) for p in prob]
    return (; s = Float64.(s), fail = Float64.(trials .- s), x, z, g), alpha
end

@testset "binomial latent SD regression: q1 fit and reduction" begin
    data, alpha = _binomial_sd_fixture()
    variable = bf(@formula(cbind(s, fail) ~ x + (1 | g)), @formula(sd(g) ~ z))
    fit = drm(variable, Binomial(); data = data)
    @test fit.converged
    @test length(coef(fit, :sd)) == 2
    @test coef(fit, :sd)[2] > 0
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.55
    @test all(isfinite, coef(fit))
    @test_throws ArgumentError re_sd(fit)
    @test fit.scales[:sd_group] ≈ exp.(hcat(ones(28), [zeros(14); ones(14)]) * coef(fit, :sd))
    @test Int.(fit.scales[:sd_group_index]) == data.g

    constant = bf(@formula(cbind(s, fail) ~ x + (1 | g)), @formula(sd(g) ~ 1))
    f_sd = drm(constant, Binomial(); data = data)
    f_old = drm(bf(@formula(cbind(s, fail) ~ x + (1 | g))), Binomial(); data = data)
    @test f_sd.converged && f_old.converged
    @test isapprox(f_sd.loglik, f_old.loglik; atol = 1e-5)
    @test isapprox(coef(f_sd, :mu), coef(f_old, :mu); atol = 1e-4)
    @test isapprox(coef(f_sd, :sd)[1], coef(f_old, :resd)[1]; atol = 1e-4)
end

@testset "Binomial non-logit links retain predictor-dependent random-effect SDs" begin
    # Independent Bernoulli data keep this a link check rather than a change
    # to the first-delivery trial-count model. Each link must use the same
    # group-level log-SD design and report the link used by the fitted model.
    rng = StableRNG(20261024)
    G, m = 48, 14
    g = repeat(1:G, inner = m)
    wgroup = repeat([-1.0, 1.0], inner = G ÷ 2)
    w = wgroup[g]
    x = randn(rng, G * m)
    alpha = [-0.65, 0.5]
    b = exp.(alpha[1] .+ alpha[2] .* wgroup) .* randn(rng, G)
    eta = -0.1 .+ 0.3 .* x .+ b[g]
    formulas = bf(@formula(y ~ x + (1 | g)), @formula(sd(g) ~ w))

    for link in (:probit, :cloglog)
        family = Binomial(link = link)
        probability = if link === :probit
            Distributions.cdf.(Distributions.Normal(), eta)
        else
            -expm1.(-exp.(eta))
        end
        y = Float64.(rand.(Ref(rng), Distributions.Bernoulli.(probability)))
        data = (; y, x, w, g)
        fixed_fit = drm(bf(@formula(y ~ x)), family; data)
        @test fixed_fit.converged
        @test fixed_fit.family.link === link
        fixed_eta = predict(fixed_fit, data; type = :link)
        @test predict(fixed_fit, data) ≈ DRModels._binomial_mean.(Ref(family), fixed_eta)
        fit = drm(formulas, family; data, nq = 9)

        @test fit.converged
        @test fit.family.link === link
        @test length(coef(fit, :sd)) == 2
        @test coef(fit, :sd)[2] > 0
        @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.8
        @test fit.scales[:sd_group] ≈ exp.(hcat(ones(G), wgroup) * coef(fit, :sd))
        eta_hat = predict(fit, data; type = :link)
        @test predict(fit, data) ≈ DRModels._binomial_mean.(Ref(family), eta_hat)
        simulator = DRModels._marginal_simulator_build(fit, data)
        simulated = simulator(MersenneTwister(2026102400 + (link === :probit)))
        @test length(simulated) == length(y)
        @test all(value -> value == 0.0 || value == 1.0, simulated)
    end
end

@testset "Bernoulli independent random-slope SD regression" begin
    rng = StableRNG(20261018)
    # The slope-variance regression needs many independent groups; keep this
    # recovery check in a high-information regime rather than stress-testing
    # boundary behaviour in a small sample.
    G, m = 100, 30
    g = repeat(1:G, inner = m)
    zgroup = repeat([0.0, 1.0], inner = G ÷ 2)
    z = zgroup[g]
    x = randn(rng, G * m)
    alpha = [-0.8, 0.65]
    b = exp.(alpha[1] .+ alpha[2] .* zgroup) .* randn(rng, G)
    eta = 0.2 .+ 0.35 .* x .+ b[g] .* x
    p = 1.0 ./ (1.0 .+ exp.(-eta))
    y = Float64.(rand.(Ref(rng), Distributions.Bernoulli.(p)))
    data = (; y, x, z, g)
    formula = bf(@formula(y ~ x + (0 + x | g)), @formula(sd(g) ~ z))
    fit = DRModels._drm_binomial_sd_regression(formula, Binomial(); data, nq = 31)
    public_fit = drm(formula, Binomial(); data)
    @test fit.converged
    @test public_fit.converged
    @test length(coef(fit, :sd)) == 2
    @test coef(fit, :sd)[2] > 0
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.4
    @test fit.scales[:sd_group] ≈ exp.(hcat(ones(G), zgroup) * coef(fit, :sd))
    @test fit.scales[:sd] ≈ fit.scales[:sd_group][g]
    @test length(coef(public_fit, :sd)) == 2
    @test coef(public_fit, :sd)[2] > 0
    @test public_fit.scales[:sd_group] ≈ exp.(hcat(ones(G), zgroup) * coef(public_fit, :sd))
    public_prediction = predict_parameters(public_fit, data)
    @test public_prediction[:sd] ≈ public_fit.scales[:sd_group][g]

    # Independent scalar-normal integration at fixed parameters checks that
    # the random-slope covariate enters each group's linear predictor.
    theta = [0.2, 0.35, -0.8, 0.65]
    reference = 0.0
    normal = Distributions.Normal()
    for gi in 1:G
        rows = findall(==(gi), g)
        sdg = exp(theta[3] + theta[4] * zgroup[gi])
        integrand = u -> begin
            ll = sum(rows) do i
                pi = 1 / (1 + exp(-(theta[1] + theta[2] * x[i] + sdg * u * x[i])))
                y[i] == 1 ? log(pi) : log1p(-pi)
            end
            exp(ll) * Distributions.pdf(normal, u)
        end
        value, _ = QuadGK.quadgk(integrand, -10.0, 10.0; rtol = 1e-10)
        reference += log(value)
    end
    @test isapprox(-fit.nll(theta), reference; atol = 2e-4)
    # The public route keeps its lower default quadrature order; compare its
    # fixed-parameter likelihood with the same independent integral at a
    # tolerance informed by the default rule's observed approximation error.
    @test isapprox(-public_fit.nll(theta), reference; atol = 0.02)
end

@testset "Bernoulli response equals one-trial binomial" begin
    counts, _ = _binomial_sd_fixture(; G = 28, m = 8, trials = 1)
    bernoulli = (; y = counts.s, x = counts.x, z = counts.z, g = counts.g)
    bfit = drm(
        bf(@formula(y ~ x + (1 | g)), @formula(sd(g) ~ z)), Binomial(); data = bernoulli)
    cfit = drm(
        bf(@formula(cbind(s, fail) ~ x + (1 | g)), @formula(sd(g) ~ z)),
        Binomial(); data = counts)
    @test bfit.converged && cfit.converged
    @test isapprox(bfit.loglik, cfit.loglik; atol = 1e-8)
    @test isapprox(coef(bfit), coef(cfit); atol = 1e-8)
    @test all(0 .< fitted(bfit; re = :zero) .< 1)
end

@testset "binomial latent SD regression: independent fixed-parameter integral" begin
    data, _ = _binomial_sd_fixture(; G = 28, m = 4, trials = 2)
    formula = bf(@formula(cbind(s, fail) ~ x + (1 | g)), @formula(sd(g) ~ z))
    fit5 = DRModels._drm_binomial_sd_regression(formula, Binomial(); data = data, nq = 5)
    fit = drm(formula, Binomial(); data = data)
    fit13 = DRModels._drm_binomial_sd_regression(formula, Binomial(); data = data, nq = 13)
    theta = [0.15, 0.45, -0.55, 0.55]
    normal = Distributions.Normal()
    reference = 0.0
    for gi in 1:28
        rows = findall(==(gi), data.g)
        sdg = exp(theta[3] + theta[4] * data.z[first(rows)])
        integrand = u -> begin
            ll = sum(rows) do i
                p = 1 / (1 + exp(-(theta[1] + theta[2] * data.x[i] + sdg * u)))
                Distributions.logpdf(Distributions.Binomial(2, p), Int(data.s[i]))
            end
            exp(ll) * Distributions.pdf(normal, u)
        end
        value, _ = QuadGK.quadgk(integrand, -10.0, 10.0; rtol = 1e-11)
        reference += log(value)
    end
    @test isapprox(-fit.nll(theta), reference; atol = 2e-3)
    @test abs(fit5.nll(theta) - fit13.nll(theta)) < 2e-3
    @test abs(fit.nll(theta) - fit13.nll(theta)) < 2e-3
    @test fit.scales[:aghq_nodes] == [Float64(DRModels._BINOMIAL_RANEF_AGHQ_K)]
    @test DRModels._bootstrap_refit_kwargs(fit13).nq == 13
    # This fixture's SD intercept sits at the zero-variance boundary, so a
    # single-replicate refit is degenerate on a sizeable fraction of draws and
    # `MersenneTwister(seed)` streams differ across Julia versions (the package
    # draws each replicate from it). Try the recorded seed first, then the next
    # few, until one replicate refits; the exact-seed checks below use that run.
    bootstrap13 = nothing
    for offset in 0:9
        try
            bootstrap13 = bootstrap_result(fit13; data, B = 1,
                rng = MersenneTwister(2026102201 + offset), failures = :error)
            break
        catch err
            (err isa ErrorException && occursin("bootstrap", err.msg) && offset < 9) ||
                rethrow()
        end
    end
    @test bootstrap13.attempted == 1
    @test bootstrap13.used == 1
    @test bootstrap13.failed == 0

    # A successful refit alone would not prove the fitted quadrature order was
    # preserved. Recreate the exact bootstrap response from its recorded seed,
    # refit it with 13 nodes, and compare that refit's coefficients with the
    # one-draw bootstrap quantiles. This catches a refit that silently falls
    # back to the default node count.
    replicate_rng = MersenneTwister(only(bootstrap13.seeds))
    simulator13 = DRModels._marginal_simulator_build(fit13, data)
    simulated_response = simulator13(replicate_rng)
    replicate_data = DRModels._bootstrap_data(formula, data, simulated_response)
    expected_refit13 = DRModels._drm_binomial_sd_regression(
        formula, Binomial(); data = replicate_data, nq = 13,
    )
    expected_refit5 = DRModels._drm_binomial_sd_regression(
        formula, Binomial(); data = replicate_data, nq = 5,
    )
    expected_coefficients13 = coef(expected_refit13)
    @test maximum(abs.(expected_coefficients13 .- coef(expected_refit5))) > 1e-8
    @test length(bootstrap13.summary) == length(expected_coefficients13)
    @test [row.lower for row in bootstrap13.summary] ≈ expected_coefficients13 atol = 1e-8
    @test [row.upper for row in bootstrap13.summary] ≈ expected_coefficients13 atol = 1e-8
end

@testset "binomial latent SD regression: unsupported cells refuse explicitly" begin
    data, _ = _binomial_sd_fixture(; G = 6, m = 4, trials = 2)
    route(f; kwargs...) = drm(f, Binomial(); data = data, kwargs...)
    @test_throws ArgumentError route(bf(@formula(cbind(s, fail) ~ x + (1 | g)),
                                        @formula(sd(h) ~ z)))
    @test_throws ArgumentError route(bf(@formula(cbind(s, fail) ~ x + (1 + x | g)),
                                        @formula(sd(g) ~ z)))
    @test_throws ArgumentError route(bf(@formula(cbind(s, fail) ~ x + (1 | g)),
                                        @formula(sd(g) ~ x)))
    @test_throws ArgumentError route(bf(@formula(cbind(s, fail) ~ x + (1 | g)),
                                        @formula(sigma ~ x), @formula(sd(g) ~ z)))
    # `marginal = :Laplace` now fits this shape (opt-in drmTMB parity route).
    laplace_fit = route(bf(@formula(cbind(s, fail) ~ x + (1 | g)),
                           @formula(sd(g) ~ z)); marginal = :Laplace)
    @test laplace_fit.marginal === :Laplace

    wrong_dpar = bf(@formula(cbind(s, fail) ~ x + (1 | g)),
                    @formula(sd(g) ~ z))
    sd_key = only(keys(wrong_dpar.sd_targets))
    original_target = wrong_dpar.sd_targets[sd_key]
    wrong_dpar.sd_targets[sd_key] = merge(original_target, (; dpar = :sigma))
    @test_throws ArgumentError route(wrong_dpar)
    wrong_dpar.sd_targets[sd_key] = merge(original_target, (; level = :spatial))
    @test_throws ArgumentError route(wrong_dpar)
    wrong_dpar.sd_targets[sd_key] = merge(original_target, (; block = :p))
    @test_throws ArgumentError route(wrong_dpar)
    wrong_dpar.sd_targets[sd_key] = merge(original_target, (; member = :not_a_coefficient))
    @test_throws ArgumentError route(wrong_dpar)
end

@testset "Bernoulli structured SD target metadata" begin
    formula = bf(
        @formula(y ~ x + animal(1 | g)),
        @formula(sd(g, animal) ~ z),
    )
    targets = DRModels._sd_target_parts(formula)
    @test length(targets) == 1
    @test only(targets).target == (
        group = :g, level = :animal, member = nothing,
        dpar = :mu, block = nothing,
    )
    @test only(targets).rhs == @formula(y ~ z).rhs
end

@testset "Bernoulli animal-intercept SD regression" begin
    rng = StableRNG(20261021)
    G, m = 40, 16
    group = repeat(1:G, inner = m)
    z_group = repeat([-1.0, 1.0], inner = G ÷ 2)
    z = z_group[group]
    x = repeat(range(-1.0, 1.0; length = m), outer = G)
    A = [0.25^abs(i - j) for i in 1:G, j in 1:G]
    u = cholesky(Symmetric(A)).L * randn(rng, G)
    alpha = [-0.55, 0.6]
    tau = exp.(alpha[1] .+ alpha[2] .* z_group)
    eta = 0.1 .+ 0.3 .* x .+ tau[group] .* u[group]
    p = 1 ./ (1 .+ exp.(-eta))
    y = Float64.(rand.(Ref(rng), Distributions.Bernoulli.(p)))
    data = (; y, x, z, group)
    formula = bf(
        @formula(y ~ x + animal(1 | group)),
        @formula(sd(group, animal) ~ z),
    )

    fit = drm(formula, Binomial(); data, A)
    @test fit.converged
    @test all(isfinite, vcov(fit))
    @test all(>(0), diag(vcov(fit)))
    @test length(coef(fit, :sd)) == 2
    @test coef(fit, :sd)[2] > 0
    _SLOW_BINOMIAL_SD && @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.45
    @test fit.scales[:sd_group] ≈ exp.(hcat(ones(G), z_group) * coef(fit, :sd))
    @test fit.scales[:sd] ≈ fit.scales[:sd_group][group]
    @test predict_parameters(fit, data; se = false)[:sd] ≈ fit.scales[:sd]

    # An independently parameterised dense latent-b oracle checks the sparse
    # route's D*A*D covariance and Laplace determinant terms.
    theta_oracle = [0.1, 0.3, alpha...]
    tau_oracle = exp.(alpha[1] .+ alpha[2] .* z_group)
    Sigma = Diagonal(tau_oracle) * A * Diagonal(tau_oracle)
    Q = inv(Sigma)
    s_int = round.(Int, y)
    aux = (s = s_int, ntr = ones(Int, length(y)),
           logchoose = zeros(length(y)))
    eta_fixed = 0.1 .+ 0.3 .* x
    function dense_latent_nll(b)
        total = 0.5 * dot(b, Q * b)
        for i in eachindex(y)
            total += DRModels._laplace_value(
                Val(:binomial), aux, i, eta_fixed[i] + b[group[i]],
            )
        end
        total
    end
    result = DRModels.Optim.optimize(
        dense_latent_nll, zeros(G), DRModels.Optim.LBFGS(),
        DRModels.Optim.Options(g_tol = 1e-9, iterations = 400);
        autodiff = :forward,
    )
    b_mode = DRModels.Optim.minimizer(result)
    H = DRModels.ForwardDiff.hessian(dense_latent_nll, b_mode)
    dense_laplace = dense_latent_nll(b_mode) -
        0.5 * logdet(cholesky(Symmetric(Q))) +
        0.5 * logdet(cholesky(Symmetric(H)))
    @test isapprox(fit.nll(theta_oracle), dense_laplace; atol = 2e-5)
    @test isposdef(Symmetric(Sigma))

    @test_throws ArgumentError drm(formula, Binomial(); data, se = false)
    @test_throws ArgumentError drm(formula, Binomial(); data, A = A[1:end-1, 1:end-1], se = false)
    A_asymmetric = copy(A)
    A_asymmetric[1, 2] += 0.1
    @test_throws ArgumentError drm(formula, Binomial(); data, A = A_asymmetric, se = false)

    data_bad_sd = merge(data, (; z = collect(1.0:length(z))))
    @test_throws ArgumentError drm(formula, Binomial(); data = data_bad_sd, A, se = false)

    data_zero_slope = merge(data, (; x = zeros(length(x))))
    formula_zero_slope = bf(
        @formula(y ~ 1 + animal(0 + x | group)),
        @formula(sd(group, animal) ~ z),
    )
    @test_throws ArgumentError drm(formula_zero_slope, Binomial();
                                   data = data_zero_slope, A, se = false)
end

@testset "Bernoulli animal random-slope SD regression" begin
    rng = StableRNG(20261022)
    G, m = 64, 32
    group_order = randperm(rng, G)
    row_group_index = repeat(1:G, inner = m)
    group = repeat(group_order, inner = m)
    z_group = ifelse.(randperm(rng, G) .<= G ÷ 2, -1.0, 1.0)
    z = z_group[group]
    x_within = collect(range(-1.2, 1.2; length = m))
    x = repeat(x_within, outer = G)
    A = [0.35^abs(group_order[i] - group_order[j]) for i in 1:G, j in 1:G]
    alpha = [-0.45, 0.65]
    u = cholesky(Symmetric(A)).L * randn(rng, G)
    tau = exp.(alpha[1] .+ alpha[2] .* z_group[group_order])
    eta = -0.1 .+ 0.3 .* x .+ x .* (tau .* u)[row_group_index]
    y = Float64.(rand.(Ref(rng), Distributions.Bernoulli.(1 ./ (1 .+ exp.(-eta)))))
    data = (; y, x, z, group)
    formula = bf(
        @formula(y ~ x + animal(0 + x | group)),
        @formula(sd(group, animal) ~ z),
    )

    fit = drm(formula, Binomial(); data, A)
    @test fit.converged
    @test coef(fit, :sd)[2] > 0
    _SLOW_BINOMIAL_SD && @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.35
    @test fit.scales[:sd_group] ≈ exp.(hcat(ones(G), z_group[group_order]) * coef(fit, :sd))
    @test fit.scales[:sd] ≈ fit.scales[:sd_group][row_group_index]
    @test predict_parameters(fit, data; se = false)[:sd] ≈ fit.scales[:sd]
    pred_sd = predict_parameters(fit, data; se = true)[:sd]
    @test pred_sd.value ≈ fit.scales[:sd]
    @test all(isfinite, pred_sd.se)

    # Dense latent-variable Laplace oracle: b = D*u, u ~ N(0, A), and the
    # observation-level slope contributes x_i*b[group_i] to the logit.
    theta = [-0.1, 0.3, alpha...]
    D = Diagonal(exp.(alpha[1] .+ alpha[2] .* z_group[group_order]))
    Sigma = D * A * D
    Q = inv(Sigma)
    aux = (s = round.(Int, y), ntr = ones(Int, length(y)),
           logchoose = zeros(length(y)))
    eta_fixed = -0.1 .+ 0.3 .* x
    function dense_latent_nll(b)
        total = 0.5 * dot(b, Q * b)
        for i in eachindex(y)
            total += DRModels._laplace_value(
                Val(:binomial), aux, i, eta_fixed[i] + x[i] * b[row_group_index[i]],
            )
        end
        total
    end
    result = DRModels.Optim.optimize(
        dense_latent_nll, zeros(G), DRModels.Optim.LBFGS(),
        DRModels.Optim.Options(g_tol = 1e-9, iterations = 400);
        autodiff = :forward,
    )
    b_mode = DRModels.Optim.minimizer(result)
    H = DRModels.ForwardDiff.hessian(dense_latent_nll, b_mode)
    dense_laplace = dense_latent_nll(b_mode) -
        0.5 * logdet(cholesky(Symmetric(Q))) +
        0.5 * logdet(cholesky(Symmetric(H)))
    @test isapprox(fit.nll(theta), dense_laplace; atol = 2e-5)
    @test isposdef(Symmetric(Sigma))

    boot = bootstrap_result(fit; data, B = 1, A,
        rng = MersenneTwister(20261023), failures = :error)
    @test boot.attempted == 1
    @test boot.used == 1
    @test boot.failed == 0

    sd_profile_target = only(filter(profile_targets(fit)) do target
        target.param === :sd && target.parm == "z"
    end)
    @test sd_profile_target.profile_ready
    @test_throws ArgumentError profile_result(
        fit; parm = :sd => "z", threads = true,
    )
    sd_profile = profile_result(fit; parm = :sd => "z")
    @test sd_profile.attempted == 1
    sd_profile_stats = only(sd_profile.stats)
    @test sd_profile.failed == Int(
        sd_profile_stats.lower_endpoint_failed || sd_profile_stats.upper_endpoint_failed,
    )
    sd_profile_ci = only(sd_profile.ci)
    @test sd_profile_ci.param === :sd
    @test sd_profile_ci.coef == "z"
    @test sd_profile_stats.lower_endpoint_failed || sd_profile_stats.lower_unbounded ||
          isfinite(sd_profile_ci.lower)
    @test sd_profile_stats.upper_endpoint_failed || sd_profile_stats.upper_unbounded ||
          isfinite(sd_profile_ci.upper)
    @test sd_profile_ci.lower <= coef(fit, :sd)[2] <= sd_profile_ci.upper
    # Profile-endpoint status is a workflow smoke check, not interval calibration.

    # Exercise derived SD-slope interval plumbing independently. Two successful
    # refits are only a software smoke check, not interval calibration.
    sd_boot_ci = bootstrap_ci(fit; data, B = 2, A,
        rng = MersenneTwister(202610251), failures = :error)
    sd_boot_slope_ci = filter(row -> row.param === :sd && row.coef == "z", sd_boot_ci)
    @test length(sd_boot_slope_ci) == 1
    @test isfinite(only(sd_boot_slope_ci).lower)
    @test isfinite(only(sd_boot_slope_ci).upper)

    # Reproduce one fresh marginal draw from the stated covariance contract
    # b = D*L_A*z, followed by the row-level slope loading x_i*b[group_i].
    # Exact seeded agreement checks the simulator path without treating a
    # Monte Carlo coverage exercise as an implementation test.
    draw_seed = 20261024
    simulator = DRModels._marginal_simulator_build(fit, data; A)
    actual_draw = simulator(MersenneTwister(draw_seed))
    oracle_rng = MersenneTwister(draw_seed)
    group_index, n_group = DRModels._group_index(group)
    @test group_order != collect(1:G)
    @test group_index == row_group_index
    group_sd = exp.(hcat(ones(G), z_group[group_order]) * coef(fit, :sd))
    latent_effect = group_sd .* (cholesky(Symmetric(A)).L * randn(oracle_rng, G))
    eta_fixed = predict(fit, data; type = :link)
    expected_mu = DRModels._mean_response(
        fit.family, eta_fixed .+ x .* latent_effect[group_index],
    )
    trials = round.(Int, DRModels._scale_vector(fit, :trials))
    expected_draw = Float64[
        rand(oracle_rng, Distributions.Binomial(
            trials[i], clamp(expected_mu[i], eps(), 1 - eps()),
        )) for i in eachindex(y)
    ]
    @test n_group == G
    @test actual_draw == expected_draw

    @test_throws ArgumentError drm(
        bf(@formula(y ~ x + animal(1 + x | group)),
           @formula(sd(group, animal) ~ z)), Binomial(); data, A, se = false,
    )
    data_bad_sd = merge(data, (; z = collect(1.0:length(z))))
    @test_throws ArgumentError drm(formula, Binomial(); data = data_bad_sd, A, se = false)
end
